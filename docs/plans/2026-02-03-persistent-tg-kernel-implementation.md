# Persistent TG Kernel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve Token Generation throughput from ~16 t/s to ~40 t/s by eliminating kernel launch overhead with a persistent kernel integrated into the UnifiedKernel class.

**Architecture:** Full persistent kernel with hierarchical work distribution. All work-groups cooperate on one operation at a time, work-steal tiles within operations, use split barriers for synchronization. The UnifiedKernel class encapsulates all complexity as a black box.

**Tech Stack:** SYCL 2020, Intel oneAPI, ESIMD intrinsics, Split Barriers (SPIR-V), Unified Cache

**Design Document:** `docs/plans/2026-02-03-persistent-tg-kernel-design.md`

---

## Phase 1: UnifiedKernel Class Foundation

### Task 1.1: Create UnifiedKernel Class Scaffold

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.hpp`
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Add forward declarations and enums to unified-kernel.hpp**

Add after the existing includes (around line 30):

```cpp
// Forward declarations
class UnifiedCache;

// Operation types for persistent kernel
enum class OperationType {
    RMS_NORM,
    MATMUL_Q_PROJ,
    MATMUL_K_PROJ,
    MATMUL_V_PROJ,
    MATMUL_OUT_PROJ,
    MATMUL_GATE,
    MATMUL_UP,
    MATMUL_DOWN,
    ROPE,
    ATTENTION,
    SILU_MUL,
    SOFTMAX
};

// Matmul type classification
enum class MatmulType {
    Q_PROJ,
    K_PROJ,
    V_PROJ,
    OUT_PROJ,
    GATE,
    UP,
    DOWN,
    GENERIC
};

// Prefetch priority for cache streaming
enum class PrefetchPriority {
    LOW,
    NORMAL,
    HIGH
};
```

**Step 2: Run build to verify no syntax errors**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -20`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.hpp
git commit -m "sycl: Add operation type enums for persistent kernel"
```

---

### Task 1.2: Add Operation Descriptor Structures

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.hpp`

**Step 1: Add descriptor structures after the enums**

```cpp
// Descriptor for RMS normalization
struct RmsNormDescriptor {
    const void* input;      // Input tensor [hidden_dim]
    const void* weights;    // Norm weights [hidden_dim]
    void* output;           // Output tensor [hidden_dim]
    int hidden_dim;
    float eps;
};

// Descriptor for attention operation
struct AttentionDescriptor {
    const void* q;          // Query [n_heads, head_dim]
    const void* k_cache;    // Key cache [n_heads, seq_len, head_dim]
    const void* v_cache;    // Value cache [n_heads, seq_len, head_dim]
    void* output;           // Output [n_heads, head_dim]
    int n_heads;
    int n_kv_heads;         // For GQA
    int head_dim;
    int seq_len;
    float scale;            // 1/sqrt(head_dim)
};

// Descriptor for RoPE operation
struct RopeDescriptor {
    void* q;                // Query to rotate in-place
    void* k;                // Key to rotate in-place
    const float* cos_cache; // Cosine cache
    const float* sin_cache; // Sine cache
    int n_heads;
    int head_dim;
    int position;           // Current position in sequence
};

// Descriptor for a single operation in the persistent plan
struct OperationDescriptor {
    OperationType type;
    int layer;

    // Pointers (interpretation depends on type)
    const void* weights;
    const void* input;
    void* output;
    void* aux;              // Secondary input (e.g., 'up' for silu_mul)

    // Dimensions
    int M, N, K;
    int hidden_dim;
    int intermediate_dim;

    // Operation-specific params
    float eps;              // For RMS norm
    float scale;            // For attention
    int quant_type;         // GGML_TYPE_*
};
```

**Step 2: Run build to verify no syntax errors**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -20`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.hpp
git commit -m "sycl: Add operation descriptor structures for persistent kernel"
```

---

### Task 1.3: Add PersistentPlan Structure

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.hpp`

**Step 1: Add PersistentPlan structure**

```cpp
// Statistics from persistent kernel execution
struct PersistentStats {
    int n_operations;
    int n_layers;
    int total_tiles;
    double kernel_time_ms;
    double memory_bandwidth_gbps;
};

// Persistent execution plan (opaque to callers)
struct PersistentPlan {
    int n_layers;
    int batch_size;
    int hidden_dim;
    int intermediate_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int quant_type;

    // Operation list
    std::vector<OperationDescriptor> operations;

    // Device-side buffers
    void* intermediate_buffers[4];  // Ping-pong buffers
    int* tile_counter;              // Atomic work-stealing counter

    // Weight pointer table (for cache integration)
    void* weight_table;

    bool is_valid() const { return n_layers > 0 && !operations.empty(); }
};
```

**Step 2: Run build to verify no syntax errors**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -20`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.hpp
git commit -m "sycl: Add PersistentPlan structure for operation batching"
```

---

### Task 1.4: Create UnifiedKernel Class Declaration

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.hpp`

**Step 1: Add the UnifiedKernel class declaration at end of namespace**

```cpp
// =============================================================================
// UnifiedKernel Class
// =============================================================================
// Black-box kernel manager that handles both single operations and persistent
// execution. Callers describe WHAT to compute, the class handles HOW.

class UnifiedKernel {
public:
    // =========================================================================
    // Construction & Configuration
    // =========================================================================

    explicit UnifiedKernel(sycl::queue& queue);
    ~UnifiedKernel();

    // Non-copyable, non-movable (owns device resources)
    UnifiedKernel(const UnifiedKernel&) = delete;
    UnifiedKernel& operator=(const UnifiedKernel&) = delete;

    // Configure hardware-specific settings
    void configure(const XMXConfig& xmx_config);

    // =========================================================================
    // Single Operation API
    // =========================================================================

    void matmul(const UnifiedKernelArgs& args);
    void rms_norm(const RmsNormDescriptor& desc);
    void rope(const RopeDescriptor& desc);
    void silu_mul(const void* gate, const void* up, void* output, int dim);
    void softmax(const void* input, void* output, int n, int stride);

    // =========================================================================
    // Persistent Execution API
    // =========================================================================

    // Begin building a persistent execution plan
    void begin_persistent(int n_layers, int batch_size, int hidden_dim,
                          int intermediate_dim, int n_heads, int n_kv_heads,
                          int head_dim, int quant_type);

    // Add operations to the plan (order matters)
    void add_rms_norm(int layer, const void* weights, const void* input, void* output);
    void add_matmul(int layer, const void* weights, const void* input,
                    void* output, MatmulType type, int M, int N, int K);
    void add_attention(int layer, const AttentionDescriptor& desc);
    void add_silu_mul(int layer, const void* gate, const void* up, void* output);
    void add_rope(int layer, const RopeDescriptor& desc);

    // Finalize and execute the entire plan in one kernel launch
    void execute_persistent();

    // Cancel current plan without executing
    void cancel_persistent();

    // =========================================================================
    // State & Diagnostics
    // =========================================================================

    bool supports_persistent() const;
    bool is_building_plan() const;
    PersistentStats get_last_stats() const;

private:
    sycl::queue& queue_;
    XMXConfig xmx_config_;
    bool xmx_configured_ = false;

    // Persistent execution state
    std::unique_ptr<PersistentPlan> current_plan_;
    PersistentStats last_stats_;

    // Device-side persistent buffers (allocated on first use)
    void* persistent_buffers_[4] = {nullptr, nullptr, nullptr, nullptr};
    int* tile_counter_ = nullptr;
    size_t persistent_buffer_size_ = 0;

    // Internal methods
    void allocate_persistent_buffers(int hidden_dim, int intermediate_dim);
    void free_persistent_buffers();
    void launch_persistent_kernel();
};
```

**Step 2: Run build to verify syntax**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -30`
Expected: May have linker errors (implementation missing) but no syntax errors in header

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.hpp
git commit -m "sycl: Add UnifiedKernel class declaration"
```

---

### Task 1.5: Implement UnifiedKernel Constructor and Destructor

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Add includes at top of file if not present**

```cpp
#include <memory>
#include <chrono>
```

**Step 2: Add constructor and destructor implementation at end of file**

```cpp
// =============================================================================
// UnifiedKernel Class Implementation
// =============================================================================

UnifiedKernel::UnifiedKernel(sycl::queue& queue)
    : queue_(queue) {
    // Initialize XMX config with defaults
    xmx_config_ = {};
    xmx_config_.supported = false;

    // Zero-initialize stats
    last_stats_ = {};
}

UnifiedKernel::~UnifiedKernel() {
    free_persistent_buffers();
}

void UnifiedKernel::configure(const XMXConfig& xmx_config) {
    xmx_config_ = xmx_config;
    xmx_configured_ = true;
}

bool UnifiedKernel::supports_persistent() const {
    // Require XMX support and sufficient SLM
    if (!xmx_configured_ || !xmx_config_.supported) {
        return false;
    }
    // Need at least 32KB SLM for efficient tiling
    if (xmx_config_.slm_size < 32 * 1024) {
        return false;
    }
    return true;
}

bool UnifiedKernel::is_building_plan() const {
    return current_plan_ != nullptr;
}

PersistentStats UnifiedKernel::get_last_stats() const {
    return last_stats_;
}

void UnifiedKernel::allocate_persistent_buffers(int hidden_dim, int intermediate_dim) {
    // Calculate required buffer sizes
    // Need space for: hidden state, Q/K/V, attention output, FFN intermediate
    size_t hidden_size = hidden_dim * sizeof(sycl::half);
    size_t ffn_size = intermediate_dim * sizeof(sycl::half);
    size_t required_size = std::max(hidden_size * 4, ffn_size * 2);

    // Only reallocate if needed
    if (persistent_buffer_size_ >= required_size) {
        return;
    }

    // Free old buffers
    free_persistent_buffers();

    // Allocate new buffers
    for (int i = 0; i < 4; i++) {
        persistent_buffers_[i] = sycl::malloc_device(required_size, queue_);
    }

    // Allocate tile counter
    tile_counter_ = sycl::malloc_device<int>(1, queue_);
    queue_.memset(tile_counter_, 0, sizeof(int)).wait();

    persistent_buffer_size_ = required_size;
}

void UnifiedKernel::free_persistent_buffers() {
    for (int i = 0; i < 4; i++) {
        if (persistent_buffers_[i]) {
            sycl::free(persistent_buffers_[i], queue_);
            persistent_buffers_[i] = nullptr;
        }
    }
    if (tile_counter_) {
        sycl::free(tile_counter_, queue_);
        tile_counter_ = nullptr;
    }
    persistent_buffer_size_ = 0;
}
```

**Step 3: Run build**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -30`
Expected: May have linker errors for unimplemented methods, but constructor/destructor should compile

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Implement UnifiedKernel constructor and buffer management"
```

---

### Task 1.6: Implement Persistent Plan Building Methods

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Add plan building methods**

```cpp
void UnifiedKernel::begin_persistent(int n_layers, int batch_size, int hidden_dim,
                                      int intermediate_dim, int n_heads, int n_kv_heads,
                                      int head_dim, int quant_type) {
    // Cancel any existing plan
    cancel_persistent();

    // Create new plan
    current_plan_ = std::make_unique<PersistentPlan>();
    current_plan_->n_layers = n_layers;
    current_plan_->batch_size = batch_size;
    current_plan_->hidden_dim = hidden_dim;
    current_plan_->intermediate_dim = intermediate_dim;
    current_plan_->n_heads = n_heads;
    current_plan_->n_kv_heads = n_kv_heads;
    current_plan_->head_dim = head_dim;
    current_plan_->quant_type = quant_type;

    // Reserve space for operations (estimate: ~10 ops per layer)
    current_plan_->operations.reserve(n_layers * 10);

    // Allocate persistent buffers
    allocate_persistent_buffers(hidden_dim, intermediate_dim);
}

void UnifiedKernel::add_rms_norm(int layer, const void* weights,
                                  const void* input, void* output) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_rms_norm called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::RMS_NORM;
    op.layer = layer;
    op.weights = weights;
    op.input = input;
    op.output = output;
    op.hidden_dim = current_plan_->hidden_dim;
    op.eps = 1e-5f;  // Default RMS norm epsilon

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_matmul(int layer, const void* weights, const void* input,
                                void* output, MatmulType type, int M, int N, int K) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_matmul called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};

    // Map MatmulType to OperationType
    switch (type) {
        case MatmulType::Q_PROJ:    op.type = OperationType::MATMUL_Q_PROJ; break;
        case MatmulType::K_PROJ:    op.type = OperationType::MATMUL_K_PROJ; break;
        case MatmulType::V_PROJ:    op.type = OperationType::MATMUL_V_PROJ; break;
        case MatmulType::OUT_PROJ:  op.type = OperationType::MATMUL_OUT_PROJ; break;
        case MatmulType::GATE:      op.type = OperationType::MATMUL_GATE; break;
        case MatmulType::UP:        op.type = OperationType::MATMUL_UP; break;
        case MatmulType::DOWN:      op.type = OperationType::MATMUL_DOWN; break;
        default:                    op.type = OperationType::MATMUL_Q_PROJ; break;
    }

    op.layer = layer;
    op.weights = weights;
    op.input = input;
    op.output = output;
    op.M = M;
    op.N = N;
    op.K = K;
    op.quant_type = current_plan_->quant_type;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_attention(int layer, const AttentionDescriptor& desc) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_attention called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::ATTENTION;
    op.layer = layer;
    op.input = desc.q;
    op.weights = desc.k_cache;  // Repurpose weights ptr for k_cache
    op.aux = const_cast<void*>(static_cast<const void*>(desc.v_cache));
    op.output = desc.output;
    op.M = desc.seq_len;
    op.N = desc.n_heads;
    op.K = desc.head_dim;
    op.scale = desc.scale;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_silu_mul(int layer, const void* gate, const void* up, void* output) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_silu_mul called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::SILU_MUL;
    op.layer = layer;
    op.input = gate;
    op.aux = const_cast<void*>(up);
    op.output = output;
    op.intermediate_dim = current_plan_->intermediate_dim;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_rope(int layer, const RopeDescriptor& desc) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_rope called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::ROPE;
    op.layer = layer;
    op.input = desc.q;
    op.aux = desc.k;
    op.weights = desc.cos_cache;
    op.output = const_cast<float*>(desc.sin_cache);  // Repurpose for sin_cache
    op.N = desc.n_heads;
    op.K = desc.head_dim;
    op.M = desc.position;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::cancel_persistent() {
    current_plan_.reset();
}
```

**Step 2: Run build**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -30`
Expected: Compile success (execute_persistent still missing)

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Implement persistent plan building methods"
```

---

### Task 1.7: Add Stub for execute_persistent and Single Op Methods

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Add stub implementations to allow build**

```cpp
void UnifiedKernel::execute_persistent() {
    if (!current_plan_ || !current_plan_->is_valid()) {
        GGML_LOG_ERROR("UnifiedKernel: execute_persistent called with invalid plan\n");
        return;
    }

    // TODO: Implement actual persistent kernel launch
    // For now, fall back to individual operation dispatch
    GGML_LOG_WARN("UnifiedKernel: persistent execution not yet implemented, "
                  "falling back to individual ops\n");

    // Record stats
    last_stats_.n_operations = static_cast<int>(current_plan_->operations.size());
    last_stats_.n_layers = current_plan_->n_layers;
    last_stats_.total_tiles = 0;  // Will be computed during actual execution
    last_stats_.kernel_time_ms = 0.0;
    last_stats_.memory_bandwidth_gbps = 0.0;

    // Clear the plan
    current_plan_.reset();
}

void UnifiedKernel::launch_persistent_kernel() {
    // TODO: Implement the actual persistent kernel
    // This will be filled in during Phase 3
}

// Single operation wrappers (delegate to existing functions)
void UnifiedKernel::matmul(const UnifiedKernelArgs& args) {
    launch_unified_matmul(queue_, args);
}

void UnifiedKernel::rms_norm(const RmsNormDescriptor& desc) {
    // TODO: Implement using existing rms_norm_f32_sycl
    (void)desc;
}

void UnifiedKernel::rope(const RopeDescriptor& desc) {
    // TODO: Implement using existing rope functions
    (void)desc;
}

void UnifiedKernel::silu_mul(const void* gate, const void* up, void* output, int dim) {
    // TODO: Implement element-wise silu * multiply
    (void)gate; (void)up; (void)output; (void)dim;
}

void UnifiedKernel::softmax(const void* input, void* output, int n, int stride) {
    // TODO: Implement using existing softmax
    (void)input; (void)output; (void)n; (void)stride;
}
```

**Step 2: Run full build**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -30`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Add stub implementations for UnifiedKernel methods"
```

---

## Phase 2: Unit Tests for Core Operations (TDD)

### Task 2.1: Create Test File Structure

**Files:**
- Create: `ggml/src/ggml-sycl/tests/test-unified-kernel-ops.cpp`

**Step 1: Create test file with framework**

```cpp
//
// Unit tests for UnifiedKernel core operations
// TDD: These tests are written BEFORE the implementations
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>

#include "ggml-sycl.h"
#include "../unified-kernel.hpp"
#include "../common.hpp"

using namespace ggml_sycl_unified;

// =============================================================================
// Test Utilities
// =============================================================================

static constexpr float TEST_TOLERANCE = 1e-4f;

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

static void print_result(const char* test_name, bool passed, float error = 0.0f) {
    if (passed) {
        printf("  [PASS] %s", test_name);
        if (error > 0) printf(" (max_error=%.2e)", error);
        printf("\n");
    } else {
        printf("  [FAIL] %s (max_error=%.2e, tolerance=%.2e)\n",
               test_name, error, TEST_TOLERANCE);
    }
}

// =============================================================================
// Reference Implementations (CPU)
// =============================================================================

static void ref_rms_norm(const float* input, const float* weights, float* output,
                         int hidden_dim, float eps) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum_sq += input[i] * input[i];
    }

    // Compute RMS and scale
    float rms = std::sqrt(sum_sq / hidden_dim + eps);
    float scale = 1.0f / rms;

    // Normalize
    for (int i = 0; i < hidden_dim; i++) {
        output[i] = input[i] * scale * weights[i];
    }
}

static void ref_silu_mul(const float* gate, const float* up, float* output, int dim) {
    for (int i = 0; i < dim; i++) {
        float sigmoid_g = 1.0f / (1.0f + std::exp(-gate[i]));
        float silu_g = gate[i] * sigmoid_g;
        output[i] = silu_g * up[i];
    }
}

// =============================================================================
// Test Cases
// =============================================================================

static bool test_rms_norm_basic(sycl::queue& q) {
    const int hidden_dim = 128;
    const float eps = 1e-5f;

    // Allocate host memory
    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim);
    std::vector<float> h_output(hidden_dim);
    std::vector<float> h_ref_output(hidden_dim);

    // Initialize with deterministic values
    for (int i = 0; i < hidden_dim; i++) {
        h_input[i] = std::sin(i * 0.1f);
        h_weights[i] = 1.0f + 0.1f * std::cos(i * 0.05f);
    }

    // Compute reference
    ref_rms_norm(h_input.data(), h_weights.data(), h_ref_output.data(), hidden_dim, eps);

    // Allocate device memory
    float* d_input = sycl::malloc_device<float>(hidden_dim, q);
    float* d_weights = sycl::malloc_device<float>(hidden_dim, q);
    float* d_output = sycl::malloc_device<float>(hidden_dim, q);

    // Copy to device
    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();

    // Create UnifiedKernel and run
    ggml_sycl::UnifiedKernel kernel(q);
    XMXConfig config = {};
    config.supported = true;
    config.slm_size = 64 * 1024;
    kernel.configure(config);

    ggml_sycl::RmsNormDescriptor desc = {};
    desc.input = d_input;
    desc.weights = d_weights;
    desc.output = d_output;
    desc.hidden_dim = hidden_dim;
    desc.eps = eps;

    kernel.rms_norm(desc);
    q.wait();

    // Copy back
    q.memcpy(h_output.data(), d_output, hidden_dim * sizeof(float)).wait();

    // Compare
    float error = max_abs_error(h_output, h_ref_output);
    bool passed = error < TEST_TOLERANCE;

    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_output, q);

    print_result("test_rms_norm_basic", passed, error);
    return passed;
}

static bool test_rms_norm_large(sycl::queue& q) {
    const int hidden_dim = 4096;  // Mistral hidden dim
    const float eps = 1e-5f;

    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim);
    std::vector<float> h_output(hidden_dim);
    std::vector<float> h_ref_output(hidden_dim);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i] = std::sin(i * 0.01f) * 2.0f;
        h_weights[i] = 1.0f;
    }

    ref_rms_norm(h_input.data(), h_weights.data(), h_ref_output.data(), hidden_dim, eps);

    float* d_input = sycl::malloc_device<float>(hidden_dim, q);
    float* d_weights = sycl::malloc_device<float>(hidden_dim, q);
    float* d_output = sycl::malloc_device<float>(hidden_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel kernel(q);
    XMXConfig config = {};
    config.supported = true;
    config.slm_size = 64 * 1024;
    kernel.configure(config);

    ggml_sycl::RmsNormDescriptor desc = {};
    desc.input = d_input;
    desc.weights = d_weights;
    desc.output = d_output;
    desc.hidden_dim = hidden_dim;
    desc.eps = eps;

    kernel.rms_norm(desc);
    q.wait();

    q.memcpy(h_output.data(), d_output, hidden_dim * sizeof(float)).wait();

    float error = max_abs_error(h_output, h_ref_output);
    bool passed = error < TEST_TOLERANCE;

    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_output, q);

    print_result("test_rms_norm_large", passed, error);
    return passed;
}

static bool test_silu_mul_basic(sycl::queue& q) {
    const int dim = 256;

    std::vector<float> h_gate(dim);
    std::vector<float> h_up(dim);
    std::vector<float> h_output(dim);
    std::vector<float> h_ref_output(dim);

    for (int i = 0; i < dim; i++) {
        h_gate[i] = std::sin(i * 0.1f);
        h_up[i] = std::cos(i * 0.1f) + 1.0f;
    }

    ref_silu_mul(h_gate.data(), h_up.data(), h_ref_output.data(), dim);

    float* d_gate = sycl::malloc_device<float>(dim, q);
    float* d_up = sycl::malloc_device<float>(dim, q);
    float* d_output = sycl::malloc_device<float>(dim, q);

    q.memcpy(d_gate, h_gate.data(), dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel kernel(q);
    XMXConfig config = {};
    config.supported = true;
    config.slm_size = 64 * 1024;
    kernel.configure(config);

    kernel.silu_mul(d_gate, d_up, d_output, dim);
    q.wait();

    q.memcpy(h_output.data(), d_output, dim * sizeof(float)).wait();

    float error = max_abs_error(h_output, h_ref_output);
    bool passed = error < TEST_TOLERANCE;

    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_output, q);

    print_result("test_silu_mul_basic", passed, error);
    return passed;
}

static bool test_silu_mul_large(sycl::queue& q) {
    const int dim = 11008;  // Mistral intermediate dim

    std::vector<float> h_gate(dim);
    std::vector<float> h_up(dim);
    std::vector<float> h_output(dim);
    std::vector<float> h_ref_output(dim);

    for (int i = 0; i < dim; i++) {
        h_gate[i] = std::sin(i * 0.01f) * 3.0f;
        h_up[i] = std::cos(i * 0.01f) * 2.0f;
    }

    ref_silu_mul(h_gate.data(), h_up.data(), h_ref_output.data(), dim);

    float* d_gate = sycl::malloc_device<float>(dim, q);
    float* d_up = sycl::malloc_device<float>(dim, q);
    float* d_output = sycl::malloc_device<float>(dim, q);

    q.memcpy(d_gate, h_gate.data(), dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel kernel(q);
    XMXConfig config = {};
    config.supported = true;
    config.slm_size = 64 * 1024;
    kernel.configure(config);

    kernel.silu_mul(d_gate, d_up, d_output, dim);
    q.wait();

    q.memcpy(h_output.data(), d_output, dim * sizeof(float)).wait();

    float error = max_abs_error(h_output, h_ref_output);
    bool passed = error < TEST_TOLERANCE;

    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_output, q);

    print_result("test_silu_mul_large", passed, error);
    return passed;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    printf("UnifiedKernel Operation Tests\n");
    printf("==============================\n\n");

    try {
        sycl::queue q(sycl::gpu_selector_v);
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        int passed = 0;
        int failed = 0;

        printf("RMS Norm Tests:\n");
        if (test_rms_norm_basic(q)) passed++; else failed++;
        if (test_rms_norm_large(q)) passed++; else failed++;

        printf("\nSiLU Mul Tests:\n");
        if (test_silu_mul_basic(q)) passed++; else failed++;
        if (test_silu_mul_large(q)) passed++; else failed++;

        printf("\n==============================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }
}
```

**Step 2: Add to CMakeLists.txt**

Find the test section in `ggml/src/ggml-sycl/CMakeLists.txt` or create one:

```cmake
# Add after existing test definitions
if(GGML_SYCL_TEST)
    add_executable(test-unified-kernel-ops tests/test-unified-kernel-ops.cpp)
    target_link_libraries(test-unified-kernel-ops PRIVATE ggml-sycl ggml)
    target_include_directories(test-unified-kernel-ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
```

**Step 3: Build tests**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake -B build -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_TEST=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && ninja -C build test-unified-kernel-ops 2>&1 | tail -30`
Expected: Build succeeds

**Step 4: Run tests (they should FAIL - TDD)**

Run: `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-kernel-ops`
Expected: Tests fail (implementations are stubs)

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/tests/test-unified-kernel-ops.cpp
git add ggml/src/ggml-sycl/CMakeLists.txt
git commit -m "sycl: Add TDD tests for UnifiedKernel operations (failing)"
```

---

## Phase 3: Implement Core Operations

### Task 3.1: Implement RMS Norm Operation

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Implement rms_norm method with device kernel**

Replace the stub `rms_norm` method with:

```cpp
void UnifiedKernel::rms_norm(const RmsNormDescriptor& desc) {
    const int hidden_dim = desc.hidden_dim;
    const float eps = desc.eps;
    const float* input = static_cast<const float*>(desc.input);
    const float* weights = static_cast<const float*>(desc.weights);
    float* output = static_cast<float*>(desc.output);

    // Choose block size based on hidden dim
    const int block_size = (hidden_dim <= 1024) ? 256 : 512;
    const int n_warps = block_size / WARP_SIZE;

    queue_.submit([&](sycl::handler& cgh) {
        // SLM for cross-warp reduction
        sycl::local_accessor<float, 1> slm_reduce(n_warps, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int tid = item.get_local_id(0);
                const int warp_id = tid / WARP_SIZE;
                const int lane_id = tid % WARP_SIZE;
                const int n_warps_local = block_size / WARP_SIZE;

                // Phase 1: Compute sum of squares (register accumulation)
                float sum_sq = 0.0f;
                for (int i = tid; i < hidden_dim; i += block_size) {
                    float val = input[i];
                    sum_sq += val * val;
                }

                // Phase 2: Warp-level reduction using subgroup
                auto sg = item.get_sub_group();
                sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());

                // Phase 3: Cross-warp reduction via SLM
                if (lane_id == 0) {
                    slm_reduce[warp_id] = sum_sq;
                }

                // Barrier between writes and reads
                split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
                split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);

                // First warp reduces all partial sums
                if (warp_id == 0) {
                    sum_sq = (lane_id < n_warps_local) ? slm_reduce[lane_id] : 0.0f;
                    sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
                    if (lane_id == 0) {
                        slm_reduce[0] = sum_sq;
                    }
                }

                split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
                split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);

                // Phase 4: Compute scale and normalize
                const float rms = sycl::sqrt(slm_reduce[0] / hidden_dim + eps);
                const float scale = 1.0f / rms;

                for (int i = tid; i < hidden_dim; i += block_size) {
                    output[i] = input[i] * scale * weights[i];
                }
            });
    });
}
```

**Step 2: Run tests**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build test-unified-kernel-ops && ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-kernel-ops`
Expected: RMS norm tests pass

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Implement RMS norm in UnifiedKernel class"
```

---

### Task 3.2: Implement SiLU Mul Operation

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Implement silu_mul method**

Replace the stub with:

```cpp
void UnifiedKernel::silu_mul(const void* gate, const void* up, void* output, int dim) {
    const float* gate_f = static_cast<const float*>(gate);
    const float* up_f = static_cast<const float*>(up);
    float* output_f = static_cast<float*>(output);

    const int block_size = 256;
    const int n_blocks = (dim + block_size - 1) / block_size;

    queue_.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                const int gid = item.get_global_id(0);

                if (gid < dim) {
                    const float g = gate_f[gid];
                    const float sigmoid_g = 1.0f / (1.0f + sycl::exp(-g));
                    const float silu_g = g * sigmoid_g;
                    output_f[gid] = silu_g * up_f[gid];
                }
            });
    });
}
```

**Step 2: Run tests**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build test-unified-kernel-ops && ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-kernel-ops`
Expected: All tests pass

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Implement SiLU mul in UnifiedKernel class"
```

---

## Phase 4: Persistent Kernel Implementation

### Task 4.1: Add Persistent Kernel Device Code Structure

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Add device-side operation descriptor**

Add before the UnifiedKernel implementation:

```cpp
// =============================================================================
// Device-Side Persistent Kernel Structures
// =============================================================================

// Packed operation descriptor for device access
struct alignas(64) DeviceOperation {
    int type;           // OperationType as int
    int layer;
    const void* weights;
    const void* input;
    void* output;
    void* aux;
    int M, N, K;
    int hidden_dim;
    int intermediate_dim;
    float eps;
    float scale;
    int quant_type;
    int n_tiles;        // Number of tiles for this operation
    int pad[2];         // Padding to 64-byte alignment
};

// Persistent kernel arguments
struct PersistentKernelArgs {
    const DeviceOperation* operations;
    int n_operations;
    int* tile_counter;
    void* scratch_buffers[4];
    int hidden_dim;
    int intermediate_dim;
};
```

**Step 2: Add persistent kernel class template**

```cpp
// =============================================================================
// Persistent Kernel Implementation
// =============================================================================

template<int BLOCK_SIZE>
class PersistentTGKernelImpl {
public:
    PersistentTGKernelImpl(const PersistentKernelArgs& args,
                           sycl::local_accessor<float, 1> slm,
                           sycl::nd_item<1> item)
        : args_(args), slm_(slm), item_(item) {}

    void run() {
        const int local_id = item_.get_local_id(0);

        // Process each operation in sequence
        for (int op_idx = 0; op_idx < args_.n_operations; op_idx++) {
            const DeviceOperation& op = args_.operations[op_idx];

            // Work-stealing within this operation
            while (true) {
                int tile_idx = -1;

                // Thread 0 atomically claims next tile
                if (local_id == 0) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        counter(*args_.tile_counter);
                    tile_idx = counter.fetch_add(1);
                }

                // Broadcast to all threads
                tile_idx = sycl::group_broadcast(item_.get_group(), tile_idx, 0);

                // Exit if all tiles claimed
                if (tile_idx >= op.n_tiles) break;

                // Dispatch operation
                dispatch_operation(op, tile_idx);
            }

            // Barrier between operations
            split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
            split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);

            // Reset tile counter for next operation (thread 0 only)
            if (local_id == 0) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    counter(*args_.tile_counter);
                counter.store(0);
            }

            // Ensure counter reset is visible
            split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
            split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);
        }
    }

private:
    const PersistentKernelArgs& args_;
    sycl::local_accessor<float, 1> slm_;
    sycl::nd_item<1> item_;

    void dispatch_operation(const DeviceOperation& op, int tile_idx) {
        switch (static_cast<OperationType>(op.type)) {
            case OperationType::RMS_NORM:
                compute_rms_norm_tile(op, tile_idx);
                break;
            case OperationType::SILU_MUL:
                compute_silu_mul_tile(op, tile_idx);
                break;
            case OperationType::MATMUL_Q_PROJ:
            case OperationType::MATMUL_K_PROJ:
            case OperationType::MATMUL_V_PROJ:
            case OperationType::MATMUL_OUT_PROJ:
            case OperationType::MATMUL_GATE:
            case OperationType::MATMUL_UP:
            case OperationType::MATMUL_DOWN:
                compute_matmul_tile(op, tile_idx);
                break;
            case OperationType::ATTENTION:
                compute_attention_tile(op, tile_idx);
                break;
            default:
                break;
        }
    }

    void compute_rms_norm_tile(const DeviceOperation& op, int tile_idx) {
        // For RMS norm, tile_idx is ignored (single tile per norm)
        // All threads cooperate on the reduction
        (void)tile_idx;

        const int tid = item_.get_local_id(0);
        const int hidden_dim = op.hidden_dim;
        const float eps = op.eps;
        const float* input = static_cast<const float*>(op.input);
        const float* weights = static_cast<const float*>(op.weights);
        float* output = static_cast<float*>(op.output);

        // Sum of squares
        float sum_sq = 0.0f;
        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            float val = input[i];
            sum_sq += val * val;
        }

        // Warp reduction
        auto sg = item_.get_sub_group();
        sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());

        // Cross-warp reduction
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int n_warps = BLOCK_SIZE / WARP_SIZE;

        if (lane_id == 0) {
            slm_[warp_id] = sum_sq;
        }
        sycl::group_barrier(item_.get_group());

        if (warp_id == 0) {
            sum_sq = (lane_id < n_warps) ? slm_[lane_id] : 0.0f;
            sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
            if (lane_id == 0) {
                slm_[0] = sum_sq;
            }
        }
        sycl::group_barrier(item_.get_group());

        // Normalize
        const float rms = sycl::sqrt(slm_[0] / hidden_dim + eps);
        const float scale = 1.0f / rms;

        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            output[i] = input[i] * scale * weights[i];
        }
    }

    void compute_silu_mul_tile(const DeviceOperation& op, int tile_idx) {
        const int tid = item_.get_local_id(0);
        const int intermediate_dim = op.intermediate_dim;
        const int tile_size = 256;  // Elements per tile
        const int start = tile_idx * tile_size;

        const float* gate = static_cast<const float*>(op.input);
        const float* up = static_cast<const float*>(op.aux);
        float* output = static_cast<float*>(op.output);

        for (int i = start + tid; i < sycl::min(start + tile_size, intermediate_dim);
             i += BLOCK_SIZE) {
            const float g = gate[i];
            const float sigmoid_g = 1.0f / (1.0f + sycl::exp(-g));
            output[i] = g * sigmoid_g * up[i];
        }
    }

    void compute_matmul_tile(const DeviceOperation& op, int tile_idx) {
        // TODO: Implement dequantizing matmul tile
        // This will integrate with existing DMMV code
        (void)op;
        (void)tile_idx;
    }

    void compute_attention_tile(const DeviceOperation& op, int tile_idx) {
        // TODO: Implement attention tile
        (void)op;
        (void)tile_idx;
    }
};
```

**Step 3: Build to verify syntax**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -30`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Add persistent kernel device code structure"
```

---

### Task 4.2: Implement launch_persistent_kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp`

**Step 1: Implement the launch method**

```cpp
void UnifiedKernel::launch_persistent_kernel() {
    if (!current_plan_ || current_plan_->operations.empty()) {
        return;
    }

    // Build device-side operation table
    const size_t n_ops = current_plan_->operations.size();
    std::vector<DeviceOperation> host_ops(n_ops);

    int total_tiles = 0;
    for (size_t i = 0; i < n_ops; i++) {
        const auto& src = current_plan_->operations[i];
        auto& dst = host_ops[i];

        dst.type = static_cast<int>(src.type);
        dst.layer = src.layer;
        dst.weights = src.weights;
        dst.input = src.input;
        dst.output = src.output;
        dst.aux = src.aux;
        dst.M = src.M;
        dst.N = src.N;
        dst.K = src.K;
        dst.hidden_dim = src.hidden_dim;
        dst.intermediate_dim = src.intermediate_dim;
        dst.eps = src.eps;
        dst.scale = src.scale;
        dst.quant_type = src.quant_type;

        // Calculate tiles for this operation
        switch (src.type) {
            case OperationType::RMS_NORM:
                dst.n_tiles = 1;  // Single cooperative operation
                break;
            case OperationType::SILU_MUL:
                dst.n_tiles = (src.intermediate_dim + 255) / 256;
                break;
            case OperationType::MATMUL_Q_PROJ:
            case OperationType::MATMUL_K_PROJ:
            case OperationType::MATMUL_V_PROJ:
            case OperationType::MATMUL_OUT_PROJ:
            case OperationType::MATMUL_GATE:
            case OperationType::MATMUL_UP:
            case OperationType::MATMUL_DOWN:
                dst.n_tiles = (src.N + 63) / 64;  // 64 output columns per tile
                break;
            case OperationType::ATTENTION:
                dst.n_tiles = src.N;  // One tile per head
                break;
            default:
                dst.n_tiles = 1;
        }
        total_tiles += dst.n_tiles;
    }

    // Copy operation table to device
    DeviceOperation* d_ops = sycl::malloc_device<DeviceOperation>(n_ops, queue_);
    queue_.memcpy(d_ops, host_ops.data(), n_ops * sizeof(DeviceOperation)).wait();

    // Reset tile counter
    queue_.memset(tile_counter_, 0, sizeof(int)).wait();

    // Build kernel args
    PersistentKernelArgs args = {};
    args.operations = d_ops;
    args.n_operations = static_cast<int>(n_ops);
    args.tile_counter = tile_counter_;
    for (int i = 0; i < 4; i++) {
        args.scratch_buffers[i] = persistent_buffers_[i];
    }
    args.hidden_dim = current_plan_->hidden_dim;
    args.intermediate_dim = current_plan_->intermediate_dim;

    // Determine kernel configuration
    constexpr int BLOCK_SIZE = 256;
    const int n_workgroups = 64;  // TODO: Tune based on GPU
    const int slm_size = std::max(current_plan_->hidden_dim,
                                   current_plan_->intermediate_dim / 4);

    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();

    queue_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm(slm_size, cgh);

        auto args_copy = args;

        cgh.parallel_for(
            sycl::nd_range<1>(n_workgroups * BLOCK_SIZE, BLOCK_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                PersistentTGKernelImpl<BLOCK_SIZE> kernel(args_copy, slm, item);
                kernel.run();
            });
    });

    queue_.wait();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Record stats
    last_stats_.n_operations = static_cast<int>(n_ops);
    last_stats_.n_layers = current_plan_->n_layers;
    last_stats_.total_tiles = total_tiles;
    last_stats_.kernel_time_ms = elapsed_ms;

    // Cleanup
    sycl::free(d_ops, queue_);
}
```

**Step 2: Update execute_persistent to use launch_persistent_kernel**

```cpp
void UnifiedKernel::execute_persistent() {
    if (!current_plan_ || !current_plan_->is_valid()) {
        GGML_LOG_ERROR("UnifiedKernel: execute_persistent called with invalid plan\n");
        return;
    }

    // Launch the actual persistent kernel
    launch_persistent_kernel();

    // Clear the plan
    current_plan_.reset();
}
```

**Step 3: Build**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -30`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: Implement persistent kernel launch"
```

---

## Phase 5: Integration and End-to-End Testing

### Task 5.1: Create Persistent Execution Integration Test

**Files:**
- Create: `ggml/src/ggml-sycl/tests/test-unified-kernel-persistent.cpp`

**Step 1: Create integration test file**

```cpp
//
// Integration tests for UnifiedKernel persistent execution
//

#include <cmath>
#include <cstdio>
#include <vector>
#include <sycl/sycl.hpp>

#include "ggml-sycl.h"
#include "../unified-kernel.hpp"
#include "../common.hpp"

using namespace ggml_sycl_unified;

static constexpr float TEST_TOLERANCE = 1e-3f;

// Test a simple persistent plan: RMS norm -> SiLU mul
static bool test_persistent_simple_chain(sycl::queue& q) {
    printf("  test_persistent_simple_chain...\n");

    const int hidden_dim = 4096;
    const int intermediate_dim = 11008;

    // Allocate buffers
    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_norm_weights(hidden_dim, 1.0f);
    std::vector<float> h_gate(intermediate_dim);
    std::vector<float> h_up(intermediate_dim);
    std::vector<float> h_output(intermediate_dim);

    // Initialize
    for (int i = 0; i < hidden_dim; i++) {
        h_input[i] = std::sin(i * 0.01f);
    }
    for (int i = 0; i < intermediate_dim; i++) {
        h_gate[i] = std::sin(i * 0.005f);
        h_up[i] = std::cos(i * 0.005f) + 1.0f;
    }

    // Device buffers
    float* d_input = sycl::malloc_device<float>(hidden_dim, q);
    float* d_norm_weights = sycl::malloc_device<float>(hidden_dim, q);
    float* d_norm_output = sycl::malloc_device<float>(hidden_dim, q);
    float* d_gate = sycl::malloc_device<float>(intermediate_dim, q);
    float* d_up = sycl::malloc_device<float>(intermediate_dim, q);
    float* d_output = sycl::malloc_device<float>(intermediate_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_norm_weights, h_norm_weights.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_gate, h_gate.data(), intermediate_dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), intermediate_dim * sizeof(float)).wait();

    // Create kernel
    ggml_sycl::UnifiedKernel kernel(q);
    XMXConfig config = {};
    config.supported = true;
    config.slm_size = 64 * 1024;
    kernel.configure(config);

    // Build persistent plan
    kernel.begin_persistent(1, 1, hidden_dim, intermediate_dim, 32, 8, 128, GGML_TYPE_F32);
    kernel.add_rms_norm(0, d_norm_weights, d_input, d_norm_output);
    kernel.add_silu_mul(0, d_gate, d_up, d_output);

    // Execute
    kernel.execute_persistent();

    // Get stats
    auto stats = kernel.get_last_stats();
    printf("    Stats: %d ops, %.2f ms\n", stats.n_operations, stats.kernel_time_ms);

    // Verify output
    q.memcpy(h_output.data(), d_output, intermediate_dim * sizeof(float)).wait();

    // Check for NaN/Inf
    bool valid = true;
    for (int i = 0; i < intermediate_dim && valid; i++) {
        if (std::isnan(h_output[i]) || std::isinf(h_output[i])) {
            printf("    [FAIL] NaN/Inf at index %d\n", i);
            valid = false;
        }
    }

    // Check output is non-zero (sanity check)
    float sum = 0.0f;
    for (int i = 0; i < intermediate_dim; i++) {
        sum += std::abs(h_output[i]);
    }
    if (sum < 1e-6f) {
        printf("    [FAIL] Output is all zeros\n");
        valid = false;
    }

    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_norm_weights, q);
    sycl::free(d_norm_output, q);
    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_output, q);

    printf("    [%s]\n", valid ? "PASS" : "FAIL");
    return valid;
}

static bool test_persistent_multi_layer(sycl::queue& q) {
    printf("  test_persistent_multi_layer...\n");

    const int n_layers = 4;
    const int hidden_dim = 1024;
    const int intermediate_dim = 2816;

    // Simplified test: just RMS norms across layers
    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim, 1.0f);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i] = std::sin(i * 0.02f);
    }

    float* d_input = sycl::malloc_device<float>(hidden_dim, q);
    float* d_weights = sycl::malloc_device<float>(hidden_dim, q);
    float* d_output = sycl::malloc_device<float>(hidden_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel kernel(q);
    XMXConfig config = {};
    config.supported = true;
    config.slm_size = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(n_layers, 1, hidden_dim, intermediate_dim, 32, 8, 128, GGML_TYPE_F32);

    for (int layer = 0; layer < n_layers; layer++) {
        kernel.add_rms_norm(layer, d_weights, d_input, d_output);
    }

    kernel.execute_persistent();

    auto stats = kernel.get_last_stats();
    printf("    Stats: %d layers, %d ops, %.2f ms\n",
           stats.n_layers, stats.n_operations, stats.kernel_time_ms);

    bool valid = (stats.n_operations == n_layers);

    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_output, q);

    printf("    [%s]\n", valid ? "PASS" : "FAIL");
    return valid;
}

int main() {
    printf("UnifiedKernel Persistent Execution Tests\n");
    printf("=========================================\n\n");

    try {
        sycl::queue q(sycl::gpu_selector_v);
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        int passed = 0, failed = 0;

        if (test_persistent_simple_chain(q)) passed++; else failed++;
        if (test_persistent_multi_layer(q)) passed++; else failed++;

        printf("\n=========================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }
}
```

**Step 2: Add to CMakeLists.txt**

```cmake
add_executable(test-unified-kernel-persistent tests/test-unified-kernel-persistent.cpp)
target_link_libraries(test-unified-kernel-persistent PRIVATE ggml-sycl ggml)
target_include_directories(test-unified-kernel-persistent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
```

**Step 3: Build and run**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build test-unified-kernel-persistent && ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-kernel-persistent`
Expected: Tests pass

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/tests/test-unified-kernel-persistent.cpp
git add ggml/src/ggml-sycl/CMakeLists.txt
git commit -m "sycl: Add persistent execution integration tests"
```

---

## Phase 6: Graph Detection and Dispatch Integration

### Task 6.1: Add Graph Detection Functions

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Add detection helper functions**

Find the persistent kernel section (around line 24500) and add:

```cpp
// =============================================================================
// Persistent Kernel Graph Detection
// =============================================================================

static bool is_decode_phase(ggml_cgraph* cgraph) {
    // Check for M=1 in the dominant matmul operations
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT) {
            // src1 is the activation (M dimension is ne1[1])
            if (node->src[1]->ne[1] != 1) {
                return false;  // Not decode phase if M > 1
            }
        }
    }
    return true;
}

static bool has_supported_architecture(ggml_cgraph* cgraph) {
    // Check that graph has standard transformer ops
    bool has_rms_norm = false;
    bool has_mul_mat = false;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (node->op == GGML_OP_RMS_NORM) has_rms_norm = true;
        if (node->op == GGML_OP_MUL_MAT) has_mul_mat = true;
    }

    return has_rms_norm && has_mul_mat;
}

static bool is_persistent_compatible(ggml_backend_sycl_context& ctx, ggml_cgraph* cgraph) {
    // 1. Environment gate
    if (!ggml_sycl::env_persistent_tg_enabled()) {
        return false;
    }

    // 2. Hardware support check
    // TODO: Check ctx.unified_kernel->supports_persistent() when integrated

    // 3. Decode phase detection
    if (!is_decode_phase(cgraph)) {
        return false;
    }

    // 4. Architecture check
    if (!has_supported_architecture(cgraph)) {
        return false;
    }

    return true;
}
```

**Step 2: Build**

Run: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build ggml-sycl 2>&1 | tail -20`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: Add persistent kernel graph detection functions"
```

---

## Remaining Tasks (Summary)

The following tasks complete the implementation but follow the same pattern:

### Task 6.2: Implement Graph Operation Extraction
- Extract layer index from tensor names
- Map ggml ops to UnifiedKernel ops
- Build the persistent plan from graph

### Task 6.3: Integrate with ggml_backend_sycl_graph_compute
- Add persistent path before standard dispatch
- Wire up UnifiedKernel with context

### Task 7.1: Extend Unified Cache with Async Prefetch
- Add queue_layer_prefetch method
- Implement prefetch worker thread
- Add await_layer and release_layer

### Task 7.2: Implement DMMV Tile in Persistent Kernel
- Port existing DMMV dequantization logic
- Support Q4_0, Q4_K, Q6_K

### Task 7.3: Implement Attention Tile in Persistent Kernel
- Score computation with tiling
- Online softmax
- Value aggregation

### Task 8.1: End-to-End Validation
- Compare persistent vs standard output on Mistral 7B
- Verify numerical accuracy

### Task 8.2: Performance Benchmarking
- Measure TG throughput improvement
- Profile kernel launch counts
- Tune workgroup counts

---

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| TG throughput | ~16 t/s | 25+ t/s |
| Kernel launches/token | 280-392 | 1 |
| Output correctness | - | max_error < 1e-4 |
| All tests pass | - | Yes |
