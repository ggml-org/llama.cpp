# XMX Sorted MoE Kernel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a unified XMX+token-sorting MoE kernel that beats oneDNN for both prefill and decode on Intel Arc B580.

**Architecture:** Three-phase kernel: (1) GPU-side token sorting by expert, (2) XMX tiled GEMM per expert batch, (3) scatter results back. Uses hardware-queried XMX dimensions and SLM double-buffering.

**Tech Stack:** Intel SYCL, joint_matrix API, ESIMD intrinsics, llama-bench

---

## Benchmark Targets

| Model | Test | Master | Current | Target |
|-------|------|--------|---------|--------|
| GPT-OSS 20B Q8_0 | pp512 | 282 t/s | 679 t/s* | 800+ t/s |
| GPT-OSS 20B Q8_0 | tg128 | 14 t/s | 31 t/s | 35+ t/s |

*After batch threshold fix (falls back to oneDNN batching).

---

### Task 1: Add XMX Capability Query Infrastructure

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp:1556-1560`
- Modify: `ggml/src/ggml-sycl/common.cpp:327-329`

**Step 1: Add XMXCapabilities struct to common.hpp**

After line 1556 (`bool gpu_has_xmx(sycl::device & dev);`), add:

```cpp
// XMX hardware capabilities queried at runtime
struct XMXCapabilities {
    bool supported = false;

    // Tile dimensions (queried from hardware)
    size_t M = 0;  // Expected: 8
    size_t N = 0;  // Expected: 16
    size_t K = 0;  // Expected: 32

    // Supported types
    bool supports_int8 = false;
    bool supports_fp16 = false;

    // Device memory info
    size_t slm_size = 0;  // Shared local memory per work-group

    // Derived optimal config
    int optimal_tiles_m = 1;
    int optimal_tiles_n = 1;
};

XMXCapabilities query_xmx_capabilities(sycl::device& dev);
```

**Step 2: Implement query_xmx_capabilities in common.cpp**

After line 329 (end of `gpu_has_xmx`), add:

```cpp
XMXCapabilities query_xmx_capabilities(sycl::device& dev) {
    XMXCapabilities caps;

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return caps;
    }
    caps.supported = true;

    // Query SLM size
    caps.slm_size = dev.get_info<sycl::info::device::local_mem_size>();

#if defined(SYCL_EXT_ONEAPI_MATRIX_VERSION) && SYCL_EXT_ONEAPI_MATRIX_VERSION >= 1
    using namespace sycl::ext::oneapi::experimental;

    try {
        auto combinations = dev.get_info<info::device::matrix_combinations>();

        for (const auto& combo : combinations) {
            // Find int8 configuration (for Q8_0)
            if (combo.atype == matrix_type::sint8 &&
                combo.btype == matrix_type::sint8) {
                caps.supports_int8 = true;
                caps.M = combo.msize;
                caps.N = combo.nsize;
                caps.K = combo.ksize;

                GGML_SYCL_DEBUG("[XMX] int8: M=%zu, N=%zu, K=%zu\n",
                               caps.M, caps.N, caps.K);
            }

            if (combo.atype == matrix_type::fp16 &&
                combo.btype == matrix_type::fp16) {
                caps.supports_fp16 = true;
            }
        }
    } catch (const sycl::exception& e) {
        GGML_SYCL_DEBUG("[XMX] Query failed: %s\n", e.what());
    }
#else
    // Fallback: assume Intel Arc defaults
    caps.supports_int8 = true;
    caps.supports_fp16 = true;
    caps.M = 8;
    caps.N = 16;
    caps.K = 32;
    GGML_SYCL_DEBUG("[XMX] Using default config: M=8, N=16, K=32\n");
#endif

    // Compute optimal tile counts
    if (caps.M > 0) {
        caps.optimal_tiles_m = std::min(4, (int)(32 / caps.M));
        caps.optimal_tiles_n = std::min(4, (int)(64 / caps.N));
    }

    return caps;
}
```

**Step 3: Add include for matrix header in common.cpp**

At the top of common.cpp, after other includes:

```cpp
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif
```

**Step 4: Rebuild and verify compilation**

```bash
source /opt/intel/oneapi/setvars.sh --force
./scripts/quick-rebuild.sh common.cpp
```

Expected: Build succeeds without errors.

**Step 5: Test XMX query with debug output**

```bash
GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-ls-sycl-device 2>&1 | grep -i xmx
```

Expected: Output showing XMX configuration (may need to call query from device init).

**Step 6: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp ggml/src/ggml-sycl/common.cpp
git commit -m "feat(sycl): add XMX capability query infrastructure

Query joint_matrix supported configurations from hardware at runtime
instead of hardcoding XMX tile dimensions. Falls back to Intel Arc
defaults (M=8, N=16, K=32) if query API unavailable."
```

---

### Task 2: Create MoE Token Sorting Header

**Files:**
- Create: `ggml/src/ggml-sycl/moe-sort.hpp`

**Step 1: Create moe-sort.hpp with sorting structures**

```cpp
// moe-sort.hpp - GPU-side token sorting for MoE kernels
#pragma once

#include "common.hpp"
#include <sycl/sycl.hpp>

// Stores original position for scatter-back after GEMM
struct MoETokenMapping {
    int32_t original_idx;  // Original token index
    int32_t expert_idx;    // Which expert this goes to
};

// Per-expert batch info
struct MoEExpertBatch {
    int32_t offset;  // Start index in sorted buffer
    int32_t count;   // Number of tokens for this expert
};

// Sort tokens by expert ID for efficient batched GEMM
// Returns total number of (token, expert) pairs processed
template<int MAX_EXPERTS = 64>
void moe_count_tokens_per_expert(
    const int32_t* expert_ids,    // [n_tokens * n_ids] expert assignments
    int32_t* expert_counts,       // [MAX_EXPERTS] output counts
    int64_t n_tokens,
    int64_t n_ids,
    sycl::queue& queue)
{
    // Zero counts
    queue.memset(expert_counts, 0, MAX_EXPERTS * sizeof(int32_t)).wait();

    // Parallel histogram
    queue.parallel_for(
        sycl::range<1>(n_tokens * n_ids),
        [=](sycl::id<1> idx) {
            int expert = expert_ids[idx];
            if (expert >= 0 && expert < MAX_EXPERTS) {
                sycl::atomic_ref<int32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(
                        expert_counts[expert]).fetch_add(1);
            }
        }).wait();
}

// Exclusive prefix sum to compute write offsets
inline void moe_compute_expert_offsets(
    const int32_t* expert_counts,  // [n_experts] input counts
    int32_t* expert_offsets,       // [n_experts] output offsets
    int64_t n_experts,
    sycl::queue& queue)
{
    // Simple sequential scan on host for now
    // TODO: GPU parallel scan for large n_experts
    std::vector<int32_t> counts(n_experts);
    std::vector<int32_t> offsets(n_experts);

    queue.memcpy(counts.data(), expert_counts,
                 n_experts * sizeof(int32_t)).wait();

    int32_t sum = 0;
    for (int64_t i = 0; i < n_experts; i++) {
        offsets[i] = sum;
        sum += counts[i];
    }

    queue.memcpy(expert_offsets, offsets.data(),
                 n_experts * sizeof(int32_t)).wait();
}

// Gather tokens into expert-contiguous layout
template<typename T>
void moe_sort_tokens_by_expert(
    const T* tokens_in,           // [n_tokens, hidden_dim]
    T* tokens_sorted,             // [total_pairs, hidden_dim]
    const int32_t* expert_ids,    // [n_tokens * n_ids]
    int32_t* expert_write_pos,    // [n_experts] atomic write positions
    MoETokenMapping* token_map,   // [total_pairs] for scatter-back
    int64_t n_tokens,
    int64_t n_ids,
    int64_t hidden_dim,
    int64_t n_experts,
    sycl::queue& queue)
{
    // Each work-item handles one (token, expert_slot) pair
    queue.parallel_for(
        sycl::range<1>(n_tokens * n_ids),
        [=](sycl::id<1> idx) {
            int64_t token_idx = idx / n_ids;
            int expert = expert_ids[idx];

            if (expert < 0 || expert >= n_experts) return;

            // Atomically claim a slot for this expert
            int32_t write_pos = sycl::atomic_ref<int32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(
                    expert_write_pos[expert]).fetch_add(1);

            // Copy token data
            for (int64_t d = 0; d < hidden_dim; d++) {
                tokens_sorted[write_pos * hidden_dim + d] =
                    tokens_in[token_idx * hidden_dim + d];
            }

            // Record mapping for scatter-back
            token_map[write_pos].original_idx = static_cast<int32_t>(idx);
            token_map[write_pos].expert_idx = expert;
        }).wait();
}

// Scatter results back to original positions
template<typename T>
void moe_scatter_results(
    const T* sorted_output,       // [total_pairs, output_dim]
    T* final_output,              // [n_tokens * n_ids, output_dim]
    const MoETokenMapping* token_map,
    int64_t total_pairs,
    int64_t output_dim,
    sycl::queue& queue)
{
    queue.parallel_for(
        sycl::range<1>(total_pairs),
        [=](sycl::id<1> idx) {
            int32_t original_pos = token_map[idx].original_idx;

            for (int64_t d = 0; d < output_dim; d++) {
                final_output[original_pos * output_dim + d] =
                    sorted_output[idx * output_dim + d];
            }
        }).wait();
}
```

**Step 2: Rebuild to verify header compiles**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-sort.hpp
git commit -m "feat(sycl): add MoE token sorting kernels

GPU-side token sorting for efficient batched MoE GEMM:
- Parallel histogram to count tokens per expert
- Prefix sum for write offsets
- Gather tokens into expert-contiguous layout
- Scatter results back to original positions"
```

---

### Task 3: Create XMX MoE GEMM Kernel Header

**Files:**
- Create: `ggml/src/ggml-sycl/moe-xmx.hpp`

**Step 1: Create moe-xmx.hpp with XMX GEMM kernel**

```cpp
// moe-xmx.hpp - XMX-accelerated MoE GEMM kernel
#pragma once

#include "common.hpp"
#include "moe-sort.hpp"
#include <sycl/sycl.hpp>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#define SYCL_XMX_MOE_AVAILABLE 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#define SYCL_XMX_MOE_AVAILABLE 0
#endif

#if SYCL_XMX_MOE_AVAILABLE

namespace moe_xmx {

using namespace sycl::ext::oneapi::experimental::matrix;

// Configuration for XMX MoE kernel
struct MoEXMXConfig {
    // Hardware parameters (from XMXCapabilities)
    int M = 8;   // Tile rows
    int N = 16;  // Tile cols
    int K = 32;  // Reduction dim

    // Tunable parameters
    int tiles_m = 4;  // Tiles per WG in M dimension
    int tiles_n = 4;  // Tiles per WG in N dimension
    int wg_size = 256;

    // SLM allocation
    int slm_weight_bytes = 16 * 1024;  // 16KB for weight double-buffer
    int slm_token_bytes = 4 * 1024;    // 4KB for token tile

    static MoEXMXConfig from_capabilities(const XMXCapabilities& caps) {
        MoEXMXConfig cfg;
        if (caps.M > 0) cfg.M = static_cast<int>(caps.M);
        if (caps.N > 0) cfg.N = static_cast<int>(caps.N);
        if (caps.K > 0) cfg.K = static_cast<int>(caps.K);
        cfg.tiles_m = caps.optimal_tiles_m;
        cfg.tiles_n = caps.optimal_tiles_n;
        return cfg;
    }
};

// Q8_0 XMX GEMM for a single expert's token batch
// Computes: output[batch, out_dim] = tokens[batch, in_dim] @ weights[out_dim, in_dim]^T
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(
    const void* weights_qs,       // [out_dim, in_dim] int8 quantized
    const sycl::half* weights_d,  // [out_dim, in_dim/32] scales
    const sycl::half* tokens,     // [batch, in_dim]
    sycl::half* output,           // [batch, out_dim]
    int64_t batch,
    int64_t out_dim,
    int64_t in_dim,
    const MoEXMXConfig& cfg,
    sycl::queue& queue)
{
    constexpr int XMX_M = 8;
    constexpr int XMX_N = 16;
    constexpr int XMX_K = 32;
    constexpr int SG_SIZE = 16;

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{
        static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
        static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols)
    };
    sycl::range<2> local{static_cast<size_t>(cfg.wg_size), 1};

    queue.submit([&](sycl::handler& cgh) {
        // SLM for weight tiles (double-buffered)
        sycl::local_accessor<int8_t, 1> slm_weights(
            sycl::range<1>(cfg.slm_weight_bytes), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int sg_id = sg.get_group_linear_id();

                int wg_row = item.get_group(0) * wg_out_rows;
                int wg_col = item.get_group(1) * wg_out_cols;

                // Bounds check
                if (wg_row >= batch) return;

                // Initialize accumulators
                joint_matrix<sycl::sub_group, int32_t, use::accumulator,
                             XMX_M, XMX_N> acc[TILES_M][TILES_N];

                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        joint_matrix_fill(sg, acc[tm][tn], 0);
                    }
                }

                // K-dimension reduction
                const int8_t* w_ptr = static_cast<const int8_t*>(weights_qs);

                for (int64_t k = 0; k < in_dim; k += XMX_K) {
                    // Load weight and token tiles
                    joint_matrix<sycl::sub_group, int8_t, use::a,
                                 XMX_M, XMX_K, layout::row_major> mat_a;
                    joint_matrix<sycl::sub_group, int8_t, use::b,
                                 XMX_K, XMX_N, layout::row_major> mat_b;

                    // Compute tiles
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int row = wg_row + tm * XMX_M;
                            int col = wg_col + tn * XMX_N;

                            if (row < batch && col < out_dim) {
                                // Load A (tokens) - need to quantize on the fly or use pre-quantized
                                // Load B (weights)
                                joint_matrix_load(sg, mat_b,
                                    w_ptr + col * in_dim + k,
                                    static_cast<size_t>(in_dim));

                                // XMX multiply-accumulate
                                joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
                            }
                        }
                    }

                    sycl::group_barrier(item.get_group());
                }

                // Store results with scale application
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        int row = wg_row + tm * XMX_M;
                        int col = wg_col + tn * XMX_N;

                        if (row < batch && col < out_dim) {
                            // TODO: Apply Q8_0 scales and store as fp16
                            joint_matrix_store(sg, acc[tm][tn],
                                reinterpret_cast<int32_t*>(output + row * out_dim + col),
                                static_cast<size_t>(out_dim),
                                layout::row_major);
                        }
                    }
                }
            });
    }).wait();
}

} // namespace moe_xmx

#endif // SYCL_XMX_MOE_AVAILABLE
```

**Step 2: Rebuild to verify header compiles**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds (may have warnings about unused template).

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): add XMX MoE GEMM kernel skeleton

Initial XMX-accelerated MoE GEMM kernel using joint_matrix API:
- Tiled GEMM with configurable tile counts
- SLM allocation for weight double-buffering
- Q8_0 weight support (int8 accumulation)

Note: This is a skeleton - full implementation pending."
```

---

### Task 4: Integrate XMX Sorted MoE into Dispatcher

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Add includes for new headers**

Near the top of ggml-sycl.cpp, after other includes:

```cpp
#include "moe-sort.hpp"
#include "moe-xmx.hpp"
```

**Step 2: Find the mul_mat_id dispatch location**

Search for `ggml_sycl_mul_mat_id` function or `GGML_OP_MUL_MAT_ID` handling.

**Step 3: Add XMX sorted path (initially disabled)**

After the existing fused MoE dispatch logic, add:

```cpp
// XMX Sorted MoE path (experimental)
// Enabled via: GGML_SYCL_XMX_MOE=1
static bool try_xmx_sorted_moe(
    ggml_backend_sycl_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* ids,
    ggml_tensor* dst)
{
#if SYCL_XMX_MOE_AVAILABLE
    static bool enabled = getenv("GGML_SYCL_XMX_MOE") != nullptr;
    if (!enabled) return false;

    // Check XMX support
    auto& caps = ctx.device_info.xmx_caps;
    if (!caps.supported || !caps.supports_int8) {
        return false;
    }

    // Check Q8_0 K dimension matches
    if (caps.K != QK8_0) {
        GGML_SYCL_DEBUG("[XMX MoE] K=%zu != QK8_0=%d, skipping\n",
                       caps.K, QK8_0);
        return false;
    }

    GGML_SYCL_DEBUG("[XMX MoE] Using XMX sorted kernel\n");

    // TODO: Implement full dispatch
    // For now, return false to fall back to existing path
    return false;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#endif
}
```

**Step 4: Rebuild and test**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp

# Test that existing path still works
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: Benchmark runs, performance unchanged from before.

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): add XMX sorted MoE dispatch stub

Add experimental XMX sorted MoE path controlled by GGML_SYCL_XMX_MOE=1.
Currently falls back to existing path - full implementation pending."
```

---

### Task 5: Add XMX Capabilities to Device Init

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (device initialization)

**Step 1: Find ggml_sycl_device_info struct**

Search for `struct ggml_sycl_device_info` or device initialization.

**Step 2: Add XMXCapabilities to device info**

In the device info struct, add:

```cpp
XMXCapabilities xmx_caps;  // Queried at init
```

**Step 3: Query XMX caps during device init**

In the device initialization function:

```cpp
info.xmx_caps = query_xmx_capabilities(info.device);
GGML_LOG_INFO("[SYCL] Device %d XMX: %s, M=%zu N=%zu K=%zu, SLM=%zuKB\n",
              device_id,
              info.xmx_caps.supported ? "yes" : "no",
              info.xmx_caps.M, info.xmx_caps.N, info.xmx_caps.K,
              info.xmx_caps.slm_size / 1024);
```

**Step 4: Rebuild and test**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp

# Verify XMX info printed at startup
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-ls-sycl-device 2>&1 | head -20
```

Expected: XMX capability info printed during device enumeration.

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): query XMX capabilities at device init

Print XMX hardware info (M/N/K dimensions, SLM size) during
device enumeration for debugging and verification."
```

---

### Task 6: Implement Full XMX Sorted MoE Dispatch

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Implement try_xmx_sorted_moe with full logic**

Replace the stub with:

```cpp
static bool try_xmx_sorted_moe(
    ggml_backend_sycl_context& ctx,
    const ggml_tensor* src0,  // [n_experts, out_dim, in_dim]
    const ggml_tensor* src1,  // [n_tokens, in_dim]
    const ggml_tensor* ids,   // [n_tokens, n_ids]
    ggml_tensor* dst,
    sycl::queue* stream)
{
#if SYCL_XMX_MOE_AVAILABLE
    static bool enabled = getenv("GGML_SYCL_XMX_MOE") != nullptr;
    if (!enabled) return false;

    auto& caps = ctx.device_info.xmx_caps;
    if (!caps.supported || !caps.supports_int8) return false;
    if (src0->type != GGML_TYPE_Q8_0) return false;

    const int64_t n_tokens = src1->ne[1];
    const int64_t n_ids = ids->ne[1];
    const int64_t in_dim = src0->ne[0];
    const int64_t out_dim = src0->ne[1];
    const int64_t n_experts = src0->ne[2];
    const int64_t total_pairs = n_tokens * n_ids;

    GGML_SYCL_DEBUG("[XMX MoE] tokens=%ld, ids=%ld, experts=%ld\n",
                   (long)n_tokens, (long)n_ids, (long)n_experts);

    // Allocate temporary buffers
    sycl::half* tokens_sorted = sycl::malloc_device<sycl::half>(
        total_pairs * in_dim, *stream);
    MoETokenMapping* token_map = sycl::malloc_device<MoETokenMapping>(
        total_pairs, *stream);
    int32_t* expert_counts = sycl::malloc_device<int32_t>(n_experts, *stream);
    int32_t* expert_offsets = sycl::malloc_device<int32_t>(n_experts, *stream);
    sycl::half* sorted_output = sycl::malloc_device<sycl::half>(
        total_pairs * out_dim, *stream);

    // Phase 1: Sort tokens by expert
    const int32_t* expert_ids = static_cast<const int32_t*>(ids->data);

    moe_count_tokens_per_expert<64>(
        expert_ids, expert_counts, n_tokens, n_ids, *stream);

    moe_compute_expert_offsets(
        expert_counts, expert_offsets, n_experts, *stream);

    // Copy offsets for atomic writes
    int32_t* expert_write_pos = sycl::malloc_device<int32_t>(n_experts, *stream);
    stream->memcpy(expert_write_pos, expert_offsets,
                   n_experts * sizeof(int32_t)).wait();

    const sycl::half* tokens_in = static_cast<const sycl::half*>(src1->data);
    moe_sort_tokens_by_expert(
        tokens_in, tokens_sorted, expert_ids, expert_write_pos,
        token_map, n_tokens, n_ids, in_dim, n_experts, *stream);

    // Phase 2: XMX GEMM per expert
    auto xmx_cfg = moe_xmx::MoEXMXConfig::from_capabilities(caps);
    std::vector<int32_t> h_counts(n_experts), h_offsets(n_experts);
    stream->memcpy(h_counts.data(), expert_counts,
                   n_experts * sizeof(int32_t)).wait();
    stream->memcpy(h_offsets.data(), expert_offsets,
                   n_experts * sizeof(int32_t)).wait();

    for (int64_t e = 0; e < n_experts; e++) {
        if (h_counts[e] == 0) continue;

        const void* expert_weights = static_cast<const char*>(src0->data) +
            e * out_dim * in_dim;  // Simplified - need proper Q8_0 indexing
        const sycl::half* expert_scales = nullptr;  // TODO: extract scales

        moe_xmx::launch_xmx_moe_gemm_q8_0<4, 4>(
            expert_weights, expert_scales,
            tokens_sorted + h_offsets[e] * in_dim,
            sorted_output + h_offsets[e] * out_dim,
            h_counts[e], out_dim, in_dim, xmx_cfg, *stream);
    }

    // Phase 3: Scatter results
    sycl::half* final_output = static_cast<sycl::half*>(dst->data);
    moe_scatter_results(
        sorted_output, final_output, token_map,
        total_pairs, out_dim, *stream);

    // Free temporaries
    sycl::free(tokens_sorted, *stream);
    sycl::free(token_map, *stream);
    sycl::free(expert_counts, *stream);
    sycl::free(expert_offsets, *stream);
    sycl::free(expert_write_pos, *stream);
    sycl::free(sorted_output, *stream);

    return true;
#else
    return false;
#endif
}
```

**Step 2: Rebuild**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

**Step 3: Test with XMX enabled**

```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0
```

Expected: May crash or produce wrong output (kernel not fully implemented).

**Step 4: Commit work in progress**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "wip(sycl): XMX sorted MoE dispatch implementation

Implements full three-phase XMX sorted MoE:
1. GPU-side token sorting by expert
2. XMX GEMM per expert batch
3. Scatter results back

Note: Q8_0 scale handling incomplete, expect incorrect output."
```

---

### Task 7: Benchmark and Iterate

**Files:**
- Test and iterate on above files

**Step 1: Run benchmarks with both paths**

```bash
# Current path (baseline)
echo "=== Current ESIMD/oneDNN path ==="
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1

# XMX sorted path (experimental)
echo "=== XMX sorted path ==="
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Step 2: Compare correctness**

```bash
# Reference output
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# XMX path
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..." from both.

**Step 3: Profile with VTune if needed**

```bash
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  vtune -collect gpu-hotspots -result-dir /tmp/vtune_xmx_moe -- \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 0 -ngl 99 -fa 1
```

**Step 4: Iterate on kernel performance**

Based on profiling results, tune:
- Tile counts (TILES_M, TILES_N)
- SLM allocation
- Double-buffering implementation
- Scale application optimization

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | XMX Capability Query | Pending |
| 2 | MoE Token Sorting Header | Pending |
| 3 | XMX MoE GEMM Header | Pending |
| 4 | Integrate into Dispatcher | Pending |
| 5 | Device Init Query | Pending |
| 6 | Full Dispatch Implementation | Pending |
| 7 | Benchmark and Iterate | Pending |

**Environment:**
```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:1
```

**Quick rebuild:**
```bash
./scripts/quick-rebuild.sh <file.cpp>
```
