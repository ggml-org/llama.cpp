# GPU Memory Management for LLM Inference

**Module 4, Lesson 3** | **Duration: 4 hours** | **Level: Advanced**

## Table of Contents
1. [GPU Memory Types and Hierarchy](#gpu-memory-types-and-hierarchy)
2. [Memory Allocation Strategies](#memory-allocation-strategies)
3. [KV Cache GPU Management](#kv-cache-gpu-management)
4. [Model Sharding and Offloading](#model-sharding-and-offloading)
5. [Memory Pools and Optimization](#memory-pools-and-optimization)
6. [Out-of-Memory Handling](#out-of-memory-handling)

---

## Learning Objectives

By the end of this lesson, you will:
- ✅ Master GPU memory types and their use cases
- ✅ Implement efficient memory allocation strategies
- ✅ Optimize KV cache memory management
- ✅ Understand tensor parallelism and model sharding
- ✅ Profile and optimize memory usage
- ✅ Handle OOM errors gracefully

---

## GPU Memory Types and Hierarchy

### HBM (High Bandwidth Memory)

**NVIDIA A100 Memory Specs:**
```
Capacity:     80 GB (40 GB on smaller SKU)
Bandwidth:    2,039 GB/s (HBM2e)
Technology:   HBM2e (8-hi stack)
Bus Width:    5120-bit
Latency:      ~300-600 cycles
ECC:          Yes (error correction)
```

**H100 Improvements:**
```
Capacity:     80 GB
Bandwidth:    3,350 GB/s (HBM3) - 64% increase!
Technology:   HBM3 (5-hi stack)
Bus Width:    5120-bit
```

### Memory Allocation Types

```cpp
// 1. Device Memory (standard, fastest)
float* d_ptr;
cudaMalloc(&d_ptr, size);
// - Must explicitly copy to/from host
// - Fastest access from GPU
// - No CPU access

// 2. Pinned Host Memory (page-locked)
float* h_ptr;
cudaMallocHost(&h_ptr, size);  // or cudaHostAlloc
// - CPU memory that GPU can DMA to
// - Faster CPU→GPU transfers (up to 2x)
// - Limited resource (~10% of system RAM)

// 3. Unified Memory (managed)
float* u_ptr;
cudaMallocManaged(&u_ptr, size);
// - Accessible from both CPU and GPU
// - Automatic migration
// - Convenient but can be slower

// 4. Registered Host Memory
float* h_buffer = malloc(size);
cudaHostRegister(h_buffer, size, cudaHostRegisterDefault);
// - Make existing CPU allocation GPU-accessible
// - Useful for external buffers (e.g., mmap'd files)
```

**llama.cpp Usage:**
```cpp
// From ggml-cuda.cu
static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;

    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        // Unified memory for debugging/convenience
        err = cudaMallocManaged(ptr, size);
    } else {
        // Standard device memory (default, fastest)
        err = cudaMalloc(ptr, size);
    }

    return err;
}
```

---

## Memory Allocation Strategies

### Memory Layout: Row-Major vs Column-Major

**Row-Major (C/CUDA default):**
```
Matrix A[M][N]:
  Row 0: [a00, a01, a02, ..., a0N]
  Row 1: [a10, a11, a12, ..., a1N]
  ...

Memory: [a00, a01, ..., a0N, a10, a11, ..., a1N, ...]

Coalesced access pattern: threads read across columns (row)
```

**Column-Major (Fortran/cuBLAS):**
```
Matrix A[M][N]:
  Col 0: [a00, a10, a20, ..., aM0]
  Col 1: [a01, a11, a21, ..., aM1]
  ...

Memory: [a00, a10, ..., aM0, a01, a11, ..., aM1, ...]

cuBLAS expects column-major or uses CUBLAS_OP_T for row-major
```

**llama.cpp Handling:**
```cpp
// ggml tensors are row-major, but cuBLAS expects column-major
// Solution: transpose during GEMM call

// Compute C = A × B (row-major)
// Rewrite as: C^T = B^T × A^T (column-major view)
cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both inputs
    n, m, k,
    &alpha,
    B, ...,  // B^T
    A, ...,  // A^T
    &beta,
    C, ...   // C^T
);
```

### Alignment and Padding

```cpp
// From ggml-cuda/common.cuh
#define MATRIX_ROW_PADDING 512

// Why padding?
// Bad:  Row size = 4097 elements → unaligned memory access
// Good: Row size = 4608 elements (4097 rounded up to 512)
//       → all rows start at aligned addresses

const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

// GGML_PAD macro:
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
// Example: GGML_PAD(4097, 512) = 4608
```

**Benefits:**
- **Coalesced access** - aligned addresses enable 128-byte transactions
- **Bank conflicts avoided** - shared memory banks are accessed uniformly
- **Performance** - Up to 2x speedup on memory-bound kernels

### Memory Allocation Patterns

**1. Pre-allocation (Best for Production)**
```cpp
// Allocate once at startup
struct InferenceContext {
    float* temp_buffer;      // 2 GB
    float* kv_cache;         // 8 GB
    float* intermediate;     // 1 GB
};

void init(InferenceContext* ctx, size_t max_batch) {
    cudaMalloc(&ctx->temp_buffer, 2ULL << 30);      // 2 GB
    cudaMalloc(&ctx->kv_cache, 8ULL << 30);         // 8 GB
    cudaMalloc(&ctx->intermediate, 1ULL << 30);     // 1 GB
}

// Reuse across inferences (zero allocation overhead)
```

**2. Memory Pools (llama.cpp approach)**
```cpp
// From ggml-backend.c (simplified)
struct ggml_backend_cuda_buffer_pool {
    std::map<size_t, std::vector<void*>> free_buffers;
    std::map<void*, size_t> allocated_buffers;

    void* alloc(size_t size) {
        // Round up to alignment
        size_t alloc_size = (size + 255) & ~255;

        // Check for existing free buffer
        auto it = free_buffers.find(alloc_size);
        if (it != free_buffers.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            allocated_buffers[ptr] = alloc_size;
            return ptr;
        }

        // Allocate new
        void* ptr;
        cudaMalloc(&ptr, alloc_size);
        allocated_buffers[ptr] = alloc_size;
        return ptr;
    }

    void free(void* ptr) {
        auto it = allocated_buffers.find(ptr);
        if (it != allocated_buffers.end()) {
            size_t size = it->second;
            allocated_buffers.erase(it);
            free_buffers[size].push_back(ptr);  // Return to pool
        }
    }
};
```

**Benefits:**
- **Amortized allocation** - cudaMalloc only on first use
- **Reuse** - subsequent allocations from pool (microseconds vs milliseconds)
- **Reduced fragmentation** - sizes are standardized

---

## KV Cache GPU Management

### KV Cache Size Calculation

**For LLaMA-7B:**
```
Parameters:
- Layers (L):         32
- Attention heads (H): 32
- Head dimension (D):  128
- Context length (N):  2048
- Precision:           FP16 (2 bytes)

KV Cache Size = 2 × L × N × (H × D) × sizeof(FP16)
              = 2 × 32 × 2048 × 4096 × 2
              = 1,073,741,824 bytes
              = 1 GB

For batch size B=4: 4 GB
```

**LLaMA-70B:**
```
- Layers: 80
- Heads:  64 (8 KV heads with GQA)
- Dim:    128

KV Cache Size = 2 × 80 × 2048 × (8 × 128) × 2
              = 671,088,640 bytes
              = 640 MB (thanks to GQA!)

For batch=32: 20.5 GB (fits on A100!)
```

### KV Cache Layout

**Option 1: Interleaved (simple)**
```
Memory layout:
[K_layer0_seq0, V_layer0_seq0, K_layer0_seq1, V_layer0_seq1, ...]

Pros: Simple indexing
Cons: Non-contiguous K and V access
```

**Option 2: Separated (llama.cpp)**
```
Memory layout:
K cache: [K_layer0_seq0, K_layer0_seq1, ..., K_layer31_seq2047]
V cache: [V_layer0_seq0, V_layer0_seq1, ..., V_layer31_seq2047]

Pros: Contiguous access for attention computation
Cons: Two separate buffers
```

**llama.cpp Implementation:**
```cpp
// From llama.cpp (simplified)
struct llama_kv_cache {
    struct ggml_tensor * k_l[MAX_LAYERS];  // K for each layer
    struct ggml_tensor * v_l[MAX_LAYERS];  // V for each layer

    // Dimensions: [n_layer][n_kv_heads][n_ctx][head_dim]
    // For LLaMA-7B: [32][32][2048][128]
};

// Allocation
for (int il = 0; il < n_layer; il++) {
    // K cache for this layer
    kv_cache.k_l[il] = ggml_new_tensor_3d(
        ctx,
        wtype,              // FP16, Q8_0, or Q4_0
        n_embd_k_gqa,       // head_dim * kv_heads
        n_ctx,              // context length
        1                   // batch (grows dynamically)
    );

    // V cache for this layer
    kv_cache.v_l[il] = ggml_new_tensor_3d(
        ctx,
        wtype,
        n_embd_v_gqa,
        n_ctx,
        1
    );
}
```

### Quantized KV Cache

**Why Quantize KV?**
- **4x memory reduction:** FP16 (2B) → Q8_0 (1B) → Q4_0 (0.5B)
- **Minimal accuracy loss:** <0.5% perplexity increase with Q8_0
- **Enables longer context:** 8K→32K with same VRAM

**Performance Trade-off:**
```
Configuration:      FP16    Q8_0    Q4_0
─────────────────────────────────────────
KV Cache Size:      4 GB    2 GB    1 GB
Dequant Overhead:   0 ms    +2 ms   +4 ms
Attention Kernel:   8 ms    9 ms    11 ms
Total:              8 ms    11 ms   15 ms

Speedup Factor:     1.0x    0.73x   0.53x
Context Length:     8K      16K     32K
```

**When to use:**
- **Q8_0:** Long context (16K+), slight slowdown acceptable
- **Q4_0:** Very long context (32K+), significant memory constraints
- **FP16:** Short context (<8K), maximum speed

**llama.cpp Q8_0 KV Cache:**
```cpp
// block_q8_0: 32 int8 values + 1 fp16 scale
#define QK8_0 32

typedef struct {
    half  d;        // Delta (scale)
    int8_t qs[QK8_0]; // Quants (quantized values)
} block_q8_0;

// Dequantization (in attention kernel)
__device__ float dequant_q8_0(const block_q8_0* block, int idx) {
    return __half2float(block->d) * float(block->qs[idx]);
}
```

### Dynamic KV Cache Growth

**Problem:** Pre-allocating for max context wastes memory

**Solution:** Grow dynamically
```cpp
struct llama_kv_cache {
    size_t size;      // Current size
    size_t capacity;  // Max size
    size_t used;      // Used tokens

    void grow_to(size_t new_size) {
        if (new_size <= capacity) {
            size = new_size;
            return;
        }

        // Reallocate (expensive!)
        for (int il = 0; il < n_layer; il++) {
            void* old_k = k_l[il]->data;
            void* old_v = v_l[il]->data;

            cudaMalloc(&k_l[il]->data, new_size * element_size);
            cudaMalloc(&v_l[il]->data, new_size * element_size);

            // Copy existing data
            cudaMemcpy(k_l[il]->data, old_k, size * element_size, D2D);
            cudaMemcpy(v_l[il]->data, old_v, size * element_size, D2D);

            cudaFree(old_k);
            cudaFree(old_v);
        }

        capacity = new_size;
        size = new_size;
    }
};
```

**Optimization:** Use virtual memory (VMM)
```cpp
// NVIDIA Hopper+ supports Virtual Memory Management
#if defined(GGML_USE_VMM)
// Reserve large virtual address space (e.g., 64 GB)
// Commit physical memory as needed (page granularity)
// No reallocation needed!
#endif
```

---

## Model Sharding and Offloading

### Model Memory Breakdown

**LLaMA-7B FP16:**
```
Component                      Size
─────────────────────────────────
Embeddings (32K × 4096)        256 MB
32 × Attention Weights         10.5 GB
32 × FFN Weights               12.8 GB
Output Layer (4096 × 32K)      256 MB
─────────────────────────────────
TOTAL                          ~14 GB

Add KV Cache (2K ctx, batch=1): +1 GB
Add Activations:                +0.5 GB
─────────────────────────────────
TOTAL VRAM NEEDED              ~15.5 GB

Fits on: RTX 3090 (24 GB), A100 40GB
```

**LLaMA-70B FP16:**
```
Total Weights:      ~140 GB
With KV + Act:      ~145 GB

Does NOT fit on single GPU!
Solutions: Quantization, Multi-GPU, Offloading
```

### Tensor Parallelism (Model Parallel)

**Column-wise Split (Attention Projections):**
```
Original:
  Q = X · W_Q    (W_Q: 4096 × 4096)

Split across 2 GPUs:
  GPU0: Q_0 = X · W_Q[:, 0:2048]
  GPU1: Q_1 = X · W_Q[:, 2048:4096]

  Concatenate: Q = [Q_0 | Q_1]
```

**Row-wise Split (FFN):**
```
Original:
  Y = ReLU(X · W1) · W2
  W1: 4096 × 11008, W2: 11008 × 4096

Split across 2 GPUs:
  GPU0: Y_0 = ReLU(X · W1_0) · W2_0
  GPU1: Y_1 = ReLU(X · W1_1) · W2_1

  Sum: Y = Y_0 + Y_1
```

**llama.cpp Split Tensor Buffer:**
```cpp
// From ggml-cuda.h
ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(
    int main_device,
    const float * tensor_split
);

// Usage:
float split[4] = {0.25, 0.25, 0.25, 0.25};  // 4 GPUs, equal split
auto buf_type = ggml_backend_cuda_split_buffer_type(0, split);

// Tensors allocated with this buffer type are automatically split!
```

### CPU Offloading

**Partial GPU:**
```
Run first N layers on GPU, rest on CPU

Example (LLaMA-7B, 8 GB GPU):
  GPU:  Layers 0-15  (7 GB weights + 1 GB KV)
  CPU:  Layers 16-31 (7 GB weights + 0.5 GB temp)

Speedup over full CPU: ~3-5x
Slowdown vs full GPU: ~1.5-2x
```

**llama.cpp Layer Offloading:**
```cpp
// Specify number of layers to offload to GPU
llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 20;  // First 20 layers on GPU

llama_model * model = llama_load_model_from_file("model.gguf", params);

// Inference automatically routes through GPU and CPU backends
```

**Optimizations:**
1. **Async transfers:** Overlap CPU compute with GPU→CPU transfer
2. **Pinned memory:** Use cudaMallocHost for faster transfers
3. **Pipeline:** While GPU processes layer N, CPU prepares layer N+2

---

## Memory Pools and Optimization

### CUDA Memory Allocator Overhead

**cudaMalloc/cudaFree are SLOW:**
```
cudaMalloc(1 MB):  ~0.5 ms
cudaFree(1 MB):    ~0.3 ms

For inference with 100 allocations: 80 ms overhead!
(That's entire inference time for small models!)
```

**Solution: Memory Pool**

```cpp
class SimpleCudaPool {
private:
    std::vector<void*> free_list;
    std::vector<void*> allocated;
    size_t block_size;

public:
    SimpleCudaPool(size_t block_size) : block_size(block_size) {}

    void* alloc() {
        if (free_list.empty()) {
            void* ptr;
            cudaMalloc(&ptr, block_size);
            allocated.push_back(ptr);
            return ptr;
        }

        void* ptr = free_list.back();
        free_list.pop_back();
        return ptr;
    }

    void free(void* ptr) {
        free_list.push_back(ptr);  // Return to pool (instant!)
    }

    ~SimpleCudaPool() {
        for (void* ptr : allocated) {
            cudaFree(ptr);
        }
    }
};

// Usage:
SimpleCudaPool pool(16 << 20);  // 16 MB blocks

void* buf1 = pool.alloc();  // First call: cudaMalloc (slow)
// ... use buf1 ...
pool.free(buf1);

void* buf2 = pool.alloc();  // Reuses buf1 (instant!)
```

### llama.cpp CUDA Pool

**From `ggml-backend.c`:**
```cpp
struct ggml_backend_cuda_context {
    int device;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;

    // Memory pool
    ggml_cuda_pool pool;

    // Device properties
    ggml_cuda_device_info info;
};

// Pool allocation with size classes
template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool & pool;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(pool) {
        ptr = (T *) pool.alloc(size, &actual_size);
    }

    ~ggml_cuda_pool_alloc() {
        pool.free(ptr, actual_size);
    }

    T * get() { return ptr; }
};

// Automatic cleanup via RAII
{
    ggml_cuda_pool_alloc<float> temp(ctx.pool(), 1024*1024);
    my_kernel<<<grid, block>>>(temp.get(), ...);
}  // Automatically returned to pool
```

**Performance Impact:**
```
Without Pool:
  100 cudaMalloc/free: 80 ms
  Inference:           120 ms
  Total:               200 ms

With Pool:
  Pool overhead:       <1 ms
  Inference:           120 ms
  Total:               121 ms

Speedup: 1.65x!
```

---

## Out-of-Memory Handling

### Memory Estimation

```cpp
size_t estimate_memory_usage(const ModelConfig& config) {
    size_t weights = config.n_params * sizeof_dtype(config.dtype);
    size_t kv_cache = 2 * config.n_layers * config.n_ctx *
                      config.n_embd * sizeof(float16) * config.batch_size;
    size_t activations = config.n_layers * config.n_embd * config.n_ff *
                         sizeof(float16) * config.batch_size;
    size_t overhead = 0.1 * (weights + kv_cache);  // 10% buffer

    return weights + kv_cache + activations + overhead;
}

// Check available memory
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);

size_t required = estimate_memory_usage(config);
if (required > free_mem) {
    // Try: (1) smaller batch, (2) quantization, (3) offloading
}
```

### Graceful Degradation

**llama.cpp Strategy:**
```cpp
// Try configurations in order of preference
bool load_model(const char* path) {
    // 1. Try full GPU (best performance)
    if (try_load_full_gpu(path))
        return true;

    // 2. Try with quantized KV cache
    if (try_load_with_quantized_kv(path))
        return true;

    // 3. Try partial GPU offload
    if (try_load_partial_gpu(path, auto_detect_layers()))
        return true;

    // 4. Fallback to CPU
    return try_load_cpu(path);
}
```

### Memory Leak Detection

```bash
# Use cuda-memcheck
cuda-memcheck ./llama-cli -m model.gguf -p "test"

# Output:
# ========= LEAK SUMMARY =========
# cudaMalloc:  10 allocations, 8 frees → 2 LEAKS!
# Total leaked: 512 MB

# Use compute-sanitizer (newer)
compute-sanitizer --tool memcheck ./llama-cli -m model.gguf
```

**Common Causes:**
1. **Missing cudaFree** - forgetting to free allocated memory
2. **Early returns** - error paths that skip cleanup
3. **Exception safety** - C++ exceptions bypassing cleanup
4. **Stream synchronization** - freeing before kernel completes

**Fix with RAII:**
```cpp
// BAD: Manual management
float* d_ptr;
cudaMalloc(&d_ptr, size);
if (error) return;  // LEAK!
kernel<<<grid, block>>>(d_ptr);
cudaFree(d_ptr);

// GOOD: RAII wrapper
{
    CudaPtr<float> d_ptr(size);
    if (error) return;  // Automatically freed by destructor
    kernel<<<grid, block>>>(d_ptr.get());
}  // cudaFree called automatically
```

---

## Key Takeaways

1. **Memory hierarchy matters** - HBM bandwidth is often the bottleneck
2. **KV cache dominates memory** - 40-60% of total VRAM for long contexts
3. **Quantized KV cache enables long context** - 4x memory reduction
4. **Memory pools eliminate allocation overhead** - 1.5-2x speedup
5. **Tensor parallelism enables large models** - split across multiple GPUs
6. **CPU offloading extends GPU capability** - run models 2-3x larger

---

## Interview Questions

1. **Q:** How much memory does the KV cache use for LLaMA-7B at 4K context?
   **A:** 2 × 32 layers × 4096 tokens × (32 heads × 128 dim) × 2 bytes = 2 GB

2. **Q:** Why is cudaMalloc slow and how do memory pools help?
   **A:** cudaMalloc synchronizes with the GPU and manages virtual→physical mapping (~0.5ms). Pools pre-allocate and reuse buffers, reducing allocation to <1μs.

3. **Q:** What's the trade-off of Q8_0 KV cache?
   **A:** 2x memory reduction (2B→1B per value) with ~20% slower attention due to dequantization overhead. Minimal accuracy loss (<0.5% perplexity).

4. **Q:** How does tensor parallelism split attention layers?
   **A:** Column-wise split of Q/K/V projections across GPUs, each computes partial attention, then all-reduce to combine results. Requires inter-GPU communication.

5. **Q:** What happens if you don't cudaFree?
   **A:** Memory leak - allocated memory remains reserved until process exits. Can exhaust VRAM, causing future allocations to fail. Detectable with cuda-memcheck.

---

**Next Lesson:** [04-multi-gpu-inference.md](04-multi-gpu-inference.md)

**Related Labs:**
- [Lab 2: GPU Memory Profiling](../labs/lab2-gpu-memory-profiling.md)
- [Lab 3: Multi-GPU Setup](../labs/lab3-multi-gpu-setup.md)
