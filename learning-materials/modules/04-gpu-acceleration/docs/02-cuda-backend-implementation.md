# CUDA Backend Implementation in llama.cpp

**Module 4, Lesson 2** | **Duration: 5 hours** | **Level: Advanced**

## Table of Contents
1. [CUDA Backend Architecture](#cuda-backend-architecture)
2. [Matrix Multiplication Kernels](#matrix-multiplication-kernels)
3. [Flash Attention Implementation](#flash-attention-implementation)
4. [Quantized GEMM](#quantized-gemm)
5. [Kernel Dispatch and Optimization](#kernel-dispatch-and-optimization)
6. [Code Walkthrough](#code-walkthrough)

---

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand llama.cpp CUDA backend architecture
- ✅ Analyze matrix multiplication kernel implementations
- ✅ Comprehend Flash Attention GPU optimization techniques
- ✅ Master quantized GEMM on GPU
- ✅ Trace kernel dispatch and selection logic
- ✅ Read and understand production CUDA code

---

## CUDA Backend Architecture

### File Organization

```
ggml/src/ggml-cuda/
├── ggml-cuda.cu              # Main backend interface (1,600+ lines)
├── common.cuh                 # Shared utilities, macros, constants
│
├── Core Operations:
│   ├── fattn.cu              # Flash Attention dispatcher
│   ├── fattn-tile.cu         # Tiled attention (basic)
│   ├── fattn-vec.cu          # Vectorized attention (quantized KV)
│   ├── fattn-mma-f16.cu      # MMA tensor core attention
│   ├── fattn-wmma-f16.cu     # WMMA tensor core attention
│   ├── fattn-common.cuh      # Shared attention utilities
│   │
│   ├── mmq.cu                # Quantized matrix multiply (MMQ)
│   ├── mmf.cu                # FP16/FP32 matrix multiply
│   ├── mmvq.cu               # Quantized matrix-vector multiply
│   ├── mmvf.cu               # FP matrix-vector multiply
│   │
│   ├── rope.cu               # Rotary position embedding
│   ├── norm.cu               # RMS norm, layer norm
│   ├── softmax.cu            # Softmax operation
│   ├── quantize.cu           # On-GPU quantization
│   │
│   └── template-instances/   # Instantiated templates
│       ├── mmq-instance-q4_0.cu
│       ├── mmq-instance-q4_k.cu
│       ├── mmq-instance-q5_k.cu
│       └── ... (20+ more)
│
├── Element-wise Operations:
│   ├── unary.cu              # Activations (ReLU, GELU, SiLU)
│   ├── binbcast.cu           # Binary broadcast ops (add, mul)
│   ├── clamp.cu              # Value clamping
│   ├── scale.cu              # Scaling operations
│   └── ... (30+ more)
│
└── Utilities:
    ├── dequantize.cuh        # Dequantization functions
    ├── vecdotq.cuh           # Vectorized dot products
    ├── mma.cuh               # MMA helpers
    └── cp-async.cuh          # Async copy primitives
```

**Total:** 50+ CUDA files, ~30,000 lines of highly optimized GPU code

### Backend Initialization

**From `ggml-cuda.cu`:**

```cpp
static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n",
                       __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);  // Max 16 GPUs

    int64_t total_vram = 0;
    for (int id = 0; id < info.device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm       = prop.multiProcessorCount;
        info.devices[id].smpb      = prop.sharedMemPerBlock;
        info.devices[id].warp_size = prop.warpSize;

        // Compute capability (e.g., 8.0 for A100)
        info.devices[id].cc = 100 * prop.major + 10 * prop.minor;

        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d\n",
                      id, prop.name, prop.major, prop.minor);
    }

    // Normalize tensor split proportions
    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    return info;
}
```

**Key Device Info Stored:**
- **nsm**: Number of streaming multiprocessors (SMs)
- **smpb**: Shared memory per block
- **cc**: Compute capability (determines available features)
- **tensor_split**: How to distribute tensors across GPUs

### Memory Allocation

**Device Memory Allocation:**

```cpp
static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;

    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        // Unified memory (automatic migration)
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            CUDA_CHECK(cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
        }
#endif
    } else {
        // Standard device memory (faster, requires explicit copy)
        err = cudaMalloc(ptr, size);
    }

    return err;
}
```

**Memory Pool:**
llama.cpp uses **memory pools** to avoid repeated malloc/free:

```cpp
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
    T * operator->() { return ptr; }
};

// Usage:
ggml_cuda_pool_alloc<char> temp_buffer(ctx.pool(), nbytes);
my_kernel<<<grid, block>>>(temp_buffer.get(), ...);
// Automatically freed when temp_buffer goes out of scope
```

---

## Matrix Multiplication Kernels

### cuBLAS vs Custom Kernels

llama.cpp uses **two strategies** for GEMM:

1. **cuBLAS** (NVIDIA's optimized library)
   - For FP16/FP32 matrices
   - Extremely optimized (uses tensor cores)
   - Used when `GGML_CUDA_FORCE_CUBLAS` is set or for large matrices

2. **Custom MMQ kernels** (Matrix Multiply Quantized)
   - For quantized formats (Q4_0, Q4_K_M, Q5_K_S, etc.)
   - Dequantize on-the-fly during multiply
   - Better performance than dequant→cuBLAS for smaller batches

### Quantized Matrix Multiply (MMQ)

**From `mmq.cu`:**

```cpp
void ggml_cuda_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,  // Quantized weights [M, K]
    const ggml_tensor * src1,  // FP32 activations [K, N]
    const ggml_tensor * ids,   // Optional IDs for MoE
    ggml_tensor * dst          // FP32 output [M, N]
) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // Quantize src1 to Q8_1 for efficient dot products
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    // Quantize FP32 input to Q8_1
    quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
                           ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);

    // Dispatch to type-specific kernel
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
        ne00, ne01, ne1, s01, ne11, s1,
        ne02, ne12, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne1
    };

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}
```

**Key Steps:**
1. **Quantize input on-the-fly** to Q8_1 (8-bit symmetric quantization)
2. **Compute quantized×quantized** dot products
3. **Accumulate in FP32** for accuracy

### Q4_K Matrix Multiply Kernel

**Conceptual kernel structure** (simplified from actual implementation):

```cuda
template<>
__global__ void mul_mat_q_kernel<GGML_TYPE_Q4_K>(
    const void * __restrict__ vx,    // Quantized weights (Q4_K format)
    const void * __restrict__ vy,    // Quantized inputs (Q8_1 format)
    float * __restrict__ dst,        // FP32 output
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst
) {
    const int row_x = blockIdx.y;
    const int col_y = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_y >= ncols_y) return;

    const block_q4_K * x = (const block_q4_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float sum = 0.0f;

    // Each block processes QK_K elements (typically 256)
    const int num_blocks = ncols_x / QK_K;

    for (int i = 0; i < num_blocks; i++) {
        const block_q4_K * x_block = &x[row_x * num_blocks + i];
        const block_q8_1 * y_block = &y[col_y * num_blocks + i];

        // Dequantize and compute dot product
        // Q4_K has 6 scales + 1 min, 8 sub-blocks
        float sum_block = 0.0f;

        for (int j = 0; j < QK_K / 8; j++) {
            // Extract 4-bit values
            uint8_t q4_vals[8];
            for (int k = 0; k < 8; k++) {
                q4_vals[k] = (x_block->qs[j*4 + k/2] >> (4*(k%2))) & 0x0F;
            }

            // Dequantize: val = (q - 8) * scale + min
            float scale = x_block->scales[j];
            float min   = x_block->dmin * x_block->scales[j];

            // Dot product with Q8_1
            int sum_i = 0;
            for (int k = 0; k < 8; k++) {
                int q4_val = q4_vals[k] - 8;  // Convert to signed
                int q8_val = y_block->qs[j*8 + k];
                sum_i += q4_val * q8_val;
            }

            sum_block += scale * sum_i + min * 8;
        }

        sum += sum_block * x_block->d * y_block->d;  // Apply outer scales
    }

    dst[row_x * ncols_y + col_y] = sum;
}
```

**Optimizations in Real Implementation:**
1. **Vectorized loads** using `int4` (128-bit) for coalesced access
2. **Shared memory tiling** to reuse weight blocks
3. **Warp-level reductions** using shuffle instructions
4. **Fused dequantization** to avoid intermediate storage

### Performance Comparison

**LLaMA-7B, A100 80GB, Q4_K_M quantization:**

| Method | Prompt (512 tok) | Generation (128 tok) |
|--------|------------------|----------------------|
| FP16 (cuBLAS) | 22 ms | 18 ms/tok |
| Q8_0 (cuBLAS dequant) | 18 ms | 14 ms/tok |
| **Q4_K_M (MMQ)** | **12 ms** | **9 ms/tok** |
| Q4_0 (MMQ) | 10 ms | 8 ms/tok |

**Speedup:** 1.8-2.2x over FP16!

---

## Flash Attention Implementation

### Why Flash Attention?

**Standard Attention Problem:**
```
Q·K^T: [batch, heads, seq, seq] → O(n²) memory!
```

For seq=8192:
- **8192² = 67M elements** per head
- **32 heads × 67M × 2 bytes = 4.3 GB** just for attention scores!

**Flash Attention Solution:**
- **Tiling:** Process attention in blocks, never materialize full matrix
- **Kernel fusion:** Combine softmax + multiply in single kernel
- **Recomputation:** Recompute some values instead of storing

### llama.cpp Flash Attention Variants

```cpp
// From fattn.cu - kernel selection logic
static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(
    const int device, const ggml_tensor * dst
) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Prefer MMA (Matrix Multiply Accumulate) on Ampere+
    if (cc >= GGML_CUDA_CC_AMPERE &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // WMMA (Warp Matrix Multiply Accumulate) on Turing+
    if (cc >= GGML_CUDA_CC_TURING &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // Vectorized for quantized KV cache
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // Fallback to tiled implementation
    return BEST_FATTN_KERNEL_TILE;
}
```

**4 Implementations:**
1. **MMA** - Matrix Multiply Accumulate (Ampere+ tensor cores, fastest for FP16)
2. **WMMA** - Warp MMA (Turing tensor cores)
3. **VEC** - Vectorized (for quantized KV cache, e.g., Q4_0, Q8_0)
4. **TILE** - Tiled basic (fallback)

### Flash Attention MMA Kernel

**Conceptual structure** (simplified):

```cuda
template <int DKQ, int DV, int ncols1, int ncols2>
__global__ void flash_attn_ext_mma_f16(
    const __half * Q,    // Query [batch, heads, seq_q, head_dim]
    const __half * K,    // Key   [batch, kv_heads, seq_k, head_dim]
    const __half * V,    // Value [batch, kv_heads, seq_k, head_dim]
    __half * KQV,        // Output [batch, heads, seq_q, head_dim]
    const __half * mask, // Optional attention mask
    float scale          // 1/sqrt(head_dim)
) {
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int seq_idx   = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile indices
    const int tile_q = seq_idx / TILE_SIZE_Q;
    const int tile_k = 0;  // Will loop over all K tiles

    // Shared memory for Q, K, V tiles
    __shared__ __half sQ[TILE_SIZE_Q][DKQ];
    __shared__ __half sK[TILE_SIZE_K][DKQ];
    __shared__ __half sV[TILE_SIZE_K][DV];
    __shared__ float sQK[TILE_SIZE_Q][TILE_SIZE_K];  // Q·K^T scores

    // Load Q tile to shared memory
    // ... (coalesced loads)

    // Loop over K tiles
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    __half accum[DV] = {0};  // Accumulated output

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        // Load K, V tiles
        // ... (coalesced loads)

        __syncthreads();

        // Compute Q·K^T using MMA (tensor cores)
        // Uses wmma::mma_sync or inline PTX for newer archs
        fragment_a<half, 16, 16, 16> frag_Q;
        fragment_b<half, 16, 16, 16> frag_K;
        fragment_c<float, 16, 16, 16> frag_QK;

        load_matrix_sync(frag_Q, sQ, ...);
        load_matrix_sync(frag_K, sK, ...);
        mma_sync(frag_QK, frag_Q, frag_K, frag_QK);

        // Apply scale and mask
        for (int i = 0; i < frag_QK.num_elements; i++) {
            frag_QK.x[i] *= scale;
            if (mask) frag_QK.x[i] += mask[...];
        }

        // Online softmax (streaming max/sum)
        float old_max = max_score;
        float new_max = max(old_max, max(frag_QK.x));
        float exp_correction = expf(old_max - new_max);

        sum_exp = sum_exp * exp_correction;

        for (int i = 0; i < frag_QK.num_elements; i++) {
            float exp_val = expf(frag_QK.x[i] - new_max);
            frag_QK.x[i] = exp_val;
            sum_exp += exp_val;
        }

        max_score = new_max;

        // Multiply by V using MMA
        fragment_b<half, 16, 16, 16> frag_V;
        fragment_c<half, 16, 16, 16> frag_O;

        load_matrix_sync(frag_V, sV, ...);
        mma_sync(frag_O, frag_QK, frag_V, frag_O);

        // Accumulate with correction
        for (int i = 0; i < DV; i++) {
            accum[i] = accum[i] * exp_correction + frag_O.x[i];
        }

        __syncthreads();
    }

    // Final normalization
    for (int i = 0; i < DV; i++) {
        accum[i] /= sum_exp;
    }

    // Write output
    // ... (coalesced stores)
}
```

**Key Techniques:**
1. **Tensor Core Usage** - 16x16x16 matrix multiply using MMA
2. **Online Softmax** - Compute max/sum incrementally without storing full matrix
3. **Tiling** - Process attention in blocks to fit in shared memory
4. **Fused Operations** - QK, softmax, and KV multiply in single kernel

### Performance Impact

**LLaMA-7B, context=2048, A100:**

| Implementation | Latency | Memory |
|----------------|---------|--------|
| Standard Attention | 45 ms | 4.2 GB |
| **Flash Attention MMA** | **8 ms** | **0.3 GB** |

**Speedup:** 5.6x faster, 14x less memory!

---

## Quantized GEMM

### On-the-Fly Dequantization

**Why not dequantize first, then use cuBLAS?**

```
Option 1: Dequant → cuBLAS
  Q4_K (4.5 GB) → FP16 (14 GB) → cuBLAS
  Time: 50 ms dequant + 30 ms GEMM = 80 ms
  Memory: 14 GB (FP16 weights)

Option 2: Fused Quantized GEMM
  Q4_K (4.5 GB) → GEMM (dequant on-the-fly)
  Time: 35 ms GEMM
  Memory: 4.5 GB (Q4_K weights)

Speedup: 2.3x faster, 3.1x less memory!
```

### Q4_K Format Details

**From `ggml-quants.h`:**

```cpp
#define QK_K 256  // 256 values per block

typedef struct {
    uint8_t scales[QK_K/16];  // 16 scales (6-bit each, packed)
    uint8_t qs[QK_K/2];       // Quantized values (4-bit each, packed)
    __half  d;                // Super-block scale (FP16)
    __half  dmin;             // Super-block min scale (FP16)
} block_q4_K;

// Dequantization formula:
// For sub-block j, element i:
//   scale = d * scales[j]
//   min   = dmin * scales[j]
//   value = (qs[i] - 8) * scale + min
```

**Memory Layout:**
```
Block 0: [scales (16B)] [qs (128B)] [d (2B)] [dmin (2B)] = 148 bytes
Block 1: [scales (16B)] [qs (128B)] [d (2B)] [dmin (2B)] = 148 bytes
...

Bits per weight: 148 bytes / 256 weights × 8 = 4.625 bits/weight
```

### Vectorized Dequantization

**From `dequantize.cuh`:**

```cuda
template<int qk, int qr, int nf16, dequantize_kernel_t dequantize_kernel>
__global__ void dequantize_block(
    const void * __restrict__ vx,
    dst_t * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) return;

    const int64_t ib = i/qk;  // Block index
    const int iqs = (i % qk) / qr;  // Quant index within block

    // Dequantize 2 elements at once using vectorized load
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[i + 0] = v.x;
    y[i + 1] = v.y;
}

// Q4_K-specific dequantization
__device__ void dequantize_q4_K(
    const void * vx, const int64_t ib, const int iqs, dfloat2 & v
) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int idx = iqs / 8;  // Sub-block index (0-31)
    const int shift = (iqs % 8) / 2;  // Position within byte

    // Extract 4-bit values (2 per byte)
    const uint8_t q = x[ib].qs[iqs];
    const int q1 = (q >> 0) & 0x0F;
    const int q2 = (q >> 4) & 0x0F;

    // Extract scales (6-bit, packed)
    const uint8_t scale_byte = x[ib].scales[idx / 2];
    const int scale = (idx % 2 == 0) ?
        (scale_byte & 0x3F) : (scale_byte >> 6);

    const float d    = __half2float(x[ib].d);
    const float dmin = __half2float(x[ib].dmin);
    const float s = d * scale;
    const float m = dmin * scale;

    v.x = (q1 - 8) * s + m;
    v.y = (q2 - 8) * s + m;
}
```

### Quantized Dot Product Kernel

**Highly optimized version** (simplified):

```cuda
template<int vdr>
__device__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq,  // Q4_K block
    const block_q8_1 * __restrict__ bq8_1,  // Q8_1 block
    const int iqs  // Index of sub-block
) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    // Load Q8_1 values (8-bit signed)
    int v[4];
    v[0] = bq8_1->qs[iqs + 0];
    v[1] = bq8_1->qs[iqs + 1];
    v[2] = bq8_1->qs[iqs + 2];
    v[3] = bq8_1->qs[iqs + 3];

    // Load Q4_K values (4-bit)
    const uint8_t q4_byte = bq4_K->qs[iqs / 2];
    int q4[4];
    q4[0] = ((q4_byte >> 0) & 0x0F) - 8;
    q4[1] = ((q4_byte >> 4) & 0x0F) - 8;

    // Dot product
    int sum_i = q4[0]*v[0] + q4[1]*v[1] + q4[2]*v[2] + q4[3]*v[3];

    // Apply scales
    const int idx = iqs / 8;
    const float scale_q4 = get_scale_q4_K(bq4_K, idx);
    const float scale_q8 = __half2float(bq8_1->d);

    return sum_i * scale_q4 * scale_q8;
}
```

---

## Kernel Dispatch and Optimization

### Compute Capability Detection

```cpp
const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

if (cc >= GGML_CUDA_CC_AMPERE) {
    // Use MMA (tensor cores)
    use_mma_kernel();
} else if (cc >= GGML_CUDA_CC_VOLTA) {
    // Use WMMA (warp-level tensor cores)
    use_wmma_kernel();
} else if (cc >= GGML_CUDA_CC_DP4A) {
    // Use DP4A (int8 dot product)
    use_dp4a_kernel();
} else {
    // Fallback
    use_basic_kernel();
}
```

**Compute Capabilities:**
- **600** - Pascal (P100)
- **610** - Pascal with DP4A (P4, P40)
- **700** - Volta (V100) - First Tensor Cores
- **750** - Turing (T4, RTX 2080) - Improved Tensor Cores
- **800** - Ampere (A100) - 3rd Gen Tensor Cores
- **890** - Ada Lovelace (RTX 4090)
- **900** - Hopper (H100) - 4th Gen Tensor Cores

### Template Specialization

llama.cpp uses **extensive template specialization** to generate optimized code paths:

```cpp
// Generate separate kernel for each quantization type
template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);

// Explicit instantiations (each in separate .cu file)
template void mul_mat_q_case<GGML_TYPE_Q4_0>(...);
template void mul_mat_q_case<GGML_TYPE_Q4_1>(...);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(...);
template void mul_mat_q_case<GGML_TYPE_Q5_K>(...);
// ... 20+ more types
```

**Why separate files?**
- **Parallel compilation** - each .cu compiles independently
- **Reduced memory** - NVCC doesn't need to instantiate all templates at once
- **Faster builds** - change one type, rebuild one file

### Launch Configuration Optimization

```cpp
// From mmq.cu - optimized block size selection
constexpr int get_mmq_x_max_host(int cc) {
    return cc >= GGML_CUDA_CC_VOLTA ? 128 :
           cc >= GGML_CUDA_CC_PASCAL ? 64 : 32;
}

constexpr int get_mmq_y(int cc) {
    return cc >= GGML_CUDA_CC_AMPERE ? 128 :
           cc >= GGML_CUDA_CC_VOLTA  ? 64 : 32;
}

// Launch kernel with optimized configuration
dim3 block_dims(get_mmq_x_max_host(cc), get_mmq_y(cc), 1);
dim3 grid_dims(blocks_per_row_x, blocks_per_col_y, blocks_per_batch);

mul_mat_q_kernel<type><<<grid_dims, block_dims, 0, stream>>>(...);
```

---

## Code Walkthrough

### Example: RoPE (Rotary Position Embedding)

**From `rope.cu`:**

```cuda
__global__ void rope_norm_f32(
    const float * x, float * dst,
    int ne0, int n_dims, const int32_t * pos,
    float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, float beta_fast, float beta_slow
) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) return;

    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 < n_dims) {
        const int i = row*ne0 + i0;
        const int p = pos[row];

        // Compute rotation angle
        const float theta_base = powf(10000.0f, -float(i0)/float(n_dims));
        const float theta = p * theta_base * freq_scale;

        // Apply rotation
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        const float x0 = x[i + 0];
        const float x1 = x[i + 1];

        dst[i + 0] = x0*cos_theta - x1*sin_theta;
        dst[i + 1] = x0*sin_theta + x1*cos_theta;
    } else {
        // Copy unchanged dimensions
        const int i = row*ne0 + i0;
        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];
    }
}
```

**Optimizations:**
1. **Processes 2 elements per thread** (pair of dimensions to rotate)
2. **Coalesced memory access** (consecutive threads → consecutive addresses)
3. **Minimal divergence** (all threads in warp follow same path)

---

## Key Takeaways

1. **llama.cpp CUDA backend is highly modular** - 50+ specialized kernels
2. **Quantized GEMM is critical** - 2-3x faster than dequant→cuBLAS
3. **Flash Attention saves memory** - 10-20x reduction in attention memory
4. **Compute capability determines features** - tensor cores, DP4A, etc.
5. **Template specialization enables optimization** - different code paths per type
6. **Memory pools reduce allocation overhead** - reuse buffers across inferences

---

## Interview Questions

### Conceptual (6 questions)

1. **Q:** Why does llama.cpp implement custom quantized GEMM instead of using cuBLAS?
   **A:** cuBLAS only supports FP16/FP32/INT8. For Q4_K and other custom formats, dequantizing first wastes memory and time. Fused quantized GEMM dequantizes on-the-fly during multiplication, achieving 2-3x speedup and 3x memory reduction.

2. **Q:** What is Flash Attention and why is it important for long context?
   **A:** Flash Attention uses tiling and kernel fusion to avoid materializing the O(n²) attention matrix. It computes attention in blocks, using online softmax to avoid storing intermediate results. This reduces memory from O(n²) to O(n), enabling 8K+ context lengths.

3. **Q:** How does llama.cpp select which CUDA kernel to use?
   **A:** It checks: (1) compute capability (determines available features like tensor cores), (2) tensor types (FP16, Q4_K, etc.), (3) problem size (small batches vs. large), and (4) compiler flags (e.g., FORCE_CUBLAS). This dispatch logic ensures the fastest kernel for each scenario.

4. **Q:** What are tensor cores and how does llama.cpp use them?
   **A:** Specialized hardware units for mixed-precision matrix multiply (D=A×B+C). llama.cpp uses them via MMA/WMMA instructions in Flash Attention (Ampere+) and can use them via cuBLAS for FP16 GEMM. They provide 10-20x speedup over CUDA cores for matrix ops.

5. **Q:** Explain the Q4_K quantization format.
   **A:** Q4_K divides weights into blocks of 256 values, each with 16 sub-blocks. Each sub-block has a 6-bit scale. Values are stored as 4-bit integers (0-15), dequantized as: `(q - 8) * scale + min`. Achieves ~4.625 bits/weight with good accuracy preservation.

6. **Q:** Why does llama.cpp use template specialization for quantization kernels?
   **A:** Each quantization format (Q4_0, Q4_K, Q5_K, IQ3_XXS, etc.) has different layouts and dequantization formulas. Template specialization generates optimized code for each type, compiled separately for parallel builds and reduced memory usage during compilation.

### Practical (6 questions)

7. **Q:** How would you profile a specific CUDA kernel in llama.cpp?
   **A:** Use Nsight Compute: `ncu --kernel-name "fattn" --metrics sm__throughput,dram__throughput ./llama-cli -m model.gguf -p "test"`. This shows occupancy, memory bandwidth, and compute utilization for the attention kernel.

8. **Q:** What's the advantage of memory pooling in GPU inference?
   **A:** Avoids repeated cudaMalloc/cudaFree calls (expensive, can take milliseconds). Pool pre-allocates large buffer, hands out chunks, reuses freed memory. Reduces allocation overhead from ~10% to <1% of inference time.

9. **Q:** How do you check if tensor cores are being used?
   **A:** Profile with Nsight Compute: `ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`. If >50%, tensor cores are active. Also check kernel names for "mma" or "wmma".

10. **Q:** How would you add support for a new quantization format to the CUDA backend?
    **A:** (1) Define block structure in `ggml-quants.h`, (2) implement dequantization in `dequantize.cuh`, (3) create kernel in `mmq.cu` (copy from similar format), (4) add template instantiation file in `template-instances/`, (5) update dispatch in `mmq.cu`, (6) update CMakeLists.txt.

11. **Q:** What causes low GPU utilization in LLM inference?
    **A:** (1) Small batch size (insufficient parallelism), (2) memory-bound operations (waiting for memory, not compute), (3) CPU-GPU transfer overhead, (4) kernel launch overhead (too many small kernels), (5) poor occupancy (too many registers or shared memory per thread).

12. **Q:** How would you optimize attention for very long contexts (>32K)?
    **A:** (1) Use Flash Attention 2 (improved tiling), (2) quantize KV cache (Q8_0 or Q4_0), (3) implement paged attention (vLLM style) for efficient memory, (4) use multi-GPU with tensor parallelism to split KV cache, (5) consider approximate attention (sparse, local+global).

---

## Additional Resources

### llama.cpp Source Files
- **ggml-cuda.cu** - Main backend (start here)
- **fattn.cu** - Flash Attention implementation
- **mmq.cu** - Quantized matrix multiply
- **common.cuh** - Utilities and macros

### Documentation
- [CUDA Programming Guide - Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Tensor Core Programming](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)

### Papers
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)

---

**Next Lesson:** [03-gpu-memory-management.md](03-gpu-memory-management.md)

**Related Labs:**
- [Lab 2: GPU Memory Profiling](../labs/lab2-gpu-memory-profiling.md)
- [Lab 4: Kernel Optimization Challenge](../labs/lab4-kernel-optimization.md)
