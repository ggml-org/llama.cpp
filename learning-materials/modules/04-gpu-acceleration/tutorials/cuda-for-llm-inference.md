# CUDA for LLM Inference: Complete Tutorial

**Duration:** 3 hours | **Level:** Advanced

## Overview

This comprehensive tutorial walks you through GPU-accelerated LLM inference using CUDA, from basic concepts to production-level optimization. You'll learn by implementing key components of llama.cpp's CUDA backend.

## Table of Contents

1. [GPU Architecture Primer](#gpu-architecture-primer)
2. [Matrix Multiplication on GPU](#matrix-multiplication-on-gpu)
3. [Attention Mechanism Optimization](#attention-mechanism-optimization)
4. [Quantized Inference](#quantized-inference)
5. [Memory Management](#memory-management)
6. [Multi-GPU Scaling](#multi-gpu-scaling)

---

## 1. GPU Architecture Primer

### Understanding CUDA Cores vs Tensor Cores

**CUDA Cores:**
- General-purpose scalar processors
- Execute one floating-point operation per clock
- Good for: Element-wise ops, irregular workloads

**Tensor Cores (Ampere/Hopper):**
- Specialized matrix multiply units
- 4×4×4 or 16×16×16 matrix operations per clock
- 10-20x faster for matmul
- Good for: Attention, FFN layers (80%+ of LLM compute)

**Example: LLaMA-7B Forward Pass**
```
Without Tensor Cores (CUDA cores only):
  Attention:  40 ms
  FFN:        60 ms
  Other:      10 ms
  Total:      110 ms

With Tensor Cores (FP16):
  Attention:  4 ms  (10x speedup!)
  FFN:        6 ms  (10x speedup!)
  Other:      8 ms
  Total:      18 ms

Overall speedup: 6.1x
```

---

## 2. Matrix Multiplication on GPU

### Naive Implementation

```cuda
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Performance: ~100 GFLOPS on A100
// Problem: Each element of A and B loaded K times!
```

### Optimized with Shared Memory

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Performance: ~2 TFLOPS on A100 (20x speedup!)
// Shared memory reduces global memory traffic by ~32x
```

### Using cuBLAS (Tensor Cores)

```cuda
#include <cublas_v2.h>

void matmul_cublas(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    // C = alpha * A × B + beta * C
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,           // Dimensions (column-major!)
        &alpha,
        B, N,              // B matrix
        A, K,              // A matrix
        &beta,
        C, N               // C matrix (output)
    );

    cublasDestroy(handle);
}

// Performance: ~10-15 TFLOPS on A100 (100-150x over naive!)
// Tensor cores + highly optimized
```

**Lesson:** Use cuBLAS for FP16/FP32 matmul. Custom kernels only for quantized formats.

---

## 3. Attention Mechanism Optimization

### Standard Attention (Memory Bottleneck)

```cuda
// Pseudo-code for standard attention
Q = X @ W_Q  // [B, S, D]
K = X @ W_K  // [B, S, D]
V = X @ W_V  // [B, S, D]

scores = Q @ K^T  // [B, S, S] - O(S²) memory!
attn = softmax(scores / sqrt(D))
out = attn @ V

// Problem: For S=8192, stores 8192² = 67M values per head
//         32 heads × 67M × 2 bytes = 4.3 GB!
```

### Flash Attention (Tiled + Fused)

```cuda
__global__ void flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    int seq_len, int head_dim
) {
    const int TILE = 32;

    __shared__ half sQ[TILE][HEAD_DIM];
    __shared__ half sK[TILE][HEAD_DIM];
    __shared__ half sV[TILE][HEAD_DIM];

    int q_idx = blockIdx.x;  // This query

    // Load Q tile
    if (q_idx < seq_len && threadIdx.x < head_dim) {
        sQ[0][threadIdx.x] = Q[q_idx * head_dim + threadIdx.x];
    }

    float acc[HEAD_DIM] = {0};
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Loop over K/V tiles (never materialize full attention matrix!)
    for (int k_tile = 0; k_tile < (seq_len + TILE - 1) / TILE; k_tile++) {
        // Load K, V tiles
        int k_idx = k_tile * TILE + threadIdx.x;
        if (k_idx < seq_len && threadIdx.y < head_dim) {
            sK[threadIdx.x][threadIdx.y] = K[k_idx * head_dim + threadIdx.y];
            sV[threadIdx.x][threadIdx.y] = V[k_idx * head_dim + threadIdx.y];
        }
        __syncthreads();

        // Compute Q·K^T for this tile
        float scores[TILE];
        for (int k = 0; k < TILE; k++) {
            scores[k] = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                scores[k] += __half2float(sQ[0][d]) * __half2float(sK[k][d]);
            }
        }

        // Online softmax (incremental max/sum)
        float old_max = max_score;
        for (int k = 0; k < TILE; k++) {
            max_score = fmaxf(max_score, scores[k]);
        }

        float correction = expf(old_max - max_score);
        sum_exp *= correction;

        for (int k = 0; k < TILE; k++) {
            float exp_val = expf(scores[k] - max_score);
            sum_exp += exp_val;

            // Accumulate weighted V
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * correction + exp_val * __half2float(sV[k][d]);
            }
        }

        __syncthreads();
    }

    // Final normalization
    if (q_idx < seq_len && threadIdx.x < head_dim) {
        O[q_idx * head_dim + threadIdx.x] = __float2half(acc[threadIdx.x] / sum_exp);
    }
}

// Memory: O(S) instead of O(S²)!
// Speedup: 3-5x over standard attention
```

**Key Innovations:**
1. **Tiling:** Never store full S×S matrix
2. **Online Softmax:** Incremental max/sum computation
3. **Kernel Fusion:** QK^T, softmax, and V multiply in one kernel

---

## 4. Quantized Inference

### Q4_K Format (4.625 bits/weight)

```cpp
typedef struct {
    uint8_t scales[16];  // 6-bit scales, packed
    uint8_t qs[128];     // 4-bit quantized values, packed
    half d;              // Super-block scale
    half dmin;           // Super-block min
} block_q4_K;

// Block size: 256 values
// Storage: 148 bytes / 256 values = 4.625 bits/value
```

### Dequantization Kernel

```cuda
__device__ float dequant_q4_K(const block_q4_K* block, int i) {
    int sub_block = i / 8;  // 32 sub-blocks of 8 values each
    int offset = i % 8;

    // Extract 4-bit value
    uint8_t q = (block->qs[i/2] >> (4 * (i % 2))) & 0x0F;

    // Extract 6-bit scale
    uint8_t scale_byte = block->scales[sub_block / 2];
    int scale = (sub_block % 2 == 0) ? (scale_byte & 0x3F) : (scale_byte >> 6);

    float d = __half2float(block->d);
    float dmin = __half2float(block->dmin);

    // Dequantize: (q - 8) * scale * d + min
    return (int(q) - 8) * scale * d + dmin * scale;
}
```

### Quantized GEMM with On-the-Fly Dequant

```cuda
__global__ void mul_mat_q4_K(
    const block_q4_K* A,  // Quantized weights
    const float* B,        // FP32 activations
    float* C,              // FP32 output
    int M, int N, int K
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K / 256; k++) {
        const block_q4_K* block = &A[row * (K/256) + k];

        // Dequantize and accumulate
        for (int i = 0; i < 256; i++) {
            float a_val = dequant_q4_K(block, i);
            float b_val = B[col * K + k * 256 + i];
            sum += a_val * b_val;
        }
    }

    C[row * N + col] = sum;
}

// Why not dequant first?
// - Saves memory: 4.625 bits vs 32 bits = 6.9x smaller
// - Avoids extra kernel launch and global memory write
// - 2-3x faster overall!
```

---

## 5. Memory Management

### Memory Pool Pattern

```cuda
class CudaMemoryPool {
private:
    std::map<size_t, std::vector<void*>> free_buffers;
    std::map<void*, size_t> allocated;

public:
    void* allocate(size_t size) {
        size_t aligned_size = (size + 255) & ~255;  // Align to 256 bytes

        // Check free list
        auto& free_list = free_buffers[aligned_size];
        if (!free_list.empty()) {
            void* ptr = free_list.back();
            free_list.pop_back();
            return ptr;  // Instant reuse!
        }

        // Allocate new
        void* ptr;
        cudaMalloc(&ptr, aligned_size);
        allocated[ptr] = aligned_size;
        return ptr;
    }

    void free(void* ptr) {
        auto it = allocated.find(ptr);
        if (it != allocated.end()) {
            size_t size = it->second;
            allocated.erase(it);
            free_buffers[size].push_back(ptr);  // Return to pool
        }
    }

    ~CudaMemoryPool() {
        // Free all on destruction
        for (auto& [size, ptrs] : free_buffers) {
            for (void* ptr : ptrs) cudaFree(ptr);
        }
    }
};

// Usage:
CudaMemoryPool pool;

void inference() {
    void* temp1 = pool.allocate(1024 * 1024);
    kernel1<<<grid, block>>>(temp1);
    pool.free(temp1);  // Instant!

    void* temp2 = pool.allocate(1024 * 1024);  // Reuses temp1 buffer!
    kernel2<<<grid, block>>>(temp2);
    pool.free(temp2);
}

// Speedup: Eliminates cudaMalloc overhead (~0.5 ms each)
// For 100 allocations: Saves ~50 ms per inference!
```

---

## 6. Multi-GPU Scaling

### Tensor Parallelism Example

```cuda
// Split layer across 2 GPUs
void attention_multi_gpu(float* X, float* Out, int batch, int seq, int dim) {
    const int num_gpus = 2;
    const int heads_per_gpu = 32 / num_gpus;  // 16 heads each

    // GPU 0: Heads 0-15
    cudaSetDevice(0);
    attention_kernel<<<grid, block>>>(
        X, W_Q[0], W_K[0], W_V[0], W_O[0], Out_0,
        batch, seq, dim, 0, heads_per_gpu
    );

    // GPU 1: Heads 16-31
    cudaSetDevice(1);
    attention_kernel<<<grid, block>>>(
        X, W_Q[1], W_K[1], W_V[1], W_O[1], Out_1,
        batch, seq, dim, heads_per_gpu, heads_per_gpu
    );

    // All-reduce (sum outputs from both GPUs)
    ncclAllReduce(Out_0, Out_0, size, ncclFloat, ncclSum, comm, stream0);
    ncclAllReduce(Out_1, Out_1, size, ncclFloat, ncclSum, comm, stream1);

    // Merge results
    add_kernel<<<grid, block>>>(Out, Out_0, Out_1);
}

// Speedup: 1.8x with 2 GPUs (communication overhead ~10%)
```

---

## Key Takeaways

1. **Use Tensor Cores** - 10-20x speedup for matmul (80% of LLM compute)
2. **Flash Attention is critical** - Enables long context (O(n²) → O(n) memory)
3. **Quantization saves memory and time** - 4-bit weights, on-the-fly dequant
4. **Memory pools eliminate overhead** - Reuse buffers, avoid cudaMalloc
5. **Multi-GPU scales well** - NVLink enables 1.8x per GPU (up to 4-8 GPUs)

---

## Practice Exercises

1. Implement tiled matmul with different tile sizes (16, 32, 64). Which is fastest?
2. Add FP16 support to Flash Attention. Measure speedup vs FP32.
3. Profile quantized GEMM with Nsight Compute. What's the bottleneck?
4. Implement 4-GPU tensor parallelism. Measure scaling efficiency.

---

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [llama.cpp CUDA backend source](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-cuda)

**Next:** [Multi-GPU Strategies Tutorial](multi-gpu-strategies.md)
