# GPU Optimization for Machine Learning: Principles and Practices

**Topic**: GPU Architecture and ML Optimization Techniques
**Key Papers**: Various NVIDIA whitepapers, CUDA optimization guides
**Module**: 4 - GPU Acceleration & Performance
**Reading Time**: 45-60 minutes
**Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

GPU acceleration is critical for LLM inference, providing 10-100× speedup over CPU. Understanding GPU architecture, memory hierarchy, and kernel optimization enables efficient implementation of attention, matrix multiplication, and other transformer operations.

**Key Concepts**: Memory bandwidth, kernel fusion, occupancy, tensor cores

---

## 1. GPU Architecture Fundamentals

### 1.1 Streaming Multiprocessors (SMs)

```
Modern GPU (A100):
- 108 Streaming Multiprocessors (SMs)
- Each SM:
  - 64 FP32 cores
  - 32 FP64 cores
  - 4 Tensor Cores (matrix ops)
  - 64KB shared memory
  - 65,536 registers

Total: 6,912 CUDA cores
```

### 1.2 Memory Hierarchy

```
Speed and Size Trade-off:
┌─────────────────────┬──────────┬───────────┬─────────┐
│ Memory Level        │ Size     │ Bandwidth │ Latency │
├─────────────────────┼──────────┼───────────┼─────────┤
│ Registers           │ ~20 MB   │ >20 TB/s  │ 1 cycle │
│ L1 Cache/Shared Mem │ ~10 MB   │ ~15 TB/s  │ ~30 cyc │
│ L2 Cache            │ 40 MB    │ ~7 TB/s   │ ~200 cyc│
│ HBM (Global Memory) │ 40-80 GB │ 1.5-2 TB/s│ ~400 cyc│
└─────────────────────┴──────────┴───────────┴─────────┘

Key Insight: Memory bandwidth is the bottleneck!
```

**Optimization principle**: Minimize HBM accesses, maximize compute/memory ratio

---

## 2. Matrix Multiplication Optimization

### 2.1 Naive Implementation (Slow)

```cuda
// Naive GEMM: C = A × B
__global__ void naive_gemm(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // HBM access every iteration!
    }
    C[row * N + col] = sum;
}

// Performance: ~100 GFLOPS (A100 can do 19,500 GFLOPS!)
// Problem: Excessive HBM accesses
```

### 2.2 Tiled GEMM with Shared Memory

```cuda
#define TILE_SIZE 32

__global__ void tiled_gemm(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory (collaborative loading)
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute using shared memory (fast!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// Performance: ~1,000 GFLOPS (10× improvement)
// HBM accesses reduced by 32× (tile size)
```

### 2.3 Tensor Core Acceleration

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensor_core_gemm(
    half *A, half *B, float *C, int M, int N, int K
) {
    // Tensor Core fragments (16×16×16 matrix multiply)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);

    // Load and multiply-accumulate
    load_matrix_sync(a_frag, A, N);
    load_matrix_sync(b_frag, B, K);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    store_matrix_sync(C, c_frag, N, mem_row_major);
}

// Performance: ~10,000-15,000 GFLOPS (100× vs naive)
// Tensor Cores compute 16×16×16 matrix in single instruction
```

---

## 3. Attention Mechanism Optimization

### 3.1 Standard Attention (Memory Intensive)

```python
def standard_attention(Q, K, V):
    """
    Q, K, V: [batch, heads, seq_len, head_dim]

    Memory: O(n²) for attention matrix
    Time: O(n² × d)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [n, n] - HBM write!
    attn = F.softmax(scores, dim=-1)  # [n, n] - HBM read/write!
    output = torch.matmul(attn, V)    # [n, n] HBM read
    return output

# Problem: 3 passes through HBM for n×n matrix
# For n=4096: 16M × 2 bytes × 3 = 96 MB per attention head
# LLaMA 7B (32 heads × 32 layers): 96 GB total memory traffic!
```

### 3.2 Flash Attention (Optimized)

```python
# Conceptual (actual implementation in CUDA)
def flash_attention_forward(Q, K, V, block_size=128):
    """
    Memory: O(n) - no materialization of attention matrix
    Speed: 2-4× faster than standard attention

    Key idea: Compute attention incrementally using tiling
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(N)  # softmax normalizer
    m = torch.full((N,), float('-inf'))  # running max

    # Process in blocks
    for j in range(0, N, block_size):
        K_j = K[j:j+block_size]  # Load K block to SRAM
        V_j = V[j:j+block_size]

        for i in range(0, N, block_size):
            Q_i = Q[i:i+block_size]  # Load Q block to SRAM

            # Compute S_ij = Q_i K_j^T (in SRAM!)
            S_ij = Q_i @ K_j.T / sqrt(d)

            # Online softmax (numerically stable)
            m_new = max(m[i:i+block_size], S_ij.max(dim=1))
            l_new = exp(m[i:i+block_size] - m_new) * l[i:i+block_size] + \
                    exp(S_ij - m_new).sum(dim=1)

            # Update output
            O[i:i+block_size] = O[i:i+block_size] * exp(m - m_new) + \
                                exp(S_ij - m_new) @ V_j

            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new

    return O / l.unsqueeze(-1)

# HBM accesses: O(n²/M) where M = SRAM size
# For block_size=128: 128× less HBM traffic!
```

---

## 4. Kernel Fusion

### 4.1 Unfused Operations (Slow)

```python
# Separate kernels (each hits HBM)
x = layer_norm(x)        # Kernel 1: Read x, write normalized_x
x = attention(x)         # Kernel 2: Read normalized_x, write attn_out
x = x + residual         # Kernel 3: Read attn_out + residual, write result
x = activation(x)        # Kernel 4: Read result, write activated

# Total HBM accesses: 8 reads + 4 writes = 12 HBM transactions
```

### 4.2 Fused Kernel (Fast)

```cuda
__global__ void fused_attention_block(
    float *input, float *output, int N, int d
) {
    // Load input to shared memory (1 HBM read)
    __shared__ float shared_input[BLOCK_SIZE];
    shared_input[threadIdx.x] = input[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    __syncthreads();

    // All operations in shared memory/registers
    float x = shared_input[threadIdx.x];
    x = layer_norm(x);        // No HBM access
    x = attention(x);         // No HBM access
    x = x + residual;         // No HBM access
    x = activation(x);        // No HBM access

    // Write output (1 HBM write)
    output[blockIdx.x * BLOCK_SIZE + threadIdx.x] = x;
}

// HBM accesses: 1 read + 1 write = 2 (vs 12 unfused)
// Speedup: ~6× for memory-bound operations
```

---

## 5. llama.cpp GPU Optimization Techniques

### 5.1 CUDA Backend Architecture

```cpp
// ggml-cuda.cu - Key optimizations

// 1. Kernel fusion for matrix ops
void ggml_cuda_mul_mat(
    const ggml_tensor *src0,  // Weight matrix
    const ggml_tensor *src1,  // Input
    ggml_tensor *dst
) {
    // Dispatch to specialized kernels based on quantization
    if (src0->type == GGML_TYPE_Q4_K) {
        mul_mat_q4_K<<<grid, block>>>(src0, src1, dst);
    } else if (src0->type == GGML_TYPE_Q8_0) {
        mul_mat_q8_0<<<grid, block>>>(src0, src1, dst);
    }
    // ... other types
}

// 2. Dequantization + matmul fusion
__global__ void mul_mat_q4_K(/*...*/) {
    // Dequantize weights in shared memory (avoid HBM)
    __shared__ float weights_dequant[TILE_SIZE * TILE_SIZE];

    // Dequantize block
    dequantize_q4_K(weights_q4, weights_dequant);
    __syncthreads();

    // Matrix multiply with dequantized weights (in SRAM)
    // ... GEMM code ...
}
```

### 5.2 Performance Tuning

```bash
# Enable CUDA backend
cmake -B build -DGGML_CUDA=ON

# Flash Attention (for long contexts)
cmake -B build -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON

# Optimizations in llama-cli
./llama-cli \
  -m model.gguf \
  -ngl 32           # Offload 32 layers to GPU
  -fa               # Enable Flash Attention
  -b 512            # Batch size (larger for GPU)
  --n-gpu-layers 32 # Explicit GPU layer count
```

---

## 6. Key Performance Metrics

### 6.1 Compute vs Memory Bound

```
Arithmetic Intensity = FLOPs / Bytes

GPU Roofline:
- Peak compute: 19.5 TFLOPS (A100 FP16)
- Peak bandwidth: 1.5 TB/s

Balanced intensity: 19.5 TFLOPS / 1.5 TB/s = 13 FLOPs/byte

Operations:
- Matrix multiply (large): ~50 FLOPs/byte (compute-bound) ✓
- Element-wise ops: ~1 FLOP/byte (memory-bound) ✗
- Attention (standard): ~5 FLOPs/byte (memory-bound) ✗
- Attention (Flash): ~13 FLOPs/byte (balanced) ✓
```

### 6.2 Occupancy

```python
# GPU occupancy = active warps / max warps per SM

# Low occupancy example:
threads_per_block = 32
blocks_per_sm = 2048 / 32 = 64
# But if registers/shared memory limits to 16 blocks:
occupancy = 16 / 64 = 25% (bad!)

# High occupancy example:
threads_per_block = 256
blocks_per_sm = 8
occupancy = 8 / 8 = 100% (good!)

# Optimize for 50-100% occupancy
```

---

## 7. Practical Recommendations

### For llama.cpp Users

✅ **GPU Selection**:
- A100 (80GB): Best for large models (65B+)
- RTX 4090: Best price/performance for consumers
- RTX 3090: Good budget option

✅ **Memory Management**:
- Use `-ngl` to offload layers strategically
- Monitor VRAM usage (nvidia-smi)
- Leave 2-4GB headroom for KV-cache

✅ **Optimization Flags**:
```bash
# Optimal settings for RTX 4090
./llama-cli \
  -m llama-2-13b-q4_K_M.gguf \
  -ngl 40                        # Offload all layers
  -fa                            # Flash Attention
  -b 512                         # Large batch (GPUs love parallelism)
  -c 4096                        # Context window
  --flash-attn                   # Enable if compiled
```

---

## 8. Key Takeaways

**Memory Hierarchy**: Registers > Shared > L2 > HBM (1000× speed difference)
**Kernel Fusion**: Reduce HBM accesses by combining operations
**Tensor Cores**: 10× speedup for FP16 matrix multiply
**Flash Attention**: 2-4× faster attention with O(n) memory

---

## Further Reading

- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **Flash Attention Paper**: https://arxiv.org/abs/2205.14135
- **CUTLASS Library**: High-performance GEMM templates
- **llama.cpp CUDA backend**: `ggml-cuda.cu` source code

---

**Status**: Complete
