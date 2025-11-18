# GPU Computing Fundamentals for LLM Inference

**Module 4, Lesson 1** | **Duration: 3 hours** | **Level: Advanced**

## Table of Contents
1. [GPU vs CPU Architecture](#gpu-vs-cpu-architecture)
2. [CUDA Programming Model](#cuda-programming-model)
3. [GPU Memory Hierarchy](#gpu-memory-hierarchy)
4. [CUDA for LLM Operations](#cuda-for-llm-operations)
5. [Performance Considerations](#performance-considerations)
6. [Hands-On Examples](#hands-on-examples)

---

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand fundamental differences between GPU and CPU architecture
- ✅ Master CUDA programming basics (threads, blocks, grids)
- ✅ Comprehend GPU memory hierarchy and its impact on performance
- ✅ Identify which LLM operations benefit from GPU acceleration
- ✅ Write basic CUDA kernels for tensor operations
- ✅ Analyze GPU performance metrics (occupancy, bandwidth)

---

## GPU vs CPU Architecture

### Why GPUs for LLM Inference?

**CPU Characteristics:**
- **Few powerful cores** (8-64 cores typical)
- **High clock speeds** (3-5 GHz)
- **Large caches** (L1: 64KB, L2: 512KB-2MB, L3: 8-64MB)
- **Optimized for latency** - minimize time for single operation
- **Complex control logic** - branch prediction, out-of-order execution
- **Good for:** Sequential code, complex logic, irregular memory access

**GPU Characteristics:**
- **Thousands of simple cores** (2,560-16,384 CUDA cores)
- **Lower clock speeds** (1-2 GHz)
- **Smaller per-core caches** but massive memory bandwidth
- **Optimized for throughput** - maximize operations per second
- **Simple control logic** - SIMT (Single Instruction, Multiple Threads)
- **Good for:** Parallel operations, regular memory patterns, high compute density

### LLM Inference is Embarrassingly Parallel

**Matrix Multiplication (GEMM):**
```
C[i,j] = Σ(k) A[i,k] * B[k,j]
```

Each element C[i,j] can be computed independently!
- For a 4096×4096 matrix: **16.8 million independent computations**
- GPU with 10,240 cores: ~1,640 elements per core
- Massive parallelism advantage

**Real-World Example:**
```
LLaMA-7B forward pass:
- CPU (AMD EPYC 7763): ~2,000 ms
- GPU (A100 80GB):      ~15 ms   (133x speedup!)
- GPU (H100 SXM):       ~8 ms    (250x speedup!)
```

### NVIDIA GPU Architecture Evolution

```
┌─────────────────────────────────────────────────────────┐
│ NVIDIA GPU Architecture Timeline (LLM Era)              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ Pascal (2016) - GTX 1080, P100                          │
│   • FP16 support (Tensor Ops on P100)                   │
│   • 10.6 TFLOPS (FP32), 21.2 TFLOPS (FP16)             │
│   • Memory: 16GB HBM2, 732 GB/s                         │
│                                                          │
│ Volta (2017) - V100                                     │
│   • First Tensor Cores (mixed precision)                │
│   • 15.7 TFLOPS (FP32), 125 TFLOPS (Tensor)            │
│   • Memory: 32GB HBM2, 900 GB/s                         │
│                                                          │
│ Turing (2018) - RTX 2080, T4                            │
│   • 2nd Gen Tensor Cores (INT8, INT4 support)           │
│   • 14.2 TFLOPS (FP32), 57.5 TFLOPS (FP16)             │
│   • Memory: 16GB GDDR6, 320 GB/s                        │
│                                                          │
│ Ampere (2020) - A100, RTX 3090                          │
│   • 3rd Gen Tensor Cores (TF32, BF16, FP64)             │
│   • 19.5 TFLOPS (FP32), 312 TFLOPS (Tensor)            │
│   • Memory: 80GB HBM2e, 2,039 GB/s                      │
│   • Multi-Instance GPU (MIG)                            │
│                                                          │
│ Hopper (2022) - H100                                    │
│   • 4th Gen Tensor Cores (FP8, Transformer Engine)      │
│   • 67 TFLOPS (FP32), 3,958 TFLOPS (FP8 Tensor)        │
│   • Memory: 80GB HBM3, 3,350 GB/s                       │
│   • TMA (Tensor Memory Accelerator)                     │
│                                                          │
│ Blackwell (2024) - B100, B200                           │
│   • 5th Gen Tensor Cores (FP4, FP6 support)             │
│   • 10 PFLOPS (FP4 Tensor), 20 PFLOPS (FP4 sparse)     │
│   • Memory: 192GB HBM3e, 8,000 GB/s                     │
│   • RAS (Reliability, Availability, Serviceability)     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Takeaway:** Each generation brings:
1. More compute power (especially tensor operations)
2. Higher memory bandwidth
3. Better support for reduced precision (FP16 → INT8 → FP8 → FP4)

---

## CUDA Programming Model

### Thread Hierarchy

CUDA organizes parallel execution in a **3-level hierarchy**:

```
Grid (entire kernel launch)
  └── Block 0, Block 1, ..., Block N
        └── Thread 0, Thread 1, ..., Thread M
```

**Simplified Analogy:**
- **Grid** = Entire factory
- **Block** = Team of workers who can collaborate
- **Thread** = Individual worker

### CUDA Kernel Structure

```cuda
// Simple vector addition kernel
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch from host
int N = 1000000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

**Index Calculation Explained:**
```
blockIdx.x = 0, threadIdx.x = 0   → idx = 0
blockIdx.x = 0, threadIdx.x = 1   → idx = 1
blockIdx.x = 0, threadIdx.x = 255 → idx = 255
blockIdx.x = 1, threadIdx.x = 0   → idx = 256
blockIdx.x = 1, threadIdx.x = 1   → idx = 257
...
```

### Grid, Block, Thread Dimensions

CUDA supports **1D, 2D, and 3D** indexing:

```cuda
// 1D Grid (most common for vectors)
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D Grid (common for matrices)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// 3D Grid (tensors, volumes)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;
```

**Matrix Multiplication Example:**
```cuda
__global__ void matMul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Output row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output col

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch configuration for 4096×4096 matrix
dim3 threadsPerBlock(16, 16);  // 256 threads per block
dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
matMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
```

---

## GPU Memory Hierarchy

### Memory Types and Characteristics

```
┌──────────────────────────────────────────────────────────┐
│                   GPU Memory Hierarchy                    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Registers (per-thread)                                  │
│    • Size: ~65,536 per SM                                │
│    • Latency: 0 cycles                                   │
│    • Bandwidth: Highest                                  │
│    • Scope: Thread-local                                 │
│                                                           │
│  Shared Memory (per-block)                               │
│    • Size: 48-164 KB per SM                              │
│    • Latency: 1-32 cycles                                │
│    • Bandwidth: ~19 TB/s (A100)                          │
│    • Scope: Block-local, programmable cache              │
│                                                           │
│  L1 Cache (per-SM)                                       │
│    • Size: 128 KB per SM (configurable with shared mem)  │
│    • Latency: 30-50 cycles                               │
│    • Scope: Automatic, hardware-managed                  │
│                                                           │
│  L2 Cache (global)                                       │
│    • Size: 6-80 MB (A100: 40 MB)                         │
│    • Latency: 200-300 cycles                             │
│    • Scope: All SMs, shared                              │
│                                                           │
│  Global Memory (HBM/GDDR)                                │
│    • Size: 16-80 GB (H100: 80 GB)                        │
│    • Latency: 300-600 cycles                             │
│    • Bandwidth: 2-3 TB/s (A100: 2.0 TB/s)                │
│    • Scope: All threads, persistent                      │
│                                                           │
│  Host Memory (CPU RAM)                                   │
│    • Size: 64-2048 GB                                    │
│    • Latency: 10,000+ cycles (via PCIe)                  │
│    • Bandwidth: 32-64 GB/s (PCIe 4.0 x16)                │
│    • Scope: CPU and GPU (slow transfer)                  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Memory Access Patterns Matter:**

```cuda
// GOOD: Coalesced access (threads access consecutive addresses)
__global__ void coalescedRead(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // Thread 0→data[0], Thread 1→data[1], etc.
    // Process value...
}

// BAD: Strided access (performance penalty)
__global__ void stridedRead(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx * stride];  // Non-consecutive accesses
    // Process value...
}

// Performance difference: up to 10x slower for strided!
```

### Using Shared Memory for Optimization

**Naive Matrix Multiplication:**
- Each element loads A and B from global memory K times
- Total global memory accesses: 2 × M × N × K

**Optimized with Shared Memory:**
```cuda
__global__ void matMulShared(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from global to shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Wait for all threads to load

        // Compute using shared memory (fast!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();  // Wait before loading next tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Performance Improvement:**
- Reduces global memory accesses by ~TILE_SIZE factor
- For TILE_SIZE=32: ~32x fewer global memory transactions
- Real speedup: 3-5x depending on matrix size

---

## CUDA for LLM Operations

### Critical LLM Operations on GPU

**1. Matrix Multiplication (GEMM)**
- **Where:** Feed-forward layers, attention projections
- **Portion:** 80-90% of compute
- **GPU Advantage:** Massive parallelism, tensor cores
- **llama.cpp:** Uses cuBLAS, custom kernels for quantized formats

**2. Attention Mechanism**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```
- **Challenges:**
  - Memory-bound (O(n²) KV cache access)
  - Sequential softmax operation
- **GPU Optimization:** Flash Attention (tiling, fused kernels)
- **llama.cpp:** `fattn.cu` - multiple implementations (MMA, WMMA, vectorized)

**3. Element-wise Operations**
- **Examples:** ReLU, GELU, layer norm, RoPE
- **GPU Advantage:** Trivially parallel
- **llama.cpp:** Fused kernels to reduce memory transfers

**4. Quantization/Dequantization**
- **On-the-fly dequantization** during inference
- **GPU kernels:** Convert INT4/INT8 → FP16 during GEMM
- **llama.cpp:** Specialized kernels for each quant format (Q4_0, Q4_K_M, etc.)

### llama.cpp CUDA Backend Architecture

```
┌───────────────────────────────────────────────────────┐
│         llama.cpp CUDA Backend Structure              │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ggml-cuda.cu (main backend interface)                │
│    ├── Device initialization & management             │
│    ├── Memory allocation (cudaMalloc, pools)          │
│    ├── Backend API implementation                     │
│    └── Kernel dispatch                                │
│                                                        │
│  Specialized Kernels:                                 │
│    ├── fattn.cu - Flash Attention                     │
│    │     ├── fattn-tile.cu    (tiled implementation)  │
│    │     ├── fattn-vec.cu     (vectorized, quantized) │
│    │     ├── fattn-mma-f16.cu (MMA tensor cores)      │
│    │     └── fattn-wmma-f16.cu (WMMA tensor cores)    │
│    │                                                   │
│    ├── mmq.cu - Quantized Matrix Multiply             │
│    │     └── template-instances/mmq-instance-*.cu     │
│    │           (Q4_0, Q4_K, Q5_K, Q8_0, IQ*, etc.)    │
│    │                                                   │
│    ├── mmf.cu - FP16/FP32 Matrix Multiply             │
│    ├── rope.cu - Rotary Position Embedding            │
│    ├── norm.cu - RMS Norm, Layer Norm                 │
│    ├── softmax.cu - Softmax operation                 │
│    ├── quantize.cu - On-GPU quantization              │
│    └── cpy.cu - Memory copy/transpose                 │
│                                                        │
│  Utilities:                                            │
│    ├── common.cuh - CUDA helpers, macros              │
│    ├── dequantize.cuh - Dequant functions             │
│    └── vecdotq.cuh - Vectorized dot products          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**File Count:** 50+ CUDA kernels, each highly optimized!

---

## Performance Considerations

### 1. Occupancy

**Occupancy** = (Active Warps) / (Maximum Possible Warps)

**Factors Limiting Occupancy:**
1. **Registers per thread**
   - A100: 65,536 registers per SM
   - If each thread uses 64 registers → max 1,024 threads per SM

2. **Shared memory per block**
   - A100: 164 KB max per SM
   - If each block uses 48 KB → max 3 blocks per SM

3. **Threads per block**
   - Max: 1,024 threads per block
   - Too few threads → low occupancy
   - Too many → resource contention

**Optimal Configuration:**
```cuda
// Good occupancy (A100)
dim3 block(256);  // 256 threads per block
dim3 grid((N + 255) / 256);

// Achieves ~75-100% occupancy if:
// - Register usage < 64 per thread
// - Shared memory < 48 KB per block
```

**Check with Nsight Compute:**
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak ./llama-cli -m model.gguf
# Target: >50% for compute-bound, >25% for memory-bound
```

### 2. Memory Bandwidth Utilization

**Peak Bandwidth:**
- A100: 2,039 GB/s (HBM2e)
- H100: 3,350 GB/s (HBM3)

**Effective Bandwidth:**
```
Effective BW = (Bytes Transferred) / (Kernel Time)
Efficiency = (Effective BW) / (Peak BW) × 100%
```

**Example - Matrix-Vector Multiply:**
```
M × K matrix × K vector = M outputs
Bytes Read:  M × K × 2 (FP16) + K × 2 = 2MK + 2K bytes
Bytes Write: M × 2 = 2M bytes
Total:       2MK + 2K + 2M ≈ 2MK bytes (for large M, K)

Arithmetic Intensity = (2MK FLOPs) / (2MK bytes) = 1 FLOP/byte

This is VERY memory-bound! (A100 can do ~312 TFLOPS but only 2 TB/s)
```

### 3. Kernel Fusion

**Problem:** Launching multiple small kernels has overhead
```cuda
// Bad: Three kernel launches
layerNorm<<<grid, block>>>(x, gamma, beta, out1);
relu<<<grid, block>>>(out1, out2);
dropout<<<grid, block>>>(out2, out3, p);
// 3 global memory read/writes!
```

**Solution:** Fuse into single kernel
```cuda
// Good: Fused kernel
__global__ void layerNormReluDropout(float* x, float* gamma, float* beta,
                                      float* out, float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Layer norm
    float val = (x[idx] - mean) / std * gamma[idx] + beta[idx];

    // ReLU
    val = fmaxf(0.0f, val);

    // Dropout
    if (curand_uniform(&state) < p) val = 0.0f;
    else val /= (1.0f - p);

    out[idx] = val;
}
// Only 1 global memory read/write!
```

**llama.cpp Example:**
```cuda
// ggml-cuda/norm.cu - Fused RMS Norm + scaling
template<int block_size>
__global__ void rms_norm_f32(
    const float * x, float * dst, const int ncols, const float eps) {
    // Fuses: mean_square → rsqrt → multiply
    // Avoids intermediate global memory writes
}
```

---

## Hands-On Examples

### Example 1: Vector Addition (CUDA Basics)

```cuda
// vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1000000;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %.1f\n", i, h_C[i]);
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

**Compile and run:**
```bash
nvcc -o vector_add vector_add.cu
./vector_add
```

### Example 2: Timing GPU Kernels

```cuda
// timing_example.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Your setup code here...

    // Record start
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Bandwidth: %.2f GB/s\n",
           (3 * N * sizeof(float)) / (milliseconds / 1000.0) / 1e9);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

### Example 3: Checking GPU Properties

```cuda
// gpu_info.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Found %d CUDA device(s)\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n",
               prop.totalGlobalMem / 1e9);
        printf("  Shared Memory per Block: %zu KB\n",
               prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dim: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  Memory Clock Rate: %.2f GHz\n",
               prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    }

    return 0;
}
```

---

## Key Takeaways

1. **GPUs excel at parallel workloads** - LLM inference is highly parallelizable
2. **Memory hierarchy matters** - Use shared memory and coalesced access patterns
3. **CUDA threads are lightweight** - Launch thousands to millions for good occupancy
4. **Tensor cores are critical** - 10-20x speedup for mixed-precision matrix ops
5. **llama.cpp has deep CUDA integration** - 50+ optimized kernels for different operations

---

## Interview Questions

### Conceptual (5 questions)

1. **Q:** Why are GPUs faster than CPUs for LLM inference?
   **A:** GPUs have thousands of simple cores optimized for parallel throughput, whereas CPUs have few complex cores optimized for latency. LLM inference is dominated by matrix operations that are embarrassingly parallel (each output element is independent), making GPUs 100-1000x faster.

2. **Q:** What is the CUDA thread hierarchy?
   **A:** Grid → Blocks → Threads. A grid contains multiple blocks, each block contains multiple threads. Threads within a block can synchronize and share memory, but threads in different blocks cannot directly communicate.

3. **Q:** Explain memory coalescing in CUDA.
   **A:** When consecutive threads access consecutive memory addresses, the GPU can combine multiple memory requests into a single transaction. Uncoalesced access (e.g., strided patterns) requires multiple transactions, reducing bandwidth efficiency by 10x or more.

4. **Q:** What are tensor cores and why are they important for LLMs?
   **A:** Specialized hardware units for matrix multiply-accumulate operations (D = A×B+C) in mixed precision (FP16, BF16, INT8, FP8). They provide 10-20x higher throughput than CUDA cores for these operations, which dominate LLM inference.

5. **Q:** What limits GPU occupancy?
   **A:** Three main factors: (1) registers per thread, (2) shared memory per block, and (3) threads per block. If any resource is exhausted, additional warps cannot be scheduled, reducing occupancy and leaving compute units idle.

### Practical (5 questions)

6. **Q:** How do you calculate the global thread index in a 1D CUDA grid?
   **A:** `int idx = blockIdx.x * blockDim.x + threadIdx.x;`

7. **Q:** What's the difference between `cudaMalloc` and `cudaMallocManaged`?
   **A:** `cudaMalloc` allocates device memory only (requires explicit `cudaMemcpy`). `cudaMallocManaged` allocates unified memory accessible from both CPU and GPU with automatic migration, simplifying code but potentially slower.

8. **Q:** How do you time a CUDA kernel accurately?
   **A:** Use CUDA events:
   ```cuda
   cudaEvent_t start, stop;
   cudaEventCreate(&start); cudaEventCreate(&stop);
   cudaEventRecord(start);
   kernel<<<grid, block>>>(...);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float ms; cudaEventElapsedTime(&ms, start, stop);
   ```

9. **Q:** Why is shared memory faster than global memory?
   **A:** Shared memory is on-chip SRAM with ~1-2 cycle latency and 19 TB/s bandwidth (A100), while global memory is off-chip DRAM with 300-600 cycle latency and 2 TB/s bandwidth. Shared memory is ~100x lower latency, ~10x higher bandwidth.

10. **Q:** What is warp divergence and why does it hurt performance?
    **A:** When threads in a warp take different execution paths (e.g., if-else branches), the GPU must serialize execution, reducing throughput. Example: if half of threads execute `if` and half execute `else`, performance is halved.

---

## Additional Resources

### Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Architecture Whitepaper (Hopper)](https://resources.nvidia.com/en-us-hopper-architecture)

### Tools
- **Nsight Compute** - Kernel profiling
- **Nsight Systems** - System-wide timeline profiling
- **nvprof** - Legacy profiler (still useful)

### Books
- *Programming Massively Parallel Processors* by Kirk & Hwu
- *CUDA by Example* by Sanders & Kandrot

### Papers
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Transformer architecture)
- [FlashAttention](https://arxiv.org/abs/2205.14135) (GPU-optimized attention)

---

**Next Lesson:** [02-cuda-backend-implementation.md](02-cuda-backend-implementation.md) - Deep dive into llama.cpp CUDA backend

**Related Labs:**
- [Lab 1: First CUDA Kernel](../labs/lab1-first-cuda-kernel.md)
- [Lab 2: GPU Memory Profiling](../labs/lab2-gpu-memory-profiling.md)
