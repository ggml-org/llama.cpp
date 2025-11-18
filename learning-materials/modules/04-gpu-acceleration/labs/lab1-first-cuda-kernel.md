# Lab 1: First CUDA Kernel for LLM Operations

**Duration:** 2 hours | **Difficulty:** Intermediate

## Learning Objectives

- Write and compile your first CUDA kernel
- Implement RoPE (Rotary Position Embedding) on GPU
- Profile kernel performance
- Optimize memory access patterns

## Prerequisites

- CUDA Toolkit installed (11.0+)
- NVIDIA GPU (Compute Capability 6.0+)
- Basic C/C++ knowledge
- Completed Lesson 1 (GPU Computing Fundamentals)

## Lab Overview

You'll implement the RoPE operation from scratch, starting with a naive version and progressively optimizing it to match production-level performance.

---

## Part 1: Setup and Verification (15 min)

### Step 1: Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU
nvidia-smi

# Expected output: GPU info, driver version, CUDA version
```

### Step 2: Test Basic CUDA Program

Create `test_cuda.cu`:

```cuda
#include <stdio.h>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compile and run:
```bash
nvcc -o test_cuda test_cuda.cu
./test_cuda

# Expected: 8 "Hello" messages from 2 blocks × 4 threads
```

---

## Part 2: Implement Naive RoPE Kernel (30 min)

### Background: What is RoPE?

Rotary Position Embedding encodes position information by rotating embedding vectors:

```
For position p, dimension pair (i, i+1):
  x_new[i]   = x[i]   * cos(θ) - x[i+1] * sin(θ)
  x_new[i+1] = x[i] * sin(θ) + x[i+1] * cos(θ)

Where: θ = p / (10000^(2i / d))
```

### Step 1: CPU Reference Implementation

Create `rope_cpu.cpp`:

```cpp
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void rope_cpu(
    const float* x,
    float* y,
    int n_dims,
    int n_tokens,
    const int* positions
) {
    for (int t = 0; t < n_tokens; t++) {
        int pos = positions[t];

        for (int i = 0; i < n_dims; i += 2) {
            float theta_base = powf(10000.0f, -(float)i / n_dims);
            float theta = pos * theta_base;

            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            int idx = t * n_dims + i;
            float x0 = x[idx];
            float x1 = x[idx + 1];

            y[idx]     = x0 * cos_theta - x1 * sin_theta;
            y[idx + 1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

int main() {
    const int n_dims = 128;
    const int n_tokens = 512;

    float* x = (float*)malloc(n_tokens * n_dims * sizeof(float));
    float* y = (float*)malloc(n_tokens * n_dims * sizeof(float));
    int* positions = (int*)malloc(n_tokens * sizeof(int));

    // Initialize
    for (int i = 0; i < n_tokens * n_dims; i++) {
        x[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < n_tokens; i++) {
        positions[i] = i;
    }

    // Run
    rope_cpu(x, y, n_dims, n_tokens, positions);

    printf("CPU RoPE completed. y[0] = %.6f\n", y[0]);

    free(x); free(y); free(positions);
    return 0;
}
```

Compile and run:
```bash
g++ -o rope_cpu rope_cpu.cpp -lm
./rope_cpu
```

### Step 2: Naive GPU Kernel

Create `rope_gpu_naive.cu`:

```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void rope_kernel_naive(
    const float* x,
    float* y,
    int n_dims,
    int n_tokens,
    const int* positions
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];

    for (int i = 0; i < n_dims; i += 2) {
        float theta_base = powf(10000.0f, -(float)i / n_dims);
        float theta = pos * theta_base;

        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        int idx = token_idx * n_dims + i;
        float x0 = x[idx];
        float x1 = x[idx + 1];

        y[idx]     = x0 * cos_theta - x1 * sin_theta;
        y[idx + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

int main() {
    const int n_dims = 128;
    const int n_tokens = 512;
    const size_t bytes_x = n_tokens * n_dims * sizeof(float);
    const size_t bytes_pos = n_tokens * sizeof(int);

    // Allocate host memory
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_x);
    int* h_pos = (int*)malloc(bytes_pos);

    // Initialize
    for (int i = 0; i < n_tokens * n_dims; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < n_tokens; i++) {
        h_pos[i] = i;
    }

    // Allocate device memory
    float *d_x, *d_y;
    int *d_pos;
    CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_x));
    CUDA_CHECK(cudaMalloc(&d_pos, bytes_pos));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, bytes_pos, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (n_tokens + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    rope_kernel_naive<<<blocks, threads>>>(d_x, d_y, n_dims, n_tokens, d_pos);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes_x, cudaMemcpyDeviceToHost));

    printf("GPU RoPE (naive) completed in %.3f ms\\n", ms);
    printf("y[0] = %.6f\\n", h_y[0]);

    // Cleanup
    free(h_x); free(h_y); free(h_pos);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_pos);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
```

Compile and run:
```bash
nvcc -o rope_gpu_naive rope_gpu_naive.cu -arch=sm_80
./rope_gpu_naive
```

**Expected output:**
```
GPU RoPE (naive) completed in ~0.5 ms
y[0] = (some value)
```

---

## Part 3: Optimize the Kernel (45 min)

### Optimization 1: Process Pairs of Elements

**Problem:** Naive version processes tokens, not dimension pairs.

**Solution:** Process dimension pairs in parallel.

```cuda
__global__ void rope_kernel_opt1(
    const float* x,
    float* y,
    int n_dims,
    int n_tokens,
    const int* positions
) {
    // Each thread processes one dimension pair of one token
    int token_idx = blockIdx.x;
    int dim_pair = threadIdx.x * 2;  // Process 2 elements

    if (token_idx >= n_tokens || dim_pair >= n_dims) return;

    int pos = positions[token_idx];

    float theta_base = powf(10000.0f, -(float)dim_pair / n_dims);
    float theta = pos * theta_base;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    int idx = token_idx * n_dims + dim_pair;
    float x0 = x[idx];
    float x1 = x[idx + 1];

    y[idx]     = x0 * cos_theta - x1 * sin_theta;
    y[idx + 1] = x0 * sin_theta + x1 * cos_theta;
}

// Launch: rope_kernel_opt1<<<n_tokens, n_dims/2>>>(...);
```

### Optimization 2: Use Fast Math Intrinsics

```cuda
__global__ void rope_kernel_opt2(
    const float* x,
    float* y,
    int n_dims,
    int n_tokens,
    const int* positions
) {
    int token_idx = blockIdx.x;
    int dim_pair = threadIdx.x * 2;

    if (token_idx >= n_tokens || dim_pair >= n_dims) return;

    int pos = positions[token_idx];

    // Use fast intrinsics
    float theta_base = __powf(10000.0f, -(float)dim_pair / n_dims);
    float theta = pos * theta_base;

    float cos_theta, sin_theta;
    __sincosf(theta, &sin_theta, &cos_theta);  // Compute both at once!

    int idx = token_idx * n_dims + dim_pair;
    float x0 = x[idx];
    float x1 = x[idx + 1];

    y[idx]     = x0 * cos_theta - x1 * sin_theta;
    y[idx + 1] = x0 * sin_theta + x1 * cos_theta;
}
```

### Optimization 3: Coalesced Memory Access

```cuda
__global__ void rope_kernel_opt3(
    const float* x,
    float* y,
    int n_dims,
    int n_tokens,
    const int* positions
) {
    // Rearrange indexing for coalesced access
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = global_idx / (n_dims / 2);
    int dim_pair = (global_idx % (n_dims / 2)) * 2;

    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];

    float theta_base = __powf(10000.0f, -(float)dim_pair / n_dims);
    float theta = pos * theta_base;

    float cos_theta, sin_theta;
    __sincosf(theta, &sin_theta, &cos_theta);

    int idx = token_idx * n_dims + dim_pair;

    // Vectorized load (if possible)
    float2 val = *((float2*)&x[idx]);
    float x0 = val.x;
    float x1 = val.y;

    // Vectorized store
    float2 out;
    out.x = x0 * cos_theta - x1 * sin_theta;
    out.y = x0 * sin_theta + x1 * cos_theta;
    *((float2*)&y[idx]) = out;
}

// Launch: rope_kernel_opt3<<<blocks, threads>>>(...);
// Where blocks and threads are calculated for total elements
```

**Task:** Implement all three optimizations and measure speedup.

---

## Part 4: Profile and Compare (30 min)

### Step 1: Comprehensive Timing

Create a benchmark that runs all versions:

```cuda
// Add timing for each version
printf("=== RoPE Kernel Performance ===\\n");

// Naive
run_and_time("Naive", rope_kernel_naive, ...);

// Opt1: Pairs
run_and_time("Opt1 (Pairs)", rope_kernel_opt1, ...);

// Opt2: Fast math
run_and_time("Opt2 (Fast Math)", rope_kernel_opt2, ...);

// Opt3: Coalesced
run_and_time("Opt3 (Coalesced)", rope_kernel_opt3, ...);
```

### Step 2: Profile with Nsight Compute

```bash
# Profile naive version
ncu --set full --kernel-name rope_kernel_naive ./rope_gpu

# Profile optimized version
ncu --set full --kernel-name rope_kernel_opt3 ./rope_gpu

# Compare metrics:
# - sm__throughput (SM utilization)
# - dram__throughput (memory bandwidth)
# - Occupancy
# - Warp efficiency
```

### Step 3: Analyze Results

**Questions to answer:**
1. What is the speedup of each optimization?
2. Which optimization provides the biggest gain?
3. What is the bottleneck (memory or compute)?
4. What is the achieved occupancy?

**Expected speedups:**
- Opt1 vs Naive: 1.5-2x
- Opt2 vs Opt1: 1.3-1.5x
- Opt3 vs Opt2: 1.2-1.4x
- Total: 2.5-4x faster than naive!

---

## Part 5: Challenge (optional, 30 min)

### Challenge 1: Add FP16 Support

Modify the kernel to use `half` precision:
- Requires compute capability 6.0+
- Should be ~2x faster (higher throughput)
- Check accuracy loss

### Challenge 2: Implement llama.cpp Style

Study `ggml/src/ggml-cuda/rope.cu` and implement:
- Support for different RoPE variants (NeoX, GLM)
- Frequency scaling
- Context extension (yarn, etc.)

---

## Deliverables

1. **Code:**
   - All 4 kernel versions (naive + 3 optimizations)
   - Timing and benchmarking code
   - Correctness verification (compare with CPU)

2. **Report:**
   - Performance comparison table
   - Nsight Compute screenshots
   - Analysis of bottlenecks
   - Speedup breakdown

3. **Reflection:**
   - What did you learn about GPU optimization?
   - Which optimization was most effective and why?
   - How does this compare to llama.cpp implementation?

---

## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [llama.cpp rope.cu source](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/rope.cu)

---

**Next Lab:** [Lab 2: GPU Memory Profiling](lab2-gpu-memory-profiling.md)
