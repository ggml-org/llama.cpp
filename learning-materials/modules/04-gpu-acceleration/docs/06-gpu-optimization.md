# GPU Performance Optimization and Profiling

**Module 4, Lesson 6** | **Duration: 2 hours** | **Level: Expert**

## Table of Contents
1. [Profiling Tools and Techniques](#profiling-tools-and-techniques)
2. [Kernel Optimization Strategies](#kernel-optimization-strategies)
3. [Occupancy Optimization](#occupancy-optimization)
4. [Memory Bandwidth Optimization](#memory-bandwidth-optimization)
5. [Nsight Profiler Deep Dive](#nsight-profiler-deep-dive)
6. [Real-World Optimization Examples](#real-world-optimization-examples)

---

## Learning Objectives

By the end of this lesson, you will:
- ✅ Profile GPU kernels using Nsight Compute and Nsight Systems
- ✅ Optimize kernel occupancy and resource usage
- ✅ Maximize memory bandwidth utilization
- ✅ Identify and fix performance bottlenecks
- ✅ Apply optimization patterns to LLM kernels
- ✅ Measure and validate performance improvements

---

## Profiling Tools and Techniques

### NVIDIA Profiling Toolchain

**Three Main Tools:**
```
1. Nsight Compute (ncu)
   - Kernel-level profiling
   - Detailed metrics (occupancy, memory bandwidth, warp efficiency)
   - Roofline analysis
   - Bottleneck identification

2. Nsight Systems (nsys)
   - System-wide timeline
   - CPU-GPU interaction
   - Multi-GPU communication
   - API calls and synchronization

3. nvprof (legacy, deprecated)
   - Simple command-line profiler
   - Being replaced by ncu/nsys
```

### Nsight Compute Basics

**Profile a single kernel:**
```bash
# Profile specific kernel
ncu --kernel-name "flash_attn" ./llama-cli -m model.gguf -p "test"

# All metrics for kernel
ncu --set full --kernel-name "mul_mat_q4_K" ./llama-cli

# Specific metrics
ncu --metrics sm__throughput,dram__throughput,l1tex__throughput \
    --kernel-name "rope_f32" ./llama-cli

# Export report
ncu --export profile.ncu-rep ./llama-cli
```

**Key Metrics:**
```
Metric                          Meaning
─────────────────────────────────────────────────────────
sm__throughput                  SM utilization (%)
dram__throughput                Memory bandwidth (%)
l1tex__throughput               L1/Texture cache (%)
sm__warps_active                Active warps per cycle
sm__pipe_tensor_cycles_active   Tensor core usage (%)
smsp__inst_executed_pipe_tensor Tensor instructions
gpu__time_duration              Kernel execution time
```

### Nsight Systems Timeline

**Capture system trace:**
```bash
# Full system profile
nsys profile --trace=cuda,nvtx,osrt ./llama-cli -m model.gguf -p "test"

# Output: report.nsys-rep (open in Nsight Systems GUI)

# Focus on GPU activity
nsys profile --trace=cuda --cuda-memory-usage=true ./llama-cli

# Multi-GPU tracing
nsys profile --trace=cuda,nvtx,mpi ./llama-cli
```

**Timeline Visualization:**
```
CPU Timeline:
  [main thread] ───[load model]─────[inference]─────[cleanup]────
  [worker 1]    ────────────────────[process]───────────────────
  [worker 2]    ────────────────────[process]───────────────────

GPU Timeline:
  [GPU 0] ──[H2D]──[kernel]──[kernel]──[kernel]──[D2H]──[kernel]──
  [GPU 1] ──────────[kernel]──[kernel]──[kernel]──────────────────

Insights:
  • Gap between kernels → CPU bottleneck or synchronization
  • H2D/D2H transfers → Consider pinned memory
  • Short kernels → Launch overhead, try kernel fusion
```

### Custom NVTX Markers

**Annotate code for profiling:**
```cpp
#include <nvtx3/nvToolsExt.h>

void run_inference() {
    nvtxRangePush("Inference");  // Start marker

    nvtxRangePush("Load Model");
    load_model("model.gguf");
    nvtxRangePop();

    nvtxRangePush("Forward Pass");
    for (int i = 0; i < n_layers; i++) {
        nvtxRangePush("Attention");
        attention_layer(i);
        nvtxRangePop();

        nvtxRangePush("FFN");
        ffn_layer(i);
        nvtxRangePop();
    }
    nvtxRangePop();

    nvtxRangePop();  // End inference
}

// Now visible in Nsight Systems timeline with color-coded regions!
```

---

## Kernel Optimization Strategies

### 1. Minimize Divergence

**Problem: Warp Divergence**
```cuda
// BAD: Different code paths
__global__ void process(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        // Half of threads execute this
        data[idx] = expensive_computation_A(data[idx]);
    } else {
        // Other half execute this (serialized!)
        data[idx] = expensive_computation_B(data[idx]);
    }
}

// Performance: 50% of potential (one branch at a time)
```

**Solution: Predication**
```cuda
// GOOD: Same path, conditional assignment
__global__ void process(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float a = expensive_computation_A(data[idx]);
    float b = expensive_computation_B(data[idx]);

    data[idx] = (idx % 2 == 0) ? a : b;  // Predicated select (fast!)
}

// Performance: 100% (all threads follow same path)
```

**When divergence is unavoidable:**
- Keep divergent branches short
- Align divergence with warp boundaries (32 threads)
- Consider splitting into separate kernels

### 2. Coalesce Memory Access

**Problem: Uncoalesced Access**
```cuda
// BAD: Strided access
__global__ void transpose_bad(float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Writing: consecutive threads → strided addresses (BAD!)
        out[x * height + y] = in[y * width + x];
    }
}

// Memory transactions: width (many small transactions)
```

**Solution: Use Shared Memory**
```cuda
// GOOD: Tiled transpose with shared memory
__global__ void transpose_good(float* in, float* out, int width, int height) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Coalesced write
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Speedup: 5-10x!
```

### 3. Kernel Fusion

**Problem: Multiple Kernel Launches**
```cuda
// BAD: Three kernel launches
__global__ void add(float* a, float* b, float* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

__global__ void relu(float* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = fmaxf(0.0f, c[idx]);
}

__global__ void scale(float* c, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] *= s;
}

// Launch overhead + 3x global memory traffic
add<<<grid, block>>>(a, b, c);
relu<<<grid, block>>>(c);
scale<<<grid, block>>>(c, 2.0f);
```

**Solution: Fused Kernel**
```cuda
// GOOD: Single kernel
__global__ void add_relu_scale(float* a, float* b, float* c, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[idx] + b[idx];   // Add
    val = fmaxf(0.0f, val);         // ReLU
    c[idx] = val * s;               // Scale
}

// 1 launch, 1x memory traffic
add_relu_scale<<<grid, block>>>(a, b, c, 2.0f);

// Speedup: 2-3x!
```

### 4. Shared Memory Optimization

**Problem: Bank Conflicts**
```cuda
// BAD: All threads access same bank
__shared__ float shared[32][32];

__global__ void kernel() {
    int tid = threadIdx.x;
    // All threads access same column → 32-way bank conflict!
    float val = shared[tid][0];  // Bank conflict!
}

// Performance: 32x slower than ideal
```

**Solution: Padding or Layout Change**
```cuda
// GOOD: Add padding to avoid conflicts
__shared__ float shared[32][33];  // 33 instead of 32

__global__ void kernel() {
    int tid = threadIdx.x;
    float val = shared[tid][0];  // No conflict!
}

// Alternative: Change access pattern
__shared__ float shared[32][32];

__global__ void kernel() {
    int tid = threadIdx.x;
    float val = shared[0][tid];  // Access row instead → coalesced
}
```

---

## Occupancy Optimization

### Understanding Occupancy

**Occupancy = Active Warps / Max Possible Warps**

```
A100 SM:
  Max warps per SM:        64
  Max threads per SM:      2048 (64 warps × 32 threads)
  Max blocks per SM:       32
  Registers per SM:        65,536
  Shared memory per SM:    164 KB
```

**Occupancy Calculation:**
```
Kernel config: 256 threads/block = 8 warps/block

Limited by threads?
  2048 threads / 256 threads per block = 8 blocks/SM ✓

Limited by blocks?
  32 blocks max ≥ 8 blocks ✓

Limited by registers? (assume 48 registers/thread)
  65,536 registers / (256 threads × 48) = 5.3 blocks/SM ✗ (limits to 5)

Limited by shared memory? (assume 48 KB/block)
  164 KB / 48 KB = 3.4 blocks/SM ✗ (limits to 3!)

Actual: 3 blocks × 8 warps = 24 warps active
Occupancy = 24 / 64 = 37.5%
```

**Check Occupancy:**
```bash
# Nsight Compute
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name "my_kernel" ./app

# Output: "Occupancy: 37.5%"

# CUDA Occupancy Calculator (spreadsheet)
# Or use:
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel,
                                               threads, sharedMem);
```

### Improving Occupancy

**1. Reduce Register Usage**
```cuda
// BAD: Many registers per thread
__global__ void kernel() {
    float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
    // ... uses 48+ registers
}

// GOOD: Fewer registers
__global__ void kernel() {
    float a, b;  // Reuse variables
    // ... uses 16 registers
}

// Or use compiler flag:
nvcc --maxrregcount=32 kernel.cu  // Limit to 32 registers/thread
```

**2. Reduce Shared Memory**
```cuda
// BAD: Large shared memory
__shared__ float tile[64][64];  // 16 KB per block

// GOOD: Smaller tiles or dynamic shared memory
__shared__ float tile[32][32];  // 4 KB per block

// Or:
extern __shared__ float tile[];  // Allocate at launch time
kernel<<<grid, block, sharedMemSize>>>();
```

**3. Adjust Block Size**
```cuda
// Try different block sizes
dim3 block;
for (int b = 128; b <= 1024; b += 128) {
    block = dim3(b);
    // Profile and find best occupancy
}

// Common sweet spots: 128, 256, 512 threads
```

**Occupancy vs Performance:**
```
Occupancy is NOT always the goal!

Example: Memory-bound kernel
  50% occupancy, 95% memory bandwidth → Good!
  (More warps won't help, already saturating memory)

Example: Compute-bound kernel
  50% occupancy, 60% compute → Bad!
  (More warps could increase compute utilization)

Rule: Optimize for bottleneck, not occupancy
```

---

## Memory Bandwidth Optimization

### Roofline Model

**Arithmetic Intensity = FLOPs / Bytes**

```
                    Compute Bound
                    (Performance limited by FLOPs)
                   │
Peak FLOPS ────────┼──────────────
                  ╱│
                 ╱ │
                ╱  │
               ╱   │
              ╱    │ Memory Bound
             ╱     │ (Performance limited by bandwidth)
            ╱      │
───────────────────┼─────────── Arithmetic Intensity
           Ridge Point
```

**Example: Matrix Multiply**
```
Matrix size: M=N=K=4096
FLOPs: 2 × M × N × K = 137 billion FLOPs
Bytes: M×K + K×N + M×N (FP32) = 192 MB
Arithmetic Intensity = 137 GFLOPs / 192 MB = 714 FLOPs/byte

A100:
  Peak Compute: 312 TFLOPS (FP16 Tensor)
  Peak Memory:  2,039 GB/s
  Ridge Point:  312 TFLOPS / 2,039 GB/s = 153 FLOPs/byte

714 > 153 → Compute Bound (good for matmul!)
```

**Example: Vector Add**
```
Vector size: N=1M
FLOPs: 1 × N = 1 million FLOPs
Bytes: 3 × N × 4 (read A, B; write C) = 12 MB
Arithmetic Intensity = 1 MFLOPs / 12 MB = 0.083 FLOPs/byte

0.083 << 153 → Memory Bound (saturate bandwidth!)
```

### Maximizing Bandwidth

**1. Coalesced Access (covered earlier)**

**2. Vectorized Loads/Stores**
```cuda
// BAD: Scalar loads
__global__ void copy(float* dst, const float* src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[idx];  // 4-byte transaction
    }
}

// GOOD: Vectorized loads (float4 = 16 bytes)
__global__ void copy_vectorized(float* dst, const float* src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;

    if (idx4 + 3 < N) {
        float4 val = *((float4*)&src[idx4]);  // 16-byte transaction
        *((float4*)&dst[idx4]) = val;
    }
}

// Bandwidth increase: ~3-4x!
```

**llama.cpp Example:**
```cpp
// From ggml-cuda/dequantize.cuh
template<int qk, int qr>
__global__ void dequantize_block_q4_0(
    const void * vx, dst_t * y, const int64_t k
) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) return;

    const int64_t ib = i/qk;  // Block index

    // Load 2 elements at once (vectorized)
    dfloat2 v;
    dequantize_q4_0(vx, ib, i % qk, v);

    y[i + 0] = v.x;
    y[i + 1] = v.y;
}
```

**3. Asynchronous Copies**
```cuda
// BAD: Synchronous
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_ptr);

// GOOD: Async with streams
cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_ptr);
// Overlap copy with previous kernel!
```

---

## Nsight Profiler Deep Dive

### Section Analysis

**Speed of Light (SOL):**
```
Metric                     Value    Peak     %
─────────────────────────────────────────────
SM Throughput              8.2 T    10.4 T   78.8%
Memory Throughput          1.2 TB/s 2.0 TB/s 60.0%
Compute (FP16) Throughput  124 TF   312 TF   39.7%

Diagnosis: Memory bound (60% mem, 40% compute)
Action: Optimize memory access patterns
```

**Warp State Statistics:**
```
Metric                           Value
────────────────────────────────────
Active Warps per Scheduler       3.2 / 4
Eligible Warps per Scheduler     1.8
No Eligible (Memory Throttle)    45%
No Eligible (Not Selected)       12%

Diagnosis: Memory throttle (45% stalled on memory)
Action: Increase arithmetic intensity or improve bandwidth
```

**Memory Workload Analysis:**
```
Metric                   Value      Peak       %
──────────────────────────────────────────────
L1/TEX Cache Throughput  800 GB/s   1600 GB/s  50%
L2 Cache Throughput      1200 GB/s  2000 GB/s  60%
Device Memory Throughput 1100 GB/s  2039 GB/s  54%

L1 Hit Rate:             65%
L2 Hit Rate:             35%

Diagnosis: Poor cache utilization
Action: Improve data locality, use shared memory
```

### Guided Analysis

**Nsight Compute Guided Roofline:**
```bash
ncu --set roofline --kernel-name "my_kernel" ./app

# Output:
# ┌─────────────────────────────────────┐
# │ Roofline Analysis                   │
# ├─────────────────────────────────────┤
# │ Achieved Performance: 45 TFLOPS     │
# │ Arithmetic Intensity: 2.5 FLOPs/B   │
# │ Bottleneck: Memory Bandwidth        │
# │                                     │
# │ Recommendations:                    │
# │ 1. Use shared memory caching        │
# │ 2. Coalesce memory accesses         │
# │ 3. Consider kernel fusion           │
# └─────────────────────────────────────┘
```

---

## Real-World Optimization Examples

### Example 1: Optimizing RoPE Kernel

**Before:**
```cuda
__global__ void rope_f32_slow(
    const float* x, float* dst, int ne0, const int32_t* pos
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= ne0) return;

    const int p = pos[i / (ne0 / 2)];
    const float theta = powf(10000.0f, -float(i % (ne0/2)) / float(ne0/2));
    const float angle = p * theta;

    if (i < ne0 / 2) {
        const float x0 = x[i];
        const float x1 = x[i + ne0/2];
        dst[i]        = x0 * cosf(angle) - x1 * sinf(angle);
        dst[i + ne0/2] = x0 * sinf(angle) + x1 * cosf(angle);
    }
}

// Profile: 45% occupancy, many divergent branches
```

**After:**
```cuda
__global__ void rope_f32_fast(
    const float* x, float* dst, int ne0, const int32_t* pos
) {
    const int i0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (i0 >= ne0) return;

    const int p = pos[i0 / ne0];

    // Precompute angle (const expression)
    const float theta = __powf(10000.0f, -float(i0) / float(ne0));
    const float angle = p * theta;

    // Use fast math intrinsics
    float cos_theta, sin_theta;
    __sincosf(angle, &sin_theta, &cos_theta);

    // Process pair of elements
    const float x0 = x[i0];
    const float x1 = x[i0 + 1];

    dst[i0]     = x0 * cos_theta - x1 * sin_theta;
    dst[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
}

// Optimizations:
// 1. Process 2 elements per thread (reduce threads, better occupancy)
// 2. Remove divergence (all threads same path)
// 3. Use __powf and __sincosf (fast intrinsics)
// 4. Coalesced memory access

// Speedup: 2.8x!
```

### Example 2: Quantized GEMM Optimization

**Before (naive):**
```cuda
__global__ void mul_mat_q4_0_naive(
    const void* vx, const float* y, float* dst,
    int ncols_x, int nrows_x, int nrows_dst
) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < ncols_x / 32; i++) {
        // Dequantize Q4_0 block
        block_q4_0 block = ((block_q4_0*)vx)[row * (ncols_x/32) + i];

        for (int j = 0; j < 32; j++) {
            uint8_t q = (block.qs[j/2] >> (4*(j%2))) & 0x0F;
            float dequant = (q - 8) * __half2float(block.d);
            sum += dequant * y[col * ncols_x + i * 32 + j];
        }
    }

    dst[row * nrows_dst + col] = sum;
}

// Profile: 25% occupancy, 30% memory bandwidth, slow!
```

**After (optimized, from llama.cpp):**
```cuda
template<int qk, int qr, int ncols2, load_tiles_func_t load_tiles, int nwarps>
__global__ void mul_mat_q4_0_fast(/* ... */) {
    // Use shared memory for tiles
    __shared__ int   tile_x[nwarps * QI4_0 * WARP_SIZE];
    __shared__ half2 tile_y[nwarps * 2][ncols2];

    const int tid = threadIdx.x;

    // Warp-level cooperative loading
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Load tile to shared memory (coalesced)
    load_tiles(tile_x, tile_y, ...);
    __syncthreads();

    // Vectorized dot product using DP4A (int8 dot product)
    int sum_i = 0;
    #pragma unroll
    for (int i = 0; i < QI4_0; i++) {
        int vi = tile_x[warp_id * QI4_0 * WARP_SIZE + i * WARP_SIZE + lane_id];
        sum_i = __dp4a(vi, tile_y[...], sum_i);  // 4-way dot product
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum_i += __shfl_xor_sync(0xffffffff, sum_i, offset);
    }

    // Apply scale and write result
    if (lane_id == 0) {
        dst[...] = sum_i * scale;
    }
}

// Optimizations:
// 1. Shared memory tiling (reduce global memory traffic 16x)
// 2. Vectorized loads (float4, int4)
// 3. DP4A intrinsic (4x int8 multiply-add in 1 instruction)
// 4. Warp shuffle reduction (no shared memory, no sync)
// 5. Loop unrolling (#pragma unroll)

// Speedup: 18x over naive!
```

---

## Key Takeaways

1. **Profile first, optimize second** - use Nsight Compute/Systems
2. **Coalesce memory accesses** - 10x performance impact
3. **Fuse kernels** - reduce launch overhead and memory traffic
4. **Occupancy is not everything** - optimize for bottleneck
5. **Use intrinsics** - __powf, __sincosf, __dp4a for speedup
6. **Shared memory and register tuning** - balance resources

---

## Interview Questions

1. **Q:** What's the difference between Nsight Compute and Nsight Systems?
   **A:** Nsight Compute profiles individual kernels (occupancy, bandwidth, compute metrics). Nsight Systems profiles system-wide timeline (CPU-GPU interaction, multi-GPU, API calls).

2. **Q:** How do you identify a memory-bound kernel?
   **A:** Check memory throughput vs compute throughput in Nsight Compute. If memory bandwidth >70% and compute <50%, it's memory-bound. Roofline analysis also shows arithmetic intensity below ridge point.

3. **Q:** What is warp divergence and how do you fix it?
   **A:** When threads in a warp take different execution paths (if-else), they serialize, reducing performance by up to 32x. Fix by: (1) using predication instead of branches, (2) aligning divergence with warp boundaries, or (3) splitting into separate kernels.

4. **Q:** Explain the roofline model.
   **A:** Performance model with two ceilings: peak compute (TFLOPS) and peak memory (GB/s). Plot achieved performance vs arithmetic intensity (FLOPs/byte). If below memory ceiling → memory-bound, if below compute ceiling → compute-bound.

5. **Q:** How does kernel fusion improve performance?
   **A:** Reduces: (1) kernel launch overhead, (2) global memory round-trips, (3) synchronization. Example: 3 kernels (add, relu, scale) → 1 fused kernel saves 2 launches and 2 global memory passes, achieving 2-3x speedup.

---

**Module 4 Complete!**

**Next Steps:**
- Practice with [Lab 4: Kernel Optimization Challenge](../labs/lab4-kernel-optimization.md)
- Review [Module 4 Assessment](../assessment/module4-quiz.md)
- Continue to [Module 5: Advanced Inference](../../05-advanced-inference/README.md)
