# Alternative GPU Backends: ROCm, SYCL, Metal, Vulkan

**Module 4, Lesson 5** | **Duration: 2 hours** | **Level: Advanced**

## Table of Contents
1. [GPU Backend Landscape](#gpu-backend-landscape)
2. [ROCm (AMD GPUs)](#rocm-amd-gpus)
3. [SYCL (Intel GPUs)](#sycl-intel-gpus)
4. [Metal (Apple Silicon)](#metal-apple-silicon)
5. [Vulkan Compute](#vulkan-compute)
6. [Backend Comparison](#backend-comparison)

---

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand alternative GPU backends beyond CUDA
- ✅ Compare ROCm, SYCL, Metal, and Vulkan
- ✅ Configure llama.cpp for different hardware
- ✅ Optimize for AMD, Intel, and Apple GPUs
- ✅ Choose appropriate backend for deployment

---

## GPU Backend Landscape

### Why Multiple Backends?

**Hardware Diversity:**
```
Vendor        GPUs               Backend     Market Share
──────────────────────────────────────────────────────────
NVIDIA        A100, H100, RTX    CUDA        ~80% datacenter
AMD           MI250, MI300       ROCm        ~15% datacenter
Intel         Data Center Max    SYCL/oneAPI ~3% datacenter
Apple         M1/M2/M3 Ultra     Metal       ~2% edge/desktop
```

**llama.cpp Backend Support:**
```cpp
// From ggml-cuda.h
#if defined(GGML_USE_HIP)
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
```

**Build Options:**
```cmake
# CMakeLists.txt options
option(GGML_CUDA    "ggml: use CUDA"     OFF)
option(GGML_HIP     "ggml: use HIP (AMD ROCm)" OFF)
option(GGML_SYCL    "ggml: use SYCL"    OFF)
option(GGML_METAL   "ggml: use Metal"   OFF)
option(GGML_VULKAN  "ggml: use Vulkan"  OFF)
```

---

## ROCm (AMD GPUs)

### AMD GPU Architecture

**CDNA (Compute-Focused):**
```
GPU         Compute Units   VRAM     Memory BW    FP32 TFLOPS
──────────────────────────────────────────────────────────────
MI100       120             32 GB    1,228 GB/s   11.5 (23.1 FP64)
MI210       104             64 GB    1,638 GB/s   45.3
MI250X      2×110 (dual)    128 GB   3,277 GB/s   47.9 × 2
MI300X      304             192 GB   5,300 GB/s   163.4
```

**RDNA (Gaming/Consumer):**
```
GPU         Compute Units   VRAM     Memory BW    FP32 TFLOPS
──────────────────────────────────────────────────────────────
RX 6900 XT  80              16 GB    512 GB/s     23.0
RX 7900 XTX 96              24 GB    960 GB/s     61.4
```

### ROCm vs CUDA

**HIP (Heterogeneous-compute Interface for Portability):**
```cpp
// CUDA code:
cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// HIP code (nearly identical):
hipMalloc(&ptr, size);
hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
hipDeviceSynchronize();
```

**llama.cpp Abstraction:**
```cpp
// Use CUDA/HIP macros for portability
#if defined(GGML_USE_HIP)
    #define CUDA_CHECK(x) HIP_CHECK(x)
    #define cudaMalloc    hipMalloc
    #define cudaMemcpy    hipMemcpy
    #define cudaFree      hipFree
    // ... etc
#endif
```

### hipify: CUDA → HIP Translation

**Automatic conversion tool:**
```bash
# Convert CUDA file to HIP
hipify-perl file.cu > file.hip

# Or in-place
hipify-perl -inplace file.cu
```

**Example conversion:**
```cpp
// Before (CUDA):
__global__ void kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}

cudaMalloc(&d_data, size);
kernel<<<blocks, threads>>>(d_data);
cudaDeviceSynchronize();

// After (HIP) - same code works!
__global__ void kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}

hipMalloc(&d_data, size);
hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, d_data);
hipDeviceSynchronize();
```

### Building llama.cpp with ROCm

```bash
# Install ROCm
sudo apt-get install rocm-dkms rocm-dev rocblas rocrand

# Build llama.cpp
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
         -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
make -j$(nproc)

# Run
./llama-cli -m model.gguf -p "Test"
```

**Performance:**
```
Model: LLaMA-7B Q4_K_M
Hardware: AMD MI250X vs NVIDIA A100

Metric              MI250X      A100        Ratio
─────────────────────────────────────────────────
Prompt (512 tok)    18 ms       15 ms       1.2x
Generation          12 ms/tok   9 ms/tok    1.33x
Memory BW           1,638 GB/s  2,039 GB/s  0.80x

Conclusion: Competitive, within 20-30% of NVIDIA
```

### ROCm Limitations

1. **Smaller ecosystem** - Fewer libraries than CUDA
2. **Driver stability** - Can be less stable than NVIDIA drivers
3. **Compatibility** - Some features lag behind CUDA (e.g., graph capture)
4. **Documentation** - Less comprehensive than CUDA docs

---

## SYCL (Intel GPUs)

### SYCL Overview

**SYCL = Open standard for heterogeneous computing** (Khronos Group)
- Similar to OpenCL but C++ based
- Works on Intel, NVIDIA, AMD GPUs
- Intel's implementation: **oneAPI DPC++**

**Intel GPU Lineup:**
```
GPU              Compute Units   VRAM    Memory BW    FP32 TFLOPS
────────────────────────────────────────────────────────────────
Iris Xe (iGPU)   96              shared  68 GB/s      2.0
Arc A770         512             16 GB   560 GB/s     17.2
Data Center Max  128 stacks      128 GB  3,277 GB/s   52.0
```

### SYCL Programming Model

```cpp
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;  // Default device queue

    const int N = 1000000;
    sycl::buffer<float> buf_a(N), buf_b(N), buf_c(N);

    // Initialize data (omitted)

    // Launch kernel
    q.submit([&](sycl::handler& h) {
        auto a = buf_a.get_access<sycl::access::mode::read>(h);
        auto b = buf_b.get_access<sycl::access::mode::read>(h);
        auto c = buf_c.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            c[idx] = a[idx] + b[idx];
        });
    });

    q.wait();  // Synchronize
    return 0;
}
```

**SYCL vs CUDA:**
```
Feature                CUDA                SYCL
─────────────────────────────────────────────────
Vendor                 NVIDIA only         Multi-vendor
Language               C++ with extensions Standard C++17
Memory model           Explicit            Buffers (implicit)
Kernel launch          <<<grid,block>>>    parallel_for
Compilation            nvcc                dpcpp (Intel)
```

### llama.cpp SYCL Backend

**Building:**
```bash
# Install Intel oneAPI
wget https://registrationcenter-download.intel.com/...
sudo sh ./l_BaseKit_p_2024.0.0.49564_offline.sh

# Source oneAPI
source /opt/intel/oneapi/setvars.sh

# Build llama.cpp
mkdir build && cd build
cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
make -j$(nproc)
```

**Performance (Intel Data Center Max 1550):**
```
Model: LLaMA-7B Q4_K_M

Metric              Max 1550    A100        Ratio
─────────────────────────────────────────────────
Prompt (512 tok)    35 ms       15 ms       2.33x
Generation          22 ms/tok   9 ms/tok    2.44x

Conclusion: 2-2.5x slower than A100, but usable
```

**Limitations:**
1. **Newer backend** - Less mature than CUDA/ROCm
2. **Performance gap** - 2-3x slower than NVIDIA for LLMs
3. **Intel GPU availability** - Limited compared to NVIDIA/AMD

---

## Metal (Apple Silicon)

### Apple Silicon GPUs

**M-Series Chips:**
```
Chip      GPU Cores   Unified Memory   Memory BW    FP32 TFLOPS
────────────────────────────────────────────────────────────────
M1        8           8-16 GB          68 GB/s      2.6
M1 Pro    16          16-32 GB         200 GB/s     5.2
M1 Max    32          32-64 GB         400 GB/s     10.4
M1 Ultra  64          64-128 GB        800 GB/s     20.8
M2        10          8-24 GB          100 GB/s     3.6
M2 Pro    19          16-32 GB         200 GB/s     6.8
M2 Max    38          32-96 GB         400 GB/s     13.6
M2 Ultra  76          64-192 GB        800 GB/s     27.2
M3 Max    40          36-128 GB        400 GB/s     14.2
```

**Unified Memory Advantage:**
- **No CPU↔GPU transfers** - GPU directly accesses system RAM
- **Large effective VRAM** - Can use all system memory
- **Zero-copy buffers** - Efficient for CPU-GPU collaboration

### Metal Shading Language

**Metal Kernel Example:**
```metal
// vector_add.metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
```

**Host Code (Objective-C++):**
```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queue = [device newCommandQueue];
id<MTLLibrary> library = [device newDefaultLibrary];
id<MTLFunction> function = [library newFunctionWithName:@"vector_add"];

// Create pipeline
id<MTLComputePipelineState> pipeline =
    [device newComputePipelineStateWithFunction:function error:nil];

// Allocate buffers (zero-copy if using shared memory)
id<MTLBuffer> bufferA = [device newBufferWithLength:size
                         options:MTLResourceStorageModeShared];
id<MTLBuffer> bufferB = [device newBufferWithLength:size
                         options:MTLResourceStorageModeShared];
id<MTLBuffer> bufferC = [device newBufferWithLength:size
                         options:MTLResourceStorageModeShared];

// Encode kernel
id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:bufferA offset:0 atIndex:0];
[encoder setBuffer:bufferB offset:0 atIndex:1];
[encoder setBuffer:bufferC offset:0 atIndex:2];

MTLSize gridSize = MTLSizeMake(N, 1, 1);
MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
[encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

[encoder endEncoding];
[commandBuffer commit];
[commandBuffer waitUntilCompleted];
```

### llama.cpp Metal Backend

**Automatic on macOS:**
```bash
# Build (Metal enabled by default on macOS)
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run
./llama-cli -m model.gguf -p "Test" --n-gpu-layers 999
```

**Performance (M2 Ultra, 192 GB):**
```
Model              Prompt (512 tok)   Generation   Notes
───────────────────────────────────────────────────────
LLaMA-7B Q4_K_M    45 ms              18 ms/tok    Fast!
LLaMA-13B Q4_K_M   80 ms              28 ms/tok    Good
LLaMA-30B Q4_K_M   180 ms             55 ms/tok    Usable
LLaMA-70B Q4_K_M   420 ms             140 ms/tok   Slow but works

Comparison to RTX 4090 (LLaMA-7B):
  Prompt:      45 ms vs 12 ms  (3.75x slower)
  Generation:  18 ms vs 8 ms   (2.25x slower)

But: Can run 70B model with 192 GB unified memory!
     RTX 4090 (24 GB) cannot fit 70B even quantized.
```

**Optimizations in llama.cpp:**
```metal
// From ggml-metal.metal (simplified)
kernel void kernel_mul_mat_q4_K(
    device const uchar* src0 [[buffer(0)]],  // Q4_K weights
    device const float* src1 [[buffer(1)]],  // FP32 activations
    device float* dst [[buffer(2)]],         // FP32 output
    constant int& ne00 [[buffer(3)]],
    constant int& ne01 [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Use SIMD groups (32 threads on Apple GPUs)
    const int row = tgpig.y;
    const int col = tgpig.x * 32 + tiisg;

    // Dequantize Q4_K and compute dot product
    float sum = 0.0f;
    for (int i = 0; i < ne00 / 256; i++) {
        // Load Q4_K block
        device const block_q4_K* block = (device const block_q4_K*)(src0 + ...);

        // Dequantize and accumulate (optimized with SIMD)
        sum += dequant_q4_K_dot(block, src1, tiisg);
    }

    // Write result
    if (col < ne01) {
        dst[row * ne01 + col] = sum;
    }
}
```

---

## Vulkan Compute

### Vulkan Overview

**Vulkan** = Low-level, cross-platform graphics and compute API
- Successor to OpenGL
- Works on Windows, Linux, Android, macOS (via MoltenVK)
- **Compute shaders** for GPGPU

**Advantages:**
1. **Cross-platform** - Same code on NVIDIA, AMD, Intel, mobile GPUs
2. **Mobile support** - Runs on Android, iOS (via MoltenVK)
3. **Lower overhead** - Explicit control, less driver overhead

**Disadvantages:**
1. **Verbose** - ~1000 lines of boilerplate for "Hello World"
2. **Complex** - Steep learning curve
3. **Less optimized** - Vendor libraries (CUDA, Metal) often faster

### Vulkan Compute Shader

```glsl
// vector_add.comp
#version 450

layout(local_size_x = 256) in;

layout(binding = 0) buffer BufferA { float data[]; } bufferA;
layout(binding = 1) buffer BufferB { float data[]; } bufferB;
layout(binding = 2) buffer BufferC { float data[]; } bufferC;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    bufferC.data[idx] = bufferA.data[idx] + bufferB.data[idx];
}
```

**Host Code (C++, simplified):**
```cpp
// 1. Create instance, device, queue
VkInstance instance;
vkCreateInstance(&createInfo, nullptr, &instance);

VkDevice device;
vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);

// 2. Compile shader (SPIR-V bytecode)
glslangValidator -V vector_add.comp -o vector_add.spv

// 3. Create shader module
VkShaderModule shaderModule;
vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);

// 4. Create pipeline
VkPipeline pipeline;
vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo,
                         nullptr, &pipeline);

// 5. Allocate buffers
VkBuffer buffers[3];
for (int i = 0; i < 3; i++) {
    vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffers[i]);
    // ... allocate memory, bind, etc (many steps!)
}

// 6. Dispatch compute
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdBindDescriptorSets(commandBuffer, ...);
vkCmdDispatch(commandBuffer, N/256, 1, 1);

// 7. Submit and wait
vkQueueSubmit(queue, 1, &submitInfo, fence);
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
```

**Complexity:** 100x more code than CUDA for same operation!

### llama.cpp Vulkan Backend

**Status:** Experimental, limited support

**Build:**
```bash
# Install Vulkan SDK
wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
tar -xf vulkan-sdk.tar.gz
source setup-env.sh

# Build llama.cpp
cmake .. -DGGML_VULKAN=ON
make
```

**Use Case:** Mobile/embedded devices, cross-platform compatibility

---

## Backend Comparison

### Performance Summary

```
Backend   Vendor   Ecosystem   Performance   Portability   Ease of Use
─────────────────────────────────────────────────────────────────────
CUDA      NVIDIA   Excellent   100% (ref)    Low           High
ROCm      AMD      Good        80-90%        Medium        High
SYCL      Intel    Growing     40-50%        High          Medium
Metal     Apple    Apple-only  50-60%        Low           Medium
Vulkan    All      Good        60-70%        Highest       Low
```

### Recommendation

**For Production:**
- **NVIDIA GPUs:** Use CUDA (best performance)
- **AMD GPUs:** Use ROCm (80-90% of CUDA performance)
- **Intel GPUs:** Use SYCL (usable, but 2x slower)
- **Apple Silicon:** Use Metal (great unified memory)
- **Mobile/Edge:** Consider Vulkan (cross-platform)

**For Development:**
- Start with CUDA (most mature)
- Test on target hardware with appropriate backend

---

## Key Takeaways

1. **CUDA is fastest** but NVIDIA-only
2. **ROCm is competitive** for AMD GPUs (80-90% CUDA perf)
3. **Metal leverages unified memory** on Apple Silicon
4. **SYCL enables Intel GPUs** but with performance gap
5. **Vulkan is most portable** but complex and slower
6. **llama.cpp abstracts backends** - same code works everywhere

---

**Next Lesson:** [06-gpu-optimization.md](06-gpu-optimization.md)

**Related Labs:**
- [Lab 1: First CUDA Kernel](../labs/lab1-first-cuda-kernel.md)
- [Lab 4: Kernel Optimization Challenge](../labs/lab4-kernel-optimization.md)
