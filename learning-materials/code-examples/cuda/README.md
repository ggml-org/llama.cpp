# CUDA Examples for LLM Inference

This directory contains introductory CUDA examples designed to teach the fundamentals of GPU programming relevant to LLM inference. These examples are part of Module 4 (GPU Programming and Optimization) of the llama.cpp learning materials.

## Overview

Modern LLMs rely heavily on GPU acceleration for efficient inference. Understanding CUDA programming is essential for:
- Optimizing inference performance
- Implementing custom kernels
- Debugging GPU-related issues
- Contributing to projects like llama.cpp

These examples progress from basic CUDA concepts to advanced techniques used in real LLM inference engines.

## Examples

### 1. Vector Addition (`01-vector-add.cu`)

**Difficulty:** Beginner
**Concepts:** Basic CUDA syntax, memory management, kernel launches

**What you'll learn:**
- CUDA execution model (threads, blocks, grids)
- Memory allocation and transfer (host ↔ device)
- Kernel launch syntax `<<<blocks, threads>>>`
- Error checking and debugging
- Basic performance considerations

**Key code patterns:**
```cuda
// Kernel definition
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel launch
vectorAddKernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
```

**Why it matters for LLMs:**
Element-wise operations (like ReLU, LayerNorm, addition) are common in neural networks. Understanding parallel execution is the foundation for all GPU programming.

---

### 2. Matrix Multiplication (`02-matrix-multiply.cu`)

**Difficulty:** Intermediate
**Concepts:** Shared memory, 2D thread blocks, tiling, synchronization

**What you'll learn:**
- CUDA memory hierarchy (global, shared, registers)
- Shared memory optimization (`__shared__`)
- Thread synchronization (`__syncthreads()`)
- Tiled algorithms for memory reuse
- Performance analysis (GFLOPS)

**Key code patterns:**
```cuda
__global__ void matmulTiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Load tile into shared memory
    tileA[threadIdx.y][threadIdx.x] = A[...];
    __syncthreads();  // Wait for all threads

    // Compute using fast shared memory
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
}
```

**Why it matters for LLMs:**
Matrix multiplication (GEMM) is the core operation in transformers:
- Attention: `Q @ K^T`, `scores @ V`
- Feed-forward: `hidden @ W1`, `intermediate @ W2`
- Understanding optimization techniques is crucial for performance

**Performance notes:**
- Naive implementation: ~50-100 GFLOPS
- Tiled implementation: ~500-1000 GFLOPS
- cuBLAS (optimized): ~5000-8000 GFLOPS on RTX 3090
- Tensor Cores (FP16): ~15000+ GFLOPS

---

### 3. Quantized Matrix Multiplication (`03-quantized-matmul.cu`)

**Difficulty:** Advanced
**Concepts:** Quantization, INT8 operations, mixed precision, dequantization

**What you'll learn:**
- Post-training quantization techniques
- INT8 arithmetic on GPU
- Per-channel vs per-tensor quantization
- Dequantization strategies
- Memory bandwidth optimization
- Accuracy vs performance tradeoffs

**Key code patterns:**
```cuda
// Quantization
int8_t quantize_fp32_to_int8(float value, float scale, int8_t zero_point) {
    return (int8_t)roundf(value / scale) + zero_point;
}

// INT8 computation with dequantization
__global__ void matmul_int8(const int8_t* A, const int8_t* B, float* C,
                            const QuantParams* paramsA, const QuantParams* paramsB,
                            int M, int N, int K) {
    int32_t sum = 0;  // Use INT32 accumulator

    // Integer multiply-accumulate
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += (int32_t)tileA[threadIdx.y][k] * (int32_t)tileB[k][threadIdx.x];
    }

    // Dequantize final result
    C[row * N + col] = (float)sum * scaleA * scaleB;
}
```

**Why it matters for LLMs:**
Quantization is essential for deploying large models:
- **LLaMA-7B:**  FP32: ~28 GB, INT8: ~7 GB, INT4: ~3.5 GB
- **LLaMA-70B:** FP32: ~280 GB, INT8: ~70 GB, INT4: ~35 GB
- Enables running larger models on consumer GPUs
- Significant speedup on modern GPUs with INT8 support

**Quantization formats in practice:**
- **GGUF (llama.cpp):** 2-8 bit quantization, various schemes (Q4_0, Q5_K_M, Q8_0)
- **GPTQ:** 4-bit group-wise quantization
- **AWQ:** Activation-aware weight quantization
- **bitsandbytes:** 8-bit and 4-bit quantization for training

---

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability ≥ 7.5 recommended)
- Check your GPU: `nvidia-smi --query-gpu=name,compute_cap --format=csv`

### Software
- CUDA Toolkit (11.0 or later recommended)
  - Download: https://developer.nvidia.com/cuda-downloads
  - Verify installation: `nvcc --version`
- C++ compiler (gcc/g++ on Linux, MSVC on Windows)
- Make (optional, for using Makefile)

## Building the Examples

### Option 1: Using Makefile (Recommended)

```bash
# Build all examples
make

# Build specific example
make vector_add
make matmul
make quantized

# Build for your GPU architecture (e.g., RTX 3090)
make arch=sm_86

# Build and run all examples
make run-all

# Clean build artifacts
make clean

# Show help
make help
```

### Option 2: Manual Compilation

```bash
# Vector addition
nvcc -O3 -o vector_add 01-vector-add.cu

# Matrix multiplication
nvcc -O3 -o matmul 02-matrix-multiply.cu

# Quantized matmul
nvcc -O3 -o quantized_matmul 03-quantized-matmul.cu

# For specific GPU architecture
nvcc -O3 -arch=sm_86 -o vector_add 01-vector-add.cu
```

### GPU Architecture Reference

| Architecture | Compute Capability | Examples |
|--------------|-------------------|----------|
| Turing       | sm_75             | RTX 20xx, GTX 16xx, T4 |
| Ampere       | sm_80             | A100, A30 |
| Ampere       | sm_86             | RTX 30xx, A40, A10 |
| Ada Lovelace | sm_89             | RTX 40xx, L40, L4 |
| Hopper       | sm_90             | H100, H200 |

## Running the Examples

```bash
# Run vector addition
./vector_add

# Run matrix multiplication
./matmul

# Run quantized matmul
./quantized_matmul
```

Expected output includes:
- Performance metrics (execution time, GFLOPS)
- Correctness verification
- Memory usage statistics
- Sample results

## Learning Path

### For Beginners
1. Start with `01-vector-add.cu`
   - Understand the CUDA execution model
   - Learn memory management
   - Master kernel launch syntax
2. Complete the exercises at the end of the file
3. Experiment with different thread block sizes

### For Intermediate Learners
1. Study `02-matrix-multiply.cu`
   - Compare naive vs optimized implementations
   - Understand shared memory benefits
   - Analyze performance metrics
2. Try the exercises (different tile sizes, profiling)
3. Compare with cuBLAS performance

### For Advanced Learners
1. Dive into `03-quantized-matmul.cu`
   - Understand quantization techniques
   - Experiment with different bit widths
   - Analyze accuracy vs performance tradeoffs
2. Explore real-world quantization libraries
3. Implement custom quantization schemes

## Performance Profiling

### Using NVIDIA Nsight Systems

```bash
# Profile execution timeline
nsys profile --stats=true ./vector_add

# Generate report
nsys profile -o vector_add_profile ./vector_add
# Open vector_add_profile.nsys-rep in Nsight Systems GUI
```

### Using NVIDIA Nsight Compute

```bash
# Detailed kernel analysis
ncu --set full -o matmul_profile ./matmul

# Profile specific kernel
ncu --kernel-name matmulTiled ./matmul
```

### Key Metrics to Monitor
- **Kernel execution time:** How long the kernel runs
- **Memory throughput:** GB/s achieved vs theoretical peak
- **Occupancy:** Ratio of active warps to maximum warps
- **GFLOPS:** Floating-point operations per second
- **Speedup:** Performance gain from optimization

## Common Issues and Solutions

### Issue: `nvcc: command not found`
**Solution:** CUDA Toolkit not installed or not in PATH
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: `CUDA Error: invalid device function`
**Solution:** Binary compiled for wrong GPU architecture
```bash
# Rebuild for your specific architecture
make clean
make arch=sm_86  # Use your GPU's compute capability
```

### Issue: Slow performance
**Solution:** Check several factors
1. Verify GPU is being used: `nvidia-smi` during execution
2. Increase problem size (small problems have overhead)
3. Profile with nsys/ncu to identify bottlenecks
4. Ensure `-O3` optimization flag is used

### Issue: Incorrect results
**Solution:** Common causes
1. Race conditions: Missing `__syncthreads()`
2. Out-of-bounds access: Check boundary conditions
3. Integer overflow: Use larger accumulator types
4. Floating-point precision: Relaxed tolerance for comparisons

## Connection to llama.cpp

These examples teach fundamental concepts used in llama.cpp's GPU backend:

### Example 1 → llama.cpp
- **Vector operations:** Element-wise operations in `ggml_cuda.cu`
- **Memory management:** Device tensor allocation

### Example 2 → llama.cpp
- **Matrix multiplication:** Core of `ggml_cuda_op_mul_mat()`
- **Shared memory:** Used in optimized GEMM kernels
- **Tiling:** Fundamental to all matrix operations

### Example 3 → llama.cpp
- **Quantization:** GGUF format (Q4_0, Q5_K_M, Q8_0, etc.)
- **Dequantization:** `dequantize_mul_mat_vec()` kernels
- **Mixed precision:** FP16 compute with quantized storage

### Next Steps for llama.cpp Development
1. Study `ggml-cuda.cu` - Main CUDA implementation
2. Review quantization kernels in `ggml-cuda/` directory
3. Understand specific quantization formats (e.g., `dequantize_q4_0()`)
4. Explore advanced techniques: Tensor Cores, Flash Attention

## Additional Resources

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

### Quantization Research
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)

### Related Projects
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [GGML](https://github.com/ggerganov/ggml) - Tensor library
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference engine
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's LLM inference

### Online Courses
- [NVIDIA DLI: Fundamentals of Accelerated Computing with CUDA C/C++](https://www.nvidia.com/en-us/training/)
- [Udacity: Intro to Parallel Programming (CUDA)](https://www.udacity.com/course/intro-to-parallel-programming--cs344)

## Exercises and Projects

### Beginner Projects
1. **Implement vector operations:** dot product, norm, scaling
2. **Add timing:** Compare CPU vs GPU for different problem sizes
3. **Visualize results:** Export data and plot with Python

### Intermediate Projects
1. **Implement transpose:** Optimize for coalesced memory access
2. **Batch matrix multiplication:** Extend to 3D tensors
3. **Layer normalization:** Implement a transformer component

### Advanced Projects
1. **Flash Attention:** Implement efficient attention mechanism
2. **Custom quantization:** Design your own quantization scheme
3. **Kernel fusion:** Combine multiple operations into one kernel
4. **Contribute to llama.cpp:** Optimize existing kernels

## Contributing

Found a bug or have an improvement? These examples are part of the llama.cpp learning materials. Contributions are welcome!

## License

These examples are part of the llama.cpp learning materials and follow the same license as the main project.

## Acknowledgments

These examples are designed to complement the llama.cpp codebase and help developers understand the GPU optimizations used in modern LLM inference engines.

---

**Happy Learning! 🚀**

For questions or discussions, refer to the main learning materials documentation or the llama.cpp community.
