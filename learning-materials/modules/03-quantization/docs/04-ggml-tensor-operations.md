# GGML Tensor Operations Deep Dive

**Module 3, Lesson 4** | **Estimated Time**: 3 hours | **Difficulty**: Advanced

## Table of Contents
1. [Introduction](#introduction)
2. [GGML Architecture](#ggml-architecture)
3. [Tensor Basics](#tensor-basics)
4. [Core Operations](#core-operations)
5. [Computation Graph](#computation-graph)
6. [Backend System](#backend-system)
7. [Memory Management](#memory-management)
8. [Custom Operations](#custom-operations)
9. [Optimization Techniques](#optimization-techniques)
10. [Interview Questions](#interview-questions)

---

## Introduction

GGML (GPT-Generated Model Language) is the tensor library powering llama.cpp. Understanding GGML is crucial for optimizing inference, implementing custom operations, and debugging performance issues.

**Learning Objectives:**
- Understand GGML's tensor abstraction and design philosophy
- Master core tensor operations used in LLM inference
- Work with computation graphs
- Implement custom operations
- Optimize tensor operations for different backends

**Prerequisites:**
- C/C++ programming
- Linear algebra fundamentals
- Understanding of neural network operations
- Performance optimization basics

---

## GGML Architecture

### Design Philosophy

GGML is designed for:
1. **Simplicity**: Minimal dependencies, easy to understand
2. **Portability**: Run on CPU, CUDA, Metal, OpenCL, etc.
3. **Performance**: Optimized kernels for common operations
4. **Flexibility**: Easy to add new operations and backends

### Key Components

```
ggml/
├── ggml.h              # Core tensor API
├── ggml.c              # CPU implementation
├── ggml-cuda.cu        # CUDA backend
├── ggml-metal.m        # Metal backend
├── ggml-opencl.cpp     # OpenCL backend
├── ggml-quants.h       # Quantization formats
└── ggml-alloc.h        # Memory allocator
```

### Architecture Diagram

```
┌─────────────────────────────────────┐
│         High-Level API              │
│  (ggml_mul_mat, ggml_add, etc.)    │
└─────────────────────────────────────┘
                 │
┌─────────────────────────────────────┐
│       Computation Graph             │
│  (Build graph of operations)        │
└─────────────────────────────────────┘
                 │
┌─────────────────────────────────────┐
│      Backend Dispatcher             │
│  (Select CPU/CUDA/Metal/etc.)       │
└─────────────────────────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
┌──────▼──────┐    ┌────────▼────────┐
│ CPU Backend │    │  CUDA Backend   │
│  (ggml.c)   │    │ (ggml-cuda.cu)  │
└─────────────┘    └─────────────────┘
```

---

## Tensor Basics

### Tensor Structure

```c
struct ggml_tensor {
    enum ggml_type type;          // Data type (F32, F16, Q4_0, etc.)

    enum ggml_backend_type backend;  // CPU, GPU, etc.

    int     n_dims;                // Number of dimensions
    int64_t ne[GGML_MAX_DIMS];    // Number of elements in each dimension
    size_t  nb[GGML_MAX_DIMS];    // Stride in bytes for each dimension

    // Operation info
    enum ggml_op op;               // Operation type
    struct ggml_tensor * src[GGML_MAX_SRC];  // Source tensors

    // Data
    void * data;                   // Pointer to data

    char name[GGML_MAX_NAME];     // Tensor name (for debugging)

    void * extra;                  // Backend-specific data
};
```

### Tensor Types

```c
enum ggml_type {
    GGML_TYPE_F32  = 0,  // 32-bit float
    GGML_TYPE_F16  = 1,  // 16-bit float
    GGML_TYPE_Q4_0 = 2,  // 4-bit quantized (legacy)
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,  // K-quants
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    // ... more types
};
```

### Creating Tensors

```c
// Create a context (memory pool)
struct ggml_init_params params = {
    .mem_size   = 128 * 1024 * 1024,  // 128 MB
    .mem_buffer = NULL,                // Let GGML allocate
    .no_alloc   = false,               // Allocate memory
};
struct ggml_context * ctx = ggml_init(params);

// Create tensors
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 512);
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 256);

// Set names (helpful for debugging)
ggml_set_name(a, "weight_matrix");
ggml_set_name(b, "input_vector");
```

### Tensor Dimensions

GGML uses column-major order (Fortran-style):

```c
// 2D tensor: [rows, cols]
struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
// t->ne[0] = rows
// t->ne[1] = cols

// 3D tensor: [dim0, dim1, dim2]
struct ggml_tensor * t3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d0, d1, d2);

// 4D tensor: [dim0, dim1, dim2, dim3]
struct ggml_tensor * t4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d0, d1, d2, d3);
```

### Accessing Tensor Data

```c
// Get element from 2D tensor
float get_2d(struct ggml_tensor * t, int i, int j) {
    float * data = (float *)t->data;
    return data[i + j * t->ne[0]];  // Column-major
}

// Set element in 2D tensor
void set_2d(struct ggml_tensor * t, int i, int j, float value) {
    float * data = (float *)t->data;
    data[i + j * t->ne[0]] = value;
}

// Iterate over all elements
void print_tensor(struct ggml_tensor * t) {
    float * data = (float *)t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");
}
```

---

## Core Operations

### Matrix Multiplication

Most important operation in LLMs:

```c
// Matrix multiplication: C = A * B
// A: [n, k]
// B: [k, m]
// C: [n, m]
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b
);

// Example
struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 4096);
struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
struct ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);
```

**Implementation notes:**
- Optimized with BLAS, SIMD, or GPU
- Supports mixed precision (e.g., Q4_0 weights × F32 input)
- Main performance bottleneck

### Element-wise Operations

```c
// Addition: c = a + b
struct ggml_tensor * ggml_add(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b
);

// Multiplication: c = a * b (element-wise)
struct ggml_tensor * ggml_mul(ctx, a, b);

// In-place addition: a = a + b
struct ggml_tensor * ggml_add_inplace(ctx, a, b);

// Scalar operations
struct ggml_tensor * ggml_scale(ctx, a, scale_factor);
```

### Activation Functions

```c
// ReLU: max(0, x)
struct ggml_tensor * ggml_relu(ctx, x);

// GELU (used in GPT): x * Φ(x)
struct ggml_tensor * ggml_gelu(ctx, x);

// SiLU / Swish (used in LLaMA): x * sigmoid(x)
struct ggml_tensor * ggml_silu(ctx, x);

// Softmax
struct ggml_tensor * ggml_soft_max(ctx, x);
```

### Normalization

```c
// RMS Normalization (used in LLaMA)
struct ggml_tensor * ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    float                 eps  // Small constant for numerical stability
);

// Layer Normalization
struct ggml_tensor * ggml_norm(ctx, a, eps);
```

### Tensor Reshaping

```c
// Reshape (no data copy)
struct ggml_tensor * ggml_reshape_2d(ctx, a, rows, cols);
struct ggml_tensor * ggml_reshape_3d(ctx, a, d0, d1, d2);

// View (different view of same data)
struct ggml_tensor * ggml_view_2d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    int64_t               ne0,
    int64_t               ne1,
    size_t                nb1,
    size_t                offset
);

// Permute dimensions
struct ggml_tensor * ggml_permute(ctx, a, axis0, axis1, axis2, axis3);

// Transpose (2D)
struct ggml_tensor * ggml_transpose(ctx, a);
```

### Attention Operations

```c
// Scaled dot-product attention
// Q: queries [seq_len, d_k]
// K: keys [seq_len, d_k]
// V: values [seq_len, d_v]

// 1. Compute Q * K^T / sqrt(d_k)
struct ggml_tensor * qk = ggml_mul_mat(ctx, k, q);  // K^T * Q
qk = ggml_scale(ctx, qk, 1.0f / sqrt(d_k));

// 2. Apply mask (for causal attention)
qk = ggml_diag_mask_inf(ctx, qk, n_past);

// 3. Softmax
qk = ggml_soft_max(ctx, qk);

// 4. Multiply by V
struct ggml_tensor * output = ggml_mul_mat(ctx, v, qk);
```

### Rope (Rotary Position Embedding)

```c
// Apply RoPE to queries and keys
struct ggml_tensor * ggml_rope(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    int                   n_past,    // Number of past tokens
    int                   n_dims,    // Dimension to apply RoPE
    int                   mode,      // RoPE mode
    int                   n_ctx      // Context size
);

// Used in LLaMA:
q = ggml_rope(ctx, q, n_past, n_embd_head, 0, n_ctx);
k = ggml_rope(ctx, k, n_past, n_embd_head, 0, n_ctx);
```

---

## Computation Graph

### Graph-Based Execution

GGML builds a computation graph instead of immediate execution:

```c
// Build graph
struct ggml_context * ctx = ggml_init(params);

struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
struct ggml_tensor * c = ggml_add(ctx, a, b);     // Graph node
struct ggml_tensor * d = ggml_mul(ctx, c, c);     // Another node
struct ggml_tensor * e = ggml_sum(ctx, d);        // Final result

// Create computation graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, e);  // Build from output backwards

// Execute graph
ggml_graph_compute_with_ctx(ctx, gf, n_threads);
```

### Benefits of Graph-Based Execution

1. **Optimization opportunities**:
   - Operator fusion
   - Memory planning
   - Parallel execution

2. **Memory efficiency**:
   - Reuse buffers
   - Compute in-place when possible

3. **Backend flexibility**:
   - Easy to compile for different backends
   - Can offload subgraphs to GPU

### Graph Optimization

```c
// Example: Fuse operations
// Before: d = (a + b) * c  (two operations)
// After:  d = fma(a, c, b * c)  (one fused operation)

// GGML automatically fuses common patterns:
// - Add + Mul → FMA
// - Mul + Softmax → Scaled Softmax
// - MatMul + Add → MatMul with bias
```

---

## Backend System

### Backend Types

```c
enum ggml_backend_type {
    GGML_BACKEND_CPU = 0,
    GGML_BACKEND_GPU = 10,
    GGML_BACKEND_GPU_SPLIT = 20,  // Multi-GPU
};
```

### CPU Backend

Default backend, highly optimized:

```c
// CPU implementation uses:
// - BLAS (OpenBLAS, Intel MKL, Apple Accelerate)
// - SIMD (AVX2, AVX-512, NEON)
// - Multi-threading (OpenMP, pthreads)

// Example: Matrix multiply on CPU
void ggml_compute_forward_mul_mat_f32(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
) {
    // Use optimized BLAS if available
    #ifdef GGML_USE_ACCELERATE
        cblas_sgemm(...);
    #else
        // Custom implementation with SIMD
        mul_mat_avx2(...);
    #endif
}
```

### CUDA Backend

GPU acceleration for NVIDIA:

```c
// CUDA kernel for matrix multiply
__global__ void mul_mat_q4_0_f32(
    const void * vx, const float * y, float * dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst
) {
    // Each block processes a portion of the matrix
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Cooperative loading into shared memory
    __shared__ float tmp[WARP_SIZE];

    // Compute dot product
    float sum = 0.0f;
    for (int i = tid; i < ncols_x; i += WARP_SIZE) {
        sum += dequantize_q4_0(vx, i) * y[i];
    }

    // Reduce within warp
    sum = warp_reduce_sum(sum);

    if (tid == 0) {
        dst[row] = sum;
    }
}
```

### Backend Selection

```c
// Automatically select backend
struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);

// Backend selected based on:
// 1. Tensor location (CPU vs GPU memory)
// 2. Operation support (some ops CPU-only)
// 3. Performance (auto-tuning)

// Manual backend specification
ggml_backend_tensor_set(tensor, GGML_BACKEND_GPU);
```

---

## Memory Management

### Context-Based Allocation

```c
// All tensors allocated from context
struct ggml_init_params params = {
    .mem_size   = 1024 * 1024 * 1024,  // 1 GB
    .mem_buffer = NULL,                 // Auto-allocate
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);

// Tensors allocated from context pool
struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);

// Free entire context at once
ggml_free(ctx);  // Frees t1, t2, and all other tensors
```

### Graph Allocator

Efficient memory allocation for computation graphs:

```c
// Use allocator for graph
struct ggml_allocr * allocr = ggml_allocr_new(...);

// Allocate tensors as needed
ggml_allocr_alloc_graph(allocr, graph);

// Reuse memory between operations
// GGML automatically identifies when tensors can share memory
```

### Memory Mapping

For large model weights:

```c
// Map model file to memory
struct ggml_context * ctx_data = ggml_init({
    .mem_size   = file_size,
    .mem_buffer = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0),
    .no_alloc   = true,  // Don't allocate, use mmap'd memory
});

// Tensors point into mmap'd file
struct ggml_tensor * weights = ggml_new_tensor_2d(ctx_data, ...);
weights->data = (char *)mmap_ptr + offset;
```

---

## Custom Operations

### Implementing a Custom Operation

```c
// 1. Define operation type
enum ggml_op {
    // ... existing ops
    GGML_OP_CUSTOM_MY_OP = 1000,
};

// 2. Create operation function
struct ggml_tensor * ggml_my_custom_op(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b
) {
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], a->ne[1]);
    result->op = GGML_OP_CUSTOM_MY_OP;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// 3. Implement compute function
void ggml_compute_forward_my_custom_op(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
) {
    // Your implementation here
    const float * a_data = (float *)src0->data;
    const float * b_data = (float *)src1->data;
    float * dst_data = (float *)dst->data;

    for (int i = 0; i < ggml_nelements(dst); i++) {
        dst_data[i] = my_operation(a_data[i], b_data[i]);
    }
}

// 4. Register with GGML
// Add to ggml_compute_forward() switch statement
switch (tensor->op) {
    case GGML_OP_CUSTOM_MY_OP:
        ggml_compute_forward_my_custom_op(params, tensor->src[0], tensor->src[1], tensor);
        break;
    // ... other cases
}
```

### Example: Custom Activation Function

```c
// Custom activation: swish variant
struct ggml_tensor * ggml_swish_custom(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    float                 beta
) {
    // Implementation: a * sigmoid(beta * a)
    struct ggml_tensor * scaled = ggml_scale(ctx, a, beta);
    struct ggml_tensor * sig = ggml_sigmoid(ctx, scaled);
    return ggml_mul(ctx, a, sig);
}
```

---

## Optimization Techniques

### 1. In-Place Operations

```c
// Out-of-place (allocates new tensor)
struct ggml_tensor * c = ggml_add(ctx, a, b);

// In-place (modifies a, no allocation)
struct ggml_tensor * a = ggml_add_inplace(ctx, a, b);  // a = a + b

// Benefits:
// - Reduced memory usage
// - Fewer allocations
// - Better cache locality
```

### 2. Operation Fusion

```c
// Unfused (two operations)
struct ggml_tensor * x = ggml_mul_mat(ctx, w, input);  // Matrix multiply
struct ggml_tensor * y = ggml_add(ctx, x, bias);       // Add bias

// Fused (single operation)
// GGML automatically fuses common patterns
// Or implement custom fused kernel:
struct ggml_tensor * y = ggml_mul_mat_bias(ctx, w, input, bias);
```

### 3. Mixed Precision

```c
// Use lower precision for weights, higher for computation
struct ggml_tensor * weights_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ...);
struct ggml_tensor * input_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ...);

// Automatically dequantizes during computation
struct ggml_tensor * output = ggml_mul_mat(ctx, weights_q4, input_f32);
// output is F32
```

### 4. View Tensors (Zero-Copy)

```c
// Instead of copying, create view
struct ggml_tensor * full_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

// Extract slice without copying
struct ggml_tensor * slice = ggml_view_1d(
    ctx,
    full_tensor,
    256,    // Length
    512 * sizeof(float)  // Offset
);

// slice points into full_tensor's memory
```

---

## Interview Questions

### Conceptual

1. **Q: Why does GGML use a computation graph instead of immediate execution?**

   A: Benefits of graph-based execution:
   - **Optimization**: Can analyze entire graph and apply optimizations (fusion, memory planning)
   - **Memory efficiency**: Can reuse buffers, compute minimum memory needed
   - **Parallelization**: Can identify independent operations to run in parallel
   - **Backend portability**: Can compile graph for different backends (CPU/GPU/TPU)
   - **Debugging**: Can visualize and inspect entire computation

   Trade-off: Slightly more complex API vs immediate execution

2. **Q: Explain the difference between ggml_add and ggml_add_inplace.**

   A:
   - `ggml_add(ctx, a, b)`: Creates new tensor c = a + b (allocates memory)
   - `ggml_add_inplace(ctx, a, b)`: Modifies a in-place, a = a + b (no allocation)

   Use in-place when:
   - Don't need original value of `a`
   - Want to save memory
   - Working with large tensors

   Caution: In-place ops modify source, can't reuse in graph

3. **Q: How does GGML handle mixed-precision computation (e.g., Q4_0 weights × F32 input)?**

   A: GGML automatically handles type conversions:
   ```c
   tensor_q4 * tensor_f32:
   1. Dequantize Q4_0 blocks on-the-fly during matmul
   2. Perform computation in F32
   3. Output in F32

   // Implementation:
   for each block in Q4_0 tensor:
       dequantize block to F32 (8-32 values)
       compute with F32 input
       accumulate result
   ```

   This allows:
   - Keeping weights compressed
   - Accurate computation in higher precision
   - No need for explicit conversion

### Implementation

4. **Q: Implement a function to compute tensor norm (L2 norm) using GGML operations.**

   A:
   ```c
   struct ggml_tensor * ggml_norm_l2(
       struct ggml_context * ctx,
       struct ggml_tensor  * a
   ) {
       // L2 norm: sqrt(sum(a^2))

       // 1. Square each element
       struct ggml_tensor * a_squared = ggml_mul(ctx, a, a);

       // 2. Sum all elements
       struct ggml_tensor * sum = ggml_sum(ctx, a_squared);

       // 3. Square root
       struct ggml_tensor * norm = ggml_sqrt(ctx, sum);

       return norm;
   }
   ```

5. **Q: How would you implement layer normalization from scratch using GGML?**

   A:
   ```c
   struct ggml_tensor * layer_norm(
       struct ggml_context * ctx,
       struct ggml_tensor  * x,
       float                 eps
   ) {
       // LayerNorm: (x - mean) / sqrt(var + eps)

       // 1. Compute mean
       struct ggml_tensor * mean = ggml_mean(ctx, x);

       // 2. Subtract mean
       struct ggml_tensor * x_centered = ggml_sub(ctx, x, mean);

       // 3. Compute variance: mean(x^2)
       struct ggml_tensor * x_sq = ggml_mul(ctx, x_centered, x_centered);
       struct ggml_tensor * variance = ggml_mean(ctx, x_sq);

       // 4. Compute std: sqrt(var + eps)
       struct ggml_tensor * var_eps = ggml_add1(ctx, variance,
                                                 ggml_new_f32(ctx, eps));
       struct ggml_tensor * std = ggml_sqrt(ctx, var_eps);

       // 5. Normalize
       struct ggml_tensor * normalized = ggml_div(ctx, x_centered, std);

       return normalized;
   }
   ```

### Advanced

6. **Q: Explain memory layout and strides in GGML tensors. How would you access element [i,j,k] in a 3D tensor?**

   A: GGML tensors store strides in `nb[]` array:
   ```c
   struct ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
   // t->ne[0] = ne0, t->ne[1] = ne1, t->ne[2] = ne2
   // t->nb[0] = element size (e.g., 4 for F32)
   // t->nb[1] = ne0 * element_size
   // t->nb[2] = ne0 * ne1 * element_size

   // Access element [i,j,k]:
   char * data_ptr = (char *)t->data;
   float * element = (float *)(data_ptr + i * t->nb[0] +
                                            j * t->nb[1] +
                                            k * t->nb[2]);
   ```

   Strides allow for:
   - Non-contiguous views
   - Transposed tensors without data copy
   - Slicing and broadcasting

7. **Q: How would you optimize a custom GGML operation for AVX2?**

   A:
   ```c
   void compute_custom_op_avx2(const float * a, const float * b,
                                float * c, int n) {
       #ifdef __AVX2__
       int i;
       for (i = 0; i + 7 < n; i += 8) {
           __m256 va = _mm256_loadu_ps(&a[i]);
           __m256 vb = _mm256_loadu_ps(&b[i]);

           // Your custom operation (example: a^2 + b)
           __m256 va_sq = _mm256_mul_ps(va, va);
           __m256 vc = _mm256_add_ps(va_sq, vb);

           _mm256_storeu_ps(&c[i], vc);
       }

       // Handle remainder
       for (; i < n; i++) {
           c[i] = a[i] * a[i] + b[i];
       }
       #else
       // Fallback scalar version
       for (int i = 0; i < n; i++) {
           c[i] = a[i] * a[i] + b[i];
       }
       #endif
   }
   ```

   Steps:
   1. Detect AVX2 support (#ifdef or runtime check)
   2. Process 8 floats per iteration
   3. Handle remainder with scalar code
   4. Ensure data alignment if possible
   5. Benchmark and profile

---

## Summary

**Key Takeaways:**

1. **GGML is a tensor library** optimized for LLM inference
2. **Graph-based execution** enables optimizations and multi-backend support
3. **Mixed precision** handled automatically (e.g., Q4 × F32)
4. **Extensible** - easy to add custom operations
5. **Performance-critical** - most time spent in matrix multiply and attention

**Common Operations:**
- `ggml_mul_mat`: Matrix multiplication (most important)
- `ggml_add`, `ggml_mul`: Element-wise ops
- `ggml_rope`: Rotary position embedding
- `ggml_rms_norm`: RMS normalization
- `ggml_soft_max`: Attention softmax

**Next Steps:**
- Lesson 5: Benchmarking and testing
- Lab 4: Implementing custom GGML operations
- Tutorial: Profiling GGML performance

---

**Further Reading:**

- [GGML Repository](https://github.com/ggerganov/ggml)
- [GGML Header Documentation](https://github.com/ggerganov/ggml/blob/master/include/ggml.h)
- [Tensor Computation Fundamentals](https://en.wikipedia.org/wiki/Tensor_contraction)

**Author**: Agent 5 (Documentation Specialist)
**Module**: 3 - Quantization & Optimization
**Last Updated**: 2025-11-18
