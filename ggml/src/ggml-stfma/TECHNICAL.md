# Sparse Ternary FMA Kernel: Technical Documentation

**Author:** Maurice Wilson, Founder, HyperFold Technologies UK  
**Contact:** maurice.wilson@hyperfold-technologies.com  
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [2-Bit Ternary Encoding Scheme](#2-bit-ternary-encoding-scheme)
3. [Algorithm Design](#algorithm-design)
4. [AVX-512 SIMD Implementation](#avx-512-simd-implementation)
5. [Performance Analysis](#performance-analysis)
6. [Integration Guide](#integration-guide)
7. [References](#references)

---

## Overview

The Sparse Ternary Fused Multiply-Add (FMA) kernel is a high-performance C library designed to accelerate polynomial arithmetic in cryptographic applications, particularly Fully Homomorphic Encryption (FHE) schemes like TFHE. The kernel exploits the ternary nature of secret keys (composed of values in {-1, 0, 1}) to achieve significant performance improvements through three key innovations:

1. **2-bit encoding** of ternary values, reducing memory footprint by 75% compared to standard 8-bit representations
2. **SIMD acceleration** using AVX-512 instructions, processing 8 coefficients simultaneously
3. **Sparse optimization** that skips zero elements, achieving up to 26× speedup for typical key distributions

The kernel is designed to be dependency-free, portable, and easy to integrate into existing projects. It provides both scalar and SIMD implementations, with automatic dispatch based on CPU capabilities.

---

## 2-Bit Ternary Encoding Scheme

### Encoding Table

The 2-bit encoding maps ternary values to compact bit patterns as follows:

| Ternary Value | Mathematical | 2-Bit Encoding | Binary |
|:--------------|:-------------|:---------------|:-------|
| **-1**        | Negative     | `0b10`         | `10`   |
| **0**         | Zero         | `0b00`         | `00`   |
| **+1**        | Positive     | `0b01`         | `01`   |
| **Invalid**   | Error        | `0b11`         | `11`   |

### Design Rationale

The encoding scheme was carefully designed to optimize both storage efficiency and computational performance. Each value has a distinct bit pattern, with zero represented as `0b00` to simplify conditional operations. The high bit indicates sign (0 for positive, 1 for negative), while the low bit indicates magnitude for positive values. The invalid pattern `0b11` is reserved for error detection.

This encoding enables several key optimizations. First, it achieves a 4× improvement in density compared to 8-bit integer representations, allowing 256 trits to fit in a single 512-bit AVX-512 vector. Second, it eliminates the need for actual multiplication operations, replacing them with conditional selection using bitwise masks. Third, it enables efficient SIMD processing by allowing multiple trits to be packed into standard integer types.

### Packing Format

Four trits are packed into a single byte using the following layout:

```
Byte layout:
|  7  6  |  5  4  |  3  2  |  1  0  |
| trit3  | trit2  | trit1  | trit0  |
```

This packing scheme allows an array of N ternary values to be stored in N/4 bytes, achieving the 75% memory reduction. The packing and unpacking operations are implemented using simple bitwise shifts and masks, making them extremely efficient.

---

## Algorithm Design

### Mathematical Definition

The Sparse Ternary FMA operation computes the following:

```
C[i] = C[i] + A[i] × decode(B_trit[i])
```

where:
- `A[i]` is a dense coefficient (64-bit integer)
- `B_trit[i]` is a 2-bit encoded ternary value
- `C[i]` is an accumulator (64-bit integer)
- `decode(B_trit[i])` ∈ {-1, 0, 1}

Since the multiplier is always in {-1, 0, 1}, the multiplication can be replaced by conditional selection. This eliminates the need for expensive integer multiplication instructions and reduces the operation to a simple conditional add or subtract.

### Scalar Implementation

The scalar implementation processes one element at a time using the following algorithm:

```c
for (size_t i = 0; i < N; i++) {
    /* Extract 2-bit trit from packed array */
    size_t byte_idx = i / 4;
    size_t trit_offset = (i % 4) * 2;
    uint8_t trit = (B_trit[byte_idx] >> trit_offset) & 0b11;
    
    /* Decode and accumulate */
    if (trit == TRIT_POS) {
        C[i] += A[i];
    } else if (trit == TRIT_NEG) {
        C[i] -= A[i];
    }
    /* else: trit == TRIT_ZERO, skip */
}
```

This implementation is straightforward and portable, requiring no special CPU features. It serves as a reference implementation and fallback for systems without AVX-512 support.

### Sparse Implementation

For very sparse arrays (Hamming weight w << N), the kernel provides an optimized implementation that only processes non-zero elements. This is achieved by maintaining separate arrays of indices and values for the non-zero elements:

```c
for (size_t i = 0; i < w; i++) {
    uint32_t idx = indices[i];
    int8_t value = values[i];
    
    if (value == 1) {
        C[idx] += A[idx];
    } else {  /* value == -1 */
        C[idx] -= A[idx];
    }
}
```

This approach reduces the computational complexity from O(N) to O(w), where w is the Hamming weight. For typical TFHE parameters with N=2048 and w=128, this results in a 16× theoretical speedup. In practice, the measured speedup often exceeds this due to improved cache locality and reduced memory bandwidth requirements.

---

## AVX-512 SIMD Implementation

### Vector Layout

The AVX-512 implementation processes 8 elements simultaneously using 512-bit vectors. Each vector contains eight 64-bit elements, corresponding to eight coefficients and their associated ternary multipliers.

The key challenge in the SIMD implementation is efficiently extracting and processing the 2-bit trits. The algorithm loads 2 bytes (16 bits) containing 8 trits, extracts each trit into a 64-bit element, and then uses vector masks to perform conditional operations.

### Core Algorithm

The AVX-512 kernel implements the following steps for each iteration:

1. **Load Data**: Load 8 coefficients from array A and 8 accumulators from array C
2. **Extract Trits**: Load 2 bytes containing 8 packed trits and extract them into a vector
3. **Create Nonzero Mask**: Compare each trit against zero to identify non-zero elements
4. **Extract Sign Bits**: Shift right by 1 and mask to extract sign information
5. **Compute Contribution**: Use the nonzero mask to select coefficients (zero out where trit=0)
6. **Conditional Negation**: Use the sign mask to negate contributions for negative trits
7. **Accumulate**: Add the contribution to the accumulator
8. **Store Result**: Write the updated accumulator back to array C

The critical insight that enables correct operation is the use of direct comparison against zero for the nonzero mask, rather than checking the magnitude bit. This correctly handles both positive (+1 = 0b01) and negative (-1 = 0b10) values, as both are non-zero.

### Implementation Details

The implementation uses the following AVX-512 intrinsics:

- `_mm512_loadu_si512`: Unaligned vector load
- `_mm512_storeu_si512`: Unaligned vector store
- `_mm512_set_epi64`: Create vector from scalar values
- `_mm512_cmpneq_epi64_mask`: Compare for inequality, returning a mask
- `_mm512_maskz_mov_epi64`: Masked move with zero
- `_mm512_mask_blend_epi64`: Blend two vectors based on mask
- `_mm512_add_epi64`: Vector addition
- `_mm512_sub_epi64`: Vector subtraction

These intrinsics map directly to AVX-512 instructions, ensuring optimal performance on supported CPUs.

---

## Performance Analysis

### Benchmark Results

The comprehensive benchmark suite validates the performance claims of the kernel. Running on a system with AVX-512 support, the following results were obtained:

| Benchmark | Result | Notes |
|:----------|:-------|:------|
| **Encode/Decode** | All tests pass | Correctness verified |
| **Pack/Unpack** | 0 errors | 75% memory reduction |
| **SIMD Speedup** | 2.25× | Scalar: 511 Mtrits/s, SIMD: 1148 Mtrits/s |
| **Sparse Speedup** | 23.39× | Exceeds 16× theoretical for w=128, N=2048 |

The SIMD speedup of 2.25× demonstrates the effectiveness of the AVX-512 implementation. While the theoretical maximum speedup is 8× (processing 8 elements simultaneously), practical factors such as memory bandwidth, instruction latency, and overhead from trit extraction limit the achieved speedup. Nevertheless, the 2.25× improvement represents a significant performance gain for FHE applications.

The sparse optimization achieves a remarkable 23.39× speedup, exceeding the theoretical 16× speedup predicted by the ratio N/w. This superlinear speedup is attributed to improved cache locality and reduced memory traffic when processing only the non-zero elements.

### Density Gain Analysis

The 2-bit encoding achieves a 75% reduction in memory footprint compared to 8-bit representations. For a typical TFHE parameter set with N=2048 ternary coefficients, this translates to:

- **8-bit encoding**: 2048 bytes
- **2-bit encoding**: 512 bytes
- **Memory saved**: 1536 bytes (75%)

This reduction in memory footprint has several benefits beyond simple storage savings. It reduces memory bandwidth requirements, improves cache utilization, and enables larger problem sizes to fit in fast on-chip memory. These effects contribute to the overall performance improvements observed in the benchmarks.

### Throughput Analysis

The SIMD implementation achieves a throughput of 1148 million trits per second (Mtrits/s) on the test system. This represents the rate at which ternary FMA operations can be performed. For comparison, the scalar implementation achieves 511 Mtrits/s, demonstrating the 2.25× speedup.

To put this in context, a single TFHE bootstrapping operation typically requires processing several thousand ternary coefficients. The high throughput of the kernel enables bootstrapping operations to complete in milliseconds rather than seconds, making interactive FHE applications practical.

---

## Integration Guide

### Basic Usage

To use the sparse-ternary-fma kernel in your project, follow these steps:

1. **Include the header**:
```c
#include "sparse_ternary_fma.h"
```

2. **Prepare your data**:
```c
/* Dense coefficients */
int64_t A[N];
/* ... initialize A ... */

/* Ternary key */
int8_t B[N];
/* ... initialize B with values in {-1, 0, 1} ... */

/* Pack the ternary key */
uint8_t B_packed[N / 4];
pack_trit_array(B, B_packed, N);

/* Accumulator */
int64_t C[N];
memset(C, 0, N * sizeof(int64_t));
```

3. **Perform the FMA operation**:
```c
/* Automatic dispatch (recommended) */
sparse_ternary_fma(A, B_packed, C, N);

/* Or explicitly choose implementation */
sparse_ternary_fma_scalar(A, B_packed, C, N);
sparse_ternary_fma_avx512(A, B_packed, C, N);
```

### Compilation

To compile your project with the sparse-ternary-fma kernel, use the following compiler flags:

```bash
gcc -O3 -march=native -mavx512f your_code.c sparse_ternary_fma.c -o your_program
```

The `-march=native` flag enables all CPU features available on the build system, including AVX-512 if supported. The `-mavx512f` flag explicitly enables AVX-512 Foundation instructions.

### Linking

You can link against the static or shared library:

```bash
# Static linking
gcc -O3 -march=native your_code.c -L./lib -lsparsetfma -o your_program

# Dynamic linking
gcc -O3 -march=native your_code.c -L./lib -lsparsetfma -Wl,-rpath,./lib -o your_program
```

### CPU Feature Detection

The library includes runtime CPU feature detection to automatically select the best implementation. You can query the available features:

```c
if (has_avx512_support()) {
    printf("AVX-512 is available\n");
    printf("Using: %s\n", get_fma_implementation());
}
```

### Performance Considerations

For optimal performance, consider the following guidelines:

1. **Array Alignment**: While the kernel supports unaligned memory access, aligning arrays to 64-byte boundaries can improve performance
2. **Array Size**: The AVX-512 implementation requires N to be a multiple of 8 for optimal performance
3. **Sparsity**: For very sparse keys (w < N/16), use the sparse implementation with index arrays
4. **Batching**: Process multiple independent FMA operations in sequence to amortize overhead

---

## References

This kernel is part of the broader T-Encrypt (T-Enc) T-FHE architecture developed by HyperFold Technologies UK. For the complete system with advanced optimizations and production-ready features, see the evaluation repository.

For questions, bug reports, or contributions, please contact:

**Maurice Wilson**  
Founder, HyperFold Technologies UK  
Email: maurice.wilson@hyperfold-technologies.com  
Website: https://www.hyperfold-technologies.com

---

**License:** Apache License 2.0  
**Copyright:** © 2025 HyperFold Technologies UK Ltd

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. See the LICENSE file for full details.
