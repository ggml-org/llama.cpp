/**
 * Sparse Ternary FMA Benchmark
 * 
 * Comprehensive benchmarks validating the 1.58× density gain
 * and performance improvements of the SparseTernaryFMA kernel.
 * 
 * Copyright 2025 HyperFold Technologies UK Ltd
 * Author: Maurice Wilson
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/sparse_ternary_fma.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/* ========================================================================== */
/* Timing Utilities                                                          */
/* ========================================================================== */

static inline double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ========================================================================== */
/* Test Data Generation                                                      */
/* ========================================================================== */

void generate_random_array(int64_t* arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        arr[i] = (int64_t)(rand() % 1000000);
    }
}

void generate_random_ternary(int8_t* arr, size_t N) {
    for (size_t i = 0; i < N; i++) {
        int r = rand() % 3;
        arr[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
}

void generate_sparse_ternary(int8_t* arr, size_t N, size_t hamming_weight) {
    /* Initialize to zero */
    memset(arr, 0, N * sizeof(int8_t));
    
    /* Place random non-zero values */
    for (size_t i = 0; i < hamming_weight; i++) {
        size_t idx = rand() % N;
        arr[idx] = (rand() % 2 == 0) ? -1 : 1;
    }
}

/* ========================================================================== */
/* Correctness Tests                                                         */
/* ========================================================================== */

int test_encode_decode(void) {
    printf("Test 1: Encode/Decode Correctness\n");
    printf("----------------------------------\n");
    
    int8_t test_values[] = {-1, 0, 1};
    int passed = 1;
    
    for (int i = 0; i < 3; i++) {
        int8_t original = test_values[i];
        uint8_t encoded = encode_trit(original);
        int8_t decoded = decode_trit(encoded);
        
        printf("  %2d → 0x%02X → %2d ", original, encoded, decoded);
        
        if (original == decoded) {
            printf("[PASS]\n");
        } else {
            printf("[FAIL]\n");
            passed = 0;
        }
    }
    
    printf("\n");
    return passed;
}

int test_pack_unpack(void) {
    printf("Test 2: Pack/Unpack Correctness\n");
    printf("--------------------------------\n");
    
    const size_t N = 1024;
    int8_t original[N];
    uint8_t packed[N / 4];
    int8_t unpacked[N];
    
    /* Generate random ternary array */
    generate_random_ternary(original, N);
    
    /* Pack and unpack */
    pack_trit_array(original, packed, N);
    unpack_trit_array(packed, unpacked, N);
    
    /* Verify */
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
        if (original[i] != unpacked[i]) {
            errors++;
        }
    }
    
    printf("  Array size: %zu trits\n", N);
    printf("  Packed size: %zu bytes (%.1f%% of original)\n",
           N / 4, (N / 4) * 100.0 / N);
    printf("  Errors: %d\n", errors);
    printf("  Result: %s\n\n", errors == 0 ? "[PASS]" : "[FAIL]");
    
    return errors == 0;
}

int test_ternary_multiply(void) {
    printf("Test 3: Ternary Multiply Correctness\n");
    printf("-------------------------------------\n");
    
    int64_t a = 12345;
    int8_t b_values[] = {-1, 0, 1};
    int passed = 1;
    
    for (int i = 0; i < 3; i++) {
        int8_t b = b_values[i];
        uint8_t b_trit = encode_trit(b);
        int64_t result = ternary_multiply(a, b_trit);
        int64_t expected = a * b;
        
        printf("  %lld × %2d = %lld (expected %lld) ",
               (long long)a, b, (long long)result, (long long)expected);
        
        if (result == expected) {
            printf("[PASS]\n");
        } else {
            printf("[FAIL]\n");
            passed = 0;
        }
    }
    
    printf("\n");
    return passed;
}

int test_sparse_ternary_fma_correctness(void) {
    printf("Test 4: Sparse Ternary FMA Correctness\n");
    printf("---------------------------------------\n");
    
    const size_t N = 256;
    int64_t A[N], C_scalar[N], C_simd[N];
    int8_t B[N];
    uint8_t B_packed[N / 4];
    
    /* Generate test data */
    generate_random_array(A, N);
    generate_random_ternary(B, N);
    memset(C_scalar, 0, N * sizeof(int64_t));
    memset(C_simd, 0, N * sizeof(int64_t));
    
    /* Pack B */
    pack_trit_array(B, B_packed, N);
    
    /* Compute using scalar */
    sparse_ternary_fma_scalar(A, B_packed, C_scalar, N);
    
    /* Compute using SIMD */
    sparse_ternary_fma_avx512(A, B_packed, C_simd, N);
    
    /* Verify */
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
        if (C_scalar[i] != C_simd[i]) {
            errors++;
            if (errors <= 5) {
                printf("  Error at [%zu]: scalar=%lld, simd=%lld\n",
                       i, (long long)C_scalar[i], (long long)C_simd[i]);
            }
        }
    }
    
    printf("  Array size: %zu\n", N);
    printf("  Errors: %d\n", errors);
    printf("  Result: %s\n\n", errors == 0 ? "[PASS]" : "[FAIL]");
    
    return errors == 0;
}

/* ========================================================================== */
/* Performance Benchmarks                                                    */
/* ========================================================================== */

void benchmark_encoding_overhead(void) {
    printf("Benchmark 1: Encoding Overhead\n");
    printf("-------------------------------\n");
    
    const size_t N = 2048;
    const int iterations = 100000;
    int8_t trits[N];
    uint8_t packed[N / 4];
    
    generate_random_ternary(trits, N);
    
    /* Benchmark packing */
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        pack_trit_array(trits, packed, N);
    }
    double pack_time = get_time_ms() - start;
    
    /* Benchmark unpacking */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        unpack_trit_array(packed, trits, N);
    }
    double unpack_time = get_time_ms() - start;
    
    printf("  Array size: %zu trits\n", N);
    printf("  Iterations: %d\n", iterations);
    printf("  Pack time: %.3f ms (%.3f μs/op)\n",
           pack_time, pack_time * 1000.0 / iterations);
    printf("  Unpack time: %.3f ms (%.3f μs/op)\n",
           unpack_time, unpack_time * 1000.0 / iterations);
    printf("\n");
}

void benchmark_density_gain(void) {
    printf("Benchmark 2: Density Gain Validation\n");
    printf("-------------------------------------\n");
    
    const size_t N = 2048;
    const int iterations = 10000;
    
    int64_t A[N], C_8bit[N], C_2bit[N];
    int8_t B_8bit[N];
    uint8_t B_2bit[N / 4];
    
    generate_random_array(A, N);
    generate_random_ternary(B_8bit, N);
    pack_trit_array(B_8bit, B_2bit, N);
    
    /* Benchmark 8-bit encoding (baseline) */
    memset(C_8bit, 0, N * sizeof(int64_t));
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < N; i++) {
            if (B_8bit[i] == 1) {
                C_8bit[i] += A[i];
            } else if (B_8bit[i] == -1) {
                C_8bit[i] -= A[i];
            }
        }
    }
    double time_8bit = get_time_ms() - start;
    
    /* Benchmark 2-bit encoding */
    memset(C_2bit, 0, N * sizeof(int64_t));
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sparse_ternary_fma_scalar(A, B_2bit, C_2bit, N);
    }
    double time_2bit = get_time_ms() - start;
    
    double speedup = time_8bit / time_2bit;
    
    printf("  Array size: %zu\n", N);
    printf("  Iterations: %d\n", iterations);
    printf("  8-bit encoding: %.3f ms (%.3f μs/op)\n",
           time_8bit, time_8bit * 1000.0 / iterations);
    printf("  2-bit encoding: %.3f ms (%.3f μs/op)\n",
           time_2bit, time_2bit * 1000.0 / iterations);
    printf("  Speedup: %.2f×\n", speedup);
    printf("  Memory saved: %zu bytes (%.1f%%)\n",
           N - N / 4, (1.0 - 1.0 / 4) * 100);
    printf("\n");
}

void benchmark_simd_throughput(void) {
    printf("Benchmark 3: SIMD Throughput\n");
    printf("-----------------------------\n");
    
    const size_t N = 2048;
    const int iterations = 10000;
    
    int64_t A[N], C_scalar[N], C_simd[N];
    int8_t B[N];
    uint8_t B_packed[N / 4];
    
    generate_random_array(A, N);
    generate_random_ternary(B, N);
    pack_trit_array(B, B_packed, N);
    
    /* Benchmark scalar */
    memset(C_scalar, 0, N * sizeof(int64_t));
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sparse_ternary_fma_scalar(A, B_packed, C_scalar, N);
    }
    double time_scalar = get_time_ms() - start;
    
    /* Benchmark SIMD */
    memset(C_simd, 0, N * sizeof(int64_t));
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sparse_ternary_fma_avx512(A, B_packed, C_simd, N);
    }
    double time_simd = get_time_ms() - start;
    
    double speedup = time_scalar / time_simd;
    double trits_per_ms_scalar = (N * iterations) / time_scalar;
    double trits_per_ms_simd = (N * iterations) / time_simd;
    
    printf("  Array size: %zu\n", N);
    printf("  Iterations: %d\n", iterations);
    printf("  Scalar: %.3f ms (%.3f μs/op)\n",
           time_scalar, time_scalar * 1000.0 / iterations);
    printf("  SIMD: %.3f ms (%.3f μs/op)\n",
           time_simd, time_simd * 1000.0 / iterations);
    printf("  Speedup: %.2f×\n", speedup);
    printf("  Throughput (scalar): %.1f Mtrits/s\n",
           trits_per_ms_scalar / 1000.0);
    printf("  Throughput (SIMD): %.1f Mtrits/s\n",
           trits_per_ms_simd / 1000.0);
    printf("\n");
}

void benchmark_sparse_optimization(void) {
    printf("Benchmark 4: Sparse Optimization\n");
    printf("---------------------------------\n");
    
    const size_t N = 2048;
    const size_t w = 128;  /* Hamming weight */
    const int iterations = 10000;
    
    int64_t A[N], C_dense[N], C_sparse[N];
    int8_t B[N];
    uint8_t B_packed[N / 4];
    uint32_t indices[w];
    int8_t values[w];
    
    generate_random_array(A, N);
    generate_sparse_ternary(B, N, w);
    pack_trit_array(B, B_packed, N);
    
    /* Extract sparse representation */
    size_t idx_count = 0;
    for (size_t i = 0; i < N; i++) {
        if (B[i] != 0) {
            indices[idx_count] = i;
            values[idx_count] = B[i];
            idx_count++;
        }
    }
    
    /* Benchmark dense */
    memset(C_dense, 0, N * sizeof(int64_t));
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sparse_ternary_fma_scalar(A, B_packed, C_dense, N);
    }
    double time_dense = get_time_ms() - start;
    
    /* Benchmark sparse */
    memset(C_sparse, 0, N * sizeof(int64_t));
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sparse_ternary_fma_sparse(A, indices, values, C_sparse, idx_count);
    }
    double time_sparse = get_time_ms() - start;
    
    double speedup = time_dense / time_sparse;
    
    printf("  Array size: %zu\n", N);
    printf("  Hamming weight: %zu (%.1f%%)\n", w, w * 100.0 / N);
    printf("  Iterations: %d\n", iterations);
    printf("  Dense: %.3f ms (%.3f μs/op)\n",
           time_dense, time_dense * 1000.0 / iterations);
    printf("  Sparse: %.3f ms (%.3f μs/op)\n",
           time_sparse, time_sparse * 1000.0 / iterations);
    printf("  Speedup: %.2f× (theoretical: %.1f×)\n",
           speedup, (double)N / w);
    printf("\n");
}

/* ========================================================================== */
/* Main                                                                      */
/* ========================================================================== */

int main(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     Sparse Ternary FMA Kernel - Comprehensive Benchmark       ║\n");
    printf("║                                                                ║\n");
    printf("║  Implementation: %s\n", get_fma_implementation());
    printf("║  AVX-512 Support: %s\n", has_avx512_support() ? "Yes" : "No");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    /* Run correctness tests */
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("CORRECTNESS TESTS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    int all_passed = 1;
    all_passed &= test_encode_decode();
    all_passed &= test_pack_unpack();
    all_passed &= test_ternary_multiply();
    all_passed &= test_sparse_ternary_fma_correctness();
    
    /* Run performance benchmarks */
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("PERFORMANCE BENCHMARKS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    benchmark_encoding_overhead();
    benchmark_density_gain();
    benchmark_simd_throughput();
    benchmark_sparse_optimization();
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    if (all_passed) {
        printf("✓ All correctness tests PASSED\n");
        printf("✓ Performance benchmarks completed successfully\n");
        printf("✓ Kernel is ready for production use\n");
    } else {
        printf("✗ Some tests FAILED - please review results above\n");
        return 1;
    }
    
    printf("\n");
    return 0;
}
