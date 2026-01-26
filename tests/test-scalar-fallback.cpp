//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unit tests for scalar fallback path in unified kernel
//
// Tests:
// 1. should_use_scalar_fallback() logic for various argument combinations
// 2. is_partial_tile() boundary detection
// 3. Reference matmul correctness
// 4. Boundary conditions (smallest possible matrices)
// 5. Partial tile handling
//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Include the unified kernel header for type definitions and fallback functions
#include "../ggml/src/ggml-sycl/unified-kernel.hpp"

using namespace ggml_sycl_unified;

// =============================================================================
// Reference Implementation
// =============================================================================

/**
 * Reference matrix multiplication for verification.
 * C = A * B where A[M,K], B[K,N], C[M,N]
 */
void matmul_reference(const float * A, const float * B, float * C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

/**
 * Compare two matrices for approximate equality.
 */
bool matrices_equal(const float * A, const float * B, int size, float eps = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(A[i] - B[i]) > eps) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Test Cases for should_use_scalar_fallback
// =============================================================================

/**
 * Test 1: Should use scalar for small M (< 8)
 */
bool test_scalar_fallback_small_m() {
    UnifiedKernelArgs args{};
    args.M       = 4;       // Small M
    args.N       = 64;
    args.K       = 64;      // Multiple of 32
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    bool result = should_use_scalar_fallback(args);
    if (result != true) {
        printf("[FAIL] test_scalar_fallback_small_m: expected true, got false\n");
        return false;
    }

    printf("[PASS] test_scalar_fallback_small_m\n");
    return true;
}

/**
 * Test 2: Should use scalar for unaligned K (not multiple of 32)
 */
bool test_scalar_fallback_unaligned_k() {
    UnifiedKernelArgs args{};
    args.M       = 16;
    args.N       = 64;
    args.K       = 100;     // Not multiple of 32
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    bool result = should_use_scalar_fallback(args);
    if (result != true) {
        printf("[FAIL] test_scalar_fallback_unaligned_k: expected true, got false\n");
        return false;
    }

    printf("[PASS] test_scalar_fallback_unaligned_k\n");
    return true;
}

/**
 * Test 3: Should use XMX for aligned case
 */
bool test_use_xmx_aligned() {
    UnifiedKernelArgs args{};
    args.M       = 16;      // >= 8
    args.N       = 64;
    args.K       = 64;      // Multiple of 32
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;    // XMX enabled
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    bool result = should_use_scalar_fallback(args);
    if (result != false) {
        printf("[FAIL] test_use_xmx_aligned: expected false, got true\n");
        return false;
    }

    printf("[PASS] test_use_xmx_aligned\n");
    return true;
}

/**
 * Test 4: Should use scalar when XMX is disabled
 */
bool test_scalar_when_xmx_disabled() {
    UnifiedKernelArgs args{};
    args.M       = 16;
    args.N       = 64;
    args.K       = 64;      // Aligned
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = false;   // XMX disabled
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    bool result = should_use_scalar_fallback(args);
    if (result != true) {
        printf("[FAIL] test_scalar_when_xmx_disabled: expected true, got false\n");
        return false;
    }

    printf("[PASS] test_scalar_when_xmx_disabled\n");
    return true;
}

/**
 * Test 5: Edge case - M exactly 8
 */
bool test_m_exactly_8() {
    UnifiedKernelArgs args{};
    args.M       = 8;       // Exactly 8 (threshold)
    args.N       = 64;
    args.K       = 64;
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    bool result = should_use_scalar_fallback(args);
    if (result != false) {
        printf("[FAIL] test_m_exactly_8: expected false (XMX), got true (scalar)\n");
        return false;
    }

    printf("[PASS] test_m_exactly_8\n");
    return true;
}

/**
 * Test 6: Edge case - K exactly 32
 */
bool test_k_exactly_32() {
    UnifiedKernelArgs args{};
    args.M       = 16;
    args.N       = 64;
    args.K       = 32;      // Exactly 32 (aligned)
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    bool result = should_use_scalar_fallback(args);
    if (result != false) {
        printf("[FAIL] test_k_exactly_32: expected false (XMX), got true (scalar)\n");
        return false;
    }

    printf("[PASS] test_k_exactly_32\n");
    return true;
}

// =============================================================================
// Test Cases for is_partial_tile
// =============================================================================

/**
 * Test 7: Full tile (not partial)
 */
bool test_full_tile() {
    // Matrix 64x64, tile 8x16, starting at (0,0) - full tile
    bool result = is_partial_tile(0, 0, 0, 8, 16, 32, 64, 64, 64);
    if (result != false) {
        printf("[FAIL] test_full_tile: expected false (full tile), got true\n");
        return false;
    }

    printf("[PASS] test_full_tile\n");
    return true;
}

/**
 * Test 8: Partial tile at M boundary
 */
bool test_partial_tile_m_boundary() {
    // Matrix M=60, tile starting at m=56 with tile_m=8 would extend to m=64
    bool result = is_partial_tile(56, 0, 0, 8, 16, 32, 60, 64, 64);
    if (result != true) {
        printf("[FAIL] test_partial_tile_m_boundary: expected true (partial), got false\n");
        return false;
    }

    printf("[PASS] test_partial_tile_m_boundary\n");
    return true;
}

/**
 * Test 9: Partial tile at N boundary
 */
bool test_partial_tile_n_boundary() {
    // Matrix N=60, tile starting at n=48 with tile_n=16 would extend to n=64
    bool result = is_partial_tile(0, 48, 0, 8, 16, 32, 64, 60, 64);
    if (result != true) {
        printf("[FAIL] test_partial_tile_n_boundary: expected true (partial), got false\n");
        return false;
    }

    printf("[PASS] test_partial_tile_n_boundary\n");
    return true;
}

/**
 * Test 10: Partial tile at K boundary
 */
bool test_partial_tile_k_boundary() {
    // Matrix K=48, tile starting at k=32 with tile_k=32 would extend to k=64
    bool result = is_partial_tile(0, 0, 32, 8, 16, 32, 64, 64, 48);
    if (result != true) {
        printf("[FAIL] test_partial_tile_k_boundary: expected true (partial), got false\n");
        return false;
    }

    printf("[PASS] test_partial_tile_k_boundary\n");
    return true;
}

// =============================================================================
// Test Cases for Reference Matmul
// =============================================================================

/**
 * Test 11: Reference matmul with identity matrix
 */
bool test_reference_matmul_identity() {
    constexpr int M = 4, N = 4, K = 4;

    // Identity matrix * B = B
    float A[M * K] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float B[K * N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float C[M * N] = {0};

    matmul_reference(A, B, C, M, N, K);

    if (!matrices_equal(C, B, M * N)) {
        printf("[FAIL] test_reference_matmul_identity: I * B != B\n");
        for (int i = 0; i < M * N; i++) {
            printf("  C[%d] = %.6f, B[%d] = %.6f\n", i, C[i], i, B[i]);
        }
        return false;
    }

    printf("[PASS] test_reference_matmul_identity\n");
    return true;
}

/**
 * Test 12: Reference matmul with known values
 */
bool test_reference_matmul_known() {
    constexpr int M = 2, N = 2, K = 3;

    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 10], [11, 12]]
    // C = A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //           = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
    //           = [[58, 64], [139, 154]]

    float A[M * K] = {1, 2, 3, 4, 5, 6};
    float B[K * N] = {7, 8, 9, 10, 11, 12};
    float C[M * N] = {0};
    float expected[M * N] = {58, 64, 139, 154};

    matmul_reference(A, B, C, M, N, K);

    if (!matrices_equal(C, expected, M * N)) {
        printf("[FAIL] test_reference_matmul_known\n");
        for (int i = 0; i < M * N; i++) {
            printf("  C[%d] = %.6f, expected[%d] = %.6f\n", i, C[i], i, expected[i]);
        }
        return false;
    }

    printf("[PASS] test_reference_matmul_known\n");
    return true;
}

// =============================================================================
// Test Cases for Boundary Conditions
// =============================================================================

/**
 * Test 13: Smallest possible matrix (1x1x1)
 */
bool test_boundary_1x1x1() {
    float A[1] = {2.0f};
    float B[1] = {3.0f};
    float C[1] = {0};

    matmul_reference(A, B, C, 1, 1, 1);

    if (std::abs(C[0] - 6.0f) > 1e-5f) {
        printf("[FAIL] test_boundary_1x1x1: expected 6.0, got %.6f\n", C[0]);
        return false;
    }

    printf("[PASS] test_boundary_1x1x1\n");
    return true;
}

/**
 * Test 14: Single row, multiple columns (1xNxK)
 */
bool test_boundary_single_row() {
    constexpr int M = 1, N = 4, K = 3;

    float A[M * K] = {1, 2, 3};
    float B[K * N] = {
        1, 2, 3, 4,     // row 0
        5, 6, 7, 8,     // row 1
        9, 10, 11, 12   // row 2
    };
    float C[M * N] = {0};

    // C[0,n] = sum_k A[0,k] * B[k,n]
    // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
    // C[0,1] = 1*2 + 2*6 + 3*10 = 2 + 12 + 30 = 44
    // C[0,2] = 1*3 + 2*7 + 3*11 = 3 + 14 + 33 = 50
    // C[0,3] = 1*4 + 2*8 + 3*12 = 4 + 16 + 36 = 56
    float expected[M * N] = {38, 44, 50, 56};

    matmul_reference(A, B, C, M, N, K);

    if (!matrices_equal(C, expected, M * N)) {
        printf("[FAIL] test_boundary_single_row\n");
        for (int i = 0; i < M * N; i++) {
            printf("  C[%d] = %.6f, expected[%d] = %.6f\n", i, C[i], i, expected[i]);
        }
        return false;
    }

    printf("[PASS] test_boundary_single_row\n");
    return true;
}

/**
 * Test 15: Single column, multiple rows (Mx1xK)
 */
bool test_boundary_single_column() {
    constexpr int M = 3, N = 1, K = 4;

    float A[M * K] = {
        1, 2, 3, 4,     // row 0
        5, 6, 7, 8,     // row 1
        9, 10, 11, 12   // row 2
    };
    float B[K * N] = {1, 2, 3, 4};
    float C[M * N] = {0};

    // C[m,0] = sum_k A[m,k] * B[k,0]
    // C[0,0] = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    // C[1,0] = 5*1 + 6*2 + 7*3 + 8*4 = 5 + 12 + 21 + 32 = 70
    // C[2,0] = 9*1 + 10*2 + 11*3 + 12*4 = 9 + 20 + 33 + 48 = 110
    float expected[M * N] = {30, 70, 110};

    matmul_reference(A, B, C, M, N, K);

    if (!matrices_equal(C, expected, M * N)) {
        printf("[FAIL] test_boundary_single_column\n");
        for (int i = 0; i < M * N; i++) {
            printf("  C[%d] = %.6f, expected[%d] = %.6f\n", i, C[i], i, expected[i]);
        }
        return false;
    }

    printf("[PASS] test_boundary_single_column\n");
    return true;
}

// =============================================================================
// Test Cases for Partial Tile Handling
// =============================================================================

/**
 * Test 16: Partial tile (unaligned dimensions)
 */
bool test_partial_tile_unaligned() {
    // 3x3 output, 5-element inner dim (not aligned to typical tile sizes)
    constexpr int M = 3, N = 3, K = 5;

    std::vector<float> A(M * K, 1.0f);  // All ones
    std::vector<float> B(K * N, 1.0f);  // All ones
    std::vector<float> C(M * N, 0.0f);

    matmul_reference(A.data(), B.data(), C.data(), M, N, K);

    // Each output should be K (sum of K ones times ones)
    for (int i = 0; i < M * N; i++) {
        if (std::abs(C[i] - static_cast<float>(K)) > 1e-5f) {
            printf("[FAIL] test_partial_tile_unaligned: C[%d] = %.6f, expected %.6f\n",
                   i, C[i], static_cast<float>(K));
            return false;
        }
    }

    printf("[PASS] test_partial_tile_unaligned\n");
    return true;
}

/**
 * Test 17: Non-square matrix (M > N)
 */
bool test_nonsquare_m_greater() {
    constexpr int M = 8, N = 4, K = 6;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> expected(M * N);

    // Initialize with sequential values
    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>(i + 1);

    matmul_reference(A.data(), B.data(), C.data(), M, N, K);

    // Verify dimensions are handled correctly (no crashes, reasonable output)
    bool valid = true;
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(C[i]) || std::isinf(C[i])) {
            printf("[FAIL] test_nonsquare_m_greater: C[%d] is NaN or Inf\n", i);
            valid = false;
            break;
        }
    }

    if (!valid) return false;

    printf("[PASS] test_nonsquare_m_greater\n");
    return true;
}

/**
 * Test 18: Non-square matrix (N > M)
 */
bool test_nonsquare_n_greater() {
    constexpr int M = 4, N = 8, K = 6;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    // Initialize with sequential values
    for (int i = 0; i < M * K; i++) A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < K * N; i++) B[i] = static_cast<float>(i + 1);

    matmul_reference(A.data(), B.data(), C.data(), M, N, K);

    // Verify dimensions are handled correctly
    bool valid = true;
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(C[i]) || std::isinf(C[i])) {
            printf("[FAIL] test_nonsquare_n_greater: C[%d] is NaN or Inf\n", i);
            valid = false;
            break;
        }
    }

    if (!valid) return false;

    printf("[PASS] test_nonsquare_n_greater\n");
    return true;
}

// =============================================================================
// Test Cases for Validate Args
// =============================================================================

/**
 * Test 19: Valid args pass validation
 */
bool test_validate_args_valid() {
    UnifiedKernelArgs args{};
    args.M       = 16;
    args.N       = 64;
    args.K       = 64;      // Multiple of QK4_0 (32)
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    // Need non-null pointers for validation
    float dummy_weights, dummy_activations, dummy_output;
    args.weights     = &dummy_weights;
    args.activations = &dummy_activations;
    args.output      = &dummy_output;

    bool result = validate_args(args);
    if (result != true) {
        printf("[FAIL] test_validate_args_valid: expected true, got false\n");
        return false;
    }

    printf("[PASS] test_validate_args_valid\n");
    return true;
}

/**
 * Test 20: Invalid K (not multiple of QK4_0 for Q4_0)
 */
bool test_validate_args_invalid_k() {
    UnifiedKernelArgs args{};
    args.M       = 16;
    args.N       = 64;
    args.K       = 50;      // Not multiple of QK4_0 (32)
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    float dummy_weights, dummy_activations, dummy_output;
    args.weights     = &dummy_weights;
    args.activations = &dummy_activations;
    args.output      = &dummy_output;

    bool result = validate_args(args);
    if (result != false) {
        printf("[FAIL] test_validate_args_invalid_k: expected false, got true\n");
        return false;
    }

    printf("[PASS] test_validate_args_invalid_k\n");
    return true;
}

/**
 * Test 21: Null pointer fails validation
 */
bool test_validate_args_null_pointer() {
    UnifiedKernelArgs args{};
    args.M       = 16;
    args.N       = 64;
    args.K       = 64;
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    // Leave weights null
    args.weights     = nullptr;
    float dummy_activations, dummy_output;
    args.activations = &dummy_activations;
    args.output      = &dummy_output;

    bool result = validate_args(args);
    if (result != false) {
        printf("[FAIL] test_validate_args_null_pointer: expected false, got true\n");
        return false;
    }

    printf("[PASS] test_validate_args_null_pointer\n");
    return true;
}

/**
 * Test 22: Zero dimension fails validation
 */
bool test_validate_args_zero_dim() {
    UnifiedKernelArgs args{};
    args.M       = 0;       // Invalid
    args.N       = 64;
    args.K       = 64;
    args.tile_m  = 8;
    args.tile_n  = 16;
    args.tile_k  = 32;
    args.use_xmx = true;
    args.layout_mode = LAYOUT_NONE;
    args.quant_type  = QUANT_TYPE_Q4_0;

    float dummy_weights, dummy_activations, dummy_output;
    args.weights     = &dummy_weights;
    args.activations = &dummy_activations;
    args.output      = &dummy_output;

    bool result = validate_args(args);
    if (result != false) {
        printf("[FAIL] test_validate_args_zero_dim: expected false, got true\n");
        return false;
    }

    printf("[PASS] test_validate_args_zero_dim\n");
    return true;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    printf("=============================================================\n");
    printf("  Scalar Fallback Unit Tests for Unified Kernel\n");
    printf("=============================================================\n\n");

    int passed = 0;
    int failed = 0;

    // should_use_scalar_fallback tests
    printf("--- should_use_scalar_fallback tests ---\n");
    if (test_scalar_fallback_small_m()) passed++; else failed++;
    if (test_scalar_fallback_unaligned_k()) passed++; else failed++;
    if (test_use_xmx_aligned()) passed++; else failed++;
    if (test_scalar_when_xmx_disabled()) passed++; else failed++;
    if (test_m_exactly_8()) passed++; else failed++;
    if (test_k_exactly_32()) passed++; else failed++;

    // is_partial_tile tests
    printf("\n--- is_partial_tile tests ---\n");
    if (test_full_tile()) passed++; else failed++;
    if (test_partial_tile_m_boundary()) passed++; else failed++;
    if (test_partial_tile_n_boundary()) passed++; else failed++;
    if (test_partial_tile_k_boundary()) passed++; else failed++;

    // Reference matmul tests
    printf("\n--- Reference matmul tests ---\n");
    if (test_reference_matmul_identity()) passed++; else failed++;
    if (test_reference_matmul_known()) passed++; else failed++;

    // Boundary condition tests
    printf("\n--- Boundary condition tests ---\n");
    if (test_boundary_1x1x1()) passed++; else failed++;
    if (test_boundary_single_row()) passed++; else failed++;
    if (test_boundary_single_column()) passed++; else failed++;

    // Partial tile tests
    printf("\n--- Partial tile tests ---\n");
    if (test_partial_tile_unaligned()) passed++; else failed++;
    if (test_nonsquare_m_greater()) passed++; else failed++;
    if (test_nonsquare_n_greater()) passed++; else failed++;

    // Validate args tests
    printf("\n--- validate_args tests ---\n");
    if (test_validate_args_valid()) passed++; else failed++;
    if (test_validate_args_invalid_k()) passed++; else failed++;
    if (test_validate_args_null_pointer()) passed++; else failed++;
    if (test_validate_args_zero_dim()) passed++; else failed++;

    // Summary
    printf("\n=============================================================\n");
    printf("  Results: %d passed, %d failed\n", passed, failed);
    printf("=============================================================\n");

    return failed > 0 ? 1 : 0;
}
