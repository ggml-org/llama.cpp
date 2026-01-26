/**
 * @file test-op-context.cpp
 * @brief Unit tests for OperationContext builder
 *
 * Tests build_op_context() function that extracts M, N, K from tensor shapes
 * for kernel dispatch and auto-tuning.
 *
 * TDD approach: These tests are written BEFORE implementation.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include "ggml.h"

// Include the header we're testing
#include "op-context.hpp"

using namespace ggml_sycl::dispatch;

// Helper to create a mock tensor with specified dimensions
// Note: This creates a minimal tensor structure for testing shape extraction
static ggml_tensor * create_mock_tensor(ggml_context * ctx,
                                         ggml_type type,
                                         int64_t ne0, int64_t ne1,
                                         int64_t ne2 = 1, int64_t ne3 = 1) {
    return ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
}

// =============================================================================
// Test: Simple 1x4096x4096 matrix multiplication
// =============================================================================
static bool test_op_context_simple() {
    printf("  test_op_context_simple...");

    // Setup GGML context
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    // Create tensors for matmul: dst = src0 @ src1
    // GGML convention for MUL_MAT:
    //   src0 (weights): [K, N, 1, 1] - K cols, N rows
    //   src1 (activations): [K, M, 1, 1] - K cols, M rows (batch)
    //   dst: [N, M, 1, 1] - N cols, M rows
    //
    // For simple case: batch=1, hidden_dim=4096, output_dim=4096
    // M=1, N=4096, K=4096
    const int64_t K = 4096;  // Inner dimension
    const int64_t N = 4096;  // Output columns (hidden dim)
    const int64_t M = 1;     // Output rows (batch * tokens)

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q4_0, K, N);  // weights
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);   // activations
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);   // output

    // Build operation context
    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

    // Verify extracted dimensions
    bool passed = true;
    if (op_ctx.M != M) {
        printf(" FAILED: M=%lld expected %lld\n", (long long)op_ctx.M, (long long)M);
        passed = false;
    }
    if (op_ctx.N != N) {
        printf(" FAILED: N=%lld expected %lld\n", (long long)op_ctx.N, (long long)N);
        passed = false;
    }
    if (op_ctx.K != K) {
        printf(" FAILED: K=%lld expected %lld\n", (long long)op_ctx.K, (long long)K);
        passed = false;
    }
    if (op_ctx.weight_type != GGML_TYPE_Q4_0) {
        printf(" FAILED: weight_type=%d expected %d\n", op_ctx.weight_type, GGML_TYPE_Q4_0);
        passed = false;
    }
    if (op_ctx.activation_type != GGML_TYPE_F32) {
        printf(" FAILED: activation_type=%d expected %d\n", op_ctx.activation_type, GGML_TYPE_F32);
        passed = false;
    }
    if (op_ctx.batch_size != 1) {
        printf(" FAILED: batch_size=%d expected 1\n", op_ctx.batch_size);
        passed = false;
    }
    if (op_ctx.device_id != 0) {
        printf(" FAILED: device_id=%u expected 0\n", op_ctx.device_id);
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: Batched 32x4096x4096 matrix multiplication
// =============================================================================
static bool test_op_context_batched() {
    printf("  test_op_context_batched...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    // Batched matmul: batch=32
    const int64_t K = 4096;
    const int64_t N = 4096;
    const int64_t M = 32;  // batch size

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/1);

    bool passed = true;
    if (op_ctx.M != M) {
        printf(" FAILED: M=%lld expected %lld\n", (long long)op_ctx.M, (long long)M);
        passed = false;
    }
    if (op_ctx.N != N) {
        printf(" FAILED: N=%lld expected %lld\n", (long long)op_ctx.N, (long long)N);
        passed = false;
    }
    if (op_ctx.K != K) {
        printf(" FAILED: K=%lld expected %lld\n", (long long)op_ctx.K, (long long)K);
        passed = false;
    }
    if (op_ctx.batch_size != 32) {
        printf(" FAILED: batch_size=%d expected 32\n", op_ctx.batch_size);
        passed = false;
    }
    if (op_ctx.weight_type != GGML_TYPE_Q8_0) {
        printf(" FAILED: weight_type=%d expected %d\n", op_ctx.weight_type, GGML_TYPE_Q8_0);
        passed = false;
    }
    if (op_ctx.device_id != 1) {
        printf(" FAILED: device_id=%u expected 1\n", op_ctx.device_id);
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: Large batch (prompt processing scenario)
// =============================================================================
static bool test_op_context_large_batch() {
    printf("  test_op_context_large_batch...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    // Large batch (prompt processing): batch=512
    const int64_t K = 4096;
    const int64_t N = 11008;  // FFN intermediate dim (e.g., Mistral 7B)
    const int64_t M = 512;    // Large prompt

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q6_K, K, N);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

    bool passed = true;
    if (op_ctx.M != M) {
        printf(" FAILED: M=%lld expected %lld\n", (long long)op_ctx.M, (long long)M);
        passed = false;
    }
    if (op_ctx.N != N) {
        printf(" FAILED: N=%lld expected %lld\n", (long long)op_ctx.N, (long long)N);
        passed = false;
    }
    if (op_ctx.K != K) {
        printf(" FAILED: K=%lld expected %lld\n", (long long)op_ctx.K, (long long)K);
        passed = false;
    }
    if (op_ctx.batch_size != 512) {
        printf(" FAILED: batch_size=%d expected 512\n", op_ctx.batch_size);
        passed = false;
    }
    if (op_ctx.weight_type != GGML_TYPE_Q6_K) {
        printf(" FAILED: weight_type=%d expected %d\n", op_ctx.weight_type, GGML_TYPE_Q6_K);
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: Multi-dimensional batch (ne[2], ne[3] > 1)
// =============================================================================
static bool test_op_context_multidim_batch() {
    printf("  test_op_context_multidim_batch...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    // Multi-dimensional batch: ne[1]=4, ne[2]=8 -> effective batch = 4*8 = 32
    const int64_t K = 4096;
    const int64_t N = 4096;
    const int64_t M = 4;
    const int64_t batch2 = 8;

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q4_0, K, N, 1, 1);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M, batch2, 1);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M, batch2, 1);

    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

    // M should be the product of ne[1] * ne[2] * ne[3] for activations
    const int64_t expected_M = M * batch2;

    bool passed = true;
    if (op_ctx.M != expected_M) {
        printf(" FAILED: M=%lld expected %lld\n", (long long)op_ctx.M, (long long)expected_M);
        passed = false;
    }
    if (op_ctx.N != N) {
        printf(" FAILED: N=%lld expected %lld\n", (long long)op_ctx.N, (long long)N);
        passed = false;
    }
    if (op_ctx.K != K) {
        printf(" FAILED: K=%lld expected %lld\n", (long long)op_ctx.K, (long long)K);
        passed = false;
    }
    // batch_size should reflect the actual batch dimension
    if (op_ctx.batch_size != (int)expected_M) {
        printf(" FAILED: batch_size=%d expected %d\n", op_ctx.batch_size, (int)expected_M);
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: Contiguity detection
// =============================================================================
static bool test_op_context_contiguous() {
    printf("  test_op_context_contiguous...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    // Standard contiguous tensors
    const int64_t K = 4096;
    const int64_t N = 4096;
    const int64_t M = 1;

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

    bool passed = true;
    // New tensors should be contiguous
    if (!op_ctx.is_contiguous) {
        printf(" FAILED: is_contiguous=false expected true\n");
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: Different quantization types
// =============================================================================
static bool test_op_context_quant_types() {
    printf("  test_op_context_quant_types...");

    struct ggml_init_params params = {
        .mem_size   = 2 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    const int64_t K = 4096;
    const int64_t N = 4096;
    const int64_t M = 1;

    // Test various quantization types
    ggml_type quant_types[] = {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_F16,
    };

    bool passed = true;
    for (size_t i = 0; i < sizeof(quant_types) / sizeof(quant_types[0]); i++) {
        ggml_type qtype = quant_types[i];

        ggml_tensor * src0 = create_mock_tensor(ctx, qtype, K, N);
        ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);
        ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

        OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

        if (op_ctx.weight_type != qtype) {
            printf(" FAILED: weight_type=%d expected %d for type %s\n",
                   op_ctx.weight_type, qtype, ggml_type_name(qtype));
            passed = false;
        }
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: Performance (<1us per call)
// =============================================================================
static bool test_op_context_performance() {
    printf("  test_op_context_performance...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    const int64_t K = 4096;
    const int64_t N = 4096;
    const int64_t M = 32;

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

    // Warmup
    for (int i = 0; i < 100; i++) {
        OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);
        (void)op_ctx;
    }

    // Measure
    const int iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);
        (void)op_ctx;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_ns = (double)duration_ns / iterations;

    ggml_free(ctx);

    // Requirement: <1us per call (1000ns)
    // We allow some margin for different systems
    const double threshold_ns = 1000.0;  // 1 microsecond

    if (avg_ns > threshold_ns) {
        printf(" FAILED: %.1fns per call (threshold: %.1fns)\n", avg_ns, threshold_ns);
        return false;
    }

    printf(" PASSED (%.1fns per call)\n", avg_ns);
    return true;
}

// =============================================================================
// Test: Edge case - small dimensions
// =============================================================================
static bool test_op_context_small_dims() {
    printf("  test_op_context_small_dims...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    // Small model dimensions (e.g., embedding layer)
    const int64_t K = 768;   // Small hidden dim
    const int64_t N = 32000; // Vocab size
    const int64_t M = 1;

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

    bool passed = true;
    if (op_ctx.M != M) {
        printf(" FAILED: M=%lld expected %lld\n", (long long)op_ctx.M, (long long)M);
        passed = false;
    }
    if (op_ctx.N != N) {
        printf(" FAILED: N=%lld expected %lld\n", (long long)op_ctx.N, (long long)N);
        passed = false;
    }
    if (op_ctx.K != K) {
        printf(" FAILED: K=%lld expected %lld\n", (long long)op_ctx.K, (long long)K);
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Test: FP16 activations
// =============================================================================
static bool test_op_context_fp16_activations() {
    printf("  test_op_context_fp16_activations...");

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf(" FAILED (ggml_init)\n");
        return false;
    }

    const int64_t K = 4096;
    const int64_t N = 4096;
    const int64_t M = 8;

    ggml_tensor * src0 = create_mock_tensor(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * src1 = create_mock_tensor(ctx, GGML_TYPE_F16, K, M);
    ggml_tensor * dst  = create_mock_tensor(ctx, GGML_TYPE_F32, N, M);

    OperationContext op_ctx = build_op_context(src0, src1, dst, /*device_id=*/0);

    bool passed = true;
    if (op_ctx.weight_type != GGML_TYPE_F16) {
        printf(" FAILED: weight_type=%d expected %d\n", op_ctx.weight_type, GGML_TYPE_F16);
        passed = false;
    }
    if (op_ctx.activation_type != GGML_TYPE_F16) {
        printf(" FAILED: activation_type=%d expected %d\n", op_ctx.activation_type, GGML_TYPE_F16);
        passed = false;
    }

    ggml_free(ctx);

    if (passed) {
        printf(" PASSED\n");
    }
    return passed;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    printf("=== OperationContext Unit Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_op_context_simple()) passed++; else failed++;
    if (test_op_context_batched()) passed++; else failed++;
    if (test_op_context_large_batch()) passed++; else failed++;
    if (test_op_context_multidim_batch()) passed++; else failed++;
    if (test_op_context_contiguous()) passed++; else failed++;
    if (test_op_context_quant_types()) passed++; else failed++;
    if (test_op_context_performance()) passed++; else failed++;
    if (test_op_context_small_dims()) passed++; else failed++;
    if (test_op_context_fp16_activations()) passed++; else failed++;

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);

    return (failed == 0) ? 0 : 1;
}
