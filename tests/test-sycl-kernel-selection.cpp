// SYCL unified kernel selection test.
// Verifies that ggml_sycl_select_preferred_kernel() consolidates priority logic correctly.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/kernel-selection.hpp"

static bool test_invalid_device_returns_nullopt() {
    printf("  test_invalid_device_returns_nullopt: ");

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("SKIP (ggml_init failed)\n");
        return true;
    }

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 1024, 128);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1);

    // Invalid device ID -1
    auto result = ggml_sycl_select_preferred_kernel(src0, src1, -1, std::nullopt);

    ggml_free(ctx);

    if (result.has_value()) {
        printf("FAIL (expected nullopt for invalid device, got kernel)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_batch1_selects_dmmv_or_mmvq() {
    printf("  test_batch1_selects_dmmv_or_mmvq: ");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP (SYCL backend unavailable)\n");
        return true;
    }

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("SKIP (ggml_init failed)\n");
        ggml_backend_free(backend);
        return true;
    }

    // Q4_0 weight, F32 activation, batch=1
    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 1024, 128);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1);  // batch=1
    ggml_set_name(src0, "test_weight");
    ggml_set_name(src1, "test_input");

    auto result = ggml_sycl_select_preferred_kernel(src0, src1, 0, std::nullopt);

    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!result.has_value()) {
        printf("FAIL (expected kernel selection for batch=1)\n");
        return false;
    }

    // For batch=1, should select DMMV or MMVQ variant (or ONEDNN_AOS as fallback)
    // Without proper buffer setup, SOA/COALESCED layouts won't be available,
    // so AOS variants or ONEDNN fallback are expected
    auto kernel = result.value();
    bool is_valid = (kernel == ggml_sycl_mul_mat_kernel::DMMV_SOA ||
                     kernel == ggml_sycl_mul_mat_kernel::DMMV_COALESCED ||
                     kernel == ggml_sycl_mul_mat_kernel::MMVQ_SOA ||
                     kernel == ggml_sycl_mul_mat_kernel::MMVQ_COALESCED ||
                     kernel == ggml_sycl_mul_mat_kernel::MMVQ_AOS ||
                     kernel == ggml_sycl_mul_mat_kernel::MMQ_AOS ||  // MMQ_AOS also valid for batch=1
                     kernel == ggml_sycl_mul_mat_kernel::ONEDNN_AOS);

    if (!is_valid) {
        printf("FAIL (batch=1 should select DMMV, MMVQ, MMQ_AOS, or ONEDNN fallback, got kernel=%d)\n", static_cast<int>(kernel));
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_force_kernel_override() {
    printf("  test_force_kernel_override: ");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP (SYCL backend unavailable)\n");
        return true;
    }

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("SKIP (ggml_init failed)\n");
        ggml_backend_free(backend);
        return true;
    }

    // Batch size that works with MMQ (MMQ_MAX_BATCH_SIZE=32)
    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 1024, 128);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 16);  // batch=16 < MMQ_MAX_BATCH_SIZE=32
    ggml_set_name(src0, "test_weight_force");
    ggml_set_name(src1, "test_input_force");

    // Force MMQ_AOS (AOS layout works without buffer setup)
    auto result = ggml_sycl_select_preferred_kernel(
        src0, src1, 0,
        std::optional<ggml_sycl_mul_mat_kernel>(ggml_sycl_mul_mat_kernel::MMQ_AOS));

    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!result.has_value()) {
        printf("FAIL (force_kernel should return a kernel, got nullopt)\n");
        return false;
    }

    if (result.value() != ggml_sycl_mul_mat_kernel::MMQ_AOS) {
        printf("FAIL (force_kernel=MMQ_AOS should return MMQ_AOS, got kernel=%d)\n", static_cast<int>(result.value()));
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_parse_force_kernel_valid() {
    printf("  test_parse_force_kernel_valid: ");

    // Set env var
    setenv("GGML_SYCL_FORCE_KERNEL", "MMQ_COALESCED", 1);

    auto result = ggml_sycl_parse_force_kernel();

    // Clean up
    unsetenv("GGML_SYCL_FORCE_KERNEL");

    if (!result.has_value()) {
        printf("FAIL (should parse MMQ_COALESCED)\n");
        return false;
    }

    if (result.value() != ggml_sycl_mul_mat_kernel::MMQ_COALESCED) {
        printf("FAIL (parsed wrong kernel)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_parse_force_kernel_invalid() {
    printf("  test_parse_force_kernel_invalid: ");

    // Set invalid env var
    setenv("GGML_SYCL_FORCE_KERNEL", "INVALID_KERNEL", 1);

    auto result = ggml_sycl_parse_force_kernel();

    // Clean up
    unsetenv("GGML_SYCL_FORCE_KERNEL");

    if (result.has_value()) {
        printf("FAIL (invalid kernel name should return nullopt)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

static bool test_parse_force_kernel_unset() {
    printf("  test_parse_force_kernel_unset: ");

    // Ensure env var is not set
    unsetenv("GGML_SYCL_FORCE_KERNEL");

    auto result = ggml_sycl_parse_force_kernel();

    if (result.has_value()) {
        printf("FAIL (unset env var should return nullopt)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

int main() {
    printf("Running SYCL kernel selection tests...\n\n");

    int passed = 0;
    int failed = 0;
    int total = 6;

    if (test_invalid_device_returns_nullopt()) passed++; else failed++;
    if (test_batch1_selects_dmmv_or_mmvq()) passed++; else failed++;
    if (test_force_kernel_override()) passed++; else failed++;
    if (test_parse_force_kernel_valid()) passed++; else failed++;
    if (test_parse_force_kernel_invalid()) passed++; else failed++;
    if (test_parse_force_kernel_unset()) passed++; else failed++;

    printf("\n%d/%d tests passed\n", passed, total);

    return (failed == 0) ? 0 : 1;
}
