#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "../ggml/src/ggml-sycl/ab-validation.hpp"

using namespace ggml_sycl;

// Test 1: Identical arrays pass
bool test_identical_pass() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto result = compare_outputs(a, a, 4);

    assert(result.passed == true);
    assert(result.mismatches == 0);
    assert(result.max_abs_diff == 0.0f);

    std::cout << "[PASS] test_identical_pass\n";
    return true;
}

// Test 2: Small diff within tolerance
bool test_small_diff_pass() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {1.0f + 1e-7f, 2.0f, 3.0f - 1e-7f, 4.0f};

    auto result = compare_outputs(a, b, 4);
    assert(result.passed == true);

    std::cout << "[PASS] test_small_diff_pass\n";
    return true;
}

// Test 3: Large diff fails
bool test_large_diff_fail() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {1.0f, 2.5f, 3.0f, 4.0f};  // 0.5 diff at index 1

    auto result = compare_outputs(a, b, 4);
    assert(result.passed == false);
    assert(result.mismatches == 1);
    assert(result.worst_index == 1);

    std::cout << "[PASS] test_large_diff_fail\n";
    return true;
}

// Test 4: ValidationResult fields populated
bool test_result_fields() {
    float a[] = {1.0f, 2.0f, 10.0f};
    float b[] = {1.0f, 2.1f, 10.0f};  // Diff at index 1

    auto result = compare_outputs(a, b, 3);

    assert(result.total_elements == 3);
    assert(std::abs(result.max_abs_diff - 0.1f) < 1e-5f);
    assert(result.worst_index == 1);
    assert(std::abs(result.unified_at_worst - 2.0f) < 1e-6f);
    assert(std::abs(result.reference_at_worst - 2.1f) < 1e-6f);

    std::cout << "[PASS] test_result_fields\n";
    return true;
}

// Test 5: Relative tolerance
bool test_relative_tolerance() {
    float a[] = {1000.0f};
    float b[] = {1000.5f};  // 0.05% difference

    ValidationTolerance tol;
    tol.relative_pct = 0.1f;  // 0.1% tolerance

    auto result = compare_outputs(a, b, 1, tol);
    assert(result.passed == true);  // Within relative tolerance

    std::cout << "[PASS] test_relative_tolerance\n";
    return true;
}

// Test 6: ABValidationContext tracking
bool test_context_tracking() {
    ABValidationContext ctx;

    ValidationResult pass1;
    pass1.passed = true;
    pass1.total_elements = 100;
    pass1.max_abs_diff = 1e-7f;

    ValidationResult fail1;
    fail1.passed = false;
    fail1.total_elements = 100;
    fail1.mismatches = 5;
    fail1.max_abs_diff = 0.1f;

    ctx.record_result(pass1);
    ctx.record_result(fail1);

    assert(ctx.get_passed() == 1);
    assert(ctx.get_failed() == 1);
    assert(ctx.get_total() == 2);
    assert(ctx.get_pass_rate() == 50.0f);
    assert(std::abs(ctx.get_max_diff() - 0.1f) < 1e-6f);

    std::cout << "[PASS] test_context_tracking\n";
    return true;
}

// Test 7: Context reset
bool test_context_reset() {
    ABValidationContext ctx;

    ValidationResult r;
    r.passed = true;
    r.total_elements = 100;
    ctx.record_result(r);

    ctx.reset();

    assert(ctx.get_total() == 0);
    assert(ctx.get_passed() == 0);

    std::cout << "[PASS] test_context_reset\n";
    return true;
}

// Test 8: FP16 tolerance
bool test_fp16_tolerance() {
    auto tol = get_tolerance(true);  // FP16
    assert(tol.absolute_fp32 == 1e-4f);
    assert(tol.relative_pct == 0.5f);

    auto tol32 = get_tolerance(false);  // FP32
    assert(tol32.absolute_fp32 == 1e-6f);

    std::cout << "[PASS] test_fp16_tolerance\n";
    return true;
}

// Test 9: Empty array
bool test_empty_array() {
    auto result = compare_outputs(nullptr, nullptr, 0);
    assert(result.passed == true);
    assert(result.total_elements == 0);

    std::cout << "[PASS] test_empty_array\n";
    return true;
}

// Test 10: All mismatches
bool test_all_mismatch() {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {10.0f, 20.0f, 30.0f};

    auto result = compare_outputs(a, b, 3);
    assert(result.passed == false);
    assert(result.mismatches == 3);

    std::cout << "[PASS] test_all_mismatch\n";
    return true;
}

int main() {
    int passed = 0, failed = 0;

    if (test_identical_pass()) passed++; else failed++;
    if (test_small_diff_pass()) passed++; else failed++;
    if (test_large_diff_fail()) passed++; else failed++;
    if (test_result_fields()) passed++; else failed++;
    if (test_relative_tolerance()) passed++; else failed++;
    if (test_context_tracking()) passed++; else failed++;
    if (test_context_reset()) passed++; else failed++;
    if (test_fp16_tolerance()) passed++; else failed++;
    if (test_empty_array()) passed++; else failed++;
    if (test_all_mismatch()) passed++; else failed++;

    std::cout << "\n=== A/B Validation Tests ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
