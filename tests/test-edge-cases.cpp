#include <cassert>
#include <iostream>
#include <cmath>
#include "../ggml/src/ggml-sycl/edge-case-tests.hpp"

using namespace ggml_sycl;

// Mock test function that returns diff based on alignment
float mock_test_func(int64_t M, int64_t N, int64_t K, int quant_type) {
    (void)quant_type;

    // Simulate worse results for misaligned dimensions
    float base_diff = 1e-6f;

    if (M % 32 != 0) base_diff *= 10;
    if (N % 32 != 0) base_diff *= 5;
    if (K % 32 != 0) base_diff *= 10;

    // Tiny dimensions are harder
    if (M < 8) base_diff *= 2;

    return base_diff;
}

// Failing test function
float failing_test_func(int64_t M, int64_t, int64_t, int) {
    return M < 10 ? 1.0f : 1e-6f;  // Fail for small M
}

// Test 1: Test M values
bool test_m_values() {
    auto vals = EdgeCaseSelfTest::get_test_m_values();

    assert(vals.size() > 10);
    assert(vals[0] == 1);  // Includes 1

    // Check includes boundary cases
    bool has_31 = false, has_32 = false, has_33 = false;
    for (int64_t v : vals) {
        if (v == 31) has_31 = true;
        if (v == 32) has_32 = true;
        if (v == 33) has_33 = true;
    }
    if (!has_31 || !has_32 || !has_33) {
        std::cerr << "[FAIL] test_m_values: missing boundary values\n";
        return false;
    }

    std::cout << "[PASS] test_m_values\n";
    return true;
}

// Test 2: Generate test matrix
bool test_generate_matrix() {
    auto configs = EdgeCaseSelfTest::generate_test_matrix();

    if (configs.size() <= 50) {
        std::cerr << "[FAIL] test_generate_matrix: too few configs\n";
        return false;
    }

    // Check descriptions are populated
    for (size_t i = 0; i < configs.size(); i++) {
        if (configs[i].description.empty() || configs[i].M <= 0 ||
            configs[i].N <= 0 || configs[i].K <= 0) {
            std::cerr << "[FAIL] test_generate_matrix: invalid config at index " << i << "\n";
            return false;
        }
    }

    std::cout << "[PASS] test_generate_matrix (" << configs.size() << " configs)\n";
    return true;
}

// Test 3: Run sanity check
bool test_sanity_check() {
    EdgeCaseSelfTest tester;
    auto results = tester.run_sanity_check(mock_test_func);

    assert(results.size() == 5);

    int passed = 0;
    for (const auto& r : results) {
        if (r.passed) passed++;
    }

    std::cout << "[PASS] test_sanity_check (" << passed << "/" << results.size() << ")\n";
    return true;
}

// Test 4: Detect failures
bool test_detect_failures() {
    EdgeCaseSelfTest tester;
    auto results = tester.run_sanity_check(failing_test_func);

    int failed = 0;
    for (const auto& r : results) {
        if (!r.passed) failed++;
    }

    // Should detect some failures (tiny_batch fails)
    assert(failed > 0);

    std::cout << "[PASS] test_detect_failures (" << failed << " failures detected)\n";
    return true;
}

// Test 5: Power of 2 check
bool test_power_of_2() {
    assert(EdgeCaseSelfTest::is_power_of_2(1) == true);
    assert(EdgeCaseSelfTest::is_power_of_2(2) == true);
    assert(EdgeCaseSelfTest::is_power_of_2(32) == true);
    assert(EdgeCaseSelfTest::is_power_of_2(64) == true);
    assert(EdgeCaseSelfTest::is_power_of_2(31) == false);
    assert(EdgeCaseSelfTest::is_power_of_2(33) == false);
    assert(EdgeCaseSelfTest::is_power_of_2(0) == false);

    std::cout << "[PASS] test_power_of_2\n";
    return true;
}

// Test 6: Alignment check
bool test_is_aligned() {
    assert(EdgeCaseSelfTest::is_aligned(32, 32) == true);
    assert(EdgeCaseSelfTest::is_aligned(64, 32) == true);
    assert(EdgeCaseSelfTest::is_aligned(33, 32) == false);
    assert(EdgeCaseSelfTest::is_aligned(128, 64) == true);
    assert(EdgeCaseSelfTest::is_aligned(127, 64) == false);

    std::cout << "[PASS] test_is_aligned\n";
    return true;
}

// Test 7: EdgeCaseResult fields
bool test_result_fields() {
    EdgeCaseResult r;
    r.passed = true;
    r.config_desc = "test";
    r.max_diff = 1e-6f;

    assert(r.passed == true);
    assert(r.config_desc == "test");
    assert(r.max_diff > 0);

    std::cout << "[PASS] test_result_fields\n";
    return true;
}

// Test 8: EdgeCaseConfig fields
bool test_config_fields() {
    EdgeCaseConfig cfg{32, 4096, 4096, 0, "test_config"};

    assert(cfg.M == 32);
    assert(cfg.N == 4096);
    assert(cfg.K == 4096);
    assert(cfg.quant_type == 0);
    assert(cfg.description == "test_config");

    std::cout << "[PASS] test_config_fields\n";
    return true;
}

// Test 9: Custom quant types
bool test_custom_quant_types() {
    auto configs = EdgeCaseSelfTest::generate_test_matrix({0, 1, 2});

    bool has_q0 = false, has_q1 = false, has_q2 = false;
    for (size_t i = 0; i < configs.size(); i++) {
        if (configs[i].quant_type == 0) has_q0 = true;
        if (configs[i].quant_type == 1) has_q1 = true;
        if (configs[i].quant_type == 2) has_q2 = true;
    }

    if (!has_q0 || !has_q1 || !has_q2) {
        std::cerr << "[FAIL] test_custom_quant_types: missing quant types\n";
        return false;
    }

    std::cout << "[PASS] test_custom_quant_types\n";
    return true;
}

// Test 10: Run full test suite (with mock)
bool test_run_all() {
    EdgeCaseSelfTest tester;
    auto results = tester.run_all_tests(mock_test_func);

    // Should have many results
    assert(results.size() > 50);

    // Most should pass with mock function
    int passed = 0;
    for (const auto& r : results) {
        if (r.passed) passed++;
    }

    float pass_rate = 100.0f * static_cast<float>(passed) / static_cast<float>(results.size());

    std::cout << "[PASS] test_run_all (" << passed << "/" << results.size()
              << " = " << pass_rate << "%)\n";
    return true;
}

int main() {
    int passed = 0, failed = 0;

    if (test_m_values()) passed++; else failed++;
    if (test_generate_matrix()) passed++; else failed++;
    if (test_sanity_check()) passed++; else failed++;
    if (test_detect_failures()) passed++; else failed++;
    if (test_power_of_2()) passed++; else failed++;
    if (test_is_aligned()) passed++; else failed++;
    if (test_result_fields()) passed++; else failed++;
    if (test_config_fields()) passed++; else failed++;
    if (test_custom_quant_types()) passed++; else failed++;
    if (test_run_all()) passed++; else failed++;

    std::cout << "\n=== Edge Case Self-Test Tests ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
