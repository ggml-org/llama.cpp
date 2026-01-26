#pragma once
#include <vector>
#include <functional>
#include <string>
#include <cstdint>
#include <iostream>
#include <sstream>

namespace ggml_sycl {

// Test configuration
struct EdgeCaseConfig {
    int64_t M, N, K;
    int quant_type;
    std::string description;
};

// Test result
struct EdgeCaseResult {
    bool passed;
    std::string config_desc;
    std::string error_msg;
    float max_diff;
};

// Test function type: returns max difference from reference, negative on error
using EdgeCaseTestFunc = std::function<float(int64_t M, int64_t N, int64_t K, int quant_type)>;

class EdgeCaseSelfTest {
public:
    // Dimension test cases
    static std::vector<int64_t> get_test_m_values() {
        return {1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256};
    }

    static std::vector<int64_t> get_test_n_values() {
        return {32, 33, 64, 127, 128, 256, 1024, 4096, 4097};
    }

    static std::vector<int64_t> get_test_k_values() {
        return {32, 33, 64, 127, 128, 256, 1024, 4096, 4097};
    }

    // Generate all edge case configurations
    static std::vector<EdgeCaseConfig> generate_test_matrix(
        const std::vector<int>& quant_types = {0, 2})  // Q4_0, Q8_0
    {
        std::vector<EdgeCaseConfig> configs;

        for (int qt : quant_types) {
            // Critical small dimensions
            for (int64_t m : {1, 7, 15}) {
                for (int64_t n : {32, 127}) {
                    for (int64_t k : {32, 33, 127}) {
                        std::stringstream desc;
                        desc << "tiny_M=" << m << "_N=" << n << "_K=" << k << "_Q" << qt;
                        configs.push_back({m, n, k, qt, desc.str()});
                    }
                }
            }

            // Alignment boundary cases (power of 2 +/- 1)
            for (int64_t m : {31, 32, 33, 63, 64, 65, 127, 128}) {
                std::stringstream desc;
                desc << "boundary_M=" << m << "_Q" << qt;
                configs.push_back({m, 4096, 4096, qt, desc.str()});
            }

            // Non-aligned K (critical for dpas)
            for (int64_t k : {31, 33, 63, 65, 127, 129}) {
                std::stringstream desc;
                desc << "unaligned_K=" << k << "_Q" << qt;
                configs.push_back({32, 4096, k, qt, desc.str()});
            }

            // Large dimensions
            for (int64_t n : {4096, 8192}) {
                for (int64_t k : {4096, 8192}) {
                    std::stringstream desc;
                    desc << "large_N=" << n << "_K=" << k << "_Q" << qt;
                    configs.push_back({64, n, k, qt, desc.str()});
                }
            }

            // Prime dimensions (stress alignment handling)
            for (int64_t prime : {31, 127, 521, 1031}) {
                std::stringstream desc;
                desc << "prime_" << prime << "_Q" << qt;
                configs.push_back({prime, prime, prime, qt, desc.str()});
            }
        }

        return configs;
    }

    // Run all tests
    std::vector<EdgeCaseResult> run_all_tests(
        EdgeCaseTestFunc test_func,
        float tolerance = 1e-4f)
    {
        std::vector<EdgeCaseResult> results;
        auto configs = generate_test_matrix();

        for (const auto& cfg : configs) {
            EdgeCaseResult result;
            result.config_desc = cfg.description;

            float diff = test_func(cfg.M, cfg.N, cfg.K, cfg.quant_type);

            if (diff < 0) {
                result.passed = false;
                result.error_msg = "Test function returned error";
                result.max_diff = -1;
            } else if (diff > tolerance) {
                result.passed = false;
                result.error_msg = "Exceeded tolerance";
                result.max_diff = diff;
            } else {
                result.passed = true;
                result.max_diff = diff;
            }

            results.push_back(result);
        }

        return results;
    }

    // Quick sanity check (subset of tests)
    std::vector<EdgeCaseResult> run_sanity_check(
        EdgeCaseTestFunc test_func,
        float tolerance = 1e-4f)
    {
        std::vector<EdgeCaseConfig> quick_configs = {
            {1, 32, 32, 0, "single_element"},
            {7, 64, 64, 0, "tiny_batch"},
            {32, 4096, 4096, 0, "standard_small"},
            {33, 4096, 4096, 0, "misaligned_m"},
            {64, 4096, 4097, 0, "misaligned_k"},
        };

        std::vector<EdgeCaseResult> results;
        for (const auto& cfg : quick_configs) {
            EdgeCaseResult result;
            result.config_desc = cfg.description;

            float diff = test_func(cfg.M, cfg.N, cfg.K, cfg.quant_type);
            result.passed = (diff >= 0 && diff <= tolerance);
            result.max_diff = diff;

            results.push_back(result);
        }

        return results;
    }

    // Print test summary
    static void print_summary(const std::vector<EdgeCaseResult>& results) {
        int passed = 0, failed = 0;
        float max_diff = 0;

        for (const auto& r : results) {
            if (r.passed) {
                passed++;
            } else {
                failed++;
                std::cerr << "[FAIL] " << r.config_desc;
                if (!r.error_msg.empty()) {
                    std::cerr << " - " << r.error_msg;
                }
                std::cerr << " (diff=" << r.max_diff << ")\n";
            }
            if (r.max_diff > max_diff && r.max_diff > 0) {
                max_diff = r.max_diff;
            }
        }

        std::cout << "\n=== Edge Case Test Summary ===\n";
        std::cout << "Total: " << results.size() << "\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Max diff: " << max_diff << "\n";
    }

    // Check if dimension is power of 2
    static bool is_power_of_2(int64_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // Check if dimension is aligned to tile size
    static bool is_aligned(int64_t n, int64_t alignment) {
        return n % alignment == 0;
    }
};

} // namespace ggml_sycl
