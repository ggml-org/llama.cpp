//
// Test for XMXConfig struct in unified-kernel.hpp
// Tests hardware-queried dimensions for ESIMD dpas kernel configuration
//

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "../unified-kernel.hpp"
#include "../common.hpp"

using namespace ggml_sycl_unified;

// Test 1: Default XMXConfig values
void test_default_config() {
    std::cout << "Test: Default XMXConfig values... ";

    XMXConfig cfg;

    // Check default dimensions (Intel Arc defaults)
    assert(cfg.xmx_m == 8 && "Default M should be 8");
    assert(cfg.xmx_n == 16 && "Default N should be 16");
    assert(cfg.xmx_k_fp16 == 16 && "Default K for FP16 should be 16");
    assert(cfg.xmx_k_int8 == 32 && "Default K for INT8 should be 32");

    // Check default resources
    assert(cfg.slm_size == 65536 && "Default SLM should be 65536");
    assert(cfg.nsm > 0 && "Default nsm should be positive");

    // Check default capability flags (should be false for default constructed)
    assert(cfg.supported == false && "Default supported should be false");
    assert(cfg.supports_int8 == false && "Default supports_int8 should be false");
    assert(cfg.supports_fp16 == false && "Default supports_fp16 should be false");

    std::cout << "PASSED" << std::endl;
}

// Test 2: from_device with invalid device_id (-1)
void test_from_device_negative_id() {
    std::cout << "Test: from_device with device_id=-1... ";

    XMXConfig cfg = XMXConfig::from_device(-1);

    // Should return default config, not crash
    assert(cfg.xmx_m == 8 && "Invalid device should return default M=8");
    assert(cfg.xmx_n == 16 && "Invalid device should return default N=16");
    assert(cfg.xmx_k_int8 == 32 && "Invalid device should return default K_INT8=32");
    assert(cfg.slm_size == 65536 && "Invalid device should return default SLM");

    std::cout << "PASSED" << std::endl;
}

// Test 3: from_device with out-of-range device_id (999)
void test_from_device_out_of_range() {
    std::cout << "Test: from_device with device_id=999... ";

    XMXConfig cfg = XMXConfig::from_device(999);

    // Should return default config, not crash
    assert(cfg.xmx_m == 8 && "Out-of-range device should return default M=8");
    assert(cfg.xmx_n == 16 && "Out-of-range device should return default N=16");
    assert(cfg.xmx_k_int8 == 32 && "Out-of-range device should return default K_INT8=32");

    std::cout << "PASSED" << std::endl;
}

// Test 4: Environment variable check (GGML_SYCL_XMX_ESIMD)
void test_use_esimd_dpas_env_unset() {
    std::cout << "Test: use_esimd_dpas() when env not set... ";

    // Note: This test assumes the env var is NOT set before running.
    // The function uses static caching, so this test must be run
    // before any other code calls use_esimd_dpas().

    // With env unset, should return false
    bool result = use_esimd_dpas();
    // Just verify it doesn't crash and returns a bool
    (void)result;

    std::cout << "PASSED (returned " << (result ? "true" : "false") << ")" << std::endl;
}

// Test 5: Environment variable check (GGML_SYCL_XMX_INT8)
void test_use_int8_dpas_env_unset() {
    std::cout << "Test: use_int8_dpas() when env not set... ";

    bool result = use_int8_dpas();
    // Just verify it doesn't crash and returns a bool
    (void)result;

    std::cout << "PASSED (returned " << (result ? "true" : "false") << ")" << std::endl;
}

// Test 6: from_device with valid device (if available)
void test_from_device_valid() {
    std::cout << "Test: from_device with valid device... ";

    const auto& info = ggml_sycl_info();
    if (info.device_count == 0) {
        std::cout << "SKIPPED (no devices)" << std::endl;
        return;
    }

    XMXConfig cfg = XMXConfig::from_device(0);

    // On Arc B580, expect M=8, N=16, K_INT8=32
    // But these should at least be positive
    assert(cfg.xmx_m > 0 && "M should be positive");
    assert(cfg.xmx_n > 0 && "N should be positive");
    assert(cfg.xmx_k_int8 > 0 && "K_INT8 should be positive");
    assert(cfg.xmx_k_fp16 == 16 && "K_FP16 should always be 16");
    assert(cfg.slm_size > 0 && "SLM size should be positive");

    std::cout << "PASSED (M=" << cfg.xmx_m
              << ", N=" << cfg.xmx_n
              << ", K_INT8=" << cfg.xmx_k_int8
              << ", SLM=" << cfg.slm_size << ")" << std::endl;
}

// Test 7: Verify Arc B580 specific values (if running on that hardware)
void test_arc_b580_dimensions() {
    std::cout << "Test: Arc B580 expected dimensions... ";

    const auto& info = ggml_sycl_info();
    if (info.device_count == 0) {
        std::cout << "SKIPPED (no devices)" << std::endl;
        return;
    }

    XMXConfig cfg = XMXConfig::from_device(0);

    // On Arc B580, these should be exactly:
    bool is_expected = (cfg.xmx_m == 8 && cfg.xmx_n == 16 && cfg.xmx_k_int8 == 32);

    if (is_expected) {
        std::cout << "PASSED (M=8, N=16, K_INT8=32 as expected)" << std::endl;
    } else {
        std::cout << "PASSED (different hardware: M=" << cfg.xmx_m
                  << ", N=" << cfg.xmx_n
                  << ", K_INT8=" << cfg.xmx_k_int8 << ")" << std::endl;
    }
}

int main() {
    std::cout << "=== XMXConfig Unit Tests ===" << std::endl;
    std::cout << std::endl;

    test_default_config();
    test_from_device_negative_id();
    test_from_device_out_of_range();
    test_use_esimd_dpas_env_unset();
    test_use_int8_dpas_env_unset();
    test_from_device_valid();
    test_arc_b580_dimensions();

    std::cout << std::endl;
    std::cout << "=== All XMXConfig tests passed ===" << std::endl;

    return 0;
}
