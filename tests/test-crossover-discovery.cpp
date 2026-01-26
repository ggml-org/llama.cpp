#include <cassert>
#include <iostream>
#include <cmath>
#include "../ggml/src/ggml-sycl/crossover-discovery.hpp"

using namespace ggml_sycl;

// Mock benchmark functions
float mock_unified_bench(int64_t M, int64_t N, int64_t K) {
    // Unified kernel: good for small batch, degrades with size
    return 0.1f * M + 0.001f * N * K / 1e6f;
}

float mock_onednn_bench(int64_t M, int64_t N, int64_t K) {
    // oneDNN: higher overhead, but better scaling
    return 5.0f + 0.02f * M + 0.0005f * N * K / 1e6f;
}

// Test 1: Find crossover with mock benchmarks
bool test_find_crossover() {
    CrossoverDiscovery discovery;

    auto result = discovery.find_crossover(
        4096, 4096, 0,
        mock_unified_bench,
        mock_onednn_bench,
        1, 2);

    // Should find a crossover point
    assert(result.crossover_batch > 0);
    assert(result.samples_taken > 0);

    std::cout << "[PASS] test_find_crossover (crossover=" << result.crossover_batch << ")\n";
    return true;
}

// Test 2: Store and retrieve crossover
bool test_store_retrieve() {
    CrossoverDiscovery discovery;

    CrossoverKey key{4096, 4096, 0};
    CrossoverResult result;
    result.crossover_batch = 64;
    result.unified_ms = 1.0f;
    result.onednn_ms = 0.9f;

    discovery.store_crossover(key, result);

    auto* retrieved = discovery.get_crossover(4096, 4096, 0);
    assert(retrieved != nullptr);
    assert(retrieved->crossover_batch == 64);
    (void)retrieved;  // Suppress unused variable warning in release builds

    std::cout << "[PASS] test_store_retrieve\n";
    return true;
}

// Test 3: Should use oneDNN
bool test_should_use_onednn() {
    CrossoverDiscovery discovery;

    CrossoverKey key{4096, 4096, 0};
    CrossoverResult result;
    result.crossover_batch = 64;

    discovery.store_crossover(key, result);

    // Below crossover: use unified
    assert(discovery.should_use_onednn(32, 4096, 4096, 0) == false);

    // At/above crossover: use oneDNN
    assert(discovery.should_use_onednn(64, 4096, 4096, 0) == true);
    assert(discovery.should_use_onednn(128, 4096, 4096, 0) == true);

    std::cout << "[PASS] test_should_use_onednn\n";
    return true;
}

// Test 4: Unknown dimensions
bool test_unknown_dimensions() {
    CrossoverDiscovery discovery;

    // No crossover data stored
    auto* result = discovery.get_crossover(1234, 5678, 0);
    assert(result == nullptr);
    (void)result;  // Suppress unused variable warning in release builds

    // Should default to unified (not oneDNN) for unknown
    assert(discovery.should_use_onednn(100, 1234, 5678, 0) == false);

    std::cout << "[PASS] test_unknown_dimensions\n";
    return true;
}

// Test 5: Standard dimensions list
bool test_standard_dimensions() {
    auto dims = CrossoverDiscovery::get_standard_dimensions();

    assert(dims.size() >= 4);

    // Should include common model dimensions
    bool has_4096 = false;
    for (const auto& [n, k] : dims) {
        if (n == 4096 && k == 4096) has_4096 = true;
    }
    assert(has_4096);
    (void)has_4096;  // Suppress unused variable warning in release builds

    std::cout << "[PASS] test_standard_dimensions\n";
    return true;
}

// Test 6: Clear crossovers
bool test_clear() {
    CrossoverDiscovery discovery;

    CrossoverKey key{4096, 4096, 0};
    CrossoverResult result;
    result.crossover_batch = 64;

    discovery.store_crossover(key, result);
    assert(discovery.get_crossover_count() == 1);

    discovery.clear();
    assert(discovery.get_crossover_count() == 0);

    std::cout << "[PASS] test_clear\n";
    return true;
}

// Test 7: Different quant types stored separately
bool test_quant_types() {
    CrossoverDiscovery discovery;

    CrossoverResult r1, r2;
    r1.crossover_batch = 32;
    r2.crossover_batch = 128;

    discovery.store_crossover({4096, 4096, 0}, r1);  // Q4_0
    discovery.store_crossover({4096, 4096, 1}, r2);  // Q8_0

    auto* q4 = discovery.get_crossover(4096, 4096, 0);
    auto* q8 = discovery.get_crossover(4096, 4096, 1);

    assert(q4->crossover_batch == 32);
    assert(q8->crossover_batch == 128);
    (void)q4;  // Suppress unused variable warning in release builds
    (void)q8;

    std::cout << "[PASS] test_quant_types\n";
    return true;
}

// Test 8: Test batch array
bool test_batch_array() {
    assert(CrossoverDiscovery::NUM_TEST_BATCHES == 11);
    assert(CrossoverDiscovery::TEST_BATCHES[0] == 1);
    assert(CrossoverDiscovery::TEST_BATCHES[10] == 1024);

    std::cout << "[PASS] test_batch_array\n";
    return true;
}

// Test 9: Confidence calculation
bool test_confidence() {
    CrossoverDiscovery discovery;

    // Mock where crossover is clear
    auto clear_unified = [](int64_t M, int64_t, int64_t) -> float {
        return M < 64 ? 1.0f : 100.0f;  // Sudden jump
    };
    auto clear_onednn = [](int64_t, int64_t, int64_t) -> float {
        return 10.0f;  // Constant
    };

    auto result = discovery.find_crossover(4096, 4096, 0,
        clear_unified, clear_onednn, 1, 1);

    // Should have some confidence
    assert(result.confidence >= 0.0f && result.confidence <= 1.0f);
    (void)result;  // Suppress unused variable warning in release builds

    std::cout << "[PASS] test_confidence\n";
    return true;
}

// Test 10: Crossover count
bool test_crossover_count() {
    CrossoverDiscovery discovery;

    assert(discovery.get_crossover_count() == 0);

    CrossoverResult r;
    r.crossover_batch = 64;

    discovery.store_crossover({4096, 4096, 0}, r);
    assert(discovery.get_crossover_count() == 1);

    discovery.store_crossover({8192, 8192, 0}, r);
    assert(discovery.get_crossover_count() == 2);

    std::cout << "[PASS] test_crossover_count\n";
    return true;
}

int main() {
    int passed = 0, failed = 0;

    if (test_find_crossover()) passed++; else failed++;
    if (test_store_retrieve()) passed++; else failed++;
    if (test_should_use_onednn()) passed++; else failed++;
    if (test_unknown_dimensions()) passed++; else failed++;
    if (test_standard_dimensions()) passed++; else failed++;
    if (test_clear()) passed++; else failed++;
    if (test_quant_types()) passed++; else failed++;
    if (test_batch_array()) passed++; else failed++;
    if (test_confidence()) passed++; else failed++;
    if (test_crossover_count()) passed++; else failed++;

    std::cout << "\n=== Crossover Discovery Tests ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
