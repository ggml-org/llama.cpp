#include "../ggml/src/ggml-sycl/kv-cache-coordinator.hpp"

#include <cassert>
#include <iostream>

using namespace ggml_sycl;

// Test 1: Initialize with model parameters
bool test_init() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);  // 32 layers, 4096 embd, 32 heads, 8192 context

    // bytes_per_token = 2 * 32 * 4096 * 2 = 524288 bytes
    assert(coord.get_bytes_per_token() == 524288);
    assert(coord.get_current_tokens() == 0);

    std::cout << "[PASS] test_init\n";
    return true;
}

// Test 2: Current KV bytes grows with tokens
bool test_current_kv_bytes() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);

    assert(coord.get_current_kv_bytes() == 0);

    coord.on_token_generated(10ULL * 1024 * 1024 * 1024);  // 10 GB budget
    assert(coord.get_current_kv_bytes() == coord.get_bytes_per_token());

    coord.on_token_generated(10ULL * 1024 * 1024 * 1024);
    assert(coord.get_current_kv_bytes() == 2 * coord.get_bytes_per_token());

    std::cout << "[PASS] test_current_kv_bytes\n";
    return true;
}

// Test 3: Max KV bytes
bool test_max_kv_bytes() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);

    // Max = 8192 * 524288 = 4294967296 bytes (4 GB)
    size_t expected = 8192ULL * 524288;
    assert(coord.get_max_kv_bytes() == expected);
    (void) expected;  // Silence unused variable warning in release builds

    std::cout << "[PASS] test_max_kv_bytes\n";
    return true;
}

// Test 4: Sliding window caps memory
bool test_sliding_window() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);
    coord.set_sliding_window(1024);

    // Generate more tokens than window
    for (int i = 0; i < 2000; i++) {
        coord.on_token_generated(10ULL * 1024 * 1024 * 1024);
    }

    // Memory should be capped at window size
    size_t expected = 1024 * coord.get_bytes_per_token();
    assert(coord.get_current_kv_bytes() == expected);
    (void) expected;  // Silence unused variable warning in release builds

    std::cout << "[PASS] test_sliding_window\n";
    return true;
}

// Test 5: Flash attention scratch
bool test_flash_attention_scratch() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192, KVCacheModel::FLASH_ATTENTION);

    // scratch = n_head * max_context * sizeof(float) = 32 * 8192 * 4
    size_t expected = 32 * 8192 * sizeof(float);
    assert(coord.get_flash_scratch_bytes() == expected);
    (void) expected;  // Silence unused variable warning in release builds

    std::cout << "[PASS] test_flash_attention_scratch\n";
    return true;
}

// Test 6: Callback notification
bool test_callback() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);

    size_t callback_value = 0;
    coord.set_weight_cache_callback([&](size_t available) { callback_value = available; });

    size_t budget = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    coord.on_token_generated(budget);

    // Available = budget - kv_bytes
    size_t expected = budget - coord.get_current_kv_bytes();
    assert(callback_value == expected);
    (void) expected;  // Silence unused variable warning in release builds

    std::cout << "[PASS] test_callback\n";
    return true;
}

// Test 7: Predict memory for tokens
bool test_predict_memory() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);

    size_t predicted = coord.predict_memory_for_tokens(100);
    size_t expected  = 100 * coord.get_bytes_per_token();
    assert(predicted == expected);
    (void) predicted;  // Silence unused variable warning in release builds
    (void) expected;

    std::cout << "[PASS] test_predict_memory\n";
    return true;
}

// Test 8: Can generate tokens
bool test_can_generate() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);

    size_t available = 1024 * coord.get_bytes_per_token();  // Enough for 1024 tokens

    assert(coord.can_generate_tokens(1000, available) == true);
    assert(coord.can_generate_tokens(1025, available) == false);
    (void) available;  // Silence unused variable warning in release builds

    std::cout << "[PASS] test_can_generate\n";
    return true;
}

// Test 9: Reset
bool test_reset() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);

    for (int i = 0; i < 100; i++) {
        coord.on_token_generated(10ULL * 1024 * 1024 * 1024);
    }

    assert(coord.get_current_tokens() == 100);

    coord.reset();
    assert(coord.get_current_tokens() == 0);
    assert(coord.get_current_kv_bytes() == 0);

    std::cout << "[PASS] test_reset\n";
    return true;
}

// Test 10: Sliding window max bytes
bool test_sliding_window_max() {
    KVCacheCoordinator coord;
    coord.init(32, 4096, 32, 8192);
    coord.set_sliding_window(512);

    // Max should be window size, not max context
    size_t expected = 512 * coord.get_bytes_per_token();
    assert(coord.get_max_kv_bytes() == expected);
    (void) expected;  // Silence unused variable warning in release builds

    std::cout << "[PASS] test_sliding_window_max\n";
    return true;
}

int main() {
    int passed = 0, failed = 0;

    if (test_init()) {
        passed++;
    } else {
        failed++;
    }
    if (test_current_kv_bytes()) {
        passed++;
    } else {
        failed++;
    }
    if (test_max_kv_bytes()) {
        passed++;
    } else {
        failed++;
    }
    if (test_sliding_window()) {
        passed++;
    } else {
        failed++;
    }
    if (test_flash_attention_scratch()) {
        passed++;
    } else {
        failed++;
    }
    if (test_callback()) {
        passed++;
    } else {
        failed++;
    }
    if (test_predict_memory()) {
        passed++;
    } else {
        failed++;
    }
    if (test_can_generate()) {
        passed++;
    } else {
        failed++;
    }
    if (test_reset()) {
        passed++;
    } else {
        failed++;
    }
    if (test_sliding_window_max()) {
        passed++;
    } else {
        failed++;
    }

    std::cout << "\n=== KV Cache Coordinator Tests ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
