//
// Test for dense_layer_scheduler - double-buffered dense layer scheduler
// Part of: llama.cpp-2pa (Tiered Memory Architecture)
//

#include "dense-scheduler.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

// Simulated host layer storage
std::unordered_map<int, std::vector<char>> g_host_layers;

void * get_host_layer(int layer_id) {
    auto it = g_host_layers.find(layer_id);
    if (it == g_host_layers.end()) {
        return nullptr;
    }
    return it->second.data();
}

void test_dense_scheduler_basic() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 10 * 1024 * 1024;  // 10MB per layer

    // Create some host layers
    for (int i = 0; i < 5; i++) {
        g_host_layers[i].resize(LAYER_SIZE);
        std::memset(g_host_layers[i].data(), i + 1, LAYER_SIZE);  // Fill with pattern
    }

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    // Get layer 0
    void * ptr0 = scheduler.get_dense_layer(0, LAYER_SIZE);
    assert(ptr0 != nullptr && "Should get layer 0");
    assert(scheduler.layer_in_slot(scheduler.current_slot()) == 0);

    // Get same layer again (should be cached)
    void * ptr0_again = scheduler.get_dense_layer(0, LAYER_SIZE);
    assert(ptr0_again == ptr0 && "Should return same pointer");

    // Prefetch layer 1 while "computing" on layer 0
    scheduler.prefetch_next(1, LAYER_SIZE);

    // Advance slot and get layer 1
    scheduler.advance_slot();
    void * ptr1 = scheduler.get_dense_layer(1, LAYER_SIZE);
    assert(ptr1 != nullptr && "Should get layer 1");
    assert(scheduler.layer_in_slot(scheduler.current_slot()) == 1);

    std::cout << "test_dense_scheduler_basic: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_prefetch_overlap() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 50 * 1024 * 1024;  // 50MB
    constexpr int    NUM_LAYERS = 10;

    // Create host layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        g_host_layers[i].resize(LAYER_SIZE);
        std::memset(g_host_layers[i].data(), i, LAYER_SIZE);
    }

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    // Simulate layer-by-layer execution with prefetch
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        // Get current layer
        void * ptr = scheduler.get_dense_layer(layer, LAYER_SIZE);
        assert(ptr != nullptr);

        // Prefetch next layer (if not last)
        if (layer + 1 < NUM_LAYERS) {
            scheduler.prefetch_next(layer + 1, LAYER_SIZE);
        }

        // "Compute" on current layer (simulated)
        // In real code, this would be GPU kernel execution

        // Advance slot for next iteration
        if (layer + 1 < NUM_LAYERS) {
            scheduler.advance_slot();
        }
    }

    // Most loads should be prefetched (except first)
    assert(scheduler.prefetch_count() == static_cast<size_t>(NUM_LAYERS - 1));
    assert(scheduler.sync_load_count() == 1);  // Only first layer was sync

    std::cout << "test_dense_scheduler_prefetch_overlap: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_data_integrity() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 1 * 1024 * 1024;  // 1MB

    // Create layer with known pattern
    g_host_layers[0].resize(LAYER_SIZE);
    for (size_t i = 0; i < LAYER_SIZE; i++) {
        g_host_layers[0][i] = static_cast<char>(i & 0xFF);
    }

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    void * vram_ptr = scheduler.get_dense_layer(0, LAYER_SIZE);
    assert(vram_ptr != nullptr);

    // Copy back and verify
    std::vector<char> verify(LAYER_SIZE);
    q.memcpy(verify.data(), vram_ptr, LAYER_SIZE).wait();

    for (size_t i = 0; i < LAYER_SIZE; i++) {
        assert(verify[i] == static_cast<char>(i & 0xFF) && "Data mismatch");
    }

    std::cout << "test_dense_scheduler_data_integrity: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_stats() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 5 * 1024 * 1024;  // 5MB

    for (int i = 0; i < 3; i++) {
        g_host_layers[i].resize(LAYER_SIZE);
    }

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    assert(scheduler.slot_size() == LAYER_SIZE);
    assert(scheduler.current_slot() == 0);
    assert(scheduler.prefetch_count() == 0);
    assert(scheduler.sync_load_count() == 0);

    // Sync load layer 0
    scheduler.get_dense_layer(0, LAYER_SIZE);
    assert(scheduler.sync_load_count() == 1);

    // Prefetch layer 1
    scheduler.prefetch_next(1, LAYER_SIZE);
    assert(scheduler.prefetch_count() == 1);

    std::cout << "test_dense_scheduler_stats: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_size_validation() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 10 * 1024 * 1024;  // 10MB

    g_host_layers[0].resize(LAYER_SIZE * 2);         // Create oversized layer

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    // Requesting size larger than slot should fail
    void * ptr = scheduler.get_dense_layer(0, LAYER_SIZE * 2);
    assert(ptr == nullptr && "Should reject oversized layer");

    std::cout << "test_dense_scheduler_size_validation: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_no_callback() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 5 * 1024 * 1024;

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    // Note: callback NOT set

    // Should fail gracefully without callback
    void * ptr = scheduler.get_dense_layer(0, LAYER_SIZE);
    assert(ptr == nullptr && "Should fail without callback");

    std::cout << "test_dense_scheduler_no_callback: PASSED\n";
}

void test_dense_scheduler_missing_layer() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 5 * 1024 * 1024;

    // Only create layer 0, not layer 1
    g_host_layers[0].resize(LAYER_SIZE);

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    // Layer 0 should work
    void * ptr0 = scheduler.get_dense_layer(0, LAYER_SIZE);
    assert(ptr0 != nullptr && "Should get layer 0");

    // Layer 1 doesn't exist - should fail
    void * ptr1 = scheduler.get_dense_layer(1, LAYER_SIZE);
    assert(ptr1 == nullptr && "Should fail for missing layer");

    std::cout << "test_dense_scheduler_missing_layer: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_slot_reuse() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 5 * 1024 * 1024;

    // Create 4 layers
    for (int i = 0; i < 4; i++) {
        g_host_layers[i].resize(LAYER_SIZE);
        std::memset(g_host_layers[i].data(), 'A' + i, LAYER_SIZE);
    }

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    // Process layers 0, 1, 2, 3 in sequence
    // With double buffering, slots should be reused: 0->slot0, 1->slot1, 2->slot0, 3->slot1

    // Layer 0 in slot 0
    scheduler.get_dense_layer(0, LAYER_SIZE);
    scheduler.prefetch_next(1, LAYER_SIZE);
    scheduler.advance_slot();

    // Layer 1 in slot 1
    scheduler.get_dense_layer(1, LAYER_SIZE);
    scheduler.prefetch_next(2, LAYER_SIZE);
    scheduler.advance_slot();

    // Layer 2 should now be in slot 0 (was layer 0)
    void * ptr2 = scheduler.get_dense_layer(2, LAYER_SIZE);
    assert(ptr2 != nullptr);

    // Verify data integrity for layer 2
    std::vector<char> verify(LAYER_SIZE);
    q.memcpy(verify.data(), ptr2, LAYER_SIZE).wait();
    assert(verify[0] == 'C' && "Should contain layer 2 data");

    std::cout << "test_dense_scheduler_slot_reuse: PASSED\n";
    g_host_layers.clear();
}

void test_dense_scheduler_wait_prefetch() {
    sycl::queue q;

    constexpr size_t LAYER_SIZE = 20 * 1024 * 1024;  // 20MB for noticeable transfer time

    g_host_layers[0].resize(LAYER_SIZE);
    g_host_layers[1].resize(LAYER_SIZE);
    std::memset(g_host_layers[1].data(), 0x55, LAYER_SIZE);

    ggml_sycl::dense_layer_scheduler scheduler(q, LAYER_SIZE);
    assert(scheduler.is_initialized() && "Scheduler should be initialized");
    scheduler.set_host_ptr_callback(get_host_layer);

    // Get layer 0
    scheduler.get_dense_layer(0, LAYER_SIZE);

    // Start prefetch of layer 1
    scheduler.prefetch_next(1, LAYER_SIZE);

    // Explicitly wait for prefetch
    scheduler.wait_prefetch();

    // After wait, advancing and getting layer 1 should NOT require sync load
    size_t sync_before = scheduler.sync_load_count();
    scheduler.advance_slot();
    scheduler.get_dense_layer(1, LAYER_SIZE);
    size_t sync_after = scheduler.sync_load_count();

    assert(sync_before == sync_after && "Should not need sync load after wait_prefetch");

    std::cout << "test_dense_scheduler_wait_prefetch: PASSED\n";
    g_host_layers.clear();
}

int main() {
    try {
        test_dense_scheduler_basic();
        test_dense_scheduler_prefetch_overlap();
        test_dense_scheduler_data_integrity();
        test_dense_scheduler_stats();
        test_dense_scheduler_size_validation();
        test_dense_scheduler_no_callback();
        test_dense_scheduler_missing_layer();
        test_dense_scheduler_slot_reuse();
        test_dense_scheduler_wait_prefetch();
        std::cout << "\nAll tests PASSED!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
