//
// Test for PrefetchScheduler - predictive weight prefetching scheduler
// Part of: llama.cpp-6hp (Heat-Aware Memory Management)
// Task: llama.cpp-6hp.14 (Predictive prefetching)
//

#include "prefetch-scheduler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace ggml_sycl;

// Test 1: Default lookahead
bool test_default_lookahead() {
    PrefetchScheduler scheduler;
    assert(scheduler.get_lookahead() == 3);
    std::cout << "[PASS] test_default_lookahead\n";
    return true;
}

// Test 2: Set lookahead
bool test_set_lookahead() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(5);
    assert(scheduler.get_lookahead() == 5);
    std::cout << "[PASS] test_set_lookahead\n";
    return true;
}

// Test 3: Analyze graph
bool test_analyze_graph() {
    PrefetchScheduler scheduler;

    int                 data1, data2, data3;
    std::vector<void *> tensors = { &data1, &data2, &data3 };
    std::vector<size_t> sizes   = { 1024 * 1024, 2048 * 1024, 512 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2 };

    assert(scheduler.analyze_graph(tensors, sizes, layers) == 3);

    std::cout << "[PASS] test_analyze_graph\n";
    return true;
}

// Test 4: Filter small tensors
bool test_filter_small_tensors() {
    PrefetchScheduler scheduler;

    int                 data1, data2;
    std::vector<void *> tensors = { &data1, &data2 };
    std::vector<size_t> sizes   = { 100, 1024 * 1024 };  // 100 bytes too small
    std::vector<int>    layers  = { 0, 1 };

    assert(scheduler.analyze_graph(tensors, sizes, layers) == 1);  // Only the large one

    std::cout << "[PASS] test_filter_small_tensors\n";
    return true;
}

// Test 5: On kernel start triggers prefetch
bool test_on_kernel_start() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(2);

    int                 data0, data1, data2, data3;
    std::vector<void *> tensors = { &data0, &data1, &data2, &data3 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2, 3 };

    (void) scheduler.analyze_graph(tensors, sizes, layers);

    // At layer 0, should prefetch layers 1 and 2 (within lookahead=2)
    auto requests = scheduler.on_kernel_start(0);
    assert(requests.size() == 2);

    std::cout << "[PASS] test_on_kernel_start\n";
    return true;
}

// Test 6: Mark completed
bool test_mark_completed() {
    PrefetchScheduler scheduler;

    int                 data0, data1;
    std::vector<void *> tensors = { &data0, &data1 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1 };

    (void) scheduler.analyze_graph(tensors, sizes, layers);
    scheduler.on_kernel_start(0);

    assert(scheduler.get_active_count() == 1);

    scheduler.mark_completed(&data1);
    assert(scheduler.get_active_count() == 0);
    assert(scheduler.get_prefetch_hits() == 1);

    std::cout << "[PASS] test_mark_completed\n";
    return true;
}

// Test 7: Cancel all pending
bool test_cancel_all() {
    PrefetchScheduler scheduler;

    int                 data0, data1, data2;
    std::vector<void *> tensors = { &data0, &data1, &data2 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2 };

    scheduler.analyze_graph(tensors, sizes, layers);
    scheduler.on_kernel_start(0);

    scheduler.cancel_all_pending();
    assert(scheduler.get_active_count() == 0);
    assert(scheduler.get_prefetch_cancels() == 1);

    std::cout << "[PASS] test_cancel_all\n";
    return true;
}

// Test 8: Is prefetching
bool test_is_prefetching() {
    PrefetchScheduler scheduler;

    int                 data0, data1;
    std::vector<void *> tensors = { &data0, &data1 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1 };

    scheduler.analyze_graph(tensors, sizes, layers);

    assert(scheduler.is_prefetching(&data1) == false);

    scheduler.on_kernel_start(0);
    assert(scheduler.is_prefetching(&data1) == true);

    std::cout << "[PASS] test_is_prefetching\n";
    return true;
}

// Test 9: Hit rate calculation
bool test_hit_rate() {
    PrefetchScheduler scheduler;

    int                 data0, data1, data2, data3;
    std::vector<void *> tensors = { &data0, &data1, &data2, &data3 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2, 3 };

    (void) scheduler.analyze_graph(tensors, sizes, layers);
    scheduler.on_kernel_start(0);

    scheduler.mark_completed(&data1);
    scheduler.mark_completed(&data2);
    scheduler.cancel_all_pending();

    // 2 hits, 1 cancel = 66% hit rate
    assert(scheduler.get_hit_rate() > 0.65f && scheduler.get_hit_rate() < 0.68f);

    std::cout << "[PASS] test_hit_rate\n";
    return true;
}

// Test 10: Reset scheduler
bool test_reset() {
    PrefetchScheduler scheduler;

    int                 data0, data1;
    std::vector<void *> tensors = { &data0, &data1 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1 };

    scheduler.analyze_graph(tensors, sizes, layers);
    scheduler.on_kernel_start(0);

    scheduler.reset();
    assert(scheduler.get_active_count() == 0);

    // Can trigger prefetch again after reset
    auto requests = scheduler.on_kernel_start(0);
    assert(requests.size() == 1);

    std::cout << "[PASS] test_reset\n";
    return true;
}

// Test 11: No prefetch for current or past layers
bool test_no_prefetch_for_past_layers() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(2);

    int                 data0, data1, data2, data3;
    std::vector<void *> tensors = { &data0, &data1, &data2, &data3 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2, 3 };

    scheduler.analyze_graph(tensors, sizes, layers);

    // At layer 2, should only prefetch layer 3 (not 0, 1, or 2)
    auto requests = scheduler.on_kernel_start(2);
    assert(requests.size() == 1);
    assert(requests[0].layer_index == 3);

    std::cout << "[PASS] test_no_prefetch_for_past_layers\n";
    return true;
}

// Test 12: No prefetch beyond lookahead
bool test_no_prefetch_beyond_lookahead() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(1);  // Only prefetch 1 layer ahead

    int                 data0, data1, data2, data3;
    std::vector<void *> tensors = { &data0, &data1, &data2, &data3 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2, 3 };

    scheduler.analyze_graph(tensors, sizes, layers);

    // At layer 0 with lookahead=1, should only prefetch layer 1
    auto requests = scheduler.on_kernel_start(0);
    assert(requests.size() == 1);
    assert(requests[0].layer_index == 1);

    std::cout << "[PASS] test_no_prefetch_beyond_lookahead\n";
    return true;
}

// Test 13: Priority ordering (closer layers = higher priority)
bool test_priority_ordering() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(3);

    int                 data0, data1, data2, data3;
    std::vector<void *> tensors = { &data0, &data1, &data2, &data3 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2, 3 };

    (void) scheduler.analyze_graph(tensors, sizes, layers);

    auto requests = scheduler.on_kernel_start(0);
    assert(requests.size() == 3);

    // Layer 1 (distance 1) should have highest priority (3)
    // Layer 2 (distance 2) should have priority 2
    // Layer 3 (distance 3) should have priority 1
    for (size_t i = 0; i < requests.size(); i++) {
        assert(requests[i].priority == scheduler.get_lookahead() - requests[i].layer_index + 1);
    }

    std::cout << "[PASS] test_priority_ordering\n";
    return true;
}

// Test 14: Needs prefetch check
bool test_needs_prefetch() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(2);

    int                 data0, data1, data2;
    std::vector<void *> tensors = { &data0, &data1, &data2 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2 };

    scheduler.analyze_graph(tensors, sizes, layers);

    // Before prefetch starts
    assert(scheduler.needs_prefetch(&data1, 1, 0) == true);   // Layer 1 from layer 0 - yes
    assert(scheduler.needs_prefetch(&data2, 2, 0) == true);   // Layer 2 from layer 0 - yes
    assert(scheduler.needs_prefetch(&data0, 0, 0) == false);  // Current layer - no
    assert(scheduler.needs_prefetch(&data2, 2, 1) == true);   // Layer 2 from layer 1 - yes

    // Out of lookahead range - use existing data2 pointer for test
    assert(scheduler.needs_prefetch(&data2, 3, 0) == false);  // Beyond lookahead (distance=3 > lookahead=2)

    // After prefetch starts
    scheduler.on_kernel_start(0);
    assert(scheduler.needs_prefetch(&data1, 1, 0) == false);  // Already in progress

    std::cout << "[PASS] test_needs_prefetch\n";
    return true;
}

// Test 15: Multiple on_kernel_start calls (simulate layer progression)
bool test_layer_progression() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(2);

    int                 data0, data1, data2, data3, data4;
    std::vector<void *> tensors = { &data0, &data1, &data2, &data3, &data4 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1, 2, 3, 4 };

    scheduler.analyze_graph(tensors, sizes, layers);

    // Layer 0: prefetch 1, 2
    auto req0 = scheduler.on_kernel_start(0);
    assert(req0.size() == 2);

    // Layer 1: prefetch 3 (2 already in progress)
    auto req1 = scheduler.on_kernel_start(1);
    assert(req1.size() == 1);
    assert(req1[0].layer_index == 3);

    // Layer 2: prefetch 4 (3 already in progress)
    auto req2 = scheduler.on_kernel_start(2);
    assert(req2.size() == 1);
    assert(req2[0].layer_index == 4);

    // Layer 3: no more layers to prefetch within lookahead
    auto req3 = scheduler.on_kernel_start(3);
    assert(req3.size() == 0);

    std::cout << "[PASS] test_layer_progression\n";
    return true;
}

// Test 16: Empty graph
bool test_empty_graph() {
    PrefetchScheduler scheduler;

    std::vector<void *> tensors;
    std::vector<size_t> sizes;
    std::vector<int>    layers;

    assert(scheduler.analyze_graph(tensors, sizes, layers) == 0);

    auto requests = scheduler.on_kernel_start(0);
    assert(requests.empty());

    std::cout << "[PASS] test_empty_graph\n";
    return true;
}

// Test 17: Reset stats
bool test_reset_stats() {
    PrefetchScheduler scheduler;

    int                 data0, data1;
    std::vector<void *> tensors = { &data0, &data1 };
    std::vector<size_t> sizes   = { 1024 * 1024, 1024 * 1024 };
    std::vector<int>    layers  = { 0, 1 };

    scheduler.analyze_graph(tensors, sizes, layers);
    scheduler.on_kernel_start(0);
    scheduler.mark_completed(&data1);

    assert(scheduler.get_prefetch_hits() == 1);

    scheduler.reset_stats();
    assert(scheduler.get_prefetch_hits() == 0);
    assert(scheduler.get_prefetch_cancels() == 0);

    std::cout << "[PASS] test_reset_stats\n";
    return true;
}

// Test 18: Request contains correct data
bool test_request_data() {
    PrefetchScheduler scheduler;
    scheduler.set_lookahead(1);

    int                 data0, data1;
    std::vector<void *> tensors = { &data0, &data1 };
    std::vector<size_t> sizes   = { 1024 * 1024, 2048 * 1024 };
    std::vector<int>    layers  = { 0, 1 };

    scheduler.analyze_graph(tensors, sizes, layers);

    auto requests = scheduler.on_kernel_start(0);
    assert(requests.size() == 1);
    assert(requests[0].tensor_data == &data1);
    assert(requests[0].size_bytes == 2048 * 1024);
    assert(requests[0].layer_index == 1);
    assert(requests[0].status == PrefetchStatus::IN_PROGRESS);

    std::cout << "[PASS] test_request_data\n";
    return true;
}

// Test 19: Hit rate with no operations returns 0
bool test_hit_rate_empty() {
    PrefetchScheduler scheduler;
    assert(scheduler.get_hit_rate() == 0.0f);

    std::cout << "[PASS] test_hit_rate_empty\n";
    return true;
}

// Test 20: MIN_PREFETCH_SIZE boundary
bool test_min_prefetch_size_boundary() {
    PrefetchScheduler scheduler;

    int                 data1, data2, data3;
    std::vector<void *> tensors = { &data1, &data2, &data3 };
    // Test exact boundary: 1023 (below), 1024 (at), 1025 (above)
    std::vector<size_t> sizes   = { 1023, 1024, 1025 };
    std::vector<int>    layers  = { 0, 1, 2 };

    // 1023 < 1024 (MIN_PREFETCH_SIZE), so only 2 should be added
    assert(scheduler.analyze_graph(tensors, sizes, layers) == 2);

    std::cout << "[PASS] test_min_prefetch_size_boundary\n";
    return true;
}

int main() {
    int passed = 0, failed = 0;

    if (test_default_lookahead()) {
        passed++;
    } else {
        failed++;
    }
    if (test_set_lookahead()) {
        passed++;
    } else {
        failed++;
    }
    if (test_analyze_graph()) {
        passed++;
    } else {
        failed++;
    }
    if (test_filter_small_tensors()) {
        passed++;
    } else {
        failed++;
    }
    if (test_on_kernel_start()) {
        passed++;
    } else {
        failed++;
    }
    if (test_mark_completed()) {
        passed++;
    } else {
        failed++;
    }
    if (test_cancel_all()) {
        passed++;
    } else {
        failed++;
    }
    if (test_is_prefetching()) {
        passed++;
    } else {
        failed++;
    }
    if (test_hit_rate()) {
        passed++;
    } else {
        failed++;
    }
    if (test_reset()) {
        passed++;
    } else {
        failed++;
    }
    if (test_no_prefetch_for_past_layers()) {
        passed++;
    } else {
        failed++;
    }
    if (test_no_prefetch_beyond_lookahead()) {
        passed++;
    } else {
        failed++;
    }
    if (test_priority_ordering()) {
        passed++;
    } else {
        failed++;
    }
    if (test_needs_prefetch()) {
        passed++;
    } else {
        failed++;
    }
    if (test_layer_progression()) {
        passed++;
    } else {
        failed++;
    }
    if (test_empty_graph()) {
        passed++;
    } else {
        failed++;
    }
    if (test_reset_stats()) {
        passed++;
    } else {
        failed++;
    }
    if (test_request_data()) {
        passed++;
    } else {
        failed++;
    }
    if (test_hit_rate_empty()) {
        passed++;
    } else {
        failed++;
    }
    if (test_min_prefetch_size_boundary()) {
        passed++;
    } else {
        failed++;
    }

    std::cout << "\n=== Prefetch Scheduler Tests ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
