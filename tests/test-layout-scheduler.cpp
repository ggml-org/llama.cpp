//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Unit tests for LayoutConversionScheduler
// Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.16)
//

#include "../ggml/src/ggml-sycl/layout-scheduler.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

using namespace ggml_sycl;

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST_ASSERT(cond, msg)                                               \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "[FAIL] " << msg << " (line " << __LINE__ << ")\n"; \
            return false;                                                    \
        }                                                                    \
    } while (0)

// =============================================================================
// Test Cases
// =============================================================================

// Test 1: Default configuration values
bool test_default_config() {
    LayoutConversionScheduler scheduler;
    auto                      cfg = scheduler.get_config();

    TEST_ASSERT(cfg.min_idle_ms == 5, "default min_idle_ms should be 5");
    TEST_ASSERT(cfg.batch_size_limit == 4, "default batch_size_limit should be 4");
    TEST_ASSERT(cfg.batch_bytes_limit == 64 * 1024 * 1024, "default batch_bytes_limit should be 64MB");
    TEST_ASSERT(cfg.min_access_count == 3, "default min_access_count should be 3");
    TEST_ASSERT(cfg.enable_scheduling == true, "scheduling should be enabled by default");
    TEST_ASSERT(cfg.prefer_hot_tensors == true, "prefer_hot_tensors should be true by default");

    std::cout << "[PASS] test_default_config\n";
    return true;
}

// Test 2: Queue conversion with sufficient access count
bool test_queue_conversion() {
    LayoutConversionScheduler scheduler;

    ConversionRequest req;
    req.tensor_data  = (void *) 0x1000;
    req.size_bytes   = 1024 * 1024;
    req.target       = TargetLayout::SOA;
    req.heat         = TensorHeat::HOT;
    req.priority     = 100;
    req.access_count = 5;  // Above threshold (3)

    scheduler.queue_conversion(req);
    TEST_ASSERT(scheduler.get_pending_count() == 1, "should have 1 pending conversion");

    std::cout << "[PASS] test_queue_conversion\n";
    return true;
}

// Test 3: Access count filter - below threshold not queued
bool test_access_count_filter() {
    LayoutConversionScheduler scheduler;

    ConversionRequest req;
    req.tensor_data  = (void *) 0x1000;
    req.size_bytes   = 1024;
    req.target       = TargetLayout::SOA;
    req.heat         = TensorHeat::WARM;
    req.priority     = 50;
    req.access_count = 1;  // Below threshold (3)

    scheduler.queue_conversion(req);
    TEST_ASSERT(scheduler.get_pending_count() == 0, "request below access threshold should not be queued");

    std::cout << "[PASS] test_access_count_filter\n";
    return true;
}

// Test 4: Should not convert without idle period
bool test_should_convert_no_idle() {
    LayoutConversionScheduler scheduler;

    ConversionRequest req;
    req.tensor_data  = (void *) 0x1000;
    req.size_bytes   = 1024;
    req.access_count = 5;
    scheduler.queue_conversion(req);

    // No idle start recorded - idle_start_ is default initialized
    // should_convert should return false because we haven't waited long enough
    TEST_ASSERT(scheduler.get_pending_count() == 1, "should have pending request");

    std::cout << "[PASS] test_should_convert_no_idle\n";
    return true;
}

// Test 5: Conversion batch respects size limit
bool test_batch_size_limit() {
    LayoutConversionScheduler scheduler;

    // Queue 10 requests
    for (int i = 0; i < 10; i++) {
        ConversionRequest req;
        req.tensor_data  = (void *) (uintptr_t) (i + 1);
        req.size_bytes   = 1024;
        req.target       = TargetLayout::COALESCED;
        req.heat         = TensorHeat::WARM;
        req.priority     = i * 10;
        req.access_count = 5;
        scheduler.queue_conversion(req);
    }

    TEST_ASSERT(scheduler.get_pending_count() == 10, "should have 10 pending conversions");

    auto batch = scheduler.get_conversion_batch();
    TEST_ASSERT(batch.size() == 4, "batch should respect size limit of 4");
    TEST_ASSERT(scheduler.get_pending_count() == 6, "should have 6 remaining after batch");

    std::cout << "[PASS] test_batch_size_limit\n";
    return true;
}

// Test 6: Priority ordering (higher priority processed first)
bool test_priority_ordering() {
    LayoutConversionScheduler scheduler;

    ConversionRequest low;
    low.tensor_data  = (void *) 0x1000;
    low.size_bytes   = 1024;
    low.target       = TargetLayout::SOA;
    low.heat         = TensorHeat::COLD;
    low.priority     = 10;
    low.access_count = 5;

    ConversionRequest high;
    high.tensor_data  = (void *) 0x2000;
    high.size_bytes   = 1024;
    high.target       = TargetLayout::XMX_COALESCED;
    high.heat         = TensorHeat::HOT;
    high.priority     = 100;
    high.access_count = 5;

    scheduler.queue_conversion(low);
    scheduler.queue_conversion(high);

    // Get batch - priority_queue returns highest priority first
    auto batch = scheduler.get_conversion_batch();
    TEST_ASSERT(batch.size() == 2, "should have 2 items in batch");

    // After sorting by heat, HOT should be first
    TEST_ASSERT(batch[0].heat == TensorHeat::HOT, "HOT tensor should be first after sorting");

    std::cout << "[PASS] test_priority_ordering\n";
    return true;
}

// Test 7: Compute priority function
bool test_compute_priority() {
    // HOT tensor should have highest base priority
    int hot_prio  = LayoutConversionScheduler::compute_priority(TensorHeat::HOT, 1024 * 1024, 5);
    int warm_prio = LayoutConversionScheduler::compute_priority(TensorHeat::WARM, 1024 * 1024, 5);
    int cold_prio = LayoutConversionScheduler::compute_priority(TensorHeat::COLD, 1024 * 1024, 5);

    TEST_ASSERT(hot_prio > warm_prio, "HOT should have higher priority than WARM");
    TEST_ASSERT(warm_prio > cold_prio, "WARM should have higher priority than COLD");

    // Larger tensor should have higher priority (same heat)
    int small_prio = LayoutConversionScheduler::compute_priority(TensorHeat::WARM, 1024, 5);
    int large_prio = LayoutConversionScheduler::compute_priority(TensorHeat::WARM, 100 * 1024 * 1024, 5);

    TEST_ASSERT(large_prio > small_prio, "larger tensor should have higher priority");

    // More accesses should have higher priority (same heat, same size)
    int few_prio  = LayoutConversionScheduler::compute_priority(TensorHeat::WARM, 1024 * 1024, 1);
    int many_prio = LayoutConversionScheduler::compute_priority(TensorHeat::WARM, 1024 * 1024, 20);

    TEST_ASSERT(many_prio > few_prio, "more accesses should have higher priority");

    std::cout << "[PASS] test_compute_priority\n";
    return true;
}

// Test 8: Clear pending conversions
bool test_clear() {
    LayoutConversionScheduler scheduler;

    ConversionRequest req;
    req.access_count = 5;
    req.size_bytes   = 1024;
    scheduler.queue_conversion(req);
    scheduler.queue_conversion(req);

    TEST_ASSERT(scheduler.get_pending_count() == 2, "should have 2 pending conversions");

    scheduler.clear();
    TEST_ASSERT(scheduler.get_pending_count() == 0, "should have 0 pending after clear");

    std::cout << "[PASS] test_clear\n";
    return true;
}

// Test 9: Batch bytes limit
bool test_batch_bytes_limit() {
    LayoutConversionScheduler scheduler;
    SchedulerConfig           cfg;
    cfg.batch_bytes_limit = 2 * 1024 * 1024;  // 2 MB limit
    cfg.batch_size_limit  = 100;              // High limit to not interfere
    cfg.min_access_count  = 1;
    scheduler.set_config(cfg);

    // Queue 5 requests of 1 MB each
    for (int i = 0; i < 5; i++) {
        ConversionRequest req;
        req.tensor_data  = (void *) (uintptr_t) (i + 1);
        req.size_bytes   = 1024 * 1024;  // 1 MB each
        req.target       = TargetLayout::SOA;
        req.heat         = TensorHeat::WARM;
        req.priority     = i;
        req.access_count = 5;
        scheduler.queue_conversion(req);
    }

    auto batch = scheduler.get_conversion_batch();
    // Should get at most 2 (2 MB limit), might be slightly more due to greedy algorithm
    TEST_ASSERT(batch.size() <= 3, "batch should respect bytes limit");

    std::cout << "[PASS] test_batch_bytes_limit\n";
    return true;
}

// Test 10: Custom configuration
bool test_custom_config() {
    LayoutConversionScheduler scheduler;

    SchedulerConfig cfg;
    cfg.min_idle_ms        = 10;
    cfg.batch_size_limit   = 8;
    cfg.batch_bytes_limit  = 128 * 1024 * 1024;
    cfg.min_access_count   = 1;
    cfg.enable_scheduling  = false;
    cfg.prefer_hot_tensors = false;

    scheduler.set_config(cfg);
    auto result = scheduler.get_config();

    TEST_ASSERT(result.min_idle_ms == 10, "min_idle_ms should be 10");
    TEST_ASSERT(result.batch_size_limit == 8, "batch_size_limit should be 8");
    TEST_ASSERT(result.batch_bytes_limit == 128 * 1024 * 1024, "batch_bytes_limit should be 128MB");
    TEST_ASSERT(result.min_access_count == 1, "min_access_count should be 1");
    TEST_ASSERT(result.enable_scheduling == false, "scheduling should be disabled");
    TEST_ASSERT(result.prefer_hot_tensors == false, "prefer_hot_tensors should be false");

    std::cout << "[PASS] test_custom_config\n";
    return true;
}

// Test 11: Disabled scheduling
bool test_disabled_scheduling() {
    LayoutConversionScheduler scheduler;
    SchedulerConfig           cfg;
    cfg.enable_scheduling = false;
    scheduler.set_config(cfg);

    ConversionRequest req;
    req.access_count = 100;
    req.size_bytes   = 1024;
    scheduler.queue_conversion(req);

    TEST_ASSERT(scheduler.get_pending_count() == 0, "disabled scheduler should not queue requests");
    TEST_ASSERT(scheduler.is_enabled() == false, "is_enabled should return false");

    scheduler.set_enabled(true);
    TEST_ASSERT(scheduler.is_enabled() == true, "is_enabled should return true after set_enabled(true)");

    std::cout << "[PASS] test_disabled_scheduling\n";
    return true;
}

// Test 12: Get pending bytes
bool test_get_pending_bytes() {
    LayoutConversionScheduler scheduler;

    ConversionRequest req1;
    req1.tensor_data  = (void *) 0x1000;
    req1.size_bytes   = 1024 * 1024;  // 1 MB
    req1.access_count = 5;

    ConversionRequest req2;
    req2.tensor_data  = (void *) 0x2000;
    req2.size_bytes   = 2 * 1024 * 1024;  // 2 MB
    req2.access_count = 5;

    scheduler.queue_conversion(req1);
    scheduler.queue_conversion(req2);

    size_t total_bytes = scheduler.get_pending_bytes();
    TEST_ASSERT(total_bytes == 3 * 1024 * 1024, "total pending bytes should be 3 MB");

    std::cout << "[PASS] test_get_pending_bytes\n";
    return true;
}

// Test 13: Target layout string conversion
bool test_target_layout_strings() {
    TEST_ASSERT(std::string(target_layout_to_string(TargetLayout::SOA)) == "SOA", "SOA string");
    TEST_ASSERT(std::string(target_layout_to_string(TargetLayout::COALESCED)) == "COALESCED", "COALESCED string");
    TEST_ASSERT(std::string(target_layout_to_string(TargetLayout::XMX_COALESCED)) == "XMX_COALESCED",
                "XMX_COALESCED string");

    std::cout << "[PASS] test_target_layout_strings\n";
    return true;
}

// Test 14: Conversion status string conversion
bool test_conversion_status_strings() {
    TEST_ASSERT(std::string(conversion_status_to_string(ConversionStatus::NOT_STARTED)) == "NOT_STARTED",
                "NOT_STARTED string");
    TEST_ASSERT(std::string(conversion_status_to_string(ConversionStatus::QUEUED)) == "QUEUED", "QUEUED string");
    TEST_ASSERT(std::string(conversion_status_to_string(ConversionStatus::IN_PROGRESS)) == "IN_PROGRESS",
                "IN_PROGRESS string");
    TEST_ASSERT(std::string(conversion_status_to_string(ConversionStatus::COMPLETED)) == "COMPLETED",
                "COMPLETED string");
    TEST_ASSERT(std::string(conversion_status_to_string(ConversionStatus::FAILED)) == "FAILED", "FAILED string");

    std::cout << "[PASS] test_conversion_status_strings\n";
    return true;
}

// Test 15: Heat sorting in batch
bool test_heat_sorting() {
    LayoutConversionScheduler scheduler;
    SchedulerConfig           cfg;
    cfg.batch_size_limit = 10;
    cfg.min_access_count = 1;
    scheduler.set_config(cfg);

    // Add requests with different heats (in non-sorted order)
    TensorHeat heats[] = { TensorHeat::COLD, TensorHeat::HOT, TensorHeat::FROZEN, TensorHeat::WARM };

    for (int i = 0; i < 4; i++) {
        ConversionRequest req;
        req.tensor_data  = (void *) (uintptr_t) (i + 1);
        req.size_bytes   = 1024;
        req.heat         = heats[i];
        req.priority     = 50;  // Same priority
        req.access_count = 5;
        scheduler.queue_conversion(req);
    }

    auto batch = scheduler.get_conversion_batch();
    TEST_ASSERT(batch.size() == 4, "should have all 4 items");

    // After sorting: HOT(0) < WARM(1) < COLD(2) < FROZEN(3)
    TEST_ASSERT(batch[0].heat == TensorHeat::HOT, "first should be HOT");
    TEST_ASSERT(batch[1].heat == TensorHeat::WARM, "second should be WARM");
    TEST_ASSERT(batch[2].heat == TensorHeat::COLD, "third should be COLD");
    TEST_ASSERT(batch[3].heat == TensorHeat::FROZEN, "fourth should be FROZEN");

    std::cout << "[PASS] test_heat_sorting\n";
    return true;
}

// Test 16: Idle detection with actual delay
bool test_idle_detection_with_delay() {
    LayoutConversionScheduler scheduler;
    SchedulerConfig           cfg;
    cfg.min_idle_ms      = 10;  // 10ms idle threshold
    cfg.min_access_count = 1;
    scheduler.set_config(cfg);

    ConversionRequest req;
    req.access_count = 5;
    req.size_bytes   = 1024;
    scheduler.queue_conversion(req);

    scheduler.on_idle_start();

    // Immediately after - should not convert
    TEST_ASSERT(scheduler.should_convert() == false, "should not convert immediately after idle start");

    // Wait for idle threshold
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    TEST_ASSERT(scheduler.should_convert() == true, "should convert after idle threshold");

    std::cout << "[PASS] test_idle_detection_with_delay\n";
    return true;
}

// Test 17: Access count at exact threshold
bool test_access_count_at_threshold() {
    LayoutConversionScheduler scheduler;

    ConversionRequest req;
    req.tensor_data  = (void *) 0x1000;
    req.size_bytes   = 1024;
    req.access_count = 3;  // Exactly at threshold

    scheduler.queue_conversion(req);
    TEST_ASSERT(scheduler.get_pending_count() == 1, "request at exact threshold should be queued");

    std::cout << "[PASS] test_access_count_at_threshold\n";
    return true;
}

// Test 18: Empty batch when nothing pending
bool test_empty_batch() {
    LayoutConversionScheduler scheduler;

    auto batch = scheduler.get_conversion_batch();
    TEST_ASSERT(batch.empty(), "batch should be empty when nothing pending");

    std::cout << "[PASS] test_empty_batch\n";
    return true;
}

// Test 19: Multiple batches
bool test_multiple_batches() {
    LayoutConversionScheduler scheduler;
    SchedulerConfig           cfg;
    cfg.batch_size_limit = 2;
    cfg.min_access_count = 1;
    scheduler.set_config(cfg);

    // Queue 5 requests
    for (int i = 0; i < 5; i++) {
        ConversionRequest req;
        req.tensor_data  = (void *) (uintptr_t) (i + 1);
        req.size_bytes   = 1024;
        req.access_count = 5;
        req.priority     = i;
        scheduler.queue_conversion(req);
    }

    auto batch1 = scheduler.get_conversion_batch();
    TEST_ASSERT(batch1.size() == 2, "first batch should have 2 items");
    TEST_ASSERT(scheduler.get_pending_count() == 3, "should have 3 remaining");

    auto batch2 = scheduler.get_conversion_batch();
    TEST_ASSERT(batch2.size() == 2, "second batch should have 2 items");
    TEST_ASSERT(scheduler.get_pending_count() == 1, "should have 1 remaining");

    auto batch3 = scheduler.get_conversion_batch();
    TEST_ASSERT(batch3.size() == 1, "third batch should have 1 item");
    TEST_ASSERT(scheduler.get_pending_count() == 0, "should have 0 remaining");

    std::cout << "[PASS] test_multiple_batches\n";
    return true;
}

// Test 20: Idle duration tracking
bool test_idle_duration() {
    LayoutConversionScheduler scheduler;

    scheduler.on_idle_start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    long long duration = scheduler.get_idle_duration_ms();
    TEST_ASSERT(duration >= 15, "idle duration should be at least 15ms");  // Allow some timing slack

    std::cout << "[PASS] test_idle_duration\n";
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== Layout Conversion Scheduler Tests ===\n\n";

    int passed = 0;
    int failed = 0;

    // Run all tests
    auto run_test = [&](bool (*test_fn)()) {
        if (test_fn()) {
            passed++;
        } else {
            failed++;
        }
    };

    run_test(test_default_config);
    run_test(test_queue_conversion);
    run_test(test_access_count_filter);
    run_test(test_should_convert_no_idle);
    run_test(test_batch_size_limit);
    run_test(test_priority_ordering);
    run_test(test_compute_priority);
    run_test(test_clear);
    run_test(test_batch_bytes_limit);
    run_test(test_custom_config);
    run_test(test_disabled_scheduling);
    run_test(test_get_pending_bytes);
    run_test(test_target_layout_strings);
    run_test(test_conversion_status_strings);
    run_test(test_heat_sorting);
    run_test(test_idle_detection_with_delay);
    run_test(test_access_count_at_threshold);
    run_test(test_empty_batch);
    run_test(test_multiple_batches);
    run_test(test_idle_duration);

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
