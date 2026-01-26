//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Unit tests for MemoryBudget class
// Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.13)
//
// Tests:
// - Default initialization (90% of device memory)
// - Memory pressure detection (GREEN/YELLOW/RED/CRITICAL)
// - Allocation tracking
// - Deallocation tracking
// - Tier limits
// - can_allocate checks
// - get_available calculation
// - Pressure action helpers
// - Usage percentage calculation
//

#include "../ggml/src/ggml-sycl/memory-budget.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace ggml_sycl;

// =============================================================================
// Test Utilities
// =============================================================================

#define ASSERT_EQ(a, b)                                                                                 \
    do {                                                                                                \
        if ((a) != (b)) {                                                                               \
            std::cerr << "ASSERT_EQ failed: " << #a << " != " << #b << " at line " << __LINE__ << "\n"; \
            return false;                                                                               \
        }                                                                                               \
    } while (0)

#define ASSERT_TRUE(cond)                                                                    \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::cerr << "ASSERT_TRUE failed: " << #cond << " at line " << __LINE__ << "\n"; \
            return false;                                                                    \
        }                                                                                    \
    } while (0)

#define ASSERT_FALSE(cond)                                                                    \
    do {                                                                                      \
        if (cond) {                                                                           \
            std::cerr << "ASSERT_FALSE failed: " << #cond << " at line " << __LINE__ << "\n"; \
            return false;                                                                     \
        }                                                                                     \
    } while (0)

#define ASSERT_RANGE(val, lo, hi)                                                                                 \
    do {                                                                                                          \
        if ((val) < (lo) || (val) > (hi)) {                                                                       \
            std::cerr << "ASSERT_RANGE failed: " << #val << " (" << (val) << ") not in [" << (lo) << ", " << (hi) \
                      << "] at line " << __LINE__ << "\n";                                                        \
            return false;                                                                                         \
        }                                                                                                         \
    } while (0)

// =============================================================================
// Test 1: Default initialization
// =============================================================================

bool test_default_init() {
    MemoryBudget budget;
    budget.init(16ULL * 1024 * 1024 * 1024);  // 16 GB

    // Default budget is 90% of device memory = 14.4 GB
    size_t expected_budget = static_cast<size_t>(16ULL * 1024 * 1024 * 1024 * 0.90);
    ASSERT_EQ(budget.get_budget(), expected_budget);

    // Should be between 14 and 15 GB
    ASSERT_TRUE(budget.get_budget() > 14ULL * 1024 * 1024 * 1024);
    ASSERT_TRUE(budget.get_budget() < 15ULL * 1024 * 1024 * 1024);

    // Total memory should be stored
    ASSERT_EQ(budget.get_total_memory(), 16ULL * 1024 * 1024 * 1024);

    // No allocations initially
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    std::cout << "[PASS] test_default_init\n";
    return true;
}

// =============================================================================
// Test 2: Memory pressure GREEN
// =============================================================================

bool test_pressure_green() {
    MemoryBudget budget;
    budget.init(10ULL * 1024 * 1024 * 1024);  // 10 GB

    // No allocations = GREEN (0% usage)
    ASSERT_EQ(budget.get_pressure(), MemoryPressure::GREEN);

    std::cout << "[PASS] test_pressure_green\n";
    return true;
}

// =============================================================================
// Test 3: Track allocation
// =============================================================================

bool test_track_allocation() {
    MemoryBudget budget;
    budget.init(10ULL * 1024 * 1024 * 1024);

    // Track a 1KB allocation
    int dummy;
    budget.track_allocation(&dummy, 1024);

    ASSERT_EQ(budget.get_allocated(), 1024ULL);

    std::cout << "[PASS] test_track_allocation\n";
    return true;
}

// =============================================================================
// Test 4: Track free
// =============================================================================

bool test_track_free() {
    MemoryBudget budget;
    budget.init(10ULL * 1024 * 1024 * 1024);

    int dummy;
    budget.track_allocation(&dummy, 1024);
    ASSERT_EQ(budget.get_allocated(), 1024ULL);

    budget.track_free(&dummy);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    std::cout << "[PASS] test_track_free\n";
    return true;
}

// =============================================================================
// Test 5: Pressure levels
// =============================================================================

bool test_pressure_levels() {
    MemoryBudget budget;
    size_t       total = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    budget.init(total);
    size_t budget_size = budget.get_budget();         // ~9 GB

    // Allocate 60% -> GREEN (< 70%)
    char * p1 = reinterpret_cast<char *>(1);
    budget.track_allocation(p1, static_cast<size_t>(budget_size * 0.60));
    ASSERT_EQ(budget.get_pressure(), MemoryPressure::GREEN);

    // Allocate to 75% -> YELLOW (70-85%)
    char * p2 = reinterpret_cast<char *>(2);
    budget.track_allocation(p2, static_cast<size_t>(budget_size * 0.15));
    ASSERT_EQ(budget.get_pressure(), MemoryPressure::YELLOW);

    // Allocate to 90% -> RED (85-95%)
    char * p3 = reinterpret_cast<char *>(3);
    budget.track_allocation(p3, static_cast<size_t>(budget_size * 0.15));
    ASSERT_EQ(budget.get_pressure(), MemoryPressure::RED);

    // Allocate to 96% -> CRITICAL (> 95%)
    char * p4 = reinterpret_cast<char *>(4);
    budget.track_allocation(p4, static_cast<size_t>(budget_size * 0.06));
    ASSERT_EQ(budget.get_pressure(), MemoryPressure::CRITICAL);

    std::cout << "[PASS] test_pressure_levels\n";
    return true;
}

// =============================================================================
// Test 6: can_allocate
// =============================================================================

bool test_can_allocate() {
    MemoryBudget budget;
    budget.init(1024 * 1024);  // 1 MB

    // Should be able to allocate small amount
    ASSERT_TRUE(budget.can_allocate(1000));

    // Fill most of the budget
    char * p1          = reinterpret_cast<char *>(1);
    size_t budget_size = budget.get_budget();  // 900 KB
    budget.track_allocation(p1, budget_size - 100);

    // Small allocation should still work
    ASSERT_TRUE(budget.can_allocate(50));

    // Large allocation should fail
    ASSERT_FALSE(budget.can_allocate(200));

    std::cout << "[PASS] test_can_allocate\n";
    return true;
}

// =============================================================================
// Test 7: Tier limits
// =============================================================================

bool test_tier_limits() {
    MemoryBudget budget;
    budget.init(10ULL * 1024 * 1024 * 1024);  // 10 GB

    size_t budget_size = budget.get_budget();

    // HOT: 40%
    size_t hot_limit = budget.get_tier_limit(0);
    ASSERT_TRUE(hot_limit > budget_size * 0.39);
    ASSERT_TRUE(hot_limit < budget_size * 0.41);

    // WARM: 35%
    size_t warm_limit = budget.get_tier_limit(1);
    ASSERT_TRUE(warm_limit > budget_size * 0.34);
    ASSERT_TRUE(warm_limit < budget_size * 0.36);

    // COLD: 20%
    size_t cold_limit = budget.get_tier_limit(2);
    ASSERT_TRUE(cold_limit > budget_size * 0.19);
    ASSERT_TRUE(cold_limit < budget_size * 0.21);

    // Workspace: 5%
    size_t ws_limit = budget.get_tier_limit(3);
    ASSERT_TRUE(ws_limit > budget_size * 0.04);
    ASSERT_TRUE(ws_limit < budget_size * 0.06);

    // Invalid tier returns 0
    ASSERT_EQ(budget.get_tier_limit(4), 0ULL);
    ASSERT_EQ(budget.get_tier_limit(-1), 0ULL);

    std::cout << "[PASS] test_tier_limits\n";
    return true;
}

// =============================================================================
// Test 8: get_available
// =============================================================================

bool test_get_available() {
    MemoryBudget budget;
    budget.init(1024 * 1024);  // 1 MB

    size_t initial = budget.get_available();
    ASSERT_EQ(initial, budget.get_budget());

    char * p = reinterpret_cast<char *>(1);
    budget.track_allocation(p, 100);

    ASSERT_EQ(budget.get_available(), initial - 100);

    std::cout << "[PASS] test_get_available\n";
    return true;
}

// =============================================================================
// Test 9: Pressure actions
// =============================================================================

bool test_pressure_actions() {
    ASSERT_EQ(std::string(get_pressure_action(MemoryPressure::GREEN)), std::string("preload_aggressive"));
    ASSERT_EQ(std::string(get_pressure_action(MemoryPressure::YELLOW)), std::string("evict_lru"));
    ASSERT_EQ(std::string(get_pressure_action(MemoryPressure::RED)), std::string("evict_emergency"));
    ASSERT_EQ(std::string(get_pressure_action(MemoryPressure::CRITICAL)), std::string("evict_all_nonessential"));

    std::cout << "[PASS] test_pressure_actions\n";
    return true;
}

// =============================================================================
// Test 10: Usage percent
// =============================================================================

bool test_usage_percent() {
    MemoryBudget budget;
    budget.init(1000);  // Small for easy math

    char * p           = reinterpret_cast<char *>(1);
    size_t budget_size = budget.get_budget();     // 900
    budget.track_allocation(p, budget_size / 2);  // 450

    float pct = budget.get_usage_percent();
    ASSERT_TRUE(pct > 49.0f && pct < 51.0f);

    std::cout << "[PASS] test_usage_percent\n";
    return true;
}

// =============================================================================
// Test 11: Multiple allocations and frees
// =============================================================================

bool test_multiple_allocations() {
    MemoryBudget budget;
    budget.init(10ULL * 1024 * 1024);  // 10 MB

    // Track multiple allocations
    char * p1 = reinterpret_cast<char *>(1);
    char * p2 = reinterpret_cast<char *>(2);
    char * p3 = reinterpret_cast<char *>(3);

    budget.track_allocation(p1, 1024);
    budget.track_allocation(p2, 2048);
    budget.track_allocation(p3, 4096);

    ASSERT_EQ(budget.get_allocated(), 1024ULL + 2048ULL + 4096ULL);

    // Free middle allocation
    budget.track_free(p2);
    ASSERT_EQ(budget.get_allocated(), 1024ULL + 4096ULL);

    // Free first allocation
    budget.track_free(p1);
    ASSERT_EQ(budget.get_allocated(), 4096ULL);

    // Free last allocation
    budget.track_free(p3);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    std::cout << "[PASS] test_multiple_allocations\n";
    return true;
}

// =============================================================================
// Test 12: Double free handling (should be safe)
// =============================================================================

bool test_double_free() {
    MemoryBudget budget;
    budget.init(1024 * 1024);

    char * p = reinterpret_cast<char *>(1);
    budget.track_allocation(p, 1024);
    ASSERT_EQ(budget.get_allocated(), 1024ULL);

    budget.track_free(p);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    // Double free should be safe (no-op)
    budget.track_free(p);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    std::cout << "[PASS] test_double_free\n";
    return true;
}

// =============================================================================
// Test 13: Null pointer handling
// =============================================================================

bool test_null_pointer() {
    MemoryBudget budget;
    budget.init(1024 * 1024);

    // Null allocation should be no-op
    void * result = budget.track_allocation(nullptr, 1024);
    ASSERT_EQ(result, nullptr);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    // Null free should be no-op
    budget.track_free(nullptr);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    std::cout << "[PASS] test_null_pointer\n";
    return true;
}

// =============================================================================
// Test 14: Zero-size allocation
// =============================================================================

bool test_zero_size_allocation() {
    MemoryBudget budget;
    budget.init(1024 * 1024);

    char * p = reinterpret_cast<char *>(1);

    // Zero-size allocation should be no-op
    budget.track_allocation(p, 0);
    ASSERT_EQ(budget.get_allocated(), 0ULL);

    std::cout << "[PASS] test_zero_size_allocation\n";
    return true;
}

// =============================================================================
// Test 15: Pressure string conversion
// =============================================================================

bool test_pressure_to_string() {
    ASSERT_EQ(std::string(pressure_to_string(MemoryPressure::GREEN)), std::string("GREEN"));
    ASSERT_EQ(std::string(pressure_to_string(MemoryPressure::YELLOW)), std::string("YELLOW"));
    ASSERT_EQ(std::string(pressure_to_string(MemoryPressure::RED)), std::string("RED"));
    ASSERT_EQ(std::string(pressure_to_string(MemoryPressure::CRITICAL)), std::string("CRITICAL"));

    std::cout << "[PASS] test_pressure_to_string\n";
    return true;
}

// =============================================================================
// Test 16: Tier allocation configuration
// =============================================================================

bool test_tier_allocation_config() {
    MemoryBudget budget;
    budget.init(10ULL * 1024 * 1024 * 1024);

    const TierAllocation & tiers = budget.get_tier_allocation();

    // Verify default percentages
    ASSERT_TRUE(std::fabs(tiers.hot_pct - 0.40f) < 0.001f);
    ASSERT_TRUE(std::fabs(tiers.warm_pct - 0.35f) < 0.001f);
    ASSERT_TRUE(std::fabs(tiers.cold_pct - 0.20f) < 0.001f);
    ASSERT_TRUE(std::fabs(tiers.workspace_pct - 0.05f) < 0.001f);

    // Verify they sum to 100%
    float sum = tiers.hot_pct + tiers.warm_pct + tiers.cold_pct + tiers.workspace_pct;
    ASSERT_TRUE(std::fabs(sum - 1.0f) < 0.001f);

    std::cout << "[PASS] test_tier_allocation_config\n";
    return true;
}

// =============================================================================
// Test 17: Available becomes zero when over budget
// =============================================================================

bool test_available_zero_when_over() {
    MemoryBudget budget;
    budget.init(1000);

    size_t budget_size = budget.get_budget();  // 900

    // Allocate exactly the budget
    char * p = reinterpret_cast<char *>(1);
    budget.track_allocation(p, budget_size);

    ASSERT_EQ(budget.get_available(), 0ULL);

    // Allocate even more (over budget situation)
    char * p2 = reinterpret_cast<char *>(2);
    budget.track_allocation(p2, 100);

    // Available should still be 0, not underflow
    ASSERT_EQ(budget.get_available(), 0ULL);

    std::cout << "[PASS] test_available_zero_when_over\n";
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int passed = 0;
    int failed = 0;

    // Clear any existing env var to ensure default behavior
    unsetenv("GGML_SYCL_MEM_BUDGET");

    std::cout << "\n=== MemoryBudget Unit Tests ===\n\n";

    if (test_default_init()) {
        passed++;
    } else {
        failed++;
    }
    if (test_pressure_green()) {
        passed++;
    } else {
        failed++;
    }
    if (test_track_allocation()) {
        passed++;
    } else {
        failed++;
    }
    if (test_track_free()) {
        passed++;
    } else {
        failed++;
    }
    if (test_pressure_levels()) {
        passed++;
    } else {
        failed++;
    }
    if (test_can_allocate()) {
        passed++;
    } else {
        failed++;
    }
    if (test_tier_limits()) {
        passed++;
    } else {
        failed++;
    }
    if (test_get_available()) {
        passed++;
    } else {
        failed++;
    }
    if (test_pressure_actions()) {
        passed++;
    } else {
        failed++;
    }
    if (test_usage_percent()) {
        passed++;
    } else {
        failed++;
    }
    if (test_multiple_allocations()) {
        passed++;
    } else {
        failed++;
    }
    if (test_double_free()) {
        passed++;
    } else {
        failed++;
    }
    if (test_null_pointer()) {
        passed++;
    } else {
        failed++;
    }
    if (test_zero_size_allocation()) {
        passed++;
    } else {
        failed++;
    }
    if (test_pressure_to_string()) {
        passed++;
    } else {
        failed++;
    }
    if (test_tier_allocation_config()) {
        passed++;
    } else {
        failed++;
    }
    if (test_available_zero_when_over()) {
        passed++;
    } else {
        failed++;
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
