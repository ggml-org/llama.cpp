// Heat-Aware Eviction Policy unit tests
// Tests for priority-based eviction using heat classification
// Part of unified memory management system (epic llama.cpp-6hp, task llama.cpp-6hp.12)
//
// TDD: Tests written FIRST, implementation must make these pass.
//
// Heat Levels:
// - HOT (1000): Frequently accessed, NEVER evicted
// - WARM (100): Moderately accessed
// - COLD (10): Infrequently accessed, evicted first
// - FROZEN (50): Static tensors, between WARM and COLD
//
// Priority Formula:
//   priority = heat_score * frequency / (recency + 1.0) - size_penalty

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "ggml-sycl/heat-aware-eviction.hpp"

using namespace ggml_sycl;

// Helper to create a test CacheEntry
static uint64_t g_entry_id_counter = 0;

static CacheEntry make_entry(TensorHeat heat, size_t size_bytes, uint64_t access_count = 1) {
    CacheEntry entry;
    entry.tensor       = nullptr;  // No actual tensor for unit tests
    entry.heat         = heat;
    entry.size_bytes   = size_bytes;
    entry.last_access  = std::chrono::steady_clock::now();
    entry.access_count = access_count;
    entry.id           = ++g_entry_id_counter;
    return entry;
}

// Make entry with specific last access time (seconds ago)
static CacheEntry make_entry_aged(TensorHeat heat, size_t size_bytes, uint64_t access_count, int seconds_ago) {
    CacheEntry entry  = make_entry(heat, size_bytes, access_count);
    entry.last_access = std::chrono::steady_clock::now() - std::chrono::seconds(seconds_ago);
    return entry;
}

// =============================================================================
// Test 1: test_priority_heat_ordering
// HOT > WARM > COLD for same recency and access count
// =============================================================================
static bool test_priority_heat_ordering() {
    printf("TEST: test_priority_heat_ordering\n");

    reset_session_start();

    // Create entries with same recency and access count, different heat
    auto hot_entry  = make_entry(TensorHeat::HOT, 1024 * 1024, 10);
    auto warm_entry = make_entry(TensorHeat::WARM, 1024 * 1024, 10);
    auto cold_entry = make_entry(TensorHeat::COLD, 1024 * 1024, 10);

    float hot_priority  = compute_priority(hot_entry);
    float warm_priority = compute_priority(warm_entry);
    float cold_priority = compute_priority(cold_entry);

    if (hot_priority <= warm_priority) {
        printf("  FAIL: HOT priority (%.2f) should be > WARM priority (%.2f)\n", hot_priority, warm_priority);
        return false;
    }

    if (warm_priority <= cold_priority) {
        printf("  FAIL: WARM priority (%.2f) should be > COLD priority (%.2f)\n", warm_priority, cold_priority);
        return false;
    }

    printf("  PASS: Priority ordering HOT(%.2f) > WARM(%.2f) > COLD(%.2f)\n", hot_priority, warm_priority, cold_priority);
    return true;
}

// =============================================================================
// Test 2: test_evict_cold_first
// Under MODERATE pressure, only COLD tensors should be evicted
// =============================================================================
static bool test_evict_cold_first() {
    printf("TEST: test_evict_cold_first\n");

    reset_session_start();

    std::vector<CacheEntry> cache;

    // Add entries: 1 HOT, 1 WARM, 2 COLD
    // Total: 4MB, Budget: 3MB, need to evict 1MB
    cache.push_back(make_entry(TensorHeat::HOT, 1024 * 1024, 10));   // 1MB HOT
    cache.push_back(make_entry(TensorHeat::WARM, 1024 * 1024, 10));  // 1MB WARM
    cache.push_back(make_entry(TensorHeat::COLD, 1024 * 1024, 10));  // 1MB COLD (should be evicted)
    cache.push_back(make_entry(TensorHeat::COLD, 1024 * 1024, 10));  // 1MB COLD (should be evicted if needed)

    // Need to evict 1MB (from 4MB to 3MB)
    size_t target_bytes = 3 * 1024 * 1024;
    size_t freed        = evict_to_budget(cache, target_bytes);

    // Should have evicted exactly 1 COLD entry (1MB)
    if (freed != 1024 * 1024) {
        printf("  FAIL: Expected to free 1MB, freed %zu bytes\n", freed);
        return false;
    }

    // Verify cache now has 3 entries
    if (cache.size() != 3) {
        printf("  FAIL: Expected 3 entries remaining, got %zu\n", cache.size());
        return false;
    }

    // Verify HOT and WARM are still there
    int hot_count = 0, warm_count = 0;
    for (const auto & entry : cache) {
        if (entry.heat == TensorHeat::HOT)
            hot_count++;
        else if (entry.heat == TensorHeat::WARM)
            warm_count++;
    }

    if (hot_count != 1 || warm_count != 1) {
        printf("  FAIL: HOT and WARM entries should not be evicted\n");
        return false;
    }

    printf("  PASS: Only COLD entry was evicted (freed %zu bytes)\n", freed);
    return true;
}

// =============================================================================
// Test 3: test_hot_protected
// Under CRITICAL pressure, HOT tensors must remain
// =============================================================================
static bool test_hot_protected() {
    printf("TEST: test_hot_protected\n");

    reset_session_start();

    std::vector<CacheEntry> cache;

    // Add 1 HOT (2MB) and 1 COLD (1MB)
    // Total: 3MB, try to get down to 1MB
    auto hot_entry  = make_entry(TensorHeat::HOT, 2 * 1024 * 1024, 10);
    auto cold_entry = make_entry(TensorHeat::COLD, 1024 * 1024, 10);

    uint64_t hot_id = hot_entry.id;
    cache.push_back(hot_entry);
    cache.push_back(cold_entry);

    // Try to evict down to 1MB (only possible by evicting HOT, which is protected)
    size_t target_bytes = 1024 * 1024;
    size_t freed        = evict_to_budget(cache, target_bytes);

    // Should only evict the COLD entry (1MB), HOT remains
    if (freed != 1024 * 1024) {
        printf("  FAIL: Expected to free 1MB (COLD only), freed %zu bytes\n", freed);
        return false;
    }

    // Verify HOT entry is still in cache
    bool hot_found = false;
    for (const auto & entry : cache) {
        if (entry.id == hot_id) {
            hot_found = true;
            break;
        }
    }

    if (!hot_found) {
        printf("  FAIL: HOT entry was evicted (should be protected)\n");
        return false;
    }

    // Verify only 1 entry remains (the HOT one)
    if (cache.size() != 1) {
        printf("  FAIL: Expected 1 entry remaining (HOT), got %zu\n", cache.size());
        return false;
    }

    printf("  PASS: HOT tensor protected, only COLD evicted\n");
    return true;
}

// =============================================================================
// Test 4: test_lru_within_heat
// Older WARM should be evicted before newer WARM
// =============================================================================
static bool test_lru_within_heat() {
    printf("TEST: test_lru_within_heat\n");

    reset_session_start();

    std::vector<CacheEntry> cache;

    // Create two WARM entries with different ages
    auto older_warm = make_entry_aged(TensorHeat::WARM, 1024 * 1024, 10, 60);  // 60 seconds ago
    auto newer_warm = make_entry_aged(TensorHeat::WARM, 1024 * 1024, 10, 1);   // 1 second ago

    uint64_t older_id = older_warm.id;
    uint64_t newer_id = newer_warm.id;

    cache.push_back(older_warm);
    cache.push_back(newer_warm);

    // Evict 1MB (one entry)
    size_t target_bytes = 1024 * 1024;
    size_t freed        = evict_to_budget(cache, target_bytes);

    if (freed != 1024 * 1024) {
        printf("  FAIL: Expected to free 1MB, freed %zu bytes\n", freed);
        return false;
    }

    // Verify older entry was evicted
    bool older_found = false, newer_found = false;
    for (const auto & entry : cache) {
        if (entry.id == older_id)
            older_found = true;
        if (entry.id == newer_id)
            newer_found = true;
    }

    if (older_found) {
        printf("  FAIL: Older WARM entry should have been evicted\n");
        return false;
    }

    if (!newer_found) {
        printf("  FAIL: Newer WARM entry should still be in cache\n");
        return false;
    }

    printf("  PASS: Older WARM evicted before newer WARM\n");
    return true;
}

// =============================================================================
// Test 5: test_size_penalty
// Larger tensor of same heat should be evicted before smaller
// =============================================================================
static bool test_size_penalty() {
    printf("TEST: test_size_penalty\n");

    reset_session_start();

    std::vector<CacheEntry> cache;

    // Create two COLD entries with same age but different sizes
    // Large: 64MB, Small: 1MB
    auto large_cold = make_entry(TensorHeat::COLD, 64 * 1024 * 1024, 10);
    auto small_cold = make_entry(TensorHeat::COLD, 1024 * 1024, 10);

    uint64_t large_id = large_cold.id;
    uint64_t small_id = small_cold.id;

    cache.push_back(large_cold);
    cache.push_back(small_cold);

    // Check priorities - larger should have lower priority due to size penalty
    float large_priority = compute_priority(large_cold);
    float small_priority = compute_priority(small_cold);

    if (large_priority >= small_priority) {
        printf("  FAIL: Large tensor priority (%.2f) should be < small tensor priority (%.2f)\n", large_priority,
               small_priority);
        return false;
    }

    // Evict to free some space - large should be evicted first
    size_t target_bytes = 1024 * 1024;  // Keep only 1MB
    evict_to_budget(cache, target_bytes);

    // Verify large was evicted
    bool large_found = false, small_found = false;
    for (const auto & entry : cache) {
        if (entry.id == large_id)
            large_found = true;
        if (entry.id == small_id)
            small_found = true;
    }

    if (large_found) {
        printf("  FAIL: Large tensor should have been evicted\n");
        return false;
    }

    if (!small_found) {
        printf("  FAIL: Small tensor should still be in cache\n");
        return false;
    }

    printf("  PASS: Large tensor (%.2f) evicted before small tensor (%.2f) due to size penalty\n", large_priority,
           small_priority);
    return true;
}

// =============================================================================
// Test 6: test_pressure_levels
// Verify pressure calculation at different utilization levels
// =============================================================================
static bool test_pressure_levels() {
    printf("TEST: test_pressure_levels\n");

    size_t budget = 100 * 1024 * 1024;  // 100MB budget

    // Test NONE: < 70%
    {
        size_t         current = 50 * 1024 * 1024;  // 50MB used
        size_t         pending = 10 * 1024 * 1024;  // 10MB pending
        MemoryPressure pressure = get_pressure(budget, current, pending);
        if (pressure != MemoryPressure::NONE) {
            printf("  FAIL: 60%% utilization should be NONE, got %s\n", pressure_to_string(pressure));
            return false;
        }
    }

    // Test MODERATE: 70-85%
    {
        size_t         current  = 70 * 1024 * 1024;  // 70MB used
        size_t         pending  = 5 * 1024 * 1024;   // 5MB pending = 75%
        MemoryPressure pressure = get_pressure(budget, current, pending);
        if (pressure != MemoryPressure::MODERATE) {
            printf("  FAIL: 75%% utilization should be MODERATE, got %s\n", pressure_to_string(pressure));
            return false;
        }
    }

    // Test HIGH: 85-95%
    {
        size_t         current  = 85 * 1024 * 1024;  // 85MB used
        size_t         pending  = 5 * 1024 * 1024;   // 5MB pending = 90%
        MemoryPressure pressure = get_pressure(budget, current, pending);
        if (pressure != MemoryPressure::HIGH) {
            printf("  FAIL: 90%% utilization should be HIGH, got %s\n", pressure_to_string(pressure));
            return false;
        }
    }

    // Test CRITICAL: 95-100%
    {
        size_t         current  = 90 * 1024 * 1024;  // 90MB used
        size_t         pending  = 8 * 1024 * 1024;   // 8MB pending = 98%
        MemoryPressure pressure = get_pressure(budget, current, pending);
        if (pressure != MemoryPressure::CRITICAL) {
            printf("  FAIL: 98%% utilization should be CRITICAL, got %s\n", pressure_to_string(pressure));
            return false;
        }
    }

    // Test IMPOSSIBLE: > 100%
    {
        size_t         current  = 90 * 1024 * 1024;   // 90MB used
        size_t         pending  = 20 * 1024 * 1024;   // 20MB pending = 110%
        MemoryPressure pressure = get_pressure(budget, current, pending);
        if (pressure != MemoryPressure::IMPOSSIBLE) {
            printf("  FAIL: 110%% utilization should be IMPOSSIBLE, got %s\n", pressure_to_string(pressure));
            return false;
        }
    }

    printf("  PASS: All pressure levels calculated correctly\n");
    return true;
}

// =============================================================================
// Test 7: test_heat_multiplier_values
// Verify heat multiplier returns correct values
// =============================================================================
static bool test_heat_multiplier_values() {
    printf("TEST: test_heat_multiplier_values\n");

    float hot    = heat_multiplier(TensorHeat::HOT);
    float warm   = heat_multiplier(TensorHeat::WARM);
    float cold   = heat_multiplier(TensorHeat::COLD);
    float frozen = heat_multiplier(TensorHeat::FROZEN);

    if (hot != 1000.0f) {
        printf("  FAIL: HOT multiplier should be 1000, got %.2f\n", hot);
        return false;
    }

    if (warm != 100.0f) {
        printf("  FAIL: WARM multiplier should be 100, got %.2f\n", warm);
        return false;
    }

    if (cold != 10.0f) {
        printf("  FAIL: COLD multiplier should be 10, got %.2f\n", cold);
        return false;
    }

    if (frozen != 50.0f) {
        printf("  FAIL: FROZEN multiplier should be 50, got %.2f\n", frozen);
        return false;
    }

    printf("  PASS: Heat multipliers correct - HOT(1000), WARM(100), FROZEN(50), COLD(10)\n");
    return true;
}

// =============================================================================
// Test 8: test_frequency_affects_priority
// Higher access count should increase priority
// =============================================================================
static bool test_frequency_affects_priority() {
    printf("TEST: test_frequency_affects_priority\n");

    reset_session_start();

    // Same heat, same size, different access counts
    auto low_freq  = make_entry(TensorHeat::WARM, 1024 * 1024, 1);    // 1 access
    auto high_freq = make_entry(TensorHeat::WARM, 1024 * 1024, 100);  // 100 accesses

    float low_priority  = compute_priority(low_freq);
    float high_priority = compute_priority(high_freq);

    if (high_priority <= low_priority) {
        printf("  FAIL: High frequency priority (%.2f) should be > low frequency priority (%.2f)\n", high_priority,
               low_priority);
        return false;
    }

    printf("  PASS: Higher access count increases priority (%.2f > %.2f)\n", high_priority, low_priority);
    return true;
}

// =============================================================================
// Test 9: test_evict_returns_bytes_freed
// evict_to_budget should return accurate bytes freed
// =============================================================================
static bool test_evict_returns_bytes_freed() {
    printf("TEST: test_evict_returns_bytes_freed\n");

    reset_session_start();

    std::vector<CacheEntry> cache;

    // Add 5 COLD entries of 1MB each
    for (int i = 0; i < 5; i++) {
        cache.push_back(make_entry(TensorHeat::COLD, 1024 * 1024, 1));
    }

    // Evict to 2MB (should free 3MB)
    size_t target_bytes = 2 * 1024 * 1024;
    size_t freed        = evict_to_budget(cache, target_bytes);

    if (freed != 3 * 1024 * 1024) {
        printf("  FAIL: Expected to free 3MB, freed %zu bytes\n", freed);
        return false;
    }

    // Verify 2 entries remain
    if (cache.size() != 2) {
        printf("  FAIL: Expected 2 entries remaining, got %zu\n", cache.size());
        return false;
    }

    printf("  PASS: evict_to_budget returned correct bytes freed (%zu)\n", freed);
    return true;
}

// =============================================================================
// Test 10: test_no_eviction_needed
// When under budget, no eviction should occur
// =============================================================================
static bool test_no_eviction_needed() {
    printf("TEST: test_no_eviction_needed\n");

    reset_session_start();

    std::vector<CacheEntry> cache;
    cache.push_back(make_entry(TensorHeat::COLD, 1024 * 1024, 1));  // 1MB

    // Target is larger than current usage
    size_t target_bytes = 10 * 1024 * 1024;
    size_t freed        = evict_to_budget(cache, target_bytes);

    if (freed != 0) {
        printf("  FAIL: No eviction needed, but freed %zu bytes\n", freed);
        return false;
    }

    if (cache.size() != 1) {
        printf("  FAIL: Cache should still have 1 entry\n");
        return false;
    }

    printf("  PASS: No eviction when under budget\n");
    return true;
}

// =============================================================================
// Test 11: test_empty_cache_eviction
// Eviction on empty cache should return 0
// =============================================================================
static bool test_empty_cache_eviction() {
    printf("TEST: test_empty_cache_eviction\n");

    std::vector<CacheEntry> cache;

    size_t freed = evict_to_budget(cache, 0);

    if (freed != 0) {
        printf("  FAIL: Empty cache should return 0 bytes freed, got %zu\n", freed);
        return false;
    }

    printf("  PASS: Empty cache eviction returns 0\n");
    return true;
}

// =============================================================================
// Test 12: test_frozen_between_warm_and_cold
// FROZEN priority should be between WARM and COLD
// =============================================================================
static bool test_frozen_between_warm_and_cold() {
    printf("TEST: test_frozen_between_warm_and_cold\n");

    reset_session_start();

    auto warm   = make_entry(TensorHeat::WARM, 1024 * 1024, 10);
    auto frozen = make_entry(TensorHeat::FROZEN, 1024 * 1024, 10);
    auto cold   = make_entry(TensorHeat::COLD, 1024 * 1024, 10);

    float warm_priority   = compute_priority(warm);
    float frozen_priority = compute_priority(frozen);
    float cold_priority   = compute_priority(cold);

    if (frozen_priority >= warm_priority) {
        printf("  FAIL: FROZEN priority (%.2f) should be < WARM priority (%.2f)\n", frozen_priority, warm_priority);
        return false;
    }

    if (frozen_priority <= cold_priority) {
        printf("  FAIL: FROZEN priority (%.2f) should be > COLD priority (%.2f)\n", frozen_priority, cold_priority);
        return false;
    }

    printf("  PASS: FROZEN (%.2f) between WARM (%.2f) and COLD (%.2f)\n", frozen_priority, warm_priority, cold_priority);
    return true;
}

// =============================================================================
// Test 13: test_can_satisfy_request
// Verify can_satisfy_request correctly predicts eviction feasibility
// =============================================================================
static bool test_can_satisfy_request() {
    printf("TEST: test_can_satisfy_request\n");

    reset_session_start();

    std::vector<CacheEntry> cache;

    // 2MB HOT (protected) + 3MB COLD (evictable) = 5MB total
    cache.push_back(make_entry(TensorHeat::HOT, 2 * 1024 * 1024, 10));
    cache.push_back(make_entry(TensorHeat::COLD, 3 * 1024 * 1024, 1));

    size_t budget = 5 * 1024 * 1024;  // 5MB budget

    // Request 2MB - possible (evict 3MB COLD, have 2MB HOT + 2MB new = 4MB)
    if (!can_satisfy_request(cache, budget, 2 * 1024 * 1024, MemoryPressure::HIGH)) {
        printf("  FAIL: 2MB request should be satisfiable\n");
        return false;
    }

    // Request 4MB - not possible (even after evicting COLD, HOT + 4MB > 5MB)
    if (can_satisfy_request(cache, budget, 4 * 1024 * 1024, MemoryPressure::HIGH)) {
        printf("  FAIL: 4MB request should not be satisfiable\n");
        return false;
    }

    // IMPOSSIBLE pressure - always fails
    if (can_satisfy_request(cache, budget, 1024, MemoryPressure::IMPOSSIBLE)) {
        printf("  FAIL: IMPOSSIBLE pressure should always return false\n");
        return false;
    }

    printf("  PASS: can_satisfy_request correctly predicts eviction feasibility\n");
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== Heat-Aware Eviction Policy Unit Tests ===\n");
    printf("Part of unified memory management (llama.cpp-6hp/llama.cpp-6hp.12)\n\n");

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test_fn)(), const char * name) {
        bool result = test_fn();
        if (result) {
            passed++;
        } else {
            failed++;
            printf("  >>> TEST FAILED: %s\n\n", name);
        }
    };

    run_test(test_priority_heat_ordering, "test_priority_heat_ordering");
    run_test(test_evict_cold_first, "test_evict_cold_first");
    run_test(test_hot_protected, "test_hot_protected");
    run_test(test_lru_within_heat, "test_lru_within_heat");
    run_test(test_size_penalty, "test_size_penalty");
    run_test(test_pressure_levels, "test_pressure_levels");
    run_test(test_heat_multiplier_values, "test_heat_multiplier_values");
    run_test(test_frequency_affects_priority, "test_frequency_affects_priority");
    run_test(test_evict_returns_bytes_freed, "test_evict_returns_bytes_freed");
    run_test(test_no_eviction_needed, "test_no_eviction_needed");
    run_test(test_empty_cache_eviction, "test_empty_cache_eviction");
    run_test(test_frozen_between_warm_and_cold, "test_frozen_between_warm_and_cold");
    run_test(test_can_satisfy_request, "test_can_satisfy_request");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
