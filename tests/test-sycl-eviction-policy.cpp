// SYCL Eviction Policy unit tests
// Tests for priority-based LRU eviction system
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-2oy)
//
// TDD: These tests written FIRST, before implementation.
// Implementation must make these tests pass.
//
// Priority Levels (P0-P4):
// - P0: Compute buffers - NEVER evict (pinned=true)
// - P1: Active KV cache heads - evict last
// - P2: Hot experts (recently used) - evict after KV
// - P3: Warm experts (predicted) - evict before hot
// - P4: Cold experts - evict first
//
// Eviction targets (in order):
// 1. VRAM -> HOST (fast, keep in pinned RAM)
// 2. HOST -> MMAP (slower, page out)
// 3. MMAP can be discarded (reload from file)

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Include the eviction policy header (to be created)
#include "ggml-sycl/eviction-policy.hpp"

// Helper to create unique test IDs
static uint64_t g_test_id_counter = 0;

static uint64_t next_test_id() {
    return ++g_test_id_counter;
}

// =============================================================================
// Test 1: P0 items (pinned=true) are NEVER selected for eviction
// =============================================================================
static bool test_p0_never_evicted() {
    printf("TEST: test_p0_never_evicted\n");

    ggml_sycl::EvictionPolicy policy;

    // Add a P0 (compute buffer) entry - should never be evicted
    uint64_t p0_id = next_test_id();
    policy.add_entry(p0_id, ggml_sycl::EvictionPriority::P0_COMPUTE, 1024 * 1024, ggml_sycl::cache_location::DEVICE);

    // Add a P4 (cold expert) entry - should be evicted first
    uint64_t p4_id = next_test_id();
    policy.add_entry(p4_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024 * 1024,
                     ggml_sycl::cache_location::DEVICE);

    // Request eviction - should select P4, not P0
    auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);

    if (!victim.has_value()) {
        printf("  FAIL: expected a victim to be selected\n");
        return false;
    }

    if (victim->id == p0_id) {
        printf("  FAIL: P0 (compute buffer) was selected for eviction - should NEVER happen\n");
        return false;
    }

    if (victim->id != p4_id) {
        printf("  FAIL: expected P4 entry to be selected, got id=%lu\n", victim->id);
        return false;
    }

    // Now remove P4 and try again - P0 should still not be evictable
    policy.remove_entry(p4_id);
    auto victim2 = policy.select_victim(ggml_sycl::cache_location::DEVICE);

    if (victim2.has_value()) {
        printf("  FAIL: should have no evictable entries (only P0 remains), but got id=%lu\n", victim2->id);
        return false;
    }

    printf("  PASS: P0 entries are never selected for eviction\n");
    return true;
}

// =============================================================================
// Test 2: Higher priority (lower number) evicts AFTER lower priority (higher number)
// Priority order: P4 first, then P3, then P2, then P1, never P0
// =============================================================================
static bool test_priority_ordering() {
    printf("TEST: test_priority_ordering\n");

    ggml_sycl::EvictionPolicy policy;

    // Add entries with different priorities
    uint64_t p1_id = next_test_id();
    uint64_t p2_id = next_test_id();
    uint64_t p3_id = next_test_id();
    uint64_t p4_id = next_test_id();

    policy.add_entry(p1_id, ggml_sycl::EvictionPriority::P1_ACTIVE_KV, 1024, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(p2_id, ggml_sycl::EvictionPriority::P2_HOT_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(p3_id, ggml_sycl::EvictionPriority::P3_WARM_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(p4_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    // Eviction order should be: P4 -> P3 -> P2 -> P1
    auto v1 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v1.has_value() || v1->id != p4_id) {
        printf("  FAIL: first victim should be P4 (cold expert), got id=%lu\n", v1.has_value() ? v1->id : 0);
        return false;
    }
    policy.remove_entry(p4_id);

    auto v2 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v2.has_value() || v2->id != p3_id) {
        printf("  FAIL: second victim should be P3 (warm expert), got id=%lu\n", v2.has_value() ? v2->id : 0);
        return false;
    }
    policy.remove_entry(p3_id);

    auto v3 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v3.has_value() || v3->id != p2_id) {
        printf("  FAIL: third victim should be P2 (hot expert), got id=%lu\n", v3.has_value() ? v3->id : 0);
        return false;
    }
    policy.remove_entry(p2_id);

    auto v4 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v4.has_value() || v4->id != p1_id) {
        printf("  FAIL: fourth victim should be P1 (active KV), got id=%lu\n", v4.has_value() ? v4->id : 0);
        return false;
    }

    printf("  PASS: eviction order follows priority: P4 -> P3 -> P2 -> P1\n");
    return true;
}

// =============================================================================
// Test 3: Within same priority, LRU (least recently used) is evicted first
// =============================================================================
static bool test_lru_within_priority() {
    printf("TEST: test_lru_within_priority\n");

    ggml_sycl::EvictionPolicy policy;

    // Add three P4 entries with different access times
    uint64_t oldest_id = next_test_id();
    uint64_t middle_id = next_test_id();
    uint64_t newest_id = next_test_id();

    policy.add_entry(oldest_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    // Small delay to ensure different timestamps
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    policy.add_entry(middle_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    policy.add_entry(newest_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    // Touch middle one to make it most recently used
    policy.touch_entry(middle_id);

    // Eviction order should be: oldest -> newest -> middle (due to touch)
    auto v1 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v1.has_value() || v1->id != oldest_id) {
        printf("  FAIL: first victim should be oldest entry (id=%lu), got id=%lu\n", oldest_id,
               v1.has_value() ? v1->id : 0);
        return false;
    }
    policy.remove_entry(oldest_id);

    auto v2 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v2.has_value() || v2->id != newest_id) {
        printf("  FAIL: second victim should be newest entry (id=%lu), got id=%lu\n", newest_id,
               v2.has_value() ? v2->id : 0);
        return false;
    }
    policy.remove_entry(newest_id);

    auto v3 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v3.has_value() || v3->id != middle_id) {
        printf("  FAIL: third victim should be middle entry (id=%lu), got id=%lu\n", middle_id,
               v3.has_value() ? v3->id : 0);
        return false;
    }

    printf("  PASS: LRU ordering works within same priority level\n");
    return true;
}

// =============================================================================
// Test 4: Eviction target order: VRAM -> HOST -> MMAP
// When selecting from VRAM, select VRAM entries first
// =============================================================================
static bool test_eviction_target_order() {
    printf("TEST: test_eviction_target_order\n");

    ggml_sycl::EvictionPolicy policy;

    // Add P4 entries in different locations
    uint64_t vram_id = next_test_id();
    uint64_t host_id = next_test_id();
    uint64_t mmap_id = next_test_id();

    policy.add_entry(mmap_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::HOST_MMAP);
    policy.add_entry(host_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024,
                     ggml_sycl::cache_location::HOST_PINNED);
    policy.add_entry(vram_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    // Requesting eviction from DEVICE should select VRAM entries first
    auto v_device = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v_device.has_value() || v_device->id != vram_id) {
        printf("  FAIL: DEVICE eviction should select VRAM entry (id=%lu), got id=%lu\n", vram_id,
               v_device.has_value() ? v_device->id : 0);
        return false;
    }
    policy.remove_entry(vram_id);

    // Requesting eviction from HOST_PINNED should select HOST entries
    auto v_host = policy.select_victim(ggml_sycl::cache_location::HOST_PINNED);
    if (!v_host.has_value() || v_host->id != host_id) {
        printf("  FAIL: HOST eviction should select HOST entry (id=%lu), got id=%lu\n", host_id,
               v_host.has_value() ? v_host->id : 0);
        return false;
    }
    policy.remove_entry(host_id);

    // Requesting eviction from HOST_MMAP should select MMAP entries
    auto v_mmap = policy.select_victim(ggml_sycl::cache_location::HOST_MMAP);
    if (!v_mmap.has_value() || v_mmap->id != mmap_id) {
        printf("  FAIL: MMAP eviction should select MMAP entry (id=%lu), got id=%lu\n", mmap_id,
               v_mmap.has_value() ? v_mmap->id : 0);
        return false;
    }

    printf("  PASS: eviction targets correct memory tier\n");
    return true;
}

// =============================================================================
// Test 5: Empty cache returns no victim
// =============================================================================
static bool test_empty_cache() {
    printf("TEST: test_empty_cache\n");

    ggml_sycl::EvictionPolicy policy;

    // Empty policy should return no victim
    auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (victim.has_value()) {
        printf("  FAIL: empty policy should return no victim\n");
        return false;
    }

    printf("  PASS: empty cache returns no victim\n");
    return true;
}

// =============================================================================
// Test 6: Single item that is not P0 can be evicted
// =============================================================================
static bool test_single_item_eviction() {
    printf("TEST: test_single_item_eviction\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t single_id = next_test_id();
    policy.add_entry(single_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 2048, ggml_sycl::cache_location::DEVICE);

    auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!victim.has_value()) {
        printf("  FAIL: single non-P0 entry should be evictable\n");
        return false;
    }

    if (victim->id != single_id) {
        printf("  FAIL: expected single entry (id=%lu), got id=%lu\n", single_id, victim->id);
        return false;
    }

    if (victim->size != 2048) {
        printf("  FAIL: expected size 2048, got %zu\n", victim->size);
        return false;
    }

    printf("  PASS: single non-P0 item can be evicted\n");
    return true;
}

// =============================================================================
// Test 7: Single P0 item cannot be evicted (cache appears empty for eviction)
// =============================================================================
static bool test_single_p0_item() {
    printf("TEST: test_single_p0_item\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t p0_id = next_test_id();
    policy.add_entry(p0_id, ggml_sycl::EvictionPriority::P0_COMPUTE, 1024, ggml_sycl::cache_location::DEVICE);

    auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (victim.has_value()) {
        printf("  FAIL: single P0 entry should not be evictable\n");
        return false;
    }

    printf("  PASS: single P0 item cannot be evicted\n");
    return true;
}

// =============================================================================
// Test 8: Priority can be changed dynamically (e.g., expert becomes hot)
// =============================================================================
static bool test_priority_change() {
    printf("TEST: test_priority_change\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t expert_id = next_test_id();
    uint64_t other_id  = next_test_id();

    // Both start as P4 (cold expert)
    policy.add_entry(expert_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    policy.add_entry(other_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    // expert_id was added first, so it's older - should be evicted first
    auto v1 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v1.has_value() || v1->id != expert_id) {
        printf("  FAIL: older entry should be selected first\n");
        return false;
    }

    // Now promote expert_id to P2 (hot expert) - should no longer be first victim
    policy.update_priority(expert_id, ggml_sycl::EvictionPriority::P2_HOT_EXPERT);

    auto v2 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v2.has_value() || v2->id != other_id) {
        printf("  FAIL: after promotion, other_id (P4) should be victim, got id=%lu\n", v2.has_value() ? v2->id : 0);
        return false;
    }

    printf("  PASS: dynamic priority changes work correctly\n");
    return true;
}

// =============================================================================
// Test 9: Location can be updated (entry moved from VRAM to HOST)
// =============================================================================
static bool test_location_update() {
    printf("TEST: test_location_update\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t entry_id = next_test_id();
    policy.add_entry(entry_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);

    // Entry is in DEVICE, should be found when selecting from DEVICE
    auto v1 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (!v1.has_value() || v1->id != entry_id) {
        printf("  FAIL: entry in DEVICE should be found\n");
        return false;
    }

    // Update location to HOST_PINNED (simulating VRAM -> HOST eviction)
    policy.update_location(entry_id, ggml_sycl::cache_location::HOST_PINNED);

    // Now should NOT be found when selecting from DEVICE
    auto v2 = policy.select_victim(ggml_sycl::cache_location::DEVICE);
    if (v2.has_value()) {
        printf("  FAIL: entry moved to HOST should not be found in DEVICE selection\n");
        return false;
    }

    // Should be found when selecting from HOST_PINNED
    auto v3 = policy.select_victim(ggml_sycl::cache_location::HOST_PINNED);
    if (!v3.has_value() || v3->id != entry_id) {
        printf("  FAIL: entry in HOST should be found when selecting HOST\n");
        return false;
    }

    printf("  PASS: location updates work correctly\n");
    return true;
}

// =============================================================================
// Test 10: Eviction should return bytes that will be freed
// =============================================================================
static bool test_eviction_bytes_freed() {
    printf("TEST: test_eviction_bytes_freed\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t id1 = next_test_id();
    uint64_t id2 = next_test_id();

    const size_t size1 = 1024 * 1024;      // 1 MB
    const size_t size2 = 2 * 1024 * 1024;  // 2 MB

    policy.add_entry(id1, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, size1, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(id2, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, size2, ggml_sycl::cache_location::DEVICE);

    // Select victims until we have enough bytes
    size_t needed  = 3 * 1024 * 1024;  // Need 3 MB
    auto   victims = policy.select_victims_for_size(ggml_sycl::cache_location::DEVICE, needed);

    // Should select both entries (1 MB + 2 MB = 3 MB)
    if (victims.size() != 2) {
        printf("  FAIL: expected 2 victims to free 3 MB, got %zu\n", victims.size());
        return false;
    }

    size_t total_freed = 0;
    for (const auto & v : victims) {
        total_freed += v.size;
    }

    if (total_freed < needed) {
        printf("  FAIL: total freed %zu < needed %zu\n", total_freed, needed);
        return false;
    }

    printf("  PASS: eviction returns correct bytes to be freed (%zu bytes)\n", total_freed);
    return true;
}

// =============================================================================
// Test 11: Multiple entries at same priority evicted in LRU order
// =============================================================================
static bool test_multiple_same_priority_lru() {
    printf("TEST: test_multiple_same_priority_lru\n");

    ggml_sycl::EvictionPolicy policy;

    // Add 5 P3 (warm expert) entries
    std::vector<uint64_t> ids;
    for (int i = 0; i < 5; i++) {
        uint64_t id = next_test_id();
        ids.push_back(id);
        policy.add_entry(id, ggml_sycl::EvictionPriority::P3_WARM_EXPERT, 1024, ggml_sycl::cache_location::DEVICE);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Touch entries in reverse order to make oldest entries most recently used
    for (int i = 4; i >= 0; i--) {
        policy.touch_entry(ids[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Now eviction should happen in reverse order: ids[4], ids[3], ids[2], ids[1], ids[0]
    // Because ids[0] was touched last (most recent), ids[4] was touched first (least recent)
    for (int i = 4; i >= 0; i--) {
        auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);
        if (!victim.has_value() || victim->id != ids[i]) {
            printf("  FAIL: expected victim ids[%d]=%lu, got id=%lu\n", i, ids[i], victim.has_value() ? victim->id : 0);
            return false;
        }
        policy.remove_entry(ids[i]);
    }

    printf("  PASS: multiple entries at same priority follow LRU order\n");
    return true;
}

// =============================================================================
// Test 12: get_total_size returns correct sum
// =============================================================================
static bool test_total_size_tracking() {
    printf("TEST: test_total_size_tracking\n");

    ggml_sycl::EvictionPolicy policy;

    if (policy.get_total_size() != 0) {
        printf("  FAIL: empty policy should have total_size 0\n");
        return false;
    }

    uint64_t id1 = next_test_id();
    policy.add_entry(id1, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1000, ggml_sycl::cache_location::DEVICE);

    if (policy.get_total_size() != 1000) {
        printf("  FAIL: expected total_size 1000, got %zu\n", policy.get_total_size());
        return false;
    }

    uint64_t id2 = next_test_id();
    policy.add_entry(id2, ggml_sycl::EvictionPriority::P3_WARM_EXPERT, 2000, ggml_sycl::cache_location::HOST_PINNED);

    if (policy.get_total_size() != 3000) {
        printf("  FAIL: expected total_size 3000, got %zu\n", policy.get_total_size());
        return false;
    }

    policy.remove_entry(id1);

    if (policy.get_total_size() != 2000) {
        printf("  FAIL: after removal, expected total_size 2000, got %zu\n", policy.get_total_size());
        return false;
    }

    printf("  PASS: total size tracking is accurate\n");
    return true;
}

// =============================================================================
// Test 13: get_size_by_location returns correct sums per location
// =============================================================================
static bool test_size_by_location() {
    printf("TEST: test_size_by_location\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t device_id = next_test_id();
    uint64_t host_id   = next_test_id();
    uint64_t mmap_id   = next_test_id();

    policy.add_entry(device_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1000, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(host_id, ggml_sycl::EvictionPriority::P3_WARM_EXPERT, 2000,
                     ggml_sycl::cache_location::HOST_PINNED);
    policy.add_entry(mmap_id, ggml_sycl::EvictionPriority::P2_HOT_EXPERT, 3000, ggml_sycl::cache_location::HOST_MMAP);

    if (policy.get_size_by_location(ggml_sycl::cache_location::DEVICE) != 1000) {
        printf("  FAIL: DEVICE size should be 1000\n");
        return false;
    }

    if (policy.get_size_by_location(ggml_sycl::cache_location::HOST_PINNED) != 2000) {
        printf("  FAIL: HOST_PINNED size should be 2000\n");
        return false;
    }

    if (policy.get_size_by_location(ggml_sycl::cache_location::HOST_MMAP) != 3000) {
        printf("  FAIL: HOST_MMAP size should be 3000\n");
        return false;
    }

    // Update location and verify counts change
    policy.update_location(device_id, ggml_sycl::cache_location::HOST_PINNED);

    if (policy.get_size_by_location(ggml_sycl::cache_location::DEVICE) != 0) {
        printf("  FAIL: DEVICE size should be 0 after move\n");
        return false;
    }

    if (policy.get_size_by_location(ggml_sycl::cache_location::HOST_PINNED) != 3000) {
        printf("  FAIL: HOST_PINNED size should be 3000 after move\n");
        return false;
    }

    printf("  PASS: size by location tracking is accurate\n");
    return true;
}

// =============================================================================
// Test 14: Entry count tracking
// =============================================================================
static bool test_entry_count() {
    printf("TEST: test_entry_count\n");

    ggml_sycl::EvictionPolicy policy;

    if (policy.get_entry_count() != 0) {
        printf("  FAIL: empty policy should have 0 entries\n");
        return false;
    }

    uint64_t id1 = next_test_id();
    uint64_t id2 = next_test_id();
    uint64_t id3 = next_test_id();

    policy.add_entry(id1, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1000, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(id2, ggml_sycl::EvictionPriority::P3_WARM_EXPERT, 1000, ggml_sycl::cache_location::DEVICE);
    policy.add_entry(id3, ggml_sycl::EvictionPriority::P0_COMPUTE, 1000, ggml_sycl::cache_location::DEVICE);

    if (policy.get_entry_count() != 3) {
        printf("  FAIL: expected 3 entries, got %zu\n", policy.get_entry_count());
        return false;
    }

    policy.remove_entry(id2);

    if (policy.get_entry_count() != 2) {
        printf("  FAIL: expected 2 entries after removal, got %zu\n", policy.get_entry_count());
        return false;
    }

    printf("  PASS: entry count tracking is accurate\n");
    return true;
}

// =============================================================================
// Test 15: Duplicate entry ID handling (update existing)
// =============================================================================
static bool test_duplicate_entry_update() {
    printf("TEST: test_duplicate_entry_update\n");

    ggml_sycl::EvictionPolicy policy;

    uint64_t entry_id = next_test_id();

    // Add initial entry
    policy.add_entry(entry_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 1000, ggml_sycl::cache_location::DEVICE);

    if (policy.get_entry_count() != 1) {
        printf("  FAIL: expected 1 entry after first add\n");
        return false;
    }

    // Add same ID again with different properties - should update, not duplicate
    policy.add_entry(entry_id, ggml_sycl::EvictionPriority::P2_HOT_EXPERT, 2000,
                     ggml_sycl::cache_location::HOST_PINNED);

    if (policy.get_entry_count() != 1) {
        printf("  FAIL: expected 1 entry after duplicate add (update)\n");
        return false;
    }

    // Verify updated properties
    if (policy.get_total_size() != 2000) {
        printf("  FAIL: size should be updated to 2000\n");
        return false;
    }

    if (policy.get_size_by_location(ggml_sycl::cache_location::HOST_PINNED) != 2000) {
        printf("  FAIL: entry should be in HOST_PINNED now\n");
        return false;
    }

    printf("  PASS: duplicate entry ID updates existing entry\n");
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== Eviction Policy Unit Tests ===\n");
    printf("Part of unified memory management (llama.cpp-v3n/llama.cpp-2oy)\n\n");

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

    run_test(test_p0_never_evicted, "test_p0_never_evicted");
    run_test(test_priority_ordering, "test_priority_ordering");
    run_test(test_lru_within_priority, "test_lru_within_priority");
    run_test(test_eviction_target_order, "test_eviction_target_order");
    run_test(test_empty_cache, "test_empty_cache");
    run_test(test_single_item_eviction, "test_single_item_eviction");
    run_test(test_single_p0_item, "test_single_p0_item");
    run_test(test_priority_change, "test_priority_change");
    run_test(test_location_update, "test_location_update");
    run_test(test_eviction_bytes_freed, "test_eviction_bytes_freed");
    run_test(test_multiple_same_priority_lru, "test_multiple_same_priority_lru");
    run_test(test_total_size_tracking, "test_total_size_tracking");
    run_test(test_size_by_location, "test_size_by_location");
    run_test(test_entry_count, "test_entry_count");
    run_test(test_duplicate_entry_update, "test_duplicate_entry_update");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
