// SYCL MoE Expert Parallelism Unit Tests
//
// Pure C++ unit tests for MoE expert dispatch data structures:
// 1. ExpertPlacementTable: init, set/get roundtrip, thread safety
// 2. Key generation: make_key uniqueness for different (layer_id, expert_id) pairs
// 3. N-device routing: mock placement table with 3+ devices, verify partitioning
//
// No GPU dependency — tests only the CPU-side data structures.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>
#include <unordered_set>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl/unified-cache.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

// Test counters
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "  FAIL: %s\n", msg);                              \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        g_tests_run++;                                                         \
        if (fn()) {                                                            \
            g_tests_passed++;                                                  \
            printf("  PASS: %s\n", #fn);                                       \
        } else {                                                               \
            g_tests_failed++;                                                  \
            fprintf(stderr, "  FAIL: %s\n", #fn);                              \
        }                                                                      \
    } while (0)

// ========================================================================
// Test 1: ExpertPlacementTable basic init
// ========================================================================
static bool test_placement_table_init() {
    ggml_sycl::ExpertPlacementTable table;
    TEST_ASSERT(!table.is_initialized(), "table should not be initialized before init()");
    TEST_ASSERT(table.n_layers() == 0, "n_layers should be 0 before init");
    TEST_ASSERT(table.n_experts() == 0, "n_experts should be 0 before init");

    table.init(32, 8);
    TEST_ASSERT(table.is_initialized(), "table should be initialized after init()");
    TEST_ASSERT(table.n_layers() == 32, "n_layers should be 32");
    TEST_ASSERT(table.n_experts() == 8, "n_experts should be 8");

    return true;
}

// ========================================================================
// Test 2: Set/get roundtrip
// ========================================================================
static bool test_placement_set_get_roundtrip() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(4, 16);

    // Create a placement with known values
    ggml_sycl::ExpertPlacement p{};
    p.device_id       = 0;
    p.device_ptr      = reinterpret_cast<void *>(0xDEAD0000);
    p.host_ptr        = reinterpret_cast<void *>(0xBEEF0000);
    p.weight_bytes    = 1024;
    p.popularity_rank = 3;

    const int layer_id  = 42;   // FNV hash-based IDs can be arbitrary
    const int expert_id = 7;
    table.set(layer_id, expert_id, p);

    auto got = table.get(layer_id, expert_id);
    TEST_ASSERT(got.device_id == 0, "device_id roundtrip");
    TEST_ASSERT(got.device_ptr == reinterpret_cast<void *>(0xDEAD0000), "device_ptr roundtrip");
    TEST_ASSERT(got.host_ptr == reinterpret_cast<void *>(0xBEEF0000), "host_ptr roundtrip");
    TEST_ASSERT(got.weight_bytes == 1024, "weight_bytes roundtrip");
    TEST_ASSERT(got.popularity_rank == 3, "popularity_rank roundtrip");
    TEST_ASSERT(got.is_valid(), "placement should be valid (host_ptr != nullptr)");

    // Getting a non-existent entry returns default (invalid) placement
    auto missing = table.get(999, 999);
    TEST_ASSERT(missing.device_id == -1, "missing entry device_id should be -1");
    TEST_ASSERT(missing.device_ptr == nullptr, "missing entry device_ptr should be nullptr");
    TEST_ASSERT(!missing.is_valid(), "missing entry should not be valid");

    return true;
}

// ========================================================================
// Test 3: set_device_ptr and set_popularity updates
// ========================================================================
static bool test_placement_update_methods() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(4, 8);

    const int layer_id  = 100;
    const int expert_id = 3;

    // Initial placement: CPU-only
    ggml_sycl::ExpertPlacement p{};
    p.device_id       = -1;
    p.host_ptr        = reinterpret_cast<void *>(0x1000);
    p.weight_bytes    = 512;
    p.popularity_rank = -1;
    table.set(layer_id, expert_id, p);

    // Update device pointer (simulating VRAM upload)
    void * fake_dev_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(0xABCD0000));
    table.set_device_ptr(layer_id, expert_id, 0, fake_dev_ptr);

    auto got = table.get(layer_id, expert_id);
    TEST_ASSERT(got.device_id == 0, "device_id should be updated to 0");
    TEST_ASSERT(got.device_ptr == fake_dev_ptr, "device_ptr should be updated");

    // Update popularity
    table.set_popularity(layer_id, expert_id, 5);
    got = table.get(layer_id, expert_id);
    TEST_ASSERT(got.popularity_rank == 5, "popularity_rank should be updated to 5");

    return true;
}

// ========================================================================
// Test 4: Key uniqueness — different (layer_id, expert_id) pairs must
//         produce different keys. Tests the make_key() logic indirectly
//         by verifying that set/get with different IDs are independent.
// ========================================================================
static bool test_key_uniqueness() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(100, 64);

    // Insert entries with varying layer_id and expert_id
    const int n_layers  = 32;
    const int n_experts = 8;
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            ggml_sycl::ExpertPlacement p{};
            p.device_id    = l % 3;  // Distribute across 3 devices
            p.host_ptr     = reinterpret_cast<void *>(static_cast<uintptr_t>(l * 1000 + e));
            p.weight_bytes = static_cast<size_t>(l * 100 + e);
            table.set(l, e, p);
        }
    }

    // Verify each entry is independent
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            auto got = table.get(l, e);
            TEST_ASSERT(got.device_id == l % 3, "device_id should match for each entry");
            TEST_ASSERT(got.weight_bytes == static_cast<size_t>(l * 100 + e),
                        "weight_bytes should be unique per entry");
        }
    }

    // Verify that swapped (layer_id, expert_id) are different keys
    // e.g., (1, 2) != (2, 1)
    {
        ggml_sycl::ExpertPlacement p1{};
        p1.device_id = 0;
        p1.host_ptr  = reinterpret_cast<void *>(0xA);
        table.set(1001, 2002, p1);

        ggml_sycl::ExpertPlacement p2{};
        p2.device_id = 1;
        p2.host_ptr  = reinterpret_cast<void *>(0xB);
        table.set(2002, 1001, p2);

        auto got1 = table.get(1001, 2002);
        auto got2 = table.get(2002, 1001);
        TEST_ASSERT(got1.device_id == 0, "swapped key 1 device_id");
        TEST_ASSERT(got2.device_id == 1, "swapped key 2 device_id");
        TEST_ASSERT(got1.host_ptr != got2.host_ptr, "swapped keys should have different host_ptr");
    }

    return true;
}

// ========================================================================
// Test 5: FNV hash-based layer IDs (large values typical of moe_cache_layer_id)
// ========================================================================
static bool test_hash_based_layer_ids() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(100, 64);

    // Simulate FNV-1a 32-bit hash values (these can be large negative when
    // stored as int, or large positive as uint32_t cast to int).
    // All values must be UNIQUE to avoid overwriting.
    const int layer_ids[] = {
        static_cast<int>(2166136261u),  // FNV offset basis
        static_cast<int>(0xCAFEBABEu),  // Arbitrary hash
        static_cast<int>(0xDEADBEEFu),  // Arbitrary hash
        static_cast<int>(0x00000001u),  // Small hash
        static_cast<int>(0xFFFFFFFFu),  // Max hash (becomes -1 as signed int)
        0,                              // Zero hash
    };

    for (int i = 0; i < 6; i++) {
        ggml_sycl::ExpertPlacement p{};
        p.device_id = i % 3;
        p.host_ptr  = reinterpret_cast<void *>(static_cast<uintptr_t>(i + 1));
        table.set(layer_ids[i], 0, p);
    }

    for (int i = 0; i < 6; i++) {
        auto got = table.get(layer_ids[i], 0);
        TEST_ASSERT(got.device_id == i % 3, "hash-based layer_id roundtrip");
    }

    return true;
}

// ========================================================================
// Test 6: Thread safety — concurrent readers and writers
// ========================================================================
static bool test_thread_safety() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(64, 32);

    // Pre-populate some entries
    for (int l = 0; l < 64; l++) {
        for (int e = 0; e < 32; e++) {
            ggml_sycl::ExpertPlacement p{};
            p.device_id    = 0;
            p.host_ptr     = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            p.weight_bytes = 100;
            table.set(l, e, p);
        }
    }

    std::atomic<int> errors{0};
    std::atomic<bool> go{false};

    // Writer threads: update device_ptr and popularity
    auto writer_fn = [&](int thread_id) {
        while (!go.load(std::memory_order_acquire)) {}
        for (int iter = 0; iter < 1000; iter++) {
            int l = (thread_id * 7 + iter) % 64;
            int e = (thread_id * 13 + iter) % 32;
            void * ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(iter + 1));
            table.set_device_ptr(l, e, thread_id % 3, ptr);
            table.set_popularity(l, e, iter % 100);
        }
    };

    // Reader threads: get placement and verify validity
    auto reader_fn = [&](int /* thread_id */) {
        while (!go.load(std::memory_order_acquire)) {}
        for (int iter = 0; iter < 2000; iter++) {
            int l = iter % 64;
            int e = iter % 32;
            auto p = table.get(l, e);
            // host_ptr was set to non-null during init, should never be null
            if (p.host_ptr == nullptr) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
            // device_id should be in valid range
            if (p.device_id < -1 || p.device_id > 3) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    const int n_writers = 4;
    const int n_readers = 8;
    std::vector<std::thread> threads;
    threads.reserve(n_writers + n_readers);

    for (int i = 0; i < n_writers; i++) {
        threads.emplace_back(writer_fn, i);
    }
    for (int i = 0; i < n_readers; i++) {
        threads.emplace_back(reader_fn, i);
    }

    go.store(true, std::memory_order_release);

    for (auto & t : threads) {
        t.join();
    }

    TEST_ASSERT(errors.load() == 0, "no data races detected in concurrent access");
    return true;
}

// ========================================================================
// Test 7: N-device routing — mock placement with 3+ devices, verify
//         that partition logic correctly separates entries by device_id
// ========================================================================
static bool test_n_device_routing() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(4, 64);

    const int n_devices = 4;  // primary + 3 secondary GPUs
    const int n_layers  = 2;
    const int n_experts = 16;

    // Distribute experts: round-robin across devices, some on CPU
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            ggml_sycl::ExpertPlacement p{};
            if (e < 12) {
                // First 12 experts: distribute across 4 GPUs
                p.device_id  = e % n_devices;
                p.device_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(e + 1));
            } else {
                // Last 4 experts: CPU-only
                p.device_id  = -1;
                p.device_ptr = nullptr;
            }
            p.host_ptr     = reinterpret_cast<void *>(static_cast<uintptr_t>(e + 100));
            p.weight_bytes = 1024;
            table.set(l, e, p);
        }
    }

    // Simulate the partition logic from ggml-sycl.cpp MUL_MAT_ID dispatch
    struct dispatch_entry {
        int expert_id;
        int device_id;
    };

    std::vector<dispatch_entry> gpu0_entries;
    std::vector<std::vector<dispatch_entry>> per_gpu(n_devices);
    std::vector<dispatch_entry> cpu_entries;

    const int layer_id = 0;
    for (int e = 0; e < n_experts; e++) {
        auto placement = table.get(layer_id, e);
        if (placement.device_id >= 0 && placement.device_id < n_devices
            && placement.device_ptr != nullptr) {
            if (placement.device_id == 0) {
                gpu0_entries.push_back({ e, 0 });
            } else {
                per_gpu[placement.device_id].push_back({ e, placement.device_id });
            }
        } else {
            cpu_entries.push_back({ e, -1 });
        }
    }

    // Verify partition counts
    // Experts 0,4,8 → device 0 (3 entries)
    TEST_ASSERT(gpu0_entries.size() == 3, "GPU0 should have 3 entries (experts 0,4,8)");
    // Experts 1,5,9 → device 1 (3 entries)
    TEST_ASSERT(per_gpu[1].size() == 3, "GPU1 should have 3 entries (experts 1,5,9)");
    // Experts 2,6,10 → device 2 (3 entries)
    TEST_ASSERT(per_gpu[2].size() == 3, "GPU2 should have 3 entries (experts 2,6,10)");
    // Experts 3,7,11 → device 3 (3 entries)
    TEST_ASSERT(per_gpu[3].size() == 3, "GPU3 should have 3 entries (experts 3,7,11)");
    // Experts 12-15 → CPU (4 entries)
    TEST_ASSERT(cpu_entries.size() == 4, "CPU should have 4 entries (experts 12-15)");

    // Verify specific expert assignments
    TEST_ASSERT(gpu0_entries[0].expert_id == 0, "GPU0 first expert should be 0");
    TEST_ASSERT(gpu0_entries[1].expert_id == 4, "GPU0 second expert should be 4");
    TEST_ASSERT(per_gpu[1][0].expert_id == 1, "GPU1 first expert should be 1");
    TEST_ASSERT(per_gpu[2][0].expert_id == 2, "GPU2 first expert should be 2");
    TEST_ASSERT(per_gpu[3][0].expert_id == 3, "GPU3 first expert should be 3");

    return true;
}

// ========================================================================
// Test 8: get_layer_experts returns sorted by popularity_rank
// ========================================================================
static bool test_get_layer_experts_sorted() {
    ggml_sycl::ExpertPlacementTable table;
    table.init(4, 8);

    const int layer_id = 50;
    // Insert experts with varying popularity (out of order)
    int popularities[] = { 5, 2, 7, 1, 3, 6, 0, 4 };
    for (int e = 0; e < 8; e++) {
        ggml_sycl::ExpertPlacement p{};
        p.device_id       = 0;
        p.host_ptr        = reinterpret_cast<void *>(static_cast<uintptr_t>(e + 1));
        p.weight_bytes    = 100;
        p.popularity_rank = popularities[e];
        table.set(layer_id, e, p);
    }

    auto experts = table.get_layer_experts(layer_id);
    TEST_ASSERT(experts.size() == 8, "should return all 8 experts");

    // Verify sorted by popularity_rank ascending
    for (size_t i = 1; i < experts.size(); i++) {
        TEST_ASSERT(experts[i].second.popularity_rank >= experts[i - 1].second.popularity_rank,
                    "experts should be sorted by popularity_rank ascending");
    }

    // Most popular (rank 0) should be expert 6
    TEST_ASSERT(experts[0].first == 6, "most popular expert should be expert 6 (rank 0)");

    return true;
}

int main() {
    printf("=== MoE Expert Parallelism Unit Tests ===\n\n");

    RUN_TEST(test_placement_table_init);
    RUN_TEST(test_placement_set_get_roundtrip);
    RUN_TEST(test_placement_update_methods);
    RUN_TEST(test_key_uniqueness);
    RUN_TEST(test_hash_based_layer_ids);
    RUN_TEST(test_thread_safety);
    RUN_TEST(test_n_device_routing);
    RUN_TEST(test_get_layer_experts_sorted);

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}

#endif  // GGML_USE_SYCL
