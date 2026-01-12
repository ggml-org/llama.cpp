// Unit tests for unified cache bug fixes (evict accounting, realloc failure handling, unaligned hash)
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-bugs

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static bool test_evict_returns_bytes(sycl::queue & q) {
    printf("\n=== Test: evict() returns bytes freed ===\n");

    ggml_sycl::unified_cache cache(q, 4 * 1024);
    std::vector<uint8_t>     data_a(512, 0x11);
    std::vector<uint8_t>     data_b(512, 0x22);

    bool   needs_fill = false;
    void * ptr_a      = cache.ensure_cached_alloc(data_a.data(), data_a.data(), data_a.size(), data_a.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    void * ptr_b      = cache.ensure_cached_alloc(data_b.data(), data_b.data(), data_b.size(), data_b.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (!ptr_a || !ptr_b) {
        fprintf(stderr, "Failed to allocate cache entries for evict test\n");
        return false;
    }

    size_t freed = cache.evict(data_a.size());
    if (freed == 0) {
        fprintf(stderr, "evict() returned 0 bytes freed\n");
        return false;
    }

    printf("evict() freed %zu bytes\n", freed);
    return true;
}

static bool test_realloc_failure_keeps_entry(sycl::queue & q) {
    printf("\n=== Test: realloc failure preserves existing entry ===\n");

    const size_t             budget = 1ULL << 41;  // 2 TB budget to avoid budget gating
    ggml_sycl::unified_cache cache(q, budget);

    std::vector<uint8_t> data(256, 0x33);
    const void *         key_ptr   = data.data();
    const size_t         orig_size = data.size();

    bool   needs_fill = false;
    void * ptr =
        cache.ensure_cached_alloc(key_ptr, data.data(), orig_size, orig_size, ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                  -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate initial cache entry\n");
        return false;
    }

    const size_t used_before = cache.used();
    const size_t huge_alloc  = 1ULL << 40;  // 1 TB, should fail on all current devices

    void * realloc_ptr = cache.ensure_cached_alloc(key_ptr, data.data(), orig_size, huge_alloc,
                                                   ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS,
                                                   false, &needs_fill);
    if (realloc_ptr) {
        fprintf(stderr, "Unexpectedly succeeded in huge realloc\n");
        return false;
    }

    if (!cache.is_cached(key_ptr, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Existing entry was dropped after realloc failure\n");
        return false;
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Cache used() changed after realloc failure (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_realloc_eviction_failure_keeps_entry(sycl::queue & q) {
    printf("\n=== Test: realloc eviction failure preserves existing entry ===\n");

    ggml_sycl::unified_cache cache(q, 1024);
    std::vector<uint8_t>     data(512, 0x44);
    const void *             key_ptr = data.data();

    bool   needs_fill = false;
    void * ptr        = cache.ensure_cached_alloc(key_ptr, data.data(), data.size(), data.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate initial cache entry for eviction test\n");
        return false;
    }

    cache.pin(key_ptr, GGML_LAYOUT_AOS);
    const size_t used_before = cache.used();

    void * realloc_ptr =
        cache.ensure_cached_alloc(key_ptr, data.data(), data.size(), 2048, ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                  -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);

    if (realloc_ptr) {
        fprintf(stderr, "Unexpectedly succeeded in realloc with eviction failure\n");
        return false;
    }

    if (!cache.is_cached(key_ptr, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Entry dropped after eviction failure during realloc\n");
        return false;
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Cache used() changed after eviction failure (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_all_pinned_eviction_failure_new_entry(sycl::queue & q) {
    printf("\n=== Test: all-pinned eviction failure on new entry ===\n");

    ggml_sycl::unified_cache cache(q, 1024);
    std::vector<uint8_t>     data_a(512, 0x55);
    std::vector<uint8_t>     data_b(512, 0x66);
    std::vector<uint8_t>     data_c(512, 0x77);

    bool   needs_fill = false;
    void * ptr_a      = cache.ensure_cached_alloc(data_a.data(), data_a.data(), data_a.size(), data_a.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    void * ptr_b      = cache.ensure_cached_alloc(data_b.data(), data_b.data(), data_b.size(), data_b.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (!ptr_a || !ptr_b) {
        fprintf(stderr, "Failed to allocate pinned entries for eviction test\n");
        return false;
    }

    cache.pin(data_a.data(), GGML_LAYOUT_AOS);
    cache.pin(data_b.data(), GGML_LAYOUT_AOS);
    const size_t used_before = cache.used();

    void * ptr_c = cache.ensure_cached_alloc(data_c.data(), data_c.data(), data_c.size(), data_c.size(),
                                             ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                             &needs_fill);

    if (ptr_c) {
        fprintf(stderr, "Unexpectedly succeeded allocating with all entries pinned\n");
        return false;
    }

    if (!cache.is_cached(data_a.data(), GGML_LAYOUT_AOS) || !cache.is_cached(data_b.data(), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Pinned entries were evicted unexpectedly\n");
        return false;
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Cache used() changed after all-pinned eviction failure (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_partial_eviction_insufficient(sycl::queue & q) {
    printf("\n=== Test: partial eviction insufficient for new entry ===\n");

    ggml_sycl::unified_cache cache(q, 1024);
    std::vector<uint8_t>     data_a(512, 0x88);
    std::vector<uint8_t>     data_b(512, 0x99);
    std::vector<uint8_t>     data_c(1024, 0xaa);

    bool   needs_fill = false;
    void * ptr_a      = cache.ensure_cached_alloc(data_a.data(), data_a.data(), data_a.size(), data_a.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    void * ptr_b      = cache.ensure_cached_alloc(data_b.data(), data_b.data(), data_b.size(), data_b.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (!ptr_a || !ptr_b) {
        fprintf(stderr, "Failed to allocate initial entries for partial eviction test\n");
        return false;
    }

    cache.pin(data_a.data(), GGML_LAYOUT_AOS);
    const size_t used_before = cache.used();

    void * ptr_c = cache.ensure_cached_alloc(data_c.data(), data_c.data(), data_c.size(), data_c.size(),
                                             ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                             &needs_fill);

    if (ptr_c) {
        fprintf(stderr, "Unexpectedly succeeded allocating with insufficient eviction\n");
        return false;
    }

    if (!cache.is_cached(data_a.data(), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Pinned entry was evicted during partial eviction test\n");
        return false;
    }

    // Eviction is deferred; drain the queue and process deferred frees before checking accounting.
    q.wait();
    cache.evict(0);

    if (cache.used() >= used_before) {
        fprintf(stderr, "Cache used() did not drop after partial eviction (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_allocation_failure_new_entry(sycl::queue & q) {
    printf("\n=== Test: allocation failure on new entry ===\n");

    const size_t             budget = std::numeric_limits<size_t>::max() / 2;
    ggml_sycl::unified_cache cache(q, budget);

    std::vector<uint8_t> data(256, 0xbb);
    const size_t         huge_alloc = 1ULL << 40;

    bool   needs_fill = false;
    void * ptr        = cache.ensure_cached_alloc(data.data(), data.data(), data.size(), huge_alloc,
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (ptr) {
        fprintf(stderr, "Unexpectedly succeeded allocating huge entry\n");
        return false;
    }

    if (cache.is_cached(data.data(), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Cache entry created despite allocation failure\n");
        return false;
    }

    if (cache.used() != 0) {
        fprintf(stderr, "Cache used() changed after allocation failure (used=%zu)\n", cache.used());
        return false;
    }

    return true;
}

static bool test_deferred_free_stress(sycl::queue & q) {
    printf("\n=== Test: deferred free stress ===\n");

    ggml_sycl::unified_cache          cache(q, 64 * 1024);
    std::vector<std::vector<uint8_t>> payloads(32, std::vector<uint8_t>(512, 0xcc));

    bool needs_fill = false;
    for (auto & payload : payloads) {
        void * ptr = cache.ensure_cached_alloc(payload.data(), payload.data(), payload.size(), payload.size(),
                                               ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS,
                                               false, &needs_fill);
        if (!ptr) {
            fprintf(stderr, "Failed to allocate entry during deferred free stress\n");
            return false;
        }
    }

    for (auto & payload : payloads) {
        cache.remove(payload.data(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
    }

    q.wait();
    cache.evict(0);

    if (cache.used() != 0 || cache.dense_count() != 0) {
        fprintf(stderr, "Cache not fully freed after deferred free stress (used=%zu count=%zu)\n", cache.used(),
                cache.dense_count());
        return false;
    }

    return true;
}

static bool test_unaligned_hash(sycl::queue & q) {
    printf("\n=== Test: unaligned hash input ===\n");

    ggml_sycl::unified_cache cache(q, 2 * 1024 * 1024);

    std::vector<uint8_t> raw(129, 0x5a);
    uint8_t *            misaligned = raw.data() + 1;
    const size_t         size       = 127;

    void * ptr = cache.ensure_cached(misaligned, misaligned, size, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1,
                                     GGML_LAYOUT_AOS, true);

    if (!ptr) {
        fprintf(stderr, "ensure_cached failed for misaligned input\n");
        return false;
    }

    if (!cache.is_cached(misaligned, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Cache entry missing for misaligned input\n");
        return false;
    }

    return true;
}

static bool test_unpin_experts(sycl::queue & q) {
    printf("\n=== Test: unpin_experts only affects MoE entries ===\n");

    ggml_sycl::unified_cache cache(q, 2048);
    std::vector<uint8_t>     dense(128, 0x5b);
    std::vector<uint8_t>     expert(128, 0x6c);

    bool needs_fill = false;
    if (!cache.ensure_cached_alloc(dense.data(), dense.data(), dense.size(), dense.size(),
                                   ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                   &needs_fill)) {
        fprintf(stderr, "Failed to allocate dense entry for unpin test\n");
        return false;
    }
    if (!cache.ensure_cached_alloc(expert.data(), expert.data(), expert.size(), expert.size(),
                                   ggml_sycl::cache_entry_type::MOE_EXPERT, 0, 0, GGML_LAYOUT_AOS, false,
                                   &needs_fill)) {
        fprintf(stderr, "Failed to allocate expert entry for unpin test\n");
        return false;
    }

    cache.pin(dense.data(), GGML_LAYOUT_AOS);
    cache.pin(expert.data(), GGML_LAYOUT_AOS);
    cache.unpin_experts();

    if (!cache.is_pinned(dense.data(), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Dense entry was unpinned unexpectedly\n");
        return false;
    }
    if (cache.is_pinned(expert.data(), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Expert entry remained pinned after unpin_experts\n");
        return false;
    }

    return true;
}

static bool test_moe_overcommit_cap(sycl::queue & q) {
    printf("\n=== Test: MoE overcommit cap ===\n");

    const size_t             budget = 1024;
    ggml_sycl::unified_cache cache(q, budget);

    std::vector<uint8_t>            data_a(budget, 0x1a);
    ggml_sycl::cache_layout_request req{};
    req.key_ptr          = data_a.data();
    req.src_ptr          = data_a.data();
    req.src_size         = data_a.size();
    req.dst_size         = data_a.size();
    req.type             = ggml_sycl::cache_entry_type::MOE_EXPERT;
    req.layer_id         = 0;
    req.expert_id        = 0;
    req.layout           = GGML_LAYOUT_AOS;
    req.allow_overcommit = true;

    auto result = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::FAILED || !result.device_ptr) {
        fprintf(stderr, "Failed to cache initial MoE entry\n");
        return false;
    }
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    cache.pin(data_a.data(), GGML_LAYOUT_AOS);

    std::vector<uint8_t> data_b(40, 0x2b);
    req.key_ptr   = data_b.data();
    req.src_ptr   = data_b.data();
    req.src_size  = data_b.size();
    req.dst_size  = data_b.size();
    req.expert_id = 1;
    result        = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::FAILED || !result.device_ptr) {
        fprintf(stderr, "Failed to overcommit within cap\n");
        return false;
    }
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    cache.pin(data_b.data(), GGML_LAYOUT_AOS);

    std::vector<uint8_t> data_c(100, 0x3c);
    req.key_ptr   = data_c.data();
    req.src_ptr   = data_c.data();
    req.src_size  = data_c.size();
    req.dst_size  = data_c.size();
    req.expert_id = 2;
    result        = cache.ensure_cached_layout(req, {});

    if (result.status != ggml_sycl::cache_layout_status::FAILED) {
        fprintf(stderr, "Overcommit beyond cap unexpectedly succeeded\n");
        return false;
    }

    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    sycl::queue q;
    try {
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    bool ok = true;
    ok &= test_evict_returns_bytes(q);
    ok &= test_realloc_failure_keeps_entry(q);
    ok &= test_realloc_eviction_failure_keeps_entry(q);
    ok &= test_all_pinned_eviction_failure_new_entry(q);
    ok &= test_partial_eviction_insufficient(q);
    ok &= test_allocation_failure_new_entry(q);
    ok &= test_deferred_free_stress(q);
    ok &= test_unaligned_hash(q);
    ok &= test_unpin_experts(q);
    ok &= test_moe_overcommit_cap(q);

    printf("\nUnified cache bug tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
