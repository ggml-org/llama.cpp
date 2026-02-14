//
// Unified runtime allocator tests
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "../unified-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sycl/sycl.hpp>

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

using namespace ggml_sycl;

static void enable_strict_mode_env() {
#if defined(_WIN32)
    (void) _putenv_s("GGML_SYCL_UNIFIED_ALLOC_STRICT", "1");
#else
    (void) setenv("GGML_SYCL_UNIFIED_ALLOC_STRICT", "1", 1);
#endif
}

static bool reserve_allocate_success_registers_pointer(sycl::queue & q) {
    TEST_BEGIN("reserve_allocate_success_registers_pointer");
    alloc_request req;
    req.queue                          = &q;
    req.size                           = 4096;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "unified_alloc failed");
    TEST_ASSERT(h.ptr != nullptr, "allocated pointer is null");

    alloc_handle looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "lookup failed");
    TEST_ASSERT(looked.ptr == h.ptr, "lookup ptr mismatch");
    TEST_ASSERT(looked.size == h.size, "lookup size mismatch");

    TEST_ASSERT(unified_free(h), "free failed");
    TEST_PASS();
    return true;
}

static bool allocate_failure_rolls_back_budget(sycl::queue & q) {
    TEST_BEGIN("allocate_failure_rolls_back_budget");
    const int device = 0;
    const size_t before = unified_cache_get_runtime_bytes(device);

    alloc_request req;
    req.queue                          = &q;
    req.size                           = (size_t) 1 << 50;  // 1 PB-ish for deterministic fail on device alloc
    req.intent.role                    = alloc_role::GRAPH_TMP;
    req.intent.category                = runtime_category::GRAPH;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    const bool ok = unified_alloc(req, &h);
    if (ok && h.ptr != nullptr) {
        // Unexpectedly succeeded, clean up and treat as pass.
        unified_free(h);
        TEST_PASS();
        return true;
    }
    const size_t after = unified_cache_get_runtime_bytes(device);
    TEST_ASSERT(before == after, "runtime bytes did not roll back after alloc failure");
    TEST_PASS();
    return true;
}

static bool free_unknown_pointer_fails() {
    TEST_BEGIN("free_unknown_pointer_fails");
    int stack_value = 0;
    TEST_ASSERT(!unified_free_ptr(&stack_value, -1), "free unknown pointer should fail");
    TEST_PASS();
    return true;
}

static bool strict_unknown_free_fails() {
    TEST_BEGIN("strict_unknown_free_fails");
    int stack_value = 0;
    TEST_ASSERT(!unified_free_ptr(&stack_value, 0), "strict unknown free should fail");
    TEST_PASS();
    return true;
}

static bool double_free_fails(sycl::queue & q) {
    TEST_BEGIN("double_free_fails");
    alloc_request req;
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::STAGING;
    req.intent.category                = runtime_category::STAGING;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    TEST_ASSERT(unified_free(h), "first free failed");
    TEST_ASSERT(!unified_free(h), "second free should fail");
    TEST_PASS();
    return true;
}

static bool lookup_returns_correct_metadata(sycl::queue & q) {
    TEST_BEGIN("lookup_returns_correct_metadata");
    alloc_request req;
    req.queue                               = &q;
    req.size                                = 8192;
    req.intent.role                         = alloc_role::COMPUTE;
    req.intent.category                     = runtime_category::COMPUTE;
    req.intent.constraints.must_host_pinned = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    alloc_handle looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "lookup failed");
    TEST_ASSERT(looked.tier == alloc_tier::HOST_PINNED, "tier mismatch");
    TEST_ASSERT(looked.role == alloc_role::COMPUTE, "role mismatch");
    TEST_ASSERT(looked.category == runtime_category::COMPUTE, "category mismatch");
    TEST_ASSERT(unified_free(h), "free failed");
    TEST_PASS();
    return true;
}

static bool cohort_prefers_weight_tier_for_compute(sycl::queue & q) {
    TEST_BEGIN("cohort_prefers_weight_tier_for_compute");
    alloc_request seed;
    seed.queue                               = &q;
    seed.size                                = 4096;
    seed.intent.role                         = alloc_role::WEIGHT;
    seed.intent.category                     = runtime_category::OTHER;
    seed.intent.cohort_id                    = "test:cohort";
    seed.intent.constraints.must_host_pinned = true;

    alloc_handle seed_h{};
    TEST_ASSERT(unified_alloc(seed, &seed_h), "seed alloc failed");

    alloc_request req;
    req.queue                                         = &q;
    req.size                                          = 2048;
    req.intent.role                                   = alloc_role::COMPUTE;
    req.intent.category                               = runtime_category::COMPUTE;
    req.intent.cohort_id                              = "test:cohort";
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    const alloc_tier tier = unified_select_tier(req);
    TEST_ASSERT(tier == alloc_tier::HOST_PINNED, "cohort policy did not preserve host tier");

    unified_free(seed_h);
    TEST_PASS();
    return true;
}

static bool hard_constraint_overrides_cohort(sycl::queue & q) {
    TEST_BEGIN("hard_constraint_overrides_cohort");
    alloc_request req;
    req.queue                                         = &q;
    req.size                                          = 2048;
    req.intent.role                                   = alloc_role::COMPUTE;
    req.intent.category                               = runtime_category::COMPUTE;
    req.intent.cohort_id                              = "test:cohort";
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    req.intent.constraints.must_device                = true;
    const alloc_tier tier = unified_select_tier(req);
    TEST_ASSERT(tier == alloc_tier::DEVICE_VRAM, "must_device did not override cohort");
    TEST_PASS();
    return true;
}

static bool policy_never_selects_shared_usm(sycl::queue & q) {
    TEST_BEGIN("policy_never_selects_shared_usm");
    alloc_request req;
    req.queue          = &q;
    req.size           = 1024;
    req.intent.role    = alloc_role::OTHER;
    req.intent.category = runtime_category::OTHER;
    const alloc_tier tier = unified_select_tier(req);
    TEST_ASSERT(tier == alloc_tier::DEVICE_VRAM || tier == alloc_tier::HOST_PINNED, "unexpected tier selected");
    TEST_PASS();
    return true;
}

static bool strict_stale_handle_fails(sycl::queue & q) {
    TEST_BEGIN("strict_stale_handle_fails");
    alloc_request req;
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    alloc_handle stale = h;
    TEST_ASSERT(unified_free(h), "free failed");
    TEST_ASSERT(!unified_free(stale), "stale handle free should fail");
    TEST_PASS();
    return true;
}

static bool strict_device_mismatch_fails(sycl::queue & q) {
    TEST_BEGIN("strict_device_mismatch_fails");
    alloc_request req;
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    TEST_ASSERT(!unified_free_ptr(h.ptr, h.device + 1), "device mismatch free should fail");
    alloc_handle looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "allocation should remain registered after mismatch");
    TEST_ASSERT(unified_free(h), "cleanup free failed");
    TEST_PASS();
    return true;
}

static bool scoped_unified_alloc_frees_on_scope_exit(sycl::queue & q) {
    TEST_BEGIN("scoped_unified_alloc_frees_on_scope_exit");
    alloc_request req;
    req.queue                               = &q;
    req.size                                = 4096;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;

    void * ptr = nullptr;
    {
        scoped_unified_alloc scoped(req);
        TEST_ASSERT(scoped, "scoped allocation failed");
        ptr = scoped.get();
        TEST_ASSERT(ptr != nullptr, "scoped pointer null");
        alloc_handle looked{};
        TEST_ASSERT(unified_lookup(ptr, &looked), "lookup should succeed while in scope");
    }
    alloc_handle looked{};
    TEST_ASSERT(!unified_lookup(ptr, &looked), "lookup should fail after scope exit");
    TEST_PASS();
    return true;
}

static bool offload_pool_reuse_tracks_hit_miss(sycl::queue & q) {
    TEST_BEGIN("offload_pool_reuse_tracks_hit_miss");
    offload_buffer_pool_trim(-1);
    offload_stats_reset();

    offload_buffer_request req{};
    req.queue    = &q;
    req.device   = -1;
    req.size     = 4096;
    req.role     = offload_buffer_role::STAGING_SRC0;
    req.intent.role = alloc_role::STAGING;
    req.intent.category = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    req.intent.cohort_id = "test:offload_pool";

    offload_buffer_lease a{};
    TEST_ASSERT(acquire_offload_buffer(req, &a), "first acquire failed");
    TEST_ASSERT(a.valid && a.handle.ptr != nullptr, "first lease invalid");
    TEST_ASSERT(release_offload_buffer(a), "first release failed");

    offload_buffer_lease b{};
    TEST_ASSERT(acquire_offload_buffer(req, &b), "second acquire failed");
    TEST_ASSERT(b.valid && b.handle.ptr != nullptr, "second lease invalid");
    TEST_ASSERT(release_offload_buffer(b), "second release failed");

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.pool_miss_count >= 1, "expected at least one pool miss");
    TEST_ASSERT(stats.pool_hit_count >= 1, "expected at least one pool hit");
    TEST_PASS();
    return true;
}

static bool offload_pool_stale_lease_fails(sycl::queue & q) {
    TEST_BEGIN("offload_pool_stale_lease_fails");
    offload_buffer_request req{};
    req.queue    = &q;
    req.device   = -1;
    req.size     = 2048;
    req.role     = offload_buffer_role::STAGING_DST;
    req.intent.role = alloc_role::STAGING;
    req.intent.category = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;

    offload_buffer_lease lease{};
    TEST_ASSERT(acquire_offload_buffer(req, &lease), "acquire failed");
    TEST_ASSERT(release_offload_buffer(lease), "release failed");
    TEST_ASSERT(!release_offload_buffer(lease), "stale lease release should fail");
    TEST_PASS();
    return true;
}

static bool offload_pool_trim_clears_released_entries(sycl::queue & q) {
    TEST_BEGIN("offload_pool_trim_clears_released_entries");
    offload_buffer_pool_trim(-1);
    offload_stats_reset();

    offload_buffer_request req{};
    req.queue    = &q;
    req.device   = -1;
    req.size     = 1024;
    req.role     = offload_buffer_role::STAGING_SRC1;
    req.intent.role = alloc_role::STAGING;
    req.intent.category = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;

    offload_buffer_lease lease{};
    TEST_ASSERT(acquire_offload_buffer(req, &lease), "first acquire failed");
    TEST_ASSERT(release_offload_buffer(lease), "release failed");
    offload_buffer_pool_trim(-1);

    offload_buffer_lease after_trim{};
    TEST_ASSERT(acquire_offload_buffer(req, &after_trim), "acquire after trim failed");
    TEST_ASSERT(release_offload_buffer(after_trim), "release after trim failed");

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.pool_miss_count >= 2, "expected miss after trim");
    TEST_PASS();
    return true;
}

static bool offload_wait_stats_split_tracks_forced_and_fallback() {
    TEST_BEGIN("offload_wait_stats_split_tracks_forced_and_fallback");
    offload_stats_reset();

    offload_stats_note_wait(false);
    offload_stats_note_wait(true);
    offload_stats_note_wait(false);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.wait_count == 3, "expected total wait_count to be 3");
    TEST_ASSERT(stats.wait_count_forced == 2, "expected wait_count_forced to be 2");
    TEST_ASSERT(stats.wait_count_fallback == 1, "expected wait_count_fallback to be 1");
    TEST_PASS();
    return true;
}

static bool offload_cross_domain_stats_split_by_phase() {
    TEST_BEGIN("offload_cross_domain_stats_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_cross_domain_transfer(0);
    offload_stats_note_cross_domain_transfer(128);

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_cross_domain_transfer(256);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.cross_domain_transfer_count == 3, "expected three cross-domain boundaries");
    TEST_ASSERT(stats.cross_domain_transfer_count_pp == 2, "expected two PP cross-domain transfers");
    TEST_ASSERT(stats.cross_domain_transfer_count_tg == 1, "expected one TG cross-domain transfer");
    TEST_PASS();
    return true;
}

static bool offload_transfer_bytes_split_by_phase() {
    TEST_BEGIN("offload_transfer_bytes_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_transfer(true, 96);
    offload_stats_note_transfer(false, 48);

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_transfer(true, 24);
    offload_stats_note_transfer(false, 12);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.transfer_bytes_h2d == 120, "unexpected total H2D bytes");
    TEST_ASSERT(stats.transfer_bytes_d2h == 60, "unexpected total D2H bytes");
    TEST_ASSERT(stats.transfer_bytes_h2d_pp == 96, "unexpected PP H2D bytes");
    TEST_ASSERT(stats.transfer_bytes_h2d_tg == 24, "unexpected TG H2D bytes");
    TEST_ASSERT(stats.transfer_bytes_d2h_pp == 48, "unexpected PP D2H bytes");
    TEST_ASSERT(stats.transfer_bytes_d2h_tg == 12, "unexpected TG D2H bytes");
    TEST_PASS();
    return true;
}

static bool offload_dispatch_counts_split_by_phase() {
    TEST_BEGIN("offload_dispatch_counts_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_dispatch(true, false);   // CPU
    offload_stats_note_dispatch(false, false);  // GPU
    offload_stats_note_dispatch(false, true);   // GPU island

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_dispatch(true, false);   // CPU
    offload_stats_note_dispatch(false, true);   // GPU island

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.dispatch_count_cpu == 2, "unexpected total CPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu == 3, "unexpected total GPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_island == 2, "unexpected total GPU island dispatch count");
    TEST_ASSERT(stats.dispatch_count_cpu_pp == 1, "unexpected PP CPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_cpu_tg == 1, "unexpected TG CPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_pp == 2, "unexpected PP GPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_tg == 1, "unexpected TG GPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_island_pp == 1, "unexpected PP GPU-island dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_island_tg == 1, "unexpected TG GPU-island dispatch count");
    TEST_PASS();
    return true;
}

static bool offload_phase_roundtrip() {
    TEST_BEGIN("offload_phase_roundtrip");
    offload_stats_reset();
    TEST_ASSERT(offload_stats_phase() == offload_phase::UNKNOWN, "expected UNKNOWN after reset");
    offload_stats_set_phase(offload_phase::PP);
    TEST_ASSERT(offload_stats_phase() == offload_phase::PP, "expected PP phase");
    offload_stats_set_phase(offload_phase::TG);
    TEST_ASSERT(offload_stats_phase() == offload_phase::TG, "expected TG phase");
    TEST_PASS();
    return true;
}

static bool offload_transition_wait_stats_split_by_phase() {
    TEST_BEGIN("offload_transition_wait_stats_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_transition_wait(true);
    offload_stats_note_transition_wait(false);

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_transition_wait(true);
    offload_stats_note_transition_wait(false);
    offload_stats_note_transition_wait(false);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.transition_wait_count == 2, "unexpected transition wait count");
    TEST_ASSERT(stats.transition_wait_count_pp == 1, "unexpected PP transition wait count");
    TEST_ASSERT(stats.transition_wait_count_tg == 1, "unexpected TG transition wait count");
    TEST_ASSERT(stats.transition_wait_elided_count == 3, "unexpected transition wait-elided count");
    TEST_ASSERT(stats.transition_wait_elided_count_pp == 1, "unexpected PP transition wait-elided count");
    TEST_ASSERT(stats.transition_wait_elided_count_tg == 2, "unexpected TG transition wait-elided count");
    TEST_PASS();
    return true;
}

static bool offload_host_alloc_stats_split_by_tag() {
    TEST_BEGIN("offload_host_alloc_stats_split_by_tag");
    offload_stats_reset();

    offload_stats_note_host_alloc("unified_alloc:host", 128);
    offload_stats_note_host_alloc("unified_alloc:host", 64);
    offload_stats_note_host_alloc("unified_cache:host_chunk", 256);
    offload_stats_note_host_alloc("host_malloc", 512);
    offload_stats_note_host_alloc("custom:other", 32);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.host_alloc_call_count == 5, "unexpected host_alloc_call_count");
    TEST_ASSERT(stats.host_alloc_bytes == 992, "unexpected host_alloc_bytes");
    TEST_ASSERT(stats.host_alloc_calls_unified_alloc_host == 2, "unexpected unified_alloc host calls");
    TEST_ASSERT(stats.host_alloc_bytes_unified_alloc_host == 192, "unexpected unified_alloc host bytes");
    TEST_ASSERT(stats.host_alloc_calls_unified_cache_host_chunk == 1, "unexpected host_chunk calls");
    TEST_ASSERT(stats.host_alloc_bytes_unified_cache_host_chunk == 256, "unexpected host_chunk bytes");
    TEST_ASSERT(stats.host_alloc_calls_host_malloc == 1, "unexpected host_malloc calls");
    TEST_ASSERT(stats.host_alloc_bytes_host_malloc == 512, "unexpected host_malloc bytes");
    TEST_ASSERT(stats.host_alloc_calls_other == 1, "unexpected other calls");
    TEST_ASSERT(stats.host_alloc_bytes_other == 32, "unexpected other bytes");
    TEST_PASS();
    return true;
}

int main() {
    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "Unified Runtime Allocator Tests\n");
    fprintf(stderr, "===========================================\n");

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    } catch (const sycl::exception &) {
        try {
            q = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
        } catch (const sycl::exception & e) {
            fprintf(stderr, "No SYCL device available: %s\n", e.what());
            return 1;
        }
    }

    bool ok = true;
    enable_strict_mode_env();
    ok &= reserve_allocate_success_registers_pointer(q);
    ok &= allocate_failure_rolls_back_budget(q);
    ok &= free_unknown_pointer_fails();
    ok &= strict_unknown_free_fails();
    ok &= double_free_fails(q);
    ok &= lookup_returns_correct_metadata(q);
    ok &= cohort_prefers_weight_tier_for_compute(q);
    ok &= hard_constraint_overrides_cohort(q);
    ok &= policy_never_selects_shared_usm(q);
    ok &= strict_stale_handle_fails(q);
    ok &= strict_device_mismatch_fails(q);
    ok &= scoped_unified_alloc_frees_on_scope_exit(q);
    ok &= offload_pool_reuse_tracks_hit_miss(q);
    ok &= offload_pool_stale_lease_fails(q);
    ok &= offload_pool_trim_clears_released_entries(q);
    ok &= offload_wait_stats_split_tracks_forced_and_fallback();
    ok &= offload_cross_domain_stats_split_by_phase();
    ok &= offload_transfer_bytes_split_by_phase();
    ok &= offload_dispatch_counts_split_by_phase();
    ok &= offload_phase_roundtrip();
    ok &= offload_transition_wait_stats_split_by_phase();
    ok &= offload_host_alloc_stats_split_by_tag();

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
    return ok ? 0 : 1;
}
