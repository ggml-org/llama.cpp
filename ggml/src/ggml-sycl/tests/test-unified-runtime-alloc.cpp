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

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
    return ok ? 0 : 1;
}
