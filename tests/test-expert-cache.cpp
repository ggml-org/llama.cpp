// SPDX-License-Identifier: MIT
#include "ggml.h"
#include "ggml-backend.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

static void test_expert_cache_enable_disable() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(cpu);

    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);
    assert(sched);

    // Stats should be zero before enabling
    int64_t hits = -1, misses = -1, fate = -1, saved = -1, copied = -1;
    ggml_backend_sched_get_expert_cache_stats(sched, &hits, &misses, &fate, &saved, &copied);
    assert(hits == 0);
    assert(misses == 0);
    assert(fate == 0);
    assert(saved == 0);
    assert(copied == 0);

    ggml_backend_sched_set_expert_cache(sched, true);
    ggml_backend_sched_set_expert_cache(sched, false);

    ggml_backend_sched_get_expert_cache_stats(sched, &hits, &misses, &fate, &saved, &copied);
    assert(hits == 0);
    assert(misses == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_enable_disable\n");
}

static void test_expert_cache_reset_survives() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    ggml_backend_sched_set_expert_cache(sched, true);

    for (int i = 0; i < 10; i++) {
        ggml_backend_sched_reset(sched);
    }

    int64_t h, m, f, s, c;
    ggml_backend_sched_get_expert_cache_stats(sched, &h, &m, &f, &s, &c);
    assert(h == 0 && m == 0 && f == 0 && s == 0 && c == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_reset_survives\n");
}

static void test_expert_cache_toggle() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    ggml_backend_sched_set_expert_cache(sched, false);
    ggml_backend_sched_set_expert_cache(sched, true);
    ggml_backend_sched_set_expert_cache(sched, false);

    int64_t h, m, f, s, c;
    ggml_backend_sched_get_expert_cache_stats(sched, &h, &m, &f, &s, &c);
    assert(h == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_toggle\n");
}

static void test_expert_cache_null_stats() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    ggml_backend_sched_set_expert_cache(sched, true);
    ggml_backend_sched_get_expert_cache_stats(sched, nullptr, nullptr, nullptr, nullptr, nullptr);

    int64_t hits;
    ggml_backend_sched_get_expert_cache_stats(sched, &hits, nullptr, nullptr, nullptr, nullptr);
    assert(hits == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_null_stats\n");
}

static void test_expert_cache_disabled_no_crash() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    // Never enable -- sched must work normally
    ggml_backend_sched_reset(sched);

    int64_t h, m, f, s, c;
    ggml_backend_sched_get_expert_cache_stats(sched, &h, &m, &f, &s, &c);
    assert(h == 0 && m == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_disabled_no_crash\n");
}

int main() {
    printf("test-expert-cache:\n");
    test_expert_cache_enable_disable();
    test_expert_cache_reset_survives();
    test_expert_cache_toggle();
    test_expert_cache_null_stats();
    test_expert_cache_disabled_no_crash();
    printf("ALL TESTS PASSED\n");
    return 0;
}
