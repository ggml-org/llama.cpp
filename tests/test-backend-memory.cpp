#include "../ggml/src/ggml-backend-memory.h"

#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>

using planner_ptr = std::unique_ptr<ggml_backend_memory_planner, decltype(&ggml_backend_memory_planner_free)>;

static planner_ptr make_planner(size_t capacity, size_t alignment) {
    return planner_ptr(ggml_backend_memory_planner_new(capacity, alignment), ggml_backend_memory_planner_free);
}

static ggml_backend_memory_region get_region(ggml_backend_memory_planner_t planner, uint64_t id) {
    ggml_backend_memory_region region = {};
    GGML_ASSERT(ggml_backend_memory_planner_get_region(planner, id, &region));
    return region;
}

// Validate construction, transaction state, and generation changes.
static void test_lifecycle() {
    ggml_backend_memory_planner_free(nullptr);
    GGML_ASSERT(ggml_backend_memory_planner_capacity(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_planner_alignment(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_planner_region_count(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_planner_used(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_planner_high_water(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_planner_generation(nullptr) == 0);
    GGML_ASSERT(!ggml_backend_memory_planner_begin(nullptr, GGML_BACKEND_MEMORY_PLAN_NONE));

    GGML_ASSERT(ggml_backend_memory_planner_new(0, 16) == nullptr);
    GGML_ASSERT(ggml_backend_memory_planner_new(256, 0) == nullptr);
    GGML_ASSERT(ggml_backend_memory_planner_new(256, 3) == nullptr);

    auto planner = make_planner(256, 16);
    GGML_ASSERT(planner);
    GGML_ASSERT(ggml_backend_memory_planner_capacity(planner.get()) == 256);
    GGML_ASSERT(ggml_backend_memory_planner_alignment(planner.get()) == 16);
    GGML_ASSERT(ggml_backend_memory_planner_generation(planner.get()) == 0);
    GGML_ASSERT(ggml_backend_memory_planner_region_count(planner.get()) == 0);
    GGML_ASSERT(!ggml_backend_memory_planner_commit(planner.get()));
    GGML_ASSERT(!ggml_backend_memory_planner_begin(planner.get(), 1u << 8));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve(
        planner.get(), 1, 8, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));

    GGML_ASSERT(ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(!ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_planner_commit(planner.get()));
    GGML_ASSERT(ggml_backend_memory_planner_generation(planner.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_planner_region_count(planner.get()) == 0);

    GGML_ASSERT(ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    ggml_backend_memory_planner_rollback(planner.get());
    GGML_ASSERT(ggml_backend_memory_planner_generation(planner.get()) == 1);
    GGML_ASSERT(!ggml_backend_memory_planner_commit(planner.get()));
}

// Verify deterministic first-fit placement and exact reservations.
static void test_placement() {
    auto planner = make_planner(256, 16);
    GGML_ASSERT(planner);
    GGML_ASSERT(ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));

    ggml_backend_memory_region r1 = {};
    ggml_backend_memory_region r2 = {};
    ggml_backend_memory_region r3 = {};
    ggml_backend_memory_region r4 = {};
    ggml_backend_memory_region r5 = {};

    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        planner.get(), 1, 15, 0, GGML_BACKEND_MEMORY_REGION_NONE, &r1));
    GGML_ASSERT(r1.offset == 0 && r1.size == 15 && r1.alignment == 16);

    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        planner.get(), 2, 16, 64, GGML_BACKEND_MEMORY_REGION_NONE, &r2));
    GGML_ASSERT(r2.offset == 64 && r2.alignment == 64);

    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        planner.get(), 3, 32, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, &r3));
    GGML_ASSERT(r3.offset == 32);

    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        planner.get(), 4, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, &r4));
    GGML_ASSERT(r4.offset == 16);

    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        planner.get(), 5, 240, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, &r5));
    GGML_ASSERT(r5.offset == 240);

    GGML_ASSERT(ggml_backend_memory_planner_commit(planner.get()));
    GGML_ASSERT(ggml_backend_memory_planner_region_count(planner.get()) == 5);
    GGML_ASSERT(ggml_backend_memory_planner_used(planner.get()) == 79);
    GGML_ASSERT(ggml_backend_memory_planner_high_water(planner.get()) == 256);

    const size_t expected_offsets[] = {0, 16, 32, 64, 240};
    for (size_t i = 0; i < 5; ++i) {
        ggml_backend_memory_region region = {};
        GGML_ASSERT(ggml_backend_memory_planner_get_region_at(planner.get(), i, &region));
        GGML_ASSERT(region.offset == expected_offsets[i]);
    }
    ggml_backend_memory_region missing = {};
    GGML_ASSERT(!ggml_backend_memory_planner_get_region_at(planner.get(), 5, &missing));
    GGML_ASSERT(!ggml_backend_memory_planner_get_region(planner.get(), 99, &missing));
}

// Ensure failed requests do not change the staged or committed layout.
static void test_failure_atomicity() {
    auto planner = make_planner(64, 8);
    GGML_ASSERT(planner);
    GGML_ASSERT(ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));

    ggml_backend_memory_region region = {};
    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        planner.get(), 1, 8, 16, 8, GGML_BACKEND_MEMORY_REGION_NONE, &region));

    const ggml_backend_memory_region unchanged = {99, 1, 2, 4, 8};
    region = unchanged;
    GGML_ASSERT(!ggml_backend_memory_planner_reserve_at(
        planner.get(), 2, 16, 16, 8, GGML_BACKEND_MEMORY_REGION_NONE, &region));
    GGML_ASSERT(region.id == unchanged.id && region.offset == unchanged.offset &&
                region.size == unchanged.size && region.alignment == unchanged.alignment &&
                region.flags == unchanged.flags);

    GGML_ASSERT(!ggml_backend_memory_planner_reserve_at(
        planner.get(), 1, 32, 8, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve_at(
        planner.get(), 3, 3, 8, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve_at(
        planner.get(), 4, 56, 9, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve(
        planner.get(), 5, 0, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve(
        planner.get(), 6, 8, 3, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve(
        planner.get(), 7, 8, 8, 1u << 8, nullptr));

    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        planner.get(), 8, 8, 8, GGML_BACKEND_MEMORY_REGION_NONE, &region));
    GGML_ASSERT(region.offset == 0);
    GGML_ASSERT(ggml_backend_memory_planner_commit(planner.get()));
    GGML_ASSERT(ggml_backend_memory_planner_region_count(planner.get()) == 2);
    GGML_ASSERT(ggml_backend_memory_planner_used(planner.get()) == 24);
}

// Preserve only persistent regions when constructing the next layout.
static void test_persistent_transition() {
    auto planner = make_planner(128, 8);
    GGML_ASSERT(planner);
    GGML_ASSERT(ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));

    ggml_backend_memory_region persistent = {};
    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        planner.get(), 10, 32, 16, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, &persistent));
    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        planner.get(), 20, 0, 16, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_planner_commit(planner.get()));
    GGML_ASSERT(ggml_backend_memory_planner_generation(planner.get()) == 1);

    GGML_ASSERT(ggml_backend_memory_planner_begin(
        planner.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    ggml_backend_memory_region next = {};
    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        planner.get(), 30, 24, 8, GGML_BACKEND_MEMORY_REGION_NONE, &next));
    GGML_ASSERT(next.offset == 0);
    ggml_backend_memory_planner_rollback(planner.get());

    GGML_ASSERT(ggml_backend_memory_planner_generation(planner.get()) == 1);
    GGML_ASSERT(get_region(planner.get(), 20).offset == 0);
    GGML_ASSERT(!ggml_backend_memory_planner_get_region(planner.get(), 30, &next));

    GGML_ASSERT(ggml_backend_memory_planner_begin(
        planner.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        planner.get(), 30, 24, 8, GGML_BACKEND_MEMORY_REGION_NONE, &next));
    GGML_ASSERT(ggml_backend_memory_planner_commit(planner.get()));

    GGML_ASSERT(ggml_backend_memory_planner_generation(planner.get()) == 2);
    GGML_ASSERT(ggml_backend_memory_planner_region_count(planner.get()) == 2);
    GGML_ASSERT(get_region(planner.get(), 10).offset == persistent.offset);
    GGML_ASSERT(!ggml_backend_memory_planner_get_region(planner.get(), 20, &next));
    GGML_ASSERT(get_region(planner.get(), 30).offset == 0);
}

// Check exact capacity boundaries and size arithmetic near SIZE_MAX.
static void test_boundaries_and_overflow() {
    auto exact = make_planner(32, 8);
    GGML_ASSERT(exact);
    GGML_ASSERT(ggml_backend_memory_planner_begin(exact.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_planner_reserve(
        exact.get(), 1, 32, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve(
        exact.get(), 2, 1, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_planner_commit(exact.get()));
    GGML_ASSERT(ggml_backend_memory_planner_high_water(exact.get()) == 32);

    const size_t max = std::numeric_limits<size_t>::max();
    auto huge = make_planner(max, 1);
    GGML_ASSERT(huge);
    GGML_ASSERT(ggml_backend_memory_planner_begin(huge.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        huge.get(), 1, 0, max - 4, 1, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve(
        huge.get(), 2, 1, 8, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_planner_reserve_at(
        huge.get(), 3, max - 4, 4, 1, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_planner_reserve_at(
        huge.get(), 4, max, 1, 1, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_planner_commit(huge.get()));
    GGML_ASSERT(ggml_backend_memory_planner_high_water(huge.get()) == max);
}

// Exercise mixed requests and verify all committed interval invariants.
static void test_mixed_request_properties() {
    auto planner = make_planner(4096, 8);
    GGML_ASSERT(planner);
    GGML_ASSERT(ggml_backend_memory_planner_begin(planner.get(), GGML_BACKEND_MEMORY_PLAN_NONE));

    uint32_t random = 0x31415926u;
    size_t successes = 0;
    size_t failures = 0;
    for (uint64_t id = 1; id <= 300; ++id) {
        random = random*1664525u + 1013904223u;
        const size_t size = random%96 + 1;
        const size_t requested_alignment = size_t(1) << ((random >> 8)%7);
        ggml_backend_memory_region region = {999, 999, 999, 999, 999};
        bool success;
        if (random & 1) {
            success = ggml_backend_memory_planner_reserve(
                planner.get(), id, size, requested_alignment, GGML_BACKEND_MEMORY_REGION_NONE, &region);
        } else {
            const size_t effective_alignment = std::max<size_t>(8, requested_alignment);
            const size_t slots = 4096/effective_alignment;
            const size_t offset = ((random >> 16)%slots)*effective_alignment;
            success = ggml_backend_memory_planner_reserve_at(
                planner.get(), id, offset, size, requested_alignment, GGML_BACKEND_MEMORY_REGION_NONE, &region);
        }
        if (success) {
            successes++;
            GGML_ASSERT(region.id == id);
        } else {
            failures++;
            GGML_ASSERT(region.id == 999 && region.offset == 999 && region.size == 999 &&
                        region.alignment == 999 && region.flags == 999);
        }
    }
    GGML_ASSERT(successes > 0 && failures > 0);
    GGML_ASSERT(ggml_backend_memory_planner_commit(planner.get()));
    GGML_ASSERT(ggml_backend_memory_planner_region_count(planner.get()) == successes);

    size_t used = 0;
    size_t previous_end = 0;
    for (size_t i = 0; i < successes; ++i) {
        ggml_backend_memory_region region = {};
        GGML_ASSERT(ggml_backend_memory_planner_get_region_at(planner.get(), i, &region));
        GGML_ASSERT(region.size > 0 && region.offset%region.alignment == 0);
        GGML_ASSERT(region.offset >= previous_end && region.offset <= 4096);
        GGML_ASSERT(region.size <= 4096 - region.offset);
        GGML_ASSERT(region.flags == GGML_BACKEND_MEMORY_REGION_NONE);
        GGML_ASSERT(get_region(planner.get(), region.id).offset == region.offset);
        previous_end = region.offset + region.size;
        used += region.size;
    }
    GGML_ASSERT(ggml_backend_memory_planner_used(planner.get()) == used);
    GGML_ASSERT(ggml_backend_memory_planner_high_water(planner.get()) == previous_end);
}

static void run(const char * name, void (*test)()) {
    std::printf("%s ", name);
    std::fflush(stdout);
    test();
    std::printf("PASSED\n");
}

int main() {
    run("test_lifecycle", test_lifecycle);
    run("test_placement", test_placement);
    run("test_failure_atomicity", test_failure_atomicity);
    run("test_persistent_transition", test_persistent_transition);
    run("test_boundaries_and_overflow", test_boundaries_and_overflow);
    run("test_mixed_request_properties", test_mixed_request_properties);
    return 0;
}
