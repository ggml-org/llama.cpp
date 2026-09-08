#include "../ggml/src/ggml-backend-memory.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-context-workspace.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

using planner_ptr = std::unique_ptr<ggml_backend_memory_planner, decltype(&ggml_backend_memory_planner_free)>;
using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct test_buft_context {
    size_t alignment = 16;
    size_t alloc_count = 0;
    size_t physical_free_count = 0;
    size_t view_free_count = 0;
    size_t view_attempts = 0;
    size_t view_fail_after = SIZE_MAX;
    bool fail_alloc = false;
    bool support_views = true;
};

struct test_buffer_context {
    test_buft_context * owner;
    uint8_t * base;
    bool owns_data;
};

static ggml_backend_buffer_t test_buffer_view(
        ggml_backend_buffer_t buffer, size_t offset, size_t size);

static const char * test_buft_name(ggml_backend_buffer_type_t) {
    return "memory_planner_test";
}

static void test_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<test_buffer_context *>(buffer->context);
    if (context->owns_data) {
        delete[] context->base;
        context->owner->physical_free_count++;
    } else {
        context->owner->view_free_count++;
    }
    delete context;
}

static void * test_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<test_buffer_context *>(buffer->context)->base;
}

static void test_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    std::memset(test_buffer_base(buffer), value, buffer->size);
}

static ggml_backend_buffer_t test_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * owner = static_cast<test_buft_context *>(buft->context);
    if (owner->fail_alloc || size == 0) {
        return nullptr;
    }
    auto * data = new (std::nothrow) uint8_t[size];
    auto * context = new (std::nothrow) test_buffer_context {owner, data, true};
    if (data == nullptr || context == nullptr) {
        delete[] data;
        delete context;
        return nullptr;
    }

    ggml_backend_buffer_i iface = {};
    iface.free_buffer = test_buffer_free;
    iface.get_base = test_buffer_base;
    iface.clear = test_buffer_clear;
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, iface, context, size);
    buffer->view_buffer = owner->support_views ? test_buffer_view : nullptr;
    owner->alloc_count++;
    return buffer;
}

static size_t test_buft_alignment(ggml_backend_buffer_type_t buft) {
    return static_cast<test_buft_context *>(buft->context)->alignment;
}

static bool test_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static ggml_backend_buffer_type make_test_buft(test_buft_context * context) {
    ggml_backend_buffer_type buft = {};
    buft.context = context;
    buft.iface.get_name = test_buft_name;
    buft.iface.alloc_buffer = test_buft_alloc;
    buft.iface.get_alignment = test_buft_alignment;
    buft.iface.is_host = test_buft_is_host;
    return buft;
}

static ggml_backend_buffer_t test_buffer_view(
        ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    auto * parent = static_cast<test_buffer_context *>(buffer->context);
    parent->owner->view_attempts++;
    if (parent->owner->view_attempts > parent->owner->view_fail_after) {
        return nullptr;
    }
    auto * context = new (std::nothrow) test_buffer_context {
        parent->owner,
        parent->base + offset,
        false,
    };
    if (context == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_t view = ggml_backend_buffer_init(buffer->buft, buffer->iface, context, size);
    view->view_buffer = test_buffer_view;
    return view;
}


static planner_ptr make_planner(size_t capacity, size_t alignment) {
    return planner_ptr(ggml_backend_memory_planner_new(capacity, alignment), ggml_backend_memory_planner_free);
}

static ggml_backend_memory_region get_region(ggml_backend_memory_planner_t planner, uint64_t id) {
    ggml_backend_memory_region region = {};
    GGML_ASSERT(ggml_backend_memory_planner_get_region(planner, id, &region));
    return region;
}

static void test_workspace_group_planning() {
    test_buft_context first_context;
    test_buft_context second_context;
    second_context.alignment = 64;
    auto first_buft = make_test_buft(&first_context);
    auto second_buft = make_test_buft(&second_context);

    ggml_backend_buffer_type_t bufts[] = {
        &first_buft,
        &second_buft,
        &first_buft,
        &second_buft,
    };
    const size_t measurements[] = {
        0, 0, 64, 17,
        0, 1, 160, 33,
    };

    size_t n_groups = 0;
    GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(
        bufts, measurements, 2, 4, nullptr, &n_groups));
    GGML_ASSERT(n_groups == 2);

    ggml_backend_memory_workspace_group groups[2] = {{nullptr, 7, 0, 0}, {}};
    n_groups = 1;
    GGML_ASSERT(!ggml_backend_memory_plan_workspace_groups(
        bufts, measurements, 2, 4, groups, &n_groups));
    GGML_ASSERT(n_groups == 2);
    GGML_ASSERT(groups[0].size == 7);

    n_groups = 2;
    GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(
        bufts, measurements, 2, 4, groups, &n_groups));
    GGML_ASSERT(n_groups == 2);
    GGML_ASSERT(groups[0].buft == &first_buft);
    GGML_ASSERT(groups[0].size == 160);
    GGML_ASSERT(groups[0].alignment == 16);
    GGML_ASSERT(groups[0].first_slot == 0);
    GGML_ASSERT(groups[1].buft == &second_buft);
    GGML_ASSERT(groups[1].size == 64);
    GGML_ASSERT(groups[1].alignment == 64);
    GGML_ASSERT(groups[1].first_slot == 1);
}

static void test_workspace_group_planning_boundaries() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    ggml_backend_buffer_type_t bufts[] = {&buft};
    size_t n_groups = 9;

    GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(
        nullptr, nullptr, 0, 0, nullptr, &n_groups));
    GGML_ASSERT(n_groups == 0);

    const size_t zero[] = {0};
    n_groups = 9;
    GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(
        bufts, zero, 1, 1, nullptr, &n_groups));
    GGML_ASSERT(n_groups == 0);

    const size_t overflow[] = {SIZE_MAX};
    n_groups = 1;
    GGML_ASSERT(!ggml_backend_memory_plan_workspace_groups(
        bufts, overflow, 1, 1, nullptr, &n_groups));

    n_groups = 1;
    GGML_ASSERT(!ggml_backend_memory_plan_workspace_groups(
        nullptr, zero, 1, 1, nullptr, &n_groups));
    GGML_ASSERT(!ggml_backend_memory_plan_workspace_groups(
        bufts, nullptr, 1, 1, nullptr, &n_groups));
    GGML_ASSERT(!ggml_backend_memory_plan_workspace_groups(
        bufts, zero, 1, 1, nullptr, nullptr));

    context.alignment = 24;
    const size_t nonzero[] = {1};
    n_groups = 1;
    GGML_ASSERT(!ggml_backend_memory_plan_workspace_groups(
        bufts, nonzero, 1, 1, nullptr, &n_groups));
}

static void test_compute_arena_binding_ownership() {
    test_buft_context context;
    auto buft = make_test_buft(&context);

    {
        llama_compute_arena_binding binding = {
            &buft,
            128,
            3,
            llama_compute_arena_ptr(ggml_backend_memory_arena_new(&buft, 128)),
        };
        GGML_ASSERT(binding.arena);
        GGML_ASSERT(binding.buft == &buft);
        GGML_ASSERT(binding.capacity == 128);
        GGML_ASSERT(binding.first_slot == 3);

        ggml_backend_memory_arena_t retained =
            ggml_backend_memory_arena_retain(binding.arena.get());
        GGML_ASSERT(retained);

        llama_compute_arena_binding moved = std::move(binding);
        GGML_ASSERT(!binding.arena);
        GGML_ASSERT(moved.arena.get() == retained);
        moved.arena.reset();
        GGML_ASSERT(context.physical_free_count == 0);
        GGML_ASSERT(ggml_backend_memory_arena_capacity(retained) == 128);
        ggml_backend_memory_arena_free(retained);
        GGML_ASSERT(context.physical_free_count == 1);
    }

    {
        llama_compute_arena_binding first = {
            &buft,
            64,
            0,
            llama_compute_arena_ptr(ggml_backend_memory_arena_new(&buft, 64)),
        };
        llama_compute_arena_binding second = {
            &buft,
            96,
            1,
            llama_compute_arena_ptr(ggml_backend_memory_arena_new(&buft, 96)),
        };
        GGML_ASSERT(first.arena && second.arena);
        second = std::move(first);
        GGML_ASSERT(!first.arena);
        GGML_ASSERT(second.capacity == 64);
        GGML_ASSERT(second.first_slot == 0);
        GGML_ASSERT(context.physical_free_count == 2);
    }
    GGML_ASSERT(context.alloc_count == 3);
    GGML_ASSERT(context.physical_free_count == 3);
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


// Verify an imported parent remains alive and views stay inside their regions.
static void test_arena_imported_parent() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    ggml_backend_buffer_t parent = ggml_backend_buft_alloc_buffer(&buft, 128);
    GGML_ASSERT(parent != nullptr);
    ggml_backend_buffer_clear(parent, 0xa5);
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(parent));

    arena_ptr arena(ggml_backend_memory_arena_new_from_buffer(parent), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena && ggml_backend_memory_arena_parent(arena.get()) == parent);
    ggml_backend_buffer_free(parent);
    GGML_ASSERT(context.physical_free_count == 0);

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 10, 32, 32, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_region_count(arena.get()) == 1);
    ggml_backend_memory_region region = {};
    GGML_ASSERT(ggml_backend_memory_arena_get_region(arena.get(), 10, &region));
    GGML_ASSERT(region.offset == 32 && region.size == 32 &&
                region.flags == GGML_BACKEND_MEMORY_REGION_PERSISTENT);

    ggml_backend_buffer_t view = ggml_backend_memory_arena_get_buffer(arena.get(), 10);
    GGML_ASSERT(view != nullptr && ggml_backend_buffer_get_base(view) == base + 32);
    ggml_backend_buffer_clear(view, 0x3c);
    for (size_t i = 0; i < 128; ++i) {
        GGML_ASSERT(base[i] == (i >= 32 && i < 64 ? 0x3c : 0xa5));
    }

    arena.reset();
    GGML_ASSERT(context.view_free_count == 1);
    GGML_ASSERT(context.physical_free_count == 1);
}

// Verify an arena-owned parent and all materialized views are released once.
static void test_arena_owned_parent() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena && context.alloc_count == 1);
    GGML_ASSERT(ggml_backend_memory_arena_capacity(arena.get()) == 128);

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 2, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_used(arena.get()) == 48);
    GGML_ASSERT(ggml_backend_memory_arena_high_water(arena.get()) == 48);

    arena.reset();
    GGML_ASSERT(context.view_free_count == 2);
    GGML_ASSERT(context.physical_free_count == 1);
}

// Reject allocation and unsupported-view backends without leaking the parent.
static void test_arena_construction_failure() {
    ggml_backend_memory_arena_free(nullptr);
    ggml_backend_memory_arena_rollback(nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_new(nullptr, 128) == nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_new_from_buffer(nullptr) == nullptr);
    GGML_ASSERT(!ggml_backend_memory_arena_begin(nullptr, GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(!ggml_backend_memory_arena_reserve(
        nullptr, 1, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_reserve_at(
        nullptr, 1, 0, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_commit(nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_parent(nullptr) == nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(nullptr, 1) == nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_capacity(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_region_count(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_used(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_high_water(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_generation(nullptr) == 0);
    ggml_backend_memory_region missing = {};
    GGML_ASSERT(!ggml_backend_memory_arena_get_region(nullptr, 1, &missing));

    test_buft_context allocation_failure;
    allocation_failure.fail_alloc = true;
    auto fail_buft = make_test_buft(&allocation_failure);
    GGML_ASSERT(ggml_backend_memory_arena_new(&fail_buft, 128) == nullptr);
    GGML_ASSERT(allocation_failure.alloc_count == 0 && allocation_failure.physical_free_count == 0);

    test_buft_context unsupported;
    unsupported.support_views = false;
    auto unsupported_buft = make_test_buft(&unsupported);
    GGML_ASSERT(ggml_backend_memory_arena_new(&unsupported_buft, 128) == nullptr);
    GGML_ASSERT(unsupported.alloc_count == 1 && unsupported.physical_free_count == 1);
}

// Keep the committed layout unchanged when any staged view fails to materialize.
static void test_arena_commit_atomicity() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 9, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    ggml_backend_buffer_t committed = ggml_backend_buffer_retain(
        ggml_backend_memory_arena_get_buffer(arena.get(), 9));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 1);

    context.view_fail_after = 2;
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 2, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(context.view_attempts == 3 && context.view_free_count == 1);
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_region_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 9) == committed);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) == nullptr);

    context.view_fail_after = SIZE_MAX;
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 3, 64, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 2);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 9) == nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 3) != nullptr);

    ggml_backend_buffer_free(committed);
    arena.reset();
    GGML_ASSERT(context.view_free_count == 3);
    GGML_ASSERT(context.physical_free_count == 1);
}

// Preserve the logical buffer object for persistent regions across commits.
static void test_arena_persistent_view_identity() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 1, 32, 16, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 2, 0, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    ggml_backend_buffer_t persistent = ggml_backend_buffer_retain(
        ggml_backend_memory_arena_get_buffer(arena.get(), 1));

    GGML_ASSERT(ggml_backend_memory_arena_begin(
        arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 3, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) == persistent);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 2) == nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 3) != nullptr);

    ggml_backend_buffer_free(persistent);
    arena.reset();
    GGML_ASSERT(context.view_free_count == 3);
    GGML_ASSERT(context.physical_free_count == 1);
}

// Exercise arena views and persistent identity on a real backend.
static void test_arena_backend(ggml_backend_t backend) {
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t parent_size = 5*alignment;
    arena_ptr arena(ggml_backend_memory_arena_new(buft, parent_size), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    ggml_backend_buffer_t parent = ggml_backend_memory_arena_parent(arena.get());
    void * base = ggml_backend_buffer_get_base(parent);
    ggml_backend_buffer_clear(parent, 0xa5);

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 1, alignment, 2*alignment, alignment,
        GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(lease && ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    ggml_backend_buffer_t view = ggml_backend_memory_lease_buffer(lease.get());
    GGML_ASSERT(view && ggml_backend_buffer_get_base(view) == static_cast<uint8_t *>(base) + alignment);

    ggml_init_params params = {
        /*.mem_size   = */ 3*ggml_tensor_overhead(),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    auto * whole = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, parent_size/sizeof(float));
    GGML_ASSERT(ggml_backend_tensor_alloc(parent, whole, base) == GGML_STATUS_SUCCESS);
    auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2*alignment/sizeof(float));
    GGML_ASSERT(ggml_backend_tensor_alloc(
        view, tensor, ggml_backend_buffer_get_base(view)) == GGML_STATUS_SUCCESS);

    ggml_backend_buffer_clear(view, 0x3c);
    std::vector<uint8_t> bytes(parent_size);
    ggml_backend_tensor_get(whole, bytes.data(), 0, bytes.size());
    for (size_t i = 0; i < parent_size; ++i) {
        GGML_ASSERT(bytes[i] == (i >= alignment && i < 3*alignment ? 0x3c : 0xa5));
    }

    std::vector<float> input(2*alignment/sizeof(float), 1.25f);
    std::vector<float> output(input.size());
    ggml_backend_tensor_set(tensor, input.data(), 0, 2*alignment);
    ggml_backend_tensor_get(tensor, output.data(), 0, 2*alignment);
    GGML_ASSERT(input == output);

    GGML_ASSERT(ggml_backend_memory_arena_begin(
        arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) == view);
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
}

static void test_arena_real_backends() {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    GGML_ASSERT(cpu);
    test_arena_backend(cpu.get());

    ggml_backend_load_all();
    ggml_backend_ptr gpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
    if (gpu) {
        test_arena_backend(gpu.get());
    }
}

// Keep arena storage alive until its final retained reference is released.
static void test_arena_reference_lifetime() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    ggml_backend_memory_arena_t retained = ggml_backend_memory_arena_retain(arena.get());
    GGML_ASSERT(retained == arena.get());

    ggml_backend_memory_arena_free(arena.release());
    GGML_ASSERT(context.physical_free_count == 0);
    arena_ptr last(retained, ggml_backend_memory_arena_free);
    GGML_ASSERT(ggml_backend_memory_arena_begin(last.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        last.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(last.get()));

    last.reset();
    GGML_ASSERT(context.view_free_count == 1 && context.physical_free_count == 1);
}

// Keep arena storage alive until the final reference to a lease is released.
static void test_lease_lifetime() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));

    ggml_backend_memory_lease_t lease = ggml_backend_memory_arena_acquire(arena.get(), 1);
    GGML_ASSERT(lease != nullptr && ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_lease_buffer(lease) ==
                ggml_backend_memory_arena_get_buffer(arena.get(), 1));
    GGML_ASSERT(ggml_backend_memory_lease_generation(lease) == 1);
    ggml_backend_memory_region region = {};
    GGML_ASSERT(ggml_backend_memory_lease_get_region(lease, &region));
    GGML_ASSERT(region.id == 1 && region.offset == 0 && region.size == 32);

    GGML_ASSERT(ggml_backend_memory_lease_retain(lease) == lease);
    ggml_backend_memory_lease_free(lease);
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);

    ggml_backend_buffer_t buffer = ggml_backend_memory_lease_buffer(lease);
    ggml_backend_memory_arena_free(arena.release());
    GGML_ASSERT(context.physical_free_count == 0);
    ggml_backend_buffer_clear(buffer, 0x5a);
    ggml_backend_memory_lease_free(lease);
    GGML_ASSERT(context.view_free_count == 1 && context.physical_free_count == 1);
}

// Count separate lease objects while a retained lease handle counts once.
static void test_multiple_leases() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));

    lease_ptr first(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    lease_ptr second(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(first && second && first.get() != second.get());
    GGML_ASSERT(ggml_backend_memory_arena_acquire(arena.get(), 2) == nullptr);
    GGML_ASSERT(ggml_backend_memory_lease_buffer(first.get()) ==
                ggml_backend_memory_lease_buffer(second.get()));
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 2);

    GGML_ASSERT(ggml_backend_memory_lease_retain(first.get()) == first.get());
    ggml_backend_memory_lease_free(first.get());
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 2);
    first.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    second.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
}

// Reject replacement of an actively leased nonpersistent region.
static void test_lease_blocks_replacement() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(lease);

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 2, 64, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) ==
                ggml_backend_memory_lease_buffer(lease.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 2) == nullptr);

    lease.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 2, 64, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 2);
}

// Allow an unchanged persistent region to remain leased across generations.
static void test_persistent_lease_transition() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 1, 32, 16, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(lease && ggml_backend_memory_lease_generation(lease.get()) == 1);
    ggml_backend_buffer_t buffer = ggml_backend_memory_lease_buffer(lease.get());

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 1, 48, 16, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) == buffer);

    context.view_fail_after = context.view_attempts;
    GGML_ASSERT(ggml_backend_memory_arena_begin(
        arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 2, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) == buffer);
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);

    context.view_fail_after = SIZE_MAX;
    GGML_ASSERT(ggml_backend_memory_arena_begin(
        arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 2, 16, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 2);
    GGML_ASSERT(ggml_backend_memory_arena_get_buffer(arena.get(), 1) == buffer);
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_lease_generation(lease.get()) == 1);
}

static void test_lease_validation() {
    GGML_ASSERT(ggml_backend_memory_arena_retain(nullptr) == nullptr);
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(nullptr) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_acquire(nullptr, 1) == nullptr);
    GGML_ASSERT(ggml_backend_memory_lease_retain(nullptr) == nullptr);
    ggml_backend_memory_lease_free(nullptr);
    GGML_ASSERT(ggml_backend_memory_lease_buffer(nullptr) == nullptr);
    GGML_ASSERT(ggml_backend_memory_lease_generation(nullptr) == 0);
    ggml_backend_memory_region region = {};
    GGML_ASSERT(!ggml_backend_memory_lease_get_region(nullptr, &region));
}

// Close the lease gate, drain existing users, and reopen after a transaction.
static void test_arena_state_lifecycle() {
    GGML_ASSERT(ggml_backend_memory_arena_get_state(nullptr) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_INVALID);
    GGML_ASSERT(!ggml_backend_memory_arena_quiesce(nullptr));
    GGML_ASSERT(!ggml_backend_memory_arena_resume(nullptr));

    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));

    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(lease);
    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);
    GGML_ASSERT(ggml_backend_memory_arena_acquire(arena.get(), 1) == nullptr);

    GGML_ASSERT(ggml_backend_memory_arena_begin(
        arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(!ggml_backend_memory_arena_resume(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);

    GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
    lease_ptr concurrent(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(concurrent);
    concurrent.reset();
    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);

    lease.reset();
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_begin(
        arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
    GGML_ASSERT(!ggml_backend_memory_arena_resume(arena.get()));
    ggml_backend_memory_arena_rollback(arena.get());
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
    GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_OPEN);
    lease_ptr resumed(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(resumed);
}

static bool wait_for_count(const std::atomic<size_t> & count, size_t expected) {
    for (size_t i = 0; i < 1 << 20; ++i) {
        if (count.load(std::memory_order_relaxed) >= expected) {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

// Serialize concurrent lease acquisition with closing the lease gate.
static void test_arena_concurrent_lease_gate() {
    test_buft_context context;
    auto buft = make_test_buft(&context);
    arena_ptr arena(ggml_backend_memory_arena_new(&buft, 128), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 32, 16, GGML_BACKEND_MEMORY_REGION_PERSISTENT, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));

    std::atomic<bool> gate_closed {false};
    std::atomic<bool> stop {false};
    std::atomic<size_t> open_successes {0};
    std::atomic<size_t> closed_attempts {0};
    std::atomic<size_t> closed_successes {0};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < 4; ++i) {
        workers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                const bool after_close = gate_closed.load(std::memory_order_acquire);
                ggml_backend_memory_lease_t lease = ggml_backend_memory_arena_acquire(arena.get(), 1);
                if (after_close) {
                    closed_attempts.fetch_add(1, std::memory_order_relaxed);
                    if (lease != nullptr) {
                        closed_successes.fetch_add(1, std::memory_order_relaxed);
                    }
                } else if (lease != nullptr) {
                    open_successes.fetch_add(1, std::memory_order_relaxed);
                }
                ggml_backend_memory_lease_free(lease);
            }
        });
    }

    GGML_ASSERT(wait_for_count(open_successes, 256));
    for (size_t i = 0; i < 8; ++i) {
        GGML_ASSERT(ggml_backend_memory_arena_begin(
            arena.get(), GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    }
    GGML_ASSERT(ggml_backend_memory_arena_generation(arena.get()) == 9);
    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
    gate_closed.store(true, std::memory_order_release);
    GGML_ASSERT(wait_for_count(closed_attempts, 256));
    stop.store(true, std::memory_order_relaxed);
    for (auto & worker : workers) {
        worker.join();
    }

    GGML_ASSERT(open_successes.load(std::memory_order_relaxed) >= 256);
    GGML_ASSERT(closed_successes.load(std::memory_order_relaxed) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
}

// Keep a workspace leased while the scheduler can access it and release it on detach or destruction.
static void test_scheduler_lease_attachment() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t workspace_size = 16*alignment;
    arena_ptr arena(ggml_backend_memory_arena_new(buft, workspace_size + alignment),
                    ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 1, 0, workspace_size, alignment, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));

    ggml_backend_t backends[] = {backend.get()};
    ggml_backend_buffer_type_t bufts[] = {buft};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
        backends, bufts, 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
    GGML_ASSERT(sched);
    GGML_ASSERT(!ggml_backend_sched_attach_memory_lease(sched.get(), backend.get(), nullptr));

    test_buft_context incompatible_context;
    auto incompatible_buft = make_test_buft(&incompatible_context);
    arena_ptr incompatible(ggml_backend_memory_arena_new(&incompatible_buft, 128),
                           ggml_backend_memory_arena_free);
    GGML_ASSERT(incompatible);
    GGML_ASSERT(ggml_backend_memory_arena_begin(incompatible.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        incompatible.get(), 9, 64, 16, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(incompatible.get()));
    lease_ptr incompatible_lease(
        ggml_backend_memory_arena_acquire(incompatible.get(), 9), ggml_backend_memory_lease_free);
    GGML_ASSERT(incompatible_lease);
    GGML_ASSERT(!ggml_backend_sched_attach_memory_lease(
        sched.get(), backend.get(), incompatible_lease.get()));
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(incompatible.get()) == 1);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == 0);

    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(lease);
    GGML_ASSERT(ggml_backend_sched_attach_memory_lease(sched.get(), backend.get(), lease.get()));
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == workspace_size);
    lease.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);

    lease_ptr duplicate(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(duplicate);
    GGML_ASSERT(!ggml_backend_sched_attach_memory_lease(sched.get(), backend.get(), duplicate.get()));
    duplicate.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));

    ggml_init_params params = {
        /*.mem_size   = */ 8*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    {
        ggml_context_ptr ctx(ggml_init(params));
        auto * graph = ggml_new_graph(ctx.get());
        auto * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        auto * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        auto * sum = ggml_add(ctx.get(), lhs, rhs);
        ggml_set_input(lhs);
        ggml_set_input(rhs);
        ggml_set_output(sum);
        ggml_build_forward_expand(graph, sum);
        GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), graph));
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

        const float a[] = {1, 2, 3, 4};
        const float b[] = {5, 6, 7, 8};
        ggml_backend_tensor_set(lhs, a, 0, sizeof(a));
        ggml_backend_tensor_set(rhs, b, 0, sizeof(b));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_sched_detach_memory_lease(sched.get(), backend.get()));
        GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
        GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                    GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
        GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == 0);
        GGML_ASSERT(!ggml_backend_sched_detach_memory_lease(sched.get(), backend.get()));

        float result[4] = {};
        ggml_backend_tensor_get(sum, result, 0, sizeof(result));
        for (size_t i = 0; i < 4; ++i) {
            GGML_ASSERT(result[i] == a[i] + b[i]);
        }
    }

    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(
        arena.get(), 2, alignment, workspace_size, alignment, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
    lease_ptr replacement(ggml_backend_memory_arena_acquire(arena.get(), 2), ggml_backend_memory_lease_free);
    GGML_ASSERT(replacement);
    GGML_ASSERT(ggml_backend_sched_attach_memory_lease(sched.get(), backend.get(), replacement.get()));
    replacement.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    {
        ggml_context_ptr ctx(ggml_init(params));
        auto * graph = ggml_new_graph(ctx.get());
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        auto * output = ggml_scale(ctx.get(), input, 3.0f);
        ggml_set_input(input);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), graph));
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

        const float values[] = {2, 4, 6, 8};
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
        sched.reset();
        GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
        GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                    GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);

        float result[4] = {};
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (size_t i = 0; i < 4; ++i) {
            GGML_ASSERT(result[i] == 3*values[i]);
        }
    }
}

// Share one lease object across aliased scheduler slots and detach it through either backend.
static void test_scheduler_shared_lease_attachment() {
    ggml_backend_ptr first_backend(ggml_backend_cpu_init());
    ggml_backend_ptr second_backend(ggml_backend_cpu_init());
    GGML_ASSERT(first_backend && second_backend);
    auto * buft = ggml_backend_get_default_buffer_type(first_backend.get());
    GGML_ASSERT(ggml_backend_get_default_buffer_type(second_backend.get()) == buft);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    arena_ptr arena(ggml_backend_memory_arena_new(buft, 8*alignment),
                    ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
    GGML_ASSERT(ggml_backend_memory_arena_reserve(
        arena.get(), 1, 8*alignment, alignment, GGML_BACKEND_MEMORY_REGION_NONE, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));

    ggml_backend_t backends[] = {first_backend.get(), second_backend.get()};
    ggml_backend_buffer_type_t bufts[] = {buft, buft};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
        backends, bufts, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
    GGML_ASSERT(lease);
    GGML_ASSERT(ggml_backend_sched_attach_memory_lease(
        sched.get(), first_backend.get(), lease.get()));
    lease.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), first_backend.get()) == 8*alignment);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), second_backend.get()) == 0);

    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
    GGML_ASSERT(ggml_backend_sched_clear_buffer_range(
        sched.get(), second_backend.get()));
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), first_backend.get()) == 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), second_backend.get()) == 0);
    GGML_ASSERT(!ggml_backend_sched_detach_memory_lease(
        sched.get(), first_backend.get()));

    GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
    lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 1));
    GGML_ASSERT(lease);
    GGML_ASSERT(ggml_backend_sched_attach_memory_lease(
        sched.get(), second_backend.get(), lease.get()));
    lease.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 1);
    GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
    sched.reset();
    GGML_ASSERT(ggml_backend_memory_arena_lease_count(arena.get()) == 0);
    GGML_ASSERT(ggml_backend_memory_arena_get_state(arena.get()) ==
                GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
}

static void run(const char * name, void (*test)()) {
    std::printf("%s ", name);
    std::fflush(stdout);
    test();
    std::printf("PASSED\n");
}

int main() {
    run("test_workspace_group_planning", test_workspace_group_planning);
    run("test_workspace_group_planning_boundaries", test_workspace_group_planning_boundaries);
    run("test_compute_arena_binding_ownership", test_compute_arena_binding_ownership);
    run("test_lifecycle", test_lifecycle);
    run("test_placement", test_placement);
    run("test_failure_atomicity", test_failure_atomicity);
    run("test_persistent_transition", test_persistent_transition);
    run("test_boundaries_and_overflow", test_boundaries_and_overflow);
    run("test_mixed_request_properties", test_mixed_request_properties);
    run("test_arena_imported_parent", test_arena_imported_parent);
    run("test_arena_owned_parent", test_arena_owned_parent);
    run("test_arena_construction_failure", test_arena_construction_failure);
    run("test_arena_commit_atomicity", test_arena_commit_atomicity);
    run("test_arena_persistent_view_identity", test_arena_persistent_view_identity);
    run("test_arena_real_backends", test_arena_real_backends);
    run("test_arena_reference_lifetime", test_arena_reference_lifetime);
    run("test_lease_lifetime", test_lease_lifetime);
    run("test_multiple_leases", test_multiple_leases);
    run("test_lease_blocks_replacement", test_lease_blocks_replacement);
    run("test_persistent_lease_transition", test_persistent_lease_transition);
    run("test_lease_validation", test_lease_validation);
    run("test_arena_state_lifecycle", test_arena_state_lifecycle);
    run("test_arena_concurrent_lease_gate", test_arena_concurrent_lease_gate);
    run("test_scheduler_lease_attachment", test_scheduler_lease_attachment);
    run("test_scheduler_shared_lease_attachment", test_scheduler_shared_lease_attachment);
    return 0;
}
