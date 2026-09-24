#include "../src/llama-kv-stream-layer-lease.h"
#include "../src/llama-kv-stream-binding.h"
#include "../src/llama-kv-stream-logical-cache.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"

#include <cstring>
#include <functional>
#include <vector>
#include <utility>
#include <stdexcept>

struct mtp_fixture {
    ggml_backend_ptr backend;
    llama_kv_stream_policy_config policy;
    llama_kv_stream_policy_state state;
    std::unique_ptr<llama_kv_stream_logical_cache> cache;
    ggml_backend_memory_arena_t arena = nullptr;
    ggml_backend_memory_lease_t pool = nullptr;
    size_t active_tokens = 3*256 - 7;

    explicit mtp_fixture(bool cuda = false, size_t tokens = 3*256 - 7) : active_tokens(tokens) {
        if (cuda) {
            ggml_backend_load_all();
            auto * device = ggml_backend_dev_by_name("CUDA0");
            GGML_ASSERT(device);
            backend.reset(ggml_backend_dev_init(device, nullptr));
        } else {
            backend.reset(ggml_backend_cpu_init());
        }
        GGML_ASSERT(backend);
        auto * device = ggml_backend_get_device(backend.get());
        auto * host_type = cuda ? llama_kv_stream_host_buffer_type(device) : ggml_backend_cpu_buffer_type();
        auto * device_type = cuda ? llama_kv_stream_device_buffer_type(device) : ggml_backend_cpu_buffer_type();
        GGML_ASSERT(host_type && device_type);
        policy.shape = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 32, 32, 2, 256, 128};
        policy.capabilities = {
            {GGML_TYPE_Q8_0, true, true, true, true},
            {GGML_TYPE_Q4_0, true, true, true, true},
            true, true,
        };
        ggml_kv_stream_execution page;
        GGML_ASSERT(ggml_kv_stream_resolve(policy.shape, policy.capabilities, 256, page).status ==
            ggml_kv_stream_status::success);
        policy.layers = 17;
        policy.caches = {{101, 16}, {202, 1}};
        policy.pool_bytes = 20*page.storage.bytes + page.conversion.bytes;
        policy.initial_ring_slots = 3;
        policy.fixed_ring = true;
        GGML_ASSERT(llama_kv_stream_policy_initialize(policy, state).status == llama_kv_stream_policy_status::success);
        GGML_ASSERT(state.resident_pages_per_layer == 1 && state.ring_slots == 3);

        llama_kv_stream_host_config config{202, policy.shape, policy.capabilities, 768, 1};
        auto host = llama_kv_stream_host::create(config, host_type);
        GGML_ASSERT(host);
        cache = llama_kv_stream_logical_cache::create(host);
        GGML_ASSERT(cache);
        llama_kv_stream_host_layer planes;
        GGML_ASSERT(host->layer(0, planes));
        auto * k = static_cast<uint8_t *>(planes.k);
        auto * v = static_cast<uint8_t *>(planes.v);
        for (size_t token = 0; token < active_tokens; ++token) {
            std::memset(k + token*host->layout().k_token_bytes, int(token % 251), host->layout().k_token_bytes);
            std::memset(v + token*host->layout().v_token_bytes, int((token + 7) % 251), host->layout().v_token_bytes);
        }
        GGML_ASSERT(cache->restore({202, 1, active_tokens}));

        arena = ggml_backend_memory_arena_new(device_type, policy.pool_bytes);
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena, GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena, 1, 0, policy.pool_bytes, 128, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena));
        pool = ggml_backend_memory_arena_acquire(arena, 1);
        GGML_ASSERT(pool);
    }

    ~mtp_fixture() {
        ggml_backend_memory_lease_free(pool);
        ggml_backend_memory_arena_free(arena);
    }

    llama_kv_stream_layer_lease_layout layout(uint64_t revision = 11) const {
        return {policy, state, active_tokens, revision, cache->identity().generation};
    }

    llama_kv_stream_complete_layer_request request(uint32_t queries = 4, uint64_t id = 202) const {
        return {16, queries, 11, cache->identity().generation, id};
    }
};

static bool matches(ggml_backend_buffer_t buffer, size_t offset, const void * expected, size_t bytes) {
    if (!buffer || !expected || !bytes || bytes > INT64_MAX) return false;
    ggml_context_ptr context(ggml_init({4096, nullptr, true}));
    if (!context) return false;
    auto * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(bytes));
    const size_t capacity = ggml_backend_buffer_get_size(buffer);
    if (!tensor || offset > capacity ||
            ggml_backend_buffer_get_alloc_size(buffer, tensor) > capacity - offset) return false;
    auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    if (ggml_backend_tensor_alloc(buffer, tensor, base + offset) != GGML_STATUS_SUCCESS) return false;
    std::vector<uint8_t> actual(bytes);
    ggml_backend_tensor_get(tensor, actual.data(), 0, bytes);
    return std::memcmp(actual.data(), expected, bytes) == 0;
}

struct upload_fault {
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::set_tensor) original;
    int calls = 0;
    int fail_on = 2;
    std::function<void()> after_copy;
    inline static upload_fault * active = nullptr;

    explicit upload_fault(ggml_backend_buffer_t buffer, int fail_on = 2,
            std::function<void()> after_copy = {}) :
        buffer(buffer), original(buffer->iface.set_tensor), fail_on(fail_on), after_copy(std::move(after_copy)) {
        GGML_ASSERT(!active);
        active = this;
        buffer->iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * tensor,
                const void * data, size_t offset, size_t size) {
            ++active->calls;
            if (active->fail_on && active->calls == active->fail_on) throw std::runtime_error("injected H2D failure");
            active->original(b, tensor, data, offset, size);
            if (active->after_copy) active->after_copy();
        };
    }
    ~upload_fault() {
        buffer->iface.set_tensor = original;
        active = nullptr;
    }
};

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
    testing t;
    t.test("mtp_layer_populates_once_and_retains_both_spans_across_query_widths", [&](testing & t) {
        mtp_fixture f(cuda);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * catchup = owner->acquire_populated(f.backend.get(), f.request(4), *f.cache);
        if (!t.assert_true(catchup != nullptr)) return;
        t.assert_equal(size_t(2), owner->ring_slots_used());
        t.assert_true(!owner->can_repartition());
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                llama_kv_stream_complete_layer_lease_plan(catchup), view))) return;
        t.assert_equal(size_t(2), view.count);
        t.assert_equal(size_t(256), view.spans[0].tokens);
        t.assert_equal(f.active_tokens - 256, view.spans[1].tokens);
        llama_kv_stream_host_layer host;
        if (!t.assert_true(f.cache->host()->layer(0, host))) return;
        for (size_t i = 0; i < view.count; ++i) {
            const auto & span = view.spans[i];
            const size_t k_bytes = span.tokens*f.cache->host()->layout().k_token_bytes;
            const size_t v_bytes = span.tokens*f.cache->host()->layout().v_token_bytes;
            t.assert_true(matches(span.k_buffer, span.k_offset,
                static_cast<const uint8_t *>(host.k) + span.token_begin*f.cache->host()->layout().k_token_bytes, k_bytes));
            t.assert_true(matches(span.v_buffer, span.v_offset,
                static_cast<const uint8_t *>(host.v) + span.token_begin*f.cache->host()->layout().v_token_bytes, v_bytes));
        }
        const auto first = llama_kv_stream_complete_layer_lease_population(catchup);
        t.assert_equal(f.active_tokens*(f.cache->host()->layout().k_token_bytes +
            f.cache->host()->layout().v_token_bytes), first.bytes);
        t.assert_equal(size_t(4), first.calls);

        llama_kv_stream_complete_layer_lease_t draft = nullptr;
        if (cuda) {
            draft = owner->acquire_populated(f.backend.get(), f.request(1), *f.cache);
        } else {
            upload_fault probe(ggml_backend_memory_lease_buffer(f.pool));
            draft = owner->acquire_populated(f.backend.get(), f.request(1), *f.cache);
            t.assert_equal(0, probe.calls);
        }
        if (!t.assert_true(draft != nullptr)) return;
        t.assert_equal(first.bytes, llama_kv_stream_complete_layer_lease_population(draft).bytes);
        t.assert_equal(first.calls, llama_kv_stream_complete_layer_lease_population(draft).calls);
        t.assert_equal(size_t(2), owner->ring_slots_used());
        auto target = f.request(1, 101);
        target.layer = 0;
        t.assert_true(owner->acquire(target) == nullptr);
        llama_kv_stream_complete_layer_lease_free(catchup);
        t.assert_equal(size_t(2), owner->ring_slots_used());
        llama_kv_stream_complete_layer_lease_free(draft);
        t.assert_equal(size_t(0), owner->ring_slots_used());
        t.assert_true(owner->can_repartition());
    });

    t.test("identity_and_copy_failure_release_the_ring_reservation", [](testing & t) {
        mtp_fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->acquire_populated(f.backend.get(), f.request(4, 101), *f.cache) == nullptr);
        t.assert_equal(size_t(0), owner->ring_slots_used());
        {
            upload_fault fault(ggml_backend_memory_lease_buffer(f.pool));
            t.assert_true(owner->acquire_populated(f.backend.get(), f.request(), *f.cache) == nullptr);
            t.assert_equal(2, fault.calls);
        }
        t.assert_equal(size_t(0), owner->ring_slots_used());
        t.assert_equal(size_t(0), owner->active_reservations());
        t.assert_true(owner->can_repartition());
        auto * retry = owner->acquire_populated(f.backend.get(), f.request(), *f.cache);
        if (!t.assert_true(retry != nullptr)) return;
        t.assert_true(owner->acquire_populated(nullptr, f.request(1), *f.cache) == nullptr);
        llama_kv_stream_complete_layer_lease_free(retry);
        t.assert_equal(size_t(0), owner->ring_slots_used());
        t.assert_true(f.cache->truncate(f.active_tokens - 1));
        t.assert_true(owner->acquire_populated(f.backend.get(), f.request(), *f.cache) == nullptr);
        t.assert_equal(size_t(0), owner->ring_slots_used());
    });
    t.test("retained_population_rejects_untracked_host_mutation", [](testing & t) {
        mtp_fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        const auto request = f.request();
        auto * current = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(current != nullptr)) return;
        t.assert_true(f.cache->content()->invalidate());
        t.assert_true(owner->acquire_populated(f.backend.get(), request, *f.cache) == nullptr);
        llama_kv_stream_complete_layer_lease_free(current);
        t.assert_equal(size_t(0), owner->ring_slots_used());
    });
    t.test("host_mutation_during_upload_prevents_publication", [](testing & t) {
        mtp_fixture f;
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        bool changed = false;
        {
            upload_fault fault(ggml_backend_memory_lease_buffer(f.pool), 0, [&] {
                if (!changed) changed = f.cache->content()->invalidate();
            });
            auto * lease = owner->acquire_populated(f.backend.get(), f.request(), *f.cache);
            t.assert_true(changed);
            t.assert_true(lease == nullptr);
            llama_kv_stream_complete_layer_lease_free(lease);
        }
        t.assert_equal(size_t(0), owner->ring_slots_used());
        t.assert_true(owner->can_repartition());
    });
    t.test("short_mtp_layer_uses_only_resident_bytes", [](testing & t) {
        mtp_fixture f(false, 128);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * lease = owner->acquire_populated(f.backend.get(), f.request(1), *f.cache);
        if (!t.assert_true(lease != nullptr)) return;
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                llama_kv_stream_complete_layer_lease_plan(lease), view))) return;
        t.assert_equal(size_t(1), view.count);
        t.assert_equal(size_t(128), view.spans[0].tokens);
        t.assert_equal(size_t(0), owner->ring_slots_used());
        const auto stats = llama_kv_stream_complete_layer_lease_population(lease);
        t.assert_equal(size_t(2), stats.calls);
        t.assert_equal(size_t(128)*(f.cache->host()->layout().k_token_bytes +
            f.cache->host()->layout().v_token_bytes), stats.bytes);
        llama_kv_stream_host_layer host;
        t.assert_true(f.cache->host()->layer(0, host));
        t.assert_true(matches(view.spans[0].k_buffer, view.spans[0].k_offset, host.k,
            size_t(128)*f.cache->host()->layout().k_token_bytes));
        t.assert_true(matches(view.spans[0].v_buffer, view.spans[0].v_offset, host.v,
            size_t(128)*f.cache->host()->layout().v_token_bytes));
        llama_kv_stream_complete_layer_lease_free(lease);
        t.assert_true(owner->can_repartition());
    });
    return t.summary();
}
