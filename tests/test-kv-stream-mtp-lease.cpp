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
    size_t committed_tokens = 3*256 - 7;

    explicit mtp_fixture(bool cuda = false, size_t tokens = 3*256 - 7, size_t committed = 0) :
        active_tokens(tokens), committed_tokens(committed ? committed : tokens) {
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

        llama_kv_stream_host_config config{202, policy.shape, policy.capabilities,
            (active_tokens + 255)/256*256, 1};
        auto host = llama_kv_stream_host::create(config, host_type);
        GGML_ASSERT(host);
        cache = llama_kv_stream_logical_cache::create(host);
        GGML_ASSERT(cache);
        llama_kv_stream_host_layer planes;
        GGML_ASSERT(host->layer(0, planes));
        auto * k = static_cast<uint8_t *>(planes.k);
        auto * v = static_cast<uint8_t *>(planes.v);
        for (size_t token = 0; token < committed_tokens; ++token) {
            std::memset(k + token*host->layout().k_token_bytes, int(token % 251), host->layout().k_token_bytes);
            std::memset(v + token*host->layout().v_token_bytes, int((token + 7) % 251), host->layout().v_token_bytes);
        }
        GGML_ASSERT(cache->restore({202, 1, committed_tokens}));

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
struct async_copy_fault {
    ggml_backend_t backend;
    decltype(ggml_backend_i::cpy_tensor_async) original;
    int calls = 0;
    int fail_on;
    inline static async_copy_fault * active = nullptr;

    async_copy_fault(ggml_backend_t backend, int fail_on) :
        backend(backend), original(backend->iface.cpy_tensor_async), fail_on(fail_on) {
        GGML_ASSERT(!active);
        active = this;
        backend->iface.cpy_tensor_async = [](ggml_backend_t src, ggml_backend_t dst,
                const ggml_tensor * from, ggml_tensor * to) {
            ++active->calls;
            if (active->calls == active->fail_on) throw std::runtime_error("injected D2D failure");
            return active->original ? active->original(src, dst, from, to) : false;
        };
    }
    ~async_copy_fault() {
        backend->iface.cpy_tensor_async = original;
        active = nullptr;
    }
};

int main(int argc, char ** argv) {
    const bool tail_only = argc > 1 && std::strcmp(argv[1], "--cuda-tail") == 0;
    const bool cuda = tail_only || (argc > 1 && std::strcmp(argv[1], "--cuda") == 0);
    testing t;
    if (tail_only) t.set_filter("retained_mtp_tail_publishes_only_new_rows_and_invalidates_old_plans");

    t.test("resident_mirror_reuse_never_claims_overwritten_ring_bytes", [&](testing & t) {
        mtp_fixture f(cuda);
        const size_t stride=f.cache->host()->layout().k_token_bytes+f.cache->host()->layout().v_token_bytes;
        auto owner=llama_kv_stream_layer_lease_owner::create(f.pool,f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto * lease=owner->acquire_populated(f.backend.get(),f.request(),*f.cache,true);
        if (!t.assert_true(lease != nullptr)) return;
        t.assert_equal(f.committed_tokens*stride,llama_kv_stream_complete_layer_lease_population(lease).bytes);
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(llama_kv_stream_complete_layer_lease_plan(lease),view))) return;
        t.assert_equal(size_t(2),view.count);
        const size_t resident=view.spans[0].tokens;
        const auto ring=view.spans[1];
        llama_kv_stream_complete_layer_lease_free(lease);
        t.assert_equal(size_t(0),owner->ring_slots_used());
        owner.reset();
        ggml_context_ptr context(ggml_init({4096,nullptr,true}));
        for (bool value : {false,true}) {
            const size_t bytes=ring.tokens*(value ? f.cache->host()->layout().v_token_bytes : f.cache->host()->layout().k_token_bytes);
            auto * tensor=ggml_new_tensor_1d(context.get(),GGML_TYPE_I8,bytes);
            auto * buffer=value ? ring.v_buffer : ring.k_buffer;
            auto * address=static_cast<char *>(ggml_backend_buffer_get_base(buffer))+(value ? ring.v_offset : ring.k_offset);
            t.assert_true(ggml_backend_tensor_alloc(buffer,tensor,address) == GGML_STATUS_SUCCESS);
            std::vector<uint8_t> overwrite(bytes,0xa5);
            ggml_backend_tensor_set(tensor,overwrite.data(),0,bytes);
        }
        owner=llama_kv_stream_layer_lease_owner::create(f.pool,f.layout());
        lease=owner->acquire_populated(f.backend.get(),f.request(),*f.cache,true);
        if (!t.assert_true(lease != nullptr)) return;
        const auto stats=llama_kv_stream_complete_layer_lease_population(lease);
        t.assert_equal((f.committed_tokens-resident)*stride,stats.bytes);
        t.assert_equal(size_t(2),stats.calls);
        llama_kv_stream_host_layer host;
        t.assert_true(f.cache->host()->layer(0,host));
        t.assert_true(matches(ring.k_buffer,ring.k_offset,static_cast<const char *>(host.k)+resident*f.cache->host()->layout().k_token_bytes,
            ring.tokens*f.cache->host()->layout().k_token_bytes));
        t.assert_true(matches(ring.v_buffer,ring.v_offset,static_cast<const char *>(host.v)+resident*f.cache->host()->layout().v_token_bytes,
            ring.tokens*f.cache->host()->layout().v_token_bytes));
        llama_kv_stream_complete_layer_lease_free(lease);
    });

    t.test("resident_population_copies_new_rows_and_retries_failed_refresh", [](testing & t) {
        mtp_fixture f(false,260,252);
        auto owner=llama_kv_stream_layer_lease_owner::create(f.pool,f.layout());
        auto request=f.request(1); request.active_tokens=f.cache->tokens();
        auto * lease=owner->acquire_populated(f.backend.get(),request,*f.cache,true);
        if (!t.assert_true(lease != nullptr)) return;
        llama_kv_stream_complete_layer_lease_free(lease);
        const auto & layout=f.cache->host()->layout();
        const size_t first=f.cache->tokens();
        std::vector<uint8_t> keys(2*layout.k_token_bytes,0x24),values(2*layout.v_token_bytes,0x35);
        llama_kv_stream_write write;
        if (!t.assert_true(f.cache->begin(2) && f.cache->content()->prepare({
                {0,ggml_kv_stream_operand::k,first*layout.k_token_bytes,keys.data(),keys.size()},
                {0,ggml_kv_stream_operand::v,first*layout.v_token_bytes,values.data(),values.size()}},write) &&
                f.cache->publish_host(write) && f.cache->finish())) return;
        owner=llama_kv_stream_layer_lease_owner::create(f.pool,f.layout());
        request=f.request(1); request.active_tokens=f.cache->tokens();
        {
            upload_fault fault(ggml_backend_memory_lease_buffer(f.pool),2);
            t.assert_true(owner->acquire_populated(f.backend.get(),request,*f.cache,true) == nullptr);
        }
        t.assert_equal(size_t(0),owner->ring_slots_used());
        lease=owner->acquire_populated(f.backend.get(),request,*f.cache,true);
        if (!t.assert_true(lease != nullptr)) return;
        t.assert_equal(2*(layout.k_token_bytes+layout.v_token_bytes),llama_kv_stream_complete_layer_lease_population(lease).bytes);
        t.assert_equal(size_t(2),llama_kv_stream_complete_layer_lease_population(lease).calls);
        llama_kv_stream_complete_layer_lease_free(lease);
        t.assert_true(f.cache->truncate(first+1));
        owner=llama_kv_stream_layer_lease_owner::create(f.pool,f.layout());
        request=f.request(1); request.active_tokens=f.cache->tokens();
        lease=owner->acquire_populated(f.backend.get(),request,*f.cache,true);
        if (!t.assert_true(lease != nullptr)) return;
        t.assert_equal(size_t(0),llama_kv_stream_complete_layer_lease_population(lease).bytes);
        llama_kv_stream_complete_layer_lease_free(lease);
    });
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
    t.test("lagging_mtp_populates_committed_prefix_but_reserves_future_ring_capacity", [&](testing & t) {
        mtp_fixture f(cuda, 3*256 + 2, 3*256 - 2);
        auto * pool = ggml_backend_memory_lease_buffer(f.pool);
        ggml_backend_buffer_clear(pool, 0xa5);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto request = f.request(4);
        request.active_tokens = f.cache->tokens();
        auto * lease = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(lease != nullptr)) return;
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                llama_kv_stream_complete_layer_lease_plan(lease), view))) return;
        t.assert_equal(f.cache->tokens(), view.active_tokens);
        t.assert_equal(size_t(3), owner->ring_slots_used());
        const auto stats = llama_kv_stream_complete_layer_lease_population(lease);
        t.assert_equal(f.cache->tokens()*(f.cache->host()->layout().k_token_bytes +
            f.cache->host()->layout().v_token_bytes), stats.bytes);
        llama_kv_stream_policy_layout physical;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                f.policy, f.state, f.active_tokens, physical).status == llama_kv_stream_policy_status::success)) return;
        const size_t ring_token = llama_kv_stream_complete_layer_lease_ring_first(lease)*256 +
            f.cache->tokens() - llama_kv_stream_complete_layer_lease_resident_tokens(lease);
        const size_t future = f.active_tokens - f.cache->tokens();
        const size_t k_bytes = future*f.cache->host()->layout().k_token_bytes;
        const size_t v_bytes = future*f.cache->host()->layout().v_token_bytes;
        std::vector<uint8_t> k_sentinel(k_bytes, 0xa5), v_sentinel(v_bytes, 0xa5);
        t.assert_true(matches(pool, ring_token*f.cache->host()->layout().k_token_bytes,
            k_sentinel.data(), k_bytes));
        t.assert_true(matches(pool, physical.ring.v_offset + ring_token*f.cache->host()->layout().v_token_bytes,
            v_sentinel.data(), v_bytes));
        llama_kv_stream_complete_layer_lease_free(lease);
        t.assert_equal(size_t(0), owner->ring_slots_used());
    });
    t.test("retained_mtp_tail_publishes_only_new_rows_and_invalidates_old_plans", [&](testing & t) {
        mtp_fixture f(cuda, 513, 512);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto request = f.request(1);
        request.active_tokens = 512;
        auto * old = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(old != nullptr)) return;
        const auto before = llama_kv_stream_complete_layer_lease_population(old);
        auto guard = owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        const uint64_t previous_generation = owner->content_generation();
        const auto & layout = f.cache->host()->layout();
        std::vector<uint8_t> key(layout.k_token_bytes, 0x73);
        std::vector<uint8_t> value(layout.v_token_bytes, 0x57);
        llama_kv_stream_write write;
        if (!t.assert_true(f.cache->begin(1) && f.cache->content()->prepare({
                {0, ggml_kv_stream_operand::k, 512*layout.k_token_bytes, key.data(), key.size()},
                {0, ggml_kv_stream_operand::v, 512*layout.v_token_bytes, value.data(), value.size()}}, write) &&
                f.cache->publish_host(write) && f.cache->finish())) return;
        if (!cuda) {
            upload_fault fault(ggml_backend_memory_lease_buffer(f.pool));
            llama_kv_stream_population_stats failed;
            t.assert_true(!owner->publish_tail(f.backend.get(), old, *f.cache, failed));
            t.assert_equal(previous_generation, owner->content_generation());
            t.assert_true(llama_kv_stream_complete_layer_lease_plan(old) != nullptr);
        }
        llama_kv_stream_population_stats delta;
        if (!t.assert_true(owner->publish_tail(f.backend.get(), old, *f.cache, delta))) return;
        t.assert_equal(layout.k_token_bytes + layout.v_token_bytes, delta.bytes);
        t.assert_equal(size_t(2), delta.calls);
        t.assert_equal(f.cache->identity().generation, owner->content_generation());
        t.assert_true(llama_kv_stream_complete_layer_lease_plan(old) == nullptr);
        request.active_tokens = 513;
        request.content_generation = f.cache->identity().generation;
        request.query_tokens = 4;
        auto * renewed = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(renewed != nullptr)) return;
        const auto total = llama_kv_stream_complete_layer_lease_population(renewed);
        t.assert_equal(before.bytes + delta.bytes, total.bytes);
        t.assert_equal(before.calls + delta.calls, total.calls);
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                llama_kv_stream_complete_layer_lease_plan(renewed), view))) return;
        llama_kv_stream_host_layer source;
        if (!t.assert_true(f.cache->host()->layer(0, source))) return;
        for (size_t i = 0; i < view.count; ++i) {
            const auto & span = view.spans[i];
            t.assert_true(matches(span.k_buffer, span.k_offset,
                static_cast<const uint8_t *>(source.k) + span.token_begin*layout.k_token_bytes,
                span.tokens*layout.k_token_bytes));
            t.assert_true(matches(span.v_buffer, span.v_offset,
                static_cast<const uint8_t *>(source.v) + span.token_begin*layout.v_token_bytes,
                span.tokens*layout.v_token_bytes));
        }
        guard.reset();
        llama_kv_stream_complete_layer_lease_free(old);
        llama_kv_stream_complete_layer_lease_free(renewed);
        t.assert_true(owner->can_repartition());
    });
    t.test("staged_mtp_copy_failure_does_not_publish_a_frontier", [](testing & t) {
        mtp_fixture f(false, 513, 512);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto request = f.request(1);
        request.active_tokens = 512;
        auto * old = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(old != nullptr)) return;
        ggml_context_ptr context(ggml_init({8192, nullptr, true}));
        if (!t.assert_true(bool(context))) return;
        auto * k = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, 64, 1);
        auto * v = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q4_0, 64, 1);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(context.get(), f.backend.get()));
        if (!t.assert_true(k && v && bool(source))) return;
        std::vector<uint8_t> key(f.cache->host()->layout().k_token_bytes, 0x73);
        std::vector<uint8_t> value(f.cache->host()->layout().v_token_bytes, 0x57);
        ggml_backend_tensor_set(k, key.data(), 0, key.size());
        ggml_backend_tensor_set(v, value.data(), 0, value.size());
        llama_kv_stream_population_stats staged;
        {
            async_copy_fault fault(f.backend.get(), 1);
            t.assert_true(!owner->stage_tail_async(f.backend.get(), old, 512, false, k, 0, 1, staged));
            t.assert_equal(1, fault.calls);
        }
        t.assert_equal(size_t(0), staged.bytes);
        {
            async_copy_fault fault(f.backend.get(), 2);
            t.assert_true(owner->stage_tail_async(f.backend.get(), old, 512, false, k, 0, 1, staged));
            t.assert_true(!owner->stage_tail_async(f.backend.get(), old, 512, true, v, 0, 1, staged));
            t.assert_equal(2, fault.calls);
        }
        t.assert_equal(key.size(), staged.bytes);
        t.assert_equal(size_t(1), staged.calls);
        t.assert_equal(size_t(512), f.cache->tokens());
        t.assert_equal(size_t(512), f.cache->frontiers().host);
        t.assert_true(llama_kv_stream_complete_layer_lease_plan(old) != nullptr);
        llama_kv_stream_complete_layer_lease_free(old);
    });
    t.test("staged_mtp_tail_requires_completed_host_publication", [&](testing & t) {
        mtp_fixture f(false, 514, 512);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto request = f.request(1);
        request.active_tokens = 512;
        auto * old = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(old != nullptr)) return;
        auto guard = owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        const auto & layout = f.cache->host()->layout();
        ggml_context_ptr source_context(ggml_init({8192, nullptr, true}));
        if (!t.assert_true(bool(source_context))) return;
        auto * k = ggml_new_tensor_2d(source_context.get(), GGML_TYPE_Q8_0, 64, 1);
        auto * v = ggml_new_tensor_2d(source_context.get(), GGML_TYPE_Q4_0, 64, 1);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(source_context.get(), f.backend.get()));
        if (!t.assert_true(k && v && bool(source))) return;
        std::vector<uint8_t> key(layout.k_token_bytes, 0x73), value(layout.v_token_bytes, 0x57);
        ggml_backend_tensor_set(k, key.data(), 0, key.size());
        ggml_backend_tensor_set(v, value.data(), 0, value.size());
        llama_kv_stream_population_stats staged;
        t.assert_true(!owner->stage_tail_async(f.backend.get(), old, 511, false, k, 0, 1, staged));
        t.assert_true(!owner->stage_tail_async(f.backend.get(), old, 512, false, k, 2, 1, staged));
        if (!t.assert_true(owner->stage_tail_async(f.backend.get(), old, 512, false, k, 0, 1, staged) &&
                owner->stage_tail_async(f.backend.get(), old, 512, true, v, 0, 1, staged))) return;
        t.assert_equal(key.size() + value.size(), staged.bytes);
        t.assert_equal(size_t(2), staged.calls);
        t.assert_equal(size_t(512), f.cache->tokens());
        t.assert_equal(size_t(512), f.cache->frontiers().host);
        t.assert_true(!owner->adopt_staged_tail(old, *f.cache, staged));
        request.active_tokens = 513;
        auto * provisional = owner->acquire(request);
        if (!t.assert_true(provisional != nullptr)) return;
        ggml_backend_synchronize(f.backend.get());
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                llama_kv_stream_complete_layer_lease_plan(provisional), view))) return;
        const auto & tail = view.spans[view.count - 1];
        if (!t.assert_true(tail.token_begin <= 512 && 512 < tail.token_begin + tail.tokens)) return;
        t.assert_true(matches(tail.k_buffer,
            tail.k_offset + (512 - tail.token_begin)*layout.k_token_bytes, key.data(), key.size()));
        t.assert_true(matches(tail.v_buffer,
            tail.v_offset + (512 - tail.token_begin)*layout.v_token_bytes, value.data(), value.size()));
        llama_kv_stream_write write;
        if (!t.assert_true(f.cache->begin(1) && f.cache->content()->prepare({
                {0, ggml_kv_stream_operand::k, 512*layout.k_token_bytes, key.data(), key.size()},
                {0, ggml_kv_stream_operand::v, 512*layout.v_token_bytes, value.data(), value.size()}}, write) &&
                f.cache->publish_host(write) && f.cache->finish())) return;
        if (!t.assert_true(owner->adopt_staged_tail(old, *f.cache, staged))) return;
        t.assert_true(llama_kv_stream_complete_layer_lease_plan(old) == nullptr);
        t.assert_true(llama_kv_stream_complete_layer_lease_plan(provisional) == nullptr);
        request.content_generation = f.cache->identity().generation;
        auto * renewed = owner->acquire(request);
        if (!t.assert_true(renewed != nullptr)) return;
        t.assert_true(llama_kv_stream_complete_layer_lease_plan(renewed) != nullptr);
        staged = {};
        if (!t.assert_true(owner->stage_tail_async(f.backend.get(), renewed, 513, false, k, 0, 1, staged) &&
                owner->stage_tail_async(f.backend.get(), renewed, 513, true, v, 0, 1, staged))) return;
        ggml_backend_synchronize(f.backend.get());
        if (!t.assert_true(f.cache->begin(1) && f.cache->cancel())) return;
        t.assert_equal(size_t(513), f.cache->tokens());
        t.assert_true(!owner->adopt_staged_tail(renewed, *f.cache, staged));
        request.active_tokens = 514;
        request.content_generation = f.cache->identity().generation;
        t.assert_true(owner->acquire(request) == nullptr);
        llama_kv_stream_complete_layer_lease_free(renewed);
        llama_kv_stream_complete_layer_lease_free(provisional);
        llama_kv_stream_complete_layer_lease_free(old);
    });
    t.test("truncated_mtp_prefix_keeps_physical_reservation_without_reupload", [&](testing & t) {
        mtp_fixture f(cuda, 513, 513);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool, f.layout());
        if (!t.assert_true(bool(owner))) return;
        auto request = f.request(4);
        auto * original = owner->acquire_populated(f.backend.get(), request, *f.cache);
        if (!t.assert_true(original != nullptr)) return;
        auto guard = owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        const auto before = llama_kv_stream_complete_layer_lease_population(original);
        const size_t used = owner->ring_slots_used();
        if (!t.assert_true(f.cache->truncate(510))) return;
        t.assert_true(owner->adopt_truncated_prefix(original, *f.cache));
        t.assert_true(llama_kv_stream_complete_layer_lease_plan(original) == nullptr);
        request.active_tokens = 510;
        request.content_generation = f.cache->identity().generation;
        auto * retained = owner->acquire(request);
        if (!t.assert_true(retained != nullptr)) return;
        t.assert_equal(used, owner->ring_slots_used());
        const auto after = llama_kv_stream_complete_layer_lease_population(retained);
        t.assert_equal(before.bytes, after.bytes);
        t.assert_equal(before.calls, after.calls);
        ggml_kv_stream_span_plan_view view;
        if (t.assert_true(ggml_kv_stream_span_plan_get_view(
                llama_kv_stream_complete_layer_lease_plan(retained), view)))
            t.assert_equal(size_t(510), view.active_tokens);
        guard.reset();
        llama_kv_stream_complete_layer_lease_free(original);
        llama_kv_stream_complete_layer_lease_free(retained);
        t.assert_true(owner->can_repartition());
    });
    t.test("future_mtp_page_uses_current_physical_layout", [&](testing & t) {
        mtp_fixture f(cuda, 3074, 3069);
        f.state.decode_active_pages = 12;
        llama_kv_stream_policy_layout current;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                f.policy, f.state, 3071, current).status == llama_kv_stream_policy_status::success)) return;
        llama_kv_stream_policy_layout future;
        t.assert_true(llama_kv_stream_policy_layout_make(
            f.policy, f.state, 3074, future).status ==
            llama_kv_stream_policy_status::invalid_observation);
        auto owner = llama_kv_stream_layer_lease_owner::create(f.pool,
            {f.policy, f.state, 3074, 11, f.cache->identity().generation, 3071});
        if (!t.assert_true(bool(owner))) return;
        auto guard = owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        t.assert_equal(current.ring.bytes, guard->ring_layout().bytes);
        t.assert_equal(size_t(f.state.ring_slots), guard->blocked_slots().size());
        guard.reset();
    });
    return t.summary();
}
