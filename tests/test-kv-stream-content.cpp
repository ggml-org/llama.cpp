#include "../src/llama-kv-stream-content.h"
#include "../src/llama-kv-stream-binding.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>
#include <stdexcept>

using operand = ggml_kv_stream_operand;
static std::shared_ptr<llama_kv_stream_host> storage(int k = GGML_TYPE_Q8_0, int v = GGML_TYPE_Q4_0,
        ggml_backend_buffer_type_t type = ggml_backend_cpu_buffer_type()) {
    llama_kv_stream_host_config c{17, {k, v, 32, 32, 1, 8, 128},
        {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, 17, 2};
    auto host = llama_kv_stream_host::create(c, type);
    GGML_ASSERT(host);
    return host;
}
static std::vector<llama_kv_stream_rows> all(const llama_kv_stream_host & host) {
    std::vector<llama_kv_stream_rows> rows;
    for (uint32_t layer = 0; layer < host.config().layers; ++layer)
        for (auto plane : {operand::k, operand::v}) rows.push_back({layer, plane, 0, host.layout().tokens});
    return rows;
}
static bool is_dirty(testing & t, const llama_kv_stream_content & content, uint32_t layer, operand plane, size_t row) {
    bool dirty = false;
    t.assert_true(content.dirty({layer, plane, row, 1}, dirty));
    return dirty;
}

int main(int argc, char ** argv) {
    testing t;
    t.test("initial_state_and_partial_mirror_ranges", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        t.assert_true(content.host() == host);
        t.assert_equal(uint64_t(1), content.generation());
        size_t calls = 0;
        t.assert_true(content.flush({{0, operand::k, 7, 3}}, [&](const auto & span) {
            ++calls;
            t.assert_equal(size_t(7), span.rows.first);
            t.assert_equal(size_t(3), span.rows.count);
            t.assert_equal(size_t(3)*host->layout().k_token_bytes, span.bytes);
            return true;
        }));
        t.assert_equal(size_t(1), calls);
        t.assert_true(!is_dirty(t, content, 0, operand::k, 8));
        t.assert_true(is_dirty(t, content, 0, operand::k, 6));
        t.assert_true(is_dirty(t, content, 0, operand::v, 8));
        t.assert_true(is_dirty(t, content, 1, operand::k, 8));
    });
    t.test("partial_encoded_rows_mark_only_intersecting_tokens", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        if (!t.assert_true(content.flush(all(*host), [](const auto &) { return true; }))) return;
        const auto stride = host->layout().k_token_bytes;
        uint8_t input[] = {3, 4, 5};
        llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({{0, operand::k, stride*8 - 1, input, 3}}, write))) return;
        input[0] = 99;
        t.assert_true(!is_dirty(t, content, 0, operand::k, 7));
        t.assert_true(content.commit(write) && !write.pending());
        t.assert_equal(uint64_t(2), content.generation());
        llama_kv_stream_host_layer layer; host->layer(0, layer);
        t.assert_equal(uint8_t(3), static_cast<uint8_t *>(layer.k)[stride*8 - 1]);
        for (size_t row = 0; row < host->layout().tokens; ++row) {
            t.assert_equal(row == 7 || row == 8, is_dirty(t, content, 0, operand::k, row));
            t.assert_true(!is_dirty(t, content, 0, operand::v, row));
        }
    });
    t.test("cancellation_and_stale_or_foreign_tickets", [](testing & t) {
        auto host = storage(); llama_kv_stream_content a(host), b(storage());
        uint8_t value = 7;
        llama_kv_stream_write first, second;
        if (!t.assert_true(a.prepare({{0, operand::v, 0, &value, 1}}, first))) return;
        t.assert_true(a.prepare({{0, operand::k, 0, &value, 1}}, second));
        t.assert_true(!b.commit(first) && first.pending());
        t.assert_true(a.commit(first));
        t.assert_true(!a.commit(second));
        second.cancel();
        t.assert_true(!second.pending() && !a.commit(second));
        llama_kv_stream_host_layer layer; host->layer(0, layer);
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(layer.k));
        t.assert_equal(uint8_t(7), *static_cast<uint8_t *>(layer.v));
    });
    t.test("invalid_batch_is_atomic_and_preserves_pending_output", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        uint8_t value = 8;
        llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({{0, operand::k, 0, &value, 1}}, write))) return;
        for (auto bad : std::vector<llama_kv_stream_write_span>{{9, operand::k, 0, &value, 1}, {0, operand::none, 0, &value, 1},
                {0, operand::v, host->layout().v_bytes, &value, 1}, {0, operand::k, SIZE_MAX, &value, 1}, {0, operand::k, 0, nullptr, 1}}) {
            t.assert_true(!content.prepare({{0, operand::v, 0, &value, 1}, bad}, write));
            t.assert_equal(uint64_t(1), content.generation());
        }
        t.assert_true(content.commit(write));
        llama_kv_stream_host_layer layer; host->layer(0, layer);
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(layer.v));
    });
    t.test("overlapping_writes_snapshot_sources_and_commit_in_order", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        llama_kv_stream_host_layer layer; host->layer(0, layer);
        uint8_t values[] = {1, 2, 3, 4};
        llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({{0, operand::k, 0, values, 4}, {0, operand::k, 1, layer.k, 2}}, write))) return;
        t.assert_true(content.commit(write));
        uint8_t expected[] = {1, 0, 0, 4};
        t.assert_true(std::memcmp(layer.k, expected, 4) == 0);
    });
    t.test("replacement_invalidates_same_id_without_an_arena_change", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        if (!t.assert_true(content.flush(all(*host), [](const auto &) { return true; }))) return;
        uint8_t value = 3; llama_kv_stream_write write;
        t.assert_true(content.prepare({{0, operand::k, 0, &value, 1}}, write));
        auto replacement = storage(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
        t.assert_true(content.replace(replacement));
        t.assert_true(content.host()->cache_id() == host->cache_id());
        t.assert_equal(uint64_t(2), content.generation());
        t.assert_true(!content.commit(write));
        t.assert_true(is_dirty(t, content, 0, operand::k, 0));
        t.assert_true(!content.replace(nullptr));
    });
    t.test("failed_or_throwing_flush_keeps_all_dirty_rows", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        size_t calls = 0;
        t.assert_true(!content.flush(all(*host), [&](const auto &) { return ++calls != 2; }));
        t.assert_equal(size_t(2), calls);
        t.assert_true(is_dirty(t, content, 0, operand::k, 0));
        bool caught = false;
        try { content.flush(all(*host), [](const auto &) -> bool { throw std::runtime_error("copy"); }); }
        catch (const std::runtime_error &) { caught = true; }
        t.assert_true(caught);
        t.assert_true(content.flush(all(*host), [](const auto &) { return true; }));
        t.assert_true(!is_dirty(t, content, 0, operand::k, 0));
    });
    t.test("mutation_and_reentrant_copy_are_blocked_during_flush", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        uint8_t value = 2; llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({{0, operand::k, 0, &value, 1}}, write))) return;
        t.assert_true(content.flush({{0, operand::k, 0, 1}}, [&](const auto &) {
            t.assert_true(!content.commit(write));
            t.assert_true(!content.replace(storage()));
            t.assert_true(!content.invalidate() && !content.reset_mirror());
            t.assert_true(!content.flush({}, {}));
            return true;
        }));
        t.assert_true(content.commit(write));
    });
    t.test("mirror_reset_does_not_cancel_pending_host_writes", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        uint8_t value = 1; llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({{0, operand::k, 0, &value, 1}}, write))) return;
        const auto epoch = content.mirror_epoch();
        t.assert_true(content.reset_mirror());
        t.assert_equal(epoch + 1, content.mirror_epoch());
        t.assert_equal(uint64_t(1), content.generation());
        t.assert_true(content.commit(write));
        t.assert_true(content.invalidate());
        t.assert_equal(uint64_t(3), content.generation());
    });
    t.test("empty_ranges_and_invalid_copy_requests", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({}, write))) return;
        t.assert_true(content.commit(write));
        t.assert_equal(uint64_t(1), content.generation());
        t.assert_true(content.flush({}, {}));
        size_t calls = 0;
        t.assert_true(!content.flush({{0, operand::k, 0, 1}, {0, operand::v, SIZE_MAX, 1}}, [&](const auto &) { ++calls; return true; }));
        t.assert_equal(size_t(0), calls);
        bool unchanged = true;
        t.assert_true(!content.dirty({0, operand::none, 0, 1}, unchanged) && unchanged);
    });
    t.test("word_boundaries_match_independent_row_oracle", [](testing & t) {
        auto c = storage()->config(); c.context_tokens = 129;
        auto host = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(host != nullptr)) return;
        llama_kv_stream_content content(host);
        const size_t n = host->layout().tokens;
        std::vector<std::vector<bool>> expected(4, std::vector<bool>(n, true));
        for (size_t step = 0; step < 136; ++step) {
            uint32_t layer = uint32_t(step % 2);
            auto plane = step % 4 < 2 ? operand::k : operand::v;
            size_t which = size_t(layer)*2 + (plane == operand::v);
            const size_t stride = plane == operand::k ? host->layout().k_token_bytes : host->layout().v_token_bytes;
            const size_t offset = (step*37 % n)*stride + step % stride;
            const size_t bytes = std::min(size_t(3)*stride, n*stride - offset);
            std::vector<uint8_t> data(bytes, uint8_t(step));
            llama_kv_stream_write write;
            t.assert_true(content.prepare({{layer, plane, offset, data.data(), data.size()}}, write));
            t.assert_true(content.commit(write));
            for (size_t r = offset/stride; r <= (offset + bytes - 1)/stride; ++r) expected[which][r] = true;
            const size_t begin = step*11 % n, count = std::min(n - begin, size_t(1) + step % 80);
            bool copied = content.flush({{layer, plane, begin, count}}, [&](const auto & span) {
                t.assert_true(span.rows.first >= begin && span.rows.first + span.rows.count <= begin + count);
                t.assert_equal(span.rows.count*stride, span.bytes);
                for (size_t r = span.rows.first; r < span.rows.first + span.rows.count; ++r) t.assert_true(expected[which][r]);
                llama_kv_stream_host_layer source; host->layer(layer, source);
                auto * base = static_cast<uint8_t *>(plane == operand::k ? source.k : source.v);
                t.assert_true(span.data == base + span.rows.first*stride);
                return step % 9 != 0;
            });
            if (copied) for (size_t r = begin; r < begin + count; ++r) expected[which][r] = false;
            for (uint32_t l = 0; l < 2; ++l) for (auto p : {operand::k, operand::v}) {
                const size_t index = size_t(l)*2 + (p == operand::v);
                for (size_t r = 0; r < n; ++r) t.assert_equal(bool(expected[index][r]), is_dirty(t, content, l, p, r));
            }
        }
    });
    t.test("quant_pairs_flush_their_own_encoded_row_sizes", [](testing & t) {
        for (int k : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL})
            for (int v : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) {
                auto host = storage(k, v); llama_kv_stream_content content(host);
                if (!t.assert_true(content.flush(all(*host), [](const auto &) { return true; }))) continue;
                const size_t stride = ggml_row_size(ggml_type(v), 32);
                uint8_t data[] = {0x21, 0x52}; llama_kv_stream_write write;
                t.assert_true(content.prepare({{1, operand::v, stride*8 - 1, data, 2}}, write));
                t.assert_true(content.commit(write));
                size_t calls = 0;
                t.assert_true(content.flush(all(*host), [&](const auto & span) {
                    ++calls;
                    t.assert_true(span.rows.layer == 1 && span.rows.operand == operand::v && span.rows.first == 7 && span.rows.count == 2);
                    t.assert_equal(stride*2, span.bytes);
                    return true;
                }));
                t.assert_equal(size_t(1), calls);
            }
    });
    t.test("external_restore_on_same_backing_invalidates_pending_write", [](testing & t) {
        auto host = storage(); llama_kv_stream_content content(host);
        t.assert_true(content.flush(all(*host), [](const auto &) { return true; }));
        uint8_t value = 3; llama_kv_stream_write write;
        if (!t.assert_true(content.prepare({{0, operand::k, 0, &value, 1}}, write))) return;
        llama_kv_stream_host_layer layer; host->layer(0, layer);
        *static_cast<uint8_t *>(layer.k) = 9;
        const auto mirror = content.mirror_epoch();
        t.assert_true(content.invalidate());
        t.assert_true(!content.commit(write));
        t.assert_equal(uint8_t(9), *static_cast<uint8_t *>(layer.k));
        t.assert_equal(mirror, content.mirror_epoch());
        t.assert_true(is_dirty(t, content, 0, operand::k, 0));
        t.assert_true(content.flush(all(*host), [](const auto &) { return true; }));
        t.assert_true(content.flush(all(*host), {}));
    });
    t.test("direct_generated_pair_defers_visibility_and_retains_backing", [](testing & t) {
        auto host=storage(); std::weak_ptr<llama_kv_stream_host> weak=host;
        llama_kv_stream_content content(host);
        if (!t.assert_true(content.flush(all(*host),[](const auto &) { return true; }))) return;
        const size_t ks=host->layout().k_token_bytes,vs=host->layout().v_token_bytes;
        const auto generation=content.generation();
        llama_kv_stream_write write;
        size_t calls=0;
        t.assert_true(content.prepare_direct_generated({
            {0,operand::k,3*ks,nullptr,ks},{0,operand::v,3*vs,nullptr,vs}},
            [&](const auto & span,void * output) {
                ++calls;
                std::memset(output,span.operand == operand::k ? 0x35 : 0x64,span.bytes);
                return true;
            },write));
        t.assert_equal(size_t(2),calls);
        t.assert_true(write.pending());
        t.assert_equal(generation,content.generation());
        t.assert_true(!is_dirty(t,content,0,operand::k,3));
        t.assert_true(!is_dirty(t,content,0,operand::v,3));
        llama_kv_stream_host_layer layer; host->layer(0,layer);
        t.assert_equal(uint8_t(0x35),static_cast<uint8_t *>(layer.k)[3*ks]);
        t.assert_equal(uint8_t(0x64),static_cast<uint8_t *>(layer.v)[3*vs]);
        host.reset();
        t.assert_true(!weak.expired());
        t.assert_true(content.commit(write));
        t.assert_equal(generation+1,content.generation());
        t.assert_true(is_dirty(t,content,0,operand::k,3));
        t.assert_true(is_dirty(t,content,0,operand::v,3));
        t.assert_true(!write.pending());

        llama_kv_stream_write cancelled;
        t.assert_true(content.prepare_direct_generated({{0,operand::k,4*ks,nullptr,ks}},
            [](const auto & span,void * output) { std::memset(output,0x71,span.bytes); return true; },cancelled));
        cancelled.cancel();
        t.assert_equal(generation+1,content.generation());
        t.assert_true(!is_dirty(t,content,0,operand::k,4));
    });

    t.test("move_cancel_and_owner_destruction_preserve_ticket_safety", [](testing & t) {
        llama_kv_stream_write pending;
        std::weak_ptr<llama_kv_stream_host> weak;
        {
            auto host = storage(); weak = host;
            llama_kv_stream_content content(host);
            uint8_t value = 1;
            if (!t.assert_true(content.prepare({{0, operand::k, 0, &value, 1}}, pending))) return;
            llama_kv_stream_write moved(std::move(pending));
            t.assert_true(!pending.pending() && moved.pending());
            pending = std::move(moved);
        }
        t.assert_true(!weak.expired());
        pending.cancel();
        t.assert_true(weak.expired());
        bool rejected = false;
        try { llama_kv_stream_content invalid(nullptr); } catch (const std::invalid_argument &) { rejected = true; }
        t.assert_true(rejected);
    });
    if (argc > 1 && std::strcmp(argv[1], "--cuda") == 0) {
        ggml_backend_load_all();
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!t.assert_true(reg && ggml_backend_reg_dev_count(reg))) return t.summary();
        auto * device = ggml_backend_reg_dev_get(reg, 0);
        t.test("cuda_synchronized_mirror_matches_dirty_host_planes", [&](testing & t) {
            auto * type = llama_kv_stream_host_buffer_type(device);
            if (!t.assert_true(type != nullptr)) return;
            auto config = storage()->config();
            config.shape.head_dim_k = config.shape.head_dim_v = 256;
            config.shape.heads = 4;
            auto host = llama_kv_stream_host::create(config, type);
            if (!t.assert_true(host != nullptr)) return;
            llama_kv_stream_content content(host);
            auto * local = llama_kv_stream_device_buffer_type(device);
            if (!t.assert_true(local != nullptr)) return;
            using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
            using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;
            arena_ptr arena(ggml_backend_memory_arena_new(local, host->bytes() + 128), ggml_backend_memory_arena_free);
            if (!t.assert_true(arena != nullptr)) return;
            t.assert_true(ggml_backend_memory_arena_begin(arena.get(), 0));
            t.assert_true(ggml_backend_memory_arena_reserve_at(arena.get(), 9, 128, host->bytes(), 128, 0, nullptr));
            if (!t.assert_true(ggml_backend_memory_arena_commit(arena.get()))) return;
            lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 9), ggml_backend_memory_lease_free);
            llama_kv_stream_binding binding(host->cache_id(), local);
            llama_kv_stream_policy_config policy;
            policy.shape = config.shape; policy.capabilities = config.capabilities;
            policy.layers = config.layers; policy.pool_bytes = host->bytes();
            struct resources : llama_memory_executable { std::shared_ptr<llama_kv_stream_host> host; };
            if (!t.assert_true(binding.bind(lease.get(), policy, [&](const auto &) {
                auto native = std::make_unique<resources>(); native->host = host; return native;
            }))) return;
            auto * buffer = binding.view()->buffer;
            ggml_context_ptr ctx(ggml_init({4096, nullptr, true}));
            if (!t.assert_true(ctx != nullptr)) return;
            auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, int64_t(host->bytes()));
            if (!t.assert_true(ggml_backend_tensor_alloc(buffer, tensor, binding.view()->base) == GGML_STATUS_SUCCESS)) return;
            auto pin = binding.acquire();
            if (!t.assert_true(bool(pin))) return;
            ggml_backend_buffer_clear(buffer, 0);
            auto copy = [&](const llama_kv_stream_copy_span & span) {
                const auto & layout = host->layout();
                const size_t stride = span.rows.operand == operand::k ? layout.k_token_bytes : layout.v_token_bytes;
                const size_t offset = span.rows.layer*host->stride() + (span.rows.operand == operand::k ? 0 : layout.v_offset) + span.rows.first*stride;
                ggml_backend_tensor_set(tensor, span.data, offset, span.bytes);
                return true;
            };
            t.assert_true(content.flush(all(*host), copy));
            std::vector<uint8_t> data(3, 0x52); llama_kv_stream_write write;
            t.assert_true(content.prepare({{1, operand::v, host->layout().v_token_bytes*8 - 1, data.data(), data.size()}}, write));
            t.assert_true(content.commit(write));
            t.assert_true(content.flush(all(*host), copy));
            std::vector<uint8_t> result(host->bytes());
            ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
            llama_kv_stream_host_layer first; host->layer(0, first);
            t.assert_true(std::memcmp(first.k, result.data(), host->bytes()) == 0);
            const auto arena_generation = ggml_backend_memory_arena_generation(arena.get());
            const auto binding_revision = binding.view()->revision;
            *static_cast<uint8_t *>(first.k) = 0x71;
            t.assert_true(content.invalidate());
            t.assert_true(content.flush(all(*host), copy));
            ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
            t.assert_true(std::memcmp(first.k, result.data(), host->bytes()) == 0);
            t.assert_equal(arena_generation, ggml_backend_memory_arena_generation(arena.get()));
            t.assert_equal(binding_revision, binding.view()->revision);
            pin.reset(); lease.reset();
            struct completion : llama_memory_executor_backend { bool drain() override { return true; } } done;
            t.assert_true(binding.detach(done).status == llama_memory_executor_status::retired);
        });
    }
    return t.summary();
}
