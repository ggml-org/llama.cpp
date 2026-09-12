#include "kv-stream-resident-test.h"

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev, nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    testing t;
    t.test("ordinary_attention_matches_scalar_reference", [&](testing & t) {
        fixture f(backend.get(), cuda);
        evaluate(t, f, false, 1, 257, 8);
    });
    t.test("resident_attention_matches_decode_and_prefill", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        for (size_t q : {size_t(1), size_t(8), size_t(33)}) evaluate(t, f, true, 1, 257, q);
        evaluate(t, f, true, 0, 769, 1); // padded keys exactly fill the four-page resident capacity
        t.assert_true(f.resident->synchronize(257));
        t.assert_equal(size_t(0), f.resident->last_upload_bytes());
    });
    t.test("resident_dirty_tail_changes_attention_without_rebinding", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        if (!evaluate(t, f, true, 1, 257, 1)) return;
        std::vector<float> row(512, 3.0f);
        std::vector<uint8_t> data(f.host->layout().v_token_bytes);
        ggml_quantize_chunk(GGML_TYPE_F16, row.data(), data.data(), 0, 2, 256, nullptr);
        llama_kv_stream_write write;
        t.assert_true(f.content->prepare({{1, ggml_kv_stream_operand::v, 256*data.size(), data.data(), data.size()}}, write));
        t.assert_true(f.content->commit(write));
        t.assert_true(!f.resident->ready(257));
        if (!evaluate(t, f, true, 1, 257, 1)) return;
        t.assert_equal(data.size(), f.resident->last_upload_bytes());
        t.assert_equal(size_t(1), f.resident->last_upload_calls());
    });
    t.test("invalid_metadata_and_streaming_requirement_are_rejected", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        t.assert_true(!f.resident->synchronize(0));
        t.assert_true(!f.resident->synchronize(770));
        ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 1, 4);
        t.assert_true(!f.resident->attention(ctx.get(), 0, q, nullptr, 257, 1.0f/16));
        t.assert_true(!f.resident->attention(ctx.get(), 9, q, nullptr, 256, 1.0f/16));
        t.assert_true(!f.resident->attention(ctx.get(), 0, q, nullptr, 256, NAN));
    });
    t.test("policy_offsets_and_flat_roots_preserve_ring_bytes", [&](testing & t) {
        fixture f(backend.get(), cuda, cuda ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, cuda ? GGML_TYPE_Q4_0 : GGML_TYPE_F16);
        auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        ggml_backend_buffer_clear(buffer, 0xa5);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 1, 4);
        auto * out = f.resident->attention(ctx.get(), 1, q, nullptr, 256, 1.0f/16);
        if (!t.assert_true(out != nullptr)) return;
        llama_kv_stream_policy_layout layout;
        const auto & binding = *f.binding->view();
        t.assert_true(llama_kv_stream_policy_layout_make(f.policy, binding.initial_policy, 0, layout).status == llama_kv_stream_policy_status::success);
        const auto & entry = layout.layers[1];
        t.assert_true(out->src[1]->data == static_cast<char *>(binding.base) + entry.offset);
        t.assert_true(out->src[2]->data == static_cast<char *>(binding.base) + entry.offset + entry.planes.v_offset);
        for (int side = 1; side <= 2; ++side) {
            auto * root = out->src[side]->view_src;
            t.assert_equal(ggml_nbytes(root), ggml_backend_buffer_get_alloc_size(buffer, root));
            t.assert_equal(side == 1 ? entry.planes.k_token_bytes : entry.planes.v_token_bytes, out->src[side]->nb[1]);
            t.assert_equal(side == 1 ? entry.planes.k_row_bytes : entry.planes.v_row_bytes, out->src[side]->nb[2]);
        }
        t.assert_true(f.resident->synchronize(256));
        auto * marker = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 128);
        t.assert_true(ggml_backend_tensor_alloc(buffer, marker, binding.base) == GGML_STATUS_SUCCESS);
        std::vector<uint8_t> bytes(128);
        ggml_backend_tensor_get(marker, bytes.data(), 0, bytes.size());
        t.assert_true(std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 0xa5; }));
    });
    t.test("insufficient_residency_and_conversion_fallback_are_not_dispatched", [&](testing & t) {
        fixture small(backend.get(), cuda);
        small.policy.pool_bytes = small.policy.pool_bytes/16*10;
        if (!t.assert_true(small.attach())) return;
        t.assert_true(small.resident->synchronize(256));
        t.assert_true(!small.resident->synchronize(257) && !small.resident->ready(256));
        fixture conversion(backend.get(), cuda);
        conversion.policy.capabilities.direct_pair = false;
        t.assert_true(!conversion.attach());
    });
    t.test("malformed_q_and_mask_reject_before_ggml_assertions", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
        for (int variant = 0; variant < 5; ++variant) {
            auto * q = ggml_new_tensor_4d(ctx.get(), variant == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                variant == 1 ? 128 : 256, 8, variant == 2 ? 3 : 4, variant == 3 ? 2 : 1);
            auto * mask = ggml_new_tensor_2d(ctx.get(), variant == 4 ? GGML_TYPE_F32 : GGML_TYPE_F16, 256, 8);
            t.assert_true(f.resident->attention(ctx.get(), 0, q, mask, 256, 1.0f/16) == nullptr);
        }
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 8, 4);
        auto * short_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 128, 8);
        t.assert_true(f.resident->attention(ctx.get(), 0, q, short_mask, 256, 1.0f/16) == nullptr);
    });
    t.test("copy_exception_stays_closed_and_can_retry", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        const auto original = buffer->iface.set_tensor;
        buffer->iface.set_tensor = [](ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) { throw std::runtime_error("copy failure"); };
        bool caught = false;
        try { f.resident->synchronize(257); } catch (const std::runtime_error &) { caught = true; }
        buffer->iface.set_tensor = original;
        t.assert_true(caught && !f.resident->ready(257));
        t.assert_true(f.resident->synchronize(257) && f.resident->ready(257));
        t.assert_equal(size_t(512)*2*(f.host->layout().k_token_bytes + f.host->layout().v_token_bytes), f.resident->last_upload_bytes());
        evaluate(t, f, true, 0, 257, 1);
    });
    t.test("shorter_replacement_refreshes_same_binding", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        if (!evaluate(t, f, true, 0, 257, 1)) return;
        const auto arena_generation = ggml_backend_memory_arena_generation(f.arena.get());
        const auto binding_revision = f.binding->view()->revision;
        auto c = f.host->config();
        auto replacement = llama_kv_stream_host::create(c, ggml_backend_buffer_get_type(f.host->buffer()));
        populate(replacement);
        llama_kv_stream_host_layer plane; replacement->layer(0, plane);
        std::vector<float> row(512, 1.5f);
        ggml_quantize_chunk(GGML_TYPE_F16, row.data(), plane.v, 0, 2, 256, nullptr);
        f.host = replacement;
        t.assert_true(f.content->replace(replacement) && !f.resident->ready(257));
        if (!evaluate(t, f, true, 0, 129, 8)) return;
        t.assert_equal(arena_generation, ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_equal(binding_revision, f.binding->view()->revision);
        t.assert_true(f.content->reset_mirror() && !f.resident->ready(129));
        t.assert_true(f.resident->synchronize(129));
        t.assert_equal(size_t(256)*2*(f.host->layout().k_token_bytes + f.host->layout().v_token_bytes), f.resident->last_upload_bytes());
    });
    t.test("factory_and_replaced_geometry_are_validated", [&](testing & t) {
        fixture f(backend.get(), cuda);
        t.assert_true(!f.binding->bind(f.lease.get(), f.policy, [&](const auto & view) -> std::unique_ptr<llama_memory_executable> {
            auto wrong = view; ++wrong.cache_id;
            t.assert_true(llama_kv_stream_resident::create(wrong, f.content, backend.get()) == nullptr);
            wrong = view; --wrong.capacity;
            t.assert_true(llama_kv_stream_resident::create(wrong, f.content, backend.get()) == nullptr);
            t.assert_true(llama_kv_stream_resident::create(view, nullptr, backend.get()) == nullptr);
            return {};
        }));
        if (!t.assert_true(f.attach())) return;
        t.assert_true(f.resident->synchronize(256));
        auto c = f.host->config();
        c.shape.type_v = GGML_TYPE_Q4_0;
        c.capabilities.v.type = GGML_TYPE_Q4_0;
        auto replacement = llama_kv_stream_host::create(c, ggml_backend_buffer_get_type(f.host->buffer()));
        if (!t.assert_true(replacement != nullptr)) return;
        t.assert_true(f.content->replace(replacement));
        t.assert_true(!f.resident->synchronize(256) && !f.resident->ready(256));
    });
    t.test("external_graph_pin_blocks_resident_destruction", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        struct completion : llama_memory_executor_backend {
            ggml_backend_t backend;
            explicit completion(ggml_backend_t b) : backend(b) {}
            bool drain() override { ggml_backend_synchronize(backend); return true; }
        } done(backend.get());
        f.lease.reset();
        t.assert_true(f.binding->detach(done).status == llama_memory_executor_status::pending);
        t.assert_true(ggml_backend_memory_arena_lease_count(f.arena.get()) != 0);
        pin.reset();
        t.assert_true(f.binding->detach(done).status == llama_memory_executor_status::retired);
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
    });
    t.test("wide_prefill_uses_ordinary_attention", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        evaluate(t, f, true, 0, 257, 257);
    });
    if (cuda) t.test("mixed_quant_resident_and_ordinary_paths_agree", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}, std::pair{GGML_TYPE_Q5_1, GGML_TYPE_Q4_1}}) {
            fixture f(backend.get(), cuda, pair.first, pair.second);
            if (!t.assert_true(f.attach())) continue;
            evaluate(t, f, false, 0, 257, 1);
            evaluate(t, f, true, 0, 257, 1);
            evaluate(t, f, true, 1, 257, 8);
        }
    });
    return t.summary();
}
