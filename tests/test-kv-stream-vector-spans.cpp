#include "kv-stream-block-test.h"

#include <algorithm>

struct span_storage {
    std::unique_ptr<block_workspace> storage;
    ggml_kv_stream_span_plan_t plan = nullptr;

    ~span_storage() {
        ggml_kv_stream_span_plan_free(plan);
    }

    static std::unique_ptr<span_storage> create(
            fixture & f, uint32_t layer, size_t active, uint32_t queries,
            const std::vector<size_t> & cuts) {
        if (cuts.size() < 2 || cuts.front() != 0 || cuts.back() != active) return {};
        auto result = std::make_unique<span_storage>();
        std::vector<ggml_kv_stream_layout> layouts;
        std::vector<size_t> offsets;
        size_t bytes = 0;
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            const size_t tokens = cuts[i + 1] - cuts[i];
            if (!tokens) return {};
            ggml_kv_stream_layout layout;
            if (ggml_kv_stream_layout_make(f.policy.shape, tokens, layout).status !=
                    ggml_kv_stream_status::success) return {};
            bytes = (bytes + 127)/128*128;
            offsets.push_back(bytes);
            layouts.push_back(layout);
            if (layout.bytes > SIZE_MAX - bytes) return {};
            bytes += layout.bytes;
        }
        result->storage = std::make_unique<block_workspace>(f, bytes, 71);
        auto * buffer = ggml_backend_memory_lease_buffer(result->storage->lease.get());
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
        llama_kv_stream_host_layer host;
        if (!f.host->layer(layer, host)) return {};
        const auto & full = f.host->layout();
        std::vector<ggml_kv_stream_span_source> sources;
        for (size_t i = 0; i < layouts.size(); ++i) {
            const size_t first = cuts[i], tokens = cuts[i + 1] - cuts[i];
            const auto & layout = layouts[i];
            const size_t k_offset = offsets[i], v_offset = offsets[i] + layout.v_offset;
            ggml_tensor k = {}, v = {};
            for (auto * tensor : {&k, &v}) {
                tensor->ne[0] = 256;
                tensor->ne[1] = int64_t(tokens);
                tensor->ne[2] = 2;
                tensor->ne[3] = 1;
                tensor->buffer = buffer;
            }
            k.type = ggml_type(f.policy.shape.type_k);
            k.nb[0] = ggml_type_size(k.type);
            k.nb[1] = layout.k_token_bytes;
            k.nb[2] = layout.k_row_bytes;
            k.nb[3] = layout.k_bytes;
            k.data = base + k_offset;
            v.type = ggml_type(f.policy.shape.type_v);
            v.nb[0] = ggml_type_size(v.type);
            v.nb[1] = layout.v_token_bytes;
            v.nb[2] = layout.v_row_bytes;
            v.nb[3] = layout.v_bytes;
            v.data = base + v_offset;
            ggml_backend_tensor_set(&k,
                static_cast<const uint8_t *>(host.k) + first*full.k_token_bytes,
                0, layout.k_bytes);
            ggml_backend_tensor_set(&v,
                static_cast<const uint8_t *>(host.v) + first*full.v_token_bytes,
                0, layout.v_bytes);
            sources.push_back({
                result->storage->lease.get(), result->storage->lease.get(),
                first, tokens, k_offset, v_offset});
        }
        if (ggml_kv_stream_span_plan_make(
                f.policy.shape, sources.data(), sources.size(), active, queries, result->plan).status !=
                ggml_kv_stream_status::success) return {};
        return result;
    }
};

static std::vector<float> ordinary(fixture & f, block_inputs & input, uint32_t layer) {
    ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
    auto * k = ggml_new_tensor_2d(ctx.get(), ggml_type(f.policy.shape.type_k), 512, input.padded);
    auto * v = ggml_new_tensor_2d(ctx.get(), ggml_type(f.policy.shape.type_v), 512, input.padded);
    auto * key = ggml_view_3d(ctx.get(), k, 256, input.padded, 2,
        ggml_row_size(k->type, 512), ggml_row_size(k->type, 256), 0);
    auto * value = ggml_view_3d(ctx.get(), v, 256, input.padded, 2,
        ggml_row_size(v->type, 512), ggml_row_size(v->type, 256), 0);
    auto * out = ggml_flash_attn_ext(ctx.get(), input.q, key, value, input.mask, 1.0f/16, 0, 0);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), f.backend));
    GGML_ASSERT(buffer);
    llama_kv_stream_host_layer host;
    GGML_ASSERT(f.host->layer(layer, host));
    ggml_backend_tensor_set(k, host.k, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, host.v, 0, ggml_nbytes(v));
    GGML_ASSERT(ggml_backend_graph_compute(f.backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    return result;
}

static std::vector<float> evaluate(
        fixture & f, const ggml_kv_stream_partial_ops * ops,
        span_storage & storage, block_inputs & input, uint32_t layer, bool & accepted) {
    ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
    auto * node = f.resident->attention(ctx.get(), layer, input.q, input.mask, input.active, 1.0f/16);
    if (!node) {
        accepted = false;
        return {};
    }
    ggml_tensor op = *node;
    op.buffer = input.output->buffer;
    op.data = input.output->data;
    ggml_kv_stream_resume_plan plan;
    if (!ops->resume_plan(
            f.backend, f.policy.shape.type_k, f.policy.shape.type_v, 4, 2,
            uint32_t(input.queries), input.active, plan)) {
        accepted = false;
        return {};
    }
    block_workspace workspace(f, plan.bytes, 79);
    accepted = ops->spans(
        f.backend, &op, storage.plan,
        ggml_backend_memory_lease_buffer(workspace.lease.get()));
    ggml_backend_synchronize(f.backend);
    return accepted ? input.read() : std::vector<float>{};
}

int main(int argc, char ** argv) {
    testing t;
    t.test("resume_layout_accounts_for_tg1_and_tg2", [](testing & t) {
        ggml_kv_stream_resume_plan one, two;
        if (!t.assert_true(ggml_kv_stream_resume_layout_make(24, 1, 3, 8, one))) return;
        if (!t.assert_true(ggml_kv_stream_resume_layout_make(24, 2, 3, 8, two))) return;
        t.assert_equal(size_t(1), size_t(one.queries));
        t.assert_equal(size_t(2), size_t(two.queries));
        t.assert_equal(2*one.state_bytes, two.state_bytes);
        t.assert_equal(2*(one.meta_offset - one.partial_offset), two.meta_offset - two.partial_offset);
        t.assert_equal(2*one.bytes, two.bytes);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "native_span_vector_contract_is_available", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        t.assert_true(get && get() && get()->version >= 6 && get()->spans);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_tg2_partition_boundaries_match_contiguous_and_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 6 && ops->spans)) return;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(257), size_t(513), size_t(1017)}) {
            for (uint32_t queries : {1u, 2u}) {
                std::vector<size_t> cuts{0, 1, 127, 128, 255, 256, 257, active - 1, active};
                cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                    [&](size_t value) { return value > active; }), cuts.end());
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
                auto contiguous = span_storage::create(f, 0, active, queries, {0, active});
                auto segmented = span_storage::create(f, 0, active, queries, cuts);
                if (!t.assert_true(contiguous && segmented && contiguous->plan && segmented->plan)) continue;
                block_inputs input(f, active, queries);
                const auto expected = ordinary(f, input, 0);
                bool accepted = false;
                const auto a = evaluate(f, ops, *contiguous, input, 0, accepted);
                if (!t.assert_true(accepted)) continue;
                const auto b = evaluate(f, ops, *segmented, input, 0, accepted);
                if (!t.assert_true(accepted)) continue;
                close_values(t, a, b, 1e-7f);
                close_values(t, expected, b, 1e-5f);
            }
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "invalid_span_plan_and_workspace_preserve_output", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        block_inputs input(f, 513, 2);
        auto wrong_width = span_storage::create(f, 0, 513, 1, {0, 256, 513});
        auto valid = span_storage::create(f, 0, 513, 2, {0, 256, 513});
        if (!t.assert_true(wrong_width && valid)) return;
        ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
        auto * node = f.resident->attention(ctx.get(), 0, input.q, input.mask, input.active, 1.0f/16);
        if (!t.assert_true(node != nullptr)) return;
        ggml_tensor op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ops->resume_plan(
                backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 4, 2, 2, 513, plan))) return;
        block_workspace workspace(f, plan.bytes, 83), tiny(f, plan.bytes - 1, 89);
        t.assert_true(!ops->spans(backend.get(), &op, nullptr,
            ggml_backend_memory_lease_buffer(workspace.lease.get())));
        t.assert_true(!ops->spans(backend.get(), &op, wrong_width->plan,
            ggml_backend_memory_lease_buffer(workspace.lease.get())));
        t.assert_true(!ops->spans(backend.get(), &op, valid->plan,
            ggml_backend_memory_lease_buffer(tiny.lease.get())));
        std::vector<float> actual = input.read();
        t.assert_true(std::all_of(actual.begin(), actual.end(), [](float value) { return value == -77; }));
    });

    return t.summary();
}
