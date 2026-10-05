#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-dispatch.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-plan.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../src/llama-kv-stream-model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

#ifdef KV_STREAM_PASCAL_CUDA_TEST
bool kv_stream_test_is_sm61_only();
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
int kv_stream_test_compiled_cc(ggml_backend_t backend);

struct pascal_test_guard {
    ggml_backend_t backend;
    int previous = 0;
    bool forced;
    // Only the SM61-only test mode changes host metadata; ordinary modern tests retain their real device.
    pascal_test_guard(ggml_backend_t backend, bool forced) : backend(backend), forced(forced) {
        if (forced) previous = kv_stream_test_override_cc(backend,610);
    }
    ~pascal_test_guard() { if (forced) kv_stream_test_override_cc(backend,previous); }
};
#endif

struct span_storage {
    std::unique_ptr<block_workspace> storage;
    ggml_kv_stream_span_plan_t plan = nullptr;

    ~span_storage() {
        ggml_kv_stream_span_plan_free(plan);
    }

    static std::unique_ptr<span_storage> create(
            fixture & f, uint32_t layer, size_t active, uint32_t queries,
            const std::vector<size_t> & cuts, bool wrapped = false) {
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
        if (wrapped) {
            bytes = 0;
            for (size_t i = 0; i < layouts.size(); ++i) {
                const size_t logical = i == 0 ? 0 : layouts.size()-i;
                bytes = (bytes+127)/128*128;
                offsets[logical] = bytes;
                bytes += layouts[logical].bytes;
            }
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
                tensor->ne[2] = f.policy.shape.heads;
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
        span_storage & storage, block_inputs & input, uint32_t layer, bool & accepted,
        size_t * used_workspace = nullptr) {
    ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
    auto * node = f.resident->attention(ctx.get(), layer, input.q, input.mask, input.active, 1.0f/16);
    if (!node) {
        accepted = false;
        return {};
    }
    ggml_tensor op = *node;
    op.buffer = input.output->buffer;
    op.data = input.output->data;
    size_t workspace_bytes = 0;
    if (!ops->spans_workspace(f.backend, &op, storage.plan, workspace_bytes)) {
        accepted = false;
        return {};
    }
    if (used_workspace) *used_workspace = workspace_bytes;
    block_workspace workspace(f, workspace_bytes, 79);
    accepted = ops->spans(
        f.backend, &op, storage.plan,
        ggml_backend_memory_lease_buffer(workspace.lease.get()));
    ggml_backend_synchronize(f.backend);
    return accepted ? input.read() : std::vector<float>{};
}


struct stock_attention_case {
    fixture & f;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    stock_attention_case(fixture & f, block_inputs & input) : f(f), context(ggml_init({65536, nullptr, true})) {
        k_storage = ggml_new_tensor_2d(context.get(), ggml_type(f.policy.shape.type_k), 512, input.padded);
        v_storage = ggml_new_tensor_2d(context.get(), ggml_type(f.policy.shape.type_v), 512, input.padded);
        k = ggml_view_3d(context.get(), k_storage, 256, input.padded, 2,
            ggml_row_size(k_storage->type, 512), ggml_row_size(k_storage->type, 256), 0);
        v = ggml_view_3d(context.get(), v_storage, 256, input.padded, 2,
            ggml_row_size(v_storage->type, 512), ggml_row_size(v_storage->type, 256), 0);
        auto * output = ggml_flash_attn_ext(context.get(), input.q, k, v, input.mask, 1.0f/16, 0, 0);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
        graph = ggml_new_graph_custom(context.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), f.backend));
        GGML_ASSERT(buffer);
        llama_kv_stream_host_layer host;
        GGML_ASSERT(f.host->layer(0, host));
        ggml_backend_tensor_set(k_storage, host.k, 0, ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage, host.v, 0, ggml_nbytes(v_storage));
    }

    void run() {
        GGML_ASSERT(ggml_backend_graph_compute(f.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(f.backend);
    }
};

struct span_attention_case {
    fixture & f;
    const ggml_kv_stream_partial_ops * ops;
    span_storage & storage;
    ggml_context_ptr context;
    ggml_tensor op = {};
    std::unique_ptr<block_workspace> workspace;

    span_attention_case(
            fixture & f, const ggml_kv_stream_partial_ops * ops, stock_attention_case & stock,
            span_storage & storage, block_inputs & input) :
        f(f), ops(ops), storage(storage), context(ggml_init({65536, nullptr, true})) {
        auto * node = ggml_flash_attn_ext(
            context.get(),input.q,stock.k,stock.v,input.mask,1.0f/16,0,0);
        GGML_ASSERT(node);
        ggml_flash_attn_ext_set_prec(node, GGML_PREC_F32);
        op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        size_t bytes = 0;
        GGML_ASSERT(ops->spans_workspace(f.backend, &op, storage.plan, bytes));
        workspace = std::make_unique<block_workspace>(f, bytes, 97);
    }

    void run() {
        GGML_ASSERT(ops->spans(
            f.backend, &op, storage.plan, ggml_backend_memory_lease_buffer(workspace->lease.get())));
        ggml_backend_synchronize(f.backend);
    }
};

template<typename F>
static double average_us(F && run, int warmup = 16, int iterations = 64) {
    for (int i = 0; i < warmup; ++i) run();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) run();
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin).count()/iterations;
}

// Synthetic one-layer timing used to compare dispatch shapes, not full-model tokens per second.
static int benchmark_spans() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    GGML_ASSERT(dev);
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    GGML_ASSERT(ops && ops->version >= 7 && ops->spans && ops->spans_workspace);
    for (size_t active : {size_t(8192), size_t(32768), size_t(65536), size_t(131072), size_t(131073)}) {
        for (uint32_t queries : {1u, 4u}) {
            const ggml_type type_k = GGML_TYPE_Q8_0;
            const ggml_type type_v = GGML_TYPE_Q4_0;
            fixture f(backend.get(), true, type_k, type_v, (active+255)/256*256);
            GGML_ASSERT(f.attach());
            auto pin = f.binding->acquire();
            GGML_ASSERT(pin);
            block_inputs input(f, active, queries);
            stock_attention_case stock(f, input);
            const double stock_us = average_us([&] { stock.run(); });
            const std::vector<std::vector<size_t>> layouts{
                {0, active},
                {0, active/2, active},
                {0, active/4, 3*active/4, active},
            };
            for (const auto & cuts : layouts) {
                auto storage = span_storage::create(f, 0, active, queries, cuts);
                GGML_ASSERT(storage && storage->plan);
                span_attention_case spans(f,ops,stock,*storage,input);
                const double span_us = average_us([&] { spans.run(); });
                std::printf(
                    "span-bench,active=%zu,tg=%u,spans=%zu,stock_us=%.3f,span_us=%.3f,overhead_pct=%.2f\n",
                    active, queries, cuts.size()-1, stock_us, span_us, 100.0*(span_us/stock_us-1.0));
            }
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    const bool pascal = argc > 1 && !std::strcmp(argv[1],"--cuda-pascal-tg1");
    if (pascal) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        if (!kv_stream_test_is_sm61_only()) {
            std::puts("SKIP: simulated Pascal TG1 requires an SM61-only build, not a mixed-target binary");
            return 77;
        }
#else
        std::puts("SKIP: CUDA test adapter is not compiled");
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) return benchmark_spans();
    testing t;
    t.test("ampere_decode_dispatch_matches_stock", [](testing & t) {
        using path = ggml_cuda_kv_stream_attention_path;
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::mma);
        t.assert_true(ggml_cuda_kv_stream_attention_select(890,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(890,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0) == path::none);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_F16,GGML_TYPE_F16) == path::none);
        t.assert_equal(4,ggml_cuda_kv_stream_mma_ncols1(2,2));
        t.assert_equal(2,ggml_cuda_kv_stream_mma_ncols1(2,6));
        t.assert_equal(2,ggml_cuda_kv_stream_mma_ncols1(2,8));
        t.assert_equal(4,ggml_cuda_kv_stream_mma_ncols1(3,8));
    });
    if (argc > 1 && !std::strcmp(argv[1], "--cuda-gqa6")) t.set_filter("qwen_ratio_six_tg3_tg4_matches_stock|qwen_24_4_tg2_aligned_and_wrapped_spans_are_exact");
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

#ifdef KV_STREAM_PASCAL_CUDA_TEST
    if (pascal || (argc > 1 && !std::strcmp(argv[1],"--cuda"))) t.test("quantized_tg1_stock_equivalence_across_spans_and_ring_waves", [&](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),pascal);
        if (pascal) t.out << "SM61 compiled code and simulated host CC610; physical device: " << ggml_backend_dev_description(dev) << '\n';
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->resume_plan && ops->spans_workspace && ops->spans)) return;
        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            if (ggml_blck_size(ggml_type(type)) > 0) continue;
            ggml_kv_stream_resume_plan untouched;
            untouched.bytes = 77;
            t.assert_true(!ops->resume_plan(backend.get(),type,GGML_TYPE_Q4_0,24,4,1,256,untouched));
            t.assert_true(!ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,type,24,4,1,256,untouched));
            t.assert_equal(size_t(77),untouched.bytes);
        }
        for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q4_1},
                std::pair{GGML_TYPE_Q4_0,GGML_TYPE_F16}, std::pair{GGML_TYPE_Q4_0,GGML_TYPE_BF16}}) {
            fixture f(backend.get(),true,pair.first,pair.second,4097,false,2,4,3);
            f.policy.initial_ring_slots = 1;
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
            auto pin = f.binding->acquire();
            fixture control(backend.get(),true,pair.first,pair.second,4097,false,2,4,35);
            control.policy.initial_ring_slots = 1;
            if (!t.assert_true(control.attach())) return;
            auto control_pin = control.binding->acquire();
            for (size_t active : {size_t(33),size_t(255),size_t(256),size_t(257),size_t(513),size_t(1024),size_t(1025),size_t(4097)}) {
                t.out << ggml_type_name(pair.first) << '/' << ggml_type_name(pair.second) << " active=" << active << '\n';
                block_inputs input(f,active,1,false,24);
                std::vector<ggml_fp16_t> mask(input.padded);
                for (size_t i = 0; i < input.padded; ++i) mask[i] = ggml_fp32_to_fp16(i < active && i%7 != 3 ? 0 : -INFINITY);
                ggml_backend_tensor_set(input.mask,mask.data(),0,mask.size()*sizeof(ggml_fp16_t));
                ggml_kv_stream_resume_plan plan;
                if (!t.assert_true("TG1 resume plan",ops->resume_plan(backend.get(),pair.first,pair.second,24,4,1,input.padded,plan))) return;
                const bool half_value = pair.second == GGML_TYPE_F16 || pair.second == GGML_TYPE_BF16;
                t.assert_equal(uint32_t(half_value ? (kv_stream_test_compiled_cc(backend.get()) < 700 ? 16 : 32) : 8),plan.values_per_thread);
                ggml_kv_stream_resume_plan checked;
                t.assert_true(ggml_kv_stream_resume_layout_make(24,1,plan.splits,plan.values_per_thread,checked));
                t.assert_equal(checked.bytes,plan.bytes);
                block_workspace workspace(f,plan.bytes), tiny(f,plan.bytes-1,89);
                auto * scratch_parent = ggml_backend_memory_arena_parent(workspace.arena.get());
                ggml_backend_buffer_clear(scratch_parent,0x5a);
                if (!t.assert_true("resident prefix synchronization",f.resident->synchronize(std::min(active,size_t(256))))) return;
                if (!t.assert_true("initial sequence admission",f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{1,true}))) return;
                if (active > 256) {
                    t.assert_true("one-byte-short scratch rejected",!f.resident->compute_streamed(0,input.q,input.mask,input.output,active,1.0f/16,tiny.lease.get(),true,1));
                    const auto untouched = input.read();
                    t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](float x) { return x == -77; }));
                    if (!t.assert_true("retry sequence after scratch rejection",f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{1,true}))) return;
                }
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    const auto expected = stock_attention(f,input,layer);
                    if (!t.assert_true("exact-sized resumed layer",f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,1))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual = input.read();
                    close_values(t,expected,actual,1e-6f);
                    if (active%256 == 0) t.assert_true(same_float_bits(expected,actual));
                    if (active > 512) t.assert_true(f.resident->last_attention_calls() > 1);
                }
                t.assert_true(!f.resident->sequence_active());
                ggml_backend_memory_region scratch_region;
                t.assert_true(ggml_backend_memory_lease_get_region(workspace.lease.get(),&scratch_region));
                ggml_tensor guard_bytes = {};
                guard_bytes.type = GGML_TYPE_I8;
                guard_bytes.buffer = scratch_parent;
                guard_bytes.data = ggml_backend_buffer_get_base(scratch_parent);
                guard_bytes.ne[0] = ggml_backend_buffer_get_size(scratch_parent);
                guard_bytes.nb[0] = 1;
                for (int i = 1; i < 4; ++i) { guard_bytes.ne[i] = 1; guard_bytes.nb[i] = guard_bytes.ne[0]; }
                std::vector<uint8_t> guards(size_t(guard_bytes.ne[0]));
                ggml_backend_tensor_get(&guard_bytes,guards.data(),0,guards.size());
                t.assert_true(std::all_of(guards.begin(),guards.begin()+scratch_region.offset,[](uint8_t x) { return x == 0x5a; }));
                t.assert_true(std::all_of(guards.begin()+scratch_region.offset+scratch_region.size,guards.end(),[](uint8_t x) { return x == 0x5a; }));
                const std::vector<size_t> cuts = active > 512 ? std::vector<size_t>{0,256,512,active} : std::vector<size_t>{0,active};
                if (!t.assert_true(control.resident->synchronize(active))) return;
                auto storage = span_storage::create(control,0,active,1,cuts,true);
                if (!t.assert_true(storage != nullptr)) return;
                bool accepted = false;
                const auto expected = stock_attention(f,input,0);
                const auto actual = evaluate(control,ops,*storage,input,0,accepted);
                if (!t.assert_true(accepted)) return;
                close_values(t,expected,actual,1e-6f);
            }
        }
    });
    if (pascal) t.test("tg1_only_session_preserves_decode_workspace_during_phase_handoff", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),true);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,3);
        ggml_kv_stream_resume_plan plan, untouched;
        if (!t.assert_true(ops && ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,1,f.host->layout().tokens,plan))) return;
        untouched.bytes = 77;
        t.assert_true(!ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,2,f.host->layout().tokens,untouched));
        t.assert_equal(size_t(77),untouched.bytes);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,24});
        if (!t.assert_true(model != nullptr)) return;
        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(model->memory_requirements(requirements))) return;
        t.assert_equal(plan.bytes,requirements.attention_decode_bytes);
        t.assert_true(requirements.attention_decode_bytes < requirements.attention_prefill_bytes);
        if (!t.assert_true(model->begin(1,1,true))) return;
        t.assert_equal(plan.bytes,model->attention_grant_bytes());
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(1,1,false));
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(1,1,true));
        model->abort();
    });
#endif

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "native_span_vector_contract_is_available", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        t.assert_true(get && get() && get()->version >= 8 && get()->spans && get()->spans_workspace && get()->convert_mma_rows);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test("stock_selected_plan_matches_workspace_and_launch", [](testing & t) {
        using status = ggml_cuda_kv_stream_plan_status;
        using family = ggml_cuda_kv_stream_kernel_family;
        using style = ggml_cuda_kv_stream_execution_style;
        using query_t = status (*)(ggml_backend_t, const ggml_tensor *, const ggml_kv_stream_span_plan_view *, style, ggml_cuda_kv_stream_attention_plan &);
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_attention_plan"));
        if (!t.assert_true(query != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025, false, 2, 4, 16);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        for (uint32_t queries : {1u, 2u, 3u, 4u}) {
            block_inputs input(f, 513, queries, false, 24);
            auto storage = span_storage::create(f, 0, 513, queries, {0, 256, 512, 513}, true);
            if (!t.assert_true(pin && storage && storage->plan)) return;
            ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
            auto * node = f.resident->attention(ctx.get(), 0, input.q, input.mask, input.active, 1.0f/16);
            if (!t.assert_true(node != nullptr)) return;
            ggml_tensor op = *node;
            op.buffer = input.output->buffer; op.data = input.output->data;
            ggml_kv_stream_span_plan_view view;
            if (!t.assert_true(ggml_kv_stream_span_plan_get_view(storage->plan, view))) return;
            ggml_cuda_kv_stream_attention_plan plan;
            if (!t.assert_true(query(backend.get(), &op, &view, style::spanned, plan) == status::success)) return;
            t.assert_true(plan.family == (queries <= 2 ? family::vector : family::mma));
            size_t bytes = 0;
            t.assert_true(ops->spans_workspace(backend.get(), &op, storage->plan, bytes));
            t.assert_equal(bytes, plan.requirements.scratch_bytes);
            t.assert_equal(ggml_nbytes(&op), plan.output_allocation_bytes);
            t.assert_true(plan.requirements.backend_scratch_known && plan.requirements.shared_bytes_known);
            ggml_cuda_kv_stream_attention_plan native;
            t.assert_true(query(backend.get(), &op, nullptr, style::native, native) == status::success);
            t.assert_true(native.family == plan.family);
            t.assert_true(!native.requirements.backend_scratch_known && !native.requirements.shared_bytes_known);
            t.assert_equal(ggml_backend_buffer_get_alloc_size(op.buffer, &op), native.output_allocation_bytes);
            if (queries >= 3) t.assert_true(native.output_allocation_bytes > ggml_nbytes(&op));
            ggml_tensor metadata_only = op, q = *op.src[0], k = *op.src[1], v = *op.src[2], mask = *op.src[3];
            metadata_only.buffer = q.buffer = k.buffer = v.buffer = mask.buffer = nullptr;
            metadata_only.data = q.data = k.data = v.data = mask.data = nullptr;
            metadata_only.src[0] = &q; metadata_only.src[1] = &k; metadata_only.src[2] = &v; metadata_only.src[3] = &mask;
            ggml_cuda_kv_stream_attention_plan described;
            t.assert_true(query(backend.get(), &metadata_only, &view, style::spanned, described) == status::success);
            t.assert_equal(bytes, described.requirements.scratch_bytes);
            ++k.nb[1];
            t.assert_true(query(backend.get(), &metadata_only, &view, style::spanned, described) == status::unsupported_geometry);
            --k.nb[1];
            t.assert_equal(bytes, described.requirements.scratch_bytes);
            auto stale = view; ++stale.query_tokens;
            t.assert_true(query(backend.get(), &op, &stale, style::spanned, described) == status::invalid_metadata);
            t.assert_equal(bytes, described.requirements.scratch_bytes);
            block_workspace exact(f, bytes, 301), short_grant(f, bytes-1, 302);
            t.assert_true(!ops->spans(backend.get(), &op, storage->plan, ggml_backend_memory_lease_buffer(short_grant.lease.get())));
            const auto sentinel = input.read();
            t.assert_true(std::all_of(sentinel.begin(), sentinel.end(), [](float value) { return value == -77; }));
            t.assert_true(ops->spans(backend.get(), &op, storage->plan, ggml_backend_memory_lease_buffer(exact.lease.get())));
            ggml_backend_synchronize(backend.get());
            const auto expected = stock_attention(f, input, 0);
            close_values(t, expected, input.read(), 1e-5f);
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test("native_fallback_cannot_use_a_bounded_output_declaration", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1024);
        block_inputs input(f, 513, 4);
        stock_attention_case stock(f, input);
        ggml_tensor op = *ggml_graph_node(stock.graph, ggml_graph_n_nodes(stock.graph)-1);
        ggml_backend_buffer_ptr parent(ggml_backend_alloc_buffer(backend.get(), 4*1024*1024));
        if (!t.assert_true(ops && parent)) return;
        ggml_backend_buffer_clear(parent.get(), 0x5a);
        op.buffer = parent.get(); op.data = ggml_backend_buffer_get_base(parent.get());
        ggml_backend_execution_set_external_workspace(&op, true);
        t.assert_true(!ops->direct(backend.get(), &op));
        ggml_tensor canary = op;
        canary.type = GGML_TYPE_I8; canary.op = GGML_OP_NONE;
        canary.data = static_cast<char *>(op.data) + ggml_nbytes(&op) + 128;
        canary.ne[0] = 128; canary.nb[0] = 1;
        for (int i = 1; i < 4; ++i) { canary.ne[i] = 1; canary.nb[i] = 128; }
        uint8_t bytes[128]; ggml_backend_tensor_get(&canary, bytes, 0, sizeof(bytes));
        for (uint8_t value : bytes) t.assert_equal(uint8_t(0x5a), value);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_tg2_partition_boundaries_match_contiguous_and_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace)) return;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1280);
        f.policy.initial_ring_slots = 4;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(257), size_t(513), size_t(1017), size_t(1280)}) {
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
                if (queries == 1) {
                    const std::vector<std::vector<size_t>> production_cuts{
                        {0,active/2,active},{0,active/3,2*active/3,active}};
                    for (const auto & physical_cuts : production_cuts) {
                        auto fixed = span_storage::create(f,0,active,queries,physical_cuts);
                        if (!t.assert_true(fixed && fixed->plan)) continue;
                        const auto value = evaluate(f,ops,*fixed,input,0,accepted);
                        if (!t.assert_true(accepted)) continue;
                        close_values(t,a,value,1e-7f);
                        close_values(t,expected,value,1e-5f);
                    }
                }
                if (active == 1280 && queries == 2) {
                    auto production = span_storage::create(f,0,active,queries,{0,256,active});
                    if (t.assert_true(production && production->plan)) {
                        const auto value=evaluate(f,ops,*production,input,0,accepted);
                        if (t.assert_true(accepted)) close_values(t,expected,value,1e-5f);
                    }
                }
                const auto b = evaluate(f, ops, *segmented, input, 0, accepted);
                if (!t.assert_true(accepted)) continue;
                close_values(t, a, b, 1e-7f);
                close_values(t, expected, b, 1e-5f);
            }
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_page_aligned_fixed_spans_match_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1024);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        block_inputs input(f,1024,1);
        const auto expected = ordinary(f,input,0);
        for (const auto & cuts : {
                std::vector<size_t>{0,1024},
                std::vector<size_t>{0,512,1024},
                std::vector<size_t>{0,256,768,1024}}) {
            auto storage = span_storage::create(f,0,1024,1,cuts);
            if (!t.assert_true(storage && storage->plan)) continue;
            bool accepted = false;
            const auto actual = evaluate(f,ops,*storage,input,0,accepted);
            if (!t.assert_true(accepted)) continue;
            close_values(t,expected,actual,1e-7f);
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_fixed_spans_accept_exact_final_tail", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        block_inputs input(f,1023,1);
        const auto expected = ordinary(f,input,0);
        auto storage = span_storage::create(f,0,1023,1,{0,512,896,1023});
        if (!t.assert_true(storage && storage->plan)) return;
        bool accepted = false;
        const auto actual = evaluate(f,ops,*storage,input,0,accepted);
        if (!t.assert_true(accepted)) return;
        close_values(t,expected,actual,1e-7f);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg3_tg4_f16_mma_spans_match_contiguous_and_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace)) return;
        fixture f(backend.get(), true, GGML_TYPE_F16, GGML_TYPE_F16, 1025);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(257), size_t(513), size_t(1017)}) {
            for (uint32_t queries : {3u, 4u}) {
                std::vector<size_t> cuts{0, 1, 63, 64, 65, 127, 128, 255, 256, 257, active - 1, active};
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
                size_t contiguous_workspace = 0, segmented_workspace = 0;
                const auto a = evaluate(f, ops, *contiguous, input, 0, accepted, &contiguous_workspace);
                if (!t.assert_true(accepted)) continue;
                const std::vector<std::vector<size_t>> production_cuts{
                    {0,active/2,active},{0,active/3,2*active/3,active}};
                for (const auto & physical_cuts : production_cuts) {
                    auto fixed = span_storage::create(f,0,active,queries,physical_cuts);
                    if (!t.assert_true(fixed && fixed->plan)) continue;
                    const auto value = evaluate(f,ops,*fixed,input,0,accepted);
                    if (!t.assert_true(accepted)) continue;
                    close_values(t,a,value,1e-5f);
                    close_values(t,expected,value,1e-5f);
                }
                const auto b = evaluate(f, ops, *segmented, input, 0, accepted, &segmented_workspace);
                if (!t.assert_true(accepted)) continue;
                ggml_kv_stream_layout gathered;
                if (t.assert_true(ggml_kv_stream_layout_make(f.policy.shape, active, gathered).status ==
                        ggml_kv_stream_status::success)) {
                    t.assert_true(contiguous_workspace < gathered.bytes);
                    t.assert_true(segmented_workspace < gathered.bytes);
                }
                close_values(t, a, b, 1e-5f);
                close_values(t, expected, b, 1e-5f);
            }
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "q8_q4_mma_tile_bytes_equal_stock_conversion", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 8 && ops->convert && ops->convert_mma_rows)) return;
        for (ggml_type type : {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0}) {
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
            auto * source = ggml_new_tensor_2d(ctx.get(),type,256,7);
            auto * stock = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,256,7);
            auto * tile = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,256,7);
            for (auto * tensor : {source,stock,tile}) tensor->nb[2] = tensor->nb[1];
            ggml_set_input(source);
            ggml_set_output(stock);
            ggml_set_output(tile);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
            if (!t.assert_true(buffer != nullptr)) continue;
            std::vector<float> values(7*256);
            for (size_t i = 0; i < values.size(); ++i) values[i] = .25f*std::sin(float(i%509)*.031f);
            std::vector<uint8_t> packed(7*ggml_row_size(type,256));
            ggml_quantize_chunk(type,values.data(),packed.data(),0,7,256,nullptr);
            ggml_backend_tensor_set(source,packed.data(),0,packed.size());
            if (!t.assert_true(ops->supports_conversion(backend.get(),source,stock) &&
                    ops->convert(backend.get(),source,stock))) continue;
            if (!t.assert_true(ops->convert_mma_rows(backend.get(),source,tile))) continue;
            ggml_backend_synchronize(backend.get());
            std::vector<ggml_fp16_t> expected(7*256),actual(7*256);
            ggml_backend_tensor_get(stock,expected.data(),0,expected.size()*sizeof(ggml_fp16_t));
            ggml_backend_tensor_get(tile,actual.data(),0,actual.size()*sizeof(ggml_fp16_t));
            size_t mismatches = 0;
            for (size_t i = 0; i < expected.size(); ++i) mismatches += expected[i] != actual[i];
            t.out << ggml_type_name(type) << " tile mismatches = " << mismatches << " / " << expected.size() << "\n";
            t.assert_equal(size_t(0),mismatches);
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg3_tg4_q8_q4_tile_conversion_matches_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025);
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace && f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        size_t maximum_workspace = 0;
        for (size_t active : {size_t(257),size_t(513),size_t(1017)}) {
            for (uint32_t queries : {3u,4u}) {
                block_inputs input(f,active,queries);
                const auto expected = ordinary(f,input,0);
                const std::vector<std::vector<size_t>> layouts{
                    {0,active},
                    {0,active/2,active},
                    {0,active/4,3*active/4,active},
                    {0,1,127,128,255,256,active},
                };
                for (auto cuts : layouts) {
                    cuts.erase(std::remove_if(cuts.begin(),cuts.end(),
                        [&](size_t value) { return value > active; }),cuts.end());
                    std::sort(cuts.begin(),cuts.end());
                    cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
                    auto storage = span_storage::create(f,0,active,queries,cuts);
                    if (!t.assert_true(storage && storage->plan)) continue;
                    bool accepted = false;
                    size_t workspace = 0;
                    const auto actual = evaluate(f,ops,*storage,input,0,accepted,&workspace);
                    if (!t.assert_true(accepted)) continue;
                    maximum_workspace = std::max(maximum_workspace,workspace);
                    t.assert_true(workspace <= 1024*1024);
                    ggml_kv_stream_layout gathered;
                    auto f16 = f.policy.shape;
                    f16.type_k = f16.type_v = GGML_TYPE_F16;
                    if (t.assert_true(ggml_kv_stream_layout_make(f16,active,gathered).status ==
                            ggml_kv_stream_status::success)) {
                        t.assert_true(workspace < gathered.bytes);
                    }
                    close_values(t,expected,actual,2e-5f);
                }
            }
        }
        t.out << "maximum Q8/Q4 MMA workspace = " << maximum_workspace << " bytes\n";
        t.assert_true(maximum_workspace <= 1024*1024);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "quantized_mma_workspace_ceiling_is_context_independent", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192);
        if (!t.assert_true(ops != nullptr)) return;
        block_inputs input(f,8192,4);
        stock_attention_case stock(f,input);
        auto storage = span_storage::create(f,0,8192,4,{0,2048,6144,8192});
        if (!t.assert_true(storage && storage->plan)) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * node = ggml_flash_attn_ext(ctx.get(),input.q,stock.k,stock.v,input.mask,1.0f/16,0,0);
        if (!t.assert_true(node != nullptr)) return;
        ggml_flash_attn_ext_set_prec(node,GGML_PREC_F32);
        ggml_tensor op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        size_t bytes = 0;
        if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,bytes))) return;
        t.out << "8K Q8/Q4 MMA workspace = " << bytes << " bytes\n";
        t.assert_true(bytes <= 1024*1024);
        ggml_kv_stream_layout gathered;
        auto f16 = f.policy.shape;
        f16.type_k = f16.type_v = GGML_TYPE_F16;
        if (t.assert_true(ggml_kv_stream_layout_make(f16,8192,gathered).status ==
                ggml_kv_stream_status::success)) t.assert_true(bytes < gathered.bytes);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "quantized_mma_undersized_workspace_preserves_output", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        block_inputs input(f,513,4);
        auto storage = span_storage::create(f,0,513,4,{0,256,513});
        if (!t.assert_true(pin && storage && storage->plan)) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * node = f.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f/16);
        if (!t.assert_true(node != nullptr)) return;
        ggml_tensor op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        size_t bytes = 0;
        if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,bytes) && bytes > 1)) return;
        block_workspace tiny(f,bytes-1,101);
        t.assert_true(!ops->spans(
            backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(tiny.lease.get())));
        const auto actual = input.read();
        t.assert_true(std::all_of(actual.begin(),actual.end(),[](float value) { return value == -77; }));
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

    if (argc > 1 && (!std::strcmp(argv[1], "--cuda") ||
            !std::strcmp(argv[1], "--cuda-gqa6"))) t.test(
            "qwen_ratio_six_tg3_tg4_matches_stock", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 9 && ops->spans &&
                ops->spans_workspace && ops->mma_workspace)) return;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
            768, false, 2, 2, 16);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        size_t maximum = 0;
        if (!t.assert_true(ops->mma_workspace(backend.get(), GGML_TYPE_Q8_0,
                GGML_TYPE_Q4_0, 12, 2, 768, 2, maximum))) return;
        for (size_t active : {size_t(257), size_t(513)}) {
            for (uint32_t queries : {3u, 4u}) {
                block_inputs input(f, active, queries, false, 12);
                const auto expected = ordinary(f, input, 0);
                auto storage = span_storage::create(f, 0, active, queries,
                    {0, 256, active});
                if (!t.assert_true(bool(storage) && storage->plan)) return;
                bool accepted = false;
                size_t workspace = 0;
                const auto actual = evaluate(f, ops, *storage, input, 0,
                    accepted, &workspace);
                if (!t.assert_true(accepted)) return;
                t.assert_true(workspace <= maximum && maximum < 2*1048576);
                close_values(t, expected, actual, 2e-5f);
            }
        }
    });
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda") ||
            !std::strcmp(argv[1], "--cuda-gqa6"))) t.test(
            "qwen_24_4_tg2_aligned_and_wrapped_spans_are_exact", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->spans && ops->spans_workspace)) return;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192,false,2,4,80);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(8192),size_t(8191)}) {
            block_inputs input(f,active,2,false,24);
            const auto expected = stock_attention(f,input,0);
            const std::vector<std::vector<size_t>> cuts{
                {0,active},{0,1024,active},{0,1024,4096,active},
                {0,127,257,active}};
            for (size_t index = 0; index < cuts.size(); ++index) {
                for (bool wrapped : {false,true}) {
                    if (wrapped && cuts[index].size() != 4) continue;
                    auto storage = span_storage::create(f,0,active,2,cuts[index],wrapped);
                    if (!t.assert_true(storage && storage->plan)) return;
                    bool accepted = false;
                    const auto actual = evaluate(f,ops,*storage,input,0,accepted);
                    if (!t.assert_true(accepted)) return;
                    if (active == 8192 && index < 3)
                        t.assert_true(same_float_bits(expected,actual));
                    else close_values(t,expected,actual,1e-5f);
                }
            }
        }
    });
    return t.summary();
}
