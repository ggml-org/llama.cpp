#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-dispatch.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-plan.h"
#include "../ggml/src/ggml-cuda/kv-stream-span.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../src/llama-kv-stream-model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

#ifdef KV_STREAM_PASCAL_CUDA_TEST
bool kv_stream_test_is_sm61_only();
bool kv_stream_test_is_arch_only(int cc);
size_t kv_stream_test_shared_limit(ggml_backend_t backend, size_t limit);
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
int kv_stream_test_compiled_cc(ggml_backend_t backend);
bool kv_stream_test_dimension_instance(ggml_backend_t backend,ggml_type key,ggml_type value,
        uint32_t width,uint32_t columns,uint32_t groups,bool mma,bool resume,bool dense,bool report=false);
bool kv_stream_test_runtime_shared_instances();
bool kv_stream_test_native_tile(ggml_backend_t backend,ggml_tensor * op);
bool kv_stream_test_tile_plan(ggml_backend_t backend,const ggml_tensor * op,size_t tokens,
        ggml_kv_stream_resume_plan & plan,size_t & bound);
bool kv_stream_test_tile_run(ggml_backend_t backend,const ggml_tensor * op,const ggml_kv_stream_span_plan_view & view,
        ggml_backend_buffer_t workspace,const ggml_kv_stream_resume_plan & plan,size_t first,size_t end);

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

struct tg1_only_test_ops {
    inline static tg1_only_test_ops * active = nullptr;
    ggml_backend_reg_t reg;
    decltype(ggml_backend_reg_i::get_proc_address) original;
    ggml_kv_stream_partial_ops limited;
    decltype(ggml_kv_stream_partial_ops::resume_plan) original_plan;

    // Retain the TG1-only handoff regression even after the real backend gains TG2 support.
    tg1_only_test_ops(ggml_backend_reg_t reg, const ggml_kv_stream_partial_ops & ops) :
        reg(reg), original(reg->iface.get_proc_address), limited(ops), original_plan(ops.resume_plan) {
        GGML_ASSERT(!active); active = this;
        // Exercise the older vector-only provider, not the new four-width workspace capability.
        limited.version=9; limited.decode_workspace=nullptr;
        limited.resume_plan = [](ggml_backend_t backend, int32_t k, int32_t v,
                uint32_t heads, uint32_t kv_heads, uint32_t queries, size_t tokens, ggml_kv_stream_resume_plan & plan) {
            return queries == 1 && active->original_plan(backend,k,v,heads,kv_heads,queries,tokens,plan);
        };
        reg->iface.get_proc_address = [](ggml_backend_reg_t reg, const char * name) -> void * {
            if (!std::strcmp(name,"ggml_backend_kv_stream_partial_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_partial_ops * { return &active->limited; });
            return active->original(reg,name);
        };
    }
    ~tg1_only_test_ops() { reg->iface.get_proc_address = original; active = nullptr; }
    tg1_only_test_ops(const tg1_only_test_ops &) = delete;
    tg1_only_test_ops & operator=(const tg1_only_test_ops &) = delete;
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
                tensor->ne[0] = tensor == &k ? f.policy.shape.head_dim_k : f.policy.shape.head_dim_v;
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
        size_t * used_workspace = nullptr, float scale = 1.0f/16) {
    ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
    auto * node = f.resident->attention(ctx.get(), layer, input.q, input.mask, input.active, scale);
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
    ggml_tensor * output = nullptr;

    stock_attention_case(fixture & f, block_inputs & input) : f(f), context(ggml_init({65536, nullptr, true})) {
        const int64_t heads=f.policy.shape.heads;
        const int64_t dk=f.policy.shape.head_dim_k, dv=f.policy.shape.head_dim_v;
        k_storage = ggml_new_tensor_2d(context.get(), ggml_type(f.policy.shape.type_k), dk*heads, input.padded);
        v_storage = ggml_new_tensor_2d(context.get(), ggml_type(f.policy.shape.type_v), dv*heads, input.padded);
        k = ggml_view_3d(context.get(), k_storage, dk, input.padded, heads,
            ggml_row_size(k_storage->type, dk*heads), ggml_row_size(k_storage->type, dk), 0);
        v = ggml_view_3d(context.get(), v_storage, dv, input.padded, heads,
            ggml_row_size(v_storage->type, dv*heads), ggml_row_size(v_storage->type, dv), 0);
        output = ggml_flash_attn_ext(context.get(), input.q, k, v, input.mask, 1.0f/16, 0, 0);
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
static int benchmark_spans(bool wide = false, ggml_type type_k = GGML_TYPE_Q8_0, ggml_type type_v = GGML_TYPE_Q4_0,
        size_t context_tokens = 0, uint32_t query_tokens = 0) {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    GGML_ASSERT(dev);
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    GGML_ASSERT(ops && ops->version >= 7 && ops->spans && ops->spans_workspace);
    const auto contexts=context_tokens ? std::vector<size_t>{context_tokens} :
        wide ? std::vector<size_t>{2048,8192,16384,65536} : std::vector<size_t>{16384,65536,131072,131073};
    const auto query_counts=query_tokens ? std::vector<uint32_t>{query_tokens} :
        wide ? std::vector<uint32_t>{1,2,3,4,8,256} : std::vector<uint32_t>{1,2,3,4,5,6,7,8};
    for (size_t active : contexts) {
        for (uint32_t queries : query_counts) {
            const uint32_t width=wide ? 512 : 256;
            fixture f(backend.get(), true, type_k, type_v, (active+255)/256*256,false,2,4,0,width,width);
            GGML_ASSERT(f.attach());
            auto pin = f.binding->acquire();
            GGML_ASSERT(pin);
            block_inputs input(f, active, queries,false,wide ? 32 : 24);
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
                stock.run();
                std::vector<float> expected(ggml_nelements(stock.output));
                ggml_backend_tensor_get(stock.output,expected.data(),0,ggml_nbytes(stock.output));
                spans.run();
                if (!same_float_bits(expected,input.read())) {
                    std::fprintf(stderr,"span benchmark numerical mismatch: active=%zu TG%u spans=%zu\n",active,queries,cuts.size()-1);
                    return 1;
                }
                const double span_us = average_us([&] { spans.run(); });
                if (wide) std::printf("wide-span-format,k=%s,v=%s,width=%u\n",ggml_type_name(type_k),ggml_type_name(type_v),width);
                std::printf(
                    "span-bench,active=%zu,tg=%u,spans=%zu,stock_us=%.3f,span_us=%.3f,overhead_pct=%.2f\n",
                    active, queries, cuts.size()-1, stock_us, span_us, 100.0*(span_us/stock_us-1.0));
            }
        }
    }
    return 0;
}

// Compare the tile decoder with the native whole-plane converter at the requested head width.
static void check_mma_tile_conversion(testing & t, int width) {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    if (!t.assert_true(ops && ops->version >= 8 && ops->convert && ops->convert_mma_rows)) return;
    for (ggml_type type : {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_F32}) {
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * source = ggml_new_tensor_2d(ctx.get(),type,width,7);
        auto * stock = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,width,7);
        auto * tile = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,width,7);
        for (auto * tensor : {source,stock,tile}) tensor->nb[2] = tensor->nb[1];
        ggml_set_input(source);
        ggml_set_output(stock);
        ggml_set_output(tile);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(buffer != nullptr)) continue;
        std::vector<float> values(7*width);
        for (size_t i = 0; i < values.size(); ++i) values[i] = .25f*std::sin(float(i%509)*.031f);
        std::vector<uint8_t> packed(7*ggml_row_size(type,width));
        ggml_quantize_chunk(type,values.data(),packed.data(),0,7,width,nullptr);
        ggml_backend_tensor_set(source,packed.data(),0,packed.size());
        if (!t.assert_true(ops->supports_conversion(backend.get(),source,stock) &&
                ops->convert(backend.get(),source,stock))) continue;
        if (!t.assert_true(ops->convert_mma_rows(backend.get(),source,tile))) continue;
        ggml_backend_synchronize(backend.get());
        std::vector<ggml_fp16_t> expected(7*width),actual(7*width);
        ggml_backend_tensor_get(stock,expected.data(),0,expected.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(tile,actual.data(),0,actual.size()*sizeof(ggml_fp16_t));
        size_t mismatches = 0;
        for (size_t i = 0; i < expected.size(); ++i) mismatches += expected[i] != actual[i];
        t.out << ggml_type_name(type) << " tile mismatches = " << mismatches << " / " << expected.size() << "\n";
        t.assert_equal(size_t(0),mismatches);
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-resources")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("wide_native_plans_load_repeatedly_without_changing_resource_identity", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            for (const auto & pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0),
                    std::make_pair(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1),std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)})
                for (const auto & geometry : {std::make_pair(1u,8u),std::make_pair(2u,8u),std::make_pair(4u,8u),
                        std::make_pair(8u,8u),std::make_pair(4u,2u),std::make_pair(8u,2u)})
                    for (bool resume : {false,true}) for (bool dense : {false,true}) {
                t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,
                    geometry.first,geometry.second,true,resume,dense,true));
                t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,
                    geometry.first,geometry.second,true,resume,dense));
            }
            t.assert_true(kv_stream_test_runtime_shared_instances());
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-tile")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("head512_tile_single_and_progressive_waves_match_native", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (auto key : formats) for (auto value : formats) {
                constexpr size_t active=2049;
                fixture f(backend.get(),true,key,value,active,false,2,4,32,512,512);
                if (!t.assert_true(f.attach())) return;
                for (uint32_t queries : {1u,2u,3u,4u,5u,6u,7u,8u,9u,16u,17u,129u,256u}) {
                    block_inputs input(f,active,queries,false,32);
                    const auto expected=stock_attention(f,input,0,1.f/16,kv_stream_test_native_tile);
                    auto storage=span_storage::create(f,0,active,queries,{0,256,768,active},true);
                    ggml_kv_stream_span_plan_view view;
                    if (!t.assert_true(storage && ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
                    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
                    auto * node=f.resident->attention(ctx.get(),0,input.q,input.mask,active,1.f/16);
                    if (!t.assert_true(node != nullptr)) return;
                    auto op=*node; op.data=input.output->data; op.buffer=input.output->buffer;
                    ggml_kv_stream_resume_plan plan;
                    size_t bound=0;
                    if (!t.assert_true(kv_stream_test_tile_plan(backend.get(),&op,input.padded,plan,bound))) return;
                    t.assert_true(plan.head_dim_k == 512 && plan.head_dim_v == 512 && plan.span_resume);
                    t.assert_true(plan.kernel_config[2]*plan.kernel_config[3] <= 16);
                    t.assert_true(plan.bytes <= bound);
                    for (size_t history : {size_t(4096),size_t(65536),size_t(262144)}) {
                        ggml_kv_stream_resume_plan other;
                        size_t other_bound=0;
                        t.assert_true(kv_stream_test_tile_plan(backend.get(),&op,history,other,other_bound));
                        t.assert_equal(bound,other_bound);
                        t.assert_true(other.bytes <= bound);
                    }
                    block_workspace workspace(f,bound,103);
                    auto * scratch=ggml_backend_memory_lease_buffer(workspace.lease.get());
                    auto * bounds=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,4,queries);
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
                    if (!t.assert_true(bool(buffer))) return;
                    std::vector<int32_t> intervals(4*queries);
                    for (uint32_t row=0;row<queries;++row) {
                        intervals[4*row]=0; intervals[4*row+1]=int32_t(active-queries+row+1);
                        intervals[4*row+2]=int32_t(input.padded); intervals[4*row+3]=1;
                    }
                    ggml_backend_tensor_set(bounds,intervals.data(),0,ggml_nbytes(bounds));
                    for (bool compact : {false,true}) {
                        auto current=op;
                        if (compact) {
                            current.src[3]=nullptr;
                            if (!t.assert_true(ggml_backend_execution_set_mask_bounds(&current,bounds))) return;
                        }
                        const auto label=std::string(ggml_type_name(key))+"/"+ggml_type_name(value)+" TG"+std::to_string(queries)+
                            (compact ? " compact" : " dense");
                        if (!t.assert_true(kv_stream_test_tile_run(backend.get(),&current,view,scratch,plan,0,plan.tokens))) return;
                        ggml_backend_synchronize(backend.get());
                        t.assert_true(label+" complete stock bits",same_float_bits(expected,input.read()));
                        std::vector<float> sentinel(expected.size(),-77);
                        ggml_backend_tensor_set(input.output,sentinel.data(),0,ggml_nbytes(input.output));
                        auto first=view; first.count=2;
                        if (!t.assert_true(kv_stream_test_tile_run(backend.get(),&current,first,scratch,plan,0,768))) return;
                        ggml_backend_synchronize(backend.get());
                        t.assert_true(label+" no early publication",same_float_bits(sentinel,input.read()));
                        auto second=view; second.spans+=2; second.count=1;
                        if (!t.assert_true(kv_stream_test_tile_run(backend.get(),&current,second,scratch,plan,768,plan.tokens))) return;
                        ggml_backend_synchronize(backend.get());
                        t.assert_true(label+" resumed stock bits",same_float_bits(expected,input.read()));
                        block_workspace short_workspace(f,plan.bytes-1,107);
                        t.assert_true(!kv_stream_test_tile_run(backend.get(),&current,view,
                            ggml_backend_memory_lease_buffer(short_workspace.lease.get()),plan,0,plan.tokens));
                        t.assert_true(same_float_bits(expected,input.read()));
                    }
                }
            }
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-convert")) {
        testing t;
        t.test("head512_tile_conversion_matches_native", [](testing & t) { check_mma_tile_conversion(t,512); });
        return t.summary();
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-spans")) {
        testing t;
        t.test("head512_spans_preserve_stock_order_for_target_verification", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(ops && ops->spans && ops->spans_workspace)) return;
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (auto key : formats) for (auto value : formats) {
                const auto pair=std::make_pair(key,value);
                constexpr size_t active=8193;
                fixture f(backend.get(),true,pair.first,pair.second,active,false,2,4,82,512,512);
                if (!t.assert_true(f.attach())) return;
                for (uint32_t queries : {1u,2u,3u,4u}) for (bool masked : {false,true}) {
                    block_inputs input(f,active,queries,masked,32);
                    const auto expected=stock_attention(f,input,0,1.0f);
                    for (const auto & cuts : {std::vector<size_t>{0,active},std::vector<size_t>{0,3328,active},
                            std::vector<size_t>{0,3328,6400,active},std::vector<size_t>{0,127,257,active}}) {
                        for (bool wrapped : {false,true}) {
                            if (wrapped && cuts.size() != 4) continue;
                            auto spans=span_storage::create(f,0,active,queries,cuts,wrapped);
                            if (!t.assert_true(bool(spans))) return;
                            bool accepted=false;
                            const auto actual=evaluate(f,ops,*spans,input,0,accepted,nullptr,1.0f);
                            if (!t.assert_true(accepted)) return;
                            t.assert_true(same_float_bits(expected,actual));
                        }
                    }
                }
            }
        });
        t.test("head512_rejects_short_workspace_and_invalid_metadata_without_writes", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1025,false,2,4,32,512,512);
            if (!t.assert_true(ops && f.attach())) return;
            block_inputs input(f,1025,3,false,32);
            auto storage=span_storage::create(f,0,1025,3,{0,256,768,1025},true);
            if (!t.assert_true(bool(storage))) return;
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
            auto * node=f.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f);
            if (!t.assert_true(node != nullptr)) return;
            auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
            size_t required=0;
            if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,required) && required > 1)) return;
            block_workspace short_workspace(f,required-1,79);
            auto * short_buffer=ggml_backend_memory_lease_buffer(short_workspace.lease.get());
            t.assert_equal(required-1,ggml_backend_buffer_get_size(short_buffer));
            const auto before=input.read();
            t.assert_true(ops->supports(backend.get(),&op));
            t.assert_true(!ops->partial(backend.get(),&op,short_buffer,false));
            t.assert_true(!ops->spans(backend.get(),&op,storage->plan,short_buffer));
            t.assert_true(same_float_bits(before,input.read()));
            auto invalid=op;
            invalid.ne[0]=256;
            size_t unchanged=77;
            t.assert_true(!ops->spans_workspace(backend.get(),&invalid,storage->plan,unchanged));
            t.assert_equal(size_t(77),unchanged);
            t.assert_true(same_float_bits(before,input.read()));
        });
        return t.summary();
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-compiled")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("compiled_wide_stock_family_instances_load_on_the_real_device", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            t.assert_true(kv_stream_test_runtime_shared_instances());
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (const auto key : formats) for (const auto value : formats)
                for (const auto geometry : {std::make_pair(1u,8u),std::make_pair(2u,8u),std::make_pair(4u,8u),
                        std::make_pair(8u,8u),std::make_pair(4u,2u),std::make_pair(8u,2u)})
                    for (bool resume : {false,true}) for (bool dense : {false,true})
                        t.assert_true(kv_stream_test_dimension_instance(backend.get(),key,value,512,
                            geometry.first,geometry.second,true,resume,dense));
            for (const auto pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),
                    std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0),
                    std::make_pair(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1),
                    std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)}) {
                for (uint32_t columns : {1u,2u,4u,8u}) for (bool resume : {false,true}) for (bool dense : {false,true})
                    t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,columns,8,true,resume,dense));
                for (uint32_t columns : {2u,4u,8u})
                    t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,columns,1,false,false,false));
            }
            t.assert_true(!kv_stream_test_dimension_instance(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL,512,1,8,true,true,false));
            t.assert_true(!kv_stream_test_dimension_instance(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,384,1,8,true,true,false));
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-geometry")) {
        testing t;
        t.test("dimension_aware_backend_prepares_full_attention_queries", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(ops && ops->version >= 15 && ops->geometry)) return;
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (auto key : formats) for (auto value : formats) {
                const auto pair=std::make_pair(key,value);
                for (uint32_t width : {1u,2u,3u,4u,256u}) {
                    const ggml_kv_stream_attention_request request{{pair.first,pair.second,512,512,4,256,128},32,width,262144};
                    ggml_kv_stream_resume_plan plan;
                    t.assert_true(ggml_kv_stream_resume_plan_query(ops,backend.get(),request,plan));
                    t.assert_equal(uint32_t(512),plan.head_dim_k);
                    t.assert_equal(uint32_t(512),plan.head_dim_v);
                    size_t bytes=0;
                    t.assert_true(ggml_kv_stream_workspace_query(ops,backend.get(),request,
                        width <= 4 ? ggml_kv_stream_workspace_kind::decode : ggml_kv_stream_workspace_kind::prefill,bytes));
                    t.assert_true(bytes > 0);
                }
            }
            const ggml_kv_stream_attention_request valid{{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,512,512,4,256,128},32,4,8192};
            for (int defect=0;defect<6;++defect) {
                auto request=valid;
                if (defect == 0) request.shape.head_dim_k=384;
                if (defect == 1) request.shape.head_dim_v=256;
                if (defect == 2) request.shape.type_v=GGML_TYPE_IQ4_NL;
                if (defect == 3) request.query_heads=31;
                if (defect == 4) request.shape.page_tokens=128;
                if (defect == 5) request.query_tokens=0;
                ggml_kv_stream_resume_plan untouched;
                untouched.bytes=77; untouched.head_dim_k=123;
                t.assert_true(!ggml_kv_stream_resume_plan_query(ops,backend.get(),request,untouched));
                t.assert_equal(size_t(77),untouched.bytes);
                t.assert_equal(uint32_t(123),untouched.head_dim_k);
                size_t bytes=77;
                t.assert_true(!ggml_kv_stream_workspace_query(ops,backend.get(),request,
                    ggml_kv_stream_workspace_kind::decode,bytes));
                t.assert_equal(size_t(77),bytes);
            }
        });
        return t.summary();
    }
    const int consumer=argc > 1 && !std::strcmp(argv[1],"--cuda-sm75") ? 750 :
        argc > 1 && !std::strcmp(argv[1],"--cuda-sm86") ? 860 :
        argc > 1 && !std::strcmp(argv[1],"--cuda-sm89") ? 890 :
        argc > 1 && !std::strcmp(argv[1],"--cuda-sm120") ? 1200 : 0;
    if (argc > 1 && !std::strncmp(argv[1],"--cuda-sm",9) && !consumer) return 2;
    ggml_backend_ptr consumer_backend;
#ifdef KV_STREAM_PASCAL_CUDA_TEST
    if (consumer) {
        if (!kv_stream_test_is_arch_only(consumer)) return 77;
        ggml_backend_load_all(); auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!dev) return 77;
        consumer_backend.reset(ggml_backend_dev_init(dev,nullptr));
        if (!consumer_backend || kv_stream_test_compiled_cc(consumer_backend.get()) != consumer) return 77;
    }
    const int previous=consumer ? kv_stream_test_override_cc(consumer_backend.get(),consumer) : 0;
    struct restore_arch {ggml_backend_t backend; int cc; ~restore_arch(){if (backend) kv_stream_test_override_cc(backend,cc);}} restore_consumer{consumer_backend.get(),previous};
#else
    if (consumer) return 77;
#endif
    const bool tg2 = argc > 1 && (!std::strcmp(argv[1],"--cuda-pascal-tg2") || !std::strcmp(argv[1],"--cuda-tg2"));
    const bool pascal = argc > 1 && (!std::strcmp(argv[1],"--cuda-pascal-tg1") || !std::strcmp(argv[1],"--cuda-pascal-tg2"));
    if (pascal) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        if (!kv_stream_test_is_sm61_only()) {
            std::puts("SKIP: simulated Pascal vector attention requires an SM61-only build, not a mixed-target binary");
            return 77;
        }
#else
        std::puts("SKIP: CUDA test adapter is not compiled");
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) return benchmark_spans();
    if (argc > 1 && !std::strcmp(argv[1], "--bench-gemma")) {
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        for (const auto & pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0),
                std::make_pair(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1),std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)}) {
            if (benchmark_spans(true,pair.first,pair.second)) return 1;
        }
        return 0;
    }
    if (argc == 5 && !std::strcmp(argv[1], "--bench-gemma-focus")) {
        char * end=nullptr;
        const auto tokens=std::strtoul(argv[2],&end,10);
        if (*end || tokens < 256 || tokens > 262144 || tokens%256) return 2;
        const auto queries=std::strtoul(argv[3],&end,10);
        if (*end || !queries || queries > 256 || queries > tokens) return 2;
        const bool f16=!std::strcmp(argv[4],"f16-f16");
        if (!f16 && std::strcmp(argv[4],"q8-q4")) return 2;
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        return benchmark_spans(true,f16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0,
            f16 ? GGML_TYPE_F16 : GGML_TYPE_Q4_0,size_t(tokens),uint32_t(queries));
    }
    testing t;
    t.test("dimensioned_inputs_preserve_independent_key_and_value_rows", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        fixture f(backend.get(),false,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,16,512,256);
        block_inputs input(f,513,4,false,32);
        t.assert_equal(int64_t(512),input.q->ne[0]);
        t.assert_equal(int64_t(256),input.output->ne[0]);
        t.assert_equal(size_t(4*32*512),input.qdata.size());
        t.assert_equal(size_t(4*32*256),input.read().size());
        t.assert_equal(size_t(544),f.host->layout().k_row_bytes);
        t.assert_equal(size_t(144),f.host->layout().v_row_bytes);
    });
    const bool quant_matrix = argc > 1 && !std::strcmp(argv[1], "--cuda-quant-matrix");
    const bool float32_only = argc > 1 && !std::strcmp(argv[1], "--cuda-f32");
    if (quant_matrix) t.set_filter("compiled_cache_pairs_execute_stock_order_spanned_attention");
    if (float32_only) t.set_filter("float32_storage_uses_bounded_stock_equivalent_attention|float32_ring_waves_preserve_native_conversion_and_resume_order");
    if (quant_matrix || (argc > 1 && !std::strcmp(argv[1], "--cuda"))) t.test(
            "compiled_cache_pairs_execute_stock_order_spanned_attention", [&](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->capabilities && ops->spans_workspace && ops->spans)) return;
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1};
        size_t pairs = 0;
        for (auto key : types) for (auto value : types) {
            if (!ops->capabilities(backend.get(), key, value).direct_pair) continue;
            fixture f(backend.get(), true, key, value, 1025, false, 2, 4, 16);
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            if (!t.assert_true(bool(pin))) return;
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                block_inputs input(f, 513, rows, false, 24);
                const auto expected = stock_attention(f, input, 0);
                auto spans = span_storage::create(f, 0, 513, rows, {0, 256, 512, 513}, true);
                ggml_context_ptr context(ggml_init({65536, nullptr, true}));
                auto * node = f.resident->attention(context.get(), 0, input.q, input.mask, input.active, 1.0f/16);
                if (!t.assert_true(spans && node)) return;
                ggml_tensor op = *node;
                op.buffer = input.output->buffer; op.data = input.output->data;
                size_t bytes = 0;
                if (!t.assert_true(ops->spans_workspace(backend.get(), &op, spans->plan, bytes))) return;
                block_workspace workspace(f, bytes, 371);
                if (!t.assert_true(ops->spans(backend.get(), &op, spans->plan,
                        ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                ggml_backend_synchronize(backend.get());
                const auto actual = input.read();
                close_values(t, expected, actual, 1e-6f);
                if (rows >= 3) t.assert_true(same_float_bits(expected, actual));
            }
            ++pairs;
        }
        t.out << "qualified compiled cache pairs=" << pairs << '\n';
        if (quant_matrix) t.assert_equal(size_t(64), pairs);
        else t.assert_true(pairs > 0);
    });
    if (argc > 1 && (!std::strcmp(argv[1],"--cuda-prepared-decode") || !std::strcmp(argv[1],"--cuda")))
        t.test("decode_preflight_covers_shortened_token_major_mma_batches", [](testing & t) {
            ggml_backend_load_all();
            auto * device=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(device != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
            if (!t.assert_true(bool(backend))) return;
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(ops && ops->decode_workspace && ops->spans_workspace && ops->spans)) return;
            for (auto pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)}) {
                size_t bytes=0;
                if (!t.assert_true(ops->decode_workspace(backend.get(),pair.first,pair.second,24,4,8,8192,bytes))) return;
                fixture f(backend.get(),true,pair.first,pair.second,8192,false,2,4,80);
                if (!t.assert_true(f.attach())) return;
                auto pin=f.binding->acquire();
                if (!t.assert_true(bool(pin))) return;
                block_workspace workspace(f,bytes,605);
                for (uint32_t queries : {3u,4u,5u,6u,7u,8u}) {
                    block_inputs input(f,8191,queries,false,24);
                    const auto expected=stock_attention(f,input,0);
                    for (bool wrapped : {false,true}) {
                        auto storage=span_storage::create(f,0,input.padded,queries,
                            wrapped ? std::vector<size_t>{0,256,768,input.padded} : std::vector<size_t>{0,input.padded},wrapped);
                        if (!t.assert_true(storage && storage->plan)) return;
                        ggml_context_ptr context(ggml_init({65536,nullptr,true}));
                        auto * node=f.resident->attention(context.get(),0,input.q,input.mask,input.active,1.0f/16);
                        if (!t.assert_true(node != nullptr)) return;
                        ggml_tensor op=*node;
                        op.buffer=input.output->buffer; op.data=input.output->data;
                        for (bool stock_conversion : {false,true}) {
                            ggml_backend_execution_set_stock_conversion(&op,stock_conversion);
                            size_t needed=0;
                            if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,needed))) return;
                            t.assert_true(needed <= bytes);
                            if (!t.assert_true(ops->spans(backend.get(),&op,storage->plan,
                                    ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                            ggml_backend_synchronize(backend.get());
                            t.assert_true(same_float_bits(expected,input.read()));
                        }
                    }
                }
            }
        });
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda-wide-spans") || !std::strcmp(argv[1], "--cuda")))
        t.test("wide_mma_dispatch_matches_stock_for_resident_and_wrapped_spans", [](testing & t) {
            ggml_backend_load_all();
            auto * dev = ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
            const auto * ops = get ? get() : nullptr;
            if (!t.assert_true(ops && ops->spans_workspace && ops->spans)) return;
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192,false,2,4,80);
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            for (size_t active : {size_t(513),size_t(8191)}) for (uint32_t queries : {5u,6u,7u,8u})
                    for (uint32_t ratio : {2u,6u,8u}) {
                block_inputs input(f, active, queries, false, ratio*4);
                const auto expected = stock_attention(f, input, 0);
                for (bool wrapped : {false,true}) {
                    auto storage = span_storage::create(f,0,active,queries,
                        wrapped ? std::vector<size_t>{0,256,512,active} : std::vector<size_t>{0,active},wrapped);
                    if (!t.assert_true(storage && storage->plan)) return;
                    ggml_context_ptr context(ggml_init({65536,nullptr,true}));
                    auto * ks = ggml_new_tensor_2d(context.get(),GGML_TYPE_Q8_0,256*4,input.padded);
                    auto * vs = ggml_new_tensor_2d(context.get(),GGML_TYPE_Q4_0,256*4,input.padded);
                    auto * k = ggml_view_3d(context.get(),ks,256,input.padded,4,
                        ggml_row_size(ks->type,256*4),ggml_row_size(ks->type,256),0);
                    auto * v = ggml_view_3d(context.get(),vs,256,input.padded,4,
                        ggml_row_size(vs->type,256*4),ggml_row_size(vs->type,256),0);
                    auto * op = ggml_flash_attn_ext(context.get(),input.q,k,v,input.mask,1.0f/16,0,0);
                    op->buffer = input.output->buffer;
                    op->data = input.output->data;
                    size_t bytes = 0;
                    if (!t.assert_true(ops->spans_workspace(backend.get(),op,storage->plan,bytes))) return;
                    block_workspace exact(f,bytes,601), short_grant(f,bytes-1,602);
                    const auto sentinel = input.read();
                    t.assert_true(!ops->spans(backend.get(),op,storage->plan,ggml_backend_memory_lease_buffer(short_grant.lease.get())));
                    t.assert_true(same_float_bits(sentinel,input.read()));
                    if (!t.assert_true(ops->spans(backend.get(),op,storage->plan,ggml_backend_memory_lease_buffer(exact.lease.get())))) return;
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(expected,input.read()));
                }
            }
        });
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda-wide-plan") || !std::strcmp(argv[1], "--cuda")))
        t.test("wide_query_plans_match_compiled_pairs_and_preserve_rejected_output", [](testing & t) {
            using status = ggml_cuda_kv_stream_plan_status;
            using family = ggml_cuda_kv_stream_kernel_family;
            using style = ggml_cuda_kv_stream_execution_style;
            using query_t = status (*)(ggml_backend_t, const ggml_tensor *, const ggml_kv_stream_span_plan_view *, style, ggml_cuda_kv_stream_attention_plan &);
            ggml_backend_load_all();
            auto * dev = ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            auto * reg = ggml_backend_dev_backend_reg(dev);
            auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_attention_plan"));
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops"));
            ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
            if (!t.assert_true(backend && query && get && get() && get()->decode_workspace)) return;
            for (auto pair : {std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q5_0, GGML_TYPE_Q4_0}}) for (uint32_t width = 5; width <= 9; ++width) {
                ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
                auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, width, 24);
                auto * ks = ggml_new_tensor_2d(ctx.get(), pair.first, 256*4, 768);
                auto * vs = ggml_new_tensor_2d(ctx.get(), pair.second, 256*4, 768);
                auto * k = ggml_view_3d(ctx.get(), ks, 256, 768, 4,
                    ggml_row_size(pair.first, 256*4), ggml_row_size(pair.first, 256), 0);
                auto * v = ggml_view_3d(ctx.get(), vs, 256, 768, 4,
                    ggml_row_size(pair.second, 256*4), ggml_row_size(pair.second, 256), 0);
                auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 768, width);
                auto * op = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/16, 0, 0);
                ggml_kv_stream_span span{};
                span.tokens = 513;
                ggml_kv_stream_span_plan_view view{{pair.first, pair.second, 256, 256, 4, 256, 128}, &span, 1, 513, width};
                ggml_cuda_kv_stream_attention_plan native;
                t.assert_true(query(backend.get(), op, nullptr, style::native, native) == status::success);
                t.assert_true(native.family == family::mma);
                t.assert_equal(int64_t(width), native.metadata.queries);
                auto unchanged = native;
                const bool available=get()->capabilities(backend.get(),pair.first,pair.second).direct_pair;
                const auto expected=available ? status::success : status::missing_streamed_implementation;
                t.assert_true(query(backend.get(), op, &view, style::spanned, unchanged) == expected);
                if (expected != status::success) t.assert_equal(native.output_allocation_bytes, unchanged.output_allocation_bytes);
                const auto spanned = unchanged;
                t.assert_true(query(backend.get(), op, &view, style::resumable, unchanged) == expected);
                if (expected == status::success)
                    t.assert_true(unchanged.requirements.scratch_bytes >= spanned.requirements.scratch_bytes);
                const auto saved=unchanged;
                size_t bytes = 77;
                const bool decode_available=available && width <= 8;
                t.assert_true(get()->decode_workspace(backend.get(), pair.first, pair.second, 24, 4, width, 768, bytes) == decode_available);
                if (!decode_available) t.assert_equal(size_t(77), bytes);
                ++q->nb[1];
                t.assert_true(query(backend.get(), op, &view, style::spanned, unchanged) == status::unsupported_geometry);
                --q->nb[1];
                auto stale = view;
                ++stale.query_tokens;
                t.assert_true(query(backend.get(), op, &stale, style::spanned, unchanged) == status::invalid_metadata);
                t.assert_equal(saved.output_allocation_bytes, unchanged.output_allocation_bytes);
            }
        });
#ifdef KV_STREAM_PASCAL_CUDA_TEST
    if (consumer || (argc > 1 && !std::strcmp(argv[1],"--cuda-resource-probe"))) t.test(
            "mma_span_resources_are_rejected_before_launch_and_requirements_are_preserved",[&](testing & t) {
        ggml_backend_load_all();
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        using status=ggml_cuda_kv_stream_plan_status;
        using style=ggml_cuda_kv_stream_execution_style;
        using query_t=status (*)(ggml_backend_t,const ggml_tensor *,const ggml_kv_stream_span_plan_view *,style,ggml_cuda_kv_stream_attention_plan &);
        auto * reg=ggml_backend_dev_backend_reg(dev);
        auto query=reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_kv_stream_attention_plan"));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,768,false,2,4,16);
        if (!t.assert_true(query && get && get() && f.attach())) return;
        auto pin=f.binding->acquire(); block_inputs input(f,513,4,false,24);
        auto storage=span_storage::create(f,0,513,4,{0,256,512,513},true);
        ggml_context_ptr context(ggml_init({65536,nullptr,true}));
        auto * node=f.resident->attention(context.get(),0,input.q,input.mask,input.active,1.0f/16);
        if (!t.assert_true(storage && node)) return;
        auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
        ggml_cuda_kv_stream_attention_plan plan;
        if (!t.assert_true(query(backend.get(),&op,&view,style::spanned,plan) == status::success)) return;
        if (!t.assert_true(plan.family == ggml_cuda_kv_stream_kernel_family::mma &&
                plan.requirements.shared_bytes > 3*sizeof(ggml_cuda_kv_span))) return;
        const auto saved=plan;
        block_workspace workspace(f,plan.requirements.scratch_bytes);
        const size_t previous=kv_stream_test_shared_limit(backend.get(),plan.requirements.shared_bytes-3*sizeof(ggml_cuda_kv_span));
        struct restore {ggml_backend_t backend; size_t limit; ~restore(){kv_stream_test_shared_limit(backend,limit);}} restore_limit{backend.get(),previous};
        t.assert_true(query(backend.get(),&op,&view,style::spanned,plan) == status::unsupported_launch_resources);
        t.assert_equal(saved.requirements.scratch_bytes,plan.requirements.scratch_bytes);
        size_t untouched=77;
        t.assert_true(!get()->spans_workspace(backend.get(),&op,storage->plan,untouched));
        t.assert_equal(size_t(77),untouched);
        t.assert_true(!get()->spans(backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(workspace.lease.get())));
        const auto output=input.read();
        t.assert_true(std::all_of(output.begin(),output.end(),[](float v){return v == -77;}));
        ggml_cuda_kv_stream_attention_plan native;
        t.assert_true(query(backend.get(),&op,nullptr,style::native,native) == status::success);
        t.assert_true(native.requirements.output_extra_bytes > 0);
        ggml_backend_execution_set_external_workspace(&op,true);
        t.assert_true(!get()->direct(backend.get(),&op));
    });
    if (consumer) t.test("consumer_rejects_unsupported_mma_geometry_and_nonpadded_workspace_probes",[&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(get && get() && get()->mma_workspace)) return;
        size_t untouched=77;
        for (auto pair : {std::pair{GGML_TYPE_Q5_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q8_0}}) {
            t.assert_true(!get()->mma_workspace(backend.get(),pair.first,pair.second,12,2,768,3,untouched));
            t.assert_equal(size_t(77),untouched);
        }
        t.assert_true(!get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8,2,768,3,untouched));
        t.assert_true(!get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,12,2,767,3,untouched));
        t.assert_equal(size_t(77),untouched);
    });
    if (consumer) t.test("consumer_tg1_tg4_stock_family_scratch_and_span_equivalence",[&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        using status=ggml_cuda_kv_stream_plan_status;
        using style=ggml_cuda_kv_stream_execution_style;
        using family=ggml_cuda_kv_stream_kernel_family;
        using query_t=status (*)(ggml_backend_t,const ggml_tensor *,const ggml_kv_stream_span_plan_view *,style,ggml_cuda_kv_stream_attention_plan &);
        auto * reg=ggml_backend_dev_backend_reg(dev);
        auto query=reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_kv_stream_attention_plan"));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(query && get && get())) return;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192,false,2,2,80);
        if (!t.assert_true(f.attach())) return;
        auto pin=f.binding->acquire();
        for (size_t active : {size_t(513),size_t(768),size_t(8191),size_t(8192)})
            for (uint32_t ratio : {2u,6u,8u}) for (uint32_t queries : {1u,2u,3u,4u}) {
                size_t maximum=0;
                if (!t.assert_true(get()->decode_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,ratio*2,2,4,8192,maximum))) return;
                block_inputs input(f,active,queries,false,ratio*2);
                const auto expected=stock_attention(f,input,0);
                for (bool wrapped : {false,true}) {
                    const std::vector<size_t> cuts=wrapped ? std::vector<size_t>{0,256,512,active} : std::vector<size_t>{0,active};
                    auto storage=span_storage::create(f,0,active,queries,cuts,wrapped);
                    if (!t.assert_true(storage != nullptr)) return;
                    ggml_context_ptr context(ggml_init({65536,nullptr,true}));
                    auto * node=f.resident->attention(context.get(),0,input.q,input.mask,active,1.0f/16);
                    if (!t.assert_true(node != nullptr)) return;
                    auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
                    ggml_kv_stream_span_plan_view view;
                    if (!t.assert_true(ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
                    ggml_cuda_kv_stream_attention_plan plan;
                    if (!t.assert_true(query(backend.get(),&op,&view,style::spanned,plan) == status::success)) return;
                    const bool vector=queries == 1 || (consumer >= 890 && queries == 2);
                    t.assert_true(plan.family == (vector ? family::vector : family::mma));
                    size_t bytes=0;
                    if (!t.assert_true(get()->spans_workspace(backend.get(),&op,storage->plan,bytes))) return;
                    t.assert_equal(plan.requirements.scratch_bytes,bytes);
                    t.assert_true(bytes <= maximum);
                    block_workspace exact(f,bytes,91), tiny(f,bytes-1,92);
                    const auto before=input.read();
                    t.assert_true(!get()->spans(backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(tiny.lease.get())));
                    t.assert_true(same_float_bits(before,input.read()));
                    if (!t.assert_true(get()->spans(backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(exact.lease.get())))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual=input.read();
                    t.out << "CC" << consumer << " GQA" << ratio << " TG" << queries << " active=" << active << " wrapped=" << wrapped << " exact=" << same_float_bits(expected,actual) << '\n';
                    // Regional vector reductions can differ on unaligned tails; keep the observed-error regression guard.
                    if (!vector || (queries == 2 && active%256 == 0) || (!wrapped && active%256 == 0))
                        t.assert_true(same_float_bits(expected,actual));
                    else close_values(t,expected,actual,1e-8f);
                }
            }
    });
    if (consumer) t.test("consumer_preserves_f16_mma_staging_and_masked_tails",[&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,8192,false,2,2,80);
        if (!t.assert_true(get && get() && f.attach())) return;
        auto pin=f.binding->acquire();
        for (size_t active : {size_t(768),size_t(8191)}) for (uint32_t queries : {3u,4u}) {
            block_inputs input(f,active,queries,false,12);
            auto storage=span_storage::create(f,0,active,queries,{0,256,512,active},true);
            if (!t.assert_true(storage != nullptr)) return;
            bool accepted=false;
            const auto actual=evaluate(f,get(),*storage,input,0,accepted);
            if (!t.assert_true(accepted)) return;
            t.assert_true(same_float_bits(stock_attention(f,input,0),actual));
        }
    });
#endif
    t.test("ampere_decode_dispatch_matches_stock", [](testing & t) {
        using path = ggml_cuda_kv_stream_attention_path;
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::mma);
        t.assert_true(ggml_cuda_kv_stream_attention_select(890,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(890,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0) == path::mma);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_F16,GGML_TYPE_F16) == path::mma);
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
    if (pascal || tg2 || (argc > 1 && !std::strcmp(argv[1],"--cuda"))) t.test("quantized_vector_stock_equivalence_across_spans_and_ring_waves", [&](testing & t) {
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
        const uint32_t queries = tg2 ? 2 : 1;
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
            const auto capabilities = ops->capabilities(backend.get(),pair.first,pair.second);
            if (!capabilities.direct_pair) {
                ggml_kv_stream_resume_plan unavailable;
                unavailable.bytes = 77;
                t.assert_true(!ops->resume_plan(backend.get(),pair.first,pair.second,24,4,queries,256,unavailable));
                t.assert_equal(size_t(77),unavailable.bytes);
                t.out << "pair not compiled: " << ggml_type_name(pair.first) << '/' << ggml_type_name(pair.second) << '\n';
                continue;
            }
            t.assert_true(capabilities.direct_pair && capabilities.k.direct_attention && capabilities.v.direct_attention);
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
                block_inputs input(f,active,queries,false,24);
                std::vector<ggml_fp16_t> mask(input.padded*queries);
                for (size_t q = 0; q < queries; ++q) for (size_t i = 0; i < input.padded; ++i)
                    mask[q*input.padded+i] = ggml_fp32_to_fp16(i <= active-queries+q && i%7 != 3 ? 0 : -INFINITY);
                ggml_backend_tensor_set(input.mask,mask.data(),0,mask.size()*sizeof(ggml_fp16_t));
                ggml_kv_stream_resume_plan plan;
                if (!t.assert_true("vector resume plan",ops->resume_plan(backend.get(),pair.first,pair.second,24,4,queries,input.padded,plan))) return;
                if (queries == 2) {
                    ggml_kv_stream_resume_plan one;
                    t.assert_true(ggml_kv_stream_resume_layout_make(24,1,plan.splits,plan.values_per_thread,one));
                    t.assert_equal(2*one.bytes,plan.bytes);
                    t.assert_equal(2*one.state_bytes,plan.state_bytes);
                }
                const bool half_value = pair.second == GGML_TYPE_F16 || pair.second == GGML_TYPE_BF16;
                t.assert_equal(uint32_t(half_value ? (kv_stream_test_compiled_cc(backend.get()) < 700 ? 16 : 32) : 8),plan.values_per_thread);
                ggml_kv_stream_resume_plan checked;
                t.assert_true(ggml_kv_stream_resume_layout_make(24,queries,plan.splits,plan.values_per_thread,checked));
                t.assert_equal(checked.bytes,plan.bytes);
                block_workspace workspace(f,plan.bytes), tiny(f,plan.bytes-1,89);
                auto * scratch_parent = ggml_backend_memory_arena_parent(workspace.arena.get());
                ggml_backend_buffer_clear(scratch_parent,0x5a);
                if (!t.assert_true("resident prefix synchronization",f.resident->synchronize(std::min(active,size_t(256))))) return;
                if (!t.assert_true("initial sequence admission",f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return;
                if (active > 256) {
                    t.assert_true("one-byte-short scratch rejected",!f.resident->compute_streamed(0,input.q,input.mask,input.output,active,1.0f/16,tiny.lease.get(),true,1));
                    const auto untouched = input.read();
                    t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](float x) { return x == -77; }));
                    if (!t.assert_true("retry sequence after scratch rejection",f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return;
                }
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    const auto expected = stock_attention(f,input,layer);
                    if (!t.assert_true("exact-sized resumed layer",f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,1))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual = input.read();
                    close_values(t,expected,actual,1e-6f);
                    if (queries == 2) t.assert_true(!std::equal(actual.begin(),actual.begin()+actual.size()/2,actual.begin()+actual.size()/2));
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
                auto storage = span_storage::create(control,0,active,queries,cuts,true);
                if (!t.assert_true(storage != nullptr)) return;
                if (queries == 2) {
                    auto stale = span_storage::create(control,0,active,1,cuts,true);
                    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
                    auto * node = control.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f/16);
                    if (!t.assert_true(stale && node)) return;
                    auto op = *node; op.buffer = input.output->buffer; op.data = input.output->data;
                    size_t untouched = 77;
                    t.assert_true(!ops->spans_workspace(backend.get(),&op,stale->plan,untouched));
                    t.assert_equal(size_t(77),untouched);
                    const auto before = input.read();
                    t.assert_true(!ops->spans(backend.get(),&op,stale->plan,ggml_backend_memory_lease_buffer(workspace.lease.get())));
                    t.assert_true(same_float_bits(before,input.read()));
                    ggml_kv_stream_resume_plan wrong_width;
                    t.assert_true(ggml_kv_stream_resume_layout_make(24,1,plan.splits,plan.values_per_thread,wrong_width));
                    wrong_width.tokens = plan.tokens;
                    t.assert_true(!ops->resume(backend.get(),&op,ggml_backend_memory_lease_buffer(workspace.lease.get()),wrong_width,plan.tokens,0,true));
                    t.assert_true(same_float_bits(before,input.read()));
                }
                bool accepted = false;
                const auto expected = stock_attention(f,input,0);
                const auto actual = evaluate(control,ops,*storage,input,0,accepted);
                if (!t.assert_true(accepted)) return;
                close_values(t,expected,actual,1e-6f);
            }
        }
    });
    if (tg2) t.test("two_query_tail_visibility_and_rejection_replay_preserve_prefix", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),pascal);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,3);
        f.policy.initial_ring_slots = 1;
        if (!t.assert_true(ops && f.attach() && f.resident->configure_native_graph_attention(true) && f.resident->configure_resumed_decode(true))) return;
        auto pin = f.binding->acquire();
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,2,768,plan))) return;
        block_workspace workspace(f,plan.bytes);
        const auto run = [&](size_t active, uint32_t queries) {
            block_inputs input(f,active,queries,false,24);
            std::vector<float> first;
            if (!t.assert_true(f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return first;
            for (uint32_t layer = 0; layer < 2; ++layer) {
                const auto expected = stock_attention(f,input,layer);
                if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,1))) return std::vector<float>{};
                ggml_backend_synchronize(backend.get());
                const auto actual = input.read();
                close_values(t,expected,actual,1e-6f);
                if (!layer) first = actual;
            }
            t.assert_true(!f.resident->sequence_active());
            return first;
        };
        const auto baseline = run(257,2);
        if (!t.assert_equal(size_t(2*24*256),baseline.size())) return;
        llama_kv_stream_host_layer host;
        if (!t.assert_true(f.host->layer(0,host))) return;
        const auto & layout = f.host->layout();
        auto * k = static_cast<uint8_t *>(host.k)+256*layout.k_token_bytes;
        auto * v = static_cast<uint8_t *>(host.v)+256*layout.v_token_bytes;
        std::vector<uint8_t> saved_k(k,k+layout.k_token_bytes), saved_v(v,v+layout.v_token_bytes);
        std::vector<float> changed(4*256,1.0f);
        ggml_quantize_chunk(GGML_TYPE_Q8_0,changed.data(),k,0,4,256,nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0,changed.data(),v,0,4,256,nullptr);
        if (!t.assert_true(f.content->invalidate_suffix(256))) return;
        const auto modified = run(257,2);
        if (!t.assert_equal(baseline.size(),modified.size())) return;
        t.assert_true(std::equal(baseline.begin(),baseline.begin()+baseline.size()/2,modified.begin()));
        t.assert_true(!std::equal(baseline.begin()+baseline.size()/2,baseline.end(),modified.begin()+modified.size()/2));
        std::memcpy(k,saved_k.data(),saved_k.size()); std::memcpy(v,saved_v.data(),saved_v.size());
        if (!t.assert_true(f.content->invalidate_suffix(256))) return;
        t.assert_equal(size_t(24*256),run(256,1).size());
        t.assert_true(same_float_bits(baseline,run(257,2)));
    });
    if (pascal) t.test("tg1_only_session_preserves_decode_workspace_during_phase_handoff", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),true);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops != nullptr)) return;
        tg1_only_test_ops limited(ggml_backend_dev_backend_reg(dev),*ops);
        ops = &limited.limited;
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
    if (pascal && tg2) t.test("two_query_session_handoff_reserves_the_complete_decode_plan", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),true);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,3);
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ops && ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,2,f.host->layout().tokens,plan))) return;
        size_t required=plan.bytes;
        if (ops->version >= 10) {
            if (!t.assert_true(ops->decode_workspace && ops->decode_workspace(backend.get(),GGML_TYPE_Q8_0,
                    GGML_TYPE_Q4_0,24,4,4,f.host->layout().tokens,required))) return;
            t.assert_true(required >= plan.bytes);
        }
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,24});
        if (!t.assert_true(model != nullptr)) return;
        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(model->memory_requirements(requirements))) return;
        t.assert_equal(required,requirements.attention_decode_bytes);
        if (!t.assert_true(model->begin(2,2,true))) return;
        t.assert_equal(required,model->attention_grant_bytes());
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(1,1,true));
        t.assert_equal(required,model->attention_grant_bytes());
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(2,2,false));
        t.assert_equal(requirements.attention_prefill_bytes,model->attention_grant_bytes());
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

    if (float32_only || (argc > 1 && !std::strcmp(argv[1], "--cuda"))) t.test("float32_storage_uses_bounded_stock_equivalent_attention", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->capabilities && ops->decode_workspace)) return;
        for (auto pair : {std::pair{GGML_TYPE_F32, GGML_TYPE_F32}, std::pair{GGML_TYPE_F32, GGML_TYPE_Q8_0},
                std::pair{GGML_TYPE_Q5_1, GGML_TYPE_F32}}) for (uint32_t query_heads : {8u, 24u}) {
            const auto caps = ops->capabilities(backend.get(), pair.first, pair.second);
            t.assert_true(caps.direct_pair && caps.k.online_write && caps.v.online_write);
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                size_t bytes = 0;
                t.assert_true(ops->decode_workspace(backend.get(), pair.first, pair.second, query_heads, 4, rows, 262144, bytes));
                t.assert_true(bytes > 0 && bytes < 32*1024*1024);
            }
            fixture f(backend.get(), true, pair.first, pair.second, 1025, false, 2, 4, 16);
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                block_inputs input(f, 513, rows, false, query_heads);
                auto storage = span_storage::create(f, 0, 513, rows, {0, 256, 512, 513}, true);
                ggml_context_ptr context(ggml_init({65536, nullptr, true}));
                auto * node = f.resident->attention(context.get(), 0, input.q, input.mask, input.active, 1.0f/16);
                if (!t.assert_true(pin && storage && node)) return;
                ggml_tensor op = *node;
                op.buffer = input.output->buffer;
                op.data = input.output->data;
                size_t bytes = 0;
                if (!t.assert_true(ops->spans_workspace(backend.get(), &op, storage->plan, bytes))) return;
                block_workspace workspace(f, bytes, 351);
                if (!t.assert_true(ops->spans(backend.get(), &op, storage->plan,
                        ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                ggml_backend_synchronize(backend.get());
                const auto expected = stock_attention(f, input, 0);
                const auto actual = input.read();
                t.assert_true(expected.size() == actual.size() &&
                    std::memcmp(expected.data(), actual.data(), expected.size()*sizeof(float)) == 0);
            }
        }
    });

    if (float32_only || (argc > 1 && !std::strcmp(argv[1], "--cuda"))) t.test("float32_ring_waves_preserve_native_conversion_and_resume_order", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->decode_workspace)) return;
        for (auto pair : {std::pair{GGML_TYPE_F32, GGML_TYPE_F32}, std::pair{GGML_TYPE_F32, GGML_TYPE_Q8_0},
                std::pair{GGML_TYPE_Q5_1, GGML_TYPE_F32}}) for (uint32_t query_heads : {8u, 24u}) {
            fixture f(backend.get(), true, pair.first, pair.second, 2049, false, 2, 4, 7);
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
            auto pin = f.binding->acquire();
            if (!t.assert_true(bool(pin))) return;
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                block_inputs input(f, 2049, rows, false, query_heads);
                size_t bytes = 0;
                if (!t.assert_true(ops->decode_workspace(backend.get(), pair.first, pair.second, query_heads, 4, rows, input.padded, bytes))) return;
                block_workspace workspace(f, bytes, 361);
                if (!t.assert_true(f.resident->synchronize(256) &&
                        f.resident->begin_sequence({0,1}, input.active, 1, SIZE_MAX, {rows, true}))) return;
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    const auto expected = stock_attention(f, input, layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer, input.q, input.mask, input.output,
                            input.active, 1.0f/16, workspace.lease.get(), true, 1))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual = input.read();
                    close_values(t, expected, actual, 1e-6f);
                    if (rows >= 3 || pair.first == pair.second) t.assert_true(same_float_bits(expected, actual));
                    t.assert_true(f.resident->last_attention_calls() > 1);
                }
                t.assert_true(!f.resident->sequence_active());
            }
        }
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
                const auto before = input.read();
                const auto b = evaluate(f, ops, *segmented, input, 0, accepted, &segmented_workspace);
                if (cuts.size() > 4) {
                    t.assert_true(!accepted && b.empty());
                    t.assert_true(same_float_bits(before,input.read()));
                    continue;
                }
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
            "encoded_mma_tile_bytes_equal_stock_conversion", [](testing & t) { check_mma_tile_conversion(t,256); });

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
                    const auto before = input.read();
                    const auto actual = evaluate(f,ops,*storage,input,0,accepted,&workspace);
                    if (cuts.size() > 4) {
                        t.assert_true(!accepted && actual.empty());
                        t.assert_true(same_float_bits(before,input.read()));
                        continue;
                    }
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
