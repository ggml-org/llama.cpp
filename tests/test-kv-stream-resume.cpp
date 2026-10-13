#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../src/llama-kv-stream-model.h"
#include <chrono>
#include <cstdio>

static std::vector<float> ordinary(fixture & f, block_inputs & input, uint32_t layer) {
    return stock_attention(f,input,layer);
}

static int benchmark_ring_decode(bool long_context) {
    ggml_backend_load_all();
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!dev) return 1;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    const size_t active=long_context ? 163840 : 25601;
    const size_t ring_slots=long_context ? 273 : 96;
    const size_t pool_pages=long_context ? 1063 : 112;
    for (uint32_t queries : {2u,3u,4u}) for (bool native : {false,true}) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,8,pool_pages);
        f.policy.initial_ring_slots=ring_slots;
        llama_kv_stream_policy_state placement;
        GGML_ASSERT(llama_kv_stream_policy_initialize(f.policy,placement).status == llama_kv_stream_policy_status::success);
        GGML_ASSERT(f.attach(&placement) && f.resident->configure_native_graph_attention(native) &&
            f.resident->configure_resumed_decode(native));
        auto pin=f.binding->acquire();
        block_workspace workspace(f,64*1024*1024);
        block_inputs input(f,active,queries,false,64);
        std::vector<double> samples;
        size_t copy_calls=0, attention_calls=0;
        for (int trial=0;trial<15;++trial) {
            const auto start=std::chrono::steady_clock::now();
            GGML_ASSERT(f.resident->begin_sequence({0,1},active,32,SIZE_MAX,{queries,true}));
            for (uint32_t layer=0;layer<2;++layer)
                GGML_ASSERT(f.resident->compute_streamed(layer,input.q,input.mask,input.output,
                    active,1.0f/16,workspace.lease.get(),true,32));
            ggml_backend_synchronize(backend.get());
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if (trial >= 3) samples.push_back(ms);
            copy_calls=f.resident->sequence_stats().copy_calls;
            attention_calls=f.resident->last_attention_calls();
        }
        std::sort(samples.begin(),samples.end());
        std::printf("ring-decode,active=%zu,tg=%u,mode=%s,median_ms=%.3f,copy_calls=%zu,last_layer_launches=%zu\n",
            active,queries,native ? "native" : "partial",samples[samples.size()/2],copy_calls,attention_calls);
    }
    return 0;
}

static int verify_long_ring_exact() {
    ggml_backend_load_all();
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!dev) return 1;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    constexpr size_t active=163840;
    fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,8,1063);
    f.policy.initial_ring_slots=273;
    llama_kv_stream_policy_state placement;
    GGML_ASSERT(llama_kv_stream_policy_initialize(f.policy,placement).status == llama_kv_stream_policy_status::success);
    GGML_ASSERT(f.attach(&placement) && f.resident->configure_native_graph_attention(true) &&
        f.resident->configure_resumed_decode(true));
    auto pin=f.binding->acquire();
    block_workspace workspace(f,64*1024*1024);
    for (uint32_t queries : {2u,3u,4u}) {
        block_inputs input(f,active,queries,false,64);
        GGML_ASSERT(f.resident->begin_sequence({0,1},active,32,SIZE_MAX,{queries,true}));
        for (uint32_t layer=0;layer<2;++layer) {
            const auto expected=stock_attention(f,input,layer);
            GGML_ASSERT(f.resident->compute_streamed(layer,input.q,input.mask,input.output,
                active,1.0f/16,workspace.lease.get(),true,32));
            ggml_backend_synchronize(backend.get());
            const auto actual=input.read();
            const bool exact=same_float_bits(expected,actual);
            std::printf("long-ring-exact,active=%zu,tg=%u,layer=%u,exact=%d\n",active,queries,layer,int(exact));
            if (!exact) return 1;
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && !std::strcmp(argv[1],"--verify-long-ring")) return verify_long_ring_exact();
    if (argc > 1 && !std::strcmp(argv[1],"--bench-ring-decode")) return benchmark_ring_decode(argc > 2 && !std::strcmp(argv[2],"160k"));
    testing t;
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("prefill_device_requirements_do_not_grow_with_history", [&](testing & t) {
        ggml_backend_load_all(); auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        size_t first=0;
        for (size_t context : {size_t(4096),size_t(65536)}) {
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,context,false,2,2,12);
            llama_kv_stream_model_config config;
            config.backend=backend.get(); config.host=f.host->config(); config.host.cache_id=0;
            config.pool_bytes=f.policy.pool_bytes; config.max_batch_rows=256; config.query_heads=12;
            auto stream=llama_kv_stream_model::create(config);
            if (!t.assert_true(bool(stream) && stream->bounded_prefill())) return;
            llama_kv_stream_memory_requirements requirements;
            if (!t.assert_true(stream->memory_requirements(requirements))) return;
            if (!first) first=requirements.attention_prefill_bytes;
            t.assert_equal(first,requirements.attention_prefill_bytes);
        }
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("bounded_prefill_waves_match_stock_without_gather_or_output_extras", [&](testing & t) {
        ggml_backend_load_all();
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(get && get() && get()->version >= 13 && get()->prefill_workspace)) return;
        for (auto pair : {std::pair{GGML_TYPE_F16,GGML_TYPE_F16}, {GGML_TYPE_BF16,GGML_TYPE_BF16},
                {GGML_TYPE_Q8_0,GGML_TYPE_Q8_0}, {GGML_TYPE_Q8_0,GGML_TYPE_Q5_0}, {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},
                {GGML_TYPE_Q5_0,GGML_TYPE_Q8_0}, {GGML_TYPE_Q5_0,GGML_TYPE_Q5_0}, {GGML_TYPE_Q5_0,GGML_TYPE_Q4_0},
                {GGML_TYPE_Q4_0,GGML_TYPE_Q8_0}, {GGML_TYPE_Q4_0,GGML_TYPE_Q5_0}, {GGML_TYPE_Q4_0,GGML_TYPE_Q4_0},
                {GGML_TYPE_Q5_1,GGML_TYPE_Q4_1}, {GGML_TYPE_Q4_1,GGML_TYPE_Q5_1},
                {GGML_TYPE_Q4_0,GGML_TYPE_F16}, {GGML_TYPE_F16,GGML_TYPE_Q4_0}})
            for (size_t active : {size_t(2049),size_t(32769)}) {
            if (active > 2049 && pair != std::pair{GGML_TYPE_F16,GGML_TYPE_F16} &&
                    pair != std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0}) continue;
            const size_t capacity=(active+255)/256*256;
            fixture f(backend.get(),true,pair.first,pair.second,capacity,false,2,2,12);
            size_t scratch=0;
            if (!t.assert_true(get()->prefill_workspace(backend.get(),pair.first,pair.second,12,2,1024,capacity,scratch))) return;
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_prefill_workspace(scratch))) return;
            t.assert_true(!f.resident->configure_prefill_workspace(SIZE_MAX));
            auto pin=f.binding->acquire();
            block_workspace workspace(f,scratch);
            {
                block_workspace short_workspace(f,scratch-1,23);
                block_inputs input(f,active,9,false,12);
                const auto before=input.read();
                if (!t.assert_true(f.resident->begin_sequence({0,1},active,4,SIZE_MAX,{9,false}))) return;
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,active,1.0f/16,
                    short_workspace.lease.get(),true,4));
                t.assert_true(same_float_bits(before,input.read()));
            }
            for (uint32_t queries : {1u,2u,3u,4u,8u,9u,64u,129u,256u,512u,1024u}) {
                block_inputs input(f,active,queries,false,12);
                if (!t.assert_true(f.resident->begin_sequence({0,1},active,4,SIZE_MAX,{queries,false}))) return;
                for (uint32_t layer=0;layer<2;++layer) {
                    const auto expected=ordinary(f,input,layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,
                            active,1.0f/16,workspace.lease.get(),true,4))) return;
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(expected,input.read()));
                }
                ggml_context_ptr bounds_context(ggml_init({8192,nullptr,true}));
                auto * bounds=ggml_new_tensor_2d(bounds_context.get(),GGML_TYPE_I32,4,queries);
                ggml_backend_buffer_ptr bounds_buffer(ggml_backend_alloc_ctx_tensors(bounds_context.get(),backend.get()));
                if (!t.assert_true(bool(bounds_buffer))) return;
                std::vector<ggml_backend_execution_mask_interval> intervals(queries);
                for (size_t row=0; row<queries; ++row)
                    intervals[row]={0,int32_t(active-queries+row+1),int32_t(capacity),1};
                ggml_backend_tensor_set(bounds,intervals.data(),0,ggml_nbytes(bounds));
                if (queries == 9 && active == 2049) {
                    const auto before=input.read();
                    auto * previous=bounds->data;
                    bounds->data=static_cast<char *>(ggml_backend_buffer_get_base(bounds_buffer.get()))+
                        ggml_backend_buffer_get_size(bounds_buffer.get())-sizeof(int32_t);
                    if (!t.assert_true(f.resident->begin_sequence({0,1},active,4,SIZE_MAX,{queries,false}))) return;
                    t.assert_true(!f.resident->compute_streamed(0,input.q,bounds,input.output,
                        active,1.f/16,workspace.lease.get(),true,4));
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(before,input.read()));
                    bounds->data=previous;
                    block_workspace short_workspace(f,scratch-1,24);
                    if (!t.assert_true(f.resident->begin_sequence({0,1},active,4,SIZE_MAX,{queries,false}))) return;
                    t.assert_true(!f.resident->compute_streamed(0,input.q,bounds,input.output,
                        active,1.f/16,short_workspace.lease.get(),true,4));
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(before,input.read()));
                }
                if (!t.assert_true(f.resident->begin_sequence({0,1},active,4,SIZE_MAX,{queries,false}))) return;
                for (uint32_t layer=0; layer<2; ++layer) {
                    const auto expected=ordinary(f,input,layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,bounds,input.output,
                            active,1.f/16,workspace.lease.get(),true,4))) return;
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(expected,input.read()));
                }
            }
        }
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("compiled_decode_pairs_share_complete_span_routing", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        for (auto key : {GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0})
            for (auto value : {GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0}) {
            fixture f(backend.get(),true,key,value,2049,false,2,2,12);
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
            auto pin=f.binding->acquire();
            size_t bytes=0;
            if (!t.assert_true(get()->decode_workspace(backend.get(),key,value,12,2,8,2304,bytes))) return;
            block_workspace workspace(f,bytes);
            for (uint32_t queries : {3u,4u,8u}) {
                block_inputs input(f,2049,queries,false,12);
                if (!t.assert_true(f.resident->begin_sequence({0,1},2049,4,SIZE_MAX,{queries,true}))) return;
                for (uint32_t layer=0; layer<2; ++layer) {
                    const auto expected=ordinary(f,input,layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,
                            2049,1.f/16,workspace.lease.get(),true,4))) return;
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(expected,input.read()));
                    t.assert_equal(size_t(1),f.resident->last_attention_calls());
                }
            }
        }
    });
    t.test("resume_layout_is_bounded_and_transactional", [&](testing & t) {
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ggml_kv_stream_resume_layout_make(24,1,3,8,plan))) return;
        t.assert_equal(size_t(368640),plan.state_bytes);
        t.assert_equal(size_t(368640),plan.partial_offset);
        t.assert_equal(size_t(442368),plan.meta_offset);
        t.assert_equal(size_t(442944),plan.bytes);
        for (auto dims : {std::array<uint32_t,3>{0,3,8},{24,0,8},{24,3,0},{UINT32_MAX,UINT32_MAX,UINT32_MAX}}) {
            t.assert_true(!ggml_kv_stream_resume_layout_make(dims[0],1,dims[1],dims[2],plan));
            t.assert_equal(size_t(442944),plan.bytes);
        }
    });
    t.test("resume_layout_covers_legacy_half_value_accumulators", [&](testing & t) {
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ggml_kv_stream_resume_layout_make(24,1,3,16,plan))) return;
        t.assert_equal(size_t(663552),plan.state_bytes);
        t.assert_equal(size_t(663552),plan.partial_offset);
        t.assert_equal(size_t(737280),plan.meta_offset);
        t.assert_equal(size_t(737856),plan.bytes);
        for (uint32_t values : {0u,1u,4u,9u,17u,64u}) {
            t.assert_true(!ggml_kv_stream_resume_layout_make(24,1,3,values,plan));
            t.assert_equal(size_t(737856),plan.bytes);
        }
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("native_resume_contract_is_available", [&](testing & t) {
        ggml_backend_load_all(); auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(get && get() && get()->version >= 9 && get()->mma_workspace)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        size_t bytes=77;
        t.assert_true(!get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,63,8,25856,3,bytes));
        t.assert_equal(size_t(77),bytes);
        t.assert_true(!get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,64,8,25856,4,bytes));
        t.assert_equal(size_t(77),bytes);
        t.assert_true(get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,64,8,25856,3,bytes));
        t.assert_true(bytes > 0);
        t.assert_equal(size_t(4526336),bytes);
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("resident_and_ring_spans_preserve_native_decode", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q4_1},std::pair{GGML_TYPE_Q4_0,GGML_TYPE_F16}}) {
            fixture f(backend.get(),true,pair.first,pair.second,4097);
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            f.policy.pool_bytes=page.bytes*3; f.policy.initial_ring_slots=1;
            if (!t.assert_true(f.attach() && f.resident->configure_resumed_decode(true))) return;
            auto pin=f.binding->acquire();
            for (size_t active : {size_t(513),size_t(769),size_t(4097)}) {
                t.out << ggml_type_name(pair.first) << '/' << ggml_type_name(pair.second) << " active=" << active << '\n';
                block_inputs input(f,active,1);
                ggml_kv_stream_resume_plan plan;
                if (!t.assert_true(get()->resume_plan(backend.get(),pair.first,pair.second,4,2,1,input.padded,plan))) return;
                block_workspace workspace(f,plan.bytes);
                if (!t.assert_true(f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{1,true}))) return;
                for (uint32_t layer=0;layer<2;++layer) {
                    const auto expected=ordinary(f,input,layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,1))) return;
                    ggml_backend_synchronize(backend.get());
                    close_values(t,expected,input.read(),1e-6f);
                }
                t.assert_true(!f.resident->sequence_active());
            }
        }
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("native_streaming_routes_tg1_through_tg4_over_complete_spans", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1025);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        f.policy.pool_bytes=page.bytes*6; f.policy.initial_ring_slots=4;
        if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                f.resident->configure_resumed_decode(true))) return;
        auto pin=f.binding->acquire();
        block_workspace workspace(f,512*1024);
        for (size_t active : {size_t(769),size_t(1025)}) for (uint32_t queries : {1u,2u,3u,4u}) {
            block_inputs input(f,active,queries);
            // Multi-query verification is decode work, not a strict-gather prefill.
            if (!t.assert_true(f.resident->begin_sequence({0,1},active,4,SIZE_MAX,{queries,true}))) return;
            for (uint32_t layer=0;layer<2;++layer) {
                const auto expected=ordinary(f,input,layer);
                if (!t.assert_true(f.resident->compute_streamed(
                        layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,4))) return;
                ggml_backend_synchronize(backend.get());
                close_values(t,expected,input.read(),1e-6f);
                const size_t calls=queries <= 2 ? (active == 769 && layer == 1 ? 3 : 2) : 1;
                t.assert_equal(calls,f.resident->last_attention_calls());
            }
            t.assert_true(!f.resident->sequence_active());
            t.assert_equal(active == 769 ? size_t(6) : size_t(4),f.resident->sequence_stats().copy_calls);
        }
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("production_head_geometry_uses_segmented_spans", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1025,false,4,8);
        llama_kv_stream_policy_state initial; llama_kv_stream_policy_decision decision;
        if (!t.assert_true(llama_kv_stream_policy_initialize(f.policy,initial).status == llama_kv_stream_policy_status::success) ||
                !t.assert_true(llama_kv_stream_policy_step(f.policy,initial,{1025,1,{},false},decision).status == llama_kv_stream_policy_status::success) ||
                !t.assert_true(f.attach(&decision.next)) ||
                !t.assert_true(f.resident->configure_native_graph_attention(true))) return;
        auto pin=f.binding->acquire(); block_inputs input(f,1025,1,false,64); block_workspace workspace(f,4*1024*1024);
        if (!t.assert_true(f.resident->begin_sequence({0,1,2},1025,decision.next.ring_slots,SIZE_MAX,{1,true}))) return;
        for (uint32_t layer=0;layer<3;++layer) {
            const auto expected=ordinary(f,input,layer);
            if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,decision.next.ring_slots))) return;
            ggml_backend_synchronize(backend.get()); close_values(t,expected,input.read(),1e-6f);
            t.assert_equal(size_t(1),f.resident->last_attention_calls());
        }
        t.assert_true(!f.resident->sequence_active());
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("large_suffix_that_fits_ring_uses_stock_split_resume", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        constexpr size_t active=41473, ring_pages=161;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,2,165);
        f.policy.initial_ring_slots=ring_pages;
        if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                f.resident->configure_resumed_decode(true))) return;
        auto pin=f.binding->acquire(); block_inputs input(f,active,1); block_workspace workspace(f,4*1024*1024);
        if (!t.assert_true(f.resident->begin_sequence({0,1},active,ring_pages,SIZE_MAX,{1,true}))) return;
        for (uint32_t layer=0;layer<2;++layer) {
            const auto expected=ordinary(f,input,layer);
            if (!t.assert_true(f.resident->compute_streamed(
                    layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,ring_pages))) return;
            ggml_backend_synchronize(backend.get()); close_values(t,expected,input.read(),1e-6f);
            t.assert_equal(size_t(2),f.resident->last_attention_calls());
        }
        t.assert_true(!f.resident->sequence_active());
    });

    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("thirty_two_page_dma_batches_share_one_stock_split_resume", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        constexpr size_t active=25601, transfer_pages=32, ring_pages=96;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,8,112);
        f.policy.initial_ring_slots=ring_pages;
        llama_kv_stream_policy_state placement;
        const auto initialized=llama_kv_stream_policy_initialize(f.policy,placement);
        if (!t.assert_true(initialized.status == llama_kv_stream_policy_status::success) ||
                !t.assert_equal(ring_pages,size_t(placement.ring_slots)) ||
                !t.assert_equal(size_t(8),size_t(placement.resident_pages_per_layer)) ||
                !t.assert_true(f.attach(&placement) && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
        auto pin=f.binding->acquire(); block_workspace workspace(f,64*1024*1024);
        for (uint32_t queries : {3u,4u,1u,2u}) {
            block_inputs input(f,active,queries,false,64);
            if (!t.assert_true(f.resident->begin_sequence({0,1},active,transfer_pages,SIZE_MAX,{queries,true}))) return;
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            ggml_kv_stream_resume_plan planned;
            const bool planned_ok=get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,
                64,8,queries,input.padded,planned);
            if (queries == 1 && !t.assert_true(planned_ok)) return;
            if (queries == 2 && !planned_ok) {
                size_t mma_bytes=0;
                if (!t.assert_true(get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,
                        64,8,input.padded,3,mma_bytes) && mma_bytes > 0)) return;
            }
            for (uint32_t layer=0;layer<2;++layer) {
                const auto expected=ordinary(f,input,layer);
                if (!t.assert_true(f.resident->compute_streamed(
                        layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,transfer_pages))) return;
                ggml_backend_synchronize(backend.get());
                const auto actual=input.read();
                close_values(t,expected,actual,1e-6f);
                t.assert_true(same_float_bits(expected,actual));
                const size_t calls = queries <= 2 && planned_ok ? (layer ? size_t(3) : size_t(2)) : size_t(1);
                t.assert_equal(calls,f.resident->last_attention_calls());
            }
            t.assert_true(!f.resident->sequence_active());
            t.assert_equal(size_t(14),f.resident->sequence_stats().copy_calls);
        }
    });

    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("zero_resident_tg3_tg4_use_exact_ring_spans", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        constexpr size_t active=25601;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,8,112);
        llama_kv_stream_policy_state placement;
        if (!t.assert_true(llama_kv_stream_policy_initialize(f.policy,placement).status == llama_kv_stream_policy_status::success)) return;
        placement.resident_pages_per_layer=0;
        placement.ring_slots=placement.budget.pages;
        placement.decode_active_pages=(active+255)/256;
        if (!t.assert_true(f.attach(&placement) && f.resident->configure_native_graph_attention(true))) return;
        auto pin=f.binding->acquire(); block_workspace workspace(f,64*1024*1024);
        for (uint32_t queries : {3u,4u}) {
            block_inputs input(f,active,queries,false,64);
            if (!t.assert_true(f.resident->begin_sequence({0,1},active,32,SIZE_MAX,{queries,true}))) return;
            for (uint32_t layer=0;layer<2;++layer) {
                const auto expected=stock_attention(f,input,layer);
                if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,
                        active,1.0f/16,workspace.lease.get(),true,32))) return;
                ggml_backend_synchronize(backend.get());
                t.assert_true(same_float_bits(expected,input.read()));
                t.assert_equal(size_t(1),f.resident->last_attention_calls());
            }
            t.assert_true(!f.resident->sequence_active());
        }
    });

    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("invalid_resume_inputs_preserve_output", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0"); ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769);
        if (!t.assert_true(f.attach() && f.resident->synchronize(769))) return;
        block_inputs input(f,769,1); ggml_kv_stream_resume_plan plan;
        t.assert_true(get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,2,1,input.padded,plan));
        const auto bytes=plan.bytes;
        t.assert_true(!get()->resume_plan(nullptr,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,2,1,input.padded,plan));
        t.assert_true(!get()->resume_plan(backend.get(),GGML_TYPE_F16,GGML_TYPE_F16,4,2,1,input.padded,plan));
        t.assert_true(!get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,3,2,1,input.padded,plan));
        t.assert_equal(bytes,plan.bytes);
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * op=f.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f/16);
        ggml_backend_buffer_ptr output(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        std::vector<float> sentinel(ggml_nelements(op),-77); ggml_backend_tensor_set(op,sentinel.data(),0,ggml_nbytes(op));
        block_workspace workspace(f,bytes),tiny(f,bytes-1,29);
        t.assert_true(!get()->resume(backend.get(),op,ggml_backend_memory_lease_buffer(tiny.lease.get()),plan,input.padded,0,true));
        t.assert_true(!get()->resume(backend.get(),op,ggml_backend_memory_lease_buffer(workspace.lease.get()),plan,input.padded,1,true));
        t.assert_true(!get()->resume(backend.get(),op,ggml_backend_memory_lease_buffer(workspace.lease.get()),plan,input.padded,0,false));
        auto bad=plan; ++bad.meta_offset;
        t.assert_true(!get()->resume(backend.get(),op,ggml_backend_memory_lease_buffer(workspace.lease.get()),bad,input.padded,0,true));
        std::vector<float> actual(sentinel.size()); ggml_backend_tensor_get(op,actual.data(),0,ggml_nbytes(op));
        t.assert_true(actual==sentinel);
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("tg1_only_resume_reclaims_decode_attention", [&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(get && get() && get()->resume_plan && get()->mma_workspace)) return;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513);
        ggml_kv_stream_resume_plan one,two;
        const bool tg1=get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,2,1,
            f.host->layout().tokens,one);
        const bool tg2=get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,2,2,
            f.host->layout().tokens,two);
        if (tg2) return;
        if (!t.assert_true(tg1)) return;
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape,256,page).status == ggml_kv_stream_status::success)) return;
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(bool(model) && model->memory_requirements(requirements))) return;
        t.assert_true(requirements.attention_decode_bytes < requirements.attention_prefill_bytes);
        size_t mma_bytes=0;
        t.assert_true(get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,2,
            f.host->layout().tokens,3,mma_bytes) && mma_bytes <= requirements.attention_decode_bytes);
    });
    return t.summary();
}
