#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-session.h"
#include "../src/llama-kv-stream-layer-lease.h"

struct session_inputs {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k, * v;
    session_inputs(ggml_backend_t backend, size_t rows, size_t width = 512) {
        ctx.reset(ggml_init({4096,nullptr,true}));
        k = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,int64_t(width),int64_t(rows));
        v = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,int64_t(width),int64_t(rows));
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),backend)); GGML_ASSERT(buffer);
        std::vector<float> data(rows*width);
        for (size_t i = 0; i < data.size(); ++i) data[i] = .25f*std::cos(float(i%541)*.07f);
        ggml_backend_tensor_set(k,data.data(),0,data.size()*sizeof(float));
        ggml_backend_tensor_set(v,data.data(),0,data.size()*sizeof(float));
    }
};

int main(int argc, char ** argv) {
    const bool physical_map_only = argc > 1 && !std::strcmp(argv[1],"--cuda-physical-map");
    const bool guard_only = argc > 1 && !std::strcmp(argv[1],"--cuda-guard-handoff");
    const bool cuda = physical_map_only || guard_only || (argc > 1 && !std::strcmp(argv[1],"--cuda"));
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev,nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    testing t;
    if (guard_only) t.set_filter("guarded_target_session_preserves_mtp_ring_and_releases_for_replan");
    if (physical_map_only) t.set_filter("shared_physical_pool_publishes_only_target_pairs");
    t.test("unsupported_and_missing_session_dependencies_are_rejected", [&](testing & t) {
        fixture f(backend.get(),cuda);
        t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,{f.policy,33,4,false},nullptr,nullptr,nullptr));
        if (!cuda) {
            block_workspace writer(f,32768,19), partial(f,1024*1024,29);
            t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,{f.policy,33,4,false},f.lease.get(),writer.lease.get(),partial.lease.get()));
        }
    });
    if (cuda) t.test("serial_appends_preserve_content_across_policy_rebinding", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,false,4);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        f.policy.pool_bytes = 10*page.bytes; f.policy.initial_ring_slots = 6;
        ggml_kv_stream_block_layout workspace;
        t.assert_true(ggml_kv_stream_block_layout_make(256*4,256,workspace).status == ggml_kv_stream_partial_status::success);
        block_workspace writer(f,32768,19), partial(f,workspace.bytes,29);
        const size_t grant = ggml_backend_buffer_get_size(ggml_backend_memory_lease_buffer(f.lease.get()))+32768+workspace.bytes;
        auto * pool_buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        auto session = llama_kv_stream_session::create(backend.get(),f.content,{f.policy,256,4,false},f.lease.get(),writer.lease.get(),partial.lease.get());
        if (!t.assert_true(bool(session))) return;
        t.assert_equal(grant,session->granted_bytes());
        t.assert_true(!session->set_attention_workspace(f.lease.get(),false));
        t.assert_true(session->set_attention_workspace(nullptr,false));
        t.assert_true(!session->begin(1,1,false));
        t.assert_true(session->set_attention_workspace(partial.lease.get(),false));
        t.assert_equal(grant,session->granted_bytes());
        f.lease.reset(); writer.lease.reset(); partial.lease.reset();
        size_t active = 0;
        for (uint32_t rows : {256u,32u,224u,1u,256u}) {
            active += rows; const bool decode = rows == 1;
            session_inputs input(backend.get(),rows); block_inputs attn(f,active,rows);
            if (decode) {
                const auto revision = session->layout_revision();
                const auto init = pool_buffer->iface.init_tensor;
                pool_buffer->iface.init_tensor = [](ggml_backend_buffer_t,ggml_tensor *) { return GGML_STATUS_ALLOC_FAILED; };
                const bool accepted = session->begin(active,rows,decode);
                pool_buffer->iface.init_tensor = init;
                t.assert_true(!accepted && !session->failed());
                t.assert_equal(revision,session->layout_revision());
                t.assert_equal(active-rows,session->tokens());
            }
            if (!t.assert_true(session->begin(active,rows,decode))) return;
            auto publication = session->publication_frontiers();
            t.assert_equal(active,publication.reserved);
            t.assert_equal(active-rows,publication.device);
            t.assert_equal(active-rows,publication.host);
            t.assert_equal(active-rows,publication.committed);
            t.assert_true(!session->set_attention_workspace(nullptr,decode));
            if (!decode) t.assert_equal(uint32_t(0),session->policy().decode_active_pages);
            t.assert_true(!session->begin(active,rows,decode));
            t.assert_true(!session->attention(0,attn.q,attn.mask,attn.output,1.0f/16));
            for (uint32_t layer = 0; layer < 4; ++layer) {
                if (!t.assert_true(session->produce(layer,input.k,input.v))) return;
                publication = session->publication_frontiers();
                const size_t published = layer+1 == 4 ? active : active-rows;
                t.assert_equal(published,publication.device);
                t.assert_equal(active-rows,publication.host);
                t.assert_equal(active-rows,publication.committed);
                t.assert_equal(active-rows,session->tokens());
                t.assert_true(!session->produce(layer,input.k,input.v));
                if (!t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16))) return;
                close_values(t,oracle(f,layer,active,rows,attn.qdata),attn.read(),1e-3f);
                publication=session->publication_frontiers();
                t.assert_equal(published,publication.host);
                t.assert_equal(published,publication.committed);
            }
            t.assert_equal(active,session->tokens()); t.assert_true(!session->active() && !session->failed());
        }
        t.assert_true(session->layout_revision() > 1);
        t.assert_true(!session->begin(770,1,true));
        session.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(writer.arena.get()));
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(partial.arena.get()));
    });
    if (cuda) t.test("invalid_grants_and_shapes_are_rejected_before_session_creation", [&](testing & t) {
        fixture f(backend.get(),true); block_workspace writer(f,32768,19), partial(f,65536,29), duplicate(f,65536,19), tiny(f,8,39);
        const llama_kv_stream_session_config config{f.policy,1,4,false};
        t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,config,f.lease.get(),writer.lease.get(),duplicate.lease.get()));
        t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,config,f.lease.get(),f.lease.get(),partial.lease.get()));
        t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,config,f.lease.get(),writer.lease.get(),tiny.lease.get()));
        auto wrong = config; wrong.query_heads = 3;
        t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,wrong,f.lease.get(),writer.lease.get(),partial.lease.get()));
        wrong = config; wrong.max_batch_rows = 0;
        t.assert_true(!llama_kv_stream_session::create(backend.get(),f.content,wrong,f.lease.get(),writer.lease.get(),partial.lease.get()));
    });
    if (cuda) t.test("abort_and_external_mutation_close_append_admission", [&](testing & t) {
        for (bool mutate : {false,true}) {
            fixture f(backend.get(),true); block_workspace writer(f,32768,19), partial(f,65536,29);
            auto session = llama_kv_stream_session::create(backend.get(),f.content,{f.policy,1,4,false},f.lease.get(),writer.lease.get(),partial.lease.get());
            if (!t.assert_true(bool(session))) return;
            t.assert_true(!session->begin(2,1,true));
            t.assert_true(!session->failed());
            if (mutate) { t.assert_true(f.content->invalidate()); t.assert_true(!session->begin(1,1,true)); }
            else { t.assert_true(session->begin(1,1,false)); session->abort(); }
            t.assert_true(session->failed() && !session->active());
            t.assert_true(!session->begin(1,1,false));
            t.assert_equal(size_t(0),session->tokens());
        }
    });
    if (cuda) t.test("restored_frontier_reopens_serial_append", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,false,4);
        block_workspace writer(f,32768,19), partial(f,f.host->layout().bytes,29);
        auto session=llama_kv_stream_session::create(backend.get(),f.content,{f.policy,256,4,false,true,true},
            f.lease.get(),writer.lease.get(),partial.lease.get());
        if (!t.assert_true(bool(session))) return;
        t.assert_true(!session->restore(770));
        t.assert_true(session->restore(513));
        t.assert_equal(size_t(513),session->tokens());
        t.assert_true(!session->restore(512));
        t.assert_true(session->begin(514,1,true));
        t.assert_true(!session->restore(0));
        session->abort();
        t.assert_true(!session->restore(513));
    });
    if (cuda) t.test("two_token_decode_uses_bounded_resume_workspace", [&](testing & t) {
        constexpr size_t active=25601;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,2,112);
        f.policy.initial_ring_slots=96;
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get())),"ggml_backend_kv_stream_partial_ops"));
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(get && get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,
                4,2,2,f.host->layout().tokens,plan))) return;
        block_workspace writer(f,32768,19), attention(f,plan.bytes,29);
        llama_kv_stream_session_config config{f.policy,2,4,false,true,true};
        config.initial_decode=true;
        auto session=llama_kv_stream_session::create(backend.get(),f.content,config,
            f.lease.get(),writer.lease.get(),attention.lease.get());
        if (!t.assert_true(bool(session) && session->restore(active-2))) return;
        session_inputs kv(backend.get(),2);
        block_inputs input(f,active,2);
        if (!t.assert_true(session->begin(active,2,true))) return;
        std::array<std::vector<float>,2> outputs;
        for (uint32_t layer=0;layer<2;++layer) {
            if (!t.assert_true(session->produce(layer,kv.k,kv.v) &&
                    session->attention(layer,input.q,input.mask,input.output,1.0f/16))) return;
            ggml_backend_synchronize(backend.get());
            outputs[layer]=input.read();
        }
        t.assert_equal(active,session->tokens());
        t.assert_true(!session->active() && !session->failed());
        t.assert_true(session->sequence_stats().copy_calls > 0 && session->sequence_stats().copy_calls <= 14);
        for (uint32_t layer=0;layer<2;++layer) close_values(t,oracle(f,layer,active,2,input.qdata),outputs[layer],1e-3f);
        for (uint32_t layer=0;layer<2;++layer) t.assert_true(same_float_bits(stock_attention(f,input,layer),outputs[layer]));
        t.assert_true(session->attention_workspace_bytes() == plan.bytes);
    });

    if (cuda) t.test("qwen_tg3_tg4_decode_uses_stock_mma_and_bounded_ring", [&](testing & t) {
        constexpr size_t active=25601;
        for (uint32_t queries : {3u,4u}) {
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,8,112);
            f.policy.initial_ring_slots=96;
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get())),"ggml_backend_kv_stream_partial_ops"));
            ggml_kv_stream_resume_plan vector_plan;
            size_t mma_bytes=0;
            if (!t.assert_true(get && get()->version >= 9 && get()->mma_workspace &&
                    get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,64,8,2,f.host->layout().tokens,vector_plan) &&
                    get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,64,8,f.host->layout().tokens,3,mma_bytes))) return;
            const size_t scratch=std::max(vector_plan.bytes,mma_bytes);
            block_workspace writer(f,1024*1024,19), attention(f,scratch,29);
            llama_kv_stream_session_config config{f.policy,4,64,false,true,true};
            config.initial_decode=true;
            auto session=llama_kv_stream_session::create(backend.get(),f.content,config,
                f.lease.get(),writer.lease.get(),attention.lease.get());
            if (!t.assert_true(bool(session) && session->restore(active-queries))) return;
            session_inputs kv(backend.get(),queries,2048);
            block_inputs input(f,active,queries,false,64);
            if (!t.assert_true(session->begin(active,queries,true))) return;
            std::array<std::vector<float>,2> outputs;
            for (uint32_t layer=0;layer<2;++layer) {
                if (!t.assert_true(session->produce(layer,kv.k,kv.v) &&
                        session->attention(layer,input.q,input.mask,input.output,1.0f/16))) return;
                ggml_backend_synchronize(backend.get());
                outputs[layer]=input.read();
            }
            for (uint32_t layer=0;layer<2;++layer)
                t.assert_true(same_float_bits(stock_attention(f,input,layer),outputs[layer]));
            t.assert_equal(active,session->tokens());
            t.assert_true(!session->active() && !session->failed());
            t.assert_true(session->sequence_stats().copy_calls > 0 && session->sequence_stats().copy_calls <= 14);
            t.assert_equal(scratch,session->attention_workspace_bytes());
        }
    });

    if (cuda) t.test("tg4_decode_rejects_incomplete_ring_without_mutation", [&](testing & t) {
        constexpr size_t active=25601;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,2,8,112);
        f.policy.initial_ring_slots=4; f.policy.fixed_ring=true;
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get())),"ggml_backend_kv_stream_partial_ops"));
        ggml_kv_stream_resume_plan vector_plan;
        size_t mma_bytes=0;
        if (!t.assert_true(get && get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,
                64,8,2,f.host->layout().tokens,vector_plan) && get()->mma_workspace(backend.get(),
                GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,64,8,f.host->layout().tokens,3,mma_bytes))) return;
        block_workspace writer(f,1024*1024,19), attention(f,std::max(vector_plan.bytes,mma_bytes),29);
        llama_kv_stream_session_config config{f.policy,4,64,false,true,true};
        config.initial_decode=true;
        auto session=llama_kv_stream_session::create(backend.get(),f.content,config,
            f.lease.get(),writer.lease.get(),attention.lease.get());
        if (!t.assert_true(bool(session) && session->restore(active-4))) return;
        const auto revision=session->layout_revision();
        t.assert_true(!session->begin(active,4,true));
        t.assert_true(!session->failed() && !session->active());
        t.assert_equal(active-4,session->tokens());
        t.assert_equal(revision,session->layout_revision());
    });

    if (cuda) t.test("failure_after_production_never_commits_the_token_frontier", [&](testing & t) {
        fixture f(backend.get(),true); block_workspace writer(f,32768,19), partial(f,65536,29);
        auto session = llama_kv_stream_session::create(backend.get(),f.content,{f.policy,1,4,false},f.lease.get(),writer.lease.get(),partial.lease.get());
        if (!t.assert_true(bool(session))) return;
        session_inputs input(backend.get(),1); block_inputs attn(f,1,1);
        t.assert_true(session->begin(1,1,false));
        t.assert_true(!session->produce(1,input.k,input.v));
        t.assert_true(session->produce(0,input.k,input.v));
        auto invalid = *attn.q; invalid.ne[2] = 3;
        t.assert_true(!session->attention(0,&invalid,attn.mask,attn.output,1.0f/16));
        t.assert_true(session->failed() && !session->active());
        t.assert_equal(size_t(0),session->tokens());
        t.assert_true(!session->produce(0,input.k,input.v));
    });

    if (cuda) t.test("matching_decode_adopts_cross_token_prefetch_and_mismatch_drains_it", [&](testing & t) {
        const auto run=[&](bool carry) {
            std::vector<std::vector<float>> outputs;
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,false,4);
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            f.policy.pool_bytes=8*page.bytes; f.policy.initial_ring_slots=4;
            block_workspace writer(f,32768,19),partial(f,f.host->layout().bytes,29);
            llama_kv_stream_session_config config{f.policy,256,4,false,true,true};
            config.cross_token_prefetch=carry;
            auto session=llama_kv_stream_session::create(backend.get(),f.content,config,
                f.lease.get(),writer.lease.get(),partial.lease.get());
            if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return outputs;
            for (size_t active : {size_t(514),size_t(515)}) {
                const bool had_prime=session->prefetch_primed();
                const auto before=session->sequence_stats();
                session_inputs kv(backend.get(),1); block_inputs attn(f,active,1);
                if (!t.assert_true(session->begin(active,1,true))) return std::vector<std::vector<float>>{};
                if (carry && active == 515) {
                    t.assert_true(had_prime && before.primed && before.copy_calls>0);
                    const auto adopted=session->sequence_stats();
                    t.assert_true(adopted.adopted && adopted.copy_calls==before.copy_calls);
                } else t.assert_true(!had_prime);
                for (uint32_t layer=0;layer<4;++layer) {
                    if (!t.assert_true(session->produce(layer,kv.k,kv.v)) ||
                            !t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16)))
                        return std::vector<std::vector<float>>{};
                    ggml_backend_synchronize(backend.get());
                    if (active == 515) outputs.push_back(attn.read());
                }
                t.assert_equal(carry,session->prefetch_primed());
            }
            if (carry) {
                t.assert_true(session->set_attention_workspace(partial.lease.get(),true));
                t.assert_true(!session->prefetch_primed());
                session_inputs kv(backend.get(),2); block_inputs attn(f,517,2);
                if (!t.assert_true(session->begin(517,2,false))) return std::vector<std::vector<float>>{};
                for (uint32_t layer=0;layer<4;++layer) {
                    if (!t.assert_true(session->produce(layer,kv.k,kv.v)) ||
                            !t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16)))
                        return std::vector<std::vector<float>>{};
                }
                t.assert_true(!session->prefetch_primed());
            }
            return outputs;
        };
        const auto control=run(false),carried=run(true);
        if (!t.assert_equal(control.size(),carried.size()) || !t.assert_equal(size_t(4),control.size())) return;
        for (size_t layer=0;layer<control.size();++layer) close_values(t,control[layer],carried[layer],1e-6f);
    });

    if (cuda) t.test("cross_token_feedback_preserves_continuity_without_false_growth", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1281,false,4);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        f.policy.pool_bytes=16*page.bytes; f.policy.initial_ring_slots=4;
        f.policy.grow_evaluations=100; f.policy.cooldown_evaluations=1;
        block_workspace writer(f,32768,19),partial(f,f.host->layout().bytes,29);
        llama_kv_stream_session_config config{f.policy,256,4,true,true,true};
        config.cross_token_prefetch=true;
        auto session=llama_kv_stream_session::create(backend.get(),f.content,config,
            f.lease.get(),writer.lease.get(),partial.lease.get());
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(1024))) return;
        const auto initial=session->policy();
        t.assert_equal(uint32_t(3),initial.resident_pages_per_layer);
        t.assert_equal(uint32_t(4),initial.ring_slots);
        for (size_t active=1025;active<=1032;++active) {
            session_inputs kv(backend.get(),1); block_inputs attn(f,active,1);
            if (!t.assert_true(session->begin(active,1,true))) return;
            for (uint32_t layer=0;layer<4;++layer) {
                if (!t.assert_true(session->produce(layer,kv.k,kv.v)) ||
                        !t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16))) return;
            }
            ggml_backend_synchronize(backend.get());
        }
        t.assert_true(session->policy().samples > 0);
        t.assert_equal(initial.resident_pages_per_layer,session->policy().resident_pages_per_layer);
        t.assert_equal(initial.ring_slots,session->policy().ring_slots);
    });

    if (cuda) t.test("external_growth_rebinds_and_lazily_restores_host", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,false,4);
        block_workspace writer(f,32768,19), partial(f,f.host->layout().bytes,29);
        auto session=llama_kv_stream_session::create(backend.get(),f.content,{f.policy,256,4,false,true,true},
            f.lease.get(),writer.lease.get(),partial.lease.get());
        if (!t.assert_true(bool(session))) return;
        if (!t.assert_true(session->restore(513))) return;
        const auto old_view=session->binding_view();
        const auto old_policy=session->policy();
        const auto old_revision=session->layout_revision();
        const auto content_generation=f.content->generation();
        const auto mirror_epoch=f.content->mirror_epoch();

        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        const size_t grown_bytes=f.policy.pool_bytes+8*page.bytes;
        block_workspace grown(f,grown_bytes,41);
        const auto arena_generation=ggml_backend_memory_arena_generation(grown.arena.get());
        t.assert_true(!session->grow_pool(f.lease.get(),f.policy.pool_bytes,true));
        t.assert_true(!session->grow_pool(writer.lease.get(),grown_bytes,true));
        t.assert_equal(old_revision,session->layout_revision());
        t.assert_equal(old_view.base,session->binding_view().base);
        f.lease.reset();
        if (!t.assert_true(session->grow_pool(grown.lease.get(),grown_bytes,true))) return;
        const auto new_view=session->binding_view();
        t.assert_equal(size_t(513),session->tokens());
        t.assert_equal(content_generation,f.content->generation());
        t.assert_equal(mirror_epoch,f.content->mirror_epoch());
        t.assert_equal(old_revision+1,session->layout_revision());
        t.assert_equal(old_view.revision+1,new_view.revision);
        t.assert_equal(old_view.cache_id,new_view.cache_id);
        t.assert_true(old_view.base!=new_view.base);
        t.assert_equal(grown_bytes,new_view.capacity);
        t.assert_equal(grown_bytes+32768+
            ggml_backend_buffer_get_size(ggml_backend_memory_lease_buffer(partial.lease.get())),
            session->granted_bytes());
        const auto publication=session->publication_frontiers();
        t.assert_equal(size_t(513),publication.reserved);
        t.assert_equal(size_t(513),publication.device);
        t.assert_equal(size_t(513),publication.host);
        t.assert_equal(size_t(513),publication.committed);
        t.assert_true(session->policy().budget.pages>old_policy.budget.pages);
        t.assert_true(session->policy().resident_pages_per_layer>=old_policy.resident_pages_per_layer);
        t.assert_equal(arena_generation,ggml_backend_memory_arena_generation(grown.arena.get()));
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(f.arena.get()));

        session_inputs input(backend.get(),1); block_inputs attn(f,514,1);
        if (!t.assert_true(session->begin(514,1,true))) return;
        t.assert_true(f.content->mirror_epoch()>mirror_epoch);
        std::vector<std::vector<float>> outputs;
        for (uint32_t layer=0;layer<4;++layer) {
            if (!t.assert_true(session->produce(layer,input.k,input.v)) ||
                    !t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16))) return;
            ggml_backend_synchronize(backend.get());
            outputs.push_back(attn.read());
        }
        t.assert_equal(size_t(514),session->tokens());
        // Host KV becomes authoritative after the complete asynchronous append commits.
        for (uint32_t layer=0;layer<4;++layer) {
            t.out << "grown-pool stock comparison layer " << layer << ": ";
            close_values(t,stock_attention(f,attn,layer),outputs[layer],1e-5f);
            close_values(t,oracle(f,layer,514,1,attn.qdata),outputs[layer],1e-3f);
        }
        grown.lease.reset();
        session.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grown.arena.get()));
    });

    if (cuda) t.test("external_shrink_drains_old_work_and_preserves_cache", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,false,4);
        block_workspace writer(f,32768,19),partial(f,f.host->layout().bytes,29);
        auto session=llama_kv_stream_session::create(backend.get(),f.content,{f.policy,256,4,false,true,false},
            f.lease.get(),writer.lease.get(),partial.lease.get());
        if (!t.assert_true(bool(session))||!t.assert_true(session->restore(513))) return;
        const auto old_view=session->binding_view();
        const auto old_revision=session->layout_revision();
        const auto content_generation=f.content->generation();
        const auto mirror_epoch=f.content->mirror_epoch();

        ggml_kv_stream_layout page;ggml_kv_stream_layout_make(f.policy.shape,256,page);
        ggml_kv_stream_execution execution;ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,execution);
        const size_t minimum=(f.policy.layers+1)*page.bytes+execution.conversion.bytes;
        block_workspace shrunk(f,minimum,51);

        ggml_context_ptr pending_ctx(ggml_init({4096,nullptr,true}));
        auto * pending_tensor=ggml_new_tensor_1d(pending_ctx.get(),GGML_TYPE_F32,64);
        if (!t.assert_true(ggml_backend_tensor_alloc(
                old_view.buffer,pending_tensor,old_view.base)==GGML_STATUS_SUCCESS)) return;
        std::vector<float> expected(64),copied(64);
        for (size_t i=0;i<expected.size();++i) expected[i]=float(i)+.5f;
        ggml_backend_tensor_set_async(backend.get(),pending_tensor,expected.data(),0,expected.size()*sizeof(float));
        ggml_backend_tensor_get_async(backend.get(),pending_tensor,copied.data(),0,copied.size()*sizeof(float));

        t.assert_true(!session->shrink_pool(shrunk.lease.get(),f.policy.pool_bytes,true));
        t.assert_equal(old_revision,session->layout_revision());
        f.lease.reset();
        if (!t.assert_true(session->shrink_pool(shrunk.lease.get(),minimum,true))) return;
        const auto new_view=session->binding_view();
        t.assert_true(copied==expected);
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(f.arena.get()));
        t.assert_equal(old_revision+1,session->layout_revision());
        t.assert_equal(old_view.revision+1,new_view.revision);
        t.assert_true(old_view.base!=new_view.base);
        t.assert_equal(minimum,new_view.capacity);
        t.assert_equal(uint32_t(5),session->policy().budget.pages);
        t.assert_equal(uint32_t(1),session->policy().resident_pages_per_layer);
        t.assert_equal(content_generation,f.content->generation());
        t.assert_equal(mirror_epoch,f.content->mirror_epoch());
        const auto publication=session->publication_frontiers();
        t.assert_equal(size_t(513),publication.reserved);
        t.assert_equal(size_t(513),publication.device);
        t.assert_equal(size_t(513),publication.host);
        t.assert_equal(size_t(513),publication.committed);

        session_inputs input(backend.get(),1);block_inputs attn(f,514,1);
        if (!t.assert_true(session->begin(514,1,true))) return;
        t.assert_true(f.content->mirror_epoch()>mirror_epoch);
        for (uint32_t layer=0;layer<4;++layer) {
            if (!t.assert_true(session->produce(layer,input.k,input.v))||
                    !t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16))) return;
            close_values(t,oracle(f,layer,514,1,attn.qdata),attn.read(),1e-3f);
        }
        t.assert_equal(size_t(514),session->tokens());
        shrunk.lease.reset();session.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(shrunk.arena.get()));
    });
    if (cuda) t.test("shared_physical_pool_publishes_only_target_pairs", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_F16, GGML_TYPE_F16, 769, false, 2, 2, 16);
        f.policy.layers = 3;
        f.policy.caches = {{999, 1}, {f.host->cache_id(), 2}};
        auto * pool = ggml_backend_memory_lease_buffer(f.lease.get());
        ggml_backend_buffer_clear(pool, 0xa5);
        ggml_kv_stream_block_layout work;
        if (!t.assert_true(ggml_kv_stream_block_layout_make(4, 256, work).status ==
                ggml_kv_stream_partial_status::success)) return;
        block_workspace writer(f, 32768, 19), attention(f, work.bytes, 29);
        auto session = llama_kv_stream_session::create(backend.get(), f.content,
            {f.policy, 1, 4, false}, f.lease.get(), writer.lease.get(), attention.lease.get());
        if (!t.assert_true(bool(session))) return;
        session_inputs kv(backend.get(), 1);
        block_inputs input(f, 1, 1);
        if (!t.assert_true(session->begin(1, 1, false))) return;
        for (uint32_t local = 0; local < 2; ++local) {
            if (!t.assert_true(session->produce(local, kv.k, kv.v) &&
                    session->attention(local, input.q, input.mask, input.output, 1.0f/16))) return;
            close_values(t, oracle(f, local, 1, 1, input.qdata), input.read(), 1e-3f);
        }
        t.assert_equal(size_t(1), session->tokens());
        t.assert_equal(size_t(1), session->publication_frontiers().committed);
        t.assert_true(!session->active() && !session->failed());
        llama_kv_stream_policy_layout physical;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                f.policy, session->policy(), 1, physical).status == llama_kv_stream_policy_status::success)) return;
        ggml_context_ptr context(ggml_init({8192, nullptr, true}));
        for (size_t offset : {physical.layers[0].offset,
                physical.layers[0].offset + physical.layers[0].planes.v_offset}) {
            auto * marker = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, 128);
            if (!t.assert_true(ggml_backend_tensor_alloc(pool, marker,
                    static_cast<char *>(ggml_backend_buffer_get_base(pool)) + offset) == GGML_STATUS_SUCCESS)) return;
            std::vector<uint8_t> bytes(128);
            ggml_backend_tensor_get(marker, bytes.data(), 0, bytes.size());
            t.assert_true(std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0xa5; }));
        }
    });
    if (cuda) t.test("guarded_target_session_preserves_mtp_ring_and_releases_for_replan", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 513, false, 2, 2, 7);
        f.policy.layers = 3;
        f.policy.caches = {{999, 1}, {f.host->cache_id(), 2}};
        f.policy.initial_ring_slots = 3;
        f.policy.fixed_ring = true;
        ggml_kv_stream_block_layout work;
        if (!t.assert_true(ggml_kv_stream_block_layout_make(256*4, 256, work).status ==
                ggml_kv_stream_partial_status::success)) return;
        block_workspace writer(f, 32768, 19), attention(f, work.bytes, 29);
        auto session = llama_kv_stream_session::create(backend.get(), f.content,
            {f.policy, 256, 4, false}, f.lease.get(), writer.lease.get(), attention.lease.get());
        if (!t.assert_true(bool(session))) return;
        const auto append = [&](size_t active, uint32_t rows, bool decode) {
            session_inputs kv(backend.get(), rows);
            block_inputs input(f, active, rows);
            if (!t.assert_true(session->begin(active, rows, decode))) return false;
            for (uint32_t layer = 0; layer < 2; ++layer) {
                if (!t.assert_true(session->produce(layer, kv.k, kv.v) &&
                        session->attention(layer, input.q, input.mask, input.output, 1.0f/16))) return false;
            }
            return t.assert_equal(active, session->tokens());
        };
        if (!append(256, 256, false)) return;
        const auto before_repartition = session->layout_revision();
        auto initial_owner = llama_kv_stream_layer_lease_owner::create(f.lease.get(),
            {f.policy, session->policy(), 256, before_repartition, f.content->generation()});
        if (!t.assert_true(bool(initial_owner))) return;
        auto * initial_mtp = initial_owner->acquire(
            {0, 1, before_repartition, f.content->generation(), 999, 256});
        if (!t.assert_true(initial_mtp != nullptr)) return;
        auto initial_guard = initial_owner->hold_ring();
        if (!t.assert_true(session->set_ring_guard(initial_guard))) return;
        t.assert_true(!session->begin(257, 1, true));
        t.assert_true(!session->failed());
        t.assert_equal(before_repartition, session->layout_revision());
        t.assert_true(session->set_ring_guard({}));
        initial_guard.reset();
        llama_kv_stream_complete_layer_lease_free(initial_mtp);
        t.assert_true(initial_owner->can_repartition());
        if (!append(257, 1, true)) return;
        t.assert_true(session->layout_revision() > before_repartition);
        const auto revision = session->layout_revision();
        auto owner = llama_kv_stream_layer_lease_owner::create(f.lease.get(),
            {f.policy, session->policy(), 257, revision, f.content->generation()});
        if (!t.assert_true(bool(owner))) return;
        auto * mtp = owner->acquire({0, 1, revision, f.content->generation(), 999, 257});
        if (!t.assert_true(mtp != nullptr)) return;
        auto guard = owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        t.assert_true(owner->ring_slots_used() > 0);
        fixture foreign(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 513, false, 2, 2, 7);
        auto foreign_owner = llama_kv_stream_layer_lease_owner::create(foreign.lease.get(),
            {f.policy, session->policy(), 257, revision, f.content->generation()});
        if (t.assert_true(bool(foreign_owner))) {
            t.assert_true(!session->set_ring_guard(foreign_owner->hold_ring()));
        }
        if (!t.assert_true(session->set_ring_guard(guard))) return;
        llama_kv_stream_policy_layout physical;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                f.policy, session->policy(), 257, physical).status == llama_kv_stream_policy_status::success)) return;
        auto * pool = ggml_backend_memory_lease_buffer(f.lease.get());
        const size_t first = llama_kv_stream_complete_layer_lease_ring_first(mtp);
        const size_t page = size_t(f.policy.shape.page_tokens);
        const size_t k_offset = first*page*physical.ring.k_token_bytes;
        const size_t v_offset = physical.ring.v_offset + first*page*physical.ring.v_token_bytes;
        ggml_context_ptr marker_ctx(ggml_init({8192, nullptr, true}));
        std::array<ggml_tensor *, 2> markers{};
        const std::vector<uint8_t> sentinel(128, 0xa5);
        for (size_t i = 0; i < markers.size(); ++i) {
            markers[i] = ggml_new_tensor_1d(marker_ctx.get(), GGML_TYPE_I8, sentinel.size());
            const size_t offset = i ? v_offset : k_offset;
            if (!t.assert_true(ggml_backend_tensor_alloc(pool, markers[i],
                    static_cast<char *>(ggml_backend_buffer_get_base(pool)) + offset) == GGML_STATUS_SUCCESS)) return;
            ggml_backend_tensor_set(markers[i], sentinel.data(), 0, sentinel.size());
        }
        if (!append(258, 1, true)) return;
        t.assert_equal(revision, session->layout_revision());
        const auto prefetch = session->sequence_stats();
        t.assert_true(prefetch.copy_bytes > 0);
        t.assert_true(prefetch.peak_pages <=
            session->policy().ring_slots - owner->ring_slots_used());
        for (auto * marker : markers) {
            std::vector<uint8_t> actual(sentinel.size());
            ggml_backend_tensor_get(marker, actual.data(), 0, actual.size());
            t.assert_true(actual == sentinel);
        }
        t.assert_true(session->set_ring_guard({}));
        guard.reset();
        llama_kv_stream_complete_layer_lease_free(mtp);
        t.assert_true(owner->can_repartition());
        t.assert_true(!session->failed());
    });
    return t.summary();
}
