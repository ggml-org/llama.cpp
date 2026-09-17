#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-session.h"

struct session_inputs {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k, * v;
    session_inputs(ggml_backend_t backend, size_t rows) {
        ctx.reset(ggml_init({4096,nullptr,true}));
        k = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,int64_t(rows));
        v = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,int64_t(rows));
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),backend)); GGML_ASSERT(buffer);
        std::vector<float> data(rows*512);
        for (size_t i = 0; i < data.size(); ++i) data[i] = .25f*std::cos(float(i%541)*.07f);
        ggml_backend_tensor_set(k,data.data(),0,data.size()*sizeof(float));
        ggml_backend_tensor_set(v,data.data(),0,data.size()*sizeof(float));
    }
};

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && !std::strcmp(argv[1],"--cuda");
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev,nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    testing t;
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
        for (uint32_t layer=0;layer<4;++layer) {
            if (!t.assert_true(session->produce(layer,input.k,input.v)) ||
                    !t.assert_true(session->attention(layer,attn.q,attn.mask,attn.output,1.0f/16))) return;
            close_values(t,oracle(f,layer,514,1,attn.qdata),attn.read(),1e-3f);
        }
        t.assert_equal(size_t(514),session->tokens());
        grown.lease.reset();
        session.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grown.arena.get()));
    });
    return t.summary();
}
