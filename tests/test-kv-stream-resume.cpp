#include "kv-stream-block-test.h"

static std::vector<float> ordinary(fixture & f, block_inputs & input, uint32_t layer) {
    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
    auto * k = ggml_new_tensor_2d(ctx.get(),ggml_type(f.policy.shape.type_k),512,input.padded);
    auto * v = ggml_new_tensor_2d(ctx.get(),ggml_type(f.policy.shape.type_v),512,input.padded);
    auto * key = ggml_view_3d(ctx.get(),k,256,input.padded,2,ggml_row_size(k->type,512),ggml_row_size(k->type,256),0);
    auto * value = ggml_view_3d(ctx.get(),v,256,input.padded,2,ggml_row_size(v->type,512),ggml_row_size(v->type,256),0);
    auto * out = ggml_flash_attn_ext(ctx.get(),input.q,key,value,input.mask,1.0f/16,0,0);
    ggml_flash_attn_ext_set_prec(out,GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),f.backend)); GGML_ASSERT(buffer);
    llama_kv_stream_host_layer host; GGML_ASSERT(f.host->layer(layer,host));
    ggml_backend_tensor_set(k,host.k,0,ggml_nbytes(k)); ggml_backend_tensor_set(v,host.v,0,ggml_nbytes(v));
    GGML_ASSERT(ggml_backend_graph_compute(f.backend,graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(ggml_nelements(out)); ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out)); return result;
}

int main(int argc, char ** argv) {
    testing t;
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
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("native_resume_contract_is_available", [&](testing & t) {
        ggml_backend_load_all(); auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        t.assert_true(get && get() && get()->version >= 5);
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
    return t.summary();
}
