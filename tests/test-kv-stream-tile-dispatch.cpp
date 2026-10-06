#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-model.h"

#ifdef KV_STREAM_TILE_NATIVE_TEST
bool kv_stream_test_is_sm61_only();
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
#endif

// Keep the accepted legacy rounding bound; refill boundaries themselves may not add error.
static void qualify(testing & t,const std::vector<float> & expected,const std::vector<float> & actual) {
    if (!t.assert_equal(expected.size(),actual.size())) return;
    double squared=0,reference=0,maximum=0;
    bool finite=true;
    for (size_t i=0;i<actual.size();++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const double error=double(actual[i])-expected[i];
        maximum=std::max(maximum,std::abs(error)); squared+=error*error; reference+=double(expected[i])*expected[i];
    }
    const double relative=reference ? std::sqrt(squared/reference) : squared ? INFINITY : 0;
    t.out << "max_abs=" << maximum << " normalized_l2=" << relative << '\n';
    t.assert_true(finite && maximum <= 1e-8 && relative <= 8*std::numeric_limits<float>::epsilon());
}

int main(int argc, char ** argv) {
    const bool pascal=argc == 2 && !std::strcmp(argv[1],"--pascal");
    if (!pascal && (argc != 2 || std::strcmp(argv[1],"--cuda"))) return 77;
#ifndef KV_STREAM_TILE_NATIVE_TEST
    return 77;
#else
    if (pascal && !kv_stream_test_is_sm61_only()) return 77;
    ggml_backend_load_all();
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!dev) return 77;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    const int previous=pascal ? kv_stream_test_override_cc(backend.get(),610) : 0;
    struct restore {
        ggml_backend_t backend; int cc; bool enabled;
        ~restore() {if (enabled) kv_stream_test_override_cc(backend,cc);}
    } restore_cc{backend.get(),previous,pascal};
    auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
    testing t;
    t.test("tile_decode_is_admitted_and_sizes_bounded_scratch",[&](testing & t) {
        if (!t.assert_true(get && get() && get()->version >= 10 && get()->resume_plan && get()->decode_workspace)) return;
        size_t maximum=0;
        if (!t.assert_true(get()->decode_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,4,1280,maximum))) return;
        for (uint32_t queries : {1u,2u,3u,4u}) {
            ggml_kv_stream_resume_plan plan;
            const bool admitted=get()->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,queries,1280,plan);
            if (pascal || queries <= 2) {
                if (!t.assert_true(admitted)) return;
                t.assert_true(plan.bytes > 0 && plan.bytes <= maximum);
                t.assert_equal(uint32_t(pascal && queries >= 3),plan.kernel_config[0]);
            }
        }
        size_t untouched=77;
        t.assert_true(!get()->decode_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,5,1280,untouched));
        t.assert_equal(size_t(77),untouched);
    });
    if (pascal) t.test("target_tile_decode_survives_narrow_ring_refills_and_all_resident",[&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_F16,GGML_TYPE_F16},
                std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q4_1}}) for (size_t pool_pages : {size_t(3),size_t(40)}) {
            fixture f(backend.get(),true,pair.first,pair.second,4097,false,2,4,pool_pages);
            f.policy.initial_ring_slots=1;
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
            auto pin=f.binding->acquire();
            for (size_t active : {size_t(513),size_t(4097)}) for (uint32_t queries : {1u,2u,3u,4u}) {
                block_inputs input(f,active,queries,false,24);
                size_t bytes=0;
                if (!t.assert_true(get()->decode_workspace(backend.get(),pair.first,pair.second,24,4,4,input.padded,bytes))) return;
                block_workspace workspace(f,bytes);
                if (!t.assert_true(f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return;
                for (uint32_t layer=0;layer<2;++layer) {
                    const auto expected=stock_attention(f,input,layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,
                            active,1.0f/16,workspace.lease.get(),true,1))) return;
                    ggml_backend_synchronize(backend.get()); qualify(t,expected,input.read());
                }
                t.assert_true(!f.resident->sequence_active());
            }
        }
    });
    if (pascal) t.test("mtp_spans_preserve_stock_padded_geometry_and_reject_short_scratch",[&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1025,false,2,4,16);
        if (!t.assert_true(f.attach())) return;
        auto pin=f.binding->acquire();
        block_workspace storage(f,2*1024*1024,80);
        auto * buffer=ggml_backend_memory_lease_buffer(storage.lease.get());
        const auto base=uintptr_t(ggml_backend_buffer_get_base(buffer));
        for (size_t active : {size_t(257),size_t(513),size_t(1025)}) for (uint32_t queries : {1u,2u,3u,4u}) {
            t.out << "MTP active=" << active << " TG" << queries << '\n';
            block_inputs input(f,active,queries,false,24);
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            std::vector<ggml_kv_stream_span_source> sources;
            size_t offset=0;
            const std::vector<size_t> cuts{0,256,active};
            for (size_t i=0;i<2;++i) {
                ggml_kv_stream_layout layout;
                if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape,cuts[i+1]-cuts[i],layout).status == ggml_kv_stream_status::success)) return;
                auto transfer=[&](bool value) {
                    ggml_tensor dst{}; dst.type=GGML_TYPE_I8; dst.buffer=buffer;
                    dst.data=reinterpret_cast<void *>(base+offset+(value ? layout.v_offset : 0));
                    dst.ne[0]=int64_t(value ? layout.v_bytes : layout.k_bytes); dst.nb[0]=1;
                    for (int d=1;d<4;++d) {dst.ne[d]=1; dst.nb[d]=size_t(dst.ne[0]);}
                    ggml_backend_tensor_set(&dst,static_cast<const char *>(value ? host.v : host.k)+
                        cuts[i]*(value ? layout.v_token_bytes : layout.k_token_bytes),0,size_t(dst.ne[0]));
                };
                transfer(false); transfer(true);
                sources.push_back({storage.lease.get(),storage.lease.get(),cuts[i],cuts[i+1]-cuts[i],offset,offset+layout.v_offset});
                offset=(offset+layout.bytes+127)/128*128;
            }
            ggml_kv_stream_span_plan_t raw=nullptr;
            if (!t.assert_true(ggml_kv_stream_span_plan_make(f.policy.shape,sources.data(),sources.size(),active,queries,raw).status == ggml_kv_stream_status::success)) return;
            std::unique_ptr<ggml_kv_stream_span_plan,decltype(&ggml_kv_stream_span_plan_free)> plan(raw,ggml_kv_stream_span_plan_free);
            ggml_kv_stream_layout logical;
            if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape,input.padded,logical).status == ggml_kv_stream_status::success)) return;
            ggml_tensor k{},v{},op=*input.output;
            for (auto * tensor : {&k,&v}) {
                tensor->type=tensor == &k ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
                tensor->ne[0]=256; tensor->ne[1]=int64_t(input.padded); tensor->ne[2]=4; tensor->ne[3]=1;
                tensor->nb[0]=ggml_type_size(tensor->type);
                tensor->nb[1]=tensor == &k ? logical.k_token_bytes : logical.v_token_bytes;
                tensor->nb[2]=tensor == &k ? logical.k_row_bytes : logical.v_row_bytes;
                tensor->nb[3]=tensor == &k ? logical.k_bytes : logical.v_bytes;
            }
            op.op=GGML_OP_FLASH_ATTN_EXT; op.src[0]=input.q; op.src[1]=&k; op.src[2]=&v; op.src[3]=input.mask;
            const float scale=1.0f/16; std::memcpy(op.op_params,&scale,sizeof(scale));
            ggml_flash_attn_ext_set_prec(&op,GGML_PREC_F32);
            size_t bytes=0;
            if (!t.assert_true(get()->spans_workspace(backend.get(),&op,raw,bytes))) return;
            block_workspace exact(f,bytes,81), tiny(f,bytes-1,82);
            if (queries >= 3) {
                t.assert_true(!get()->spans(backend.get(),&op,raw,ggml_backend_memory_lease_buffer(tiny.lease.get())));
                const auto unchanged=input.read(); t.assert_true(std::all_of(unchanged.begin(),unchanged.end(),[](float v){return v == -77;}));
            }
            if (!t.assert_true(get()->spans(backend.get(),&op,raw,ggml_backend_memory_lease_buffer(exact.lease.get())))) return;
            ggml_backend_synchronize(backend.get()); qualify(t,stock_attention(f,input,0),input.read());
        }
    });
    if (pascal) t.test("tile_model_owner_reserves_all_decode_widths_without_a_kv_gather",[&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4097,false,2,4,16);
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,24});
        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(bool(model) && model->memory_requirements(requirements))) return;
        size_t bytes=0;
        t.assert_true(get()->decode_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,4,f.host->layout().tokens,bytes));
        t.assert_equal(bytes,requirements.attention_decode_bytes);
        t.assert_true(bytes < requirements.attention_prefill_bytes);
    });
    if (pascal) t.test("tile_resume_rejects_stale_configuration_and_aliases_before_publication",[&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,513,false,2,4,40);
        if (!t.assert_true(f.attach())) return;
        auto pin=f.binding->acquire(); block_inputs input(f,513,3,false,24);
        if (!t.assert_true(f.resident->synchronize(input.active))) return;
        ggml_context_ptr context(ggml_init({65536,nullptr,true}));
        auto * node=f.resident->attention(context.get(),0,input.q,input.mask,input.active,1.0f/16);
        if (!t.assert_true(node != nullptr)) return;
        auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(get()->resume_plan(backend.get(),GGML_TYPE_F16,GGML_TYPE_F16,24,4,3,input.padded,plan))) return;
        block_workspace scratch(f,plan.bytes,90), short_grant(f,plan.bytes-1,91);
        auto * buffer=ggml_backend_memory_lease_buffer(scratch.lease.get());
        for (int fault=0;fault<7;++fault) {
            auto bad=plan;
            if (fault == 0) ++bad.kernel_config[4];
            if (fault == 1) ++bad.kernel_config[2];
            if (fault == 2) bad.splits=0;
            if (fault == 3) ++bad.state_bytes;
            if (fault == 4) ++bad.bytes;
            if (fault == 5) ++bad.meta_offset;
            if (fault == 6) bad.resume_resident=false;
            t.assert_true(!get()->resume(backend.get(),&op,buffer,bad,input.padded,0,true));
        }
        t.assert_true(!get()->resume(backend.get(),&op,ggml_backend_memory_lease_buffer(short_grant.lease.get()),plan,input.padded,0,true));
        t.assert_true(!get()->resume(backend.get(),&op,buffer,plan,input.padded,1,true));
        t.assert_true(!get()->resume(backend.get(),&op,buffer,plan,input.padded,0,false));
        auto alias=op; alias.data=input.q->data; alias.buffer=input.q->buffer;
        t.assert_true(!get()->resume(backend.get(),&alias,buffer,plan,input.padded,0,true));
        const auto actual=input.read();
        t.assert_true(std::all_of(actual.begin(),actual.end(),[](float v){return v == -77;}));
        t.assert_true(get()->resume(backend.get(),&op,buffer,plan,input.padded,0,true));
        ggml_backend_synchronize(backend.get()); qualify(t,stock_attention(f,input,0),input.read());
    });
    return t.summary();
#endif
}
