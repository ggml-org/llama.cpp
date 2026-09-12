// Include the implementation to check trait ownership and view admission.
#include "ggml-cpu/repack.cpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
static int checks=0;
static void require_test(bool ok,const char *message) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);} ++checks;
}
static void graph_checks() {
    // Feed canonical Q4_K through both repackers, then compare real CPU graphs.
    for (int n : {256, 512}) {
        for (int nr : {1, 4, 5, 20}) {
            for (bool view : {false, true}) {
                auto *weights = ggml_init({1024*1024, nullptr, true});
                auto *ctx = ggml_init({8*1024*1024, nullptr, false});
                auto *w = ggml_new_tensor_2d(weights, GGML_TYPE_Q4_K, n, 24);
                auto *ref = ggml_new_tensor_2d(weights, GGML_TYPE_Q4_K, n, 24);
                auto *buft = ggml_backend_cpu_repack_buffer_type();
                auto *bw = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(w));
                auto *br = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(ref));
                require_test(bw && br, "graph weight allocation");
                require_test(ggml_backend_tensor_alloc(bw, w, ggml_backend_buffer_get_base(bw)) == GGML_STATUS_SUCCESS,
                             "graph candidate initialization");
                require_test(ggml_backend_tensor_alloc(br, ref, ggml_backend_buffer_get_base(br)) == GGML_STATUS_SUCCESS,
                             "graph reference initialization");
                static const ggml::cpu::repack::tensor_traits<block_q4_K,8,8,GGML_TYPE_Q8_K> original_trait;
                ref->extra = const_cast<ggml::cpu::repack::tensor_traits<block_q4_K,8,8,GGML_TYPE_Q8_K> *>(&original_trait);
                auto *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, nr);
                for (int i = 0; i < n*nr; ++i) {
                    static_cast<float *>(a->data)[i] = float((i*13+7)%251-125)/127.0f;
                }
                std::vector<float> floats(n*24);
                std::vector<uint8_t> canonical(ggml_nbytes(w));
                // Repeated loading must repack fresh canonical data, not recode P6 twice.
                for (int load = 0; load < 2; ++load) {
                    for (size_t i = 0; i < floats.size(); ++i) {
                        floats[i] = float(int((i*17+load*23)%257)-128)/129.0f;
                    }
                    require_test(ggml_quantize_chunk(GGML_TYPE_Q4_K, floats.data(), canonical.data(),
                                 0, 24, n, nullptr) == canonical.size(), "canonical quantization size");
                    ggml_backend_tensor_set(w, canonical.data(), 0, canonical.size());
                    ggml_backend_tensor_set(ref, canonical.data(), 0, canonical.size());
                    auto *wv = w;
                    auto *rv = ref;
                    if (view) {
                        wv = ggml_view_2d(weights, w, n, 8, w->nb[1], 8*w->nb[1]);
                        rv = ggml_view_2d(weights, ref, n, 8, ref->nb[1], 8*ref->nb[1]);
                        require_test(ggml_backend_view_init(wv) == GGML_STATUS_SUCCESS, "graph candidate view");
                        require_test(ggml_backend_view_init(rv) == GGML_STATUS_SUCCESS, "graph reference view");
                        rv->extra = ref->extra;
                    }
                    auto *out = ggml_mul_mat(ctx, wv, a);
                    auto *expected = ggml_mul_mat(ctx, rv, a);
                    auto *graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, expected);
                    ggml_build_forward_expand(graph, out);
                    const auto calls = q4kp_runtime_stat(2);
                    require_test(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS, "CPU graph compute");
                    require_test(std::memcmp(out->data, expected->data, ggml_nbytes(out)) == 0, "graph output bits");
                    require_test(q4kp_runtime_stat(2) - calls == uint64_t(q4kp_runtime_enabled()), "candidate graph dispatch count");
                }
                ggml_backend_buffer_free(bw);
                ggml_backend_buffer_free(br);
                ggml_free(weights);
                ggml_free(ctx);
            }
        }
    }
}
int main() {
    const bool enabled=q4kp_runtime_enabled();
    require_test(__builtin_cpu_supports("bmi2") && ggml_cpu_has_avx2() &&
        ggml_cpu_has_fma() && ggml_cpu_has_f16c(),"required test CPU features");
    ggml_init_params p={1024*1024,nullptr,true};
    auto *ctx=ggml_init(p);
    auto *w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,256,16);
    require_test(q4kp_eligible(w)==enabled,"owned eligibility");
    auto *buft=ggml_backend_cpu_repack_buffer_type();
    require_test(ggml_backend_buft_get_alloc_size(buft,w)==2304,"unchanged allocation size");
    auto *buffer=ggml_backend_buft_alloc_buffer(buft,4096);
    require_test(buffer!=nullptr,"buffer allocated");
    require_test(ggml_backend_tensor_alloc(buffer,w,ggml_backend_buffer_get_base(buffer))==GGML_STATUS_SUCCESS,"owned init");
    require_test(q4kp_has_layout(w)==enabled,"explicit owned layout");
    std::vector<uint8_t> original(ggml_nbytes(w),0);
    ggml_backend_tensor_set(w,original.data(),0,original.size());
    require_test(q4kp_runtime_stat(0)==uint64_t(enabled),"one conversion recorded");
    require_test(q4kp_runtime_stat(1)==uint64_t(enabled?192:0),"in-place byte accounting");
    require_test(q4kp_runtime_stat(3)==0,"no extra allocation");
    auto *v=ggml_view_2d(ctx,w,256,8,w->nb[1],8*w->nb[1]);
    require_test(!q4kp_eligible(v),"views never independently recoded");
    require_test(ggml_backend_view_init(v)==GGML_STATUS_SUCCESS,"aligned view initialized");
    require_test(q4kp_has_layout(v)==enabled,"view inherits parent encoding");
    require_test(v->extra==w->extra,"same trait for row-aligned view");
    auto *nested=ggml_view_2d(ctx,v,256,8,v->nb[1],0);
    require_test(ggml_backend_view_init(nested)==GGML_STATUS_SUCCESS && nested->extra==w->extra,"nested view encoding");
    auto *bad=ggml_view_2d(ctx,w,256,8,w->nb[1],w->nb[1]);
    require_test(!q4kp_compatible_view(bad),"misaligned row view incompatible");
    require_test(ggml_backend_view_init(bad)==(enabled?GGML_STATUS_FAILED:GGML_STATUS_SUCCESS),"bad recoded view refused before compute");
    ggml_tensor copy=*v;copy.ne[0]/=2;
    require_test(!q4kp_compatible_view(&copy),"partial-column view incompatible");
    copy=*v;copy.view_offs=ggml_nbytes(w);
    require_test(!q4kp_compatible_view(&copy),"out-of-range view incompatible");
    const auto saved=w->extra;
    w->extra=nullptr;
    require_test(!q4kp_has_layout(v),"layout not inferred from shape or environment");
    w->extra=saved;
    require_test(!q4kp_eligible(ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,256,15)),"nonaligned rows stay standard");
    require_test(!q4kp_eligible(ggml_new_tensor_3d(ctx,GGML_TYPE_Q4_K,256,8,2)),"MoE stays standard");
    require_test(!q4kp_eligible(ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,16)),"unrelated dtype stays standard");
    require_test(q4kp_runtime_stat(2)==0,"no inference during allocation tests");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
    graph_checks();
    std::printf("PASS: %d layout/allocation/view/graph checks, mode=%d\n",checks,int(enabled));
}
