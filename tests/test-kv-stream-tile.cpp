#include "ggml-cpp.h"
#include "testing.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-plan.h"
#include "../ggml/src/ggml-kv-stream-device.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

#ifdef KV_STREAM_TILE_NATIVE_TEST
bool kv_stream_test_is_sm61_only();
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
#endif

// Encode actual input bytes once; comparisons include stock's native KV-to-F16 conversion.
static void fill_tensor(ggml_tensor * tensor, float phase) {
    std::vector<float> values(ggml_nelements(tensor));
    for (size_t i = 0; i < values.size(); ++i) values[i] = .2f*std::sin(float(i%401)*.13f+phase);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    if (tensor->type == GGML_TYPE_F32) std::memcpy(bytes.data(),values.data(),bytes.size());
    else ggml_quantize_chunk(tensor->type,values.data(),bytes.data(),0,tensor->ne[1],tensor->ne[0],nullptr);
    ggml_backend_tensor_set(tensor,bytes.data(),0,bytes.size());
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::puts("SKIP: use --record or --compare with a snapshot directory, optionally --pascal"); return 77; }
    const bool record = !std::strcmp(argv[1],"--record"), compare = !std::strcmp(argv[1],"--compare");
    const bool pascal = argc == 4 && !std::strcmp(argv[3],"--pascal");
    if ((!record && !compare) || argc > 4) return 2;
#ifndef KV_STREAM_TILE_NATIVE_TEST
    std::puts("SKIP: CUDA native adapter is not compiled"); return 77;
#else
    if (pascal && !kv_stream_test_is_sm61_only()) { std::puts("SKIP: Pascal requires an SM61-only build"); return 77; }
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) return 77;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    const int previous = pascal ? kv_stream_test_override_cc(backend.get(),610) : 0;
    struct restore {
        ggml_backend_t backend; int previous; bool enabled;
        ~restore() { if (enabled) kv_stream_test_override_cc(backend,previous); }
    } restore_cc{backend.get(),previous,pascal};
    using status = ggml_cuda_kv_stream_plan_status;
    using style = ggml_cuda_kv_stream_execution_style;
    using query_t = status (*)(ggml_backend_t,const ggml_tensor *,const ggml_kv_stream_span_plan_view *,style,ggml_cuda_kv_stream_attention_plan &);
    auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev),"ggml_backend_cuda_kv_stream_attention_plan"));
    if (!query) return 1;
    testing t;
    size_t index = 0, admitted = 0;
    for (int dim : (pascal ? std::vector<int>{64,256} : std::vector<int>{40,72}))
        for (auto pair : {std::pair{GGML_TYPE_F16,GGML_TYPE_F16},std::pair{GGML_TYPE_BF16,GGML_TYPE_F32},
                std::pair{GGML_TYPE_F32,GGML_TYPE_F16},std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0}}) {
        if (dim%32 && (ggml_is_quantized(pair.first) || ggml_is_quantized(pair.second))) continue;
        for (int tokens : {31,256,513,8192}) for (int queries : {1,2,3,4}) for (bool masked : {false,true}) {
            const size_t id = index++;
            const std::string name = std::to_string(dim)+"/"+ggml_type_name(pair.first)+"/"+ggml_type_name(pair.second)+
                "/ctx"+std::to_string(tokens)+"/tg"+std::to_string(queries)+(masked ? "/mask" : "/none");
            t.test(name,[&](testing & t) {
                ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
                auto * q = ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,dim,queries,6);
                auto * ks = ggml_new_tensor_2d(ctx.get(),pair.first,dim*2,tokens);
                auto * vs = ggml_new_tensor_2d(ctx.get(),pair.second,dim*2,tokens);
                auto * k = ggml_view_3d(ctx.get(),ks,dim,tokens,2,ggml_row_size(pair.first,dim*2),ggml_row_size(pair.first,dim),0);
                auto * v = ggml_view_3d(ctx.get(),vs,dim,tokens,2,ggml_row_size(pair.second,dim*2),ggml_row_size(pair.second,dim),0);
                auto * mask = masked ? ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,(tokens+31)/32*32,queries) : nullptr;
                auto * output = ggml_flash_attn_ext(ctx.get(),q,k,v,mask,1.0f/std::sqrt(float(dim)),0,0);
                ggml_flash_attn_ext_set_prec(output,GGML_PREC_F32);
                auto * graph = ggml_new_graph_custom(ctx.get(),32,false);
                ggml_build_forward_expand(graph,output);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
                if (!t.assert_true(buffer != nullptr)) return;
                fill_tensor(q,.1f); fill_tensor(ks,.3f); fill_tensor(vs,.7f);
                if (mask) {
                    std::vector<ggml_fp16_t> data(ggml_nelements(mask));
                    for (int row = 0; row < queries; ++row) for (int col = 0; col < mask->ne[0]; ++col)
                        data[row*mask->ne[0]+col] = ggml_fp32_to_fp16(col <= tokens-queries+row && col%7 != 3 ? 0 : -INFINITY);
                    ggml_backend_tensor_set(mask,data.data(),0,data.size()*sizeof(ggml_fp16_t));
                }
                ggml_cuda_kv_stream_attention_plan plan;
                if (!t.assert_true(query(backend.get(),output,nullptr,style::native,plan) == status::success)) return;
                const std::string path = std::string(argv[2])+"/case-"+std::to_string(id)+".bin";
                if (plan.family != ggml_cuda_kv_stream_kernel_family::tile) {
                    if (compare) { std::ifstream previous(path,std::ios::binary); t.assert_true(!previous.good()); }
                    return;
                }
                ++admitted;
                for (int warm = 0; warm < 4; ++warm)
                    if (!t.assert_true(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS)) return;
                const auto begin = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < 16; ++repeat)
                    if (!t.assert_true(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS)) return;
                const double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/16;
                std::vector<float> actual(ggml_nelements(output));
                ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
                t.assert_true(std::all_of(actual.begin(),actual.end(),[](float x) { return std::isfinite(x); }));
                if (record) {
                    std::ofstream file(path,std::ios::binary);
                    file.write(reinterpret_cast<const char *>(actual.data()),actual.size()*sizeof(float));
                    t.assert_true(file.good());
                } else {
                    std::ifstream file(path,std::ios::binary);
                    std::vector<char> expected((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
                    if (!t.assert_equal(actual.size()*sizeof(float),expected.size())) return;
                    if (!t.assert_true(std::memcmp(actual.data(),expected.data(),expected.size()) == 0)) {
                        float max_error = 0;
                        for (size_t i = 0; i < actual.size(); ++i) {
                            float reference;
                            std::memcpy(&reference,expected.data()+i*sizeof(float),sizeof(float));
                            max_error = std::max(max_error,std::abs(reference-actual[i]));
                        }
                        std::printf("tile-error,id=%zu,max_abs=%.9g\n",id,max_error);
                    }
                }
                std::printf("tile-native,id=%zu,dim=%d,ctx=%d,tg=%d,mask=%d,us=%.3f,extra=%zu\n",
                    id,dim,tokens,queries,int(masked),us,plan.requirements.output_extra_bytes);
            });
        }
    }
    t.assert_true(admitted > 0);
    std::printf("tile-native admitted=%zu cases=%zu\n",admitted,index);
    return t.summary();
#endif
}
