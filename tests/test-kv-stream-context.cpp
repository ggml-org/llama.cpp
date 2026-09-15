#include "llama.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-io.h"
#include "testing.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using context_ptr = std::unique_ptr<llama_context,decltype(&llama_free)>;
using model_ptr = std::unique_ptr<llama_model,decltype(&llama_model_free)>;
static bool f16_control = false;
static bool resident_control = false, trace_control = false;

struct recurrent_snapshot : llama_io_write_i {
    std::vector<uint8_t> metadata;
    std::vector<std::vector<float>> tensors;
    size_t count = 0;
    void write(const void * p,size_t n) override {
        auto * bytes = static_cast<const uint8_t *>(p); metadata.insert(metadata.end(),bytes,bytes+n); count += n;
    }
    void write_tensor(ggml_tensor * tensor,size_t offset,size_t bytes) override {
        GGML_ASSERT(tensor->type == GGML_TYPE_F32 && bytes%sizeof(float) == 0);
        tensors.emplace_back(bytes/sizeof(float));
        ggml_backend_tensor_get(tensor,tensors.back().data(),offset,bytes); count += bytes;
    }
    size_t n_bytes() override { return count; }
};
struct run_result {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> continuation;
    recurrent_snapshot recurrent;
    std::vector<std::pair<std::string,std::vector<float>>> trace;
};

static run_result evaluate(testing & t,llama_model * model,const std::vector<llama_token> & prompt,
        uint32_t ubatch,size_t pool,const std::vector<llama_token> & forced) {
    run_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 1024; params.n_batch = 512; params.n_ubatch = ubatch;
    params.n_threads = params.n_threads_batch = 8; params.n_seq_max = 1;
    params.type_k = f16_control ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    params.type_v = f16_control ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; params.kv_stream_pool_bytes = pool;
    if (trace_control) {
        params.cb_eval_user_data = &result;
        params.cb_eval = [](ggml_tensor * tensor,bool ask,void * opaque) {
            auto & trace = static_cast<run_result *>(opaque)->trace;
            const std::string name = tensor->name;
            if (tensor->type != GGML_TYPE_F32 || (name != "Qcur-3" && name != "Kcur-3" && name != "Vcur-3" && name != "attn_pregate-3")) return false;
            if (std::count_if(trace.begin(),trace.end(),[&](const auto & v) { return v.first == name; }) >= 2) return false;
            if (ask) return true;
            trace.emplace_back(name,std::vector<float>(ggml_nbytes(tensor)/sizeof(float)));
            ggml_backend_tensor_get(tensor,trace.back().second.data(),0,ggml_nbytes(tensor)); return true;
        };
    }
    context_ptr context(llama_init_from_model(model,params),llama_free);
    if (!t.assert_true(bool(context))) return result;
    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
    t.assert_true(bool(hybrid->get_mem_attn()->get_kv_stream()) == (pool != 0));
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch = llama_batch_init(512,0,1);
    struct batch_guard { llama_batch & batch; ~batch_guard() { llama_batch_free(batch); } } batch_owner{batch};
    const auto submit = [&](const std::vector<llama_token> & tokens,size_t first,bool decode) {
        batch.n_tokens = int32_t(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = llama_pos(first+i);
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i+1 == tokens.size();
        }
        llama_set_kv_stream_decode(context.get(),decode);
        return llama_decode(context.get(),batch) == 0;
    };
    for (size_t first = 0; first < prompt.size(); first += 512) {
        const size_t count = std::min(size_t(512),prompt.size()-first);
        if (!t.assert_true(submit({prompt.begin()+first,prompt.begin()+first+count},first,false))) return {};
    }
    for (size_t step = 0; step < 32; ++step) {
        auto * logits = llama_get_logits_ith(context.get(),-1);
        result.logits.emplace_back(logits,logits+vocab);
        const auto token = forced.empty() ? llama_token(std::max_element(logits,logits+vocab)-logits) : forced[step];
        result.continuation.push_back(token);
        if (!t.assert_true(submit({token},prompt.size()+step,true))) return {};
    }
    llama_synchronize(context.get()); hybrid->get_mem_recr()->state_write(result.recurrent,0,0);
    if (pool) {
        auto * stream = hybrid->get_mem_attn()->get_kv_stream();
        t.assert_equal(prompt.size()+32,stream->tokens());
        t.out << "resident attention captures=" << stream->captured_layers() << '\n';
        if ((f16_control || resident_control) && !std::getenv("GGML_CUDA_DISABLE_GRAPHS")) {
            t.assert_equal(hybrid->get_mem_attn()->get_layer_ids().size(),stream->captured_layers());
        }
        t.assert_true(!llama_memory_seq_rm(llama_get_memory(context.get()),0,llama_pos(stream->tokens()-1),-1));
        std::vector<llama_token> overflow(1024-stream->tokens()+1,prompt.front());
        t.assert_true(llama_decode(context.get(),llama_batch_get_one(overflow.data(),int32_t(overflow.size()))) != 0);
        llama_memory_seq_cp(llama_get_memory(context.get()),0,1,0,-1);
        llama_memory_seq_keep(llama_get_memory(context.get()),1);
        llama_memory_seq_add(llama_get_memory(context.get()),0,0,-1,1);
        llama_memory_seq_div(llama_get_memory(context.get()),0,0,-1,2);
        recurrent_snapshot unchanged; hybrid->get_mem_recr()->state_write(unchanged,0,0);
        t.assert_true(unchanged.metadata == result.recurrent.metadata && unchanged.tensors == result.recurrent.tensors);
        const size_t state_size = llama_state_seq_get_size(context.get(),0);
        t.assert_equal(size_t(0),llama_state_seq_get_size_ext(context.get(),0,LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        std::vector<uint8_t> saved(state_size);
        t.assert_equal(state_size,llama_state_seq_get_data(context.get(),saved.data(),saved.size(),0));
        t.assert_equal(size_t(0),llama_state_seq_set_data_ext(context.get(),saved.data(),saved.size(),0,LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const auto checkpoint_tokens=stream->tokens();
        t.assert_true(submit({prompt.front()},checkpoint_tokens,true));
        std::vector<float> expected_logits(llama_get_logits_ith(context.get(),-1),llama_get_logits_ith(context.get(),-1)+vocab);
        recurrent_snapshot expected_state; hybrid->get_mem_recr()->state_write(expected_state,0,0);
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        t.assert_equal(checkpoint_tokens,stream->tokens());
        t.assert_true(submit({prompt.front()},checkpoint_tokens,true));
        auto * restored_logits=llama_get_logits_ith(context.get(),-1);
        t.assert_true(std::equal(expected_logits.begin(),expected_logits.end(),restored_logits));
        recurrent_snapshot restored_state; hybrid->get_mem_recr()->state_write(restored_state,0,0);
        t.assert_true(expected_state.metadata==restored_state.metadata && expected_state.tensors==restored_state.tensors);
        t.assert_equal(size_t(0),llama_state_seq_set_data(context.get(),saved.data(),saved.size()-1,0));
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        auto sparse=saved;
        llama_pos wrong_first=1;
        const size_t first_position=sizeof(uint32_t)+sizeof(llama_seq_id)+2*sizeof(uint32_t);
        std::memcpy(sparse.data()+first_position,&wrong_first,sizeof(wrong_first));
        t.assert_equal(size_t(0),llama_state_seq_set_data(context.get(),sparse.data(),sparse.size(),0));
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        t.assert_equal(prompt.size()+32,stream->tokens());
        struct abort_data { int calls=0; } abort;
        llama_set_abort_callback(context.get(),[](void * p) { return ++static_cast<abort_data *>(p)->calls > 0; },&abort);
        auto abort_token=prompt.front();
        t.assert_equal(int32_t(2),llama_decode(context.get(),llama_batch_get_one(&abort_token,1)));
        llama_set_abort_callback(context.get(),nullptr,nullptr);
        t.assert_true(abort.calls>0);
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        t.assert_true(submit({prompt.front()},stream->tokens(),true));
        restored_logits=llama_get_logits_ith(context.get(),-1);
        t.assert_true(std::equal(expected_logits.begin(),expected_logits.end(),restored_logits));
        if (ubatch==256) {
            const auto whole_tokens=stream->tokens();
            const size_t whole_size=llama_state_get_size(context.get());
            std::vector<uint8_t> whole(whole_size);
            t.assert_equal(whole_size,llama_state_get_data(context.get(),whole.data(),whole.size()));
            t.assert_true(submit({prompt.front()},whole_tokens,true));
            std::vector<float> whole_logits(llama_get_logits_ith(context.get(),-1),llama_get_logits_ith(context.get(),-1)+vocab);
            recurrent_snapshot whole_state; hybrid->get_mem_recr()->state_write(whole_state,0,0);
            t.assert_equal(whole_size,llama_state_set_data(context.get(),whole.data(),whole.size()));
            t.assert_equal(whole_tokens,stream->tokens());
            t.assert_true(submit({prompt.front()},whole_tokens,true));
            t.assert_true(std::equal(whole_logits.begin(),whole_logits.end(),llama_get_logits_ith(context.get(),-1)));
            recurrent_snapshot whole_restored; hybrid->get_mem_recr()->state_write(whole_restored,0,0);
            t.assert_true(whole_state.metadata==whole_restored.metadata && whole_state.tensors==whole_restored.tensors);
        }
        // A no-op adapter reset still forces scheduler/workspace reconstruction in the real context.
        t.assert_equal(int32_t(0),llama_set_adapter_cvec(context.get(),nullptr,0,0,-1,-1));
        t.assert_true(submit({prompt.front()},stream->tokens(),true));
        llama_memory_clear(llama_get_memory(context.get()),true);
        t.assert_equal(size_t(0),stream->tokens());
        t.assert_true(submit({prompt.front()},0,false));
        t.assert_equal(size_t(1),stream->tokens());
    }
    return result;
}

int main(int argc,char ** argv) {
    testing t;
    t.test("streaming_is_disabled_by_default", [&](testing & t) { t.assert_equal(size_t(0),llama_context_default_params().kv_stream_pool_bytes); });
    if (argc < 3 || std::strcmp(argv[1],"--model")) return t.summary();
    f16_control = argc > 3 && !std::strcmp(argv[3],"--f16");
    resident_control = argc > 3 && !std::strcmp(argv[3],"--resident");
    trace_control = argc > 3 && !std::strcmp(argv[argc-1],"--trace");
    ggml_backend_load_all(); llama_backend_init();
    auto mparams = llama_model_default_params(); mparams.n_gpu_layers = 999;
    model_ptr model(llama_model_load_from_file(argv[2],mparams),llama_model_free);
    if (!t.assert_true(bool(model))) return t.summary();
    const auto * vocab = llama_model_get_vocab(model.get());
    std::string text;
    for (int i = 0; i < 200; ++i) text += "The capital of France is Paris. We are testing a serial language model with a bounded key and value cache. ";
    const int size = -llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,true,false);
    std::vector<llama_token> prompt(size);
    const int tokens = llama_tokenize(vocab,text.data(),int(text.size()),prompt.data(),size,true,false);
    if (!t.assert_true(tokens >= 640)) return t.summary();
    prompt.resize(640);
    t.test("invalid_streaming_context_modes_fail_before_execution", [&](testing & t) {
        for (int mode = 0; mode < 5; ++mode) {
            auto p = llama_context_default_params(); p.kv_stream_pool_bytes = 16*1048576; p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            if (mode == 0) p.n_seq_max = 2;
            if (mode == 1) p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            if (mode == 2) p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            if (mode == 3) p.offload_kqv = false;
            if (mode == 4) p.n_rs_seq = 1;
            context_ptr context(llama_init_from_model(model.get(),p),llama_free); t.assert_true(!context);
        }
    });
    for (uint32_t ubatch : {256u,512u}) t.test("hybrid_prefill_decode_and_recurrent_state_match_ub_"+std::to_string(ubatch), [&](testing & t) {
        const auto baseline = evaluate(t,model.get(),prompt,ubatch,0,{});
        if (!t.assert_equal(size_t(32),baseline.continuation.size())) return;
        const auto streamed = evaluate(t,model.get(),prompt,ubatch,(f16_control || resident_control ? 64 : 16)*1048576,baseline.continuation);
        if (trace_control && baseline.trace.size() == streamed.trace.size()) for (size_t n = 0; n < baseline.trace.size(); ++n) {
            const auto & a = baseline.trace[n]; const auto & b = streamed.trace[n];
            if (a.first != b.first || a.second.size() != b.second.size()) continue;
            double delta = 0, energy = 0; float maximum = 0;
            for (size_t i = 0; i < a.second.size(); ++i) { const auto d = a.second[i]-b.second[i]; delta += double(d)*d; energy += double(a.second[i])*a.second[i]; maximum = std::max(maximum,std::abs(d)); }
            t.out << "trace " << a.first << " max=" << maximum << " relative=" << std::sqrt(delta/std::max(energy,1e-30)) << '\n';
        }
        if (!t.assert_equal(baseline.logits.size(),streamed.logits.size())) return;
        float maximum = 0; size_t same_top = 0;
        for (size_t step = 0; step < baseline.logits.size(); ++step) {
            const auto & a = baseline.logits[step]; const auto & b = streamed.logits[step];
            if (!t.assert_equal(a.size(),b.size())) return;
            for (size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) { t.assert_true(false); return; } maximum = std::max(maximum,std::abs(a[i]-b[i])); }
            same_top += std::max_element(a.begin(),a.end())-a.begin() == std::max_element(b.begin(),b.end())-b.begin();
        }
        t.out << "logit max error=" << maximum << ", matching top tokens=" << same_top << "/32\n";
        t.assert_true(maximum < .05f);
        t.assert_true(baseline.recurrent.metadata == streamed.recurrent.metadata);
        if (!t.assert_equal(baseline.recurrent.tensors.size(),streamed.recurrent.tensors.size())) return;
        double delta = 0, energy = 0;
        for (size_t n = 0; n < baseline.recurrent.tensors.size(); ++n) {
            const auto & a = baseline.recurrent.tensors[n]; const auto & b = streamed.recurrent.tensors[n];
            if (!t.assert_equal(a.size(),b.size())) return;
            for (size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) { t.assert_true(false); return; } delta += double(a[i]-b[i])*(a[i]-b[i]); energy += double(a[i])*a[i]; }
        }
        t.out << "recurrent relative L2=" << std::sqrt(delta/std::max(energy,1e-30)) << '\n';
        t.assert_true(delta < std::max(energy,1e-30)*1e-4);
    });
    return t.summary();
}
