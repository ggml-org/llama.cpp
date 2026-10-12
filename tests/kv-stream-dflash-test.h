#pragma once

#include "kv-stream-bit-exact-test.h"
#include "../src/llama-kv-cache-iswa.h"

// Sort candidate axes by vocabulary ID without changing or rounding any selector score.
static std::vector<uint32_t> dflash_exact_lattice(const float * data,size_t rows,size_t width,size_t k,size_t vocab) {
    if (!data || rows < 2 || rows > 8 || !k || k > 16 || width < k+k*k || width > 5120)
        throw std::runtime_error("invalid exact lattice geometry");
    std::vector<uint32_t> result{uint32_t(rows),uint32_t(width),uint32_t(k)};
    for (size_t i=0;i<width;++i) if (data[i] != 0) throw std::runtime_error("nonzero selector anchor");
    std::vector<size_t> previous;
    for (size_t row=1;row<rows;++row) {
        const auto * values=data+row*width;
        std::vector<size_t> order;
        for (size_t i=0;i<k;++i) {
            if (!std::isfinite(values[i]) || values[i] < 0 || values[i] >= vocab || float(uint32_t(values[i])) != values[i])
                throw std::runtime_error("invalid selector candidate");
            order.push_back(i);
        }
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b) { return values[a] < values[b]; });
        for (size_t i=1;i<k;++i) if (values[order[i]] == values[order[i-1]]) throw std::runtime_error("duplicate selector ID");
        for (size_t i=k+k*k;i<width;++i) if (values[i] != 0) throw std::runtime_error("nonzero selector padding");
        const auto & predecessor=row == 1 ? std::vector<size_t>{0} : previous;
        for (auto p : predecessor) for (auto c : order) {
            if (row > 1) result.push_back(uint32_t(data[(row-1)*width+p]));
            result.push_back(uint32_t(values[c]));
            const float score=values[k+p*k+c];
            if (!std::isfinite(score)) throw std::runtime_error("nonfinite selector score");
            uint32_t bits; std::memcpy(&bits,&score,sizeof(bits)); result.push_back(bits);
        }
        if (row == 1) for (size_t p=1;p<k;++p) if (std::memcmp(values+k,values+k+p*k,k*sizeof(float)))
            throw std::runtime_error("anchor selector scores are not repeated");
        previous=std::move(order);
    }
    return result;
}

// Match the same block execution and force decisions independently of sampler acceptance.
static void test_kv_stream_dflash_exact(testing & t,llama_model * model,const char * sidecar) {
    const auto vocabulary=llama_model_get_vocab(model);
    const auto vocab=llama_vocab_n_tokens(vocabulary);
    const auto seed=common_tokenize(vocabulary,"A controlled comparison of windowed drafting and target recovery. ",true);
    for (const auto profile : {std::array<size_t,3>{1,2304,8192},{3,2304,8192},{7,2304,8192},{7,16000,16384}}) {
        for (bool quantized : {false,true}) {
            kv_stream_bit_oracle oracle;
            for (bool streaming : {false,true}) {
                if (streaming) oracle.compare();
                auto cp=llama_context_default_params();
                cp.n_ctx=uint32_t(profile[2]); cp.n_batch=cp.n_ubatch=256;
                cp.n_seq_max=1; cp.n_rs_seq=uint32_t(profile[0]); cp.n_outputs_max=uint32_t(profile[0]+1);
                cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
                cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
                if (streaming) cp.shared_device_memory_bytes=512u*1024u*1024u;
                context_ptr target(llama_init_from_model(model,cp),llama_free);
                if (!t.assert_true(bool(target))) return;
                common_params base;
                base.n_ctx=cp.n_ctx; base.n_batch=base.n_ubatch=256; base.n_parallel=1; base.n_gpu_layers=999;
                base.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
                base.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
                base.speculative.draft.n_max=int(profile[0]); base.speculative.draft.n_min=0; base.speculative.draft.p_min=0;
                base.speculative.draft.n_gpu_layers=999; base.speculative.draft.mparams.path=sidecar;
                base.speculative.draft.cache_type_k=quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
                base.speculative.draft.cache_type_v=quantized ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
                auto params=common_base_params_to_speculative(base);
                auto draft=common_speculative_init_from_params(params,model,target.get());
                if (!t.assert_true(draft && draft->context())) return;
                params.speculative.draft.ctx_tgt=target.get(); params.speculative.draft.ctx_dft=draft->context();
                common_speculative_ptr spec(common_speculative_init(params.speculative,1));
                if (!t.assert_true(bool(spec))) return;
                auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
                auto * window=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(draft->context()));
                if (!t.assert_true(hybrid && window)) return;
                auto * stream=hybrid->get_mem_attn()->get_kv_stream();
                bool observed_streaming=false;
                llama_tokens history;
                size_t position=0;
                const auto submit=[&](const std::string & name,const llama_tokens & tokens,bool decode,bool record) {
                    common_batch batch(target.get());
                    for (size_t i=0;i<tokens.size();++i) batch.add(tokens[i],llama_pos(position+i),0,decode || i+1 == tokens.size());
                    llama_set_kv_stream_decode(target.get(),decode);
                    if (llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()) != 0)
                        throw std::runtime_error("target failed: "+name);
                    if (record) {
                        for (size_t i=0;i<tokens.size();++i) oracle.bytes(name+"/logits/"+std::to_string(i),
                            llama_get_logits_ith(target.get(),int(i)),size_t(vocab)*sizeof(float));
                        const auto * ids=llama_model_target_layer_ids(draft->model());
                        for (uint32_t i=0;i<llama_model_target_layer_ids_n(draft->model());++i)
                            oracle.bytes(name+"/features/"+std::to_string(i),llama_get_embeddings_layer_inp(target.get(),uint32_t(ids[i])),
                                tokens.size()*size_t(llama_model_n_embd(model))*sizeof(float));
                    }
                    if (!common_speculative_process(spec.get(),batch)) throw std::runtime_error("injection failed: "+name);
                    llama_synchronize(target.get()); llama_synchronize(draft->context());
                    position+=tokens.size();
                    if (stream) {
                        llama_kv_stream_runtime_diagnostics status;
                        if (!stream->runtime_diagnostics(status)) throw std::runtime_error("missing target diagnostics");
                        observed_streaming|=decode && status.streaming_active;
                    }
                };
                while (position < profile[1]) {
                    llama_tokens part(std::min(size_t(256),profile[1]-position));
                    for (size_t i=0;i<part.size();++i) part[i]=seed[(position+i)%seed.size()];
                    submit("prefill",part,false,false); history.insert(history.end(),part.begin(),part.end());
                }
                llama_token anchor=seed[0];
                const std::array<size_t,8> lengths{7,4,6,5,1,3,2,0};
                for (size_t round=0;round<lengths.size();++round) {
                    const auto name="round/"+std::to_string(round);
                    const auto first=position;
                    llama_tokens predicted;
                    auto & dp=common_speculative_get_draft_params(spec.get(),0);
                    dp.drafting=true; dp.n_max=int(profile[0]); dp.pos0=int(position); dp.id_last=anchor;
                    dp.prompt=&history; dp.result=&predicted;
                    common_speculative_draft(spec.get()); llama_synchronize(draft->context());
                    if (!t.assert_equal(profile[0],predicted.size())) return;
                    oracle.bytes(name+"/proposals",predicted.data(),predicted.size()*sizeof(llama_token));
                    const auto lattice=dflash_exact_lattice(llama_get_embeddings_nextn(draft->context()),profile[0]+1,
                        size_t(llama_model_n_embd(draft->model())),size_t(llama_model_dflash_selector_top_k(draft->model())),size_t(vocab));
                    oracle.bytes(name+"/lattice",lattice.data(),lattice.size()*sizeof(uint32_t));
                    predicted.resize(std::min(profile[0],lengths[round]));
                    if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(first),-1))) return;
                    if (predicted.size() > 1) predicted.back()=(predicted.back()+1)%vocab;
                    llama_tokens verified{anchor}; verified.insert(verified.end(),predicted.begin(),predicted.end());
                    submit(name,verified,true,true);
                    const size_t accepted=round%3 == 0 ? 0 : round%3 == 1 ? predicted.size()/2 : predicted.size();
                    const size_t keep=first+accepted+1;
                    if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(keep),-1) &&
                            llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(keep),-1))) return;
                    position=keep;
                    kv_stream_exact_kv(oracle,name+"/target_KV",hybrid->get_mem_attn(),first,accepted+1);
                    kv_stream_exact_kv(oracle,name+"/draft_KV",window->get_swa(),first,accepted+1);
                    kv_stream_exact_state state;
                    hybrid->get_mem_recr()->state_write(state,0,0); state.compare_with(oracle,name+"/recurrent");
                    history.insert(history.end(),verified.begin(),verified.begin()+accepted+1);
                    anchor=seed[(round+3)%seed.size()];
                    if (accepted < predicted.size() && anchor == predicted[accepted]) anchor=(anchor+1)%vocab;
                }
                if (streaming) {
                    if (profile[1] == 16000 && !t.assert_true(observed_streaming)) return;
                    oracle.finish();
                }
                t.out << "DFlash exact D=" << profile[0] << " prefix=" << profile[1] << " quant=" << quantized
                    << " streamed=" << streaming << " ring=" << observed_streaming << '\n';
            }
        }
    }
}
