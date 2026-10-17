#pragma once

#include "kv-stream-bit-exact-test.h"
#include "../src/llama-kv-cache-iswa.h"

static std::vector<std::array<size_t,3>> dflash_exact_profiles(
        bool lifecycle, bool context_end, bool sustained, bool short_profile, ggml_type key, ggml_type value) {
    if (short_profile) {
        if (lifecycle || context_end || sustained) throw std::runtime_error("short DFlash profile cannot replace lifecycle/context-end coverage");
        const size_t bytes = ggml_row_size(key,1024)+ggml_row_size(value,1024);
        const size_t capacity = bytes > 4096 ? 4096 : 8192;
        return {{1,2047,capacity},{3,2047,capacity},{7,2047,capacity},{7,capacity-192,capacity}};
    }
    return sustained ? std::vector<std::array<size_t,3>>{{7,8176,8448}} : context_end ?
        std::vector<std::array<size_t,3>>{{1,8176,8192},{3,8176,8192},{7,8176,8192}} : lifecycle ?
        std::vector<std::array<size_t,3>>{{7,2047,16384},{7,2048,16384},{7,2049,16384},{7,2304,16384},{7,16000,16384}} :
        std::vector<std::array<size_t,3>>{{1,2304,8192},{3,2304,8192},{7,2304,8192},{7,16000,16384}};
}

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
static void test_kv_stream_dflash_exact(testing & t,llama_model * model,const char * sidecar,bool lifecycle=false,bool context_end=false,bool sustained_rejection=false,
        ggml_type key=GGML_TYPE_Q8_0,ggml_type value=GGML_TYPE_Q4_0,bool matching_cache=false,bool short_profile=false) {
    const auto vocabulary=llama_model_get_vocab(model);
    const auto vocab=llama_vocab_n_tokens(vocabulary);
    const auto seed=common_tokenize(vocabulary,"A controlled comparison of windowed drafting and target recovery. ",true);
    const auto profiles = dflash_exact_profiles(lifecycle,context_end,sustained_rejection,short_profile,key,value);
    for (const auto profile : profiles) {
        for (bool quantized : {false,true}) {
            if (matching_cache && !quantized) continue;
            if (lifecycle && !quantized && profile[1] != 2049) continue;
            kv_stream_bit_oracle oracle;
            for (bool streaming : {false,true}) {
                if (streaming) oracle.compare();
                auto cp=llama_context_default_params();
                cp.n_ctx=uint32_t(profile[2]); cp.n_batch=cp.n_ubatch=256;
                cp.n_seq_max=1; cp.n_rs_seq=uint32_t(profile[0]); cp.n_outputs_max=uint32_t(profile[0]+1);
                cp.type_k=key; cp.type_v=value;
                cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
                if (streaming) {
                    const bool wide = ggml_row_size(key,1024)+ggml_row_size(value,1024) > 4096;
                    cp.shared_device_memory_bytes=(short_profile ? (wide ? 384u : 228u) : context_end ? 224u : matching_cache ? 384u : 512u)*1024u*1024u;
                }
                context_ptr target(llama_init_from_model(model,cp),llama_free);
                if (!t.assert_true(bool(target))) return;
                common_params base;
                base.n_ctx=cp.n_ctx; base.n_batch=base.n_ubatch=256; base.n_parallel=1; base.n_gpu_layers=999;
                base.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
                base.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
                base.speculative.draft.n_max=int(profile[0]); base.speculative.draft.n_min=0; base.speculative.draft.p_min=0;
                base.speculative.draft.n_gpu_layers=999; base.speculative.draft.mparams.path=sidecar;
                base.speculative.draft.cache_type_k=matching_cache ? key : quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
                base.speculative.draft.cache_type_v=matching_cache ? value : quantized ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
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
                if (context_end) {
                    while (position <= profile[2]) {
                        const size_t remaining=profile[2]-position;
                        const size_t expected=remaining ? std::min(profile[0],remaining-1) : 0;
                        const auto first=position;
                        llama_tokens proposed;
                        auto & dp=common_speculative_get_draft_params(spec.get(),0);
                        dp.drafting=true; dp.n_max=-1; dp.pos0=int(position); dp.id_last=anchor;
                        dp.prompt=&history; dp.result=&proposed;
                        common_speculative_draft(spec.get());
                        llama_synchronize(draft->context());
                        if (!t.assert_equal(expected,proposed.size())) return;
                        const auto name="context_end/"+std::to_string(remaining);
                        if (!proposed.empty()) oracle.bytes(name+"/proposals",proposed.data(),proposed.size()*sizeof(llama_token));
                        if (!remaining) break;
                        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(first),-1))) return;
                        llama_tokens verified{anchor};
                        verified.insert(verified.end(),proposed.begin(),proposed.end());
                        submit(name,verified,true,true);
                        // Force rejection while approaching the boundary one accepted row at a time.
                        position=first+1;
                        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(position),-1) &&
                                llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(position),-1))) return;
                        history.push_back(anchor);
                        anchor=seed[position%seed.size()];
                    }
                    kv_stream_exact_kv(oracle,"context_end/target_KV",hybrid->get_mem_attn(),position-8,8);
                    kv_stream_exact_kv(oracle,"context_end/draft_KV",window->get_swa(),position-8,8);
                    kv_stream_exact_state state;
                    hybrid->get_mem_recr()->state_write(state,0,0);
                    state.compare_with(oracle,"context_end/recurrent");
                    if (streaming) {
                        if (!t.assert_true(observed_streaming)) return;
                        oracle.finish();
                    }
                    t.out << "DFlash context end D=" << profile[0] << " quant=" << quantized
                          << " streamed=" << streaming << " forced_rejections=" << profile[2]-profile[1] << '\n';
                    continue;
                }
                if (lifecycle) {
                    const size_t saved_position=position;
                    const auto saved_history=history;
                    const auto save=[&](llama_context * ctx) {
                        std::vector<uint8_t> data(llama_state_seq_get_size(ctx,0));
                        if (data.empty() || llama_state_seq_get_data(ctx,data.data(),data.size(),0) != data.size())
                            throw std::runtime_error("DFlash lifecycle state save failed");
                        return data;
                    };
                    const auto target_state=save(target.get());
                    const auto draft_state=save(draft->context());
                    const auto restore=[&] {
                        if (llama_state_seq_set_data(target.get(),target_state.data(),target_state.size(),0) != target_state.size() ||
                                llama_state_seq_set_data(draft->context(),draft_state.data(),draft_state.size(),0) != draft_state.size())
                            throw std::runtime_error("DFlash lifecycle state restore failed");
                        position=saved_position; history=saved_history;
                        common_speculative_begin(spec.get(),0,history);
                    };
                    const auto predict=[&] {
                        llama_tokens result;
                        auto & dp=common_speculative_get_draft_params(spec.get(),0);
                        dp.drafting=true; dp.n_max=int(profile[0]); dp.pos0=int(position); dp.id_last=anchor;
                        dp.prompt=&history; dp.result=&result;
                        common_speculative_draft(spec.get()); llama_synchronize(draft->context());
                        return result;
                    };
                    const auto probe=[&](kv_stream_bit_oracle & replay) {
                        const auto first=position;
                        const auto proposed=predict();
                        if (proposed.size() != profile[0]) throw std::runtime_error("DFlash lifecycle prediction failed");
                        replay.bytes("proposals",proposed.data(),proposed.size()*sizeof(llama_token));
                        const auto lattice=dflash_exact_lattice(llama_get_embeddings_nextn(draft->context()),profile[0]+1,
                            size_t(llama_model_n_embd(draft->model())),size_t(llama_model_dflash_selector_top_k(draft->model())),size_t(vocab));
                        replay.bytes("selector",lattice.data(),lattice.size()*sizeof(uint32_t));
                        if (!llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(first),-1))
                            throw std::runtime_error("DFlash lifecycle temporary suffix retirement failed");
                        llama_tokens verified{anchor}; verified.insert(verified.end(),proposed.begin(),proposed.end());
                        submit("lifecycle",verified,true,false);
                        for (size_t i=0;i<verified.size();++i)
                            replay.bytes("logits/"+std::to_string(i),llama_get_logits_ith(target.get(),int(i)),size_t(vocab)*sizeof(float));
                        const auto * ids=llama_model_target_layer_ids(draft->model());
                        for (uint32_t i=0;i<llama_model_target_layer_ids_n(draft->model());++i)
                            replay.bytes("features/"+std::to_string(i),llama_get_embeddings_layer_inp(target.get(),uint32_t(ids[i])),
                                verified.size()*size_t(llama_model_n_embd(model))*sizeof(float));
                        const auto keep=first+3;
                        if (!llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(keep),-1) ||
                                !llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(keep),-1))
                            throw std::runtime_error("DFlash lifecycle rejection failed");
                        position=keep;
                        kv_stream_exact_kv(replay,"target_KV",hybrid->get_mem_attn(),first,3);
                        kv_stream_exact_kv(replay,"draft_KV",window->get_swa(),first,3);
                        kv_stream_exact_state state;
                        hybrid->get_mem_recr()->state_write(state,0,0); state.compare_with(replay,"recurrent");
                    };
                    kv_stream_bit_oracle retained;
                    const auto window_tokens=std::min(size_t(2048),saved_position);
                    const auto target_tokens=std::min(size_t(256),saved_position);
                    const auto retained_rows=[&] {
                        kv_stream_exact_kv(retained,"retained_target",hybrid->get_mem_attn(),saved_position-target_tokens,target_tokens);
                        kv_stream_exact_kv(retained,"retained_draft",window->get_swa(),saved_position-window_tokens,window_tokens);
                    };
                    retained_rows();
                    const auto old_cells=window->get_swa()->get_cells(0).seq_pos_get(0).size();
                    // Stock sequence snapshots compact masked SWA rows; both probes must use that same layout.
                    restore();
                    retained.compare(); retained_rows(); retained.finish();
                    t.out << "DFlash checkpoint cells " << old_cells << " -> "
                        << window->get_swa()->get_cells(0).seq_pos_get(0).size() << '\n';
                    kv_stream_bit_oracle replay;
                    probe(replay);
                    restore();
                    // Exercise another prompt and window wrap before restoring the original branch.
                    llama_memory_clear(llama_get_memory(target.get()),true);
                    llama_memory_clear(llama_get_memory(draft->context()),true);
                    position=0; history.clear();
                    while (position < 15872) {
                        llama_tokens part(std::min(size_t(256),size_t(15872)-position));
                        for (size_t i=0;i<part.size();++i) part[i]=seed[(position+i+7)%seed.size()];
                        submit("other_prompt",part,false,false); history.insert(history.end(),part.begin(),part.end());
                    }
                    restore();
                    if (!t.assert_equal(size_t(0),llama_state_seq_set_data(target.get(),target_state.data(),target_state.size()-1,0))) return;
                    restore();
                    if (!t.assert_equal(size_t(0),llama_state_seq_set_data(draft->context(),draft_state.data(),draft_state.size()-1,0))) return;
                    restore();
                    for (bool cancel_draft : {false,true}) {
                        auto * ctx=cancel_draft ? draft->context() : target.get();
                        int calls=0;
                        llama_set_abort_callback(ctx,[](void * data) { ++*static_cast<int *>(data); return true; },&calls);
                        if (cancel_draft) {
                            const auto proposed=predict();
                            llama_set_abort_callback(ctx,nullptr,nullptr);
                            if (!t.assert_true(proposed.empty() && calls > 0)) return;
                        } else {
                            common_batch batch(ctx);
                            batch.add(anchor,llama_pos(position),0,true);
                            llama_set_kv_stream_decode(ctx,true);
                            const auto status=llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch.get());
                            llama_set_abort_callback(ctx,nullptr,nullptr);
                            if (!t.assert_true(status == 2 && calls > 0)) return;
                        }
                        restore();
                    }
                    replay.compare(); probe(replay); replay.finish();
                    restore();
                    t.out << "DFlash lifecycle restored prefix=" << saved_position << " quant=" << quantized
                        << " streamed=" << streaming << '\n';
                }
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
                    if ((profile[1] == 16000 || (short_profile && profile == profiles.back())) && !t.assert_true(observed_streaming)) return;
                    oracle.finish();
                }
                t.out << "DFlash exact D=" << profile[0] << " prefix=" << profile[1] << " quant=" << quantized
                    << " streamed=" << streaming << " ring=" << observed_streaming << '\n';
            }
        }
    }
}
