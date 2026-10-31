#pragma once

#include "kv-stream-bit-exact-test.h"
#include "../src/llama-kv-cache-iswa.h"

#include <set>

static void test_dflash_prefetch_intent(testing & t, llama_model * model, const char * sidecar) {
    for (bool custom : {true, false}) {
        auto cp = llama_context_default_params();
        cp.n_ctx = custom ? 4096 : 8192; cp.n_batch = cp.n_ubatch = 256;
        cp.n_seq_max = 1; cp.n_rs_seq = 7;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 8;
        cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q4_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes = size_t(custom ? 1024 : 228)*1024*1024;
        context_ptr target(llama_init_from_model(model, cp), llama_free);
        if (!t.assert_true(bool(target))) return;
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
        auto * source = hybrid ? hybrid->get_mem_attn()->get_kv_stream() : nullptr;
        if (!t.assert_true(source != nullptr)) return;
        const auto initial_append = source->next_consumer(llama_kv_stream_access_role::append);
        const auto initial_read = source->next_consumer(llama_kv_stream_access_role::read);
        llama_kv_stream_next_consumer previous;
        previous.cache_id = custom ? source->host()->cache_id() : 0; previous.future_tokens = custom ? 3 : 0;
        if (!t.assert_true(source->schedule_next_consumer(llama_kv_stream_access_role::append, previous))) return;
        auto previous_read = previous; previous_read.cache_id = source->host()->cache_id(); previous_read.future_tokens = 1;
        if (!t.assert_true(source->schedule_next_consumer(llama_kv_stream_access_role::read, previous_read))) return;
        common_params base;
        base.n_ctx = cp.n_ctx; base.n_batch = base.n_ubatch = 256;
        base.n_parallel = 1; base.n_gpu_layers = 999;
        base.shared_device_memory_bytes = cp.shared_device_memory_bytes;
        base.fit_params = false;
        base.cache_type_k = cp.type_k; base.cache_type_v = cp.type_v;
        base.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        base.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        if (custom) base.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        base.speculative.draft.n_max = 3; base.speculative.draft.n_min = 0; base.speculative.draft.p_min = 0;
        base.speculative.draft.n_gpu_layers = 999; base.speculative.draft.mparams.path = sidecar;
        base.speculative.ngram_map_k4v.size_m = 7;
        auto params = common_base_params_to_speculative(base);
        params.n_ubatch = 8;
        auto owned = common_speculative_init_from_params(params, model, target.get());
        if (!t.assert_true(owned && owned->context())) return;
        const auto flexible = [&] {
            const auto next = source->next_consumer(llama_kv_stream_access_role::append);
            return next.access == llama_kv_stream_access_role::append &&
                next.cache_id == source->host()->cache_id() && next.future_tokens == 0;
        };
        if (!t.assert_true("serial block drafter owns flexible next-target prefetch", flexible())) return;
        auto * parent = target->get_compute_memory()->shared_parent();
        params.speculative.draft.ctx_tgt = target.get();
        params.speculative.draft.ctx_dft = owned->context();
        common_speculative_ptr first(common_speculative_init(params.speculative, 1));
        common_speculative_ptr second(common_speculative_init(params.speculative, 1));
        if (!t.assert_true(first && second && flexible())) return;
        first.reset();
        t.assert_true("one probe driver cannot remove the context owner's prefetch intent", flexible());
        size_t position = 0;
        if (!custom) {
            const auto submit = [&](size_t rows, bool decode) {
                common_batch batch(target.get());
                for (size_t i = 0; i < rows; ++i) batch.add(1000+llama_token((position+i)%17),
                    llama_pos(position+i), 0, decode || i+1 == rows);
                llama_set_kv_stream_decode(target.get(), decode);
                if (!t.assert_equal(0, llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) ||
                        !t.assert_true(common_speculative_process(second.get(), batch))) return false;
                position += rows;
                return true;
            };
            while (position < 6143) if (!submit(std::min(size_t(256),6143-position), false)) return;
            if (!submit(1, true)) return;
            t.assert_true("lookahead does not cross a physical KV layout boundary", !source->prefetch_primed());
            if (!submit(1, true)) return;
            if (!t.assert_true("the committed ring source is primed for a variable verification width", source->prefetch_primed())) return;
            if (!submit(8, true)) return;
            llama_synchronize(target.get()); llama_synchronize(owned->context());
            llama_kv_stream_runtime_diagnostics diagnostics;
            if (!t.assert_true(source->runtime_diagnostics(diagnostics) && diagnostics.streaming_active)) return;
            t.assert_true("actual TG8 adopts the existing flexible ring prime",
                source->sequence_stats(llama_kv_stream_access_role::append).adopted);
            t.assert_true("child teardown is exercised with a retained idle prime", source->prefetch_primed());
        }
        second.reset();
        t.assert_true("prefetch intent outlives all drivers sharing its owned child", flexible());
        owned.reset();
        const auto restored = source->next_consumer(llama_kv_stream_access_role::append);
        const auto restored_read = source->next_consumer(llama_kv_stream_access_role::read);
        t.assert_equal(previous.cache_id, restored.cache_id);
        t.assert_equal(previous.future_tokens, restored.future_tokens);
        t.assert_equal(previous_read.cache_id, restored_read.cache_id);
        t.assert_equal(previous_read.future_tokens, restored_read.future_tokens);
        t.assert_true(!target->get_compute_memory()->has_speculative_consumer());
        t.assert_true(parent == target->get_compute_memory()->shared_parent());
        params.speculative.draft.ctx_dft = nullptr;
        auto invalid = params;
        invalid.speculative.draft.n_max = 8;
        bool rejected = false;
        try { auto failure = common_speculative_init_from_params(invalid, model, target.get()); }
        catch (const std::invalid_argument &) { rejected = true; }
        t.assert_true("failed admission preserves the previous prefetch owner", rejected &&
            source->next_consumer(llama_kv_stream_access_role::append).future_tokens == previous.future_tokens &&
            source->next_consumer(llama_kv_stream_access_role::read).future_tokens == previous_read.future_tokens &&
            !target->get_compute_memory()->has_speculative_consumer());
        t.assert_true(source->schedule_next_consumer(llama_kv_stream_access_role::append, initial_append));
        t.assert_true(source->schedule_next_consumer(llama_kv_stream_access_role::read, initial_read));
        common_batch resumed(target.get()); resumed.add(1000, llama_pos(position), 0, true);
        llama_set_kv_stream_decode(target.get(), true);
        if (!t.assert_equal(0, llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, resumed.get()))) return;
        llama_synchronize(target.get());
        const auto * logits = llama_get_logits_ith(target.get(), -1);
        const auto vocab = size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
        t.assert_true(logits && std::all_of(logits, logits+vocab, [](float x) { return std::isfinite(x); }));
    }
}

static llama_context_params dflash_independent_child_params(llama_context_params target, llama_context * parent) {
    target.shared_device_memory_bytes = 0;
    target.n_rs_seq = 0;
    target.swa_full = false;
    target.n_outputs_max = target.n_outputs_max_per_seq = 4;
    target.ctx_other = parent;
    return target;
}

static void test_dflash_independent_quota(testing & t, llama_model * model, const char * sidecar) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model_ptr draft_model(llama_model_load_from_file(sidecar, mp), llama_model_free);
    if (!t.assert_true(bool(draft_model))) return;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = cp.n_ubatch = 256;
    cp.n_seq_max = 1; cp.n_rs_seq = 7; cp.n_outputs_max = cp.n_outputs_max_per_seq = 8;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.shared_device_memory_bytes = size_t(512)*1024*1024;
    const auto make_draft_params = [&](llama_context * target) {
        return dflash_independent_child_params(cp,target);
    };
    const auto exercise = [&](llama_context * target, llama_context * child) {
        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        params.draft.n_max = 3; params.draft.n_min = 0; params.draft.p_min = 0;
        params.draft.ctx_tgt = target; params.draft.ctx_dft = child;
        common_speculative_ptr spec(common_speculative_init(params, 1));
        if (!t.assert_true(bool(spec))) return false;
        const auto seed = common_tokenize(target, "A bounded serial resource and rollback control. ", true);
        if (!t.assert_true(!seed.empty())) return false;
        llama_tokens history;
        const auto submit = [&](size_t first, size_t rows, bool decode) {
            common_batch batch(target);
            for (size_t i=0;i<rows;++i) batch.add(seed[(first+i)%seed.size()],llama_pos(first+i),0,decode || i+1 == rows);
            llama_set_kv_stream_decode(target,decode);
            if (!t.assert_equal(0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return false;
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return false;
            llama_synchronize(target); llama_synchronize(child);
            for (size_t i=0;i<rows;++i) history.push_back(seed[(first+i)%seed.size()]);
            return true;
        };
        if (!submit(0,16,false)) return false;
        llama_tokens proposed;
        auto & request = common_speculative_get_draft_params(spec.get(),0);
        request.drafting=true; request.n_max=3; request.pos0=16;
        request.id_last=seed[16%seed.size()]; request.prompt=&history; request.result=&proposed;
        common_speculative_draft(spec.get()); llama_synchronize(child);
        if (!t.assert_equal(size_t(3),proposed.size())) return false;
        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(child),0,16,-1))) return false;
        if (!submit(16,1,false) || !submit(17,8,true)) return false;
        const auto vocab = size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
        for (int row=0;row<8;++row) {
            const auto * logits = llama_get_logits_ith(target,row);
            if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }))) return false;
        }
        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target),0,18,-1) &&
                llama_memory_seq_rm(llama_get_memory(child),0,18,-1))) return false;
        history.resize(18);
        return submit(18,1,false);
    };
    size_t exact = 0;
    {
        context_ptr target(new llama_context(*model,cp),llama_free);
        const auto features = target->get_cparams().embeddings_layer_inp;
        const auto revision = target->get_compute_memory()->graph_binding_revision();
        for (uint32_t rows : {1u, 9u}) {
            bool rejected = false;
            try {
                context_ptr invalid(new llama_context(*draft_model,make_draft_params(target.get()),rows),llama_free);
            } catch (const std::runtime_error & error) {
                rejected = std::string(error.what()).find("incompatible serial DFlash2 context") != std::string::npos;
            }
            t.assert_true("invalid prediction bounds reject before target mutation", rejected);
            t.assert_equal(revision, target->get_compute_memory()->graph_binding_revision());
            t.assert_true(features == target->get_cparams().embeddings_layer_inp);
        }
        context_ptr child(new llama_context(*draft_model,make_draft_params(target.get()),4),llama_free);
        if (!exercise(target.get(),child.get())) return;
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
        auto * stream = hybrid ? hybrid->get_mem_attn()->get_kv_stream() : nullptr;
        if (!t.assert_true(stream != nullptr)) return;
        const auto & config = stream->host()->config();
        llama_kv_stream_policy_config policy;
        policy.shape=config.shape; policy.capabilities=config.capabilities; policy.layers=config.layers;
        llama_context_memory_diagnostics usage;
        size_t minimum = 0;
        if (!t.assert_true(target->get_compute_memory()->diagnostics(usage) && usage.unused_bytes == 0 &&
                usage.phase == llama_memory_text_phase::prefill &&
                llama_kv_stream_policy_minimum_pool_bytes(policy,minimum).status == llama_kv_stream_policy_status::success &&
                minimum <= usage.kv_pool_bytes && usage.kv_pool_bytes <= usage.parent_bytes)) return;
        exact=usage.parent_bytes-usage.kv_pool_bytes+minimum;
        std::fprintf(stderr,"DFlash independent quota: target_rows=8 prediction_rows=4 required=%zu minimum_pool=%zu\n",exact,minimum);
    }
    if (!t.assert_true(exact > 1)) return;
    for (bool sufficient : {false,true}) {
        cp.shared_device_memory_bytes=exact-(sufficient ? 0 : 1);
        context_ptr target(nullptr,llama_free),child(nullptr,llama_free);
        std::string error;
        try {
            target.reset(new llama_context(*model,cp));
            child.reset(new llama_context(*draft_model,make_draft_params(target.get()),4));
        } catch (const std::runtime_error & failure) {
            error=failure.what();
        }
        if (!sufficient) {
            const auto fields=error.find("required=");
            size_t required=0,available=0,additional=0;
            if (!t.assert_true("one-byte-short DFlash joint quota rejects with exact cause and no private fallback",
                    error.find("insufficient arena quota") != std::string::npos && fields != std::string::npos &&
                    std::sscanf(error.c_str()+fields,"required=%zu, available=%zu, additional=%zu",&required,&available,&additional)==3)) {
                t.out << "DFlash short quota error: " << error << '\n'; return;
            }
            t.assert_equal(exact,required); t.assert_equal(exact-1,available); t.assert_equal(size_t(1),additional);
            continue;
        }
        if (!t.assert_true("exact DFlash joint quota admits both contexts",error.empty() && target && child)) {
            t.out << "DFlash exact quota error: " << error << '\n'; return;
        }
        t.assert_equal(exact,target->get_compute_memory()->shared_parent_capacity());
        t.assert_true(target->get_compute_memory()->shared_parent() == child->get_compute_memory()->shared_parent());
        t.assert_equal(uint32_t(4),child->get_cparams().serial_block_rows);
        t.assert_equal(uint32_t(8),target->get_cparams().kv_stream_decode_rows());
        if (!exercise(target.get(),child.get())) return;
        auto * parent=target->get_compute_memory()->shared_parent();
        child.reset();
        t.assert_true(!target->get_compute_memory()->has_speculative_consumer());
        t.assert_true(parent==target->get_compute_memory()->shared_parent());
        const auto next=llama_memory_seq_pos_max(llama_get_memory(target.get()),0)+1;
        common_batch resumed(target.get());
        resumed.add(1000,next,0,true);
        llama_set_kv_stream_decode(target.get(),true);
        if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,resumed.get()))) return;
        t.assert_equal(next,llama_memory_seq_pos_max(llama_get_memory(target.get()),0));
    }
}

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
        ggml_type key=GGML_TYPE_Q8_0,ggml_type value=GGML_TYPE_Q4_0,bool matching_cache=false,bool short_profile=false,bool independent_rows=false,
        bool small_injection_batch=false,uint32_t only_depth=0) {
    const auto vocabulary=llama_model_get_vocab(model);
    const auto vocab=llama_vocab_n_tokens(vocabulary);
    const auto seed=common_tokenize(vocabulary,"A controlled comparison of windowed drafting and target recovery. ",true);
    const auto profiles = dflash_exact_profiles(lifecycle,context_end,sustained_rejection,short_profile,key,value);
    for (const auto profile : profiles) {
        if (only_depth && profile[0] != only_depth) continue;
        for (bool quantized : {false,true}) {
            if (matching_cache && !quantized) continue;
            if (lifecycle && !quantized && profile[1] != 2049) continue;
            kv_stream_bit_oracle oracle;
            for (bool streaming : {false,true}) {
                if (streaming) oracle.compare();
                auto cp=llama_context_default_params();
                cp.n_ctx=uint32_t(profile[2]); cp.n_batch=cp.n_ubatch=256;
                cp.n_seq_max=1; cp.n_rs_seq=independent_rows ? 7u : uint32_t(profile[0]); cp.n_outputs_max=cp.n_rs_seq+1;
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
                if (small_injection_batch) params.n_ubatch=8;
                if (independent_rows) std::fprintf(stderr,
                    "DFlash independent profile: prediction=%zu verification=%u context=%u prefix=%zu target_ubatch=%u draft_ubatch=%d arena=%zu\n",
                    profile[0],cp.n_rs_seq+1,cp.n_ctx,profile[1],cp.n_ubatch,params.n_ubatch,cp.shared_device_memory_bytes);
                auto draft=common_speculative_init_from_params(params,model,target.get());
                if (!t.assert_true(draft && draft->context())) return;
                if (small_injection_batch) {
                    t.assert_equal(uint32_t(8),llama_n_ubatch(draft->context()));
                    t.assert_equal(uint32_t(256),llama_n_ubatch(target.get()));
                    t.assert_equal(uint32_t(256),draft->context()->get_cparams().n_batch);
                    t.assert_true(profile[0]+1<=llama_n_ubatch(draft->context()));
                }
                params.speculative.draft.ctx_tgt=target.get(); params.speculative.draft.ctx_dft=draft->context();
                common_speculative_ptr spec(common_speculative_init(params.speculative,1));
                if (!t.assert_true(bool(spec))) return;
                auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
                auto * window=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(draft->context()));
                if (!t.assert_true(hybrid && window)) return;
                auto * stream=hybrid->get_mem_attn()->get_kv_stream();
                if (streaming && independent_rows) {
                    const auto * parent = llama_context_compute_memory(target.get());
                    const auto * child = llama_context_compute_memory(draft->context());
                    if (!t.assert_true(parent && child && child->borrows_serial_parent() &&
                            parent->shared_parent() == child->shared_parent())) return;
                    t.assert_equal(uint32_t(profile[0]+1), draft->context()->get_cparams().serial_block_rows);
                    t.assert_equal(uint32_t(8), target->get_cparams().kv_stream_decode_rows());
                }
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
                if (independent_rows) {
                    for (size_t width=1;width<=8;++width) {
                        const auto first = position;
                        const auto name = "independent_width/"+std::to_string(width);
                        llama_tokens verification(width);
                        for (size_t i=0;i<verification.size();++i) verification[i]=seed[(first+i)%seed.size()];
                        verification[0]=anchor;
                        submit(name,verification,true,true);
                        if (streaming) {
                            llama_context_memory_diagnostics usage;
                            if (!t.assert_true("owning shared arena remains in decode during feature injection",
                                    llama_context_compute_memory(target.get())->diagnostics(usage) &&
                                    usage.phase == llama_memory_text_phase::decode)) return;
                            common_batch oversized(draft->context());
                            for (size_t i=0;i<profile[0]+2;++i) oversized.add(anchor,llama_pos(position+i),0,false);
                            const auto frontier = llama_memory_seq_pos_max(llama_get_memory(draft->context()),0);
                            if (!t.assert_equal(-1,llama_process(draft->context(),LLAMA_PROCESS_TYPE_DECODE,oversized.get()))) return;
                            t.assert_equal(frontier,llama_memory_seq_pos_max(llama_get_memory(draft->context()),0));
                        }
                        position=first+1;
                        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(position),-1) &&
                                llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(position),-1))) return;
                        history.push_back(anchor);
                        kv_stream_exact_kv(oracle,name+"/target_KV",hybrid->get_mem_attn(),first,1);
                        kv_stream_exact_kv(oracle,name+"/draft_KV",window->get_swa(),first,1);
                        kv_stream_exact_state state;
                        hybrid->get_mem_recr()->state_write(state,0,0); state.compare_with(oracle,name+"/recurrent");
                    }
                }
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

static void test_dflash_mixed_switching(testing & t,llama_model * model,const char * sidecar,
        ggml_type key=GGML_TYPE_Q8_0,ggml_type value=GGML_TYPE_Q4_0,
        const char * export_native=nullptr,const char * import_native=nullptr,bool all_widths=false,
        bool streamed_control=false,int predictions=3) {
    if (!t.assert_true(!(export_native && import_native) && (predictions == 3 || predictions == 7))) return;
    const size_t prefix = streamed_control ? 6143 : 2047;
    struct round {
        std::string name;
        int key;
        size_t accepted;
        int limit=7;
        bool probe=false;
    };
    struct decision {
        llama_token anchor;
        common_speculative_type source;
        llama_tokens proposed;
    };
    std::vector<round> program{
        {"hit_0",0,7},{"hit_1",0,7},{"hit_2",0,7,7,true},
        {"miss_0",-1,3},{"miss_1",-1,2},{"hit_after_miss_0",1,7},{"hit_after_miss_1",1,7},
        {"ambiguous",-2,1},
    };
    for (int accepted=0;accepted<=7;++accepted)
        program.push_back({"prefix_"+std::to_string(accepted),accepted,size_t(accepted)});
    program.push_back({"suppressed_0",-3,0});
    program.push_back({"suppressed_1",-3,3});
    program.push_back({"short_ngram",8,1,2});
    program.push_back({"short_neural",-1,0,1});
    if (all_widths) for (int limit=7;limit>=1;--limit)
        program.push_back({"verification_width_"+std::to_string(limit+1),7,size_t(limit),limit});
    program.push_back({"both_empty",-4,0});
    llama_tokens pattern;
    for (int group=0;group<9;++group) {
        for (int offset=0;offset<9;++offset) pattern.push_back(100+20*group+offset);
        pattern.push_back(999);
    }
    for (int variant=0;variant<2;++variant) {
        pattern.insert(pattern.end(),{400,401});
        for (int offset=0;offset<7;++offset) pattern.push_back(410+20*variant+offset);
        pattern.push_back(999);
    }
    const auto vocab=size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    if (!t.assert_true(vocab>6000)) return;
    kv_stream_bit_oracle matched,continuity;
    const std::array<uint64_t,16> profile{uint64_t(streamed_control ? 2 : 1),8192,prefix,vocab,uint64_t(llama_model_n_embd(model)),uint64_t(key),
        uint64_t(value),uint64_t(predictions),7,2,1,256,8,program.size(),228,uint64_t(streamed_control)};
    if (import_native) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> input(std::fopen(import_native,"rb"));
        if (!input) throw std::runtime_error("cannot open mixed native control");
        matched.load_control(input.get(),profile);
    }
    std::vector<decision> transcript;
    for (int mode=0;mode<3;++mode) {
        if (import_native && mode!=2) continue;
        if (streamed_control && (mode==1 || (export_native && mode==2))) continue;
        const bool reference=mode==1,streamed=mode==2 || streamed_control;
        if (reference) continuity.compare();
        if (mode==2) matched.compare();
        auto cp=llama_context_default_params();
        cp.n_ctx=8192; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1; cp.n_rs_seq=7;
        cp.n_outputs_max=cp.n_outputs_max_per_seq=8;
        cp.type_k=key; cp.type_v=value; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? size_t(228)*1024*1024 : 0;
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(target))) return;
        common_params base;
        base.n_ctx=8192; base.n_batch=base.n_ubatch=256; base.n_parallel=1; base.n_gpu_layers=999;
        base.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        base.speculative.types=reference ? std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH} :
            std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};
        if (streamed) std::reverse(base.speculative.types.begin(),base.speculative.types.end());
        base.speculative.draft.n_max=predictions; base.speculative.draft.n_min=0; base.speculative.draft.p_min=0;
        base.speculative.draft.n_gpu_layers=999; base.speculative.draft.mparams.path=sidecar;
        base.speculative.draft.cache_type_k=key; base.speculative.draft.cache_type_v=value;
        base.speculative.ngram_map_k4v.size_n=2; base.speculative.ngram_map_k4v.size_m=7; base.speculative.ngram_map_k4v.min_hits=1;
        auto params=common_base_params_to_speculative(base);
        params.n_ubatch=8;
        auto draft=common_speculative_init_from_params(params,model,target.get());
        if (!t.assert_true(draft && draft->context())) return;
        params.speculative.draft.ctx_tgt=target.get(); params.speculative.draft.ctx_dft=draft->context();
        common_speculative_ptr spec(common_speculative_init(params.speculative,1));
        auto probe_params=params.speculative;
        probe_params.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        common_speculative_ptr probe(common_speculative_init(probe_params,1));
        auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
        auto * window=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(draft->context()));
        if (!t.assert_true(spec && probe && hybrid && window)) return;
        auto * stream=hybrid->get_mem_attn()->get_kv_stream();
        auto * allocation=streamed ? target->get_compute_memory()->shared_parent() : nullptr;
        bool ring=false;
        std::set<size_t> verified_widths;
        size_t position=0;
        llama_tokens history;
        const auto pair_bytes=[&](const std::string & label,const void * data,size_t bytes) {
            if (!reference) matched.bytes(label,data,bytes);
        };
        const auto shared_bytes=[&](const std::string & label,const void * data,size_t bytes) {
            if (!reference) matched.bytes(label,data,bytes);
            if (!streamed) continuity.bytes(label,data,bytes);
        };
        const auto record_state=[&](const std::string & label) {
            if (!reference) {
                kv_stream_exact_kv(matched,label+"/target_KV",hybrid->get_mem_attn(),0,position);
                kv_stream_exact_state state;
                hybrid->get_mem_recr()->state_write(state,0,0);
                state.compare_with(matched,label+"/recurrent");
            }
            const auto visible=std::min(position,size_t(draft->model()->hparams.n_swa-1));
            if (!reference) kv_stream_exact_kv(matched,label+"/draft_KV",window->get_swa(),position-visible,visible);
            if (!streamed) kv_stream_exact_kv(continuity,label+"/draft_KV",window->get_swa(),position-visible,visible);
            if (!t.assert_equal(llama_pos(position-1),llama_memory_seq_pos_max(llama_get_memory(target.get()),0)) ||
                    !t.assert_equal(llama_pos(position-1),llama_memory_seq_pos_max(llama_get_memory(draft->context()),0))) return false;
            if (streamed && !t.assert_true(target->get_compute_memory()->shared_parent()==allocation &&
                    draft->context()->get_compute_memory()->shared_parent()==allocation &&
                    draft->context()->get_compute_memory()->borrows_serial_parent())) return false;
            return true;
        };
        const auto submit=[&](const std::string & name,const llama_tokens & tokens,bool decode,bool record,bool stale=false) {
            common_batch batch(target.get());
            for (size_t i=0;i<tokens.size();++i) batch.add(tokens[i],llama_pos(position+i),0,decode || i+1==tokens.size());
            llama_set_kv_stream_decode(target.get(),decode);
            if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return false;
            if (record) {
                for (size_t row=0;row<tokens.size();++row) {
                    const auto * logits=llama_get_logits_ith(target.get(),int(row));
                    if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }))) return false;
                    pair_bytes(name+"/logits/"+std::to_string(row),logits,vocab*sizeof(float));
                }
                const auto * ids=llama_model_target_layer_ids(draft->model());
                for (uint32_t i=0;i<llama_model_target_layer_ids_n(draft->model());++i) {
                    const auto * features=llama_get_embeddings_layer_inp(target.get(),uint32_t(ids[i]));
                    const auto count=tokens.size()*size_t(llama_model_n_embd(model));
                    if (!t.assert_true(features && std::all_of(features,features+count,[](float value) { return std::isfinite(value); }))) return false;
                    shared_bytes(name+"/features/"+std::to_string(i),features,count*sizeof(float));
                }
            }
            if (stale && streamed) {
                common_batch wrong(target.get());
                for (size_t i=0;i<tokens.size();++i) wrong.add(tokens[i],llama_pos(position+i+1),0,true);
                const auto frontier=llama_memory_seq_pos_max(llama_get_memory(draft->context()),0);
                if (!t.assert_true("stale target features reject before child writes",!common_speculative_process(spec.get(),wrong))) return false;
                t.assert_equal(frontier,llama_memory_seq_pos_max(llama_get_memory(draft->context()),0));
            }
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return false;
            llama_synchronize(target.get()); llama_synchronize(draft->context());
            position+=tokens.size();
            if (stream) {
                llama_kv_stream_runtime_diagnostics status;
                if (!t.assert_true(stream->runtime_diagnostics(status))) return false;
                ring|=decode && status.streaming_active;
            }
            return true;
        };
        const auto fill=[&](size_t end) {
            while (position<end) {
                llama_tokens tokens(std::min(size_t(256),end-position));
                for (size_t i=0;i<tokens.size();++i) tokens[i]=pattern[(position+i)%pattern.size()];
                if (!submit("fill",tokens,false,false)) return false;
                history.insert(history.end(),tokens.begin(),tokens.end());
            }
            return true;
        };
        if (!fill(prefix) || !record_state("prefill")) return;
        common_speculative_begin(spec.get(),0,history);
        const auto predict=[&](common_speculative * drafter,llama_token anchor,int limit) {
            llama_tokens result;
            auto & request=common_speculative_get_draft_params(drafter,0);
            request.drafting=true; request.n_max=limit; request.pos0=llama_pos(position); request.id_last=anchor;
            request.prompt=&history; request.result=&result;
            common_speculative_draft(drafter); llama_synchronize(draft->context());
            request.prompt=nullptr; request.result=nullptr;
            return result;
        };
        const auto selector=[&](const std::string & label,const llama_tokens & proposed) {
            shared_bytes(label+"/proposals",proposed.data(),proposed.size()*sizeof(llama_token));
            const auto lattice=dflash_exact_lattice(llama_get_embeddings_nextn(draft->context()),size_t(predictions+1),
                size_t(llama_model_n_embd(draft->model())),size_t(llama_model_dflash_selector_top_k(draft->model())),vocab);
            shared_bytes(label+"/selector",lattice.data(),lattice.size()*sizeof(uint32_t));
        };
        for (size_t index=0;index<program.size();++index) {
            const auto & step=program[index];
            std::fprintf(stderr,"DFlash mixed mode=%d round=%s position=%zu\n",mode,step.name.c_str(),position);
            if (step.key==-4) {
                if (!fill(8191)) return;
            } else {
                const llama_token bridge=step.key>=0 ? 100+20*step.key : step.key==-2 ? 400 : step.key==-3 ? 100 : 998;
                if (!submit(step.name+"/bridge",{bridge},true,true)) return;
                history.push_back(bridge);
            }
            llama_token anchor=step.key>=0 ? 101+20*step.key : step.key==-2 ? 401 : step.key==-3 ? 101 : llama_token(5000+index);
            if (step.key==-1 || step.key==-4) while (std::find(history.begin(),history.end(),anchor)!=history.end()) ++anchor;
            const auto expected=step.key>=0 ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V :
                step.key==-4 ? COMMON_SPECULATIVE_TYPE_NONE : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
            const auto before=common_speculative_get_stats(spec.get());
            llama_tokens proposed;
            auto supplied=expected;
            if (reference && expected==COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V) {
                if (!t.assert_equal(transcript[index].anchor,anchor)) return;
                proposed=transcript[index].proposed;
            } else {
                proposed=predict(spec.get(),anchor,step.limit);
                supplied=common_speculative_get_last_type(spec.get(),0);
                if (!t.assert_equal("actual selected proposal source",expected,supplied)) return;
            }
            const size_t length=expected==COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V ? std::min(7,step.limit) :
                expected==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ? std::min(predictions,step.limit) : 0;
            if (!t.assert_equal(length,proposed.size()) || !t.assert_true(step.accepted<=proposed.size())) return;
            if (mode==0) transcript.push_back({anchor,supplied,proposed});
            if (expected==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) selector(step.name+"/predict",proposed);
            const uint32_t type=uint32_t(supplied);
            pair_bytes(step.name+"/source",&type,sizeof(type));
            pair_bytes(step.name+"/selected",proposed.data(),proposed.size()*sizeof(llama_token));
            if (!reference) {
                const auto after=common_speculative_get_stats(spec.get());
                if (!t.assert_true(before.size()==2 && after.size()==2)) return;
                t.assert_equal(before[0].n_call_draft+1,after[0].n_call_draft);
                t.assert_equal(before[1].n_call_draft+(expected!=COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V),after[1].n_call_draft);
                t.assert_equal(before[1].n_predict+(expected==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH),after[1].n_predict);
            }
            if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(position),-1))) return;
            const auto first=position;
            llama_tokens verified{anchor}; verified.insert(verified.end(),proposed.begin(),proposed.end());
            if (!submit(step.name+"/verify",verified,true,true,index==0)) return;
            verified_widths.insert(verified.size());
            position=first+step.accepted+1;
            if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(position),-1) &&
                    llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(position),-1))) return;
            if (!reference || expected!=COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V)
                common_speculative_accept(spec.get(),0,uint16_t(step.accepted));
            history.insert(history.end(),verified.begin(),verified.begin()+step.accepted+1);
            if (!record_state(step.name)) return;
            if (!reference) {
                const auto stats=common_speculative_get_stats(spec.get());
                std::vector<uint64_t> counters;
                for (const auto & item:stats) counters.insert(counters.end(),{uint64_t(item.type),item.n_call_begin,item.n_call_draft,
                    item.n_call_accept,item.n_gen_drafts,item.n_acc_drafts,item.n_gen_tokens,item.n_acc_tokens,item.n_predict});
                matched.bytes(step.name+"/counters",counters.data(),counters.size()*sizeof(uint64_t));
            }
            if (step.probe) {
                const auto proposed=predict(probe.get(),777,3);
                if (!t.assert_equal(size_t(3),proposed.size())) return;
                selector(step.name+"/continuity_probe",proposed);
                if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(draft->context()),0,llama_pos(position),-1))) return;
                if (!record_state(step.name+"/after_probe")) return;
            }
        }
        if (all_widths && !t.assert_true("mixed sources execute every verification width TG1-TG8",
                verified_widths==std::set<size_t>({1,2,3,4,5,6,7,8}))) return;
        if (reference) continuity.finish();
        if (mode==0 && export_native) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> output(std::fopen(export_native,"wbx"));
            if (!output) throw std::runtime_error("cannot create mixed native control");
            matched.save_control(output.get(),profile);
        }
        if (mode==2) {
            if (!t.assert_true("actual mixed protocol reaches streaming ring",ring)) return;
            matched.finish();
        }
        t.out << "DFlash mixed mode=" << mode << " rounds=" << program.size() << " position=" << position
              << " ring=" << ring << " matched_records=" << matched.records() << " matched_bytes=" << matched.byte_count()
              << " continuity_records=" << continuity.records() << " continuity_bytes=" << continuity.byte_count()
              << " verification_widths=" << verified_widths.size() << '\n';
    }
}
