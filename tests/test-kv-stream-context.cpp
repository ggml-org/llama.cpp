#include "llama.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-memory-recurrent-spill.h"
#include "../src/llama-memory-bounded-kv.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-io.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "testing.h"
#include "kv-stream-bit-exact-test.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

#ifdef KV_STREAM_TILE_NATIVE_TEST
bool kv_stream_test_is_sm61_only();
bool kv_stream_test_is_arch_only(int cc);
int kv_stream_test_compiled_cc(ggml_backend_t backend);
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
bool kv_stream_test_native_graphs_enabled();
#endif

using context_ptr = std::unique_ptr<llama_context,decltype(&llama_free)>;
using model_ptr = std::unique_ptr<llama_model,decltype(&llama_model_free)>;
#include "kv-stream-dflash-test.h"
static bool auxiliary_control = false, embedded_mtp_control = false, target_tg3_control = false, target_stream_tg4_control = false;
static bool suspend_control = false;
static bool suspend_resident_only = false;
static bool resume_control = false, resume_resident_only = false;
static bool target_with_stock_draft = false;
static bool f16_control = false, shared_budget_control = false;
static bool resident_control = false, trace_control = false;
static bool mtp_memory_probe = false, mtp_sustained_control = false, mtp_sustained_rs_control = false;
static bool target_drift_control = false, target_drift_rs_control = false;
static bool rollback_probe = false;
static bool wide_rollback_control = false;
static bool vision_mtp_handoff = false;
static bool serial_workspace_growth = false;
static bool mtp_bit_exact_control = false;
static const char * dflash_native_model = nullptr;
static bool dflash_shared_workspace_control = false;
static bool dflash_exact_control = false;
static bool dflash_short_control = false;
static bool test_cache_explicit = false;
static ggml_type test_cache_k = GGML_TYPE_Q8_0, test_cache_v = GGML_TYPE_Q4_0;

// Qualify ordinary feature injection without a stream pool or a shared phase arena.
static void test_dflash_native(testing & t, llama_model * model) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 1;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (dflash_shared_workspace_control) {
        cp.shared_device_memory_bytes=512u*1024u*1024u;
        cp.n_rs_seq=3;
    }
    context_ptr target(llama_init_from_model(model, cp), llama_free);
    if (!t.assert_true(bool(target))) return;
    common_params params;
    params.n_ctx = 4096;
    params.n_batch = params.n_ubatch = 256;
    params.n_parallel = 1;
    params.n_gpu_layers = 999;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    params.speculative.draft.n_max = 1;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    params.speculative.draft.n_gpu_layers = 999;
    params.speculative.draft.mparams.path = dflash_native_model;
    auto draft = common_speculative_init_from_params(params, model, target.get());
    if (!t.assert_true(bool(draft) && draft->context())) return;
    if (dflash_shared_workspace_control) {
        auto * target_owner=llama_context_compute_memory(target.get());
        auto * draft_owner=llama_context_compute_memory(draft->context());
        if (!t.assert_true(target_owner && draft_owner && draft_owner->borrows_serial_parent())) return;
        t.assert_true(!draft->context()->get_cparams().mtp_publish_host);
        t.assert_true(target_owner->shared_parent() == draft_owner->shared_parent());
    }
    auto * stock=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(draft->context()));
    if (!t.assert_true(stock && stock->get_base()->get_layer_ids().empty())) return;
    if (dflash_shared_workspace_control) {
        auto * k=stock->get_swa()->get_k_storage(0);
        auto * parent=llama_context_compute_memory(target.get())->shared_parent();
        const auto base=uintptr_t(ggml_backend_buffer_get_base(parent));
        const auto bytes=ggml_backend_buffer_get_size(parent);
        if (!t.assert_true(ggml_backend_buffer_is_view(k->buffer) && uintptr_t(k->data) >= base &&
                uintptr_t(k->data)-base < bytes && ggml_nbytes(k) <= bytes-(uintptr_t(k->data)-base))) return;
        auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
        const auto * bank=hybrid->get_mem_recr()->spill_bank();
        auto * stage=bank ? bank->staging_tensor(0,true,1) : nullptr;
        if (!t.assert_true(stage && ggml_backend_buffer_is_view(stage->buffer) && uintptr_t(stage->data) >= base &&
                uintptr_t(stage->data)-base < bytes && ggml_nbytes(stage) <= bytes-(uintptr_t(stage->data)-base))) return;
    }
    llama_memory_bounded_kv_layout storage;
    if (!t.assert_true(stock->get_swa()->describe_bounded_storage(903,1,4096,256,storage))) return;
    t.assert_equal(uint32_t(2304),storage.allocation.capacity_rows);
    t.assert_equal(size_t(5),storage.layers.size());
    t.assert_equal(uint32_t(128),storage.cache.format.key_head_dim);
    t.assert_equal(uint32_t(8),storage.cache.format.kv_heads);
    t.assert_true(!stock->get_base()->describe_bounded_storage(904,1,4096,256,storage));
    auto cache_owner=stock->get_swa()->retain_bounded_storage(903,1,4096,256);
    if (!t.assert_true(bool(cache_owner))) return;
    auto source=cache_owner->read({903,1});
    if (!t.assert_true(bool(source))) return;
    t.assert_equal(storage.allocation.bytes,ggml_backend_buffer_get_size(source.buffer()));
    t.assert_equal(uint64_t(0),cache_owner->begin_write({903,1}));
    source.reset();
    params.speculative.draft.ctx_tgt = target.get();
    params.speculative.draft.ctx_dft = draft->context();
    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    if (!t.assert_true(bool(spec))) return;
    const auto tokens = common_tokenize(target.get(), "An ordinary DFlash2 feature injection control. ", true);
    common_batch batch(target.get());
    for (int i = 0; i < 256; ++i) batch.add(tokens[size_t(i)%tokens.size()], i, 0, i == 255);
    if (dflash_shared_workspace_control) t.assert_true(!llama_serial_features_match(target.get(),batch.get()));
    if (!t.assert_equal(0, llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()))) return;
    if (dflash_shared_workspace_control) {
        t.assert_true(llama_serial_features_match(target.get(),batch.get()));
        common_batch wrong_position(target.get());
        for (int i=0;i<256;++i) wrong_position.add(tokens[size_t(i)%tokens.size()],i+1,0,i == 255);
        t.assert_true(!llama_serial_features_match(target.get(),wrong_position.get()));
    }
    const auto injection=cache_owner->begin_write({903,1});
    if (!t.assert_true(injection != 0)) return;
    const auto injected=common_speculative_process(spec.get(), batch);
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(injection,injected));
    if (!t.assert_true(injected)) return;
    t.assert_equal(llama_pos(255), llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0));
    llama_tokens proposals;
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.drafting = true;
    dp.n_max = 1;
    dp.pos0 = 256;
    dp.id_last = tokens[0];
    dp.prompt = &tokens;
    dp.result = &proposals;
    const auto prediction=cache_owner->begin_write(cache_owner->identity());
    if (!t.assert_true(prediction != 0)) return;
    auto * draft_workspace=llama_context_compute_memory(draft->context());
    if (dflash_shared_workspace_control) t.assert_equal(int32_t(0),llama_set_adapter_cvec(draft->context(),nullptr,0,0,-1,-1));
    common_speculative_draft(spec.get());
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(prediction,proposals.size() == 1));
    t.assert_equal(size_t(1), proposals.size());
    source=cache_owner->read(cache_owner->identity());
    t.assert_true(bool(source));
    source.reset();
    const size_t state_bytes=llama_state_seq_get_size(draft->context(),0);
    std::vector<uint8_t> saved(state_bytes);
    if (!t.assert_true(state_bytes && llama_state_seq_get_data(draft->context(),saved.data(),saved.size(),0) == state_bytes)) return;
    if (dflash_shared_workspace_control) {
        t.assert_true(draft_workspace == llama_context_compute_memory(draft->context()));
        auto * parent=llama_context_compute_memory(target.get());
        auto * allocation=parent->shared_parent();
        t.assert_equal(int32_t(0),llama_set_adapter_cvec(target.get(),nullptr,0,0,-1,-1));
        batch.clear(); batch.add(tokens[0],256,0,true);
        llama_set_kv_stream_decode(target.get(),true);
        if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        batch.clear(); for (int i=0;i<256;++i) batch.add(tokens[0],257+i,0,i == 255);
        llama_set_kv_stream_decode(target.get(),false);
        if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        t.assert_true(parent == llama_context_compute_memory(target.get()) && allocation == parent->shared_parent());
        std::vector<uint8_t> retained(state_bytes);
        t.assert_equal(state_bytes,llama_state_seq_get_data(draft->context(),retained.data(),retained.size(),0));
        t.assert_true(retained == saved);
        common_batch oversized(draft->context());
        for (int i=0;i<9;++i) oversized.add(tokens[0],258+i,0,true);
        t.assert_true(llama_process(draft->context(),LLAMA_PROCESS_TYPE_DECODE,oversized.get()) != 0);
    }
    const auto restore=cache_owner->begin_write(cache_owner->identity());
    if (!t.assert_true(restore != 0)) return;
    const bool restored=llama_state_seq_set_data(draft->context(),saved.data(),saved.size(),0) == state_bytes;
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(restore,restored));
    t.assert_true(restored);
    std::vector<uint8_t> readback(state_bytes);
    t.assert_equal(state_bytes,llama_state_seq_get_data(draft->context(),readback.data(),readback.size(),0));
    t.assert_true(saved == readback);
    const auto reset=cache_owner->begin_write(cache_owner->identity());
    if (!t.assert_true(reset != 0)) return;
    llama_memory_clear(llama_get_memory(draft->context()),true);
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(reset,true));
    source=cache_owner->read(cache_owner->identity());
    auto full_params=cp;
    full_params.n_rs_seq=0;
    full_params.swa_full=true;
    full_params.ctx_other=target.get();
    full_params.shared_device_memory_bytes=0;
    context_ptr full(llama_init_from_model(draft->model(),full_params),llama_free);
    if (dflash_shared_workspace_control) {
        t.assert_true(!full);
        cache_owner.reset(); spec.reset(); draft.reset();
        t.assert_true(bool(source));
        return;
    }
    if (!t.assert_true(bool(full))) return;
    auto * full_stock=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(full.get()));
    if (!t.assert_true(full_stock != nullptr)) return;
    t.assert_true(!full_stock->get_swa()->describe_bounded_storage(905,1,4096,256,storage));
    t.assert_true(!full_stock->get_swa()->retain_bounded_storage(905,1,4096,256));
    full.reset();
    cache_owner.reset(); spec.reset(); draft.reset();
    t.assert_true(bool(source) && source.layout()->allocation.bytes == storage.allocation.bytes);
}

// Fail one candidate view after the draft's initial view, then permit parent recovery.
struct serial_parent_view_failure {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t,size_t,size_t);
    inline static serial_parent_view_failure * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t calls = 0;
    explicit serial_parent_view_failure(ggml_backend_buffer_t parent) : parent(parent),original(parent->view_buffer) {
        GGML_ASSERT(original && !active);
        active = this;
        parent->view_buffer = [](ggml_backend_buffer_t buffer,size_t offset,size_t bytes) {
            GGML_ASSERT(active && buffer == active->parent);
            if (++active->calls == 2) return static_cast<ggml_backend_buffer_t>(nullptr);
            return active->original(buffer,offset,bytes);
        };
    }
    ~serial_parent_view_failure() { parent->view_buffer = original; active = nullptr; }
};

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

// Compare matching target forwards, selected rollback state and replay without allocating a draft context.
static void test_wide_target_rollback(testing & t,llama_model * model) {
    kv_stream_exact_config cfg;
    std::ifstream input(cfg.corpus,std::ios::binary);
    if (!t.assert_true(bool(input))) return;
    const std::string article((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
    const auto * vocabulary=llama_model_get_vocab(model);
    const int count=-llama_tokenize(vocabulary,article.data(),int(article.size()),nullptr,0,false,false);
    if (!t.assert_true(count >= 16381)) return;
    std::vector<llama_token> corpus(size_t(count),0);
    if (!t.assert_equal(count,llama_tokenize(vocabulary,article.data(),int(article.size()),corpus.data(),count,false,false))) return;
    const size_t vocab_size=llama_vocab_n_tokens(vocabulary);
    for (uint32_t depth : {1u,3u,7u}) for (size_t prefix : {249u,505u,2041u,16381u}) {
        kv_stream_bit_oracle oracle;
        for (bool streamed : {false,true}) {
            if (streamed) oracle.compare();
            auto params=llama_context_default_params();
            params.n_ctx=uint32_t((prefix+6*(depth+2)+255)/256*256);
            params.n_batch=params.n_ubatch=256; params.n_seq_max=1; params.n_rs_seq=depth;
            params.n_threads=params.n_threads_batch=8;
            params.n_outputs_max=params.n_outputs_max_per_seq=depth+1;
            params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
            params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            // Include a real phase-arena handoff while other streamed cases force a small physical ring.
            if (streamed && depth == 7 && prefix == 505) params.shared_device_memory_bytes=size_t(2240)*1048576;
            else params.kv_stream_pool_bytes=streamed ? 16*1048576 : 0;
            context_ptr ctx(llama_init_from_model(model,params),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!t.assert_true(hybrid != nullptr)) return;
            auto * recurrent=hybrid->get_mem_recr();
            auto * kv=hybrid->get_mem_attn();
            auto * stream=kv->get_kv_stream();
            t.assert_equal(streamed,recurrent->spill_enabled());
            if (streamed) {
                if (!t.assert_true(stream && !stream->auxiliary_cache())) return;
                t.out << "rollback D=" << depth << " prefix=" << prefix << " stage MiB=" <<
                    double(recurrent->spill_bank()->staged_device_bytes())/1048576 << '\n';
            }
            const auto drain=[&] {
                llama_synchronize(ctx.get());
                if (!recurrent->complete_spill() || !recurrent->complete_restore_spill())
                    throw std::runtime_error("failed to drain rollback test");
            };
            const auto state=[&](const std::string & name) {
                drain();
                kv_stream_exact_state snapshot;
                recurrent->state_write(snapshot,0,0);
                snapshot.compare_with(oracle,name+"/state");
            };
            const auto submit=[&](size_t first,size_t tokens,bool decode,bool proposals=false) {
                common_batch batch(ctx.get());
                for (size_t i=0;i<tokens;++i) {
                    // A rejected proposal differs from its correction, exercising actual KV overwrite.
                    const auto token=proposals && i ? llama_token((size_t(corpus[first+i])+1+i)%vocab_size) : corpus[first+i];
                    batch.add(token,llama_pos(first+i),0,decode || i+1==tokens);
                }
                llama_set_kv_stream_decode(ctx.get(),decode);
                return llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
            };
            for (size_t first=0;first<prefix;first+=256)
                if (!t.assert_equal(0,submit(first,std::min(size_t(256),prefix-first),false))) return;
            state("prefill");
            kv_stream_exact_kv(oracle,"prefill/KV",kv,0,prefix);
            size_t position=prefix;
            const uint32_t short_widths[6]={8,5,7,6,2,4};
            for (uint32_t round=0;round<6;++round) {
                const auto name="round/"+std::to_string(round);
                const size_t width=depth == 7 && prefix == 2041 ? short_widths[round] : depth+1;
                if (!t.assert_equal(0,submit(position,width,true,true))) return;
                drain();
                for (size_t q=0;q<width;++q) {
                    const auto * logits=llama_get_logits_ith(ctx.get(),int32_t(q));
                    if (!t.assert_true(logits != nullptr)) return;
                    oracle.bytes(name+"/logits/"+std::to_string(q),logits,vocab_size*sizeof(float));
                }
                state(name+"/verify");
                const size_t proposals=width-1;
                const size_t accepted=round == 0 || round == 3 ? 0 : round == 1 ? proposals/2 : round == 4 ? proposals-1 : proposals;
                const size_t kept=accepted+1;
                if (kept < width && !t.assert_true(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,llama_pos(position+kept),-1))) return;
                t.assert_equal(llama_pos(position+kept-1),llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0));
                state(name+"/selected");
                kv_stream_exact_kv(oracle,name+"/retained_KV",kv,position,kept);
                position+=kept;
                if (!t.assert_equal(0,submit(position,1,true))) return;
                drain();
                oracle.bytes(name+"/correction_logits",llama_get_logits_ith(ctx.get(),0),vocab_size*sizeof(float));
                state(name+"/correction");
                kv_stream_exact_kv(oracle,name+"/corrected_KV",kv,position,1);
                ++position;
                if (streamed) t.assert_equal(position,stream->tokens());
            }
            const size_t capacity=llama_n_ctx(ctx.get());
            for (;position<capacity;) {
                const size_t rows=std::min(size_t(256),capacity-position);
                if (!t.assert_equal(0,submit(position,rows,false))) return;
                position+=rows;
            }
            state("context_end");
            if (!t.assert_true(submit(position,depth+1,true) != 0)) return;
            t.assert_equal(llama_pos(capacity-1),llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0));
            state("rejected_context_end");
            if (streamed) oracle.finish();
        }
        t.out << "D=" << depth << " prefix=" << prefix << " native/spilled rollback bits match (" <<
            oracle.records() << " records, " << oracle.byte_count() << " bytes)\n";
    }
}

static run_result evaluate(testing & t,llama_model * model,const std::vector<llama_token> & prompt,
        uint32_t ubatch,size_t pool,const std::vector<llama_token> & forced) {
    run_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 1024; params.n_batch = 512; params.n_ubatch = ubatch;
    params.n_threads = params.n_threads_batch = 8; params.n_seq_max = 1;
    params.type_k = f16_control ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    params.type_v = f16_control ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (shared_budget_control) params.shared_device_memory_bytes = pool;
    else params.kv_stream_pool_bytes = pool;
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
        llama_pos wrong_first=-2;
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

struct serial_phase_result {
    std::vector<std::vector<float>> logits;
    recurrent_snapshot recurrent;
    std::vector<size_t> pool_grants;
    std::vector<int64_t> prefill_after_decode_us;
};

static serial_phase_result evaluate_serial_phases(testing & t, llama_model * model,
        const std::vector<llama_token> & prompt, size_t pool) {
    serial_phase_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 2048; params.n_batch = 512; params.n_ubatch = 256;
    params.n_threads = params.n_threads_batch = 8; params.n_seq_max = 1;
    params.type_k = GGML_TYPE_Q8_0; params.type_v = GGML_TYPE_Q4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (shared_budget_control) params.shared_device_memory_bytes = pool;
    else params.kv_stream_pool_bytes = pool;
    context_ptr context(llama_init_from_model(model,params),llama_free);
    if (!t.assert_true(bool(context))) return result;
    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
    auto * stream = hybrid->get_mem_attn()->get_kv_stream();
    if (!t.assert_true(bool(stream) == (pool != 0))) return result;
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch = llama_batch_init(512,0,1);
    struct batch_guard {
        llama_batch & batch;
        ~batch_guard() { llama_batch_free(batch); }
    } batch_owner{batch};
    size_t position = 0;

    const auto submit = [&](const std::vector<llama_token> & tokens, bool decode) {
        if (tokens.empty() || tokens.size() > 512) return false;
        batch.n_tokens = int32_t(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = llama_pos(position+i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i+1 == tokens.size();
        }
        llama_set_kv_stream_decode(context.get(),decode);
        if (llama_decode(context.get(),batch) != 0) return false;
        position += tokens.size();
        return true;
    };
    const auto capture = [&] {
        auto * logits = llama_get_logits_ith(context.get(),-1);
        result.logits.emplace_back(logits,logits+vocab);
        if (stream) result.pool_grants.push_back(stream->pool_grant_bytes());
    };
    const auto append = [&](size_t first,size_t count,bool decode) {
        if (first > prompt.size() || count > prompt.size()-first) return false;
        return submit({prompt.begin()+first,prompt.begin()+first+count},decode);
    };
    const auto append_prompt = [&](size_t count) {
        for (size_t first = 0; first < count; first += 512) {
            const size_t chunk = std::min(size_t(512),count-first);
            if (!append(first,chunk,false)) return false;
        }
        return true;
    };
    const auto fixed_decode = [&](size_t count,size_t seed) {
        for (size_t i = 0; i < count; ++i) {
            if (!append((seed+i)%prompt.size(),1,true)) return false;
        }
        return true;
    };
    const auto timed_prefill = [&](size_t first,size_t count) {
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = append(first,count,false);
        const auto end = std::chrono::steady_clock::now();
        result.prefill_after_decode_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
        return ok;
    };
    const auto clear = [&] {
        llama_memory_clear(llama_get_memory(context.get()),true);
        position = 0;
        return !stream || stream->tokens() == 0;
    };
    const auto cancel_and_restore = [&](size_t first,size_t count,bool decode) {
        if (first > prompt.size() || count > prompt.size()-first || !count || count > 512) return false;
        const size_t state_size = llama_state_get_size(context.get());
        std::vector<uint8_t> state(state_size);
        if (llama_state_get_data(context.get(),state.data(),state.size()) != state_size) return false;
        batch.n_tokens = int32_t(count);
        for (size_t i = 0; i < count; ++i) {
            batch.token[i] = prompt[first+i];
            batch.pos[i] = llama_pos(position+i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i+1 == count;
        }
        struct abort_data { int calls = 0; } abort;
        llama_set_kv_stream_decode(context.get(),decode);
        llama_set_abort_callback(context.get(),[](void * opaque) {
            return ++static_cast<abort_data *>(opaque)->calls > 0;
        },&abort);
        const int32_t status = llama_decode(context.get(),batch);
        llama_set_abort_callback(context.get(),nullptr,nullptr);
        if (status != 2 || abort.calls == 0 ||
                llama_state_set_data(context.get(),state.data(),state.size()) != state_size) return false;
        return !stream || stream->tokens() == position;
    };

    if (!t.assert_true(append_prompt(640))) return {};
    capture();
    if (!t.assert_true(fixed_decode(192,3))) return {};
    capture();
    if (!t.assert_true(timed_prefill(80,7))) return {};
    capture();
    if (!t.assert_true(fixed_decode(16,101))) return {};
    capture();
    if (!t.assert_true(timed_prefill(160,192))) return {};
    capture();
    if (!t.assert_true(fixed_decode(16,211))) return {};
    capture();
    if (!t.assert_true(timed_prefill(17,3))) return {};
    capture();
    if (!t.assert_true(fixed_decode(8,307))) return {};
    capture();

    if (!t.assert_true(clear() && append_prompt(17))) return {};
    capture();
    if (!t.assert_true(fixed_decode(8,401))) return {};
    capture();
    if (!t.assert_true(clear() && append_prompt(640))) return {};
    capture();
    if (!t.assert_true(fixed_decode(8,503))) return {};
    capture();

    if (!t.assert_true(cancel_and_restore(80,7,false))) return {};
    if (!t.assert_true(append(80,7,false))) return {};
    capture();
    if (!t.assert_true(cancel_and_restore(503,1,true))) return {};
    if (!t.assert_true(append(503,1,true))) return {};
    capture();
    llama_synchronize(context.get());
    hybrid->get_mem_recr()->state_write(result.recurrent,0,0);
    if (stream) t.assert_equal(position,stream->tokens());
    return result;
}

int main(int argc,char ** argv) {
    int baseline=0;
    bool exact_options = false, cache_options = false;
    for (int i = 1; i < argc; ++i) {
        mtp_bit_exact_control |= !std::strcmp(argv[i], "--mtp-bit-exact");
        exact_options |= !std::strncmp(argv[i], "--exact-", 8);
        cache_options |= !std::strncmp(argv[i], "--test-cache-", 13);
    }
    if (((mtp_bit_exact_control || cache_options) && (argc < 3 || std::strcmp(argv[1], "--model"))) ||
            (exact_options && !mtp_bit_exact_control)) {
        std::fprintf(stderr, "strict MTP usage: %s --model GGUF --mtp-bit-exact [--test-cache-k TYPE --test-cache-v TYPE --exact-prefill N --exact-rounds N --exact-pool-mib N --exact-batch-size N --exact-length N --exact-prefill-file FILE]\n", argv[0]);
        return 1;
    }
    for (int i=3;i<argc;++i) {
        if (!std::strcmp(argv[i],"--pascal")) baseline=610;
        if (!std::strcmp(argv[i],"--sm75")) baseline=750;
        if (!std::strcmp(argv[i],"--sm86")) baseline=860;
        if (!std::strcmp(argv[i],"--sm89")) baseline=890;
        if (!std::strcmp(argv[i],"--sm120")) baseline=1200;
    }
#ifdef KV_STREAM_TILE_NATIVE_TEST
    if (baseline && !kv_stream_test_is_arch_only(baseline)) return 77;
#else
    if (baseline) return 77;
#endif
    testing t;
    test_kv_stream_bit_exact_comparator(t);
    t.test("short_dflash_profiles_keep_native_controls_physically_bounded", [&](testing & t) {
        const auto original = dflash_exact_profiles(false, false, false, false, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_equal(size_t(16000), original.back()[1]);
        t.assert_equal(size_t(16384), original.back()[2]);
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1};
        for (auto key : types) for (auto value : types) {
            const auto profiles = dflash_exact_profiles(false, false, false, true, key, value);
            t.assert_equal(size_t(4), profiles.size());
            const size_t maximum = ggml_row_size(key,1024)+ggml_row_size(value,1024) > 4096 ? 4096 : 8192;
            for (const auto & profile : profiles) {
                t.assert_true(profile[2] <= maximum && profile[1]+64 <= profile[2]);
                t.assert_true(profile[0] == 1 || profile[0] == 3 || profile[0] == 7);
            }
            t.assert_equal(size_t(2047), profiles.front()[1]);
            t.assert_equal(maximum-192, profiles.back()[1]);
        }
    });
    t.test("selector_lattice_permutations_preserve_scores_but_one_bit_changes_do_not",[](testing & t) {
        std::vector<float> a{0,0,0,0,0,0,4,2,0.2f,0.4f,0.2f,0.4f};
        auto b=a;
        b[6]=2; b[7]=4; b[8]=b[10]=0.4f; b[9]=b[11]=0.2f;
        const auto canonical=dflash_exact_lattice(a.data(),2,6,2,16);
        t.assert_true(canonical == dflash_exact_lattice(b.data(),2,6,2,16));
        uint32_t bits; std::memcpy(&bits,&b[8],sizeof(bits)); bits^=1;
        std::memcpy(&b[8],&bits,sizeof(bits)); b[10]=b[8];
        t.assert_true(canonical != dflash_exact_lattice(b.data(),2,6,2,16));
        a.insert(a.end(),{3,5,0.1f,0.2f,0.3f,0.4f});
        b={0,0,0,0,0,0,2,4,0.4f,0.2f,0.4f,0.2f,5,3,0.4f,0.3f,0.2f,0.1f};
        t.assert_true(dflash_exact_lattice(a.data(),3,6,2,16) == dflash_exact_lattice(b.data(),3,6,2,16));
    });
    t.test("streaming_is_disabled_by_default", [&](testing & t) {
        const auto defaults = llama_context_default_params();
        t.assert_equal(size_t(0),defaults.kv_stream_pool_bytes);
        t.assert_equal(size_t(0),defaults.shared_device_memory_bytes);
        t.assert_equal(uint32_t(0), defaults.kv_stream_auxiliary_layers);
    });
    if (argc < 3 || std::strcmp(argv[1],"--model")) return t.summary();
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--test-cache-k") || !std::strcmp(argv[i], "--test-cache-v")) {
            const bool key = !std::strcmp(argv[i], "--test-cache-k");
            if (++i == argc) {
                std::fprintf(stderr, "missing test cache type\n");
                return 1;
            }
            try {
                const auto type = kv_stream_exact_cache_type(argv[i]);
                (key ? test_cache_k : test_cache_v) = type;
                test_cache_explicit = true;
            } catch (const std::exception & error) {
                std::fprintf(stderr, "%s\n", error.what());
                return 1;
            }
            continue;
        }
        if (!std::strcmp(argv[i], "--dflash-native") && i+1 < argc) dflash_native_model = argv[++i];
        if (!std::strcmp(argv[i], "--dflash-shared-workspace")) dflash_shared_workspace_control = true;
        if (!std::strcmp(argv[i], "--dflash-bit-exact")) dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-short")) dflash_short_control = dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-lifecycle")) dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-context-end")) dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-rejection-long")) dflash_exact_control = true;
        f16_control = f16_control || !std::strcmp(argv[i],"--f16");
        resident_control = resident_control || !std::strcmp(argv[i],"--resident");
        shared_budget_control = shared_budget_control || !std::strcmp(argv[i],"--shared-budget");
        suspend_control = suspend_control || !std::strcmp(argv[i],"--suspend-only");
        suspend_resident_only = suspend_resident_only || !std::strcmp(argv[i],"--suspend-resident-only");
        suspend_control = suspend_control || suspend_resident_only;
        resume_control = resume_control || !std::strcmp(argv[i],"--resume-only");
        resume_resident_only = resume_resident_only || !std::strcmp(argv[i],"--resume-resident-only");
        resume_control = resume_control || resume_resident_only;
        trace_control = trace_control || !std::strcmp(argv[i],"--trace");
        auxiliary_control = auxiliary_control || !std::strcmp(argv[i],"--auxiliary-only");
        embedded_mtp_control = embedded_mtp_control || !std::strcmp(argv[i],"--embedded-mtp-pair");
        target_tg3_control = target_tg3_control || !std::strcmp(argv[i],"--target-tg3-only");
        target_stream_tg4_control = target_stream_tg4_control || !std::strcmp(argv[i],"--target-stream-tg4");
        target_with_stock_draft = target_with_stock_draft || !std::strcmp(argv[i],"--target-with-stock-draft");
        mtp_memory_probe = mtp_memory_probe || !std::strcmp(argv[i],"--mtp-memory-probe");
        mtp_sustained_control = mtp_sustained_control || !std::strcmp(argv[i],"--embedded-mtp-sustained");
        mtp_sustained_rs_control = mtp_sustained_rs_control || !std::strcmp(argv[i],"--embedded-mtp-sustained-rs");
        mtp_sustained_control = mtp_sustained_control || mtp_sustained_rs_control;
        target_drift_control = target_drift_control || !std::strcmp(argv[i],"--mtp-target-drift");
        target_drift_rs_control = target_drift_rs_control || !std::strcmp(argv[i],"--mtp-target-drift-rs");
        target_drift_control = target_drift_control || target_drift_rs_control;
        rollback_probe = rollback_probe || !std::strcmp(argv[i],"--mtp-rs-rollback-probe");
        wide_rollback_control = wide_rollback_control || !std::strcmp(argv[i],"--target-rs-rollback");
        vision_mtp_handoff = vision_mtp_handoff || !std::strcmp(argv[i],"--vision-mtp-handoff");
        serial_workspace_growth = serial_workspace_growth || !std::strcmp(argv[i],"--serial-workspace-growth");
    }
    embedded_mtp_control |= vision_mtp_handoff;
    if (test_cache_explicit && !mtp_bit_exact_control && !dflash_exact_control && !vision_mtp_handoff) {
        std::fprintf(stderr, "test cache types require a numerical oracle or vision handoff profile\n");
        return 1;
    }
    ggml_backend_load_all(); llama_backend_init();
    ggml_backend_ptr baseline_backend;
#ifdef KV_STREAM_TILE_NATIVE_TEST
    if (baseline) {
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!device) return 77;
        baseline_backend.reset(ggml_backend_dev_init(device,nullptr));
        if (!baseline_backend || kv_stream_test_compiled_cc(baseline_backend.get()) != baseline) return 77;
    }
    const int previous=baseline ? kv_stream_test_override_cc(baseline_backend.get(),baseline) : 0;
    struct restore {
        ggml_backend_t backend; int previous;
        ~restore() {if (backend) kv_stream_test_override_cc(backend,previous);}
    } restore_cc{baseline_backend.get(),previous};
#endif
    auto mparams = llama_model_default_params(); mparams.n_gpu_layers = 999;
    mparams.load_mtp = embedded_mtp_control || mtp_memory_probe || mtp_sustained_control || target_drift_control || rollback_probe || serial_workspace_growth || mtp_bit_exact_control;
    model_ptr model(llama_model_load_from_file(argv[2],mparams),llama_model_free);
    if (dflash_native_model && !dflash_exact_control) {
        if (!t.assert_true(bool(model))) return t.summary();
        t.test("ordinary_dflash_feature_workspace", [&](testing & t) { test_dflash_native(t, model.get()); });
        return t.summary();
    }
    if (!t.assert_true(bool(model))) return t.summary();
    if (dflash_exact_control) {
        if (!t.assert_true(dflash_native_model != nullptr)) return t.summary();
        const bool lifecycle=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--dflash-lifecycle"); });
        const bool context_end=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--dflash-context-end"); });
        const bool sustained=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--dflash-rejection-long"); });
        t.test("streamed_DFlash2_matches_native_forced_decisions",[&](testing & t) {
            test_kv_stream_dflash_exact(t,model.get(),dflash_native_model,lifecycle,context_end || sustained,sustained,
                test_cache_k,test_cache_v,test_cache_explicit || dflash_short_control,dflash_short_control);
        });
        return t.summary();
    }
    if (wide_rollback_control) {
        t.test("target_TG1_TG8_rollback_matches_native_without_draft_KV", [&](testing & t) {
            test_wide_target_rollback(t,model.get());
        });
        return t.summary();
    }
    if (mtp_bit_exact_control) {
        t.test("streamed_and_native_mtp_bit_exact", [&](testing & t) {
            test_kv_stream_mtp_bit_exact(t, model.get(), kv_stream_exact_arguments(argc, argv));
        });
        return t.summary();
    }
    if (serial_workspace_growth) {
        t.test("serial_draft_growth_rebuilds_target_and_preserves_long_gather", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 8192; p.n_batch = p.n_ubatch = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.shared_device_memory_bytes = size_t(2240)*1048576;
            p.n_outputs_max = p.n_outputs_max_per_seq = 4;
            p.kv_stream_auxiliary_layers = 1;
            context_ptr target(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(target))) return;
            llama_token token = 1000;
            llama_set_kv_stream_decode(target.get(),false);
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            auto checkpoint = std::vector<uint8_t>(llama_state_get_size(target.get()));
            if (!t.assert_equal(checkpoint.size(),llama_state_get_data(target.get(),checkpoint.data(),checkpoint.size()))) return;
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            const size_t vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            const auto * logits = llama_get_logits_ith(target.get(),-1);
            std::vector<float> expected(logits,logits+vocabulary);
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
            recurrent_snapshot expected_recurrent;
            hybrid->get_mem_recr()->state_write(expected_recurrent,0,0);
            if (!t.assert_equal(checkpoint.size(),llama_state_set_data(target.get(),checkpoint.data(),checkpoint.size()))) return;
            const auto before = target->get_compute_memory()->phase_transition_count();
            auto * parent = target->get_compute_memory()->shared_parent();
            auto dp = p;
            dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            dp.ctx_other = target.get();
            dp.shared_device_memory_bytes = 0;
            dp.kv_stream_auxiliary_layers = 0;
            const auto old_revision = target->get_compute_memory()->graph_binding_revision();
            {
                serial_parent_view_failure fault(parent);
                context_ptr fallback(llama_init_from_model(model.get(),dp),llama_free);
                t.assert_true(fault.calls >= 2);
                // Recovery must remain usable even if separate fallback scratch cannot fit the GPU.
                if (fallback) t.assert_true(!fallback->get_compute_memory()->borrows_serial_parent());
            }
            t.assert_true(target->get_compute_memory()->valid());
            t.assert_equal(before,target->get_compute_memory()->phase_transition_count());
            t.assert_true(target->get_compute_memory()->graph_binding_revision() > old_revision);
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            logits = llama_get_logits_ith(target.get(),-1);
            t.assert_true(std::equal(expected.begin(),expected.end(),logits));
            recurrent_snapshot recovered_recurrent;
            hybrid->get_mem_recr()->state_write(recovered_recurrent,0,0);
            t.assert_true(expected_recurrent.metadata == recovered_recurrent.metadata && expected_recurrent.tensors == recovered_recurrent.tensors);
            if (!t.assert_equal(checkpoint.size(),llama_state_set_data(target.get(),checkpoint.data(),checkpoint.size()))) return;
            context_ptr draft(llama_init_from_model(model.get(),dp),llama_free);
            if (!t.assert_true(bool(draft))) return;
            t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
            t.assert_true(target->get_compute_memory()->phase_transition_count() > before);
            t.assert_true(parent == target->get_compute_memory()->shared_parent());
            t.assert_equal(size_t(p.shared_device_memory_bytes),target->get_compute_memory()->shared_parent_capacity());
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            logits = llama_get_logits_ith(target.get(),-1);
            t.assert_true(std::equal(expected.begin(),expected.end(),logits));
            auto batch = llama_batch_init(256,llama_model_n_embd(model.get()),1);
            std::vector<llama_token> tokens(256,token);
            batch.token = tokens.data();
            struct cleanup {
                llama_batch & batch;
                ~cleanup() { batch.token = nullptr; llama_batch_free(batch); }
            } guard{batch};
            std::fill(batch.embd,batch.embd+256*llama_model_n_embd(model.get()),0.0f);
            for (int first = 0; first < 7936; first += 256) {
                batch.n_tokens = 256;
                for (int row = 0; row < 256; ++row) {
                    batch.pos[row] = first+row;
                    batch.n_seq_id[row] = 1;
                    batch.seq_id[row][0] = 0;
                    batch.logits[row] = row == 255;
                }
                if (!t.assert_equal(0,llama_decode(draft.get(),batch))) return;
            }
            auto * kv = static_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
            t.assert_equal(size_t(7936),kv->mtp_auxiliary_cache()->tokens());
            t.assert_equal(size_t(p.shared_device_memory_bytes),target->get_compute_memory()->shared_parent_capacity());
        });
        return t.summary();
    }
    if (rollback_probe) {
        t.test("streamed_mtp_uses_bounded_host_spilled_recurrent_rollback", [&](testing & t) {
            auto p=llama_context_default_params();
            p.n_ctx=1024; p.n_batch=p.n_ubatch=256;
            p.n_threads=p.n_threads_batch=8;
            p.type_k=GGML_TYPE_Q8_0; p.type_v=GGML_TYPE_Q4_0;
            p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes=16*1048576;
            p.kv_stream_auxiliary_layers=1;
            p.n_rs_seq=3;
            auto invalid=p;
            invalid.n_rs_seq=8;
            context_ptr unsafe(llama_init_from_model(model.get(),invalid),llama_free);
            t.assert_true(!unsafe);
            invalid=p;
            invalid.n_rs_seq=4;
            context_ptr too_deep(llama_init_from_model(model.get(),invalid),llama_free);
            t.assert_true(!too_deep);
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            t.assert_equal(uint32_t(3),llama_n_rs_seq(ctx.get()));
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!t.assert_true(hybrid != nullptr && hybrid->get_mem_attn()->get_kv_stream())) return;
            auto * spill=hybrid->get_mem_recr()->spill_bank();
            if (!t.assert_true(spill != nullptr)) return;
            t.assert_true(spill->host_bytes() >= 448*1048576 && spill->host_bytes() <= 450*1048576);
            t.assert_true(spill->staged_device_bytes() >= 18*1048576 && spill->staged_device_bytes() <= 20*1048576);
            auto batch=llama_batch_init(256,0,1);
            for (int i=0; i<256; ++i) {
                batch.token[i]=1; batch.pos[i]=i; batch.n_seq_id[i]=1;
                batch.seq_id[i][0]=0; batch.logits[i]=i==255;
            }
            batch.n_tokens=256;
            llama_set_kv_stream_decode(ctx.get(),false);
            t.assert_equal(0,llama_decode(ctx.get(),batch));
            batch.n_tokens=4;
            for (int i=0; i<4; ++i) {
                batch.token[i]=1; batch.pos[i]=256+i; batch.logits[i]=i==3;
            }
            llama_set_kv_stream_decode(ctx.get(),true);
            t.assert_equal(0,llama_decode(ctx.get(),batch));
            llama_batch_free(batch);
        });
        t.test("independent_target_rollback_needs_no_auxiliary_kv_layer", [&](testing & t) {
            auto p=llama_context_default_params();
            p.n_ctx=1024; p.n_batch=p.n_ubatch=256; p.n_seq_max=1;
            p.n_threads=p.n_threads_batch=8;
            p.n_rs_seq=3;
            p.n_outputs_max=p.n_outputs_max_per_seq=4;
            p.type_k=GGML_TYPE_Q8_0; p.type_v=GGML_TYPE_Q4_0;
            p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes=16*1048576;
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!t.assert_true(hybrid && hybrid->get_mem_recr()->spill_enabled())) return;
            auto * target=hybrid->get_mem_attn()->get_kv_stream();
            if (!t.assert_true(target && !target->auxiliary_cache())) return;
            t.assert_equal(uint32_t(4),ctx->get_cparams().kv_stream_decode_rows());
            auto batch=llama_batch_init(256,0,1);
            struct release { llama_batch & batch; ~release() { llama_batch_free(batch); } } guard{batch};
            const auto submit=[&](uint32_t width,llama_pos first,bool decode) {
                batch.n_tokens=int32_t(width);
                for (uint32_t i=0;i<width;++i) {
                    batch.token[i]=1; batch.pos[i]=first+llama_pos(i);
                    batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=decode || i+1==width;
                }
                llama_set_kv_stream_decode(ctx.get(),decode);
                return llama_decode(ctx.get(),batch);
            };
            if (!t.assert_equal(0,submit(256,0,false))) return;
            llama_pos position=256;
            for (uint32_t width : {1u,2u,3u,4u}) {
                if (!t.assert_equal(0,submit(width,position,true))) return;
                position+=llama_pos(width);
                t.assert_equal(size_t(position),target->tokens());
            }
            t.assert_true(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,position-3,-1));
            t.assert_equal(size_t(position-3),target->tokens());
            t.assert_equal(0,submit(1,position-3,true));
        });
        return t.summary();
    }
    if (target_drift_control) {
        t.test("teacher_forced_target_verification_and_replay", [&](testing & t) {
            if (!t.assert_true(argc >= 5)) return;
            std::ifstream input(argv[4]);
            if (!t.assert_true(bool(input))) return;
            const std::string article((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            const size_t prefill_tokens = std::getenv("LLAMA_DIAG_PREFILL_TOKENS") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_PREFILL_TOKENS"))) : 7932;
            const size_t decode_tokens = std::getenv("LLAMA_DIAG_DECODE_TOKENS") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_DECODE_TOKENS"))) : 16;
            if (!t.assert_true(prefill_tokens >= 256 && prefill_tokens <= 261888 &&
                    decode_tokens >= 16 && decode_tokens <= 256 && prefill_tokens+decode_tokens <= 262144)) return;
            const auto * vocab = llama_model_get_vocab(model.get());
            const int needed = -llama_tokenize(vocab, article.data(), int(article.size()), nullptr, 0, false, false);
            std::vector<llama_token> tokens(needed);
            if (!t.assert_true(llama_tokenize(vocab, article.data(), int(article.size()),
                    tokens.data(), needed, false, false) >= int32_t(prefill_tokens))) return;
            auto p = llama_context_default_params();
            p.n_ctx = uint32_t((prefill_tokens+decode_tokens+255)/256*256); p.n_batch = p.n_ubatch = 256;
            p.n_rs_seq = target_drift_rs_control ? 3 : 0;
            p.n_threads = p.n_threads_batch = 8;
            p.n_outputs_max = p.n_outputs_max_per_seq = 4;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            struct diagnostic_trace {
                bool enabled = false;
                size_t stable = 0;
                std::vector<std::pair<std::string,std::vector<float>>> values;
                struct cache { std::string name; std::vector<uint8_t> bytes; size_t stride; };
                std::vector<cache> caches;
            } trace_stock, trace_streamed;
            p.cb_eval_user_data = &trace_stock;
            p.cb_eval = [](ggml_tensor * tensor, bool ask, void * opaque) {
                auto & trace = *static_cast<diagnostic_trace *>(opaque);
                std::string name = tensor->name;
                bool first_attention = false;
                if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                    const ggml_tensor * q = tensor->src[0];
                    for (int depth = 0; q && depth < 8; ++depth, q = q->src[0]) {
                        if (std::strcmp(q->name,"Qcur-3") == 0) { first_attention = true; break; }
                    }
                }
                if (first_attention) name = "FA-3";
                if (!trace.enabled || tensor->type != GGML_TYPE_F32 ||
                        (name != "Qcur-3" && name != "Kcur-3" && name != "Vcur-3" &&
                         !first_attention && name != "attn_pregate-3")) return false;
                if (ask) return true;
                trace.values.emplace_back(name,std::vector<float>(ggml_nbytes(tensor)/sizeof(float)));
                ggml_backend_tensor_get(tensor,trace.values.back().second.data(),0,ggml_nbytes(tensor));
                if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                    for (int plane = 1; plane <= 2; ++plane) {
                        const auto * source = tensor->src[plane];
                        trace.caches.push_back({plane == 1 ? "Kcache" : "Vcache",
                            std::vector<uint8_t>(trace.stable*source->nb[1]),source->nb[1]});
                        ggml_backend_tensor_get(source,trace.caches.back().bytes.data(),0,trace.caches.back().bytes.size());
                    }
                }
                return true;
            };
            context_ptr stock(llama_init_from_model(model.get(), p), llama_free);
            const size_t arena_mib = std::getenv("LLAMA_DIAG_ARENA_MIB") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_ARENA_MIB"))) : 512;
            if (!t.assert_true(arena_mib >= 128 && arena_mib <= 4096)) return;
            p.shared_device_memory_bytes = arena_mib*1048576;
            p.kv_stream_auxiliary_layers = 1;
            p.cb_eval_user_data = &trace_streamed;
            context_ptr streamed(llama_init_from_model(model.get(), p), llama_free);
            if (!t.assert_true(stock && streamed)) return;
            llama_set_embeddings_nextn(stock.get(), true, false);
            llama_set_embeddings_nextn(streamed.get(), true, false);
            auto batch = llama_batch_init(256, 0, 1);
            struct cleanup { llama_batch & batch; ~cleanup() { llama_batch_free(batch); } } owner{batch};
            const auto submit = [&](llama_context * ctx, const std::vector<llama_token> & rows, size_t first, bool decode) {
                auto & trace = ctx == stock.get() ? trace_stock : trace_streamed;
                trace.enabled = decode; trace.stable = first;
                trace.values.clear(); trace.caches.clear();
                batch.n_tokens = int32_t(rows.size());
                for (size_t i = 0; i < rows.size(); ++i) {
                    batch.token[i] = rows[i]; batch.pos[i] = llama_pos(first+i);
                    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = decode || i+1 == rows.size();
                }
                llama_set_kv_stream_decode(ctx, decode);
                const bool result = t.assert_equal(0, llama_decode(ctx, batch));
                llama_synchronize(ctx);
                return result;
            };
            const size_t vocabulary = llama_vocab_n_tokens(vocab);
            const auto compare = [&](const char * label, size_t first, size_t rows) {
                if (!t.assert_equal(trace_stock.values.size(),trace_streamed.values.size()) ||
                        !t.assert_equal(trace_stock.caches.size(),trace_streamed.caches.size())) return;
                for (size_t i = 0; i < std::min(trace_stock.values.size(),trace_streamed.values.size()); ++i) {
                    const auto & a = trace_stock.values[i]; const auto & b = trace_streamed.values[i];
                    if (!t.assert_equal(a.first,b.first) || !t.assert_equal(a.second.size(),b.second.size())) return;
                    float error = 0;
                    for (size_t j = 0; j < a.second.size(); ++j) error=std::max(error,std::abs(a.second[j]-b.second[j]));
                    t.out << label << " pos=" << first << " tensor=" << a.first << " max=" << error << '\n';
                }
                for (size_t i = 0; i < std::min(trace_stock.caches.size(),trace_streamed.caches.size()); ++i) {
                    const auto & a = trace_stock.caches[i]; const auto & b = trace_streamed.caches[i];
                    if (!t.assert_equal(a.bytes.size(),b.bytes.size())) return;
                    const auto mismatch=std::mismatch(a.bytes.begin(),a.bytes.end(),b.bytes.begin());
                    const size_t byte=size_t(mismatch.first-a.bytes.begin());
                    t.out << label << " pos=" << first << " cache=" << a.name << " first_different_token="
                        << (byte == a.bytes.size() ? -1 : int64_t(byte/a.stride)) << '\n';
                }
                for (size_t row = 0; row < rows; ++row) {
                    const float * a = llama_get_logits_ith(stock.get(), int32_t(row)-int32_t(rows));
                    const float * b = llama_get_logits_ith(streamed.get(), int32_t(row)-int32_t(rows));
                    float error = 0;
                    for (size_t i = 0; i < vocabulary; ++i) {
                        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { t.assert_true(false); return; }
                        error = std::max(error, std::abs(a[i]-b[i]));
                    }
                    t.assert_true(error == 0);
                    t.out << label << " pos=" << first+row << " logit_max=" << error
                        << " stock_top=" << std::max_element(a,a+vocabulary)-a
                        << " streamed_top=" << std::max_element(b,b+vocabulary)-b << '\n';
                }
                recurrent_snapshot a, b;
                llama_synchronize(stock.get()); llama_synchronize(streamed.get());
                static_cast<llama_memory_hybrid *>(llama_get_memory(stock.get()))->get_mem_recr()->state_write(a,0,0);
                static_cast<llama_memory_hybrid *>(llama_get_memory(streamed.get()))->get_mem_recr()->state_write(b,0,0);
                if (!t.assert_equal(a.tensors.size(), b.tensors.size())) return;
                size_t different = 0; float worst = 0;
                for (size_t part = 0; part < a.tensors.size(); ++part) {
                    if (!t.assert_equal(a.tensors[part].size(), b.tensors[part].size())) return;
                    float error = 0;
                    for (size_t i = 0; i < a.tensors[part].size(); ++i) {
                        if (!std::isfinite(a.tensors[part][i]) || !std::isfinite(b.tensors[part][i])) { t.assert_true(false); return; }
                        error = std::max(error, std::abs(a.tensors[part][i]-b.tensors[part][i]));
                    }
                    if (error) {
                        if (!different) t.out << label << " first_state_part=" << part << " error=" << error << '\n';
                        ++different; worst = std::max(worst,error);
                    }
                }
                t.out << label << " different_state_parts=" << different << " state_max=" << worst << '\n';
                t.assert_equal(size_t(0), different);
            };
            for (size_t first = 0; first < prefill_tokens; first += 256) {
                const size_t n = std::min<size_t>(256,prefill_tokens-first);
                const std::vector<llama_token> rows(tokens.begin()+first,tokens.begin()+first+n);
                if (!submit(stock.get(),rows,first,false) || !submit(streamed.get(),rows,first,false)) return;
            }
            compare("prefill", prefill_tokens-1, 1);
            const std::vector<llama_token> pattern{264,4927,5253,383,2919,4802,11,4927,5253,383,4128,958,11,4927,5253,383};
            std::vector<llama_token> continuation(decode_tokens);
            for (size_t i = 0; i < decode_tokens; ++i) continuation[i] = pattern[i%pattern.size()];
            const size_t width = std::getenv("LLAMA_DIAG_QUERY_WIDTH") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_QUERY_WIDTH"))) : 2;
            if (!t.assert_true(width >= 1 && width <= 4)) return;
            for (size_t first = 0; first < continuation.size(); first += width) {
                const size_t count = std::min(width, continuation.size()-first);
                const auto save = [&](llama_context * ctx) {
                    std::vector<uint8_t> result(llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    t.assert_equal(result.size(),llama_state_seq_get_data_ext(ctx,result.data(),result.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    return result;
                };
                auto a = save(stock.get()), b = save(streamed.get());
                const std::vector<llama_token> rows(continuation.begin()+first,continuation.begin()+first+count);
                if (!submit(stock.get(),rows,prefill_tokens+first,true) || !submit(streamed.get(),rows,prefill_tokens+first,true)) return;
                compare("verify",prefill_tokens+first,count);
                if (first%32 == 4 || first%32 == 12) {
                    const auto restore = [&](llama_context * ctx, const std::vector<uint8_t> & state) {
                        return t.assert_equal(state.size(),llama_state_seq_set_data_ext(ctx,state.data(),state.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) &&
                            t.assert_true(llama_memory_seq_rm(llama_get_memory(ctx),0,llama_pos(prefill_tokens+first),-1));
                    };
                    if (!restore(stock.get(),a) || !restore(streamed.get(),b)) return;
                    if (!submit(stock.get(),rows,prefill_tokens+first,true) || !submit(streamed.get(),rows,prefill_tokens+first,true)) return;
                    compare("replay",prefill_tokens+first,count);
                }
            }
        });
        return t.summary();
    }
    if (mtp_memory_probe) {
        t.set_filter("native_context_serial_parent_probe");
        t.test("native_context_serial_parent_probe", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 262144;
            p.n_batch = p.n_ubatch = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = GGML_TYPE_Q8_0;
            p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_auxiliary_layers = 1;
            p.shared_device_memory_bytes = size_t(1800)*1048576;
            context_ptr target(llama_init_from_model(model.get(), p), llama_free);
            if (!t.assert_true(bool(target) && target->get_compute_memory())) return;
            t.assert_true(target->get_cparams().embeddings_nextn && !target->get_cparams().embeddings_nextn_masked);
            auto d = p;
            d.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            d.ctx_other = target.get();
            d.kv_stream_auxiliary_layers = 0;
            d.shared_device_memory_bytes = 0;
            context_ptr draft(llama_init_from_model(model.get(), d), llama_free);
            if (!t.assert_true(bool(draft) && draft->get_compute_memory())) return;
            t.assert_true(draft->get_cparams().embeddings_nextn && draft->get_cparams().embeddings_nextn_masked);
            auto * draft_kv = static_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
            const int mtp_layer = llama_model_n_layer(model.get());
            t.assert_true(ggml_backend_buffer_is_host(draft_kv->get_k_storage(mtp_layer)->buffer));
            t.assert_true(ggml_backend_buffer_is_host(draft_kv->get_v_storage(mtp_layer)->buffer));
            auto * physical = target->get_compute_memory()->shared_parent();
            if (!t.assert_true(physical != nullptr)) return;
            bool found = false;
            for (auto * lease : draft->get_compute_memory()->workspace_leases()) {
                auto * buffer = lease ? ggml_backend_memory_lease_buffer(lease) : nullptr;
                if (!buffer || ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(physical)) continue;
                ggml_backend_memory_region region{};
                if (!t.assert_true(ggml_backend_memory_lease_get_region(lease, &region))) return;
                found = true;
                t.out << "native MTP draft workspace MiB=" << double(region.size)/1048576
                    << ", target parent MiB=" << double(ggml_backend_buffer_get_size(physical))/1048576 << '\n';
                t.assert_true(ggml_backend_buffer_get_base(buffer) == ggml_backend_buffer_get_base(physical));
            }
            t.assert_true(found);
            t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
            auto target_batch = llama_batch_init(256, 0, 1);
            auto mtp_batch = llama_batch_init(256, llama_model_n_embd(model.get()), 1);
            std::vector<llama_token> mtp_tokens(256, 1);
            mtp_batch.token = mtp_tokens.data();
            struct batch_cleanup {
                llama_batch & target, & mtp;
                ~batch_cleanup() { mtp.token = nullptr; llama_batch_free(target); llama_batch_free(mtp); }
            } cleanup{target_batch, mtp_batch};
            std::fill(mtp_batch.embd, mtp_batch.embd + 256*llama_model_n_embd(model.get()), 0.0f);
            target_batch.n_tokens = mtp_batch.n_tokens = 256;
            for (int i = 0; i < 256; ++i) {
                target_batch.token[i] = mtp_batch.token[i] = 1;
                target_batch.pos[i] = mtp_batch.pos[i] = i;
                target_batch.n_seq_id[i] = mtp_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = mtp_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = mtp_batch.logits[i] = i == 255;
            }
            llama_set_kv_stream_decode(target.get(), false);
            if (!t.assert_equal(0, llama_decode(target.get(), target_batch))) return;
            llama_context_memory_diagnostics before{};
            if (!t.assert_true(target->get_compute_memory()->diagnostics(before))) return;
            if (!t.assert_equal(0, llama_decode(draft.get(), mtp_batch))) return;
            target_batch.n_tokens = 1;
            target_batch.token[0] = 1;
            target_batch.pos[0] = 256;
            target_batch.logits[0] = 1;
            llama_set_kv_stream_decode(target.get(), true);
            if (!t.assert_equal(0, llama_decode(target.get(), target_batch))) return;
            llama_context_memory_diagnostics after{};
            if (!t.assert_true(target->get_compute_memory()->diagnostics(after))) return;
            t.assert_true(after.kv_pool_bytes > before.kv_pool_bytes);
            if (!t.assert_true(llama_kv_stream_mtp_prepare(target.get(), 0))) return;
            mtp_batch.n_tokens = 1;
            mtp_batch.token[0] = 1;
            mtp_batch.pos[0] = 256;
            mtp_batch.logits[0] = 1;
            if (!t.assert_equal(0, llama_decode(draft.get(), mtp_batch))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
            t.assert_equal(size_t(257), hybrid->get_mem_attn()->get_kv_stream()->tokens());
            t.assert_equal(size_t(257), hybrid->get_mem_attn()->get_kv_stream()->auxiliary_cache()->tokens());
            auto * stream=hybrid->get_mem_attn()->get_kv_stream();
            t.assert_true(stream->has_mtp_layer());
            target_batch.pos[0]=257;
            if (!t.assert_equal(0,llama_decode(target.get(),target_batch))) return;
            t.assert_true(!stream->has_mtp_layer());
            const bool target_removed = llama_memory_seq_rm(llama_get_memory(target.get()), 0, 0, -1);
            t.out << "native target full remove=" << target_removed << '\n';
            t.assert_true(target_removed);
            const bool draft_removed = llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1);
            t.out << "native draft full remove=" << draft_removed << '\n';
            t.assert_true(draft_removed);
        });
        return t.summary();
    }
    const char * auxiliary_test_name = embedded_mtp_control ?
        "target_and_mtp_contexts_share_auxiliary_identity" :
        "target_context_owns_distinct_auxiliary_mtp_host_cache";
    if (auxiliary_control || embedded_mtp_control || target_tg3_control || target_stream_tg4_control) t.set_filter(auxiliary_test_name);
    const auto * vocab = llama_model_get_vocab(model.get());
    std::string text;
    for (int i = 0; i < 200; ++i) text += "The capital of France is Paris. We are testing a serial language model with a bounded key and value cache. ";
    const int size = -llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,true,false);
    std::vector<llama_token> prompt(size);
    const int tokens = llama_tokenize(vocab,text.data(),int(text.size()),prompt.data(),size,true,false);
    if (!t.assert_true(tokens >= (target_stream_tg4_control ? 3005 : 640))) return t.summary();
    prompt.resize(target_stream_tg4_control ? 3005 : 640);
    if (resume_control) {
        for (bool streamed : {false,true}) {
            if (streamed && resume_resident_only) continue;
            for (auto phase : {llama_memory_text_phase::decode,llama_memory_text_phase::prefill}) {
                const auto name = std::string(streamed ? "streamed" : "resident") +
                    "_hybrid_resume_matches_uninterrupted_" + (phase == llama_memory_text_phase::decode ? "decode" : "prefill");
                t.test(name, [&](testing & t) {
                    auto p = llama_context_default_params();
                    p.n_ctx = streamed ? 8192 : 1024;
                    p.n_batch = p.n_ubatch = streamed ? 64 : 256;
                    p.n_seq_max = 1; p.n_rs_seq = 0;
                    p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
                    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                    p.kv_stream_pool_bytes = 16*1048576;
                    context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
                    if (!t.assert_true(bool(ctx))) return;
                    std::vector<llama_token> history(streamed ? 6144 : 256);
                    for (size_t i = 0; i < history.size(); ++i) history[i] = prompt[i%prompt.size()];
                    llama_set_kv_stream_decode(ctx.get(),false);
                    for (size_t first = 0; first < history.size(); first += p.n_ubatch)
                        if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(history.data()+first,p.n_ubatch)))) return;
                    const auto checkpoint = [&] {
                        std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
                        data.resize(llama_state_get_data(ctx.get(),data.data(),data.size())); return data;
                    };
                    const auto prefill_state = checkpoint();
                    llama_set_kv_stream_decode(ctx.get(),true);
                    for (size_t i = 0; i < 4; ++i)
                        if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
                    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
                    auto * stream = hybrid->get_mem_attn()->get_kv_stream();
                    auto snapshot = [&] {
                        std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
                        data.resize(llama_state_get_data(ctx.get(),data.data(),data.size())); return data;
                    };
                    const auto saved = snapshot();
                    const auto initial_frontier = stream->tokens();
                    const auto host = stream->host();
                    const auto consumer = stream->memory_consumer();
                    const auto next = [&]() {
                        llama_set_kv_stream_decode(ctx.get(),phase == llama_memory_text_phase::decode);
                        const int32_t rows = phase == llama_memory_text_phase::decode ? 1 : 3;
                        return llama_decode(ctx.get(),llama_batch_get_one(prompt.data()+4,rows)) == 0;
                    };
                    const auto generate = [&]() {
                        std::vector<llama_token> generated;
                        llama_set_kv_stream_decode(ctx.get(),true);
                        for (size_t i = 0; i < 16; ++i) {
                            const auto * logits = llama_get_logits_ith(ctx.get(),-1);
                            auto token = llama_token(std::max_element(logits,logits+llama_vocab_n_tokens(vocab))-logits);
                            generated.push_back(token);
                            if (llama_decode(ctx.get(),llama_batch_get_one(&token,1)) != 0) { generated.clear(); break; }
                        }
                        return generated;
                    };
                    if (!t.assert_true(next())) return;
                    std::vector<float> expected_next(llama_get_logits_ith(ctx.get(),-1),
                        llama_get_logits_ith(ctx.get(),-1)+llama_vocab_n_tokens(vocab));
                    const auto expected_tokens = generate();
                    if (!t.assert_equal(size_t(16),expected_tokens.size())) return;
                    recurrent_snapshot expected_recurrent;
                    hybrid->get_mem_recr()->state_write(expected_recurrent,0,0);
                    std::vector<float> expected_logits(llama_get_logits_ith(ctx.get(),-1),
                        llama_get_logits_ith(ctx.get(),-1)+llama_vocab_n_tokens(vocab));
                    if (!t.assert_equal(saved.size(),llama_state_set_data(ctx.get(),saved.data(),saved.size()))) return;
                    if (!t.assert_true(next())) return;
                    const auto control_tokens = generate();
                    t.out << "checkpoint-only matching tokens=" << (control_tokens == expected_tokens) << '\n';
                    if (!t.assert_true(control_tokens == expected_tokens) ||
                            !t.assert_equal(prefill_state.size(),llama_state_set_data(ctx.get(),prefill_state.data(),prefill_state.size()))) return;
                    llama_set_kv_stream_decode(ctx.get(),true);
                    for (size_t i = 0; i < 4; ++i)
                        if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
                    t.assert_true(snapshot() == saved);
                    if (!streamed) {
#ifdef KV_STREAM_TILE_NATIVE_TEST
                        if (kv_stream_test_native_graphs_enabled()) t.assert_true(stream->captured_layers() > 0);
                        else t.assert_equal(size_t(0),stream->captured_layers());
#endif
                    }
                    const auto revision = stream->binding_view().revision;
                    if (!t.assert_true(llama_context_suspend_kv_device(ctx.get()))) return;
                    t.assert_equal(size_t(0),stream->captured_layers());
                    if (!t.assert_true(llama_context_resume_kv_device(ctx.get(),phase))) return;
                    t.assert_true(stream->host() == host && stream->memory_consumer() == consumer);
                    t.assert_equal(initial_frontier,stream->tokens());
                    t.assert_true(stream->binding_view().revision > revision);
                    t.assert_equal(size_t(0),stream->captured_layers());
                    t.assert_true(snapshot() == saved);
                    if (!t.assert_true(next())) return;
                    float next_error = 0;
                    const auto * next_logits = llama_get_logits_ith(ctx.get(),-1);
                    for (size_t i = 0; i < expected_next.size(); ++i) next_error = std::max(next_error,std::abs(next_logits[i]-expected_next[i]));
                    t.out << "resume first-batch error=" << next_error << '\n';
                    t.assert_equal(0.0f,next_error);
                    const auto actual_tokens = generate();
                    t.assert_true(actual_tokens == expected_tokens);
                    const auto * actual_logits = llama_get_logits_ith(ctx.get(),-1);
                    float maximum = 0;
                    for (size_t i = 0; i < expected_logits.size(); ++i) maximum = std::max(maximum,std::abs(actual_logits[i]-expected_logits[i]));
                    t.out << "resume max_logit_error=" << maximum << ", matching tokens=" << (actual_tokens == expected_tokens) << '\n';
                    t.assert_equal(0.0f,maximum);
                    recurrent_snapshot actual_recurrent;
                    hybrid->get_mem_recr()->state_write(actual_recurrent,0,0);
                    t.assert_true(expected_recurrent.metadata == actual_recurrent.metadata && expected_recurrent.tensors == actual_recurrent.tensors);
                });
            }
        }
        return t.summary();
    }
    if (suspend_control) {
        t.test("populated_hybrid_state_survives_device_suspension", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 1024; p.n_batch = p.n_ubatch = 256;
            p.n_seq_max = 1; p.n_rs_seq = 0;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = 16*1048576;
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            llama_set_kv_stream_decode(ctx.get(),false);
            if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(prompt.data(),256)))) return;
            llama_set_kv_stream_decode(ctx.get(),true);
            for (size_t i = 256; i < 260; ++i)
                if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            auto * stream = hybrid->get_mem_attn()->get_kv_stream();
            if (!t.assert_true(stream && stream->complete())) return;
            auto snapshot = [&] {
                std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
                const size_t written = llama_state_get_data(ctx.get(),data.data(),data.size());
                data.resize(written); return data;
            };
            const auto before = snapshot();
            recurrent_snapshot recurrent;
            hybrid->get_mem_recr()->state_write(recurrent,0,0);
            const auto host = stream->host();
            const auto frontier = llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0);
            if (!t.assert_true(llama_context_suspend_kv_device(ctx.get()))) return;
            t.assert_true(stream->host() == host && stream->device_suspended());
            t.assert_equal(size_t(260),stream->tokens());
            t.assert_equal(size_t(0),stream->device_grant_bytes());
            t.assert_true(snapshot() == before);
            recurrent_snapshot after;
            hybrid->get_mem_recr()->state_write(after,0,0);
            t.assert_true(recurrent.metadata == after.metadata && recurrent.tensors == after.tensors);
            t.assert_equal(frontier,llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0));
            t.assert_true(llama_decode(ctx.get(),llama_batch_get_one(&prompt[260],1)) != 0);
            t.assert_true(snapshot() == before);
            t.assert_true(llama_context_suspend_kv_device(ctx.get()));
        });
        if (!suspend_resident_only) t.test("streamed_hybrid_suspension_retires_cross_token_prefetch", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 8192; p.n_batch = p.n_ubatch = 64;
            p.n_seq_max = 1; p.n_rs_seq = 0;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = 16*1048576;
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            std::vector<llama_token> history(6144);
            for (size_t i = 0; i < history.size(); ++i) history[i] = prompt[i%prompt.size()];
            llama_set_kv_stream_decode(ctx.get(),false);
            for (size_t first = 0; first < history.size(); first += 64)
                if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(history.data()+first,64)))) return;
            llama_set_kv_stream_decode(ctx.get(),true);
            for (size_t i = 0; i < 4; ++i)
                if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            auto * stream = hybrid->get_mem_attn()->get_kv_stream();
            llama_kv_stream_runtime_diagnostics before;
            if (!t.assert_true(stream && stream->runtime_diagnostics(before) && before.streaming_active)) return;
            t.assert_true(stream->prefetch_primed());
            const auto host = stream->host();
            const auto consumer = stream->memory_consumer();
            std::vector<uint8_t> bytes(host->bytes());
            std::memcpy(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size());
            recurrent_snapshot recurrent;
            hybrid->get_mem_recr()->state_write(recurrent,0,0);
            if (!t.assert_true(llama_context_suspend_kv_device(ctx.get()))) return;
            t.assert_true(stream->device_suspended() && !stream->prefetch_primed());
            t.assert_equal(size_t(6148),stream->tokens());
            t.assert_true(stream->memory_consumer() == consumer && stream->host() == host);
            t.assert_equal(size_t(0),stream->device_grant_bytes());
            t.assert_true(std::memcmp(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size()) == 0);
            recurrent_snapshot after;
            hybrid->get_mem_recr()->state_write(after,0,0);
            t.assert_true(recurrent.metadata == after.metadata && recurrent.tensors == after.tensors);
            t.assert_true(llama_decode(ctx.get(),llama_batch_get_one(&prompt[4],1)) != 0);
        });
        return t.summary();
    }
    if (mtp_sustained_control) {
        t.set_filter("mtp_serial_requests_change_prefill_and_reset_prompt_cache");
        t.test("mtp_serial_requests_change_prefill_and_reset_prompt_cache", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 1024;
            p.n_batch = p.n_ubatch = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = GGML_TYPE_Q8_0;
            p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = 16*1048576;
            p.kv_stream_auxiliary_layers = 1;
            p.n_rs_seq = mtp_sustained_rs_control ? 3 : 0;
            context_ptr target_ctx(llama_init_from_model(model.get(), p), llama_free);
            if (!t.assert_true(bool(target_ctx))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target_ctx.get()));
            auto * stream = hybrid->get_mem_attn()->get_kv_stream();
            if (!t.assert_true(stream != nullptr)) return;
            auto cache = stream->auxiliary_cache();
            if (!t.assert_true(bool(cache))) return;
            auto draft_params = p;
            draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            draft_params.ctx_other = target_ctx.get();
            draft_params.kv_stream_pool_bytes = 0;
            draft_params.kv_stream_auxiliary_layers = 0;
            draft_params.n_rs_seq = 0;
            context_ptr draft_ctx(llama_init_from_model(model.get(), draft_params), llama_free);
            if (!t.assert_true(bool(draft_ctx))) return;
            draft_params.ctx_other = nullptr;
            context_ptr stock_ctx(llama_init_from_model(model.get(), draft_params), llama_free);
            if (!t.assert_true(bool(stock_ctx))) return;
            auto target_batch = llama_batch_init(256, 0, 1);
            auto mtp_batch = llama_batch_init(256, llama_model_n_embd(model.get()), 1);
            std::vector<llama_token> mtp_tokens(256, prompt.front());
            mtp_batch.token = mtp_tokens.data();
            struct batch_cleanup {
                llama_batch & target, & mtp;
                ~batch_cleanup() { mtp.token = nullptr; llama_batch_free(target); llama_batch_free(mtp); }
            } cleanup{target_batch, mtp_batch};
            std::fill(mtp_batch.embd, mtp_batch.embd + 256*llama_model_n_embd(model.get()), 0.0f);
            const size_t vocab_size = llama_vocab_n_tokens(vocab);
            size_t settled_free = 0;
            float worst = 0;
            for (size_t cycle = 0; cycle < 24; ++cycle) {
                const size_t prefill = std::array<size_t, 8>{32, 128, 256, 64, 192, 32, 256, 96}[cycle % 8];
                if (cycle) {
                    llama_synchronize(stock_ctx.get());
                    llama_synchronize(draft_ctx.get());
                    llama_synchronize(target_ctx.get());
                    llama_memory_clear(llama_get_memory(stock_ctx.get()), true);
                    llama_memory_clear(llama_get_memory(draft_ctx.get()), true);
                    llama_memory_clear(llama_get_memory(target_ctx.get()), true);
                    if (!t.assert_equal(size_t(0), stream->tokens()) ||
                            !t.assert_equal(size_t(0), cache->tokens()) ||
                            !t.assert_true(!stream->has_mtp_layer())) return;
                }
                target_batch.n_tokens = mtp_batch.n_tokens = int32_t(prefill);
                for (size_t i = 0; i < prefill; ++i) {
                    const auto token = prompt[(cycle*37 + i)%prompt.size()];
                    target_batch.token[i] = mtp_tokens[i] = token;
                    target_batch.pos[i] = mtp_batch.pos[i] = llama_pos(i);
                    target_batch.n_seq_id[i] = mtp_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = mtp_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = mtp_batch.logits[i] = i + 1 == prefill;
                }
                llama_set_kv_stream_decode(target_ctx.get(), false);
                if (!t.assert_equal(0, llama_decode(target_ctx.get(), target_batch)) ||
                        !t.assert_equal(0, llama_decode(draft_ctx.get(), mtp_batch)) ||
                        !t.assert_equal(0, llama_decode(stock_ctx.get(), mtp_batch))) return;
                if (!t.assert_equal(prefill, stream->tokens()) ||
                        !t.assert_equal(prefill, cache->tokens())) return;
                target_batch.n_tokens = mtp_batch.n_tokens = 1;
                target_batch.token[0] = mtp_tokens[0] = prompt[(cycle*53 + prefill)%prompt.size()];
                target_batch.pos[0] = mtp_batch.pos[0] = llama_pos(prefill);
                target_batch.logits[0] = mtp_batch.logits[0] = 1;
                llama_set_kv_stream_decode(target_ctx.get(), true);
                if (!t.assert_equal(0, llama_decode(target_ctx.get(), target_batch)) ||
                        !t.assert_true(llama_kv_stream_mtp_prepare(target_ctx.get(), 1))) return;
                for (int attempt = 0; attempt < 2; ++attempt) {
                    if (attempt) {
                        if (!t.assert_true(llama_memory_seq_rm(
                                llama_get_memory(draft_ctx.get()), 0, llama_pos(prefill), -1) &&
                                llama_memory_seq_rm(
                                llama_get_memory(stock_ctx.get()), 0, llama_pos(prefill), -1))) return;
                        if (!t.assert_equal(prefill, cache->tokens())) return;
                        mtp_tokens[0] = prompt[(cycle*53 + prefill + 1)%prompt.size()];
                    }
                    if (!t.assert_equal(0, llama_decode(draft_ctx.get(), mtp_batch)) ||
                            !t.assert_equal(0, llama_decode(stock_ctx.get(), mtp_batch))) return;
                    llama_synchronize(draft_ctx.get());
                    llama_synchronize(stock_ctx.get());
                    const float * actual = llama_get_logits_ith(draft_ctx.get(), -1);
                    const float * expected = llama_get_logits_ith(stock_ctx.get(), -1);
                    if (!t.assert_true(actual && expected)) return;
                    float error = 0;
                    for (size_t i = 0; i < vocab_size; ++i) {
                        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
                            t.assert_true(false); return;
                        }
                        error = std::max(error, std::abs(actual[i] - expected[i]));
                    }
                    worst = std::max(worst, error);
                    t.assert_true(error < .05f);
                    t.assert_equal(
                        std::max_element(actual, actual + vocab_size) - actual,
                        std::max_element(expected, expected + vocab_size) - expected);
                    t.assert_equal(prefill + 1, cache->tokens());
                    t.assert_true(stream->has_mtp_layer());
                }
                auto * device = ggml_backend_dev_by_name("CUDA0");
                size_t free_bytes = 0, total_bytes = 0;
                if (device) ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
                if (cycle == 7) settled_free = free_bytes;
                if (cycle > 7 && device) t.assert_true(free_bytes + 16*1048576 >= settled_free);
                t.out << "MTP serial cycle " << cycle << " prefill=" << prefill
                    << " free VRAM MiB=" << free_bytes/1048576 << '\n';
            }
            t.out << "MTP serial max stock-logit error=" << worst << '\n';
        });
        return t.summary();
    }
    t.test("invalid_streaming_context_modes_fail_before_execution", [&](testing & t) {
        for (int mode = 0; mode < 6; ++mode) {
            auto p = llama_context_default_params(); p.kv_stream_pool_bytes = 16*1048576; p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            if (mode == 0) p.n_seq_max = 2;
            if (mode == 1) p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            if (mode == 2) p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            if (mode == 3) p.offload_kqv = false;
            if (mode == 4) p.n_rs_seq = 1;
            if (mode == 5) p.shared_device_memory_bytes = 640*1048576;
            context_ptr context(llama_init_from_model(model.get(),p),llama_free); t.assert_true(!context);
        }
    });
    if (auxiliary_control || embedded_mtp_control || target_tg3_control || target_stream_tg4_control) t.test(auxiliary_test_name, [&](testing & t) {
        auto p = llama_context_default_params();
        p.n_ctx = target_tg3_control ? 8192 : target_stream_tg4_control ? 4096 : 1024; p.n_batch = p.n_ubatch = 256;
        if (target_stream_tg4_control) p.n_outputs_max = p.n_outputs_max_per_seq = 4;
        p.n_threads = p.n_threads_batch = 8;
        p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        p.kv_stream_pool_bytes = 16*1048576;
        if (vision_mtp_handoff) {
            p.type_k = test_cache_k;
            p.type_v = test_cache_v;
            p.n_ctx = 8192;
            p.kv_stream_pool_bytes = 0;
            p.shared_device_memory_bytes = 640*1048576;
        }
        if (target_tg3_control) {
            p.kv_stream_pool_bytes = 0;
            p.shared_device_memory_bytes = 1800*1048576;
        }
        p.kv_stream_auxiliary_layers = 1;
        context_ptr context(llama_init_from_model(model.get(), p), llama_free);
        if (!t.assert_true(bool(context))) return;
        auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
        auto * target = hybrid->get_mem_attn()->get_kv_stream();
        if (!t.assert_true(target != nullptr)) return;
        auto mtp = target->auxiliary_cache();
        if (!t.assert_true(bool(mtp))) return;
        t.assert_true(target->host()->cache_id() != mtp->host()->cache_id());
        t.assert_equal(uint32_t(1), mtp->host()->config().layers);
        t.assert_equal(size_t(0), mtp->tokens());
        const auto view = target->binding_view();
        t.assert_equal(target->host()->config().layers + 1, view.config.layers);
        if (!t.assert_equal(size_t(2), view.config.caches.size())) return;
        t.assert_equal(target->host()->cache_id(), view.config.caches[0].id);
        t.assert_equal(mtp->host()->cache_id(), view.config.caches[1].id);
        const auto breakdown = hybrid->get_mem_attn()->memory_breakdown();
        const auto host_type = ggml_backend_buffer_get_type(target->host()->buffer());
        t.assert_equal(target->host()->bytes() + mtp->host()->bytes(), breakdown.at(host_type));
        llama_token token = prompt.front();
        llama_set_kv_stream_decode(context.get(), false);
        t.assert_equal(0, llama_decode(context.get(), llama_batch_get_one(&token, 1)));
        t.assert_equal(size_t(1), target->tokens());
        t.assert_equal(size_t(0), mtp->tokens());
        if (!embedded_mtp_control && !target_tg3_control && !target_stream_tg4_control) return;
        auto target_batch = llama_batch_init(256, 0, 1);
        struct target_batch_guard {
            llama_batch & batch;
            ~target_batch_guard() { llama_batch_free(batch); }
        } target_batch_owner{target_batch};
        target_batch.n_tokens = 256;
        for (int i = 0; i < 256; ++i) {
            target_batch.token[i] = prompt[size_t(i + 1)];
            target_batch.pos[i] = i + 1;
            target_batch.n_seq_id[i] = 1;
            target_batch.seq_id[i][0] = 0;
            target_batch.logits[i] = i == 255;
        }
        if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
        if (target_stream_tg4_control) {
            for (size_t first = 257; first < 3001; first += 256) {
                const size_t count = std::min<size_t>(256, 3001 - first);
                target_batch.n_tokens = int(count);
                for (size_t i = 0; i < count; ++i) {
                    target_batch.token[i] = prompt[first + i];
                    target_batch.pos[i] = llama_pos(first + i);
                    target_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = i + 1 == count;
                }
                llama_set_kv_stream_decode(context.get(), false);
                if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            }
            t.assert_equal(size_t(3001), target->tokens());
            target_batch.n_tokens = 4;
            for (int i = 0; i < 4; ++i) {
                target_batch.token[i] = prompt[size_t(3001 + i)];
                target_batch.pos[i] = 3001 + i;
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i == 3;
            }
            llama_set_kv_stream_decode(context.get(), true);
            if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            t.assert_equal(size_t(3005), target->tokens());
            llama_kv_stream_runtime_diagnostics diagnostics;
            if (t.assert_true(target->runtime_diagnostics(diagnostics)))
                t.assert_true(diagnostics.streaming_active);
            llama_synchronize(context.get());
            const float * streamed_logits = llama_get_logits_ith(context.get(), -1);
            const size_t vocab_size = llama_vocab_n_tokens(vocab);
            if (!t.assert_true(streamed_logits != nullptr)) return;
            const std::vector<float> streamed(streamed_logits, streamed_logits + vocab_size);
            auto stock_params = p;
            stock_params.kv_stream_pool_bytes = 0;
            stock_params.kv_stream_auxiliary_layers = 0;
            context_ptr stock(llama_init_from_model(model.get(), stock_params), llama_free);
            if (!t.assert_true(bool(stock))) return;
            target_batch.n_tokens = 1;
            target_batch.token[0] = prompt[0];
            target_batch.pos[0] = 0;
            target_batch.n_seq_id[0] = 1;
            target_batch.seq_id[0][0] = 0;
            target_batch.logits[0] = 1;
            if (!t.assert_equal(0, llama_decode(stock.get(), target_batch))) return;
            for (size_t first = 1; first < 3001; first += 256) {
                const size_t count = std::min<size_t>(256, 3001 - first);
                target_batch.n_tokens = int(count);
                for (size_t i = 0; i < count; ++i) {
                    target_batch.token[i] = prompt[first + i];
                    target_batch.pos[i] = llama_pos(first + i);
                    target_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = i + 1 == count;
                }
                if (!t.assert_equal(0, llama_decode(stock.get(), target_batch))) return;
            }
            target_batch.n_tokens = 4;
            for (int i = 0; i < 4; ++i) {
                target_batch.token[i] = prompt[size_t(3001 + i)];
                target_batch.pos[i] = 3001 + i;
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i == 3;
            }
            if (!t.assert_equal(0, llama_decode(stock.get(), target_batch))) return;
            llama_synchronize(stock.get());
            const float * expected = llama_get_logits_ith(stock.get(), -1);
            if (!t.assert_true(expected != nullptr)) return;
            float worst = 0;
            for (size_t i = 0; i < vocab_size; ++i)
                worst = std::max(worst, std::abs(streamed[i] - expected[i]));
            t.out << "streamed TG4 at 3005 tokens max logit error=" << worst << '\n';
            t.assert_true(worst < .05f);
            t.assert_equal(
                std::max_element(streamed.begin(), streamed.end()) - streamed.begin(),
                std::max_element(expected, expected + vocab_size) - expected);
            return;
        }
            context_ptr stock_draft(nullptr, llama_free);
            if (target_with_stock_draft) {
                auto draft_only = p;
                draft_only.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                draft_only.ctx_other = nullptr;
                draft_only.kv_stream_pool_bytes = 0;
                draft_only.kv_stream_auxiliary_layers = 0;
                stock_draft.reset(llama_init_from_model(model.get(), draft_only));
                if (!t.assert_true(bool(stock_draft))) return;
            }
        if (target_tg3_control) {
            for (size_t first = 257; first < 7933; first += 256) {
                const size_t count = std::min<size_t>(256, 7933 - first);
                target_batch.n_tokens = int(count);
                for (size_t i = 0; i < count; ++i) {
                    target_batch.token[i] = prompt[(first + i) % prompt.size()];
                    target_batch.pos[i] = llama_pos(first + i);
                    target_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = i + 1 == count;
                }
                llama_set_kv_stream_decode(context.get(), false);
                if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            }
            target_batch.n_tokens = 3;
            for (int i = 0; i < 3; ++i) {
                target_batch.token[i] = prompt[size_t(7933 + i) % prompt.size()];
                target_batch.pos[i] = 7933 + i;
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i == 2;
            }
            llama_set_kv_stream_decode(context.get(), true);
            t.assert_equal(0, llama_decode(context.get(), target_batch));
            return;
        }
        t.assert_equal(size_t(257), target->tokens());
        if (vision_mtp_handoff) t.assert_true(!context->get_compute_memory()->can_suspend_for_vision());
        auto draft_params = p;
        draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        draft_params.ctx_other = context.get();
        draft_params.kv_stream_pool_bytes = 0;
        draft_params.shared_device_memory_bytes = 0;
        draft_params.kv_stream_auxiliary_layers = 0;
        context_ptr draft(llama_init_from_model(model.get(), draft_params), llama_free);
        if (!t.assert_true(bool(draft))) return;
        if (vision_mtp_handoff) {
            t.assert_true("attached draft permits vision", context->get_compute_memory()->can_suspend_for_vision());
            t.assert_true("draft cannot lend target parent", !draft->get_compute_memory()->can_suspend_for_vision());
        }
        auto ordinary_params = draft_params;
        ordinary_params.ctx_other = nullptr;
        context_ptr ordinary_draft(llama_init_from_model(model.get(), ordinary_params), llama_free);
        if (!t.assert_true(bool(ordinary_draft))) return;
        auto * ordinary_kv = static_cast<llama_kv_cache *>(llama_get_memory(ordinary_draft.get()));
        t.assert_true(ordinary_kv->mtp_auxiliary_cache() == nullptr);
        auto * draft_kv = static_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
        t.assert_true(draft_kv->get_kv_stream() == nullptr);
        t.assert_true(draft_kv->mtp_auxiliary_cache() == mtp);
        t.assert_equal(size_t(0), draft_kv->mtp_auxiliary_cache()->tokens());
        t.assert_equal(mtp->host()->cache_id(), draft_kv->mtp_auxiliary_cache()->host()->cache_id());
        auto mtp_batch = llama_batch_init(257, llama_model_n_embd(model.get()), 1);
        std::vector<llama_token> mtp_tokens(257, prompt.front());
        mtp_batch.token = mtp_tokens.data();
        struct mtp_batch_guard {
            llama_batch & batch;
            ~mtp_batch_guard() { batch.token = nullptr; llama_batch_free(batch); }
        } mtp_batch_owner{mtp_batch};
        std::fill(mtp_batch.embd, mtp_batch.embd + 257*llama_model_n_embd(model.get()), 0.0f);
        const auto append_mtp = [&](int first, int count) {
            mtp_batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                mtp_batch.pos[i] = first + i;
                mtp_batch.n_seq_id[i] = 1;
                mtp_batch.seq_id[i][0] = 0;
                mtp_batch.logits[i] = i + 1 == count;
            }
            const int attached_status = llama_decode(draft.get(), mtp_batch);
            const int stock_status = llama_decode(ordinary_draft.get(), mtp_batch);
            return attached_status == 0 && stock_status == 0;
        };
        if (!t.assert_true(append_mtp(0, 256))) return;
        t.assert_equal(size_t(256), mtp->tokens());
        t.assert_equal(size_t(0), mtp->frontiers().device);
        if (vision_mtp_handoff) {
            auto * owner = llama_context_compute_memory(context.get());
            if (!t.assert_true(owner && owner->text_phase().phase == llama_memory_text_phase::prefill)) return;
            if (!t.assert_true(llama_kv_stream_mtp_prepare(context.get(), 0))) return;
            if (!t.assert_true(owner->text_phase().phase == llama_memory_text_phase::decode)) return;
        } else if (!t.assert_true(target->acquire_mtp_layer())) return;
        const auto initial_upload = target->mtp_layer_population();
        t.assert_true(initial_upload.bytes > 0);
        if (!t.assert_true(append_mtp(256, 1))) return;
        t.assert_equal(size_t(257), mtp->tokens());
        t.assert_equal(size_t(1), draft_kv->mtp_span_attention_calls());
        t.assert_equal(size_t(0), mtp->frontiers().device);
        const auto & tail_layout = mtp->host()->layout();
        t.assert_true(target->has_mtp_layer());
        t.assert_equal(initial_upload.bytes + tail_layout.k_token_bytes +
            tail_layout.v_token_bytes, target->mtp_layer_population().bytes);
        ggml_kv_stream_span_plan_view advanced;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                target->mtp_layer_plan(1), advanced))) return;
        t.assert_equal(size_t(257), advanced.active_tokens);
        llama_synchronize(draft.get());
        llama_synchronize(ordinary_draft.get());
        const float * attached_logits = llama_get_logits_ith(draft.get(), -1);
        const float * stock_logits = llama_get_logits_ith(ordinary_draft.get(), -1);
        if (!t.assert_true(attached_logits != nullptr && stock_logits != nullptr)) return;
        const size_t vocab_size = llama_vocab_n_tokens(vocab);
        float maximum_logit_error = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!std::isfinite(attached_logits[i]) || !std::isfinite(stock_logits[i])) {
                t.assert_true(false);
                return;
            }
            maximum_logit_error = std::max(maximum_logit_error,
                std::abs(attached_logits[i] - stock_logits[i]));
        }
        t.out << "real MTP TG1 max logit error=" << maximum_logit_error << '\n';
        t.assert_true(maximum_logit_error < .05f);
        t.assert_equal(
            std::max_element(attached_logits, attached_logits + vocab_size) - attached_logits,
            std::max_element(stock_logits, stock_logits + vocab_size) - stock_logits);
        llama_kv_stream_host_layer host_rows;
        if (!t.assert_true(mtp->host()->layer(0, host_rows))) return;
        const auto & kv_layout = mtp->host()->layout();
        const int mtp_layer = llama_model_n_layer(model.get());
        std::vector<uint8_t> stock_k(257*kv_layout.k_token_bytes);
        std::vector<uint8_t> stock_v(257*kv_layout.v_token_bytes);
        ggml_backend_tensor_get(draft_kv->get_k_storage(mtp_layer), stock_k.data(), 0, stock_k.size());
        ggml_backend_tensor_get(draft_kv->get_v_storage(mtp_layer), stock_v.data(), 0, stock_v.size());
        t.assert_true(std::memcmp(stock_k.data(), host_rows.k, stock_k.size()) == 0);
        t.assert_true(std::memcmp(stock_v.data(), host_rows.v, stock_v.size()) == 0);
        const auto prior_generation = mtp->identity().generation;
        const std::vector<uint8_t> prior_k(static_cast<const uint8_t *>(host_rows.k),
                static_cast<const uint8_t *>(host_rows.k) + stock_k.size());
        const std::vector<uint8_t> prior_v(static_cast<const uint8_t *>(host_rows.v),
                static_cast<const uint8_t *>(host_rows.v) + stock_v.size());
        const auto before_rewrite = target->mtp_layer_population();
        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(ordinary_draft.get()), 0, 256, -1))) return;
        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 256, -1))) return;
        t.assert_true(target->has_mtp_layer());
        t.assert_equal(before_rewrite.bytes, target->mtp_layer_population().bytes);
        ggml_kv_stream_span_plan_view rewritten_prefix;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                target->mtp_layer_plan(1), rewritten_prefix))) return;
        t.assert_equal(size_t(256), rewritten_prefix.active_tokens);

        mtp_tokens[0] = prompt[1];
        if (!t.assert_true(append_mtp(256, 1))) return;
        t.assert_equal(size_t(257), mtp->tokens());
        t.assert_true(mtp->identity().generation > prior_generation);
        t.assert_equal(size_t(2), draft_kv->mtp_span_attention_calls());
        t.assert_true(std::memcmp(prior_k.data(), host_rows.k, 256*kv_layout.k_token_bytes) == 0);
        t.assert_true(std::memcmp(prior_v.data(), host_rows.v, 256*kv_layout.v_token_bytes) == 0);
        t.assert_true(std::memcmp(prior_k.data() + 256*kv_layout.k_token_bytes,
                static_cast<const uint8_t *>(host_rows.k) + 256*kv_layout.k_token_bytes,
                kv_layout.k_token_bytes) != 0);
        t.assert_true(target->has_mtp_layer());
        t.assert_equal(before_rewrite.bytes + kv_layout.k_token_bytes +
            kv_layout.v_token_bytes, target->mtp_layer_population().bytes);
        size_t verified = 257;
        for (uint32_t width : {2u, 3u, 4u}) {
            target_batch.n_tokens = int(width);
            for (uint32_t i = 0; i < width; ++i) {
                target_batch.token[i] = prompt[verified + i];
                target_batch.pos[i] = llama_pos(verified + i);
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i + 1 == width;
            }
            llama_set_kv_stream_decode(context.get(), true);
            t.out << "real target TG" << width << " verify begin\n";
            if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            if (width == 2) t.assert_true(!target->has_mtp_layer());
            t.assert_equal(verified + width, target->tokens());
            if (!t.assert_true(llama_kv_stream_mtp_prepare(context.get(), width == 4 ? 4 : 0))) return;
            const auto before_tail = target->mtp_layer_population();
            mtp_tokens[0] = prompt.front();
            t.out << "real MTP TG" << width << " catch-up begin\n";
            if (!t.assert_true(append_mtp(int(verified), int(width)))) return;
            verified += width;
            t.assert_equal(verified, mtp->tokens());
            t.assert_equal(size_t(width + 1), draft_kv->mtp_span_attention_calls());
            t.assert_equal(before_tail.bytes + width*(kv_layout.k_token_bytes +
                kv_layout.v_token_bytes), target->mtp_layer_population().bytes);
            llama_synchronize(draft.get());
            llama_synchronize(ordinary_draft.get());
            const float * actual = llama_get_logits_ith(draft.get(), -1);
            const float * expected = llama_get_logits_ith(ordinary_draft.get(), -1);
            if (!t.assert_true(actual && expected)) return;
            float worst = 0;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
                    t.assert_true(false);
                    return;
                }
                worst = std::max(worst, std::abs(actual[i] - expected[i]));
            }
            t.out << "real MTP TG" << width << " max logit error=" << worst << '\n';
            t.assert_true(worst < .05f);
            t.assert_equal(
                std::max_element(actual, actual + vocab_size) - actual,
                std::max_element(expected, expected + vocab_size) - expected);
            if (width == 4) {
                t.assert_equal(verified + 4, target->mtp_reserved_tokens());
                const auto after_catchup = target->mtp_layer_population();
                const size_t calls = draft_kv->mtp_span_attention_calls();
                for (size_t step = 0; step < 2; ++step) {
                    if (!t.assert_true(append_mtp(int(verified + step), 1))) return;
                    t.assert_equal(verified, target->tokens());
                    t.assert_equal(verified + step + 1, mtp->tokens());
                    t.assert_equal(calls + step + 1, draft_kv->mtp_span_attention_calls());
                    t.assert_true(target->has_mtp_layer());
                    const auto population = target->mtp_layer_population();
                    t.assert_equal(after_catchup.bytes + (step + 1)*
                        (kv_layout.k_token_bytes + kv_layout.v_token_bytes), population.bytes);
                    ggml_kv_stream_span_plan_view draft_plan;
                    if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                            target->mtp_layer_plan(1), draft_plan))) return;
                    t.assert_equal(verified + step + 1, draft_plan.active_tokens);
                    llama_synchronize(draft.get());
                    llama_synchronize(ordinary_draft.get());
                    const float * live = llama_get_logits_ith(draft.get(), -1);
                    const float * stock = llama_get_logits_ith(ordinary_draft.get(), -1);
                    if (!t.assert_true(live && stock)) return;
                    float error = 0;
                    for (size_t i = 0; i < vocab_size; ++i)
                        error = std::max(error, std::abs(live[i] - stock[i]));
                    t.out << "real MTP draft step " << step << " max logit error=" << error << '\n';
                    t.assert_true(error < .05f);
                }
            }
            if (width == 4) {
                const auto before_partial = target->mtp_layer_population();
                if (!t.assert_true(llama_memory_seq_rm(
                        llama_get_memory(ordinary_draft.get()), 0, llama_pos(verified + 1), -1))) return;
                if (!t.assert_true(llama_memory_seq_rm(
                        llama_get_memory(draft.get()), 0, llama_pos(verified + 1), -1))) return;
                t.assert_equal(verified + 1, mtp->tokens());
                t.assert_true(target->has_mtp_layer());
                t.assert_equal(verified + 4, target->mtp_reserved_tokens());
                t.assert_equal(before_partial.bytes, target->mtp_layer_population().bytes);
                ggml_kv_stream_span_plan_view partial_plan;
                if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                        target->mtp_layer_plan(1), partial_plan))) return;
                t.assert_equal(verified + 1, partial_plan.active_tokens);

                for (size_t accepted = 0; accepted <= 4; ++accepted) {
                    if (mtp->tokens() != verified) {
                        if (!t.assert_true(llama_memory_seq_rm(
                                llama_get_memory(ordinary_draft.get()), 0, llama_pos(verified), -1) &&
                                llama_memory_seq_rm(
                                llama_get_memory(draft.get()), 0, llama_pos(verified), -1))) return;
                    }
                    t.assert_equal(verified, mtp->tokens());
                    if (target->has_mtp_layer() && !t.assert_true(target->release_mtp_layer())) return;
                    if (!t.assert_true(target->acquire_mtp_layer(4))) return;
                    const size_t calls_before = draft_kv->mtp_span_attention_calls();
                    for (size_t i = 0; i < 4; ++i) {
                        if (!t.assert_true(append_mtp(int(verified + i), 1))) return;
                        t.assert_equal(calls_before + i + 1,
                            draft_kv->mtp_span_attention_calls());
                    }
                    t.assert_equal(verified + 4, mtp->tokens());
                    t.assert_equal(verified, target->tokens());
                    if (accepted == 4) {
                        // Interior holes cannot be represented by the logical prefix frontier.
                        t.assert_true(!llama_memory_seq_rm(
                            llama_get_memory(draft.get()), 0, llama_pos(verified + 1),
                            llama_pos(verified + 2)));
                        t.assert_equal(verified + 4, mtp->tokens());
                        t.assert_true(target->has_mtp_layer());
                    }
                    const uint64_t generation = mtp->identity().generation;
                    const auto before_acceptance = target->mtp_layer_population();
                    if (!t.assert_true(llama_memory_seq_rm(
                            llama_get_memory(ordinary_draft.get()), 0,
                            llama_pos(verified + accepted), -1) &&
                            llama_memory_seq_rm(
                            llama_get_memory(draft.get()), 0,
                            llama_pos(verified + accepted), -1))) return;
                    t.assert_equal(verified + accepted, mtp->tokens());
                    t.assert_true(target->has_mtp_layer());
                    t.assert_equal(verified + 4, target->mtp_reserved_tokens());
                    t.assert_equal(before_acceptance.bytes, target->mtp_layer_population().bytes);
                    ggml_kv_stream_span_plan_view accepted_plan;
                    if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                            target->mtp_layer_plan(1), accepted_plan))) return;
                    t.assert_equal(verified + accepted, accepted_plan.active_tokens);

                    if (accepted == 4) t.assert_equal(generation, mtp->identity().generation);
                    else t.assert_true(mtp->identity().generation > generation);
                }
                llama_synchronize(draft.get());
                const size_t state_bytes = llama_state_seq_get_size(draft.get(), 0);
                if (!t.assert_true(state_bytes > 0)) return;
                std::vector<uint8_t> saved_state(state_bytes);
                if (!t.assert_equal(state_bytes, llama_state_seq_get_data(
                        draft.get(), saved_state.data(), saved_state.size(), 0))) return;
                if (!t.assert_true(mtp->host()->layer(0, host_rows))) return;
                const size_t prefix_k_bytes = mtp->tokens()*kv_layout.k_token_bytes;
                const size_t prefix_v_bytes = mtp->tokens()*kv_layout.v_token_bytes;
                const std::vector<uint8_t> saved_k(static_cast<const uint8_t *>(host_rows.k),
                    static_cast<const uint8_t *>(host_rows.k) + prefix_k_bytes);
                const std::vector<uint8_t> saved_v(static_cast<const uint8_t *>(host_rows.v),
                    static_cast<const uint8_t *>(host_rows.v) + prefix_v_bytes);
                llama_memory_clear(llama_get_memory(draft.get()), true);
                t.assert_equal(size_t(0), mtp->tokens());
                t.assert_true(!target->has_mtp_layer());
                t.assert_equal(size_t(0), target->mtp_reserved_tokens());
                ggml_backend_buffer_clear(mtp->host()->buffer(), 0);
                if (!t.assert_equal(state_bytes, llama_state_seq_set_data(
                        draft.get(), saved_state.data(), saved_state.size(), 0))) return;
                t.assert_equal(verified + 4, mtp->tokens());
            } else if (!t.assert_true(llama_kv_stream_mtp_release(context.get()))) return;
        }
        if (vision_mtp_handoff) {
            auto * owner = llama_context_compute_memory(context.get());
            const auto snapshot = [](llama_context * ctx) {
                std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, 0));
                bytes.resize(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), 0));
                return bytes;
            };
            const auto target_state = snapshot(context.get()), draft_state = snapshot(draft.get());
            const auto tokens = mtp->tokens();
            const auto parent = owner->shared_parent();
            if (!t.assert_true("populated draft permits suspension", owner->can_suspend_for_vision() && llama_context_suspend_kv_device(context.get()))) return;
            t.assert_true("suspension retires MTP ring", owner->kv_device_suspended() && !target->has_mtp_layer());
            t.assert_equal(size_t(0), target->device_grant_bytes());
            t.assert_true("suspension preserves both states", snapshot(context.get()) == target_state && snapshot(draft.get()) == draft_state);
            std::vector<ggml_backend_memory_lease_t> loans;
            if (!t.assert_true(owner->lend_suspended({1024}, loans))) return;
            ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(loans.front()), 0xa5);
            t.assert_true(!llama_context_resume_kv_device(context.get(), llama_memory_text_phase::prefill));
            for (auto * loan : loans) ggml_backend_memory_lease_free(loan);
            if (!t.assert_true(llama_context_resume_kv_device(context.get(), llama_memory_text_phase::prefill))) return;
            t.assert_true("resume preserves parent and admission", owner->shared_parent() == parent && owner->can_suspend_for_vision());
            t.assert_equal(tokens, mtp->tokens());
            t.assert_true("resume preserves both states", snapshot(context.get()) == target_state && snapshot(draft.get()) == draft_state);
            t.assert_true(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, llama_pos(verified), -1));
            t.assert_true("resume permits new MTP lease", llama_kv_stream_mtp_prepare(context.get(), 4));
            t.assert_true(llama_kv_stream_mtp_release(context.get()));
        }
        auto wrong_ctx = draft_params;
        wrong_ctx.n_ctx = 2048;
        context_ptr mismatched_context(llama_init_from_model(model.get(), wrong_ctx), llama_free);
        t.assert_true(!mismatched_context);
        auto wrong_quant = draft_params;
        wrong_quant.type_k = GGML_TYPE_F16;
        context_ptr mismatched_quant(llama_init_from_model(model.get(), wrong_quant), llama_free);
        t.assert_true(!mismatched_quant);
    });
    t.test("serial_phase_alternation_and_boundary_cancellation_match_stock", [&](testing & t) {
        const auto baseline = evaluate_serial_phases(t,model.get(),prompt,0);
        const auto streamed = evaluate_serial_phases(t,model.get(),prompt,(shared_budget_control ? 640 : 16)*1048576);
        if (!t.assert_equal(baseline.logits.size(),streamed.logits.size()) ||
                !t.assert_equal(size_t(14),streamed.logits.size()) ||
                !t.assert_equal(streamed.logits.size(),streamed.pool_grants.size()) ||
                !t.assert_equal(size_t(3),streamed.prefill_after_decode_us.size()) ||
                !t.assert_equal(baseline.prefill_after_decode_us.size(),
                    streamed.prefill_after_decode_us.size())) return;
        float maximum = 0;
        size_t matching = 0;
        for (size_t phase = 0; phase < baseline.logits.size(); ++phase) {
            if (!t.assert_equal(baseline.logits[phase].size(),streamed.logits[phase].size())) return;
            for (size_t token = 0; token < baseline.logits[phase].size(); ++token) {
                const float actual = streamed.logits[phase][token];
                if (!std::isfinite(actual)) { t.assert_true(false); return; }
                maximum = std::max(maximum,std::abs(baseline.logits[phase][token]-actual));
            }
            matching += std::max_element(baseline.logits[phase].begin(),baseline.logits[phase].end())-
                baseline.logits[phase].begin() ==
                std::max_element(streamed.logits[phase].begin(),streamed.logits[phase].end())-
                streamed.logits[phase].begin();
        }
        t.out << "serial phase max logit error=" << maximum
              << ", matching boundaries=" << matching << '/' << baseline.logits.size() << '\n';
        t.assert_true(maximum < .05f);
        t.assert_equal(baseline.logits.size(),matching);
        t.assert_true(baseline.recurrent.metadata == streamed.recurrent.metadata);
        t.assert_true(baseline.recurrent.tensors == streamed.recurrent.tensors);
        const size_t prefill_pool = streamed.pool_grants.front();
        const size_t decode_pool = streamed.pool_grants[1];
        t.assert_true(decode_pool > prefill_pool);
        for (size_t i = 0; i < 8; ++i) {
            t.assert_equal(i%2 ? decode_pool : prefill_pool,streamed.pool_grants[i]);
        }
        t.assert_equal(prefill_pool,streamed.pool_grants[8]);
        t.assert_equal(decode_pool,streamed.pool_grants[9]);
        t.assert_equal(prefill_pool,streamed.pool_grants[10]);
        t.assert_equal(decode_pool,streamed.pool_grants[11]);
        t.assert_equal(prefill_pool,streamed.pool_grants[12]);
        t.assert_equal(decode_pool,streamed.pool_grants[13]);
        t.out << "decode-to-prefill baseline -> streamed us:";
        for (size_t i = 0; i < streamed.prefill_after_decode_us.size(); ++i) {
            t.out << ' ' << baseline.prefill_after_decode_us[i] << "->"
                  << streamed.prefill_after_decode_us[i] << " (delta "
                  << streamed.prefill_after_decode_us[i]-baseline.prefill_after_decode_us[i] << ')';
        }
        t.out << '\n';
    });
    for (uint32_t ubatch : {256u,512u}) t.test("hybrid_prefill_decode_and_recurrent_state_match_ub_"+std::to_string(ubatch), [&](testing & t) {
        const auto baseline = evaluate(t,model.get(),prompt,ubatch,0,{});
        if (!t.assert_equal(size_t(32),baseline.continuation.size())) return;
        const auto streamed = evaluate(t,model.get(),prompt,ubatch,(shared_budget_control ? 640 : (f16_control || resident_control ? 64 : 16))*1048576,baseline.continuation);
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
