#include "llama.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-memory-recurrent-spill.h"
#include "../src/llama-memory-bounded-kv.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-io.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "testing.h"
#include "kv-stream-bit-exact-test.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

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
static size_t gemma_test_arena_size = size_t(1024)*1048576;

// Keep instrumentation budgets explicit without changing ordinary model qualification.
static size_t gemma_test_arena_bytes(const char * text) {
    if (!text) return size_t(1024)*1048576;
    size_t mib=0;
    if (!*text) throw std::invalid_argument("missing Gemma test arena MiB");
    for (const char * digit=text;*digit;++digit) {
        if (*digit < '0' || *digit > '9' || mib > (4095u-size_t(*digit-'0'))/10)
            throw std::invalid_argument("Gemma test arena must be 1..4095 MiB");
        mib=mib*10+size_t(*digit-'0');
    }
    if (!mib) throw std::invalid_argument("Gemma test arena must be 1..4095 MiB");
    return mib*1048576;
}

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

// Fail one candidate view, then permit parent recovery.
struct serial_parent_view_failure {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t,size_t,size_t);
    inline static serial_parent_view_failure * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t calls = 0;
    size_t fail_call;
    explicit serial_parent_view_failure(ggml_backend_buffer_t parent,size_t fail_call=2) :
        parent(parent),original(parent->view_buffer),fail_call(fail_call) {
        GGML_ASSERT(original && !active);
        active = this;
        parent->view_buffer = [](ggml_backend_buffer_t buffer,size_t offset,size_t bytes) {
            GGML_ASSERT(active && buffer == active->parent);
            if (++active->calls == active->fail_call) return static_cast<ggml_backend_buffer_t>(nullptr);
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

// Keep exact model outputs across fresh processes without accepting truncated or mismatched controls.
static void gemma_logits_file(FILE * file,bool writing,const std::array<uint64_t,8> & profile,
        std::vector<std::vector<float>> & values) {
    constexpr uint64_t magic=0x474d344c4f474954ull;
    if (!file || profile[0] != magic || profile[1] != 1 || !profile[2] || profile[2] > INT32_MAX ||
            profile[3] >= profile[2] || !profile[4] || profile[4] > 256 || profile[4] > profile[2]-profile[3] ||
            !profile[5] || profile[5] > INT32_MAX || profile[5] > (uint64_t(32)<<30)/sizeof(float)/profile[4] ||
            profile[6] >= GGML_TYPE_COUNT || profile[7] >= GGML_TYPE_COUNT)
        throw std::runtime_error("invalid Gemma logit control profile");
    if (writing) {
        if (values.size() != profile[4]) throw std::runtime_error("invalid Gemma logit control rows");
        for (const auto & row : values) if (row.size() != profile[5])
            throw std::runtime_error("invalid Gemma logit control columns");
        if (std::fwrite(profile.data(),1,sizeof(profile),file) != sizeof(profile))
            throw std::runtime_error("cannot write Gemma logit control header");
        for (const auto & row : values) if (std::fwrite(row.data(),sizeof(float),row.size(),file) != row.size())
            throw std::runtime_error("cannot write Gemma logit control values");
        const uint64_t footer=~magic;
        if (std::fwrite(&footer,1,sizeof(footer),file) != sizeof(footer) || std::fflush(file))
            throw std::runtime_error("cannot finish Gemma logit control");
        return;
    }
    std::array<uint64_t,8> actual{};
    if (std::fread(actual.data(),1,sizeof(actual),file) != sizeof(actual) || actual != profile)
        throw std::runtime_error("Gemma logit control header differs");
    auto next=std::vector<std::vector<float>>(size_t(profile[4]),std::vector<float>(size_t(profile[5])));
    for (auto & row : next) if (std::fread(row.data(),sizeof(float),row.size(),file) != row.size())
        throw std::runtime_error("truncated Gemma logit control values");
    uint64_t footer=0;
    if (std::fread(&footer,1,sizeof(footer),file) != sizeof(footer) || footer != ~magic || std::fgetc(file) != EOF || std::ferror(file))
        throw std::runtime_error("invalid Gemma logit control footer");
    values=std::move(next);
}

// Qualify the model graph, not just individual cache owners or attention operators.
static void test_gemma_target_context(testing & t, llama_model * model,uint32_t capacity=8192,
        uint32_t prompt=1333,uint32_t outputs=8,bool native_control=true,bool expect_streaming=false,
        const char * export_native=nullptr,const char * import_native=nullptr) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    if (!t.assert_true(capacity <= model->hparams.n_ctx_train && prompt > 1024 &&
            prompt < capacity && outputs > 0 && outputs <= capacity-prompt)) return;
    std::vector<std::vector<float>> reference;
    const auto vocab=size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    const std::array<uint64_t,8> profile{0x474d344c4f474954ull,1,capacity,prompt,outputs,vocab,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
    if (!t.assert_true(!(export_native && import_native) && (!(export_native || import_native) || native_control))) return;
    if (import_native) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(import_native,"rb"));
        gemma_logits_file(file.get(),false,profile,reference);
    }
    for (bool streaming : {false,true}) {
        if (!streaming && !native_control) continue;
        if ((!streaming && import_native) || (streaming && export_native)) continue;
        auto cp=llama_context_default_params();
        cp.n_ctx=capacity; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (streaming) cp.shared_device_memory_bytes=gemma_test_arena_size;
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(streaming ? "streamed Gemma target context is admitted" : "native Gemma control starts",bool(context))) return;
        auto * groups=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(context.get()));
        if (!t.assert_true(groups && !groups->get_swa()->bounded_storage_pending())) return;
        if (streaming) {
            auto * owner=context->get_compute_memory();
            if (!t.assert_true(owner && owner->shares_kv_memory() && owner->has_bounded_cache())) return;
            t.assert_equal(cp.shared_device_memory_bytes,owner->shared_parent_capacity());
            t.assert_true(ggml_backend_buffer_is_view(groups->get_swa()->get_k_storage(0)->buffer));
        }
        common_batch batch(context.get());
        for (int32_t first=0;first<int32_t(prompt);) {
            const auto rows=std::min(256,int32_t(prompt)-first);
            batch.clear();
            for (int32_t row=0;row<rows;++row) batch.add(llama_token(10+(first+row)%64),first+row,0,row == rows-1);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            first+=rows;
            if (first%8192 == 0 || first == int32_t(prompt)) t.out << "Gemma target streaming=" << streaming
                << " prefill=" << first << '/' << prompt << std::endl;
        }
        if (streaming) llama_set_kv_stream_decode(context.get(),true);
        for (uint32_t step=0;step<outputs;++step) {
            const auto * logits=llama_get_logits_ith(context.get(),-1);
            if (!t.assert_true(logits != nullptr)) return;
            t.assert_true("Gemma model logits remain finite",
                std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }));
            if (!streaming) reference.emplace_back(logits,logits+vocab);
            else if (native_control) t.assert_true("native and streamed Gemma logits are bit-exact",
                std::memcmp(reference[size_t(step)].data(),logits,vocab*sizeof(float)) == 0);
            else t.assert_true(std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }));
            batch.clear(); batch.add(llama_token(80+step),llama_pos(prompt+step),0,true);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
        llama_synchronize(context.get());
        if (streaming) {
            auto * stream=groups->get_base()->get_kv_stream();
            t.assert_true(stream && stream->complete());
            t.assert_equal(size_t(prompt)+outputs,stream->tokens());
            llama_context_memory_diagnostics usage;
            if (!t.assert_true(context->get_compute_memory()->diagnostics(usage))) return;
            t.assert_equal(size_t(425984000),usage.persistent_cache_bytes);
            t.assert_equal(cp.shared_device_memory_bytes,usage.parent_bytes);
            t.assert_true(usage.phase == llama_memory_text_phase::decode);
            if (expect_streaming) t.assert_true("longer-than-pool target uses streaming",usage.streaming_active);
            t.out << "Gemma target context=" << capacity << " prompt=" << prompt << " outputs=" << outputs
                << " native_control=" << native_control << " decode_pool=" << usage.kv_pool_bytes
                << " resident/ring=" << usage.resident_pages_per_layer << '/' << usage.ring_slots
                << " active=" << usage.active_pages << '\n';
        }
    }
    if (export_native) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(export_native,"wbx"));
        gemma_logits_file(file.get(),true,profile,reference);
    }
}

// Constructor failures must explain the rejected profile and leave a valid context constructible afterward.
static void test_gemma_target_admission(testing & t, llama_model * model,bool include_forward=true) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    for (uint32_t invalid=0;invalid<6;++invalid) {
        auto cp=llama_context_default_params();
        cp.n_ctx=8192; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=gemma_test_arena_size;
        const char * expected="KV streaming requires one CUDA device, serial Qwen35 or Gemma4 target context";
        if (invalid == 0) {
            cp.swa_full=true;
            expected="Gemma arena streaming requires bounded SWA storage";
        } else if (invalid == 1) cp.n_seq_max=2;
        else if (invalid == 2) cp.embeddings=true;
        else if (invalid == 3) cp.offload_kqv=false;
        else if (invalid == 4) {
            cp.shared_device_memory_bytes=256u*1024u*1024u;
            expected="workspace admission: insufficient arena quota";
        } else {
            cp.shared_device_memory_bytes=512u*1024u*1024u;
            expected="persistent local cache does not fit the declared shared arena";
        }
        bool rejected=false;
        try {
            auto context=std::make_unique<llama_context>(*model,cp);
        } catch (const std::runtime_error & error) {
            t.out << "Gemma admission case=" << invalid << ": " << error.what() << '\n';
            rejected=std::string(error.what()).find(expected) != std::string::npos;
            if (invalid == 4 || invalid == 5) {
                const std::string message=error.what();
                const auto start=message.find("required=");
                size_t required=0,available=0,additional=0;
                const bool parsed=start!=std::string::npos && std::sscanf(message.c_str()+start,
                    "required=%zu, available=%zu, additional=%zu",&required,&available,&additional)==3;
                t.assert_true("quota diagnostic carries exact byte fields",parsed);
                if (parsed) {
                    t.assert_equal(cp.shared_device_memory_bytes,available);
                    t.assert_true(required>available);
                    t.assert_equal(required-available,additional);
                }
                if (invalid==5) t.assert_true(message.find("phase=")!=std::string::npos && message.find("resource=")!=std::string::npos);
            }
        }
        t.assert_true(std::string("startup rejects profile with the expected cause: ")+expected,rejected);
    }
    {
        auto cp=llama_context_default_params();
        cp.n_ctx=8192; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=gemma_test_arena_size; cp.n_rs_seq=3;
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true("Gemma keeps upstream's ignored recurrent rollback setting",bool(context))) return;
        t.assert_equal(uint32_t(0),context->get_cparams().n_rs_seq);
        t.assert_true(!context->get_compute_memory()->has_speculative_consumer());
    }
    if (include_forward) test_gemma_target_context(t,model);
}

// A rejected streamed suffix operation must not change the independent local group's cells.
static void test_gemma_lifecycle(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.kv_stream_pool_bytes=64u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get();
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
        8192,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
    llama_batch_allocr allocator(1);
    auto batch=allocator.ubatch_reserve(64,1);
    batch.data->seq_id_data.resize(batch.n_tokens,0);
    for (uint32_t row=0;row<batch.n_tokens;++row) {
        batch.pos[row]=llama_pos(row); batch.n_seq_id[row]=1;
        batch.seq_id[row]=&batch.data->seq_id_data[row];
    }
    for (auto * group : {cache.get_base(),cache.get_swa()}) {
        auto slots=group->prepare({batch});
        if (!t.assert_true(slots.size() == 1)) return;
        group->apply_ubatch(slots[0],batch);
    }
    auto * stream=cache.get_base()->get_kv_stream();
    if (!t.assert_true(stream != nullptr)) return;
    ggml_backend_buffer_clear(stream->buffer(),0);
    if (!t.assert_true("imported host frontier is restored",stream->restore(64))) return;
    const auto positions=[](const llama_kv_cache & group) {
        std::vector<llama_pos> result;
        const auto & cells=group.get_cells(0);
        for (uint32_t row=0;row<group.get_size();++row)
            result.push_back(cells.is_empty(row) ? -1 : cells.pos_get(row));
        return result;
    };
    const auto full=positions(*cache.get_base()), local=positions(*cache.get_swa());
    t.assert_true(!cache.seq_rm(0,10,20));
    t.assert_equal(size_t(64),stream->tokens());
    t.assert_true(full == positions(*cache.get_base()));
    t.assert_true("rejected interior removal preserves local cells",local == positions(*cache.get_swa()));

    struct image_writer : llama_io_write_i {
        std::vector<uint8_t> bytes;
        void write(const void * data,size_t size) override {
            const auto * begin=static_cast<const uint8_t *>(data);
            bytes.insert(bytes.end(),begin,begin+size);
        }
        void write_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
            const auto start=bytes.size(); bytes.resize(start+size);
            ggml_backend_tensor_get(tensor,bytes.data()+start,offset,size);
        }
        size_t n_bytes() override { return bytes.size(); }
    };
    struct image_reader : llama_io_read_i {
        const std::vector<uint8_t> & bytes;
        size_t cursor=0;
        explicit image_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
        void read(void * data,size_t size) override {
            if (size > bytes.size()-cursor) throw std::runtime_error("truncated lifecycle fixture");
            std::memcpy(data,bytes.data()+cursor,size); cursor+=size;
        }
        void read_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
            if (size > bytes.size()-cursor) throw std::runtime_error("truncated lifecycle tensor");
            ggml_backend_tensor_set(tensor,bytes.data()+cursor,offset,size); cursor+=size;
        }
        size_t n_bytes() override { return cursor; }
    };
    const auto save=[&] {
        image_writer image;
        cache.state_write(image,0,0);
        return image.bytes;
    };
    // Exercise stock placement/serialization with encoded rows, without claiming a model-forward numerical test.
    const auto append=[&](uint32_t first,uint32_t count,uint32_t seed) {
        for (uint32_t done=0;done<count;) {
            const uint32_t rows=std::min(256u,count-done);
            auto batch=allocator.ubatch_reserve(rows,1);
            batch.data->seq_id_data.resize(rows,0);
            for (uint32_t row=0;row<rows;++row) {
                batch.pos[row]=llama_pos(first+done+row); batch.token[row]=llama_token(seed);
                batch.n_seq_id[row]=1; batch.seq_id[row]=&batch.data->seq_id_data[row];
            }
            for (auto * group : {cache.get_base(),cache.get_swa()}) {
                auto slots=group->prepare({batch});
                if (!t.assert_true(slots.size() == 1)) return false;
                group->apply_ubatch(slots[0],batch);
                const int32_t id=group == cache.get_base() ? 5 : 0;
                for (auto * tensor : {group->get_k_storage(id),group->get_v_storage(id)}) {
                    const auto row_bytes=ggml_row_size(tensor->type,tensor->ne[0]);
                    std::vector<float> floats(size_t(tensor->ne[0]));
                    std::vector<uint8_t> encoded(row_bytes);
                    for (uint32_t row=0;row<rows;++row) {
                        std::fill(floats.begin(),floats.end(),float(seed+uint32_t(batch.pos[row])%17)*0.0625f);
                        ggml_quantize_chunk(tensor->type,floats.data(),encoded.data(),0,1,tensor->ne[0],nullptr);
                        ggml_backend_tensor_set(tensor,encoded.data(),slots[0].idxs[0][row]*row_bytes,row_bytes);
                    }
                }
            }
            if (!t.assert_true(stream->restore(first+done+rows))) return false;
            done+=rows;
        }
        return true;
    };
    const auto local_window=[&] {
        auto * group=cache.get_swa();
        const auto & cells=group->get_cells(0);
        std::map<llama_pos,std::vector<uint8_t>> result;
        for (auto * tensor : {group->get_k_storage(0),group->get_v_storage(0)}) {
            std::vector<uint8_t> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
            const auto row_bytes=ggml_row_size(tensor->type,tensor->ne[0]);
            for (uint32_t row=0;row<group->get_size();++row) {
                if (cells.is_empty(row)) continue;
                const auto pos=cells.pos_get(row);
                if (llama_hparams::is_masked_swa(model->hparams.n_swa,model->hparams.swa_type,pos,cells.seq_pos_max(0))) continue;
                auto & entry=result[pos];
                entry.insert(entry.end(),bytes.data()+row*row_bytes,bytes.data()+(row+1)*row_bytes);
            }
        }
        return result;
    };
    cache.clear(true);
    if (!t.assert_true(append(0,64,1))) return;
    const auto saved=save();
    t.assert_true(!cache.seq_rm(0,0,16) && saved == save());
    t.assert_true(cache.seq_rm(0,16,-1));
    t.assert_equal(size_t(16),stream->tokens());
    if (!t.assert_true(append(16,48,2))) return;
    t.assert_true("equal-length requests with different content are distinct",saved != save());
    cache.clear(true);
    t.assert_equal(size_t(0),stream->tokens());
    t.assert_equal(uint32_t(0),cache.get_swa()->get_cells(0).get_used());
    image_reader reader(saved);
    cache.state_read(reader,0,0);
    t.assert_equal(saved.size(),reader.n_bytes());
    t.assert_equal(size_t(64),stream->tokens());
    t.assert_true("both groups round-trip through saved state",saved == save());
    t.assert_true(cache.seq_rm(0,16,-1));
    if (!t.assert_true(append(16,48,2))) return;
    const auto rewritten=save();
    t.assert_true(rewritten != saved);
    t.assert_true(cache.seq_rm(0,64,64) && rewritten == save());

    cache.clear(true);
    if (!t.assert_true(append(0,2304,3))) return;
    const auto baseline=local_window();
    t.assert_equal(size_t(model->hparams.n_swa),baseline.size());
    for (uint32_t accepted=0;accepted<=4;++accepted) {
        if (!t.assert_true(append(2304,4,7))) return;
        const auto provisional=local_window();
        if (!t.assert_true(cache.seq_rm(0,llama_pos(2304+accepted),-1))) return;
        t.assert_equal(size_t(2304+accepted),stream->tokens());
        const auto retained=local_window();
        t.assert_equal(baseline.size(),retained.size());
        for (const auto & entry : retained) {
            const auto & reference=entry.first < 2304 ? baseline : provisional;
            const auto found=reference.find(entry.first);
            t.assert_true("accepted window bytes survive rejected writes",found != reference.end() && found->second == entry.second);
        }
        t.assert_true(cache.seq_rm(0,2304,-1));
        t.assert_true("zero acceptance and cancellation recover the complete local window",local_window() == baseline);
    }
    const auto full_before_old_prefix=positions(*cache.get_base());
    t.assert_true("a prefix whose local history was evicted cannot be reused",!cache.seq_rm(0,64,-1));
    t.assert_equal(size_t(2304),stream->tokens());
    t.assert_true(full_before_old_prefix == positions(*cache.get_base()));
    t.assert_true(local_window() == baseline);
    cache.clear(true);
    if (!t.assert_true(append(0,17,9))) return;
    t.assert_equal(llama_pos(16),cache.seq_pos_max(0));
    t.assert_equal(size_t(17),stream->tokens());
}

// Measure native local attention before storage exists, then reuse one parent through both text phases.
static void test_gemma_phase_parent(testing & t, llama_model * model, bool shared_reads=false) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr)), cpu(ggml_backend_cpu_init());
    auto * buft=llama_kv_stream_device_buffer_type(dev);
    std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
    std::vector<ggml_backend_buffer_type_t> types{buft,ggml_backend_cpu_buffer_type()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types.data(),2,1024,false,true));
    if (!t.assert_true(bool(sched))) return;
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.shared_device_memory_bytes=768u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get(); params.defer_bounded_cache=true;
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    const uint32_t capacity=shared_reads ? 131072 : 32768;
    llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
        capacity,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
    auto * local=cache.get_swa(); auto * stream=cache.get_base()->get_kv_stream();
    if (!t.assert_true(local->bounded_storage_pending() && stream)) return;
    auto * placeholder=local->get_k_storage(0);
    struct attention_graph {
        ggml_context_ptr ctx;
        ggml_cgraph * graph=nullptr;
        ggml_tensor * query=nullptr, * mask=nullptr, * output=nullptr;
    };
    const auto build=[&](uint32_t rows) {
        attention_graph graph;
        graph.ctx.reset(ggml_init({1024*1024,nullptr,true}));
        auto * k=local->get_k_storage(0); auto * v=local->get_v_storage(0);
        auto * keys=ggml_view_3d(graph.ctx.get(),k,256,k->ne[1],16,k->nb[1],ggml_row_size(k->type,256),0);
        auto * values=ggml_view_3d(graph.ctx.get(),v,256,v->ne[1],16,v->nb[1],ggml_row_size(v->type,256),0);
        graph.query=ggml_new_tensor_3d(graph.ctx.get(),GGML_TYPE_F32,256,rows,32);
        graph.mask=ggml_new_tensor_2d(graph.ctx.get(),GGML_TYPE_F16,k->ne[1],GGML_PAD(rows,32));
        ggml_set_input(graph.query); ggml_set_input(graph.mask);
        graph.output=ggml_flash_attn_ext(graph.ctx.get(),graph.query,keys,values,graph.mask,1.0f/16,0,0);
        ggml_prec_set_acc(graph.output,GGML_PREC_F32);
        graph.graph=ggml_new_graph_custom(graph.ctx.get(),64,false);
        ggml_build_forward_expand(graph.graph,graph.output);
        ggml_backend_sched_set_tensor_backend(sched.get(),graph.output,backend.get());
        return graph;
    };
    std::vector<size_t> measurements(4);
    for (size_t phase=0;phase<2;++phase) {
        ggml_backend_sched_reset(sched.get());
        auto graph=build(phase ? 1 : 256);
        ggml_backend_sched_reserve_size(sched.get(),graph.graph,measurements.data()+2*phase);
        t.assert_true(local->bounded_storage_pending() && !placeholder->data &&
            ggml_backend_buffer_get_size(placeholder->buffer) == 0);
        ggml_backend_sched_reset(sched.get());
    }
    llama_compute_workspace_plan plan;
    if (!t.assert_true(llama_compute_workspace_plan_make(types,measurements,2,plan))) return;
    auto owner=llama_context_memory::create(sched.get(),backends,plan,stream);
    if (!t.assert_true(bool(owner))) return;
    t.assert_equal(params.shared_device_memory_bytes,owner->shared_parent_capacity());
    llama_cparams cp{}; cp.n_ctx_seq=capacity; cp.n_ubatch=256; cp.n_seq_max=1; cp.flash_attn=true;
    llama_memory_bounded_kv_layout quote;
    if (!t.assert_true(llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,cp,buft,{609,1},quote))) return;
    auto too_large=quote; too_large.allocation.bytes=params.shared_device_memory_bytes;
    t.assert_true(!owner->reserve_bounded_cache(too_large) && owner->valid() && !owner->has_bounded_cache());
    auto loan=owner->reserve_bounded_cache(quote);
    if (!t.assert_true(loan && owner->retire_graph())) return;
    ggml_backend_sched_reset(sched.get());
    if (!t.assert_true(local->bind_bounded_storage(loan->lease()))) return;
    auto * parent=owner->shared_parent();
    const auto parent_base=uintptr_t(ggml_backend_buffer_get_base(parent));
    std::vector<uintptr_t> addresses;
    for (const auto & layer : quote.layers) for (bool value : {false,true}) {
        auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
        addresses.push_back(uintptr_t(tensor->data));
        t.assert_true(ggml_backend_buffer_is_view(tensor->buffer) && uintptr_t(tensor->data) >= parent_base &&
            uintptr_t(tensor->data)-parent_base <= params.shared_device_memory_bytes-ggml_nbytes(tensor));
    }
    auto * v=local->get_v_storage(0);
    std::vector<float> values(ggml_nelements(v));
    for (size_t i=0;i<values.size();++i) values[i]=float(int(i%31)-15)*0.0625f;
    std::vector<uint8_t> encoded(ggml_nbytes(v));
    ggml_quantize_chunk(v->type,values.data(),encoded.data(),0,v->ne[1],v->ne[0],nullptr);
    ggml_backend_tensor_set(v,encoded.data(),0,encoded.size());
    const auto verify_local=[&] {
        size_t ordinal=0;
        for (const auto & layer : quote.layers) for (bool value : {false,true}) {
            auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
            t.assert_equal(addresses[ordinal++],uintptr_t(tensor->data));
            std::vector<uint8_t> data(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor,data.data(),0,data.size());
            t.assert_true(value && layer.id == 0 ? data == encoded :
                std::all_of(data.begin(),data.end(),[](uint8_t byte) { return byte == 0; }));
        }
        t.assert_equal(uint32_t(0),local->get_cells(0).get_used());
    };
    verify_local();
    ggml_backend_buffer_clear(stream->buffer(),0);
    if (!t.assert_true(stream->restore(32000))) return;
    std::vector<float> prefill_output;
    llama_context_memory_diagnostics before{}, decode{};
    for (int step=0;step<3;++step) {
        const bool decoding=step == 1;
        const uint32_t rows=decoding ? 1 : 256;
        if (!t.assert_true(owner->retire_graph())) return;
        ggml_backend_sched_reset(sched.get());
        const auto phase=decoding ? llama_memory_text_phase::decode : llama_memory_text_phase::prefill;
        if (decoding) {
            const auto previous=owner->text_phase();
            {
                serial_parent_view_failure fault(parent,1);
                t.assert_true(owner->signal_text_phase({phase,rows,true,true,false}).status ==
                    llama_memory_text_phase_status::transition_failed);
                t.assert_true(fault.calls > 0);
            }
            t.assert_true(owner->valid() && stream->complete());
            t.assert_equal(previous.revision,owner->text_phase().revision);
            t.assert_equal(before.kv_pool_bytes,stream->pool_grant_bytes());
            t.assert_equal(size_t(32000),stream->tokens());
            verify_local();
        }
        if (!t.assert_true(owner->signal_text_phase({phase,rows,true,true,false}).status == llama_memory_text_phase_status::changed)) return;
        if (!t.assert_true(owner->prepare_serial_target())) return;
        auto graph=build(rows);
        if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(),graph.graph))) return;
        std::vector<float> query(ggml_nelements(graph.query),0.125f);
        std::vector<ggml_fp16_t> mask(ggml_nelements(graph.mask),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(graph.query,query.data(),0,query.size()*sizeof(float));
        ggml_backend_tensor_set(graph.mask,mask.data(),0,mask.size()*sizeof(ggml_fp16_t));
        if (!t.assert_true("native local attention executes after graph retirement",owner->compute_async(graph.graph) == GGML_STATUS_SUCCESS)) return;
        owner->synchronize();
        std::vector<float> output(ggml_nelements(graph.output));
        ggml_backend_tensor_get(graph.output,output.data(),0,output.size()*sizeof(float));
        t.assert_true(std::all_of(output.begin(),output.end(),[](float value) { return std::isfinite(value); }));
        if (step == 0) prefill_output=output;
        t.assert_true(std::any_of(output.begin(),output.end(),[](float value) { return value != 0; }));
        if (step == 2) t.assert_true(std::memcmp(output.data(),prefill_output.data(),output.size()*sizeof(float)) == 0);
        verify_local();
        llama_context_memory_diagnostics diagnostics;
        if (!t.assert_true(owner->diagnostics(diagnostics))) return;
        t.assert_equal(quote.allocation.bytes,diagnostics.persistent_cache_bytes);
        t.assert_equal(params.shared_device_memory_bytes,diagnostics.parent_bytes);
        t.assert_equal(diagnostics.parent_bytes,diagnostics.workspace_bytes+diagnostics.kv_pool_bytes+
            diagnostics.kv_writer_bytes+diagnostics.kv_attention_bytes+diagnostics.persistent_cache_bytes+diagnostics.unused_bytes);
        t.assert_true(owner->shared_parent() == parent && owner->covers_workspace(plan));
        if (step == 0) before=diagnostics;
        if (step == 1) decode=diagnostics;
        if (step == 2) t.assert_equal(before.kv_pool_bytes,diagnostics.kv_pool_bytes);
        if (!t.assert_true(owner->retire_graph())) return;
        ggml_backend_sched_reset(sched.get());
    }
    t.assert_true(decode.kv_pool_bytes > before.kv_pool_bytes && decode.workspace_bytes < before.workspace_bytes);
    t.out << "Gemma common parent bytes=" << params.shared_device_memory_bytes << " local=" << quote.allocation.bytes
        << " prefill pool=" << before.kv_pool_bytes << " decode pool=" << decode.kv_pool_bytes
        << " prefill resident/ring=" << before.resident_pages_per_layer << '/' << before.ring_slots
        << " decode resident/ring=" << decode.resident_pages_per_layer << '/' << decode.ring_slots << '\n';
    if (!shared_reads) return;
    const auto ids=cache.get_base()->get_layer_ids();
    if (!t.assert_true(!ids.empty())) return;
    const int32_t selected=int32_t(ids.back());
    const auto filter=[&](int32_t il) { return il == selected; };
    const auto share=[&](int32_t) { return selected; };
    llama_kv_cache reader(*model,model->hparams,params.type_k,params.type_v,false,true,true,
        capacity,1,256,0,LLAMA_SWA_TYPE_NONE,cache.get_base(),filter,nullptr,share);
    auto * original=reader.get_k_storage(selected);
    const size_t grants=stream->device_grant_bytes();
    t.assert_true(!reader.attach_shared_stream_reader(*owner,5));
    t.assert_true(reader.get_k_storage(selected) == original);
    if (!t.assert_true(reader.attach_shared_stream_reader(*owner,4))) return;
    t.assert_equal(grants,stream->device_grant_bytes());
    t.assert_true(!reader.attach_shared_stream_reader(*owner,4));
    t.assert_true(reader.get_k_storage(selected) != original &&
        reader.get_k_storage(selected)->data == original->data &&
        ggml_backend_buft_is_execution(ggml_backend_buffer_get_type(reader.get_k_storage(selected)->buffer)));
    bool mismatch=false;
    try {
        llama_kv_cache invalid(*model,model->hparams,params.type_k,GGML_TYPE_Q8_0,false,true,true,
            capacity,1,256,0,LLAMA_SWA_TYPE_NONE,cache.get_base(),filter,nullptr,share);
    } catch (const std::runtime_error & error) {
        mismatch=std::string(error.what()) == "shared KV source has incompatible cache geometry or format";
    }
    t.assert_true("shared cache cannot silently override requested quantization",mismatch);
    llama_kv_stream_host_layer host;
    if (!t.assert_true(stream->host()->layer(uint32_t(ids.size()-1),host))) return;
    ggml_backend_buffer_clear(stream->buffer(),0);
    for (bool value : {false,true}) {
        auto * root=value ? cache.get_base()->get_v_storage(selected) : original;
        std::vector<float> source(ggml_nelements(root));
        for (size_t i=0;i<source.size();++i) source[i]=float(int(i%37)-18)*0.0625f;
        ggml_quantize_chunk(root->type,source.data(),value ? host.v : host.k,0,root->ne[1],root->ne[0],nullptr);
    }
    const uint32_t active=128000;
    if (!t.assert_true(stream->restore(active) && owner->prepare_serial_decode() && reader.arm_shared_stream_reader())) return;
    llama_kv_stream_runtime_diagnostics runtime;
    if (!t.assert_true(stream->runtime_diagnostics(runtime))) return;
    const size_t layer_bytes=size_t(active)*(stream->host()->layout().k_token_bytes+stream->host()->layout().v_token_bytes);
    t.out << "borrowed source: rows=" << active << " layer bytes=" << layer_bytes << " entire pool=" << runtime.pool_bytes
        << " resident mean=" << runtime.resident_pages_per_layer << " ring slots=" << runtime.ring_slots << '\n';
    // Even the whole pool cannot contain this layer, so no resident/ring partition can acquire it at once.
    if (!t.assert_true("source history exceeds the entire physical KV pool",layer_bytes > runtime.pool_bytes)) return;
    llama_kv_cache::slot_info slots;
    slots.s0=slots.s1=0; slots.strm={0}; slots.idxs={{0}};
    for (bool value : {false,true}) {
        bool rejected=false;
        try {
            if (value) reader.cpy_v(nullptr,nullptr,nullptr,selected,slots);
            else reader.cpy_k(nullptr,nullptr,nullptr,selected,slots);
        } catch (const std::runtime_error & error) {
            rejected=std::string(error.what()) == "cannot write a borrowed KV source";
        }
        t.assert_true("shared reader cannot build a KV writer",rejected);
    }
    std::vector<uint8_t> saved_head(256);
    std::memcpy(saved_head.data(),host.v,saved_head.size());
    reader.clear(true);
    t.assert_true("clearing the reader does not clear its source",std::memcmp(saved_head.data(),host.v,saved_head.size()) == 0);
    t.assert_equal(size_t(active),stream->tokens());
    auto * root=reader.get_k_storage(selected);
    if (!t.assert_true(root->ne[0] == 2048 && root->ne[1] == capacity && reader.get_size() == capacity)) return;
    for (uint32_t rows : {1u,3u,4u}) {
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,512,rows,32);
        auto * mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,active,rows);
        auto * k=ggml_permute(ctx.get(),reader.get_k(ctx.get(),selected,active,slots),0,2,1,3);
        auto * v=ggml_permute(ctx.get(),reader.get_v(ctx.get(),selected,active,slots),0,2,1,3);
        auto * output=ggml_flash_attn_ext(ctx.get(),q,k,v,mask,1.0f/16,0,0);
        ggml_prec_set_acc(output,GGML_PREC_F32);
        ggml_backend_execution_set_external_workspace(output,true);
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,output);
        auto * nk=ggml_new_tensor_2d(ctx.get(),params.type_k,2048,active);
        auto * nv=ggml_new_tensor_2d(ctx.get(),params.type_v,2048,active);
        auto * native_k=ggml_view_3d(ctx.get(),nk,512,active,4,nk->nb[1],ggml_row_size(nk->type,512),0);
        auto * native_v=ggml_view_3d(ctx.get(),nv,512,active,4,nv->nb[1],ggml_row_size(nv->type,512),0);
        auto * expected=ggml_flash_attn_ext(ctx.get(),q,native_k,native_v,mask,1.0f/16,0,0);
        ggml_prec_set_acc(expected,GGML_PREC_F32);
        auto * native_graph=ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(native_graph,expected);
        ggml_backend_buffer_ptr allocation(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(allocation))) return;
        std::vector<float> queries(ggml_nelements(q),0.125f);
        std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(q,queries.data(),0,ggml_nbytes(q));
        ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
        ggml_backend_tensor_set(nk,host.k,0,ggml_nbytes(nk));
        ggml_backend_tensor_set(nv,host.v,0,ggml_nbytes(nv));
        if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),native_graph))) return;
        std::vector<float> want(ggml_nelements(expected)),got(want.size());
        ggml_backend_tensor_get(expected,want.data(),0,ggml_nbytes(expected));
        for (int repeat=0;repeat<2;++repeat) {
            if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph))) return;
            ggml_backend_tensor_get(output,got.data(),0,ggml_nbytes(output));
            t.assert_true("cache-view borrowed attention matches stock bits",std::memcmp(want.data(),got.data(),ggml_nbytes(output)) == 0);
            t.assert_equal(size_t(active),stream->tokens());
        }
        if (!t.assert_true(reader.release_shared_stream_reader())) return;
        t.assert_equal(GGML_STATUS_FAILED,ggml_backend_graph_compute(backend.get(),graph));
        ggml_backend_tensor_get(output,got.data(),0,ggml_nbytes(output));
        t.assert_true("unarmed shared reader preserves outputs",std::memcmp(want.data(),got.data(),ggml_nbytes(output)) == 0);
        if (!t.assert_true(reader.arm_shared_stream_reader())) return;
        t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph));
        ggml_backend_tensor_get(output,got.data(),0,ggml_nbytes(output));
        t.assert_true("shared reader rearm keeps stock bits",std::memcmp(want.data(),got.data(),ggml_nbytes(output)) == 0);
    }
    t.assert_true(reader.release_shared_stream_reader());
    const auto local_ids=local->get_layer_ids();
    if (!t.assert_true(local_ids.size() >= 3)) return;
    const int32_t local_source=int32_t(local_ids.back());
    auto * native_key=local->get_k_storage(local_source);
    auto * native_value=local->get_v_storage(local_source);
    std::vector<float> local_values(ggml_nelements(native_value));
    for (size_t i=0;i<local_values.size();++i) local_values[i]=float(int(i%31)-15)*0.0625f;
    std::vector<uint8_t> encoded_local(ggml_nbytes(native_value));
    ggml_quantize_chunk(native_value->type,local_values.data(),encoded_local.data(),0,
        native_value->ne[1],native_value->ne[0],nullptr);
    ggml_backend_tensor_set(native_value,encoded_local.data(),0,encoded_local.size());
    auto native_owner=local->retain_bounded_storage(610,1,capacity,256);
    if (!t.assert_true(bool(native_owner))) return;
    auto native_pin=native_owner->read(native_owner->identity());
    const auto local_filter=[&](int32_t il) {
        return std::find(local_ids.end()-3,local_ids.end(),uint32_t(il)) != local_ids.end();
    };
    const auto local_share=[&](int32_t) { return local_source; };
    llama_kv_cache local_reader(*model,model->hparams,params.type_k,params.type_v,false,true,true,
        local->get_size(),1,256,model->hparams.n_swa,model->hparams.swa_type,local,local_filter,nullptr,local_share);
    auto * original_local=local_reader.get_k_storage(local_source);
    llama_memory_bounded_kv_read empty;
    t.assert_true(!local_reader.attach_shared_bounded_reader(empty));
    t.assert_true(local_reader.get_k_storage(local_source) == original_local);
    if (!t.assert_true(local_reader.attach_shared_bounded_reader(native_pin))) return;
    t.assert_true(!local_reader.attach_shared_bounded_reader(native_pin));
    if (!t.assert_true(local_reader.arm_shared_bounded_reader(native_pin))) return;
    for (auto id : local_reader.get_layer_ids()) {
        auto * key=local_reader.get_k_storage(int32_t(id)),* value=local_reader.get_v_storage(int32_t(id));
        t.assert_true(key != native_key && value != native_value && key->view_src == native_key && value->view_src == native_value);
        t.assert_true(key->data == native_key->data && value->data == native_value->data);
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,1,32);
        auto * mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,local->get_size(),1);
        const auto attention=[&](ggml_tensor * k,ggml_tensor * v) {
            auto * keys=ggml_view_3d(ctx.get(),k,256,local->get_size(),16,k->nb[1],ggml_row_size(k->type,256),0);
            auto * values=ggml_view_3d(ctx.get(),v,256,local->get_size(),16,v->nb[1],ggml_row_size(v->type,256),0);
            auto * out=ggml_flash_attn_ext(ctx.get(),q,keys,values,mask,1.0f/16,0,0);
            ggml_prec_set_acc(out,GGML_PREC_F32);
            return out;
        };
        auto * result=attention(key,value),* reference=attention(native_key,native_value);
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
        ggml_build_forward_expand(graph,result); ggml_build_forward_expand(graph,reference);
        ggml_backend_buffer_ptr allocation(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(allocation))) return;
        std::vector<float> query(ggml_nelements(q),0.125f);
        std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(q,query.data(),0,ggml_nbytes(q));
        ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
        if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph))) return;
        std::vector<float> got(ggml_nelements(result)),want(got.size());
        ggml_backend_tensor_get(result,got.data(),0,ggml_nbytes(result));
        ggml_backend_tensor_get(reference,want.data(),0,ggml_nbytes(reference));
        t.assert_true("three local readers preserve native attention bits",std::memcmp(want.data(),got.data(),ggml_nbytes(result)) == 0);
    }
    local_reader.clear(true);
    std::vector<uint8_t> after_local(encoded_local.size());
    ggml_backend_tensor_get(native_value,after_local.data(),0,after_local.size());
    t.assert_true(encoded_local == after_local);
    native_pin.reset();
    t.assert_equal(uint64_t(0),native_owner->begin_write(native_owner->identity()));
    {
        auto wrong_owner=local->retain_bounded_storage(611,1,capacity,256);
        if (!t.assert_true(bool(wrong_owner))) return;
        auto wrong=wrong_owner->read(wrong_owner->identity());
        t.assert_true(!local_reader.arm_shared_bounded_reader(wrong));
        t.assert_equal(uint64_t(0),native_owner->begin_write(native_owner->identity()));
    }
    local_reader.release_shared_bounded_reader();
    const auto write=native_owner->begin_write(native_owner->identity());
    if (!t.assert_true(write != 0 && native_owner->finish_write(write,true))) return;
    native_pin=native_owner->read(native_owner->identity());
    t.assert_true(local_reader.arm_shared_bounded_reader(native_pin));
    local_reader.release_shared_bounded_reader();
}

static void test_gemma_deferred_local(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.kv_stream_pool_bytes=64u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get(); params.defer_bounded_cache=true;
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
        8192,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
    auto * local=cache.get_swa();
    if (!t.assert_true(local->bounded_storage_pending() && cache.get_base()->get_kv_stream())) return;
    auto * placeholder=local->get_k_storage(0);
    t.assert_true(placeholder && !placeholder->data && ggml_backend_buffer_get_size(placeholder->buffer) == 0);
    local->clear(true);
    t.assert_true(local->bounded_storage_pending() && local->prepare({}).empty() &&
        local->init_update(nullptr,false)->get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    for (auto id : local->get_layer_ids()) t.assert_true(!local->get_k_storage(int32_t(id))->data &&
        !local->get_v_storage(int32_t(id))->data);
    llama_cparams cp{}; cp.n_ctx_seq=8192; cp.n_ubatch=256; cp.n_seq_max=1; cp.flash_attn=true;
    auto * buft=llama_kv_stream_device_buffer_type(dev);
    llama_memory_bounded_kv_layout quote;
    if (!t.assert_true(llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,cp,buft,{608,1},quote))) return;
    using arena_ptr=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
    using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
    const auto bytes=quote.allocation.bytes, alignment=quote.allocation.alignment;
    arena_ptr arena(ggml_backend_memory_arena_new(buft,bytes+2*alignment),ggml_backend_memory_arena_free);
    if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
            ggml_backend_memory_arena_reserve_at(arena.get(),78,alignment,bytes,alignment,
                GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
    auto * parent=ggml_backend_memory_arena_parent(arena.get());
    ggml_backend_buffer_clear(parent,0x5a);
    ggml_backend_buffer_ptr short_buffer(ggml_backend_buffer_view(parent,alignment,bytes-1));
    arena_ptr short_arena(ggml_backend_memory_arena_new_from_buffer(short_buffer.get()),ggml_backend_memory_arena_free);
    if (!t.assert_true(short_arena && ggml_backend_memory_arena_begin(short_arena.get(),0) &&
            ggml_backend_memory_arena_reserve_at(short_arena.get(),78,0,bytes-1,alignment,
                GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(short_arena.get()))) return;
    lease_ptr short_lease(ggml_backend_memory_arena_acquire(short_arena.get(),78),ggml_backend_memory_lease_free);
    t.assert_true(!local->bind_bounded_storage(short_lease.get()));
    t.assert_true(local->bounded_storage_pending() && local->get_k_storage(0) == placeholder && !placeholder->data);
    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(),78),ggml_backend_memory_lease_free);
    if (!t.assert_true(local->bind_bounded_storage(lease.get()))) return;
    t.assert_true(!local->bounded_storage_pending() && placeholder->data == nullptr);
    const auto base=uintptr_t(ggml_backend_buffer_get_base(parent))+alignment;
    for (const auto & layer : quote.layers) {
        t.assert_equal(base+layer.k_offset,uintptr_t(local->get_k_storage(int32_t(layer.id))->data));
        t.assert_equal(base+layer.v_offset,uintptr_t(local->get_v_storage(int32_t(layer.id))->data));
    }
    std::vector<uint8_t> marker(256,0x39), after(marker.size());
    ggml_backend_tensor_set(local->get_k_storage(0),marker.data(),0,marker.size());
    t.assert_true(local->bind_bounded_storage(lease.get()));
    ggml_backend_tensor_get(local->get_k_storage(0),after.data(),0,after.size());
    t.assert_true(marker == after);
    ggml_tensor raw{}; raw.type=GGML_TYPE_I8;
    raw.ne[0]=int64_t(bytes+2*alignment); raw.ne[1]=raw.ne[2]=raw.ne[3]=1;
    raw.nb[0]=1; raw.nb[1]=raw.nb[2]=raw.nb[3]=size_t(raw.ne[0]);
    raw.buffer=parent; raw.data=ggml_backend_buffer_get_base(parent);
    std::vector<uint8_t> guard(alignment);
    ggml_backend_tensor_get(&raw,guard.data(),0,guard.size());
    t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t value) { return value == 0x5a; }));
    ggml_backend_tensor_get(&raw,guard.data(),alignment+bytes,guard.size());
    t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t value) { return value == 0x5a; }));
    llama_memory_bounded_kv_layout bound;
    t.assert_true(local->describe_bounded_storage(608,1,8192,256,bound));
    t.assert_equal(bytes,bound.allocation.bytes);
}

static void test_gemma_stream_groups(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.kv_stream_pool_bytes=64u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get();
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    try {
        llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
            8192,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
        auto * base=cache.get_base(); auto * local=cache.get_swa();
        auto * stream=base->get_kv_stream();
        if (!t.assert_true(stream && !local->get_kv_stream())) return;
        const auto resources=llama_memory_kv_resources_get(&cache);
        t.assert_true(resources.full == base && resources.bounded == local && !resources.recurrent);
        const auto & host=stream->host()->config();
        t.assert_equal(10u,host.layers);
        t.assert_equal(size_t(8192),host.context_tokens);
        t.assert_equal(int64_t(512),host.shape.head_dim_k);
        t.assert_equal(int64_t(512),host.shape.head_dim_v);
        t.assert_equal(int64_t(4),host.shape.heads);
        t.assert_equal(size_t(272629760),stream->host()->bytes());
        const auto full=base->get_layer_ids();
        t.assert_equal(size_t(10),full.size());
        for (size_t ordinal=0;ordinal<full.size();++ordinal) {
            const uint32_t id=uint32_t(5+6*ordinal);
            t.assert_equal(id,full[ordinal]);
            llama_kv_stream_host_layer planes;
            if (!t.assert_true(stream->host()->layer(uint32_t(ordinal),planes))) return;
            t.assert_true(base->get_k_storage(int32_t(id))->data == planes.k);
            t.assert_true(base->get_v_storage(int32_t(id))->data == planes.v);
            t.assert_true(base->get_k_storage(int32_t(id))->buffer == stream->buffer());
        }
        t.assert_equal(size_t(50),local->get_layer_ids().size());
        t.assert_equal(uint32_t(1280),local->get_size());
        for (auto id : local->get_layer_ids()) {
            auto * tensor=local->get_k_storage(int32_t(id));
            t.assert_true(tensor->buffer != stream->buffer() && !ggml_backend_buffer_is_host(tensor->buffer));
        }
        t.assert_true(stream->bounded_prefill() && stream->tokens() == 0 && stream->complete());
        for (int invalid=0;invalid<3;++invalid) {
            auto bad=params;
            if (invalid == 1) bad.kv_stream_auxiliary_layers=1;
            const llama_memory_i::layer_reuse_cb reuse=invalid == 2 ?
                +[](int32_t il) { return il == 5 ? int32_t(11) : int32_t(-1); } : +[](int32_t) { return int32_t(-1); };
            bool rejected=false;
            try {
                llama_kv_cache_iswa invalid_cache(*model,bad.type_k,bad.type_v,false,true,invalid == 0,true,
                    8192,1,256,256,nullptr,nullptr,reuse,nullptr,&bad);
            } catch (const std::runtime_error & error) {
                const char * expected=invalid == 0 ? "KV streaming requires bounded SWA storage (swa_full=false)" :
                    invalid == 1 ? "Gemma shared-KV drafting does not allocate auxiliary KV histories" :
                    "KV streaming cannot alias selected layers";
                rejected=std::string(error.what()) == expected;
            }
            t.assert_true(rejected);
        }
    } catch (const std::exception & error) {
        t.out << "Gemma private streaming: " << error.what() << '\n';
        t.assert_true(false);
    }
}

static void test_gemma_assistant_source(testing & t, llama_model * model, const char * path,
        bool forced=false, bool compact=false, bool verify=false, bool quotas=false) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4 && path)) return;
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch == LLM_ARCH_GEMMA4_ASSISTANT)) return;
    const size_t width=model->hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(assistant.get()));
    if (!t.assert_true(width == assistant->hparams.n_embd_inp() && width == assistant->hparams.n_embd_out())) return;
    std::vector<std::vector<float>> logits[2],hidden_states[2];
    std::vector<llama_token> proposals[2];
    llama_context_params quota_params{};
    size_t exact_budget=0;
    const int prefix=forced ? 6143 : compact ? 2305 : 1333;
    for (bool streamed : {false,true}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=forced ? 6144 : 4096;
        cp.n_batch=cp.n_ubatch=forced ? 1 : compact ? 32 : 256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? size_t(forced ? 424 : compact ? 480 : 1024)*1048576 : 0;
        cp.kv_stream_verify_max=streamed && verify ? 4 : 0;
        if (streamed && verify) for (int invalid=0;invalid<4;++invalid) {
            auto bad=cp;
            if (invalid == 0) bad.kv_stream_verify_max=5;
            if (invalid == 1) bad.n_ubatch=3;
            if (invalid == 2) bad.shared_device_memory_bytes=0;
            if (invalid == 3) bad.n_seq_max=2;
            bool rejected=false;
            try { llama_context wrong(*model,bad); }
            catch (const std::runtime_error & error) {
                rejected=std::string(error.what()) == "explicit streamed verification requires a serial Gemma4 target and 1-4 queries within n_ubatch";
            }
            if (!t.assert_true("invalid verification declaration rejects before target allocation",rejected)) return;
        }
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(streamed ? "streamed target for actual assistant" : "native target for actual assistant",bool(target))) return;
        llama_set_embeddings_nextn(target.get(),true,false);
        const auto resources=llama_memory_kv_resources_get(llama_get_memory(target.get()));
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent)) return;
        auto * stream=resources.full->get_kv_stream();
        if (streamed && verify) {
            t.assert_equal(uint32_t(0),target->get_cparams().n_rs_seq);
            t.assert_equal(uint32_t(4),target->get_cparams().kv_stream_decode_rows());
            t.assert_equal(uint32_t(4),stream->max_decode_rows());
        }
        auto dp=cp;
        dp.n_batch=dp.n_ubatch=1; dp.shared_device_memory_bytes=0;
        dp.kv_stream_verify_max=0;
        dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
        if (streamed && !forced) {
            const auto revision=target->get_compute_memory()->graph_binding_revision();
            for (int profile=0;profile<10;++profile) {
                auto invalid=dp;
                switch (profile) {
                    case 0: invalid.n_batch=2; break;
                    case 1: invalid.n_ctx+=256; break;
                    case 2: invalid.swa_full=true; break;
                    case 3: invalid.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; break;
                    case 4: invalid.embeddings=true; break;
                    case 5: invalid.offload_kqv=false; break;
                    case 6: invalid.attention_type=LLAMA_ATTENTION_TYPE_NON_CAUSAL; break;
                    case 7: invalid.ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT; break;
                    case 8: invalid.type_k=GGML_TYPE_Q4_0; break;
                    case 9: invalid.type_v=GGML_TYPE_Q8_0; break;
                }
                bool rejected=false;
                try { llama_context wrong(*assistant,invalid); }
                catch (const std::runtime_error & error) {
                    const char * expected=profile < 8 ?
                        "shared Gemma KV reads require a matching serial TG1 MTP context and target phase arena" :
                        "shared KV source has incompatible cache geometry or format";
                    rejected=std::string(error.what()) == expected;
                    if (!rejected) t.out << "unexpected assistant rejection: " << error.what() << '\n';
                }
                if (!t.assert_true("incompatible assistant rejected before changing target",rejected)) return;
                t.assert_true(stream->complete() && stream->tokens() == 0);
                t.assert_equal(uint32_t(0),resources.full->get_cells(0).get_used());
                t.assert_equal(uint32_t(0),resources.bounded->get_cells(0).get_used());
                t.assert_equal(revision,target->get_compute_memory()->graph_binding_revision());
            }
        }
        size_t free=0,total=0;
        ggml_backend_dev_memory(model->devices.front().dev,&free,&total);
        std::fprintf(stderr,"Gemma actual assistant: streamed=%d arena=%zu device_free_before_context=%zu\n",int(streamed),cp.shared_device_memory_bytes,free);
        context_ptr draft(llama_init_from_model(assistant.get(),dp),llama_free);
        if (!t.assert_true(streamed ? "streamed Gemma assistant context admitted" : "native Gemma assistant context admitted",bool(draft))) return;
        if (streamed) {
            const auto * child=draft->get_compute_memory();
            const auto * parent=target->get_compute_memory();
            if (!t.assert_true("Gemma assistant borrows the existing serial target workspace",
                    child && parent && child->borrows_serial_parent() && child->shared_parent() == parent->shared_parent())) return;
            t.assert_true(parent->has_speculative_consumer());
            if (quotas) {
                llama_context_memory_diagnostics usage;
                const auto & config=stream->host()->config();
                llama_kv_stream_policy_config policy;
                policy.shape=config.shape; policy.capabilities=config.capabilities; policy.layers=config.layers;
                size_t minimum=0;
                if (!t.assert_true(parent->diagnostics(usage) && usage.unused_bytes == 0 &&
                        llama_kv_stream_policy_minimum_pool_bytes(policy,minimum).status == llama_kv_stream_policy_status::success &&
                        minimum <= usage.kv_pool_bytes && usage.kv_pool_bytes <= usage.parent_bytes)) return;
                exact_budget=usage.parent_bytes-usage.kv_pool_bytes+minimum;
                quota_params=cp;
                std::fprintf(stderr,"Gemma joint arena quota: required=%zu current=%zu minimum_pool=%zu\n",
                    exact_budget,usage.parent_bytes,minimum);
            }
        }
        llama_set_embeddings_nextn(draft.get(),true,true);
        const auto words=common_tokenize(target.get(),"A woodland stream flows past trees and birds. ",true);
        if (!t.assert_true(!words.empty())) return;
        common_batch prompt(target.get());
        int last_count=0;
        for (int first=0;first<prefix;first+=int(cp.n_ubatch)) {
            prompt.clear(); last_count=std::min(int(cp.n_ubatch),prefix-first);
            for (int i=0;i<last_count;++i) prompt.add(words[size_t(first+i)%words.size()],first+i,0,i == last_count-1);
            if (first < 4) std::fprintf(stderr,"Gemma assistant prefill submit: streamed=%d first=%d rows=%d\n",int(streamed),first,last_count);
            if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            if (compact) llama_synchronize(target.get());
            if (first < 4 || (compact && (first+last_count)%128 == 0) || (first+last_count)%512 == 0 || first+last_count == prefix)
                std::fprintf(stderr,"Gemma assistant control prefill: streamed=%d progress=%d/%d\n",int(streamed),first+last_count,prefix);
        }
        llama_synchronize(target.get());
        const auto * h=llama_get_embeddings_nextn_ith(target.get(),last_count-1);
        const auto * initial=llama_get_logits_ith(target.get(),-1);
        if (!t.assert_true(h && initial)) return;
        std::vector<float> hidden(h,h+width);
        const auto target_hidden=hidden;
        const std::vector<float> target_logits(initial,initial+vocab);
        const llama_token first_token=llama_token(std::max_element(initial,initial+vocab)-initial);
        auto token=first_token;
        const auto full_ids=resources.full->get_layer_ids(),local_ids=resources.bounded->get_layer_ids();
        if (!t.assert_true(!full_ids.empty() && !local_ids.empty())) return;
        const std::array<ggml_tensor *,4> source_tensors{
            resources.full->get_k_storage(int32_t(full_ids.back())),resources.full->get_v_storage(int32_t(full_ids.back())),
            resources.bounded->get_k_storage(int32_t(local_ids.back())),resources.bounded->get_v_storage(int32_t(local_ids.back()))};
        const auto snapshot=[&] {
            std::vector<std::vector<uint8_t>> bytes;
            for (auto * tensor : source_tensors) {
                bytes.emplace_back(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,bytes.back().data(),0,bytes.back().size());
            }
            return bytes;
        };
        const auto before=snapshot();
        const auto full_used=resources.full->get_cells(0).get_used(),local_used=resources.bounded->get_cells(0).get_used();
        if (streamed && !forced) {
            common_batch valid(draft.get());
            const auto id=valid.add(token,prefix,0,true);
            valid.set_embd(id,{hidden.data(),1,width});
            const auto revision=target->get_compute_memory()->graph_binding_revision();
            llama_memory_cache_requirement original;
            if (!t.assert_true(stream->read_source(original))) return;
            for (int profile=0;profile<11;++profile) {
                auto invalid=*valid.get();
                switch (profile) {
                    case 0: invalid.tokens.front().has_embd=false; break;
                    case 1: invalid.tokens.front().id=LLAMA_TOKEN_NULL; break;
                    case 2: invalid.tokens.front().id=llama_token(vocab); break;
                    case 3: invalid.tokens.front().pos[0]=-1; break;
                    case 4: --invalid.tokens.front().pos[0]; break;
                    case 5: ++invalid.tokens.front().pos[0]; break;
                    case 6: invalid.tokens.front().seq_ids={1}; break;
                    case 7: invalid.tokens.front().embd_off=SIZE_MAX; break;
                    case 8: invalid.embd.pop_back(); break;
                    case 9: invalid.n_embd=width-1; break;
                    case 10: invalid.tokens.push_back(invalid.tokens.front()); break;
                }
                if (!t.assert_equal("invalid assistant payload rejects before source acquisition",-1,
                        llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,&invalid))) return;
                llama_memory_cache_requirement unchanged;
                t.assert_true(stream->read_source(unchanged) && unchanged.identity.id == original.identity.id &&
                    unchanged.identity.generation == original.identity.generation);
                t.assert_true("invalid payload leaves all shared source bytes unchanged",snapshot() == before);
                t.assert_equal(full_used,resources.full->get_cells(0).get_used());
                t.assert_equal(local_used,resources.bounded->get_cells(0).get_used());
                t.assert_equal(revision,target->get_compute_memory()->graph_binding_revision());
            }
            llama_memory_clear(llama_get_memory(draft.get()),true);
            t.assert_true("assistant clear cannot clear target KV",snapshot() == before);
            t.assert_equal(full_used,resources.full->get_cells(0).get_used());
            t.assert_equal(local_used,resources.bounded->get_cells(0).get_used());
        }
        for (int step=0;step<3;++step) {
            common_batch batch(draft.get());
            const auto id=batch.add(token,prefix,0,true);
            batch.set_embd(id,{hidden.data(),1,width});
            if (!t.assert_equal("actual assistant forward",0,llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            llama_synchronize(draft.get());
            const auto * values=llama_get_logits_ith(draft.get(),-1);
            const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
            if (!t.assert_true(values && next)) return;
            if (!t.assert_true(std::all_of(values,values+vocab,[](float value) { return std::isfinite(value); }))) return;
            logits[streamed].emplace_back(values,values+vocab);
            hidden_states[streamed].emplace_back(next,next+width);
            token=llama_token(std::max_element(values,values+vocab)-values);
            proposals[streamed].push_back(token);
            hidden.assign(next,next+width);
            const auto * retained_h=llama_get_embeddings_nextn_ith(target.get(),last_count-1);
            const auto * retained_logits=llama_get_logits_ith(target.get(),-1);
            t.assert_true("assistant preserves retained target hidden output",retained_h &&
                !std::memcmp(retained_h,target_hidden.data(),width*sizeof(float)));
            t.assert_true("assistant preserves retained target logits",retained_logits &&
                !std::memcmp(retained_logits,target_logits.data(),vocab*sizeof(float)));
            t.assert_true("assistant leaves source KV bytes unchanged",snapshot() == before);
            t.assert_equal(full_used,resources.full->get_cells(0).get_used());
            t.assert_equal(local_used,resources.bounded->get_cells(0).get_used());
            if (stream) {
                t.assert_true(stream->complete() && stream->tokens() == size_t(prefix));
                if (compact) {
                    llama_context_memory_diagnostics usage;
                    t.assert_true(target->get_compute_memory()->diagnostics(usage) && usage.streaming_active);
                }
                if (forced) {
                    const auto & layout=stream->host()->layout();
                    const size_t layer_bytes=size_t(prefix)*(layout.k_token_bytes+layout.v_token_bytes);
                    std::fprintf(stderr,"Gemma assistant forced ring: step=%d source_bytes=%zu entire_pool=%zu\n",step,layer_bytes,stream->pool_grant_bytes());
                    if (!t.assert_true("actual assistant source exceeds the entire pool",layer_bytes > stream->pool_grant_bytes())) return;
                }
            }
        }
        if (!forced) {
            llama_memory_cache_requirement old_source;
            if (stream && !t.assert_true(stream->read_source(old_source))) return;
            if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,prefix-1,-1))) return;
            if (stream) {
                common_batch stale(draft.get());
                const auto id=stale.add(token,prefix,0,true);
                stale.set_embd(id,{hidden.data(),1,width});
                if (!t.assert_equal("old frontier is not readable after truncation",-1,
                        llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,stale.get()))) return;
            }
            prompt.clear(); prompt.add(words[size_t(prefix-1)%words.size()],prefix-1,0,true);
            if (!t.assert_equal("target rewrites while assistant metadata stays attached",0,
                    llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            h=llama_get_embeddings_nextn_ith(target.get(),0);
            initial=llama_get_logits_ith(target.get(),-1);
            if (!t.assert_true(h && initial)) return;
            hidden.assign(h,h+width);
            token=llama_token(std::max_element(initial,initial+vocab)-initial);
            if (stream) {
                llama_memory_cache_requirement refreshed;
                if (!t.assert_true(stream->read_source(refreshed) && refreshed.identity.id == old_source.identity.id &&
                        refreshed.identity.generation > old_source.identity.generation && stream->tokens() == size_t(prefix))) return;
            }
            const auto rewritten=snapshot();
            common_batch fresh(draft.get());
            const auto id=fresh.add(token,prefix,0,true);
            fresh.set_embd(id,{hidden.data(),1,width});
            if (!t.assert_equal("assistant rebinds the new target generation",0,
                    llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,fresh.get()))) return;
            const auto * values=llama_get_logits_ith(draft.get(),-1);
            const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
            if (!t.assert_true(values && next)) return;
            logits[streamed].emplace_back(values,values+vocab);
            hidden_states[streamed].emplace_back(next,next+width);
            t.assert_true("refreshed assistant read preserves rewritten sources",snapshot() == rewritten);
        }
        if (verify) {
            llama_set_kv_stream_decode(target.get(),true);
            if (stream) {
                const auto unchanged=snapshot();
                llama_memory_cache_requirement before_source,after_source;
                if (!t.assert_true(stream->read_source(before_source))) return;
                prompt.clear();
                for (int row=0;row<5;++row) prompt.add(words[size_t(row)%words.size()],prefix+row,0,true);
                if (!t.assert_equal("verification wider than the declared profile rejects before execution",-1,
                        llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
                t.assert_equal(size_t(prefix),stream->tokens());
                t.assert_true(stream->read_source(after_source) && after_source.identity.id == before_source.identity.id &&
                    after_source.identity.generation == before_source.identity.generation);
                t.assert_true(snapshot() == unchanged);
            }
            for (int count=1;count<=4;++count) {
                prompt.clear();
                for (int row=0;row<count;++row) prompt.add(words[size_t(row)%words.size()],prefix+row,0,true);
                if (!t.assert_equal("declared target verification width executes",0,
                        llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
                for (int row=0;row<count;++row) {
                    const auto * values=llama_get_logits_ith(target.get(),row);
                    const auto * next=llama_get_embeddings_nextn_ith(target.get(),row);
                    if (!t.assert_true(values && next)) return;
                    logits[streamed].emplace_back(values,values+vocab);
                    hidden_states[streamed].emplace_back(next,next+width);
                    if (row+1 == count) {
                        hidden.assign(next,next+width);
                        token=llama_token(std::max_element(values,values+vocab)-values);
                    }
                }
                if (stream && !t.assert_equal(size_t(prefix+count),stream->tokens())) return;
                const auto verified=snapshot();
                common_batch read(draft.get());
                const auto id=read.add(token,prefix+count,0,true);
                read.set_embd(id,{hidden.data(),1,width});
                if (!t.assert_equal("assistant uses the completed verification frontier",0,
                        llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,read.get()))) return;
                const auto * values=llama_get_logits_ith(draft.get(),-1);
                const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
                if (!t.assert_true(values && next)) return;
                logits[streamed].emplace_back(values,values+vocab);
                hidden_states[streamed].emplace_back(next,next+width);
                t.assert_true("post-verification assistant leaves sources unchanged",snapshot() == verified);
                if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,prefix,-1))) return;
            }
        }
        draft.reset();
        llama_set_kv_stream_decode(target.get(),true);
        prompt.clear(); prompt.add(first_token,prefix,0,true);
        if (!t.assert_equal("target resumes after reader destruction",0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
        const auto * values=llama_get_logits_ith(target.get(),-1);
        if (!t.assert_true(values != nullptr)) return;
        logits[streamed].emplace_back(values,values+vocab);
        std::fprintf(stderr,"Gemma actual assistant control complete: streamed=%d predictions=3 verify_max=%d\n",int(streamed),verify ? 4 : 0);
    }
    if (quotas) {
        if (!t.assert_true(exact_budget > 1)) return;
        for (bool sufficient : {false,true}) {
            auto cp=quota_params; cp.shared_device_memory_bytes=exact_budget-(sufficient ? 0 : 1);
            context_ptr target(nullptr,llama_free), draft(nullptr,llama_free);
            std::string error;
            try {
                target.reset(new llama_context(*model,cp));
                auto dp=cp; dp.n_batch=dp.n_ubatch=1;
                dp.shared_device_memory_bytes=0; dp.kv_stream_verify_max=0;
                dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
                draft.reset(new llama_context(*assistant,dp));
            } catch (const std::runtime_error & failure) { error=failure.what(); }
            if (!sufficient) {
                const auto fields=error.find("required=");
                size_t required=0,available=0,additional=0;
                if (!t.assert_true("one-byte-short joint arena rejects without private workspace fallback",
                        error.find("persistent local cache does not fit the declared shared arena: insufficient arena quota;")==0 &&
                        fields!=std::string::npos && std::sscanf(error.c_str()+fields,
                            "required=%zu, available=%zu, additional=%zu",&required,&available,&additional)==3)) {
                    t.out << "one-byte-short joint arena error: " << error << '\n'; return;
                }
                t.assert_equal(exact_budget,required);
                t.assert_equal(exact_budget-1,available);
                t.assert_equal(size_t(1),additional);
                continue;
            }
            if (!t.assert_true("exact joint arena admits both contexts",error.empty() && target && draft)) {
                t.out << "exact joint arena error: " << error << '\n'; return;
            }
            t.assert_equal(exact_budget,target->get_compute_memory()->shared_parent_capacity());
            t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
            const auto words=common_tokenize(target.get(),"A woodland stream flows past trees. ",true);
            if (!t.assert_true(!words.empty())) return;
            common_batch prompt(target.get());
            for (int row=0;row<16;++row) prompt.add(words[size_t(row)%words.size()],row,0,row == 15);
            if (!t.assert_equal("exact joint arena short prefill",0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            const auto * h=llama_get_embeddings_nextn_ith(target.get(),15);
            const auto * values=llama_get_logits_ith(target.get(),-1);
            if (!t.assert_true(h && values)) return;
            common_batch read(draft.get());
            const auto id=read.add(llama_token(std::max_element(values,values+vocab)-values),16,0,true);
            read.set_embd(id,{h,1,width});
            t.assert_equal("exact joint arena assistant handoff",0,llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,read.get()));
        }
    }
    t.assert_true(proposals[0] == proposals[1]);
    if (!t.assert_equal(logits[0].size(),logits[1].size())) return;
    for (size_t row=0;row<logits[0].size();++row) t.assert_true("actual assistant/target logits match native bits",
        std::memcmp(logits[0][row].data(),logits[1][row].data(),vocab*sizeof(float)) == 0);
    for (size_t row=0;row<hidden_states[0].size();++row) t.assert_true("actual assistant hidden rows match native bits",
        std::memcmp(hidden_states[0][row].data(),hidden_states[1][row].data(),width*sizeof(float)) == 0);
}

// Observe the real graph inputs and every assistant block without changing the draft loop.
struct gemma_candidate_trace {
    struct record {
        std::string name;
        std::vector<uint8_t> bytes;
        bool operator==(const record & other) const { return name == other.name && bytes == other.bytes; }
    };
    std::vector<record> records;
    bool valid=true;

    void capture(const char * name,ggml_tensor * tensor,ggml_type type) {
        if (!tensor || tensor->type != type || !ggml_is_contiguous(tensor)) {
            std::fprintf(stderr,"Gemma candidate trace cannot read %s: tensor=%s expected_type=%d actual_type=%d\n",
                name,tensor ? tensor->name : "null",int(type),tensor ? int(tensor->type) : -1);
            valid=false; return;
        }
        records.push_back({name,std::vector<uint8_t>(ggml_nbytes(tensor))});
        ggml_backend_tensor_get(tensor,records.back().bytes.data(),0,records.back().bytes.size());
    }
    static bool callback(ggml_tensor * tensor,bool ask,void * opaque) {
        auto & trace=*static_cast<gemma_candidate_trace *>(opaque);
        const std::string name=tensor->name;
        const bool token_input=tensor->op == GGML_OP_GET_ROWS && tensor->src[1] &&
            !std::strncmp(tensor->src[1]->name,"inp_tokens",10);
        const bool wanted=token_input || name == "inp_xh" || name == "Qcur_pos-0" || name == "result_output" ||
            name == "h_nextn" || name == "out_scaled-0" || name == "out_scaled-1" ||
            name == "out_scaled-2" || name == "out_scaled-3";
        if (!wanted) return false;
        if (ask) return true;
        if (token_input) {
            trace.capture("tokens",tensor->src[1],GGML_TYPE_I32);
        } else if (name == "inp_xh") {
            trace.capture("hidden",tensor->src[1],GGML_TYPE_F32);
        } else if (name == "Qcur_pos-0") {
            trace.capture("position",tensor->src[1],GGML_TYPE_I32);
        } else {
            trace.capture(name.c_str(),tensor,GGML_TYPE_F32);
        }
        return true;
    }
};

// Match the common driver to explicit sequential stock predictions and frozen target sources.
static void test_gemma_candidate_loop(testing & t,llama_model * model,const char * path,bool compact) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4 && path)) return;
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch == LLM_ARCH_GEMMA4_ASSISTANT && assistant->hparams.n_layer_nextn == 4)) return;
    const size_t width=model->hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(assistant.get()));
    kv_stream_bit_oracle oracle;
    for (bool streamed : {false,true}) {
        if (streamed) oracle.compare();
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=compact ? 32 : 256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? size_t(compact ? 480 : 1024)*1048576 : 0;
        cp.kv_stream_verify_max=streamed ? 4 : 0;
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(target))) return;
        gemma_candidate_trace trace;
        auto dp=cp; dp.n_batch=dp.n_ubatch=1;
        dp.shared_device_memory_bytes=0; dp.kv_stream_verify_max=0;
        dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
        dp.cb_eval=gemma_candidate_trace::callback; dp.cb_eval_user_data=&trace;
        context_ptr draft(llama_init_from_model(assistant.get(),dp),llama_free);
        if (!t.assert_true(bool(draft))) return;
        llama_set_embeddings_nextn(target.get(),true,false);
        llama_set_embeddings_nextn(draft.get(),true,true);
        const auto words=common_tokenize(target.get(),"A woodland stream flows past trees and birds. ",true);
        if (!t.assert_true(!words.empty())) return;
        const int prefix=compact ? 2305 : 1333;
        llama_tokens prompt;
        common_batch batch(target.get());
        for (int first=0;first<prefix;first+=int(cp.n_ubatch)) {
            batch.clear();
            const int count=std::min(int(cp.n_ubatch),prefix-first);
            for (int row=0;row<count;++row) {
                const auto token=words[size_t(first+row)%words.size()];
                prompt.push_back(token); batch.add(token,first+row,0,row+1 == count);
            }
            if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
        llama_set_kv_stream_decode(target.get(),true);
        const auto * h=llama_get_embeddings_nextn_ith(target.get(),batch.size()-1);
        const auto * logits=llama_get_logits_ith(target.get(),-1);
        if (!t.assert_true(h && logits)) return;
        const std::vector<float> hidden(h,h+width);
        const llama_token anchor=llama_token(std::max_element(logits,logits+vocab)-logits);
        const auto resources=llama_memory_kv_resources_get(llama_get_memory(target.get()));
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent)) return;
        const auto snapshot=[&] {
            std::vector<std::vector<uint8_t>> bytes;
            for (auto * cache : {resources.full,resources.bounded}) {
                const auto ids=cache->get_layer_ids();
                for (auto * tensor : {cache->get_k_storage(int32_t(ids.back())),cache->get_v_storage(int32_t(ids.back()))}) {
                    bytes.emplace_back(ggml_nbytes(tensor));
                    ggml_backend_tensor_get(tensor,bytes.back().data(),0,bytes.back().size());
                }
            }
            return bytes;
        };
        const auto source_bytes=snapshot();
        auto * physical_parent=target->get_compute_memory()->shared_parent();
        llama_memory_cache_requirement source;
        auto * stream=resources.full->get_kv_stream();
        if (stream && !t.assert_true(stream->read_source(source))) return;
        trace.records.clear();
        llama_tokens explicit_tokens;
        auto token=anchor;
        auto next_hidden=hidden;
        for (int step=0;step<3;++step) {
            common_batch read(draft.get());
            const auto id=read.add(token,prefix,0,true);
            read.set_embd(id,{next_hidden.data(),1,width});
            if (!t.assert_equal(0,llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,read.get()))) return;
            const auto * values=llama_get_logits_ith(draft.get(),-1);
            const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
            if (!t.assert_true(values && next)) return;
            const size_t offset=size_t(step)*9;
            if (!t.assert_true(trace.valid && trace.records.size() == offset+9)) return;
            const auto & token_record=trace.records[offset];
            const auto & hidden_record=trace.records[offset+1];
            const auto & position_record=trace.records[offset+2];
            const llama_pos position=prefix;
            t.assert_true("observed token is the current sequential prediction",token_record.name == "tokens" &&
                kv_stream_bytes_equal(&token,sizeof(token),token_record.bytes.data(),token_record.bytes.size()));
            t.assert_true("observed hidden state evolves from the previous prediction",hidden_record.name == "hidden" &&
                kv_stream_bytes_equal(next_hidden.data(),width*sizeof(float),hidden_record.bytes.data(),hidden_record.bytes.size()));
            t.assert_true("all predictions use the fixed target frontier",position_record.name == "position" &&
                kv_stream_bytes_equal(&position,sizeof(position),position_record.bytes.data(),position_record.bytes.size()));
            for (size_t layer=0;layer<4;++layer) t.assert_equal("out_scaled-"+std::to_string(layer),trace.records[offset+3+layer].name);
            token=llama_token(std::max_element(values,values+vocab)-values);
            explicit_tokens.push_back(token); next_hidden.assign(next,next+width);
        }
        if (!trace.valid || trace.records.size() != 27) {
            t.out << "Gemma candidate trace valid=" << trace.valid << " records=" << trace.records.size() << '\n';
            for (const auto & record : trace.records) t.out << record.name << " bytes=" << record.bytes.size() << '\n';
        }
        if (!t.assert_true("three sequential passes observe all inputs, four blocks and outputs",trace.valid && trace.records.size() == 27)) return;
        const auto explicit_trace=trace.records;
        for (bool backend : {false,true}) for (int depth=1;depth<=3;++depth) for (int scenario=0;scenario<4;++scenario) {
            common_params_speculative params;
            params.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params.draft.ctx_tgt=target.get(); params.draft.ctx_dft=draft.get();
            params.draft.n_max=depth; params.draft.backend_sampling=backend;
            params.draft.p_min=scenario == 1 ? 1.0f : 0.0f;
            params.draft.n_min=scenario == 2 ? depth+1 : 0;
            common_speculative_ptr spec(common_speculative_init(params,1));
            if (!t.assert_true(bool(spec))) return;
            trace.records.clear();
            common_speculative_begin(spec.get(),0,prompt);
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            if (!t.assert_true("shared KV process performs no assistant catch-up",trace.records.empty())) return;
            std::vector<uint8_t> saved;
            if (!t.assert_true(common_speculative_get_state(spec.get(),0,saved) && saved.size() == sizeof(uint32_t)+width*sizeof(float))) return;
            if (!t.assert_true("draft carry starts at the target's final hidden row",!std::memcmp(saved.data()+sizeof(uint32_t),hidden.data(),width*sizeof(float)))) return;
            auto stale=saved;
            std::fill(stale.begin()+sizeof(uint32_t),stale.end(),0);
            common_speculative_set_state(spec.get(),0,stale);
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            std::vector<uint8_t> refreshed;
            if (!t.assert_true("processing current target outputs replaces stale draft carry",
                    common_speculative_get_state(spec.get(),0,refreshed) && refreshed == saved && trace.records.empty())) return;
            for (int repeat=0;repeat<2;++repeat) {
                if (repeat) common_speculative_set_state(spec.get(),0,saved);
                trace.records.clear();
                llama_tokens result;
                auto & request=common_speculative_get_draft_params(spec.get(),0);
                request.drafting=true; request.n_max=scenario == 3 ? 1 : -1;
                request.pos0=prefix; request.id_last=anchor; request.prompt=&prompt; request.result=&result;
                common_speculative_draft(spec.get());
                if (streamed && !t.assert_true("sampler graph changes keep the same shared physical parent",
                        target->get_compute_memory()->shared_parent() == physical_parent &&
                        draft->get_compute_memory()->borrows_serial_parent() &&
                        draft->get_compute_memory()->shared_parent() == physical_parent)) return;
                const size_t calls=scenario == 1 ? 1 : size_t(depth);
                const size_t returned=(scenario == 1 || scenario == 2) ? 0 : scenario == 3 ? 1 : size_t(depth);
                if (!t.assert_true("stock driver observes the explicit fixed-position token/hidden chain",trace.valid &&
                        trace.records.size() == calls*9 && std::equal(trace.records.begin(),trace.records.end(),explicit_trace.begin()))) return;
                if (!t.assert_true("configured length, confidence and minimum-length rules match stock",result ==
                        llama_tokens(explicit_tokens.begin(),explicit_tokens.begin()+returned))) return;
                if (!t.assert_true("all draft passes leave target KV unchanged",snapshot() == source_bytes)) return;
                if (stream) {
                    llama_memory_cache_requirement after;
                    t.assert_true(stream->read_source(after) && after.identity.id == source.identity.id &&
                        after.identity.generation == source.identity.generation && stream->tokens() == size_t(prefix));
                    if (compact) {
                        llama_context_memory_diagnostics usage;
                        t.assert_true("candidate memory profile exercises the ring",target->get_compute_memory()->diagnostics(usage) && usage.streaming_active);
                    }
                }
                const std::string label=std::to_string(int(backend))+"/"+std::to_string(depth)+"/"+std::to_string(scenario)+"/"+std::to_string(repeat);
                oracle.bytes(label+"/candidates",result.data(),result.size()*sizeof(llama_token));
                for (size_t row=0;row<trace.records.size();++row) {
                    const auto & record=trace.records[row];
                    oracle.bytes(label+"/"+std::to_string(row)+"/"+record.name,record.bytes.data(),record.bytes.size());
                }
            }
        }
        if (streamed) oracle.finish();
        std::fprintf(stderr,"Gemma candidate loop: streamed=%d matched_records=%zu bytes=%llu\n",int(streamed),oracle.records(),(unsigned long long)oracle.byte_count());
    }
}

// Exercise the server's accepted-prefix contract with real target writes and the stock MTP driver.
static void test_gemma_acceptance(testing & t,llama_model * model,const char * path,bool compact) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4 && path)) return;
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch == LLM_ARCH_GEMMA4_ASSISTANT)) return;
    const size_t width=model->hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    kv_stream_bit_oracle oracle;
    for (bool streamed : {false,true}) {
        if (streamed) oracle.compare();
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=compact ? 32 : 256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? size_t(compact ? 480 : 1024)*1048576 : 0;
        cp.kv_stream_verify_max=streamed ? 4 : 0;
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(target))) return;
        auto dp=cp; dp.n_batch=dp.n_ubatch=1;
        dp.shared_device_memory_bytes=0; dp.kv_stream_verify_max=0;
        dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
        context_ptr draft(llama_init_from_model(assistant.get(),dp),llama_free);
        if (!t.assert_true(bool(draft))) return;
        llama_set_embeddings_nextn(target.get(),true,false);
        llama_set_embeddings_nextn(draft.get(),true,true);
        const auto resources=llama_memory_kv_resources_get(llama_get_memory(target.get()));
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent)) return;
        auto * stream=resources.full->get_kv_stream();
        auto * physical_parent=target->get_compute_memory()->shared_parent();
        const auto words=common_tokenize(target.get(),"A woodland stream flows past trees and birds. ",true);
        if (!t.assert_true(!words.empty())) return;
        llama_tokens history;
        common_batch batch(target.get());
        common_speculative_ptr spec;
        const auto carry=[&] {
            std::vector<uint8_t> bytes;
            if (!common_speculative_get_state(spec.get(),0,bytes)) throw std::runtime_error("missing Gemma carry state");
            return bytes;
        };
        const auto canonical=[&](kv_stream_bit_oracle & out,const std::string & label) {
            llama_synchronize(target.get()); llama_synchronize(draft.get());
            const size_t frontier=history.size();
            for (auto * cache : {resources.full,resources.bounded}) {
                const bool local=cache == resources.bounded;
                const auto tag=label+(local ? "/local" : "/full");
                const size_t first=local && frontier > model->hparams.n_swa ? frontier-model->hparams.n_swa : 0;
                const auto & cells=cache->get_cells(0);
                std::vector<int64_t> metadata;
                const bool extended=model->hparams.n_pos_per_embd() > 1 || model->hparams.ple_n_heads > 0;
                for (const auto & item : cells.seq_pos_get(0)) {
                    if (item.first < 0 || size_t(item.first) >= frontier) throw std::runtime_error("Gemma cache retained a rejected suffix");
                    if (size_t(item.first) < first) continue;
                    metadata.push_back(item.first);
                    if (extended) {
                        const auto & ext=cells.ext_get(item.second);
                        metadata.insert(metadata.end(),{ext.x,ext.y,ext.tok});
                    }
                }
                if (metadata.size() != (frontier-first)*(extended ? 4 : 1)) throw std::runtime_error("incomplete Gemma visible cache metadata");
                out.bytes(tag+"/positions",metadata.data(),metadata.size()*sizeof(int64_t));
                kv_stream_exact_kv(out,tag,cache,first,frontier-first);
            }
        };
        const auto submit=[&](const std::string & label,const llama_tokens & tokens,bool decode) {
            batch.clear();
            for (size_t row=0;row<tokens.size();++row) batch.add(tokens[row],llama_pos(history.size()+row),0,decode || row+1 == tokens.size());
            llama_set_kv_stream_decode(target.get(),decode);
            if (!t.assert_equal(label,0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return false;
            if (spec && !t.assert_true(common_speculative_process(spec.get(),batch))) return false;
            history.insert(history.end(),tokens.begin(),tokens.end());
            if (decode) for (size_t row=0;row<tokens.size();++row) {
                const auto * values=llama_get_logits_ith(target.get(),int(row));
                const auto * hidden=llama_get_embeddings_nextn_ith(target.get(),int(row));
                if (!t.assert_true(values && hidden)) return false;
                oracle.bytes(label+"/logits/"+std::to_string(row),values,vocab*sizeof(float));
                oracle.bytes(label+"/hidden/"+std::to_string(row),hidden,width*sizeof(float));
            }
            return true;
        };
        const auto prefill=[&](size_t frontier,size_t salt) {
            while (history.size() < frontier) {
                llama_tokens tokens(std::min(size_t(cp.n_ubatch),frontier-history.size()));
                for (size_t row=0;row<tokens.size();++row) tokens[row]=words[(history.size()+row+salt)%words.size()];
                if (!submit("prefill/"+std::to_string(history.size()),tokens,false)) return false;
            }
            return true;
        };
        if (!prefill(compact ? 2303 : 1279,0)) return;
        llama_token next=0;
        const auto sample=[&](int row) {
            const auto * values=llama_get_logits_ith(target.get(),row);
            if (!values) throw std::runtime_error("missing Gemma correction logits");
            return llama_token(std::max_element(values,values+vocab)-values);
        };
        next=sample(-1);
        const auto predict=[&](const std::string & label,int depth,llama_token anchor) {
            llama_tokens result;
            common_speculative_get_draft_params(spec.get(),0)={true,depth,llama_pos(history.size()),anchor,&history,&result,nullptr,0.0f,123};
            common_speculative_draft(spec.get());
            if (result.size() != size_t(depth)) throw std::runtime_error("incomplete Gemma acceptance draft");
            oracle.bytes(label,result.data(),result.size()*sizeof(llama_token));
            return result;
        };
        for (int depth=1;depth<=3;++depth) {
            spec.reset();
            common_params_speculative params;
            params.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params.draft.ctx_tgt=target.get(); params.draft.ctx_dft=draft.get();
            params.draft.n_max=depth; params.draft.backend_sampling=true;
            spec.reset(common_speculative_init(params,1));
            if (!t.assert_true(bool(spec))) return;
            common_speculative_begin(spec.get(),0,history);
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            for (int accepted=0;accepted<=depth;++accepted) {
                const std::string label="depth/"+std::to_string(depth)+"/accept/"+std::to_string(accepted);
                const size_t first=history.size();
                const auto proposed=predict(label+"/proposals",depth,next);
                llama_tokens verified{next}; verified.insert(verified.end(),proposed.begin(),proposed.end());
                if (!submit(label+"/verify",verified,true)) return;
                const auto correction=sample(accepted);
                const auto * hidden=llama_get_embeddings_nextn_ith(target.get(),accepted);
                if (!t.assert_true(hidden != nullptr)) return;
                const std::vector<float> selected(hidden,hidden+width);
                common_speculative_accept(spec.get(),0,uint16_t(accepted));
                const auto selected_carry=carry();
                if (!t.assert_true("acceptance chooses the matching target hidden row",selected_carry.size() == sizeof(uint32_t)+width*sizeof(float) &&
                        !std::memcmp(selected_carry.data()+sizeof(uint32_t),selected.data(),width*sizeof(float)))) return;
                const size_t frontier=first+size_t(accepted)+1;
                if (!t.assert_true("accepted prefix truncation succeeds",llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(frontier),-1))) return;
                history.resize(frontier);
                canonical(oracle,label+"/accepted-cache");
                oracle.bytes(label+"/accepted-carry",selected_carry.data(),selected_carry.size());
                (void)predict(label+"/after-accept",depth,correction);
                if (!submit(label+"/correction",{correction},true)) return;
                next=sample(-1);
                if (stream) {
                    t.assert_equal(history.size(),stream->tokens());
                    t.assert_true(target->get_compute_memory()->shared_parent() == physical_parent &&
                        draft->get_compute_memory()->shared_parent() == physical_parent);
                    if (compact) {
                        llama_context_memory_diagnostics usage;
                        t.assert_true("acceptance profile exercises streaming",target->get_compute_memory()->diagnostics(usage) && usage.streaming_active);
                    }
                }
            }
            const std::string label="depth/"+std::to_string(depth)+"/cancel";
            const size_t first=history.size();
            const auto saved_carry=carry();
            common_prompt_checkpoint checkpoint;
            checkpoint.update_tgt(target.get(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            kv_stream_bit_oracle cancelled;
            canonical(cancelled,"entry");
            const auto proposed=predict(label+"/proposals",depth,next);
            llama_tokens verified{next}; verified.insert(verified.end(),proposed.begin(),proposed.end());
            if (!submit(label+"/provisional",verified,true)) return;
            if (!t.assert_true("cancelled verification restores the saved target prefix",
                    checkpoint.load_tgt(target.get(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
                    llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(first),-1))) return;
            history.resize(first);
            common_speculative_set_state(spec.get(),0,saved_carry);
            cancelled.compare(); canonical(cancelled,"entry"); cancelled.finish();
            if (!t.assert_true("cancellation retry produces the same draft",predict(label+"/retry",depth,next) == proposed)) return;
            canonical(oracle,label+"/restored-cache");
            if (!submit(label+"/resume",{next},true)) return;
            next=sample(-1);
        }
        // Match the server's extra-token headroom before the final speculative round.
        if (!prefill(cp.n_ctx-3,0)) return;
        next=sample(-1);
        kv_stream_bit_oracle boundary;
        canonical(boundary,"end-prefix");
        batch.clear();
        for (int row=0;row<4;++row) batch.add(words[size_t(row)%words.size()],llama_pos(history.size()+row),0,true);
        llama_set_kv_stream_decode(target.get(),true);
        if (!t.assert_true("oversized final verification rejects before mutation",llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()) != 0)) return;
        boundary.compare(); canonical(boundary,"end-prefix"); boundary.finish();
        const auto proposed=predict("end/proposals",1,next);
        if (!submit("end/verify",{next,proposed.front()},true)) return;
        common_speculative_accept(spec.get(),0,1);
        if (!submit("end/correction",{sample(-1)},true)) return;
        t.assert_equal(size_t(cp.n_ctx),history.size());
        const auto empty_requests=history;
        common_speculative_get_draft_params(spec.get(),0).drafting=false;
        common_speculative_draft(spec.get());
        t.assert_true("no drafting is scheduled once context headroom is exhausted",history == empty_requests);
        for (size_t request=0;request<2;++request) {
            llama_memory_clear(llama_get_memory(target.get()),true);
            llama_memory_clear(llama_get_memory(draft.get()),true);
            history.clear();
            const llama_tokens short_prompt(17,words[(request+1)%words.size()]);
            common_speculative_begin(spec.get(),0,short_prompt);
            const auto empty=carry();
            t.assert_true("new empty request resets hidden carry",empty.size() == sizeof(uint32_t)+width*sizeof(float) &&
                std::all_of(empty.begin()+sizeof(uint32_t),empty.end(),[](uint8_t value) { return value == 0; }));
            if (!submit("request/"+std::to_string(request)+"/prefill",short_prompt,false)) return;
            canonical(oracle,"request/"+std::to_string(request)+"/cache");
            const auto proposed=predict("request/"+std::to_string(request)+"/proposals",3,sample(-1));
            t.assert_equal(size_t(3),proposed.size());
            const auto request_carry=carry();
            oracle.bytes("request/"+std::to_string(request)+"/carry",request_carry.data(),request_carry.size());
        }
        spec.reset();
        if (streamed) oracle.finish();
        std::fprintf(stderr,"Gemma acceptance: streamed=%d records=%zu bytes=%llu\n",int(streamed),oracle.records(),(unsigned long long)oracle.byte_count());
    }
}

struct warmup_compute_failure {
    inline static warmup_compute_failure * active=nullptr;
    ggml_backend_dev_t device;
    decltype(ggml_backend_device_i::init_backend) initialize;
    std::map<ggml_backend_t,decltype(ggml_backend_i::graph_compute)> compute;
    ggml_status status;
    size_t calls=0;
    warmup_compute_failure(ggml_backend_dev_t device,ggml_status status) :
        device(device),initialize(device->iface.init_backend),status(status) {
        GGML_ASSERT(!active); active=this;
        device->iface.init_backend=[](ggml_backend_dev_t dev,const char * params) {
            auto * backend=active->initialize(dev,params);
            if (backend) {
                active->compute[backend]=backend->iface.graph_compute;
                backend->iface.graph_compute=[](ggml_backend_t backend,ggml_cgraph *) {
                    backend->iface.graph_compute=active->compute.at(backend);
                    ++active->calls;
                    return active->status;
                };
            }
            return backend;
        };
    }
    ~warmup_compute_failure() { device->iface.init_backend=initialize; active=nullptr; }
};

static void test_gemma_warmup_failures(testing & t,const char * target_path,int mode=-1,const char * logit_path=nullptr) {
    auto * reg=ggml_backend_reg_by_name("CUDA");
    if (!t.assert_true(reg && ggml_backend_reg_dev_count(reg))) return;
    auto * device=ggml_backend_reg_dev_get(reg,0);
    std::vector<float> controls[2];
    for (bool streamed : {false,true}) {
        if (mode>=0 && int(streamed)!=mode) continue;
        common_params params;
        params.model.path=target_path;
        params.n_ctx=4096; params.n_batch=params.n_ubatch=256; params.n_parallel=1;
        params.n_gpu_layers=999; params.fit_params=false; params.swa_full=false;
        params.shared_device_memory_bytes=streamed ? size_t(1024)*1048576 : 0;
        params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.cache_type_k=GGML_TYPE_Q8_0; params.cache_type_v=GGML_TYPE_Q4_0;
        params.cpuparams.n_threads=params.cpuparams_batch.n_threads=8;
        params.warmup=true;
        for (auto status : {GGML_STATUS_ABORTED,GGML_STATUS_ALLOC_FAILED,GGML_STATUS_FAILED}) {
            warmup_compute_failure fault(device,status);
            auto result=common_init_from_params(params);
            t.out << "warmup failure streamed=" << streamed << " status=" << status << " calls=" << fault.calls << '\n';
            if (!t.assert_equal("warmup reaches the injected backend failure",size_t(1),fault.calls)) return;
            t.assert_true("warmup failure retains the initialization result and loaded model",result && result->model());
            t.assert_true("failed warmup must not publish a usable context",result && !result->context());
        }
        auto retry=common_init_from_params(params);
        if (!t.assert_true("startup succeeds after failed warmup teardown",retry && retry->context())) return;
        t.assert_equal("successful warmup leaves no request cells",llama_pos(-1),
            llama_memory_seq_pos_max(llama_get_memory(retry->context()),0));
        const auto words=common_tokenize(retry->context(),"A woodland stream flows past trees. ",true);
        if (!t.assert_true(!words.empty())) return;
        common_batch prompt(retry->context());
        for (int row=0;row<2;++row) prompt.add(words[size_t(row)%words.size()],row,0,row==1);
        if (!t.assert_equal("fresh request computes after failed startup cleanup",0,
                llama_process(retry->context(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
        const auto * logits=llama_get_logits_ith(retry->context(),-1);
        const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(retry->model()));
        if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }))) return;
        controls[streamed].assign(logits,logits+vocab);
        retry.reset();
        for (bool model_only : {false,true}) {
            params.warmup=model_only;
            warmup_compute_failure fault(device,GGML_STATUS_FAILED);
            auto bypass=common_init_from_params(params,model_only);
            t.assert_true("disabled warmup and model-only loading do not execute a graph",bypass && bypass->model() &&
                bool(bypass->context())!=model_only);
            t.assert_equal(size_t(0),fault.calls);
        }
    }
    if (mode<0) {
        t.assert_true("recovered native and streamed startup produce exact logits",kv_stream_bytes_equal(
            controls[0].data(),controls[0].size()*sizeof(float),controls[1].data(),controls[1].size()*sizeof(float)));
    } else {
        if (!t.assert_true(logit_path && !controls[mode].empty())) return;
        const std::array<uint64_t,8> profile{0x474d344c4f474954ull,1,4096,2,1,controls[mode].size(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(logit_path,mode==0 ? "wb" : "rb"));
        if (!t.assert_true(bool(file))) return;
        std::vector<std::vector<float>> values;
        if (mode==0) values.push_back(controls[0]);
        gemma_logits_file(file.get(),mode==0,profile,values);
        if (mode==1) t.assert_true("fresh-process recovered logits match every native bit",values.size()==1 &&
            kv_stream_bytes_equal(values[0].data(),values[0].size()*sizeof(float),controls[1].data(),controls[1].size()*sizeof(float)));
    }
}

static void test_gemma_common_startup(testing & t,const char * target_path,const char * assistant_path) {
    common_params params;
    params.model.path=target_path;
    params.n_ctx=4096; params.n_batch=params.n_ubatch=256; params.n_parallel=1;
    params.n_gpu_layers=999; params.fit_params=false; params.swa_full=false;
    params.shared_device_memory_bytes=size_t(1024)*1048576;
    params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_type_k=GGML_TYPE_Q8_0; params.cache_type_v=GGML_TYPE_Q4_0;
    params.cpuparams.n_threads=params.cpuparams_batch.n_threads=8;
    params.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max=3;
    params.speculative.draft.n_gpu_layers=999;
    params.speculative.draft.mparams.path=assistant_path;
    auto main=common_init_from_params(params);
    auto * target=main ? main->context() : nullptr;
    if (!t.assert_true("common Gemma target startup succeeds",target != nullptr)) return;
    const auto * parent=target->get_compute_memory();
    if (!t.assert_true(parent && parent->serial_ready())) return;
    t.assert_equal(uint32_t(0),target->get_cparams().n_rs_seq);
    t.assert_equal(uint32_t(4),target->get_cparams().kv_stream_verify_max);
    struct load_probe {
        const llama_context_memory * parent;
        std::atomic<uint32_t> observed{0};
    } probe{parent};
    auto draft_params=common_base_params_to_speculative(params);
    draft_params.load_progress_callback=[](float,void * opaque) {
        auto & p=*static_cast<load_probe *>(opaque);
        p.observed.fetch_or(p.parent->serial_ready() ? 3u : 1u,std::memory_order_relaxed);
        return true;
    };
    draft_params.load_progress_callback_user_data=&probe;
    auto speculative=common_speculative_init_from_params(draft_params,main->model(),target);
    auto * draft=speculative ? speculative->context() : nullptr;
    if (!t.assert_true("common Gemma assistant startup succeeds",draft != nullptr)) return;
    auto * full=llama_memory_kv_resources_get(llama_get_memory(target)).full;
    auto * reader_source=full ? full->get_kv_stream() : nullptr;
    if (!t.assert_true(reader_source && llama_kv_stream_schedule_next(target,draft,3))) return;
    const auto next_read=reader_source->next_consumer(llama_kv_stream_access_role::append);
    t.assert_true("shared MTP schedules an immutable source read, not a new append",
        next_read.access == llama_kv_stream_access_role::read && next_read.cache_id == reader_source->host()->cache_id() &&
        next_read.layer+1 == full->get_layer_ids().size() && next_read.query_tokens == 1 && next_read.future_tokens == 0);
    t.assert_equal("target warmup graphs are retired before assistant weight loading",uint32_t(1),
        probe.observed.load(std::memory_order_relaxed));
    t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
    t.assert_true(draft->get_compute_memory()->shared_parent() == parent->shared_parent());
    t.assert_equal(uint32_t(1),draft->n_batch());
    t.assert_equal(uint32_t(1),draft->n_ubatch());
    const auto words=common_tokenize(target,"A woodland stream flows past trees and birds. ",true);
    if (!t.assert_true(!words.empty())) return;
    common_batch batch(target);
    for (int row=0;row<64;++row) batch.add(words[size_t(row)%words.size()],row,0,row == 63);
    if (!t.assert_equal("short prefill after common warmup",0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
    const size_t width=target->get_model().hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(main->model()));
    const auto predict=[&](int position,int hidden_row) {
        const auto * hidden=llama_get_embeddings_nextn_ith(target,hidden_row);
        const auto * logits=llama_get_logits_ith(target,-1);
        if (!t.assert_true(hidden && logits)) return false;
        common_batch read(draft);
        const auto id=read.add(llama_token(std::max_element(logits,logits+vocab)-logits),position,0,true);
        read.set_embd(id,{hidden,1,width});
        if (!t.assert_true(llama_kv_stream_schedule_next(draft,target,0))) return false;
        if (!t.assert_equal("common assistant reads target outputs",0,llama_process(draft,LLAMA_PROCESS_TYPE_DECODE,read.get()))) return false;
        if (!t.assert_true("actual assistant adopts source lookahead",reader_source->sequence_stats(llama_kv_stream_access_role::read).adopted)) return false;
        const auto * result=llama_get_logits_ith(draft,-1);
        return t.assert_true(result && std::all_of(result,result+vocab,[](float value) { return std::isfinite(value); }));
    };
    if (!predict(64,63)) return;
    llama_memory_cache_requirement before_probe,after_probe;
    if (!t.assert_true(reader_source && reader_source->read_source(before_probe))) return;
    const auto revision=parent->graph_binding_revision();
    t.assert_equal("shared assistant rollback classification performs no ordinary decode probe",
        COMMON_CONTEXT_SEQ_RM_TYPE_PART,common_context_can_seq_rm(draft));
    t.assert_true(reader_source->read_source(after_probe) &&
        before_probe.identity.id == after_probe.identity.id && before_probe.identity.generation == after_probe.identity.generation &&
        reader_source->tokens() == 64);
    t.assert_equal(revision,parent->graph_binding_revision());
    llama_set_kv_stream_decode(target,true);
    batch.clear();
    for (int row=0;row<4;++row) batch.add(words[size_t(row)%words.size()],64+row,0,true);
    if (!t.assert_equal("common target verifies declared four-query batch",0,
            llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
    t.assert_true("target verification hands its committed source to MTP",reader_source->prefetch_primed());
    if (!predict(68,3)) return;
    t.assert_true(llama_kv_stream_schedule_next(target,nullptr,0));
    speculative.reset();
    t.assert_true(!parent->has_speculative_consumer());
    batch.clear(); batch.add(words.front(),68,0,true);
    t.assert_equal("common target resumes after assistant destruction",0,
        llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()));
}

static void test_gemma_bounded_cache(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    std::vector<std::vector<float>> controls[2];
    for (bool borrowed : {false,true}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=2048; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1;
        cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(context))) return;
        auto resources=llama_memory_kv_resources_get(llama_get_memory(context.get()));
        auto * local=resources.bounded;
        if (!t.assert_true(local && resources.full && !resources.recurrent)) return;
        auto * storage=local->get_k_storage(0)->buffer;
        llama_memory_params params{}; params.type_k=cp.type_k; params.type_v=cp.type_v;
        llama_memory_bounded_kv_layout quote, actual;
        if (!t.assert_true(llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,
                context->get_cparams(),ggml_backend_buffer_get_type(storage),{607,1},quote) &&
                local->describe_bounded_storage(607,1,2048,256,actual))) return;
        t.assert_equal(size_t(50),quote.layers.size());
        t.assert_equal(uint32_t(1280),quote.allocation.capacity_rows);
        t.assert_equal(size_t(425984000),quote.allocation.bytes);
        t.assert_equal(actual.allocation.bytes,quote.allocation.bytes);
        using arena_ptr=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
        using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
        arena_ptr arena(nullptr,ggml_backend_memory_arena_free);
        lease_ptr lease(nullptr,ggml_backend_memory_lease_free);
        std::unique_ptr<llama_memory_bounded_kv> owner;
        if (borrowed) {
            arena.reset(ggml_backend_memory_arena_new_from_buffer(storage));
            if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),77,0,quote.allocation.bytes,quote.allocation.alignment,
                        GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
            lease.reset(ggml_backend_memory_arena_acquire(arena.get(),77));
            params.bounded_cache_lease=lease.get();
            const auto filter=[&](int32_t il) { return model->hparams.is_swa(uint32_t(il)); };
            const auto no_reuse=[](int32_t) { return int32_t(-1); };
            bool alias_rejected=false;
            try {
                const auto alias=[](int32_t il) { return il == 0 ? int32_t(1) : int32_t(-1); };
                llama_kv_cache invalid(*model,model->hparams,cp.type_k,cp.type_v,false,true,true,1280,1,256,
                    model->hparams.n_swa,model->hparams.swa_type,nullptr,filter,alias,nullptr,"",&params);
            } catch (const std::runtime_error & error) {
                alias_rejected=std::string(error.what()) == "bounded cache lease cannot alias selected layers";
            }
            if (!t.assert_true(alias_rejected)) return;
            ggml_backend_buffer_ptr short_buffer(ggml_backend_buffer_view(storage,0,quote.allocation.bytes-1));
            arena_ptr short_arena(ggml_backend_memory_arena_new_from_buffer(short_buffer.get()),ggml_backend_memory_arena_free);
            if (!t.assert_true(short_arena && ggml_backend_memory_arena_begin(short_arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(short_arena.get(),77,0,quote.allocation.bytes-1,quote.allocation.alignment,
                        GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(short_arena.get()))) return;
            lease_ptr short_lease(ggml_backend_memory_arena_acquire(short_arena.get(),77),ggml_backend_memory_lease_free);
            auto short_params=params; short_params.bounded_cache_lease=short_lease.get();
            bool short_rejected=false;
            try {
                llama_kv_cache invalid(*model,model->hparams,cp.type_k,cp.type_v,false,true,true,1280,1,256,
                    model->hparams.n_swa,model->hparams.swa_type,nullptr,filter,no_reuse,nullptr,"",&short_params);
            } catch (const std::runtime_error & error) {
                short_rejected=std::string(error.what()) == "bounded SWA cache exceeds its measured grant";
            }
            if (!t.assert_true(short_rejected)) return;
            try {
                llama_kv_cache cache(*model,model->hparams,cp.type_k,cp.type_v,false,true,true,1280,1,256,
                    model->hparams.n_swa,model->hparams.swa_type,nullptr,filter,no_reuse,nullptr,"",&params);
                t.assert_true(local->get_layer_ids() == cache.get_layer_ids());
                t.assert_true(cache.get_k_storage(0)->data == local->get_k_storage(0)->data);
            } catch (const std::exception & error) {
                t.out << "no-op target reuse: " << error.what() << '\n';
                t.assert_true(false); return;
            }
            owner=std::make_unique<llama_memory_bounded_kv>(quote,
                llama_memory_resource{77,1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,llama_memory_content::preserve});
            if (!t.assert_true(owner->bind(lease.get()))) return;
            llama_memory_execution_plan plan;
            plan.domains.push_back({1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            plan.stages={{1,{},{}},{2,{1},{}}};
            if (!t.assert_true(owner->register_resources(plan,{1,2}))) return;
            for (const auto & stage : plan.stages) t.assert_equal(quote.allocation.bytes,stage.requirements.front().size_min);
        }
        const auto words=common_tokenize(context.get(),"A forest has trees, birds, and flowing water. ",true);
        if (!t.assert_true(!words.empty())) return;
        uint64_t ticket=owner ? owner->begin_write(owner->identity()) : 0;
        if (owner && !t.assert_true(ticket != 0)) return;
        common_batch batch(context.get());
        for (int first=0;first<1333;first+=256) {
            batch.clear();
            const int count=std::min(256,1333-first);
            for (int i=0;i<count;++i) batch.add(words[size_t(first+i)%words.size()],first+i,0,i == count-1);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
        llama_synchronize(context.get());
        if (owner) {
            if (!t.assert_true(owner->finish_write(ticket,true))) return;
            std::vector<std::vector<uint8_t>> bytes;
            for (auto id : local->get_layer_ids()) for (auto * tensor : {local->get_k_storage(int32_t(id)),local->get_v_storage(int32_t(id))}) {
                bytes.emplace_back(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,bytes.back().data(),0,bytes.back().size());
            }
            std::vector<llama_pos> positions;
            for (uint32_t cell=0;cell<local->get_size();++cell) positions.push_back(
                local->get_cells(0).is_empty(cell) ? -1 : local->get_cells(0).pos_get(cell));
            if (!t.assert_true(owner->bind(lease.get()))) return;
            size_t index=0;
            for (auto id : local->get_layer_ids()) for (auto * tensor : {local->get_k_storage(int32_t(id)),local->get_v_storage(int32_t(id))}) {
                std::vector<uint8_t> after(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,after.data(),0,after.size());
                t.assert_true(bytes[index++] == after);
            }
            for (uint32_t cell=0;cell<local->get_size();++cell) t.assert_equal(positions[cell],
                local->get_cells(0).is_empty(cell) ? -1 : local->get_cells(0).pos_get(cell));
            t.assert_true(bool(owner->read(owner->identity())));
            {
                auto source=owner->read(owner->identity());
                ggml_context_ptr metadata(ggml_init({4096,nullptr,true}));
                ggml_tensor * k=nullptr,* v=nullptr;
                auto * native_k=local->get_k_storage(58),* native_v=local->get_v_storage(58);
                if (!t.assert_true(source.bind_layer_views(metadata.get(),58,native_k,native_v,k,v))) return;
                t.assert_true(k != native_k && v != native_v && k->view_src == native_k && v->view_src == native_v);
                t.assert_true(k->data == native_k->data && v->data == native_v->data);
                t.assert_equal(int64_t(4096),k->ne[0]); t.assert_equal(int64_t(1280),k->ne[1]);
                t.assert_equal(uint64_t(0),owner->begin_write(owner->identity()));
                auto * saved_k=k; auto * saved_v=v;
                t.assert_true(!source.bind_layer_views(metadata.get(),58,local->get_k_storage(57),native_v,k,v));
                t.assert_true(k == saved_k && v == saved_v);
                for (auto item : {std::pair{k,bytes.size()-2},std::pair{v,bytes.size()-1}}) {
                    std::vector<uint8_t> copy(ggml_nbytes(item.first));
                    ggml_backend_tensor_get(item.first,copy.data(),0,copy.size());
                    t.assert_true(copy == bytes[item.second]);
                }
            }
        }
        const size_t vocab=size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
        for (int step=0;step<8;++step) {
            auto * logits=llama_get_logits_ith(context.get(),-1);
            if (!t.assert_true(logits != nullptr)) return;
            controls[borrowed].emplace_back(logits,logits+vocab);
            const auto token=llama_token(std::max_element(logits,logits+vocab)-logits);
            batch.clear(); batch.add(token,1333+step,0,true);
            ticket=owner ? owner->begin_write(owner->identity()) : 0;
            if (owner && !t.assert_true(ticket != 0)) return;
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            llama_synchronize(context.get());
            if (owner && !t.assert_true(owner->finish_write(ticket,true))) return;
        }
    }
    if (!t.assert_equal(controls[0].size(),controls[1].size())) return;
    for (size_t row=0;row<controls[0].size();++row) t.assert_true(std::memcmp(controls[0][row].data(),
        controls[1][row].data(),controls[0][row].size()*sizeof(float)) == 0);
    auto full=llama_context_default_params();
    full.n_ctx=2048; full.n_batch=full.n_ubatch=256; full.n_seq_max=1;
    full.type_k=GGML_TYPE_Q8_0; full.type_v=GGML_TYPE_Q4_0; full.swa_full=true;
    full.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_ptr context(llama_init_from_model(model,full),llama_free);
    if (!t.assert_true(bool(context))) return;
    auto * local=llama_memory_kv_resources_get(llama_get_memory(context.get())).bounded;
    if (!t.assert_true(local != nullptr)) return;
    t.assert_equal(uint32_t(2048),local->get_size());
    llama_memory_bounded_kv_layout unchanged;
    unchanged.allocation.bytes=17;
    llama_memory_params params{}; params.type_k=full.type_k; params.type_v=full.type_v; params.swa_full=true;
    t.assert_true(!llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,context->get_cparams(),
        ggml_backend_buffer_get_type(local->get_k_storage(0)->buffer),{607,1},unchanged));
    t.assert_equal(size_t(17),unchanged.allocation.bytes);
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
    t.test("memory_resources_resolve_cache_roles_without_hybrid_casts", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        auto & hp = model->hparams;
        hp = {};
        hp.n_layer_all = 2;
        hp.n_embd = 64;
        hp.n_head_arr.fill(2);
        hp.n_head_kv_arr.fill(1);
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
        hp.n_embd_head_k_swa = hp.n_embd_head_v_swa = 32;
        hp.n_swa = 128;
        hp.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hp.is_swa_impl[0] = true;
        llama_kv_cache_iswa local(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, false,
                true, 256, 1, 64, 1, nullptr, nullptr, nullptr, nullptr);
        const auto view = llama_memory_kv_resources_get(&local);
        t.assert_true(view.full == local.get_base());
        t.assert_true(view.bounded == local.get_swa());
        t.assert_true(view.recurrent == nullptr);
        auto ctx = local.init_full();
        const auto * iswa = dynamic_cast<llama_kv_cache_iswa_context *>(ctx.get());
        t.assert_true(iswa && llama_memory_kv_context_get(ctx.get()) == iswa->get_base());
        t.assert_true(llama_memory_kv_resources_get(view.full).full == view.full);
        auto full_ctx = view.full->init_full();
        t.assert_true(llama_memory_kv_context_get(full_ctx.get()) == dynamic_cast<llama_kv_cache_context *>(full_ctx.get()));
        t.assert_true(!llama_memory_kv_resources_get(nullptr).full);
        t.assert_true(!llama_memory_kv_context_get(nullptr));

        const auto skip = [](int32_t) { return false; };
        llama_memory_hybrid hybrid(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 256, 1, 0,
                LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, 0,
                false, true, skip, skip);
        const auto hybrid_view = llama_memory_kv_resources_get(&hybrid);
        t.assert_true(hybrid_view.full == hybrid.get_mem_attn());
        t.assert_true(hybrid_view.recurrent == hybrid.get_mem_recr());
        t.assert_true(hybrid_view.bounded == nullptr);
        auto hybrid_ctx = hybrid.init_full();
        const auto * hybrid_cast = dynamic_cast<llama_memory_hybrid_context *>(hybrid_ctx.get());
        t.assert_true(hybrid_cast && llama_memory_kv_context_get(hybrid_ctx.get()) == hybrid_cast->get_attn());
        t.assert_true(!llama_memory_kv_resources_get(hybrid.get_mem_recr()).full);
        auto recurrent_ctx = hybrid.get_mem_recr()->init_full();
        t.assert_true(!llama_memory_kv_context_get(recurrent_ctx.get()));
    });
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
    t.test("gemma_logit_files_validate_profile_framing_and_exact_payload", [&](testing & t) {
        const std::array<uint64_t,8> profile{0x474d344c4f474954ull,1,16,4,2,3,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
        std::vector<std::vector<float>> source{{0.0f,-0.0f,1.5f},{-2.0f,3.0f,4.0f}}, result;
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        gemma_logits_file(file.get(),true,profile,source);
        std::rewind(file.get());
        gemma_logits_file(file.get(),false,profile,result);
        t.assert_equal(source.size(),result.size());
        for (size_t row=0;row<source.size();++row) t.assert_true(kv_stream_bytes_equal(
            source[row].data(),source[row].size()*sizeof(float),result[row].data(),result[row].size()*sizeof(float)));
        const auto unchanged=result;
        const auto rejects=[&](const std::array<uint64_t,8> & expected) {
            std::rewind(file.get());
            try { gemma_logits_file(file.get(),false,expected,result); }
            catch (const std::runtime_error &) { return true; }
            return false;
        };
        auto wrong=profile; wrong[2]+=1;
        t.assert_true(rejects(wrong) && result == unchanged);
        wrong=profile; wrong[5]=UINT64_MAX;
        t.assert_true(rejects(wrong) && result == unchanged);
        const unsigned char extra=7;
        std::fseek(file.get(),0,SEEK_END); std::fwrite(&extra,1,1,file.get()); std::fflush(file.get());
        t.assert_true(rejects(profile) && result == unchanged);
        file.reset(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        std::fwrite(profile.data(),1,sizeof(profile),file.get()); std::fflush(file.get());
        t.assert_true(rejects(profile) && result == unchanged);
        file.reset(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        gemma_logits_file(file.get(),true,profile,source);
        std::rewind(file.get()); std::fwrite(&extra,1,1,file.get()); std::fflush(file.get());
        t.assert_true(rejects(profile) && result == unchanged);
    });
    t.test("gemma_test_arena_override_is_explicit_and_bounded", [](testing & t) {
        t.assert_equal(size_t(1024)*1048576,gemma_test_arena_bytes(nullptr));
        t.assert_equal(size_t(768)*1048576,gemma_test_arena_bytes("768"));
        t.assert_equal(size_t(1)*1048576,gemma_test_arena_bytes("1"));
        t.assert_equal(size_t(4095)*1048576,gemma_test_arena_bytes("4095"));
        for (const char * invalid : {"", "0", "-1", "+768", " 768", "768 ", "768x", "4096", "18446744073709551616"}) {
            bool rejected=false;
            try { (void)gemma_test_arena_bytes(invalid); }
            catch (const std::invalid_argument &) { rejected=true; }
            t.assert_true(rejected);
        }
    });
    t.test("streaming_is_disabled_by_default", [&](testing & t) {
        const auto defaults = llama_context_default_params();
        t.assert_equal(size_t(0),defaults.kv_stream_pool_bytes);
        t.assert_equal(size_t(0),defaults.shared_device_memory_bytes);
        t.assert_equal(uint32_t(0), defaults.kv_stream_auxiliary_layers);
    });
    t.test("streamed_verification_width_is_independent_of_recurrent_storage", [&](testing & t) {
        t.assert_equal(uint32_t(0),llama_context_default_params().kv_stream_verify_max);
        llama_cparams params{};
        params.n_seq_max=1;
        t.assert_equal(uint32_t(1),params.kv_stream_decode_rows());
        for (uint32_t rows=1;rows<=4;++rows) {
            params.kv_stream_verify_max=rows;
            params.n_rs_seq=0;
            t.assert_equal(rows,params.kv_stream_decode_rows());
            params.n_rs_seq=3;
            t.assert_equal(rows,params.kv_stream_decode_rows());
        }
        params.kv_stream_verify_max=0;
        t.assert_equal(uint32_t(4),params.kv_stream_decode_rows());
        params.n_rs_seq=0; params.kv_stream_auxiliary_layers=1;
        t.assert_equal(uint32_t(4),params.kv_stream_decode_rows());
    });
    if (argc < 3 || std::strcmp(argv[1],"--model")) return t.summary();
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i],"--gemma-test-arena-mib")) {
            if (++i == argc) { std::fprintf(stderr,"missing Gemma test arena MiB\n"); return 1; }
            try { gemma_test_arena_size=gemma_test_arena_bytes(argv[i]); }
            catch (const std::invalid_argument & error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
            continue;
        }
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
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-warmup-native-logits") || !std::strcmp(argv[i],"--gemma-warmup-streamed-logits")) {
        if (i+1==argc) { std::fprintf(stderr,"missing Gemma warmup logit path\n"); return 1; }
        const int mode=!std::strcmp(argv[i],"--gemma-warmup-native-logits") ? 0 : 1;
        t.test("Gemma_warmup_failure_and_recovery_matches_framed_native_control",
            [&](testing & t) { test_gemma_warmup_failures(t,argv[2],mode,argv[i+1]); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-warmup-failures"); })) {
        t.test("Gemma_common_startup_rejects_failed_warmup_and_can_retry",
            [&](testing & t) { test_gemma_warmup_failures(t,argv[2]); });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-common-startup")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        t.test("Gemma_common_startup_and_short_prefill_borrow_workspace",[&](testing & t) {
            test_gemma_common_startup(t,argv[2],argv[i+1]);
        });
        return t.summary();
    }
    auto mparams = llama_model_default_params(); mparams.n_gpu_layers = 999;
    const llama_model_tensor_buft_override cache_only_weights[]={{".*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-read-source-cache"); })) {
        // This cache-only oracle needs no weight computation; leave VRAM for the stock full-history attention graph.
        mparams.tensor_buft_overrides=cache_only_weights;
    }
    mparams.load_mtp = embedded_mtp_control || mtp_memory_probe || mtp_sustained_control || target_drift_control || rollback_probe || serial_workspace_growth || mtp_bit_exact_control;
    model_ptr model(llama_model_load_from_file(argv[2],mparams),llama_model_free);
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-acceptance")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        const bool compact=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-memory"); });
        t.test("Gemma_acceptance_rollback_and_request_reuse_match_native",[&](testing & t) {
            test_gemma_acceptance(t,model.get(),argv[i+1],compact);
        });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-candidate-loop")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        t.test("Gemma_common_candidate_loop_matches_native_fixed_position_predictions",[&](testing & t) {
            const bool compact=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-memory"); });
            test_gemma_candidate_loop(t,model.get(),argv[i+1],compact);
        });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-assistant-source")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        t.test("Gemma_actual_assistant_reads_target_sources_without_new_KV",[&](testing & t) {
            const bool forced=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-forced"); });
            const bool compact=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-memory"); });
            const bool verify=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-verify"); });
            const bool quotas=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-quotas"); });
            test_gemma_assistant_source(t,model.get(),argv[i+1],forced,compact,verify,quotas);
        });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-admission-only"); })) {
        t.test("Gemma_target_admission_memory_boundaries",
            [&](testing & t) { test_gemma_target_admission(t,model.get(),false); });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-native-logits") || !std::strcmp(argv[i],"--gemma-streamed-logits")) {
        const bool native=!std::strcmp(argv[i],"--gemma-native-logits");
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma logit control path\n"); return 1; }
        const char * path=argv[i+1];
        t.test(native ? "Gemma_native_logit_control_export" : "Gemma_streamed_target_matches_external_native_control",
            [&](testing & t) { test_gemma_target_context(t,model.get(),8192,1333,8,true,false,
                native ? path : nullptr,native ? nullptr : path); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-admission-guards"); })) {
        t.test("Gemma_target_rejects_invalid_profiles_and_recovers_after_quota_failures",
            [&](testing & t) { test_gemma_target_admission(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target"); })) {
        t.test("Gemma_streamed_target_matches_native_model_execution",
            [&](testing & t) { test_gemma_target_context(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-medium"); })) {
        t.test("Gemma_streamed_target_ring_matches_native_model_execution",
            [&](testing & t) { test_gemma_target_context(t,model.get(),32768,32000,16,true,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-medium-fit"); })) {
        t.test("Gemma_streamed_target_light_ring_matches_fitting_native_context",
            [&](testing & t) { test_gemma_target_context(t,model.get(),24576,23552,16,true,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-ring"); })) {
        t.test("Gemma_streamed_target_ring_exceeds_measured_native_context_capacity",
            [&](testing & t) { test_gemma_target_context(t,model.get(),32768,32000,16,false,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-long"); })) {
        t.test("Gemma_streamed_target_long_history_fits_fixed_physical_budget",
            [&](testing & t) { test_gemma_target_context(t,model.get(),262144,65536,16,false,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-lifecycle"); })) {
        t.test("Gemma_serial_lifecycle_preserves_both_cache_groups",
            [&](testing & t) { test_gemma_lifecycle(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-phase-parent"); })) {
        t.test("Gemma_native_local_cache_and_streamed_full_cache_share_one_phase_parent",
            [&](testing & t) { test_gemma_phase_parent(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-read-source-cache"); })) {
        t.test("Gemma_shared_full_cache_reads_use_progressive_target_storage",
            [&](testing & t) { test_gemma_phase_parent(t,model.get(),true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-deferred-local"); })) {
        t.test("Gemma_local_cache_metadata_binds_transactionally_without_private_storage",
            [&](testing & t) { test_gemma_deferred_local(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-stream-groups"); })) {
        t.test("Gemma_streams_only_owned_full_attention_cache_roots",
            [&](testing & t) { test_gemma_stream_groups(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-bounded"); })) {
        t.test("Gemma_target_local_cache_reuses_bounded_storage_without_changing_causal_execution",
            [&](testing & t) { test_gemma_bounded_cache(t,model.get()); });
        return t.summary();
    }
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
