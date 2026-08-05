//
// Runtime speculative-decoding engine for the Tessera app.
// See common/tessera-runtime.h for the extern-C contract and
// docs/tessera-runtime-traces-design.md sections 5 and 8.
//
// The generation loop mirrors examples/speculative-simple (the live-loop
// API: common_speculative_begin/draft/process/accept bookkeeping) and adds
// per-step telemetry capture through common_spec_telemetry_record().
//

#include "tessera-runtime.h"

#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative-calibration.h"
#include "speculative.h"

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

// last error, per thread
static thread_local std::string g_tessera_rt_err;

static void tessera_rt_set_err(const std::string & msg) {
    g_tessera_rt_err = msg;
}

// device-local random session id (uuid v4), one per generate call
static std::string tessera_rt_uuid4() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    uint64_t a = dist(gen);
    uint64_t b = dist(gen);

    a = (a & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL; // version 4
    b = (b & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL; // variant 10

    char buf[37];
    snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
            (uint32_t) (a >> 32),
            (uint32_t) ((a >> 16) & 0xFFFF),
            (uint32_t) (a & 0xFFFF),
            (uint32_t) (b >> 48),
            (unsigned long long) (b & 0xFFFFFFFFFFFFULL));
    return std::string(buf);
}

struct tessera_rt {
    common_params params;

    // note: declaration order matters, destructors run bottom-to-top:
    // the verifier sampler and the spec handle (whose impl owns samplers
    // bound to the draft model) must die before the contexts and models
    common_init_result_ptr             llama_init; // trunk model + context
    common_speculative_init_result_ptr dft_init;   // drafter model + context
    common_speculative_ptr             spec;
    common_sampler_ptr                 smpl;       // greedy verifier sampler

    llama_batch batch;
    bool        batch_inited = false;

    int32_t draft_max = 3;
};

tessera_rt * tessera_rt_load(
        const char * trunk_path,
        const char * draft_path,
        uint32_t n_ctx,
        int32_t  n_threads,
        int32_t  n_gpu_layers,
        int32_t  draft_max) {

    g_tessera_rt_err.clear();

    if (trunk_path == nullptr || trunk_path[0] == '\0') {
        tessera_rt_set_err("tessera_rt_load: trunk_path is empty");
        return nullptr;
    }
    if (draft_path == nullptr || draft_path[0] == '\0') {
        tessera_rt_set_err("tessera_rt_load: draft_path is empty");
        return nullptr;
    }
    if (n_ctx == 0) {
        tessera_rt_set_err("tessera_rt_load: n_ctx must be > 0");
        return nullptr;
    }
    if (draft_max <= 0) {
        tessera_rt_set_err("tessera_rt_load: draft_max must be > 0");
        return nullptr;
    }

    std::unique_ptr<tessera_rt> rt(new tessera_rt());

    common_params & params = rt->params;

    params.model.path   = trunk_path;
    params.n_ctx        = n_ctx;
    params.n_gpu_layers = n_gpu_layers;

    if (n_threads > 0) {
        params.cpuparams.n_threads       = n_threads;
        params.cpuparams_batch.n_threads = n_threads;
    }

    // greedy verifier: keep a single candidate and take it
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.sampling.top_k    = 1;
    params.sampling.no_perf  = true;

    params.speculative.draft.n_max = draft_max;
    params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    // default (-1) means "auto", which would let the drafter float to a
    // GPU even when the caller pinned the trunk to CPU; inherit instead
    params.speculative.draft.n_gpu_layers = n_gpu_layers;

    rt->draft_max = draft_max;

    try {
        rt->llama_init = common_init_from_params(params);
    } catch (const std::exception & e) {
        tessera_rt_set_err(std::string("tessera_rt_load: trunk init threw: ") + e.what());
        return nullptr;
    }
    if (!rt->llama_init) {
        tessera_rt_set_err("tessera_rt_load: failed to load trunk model '" + std::string(trunk_path) + "'");
        return nullptr;
    }

    llama_model   * model_tgt = rt->llama_init->model();
    llama_context * ctx_tgt   = rt->llama_init->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        tessera_rt_set_err("tessera_rt_load: trunk model or context is null");
        return nullptr;
    }

    // standalone drafter (the imatrix calibration path's setup)
    try {
        common_params params_dft = common_base_params_to_speculative(params);
        params_dft.speculative.draft.mparams.path = draft_path;
        params_dft.speculative.types              = params.speculative.types;
        // the verification batch decodes draft_max+1 tokens with logits on;
        // common_base_params_to_speculative sizes drafter outputs at
        // n_parallel (1), which the output reserve asserts against
        params_dft.n_outputs_max                  = draft_max + 1;

        rt->dft_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    } catch (const std::exception & e) {
        tessera_rt_set_err(std::string("tessera_rt_load: drafter init threw: ") + e.what());
        return nullptr;
    }
    if (!rt->dft_init || rt->dft_init->model() == nullptr || rt->dft_init->context() == nullptr) {
        tessera_rt_set_err("tessera_rt_load: failed to load draft model '" + std::string(draft_path) + "'");
        return nullptr;
    }

    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = rt->dft_init->context();

    try {
        rt->spec.reset(common_speculative_init(params.speculative, /*n_seq=*/1));
    } catch (const std::exception & e) {
        tessera_rt_set_err(std::string("tessera_rt_load: spec init threw: ") + e.what());
        return nullptr;
    }
    if (!rt->spec) {
        tessera_rt_set_err("tessera_rt_load: failed to create spec context");
        return nullptr;
    }

    rt->smpl.reset(common_sampler_init(model_tgt, params.sampling));
    if (!rt->smpl) {
        tessera_rt_set_err("tessera_rt_load: failed to create verifier sampler");
        return nullptr;
    }

    rt->batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    rt->batch_inited = true;

    return rt.release();
}

int32_t tessera_rt_generate(
        tessera_rt * rt,
        const char * prompt,
        int32_t max_tokens,
        int32_t telemetry_topk,
        tessera_rt_token_cb on_token,
        tessera_rt_trace_cb on_trace,
        void * ud) {

    g_tessera_rt_err.clear();

    if (rt == nullptr) {
        tessera_rt_set_err("tessera_rt_generate: rt is null");
        return -1;
    }
    if (prompt == nullptr) {
        tessera_rt_set_err("tessera_rt_generate: prompt is null");
        return -1;
    }
    if (telemetry_topk < 0) {
        telemetry_topk = 0;
    }

    llama_model   * model_tgt = rt->llama_init->model();
    llama_context * ctx_tgt   = rt->llama_init->context();
    llama_context * ctx_dft   = rt->dft_init->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    common_speculative * spec = rt->spec.get();

    const llama_seq_id seq_id = 0;
    const int32_t n_ctx    = (int32_t) llama_n_ctx(ctx_tgt);
    const int32_t n_vocab  = llama_vocab_n_tokens(vocab);
    const int32_t n_batch  = std::min(llama_n_batch(ctx_tgt), llama_n_batch(ctx_dft));

    // telemetry only when requested AND a sink is attached
    const bool capture = telemetry_topk > 0 && on_trace != nullptr;

    // one session id per generation call
    const std::string sid = tessera_rt_uuid4();

    // fresh KV state for both contexts; the engine is reused across calls
    llama_memory_clear(llama_get_memory(ctx_tgt), false);
    llama_memory_clear(llama_get_memory(ctx_dft), false);

    common_sampler_reset(rt->smpl.get());

    const llama_tokens inp = common_tokenize(ctx_tgt, std::string(prompt), /*add_special=*/true, /*parse_special=*/true);
    if (inp.empty()) {
        tessera_rt_set_err("tessera_rt_generate: prompt tokenized to zero tokens");
        return -1;
    }
    if ((int) inp.size() + 1 > n_ctx) {
        tessera_rt_set_err("tessera_rt_generate: prompt exceeds context size");
        return -1;
    }

    // prefill both contexts in n_batch chunks; keep the last token separate.
    // the drafter side goes through common_speculative_process (for
    // DRAFT_SIMPLE that is the drafter's decode of the same batch)
    for (int i = 0; i + 1 < (int) inp.size(); i += n_batch) {
        const int n = std::min(n_batch, (int) inp.size() - 1 - i);

        llama_batch chunk = llama_batch_get_one(const_cast<llama_token *>(inp.data()) + i, n);

        if (llama_decode(ctx_tgt, chunk) != 0) {
            tessera_rt_set_err("tessera_rt_generate: trunk prefill decode failed");
            return -1;
        }
        if (!common_speculative_process(spec, chunk)) {
            tessera_rt_set_err("tessera_rt_generate: drafter prefill decode failed");
            return -1;
        }
    }

    llama_token id_last = inp.back();

    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(n_ctx);

    int n_past = (int) inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    // contexts that cannot partially remove sequences fall back to
    // full-state checkpoints (same scheme as examples/speculative-simple)
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = (common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    llama_batch batch_tgt = rt->batch;

    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    int  n_generated = 0;
    bool stop        = false;
    int  step        = 0;

    while (!stop) {
        if (max_tokens > 0 && n_generated >= max_tokens) {
            break;
        }

        // keep headroom for one full spec step
        if (n_past + rt->draft_max + 1 >= n_ctx) {
            break;
        }

        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ rt->draft_max,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(spec);

            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // roll the drafter's KV back to the pre-draft state; the
            // verification batch below re-adds the accepted prefix
            {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // partial draft reused after checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        const size_t      n_draft  = draft.size();
        const llama_token id_prime = id_last; // prime token for the record

        // evaluate [id_last, draft0, ..., draftN-1] with the target model
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (llama_pos) i, { seq_id }, true);
        }

        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            tessera_rt_set_err("tessera_rt_generate: trunk verification decode failed");
            return -1;
        }

        // the drafter re-evaluates the same batch so its KV stays in sync
        if (llama_decode(ctx_dft, batch_tgt) != 0) {
            tessera_rt_set_err("tessera_rt_generate: drafter verification decode failed");
            return -1;
        }

        // capture the per-position verifier and drafter rows before the next
        // decode overwrites them. row i is the distribution conditioned on
        // [prompt..., id_last, draft[0..i-1]], matching the calibration
        // emitter's row semantics. nullptr rows (unavailable outputs)
        // serialize as argmax 0 / empty top-k, same as calibration.
        std::vector<std::vector<float>> row_storage;
        std::vector<const float *> v_rows(n_draft + 1, nullptr);
        std::vector<const float *> d_rows(n_draft + 1, nullptr);
        if (capture) {
            row_storage.reserve(2 * (n_draft + 1));
            for (size_t i = 0; i <= n_draft; ++i) {
                const float * row = llama_get_logits_ith(ctx_tgt, (int32_t) i);
                if (row != nullptr) {
                    row_storage.emplace_back(row, row + n_vocab);
                    v_rows[i] = row_storage.back().data();
                }
            }
            for (size_t i = 0; i <= n_draft; ++i) {
                const float * row = llama_get_logits_ith(ctx_dft, (int32_t) i);
                if (row != nullptr) {
                    row_storage.emplace_back(row, row + n_vocab);
                    d_rows[i] = row_storage.back().data();
                }
            }
        }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(rt->smpl.get()));
        }

        const llama_tokens ids = common_sampler_sample_and_accept_n(rt->smpl.get(), ctx_tgt, draft);

        GGML_ASSERT(!ids.empty()); // at least the bonus token is always sampled

        const size_t n_acc = ids.size() - 1;

        if (use_ckpt_tgt && n_acc < draft.size()) {
            // partial acceptance on a checkpoint context: restore the state
            // and re-verify with the accepted prefix as the new draft
            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            rt->smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        if (capture) {
            const std::string line = common_spec_telemetry_record(
                    step, id_prime, draft, n_acc,
                    v_rows, d_rows, n_vocab, telemetry_topk,
                    /*provenance=*/"runtime", /*sid=*/sid.c_str());
            on_trace(line.c_str(), ud);
        }

        // accept() asserts an impl produced the draft, so only call it when
        // this step actually drafted (a failed draft has nothing to account)
        if (!draft.empty()) {
            common_speculative_accept(spec, seq_id, (uint16_t) n_acc);
        }

        n_past += (int) ids.size() - 1;

        // commit and stream the accepted tokens
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                stop = true;
                break;
            }
            if (max_tokens > 0 && n_generated >= max_tokens) {
                stop = true;
                break;
            }

            if (on_token != nullptr) {
                const std::string piece = common_token_to_piece(ctx_tgt, id_last);
                on_token(piece.c_str(), id_last, ud);
            }

            ++n_generated;
        }

        draft.clear();

        // trim rejected drafts from both KV caches
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);

        ++step;
    }

    LOG_DBG("%s: steps=%d generated=%d\n", __func__, step, n_generated);

    return n_generated;
}

void tessera_rt_free(tessera_rt * rt) {
    if (rt == nullptr) {
        return;
    }

    if (rt->batch_inited) {
        llama_batch_free(rt->batch);
    }

    delete rt;
}

const char * tessera_rt_last_error(void) {
    return g_tessera_rt_err.c_str();
}
