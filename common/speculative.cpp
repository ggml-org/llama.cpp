#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "spec_sidecar.h"
#include "speculative-sidecar-cap.h"
#include "../include/spec_sidecar/sidecar_abi.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <random>
#include <cinttypes>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

static bool common_speculative_rdna2_auto_enabled() {
    const char * value = std::getenv("GGML_HIP_RDNA2_AUTO");
    return value == nullptr ||
           (std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "false") != 0);
}

// Sidecar acceleration is an explicit experimental opt-in. Artifact paths or
// libraries in the environment must never activate it by themselves.
static bool common_speculative_sidecar_enabled() {
    const char * value = std::getenv("SPEC_SIDECAR");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static uint64_t common_spec_sidecar_stochastic_key(uint32_t seed, llama_seq_id seq_id,
        llama_pos n_past, uint32_t kind) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    uint64_t key = (uint64_t) seed << 32;
    key ^= (uint64_t) (uint32_t) seq_id * UINT64_C(0x632be59bd9b4e019);
    key ^= (uint64_t) (uint32_t) n_past * UINT64_C(0x8cb92ba72f3d8dd7);
    key ^= (uint64_t) kind * UINT64_C(0x9e3779b97f4a7c15);
    return spec_sidecar_stochastic_mix64(key);
}

static bool common_spec_sidecar_validate_distribution(const int32_t * ids, const float * probs,
        int count, int32_t n_vocab, common_speculative_token_dist & dist) {
    if (ids == nullptr || probs == nullptr || count <= 0) {
        return false;
    }
    dist.ids.resize((size_t) count);
    dist.probs.resize((size_t) count);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= n_vocab || !std::isfinite(probs[i]) || probs[i] < 0.0f) {
            return false;
        }
        dist.ids[i] = (llama_token) ids[i];
        dist.probs[i] = probs[i];
        sum += probs[i];
    }
    if (!std::isfinite(sum) || sum <= 0.0f) {
        return false;
    }
    // Normalize defensively at the ABI boundary. The sidecar samples the same
    // top-k row before returning it, while this removes harmless reduction
    // round-off before the target residual sampler consumes q.
    for (float & value : dist.probs) {
        value /= sum;
    }
    return true;
}

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

// Sidecar MTP+n-gram stacks use a fixed n-gram proposal cap.  The cap is
// derived from the configured MTP width, preserving the anti-stutter policy
// without acceptance-driven width changes.
static common_speculative_sidecar_cap_config common_speculative_sidecar_cap_config_for(
        const common_params_speculative & params, int ceiling) {
    if (!params.draft.sidecar_only ||
            params.draft.sidecar_type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP ||
            params.draft.n_max <= 0 || ceiling <= params.draft.n_max) {
        return {};
    }

    return { std::max(1, params.draft.n_max) };
}

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    // Lets sidecar-backed implementations inspect the per-request sampling
    // mode before process() decides whether native draft state can be skipped.
    // Existing implementations do not need this hook.
    virtual void prepare_process(const common_speculative_draft_params_vec & /*dparams*/) {}

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    // Sidecar implementations serialize only a small logical cursor; their
    // large device KV allocation stays resident and is never copied here.
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) { return false; }
    virtual bool set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) { return true; }
    virtual bool state_required(llama_seq_id /*seq_id*/) const { return false; }
    virtual bool reset_state(llama_seq_id /*seq_id*/) { return true; }
    // Request release normally invalidates implementation-local state. Stateful
    // sidecars may keep a committed cursor here and validate it against the next
    // resident prompt before any target prefix is reused.
    virtual void release_state(llama_seq_id seq_id) { reset_state(seq_id); }
    virtual bool prepare_prompt_state(
            llama_seq_id /*seq_id*/, llama_pos /*pos_next*/, bool /*can_reuse_resident*/) { return true; }
    virtual bool truncate_state(llama_seq_id /*seq_id*/, llama_pos /*pos_max*/) { return true; }
    virtual bool commit_state(llama_seq_id /*seq_id*/, llama_pos /*pos_max*/) { return true; }
    virtual bool rebase_state(llama_seq_id /*seq_id*/, llama_pos /*pos_min*/, llama_pos /*pos_max*/, llama_pos /*delta*/) { return true; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return true;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return false;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
        return true;
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch = {};        // noise tokens
    llama_batch batch_inject = {}; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;
    std::vector<std::mt19937> selector_rng;
    std::vector<bool> selector_reset;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;
    int32_t         n_layer_tgt        = 0;       // extract id == n_layer_tgt -> pre-final-norm state (nextn)

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<llama_pos> verify_pos_first;
    std::vector<int32_t> verify_rows;
    // Sequences whose sidecar KV no longer mirrors the target prefix (e.g. an
    // M-RoPE image prompt made target positions diverge from the dense draft
    // rows). Stale sequences skip sidecar drafting until the next state reset
    // instead of disabling the sidecar for the whole process.
    std::vector<bool> sidecar_stale;

    common_spec_sidecar_dflash sidecar;
    bool sidecar_target_only = false; // runtime failure or unsupported sampling mode

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        const bool sidecar_only = this->params.sidecar_only &&
                this->params.sidecar_type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
        const common_spec_sidecar_profile * sidecar_profile = this->params.sidecar_profile;
        GGML_ASSERT(ctx_tgt && (ctx_dft != nullptr || sidecar_only) &&
                "DFlash requires a target context or a validated sidecar-only mode");

        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_dft = ctx_dft != nullptr ? llama_get_model(ctx_dft) : nullptr;

        if (model_dft != nullptr) {
            target_layer_ids   = llama_model_target_layer_ids  (model_dft);
            target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        } else if (sidecar_only && sidecar_profile != nullptr &&
                sidecar_profile->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH) {
            target_layer_ids   = sidecar_profile->dflash_target_layer_ids;
            target_layer_ids_n = sidecar_profile->dflash_target_layer_ids_n;
        }
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = model_dft != nullptr ? llama_model_n_embd(model_dft) :
                (sidecar_profile != nullptr ? sidecar_profile->dflash_decoder_width : 0);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // Read trained block metadata from a native model, or use the provider
        // profile when no host draft exists.
        block_size = sidecar_only && sidecar_profile != nullptr
                ? sidecar_profile->dflash_block_size : 16;
        if (model_dft != nullptr) {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
            selector_top_k = llama_model_dflash_selector_top_k(model_dft);
            is_dflash2 = selector_top_k > 0;
        } else if (sidecar_only && sidecar_profile != nullptr &&
                sidecar_profile->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH) {
            selector_top_k = sidecar_profile->dflash_selector_top_k;
            is_dflash2 = selector_top_k > 0;
        }
        // In sidecar-only mode there is no host draft model; the target shares
        // the drafter vocabulary, so its <mask> token is authoritative. Token 0
        // is a real vocabulary entry and must not be used as a mask filler.
        mask_token_id = model_dft != nullptr
                ? llama_vocab_mask(llama_model_get_vocab(model_dft))
                : llama_vocab_mask(llama_model_get_vocab(model_tgt));
        if (mask_token_id == LLAMA_TOKEN_NULL) {
            mask_token_id = 0;
        }
        n_layer_tgt = llama_model_n_layer(model_tgt);

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        // speculative sidecar's DFlash DLL is selected only after the preflight probe.
        // A sidecar-only construction has no native draft context to fall back
        // to if HIP initialization fails, so it enters target-only mode.
        // The provider profile is selected only after the preflight probe.
        // A sidecar-only construction has no native draft context to fall back
        // to if HIP initialization fails, so it enters target-only mode.
        if (sidecar_only && is_dflash2 && sidecar_profile != nullptr &&
                sidecar_profile->kind == COMMON_SPEC_SIDECAR_KIND_DFLASH) {
            common_spec_sidecar_paths paths;
            std::string error;
            if (common_spec_sidecar_get_paths(*sidecar_profile, paths, error) &&
                    n_embd_enc == sidecar_profile->dflash_encoded_width &&
                    block_size == sidecar_profile->dflash_block_size) {
                if (sidecar.load(paths.library, paths.artifact_dir,
                        sidecar_profile->dflash_encoded_width,
                        sidecar_profile->dflash_block_size, (int32_t) n_seq, error)) {
                    for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                        if (target_layer_ids[k] == n_layer_tgt) {
                            llama_set_embeddings_nextn_device_preferred(ctx_tgt, true);
                        } else {
                            llama_set_embeddings_layer_inp_device_preferred(
                                    ctx_tgt, (uint32_t) target_layer_ids[k], true);
                        }
                    }
                    SPC_INF("DFlash sidecar active: %s\n", paths.library.c_str());
                } else {
                    sidecar_target_only = true;
                    SPC_WRN("DFlash sidecar unavailable (%s); target-only mode\n", error.c_str());
                }
            } else {
                sidecar_target_only = true;
                if (error.empty()) {
                    error = "provider profile dimensions do not match the target";
                }
                SPC_WRN("DFlash sidecar unavailable (%s); target-only mode\n", error.c_str());
            }
        }
        if (sidecar_only && !sidecar.active()) {
            sidecar_target_only = true;
        }

        if (ctx_dft != nullptr) {
            batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
            batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

            // Embedding batches for an M-RoPE draft carry four position rows.
            is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
            if (is_mrope) {
                free(batch_inject.pos);
                batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
            }

            smpls.resize(n_seq);
            for (auto & s : smpls) {
                common_params_sampling sparams;
                sparams.no_perf  = false;
                sparams.top_k    = is_dflash2 ? selector_top_k : 10;
                sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
                s.reset(common_sampler_init(model_dft, sparams));
            }
        }

        selector_rng.resize(n_seq);
        selector_reset.assign(n_seq, true);
        verify_pos_first.assign(n_seq, -1);
        verify_rows.assign(n_seq, 0);
        sidecar_stale.assign(n_seq, false);

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings; an id equal
        // to the target's layer count means the pre-final-norm hidden state,
        // which is captured through the unmasked nextn path instead
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            }
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        // Legacy Laguna drafters retain their causal noise-block attention.
        if (ctx_dft != nullptr) {
            llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
            llama_set_causal_attn(ctx_dft, causal_attn);
        }
    }

    void prepare_process(const common_speculative_draft_params_vec & dparams) override {
        if (!sidecar.active()) {
            return;
        }
        for (const auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }
            if (dp.temperature > 0.0f && dp.dists == nullptr) {
                // A stochastic proposal must carry q to the verifier. Never
                // fall back to equality-only verification, which would bias
                // the target distribution.
                sidecar.disable();
                sidecar_target_only = true;
                SPC_WRN("%s", "DFlash sidecar requires proposal distributions for stochastic sampling; target-only mode\n");
                break;
            }
            if (dp.temperature <= 0.0f && this->params.p_min > 0.0f) {
                sidecar.disable();
                sidecar_target_only = true;
                SPC_WRN("%s", "DFlash sidecar does not support p_min in greedy mode; target-only mode\n");
                break;
            }
        }
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        selector_reset[seq_id] = true;

        if (params.ctx_dft == nullptr) {
            return;
        }
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process_sidecar(const llama_batch & batch_in,
            const std::vector<int32_t> & i_batch_beg,
            const std::vector<int32_t> & i_batch_end) {
        auto * ctx_tgt = this->params.ctx_tgt;
        std::vector<llama_device_view> layers(target_layer_ids_n);
        bool direct = false;
        void * stream = nullptr;
        int32_t device = -1;
#if defined(GGML_USE_HIP)
        direct = true;
#endif
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const bool is_nextn = target_layer_ids[k] == n_layer_tgt;
            const bool has_device_layer = is_nextn
                    ? llama_get_embeddings_nextn_device(ctx_tgt, &layers[k])
                    : llama_get_embeddings_layer_inp_device(ctx_tgt, (uint32_t) target_layer_ids[k], &layers[k]);
            if (!has_device_layer || layers[k].row_stride != (size_t) n_embd_tgt * sizeof(float) ||
                    layers[k].n_rows < (uint32_t) batch_in.n_tokens) {
                direct = false;
                break;
            }
            if (stream == nullptr) {
                stream = layers[k].stream;
                device = layers[k].device;
            } else if (layers[k].stream != stream || layers[k].device != device) {
                direct = false;
                break;
            }
        }
        if (direct && !sidecar.attach_target_stream(stream, device)) {
            direct = false;
        }

        std::vector<bool> contiguous(n_seq, true);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) continue;
            for (int32_t i = i_batch_beg[seq_id]; i <= i_batch_end[seq_id]; ++i) {
                if (batch_in.seq_id[i][0] != seq_id) {
                    contiguous[seq_id] = false;
                    break;
                }
            }
        }

        std::vector<std::vector<float>> host_layer_rows;
        if (!direct) {
            host_layer_rows.resize(target_layer_ids_n);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                float * layer = target_layer_ids[k] == n_layer_tgt
                        ? llama_get_embeddings_nextn(ctx_tgt)
                        : llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                if (layer == nullptr) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    SPC_ERR("%s", "DFlash sidecar target layer output unavailable; entering target-only mode\n");
                    return true;
                }
                host_layer_rows[k].assign(layer, layer +
                        (size_t) batch_in.n_tokens * n_embd_tgt);
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < beg) continue;
            std::vector<int32_t> indices;
            if (!contiguous[seq_id]) {
                for (int32_t i = 0; i < batch_in.n_tokens; ++i) {
                    if (batch_in.seq_id[i][0] == seq_id) indices.push_back(i);
                }
            }
            const int32_t n_rows = contiguous[seq_id] ? end - beg + 1 : (int32_t) indices.size();
            if (n_rows <= 0) continue;
            if (sidecar_stale[seq_id]) {
                verify_rows[seq_id] = 0;
                verify_pos_first[seq_id] = -1;
                continue;
            }
            verify_rows[seq_id] = n_rows;
            verify_pos_first[seq_id] = batch_in.pos[contiguous[seq_id] ? beg : indices.front()];

            int rc = -1;
            if (direct && contiguous[seq_id]) {
                std::vector<const void *> layer_ptrs(target_layer_ids_n);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    layer_ptrs[k] = static_cast<const char *>(layers[k].data) +
                            (size_t) beg * layers[k].row_stride;
                }
                rc = sidecar.chunk_device(seq_id, batch_in.pos + beg, layer_ptrs.data(),
                        (int) target_layer_ids_n, n_embd_tgt, n_rows);
            } else {
                features_buf.resize((size_t) n_rows * n_embd_enc);
                std::vector<int32_t> positions((size_t) n_rows);
                for (int32_t r = 0; r < n_rows; ++r) {
                    const int32_t i = contiguous[seq_id] ? beg + r : indices[r];
                    positions[r] = batch_in.pos[i];
                    for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                        std::memcpy(features_buf.data() + (size_t) r * n_embd_enc +
                                (size_t) k * n_embd_tgt,
                                host_layer_rows[k].data() + (size_t) i * n_embd_tgt,
                                (size_t) n_embd_tgt * sizeof(float));
                    }
                }
                size_t n_bad = 0;
                for (float & value : features_buf) {
                    if (!std::isfinite(value)) {
                        value = value != value ? 0.0f : (value > 0.0f ? 65504.0f : -65504.0f);
                        ++n_bad;
                    }
                }
                if (n_bad > 0) {
                    static bool warned = false;
                    if (!warned) {
                        SPC_WRN("%s", "sanitized non-finite target features for DFlash sidecar\n");
                        warned = true;
                    }
                }
                rc = sidecar.chunk(seq_id, positions.data(), features_buf.data(), n_rows);
            }
            if (rc != 0) {
                // Position mismatch (e.g. M-RoPE image prompt) or transient
                // failure: pause this sequence until reset, keep the sidecar.
                sidecar_stale[seq_id] = true;
                verify_rows[seq_id] = 0;
                verify_pos_first[seq_id] = -1;
                SPC_WRN("DFlash sidecar: seq %d target feature update rejected; drafting paused until reset\n", (int) seq_id);
            }
        }
        return true;
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // skipping the embedding batches leaves a hole in the draft's cache and
        // the next injection fails to initialize.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            if (sidecar.active()) {
                // Image/multimodal embedding batches are not representable in the
                // sidecar's dense KV. Mark the affected sequences stale so they
                // skip sidecar drafting until the next reset; other sequences and
                // future requests keep the sidecar.
                for (int32_t k = 0; k < batch_in.n_tokens; ++k) {
                    const llama_seq_id sid = batch_in.seq_id != nullptr ? batch_in.seq_id[k][0] : 0;
                    if (sid >= 0 && sid < (llama_seq_id) n_seq && !sidecar_stale[sid]) {
                        sidecar_stale[sid] = true;
                        SPC_WRN("DFlash sidecar: seq %d stale after non-token batch; drafting paused until reset\n", (int) sid);
                    }
                }
            }
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        if (sidecar.active()) {
            if (batch_in.pos == nullptr) {
                sidecar.disable();
                sidecar_target_only = true;
                SPC_ERR("%s", "DFlash sidecar requires explicit target positions; entering target-only mode\n");
                return true;
            }
            return process_sidecar(batch_in, i_batch_beg, i_batch_end);
        }
        if (sidecar_target_only) {
            return true;
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_rows[seq_id] = n_rows;
            verify_pos_first[seq_id] = batch_in.pos != nullptr
                    ? batch_in.pos[i_batch_beg[seq_id]] : -1;

            // M-RoPE target positions are tuples whose temporal component can repeat across image
            // tokens. The 1-D draft cache instead keeps exactly one dense row per target token.
            // Server-side draft trimming guarantees that its current tail is the next token index.
            llama_pos pos_next = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) + 1;

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = target_layer_ids[k] == n_layer_tgt
                        ? llama_get_embeddings_nextn(ctx_tgt)
                        : llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                // sanitize non-finite feature values before fusing. on Metal, the
                // mat-mat kernels stage f32 activations as f16 for the simdgroup
                // multiply; Laguna's massive-activation rows (attention-sink tokens,
                // |x| ~ 1e6 in the pre-final-norm residual) overflow f16 -> inf/nan.
                // one poisoned row would otherwise NaN the whole drafter KV cache.
                {
                    size_t n_bad = 0;
                    for (auto & v : features_buf) {
                        if (!std::isfinite(v)) {
                            v = v != v ? 0.0f : (v > 0.0f ? 65504.0f : -65504.0f);
                            n_bad++;
                        }
                    }
                    if (n_bad > 0) {
                        static bool warned = false;
                        if (!warned) {
                            LOG_WRN("%s: sanitized %zu non-finite target feature values (f16 overflow on massive activations); "
                                    "draft quality may degrade slightly on affected rows\n", __func__, n_bad);
                            warned = true;
                        }
                    }
                }

                if (sidecar_target_only) {
                    continue;
                }

                // fuse extracted features through DFlash encoder
                // M-RoPE drafts read 4 position rows per token from embd batches, so pass them explicitly
                std::vector<llama_pos> enc_pos;
                if (is_mrope) {
                    enc_pos.resize((size_t) 4 * n_chunk);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const llama_pos p = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                        enc_pos[0 * n_chunk + i] = p;
                        enc_pos[1 * n_chunk + i] = p;
                        enc_pos[2 * n_chunk + i] = p;
                        enc_pos[3 * n_chunk + i] = 0;
                    }
                }

                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ is_mrope ? enc_pos.data() : nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // Inject the DFlash decoder K/V cache in dense token space. For text-only batches
                // this is identical to the target positions; for M-RoPE it avoids repeated positions.
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        batch_inject.pos[1 * n_chunk + i] = p;
                        batch_inject.pos[2 * n_chunk + i] = p;
                        batch_inject.pos[3 * n_chunk + i] = 0;
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        if (sidecar_target_only) {
            return;
        }

        if (sidecar.active()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & dp = dparams[seq_id];
                if (!dp.drafting || sidecar_stale[seq_id]) {
                    continue;
                }
                const bool stochastic = dp.temperature > 0.0f;
                if (!stochastic && params.p_min > 0.0f) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    return;
                }
                if (stochastic && dp.dists == nullptr) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    SPC_ERR("%s", "DFlash sidecar has no proposal-distribution output; entering target-only mode\n");
                    return;
                }

                const int limit = std::min(params.n_max, dp.n_max > 0 ? dp.n_max : params.n_max);
                if (limit < 1) {
                    return;
                }
                int32_t ids[7] = {};
                std::vector<int32_t> dist_ids;
                std::vector<float> dist_probs;
                if (stochastic) {
                    dist_ids.resize((size_t) limit * SPEC_SIDECAR_DFLASH_DRAFT_TOP_K, -1);
                    dist_probs.resize((size_t) limit * SPEC_SIDECAR_DFLASH_DRAFT_TOP_K, 0.0f);
                } else if (dp.dists != nullptr) {
                    dp.dists->clear();
                }
                const int n = stochastic
                        ? sidecar.draft_stochastic(seq_id, dp.id_last, (int32_t) dp.n_past,
                                dp.temperature, params.p_min,
                                common_spec_sidecar_stochastic_key(dp.seed, seq_id, dp.n_past, 2),
                                limit, ids, dist_ids.data(), dist_probs.data())
                        : sidecar.draft(seq_id, dp.id_last, (int32_t) dp.n_past, ids);
                if (n < 0 || n > (stochastic ? limit : 7)) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    SPC_ERR("%s", "DFlash sidecar draft failed; entering target-only mode\n");
                    return;
                }

                // The greedy DFlash ABI always fills the trained seven-token
                // block, while the request may reserve a smaller speculative
                // width. Only expose the requested prefix to target verification.
                const int n_emit = std::min(n, limit);

                const int32_t n_vocab = llama_vocab_n_tokens(
                        llama_model_get_vocab(llama_get_model(params.ctx_tgt)));
                auto & result = *dp.result;
                if (stochastic) {
                    dp.dists->clear();
                    dp.dists->reserve((size_t) n_emit);
                }
                for (int i = 0; i < n_emit; ++i) {
                    if (ids[i] < 0 || ids[i] >= n_vocab) {
                        result.clear();
                        if (dp.dists != nullptr) dp.dists->clear();
                        sidecar.disable();
                        sidecar_target_only = true;
                        SPC_ERR("%s", "DFlash sidecar returned invalid token ID; entering target-only mode\n");
                        return;
                    }
                    result.push_back((llama_token) ids[i]);
                    if (stochastic) {
                        common_speculative_token_dist dist;
                        if (!common_spec_sidecar_validate_distribution(
                                    dist_ids.data() + (size_t) i * SPEC_SIDECAR_DFLASH_DRAFT_TOP_K,
                                    dist_probs.data() + (size_t) i * SPEC_SIDECAR_DFLASH_DRAFT_TOP_K,
                                    SPEC_SIDECAR_DFLASH_DRAFT_TOP_K, n_vocab, dist)) {
                            result.clear();
                            dp.dists->clear();
                            sidecar.disable();
                            sidecar_target_only = true;
                            SPC_ERR("%s", "DFlash sidecar returned invalid proposal distribution; entering target-only mode\n");
                            return;
                        }
                        dp.dists->push_back(std::move(dist));
                    }
                }
                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (dp.dists != nullptr) dp.dists->clear();
                }
            }
            return;
        }

        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.n_past;

            // the previous block left its noise tokens' K/V in the draft region. the decoder runs
            // non-causal, so those cells are not masked out by position and the new block would
            // attend to them. only the injected target states (positions < n_past) may persist.
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n, -1);

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (dp.dists) {
                dp.dists->clear();
            }

            if (is_dflash2) {
                GGML_ASSERT(dp.temperature <= 0.0f || dp.dists);
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                if (selector_reset[seq_id]) {
                    uint32_t seed = dp.seed;
                    if (seed == LLAMA_DEFAULT_SEED) {
                        seed = (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
                    }
                    selector_rng[seq_id].seed(seed ^ 0x85ebca6bU);
                    selector_reset[seq_id] = false;
                }

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    if (dp.temperature > 0.0f) {
                        common_speculative_token_dist dist;
                        dist.ids.resize(selector_top_k);
                        dist.probs.resize(selector_top_k);
                        const float max_score = *std::max_element(scores, scores + selector_top_k);
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            dist.ids[k] = (llama_token) row[k];
                            dist.probs[k] = std::exp((scores[k] - max_score) / dp.temperature);
                            sum += dist.probs[k];
                        }
                        for (float & p : dist.probs) {
                            p /= sum;
                        }
                        std::discrete_distribution<int32_t> sample(dist.probs.begin(), dist.probs.end());
                        predecessor = sample(selector_rng[seq_id]);
                        if (dist.probs[predecessor] < params.p_min) {
                            break;
                        }
                        result.push_back(dist.ids[predecessor]);
                        dp.dists->push_back(std::move(dist));
                    } else {
                        predecessor = (int32_t) std::distance(scores,
                                std::max_element(scores, scores + selector_top_k));
                        if (params.p_min > 0.0f) {
                            // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                            float sum = 0.0f;
                            for (int32_t k = 0; k < selector_top_k; ++k) {
                                sum += std::exp(scores[k] - scores[predecessor]);
                            }
                            if (1.0f / sum < params.p_min) {
                                break;
                            }
                        }
                        result.push_back((llama_token) row[predecessor]);
                    }
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (dp.dists) {
                        dp.dists->clear();
                    }
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    bool state_required(llama_seq_id /*seq_id*/) const override {
        return sidecar.active();
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) override {
        if (!sidecar.active()) return false;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && sidecar_stale[seq_id]) return false;
        if (!sidecar.get_state(seq_id, data)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "DFlash sidecar state snapshot failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!sidecar.active()) return true;
        if (!sidecar.set_state(seq_id, data)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "DFlash sidecar state restore failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool reset_state(llama_seq_id seq_id) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && sidecar_stale[seq_id]) {
            sidecar_stale[seq_id] = false;
            SPC_INF("DFlash sidecar: seq %d re-armed after reset\n", (int) seq_id);
        }
        if (!sidecar.reset_state(seq_id)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "DFlash sidecar state reset failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool truncate_state(llama_seq_id seq_id, llama_pos pos_max) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && sidecar_stale[seq_id]) return true;
        if (!sidecar.truncate_state(seq_id, pos_max)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "DFlash sidecar state truncate failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool commit_state(llama_seq_id seq_id, llama_pos pos_max) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && sidecar_stale[seq_id]) return true;
        if (!sidecar.commit_state(seq_id, pos_max)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "DFlash sidecar state commit failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool rebase_state(llama_seq_id seq_id, llama_pos pos_min, llama_pos pos_max, llama_pos delta) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && sidecar_stale[seq_id]) return true;
        if (!sidecar.rebase_state(seq_id, pos_min, pos_max, delta)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "DFlash sidecar state rebase failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || verify_pos_first[seq_id] < 0 || verify_rows[seq_id] <= 0) {
            return;
        }
        const int32_t n_commit = std::min<int32_t>((int32_t) n_accepted + 1, verify_rows[seq_id]);
        commit_state(seq_id, verify_pos_first[seq_id] + n_commit);
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    common_spec_sidecar_mtp sidecar;
    common_spec_sidecar_paths sidecar_paths;
    int32_t sidecar_embedding_width = 0;
    int32_t sidecar_head_rows = 0;
    bool sidecar_load_pending = false;
    bool sidecar_target_only = false; // runtime failure or unsupported sampling mode
    bool sidecar_catchup_deferred = false;
    bool sidecar_catchup_deferred_logged = false;
    std::vector<std::vector<int32_t>> sidecar_deferred_tokens;
    std::vector<std::vector<int32_t>> sidecar_deferred_pos;
    std::vector<const float *> verify_h_device;
    std::vector<size_t> verify_h_device_stride;

    llama_batch batch = {};

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;
    std::vector<llama_pos> verify_pos_first;
    // Sequences whose sidecar cursor no longer mirrors the target prefix
    // (e.g. an image prompt delivered embedding batches with no token ids).
    // Stale sequences skip sidecar catch-up/drafting until the next reset
    // instead of disabling the sidecar for the whole process.
    std::vector<bool> mtp_sidecar_stale;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    std::vector<llama_token> deferred_tokens;
    std::vector<llama_pos> deferred_pos;
    std::vector<float> deferred_embd;
    bool deferred_catchup_ready = false;
    bool deferred_catchup_logged = false;
    bool deferred_auto_model = false;
    bool deferred_replay_logged = false;

    void clear_deferred_catchup() {
        deferred_tokens.clear();
        deferred_pos.clear();
        deferred_embd.clear();
        deferred_catchup_ready = false;
    }

    void clear_sidecar_deferred_catchup() {
        for (auto & values : sidecar_deferred_tokens) values.clear();
        for (auto & values : sidecar_deferred_pos) values.clear();
        sidecar_catchup_deferred = false;
    }

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        const bool sidecar_only = this->params.sidecar_only &&
                this->params.sidecar_type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
        const common_spec_sidecar_profile * sidecar_profile = this->params.sidecar_profile;
        GGML_ASSERT(ctx_tgt && (ctx_dft != nullptr || sidecar_only) &&
                "MTP requires a target context or a validated sidecar-only mode");

        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_dft = ctx_dft != nullptr ? llama_get_model(ctx_dft) : model_tgt;
        n_embd = llama_model_n_embd_out(model_dft);
        GGML_ASSERT(n_embd == llama_model_n_embd_out(model_tgt) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(model_dft));
        char target_arch[32] = {};
        const int32_t arch_len = llama_model_meta_val_str(
            llama_get_model(ctx_tgt), "general.architecture", target_arch, sizeof(target_arch));
        deferred_auto_model = common_speculative_rdna2_auto_enabled() &&
                arch_len >= 0 && std::strcmp(target_arch, "qwen35") == 0 && n_embd == 5120;

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        if (ctx_dft != nullptr) {
            const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
            batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
            // llama_batch_init allocates only one of token/embd; MTP needs both.
            // TODO: fix, how to call without malloc
            batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);
        }

        if (ctx_dft != nullptr) {
            smpls.resize(n_seq);
            for (auto & s : smpls) {
                common_params_sampling sparams;
                sparams.no_perf  = false;
                sparams.top_k    = 10;
                sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
                s.reset(common_sampler_init(model_dft, sparams));
            }
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (ctx_dft != nullptr && this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        if (ctx_dft != nullptr) {
            llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        }

        is_mem_shared = ctx_dft != nullptr && llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        if (sidecar_only && n_mtp_layers == 1 && !chain_heads && !is_mem_shared &&
                sidecar_profile != nullptr && sidecar_profile->kind == COMMON_SPEC_SIDECAR_KIND_MTP) {
            std::string error;
            if (common_spec_sidecar_get_paths(*sidecar_profile, sidecar_paths, error)) {
                sidecar_embedding_width = sidecar_profile->mtp_embedding_width;
                sidecar_head_rows = sidecar_profile->mtp_head_rows;
                // Retain h_nextn on the backend from the first evaluation;
                // the actual device ID is still bound lazily from its view.
                llama_set_embeddings_nextn_device_preferred(ctx_tgt, true);
                sidecar_load_pending = true;
            } else {
                sidecar_target_only = true;
                SPC_WRN("MTP sidecar unavailable (%s); target-only mode\n", error.c_str());
            }
        }
        if (sidecar_only && !sidecar.active() && !sidecar_load_pending) {
            sidecar_target_only = true;
        }

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
        verify_pos_first.assign(n_seq, -1);
        mtp_sidecar_stale.assign(n_seq, false);
        verify_h_device.assign(n_seq, nullptr);
        verify_h_device_stride.assign(n_seq, 0);
        sidecar_deferred_tokens.assign(n_seq, {});
        sidecar_deferred_pos.assign(n_seq, {});
    }

    void prepare_process(const common_speculative_draft_params_vec & dparams) override {
        if (!sidecar.active() && !sidecar_load_pending) {
            return;
        }
        for (const auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }
            if (dp.temperature > 0.0f && dp.dists == nullptr) {
                sidecar.disable();
                sidecar_load_pending = false;
                sidecar_target_only = true;
                SPC_WRN("%s", "MTP sidecar requires proposal distributions for stochastic sampling; target-only mode\n");
                break;
            }
            if (dp.temperature <= 0.0f && params.p_min > 0.0f) {
                sidecar.disable();
                sidecar_load_pending = false;
                sidecar_target_only = true;
                SPC_WRN("%s", "MTP sidecar does not support p_min in greedy mode; target-only mode\n");
                break;
            }
        }
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        if (ctx_dft == nullptr) {
            return;
        }
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process_sidecar(const llama_batch & batch_in,
            const std::vector<int32_t> & i_batch_beg,
            const std::vector<int32_t> & i_batch_end,
            bool defer_catchup) {
        auto * ctx_tgt = this->params.ctx_tgt;
        llama_device_view device_view;
        bool have_device_view = false;
        bool direct = false;
        int32_t target_device = -1;
#if defined(GGML_USE_HIP)
        have_device_view = llama_get_embeddings_nextn_device(ctx_tgt, &device_view);
        if (have_device_view) {
            target_device = device_view.device;
        }
#endif

        if (sidecar_load_pending) {
            std::string error;
            // The first target evaluation gives us the backend/device that
            // owns h_nextn. Bind the sidecar there before accepting any rows.
            if (!sidecar.load(sidecar_paths.library, sidecar_paths.artifact_dir,
                    sidecar_paths.ids, sidecar_embedding_width, sidecar_head_rows,
                    (int32_t) n_seq, error, target_device)) {
                sidecar_load_pending = false;
                sidecar_target_only = true;
                SPC_WRN("MTP sidecar unavailable on target device %d (%s); target-only mode\n",
                        target_device, error.c_str());
                return true;
            }
            sidecar_load_pending = false;
            llama_set_embeddings_nextn_device_preferred(ctx_tgt, true);
            SPC_INF("MTP sidecar active: %s (bound device=%d)\n",
                    sidecar_paths.library.c_str(), target_device);
        }

#if defined(GGML_USE_HIP)
        const bool view_shape_ok = have_device_view &&
                device_view.row_stride == (size_t) n_embd * sizeof(float) &&
                device_view.n_rows >= (uint32_t) batch_in.n_tokens;
        direct = view_shape_ok && sidecar.attach_target_stream(device_view.stream, device_view.device);
#endif
        if (std::getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr) {
            SPC_DBG("MTP catch-up input mode: %s (target device=%d)\n",
                    direct ? "DIRECT D2D" : "HOST FALLBACK", target_device);
        }

        // If any sequence is interleaved, direct rows cannot be represented by
        // one contiguous pointer for that sequence. The host fallback gathers
        // only that sequence while other contiguous sequences retain D2D input.
        std::vector<bool> contiguous(n_seq, true);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) continue;
            for (int32_t i = i_batch_beg[seq_id]; i <= i_batch_end[seq_id]; ++i) {
                if (batch_in.seq_id[i][0] != seq_id) {
                    contiguous[seq_id] = false;
                    break;
                }
            }
        }
        bool need_host = false;
        for (bool value : contiguous) need_host = need_host || !value;
        float * host_h = nullptr;
        if (!direct || need_host) {
            host_h = llama_get_embeddings_nextn(ctx_tgt);
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < beg) continue;

            std::vector<int32_t> indices;
            if (!contiguous[seq_id]) {
                for (int32_t i = 0; i < batch_in.n_tokens; ++i) {
                    if (batch_in.seq_id[i][0] == seq_id) indices.push_back(i);
                }
            }
            const int32_t n_rows = contiguous[seq_id] ? end - beg + 1 : (int32_t) indices.size();
            if (n_rows <= 0) continue;
            if (mtp_sidecar_stale[seq_id]) {
                verify_h_rows[seq_id] = 0;
                verify_pos_first[seq_id] = -1;
                sidecar_deferred_tokens[seq_id].clear();
                sidecar_deferred_pos[seq_id].clear();
                continue;
            }
            verify_h_rows[seq_id] = n_rows;
            verify_pos_first[seq_id] = batch_in.pos[contiguous[seq_id] ? beg : indices.front()];
            verify_h_device[seq_id] = nullptr;
            verify_h_device_stride[seq_id] = 0;
            verify_h[seq_id].clear();
            sidecar_deferred_tokens[seq_id].clear();
            sidecar_deferred_pos[seq_id].clear();

            int rc = -1;
            if (direct && contiguous[seq_id]) {
                const float * rows = static_cast<const float *>(device_view.data) +
                        (size_t) beg * device_view.row_stride / sizeof(float);
                verify_h_device[seq_id] = rows;
                verify_h_device_stride[seq_id] = device_view.row_stride;
                sidecar_deferred_tokens[seq_id].assign(
                        batch_in.token + beg, batch_in.token + beg + n_rows);
                sidecar_deferred_pos[seq_id].assign(
                        batch_in.pos + beg, batch_in.pos + beg + n_rows);
                rc = defer_catchup ? 0 : sidecar.catchup_device(
                        seq_id, sidecar_deferred_tokens[seq_id].data(),
                        sidecar_deferred_pos[seq_id].data(), rows, n_rows);
            } else {
                std::vector<int32_t> tokens((size_t) n_rows);
                std::vector<int32_t> positions((size_t) n_rows);
                verify_h[seq_id].resize((size_t) n_rows * n_embd);
                for (int32_t r = 0; r < n_rows; ++r) {
                    const int32_t i = contiguous[seq_id] ? beg + r : indices[r];
                    tokens[r] = batch_in.token[i];
                    positions[r] = batch_in.pos[i];
                    std::memcpy(verify_h[seq_id].data() + (size_t) r * n_embd,
                            host_h + (size_t) i * n_embd, (size_t) n_embd * sizeof(float));
                }
                sidecar_deferred_tokens[seq_id] = tokens;
                sidecar_deferred_pos[seq_id] = positions;
                rc = defer_catchup ? 0 : sidecar.catchup(
                        seq_id, tokens.data(), positions.data(),
                        verify_h[seq_id].data(), n_rows);
            }
            if (rc != 0) {
                // Position/token mismatch (e.g. image prompt) or transient
                // failure: pause this sequence until reset, keep the sidecar.
                mtp_sidecar_stale[seq_id] = true;
                verify_h_rows[seq_id] = 0;
                verify_pos_first[seq_id] = -1;
                sidecar_deferred_tokens[seq_id].clear();
                sidecar_deferred_pos[seq_id].clear();
                SPC_WRN("MTP sidecar: seq %d catch-up rejected; drafting paused until reset\n", (int) seq_id);
            }
        }
        if (defer_catchup) {
            sidecar_catchup_deferred = true;
            if (!sidecar_catchup_deferred_logged) {
                SPC_INF("%s", "using deferred MTP sidecar catch-up scheduling\n");
                sidecar_catchup_deferred_logged = true;
            }
        }
        return true;
    }

    bool process(const llama_batch & batch_in) override {
        // If accept() did not consume a captured batch, the server took its
        // checkpoint/replay branch and restored the draft context. Do not let
        // that stale capture leak into the replay or the next request.
        if (deferred_catchup_ready) {
            if (!deferred_replay_logged) {
                SPC_INF("%s", "discarding deferred MTP catch-up after checkpoint/replay restore\n");
                deferred_replay_logged = true;
            }
            clear_deferred_catchup();
        }
        if (sidecar_catchup_deferred) {
            SPC_INF("%s", "discarding deferred MTP sidecar catch-up after checkpoint/replay restore\n");
            clear_sidecar_deferred_catchup();
        }

        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Image/multimodal embedding batches carry no token ids, so the MTP
        // catch-up cannot represent them. Pause the affected sequences until
        // the next reset instead of disabling the sidecar for the process.
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            if (sidecar.active()) {
                for (int32_t k = 0; k < batch_in.n_tokens; ++k) {
                    const llama_seq_id sid = batch_in.seq_id != nullptr ? batch_in.seq_id[k][0] : 0;
                    if (sid >= 0 && sid < (llama_seq_id) n_seq && !mtp_sidecar_stale[sid]) {
                        mtp_sidecar_stale[sid] = true;
                        SPC_WRN("MTP sidecar: seq %d stale after non-token batch; drafting paused until reset\n", (int) sid);
                    }
                }
            }
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        const char * sidecar_defer_env = std::getenv("GGML_MTP_DEFER_CATCHUP");
        const bool sidecar_defer_force = sidecar_defer_env != nullptr &&
                std::strcmp(sidecar_defer_env, "1") == 0;
        const bool sidecar_defer_auto = sidecar_defer_env == nullptr ||
                std::strcmp(sidecar_defer_env, "auto") == 0;
        const bool sidecar_defer_requested = common_speculative_rdna2_auto_enabled() &&
                (sidecar_defer_force || (sidecar_defer_auto && deferred_auto_model));
        bool sidecar_all_logits = batch_in.logits != nullptr;
        for (int32_t k = 0; sidecar_all_logits && k < n_tokens; ++k) {
            sidecar_all_logits = batch_in.logits[k] != 0;
        }
        const bool sidecar_defer_catchup = (sidecar.active() || sidecar_load_pending) &&
                sidecar_defer_requested && n_seq == 1 && n_mtp_layers == 1 &&
                !chain_heads && !is_mem_shared && this->params.n_max == 4 &&
                n_tokens == 5 && sidecar_all_logits;

        if (sidecar.active() || sidecar_load_pending) {
            if (batch_in.pos == nullptr) {
                sidecar.disable();
                sidecar_load_pending = false;
                sidecar_target_only = true;
                SPC_ERR("%s", "MTP sidecar requires explicit target positions; entering target-only mode\n");
                return true;
            }
            return process_sidecar(batch_in, i_batch_beg, i_batch_end,
                    sidecar_defer_catchup);
        }
        if (sidecar_target_only) {
            return true;
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // Opt-in Qwen MTP path: preserve the exact full-width verification
        // batch but schedule draft-context catch-up after target acceptance.
        // Requiring logits on all rows distinguishes verification from an
        // unrelated five-token prompt/prefill batch.
        const char * defer_env = std::getenv("GGML_MTP_DEFER_CATCHUP");
        const bool defer_force = defer_env != nullptr && std::strcmp(defer_env, "1") == 0;
        const bool defer_auto = defer_env == nullptr || std::strcmp(defer_env, "auto") == 0;
        const bool defer_requested = common_speculative_rdna2_auto_enabled() &&
                (defer_force || (defer_auto && deferred_auto_model));
        bool all_logits = batch_in.logits != nullptr;
        for (int32_t k = 0; all_logits && k < n_tokens; ++k) {
            all_logits = batch_in.logits[k] != 0;
        }
        const bool defer_catchup = defer_requested &&
                n_seq == 1 && n_mtp_layers == 1 && !chain_heads && !is_mem_shared &&
                this->params.n_max == 4 && n_tokens == 5 && all_logits;

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            if (defer_catchup) {
                deferred_tokens.assign(batch.token, batch.token + batch.n_tokens);
                deferred_pos.assign(batch.pos, batch.pos + batch.n_tokens);
                deferred_embd.assign(batch.embd, batch.embd + (size_t) batch.n_tokens * n_embd);
                deferred_catchup_ready = true;
                if (!deferred_catchup_logged) {
                    SPC_INF("%s", "using deferred full-width MTP draft-context catch-up scheduling\n");
                    deferred_catchup_logged = true;
                }
            } else {
                auto * mem_dft = llama_get_memory(ctx_dft);

                bool ok = true;
                for (int head = 0; head < n_mtp_layers; ++head) {
                    if (chain_heads) {
                        // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                            if (i_batch_beg[seq_id] < 0) {
                                continue;
                            }
                            llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                        }
                        llama_set_nextn_layer_offset(ctx_dft, head);
                    }

                    const int32_t rc = llama_decode(ctx_dft, batch);
                    if (rc != 0) {
                        SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                                head, (int) rc, (int) batch_in.pos[0]);
                        ok = false;
                        break;
                    }
                }

                if (chain_heads) {
                    llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
                }
                if (!ok) {
                    return false;
                }
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_pos_first[seq_id] = batch_in.pos != nullptr
                    ? batch_in.pos[i_batch_beg[seq_id]] : -1;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        if (sidecar_target_only) {
            return;
        }

        if (sidecar.active()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & dp = dparams[seq_id];
                if (!dp.drafting || mtp_sidecar_stale[seq_id]) {
                    continue;
                }
                const bool stochastic = dp.temperature > 0.0f;
                if (!stochastic && params.p_min > 0.0f) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    return;
                }
                if (stochastic && dp.dists == nullptr) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    SPC_ERR("%s", "MTP sidecar has no proposal-distribution output; entering target-only mode\n");
                    return;
                }

                const int limit = std::min(params.n_max, dp.n_max > 0 ? dp.n_max : params.n_max);
                if (limit < 1) {
                    return;
                }
                std::vector<int32_t> ids((size_t) limit, -1);
                std::vector<int32_t> dist_ids;
                std::vector<float> dist_probs;
                if (stochastic) {
                    dist_ids.resize((size_t) limit * SPEC_SIDECAR_MTP_DRAFT_TOP_K, -1);
                    dist_probs.resize((size_t) limit * SPEC_SIDECAR_MTP_DRAFT_TOP_K, 0.0f);
                } else if (dp.dists != nullptr) {
                    dp.dists->clear();
                }
                const int n = stochastic
                        ? (verify_h_device[seq_id] != nullptr
                            ? sidecar.draft_stochastic_device(seq_id, dp.id_last, (int32_t) dp.n_past,
                                    dp.temperature, params.p_min,
                                    common_spec_sidecar_stochastic_key(dp.seed, seq_id, dp.n_past, 1),
                                    limit, ids.data(), dist_ids.data(), dist_probs.data())
                            : sidecar.draft_stochastic(seq_id, dp.id_last, (int32_t) dp.n_past,
                                    pending_h[seq_id].data(), dp.temperature, params.p_min,
                                    common_spec_sidecar_stochastic_key(dp.seed, seq_id, dp.n_past, 1),
                                    limit, ids.data(), dist_ids.data(), dist_probs.data()))
                        : (verify_h_device[seq_id] != nullptr
                            ? sidecar.draft_device(seq_id, dp.id_last, (int32_t) dp.n_past, limit, ids.data())
                            : sidecar.draft(seq_id, dp.id_last, (int32_t) dp.n_past,
                                    pending_h[seq_id].data(), limit, ids.data()));
                if (n < 0 || n > limit) {
                    sidecar.disable();
                    sidecar_target_only = true;
                    SPC_ERR("%s", "MTP sidecar draft failed; entering target-only mode\n");
                    return;
                }

                const int32_t n_vocab = llama_vocab_n_tokens(
                        llama_model_get_vocab(llama_get_model(params.ctx_tgt)));
                auto & result = *dp.result;
                if (stochastic) {
                    dp.dists->clear();
                    dp.dists->reserve((size_t) n);
                }
                for (int i = 0; i < n; ++i) {
                    if (ids[i] < 0 || ids[i] >= n_vocab) {
                        result.clear();
                        if (dp.dists != nullptr) dp.dists->clear();
                        sidecar.disable();
                        sidecar_target_only = true;
                        SPC_ERR("%s", "MTP sidecar returned invalid token ID; entering target-only mode\n");
                        return;
                    }
                    result.push_back((llama_token) ids[i]);
                    if (stochastic) {
                        common_speculative_token_dist dist;
                        if (!common_spec_sidecar_validate_distribution(
                                    dist_ids.data() + (size_t) i * SPEC_SIDECAR_MTP_DRAFT_TOP_K,
                                    dist_probs.data() + (size_t) i * SPEC_SIDECAR_MTP_DRAFT_TOP_K,
                                    SPEC_SIDECAR_MTP_DRAFT_TOP_K, n_vocab, dist)) {
                            result.clear();
                            dp.dists->clear();
                            sidecar.disable();
                            sidecar_target_only = true;
                            SPC_ERR("%s", "MTP sidecar returned invalid proposal distribution; entering target-only mode\n");
                            return;
                        }
                        dp.dists->push_back(std::move(dist));
                    }
                }
                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (dp.dists != nullptr) dp.dists->clear();
                }
            }
            return;
        }

        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }
            // The MTP output row is copied asynchronously by the backend, but
            // the next sampler step consumes it immediately on the host.
            llama_synchronize(ctx_dft);

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    bool state_required(llama_seq_id /*seq_id*/) const override {
        return sidecar.active();
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) override {
        if (!sidecar.active()) return false;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && mtp_sidecar_stale[seq_id]) return false;
        if (!sidecar.get_state(seq_id, data)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state snapshot failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!sidecar.active()) return true;
        if (!sidecar.set_state(seq_id, data)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state restore failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool reset_state(llama_seq_id seq_id) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && mtp_sidecar_stale[seq_id]) {
            mtp_sidecar_stale[seq_id] = false;
            SPC_INF("MTP sidecar: seq %d re-armed after reset\n", (int) seq_id);
        }
        if (!sidecar.reset_state(seq_id)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state reset failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    void release_state(llama_seq_id /*seq_id*/) override {
        // The server can retain this slot's target KV and prompt after request
        // completion. Keep the committed sidecar KV with it; prepare_prompt_state()
        // validates the exact target/sidecar boundary before the next reuse.
    }

    bool prepare_prompt_state(
            llama_seq_id seq_id, llama_pos pos_next, bool can_reuse_resident) override {
        if (sidecar_load_pending) {
            // A lazily loaded sidecar has no prior KV. Position zero is the only
            // boundary that can be seeded without replaying the target prompt.
            return can_reuse_resident && pos_next == 0;
        }
        if (!sidecar.active()) {
            return true;
        }

        bool cursor_matches = false;
        if (can_reuse_resident) {
            std::vector<uint8_t> data;
            spec_sidecar_state state = {};
            if (sidecar.get_state(seq_id, data) && data.size() == sizeof(state)) {
                std::memcpy(&state, data.data(), sizeof(state));
                cursor_matches =
                        state.magic   == SPEC_SIDECAR_STATE_MAGIC &&
                        state.version == SPEC_SIDECAR_STATE_VERSION &&
                        state.kind    == SPEC_SIDECAR_STATE_KIND_MTP &&
                        state.pos_min >= 0 && state.pos_min <= state.pos_max &&
                        state.pos_max == pos_next;
            }
        }

        if (cursor_matches && sidecar.truncate_state(seq_id, pos_next)) {
            // Truncating to the current tip is a no-op for committed KV/hidden
            // state, but discards any uncommitted catch-up rows from a cancelled
            // request before the slot is reused.
            SPC_DBG("reusing MTP sidecar state: seq=%d, pos=%d\n", (int) seq_id, (int) pos_next);
            return true;
        }

        if (!reset_state(seq_id)) {
            return false;
        }
        SPC_DBG("reset MTP sidecar state for prompt replay: seq=%d, target_pos=%d, resident=%d\n",
                (int) seq_id, (int) pos_next, (int) can_reuse_resident);
        return false;
    }

    bool truncate_state(llama_seq_id seq_id, llama_pos pos_max) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && mtp_sidecar_stale[seq_id]) return true;
        if (!sidecar.truncate_state(seq_id, pos_max)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state truncate failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool commit_state(llama_seq_id seq_id, llama_pos pos_max) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && mtp_sidecar_stale[seq_id]) return true;
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state commit sequence invalid; entering target-only mode\n");
            return false;
        }
        const int32_t first = verify_pos_first[seq_id];
        const int32_t rows = verify_h_rows[seq_id];
        if (first < 0 || rows <= 0 ||
                pos_max < first || pos_max > first + rows) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state commit boundary invalid; entering target-only mode\n");
            return false;
        }
        const int32_t row = (int32_t) (pos_max - first) - 1;
        const float * hidden_device = verify_h_device[seq_id] != nullptr && row >= 0
                ? verify_h_device[seq_id] + (size_t) row * verify_h_device_stride[seq_id] / sizeof(float)
                : nullptr;
        if (!sidecar.commit_state(seq_id, pos_max, hidden_device)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state commit failed; entering target-only mode\n");
            return false;
        }
        if (hidden_device == nullptr && row >= 0 &&
                (size_t) (row + 1) * n_embd <= verify_h[seq_id].size()) {
            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) row * n_embd,
                    (size_t) n_embd * sizeof(float));
        }
        return true;
    }

    bool rebase_state(llama_seq_id seq_id, llama_pos pos_min, llama_pos pos_max, llama_pos delta) override {
        if (!sidecar.active()) return true;
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && mtp_sidecar_stale[seq_id]) return true;
        if (!sidecar.rebase_state(seq_id, pos_min, pos_max, delta)) {
            sidecar.disable();
            sidecar_target_only = true;
            SPC_ERR("%s", "MTP sidecar state rebase failed; entering target-only mode\n");
            return false;
        }
        return true;
    }

    bool flush_sidecar_deferred_catchup() {
        if (!sidecar_catchup_deferred) return true;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int count = (int) sidecar_deferred_tokens[seq_id].size();
            if (count <= 0 || sidecar_deferred_pos[seq_id].size() != (size_t) count) {
                continue;
            }
            const int rc = verify_h_device[seq_id] != nullptr
                    ? sidecar.catchup_device(seq_id,
                            sidecar_deferred_tokens[seq_id].data(),
                            sidecar_deferred_pos[seq_id].data(),
                            verify_h_device[seq_id], count)
                    : sidecar.catchup(seq_id,
                            sidecar_deferred_tokens[seq_id].data(),
                            sidecar_deferred_pos[seq_id].data(),
                            verify_h[seq_id].data(), count);
            if (rc != 0) {
                sidecar.disable();
                sidecar_target_only = true;
                clear_sidecar_deferred_catchup();
                SPC_ERR("%s", "MTP sidecar deferred catch-up failed; entering target-only mode\n");
                return false;
            }
        }
        clear_sidecar_deferred_catchup();
        return true;
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        if (sidecar.active()) {
            if (!flush_sidecar_deferred_catchup()) {
                return;
            }
            const int32_t n_commit = std::min<int32_t>((int32_t) n_accepted + 1, n_rows);
            commit_state(seq_id, verify_pos_first[seq_id] + n_commit);
        }

        if (deferred_catchup_ready) {
            if (seq_id != 0 || deferred_tokens.size() != 5 || deferred_pos.size() != 5 ||
                    deferred_embd.size() != (size_t) 5 * n_embd) {
                SPC_ERR("%s", "discarding malformed deferred MTP catch-up batch\n");
                clear_deferred_catchup();
                return;
            }
            common_batch_clear(batch);
            for (int32_t i = 0; i < 5; ++i) {
                common_batch_add(batch, deferred_tokens[i], deferred_pos[i], { seq_id }, 0);
                std::memcpy(batch.embd + (size_t) i * n_embd,
                        deferred_embd.data() + (size_t) i * n_embd, (size_t) n_embd * sizeof(float));
            }
            const int32_t rc = llama_decode(params.ctx_dft, batch);
            clear_deferred_catchup();
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed for deferred MTP catch-up rc=%d\n", (int) rc);
                return;
            }
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        if (verify_h_device[seq_id] == nullptr) {
            std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
        }
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;
    common_speculative_sidecar_cap_config sidecar_cap;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config,
            common_speculative_sidecar_cap_config cap_cfg = {})
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
        , sidecar_cap(cap_cfg)
    {

        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
            common_speculative_sidecar_cap_trim(sidecar_cap, dp, *dp.result);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;
    common_speculative_sidecar_cap_config sidecar_cap;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq,
            common_speculative_sidecar_cap_config cap_cfg = {})
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
        , sidecar_cap(cap_cfg)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }



        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);


        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const bool cap_request = common_speculative_sidecar_cap_request_enabled(
                    sidecar_cap, dp);
            config[seq_id].draft_limit = cap_request
                    ? (uint16_t) std::min(common_speculative_sidecar_cap_limit(
                            sidecar_cap, dp), (int) UINT16_MAX)
                    : 0;
            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
            common_speculative_sidecar_cap_trim(sidecar_cap, dp, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) config.size()) {
            return;
        }
        if (!is_other) {
            common_ngram_map_accept(config[seq_id], n_accepted);
        }
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;
    common_speculative_sidecar_cap_config sidecar_cap;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq,
            common_speculative_sidecar_cap_config cap_cfg = {})
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr)
        , sidecar_cap(cap_cfg)
        {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);

    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;


        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
            common_speculative_sidecar_cap_trim(sidecar_cap, dp, *dp.result);
            sinfos[seq_id].n_draft_last = dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;
    common_speculative_sidecar_cap_config sidecar_cap;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static,
            common_speculative_sidecar_cap_config cap_cfg = {})
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
        , sidecar_cap(cap_cfg)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);


        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
            common_speculative_sidecar_cap_trim(sidecar_cap, dp, *dp.result);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic,
        common_speculative_sidecar_cap_config cap_cfg = {}) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static,
            path_dynamic, save_static, save_dynamic, cap_cfg);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static bool common_speculative_has_type(const common_params_speculative & params,
        common_speculative_type type) {
    return std::find(params.types.begin(), params.types.end(), type) != params.types.end();
}

static bool common_speculative_has_host_draft_type(const common_params_speculative & params) {
    return common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) ||
           common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ||
           common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
}

bool common_speculative_sidecar_candidate(const common_params_speculative & params,
        const std::string & target_model_path, uint32_t n_seq) {
    if (!common_speculative_sidecar_enabled()) {
        return false;
    }

    const bool has_mtp = common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    const bool has_dflash = common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
    if (common_speculative_has_host_draft_type(params) || (has_mtp && has_dflash)) {
        return false;
    }

    std::string error;
    if (has_mtp) {
        const auto * profile = common_spec_sidecar_profile_for_target_file(
                COMMON_SPEC_SIDECAR_KIND_MTP, target_model_path, error);
        if (profile != nullptr && common_spec_sidecar_probe(*profile, n_seq, error)) {
            return true;
        }
    }
    if (has_dflash) {
        const auto * profile = common_spec_sidecar_profile_for_target_file(
                COMMON_SPEC_SIDECAR_KIND_DFLASH, target_model_path, error);
        if (profile != nullptr && common_spec_sidecar_probe(*profile, n_seq, error)) {
            return true;
        }
    }
    return false;
}

common_speculative_type common_speculative_sidecar_preflight(
        common_params_speculative & params, const llama_model * model_tgt,
        uint32_t n_seq, std::string & error) {
    if (!common_speculative_sidecar_enabled()) {
        params.draft.sidecar_only = false;
        params.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_NONE;
        params.draft.sidecar_profile = nullptr;
        error.clear();
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    params.draft.sidecar_only = false;
    params.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_NONE;
    params.draft.sidecar_profile = nullptr;
    error.clear();
    const bool has_mtp = common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    const bool has_dflash = common_speculative_has_type(params, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
    if (model_tgt == nullptr || n_seq < 1 || n_seq > 8 ||
            common_speculative_has_host_draft_type(params) || (has_mtp && has_dflash)) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    std::string probe_error;
    if (has_mtp) {
        const auto * profile = common_spec_sidecar_profile_for_model(
                COMMON_SPEC_SIDECAR_KIND_MTP, model_tgt, probe_error);
        if (profile != nullptr && common_spec_sidecar_probe(*profile, n_seq, probe_error)) {
            params.draft.sidecar_only = true;
            params.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
            params.draft.sidecar_profile = profile;
            return params.draft.sidecar_type;
        }
    }

    if (has_dflash) {
        const auto * profile = common_spec_sidecar_profile_for_model(
                COMMON_SPEC_SIDECAR_KIND_DFLASH, model_tgt, probe_error);
        if (profile != nullptr && common_spec_sidecar_probe(*profile, n_seq, probe_error)) {
            params.draft.sidecar_only = true;
            params.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
            params.draft.sidecar_profile = profile;
            return params.draft.sidecar_type;
        }
    }

    if (!probe_error.empty()) {
        error = probe_error;
    }
    return COMMON_SPECULATIVE_TYPE_NONE;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // Preserve the target device selection unless the draft devices were
        // explicitly configured. Falling back to library auto-selection can
        // put a shared DFlash tensor on a backend the draft scheduler does not
        // own, which aborts on heterogeneous or multi-GPU systems.
        if (!params_spec.devices.empty()) {
            result.devices = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        // The noise block is decoded as a whole under non-causal attention, so it must fit in a single ubatch.
        const int32_t n_block = params.n_parallel * per_seq;
        result.n_outputs_max = n_block;
        result.n_batch = std::max(result.n_batch, n_block);

        // The draft only decodes a small block during generation, so avoid reserving a compute buffer for the
        // full inherited ubatch. The explicit override wins, while every value is floored to the noise block.
        const int32_t n_ubatch_dft_cap = 128;
        const int32_t n_ubatch_dft_req = params_spec.n_ubatch > 0 ? params_spec.n_ubatch
                                                                  : std::min(result.n_ubatch, n_ubatch_dft_cap);
        result.n_ubatch = std::max(n_ubatch_dft_req, n_block);

        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
    bool sidecar_only = false;
    common_speculative_type sidecar_type = COMMON_SPECULATIVE_TYPE_NONE;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    // Draft placement/cache/device flags live in speculative.draft; build the
    // draft params from that sub-structure rather than inheriting target-only
    // tensor overrides and paths.
    common_params params_dft = common_base_params_to_speculative(params);
    auto mparams = common_model_params_to_llama(params_dft);
    auto cparams = common_context_params_to_llama(params_dft);

    // DFlash and DSpark draft weights do not define tensor-parallel split semantics.
    // Keep the target tensor-parallel while distributing the draft by whole layers;
    // this also keeps the draft scheduler compatible with the shared target output head.
    const bool spec_dflash = std::find(params.speculative.types.begin(),
            params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) !=
            params.speculative.types.end();
    const bool spec_dspark = std::find(params.speculative.types.begin(),
            params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) !=
            params.speculative.types.end();
    if ((spec_dflash || spec_dspark) && mparams.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
        LOG_INF("%s: using layer split for DFlash/DSpark draft weights while retaining target tensor split\n", __func__);
    }

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string sidecar_error;
    pimpl->sidecar_type = common_speculative_sidecar_preflight(
            params.speculative, model_tgt, params.n_parallel, sidecar_error);
    pimpl->sidecar_only = pimpl->sidecar_type != COMMON_SPECULATIVE_TYPE_NONE;
    if (pimpl->sidecar_only) {
        LOG_INF("%s: sidecar-only draft selected (%s); skipping host draft model/context load\n",
                __func__, common_speculative_type_to_str(pimpl->sidecar_type).c_str());
    } else if (!sidecar_error.empty()) {
        LOG_WRN("%s: sidecar probe unavailable; retaining native draft loading: %s\n",
                __func__, sidecar_error.c_str());
    }

    // If the path-based candidate probe promised a sidecar at target-model load
    // time (allowing the shared output head to be sharded), a preflight failure
    // must fail closed: loading the head-sharing host draft against a sharded
    // head would be incorrect.
    if (!pimpl->sidecar_only && (spec_dflash || spec_dspark) &&
            common_speculative_sidecar_candidate(params.speculative,
                params.model.path, params.n_parallel)) {
        LOG_ERR("%s: sidecar candidate accepted at model load but preflight failed (%s); "
                "refusing host draft fallback against a possibly sharded output head\n",
                __func__, sidecar_error.c_str());
        return;
    }

    std::string model_path;
    if (has_draft && !pimpl->sidecar_only) {
        model_path = params_dft.model.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp && !pimpl->sidecar_only) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

bool common_speculative_init_result::sidecar_only() const {
    return pimpl->sidecar_only;
}

common_speculative_type common_speculative_init_result::sidecar_type() const {
    return pimpl->sidecar_type;
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        const bool has_draft_ctx = params.draft.ctx_dft != nullptr;
        const bool mtp_sidecar = params.draft.sidecar_only &&
                params.draft.sidecar_type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
        const bool dflash_sidecar = params.draft.sidecar_only &&
                params.draft.sidecar_type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, has_draft_ctx);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, has_draft_ctx);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, has_draft_ctx || mtp_sidecar);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, has_draft_ctx || dflash_sidecar);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, has_draft_ctx);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    const bool mtp_sidecar = params.draft.sidecar_only &&
            params.draft.sidecar_type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
    const bool has_ngram = std::any_of(params.types.begin(), params.types.end(), [](common_speculative_type type) {
        return type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE ||
               type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K ||
               type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V ||
               type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD ||
               type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
    });
    if (mtp_sidecar && has_ngram && params.draft.n_max > 0) {
        SPC_INF("sidecar ngram verification fixed at MTP width %d; explicit speculative.n_max remains authoritative\n",
                params.draft.n_max);
    }

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple,
                    /* .cap_cfg = */ common_speculative_sidecar_cap_config_for(
                        params, config_simple.size_mgram)
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq,
                            common_speculative_sidecar_cap_config_for(
                                params, config.params.ngram_map_k.size_m)));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq,
                            common_speculative_sidecar_cap_config_for(
                                params, config.params.ngram_map_k4v.size_m)));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(
                            config.params, n_seq,
                            common_speculative_sidecar_cap_config_for(
                                params, config.params.ngram_mod.n_max)));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic,
                        common_speculative_sidecar_cap_config_for(params, 8));
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        impl->prepare_process(spec->dparams);
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// Speculative checkpoints can contain state for more than one implementation
// (for example, an MTP sidecar plus an n-gram fallback). Keep the envelope
// compact and implementation-keyed so a restore never feeds one format to a
// different implementation. Sidecar payloads are only a fixed-size cursor.
static constexpr uint32_t COMMON_SPECULATIVE_STATE_MAGIC   = UINT32_C(0x53504353); // "SPCS"
static constexpr uint32_t COMMON_SPECULATIVE_STATE_VERSION = 1;

static void common_speculative_state_put_u32(std::vector<uint8_t> & data, uint32_t value) {
    data.push_back((uint8_t) (value & 0xffU));
    data.push_back((uint8_t) ((value >> 8) & 0xffU));
    data.push_back((uint8_t) ((value >> 16) & 0xffU));
    data.push_back((uint8_t) ((value >> 24) & 0xffU));
}

static bool common_speculative_state_get_u32(const std::vector<uint8_t> & data,
        size_t & offset, uint32_t & value) {
    if (offset > data.size() || data.size() - offset < sizeof(uint32_t)) {
        return false;
    }
    value = (uint32_t) data[offset] |
            ((uint32_t) data[offset + 1] << 8) |
            ((uint32_t) data[offset + 2] << 16) |
            ((uint32_t) data[offset + 3] << 24);
    offset += sizeof(uint32_t);
    return true;
}

bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    data.clear();
    if (spec == nullptr) {
        return false;
    }

    struct state_entry {
        uint32_t type;
        std::vector<uint8_t> data;
    };
    std::vector<state_entry> entries;

    for (auto & impl : spec->impls) {
        std::vector<uint8_t> impl_data;
        if (!impl->get_state(seq_id, impl_data)) {
            continue;
        }
        if (impl_data.size() > UINT32_MAX) {
            SPC_ERR("speculative state for %s is too large to checkpoint\n",
                    common_speculative_type_to_str(impl->type).c_str());
            data.clear();
            return false;
        }
        entries.push_back({ (uint32_t) impl->type, std::move(impl_data) });
    }

    if (entries.empty() || entries.size() > UINT32_MAX) {
        return false;
    }

    size_t total = 3 * sizeof(uint32_t);
    for (const auto & entry : entries) {
        if (entry.data.size() > UINT32_MAX || total > SIZE_MAX - 2 * sizeof(uint32_t) - entry.data.size()) {
            data.clear();
            return false;
        }
        total += 2 * sizeof(uint32_t) + entry.data.size();
    }
    data.reserve(total);
    common_speculative_state_put_u32(data, COMMON_SPECULATIVE_STATE_MAGIC);
    common_speculative_state_put_u32(data, COMMON_SPECULATIVE_STATE_VERSION);
    common_speculative_state_put_u32(data, (uint32_t) entries.size());
    for (const auto & entry : entries) {
        common_speculative_state_put_u32(data, entry.type);
        common_speculative_state_put_u32(data, (uint32_t) entry.data.size());
        data.insert(data.end(), entry.data.begin(), entry.data.end());
    }
    return true;
}

bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return true;
    }

    // Preserve compatibility with the pre-envelope EAGLE3 state that may be
    // supplied by an embedding application. New checkpoints always use the
    // keyed format above; a sidecar rejects legacy bytes rather than guessing.
    size_t offset = 0;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t n_entries = 0;
    if (!common_speculative_state_get_u32(data, offset, magic) ||
            magic != COMMON_SPECULATIVE_STATE_MAGIC) {
        bool result = true;
        for (auto & impl : spec->impls) {
            result = impl->set_state(seq_id, data) && result;
        }
        return result;
    }

    if (!common_speculative_state_get_u32(data, offset, version) ||
            !common_speculative_state_get_u32(data, offset, n_entries) ||
            version != COMMON_SPECULATIVE_STATE_VERSION ||
            offset > data.size() ||
            n_entries > (data.size() - offset) / (2 * sizeof(uint32_t))) {
        for (auto & impl : spec->impls) {
            if (impl->state_required(seq_id)) {
                impl->set_state(seq_id, {});
            }
        }
        return false;
    }

    struct state_entry_view {
        uint32_t type;
        const uint8_t * data;
        size_t size;
    };
    std::vector<state_entry_view> entries;
    entries.reserve(n_entries);
    for (uint32_t i = 0; i < n_entries; ++i) {
        uint32_t type = 0;
        uint32_t size = 0;
        if (!common_speculative_state_get_u32(data, offset, type) ||
                !common_speculative_state_get_u32(data, offset, size) ||
                offset > data.size() || data.size() - offset < size) {
            for (auto & impl : spec->impls) {
                if (impl->state_required(seq_id)) {
                    impl->set_state(seq_id, {});
                }
            }
            return false;
        }
        entries.push_back({ type, data.data() + offset, size });
        offset += size;
    }
    if (offset != data.size()) {
        for (auto & impl : spec->impls) {
            if (impl->state_required(seq_id)) {
                impl->set_state(seq_id, {});
            }
        }
        return false;
    }

    bool result = true;
    for (auto & impl : spec->impls) {
        const auto it = std::find_if(entries.begin(), entries.end(), [&](const state_entry_view & entry) {
            return entry.type == (uint32_t) impl->type;
        });
        if (it == entries.end()) {
            if (impl->state_required(seq_id)) {
                result = impl->set_state(seq_id, {}) && result;
            }
            continue;
        }

        const std::vector<uint8_t> impl_data(it->data, it->data + it->size);
        result = impl->set_state(seq_id, impl_data) && result;
    }
    return result;
}

void common_speculative_reset_state(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->reset_state(seq_id);
    }
}

void common_speculative_release_state(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->release_state(seq_id);
    }
}

bool common_speculative_prepare_prompt_state(
        common_speculative * spec, llama_seq_id seq_id, llama_pos pos_next, bool can_reuse_resident) {
    if (spec == nullptr) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->prepare_prompt_state(seq_id, pos_next, can_reuse_resident) && result;
    }
    return result;
}

bool common_speculative_truncate_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos_max) {
    if (spec == nullptr) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->truncate_state(seq_id, pos_max) && result;
    }
    return result;
}

bool common_speculative_commit_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos_max) {
    if (spec == nullptr) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->commit_state(seq_id, pos_max) && result;
    }
    return result;
}

bool common_speculative_rebase_state(common_speculative * spec, llama_seq_id seq_id,
        llama_pos pos_min, llama_pos pos_max, llama_pos delta) {
    if (spec == nullptr) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->rebase_state(seq_id, pos_min, pos_max, delta) && result;
    }
    return result;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
