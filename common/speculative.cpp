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
#include "speculative-adaptive.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-mtp-adaptive", COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE},
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

static bool common_speculative_mtp_chain_enabled(const common_params_speculative_draft & params) {
    const char * env = getenv("LLAMA_SPEC_CHAIN");
    return params.chain || (env != nullptr && std::strcmp(env, "0") != 0);
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

bool common_speculative_are_compatible(
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
// A draft context that cannot hold what the target hands it fails here, once, instead of per row:
// process() mirrors whole target batches, so the draft batch must fit the target's batch and the
// speculative sequences. With need_vocab, a draft that decodes target token ids in process() must
// also cover the target vocab: EAGLE3 (and MTP, which runs on the target's own model). The EAGLE3
// and DFlash converters take the tokenizer from the target model, but pad it to the draft config's
// vocab_size, so a smaller draft vocab means a misconverted or differently padded draft. DFlash
// needs no vocab rule, its process() injects features only and a rejected seed in draft() just
// skips the round; draft-simple handles a token outside its vocab per sequence.
static void spec_check_draft_ctx(llama_context * ctx_tgt, llama_context * ctx_dft, uint32_t n_seq, const char * impl, bool need_vocab = false) {
    if (!ctx_dft || ctx_dft == ctx_tgt) {
        return;
    }
    if (need_vocab && ctx_tgt) {
        const int32_t n_vocab_tgt = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_tgt)));
        const int32_t n_vocab_dft = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
        if (n_vocab_dft < n_vocab_tgt) {
            throw std::runtime_error(std::string(impl) + ": the draft vocab (" + std::to_string(n_vocab_dft) +
                    " tokens) is smaller than the target's (" + std::to_string(n_vocab_tgt) + "); target tokens past it cannot be mirrored");
        }
    }
    if (ctx_tgt && llama_n_batch(ctx_dft) < llama_n_batch(ctx_tgt)) {
        throw std::runtime_error(std::string(impl) + ": the draft context batch size (" + std::to_string(llama_n_batch(ctx_dft)) +
                ") is smaller than the target's (" + std::to_string(llama_n_batch(ctx_tgt)) + ")");
    }
    if (llama_n_seq_max(ctx_dft) < n_seq) {
        throw std::runtime_error(std::string(impl) + ": the draft context holds " + std::to_string(llama_n_seq_max(ctx_dft)) +
                " sequences, the speculative setup needs " + std::to_string(n_seq));
    }
}

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

    // add a row with an optional paired embedding to a draft batch; false when the draft cannot
    // take it, and the batch is left as it was (a rejected embedding rolls the token row back)
    // idx_out is written only on success
    static bool draft_add(common_batch & b, llama_token id, llama_pos pos, llama_seq_id seq_id, bool output, llama_embd embd, int32_t * idx_out = nullptr) {
        const int32_t idx = b.add(id, pos, seq_id, output);
        if (idx < 0) {
            return false;
        }
        if (embd.data && !b.set_embd(idx, embd)) {
            b.remove_last();
            return false;
        }
        if (idx_out) {
            *idx_out = idx;
        }
        return true;
    }

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const common_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    common_batch batch;

    // zero row at the draft input width, stands in for target embeddings the draft cannot read
    std::vector<float> zeros;

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
        spec_check_draft_ctx(this->params.ctx_tgt, ctx_dft, n_seq, "draft-simple");

        zeros.assign(llama_model_n_embd_inp(llama_get_model(ctx_dft)), 0.0f);

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = common_batch(ctx_dft);

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
            params.top_k = LLAMA_DRAFT_TOP_K;
            params.samplers.assign(1, COMMON_SAMPLER_TYPE_TOP_K);

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

    // log a token the draft could not take once per occurrence: the verify batch that follows a
    // rejected sampled token carries the same token again
    std::vector<llama_token> last_rejected;

    void report_rejected(llama_seq_id seq_id, llama_token id) {
        if (last_rejected.empty()) {
            last_rejected.assign(n_seq, LLAMA_TOKEN_NULL);
        }
        if (last_rejected[seq_id] == id) {
            SPC_DBG("seq %d: token %d rejected by the draft again\n", seq_id, id);
            return;
        }
        last_rejected[seq_id] = id;
        SPC_WRN("seq %d: the draft cannot take token %d (outside the draft vocab), the sequence is not mirrored "
                "while its draft memory lags the target\n", seq_id, id);
    }

    // Decide whether rows of seq_id starting at position pos continue its draft memory, the same way
    // the draft's batch allocator (llama_batch_allocr::init) will judge them: an M-RoPE draft accepts
    // a forward jump (image rows advance positions by the grid size), and for embedding rows also
    // pos == pos_max (an image split across sub-batches repeats its temporal position); any other
    // draft needs pos == pos_max + 1. Rows behind that are stale draft rows, a bug upstream of here:
    // trim them and say so. Rows ahead of it on a non-M-RoPE draft mean the draft lags (a row it
    // could not take), so skip the sequence. An M-RoPE draft cannot tell a lag from an image jump
    // and keeps mirroring across the gap.
    bool continues_draft(llama_context * ctx_dft, llama_seq_id seq_id, llama_pos pos, bool embd_row) {
        auto * mem = llama_get_memory(ctx_dft);
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);
        const bool mrope = batch.n_pos > 1;
        if (mrope && embd_row && pos == pos_max) {
            return true;
        }
        if (pos <= pos_max) {
            SPC_WRN("seq %d: stale draft memory (ends at %d, target rows start at %d), trimming\n", seq_id, pos_max, pos);
            if (!llama_memory_seq_rm(mem, seq_id, pos, -1)) {
                SPC_WRN("seq %d: the draft memory cannot be trimmed, the sequence is not mirrored\n", seq_id);
                return false;
            }
            return pos == llama_memory_seq_pos_max(mem, seq_id) + 1 || batch.n_pos > 1;
        }
        if (batch.n_pos > 1 || pos == pos_max + 1) {
            if (!last_rejected.empty()) {
                last_rejected[seq_id] = LLAMA_TOKEN_NULL; // the sequence mirrors cleanly again
            }
            return true;
        }
        SPC_DBG("seq %d: draft memory ends at %d, target rows start at %d, not mirrored\n", seq_id, pos_max, pos);
        return false;
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop: whether a sequence is mirrored follows from its draft memory, see process()
    }

    bool process(const common_batch & batch_in) override {
        auto * ctx_dft = params.ctx_dft;

        // The draft mirrors the target, so a sequence's rows must continue its draft memory (see
        // continues_draft). A sequence whose draft memory lags (an earlier row the draft could not
        // take, e.g. a token outside a smaller draft vocab) is left alone: its rows are not decoded
        // and draft() gives it nothing, until a batch continues its draft memory again, e.g. after
        // the server resets it for a new request. A prompt-cache entry that stores the lagging draft
        // memory keeps that prefix undrafted. No state of its own to save or restore.
        std::vector<int8_t> mirror(n_seq, 0); // 0 not seen yet, 1 mirrored, -1 skipped

        // copy the entries to a batch owned by the draft context, only the last token is output
        batch.clear();
        const int32_t n_tokens = batch_in.size();
        for (int32_t k = 0; k < n_tokens; ++k) {
            const auto & t = batch_in.tokens[k];
            const llama_seq_id s = t.seq_id;
            if (s < 0 || s >= (llama_seq_id) n_seq) {
                continue;
            }
            if (mirror[s] == 0) {
                mirror[s] = continues_draft(ctx_dft, s, t.pos[0], /*embd_row =*/ t.id == LLAMA_TOKEN_NULL) ? 1 : -1;
            }
            if (mirror[s] < 0) {
                continue;
            }
            const bool output = k == n_tokens - 1;
            bool ok;
            if (t.id != LLAMA_TOKEN_NULL) {
                // a paired target embedding of a different width is dropped, the draft decodes the token alone
                const bool same_width = t.embd.data == nullptr || t.embd.n_rows * t.embd.n_embd == zeros.size();
                ok = draft_add(batch, t.id, t.pos[0], s, output, same_width ? t.embd : llama_embd{ nullptr, 0, 0 });
            } else {
                // a draft with a different width (e.g. a smaller model) gets zeros instead, keeping its positions contiguous
                const bool same_width = t.embd.n_rows * t.embd.n_embd == zeros.size();
                const llama_embd embd = same_width ? t.embd : llama_embd{ zeros.data(), 1, zeros.size() };
                ok = batch.add_embd(embd, t.pos.data(), s, output) >= 0;
            }
            if (!ok) {
                // the rows before this one are decoded; the sequence's draft memory then lags the
                // target and the check above keeps it out until the next request resets it
                report_rejected(s, t.id);
                mirror[s] = -1;
            }
        }

        if (batch.size() == 0) {
            return true;
        }

        const int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            // the seed must continue the draft memory, otherwise this sequence is not mirrored
            if (!continues_draft(ctx_dft, seq_id, dp.pos0, /*embd_row =*/ false)) {
                continue;
            }

            if (batch.add(dp.id_last, dp.pos0, seq_id, true) < 0) {
                report_rejected(seq_id, dp.id_last);
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());
        }

        if (batch.size() == 0) {
            return;
        }

        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            SPC_ERR("llama_process returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            batch.clear();

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

                if (batch.add(id, dp.pos0 + i + 1, seq_id, true) < 0) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
            }

            if (batch.size() == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
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
    common_batch batch;     // decoder input, (token, g_embd) pairs
    common_batch batch_enc; // encoder input, built from the extracted target features

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
        spec_check_draft_ctx(ctx_tgt, ctx_dft, n_seq, "draft-eagle3", /*need_vocab =*/ true);

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

        batch     = common_batch(ctx_dft);
        batch_enc = common_batch(ctx_dft);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = LLAMA_DRAFT_TOP_K;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(LLAMA_DRAFT_TOP_K));

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

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        if (!batch_in.has_token() || batch_in.has_embd()) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
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

        // llama_process() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            // the per-token encoder does not use positions, generate placeholder ones from the memory state
            batch_enc.clear();
            llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) + 1;
            for (int32_t j = 0; j < n_chunk; ++j) {
                // add_embd() reads n_pos entries (4 on an M-RoPE draft), so pass a full position row
                const llama_pos pos_arr[GGML_MROPE_SECTIONS] = { pos, pos, pos, 0 };
                if (batch_enc.add_embd({ features_buf.data() + (size_t) (i + j) * n_embd_enc, 1, (size_t) n_embd_enc }, pos_arr, 0, true) < 0) {
                    SPC_ERR("encoder batch rejected feature row %d of %d\n", (int) (i + j), (int) n_tokens);
                    return false;
                }
                pos++;
            }

            const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_ENCODE, batch_enc.get());
            if (rc != 0) {
                SPC_ERR("llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
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
        batch.clear();

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
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.tokens[beg].pos[0]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    if (!draft_add(batch, batch_in.tokens[beg].id, pending_pos, seq_id, /*output=*/ false, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec })) {
                        SPC_ERR("seq %d: the draft batch rejected the deferred boundary row\n", seq_id);
                        return false;
                    }
                }
            }

            bool ok = true;
            for (int32_t k = beg; k < end && ok; ++k) {
                ok = draft_add(batch, batch_in.tokens[k + 1].id, batch_in.tokens[k].pos[0], seq_id, /*output=*/ false, { g_embd + (size_t) k * n_embd_dec, 1, (size_t) n_embd_dec });
            }
            if (!ok) {
                SPC_ERR("seq %d: the draft batch rejected a target row\n", seq_id);
                return false;
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.tokens[beg].pos[0];
            pending_pos_last[seq_id] = batch_in.tokens[end].pos[0];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.size() > 0) {
            const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) {
                SPC_ERR("llama_process(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.size(), (int) batch_in.tokens[0].pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

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

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            if (!draft_add(batch, dp.id_last, pending_pos_last[seq_id], seq_id, true, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec })) {
                SPC_ERR("seq %d: the draft batch rejected the seed row, no draft this round\n", seq_id);
                return;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());
        }

        if (batch.size() == 0) {
            return;
        }

        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            SPC_ERR("llama_process returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            batch.clear();

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

                if (!draft_add(batch, id, pending_pos_last[seq_id] + (i + 1), seq_id, true, { prenorm, 1, (size_t) n_embd_dec })) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
            }

            if (batch.size() == 0) {
                break;
            }

            ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
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

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
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

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    common_batch batch;        // noise tokens
    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

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

    // Initialize per-sequence DFlash/DSpark drafting and enable the required target
    // feature extraction. Clamp draft lengths to the trained block's capacity and
    // configure draft embeddings, attention, and optional backend sampling.
    // Throws std::runtime_error if DSpark confidence filtering is requested without
    // a confidence head. Target/draft contexts and target layer IDs are required.
    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");
        spec_check_draft_ctx(ctx_tgt, ctx_dft, n_seq, "draft-dflash");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.decoder_arch", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "laguna") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

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
        // draft() decodes every drafting sequence's noise block (n_max rows, plus the anchor unless
        // anchor-first DSpark) in one batch, so n_seq blocks must fit the draft batch
        {
            const int32_t n_extra  = is_dspark && sample_from_anchor ? 0 : 1;
            const int32_t n_fit    = (int32_t) llama_n_batch(ctx_dft) / (int32_t) std::max<uint32_t>(1, n_seq) - n_extra;
            if (n_fit < 1) {
                throw std::runtime_error("draft-dflash: the draft batch (" + std::to_string(llama_n_batch(ctx_dft)) +
                        ") cannot hold one noise block per sequence (" + std::to_string(n_seq) + " sequences)");
            }
            if (this->params.n_max > n_fit) {
                LOG_WRN("%s: %u sequences of n_max=%d do not fit the draft batch %u -- clamping n_max to %d\n",
                        __func__, n_seq, this->params.n_max, llama_n_batch(ctx_dft), n_fit);
                this->params.n_max = n_fit;
                this->params.n_min = std::min(this->params.n_min, n_fit);
            }
        }
        this->n_max = this->params.n_max;

        batch        = common_batch(ctx_dft);
        batch_inject = common_batch(ctx_dft);

        // embd batches on an M-RoPE draft carry 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = LLAMA_DRAFT_TOP_K;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(LLAMA_DRAFT_TOP_K));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        // Enable extraction of target model layer outputs for DFlash encoder; an id
        // equal to the target's layer count means the pre-final-norm hidden state,
        // which is captured through the unmasked nextn path instead
        n_layer_tgt = llama_model_n_layer(model_tgt);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            }
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
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
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    // Encode features from the just-evaluated target batch and inject draft KV at
    // the target positions. Each token must have one sequence ID, with each sequence's
    // tokens contiguous in the batch. Empty or embedding batches are skipped successfully.
    // Replace NaN features with zero and infinities with signed 65504 before encoding.
    // Return false on a nonzero encoder/decoder status; earlier chunks may already be cached.
    // Missing extracted features or encoder output abort instead of returning false.
    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings (image chunks).
        // Image chunks are not mirrored into the draft: their target-layer features are
        // vision states the draft was never trained on, and their M-RoPE positions collapse
        // onto one temporal position, so injecting them poisons every draft after the
        // image (acceptance fell from ~0.5 to ~0.03 for the rest of the conversation).
        // Skipping them leaves a gap in the draft's cache between the text before and after
        // the image; the draft keeps the target's positions, and the text tokens after the
        // image carry the image's influence in their injected features. The gap is fine for
        // the default sliding-window draft cache: every batch it does see has consecutive
        // positions, which is all find_slot requires (the crash was the image batch itself,
        // whose tokens all carry one temporal position). No server-side change is involved.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.has_token();
        const bool has_embeddings = batch_in.has_embd();
        if (!has_tokens || has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
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

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

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

                // clamp overflow-prone feature values before fusing. backends that
                // stage these f32 activations as f16 for the matmul (e.g. a
                // simdgroup/subgroup multiply) hit Laguna's massive-activation rows
                // (attention-sink tokens, |x| ~ 1e6 in the pre-final-norm residual):
                // finite in f32, but overflowing f16's +-65504 range -> inf/nan.
                // Clamp those finite-but-out-of-range values too, not just actual
                // NaN/Inf, or one poisoned row still NaNs the whole drafter KV cache.
                {
                    size_t n_bad = 0;
                    for (auto & v : features_buf) {
                        if (!std::isfinite(v) || v > 65504.0f || v < -65504.0f) {
                            v = v != v ? 0.0f : (v > 0.0f ? 65504.0f : -65504.0f);
                            n_bad++;
                        }
                    }
                    if (n_bad > 0) {
                        static bool warned = false;
                        if (!warned) {
                            LOG_WRN("%s: sanitized %zu non-finite/f16-overflow target feature values (massive activations); "
                                    "draft quality may degrade slightly on affected rows\n", __func__, n_bad);
                            warned = true;
                        }
                    }
                }

                // inject the DFlash decoder K/V cache at the tokens' target positions.
                // The injection graph (see the "KV cache injection" branches in
                // src/models/dflash.cpp) applies build_dflash_features() itself, so
                // the batch_inject rows must carry the same raw, n_embd_enc-sized target
                // features as features_buf - not a separately fused n_embd_dec-sized
                // encoder output, which would be fused a second time and also
                // overrun/underrun the n_embd_enc-sized row allocation.
                batch_inject.clear();
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.tokens[i_batch_beg[seq_id] + offset + i].pos[0];
                    const llama_pos pos_arr[4] = { p, p, p, 0 };
                    if (batch_inject.add_embd({ features_buf.data() + (size_t) i * n_embd_enc, 1, (size_t) n_embd_enc }, pos_arr, seq_id, false) < 0) {
                        LOG_ERR("%s: injection batch rejected feature row %d of %d\n", __func__, (int) i, (int) n_chunk);
                        return false;
                    }
                }
                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
                if (rc != 0) {
                    LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

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

            // dp.pos0 is the slot's token count. For an M-RoPE target that is no longer the
            // next position once an image has been in the prompt (image tokens advance the
            // position by the grid size, not by their count), and a noise block placed at the
            // token count sits hundreds of positions past the draft's own cache. Take the next
            // position from the target's memory instead; without images the two are equal.
            const llama_pos pos_max_tgt = llama_memory_seq_pos_max(llama_get_memory(params.ctx_tgt), seq_id);
            const int32_t n = pos_max_tgt >= 0 ? (int32_t) pos_max_tgt + 1 : (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            const int32_t beg = batch.size();
            bool ok = true;
            for (int32_t i = 0; i < n_block_tokens && ok; ++i) {
                ok = batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, seq_id, !is_dflash2) >= 0;
            }
            if (!ok) {
                SPC_ERR("seq %d: the draft batch rejected a noise-block row, no draft this round\n", seq_id);
                return;
            }
            i_block_beg[seq_id] = beg;
            n_block    [seq_id] = n_block_tokens;
        }

        if (batch.size() == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
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

            if (is_dflash2) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

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

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
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

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    common_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;
    int32_t batch_capacity = 0;
    int32_t ubatch_capacity = 0;

    std::vector<float> zeros; // one n_embd row of zeros for the chained draft rows

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    same_position_draft = false; // gemma4-assistant only: every draft row in a round shares pos0
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    // catch-up rows deferred from process() and prepended to the first draft
    // decode, which merges the two evals. Only for the single-head own-memory
    // path; flushed standalone when drafting does not follow.
    static constexpr int32_t defer_max = 64;
    bool defer_enabled = false;
    bool chain_graph   = false;
    struct {
        std::vector<llama_token>  tok;
        std::vector<llama_pos>    pos;
        std::vector<llama_seq_id> seq;
        std::vector<float>        embd;
    } defer;

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // Adaptive draft depth (draft-mtp-adaptive), see common_speculative_adaptive
    bool adaptive = false;
    std::vector<int> n_cap;   // [n_seq] effective draft cap for the current draft() call
    std::vector<int> n_last;  // [n_seq] drafts attempted in the most recent draft() call
    std::vector<common_speculative_adaptive> adaptive_ctrl; // [n_seq] per-seq adaptive depth controller

    int32_t defer_capacity() const {
        const int32_t draft_rows = std::max(this->params.n_max, (int32_t) n_seq);
        const int32_t decode_capacity = std::min(batch_capacity, ubatch_capacity);
        return std::min(defer_max, std::max(0, decode_capacity - draft_rows));
    }

    // Initialize per-sequence MTP drafting and enable target/draft hidden-state output.
    // Shared KV requires a matching target context and an architecture that shares KV.
    // Adaptive mode starts at n_min_adaptive and aborts unless it is in [1, n_max],
    // after any limit imposed by the number of MTP heads. Contexts must be non-null
    // and their output embedding widths must match.
    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq, bool adaptive = false)
        : common_speculative_impl(adaptive ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE : COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");
        spec_check_draft_ctx(ctx_tgt, ctx_dft, n_seq, "draft-mtp", /*need_vocab =*/ true);

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        this->adaptive = adaptive;
        // n_cap/n_last are written by the shared draft loop in both modes
        n_cap.assign(n_seq, 0);
        n_last.assign(n_seq, 0);

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch_capacity = (int32_t) llama_n_batch(ctx_dft);
        ubatch_capacity = (int32_t) llama_n_ubatch(ctx_dft);
        batch = common_batch(ctx_dft);
        zeros.assign((size_t) n_embd, 0.0f);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = LLAMA_DRAFT_TOP_K;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        const bool chain_enabled = common_speculative_mtp_chain_enabled(this->params);

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt && llama_model_shares_target_kv(llama_get_model(ctx_dft));
        same_position_draft = is_mem_shared && llama_model_uses_shared_position_draft(llama_get_model(ctx_dft));
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
        chain_graph   = !is_mem_shared && !chain_heads && chain_enabled && llama_model_supports_mtp_chain(llama_get_model(ctx_dft));

        if (chain_enabled && !is_mem_shared && !chain_heads && !chain_graph) {
            SPC_WRN("%s", "chained MTP is not supported for this model; using sequential MTP\n");
        }

        // offload draft sampling to the backend (chained drafting outputs several
        // rows per sequence, which backend sampling does not support)
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !chain_graph) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(LLAMA_DRAFT_TOP_K));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // draft all n_max tokens with one chained decode (in-graph argmax
        // feeds each next step); replaces n_max sequential draft decodes
        if (chain_graph) {
            // the chain decode absorbs the deferred catch-up rows, one eval per round
            defer_enabled = true;
        }

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        if (adaptive) {
            if (this->params.n_min_adaptive < 1 || this->params.n_min_adaptive > this->params.n_max) {
                GGML_ABORT("%s: invalid adaptive draft range: n_min_adaptive=%d, n_max=%d (n_min_adaptive must be in [1, n_max])",
                        __func__, this->params.n_min_adaptive, this->params.n_max);
            }

            adaptive_ctrl.assign(n_seq, common_speculative_adaptive());
            for (uint32_t s = 0; s < n_seq; ++s) {
                // start at the floor max(1, n_min_adaptive), bounded by n_max;
                // the controller climbs from there once acceptance feedback arrives
                adaptive_ctrl[s].reset(this->params.n_max, this->params.n_min_adaptive);
            }
            SPC_TRC("%s", "adaptive draft depth enabled (draft-mtp-adaptive)\n");
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
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
    }

    // decode deferred catch-up rows standalone (no logits); used when no draft decode
    // follows to absorb them
    bool flush_deferred() {
        if (defer.tok.empty()) {
            return true;
        }

        // defer only ever grows on the !is_mem_shared path (deferral requires
        // chain_graph, which requires !is_mem_shared). Keep that invariant local:
        // the trim below on shared memory would delete target KV.
        GGML_ASSERT(!is_mem_shared);

        auto * ctx_dft = this->params.ctx_dft;

        // The draft KV can retain cells from a position rewind (context shift,
        // rollback, or a new prompt) for a sequence that had no rows in the last
        // process() batch, so its stale cells were not trimmed there. Drop them
        // here, keyed on the flush batch's own per-seq positions, before decode,
        // or the position consistency check (X < Y) fails.
        auto * mem_dft = llama_get_memory(ctx_dft);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            llama_pos pos_min = -1;
            for (size_t k = 0; k < defer.pos.size(); ++k) {
                if (defer.seq[k] == seq_id) {
                    pos_min = pos_min < 0 ? defer.pos[k] : std::min(pos_min, defer.pos[k]);
                }
            }
            if (pos_min < 0) {
                continue;
            }
            // In the normal flow the flush rows start right past the draft KV max,
            // so an overlap means a stale-state producer this trim is papering over.
            // Log it: a silently firing trim leaves no forensic signal.
            const llama_pos pos_max_pre = llama_memory_seq_pos_max(mem_dft, seq_id);
            if (pos_max_pre >= pos_min) {
                SPC_WRN("stale draft KV for seq %d at deferred flush: pos_max %d >= flush start %d, trimming\n",
                        (int) seq_id, (int) pos_max_pre, (int) pos_min);
            }
            if (!llama_memory_seq_rm(mem_dft, seq_id, pos_min, -1)) {
                SPC_ERR("failed to trim draft memory for sequence %d at deferred flush\n", (int) seq_id);
                return false;
            }
        }

        batch.clear();
        bool ok = true;
        for (size_t k = 0; k < defer.tok.size() && ok; ++k) {
            ok = draft_add(batch, defer.tok[k], defer.pos[k], defer.seq[k], false, { defer.embd.data() + k * (size_t) n_embd, 1, (size_t) n_embd });
        }
        if (!ok) {
            // the rows stay deferred; nothing was decoded
            batch.clear();
            SPC_ERR("%s", "deferred flush: the draft batch rejected a catch-up row\n");
            return false;
        }
        defer_clear();

        const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            SPC_ERR("llama_process(ctx_dft) deferred flush failed rc=%d\n", (int) rc);
            return false;
        }
        return true;
    }

    void defer_clear() {
        defer.tok.clear();
        defer.pos.clear();
        defer.seq.clear();
        defer.embd.clear();
    }

    // drop deferred rows of seq_id with pos >= pos_from; returns true if any dropped
    bool drop_deferred_from(llama_seq_id seq_id, llama_pos pos_from) {
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        size_t w = 0;
        for (size_t k = 0; k < defer.tok.size(); ++k) {
            if (defer.seq[k] == seq_id && defer.pos[k] >= pos_from) {
                continue;
            }
            if (w != k) {
                defer.tok[w] = defer.tok[k];
                defer.pos[w] = defer.pos[k];
                defer.seq[w] = defer.seq[k];
                std::memmove(defer.embd.data() + w * (size_t) n_embd,
                             defer.embd.data() + k * (size_t) n_embd, row_bytes);
            }
            w++;
        }

        const bool dropped = w < defer.tok.size();
        defer.tok.resize(w);
        defer.pos.resize(w);
        defer.seq.resize(w);
        defer.embd.resize(w * (size_t) n_embd);

        return dropped;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // note: the server calls begin() after the prefill decode, so stale defer
        // rows are already handled by the position-rewind trim in process(). Rows
        // that remain here belong to this prompt and feed the next draft decode.
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max_mem = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        llama_pos pos_max_defer = -1;
        if (defer_enabled) {
            for (size_t k = 0; k < defer.pos.size(); ++k) {
                if (defer.seq[k] == seq_id) {
                    pos_max_defer = std::max(pos_max_defer, defer.pos[k]);
                }
            }
        }
        const llama_pos pos_max = std::max(pos_max_mem, pos_max_defer);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / output flag on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (!batch_in.has_token() || batch_in.has_embd()) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (batch_in.tokens[k].seq_id == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // The draft memory can retain speculative rows after a rewind (context
        // shift, rollback, or a new prompt). Trim them before the next decode.
        if (!is_mem_shared) {
            auto * mem_dft = llama_get_memory(ctx_dft);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                const llama_pos pos_min_in = batch_in.tokens[i_batch_beg[seq_id]].pos[0];
                // The server trims draft KV to the target's pre-draft max after every
                // draft, so overlap here means an unidentified stale-state producer.
                // Log it: a silently firing trim leaves no forensic signal.
                const llama_pos pos_max_pre = llama_memory_seq_pos_max(mem_dft, seq_id);
                if (pos_max_pre >= pos_min_in) {
                    SPC_WRN("stale draft KV for seq %d at process: pos_max %d >= batch start %d, trimming\n",
                            (int) seq_id, (int) pos_max_pre, (int) pos_min_in);
                }
                if (!llama_memory_seq_rm(mem_dft, seq_id, pos_min_in, -1)) {
                    SPC_ERR("failed to trim draft memory for sequence %d from position %d\n",
                            seq_id, (int) pos_min_in);
                    return false;
                }
            }
        }

        // Deferred rows at or past the incoming batch positions are stale: either a
        // new prompt rewound the sequence, or they hold candidates a later verify
        // rejected. Drop them before any decode.
        if (defer_enabled && !defer.tok.empty()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                drop_deferred_from(seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0]);
            }
        }

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        const int32_t defer_cap = defer_capacity();
        if (!is_mem_shared && defer_enabled && n_tokens <= defer_cap) {
            // defer the catch-up rows; the first draft decode absorbs them, which saves
            // one eval per round. Flush first if rows from an undrafted round remain.
            if (!defer.tok.empty() && (int32_t) defer.tok.size() + n_tokens > defer_cap) {
                if (!flush_deferred()) {
                    return false;
                }
            }

            const size_t k0 = defer.tok.size();
            defer.tok.resize(k0 + n_tokens);
            defer.pos.resize(k0 + n_tokens);
            defer.seq.resize(k0 + n_tokens);
            defer.embd.resize((k0 + n_tokens) * (size_t) n_embd);

            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            for (int k = 0; k < n_tokens; ++k) {
                defer.tok[k0 + k] = batch_in.tokens[k].id;
                defer.pos[k0 + k] = batch_in.tokens[k].pos[0];
                defer.seq[k0 + k] = batch_in.tokens[k].seq_id;
                // shift the tgt embeddings to the right by one position (see the non-deferred path)
                if (k > 0) {
                    std::memcpy(defer.embd.data() + (k0 + k) * (size_t) n_embd, h_tgt + (size_t) (k - 1) * n_embd, row_bytes);
                }
            }
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                std::memcpy(defer.embd.data() + (k0 + i_batch_beg[seq_id]) * (size_t) n_embd, pending_h[seq_id].data(), row_bytes);
            }
        } else if (!is_mem_shared) {
            if (!flush_deferred()) {
                return false;
            }

            batch.clear();

            // pair each token with the tgt embedding shifted right by one position, and
            // the first token of each sequence with the pending embedding from a previous run
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);

            for (int k = 0; k < n_tokens; ++k) {
                const llama_seq_id seq_id = batch_in.tokens[k].seq_id;

                const float * h_row = k == i_batch_beg[seq_id]
                    ? pending_h[seq_id].data()
                    : h_tgt + (size_t) (k - 1) * n_embd;

                if (!draft_add(batch, batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false, { h_row, 1, (size_t) n_embd })) {
                    SPC_ERR("seq %d: the draft batch rejected a target row\n", seq_id);
                    return false;
                }
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (rc != 0) {
                    SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.tokens[0].pos[0]);
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

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
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
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        bool any_drafting = false;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            any_drafting = any_drafting || dparams[seq_id].drafting;
        }

        // chained drafting: one decode drafts n_max tokens for a single sequence.
        // this block must run before the generic defer merge below - it consumes
        // the deferred rows itself and prepends them to the chain batch
        if (chain_graph && any_drafting) {
            llama_seq_id seq_one = -1;
            int n_seq_drafting = 0;
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (dparams[s].drafting) {
                    n_seq_drafting++;
                    seq_one = s;
                }
            }

            if (n_seq_drafting == 1) {
                const auto & dp = dparams[seq_one];
                n_cap[seq_one] = adaptive ? adaptive_ctrl[seq_one].n_cur : params.n_max;
                if (dp.n_max > 0 && dp.n_max < n_cap[seq_one]) {
                    n_cap[seq_one] = dp.n_max;
                }
            }

            // A chain must stay in one microbatch: later rows have placeholder inputs.
            // Larger requests use the ordinary sequential path below.
            if (n_seq_drafting == 1 && n_cap[seq_one] <= std::min(batch_capacity, ubatch_capacity)) {
                auto & dp = dparams[seq_one];
                auto * smpl = smpls[seq_one].get();
                common_sampler_reset(smpl);

                const int n_chain = n_cap[seq_one];

                // deferred rows at or past pos0 hold candidates the verify
                // rejected; the committed prefix ends at pos0 - 1. Drop them.
                drop_deferred_from(seq_one, dp.pos0);

                // rows of other sequences cannot join a single-sequence chain
                // batch; decode all remaining rows standalone in that case
                bool defer_other = false;
                for (size_t k = 0; k < defer.tok.size(); ++k) {
                    defer_other = defer_other || defer.seq[k] != seq_one;
                }
                if (defer_other && !flush_deferred()) {
                    return;
                }

                batch.clear();

                // deferred catch-up rows ride along, before the chain rows
                const int n_catchup = (int) defer.tok.size();
                bool ok = true;
                for (int k = 0; k < n_catchup && ok; ++k) {
                    ok = draft_add(batch, defer.tok[k], defer.pos[k], defer.seq[k], false, { defer.embd.data() + (size_t) k * n_embd, 1, (size_t) n_embd });
                }

                for (int j = 0; j < n_chain && ok; ++j) {
                    // row 0 pairs with the pending target state, the chained rows carry zeros
                    ok = draft_add(batch, j == 0 ? dp.id_last : 0, dp.pos0 + j, seq_one, true, { j == 0 ? pending_h[seq_one].data() : zeros.data(), 1, (size_t) n_embd });
                }
                if (!ok) {
                    // the catch-up rows stay deferred for the next flush or draft; nothing was decoded
                    batch.clear();
                    SPC_ERR("%s", "chain decode: the draft batch rejected a row, no draft this round\n");
                    return;
                }
                // The chain batch starts at the deferred catch-up rows, which can sit
                // below the draft KV max from an earlier/rejected draft (position
                // rewind or a repeated decode). Drop the draft cells at/above the
                // batch's own minimum position before decoding so the position
                // consistency check (X < Y) holds; the chain re-writes that region.
                if (batch.size() > 0) {
                    llama_pos pos_min = batch.tokens[0].pos[0];
                    for (int32_t k = 1; k < batch.size(); ++k) {
                        pos_min = std::min(pos_min, batch.tokens[k].pos[0]);
                    }
                    auto * mem_dft = llama_get_memory(ctx_dft);
                    // Normal flow: the chain batch starts right past the draft KV max,
                    // so an overlap means a stale-state producer this trim is papering
                    // over. Log it: a silently firing trim leaves no forensic signal.
                    const llama_pos pos_max_pre = llama_memory_seq_pos_max(mem_dft, seq_one);
                    if (pos_max_pre >= pos_min) {
                        SPC_WRN("stale draft KV for seq %d before chain decode: pos_max %d >= batch start %d, trimming\n",
                                (int) seq_one, (int) pos_max_pre, (int) pos_min);
                    }
                    if (!llama_memory_seq_rm(mem_dft, seq_one, pos_min, -1)) {
                        SPC_ERR("failed to trim draft memory for sequence %d before chain decode\n", (int) seq_one);
                        return;
                    }
                }

                // TODO(mtp-chain): llama_set_mtp_chain() is provided by the llama API
                // backport (llama-ext.h / llama-context.cpp); the chain decode depends
                // on its in-graph chained sampling writing [token, prob] pairs to logits
                llama_set_mtp_chain(ctx_dft, true);
                const int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                llama_set_mtp_chain(ctx_dft, false);

                if (ret != 0) {
                    // nothing was decoded: the catch-up rows stay deferred for the next flush or draft
                    SPC_ERR("llama_process(chain) returned %d\n", ret);
                    return;
                }
                // the draft cache now holds the catch-up rows
                defer_clear();

                // the chain samples greedily in-graph and emits [token id, top prob]
                // pairs as 2-float rows, packed from the start of the logits buffer;
                // no host-side sampling pass runs over the draft logits
                const float * lp = llama_get_logits(ctx_dft);

                auto & result = *dp.result;
                for (int j = 0; j < n_chain; ++j) {
                    const llama_token id = (llama_token) lp[2*j + 0];
                    const float       p  =               lp[2*j + 1];

                    SPC_DBG(" - seq_id %d, chain candidate %3d: %6d (%8.3f) '%s'\n",
                            seq_one, j, id, p,
                            common_token_to_piece(ctx_dft, id).c_str());

                    if (p < params.p_min) {
                        break;
                    }

                    result.push_back(id);

                    if ((int) result.size() >= n_chain) {
                        break;
                    }
                }

                n_last[seq_one] = (int) result.size();

                // the adaptive controller decides its own depth, so the generic n_min
                // draft cutoff does not apply to it
                if (!adaptive && dp.result->size() < (size_t) params.n_min) {
                    dp.result->clear();
                }
                return;
            }
        }

        // deferred catch-up rows ride along with the first draft decode. They carry
        // earlier positions, so they must precede the draft rows: recurrent state
        // advances in batch order. If nothing drafts this round they stay deferred
        // and process() flushes them later.
        //
        // For every seq drafting this round, deferred rows at or past its pos0 are
        // candidates the verify just rejected (the committed prefix ends at
        // pos0 - 1, same as the chain path). Merging them would decode a rejected
        // token and this round's first draft row at the same position in the same
        // batch: duplicate-position draft KV cells, no X < Y error, silent quality
        // loss. Only reachable at n_parallel > 1 (a single drafting seq takes the
        // chain path, which already drops these).
        if (any_drafting && !defer.tok.empty()) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (dparams[seq_id].drafting) {
                    drop_deferred_from(seq_id, dparams[seq_id].pos0);
                }
            }
        }
        // the deferred rows leave `defer` only once the seed rows below fit as well; if anything is
        // rejected they stay deferred for the next flush or draft and nothing is decoded
        if (any_drafting && !defer.tok.empty()) {
            bool ok = true;
            for (size_t k = 0; k < defer.tok.size() && ok; ++k) {
                ok = draft_add(batch, defer.tok[k], defer.pos[k], defer.seq[k], false, { defer.embd.data() + k * (size_t) n_embd, 1, (size_t) n_embd });
            }
            if (!ok) {
                batch.clear();
                SPC_ERR("%s", "the draft batch rejected a deferred catch-up row, no draft this round\n");
                return;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            // effective draft cap for this step: adaptive depth (or the user n_max),
            // then clamped by the per-call context bound from the server
            n_cap[seq_id] = adaptive ? adaptive_ctrl[seq_id].n_cur : params.n_max;
            if (dp.n_max > 0 && dp.n_max < n_cap[seq_id]) {
                n_cap[seq_id] = dp.n_max;
            }

            int32_t idx = -1;
            if (!draft_add(batch, dp.id_last, dp.pos0, seq_id, true, { pending_h[seq_id].data(), 1, (size_t) n_embd }, &idx)) {
                batch.clear();
                SPC_ERR("seq %d: the draft batch rejected the seed row, no draft this round\n", seq_id);
                return;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            i_last[seq_id] = idx;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        // every row fit: the batch now owns the deferred catch-up rows
        defer_clear();

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            batch.clear();

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

                if (n_cap[seq_id] <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                bool ok = true;
                int32_t idx = -1;
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows && ok; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        ok = draft_add(batch, tok, dp.pos0 + t, seq_id, t == n_rows - 1, { chain_h[seq_id].data() + (size_t) t * n_embd, 1, (size_t) n_embd }, &idx);
                    }
                } else if (same_position_draft) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    ok = draft_add(batch, id, dp.pos0, seq_id, true, { h_row, 1, (size_t) n_embd }, &idx);
                } else {
                    // is_mem_shared models with a real trained NextN head (qwen35, qwen4exp, ...)
                    // still draft at incrementing positions, same as the non-shared-memory path
                    ok = draft_add(batch, id, dp.pos0 + i + 1, seq_id, true, { h_row, 1, (size_t) n_embd }, &idx);
                }
                if (!ok) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
                i_last[seq_id] = idx;
            }

            if (batch.size() == 0) {
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

            n_last[seq_id] = (int) dp.result->size();

            // the adaptive controller decides its own depth, so the generic n_min
            // draft cutoff does not apply to it
            if (!adaptive && dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // update the adaptive controller only when this implementation produced the
        // accepted draft; on is_other the stats belong to a different speculator
        if (adaptive && !is_other) {
            const int depth_before = adaptive_ctrl[seq_id].n_cur;
            adaptive_ctrl[seq_id].update(n_last[seq_id], n_accepted, params.n_max, params.n_min_adaptive);
            if (adaptive_ctrl[seq_id].n_cur != depth_before) {
                SPC_DBG("adaptive draft depth seq %d: %d -> %d (n_draft=%d, n_accepted=%d)\n",
                        (int) seq_id, depth_before, adaptive_ctrl[seq_id].n_cur, n_last[seq_id], n_accepted);
            }
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const common_batch & /*batch*/) override {
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
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
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

    bool process(const common_batch & /*batch*/) override {
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

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
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

        // consecutive zero-accept drafts and per-generation disable latch
        int  n_dead = 0;
        bool off    = false;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d, n_dead_off=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min, this->params.n_dead_off);
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
        sinfo.n_low = 0;
        sinfo.n_dead = 0;
        sinfo.off    = false;

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
        if (sinfo.off) {
            return;
        }

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

    bool process(const common_batch & /*batch*/) override {
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

            if (params.n_dead_off > 0) {
                if (n_accepted == 0) {
                    if (++sinfo.n_dead >= params.n_dead_off) {
                        if (verbose) {
                            SPC_TRC("%d dead ngram-mod fires - disabling for seq %d\n", sinfo.n_dead, seq_id);
                        }
                        sinfo.off = true;
                    }
                } else {
                    sinfo.n_dead = 0;
                }
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

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
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
        // noop
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

    bool process(const common_batch & /*batch*/) override {
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
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // the target context, used to convert legacy llama_batch inputs
    llama_context * ctx_tgt = nullptr;

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
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

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
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE: return "draft-mtp-adaptive";
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
        const std::string block_count_key = arch + ".block_count";
        const int64_t block_count_id = gguf_find_key(gguf_ctx.get(), block_count_key.c_str());
        if (block_count_id < 0 || gguf_get_kv_type(gguf_ctx.get(), block_count_id) != GGUF_TYPE_UINT32) {
            return {};
        }

        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), block_count_id);
        if (block_count == 0) {
            return {};
        }

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
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE:
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

llama_state_seq_flags common_speculative_checkpoint_flags(
        common_speculative_checkpoint_place & place,
        llama_context * ctx, llama_seq_id seq_id,
        const std::vector<size_t> & margins, const llama_model * model_margins) {
    const llama_state_seq_flags flags_host   = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const llama_state_seq_flags flags_device = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    // the host size counts the tensor data, the device size leaves it out
    const size_t size_host   = llama_state_seq_get_size_ext(ctx, seq_id, flags_host);
    const size_t size_device = llama_state_seq_get_size_ext(ctx, seq_id, flags_device);
    const size_t size_copy   = size_host > size_device ? size_host - size_device : 0;

    // the state writer frees the copy the context holds before it allocates one of another size,
    // so only the growth has to fit
    const size_t size_new = size_copy > place.size_copy ? size_copy - place.size_copy : 0;

    const double mib = 1024.0 * 1024.0;
    llama_state_seq_flags flags = flags_device;

    const llama_model * model = llama_get_model(ctx);
    for (int i = 0; size_new > 0 && i < llama_model_n_devices(model); i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue; // tensors of such a device live in host memory, as --fit assumes
            }
        }

        // the margins follow the device order of another model, which this one need not share
        size_t margin = margins.empty() ? 0 : *std::max_element(margins.begin(), margins.end());
        for (int j = 0; !margins.empty() && j < llama_model_n_devices(model_margins); j++) {
            if (llama_model_get_device(model_margins, j) == dev) {
                margin = margins[std::min((size_t) j, margins.size() - 1)];
                break;
            }
        }
        if (free < size_new + margin) {
            if (place.flags != flags_host) {
                LOG_INF("%s: seq %d checkpoint stays on the host: %s has %.1f MiB free, the device copy needs %.1f MiB on top of the %.1f MiB margin\n",
                        __func__, seq_id, ggml_backend_dev_name(dev), free / mib, size_new / mib, margin / mib);
            }
            flags = flags_host;
            break;
        }
    }

    if (flags == flags_device) {
        if (place.flags != flags_device) {
            LOG_INF("%s: seq %d checkpoint stays on the device (%.1f MiB)\n", __func__, seq_id, size_copy / mib);
        }
        place.size_copy = size_copy;
    }
    place.flags = flags;
    return flags;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.model_is_spec_draft   = true;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

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
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    // chained MTP drafting outputs logits for every chain step in one decode
    if (common_speculative_mtp_chain_enabled(params_spec)) {
        const int32_t per_seq = std::max(1, params_spec.n_max);
        result.n_outputs_max = std::max(result.n_outputs_max, params.n_parallel * per_seq);
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = params.speculative.has_mtp();
    GGML_ASSERT(has_draft || spec_mtp);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);
    // and takes whole target batches in process(): the target's n_batch can be raised above the
    // configured one (n_rs_seq + 2 for recurrent rollback), the draft context is built with n_rs_seq = 0
    cparams.n_batch = std::max(cparams.n_batch, llama_n_batch(ctx_tgt));

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
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
    } else if (spec_mtp) {
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

        // If --dflash or --eagle3 flags are set, enable the corresponding type
        if (params.draft.dflash) {
            enabled_configs |= (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
        }
        if (params.draft.eagle3) {
            enabled_configs |= (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3);
        }

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 12);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

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
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq, /*adaptive=*/ false));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq, /*adaptive=*/ true));
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
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
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
        /* .ctx_tgt     = */ params.draft.ctx_tgt,
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
    if (spec == nullptr) {
        return true;
    }

    // ngram-only setups have no target context, they do not read the batch anyway
    const common_batch tmp = spec->ctx_tgt ? common_batch_from_llama_batch(spec->ctx_tgt, batch) : common_batch();
    if (spec->ctx_tgt && tmp.size() != batch.n_tokens) {
        SPC_ERR("could not convert the legacy batch (%d of %d rows), see common_batch_from_llama_batch\n", tmp.size(), batch.n_tokens);
        return false;
    }

    return common_speculative_process(spec, tmp);
}

bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
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

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
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
