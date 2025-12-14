#include "speculative.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "common.h"
#include "sampling.h"

#include <cstring>
#include <algorithm>
#include <map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct common_speculative {
    struct llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    struct llama_context * ctx_dft;
    struct common_sampler * smpl;

    llama_batch batch;
    llama_tokens prompt_dft;
    bool vocab_dft_compatible = true; // whether retokenization is needed
    std::map<std::string, std::string> tgt_dft_replacements = {};

    // EAGLE3 specific
    struct llama_context * eagle3_encoder = nullptr;
    struct llama_context * eagle3_decoder = nullptr;
    int32_t eagle3_n_past = 0;  // number of verified positions in decoder KV cache
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    auto * result = new common_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ ctx_dft,
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ false,
    };

    // TODO: optimize or pass from outside?
#if 0
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 40;
        params.top_p = 0.9;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#else
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 10;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#endif

    result->vocab_dft_compatible = common_speculative_are_compatible(ctx_tgt, ctx_dft);
    LOG_DBG("vocab_dft_compatible = %d\n", result->vocab_dft_compatible);

    return result;
}

struct common_speculative * common_speculative_init_eagle3(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_encoder,
        struct llama_context * ctx_decoder) {

    auto * result = new common_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ nullptr,  // Not used for EAGLE3
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(llama_n_batch(ctx_decoder), 0, 1),
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ true,  // EAGLE3 uses same vocab
        /* .tgt_dft_replacements = */ {},
        /* .eagle3_encoder = */ ctx_encoder,
        /* .eagle3_decoder = */ ctx_decoder,
    };

    // Initialize sampler for EAGLE3 decoder
    {
        common_params_sampling params;
        params.no_perf = false;
        params.top_k = 10; // set 1 for greedy sampling (argmax) to match vLLM's default behavior but >1 always gets higher acceptance rate for eagle3
        params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
        result->smpl = common_sampler_init(llama_get_model(ctx_decoder), params);
    }

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    common_sampler_free(spec->smpl);

    // EAGLE3 cleanup
    if (spec->eagle3_encoder) {
        llama_free(spec->eagle3_encoder);
    }
    if (spec->eagle3_decoder) {
        llama_free(spec->eagle3_decoder);
    }

    llama_batch_free(spec->batch);

    delete spec;
}

bool common_speculative_are_compatible(
    const struct llama_context * ctx_tgt,
    const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest) {
    spec->tgt_dft_replacements[source] = dest;
}

static std::string replace_to_dft(
        struct common_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto & pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.first);
        while (pos != std::string::npos) {
            result.replace(pos, pair.first.length(), pair.second);
            pos = result.find(pair.first, pos + pair.second.length());
        }
    }
    return result;
}

static std::string replace_to_tgt(
        struct common_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto& pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.second);
        while (pos != std::string::npos) {
            result.replace(pos, pair.second.length(), pair.first);
            pos = result.find(pair.second, pos + pair.first.length());
        }
    }
    return result;
}

// EAGLE3 Draft Generation with KV Cache Reuse
//
// ============================================================================
// EXAMPLE: Two rounds of speculative decoding
// ============================================================================
//
// ROUND 1 (Initial):
//   Prompt: [t0, t1, t2, t3, t4], target generates t5
//   prompt_tgt = [t0, t1, t2, t3, t4], id_last = t5 (GENERATED)
//   n = 5, n_past = 0, n_new = 5
//
//   Step 1: Encoder
//     features: [f0, f1, f2, f3, f4] → g_embeddings: [g0, g1, g2, g3, g4]
//
//   Step 2: Decoder batch (positions 0-4)
//     tokens:     [t1, t2, t3, t4, t5]  ← prompt[1:] + id_last
//     g_embd:     [g0, g1, g2, g3, g4]
//     positions:  [0,  1,  2,  3,  4 ]
//     → KV cache: [0, 1, 2, 3, 4]
//     → sample d1 from logits[4]
//
//   Step 3: Autoregressive (positions 5, 6, ...)
//     pos 5: token=d1, g_embd=prenorm[4] → KV cache: [0,1,2,3,4,5] → d2
//     pos 6: token=d2, g_embd=prenorm    → KV cache: [0,1,2,3,4,5,6] → d3
//
//   Output: [d1, d2, d3]
//   Update: n_past = 5 (verified positions from batch decode)
//
// ROUND 2 (assuming d1 accepted, d2/d3 rejected):
//   prompt_tgt = [t0, t1, t2, t3, t4, t5, d1], id_last = t6 (new target output)
//   n = 7, n_past = 5, n_new = 2
//
//   Step 1: Clear KV cache [5, inf) - remove draft positions
//     KV cache: [0, 1, 2, 3, 4] (reuse from round 1!)
//
//   Step 2: Encoder (only new tokens)
//     features: [f5, f6] → g_embeddings: [g5, g6]
//
//   Step 3: Decoder batch (only new positions 5-6)
//     tokens:     [d1, t6]        (prompt_tgt[6], id_last)
//     g_embd:     [g5, g6]
//     positions:  [5,  6 ]
//     → KV cache: [0,1,2,3,4] + [5,6] = [0,1,2,3,4,5,6]
//     → sample d1' from logits[1] (last position in batch)
//
//   Step 4: Autoregressive...
//
// ============================================================================
//
// Key insight: Decoder KV cache stores K/V computed from (tok_embd + g_embd).
// For verified positions, both tok_embd and g_embd are fixed (encoder output),
// so KV cache can be reused. Draft positions use prenorm as g_embd, which
// differs from encoder output, so they must be cleared and recomputed.
//
static llama_tokens gen_eagle3_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt,
        llama_token id_last) {

    auto * ctx_tgt     = spec->ctx_tgt;
    auto * ctx_encoder = spec->eagle3_encoder;
    auto * ctx_decoder = spec->eagle3_decoder;
    auto * smpl        = spec->smpl;
    auto & batch       = spec->batch;

    const int n_embd = llama_model_n_embd(llama_get_model(ctx_encoder));
    const int n      = (int)prompt_tgt.size();
    const int n_new  = n - spec->eagle3_n_past;

    GGML_ASSERT(n >= 1 && "prompt_tgt is empty");
    GGML_ASSERT(n_new >= 1 && "must have at least 1 new token");

    // Clear draft positions from decoder KV cache [n_past, inf)
    llama_memory_seq_rm(llama_get_memory(ctx_decoder), 0, spec->eagle3_n_past, -1);

    // Encoder: features → g_embeddings
    const float * features = llama_get_eagle3_target_features(ctx_tgt);
    GGML_ASSERT(features && "no target features");

    llama_batch enc_batch = {
        /*.n_tokens  =*/ n_new,
        /*.token     =*/ nullptr,
        /*.embd      =*/ const_cast<float*>(features),
        /*.pos       =*/ nullptr, 
        /*.n_seq_id  =*/ nullptr,
        /*.seq_id    =*/ nullptr,
        /*.logits    =*/ nullptr,
    };
    GGML_ASSERT(llama_encode(ctx_encoder, enc_batch) == 0);

    const float * g_embd = llama_get_embeddings(ctx_encoder);
    GGML_ASSERT(g_embd && "encoder output failed");

    // Decoder batch: process new tokens with KV cache reuse
    llama_set_eagle3_g_embeddings(ctx_decoder, g_embd, n_embd, n_new);

    common_batch_clear(batch);
    for (int i = 0; i < n_new; i++) {
        const int pos = spec->eagle3_n_past + i;
        const llama_token tok = (pos < n - 1) ? prompt_tgt[pos + 1] : id_last;
        common_batch_add(batch, tok, pos, {0}, true);
    }

    GGML_ASSERT(llama_decode(ctx_decoder, batch) == 0);

    spec->eagle3_n_past = n;  // update verified positions

    // Sample draft tokens
    llama_tokens result;
    common_sampler_reset(smpl);

    // Sample and check probability (consistent with standard speculative decoding)
    auto sample_and_check = [&](int idx) -> bool {
        common_sampler_sample(smpl, ctx_decoder, idx);

        const auto * cur_p = common_sampler_get_candidates(smpl, true);
        const llama_token id = cur_p->data[0].id;

        common_sampler_accept(smpl, id, true);
        result.push_back(id);

        return cur_p->data[0].p >= params.p_min;
    };

    // First draft token from batch decode
    if (!sample_and_check(n_new - 1)) {
        return result;
    }

    // Autoregressive: use prenorm as g_embd (-1 = last output)
    const float * prenorm = llama_get_embeddings_ith(ctx_decoder, -1);

    for (int i = 1; i < params.n_draft; i++) {
        GGML_ASSERT(prenorm && "prenorm failed");
        llama_set_eagle3_g_embeddings(ctx_decoder, prenorm, n_embd, 1);

        common_batch_clear(batch);
        common_batch_add(batch, result.back(), n - 1 + i, {0}, true);
        GGML_ASSERT(llama_decode(ctx_decoder, batch) == 0);

        prenorm = llama_get_embeddings_ith(ctx_decoder, -1);

        if (!sample_and_check(0)) {
            break;
        }
    }

    return result;
}

llama_tokens common_speculative_gen_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt_main_model, // specified in target model vocab
        llama_token id_last) {

    // EAGLE3 path
    if (spec->eagle3_encoder && spec->eagle3_decoder) {
        return gen_eagle3_draft(spec, params, prompt_tgt_main_model, id_last);
    }

    // Standard draft model path
    auto & batch  = spec->batch;
    auto & ctx_tgt = spec->ctx_tgt;
    auto & ctx_dft = spec->ctx_dft;
    auto & smpl   = spec->smpl;
    auto & prompt_dft = spec->prompt_dft;

    auto * mem_dft = llama_get_memory(ctx_dft);

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(ctx_dft) - params.n_draft;

    llama_tokens prompt_tgt_draft_model;
    if (!spec->vocab_dft_compatible) {
        std::string text;
        text = common_detokenize(ctx_tgt, prompt_tgt_main_model, true);
        text = replace_to_dft(spec, text);
        LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());
        prompt_tgt_draft_model = common_tokenize(ctx_dft, text, false, true);

        // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
        const auto * model_tgt = llama_get_model(ctx_tgt);
        const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

        int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
        GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");
        text.resize(-n_chars);
        llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
        text = replace_to_dft(spec, text);

        LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
        id_last = common_tokenize(ctx_dft, text, false, true)[0];
    }
    // prompt_tgt's tokens will always be compatible with ctx_dft
    const llama_tokens &prompt_tgt =
        spec->vocab_dft_compatible ? prompt_tgt_main_model : prompt_tgt_draft_model;

    const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
    for (int i = 0; i < (int) prompt_dft.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt_tgt.size() &&
               i       + cur < (int) prompt_dft.size() &&
               prompt_tgt[i_start + cur] == prompt_dft[i + cur]) {
            cur++;
        }

        if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

    llama_tokens result;
    result.reserve(params.n_draft);

    if (reuse_n == 0) {
        llama_memory_clear(mem_dft, false);
        prompt_dft.clear();
    } else {
        // this happens when a previous draft has been discarded (for example, due to being too small), but the
        // target model agreed with it. in this case, we simply pass back the previous results to save compute
        if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
            for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                result.push_back(prompt_dft[i]);

                if (params.n_draft <= (int) result.size()) {
                    break;
                }
            }

            return result;
        }

        if (reuse_i > 0) {
            llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
            llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

            prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
        }

        if (reuse_n < (int) prompt_dft.size()) {
            llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
            prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
        }
    }

    // prepare a batch to evaluate any new tokens in the prompt
    common_batch_clear(batch);

    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_tgt[i]);
        common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);

        prompt_dft.push_back(prompt_tgt[i]);
    }

    // we should rarely end-up here during normal decoding
    if (batch.n_tokens > 0) {
        //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

        llama_decode(ctx_dft, batch);
    }

    const llama_pos n_past = prompt_dft.size();

    LOG_DBG("%s: n_past = %d\n", __func__, n_past);

    common_batch_clear(batch);
    common_batch_add  (batch, id_last, n_past, { 0 }, true);

    prompt_dft.push_back(id_last);

    LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

    llama_decode(ctx_dft, batch);

    common_sampler_reset(smpl);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < params.n_draft; ++i) {
        common_batch_clear(batch);

        common_sampler_sample(smpl, ctx_dft, 0);

        const auto * cur_p = common_sampler_get_candidates(smpl, true);

        for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
            LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                    k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
        }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        common_sampler_accept(smpl, id, true);

        result.push_back(id);

        if (params.n_draft <= (int) result.size()) {
            break;
        }

        // only collect very high-confidence draft tokens
        if (cur_p->data[0].p < params.p_min) {
            break;
        }

        common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(ctx_dft, batch);

        prompt_dft.push_back(id);
    }

    if (!spec->vocab_dft_compatible) {
        std::string detokenized = common_detokenize(ctx_dft, result, true);
        detokenized = replace_to_tgt(spec, detokenized);
        LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
        result = common_tokenize(ctx_tgt, detokenized, false, true);
        if (result.size() > (size_t)params.n_draft) {
            result.resize(params.n_draft);
        }
    }
    return result;
}
