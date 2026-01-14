#include "speculative.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "ngram-map.cpp"

#include <cstring>
#include <algorithm>
#include <map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct common_speculative_self {
    uint16_t         size_ngram = 12;      // size of n-grams to lookup in self-mode
    uint16_t         size_mgram = 48;      // size of m-grams to draft in self-mode
    const uint16_t   check_rate = 3;       // check for speculative decoding without draft model for each check_rate token
    size_t           idx_last_check  =  0; // index of last check in context history
};

struct common_speculative {
    struct llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    struct llama_context * ctx_dft;
    struct common_sampler * smpl;

    llama_batch batch;
    llama_tokens prompt_dft;
    bool vocab_dft_compatible = true; // whether retokenization is needed
    std::map<std::string, std::string> tgt_dft_replacements = {};

    const uint16_t   self_mode  = 0; // 0: off, 1: self speculative, 2: n-grams (keys) only, 3: n-grams/m-grams (key-values)
    common_ngram_map        map;        // draft ngram map for speculative decoding without draft model
    common_speculative_self self_state; // state of self-speculation (simple implementation, not ngram-map)
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft,
        uint16_t                    self_mode, // 0: off, 1: self speculative, 2: n-grams (keys) only, 3: n-grams/m-grams (key-values)
        const std::vector<uint16_t> self_cfg   // ngram size, mgram size, keys only (0|1), min hits
    ) {
    uint16_t ngram_size_key   = self_cfg.size() >= 1 ? self_cfg[0] : 12;
    uint16_t mgram_size_value = self_cfg.size() >= 2 ? self_cfg[1] : 48;
    uint16_t check_rate = self_cfg.size() >= 3 ? self_cfg[2] : 3;
    bool     key_only = (self_mode != 3);
    uint16_t min_hits = self_cfg.size() >= 4 ? self_cfg[3] : 1;
    common_ngram_map ngram_map = common_ngram_map(ngram_size_key, mgram_size_value, key_only, check_rate, min_hits);
    common_speculative_self self_state = common_speculative_self{
        /* .size_ngram      = */ ngram_size_key,
        /* .size_mgram      = */ mgram_size_value,
        /* .check_rate      = */ check_rate,
        /* .idx_last_check  = */ 0,
    };
    auto * result = new common_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ ctx_dft,
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(ctx_dft ? llama_n_batch(ctx_dft) : 64, 0, 1),
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ false,
        /* .tgt_dft_replacements = */ {},
        /* .self_mode       = */ self_mode,
        /* .map             = */ ngram_map,
        /* .self_state      = */ self_state
    };

    LOG_INF("common_speculative_init: created speculative decoder, map.n = %d\n", result->map.size_key);
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

        if (ctx_dft) {
            result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }
    }
#endif

    result->vocab_dft_compatible = common_speculative_are_compatible(ctx_tgt, ctx_dft);
    LOG_DBG("vocab_dft_compatible = %d\n", result->vocab_dft_compatible);

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    common_sampler_free(spec->smpl);

    llama_batch_free(spec->batch);

    delete spec;
}

bool common_speculative_are_compatible(
    const struct llama_context * ctx_tgt,
    const struct llama_context * ctx_dft) {
    if (ctx_tgt == nullptr && ctx_dft == nullptr) {
        return true;
    }
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

llama_tokens common_speculative_gen_self_draft(
        common_speculative * spec,
        const llama_tokens & tokens, llama_token sampled);

llama_tokens common_speculative_gen_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt_main_model, // specified in target model vocab
        llama_token id_last) {
    if (spec->self_mode) {
        // Look in the current context for a n-gram and return the following tokens as the draft.
        llama_tokens draft_self = common_speculative_gen_self_draft(spec,
                prompt_tgt_main_model, id_last);
        if (!draft_self.empty()) {
            return draft_self;
        }
    }
    if (spec == nullptr || spec->ctx_dft == nullptr) {
        return {}; // no draft model, return
    }

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

        common_sampler_sample(smpl, ctx_dft, 0, true);

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

void common_speculative_send_accepted(struct common_speculative * spec, const uint16_t n_accepted) {
    // use new function to update the ngram map statistics.
    common_ngram_map_send_accepted(spec->map, n_accepted);
}

// self-speculative decoding
//

/**
 * Perform speculative generation using the model's own token history.
 * Searches for a matching pattern in the token history and returns draft tokens.
 *
 * @param spec      configuration of speculative drafts
 * @param tokens    Token history to search in
 * @param sampled   Last sampled token
 * @return Vector of draft tokens, empty if no matching pattern is found
 */
llama_tokens common_speculative_gen_self_draft(
        common_speculative * spec,
        const llama_tokens & tokens, llama_token sampled) {

    common_ngram_map & map = spec->map;
    if (spec->self_mode != 1) {
        // Use common_ngram_map_draft to generate a draft from the current context.
        llama_tokens draft_tokens;
        common_ngram_map_draft(map, tokens, sampled, draft_tokens);
        return draft_tokens;
    }

    // Simple implementation of self-speculative decoding without draft model, without ngram-map.
    //
    common_speculative_self & self_state = spec->self_state;
    const size_t cur_len = tokens.size();
    // Only check every check_rate tokens to save compute
    // i.e., perform check if (cur_len - idx_last_check) >= check_rate
    if (self_state.idx_last_check + self_state.check_rate > cur_len) {
        llama_tokens draft_tokens;
        return draft_tokens;
    }

    size_t n_draft_min = self_state.size_ngram; // size of n-gram to lookup in token history
    size_t n_draft_max = self_state.size_mgram; // the m-gram following the found n-gram is used for draft

    // vector for tokens we want to verify.
    // return empty vector if there is no match.
    llama_tokens draft_tokens;

    // We need at least n_draft_min + n_draft_max + 1 tokens.
    if (cur_len <= static_cast<size_t>(n_draft_min + n_draft_max + 1)) {
        return draft_tokens;
    }

    // pattern search
    llama_tokens pattern;
    pattern.reserve(n_draft_min);
    for (size_t j = cur_len - n_draft_min + 1; j < cur_len; ++j) {
        pattern.push_back(tokens[j]);
    }
    pattern.push_back(sampled); // add the last token to the pattern

    // We do a search in the token history.
    self_state.idx_last_check = tokens.size();

    size_t match_pos = 0; // we ignore position 0, position 0 == no match
    // search backwards, but skip the current match (we are currently there)
    for (size_t j = cur_len - n_draft_min - 1; j > 0; --j) {
        bool match = true;
        for (size_t k = 0; k < pattern.size(); ++k) {
            if (tokens[j + k] != pattern[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            match_pos = j;
            break;
        }
    }
    if (match_pos == 0) {
        return draft_tokens;
    }

    const size_t copy_max = std::min(
        n_draft_max,
        cur_len - (match_pos + n_draft_min)
    );
    if (copy_max < n_draft_min) {
        return draft_tokens;
    }
    LOG_DBG("%s: #tokens = %zu: found matching pattern at pos %zu, length %zu, draft length %zu\n",
        __func__, cur_len,
        match_pos, pattern.size(), copy_max);

    draft_tokens.reserve(copy_max);
    for (size_t j = 0; j < copy_max; ++j) {
        draft_tokens.push_back(tokens[match_pos + n_draft_min + j]);
    }
    return draft_tokens;
}

void common_speculative_print_stats(const struct common_speculative * spec) {
    if (spec->map.drafts_generated_tokens > 0) { // only print if we have some stats
        common_ngram_map_print_stats(spec->map);
    }
}
