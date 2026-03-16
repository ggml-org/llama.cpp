#include "speculative.h"

#include "../src/llama-kv-cache-iswa.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_MTP
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

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
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}
    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;
    virtual void begin(const llama_tokens & prompt, llama_pos retained_prefix_len) {
        GGML_UNUSED(retained_prefix_len);
        begin(prompt);
    }

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) = 0;

    virtual llama_pos get_committed_prefix_len() const {
        return 0;
    }

    virtual void invalidate_retained_state() {
    }

    virtual void set_first_pass_source(
            const llama_tokens & source_tokens,
            const float *        hidden_states,
            int32_t              n_tokens,
            int32_t              n_embd,
            llama_pos            start_pos) {
        GGML_UNUSED(source_tokens);
        GGML_UNUSED(hidden_states);
        GGML_UNUSED(n_tokens);
        GGML_UNUSED(n_embd);
        GGML_UNUSED(start_pos);
    }

};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

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
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
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

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
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
        for (int i = 0; i < params.n_max; ++i) {
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

            if (params.n_max <= (int) result.size()) {
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

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        // noop
        GGML_UNUSED(batch_idxs);
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        // noop
        GGML_UNUSED(batch_idxs);
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        // noop
        GGML_UNUSED(batch_idxs);
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        GGML_UNUSED(batch_idxs);
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

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

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        GGML_UNUSED(batch_idxs);
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        // TODO: noop
        GGML_UNUSED(batch_idxs);
        GGML_UNUSED(n_accepted);
    }
};

static bool copy_hidden_state(
        llama_context *      ctx,
        int32_t              idx,
        std::vector<float> & dst,
        int32_t              dst_row,
        int32_t              n_embd) {
    float * hidden = llama_get_embeddings_ith(ctx, idx);
    if (hidden == nullptr) {
        return false;
    }

    std::memcpy(dst.data() + (int64_t) dst_row*n_embd, hidden, n_embd*sizeof(float));
    return true;
}

struct common_speculative_state_mtp : public common_speculative_state {
    enum mtp_draft_step_status : int {
        MTP_DRAFT_STEP_GUARD_STOP = -2,
        MTP_DRAFT_STEP_DECODE_FAIL = -1,
        MTP_DRAFT_STEP_STOP = 0,
        MTP_DRAFT_STEP_CONTINUE = 1,
    };

    struct mtp_round_state {
        llama_token frontier_token = 0;
        llama_pos prompt_size = 0;
        llama_tokens draft_tokens;
        std::vector<float> recurrence_hidden;
    };

    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    common_sampler * smpl = nullptr;
    llama_batch batch;

    const int32_t mtp_layer_idx;
    const int32_t n_embd;

    // retained state
    llama_pos    verified_pos_end     = 0;
    llama_pos    committed_prefix_len = 0;
    llama_kv_cache_iswa * draft_kv_iswa = nullptr;
    bool retained_state_valid = true;

    // staged sources
    llama_tokens initial_source_tokens;
    std::vector<float> initial_source_hidden_states;
    llama_pos initial_source_start_pos = 0;

    llama_tokens pending_target_tokens;
    std::vector<float> pending_hidden_states;
    llama_pos pending_start_pos = 0;

    mtp_round_state round;

    size_t n_guard_stop_rounds = 0;             // rounds that conservatively stopped because the guard found no safe reuse position
    size_t n_guard_zero_draft_rounds = 0;       // subset of guard stops that degraded to 0-draft immediately
    size_t n_guard_partial_rounds = 0;          // subset of guard stops that drafted some tokens but stopped before reaching n_max
    size_t n_guard_lost_tokens = 0;             // total draft tokens lost to guard degradation, computed as n_max - drafted
    size_t n_draft_decode_fail_rounds = 0;      // regular draft decode failures unrelated to the guard

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , batch(llama_batch_init(llama_n_batch(ctx_dft), 0, 1))
        , mtp_layer_idx(llama_model_n_layer(llama_get_model(ctx_dft)) - llama_model_n_nextn_predict_layers(llama_get_model(ctx_dft)))
        , n_embd(llama_model_n_embd(llama_get_model(ctx_dft))) {
        common_params_sampling sparams;
        sparams.no_perf = false;
        sparams.top_k = 10;
        sparams.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        auto * mem = llama_get_memory(ctx_dft);
        draft_kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(mem);

        smpl = common_sampler_init(llama_get_model(ctx_dft), sparams);
        round.recurrence_hidden.resize(n_embd);

        llama_set_embeddings(ctx_dft, true);
    }

    ~common_speculative_state_mtp() override {
        llama_perf_context_print(ctx_dft);
        llama_free(ctx_dft);
        common_sampler_free(smpl);
        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        begin(prompt, 0);
    }

    llama_pos apply_retained_prefix(llama_pos retained_prefix_len) {
        retained_prefix_len = std::max<llama_pos>(retained_prefix_len, 0);

        llama_pos retained_prefix_applied = 0;
        if (retained_state_valid) {
            retained_prefix_applied = std::min(retained_prefix_len, committed_prefix_len);
            if (retained_prefix_applied != retained_prefix_len) {
                LOG_WRN("%s: clamping retained prefix from %d to committed prefix %d\n",
                        __func__, (int) retained_prefix_len, (int) committed_prefix_len);
            }
        }

        verified_pos_end = retained_prefix_applied;
        committed_prefix_len = retained_prefix_applied;

        if (auto * mem = llama_get_memory(ctx_dft)) {
            if (retained_prefix_applied == 0) {
                llama_memory_clear(mem, true);
            } else if (!llama_memory_seq_rm(mem, 0, retained_prefix_applied, -1)) {
                LOG_WRN("%s: failed to truncate retained draft state at %d - clearing memory instead\n",
                        __func__, (int) retained_prefix_applied);
                llama_memory_clear(mem, true);
                retained_prefix_applied = 0;
                verified_pos_end = 0;
                committed_prefix_len = 0;
            }
        }

        return retained_prefix_applied;
    }

    void begin(const llama_tokens & prompt, llama_pos retained_prefix_len) override {
        GGML_UNUSED(prompt);

        // 1. Clear staged first-pass sources and round-local state.
        initial_source_tokens.clear();
        initial_source_hidden_states.clear();
        pending_start_pos = 0;
        initial_source_start_pos = 0;

        pending_target_tokens.clear();
        pending_hidden_states.clear();

        round.frontier_token = 0;
        round.prompt_size = 0;
        round.draft_tokens.clear();

        // 2. Re-apply the retained prefix boundary.
        apply_retained_prefix(retained_prefix_len);

        // 3. Reset draft-side sampler and inputs for the new round.
        retained_state_valid = true;
        common_sampler_reset(smpl);
        llama_set_mtp_op_type(ctx_dft, LLAMA_MTP_OP_NONE);
        llama_set_draft_input_hidden_state(ctx_dft, nullptr);
        llama_synchronize(ctx_dft);
    }

    llama_pos get_committed_prefix_len() const override {
        return retained_state_valid ? committed_prefix_len : 0;
    }

    void invalidate_retained_state() override {
        retained_state_valid = false;
        committed_prefix_len = 0;
        verified_pos_end = 0;
        pending_start_pos = 0;
        initial_source_start_pos = 0;
        initial_source_tokens.clear();
        initial_source_hidden_states.clear();
        pending_target_tokens.clear();
        pending_hidden_states.clear();
        round.draft_tokens.clear();

        if (auto * mem = llama_get_memory(ctx_dft)) {
            llama_memory_clear(mem, true);
        }

        common_sampler_reset(smpl);
        llama_set_mtp_op_type(ctx_dft, LLAMA_MTP_OP_NONE);
        llama_set_draft_input_hidden_state(ctx_dft, nullptr);
        llama_synchronize(ctx_dft);
    }

    void set_first_pass_source(
            const llama_tokens & source_tokens,
            const float *        hidden_states,
            int32_t              n_tokens,
            int32_t              n_embd_in,
            llama_pos            start_pos) override {
        initial_source_tokens.clear();
        initial_source_hidden_states.clear();
        initial_source_start_pos = 0;

        if (n_tokens <= 0 || source_tokens.empty()) {
            return;
        }

        if ((int32_t) source_tokens.size() != n_tokens) {
            LOG_WRN("%s: ignoring first-pass source with mismatched token count (%zu != %d)\n",
                    __func__, source_tokens.size(), n_tokens);
            return;
        }

        if (hidden_states == nullptr) {
            LOG_WRN("%s: ignoring first-pass source without hidden states\n", __func__);
            return;
        }

        if (n_embd_in != n_embd) {
            LOG_WRN("%s: ignoring first-pass source with mismatched n_embd (%d != %d)\n",
                    __func__, n_embd_in, n_embd);
            return;
        }

        initial_source_tokens = source_tokens;
        initial_source_hidden_states.assign(hidden_states, hidden_states + (int64_t) n_tokens*n_embd);
        initial_source_start_pos = start_pos;
    }

    bool select_first_pass_source(
            const llama_tokens *&       source_tokens,
            const std::vector<float> *& source_hidden_states,
            llama_pos &                 source_start_pos) const {
        if (!pending_target_tokens.empty()) {
            source_tokens = &pending_target_tokens;
            source_hidden_states = &pending_hidden_states;
            source_start_pos = pending_start_pos;
            return true;
        }

        if (!initial_source_tokens.empty()) {
            source_tokens = &initial_source_tokens;
            source_hidden_states = &initial_source_hidden_states;
            source_start_pos = initial_source_start_pos;
            return true;
        }

        return false;
    }

    bool supports_swa_guard() const {
        return draft_kv_iswa != nullptr;
    }

    void set_swa_guard(llama_pos query_pos) {
        GGML_ASSERT(draft_kv_iswa != nullptr);
        draft_kv_iswa->set_swa_reuse_guard(query_pos);
    }

    void clear_swa_guard() {
        if (draft_kv_iswa != nullptr) {
            draft_kv_iswa->clear_swa_reuse_guard();
        }
    }

    int classify_decode_failure() {
        const bool blocked = draft_kv_iswa != nullptr
            ? draft_kv_iswa->consume_swa_reuse_guard_block_prepare()
            : false;

        return blocked
            ? MTP_DRAFT_STEP_GUARD_STOP
            : MTP_DRAFT_STEP_DECODE_FAIL;
    }

    void record_guard_stop(const common_params_speculative & params, size_t drafted_tokens) {
        ++n_guard_stop_rounds;
        n_guard_lost_tokens += params.n_max > (int32_t) drafted_tokens ? params.n_max - drafted_tokens : 0;

        const bool zero_draft = drafted_tokens == 0;
        if (zero_draft) {
            ++n_guard_zero_draft_rounds;
        } else {
            ++n_guard_partial_rounds;
        }

        LOG_DBG("%s: MTP draft stopped by SWA guard: round.prompt_size = %d, drafted = %zu/%d, zero_draft = %d\n",
                __func__,
                (int) round.prompt_size,
                drafted_tokens,
                params.n_max,
                zero_draft);
    }

    void record_decode_fail() {
        ++n_draft_decode_fail_rounds;
    }

    void record_step_failure(int step_status, const common_params_speculative & params, size_t drafted_tokens) {
        GGML_ASSERT(step_status < 0);

        if (step_status == MTP_DRAFT_STEP_GUARD_STOP) {
            record_guard_stop(params, drafted_tokens);
        } else {
            record_decode_fail();
        }
    }

    int finalize_step_from_logits(
            int32_t output_idx,
            const common_params_speculative & params,
            llama_tokens & result) {
        common_sampler_sample(smpl, ctx_dft, output_idx, true);
        const auto * cur_p = common_sampler_get_candidates(smpl, true);
        if (!cur_p || cur_p->size == 0) {
            return MTP_DRAFT_STEP_DECODE_FAIL;
        }

        const llama_token id = cur_p->data[0].id;
        const float       p  = cur_p->data[0].p;

        common_sampler_accept(smpl, id, true);
        result.push_back(id);

        float * next_hidden = llama_get_embeddings_ith(ctx_dft, output_idx);
        if (next_hidden == nullptr) {
            return MTP_DRAFT_STEP_DECODE_FAIL;
        }

        std::memcpy(round.recurrence_hidden.data(), next_hidden, n_embd*sizeof(float));
        return p >= params.p_min ? MTP_DRAFT_STEP_CONTINUE : MTP_DRAFT_STEP_STOP;
    }

    int run_first_pass(
            const llama_tokens &       source_tokens,
            const std::vector<float> & source_hidden_states,
            llama_pos                  start_pos,
            llama_token                frontier_token,
            const common_params_speculative & params,
            llama_tokens &             result) {
        const int32_t n_tokens = (int32_t) source_tokens.size();
        if (n_tokens <= 0) {
            return -1;
        }

        if ((int64_t) source_hidden_states.size() != (int64_t) n_tokens*n_embd) {
            LOG_WRN("%s: first-pass hidden-state size mismatch (%zu vs %d)\n",
                    __func__, source_hidden_states.size(), n_tokens*n_embd);
            return -1;
        }

        common_batch_clear(batch);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const llama_token token = i + 1 < n_tokens ? source_tokens[i + 1] : frontier_token;
            common_batch_add(batch, token, start_pos + i, { 0 }, true);
        }

        llama_set_mtp_op_type(ctx_dft, LLAMA_MTP_OP_DRAFT_GEN);
        llama_set_mtp_layer_idx(ctx_dft, mtp_layer_idx);
        llama_set_draft_input_hidden_state(ctx_dft, source_hidden_states.data());
        const auto clear_draft_input = [&]() {
            llama_set_mtp_op_type(ctx_dft, LLAMA_MTP_OP_NONE);
            llama_set_draft_input_hidden_state(ctx_dft, nullptr);
        };

        if (llama_decode(ctx_dft, batch) != 0) {
            const int status = classify_decode_failure();
            clear_draft_input();
            return status;
        }

        const int step_status = finalize_step_from_logits(n_tokens - 1, params, result);
        if (step_status >= 0) {
            verified_pos_end = start_pos + n_tokens;
            committed_prefix_len = verified_pos_end;
            retained_state_valid = true;
        }

        clear_draft_input();
        return step_status;
    }

    int run_single_token_step(
            llama_token frontier_token,
            llama_pos   pos,
            const common_params_speculative & params,
            llama_tokens & result) {
        common_batch_clear(batch);
        common_batch_add(batch, frontier_token, pos, { 0 }, true);

        llama_set_mtp_op_type(ctx_dft, LLAMA_MTP_OP_DRAFT_GEN);
        llama_set_mtp_layer_idx(ctx_dft, mtp_layer_idx);
        llama_set_draft_input_hidden_state(ctx_dft, round.recurrence_hidden.data());
        const auto clear_draft_input = [&]() {
            llama_set_mtp_op_type(ctx_dft, LLAMA_MTP_OP_NONE);
            llama_set_draft_input_hidden_state(ctx_dft, nullptr);
        };

        if (llama_decode(ctx_dft, batch) != 0) {
            const int status = classify_decode_failure();
            clear_draft_input();
            return status;
        }

        const int step_status = finalize_step_from_logits(0, params, result);
        clear_draft_input();
        return step_status;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        // 1. round setup
        result.clear();
        result.reserve(params.n_max);

        round.frontier_token = id_last;
        round.prompt_size = (llama_pos) prompt_tgt.size();
        round.draft_tokens.clear();

        // 2. draft memory / guard
        auto * mem = llama_get_memory(ctx_dft);
        if (mem == nullptr || !supports_swa_guard()) {
            LOG_WRN("%s: MTP draft requires llama_kv_cache or llama_kv_cache_iswa memory\n", __func__);
            record_decode_fail();
            pending_target_tokens.clear();
            pending_hidden_states.clear();
            return;
        }
        set_swa_guard(round.prompt_size);

        do {
            llama_memory_seq_rm(mem, 0, verified_pos_end, -1);

            // 3. source selection + first pass
            const llama_tokens * source_tokens = nullptr;
            const std::vector<float> * source_hidden_states = nullptr;
            llama_pos source_start_pos = 0;

            if (!select_first_pass_source(source_tokens, source_hidden_states, source_start_pos)) {
                break;
            }

            const bool consumed_pending_source = source_tokens == &pending_target_tokens;
            common_sampler_reset(smpl);

            const int first_pass_status = run_first_pass(*source_tokens, *source_hidden_states, source_start_pos, id_last, params, result);

            // initial_source_* is a one-shot first-pass source, regardless of success or failure.
            initial_source_tokens.clear();
            initial_source_hidden_states.clear();
            initial_source_start_pos = 0;

            if (first_pass_status < 0) {
                record_step_failure(first_pass_status, params, result.size());

                // A failed first pass must conservatively drop any staged pending source.
                pending_target_tokens.clear();
                pending_hidden_states.clear();
                break;
            }

            if (consumed_pending_source) {
                pending_target_tokens.clear();
                pending_hidden_states.clear();
            }

            if (!result.empty()) {
                round.draft_tokens.push_back(result.back());
            }

            if (first_pass_status != MTP_DRAFT_STEP_CONTINUE) {
                break;
            }

            // 4. recurrence
            llama_pos next_pos = verified_pos_end;
            while ((int) result.size() < params.n_max) {
                const size_t result_size_prev = result.size();
                const int step_status = run_single_token_step(result.back(), next_pos, params, result);
                if (result.size() == result_size_prev) {
                    if (step_status < 0) {
                        record_step_failure(step_status, params, result.size());
                    }
                    break;
                }

                round.draft_tokens.push_back(result.back());
                next_pos++;
                if (step_status == MTP_DRAFT_STEP_STOP) {
                    break;
                }
                if (step_status < 0) {
                    record_step_failure(step_status, params, result.size());
                    break;
                }
            }
        } while (false);

        // 5. cleanup
        clear_swa_guard();
    }

    void accept(uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) override {
        // 1. clamp n_accepted to the drafted prefix that actually exists.
        const int32_t n_accepted_clamped = std::min<int32_t>(n_accepted, (int32_t) round.draft_tokens.size());

        // 2. build the pending source tokens for the next draft round.
        pending_target_tokens.clear();
        pending_hidden_states.clear();
        pending_start_pos = 0;

        pending_target_tokens.reserve(n_accepted_clamped + 1);
        pending_target_tokens.push_back(round.frontier_token);
        for (int32_t i = 0; i < n_accepted_clamped; ++i) {
            pending_target_tokens.push_back(round.draft_tokens[i]);
        }

        pending_hidden_states.resize((int64_t) pending_target_tokens.size()*n_embd);
        pending_start_pos = round.prompt_size;

        // 3. copy verifier hidden states into pending_hidden_states.
        for (int32_t i = 0; i < (int32_t) pending_target_tokens.size(); ++i) {
            if (!batch_idxs.empty() && (size_t) i >= batch_idxs.size()) {
                LOG_WRN("%s: batch_idxs missing verifier index for pending token %d\n", __func__, i);
                pending_target_tokens.clear();
                pending_hidden_states.clear();
                pending_start_pos = 0;
                break;
            }
            const int32_t hidden_idx = batch_idxs.empty() ? i : batch_idxs[i];
            if (!copy_hidden_state(ctx_tgt, hidden_idx, pending_hidden_states, i, n_embd)) {
                LOG_WRN("%s: failed to copy target hidden state %d for pending first pass\n", __func__, hidden_idx);
                pending_target_tokens.clear();
                pending_hidden_states.clear();
                pending_start_pos = 0;
                break;
            }
        }
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    auto * mem = llama_get_memory(ctx_tgt);
    if (mem == nullptr) {
        return false;
    }

    bool res = true;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = params.has_dft();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        bool has_mtp           = (params.type == COMMON_SPECULATIVE_TYPE_MTP);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_MTP, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                const llama_model * model_tgt = llama_get_model(ctx_tgt);

                if (llama_model_is_recurrent(model_tgt)) {
                    LOG_WRN("%s: MTP speculative does not support recurrent memory models yet\n", __func__);
                    break;
                }

                if (llama_model_is_hybrid(model_tgt)) {
                    LOG_WRN("%s: MTP speculative does not support hybrid memory models yet\n", __func__);
                    break;
                }

                if (llama_model_n_nextn_predict_layers(model_tgt) <= 0) {
                    LOG_WRN("%s: target model has no nextn_predict_layers\n", __func__);
                    break;
                }

                llama_context_params cparams = config.params.cparams_dft;

                if (cparams.n_ctx == 0) {
                    cparams.n_ctx = llama_n_ctx_seq(ctx_tgt);
                }
                if (cparams.n_batch == 0) {
                    cparams.n_batch = llama_n_ctx_seq(ctx_tgt);
                }
                if (cparams.n_ubatch == 0) {
                    cparams.n_ubatch = llama_n_ubatch(ctx_tgt);
                }
                if (cparams.n_threads <= 0) {
                    cparams.n_threads = llama_n_threads(ctx_tgt);
                }
                if (cparams.n_threads_batch <= 0) {
                    cparams.n_threads_batch = llama_n_threads_batch(ctx_tgt);
                }

                cparams.embeddings = true;

                llama_context * ctx_mtp = llama_init_from_model(const_cast<llama_model *>(llama_get_model(ctx_tgt)), cparams);
                if (ctx_mtp == nullptr) {
                    LOG_WRN("%s", "failed to initialize dedicated MTP draft context\n");
                    break;
                }

                auto * mem_mtp = llama_get_memory(ctx_mtp);
                if (dynamic_cast<llama_kv_cache_iswa *>(mem_mtp) == nullptr) {
                    LOG_WRN("%s: MTP draft context requires llama_kv_cache_iswa memory for current SWA/iSWA rollback guard\n",
                            __func__);
                    llama_free(ctx_mtp);
                    break;
                }

                llama_set_embeddings(ctx_mtp, true);
                impls.push_back(std::make_unique<common_speculative_state_mtp>(config.type, ctx_tgt, ctx_mtp));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    common_speculative_begin(spec, prompt, 0);
}

void common_speculative_begin(
        common_speculative * spec,
        const llama_tokens & prompt,
        llama_pos            retained_prefix_len) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt, retained_prefix_len);
        impl->n_call_begin++;
    }
}

llama_pos common_speculative_get_committed_prefix_len(
        const common_speculative * spec) {
    if (spec == nullptr) {
        return 0;
    }

    llama_pos result = 0;
    for (const auto & impl : spec->impls) {
        result = std::max(result, impl->get_committed_prefix_len());
    }

    return result;
}

void common_speculative_invalidate_retained_state(
        common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->invalidate_retained_state();
    }
}

void common_speculative_set_first_pass_source(
        common_speculative * spec,
        const llama_tokens & source_tokens,
        const float *        hidden_states,
        int32_t              n_tokens,
        int32_t              n_embd,
        llama_pos            start_pos) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_first_pass_source(source_tokens, hidden_states, n_tokens, n_embd, start_pos);
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted, const std::vector<int32_t> & batch_idxs) {
    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted, batch_idxs);
        impl->n_call_accept++;
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

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());

        if (auto * mtp = dynamic_cast<const common_speculative_state_mtp *>(impl.get())) {
            LOG_INF("statistics %s: guard stops = %zu, zero-draft guard stops = %zu, partial guard stops = %zu, guard lost tokens = %zu, draft decode fail rounds = %zu\n",
                    common_speculative_type_to_str(impl->type).c_str(),
                    mtp->n_guard_stop_rounds,
                    mtp->n_guard_zero_draft_rounds,
                    mtp->n_guard_partial_rounds,
                    mtp->n_guard_lost_tokens,
                    mtp->n_draft_decode_fail_rounds);
        }
    }
}
