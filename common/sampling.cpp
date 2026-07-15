#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler_section {
    llama_sampler * base;
    llama_sampler * reasoning;
    bool reasoning_active;
};

static llama_sampler * common_sampler_section_init(
        llama_sampler * base,
        llama_sampler * reasoning,
        bool reasoning_active = false);

static const char * common_sampler_section_name(const llama_sampler * smpl) {
    const auto * ctx = (const common_sampler_section *) smpl->ctx;
    return llama_sampler_name(ctx->base);
}

static void common_sampler_section_accept(llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_sampler_section *) smpl->ctx;
    llama_sampler_accept(ctx->base, token);
    llama_sampler_accept(ctx->reasoning, token);
}

static void common_sampler_section_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_sampler_section *) smpl->ctx;
    llama_sampler_apply(ctx->reasoning_active ? ctx->reasoning : ctx->base, cur_p);
}

static void common_sampler_section_reset(llama_sampler * smpl) {
    auto * ctx = (common_sampler_section *) smpl->ctx;
    llama_sampler_reset(ctx->base);
    llama_sampler_reset(ctx->reasoning);
}

static llama_sampler * common_sampler_section_clone(const llama_sampler * smpl) {
    const auto * ctx = (const common_sampler_section *) smpl->ctx;
    return common_sampler_section_init(
            llama_sampler_clone(ctx->base),
            llama_sampler_clone(ctx->reasoning),
            ctx->reasoning_active);
}

static void common_sampler_section_free(llama_sampler * smpl) {
    auto * ctx = (common_sampler_section *) smpl->ctx;
    llama_sampler_free(ctx->base);
    llama_sampler_free(ctx->reasoning);
    delete ctx;
}

static llama_sampler_i common_sampler_section_i = {
    /* .name              = */ common_sampler_section_name,
    /* .accept            = */ common_sampler_section_accept,
    /* .apply             = */ common_sampler_section_apply,
    /* .reset             = */ common_sampler_section_reset,
    /* .clone             = */ common_sampler_section_clone,
    /* .free              = */ common_sampler_section_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static llama_sampler * common_sampler_section_init(
        llama_sampler * base,
        llama_sampler * reasoning,
        bool reasoning_active) {
    return llama_sampler_init(
            &common_sampler_section_i,
            new common_sampler_section {
                /* .base             = */ base,
                /* .reasoning        = */ reasoning,
                /* .reasoning_active = */ reasoning_active,
            });
}

struct common_sampler_section_xtc_config {
    float probability;
    float threshold;
    size_t min_keep;
};

struct common_sampler_section_xtc {
    llama_sampler * sampler;
    common_sampler_section_xtc_config base;
    common_sampler_section_xtc_config reasoning;
    bool reasoning_active;
};

static llama_sampler * common_sampler_section_xtc_init_with_sampler(
        llama_sampler * sampler,
        common_sampler_section_xtc_config base,
        common_sampler_section_xtc_config reasoning,
        bool reasoning_active);

static bool common_sampler_section_xtc_enabled(const common_sampler_section_xtc_config & config) {
    return config.probability > 0.0f && config.threshold <= 0.5f;
}

static const char * common_sampler_section_xtc_name(const llama_sampler * /* smpl */) {
    return "xtc";
}

static void common_sampler_section_xtc_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_sampler_section_xtc *) smpl->ctx;
    const auto & config = ctx->reasoning_active ? ctx->reasoning : ctx->base;

    if (!common_sampler_section_xtc_enabled(config)) {
        return;
    }

    llama_sampler_xtc_set(ctx->sampler, config.probability, config.threshold, config.min_keep);
    llama_sampler_apply(ctx->sampler, cur_p);
}

static void common_sampler_section_xtc_reset(llama_sampler * smpl) {
    auto * ctx = (common_sampler_section_xtc *) smpl->ctx;
    llama_sampler_reset(ctx->sampler);
}

static llama_sampler * common_sampler_section_xtc_clone(const llama_sampler * smpl) {
    const auto * ctx = (const common_sampler_section_xtc *) smpl->ctx;
    return common_sampler_section_xtc_init_with_sampler(
            llama_sampler_clone(ctx->sampler),
            ctx->base,
            ctx->reasoning,
            ctx->reasoning_active);
}

static void common_sampler_section_xtc_free(llama_sampler * smpl) {
    auto * ctx = (common_sampler_section_xtc *) smpl->ctx;
    llama_sampler_free(ctx->sampler);
    delete ctx;
}

static llama_sampler_i common_sampler_section_xtc_i = {
    /* .name              = */ common_sampler_section_xtc_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ common_sampler_section_xtc_apply,
    /* .reset             = */ common_sampler_section_xtc_reset,
    /* .clone             = */ common_sampler_section_xtc_clone,
    /* .free              = */ common_sampler_section_xtc_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static llama_sampler * common_sampler_section_xtc_init_with_sampler(
        llama_sampler * sampler,
        common_sampler_section_xtc_config base,
        common_sampler_section_xtc_config reasoning,
        bool reasoning_active) {
    return llama_sampler_init(
            &common_sampler_section_xtc_i,
            new common_sampler_section_xtc {
                /* .sampler          = */ sampler,
                /* .base             = */ base,
                /* .reasoning        = */ reasoning,
                /* .reasoning_active = */ reasoning_active,
            });
}

static llama_sampler * common_sampler_section_xtc_init(
        common_sampler_section_xtc_config base,
        common_sampler_section_xtc_config reasoning,
        uint32_t seed) {
    const auto & initial = common_sampler_section_xtc_enabled(base) ? base : reasoning;
    auto * sampler = llama_sampler_init_xtc(initial.probability, initial.threshold, initial.min_keep, seed);
    return common_sampler_section_xtc_init_with_sampler(sampler, base, reasoning, false);
}

static bool common_sampler_is_section(const llama_sampler * smpl) {
    return smpl->iface == &common_sampler_section_i || smpl->iface == &common_sampler_section_xtc_i;
}

static void common_sampler_section_set_reasoning(llama_sampler * smpl, bool reasoning_active) {
    if (smpl->iface == &common_sampler_section_i) {
        ((common_sampler_section *) smpl->ctx)->reasoning_active = reasoning_active;
        return;
    }

    GGML_ASSERT(smpl->iface == &common_sampler_section_xtc_i);
    ((common_sampler_section_xtc *) smpl->ctx)->reasoning_active = reasoning_active;
}

static std::vector<llama_sampler *> common_sampler_sections(llama_sampler * chain) {
    std::vector<llama_sampler *> result;
    for (int32_t i = 0; i < llama_sampler_chain_n(chain); ++i) {
        auto * smpl = llama_sampler_chain_get(chain, i);
        if (common_sampler_is_section(smpl)) {
            result.push_back(smpl);
        }
    }
    return result;
}

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;
    std::vector<llama_sampler *> sections;

    std::vector<llama_token> prefill_tokens;
    bool grmr_prefilled;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);

        if (rbudget) {
            llama_sampler_reset(rbudget);
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(rbudget, token);
            }
        }
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

static const uint64_t COMMON_SAMPLER_REASONING_SUPPORTED =
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_K |
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_P |
        COMMON_PARAMS_SAMPLING_CONFIG_MIN_P |
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY |
        COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD |
        COMMON_PARAMS_SAMPLING_CONFIG_TEMP |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT |
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_N_SIGMA |
        COMMON_PARAMS_SAMPLING_CONFIG_TYPICAL_P |
        COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_RANGE |
        COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_EXPONENT |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_PRESENT |
        COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER |
        COMMON_PARAMS_SAMPLING_CONFIG_DRY_BASE |
        COMMON_PARAMS_SAMPLING_CONFIG_DRY_ALLOWED_LEN |
        COMMON_PARAMS_SAMPLING_CONFIG_DRY_PENALTY_LAST_N |
        COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP;

static void common_sampler_reasoning_validate(const common_params_sampling & params) {
    const uint64_t unsupported = params.reasoning_sampling & ~COMMON_SAMPLER_REASONING_SUPPORTED;
    if (unsupported != 0) {
        throw std::invalid_argument("unsupported reasoning sampling override");
    }

    if (params.mirostat != 0 && params.reasoning_sampling != 0 &&
        params.reasoning_sampling != COMMON_PARAMS_SAMPLING_CONFIG_TEMP) {
        throw std::invalid_argument("only reasoning temperature can be overridden with Mirostat");
    }
}

static common_params_sampling common_sampler_reasoning_params(const common_params_sampling & params) {
    common_params_sampling result = params;
    const uint64_t config = params.reasoning_sampling;

    if (config & COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP)           { result.min_keep           = params.reasoning_min_keep; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_TOP_K)              { result.top_k              = params.reasoning_top_k; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_TOP_P)              { result.top_p              = params.reasoning_top_p; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_MIN_P)              { result.min_p              = params.reasoning_min_p; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY)    { result.xtc_probability    = params.reasoning_xtc_probability; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD)      { result.xtc_threshold      = params.reasoning_xtc_threshold; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_TYPICAL_P)          { result.typ_p              = params.reasoning_typ_p; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_TEMP)               { result.temp               = params.reasoning_temp; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_RANGE)     { result.dynatemp_range     = params.reasoning_dynatemp_range; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_EXPONENT)  { result.dynatemp_exponent  = params.reasoning_dynatemp_exponent; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N)     { result.penalty_last_n     = params.reasoning_penalty_last_n; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT)     { result.penalty_repeat     = params.reasoning_penalty_repeat; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ)       { result.penalty_freq       = params.reasoning_penalty_freq; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_PRESENT)    { result.penalty_present    = params.reasoning_penalty_present; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER)     { result.dry_multiplier     = params.reasoning_dry_multiplier; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_DRY_BASE)           { result.dry_base           = params.reasoning_dry_base; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_DRY_ALLOWED_LEN)    { result.dry_allowed_length = params.reasoning_dry_allowed_length; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_DRY_PENALTY_LAST_N) { result.dry_penalty_last_n = params.reasoning_dry_penalty_last_n; }
    if (config & COMMON_PARAMS_SAMPLING_CONFIG_TOP_N_SIGMA)        { result.top_n_sigma        = params.reasoning_top_n_sigma; }

    return result;
}

struct common_sampler_chain_result {
    llama_sampler * chain;
    std::vector<llama_sampler *> sections;
    uint64_t reasoning_used;
};

static void common_sampler_chain_add_section(
        common_sampler_chain_result & result,
        std::vector<llama_sampler *> & samplers,
        llama_sampler * base,
        llama_sampler * reasoning,
        uint64_t config) {
    auto * section = common_sampler_section_init(base, reasoning);
    samplers.push_back(section);
    result.sections.push_back(section);
    result.reasoning_used |= config;
}

static void common_sampler_reasoning_warn_unused(uint64_t configured, uint64_t used) {
    struct override_name {
        uint64_t flag;
        const char * name;
    };

    const override_name names[] = {
        { COMMON_PARAMS_SAMPLING_CONFIG_TOP_K,             "top_k" },
        { COMMON_PARAMS_SAMPLING_CONFIG_TOP_P,             "top_p" },
        { COMMON_PARAMS_SAMPLING_CONFIG_MIN_P,             "min_p" },
        { COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY,   "xtc_probability" },
        { COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD,     "xtc_threshold" },
        { COMMON_PARAMS_SAMPLING_CONFIG_TEMP,              "temperature" },
        { COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N,    "repeat_last_n" },
        { COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT,    "repeat_penalty" },
        { COMMON_PARAMS_SAMPLING_CONFIG_TOP_N_SIGMA,       "top_n_sigma" },
        { COMMON_PARAMS_SAMPLING_CONFIG_TYPICAL_P,         "typical_p" },
        { COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_RANGE,    "dynatemp_range" },
        { COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_EXPONENT, "dynatemp_exponent" },
        { COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ,      "frequency_penalty" },
        { COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_PRESENT,   "presence_penalty" },
        { COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER,    "dry_multiplier" },
        { COMMON_PARAMS_SAMPLING_CONFIG_DRY_BASE,          "dry_base" },
        { COMMON_PARAMS_SAMPLING_CONFIG_DRY_ALLOWED_LEN,   "dry_allowed_length" },
        { COMMON_PARAMS_SAMPLING_CONFIG_DRY_PENALTY_LAST_N, "dry_penalty_last_n" },
        { COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP,           "min_keep" },
    };

    for (const auto & entry : names) {
        if ((configured & entry.flag) && !(used & entry.flag)) {
            LOG_WRN("reasoning override '%s' has no matching sampler and will be ignored\n", entry.name);
        }
    }
}

static common_sampler_chain_result common_sampler_chain_build(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const common_params_sampling reasoning = common_sampler_reasoning_params(params);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    common_sampler_chain_result result {
        /* .chain          = */ llama_sampler_chain_init(lparams),
        /* .sections       = */ {},
        /* .reasoning_used = */ 0,
    };

    std::vector<llama_sampler *> samplers;

    if (params.has_logit_bias()) {
        samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_DRY_MULTIPLIER |
                                COMMON_PARAMS_SAMPLING_CONFIG_DRY_BASE |
                                COMMON_PARAMS_SAMPLING_CONFIG_DRY_ALLOWED_LEN |
                                COMMON_PARAMS_SAMPLING_CONFIG_DRY_PENALTY_LAST_N);
                        if (config) {
                            common_sampler_chain_add_section(
                                    result,
                                    samplers,
                                    llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()),
                                    llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), reasoning.dry_multiplier, reasoning.dry_base, reasoning.dry_allowed_length, reasoning.dry_penalty_last_n, c_breakers.data(), c_breakers.size()),
                                    config);
                        } else {
                            samplers.push_back(llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    if (params.reasoning_sampling & COMMON_PARAMS_SAMPLING_CONFIG_TOP_K) {
                        common_sampler_chain_add_section(
                                result,
                                samplers,
                                llama_sampler_init_top_k(params.top_k),
                                llama_sampler_init_top_k(reasoning.top_k),
                                COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
                    } else {
                        samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    {
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_TOP_P |
                                COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP);
                        if (config) {
                            common_sampler_chain_add_section(
                                    result,
                                    samplers,
                                    llama_sampler_init_top_p(params.top_p, params.min_keep),
                                    llama_sampler_init_top_p(reasoning.top_p, reasoning.min_keep),
                                    config);
                        } else {
                            samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    if (params.reasoning_sampling & COMMON_PARAMS_SAMPLING_CONFIG_TOP_N_SIGMA) {
                        common_sampler_chain_add_section(
                                result,
                                samplers,
                                llama_sampler_init_top_n_sigma(params.top_n_sigma),
                                llama_sampler_init_top_n_sigma(reasoning.top_n_sigma),
                                COMMON_PARAMS_SAMPLING_CONFIG_TOP_N_SIGMA);
                    } else {
                        samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    {
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_MIN_P |
                                COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP);
                        if (config) {
                            common_sampler_chain_add_section(
                                    result,
                                    samplers,
                                    llama_sampler_init_min_p(params.min_p, params.min_keep),
                                    llama_sampler_init_min_p(reasoning.min_p, reasoning.min_keep),
                                    config);
                        } else {
                            samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    {
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY |
                                COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD |
                                COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP);
                        if (config) {
                            const common_sampler_section_xtc_config base_config = {
                                params.xtc_probability,
                                params.xtc_threshold,
                                (size_t) params.min_keep,
                            };
                            const common_sampler_section_xtc_config reasoning_config = {
                                reasoning.xtc_probability,
                                reasoning.xtc_threshold,
                                (size_t) reasoning.min_keep,
                            };

                            if (common_sampler_section_xtc_enabled(base_config) ||
                                common_sampler_section_xtc_enabled(reasoning_config)) {
                                auto * section = common_sampler_section_xtc_init(base_config, reasoning_config, params.seed);
                                samplers.push_back(section);
                                result.sections.push_back(section);
                            } else {
                                samplers.push_back(llama_sampler_init_xtc(
                                        params.xtc_probability,
                                        params.xtc_threshold,
                                        params.min_keep,
                                        params.seed));
                            }
                            result.reasoning_used |= config;
                        } else {
                            samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    {
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_TYPICAL_P |
                                COMMON_PARAMS_SAMPLING_CONFIG_MIN_KEEP);
                        if (config) {
                            common_sampler_chain_add_section(
                                    result,
                                    samplers,
                                    llama_sampler_init_typical(params.typ_p, params.min_keep),
                                    llama_sampler_init_typical(reasoning.typ_p, reasoning.min_keep),
                                    config);
                        } else {
                            samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    {
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_TEMP |
                                COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_RANGE |
                                COMMON_PARAMS_SAMPLING_CONFIG_DYNATEMP_EXPONENT);
                        if (config) {
                            common_sampler_chain_add_section(
                                    result,
                                    samplers,
                                    llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent),
                                    llama_sampler_init_temp_ext(reasoning.temp, reasoning.dynatemp_range, reasoning.dynatemp_exponent),
                                    config);
                        } else {
                            samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    {
                        const uint64_t config = params.reasoning_sampling & (
                                COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N |
                                COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT |
                                COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_FREQ |
                                COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_PRESENT);
                        if (config) {
                            common_sampler_chain_add_section(
                                    result,
                                    samplers,
                                    llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present),
                                    llama_sampler_init_penalties(reasoning.penalty_last_n, reasoning.penalty_repeat, reasoning.penalty_freq, reasoning.penalty_present),
                                    config);
                        } else {
                            samplers.push_back(llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                        }
                    }
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        if (params.reasoning_sampling & COMMON_PARAMS_SAMPLING_CONFIG_TEMP) {
            common_sampler_chain_add_section(
                    result,
                    samplers,
                    llama_sampler_init_temp(params.temp),
                    llama_sampler_init_temp(reasoning.temp),
                    COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
        } else {
            samplers.push_back(llama_sampler_init_temp(params.temp));
        }
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        if (params.reasoning_sampling & COMMON_PARAMS_SAMPLING_CONFIG_TEMP) {
            common_sampler_chain_add_section(
                    result,
                    samplers,
                    llama_sampler_init_temp(params.temp),
                    llama_sampler_init_temp(reasoning.temp),
                    COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
        } else {
            samplers.push_back(llama_sampler_init_temp(params.temp));
        }
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(result.chain, smpl);
    }

    common_sampler_reasoning_warn_unused(params.reasoning_sampling, result.reasoning_used);

    return result;
}

static llama_sampler * common_sampler_reasoning_budget_init(
        const llama_vocab * vocab,
        const common_params_sampling & params,
        const std::vector<llama_token> & prefill_tokens) {
    if (params.reasoning_budget_start.empty() || params.reasoning_budget_end.empty() ||
        !(params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control || params.reasoning_sampling)) {
        return nullptr;
    }

    auto * rbudget = common_reasoning_budget_init(
        vocab,
        params.reasoning_budget_start,
        params.reasoning_budget_end,
        params.reasoning_budget_forced,
        params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

    for (const auto & token : prefill_tokens) {
        llama_sampler_accept(rbudget, token);
        LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
    }

    return rbudget;
}

static std::vector<llama_token> common_sampler_prefill_tokens(
        const llama_vocab * vocab,
        const std::string & generation_prompt) {
    std::vector<llama_token> result;
    if (generation_prompt.empty()) {
        return result;
    }

    GGML_ASSERT(vocab != nullptr);
    auto tokens = common_tokenize(vocab, generation_prompt, false, true);
    for (size_t i = 0; i < tokens.size(); i++) {
        std::string piece = common_token_to_piece(vocab, tokens[i], true);
        if (i == 0 && !piece.empty() &&
            std::isspace(static_cast<unsigned char>(piece[0])) &&
            !std::isspace(static_cast<unsigned char>(generation_prompt[0]))) {
            continue;
        }
        LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
        result.push_back(tokens[i]);
    }

    return result;
}

// Feed generation prompt tokens to the grammar sampler so it advances past
// tokens the template already placed in the prompt.
// Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
// Returns true when the grammar was advanced, so callers can avoid feeding it twice.
static bool common_sampler_grammar_prefill(
        struct llama_sampler * grmr,
        const common_params_sampling & params,
        const std::vector<llama_token> & prefill_tokens) {
    if (!grmr || params.grammar_lazy || !common_grammar_needs_prefill(params.grammar) || prefill_tokens.empty()) {
        return false;
    }

    try {
        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(grmr, token);
            LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
        }
    } catch (std::exception &e) {
        LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
            common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
        throw e;
    }

    return true;
}

struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params) {
    common_sampler_reasoning_validate(params);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    auto prefill_tokens = common_sampler_prefill_tokens(vocab, params.generation_prompt);

    const bool grmr_prefilled = common_sampler_grammar_prefill(grmr, params, prefill_tokens);

    rbudget = common_sampler_reasoning_budget_init(vocab, params, prefill_tokens);

    auto chain_result = common_sampler_chain_build(model, params);

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if ((rbudget || params.reasoning_sampling) && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning sampling or budgets, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params         = */ params,
        /* .grmr           = */ grmr,
        /* .rbudget        = */ rbudget,
        /* .chain          = */ chain_result.chain,
        /* .sections       = */ std::move(chain_result.sections),
        /* .prefill_tokens = */ std::move(prefill_tokens),
        /* .grmr_prefilled = */ grmr_prefilled,
        /* .prev           = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur            = */ {},
        /* .cur_p          = */ {},
    };

    return result;
}

void common_sampler_configure_reasoning(
        struct common_sampler * gsmpl,
        const llama_vocab * vocab,
        const common_params_sampling & params) {
    if (!gsmpl) {
        return;
    }

    gsmpl->params.generation_prompt        = params.generation_prompt;
    gsmpl->params.reasoning_budget_start   = params.reasoning_budget_start;
    gsmpl->params.reasoning_budget_end     = params.reasoning_budget_end;
    gsmpl->params.reasoning_budget_forced  = params.reasoning_budget_forced;
    gsmpl->params.reasoning_budget_tokens  = params.reasoning_budget_tokens;
    gsmpl->params.reasoning_budget_message = params.reasoning_budget_message;
    gsmpl->params.reasoning_control        = params.reasoning_control;

    gsmpl->prefill_tokens = common_sampler_prefill_tokens(vocab, params.generation_prompt);

    if (!gsmpl->grmr_prefilled) {
        gsmpl->grmr_prefilled = common_sampler_grammar_prefill(gsmpl->grmr, gsmpl->params, gsmpl->prefill_tokens);
    }

    llama_sampler_free(gsmpl->rbudget);
    gsmpl->rbudget = common_sampler_reasoning_budget_init(vocab, gsmpl->params, gsmpl->prefill_tokens);
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

// true while generation is inside the reasoning block (as tracked by the
// reasoning budget sampler), i.e. after the start tag and before the end tag
static bool common_sampler_reasoning_active(const struct common_sampler * gsmpl) {
    if (!gsmpl->rbudget) {
        return false;
    }
    const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
    return state == REASONING_BUDGET_COUNTING ||
           state == REASONING_BUDGET_WAITING_UTF8 ||
           state == REASONING_BUDGET_FORCING;
}

static void common_sampler_apply_chain(common_sampler * gsmpl, llama_token_data_array * cur_p) {
    const bool reasoning_active = common_sampler_reasoning_active(gsmpl);
    for (auto * smpl : gsmpl->sections) {
        common_sampler_section_set_reasoning(smpl, reasoning_active);
    }

    llama_sampler_apply(gsmpl->chain, cur_p);
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * chain = llama_sampler_clone(gsmpl->chain);
    return new common_sampler {
        /* .params         = */ gsmpl->params,
        /* .grmr           = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget        = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain          = */ chain,
        /* .sections       = */ common_sampler_sections(chain),
        /* .prefill_tokens = */ gsmpl->prefill_tokens,
        /* .grmr_prefilled = */ gsmpl->grmr_prefilled,
        /* .prev           = */ gsmpl->prev,
        /* .cur            = */ gsmpl->cur,
        /* .cur_p          = */ gsmpl->cur_p,
    };
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    common_sampler_apply_chain(gsmpl, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    common_sampler_apply_chain(gsmpl, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
