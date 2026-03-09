#include "sampling.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>
#include <unordered_map>

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

//
// Multi-grammar state machine
//

enum common_grammar_state {
    COMMON_GRAMMAR_STATE_IDLE,      // not constraining (delayed: waiting for arm trigger)
    COMMON_GRAMMAR_STATE_ARMED,     // delayed: countdown in progress
    COMMON_GRAMMAR_STATE_ACTIVE,    // actively constraining
    COMMON_GRAMMAR_STATE_EXHAUSTED, // grammar done, no longer constraining (one-shot)
    COMMON_GRAMMAR_STATE_DEFUSED,   // delayed: countdown cancelled (one-shot)
};

struct common_grammar_instance {
    common_grammar_spec spec;
    common_grammar_state state = COMMON_GRAMMAR_STATE_IDLE;
    struct llama_sampler * sampler = nullptr;
    int32_t countdown_remaining = 0;

    // trigger matching state for arm/defuse (managed at common_sampler level)
    std::string arm_trigger_buffer;
    std::string defuse_trigger_buffer;
};

// helper: create a llama grammar sampler from a grammar spec
static llama_sampler * common_grammar_sampler_create(
        const llama_vocab * vocab,
        const common_grammar_spec & spec) {
    if (spec.grammar.empty()) {
        return nullptr;
    }

    if (spec.grammar_lazy) {
        // lazy grammar: use existing trigger mechanism
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;

        for (const auto & trigger : spec.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
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
                    trigger_tokens.push_back(trigger.token);
                    break;
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        return llama_sampler_init_grammar_lazy_patterns(vocab, spec.grammar.c_str(), "root",
                trigger_patterns_c.data(), trigger_patterns_c.size(),
                trigger_tokens.data(), trigger_tokens.size());
    }

    // regular (non-lazy) grammar
    return llama_sampler_init_grammar(vocab, spec.grammar.c_str(), "root");
}

// helper: check if a token matches any trigger in a trigger list
// returns true if a trigger matched
static bool common_grammar_check_triggers(
        const llama_vocab * vocab,
        const std::vector<common_grammar_trigger> & triggers,
        std::string & trigger_buffer,
        llama_token token) {
    if (triggers.empty()) {
        return false;
    }

    // check token triggers first (fast path)
    for (const auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN && trigger.token == token) {
            trigger_buffer.clear();
            return true;
        }
    }

    // get the token piece for text-based trigger matching
    // use llama_token_to_piece via the public API
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    std::string piece;
    if (n > 0) {
        piece.assign(buf, n);
    } else if (n < 0) {
        // buffer too small, allocate
        std::vector<char> large_buf(-n);
        llama_token_to_piece(vocab, token, large_buf.data(), large_buf.size(), 0, true);
        piece.assign(large_buf.data(), -n);
    }

    if (piece.empty()) {
        return false;
    }

    trigger_buffer += piece;

    // check word and pattern triggers against the accumulated buffer
    for (const auto & trigger : triggers) {
        switch (trigger.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            {
                if (trigger_buffer.find(trigger.value) != std::string::npos) {
                    trigger_buffer.clear();
                    return true;
                }
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            {
                try {
                    std::regex re(trigger.value);
                    if (std::regex_search(trigger_buffer, re)) {
                        trigger_buffer.clear();
                        return true;
                    }
                } catch (const std::regex_error &) {
                    LOG_WRN("%s: invalid regex pattern in trigger: %s\n", __func__, trigger.value.c_str());
                }
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            {
                try {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    std::regex re(anchored);
                    if (std::regex_match(trigger_buffer, re)) {
                        trigger_buffer.clear();
                        return true;
                    }
                } catch (const std::regex_error &) {
                    LOG_WRN("%s: invalid regex pattern in trigger: %s\n", __func__, trigger.value.c_str());
                }
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                // already handled above
                break;
            default:
                break;
        }
    }

    return false;
}

struct common_sampler {
    common_params_sampling params;

    const llama_vocab * vocab = nullptr;

    std::vector<common_grammar_instance> grammars;

    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    bool has_active_grammars() const {
        for (const auto & g : grammars) {
            if (g.state == COMMON_GRAMMAR_STATE_ACTIVE) {
                return true;
            }
        }
        return false;
    }

    void apply_active_grammars(llama_token_data_array * cur_p_) {
        for (auto & g : grammars) {
            if (g.state == COMMON_GRAMMAR_STATE_ACTIVE && g.sampler) {
                if (g.spec.non_terminating) {
                    // For non-terminating grammars, detect if the grammar is complete
                    // (only EOG tokens survive). If so, skip applying and transition out.
                    // Save logits before applying so we can restore if needed.
                    std::vector<float> saved_logits(cur_p_->size);
                    for (size_t i = 0; i < cur_p_->size; i++) {
                        saved_logits[i] = cur_p_->data[i].logit;
                    }

                    llama_sampler_apply(g.sampler, cur_p_);

                    // Check if any non-EOG token survived
                    bool has_non_eog = false;
                    for (size_t i = 0; i < cur_p_->size; i++) {
                        if (cur_p_->data[i].logit > -INFINITY
                                && !llama_vocab_is_eog(vocab, cur_p_->data[i].id)) {
                            has_non_eog = true;
                            break;
                        }
                    }

                    if (!has_non_eog) {
                        // Grammar is complete — restore logits and transition out
                        for (size_t i = 0; i < cur_p_->size; i++) {
                            cur_p_->data[i].logit = saved_logits[i];
                        }
                        if (g.spec.rearmable) {
                            g.state = COMMON_GRAMMAR_STATE_IDLE;
                            llama_sampler_reset(g.sampler);
                            g.arm_trigger_buffer.clear();
                            g.defuse_trigger_buffer.clear();
                            LOG_INF("%s: grammar '%s' completed, returning to IDLE\n",
                                    __func__, g.spec.id.c_str());
                        } else {
                            g.state = COMMON_GRAMMAR_STATE_EXHAUSTED;
                            LOG_INF("%s: grammar '%s' completed (one-shot)\n",
                                    __func__, g.spec.id.c_str());
                        }
                    }
                } else {
                    llama_sampler_apply(g.sampler, cur_p_);
                }
            }
        }
    }

    bool check_token_against_active_grammars(llama_token id) {
        for (auto & g : grammars) {
            if (g.state == COMMON_GRAMMAR_STATE_ACTIVE && g.sampler) {
                llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
                llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };
                llama_sampler_apply(g.sampler, &single_token_data_array);
                if (single_token_data_array.data[0].logit == -INFINITY) {
                    return false;
                }
            }
        }
        return true;
    }

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * v     = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(v);

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

struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * chain = llama_sampler_chain_init(lparams);

    // build the list of grammar specs to use
    std::vector<common_grammar_spec> specs;

    if (!params.grammar_specs.empty()) {
        // use the new multi-grammar specs
        specs = params.grammar_specs;
    } else if (!params.grammar.empty()) {
        // backward compat: create a single grammar spec from legacy fields
        common_grammar_spec legacy_spec;
        legacy_spec.id            = "default";
        legacy_spec.grammar       = params.grammar;
        legacy_spec.grammar_lazy  = params.grammar_lazy;
        legacy_spec.grammar_triggers = params.grammar_triggers;
        specs.push_back(std::move(legacy_spec));
    }

    // create grammar instances
    std::vector<common_grammar_instance> grammars;
    bool has_any_grammar = false;

    for (auto & spec : specs) {
        common_grammar_instance instance;
        instance.spec = spec;

        if (spec.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
            instance.sampler = llama_sampler_init_llg(vocab, "lark", spec.grammar.c_str());
            instance.state = COMMON_GRAMMAR_STATE_ACTIVE;
#else
            GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif
        } else if (spec.delayed) {
            // delayed grammar: create sampler, start state depends on arm_immediately and countdown
            instance.sampler = common_grammar_sampler_create(vocab, spec);
            instance.countdown_remaining = spec.countdown;
            if (spec.arm_immediately && spec.countdown <= 0) {
                // countdown already expired at init: start directly in ACTIVE
                instance.state = COMMON_GRAMMAR_STATE_ACTIVE;
                LOG_INF("%s: grammar '%s' starting directly in ACTIVE (countdown=%d)\n",
                        __func__, spec.id.c_str(), spec.countdown);
            } else if (spec.arm_immediately) {
                instance.state = COMMON_GRAMMAR_STATE_ARMED;
            } else {
                instance.state = COMMON_GRAMMAR_STATE_IDLE;
            }
        } else {
            // regular or lazy grammar: create sampler, start ACTIVE
            // (for lazy grammars, the internal awaiting_trigger handles the lazy behavior)
            instance.sampler = common_grammar_sampler_create(vocab, spec);
            instance.state = instance.sampler ? COMMON_GRAMMAR_STATE_ACTIVE : COMMON_GRAMMAR_STATE_IDLE;
        }

        if (instance.sampler) {
            has_any_grammar = true;
        }

        grammars.push_back(std::move(instance));
    }

    // build the sampling chain (unchanged from original)
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
                        samplers.push_back(llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
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
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (has_any_grammar && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params   = */ params,
        /* .vocab    = */ vocab,
        /* .grammars = */ std::move(grammars),
        /* .chain    = */ chain,
        /* .prev     = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur      = */ {},
        /* .cur_p    = */ {},
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    for (auto & g : gsmpl->grammars) {
        llama_sampler_free(g.sampler);
    }
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    if (accept_grammar) {
        for (auto & g : gsmpl->grammars) {
            if (!g.sampler) {
                continue;
            }

            switch (g.state) {
                case COMMON_GRAMMAR_STATE_IDLE:
                {
                    if (g.spec.delayed) {
                        // check arm triggers
                        bool armed = common_grammar_check_triggers(
                                gsmpl->vocab, g.spec.arm_triggers,
                                g.arm_trigger_buffer, token);
                        if (armed) {
                            g.state = COMMON_GRAMMAR_STATE_ARMED;
                            g.countdown_remaining = g.spec.countdown;
                            g.defuse_trigger_buffer.clear();
                            LOG_INF("%s: grammar '%s' armed, countdown = %d\n",
                                    __func__, g.spec.id.c_str(), g.countdown_remaining);
                        }
                    }
                    // for non-delayed grammars in IDLE state, nothing to do
                    // (lazy grammars handle their own triggers internally in ACTIVE state)
                    break;
                }
                case COMMON_GRAMMAR_STATE_ARMED:
                {
                    // check defuse triggers first
                    bool defused = common_grammar_check_triggers(
                            gsmpl->vocab, g.spec.defuse_triggers,
                            g.defuse_trigger_buffer, token);
                    if (defused) {
                        g.state = g.spec.rearmable ? COMMON_GRAMMAR_STATE_IDLE : COMMON_GRAMMAR_STATE_DEFUSED;
                        g.arm_trigger_buffer.clear();
                        LOG_INF("%s: grammar '%s' defused (rearmable=%d)\n",
                                __func__, g.spec.id.c_str(), g.spec.rearmable);
                        break;
                    }

                    // decrement countdown
                    if (g.spec.countdown_mode == COMMON_GRAMMAR_COUNTDOWN_TOKENS) {
                        g.countdown_remaining--;
                    } else {
                        // character countdown
                        char buf[256];
                        int n = llama_token_to_piece(gsmpl->vocab, token, buf, sizeof(buf), 0, true);
                        if (n > 0) {
                            g.countdown_remaining -= n;
                        } else if (n < 0) {
                            g.countdown_remaining -= (-n);
                        }
                    }

                    if (g.countdown_remaining <= 0) {
                        // countdown expired: activate grammar
                        llama_sampler_reset(g.sampler);
                        g.state = COMMON_GRAMMAR_STATE_ACTIVE;
                        g.defuse_trigger_buffer.clear();
                        LOG_INF("%s: grammar '%s' activated after countdown\n",
                                __func__, g.spec.id.c_str());
                    }
                    break;
                }
                case COMMON_GRAMMAR_STATE_ACTIVE:
                {
                    try {
                        llama_sampler_accept(g.sampler, token);
                    } catch (const std::runtime_error & e) {
                        if (g.spec.non_terminating) {
                            if (g.spec.rearmable) {
                                g.state = COMMON_GRAMMAR_STATE_IDLE;
                                llama_sampler_reset(g.sampler);
                                g.arm_trigger_buffer.clear();
                                g.defuse_trigger_buffer.clear();
                                LOG_INF("%s: grammar '%s' exhausted, returning to IDLE (rearmable)\n",
                                        __func__, g.spec.id.c_str());
                            } else {
                                g.state = COMMON_GRAMMAR_STATE_EXHAUSTED;
                                LOG_INF("%s: grammar '%s' exhausted (one-shot)\n",
                                        __func__, g.spec.id.c_str());
                            }
                        } else {
                            throw; // re-throw if not non-terminating
                        }
                    }
                    break;
                }
                case COMMON_GRAMMAR_STATE_EXHAUSTED:
                case COMMON_GRAMMAR_STATE_DEFUSED:
                    // terminal states, nothing to do
                    break;
            }
        }
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
    std::vector<common_grammar_instance> cloned_grammars;
    cloned_grammars.reserve(gsmpl->grammars.size());
    for (const auto & g : gsmpl->grammars) {
        common_grammar_instance cloned;
        cloned.spec                 = g.spec;
        cloned.state                = g.state;
        cloned.sampler              = g.sampler ? llama_sampler_clone(g.sampler) : nullptr;
        cloned.countdown_remaining  = g.countdown_remaining;
        cloned.arm_trigger_buffer   = g.arm_trigger_buffer;
        cloned.defuse_trigger_buffer = g.defuse_trigger_buffer;
        cloned_grammars.push_back(std::move(cloned));
    }

    return new common_sampler {
        /* .params   = */ gsmpl->params,
        /* .vocab    = */ gsmpl->vocab,
        /* .grammars = */ std::move(cloned_grammars),
        /* .chain    = */ llama_sampler_clone(gsmpl->chain),
        /* .prev     = */ gsmpl->prev,
        /* .cur      = */ gsmpl->cur,
        /* .cur_p    = */ gsmpl->cur_p,
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

        llama_memory_breakdown_print(ctx);
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

    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->has_active_grammars() && "using grammar in combination with backend sampling is not supported");

            // TODO: simplify
            gsmpl->cur.resize(1);
            gsmpl->cur[0] = { id, 0.0f, 1.0f };
            cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), 0, true };

            return id;
        }
    }

    gsmpl->set_logits(ctx, idx);

    const bool has_grammars = gsmpl->has_active_grammars();

    if (grammar_first && has_grammars) {
        gsmpl->apply_active_grammars(&cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !has_grammars) {
        return id;
    }

    // check if the sampled token fits all active grammars (grammar-based rejection sampling)
    if (gsmpl->check_token_against_active_grammars(id)) {
        return id;
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar samplers and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    gsmpl->apply_active_grammars(&cur_p);
    llama_sampler_apply(chain, &cur_p);

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

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, common_sampler_type> sampler_canonical_name_map {
        { "dry",         COMMON_SAMPLER_TYPE_DRY },
        { "top_k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top_p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min_p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
        { "xtc",         COMMON_SAMPLER_TYPE_XTC },
        { "infill",      COMMON_SAMPLER_TYPE_INFILL },
        { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES },
        { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, common_sampler_type> sampler_alt_name_map {
        { "top-k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top-p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "top-n-sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { "nucleus",     COMMON_SAMPLER_TYPE_TOP_P },
        { "typical-p",   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typical",     COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ-p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ",         COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min-p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "temp",        COMMON_SAMPLER_TYPE_TEMPERATURE },
        { "adaptive-p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        auto sampler = sampler_canonical_name_map.find(name);
        if (sampler != sampler_canonical_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        if (allow_alt_names) {
            sampler = sampler_alt_name_map.find(name);
            if (sampler != sampler_alt_name_map.end()) {
                samplers.push_back(sampler->second);
                continue;
            }
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name.c_str());
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
