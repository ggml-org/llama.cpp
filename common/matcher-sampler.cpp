#include "matcher-sampler.h"
#include "common.h"
#include "unicode.h"

#include "log.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

// --- token_matcher -----------------------------------------------------------

bool token_matcher::advance(llama_token token) {
    if (tokens.empty()) {
        return false;
    }

    if (token == tokens[pos]) {
        pos++;
        if (pos >= tokens.size()) {
            pos = 0;
            return true;
        }
    } else {
        pos = 0;
        if (token == tokens[0]) {
            pos = 1;
        }
    }
    return false;
}

// --- SSM data structures -----------------------------------------------------

struct matcher_ssm_reasoning_budget {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;

    int32_t budget;
    int32_t remaining;

    matcher_ssm_rb_state state;

    size_t force_pos;  // next position in forced_tokens to force
};

// A trigger matcher with its own replay buffer.
// Multiple triggers can be watched in parallel; the first one to complete fires.
struct tcg_trigger {
    token_matcher matcher;
    std::vector<llama_token> replay_buffer;
};

struct matcher_ssm_tool_call_grammar {
    token_matcher thinking_start_matcher;
    token_matcher thinking_end_matcher;
    std::vector<tcg_trigger> triggers;

    matcher_ssm_tcg_state state;

    // The grammar sampler to delegate to when in GRAMMAR_SAMPLING.
    // Owned by this SSM (freed on destruction, cloned on clone).
    struct llama_sampler * grammar_sampler;
};

using matcher_ssm = std::variant<matcher_ssm_reasoning_budget, matcher_ssm_tool_call_grammar>;

struct common_matcher_sampler_ctx {
    std::vector<matcher_ssm> ssms;
};

// --- Reasoning budget SSM accept/apply ---------------------------------------

static void rb_accept(matcher_ssm_reasoning_budget & rb, llama_token token) {
    switch (rb.state) {
        case MATCHER_SSM_RB_IDLE:
        {
            if (rb.start_matcher.advance(token)) {
                rb.state = MATCHER_SSM_RB_COUNTING;
                rb.remaining = rb.budget;
                LOG_INF("matcher-sampler: reasoning-budget activated, budget=%d tokens\n", rb.budget);

                if (rb.remaining <= 0) {
                    rb.state = MATCHER_SSM_RB_FORCING;
                    rb.force_pos = 0;
                    LOG_INF("matcher-sampler: reasoning-budget budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case MATCHER_SSM_RB_COUNTING:
        case MATCHER_SSM_RB_WAITING_UTF8:
        {
            if (rb.end_matcher.advance(token)) {
                rb.state = MATCHER_SSM_RB_DONE;
                LOG_INF("matcher-sampler: reasoning-budget deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (rb.vocab != nullptr) {
                const std::string piece = common_token_to_piece(rb.vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (rb.state == MATCHER_SSM_RB_WAITING_UTF8) {
                if (utf8_complete) {
                    rb.state = MATCHER_SSM_RB_FORCING;
                    rb.force_pos = 0;
                    rb.end_matcher.reset();
                    LOG_INF("matcher-sampler: reasoning-budget UTF-8 complete, now forcing end sequence\n");
                }
            } else if (rb.state == MATCHER_SSM_RB_COUNTING) {
                rb.remaining--;
                if (rb.remaining <= 0) {
                    if (utf8_complete) {
                        rb.state = MATCHER_SSM_RB_FORCING;
                        rb.force_pos = 0;
                        rb.end_matcher.reset();
                        LOG_INF("matcher-sampler: reasoning-budget exhausted, forcing end sequence\n");
                    } else {
                        rb.state = MATCHER_SSM_RB_WAITING_UTF8;
                        rb.end_matcher.reset();
                        LOG_INF("matcher-sampler: reasoning-budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case MATCHER_SSM_RB_FORCING:
            // force_pos is advanced in apply(), not here.
            break;
        case MATCHER_SSM_RB_DONE:
            break;
    }
}

static void rb_apply(matcher_ssm_reasoning_budget & rb, llama_token_data_array * cur_p) {
    if (rb.state != MATCHER_SSM_RB_FORCING) {
        return;
    }

    if (rb.force_pos >= rb.forced_tokens.size()) {
        return;
    }

    const llama_token forced = rb.forced_tokens[rb.force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }

    // advance to next forced token (done here rather than in accept so that
    // the first forced token isn't skipped when starting in FORCING state)
    rb.force_pos++;
    if (rb.force_pos >= rb.forced_tokens.size()) {
        rb.state = MATCHER_SSM_RB_DONE;
        LOG_INF("matcher-sampler: reasoning-budget forced sequence complete, done\n");
    }
}

static void rb_reset(matcher_ssm_reasoning_budget & rb) {
    rb.state = MATCHER_SSM_RB_IDLE;
    rb.remaining = rb.budget;
    rb.start_matcher.reset();
    rb.end_matcher.reset();
    rb.force_pos = 0;
}

// --- Tool call grammar SSM accept/apply --------------------------------------

static void tcg_accept(matcher_ssm_tool_call_grammar & tcg, llama_token token) {
    switch (tcg.state) {
        case MATCHER_SSM_TCG_OUT_OF_THINKING:
        {
            if (tcg.thinking_start_matcher.advance(token)) {
                tcg.state = MATCHER_SSM_TCG_IN_THINKING;
                // Clear any partial tool call matches and replay buffers
                for (auto & trigger : tcg.triggers) {
                    trigger.matcher.reset();
                    trigger.replay_buffer.clear();
                }
                LOG_INF("matcher-sampler: tool-call-grammar entered thinking block\n");
                break;
            }

            // Check all triggers in parallel; the first one to complete fires
            for (auto & trigger : tcg.triggers) {
                size_t prev_pos = trigger.matcher.matched();
                if (trigger.matcher.advance(token)) {
                    // Full match! Replay buffered tokens + this token into the grammar
                    trigger.replay_buffer.push_back(token);
                    for (const auto & t : trigger.replay_buffer) {
                        llama_sampler_accept(tcg.grammar_sampler, t);
                    }
                    trigger.replay_buffer.clear();
                    tcg.state = MATCHER_SSM_TCG_GRAMMAR_SAMPLING;
                    LOG_INF("matcher-sampler: tool-call-grammar activated, replayed %zu tokens\n",
                            trigger.matcher.tokens.size());

                    // Reset all other triggers
                    for (auto & other : tcg.triggers) {
                        if (&other != &trigger) {
                            other.matcher.reset();
                            other.replay_buffer.clear();
                        }
                    }
                    return;  // state changed, done
                }

                // Manage replay buffer based on matcher state
                size_t new_pos = trigger.matcher.matched();
                if (new_pos == 0 && prev_pos > 0) {
                    // Matcher reset — discard the buffer
                    trigger.replay_buffer.clear();
                }
                if (new_pos > 0) {
                    // We're in a partial match; buffer the token
                    if (trigger.replay_buffer.size() < new_pos) {
                        trigger.replay_buffer.push_back(token);
                    } else {
                        // Matcher reset to 1 (current token re-started match)
                        trigger.replay_buffer.clear();
                        trigger.replay_buffer.push_back(token);
                    }
                }
            }
            break;
        }
        case MATCHER_SSM_TCG_IN_THINKING:
        {
            if (tcg.thinking_end_matcher.advance(token)) {
                tcg.state = MATCHER_SSM_TCG_OUT_OF_THINKING;
                LOG_INF("matcher-sampler: tool-call-grammar exited thinking block\n");
            }
            break;
        }
        case MATCHER_SSM_TCG_GRAMMAR_SAMPLING:
        {
            llama_sampler_accept(tcg.grammar_sampler, token);
            break;
        }
    }
}

static void tcg_apply(matcher_ssm_tool_call_grammar & tcg, llama_token_data_array * cur_p) {
    if (tcg.state != MATCHER_SSM_TCG_GRAMMAR_SAMPLING) {
        return;
    }

    llama_sampler_apply(tcg.grammar_sampler, cur_p);
}

static void tcg_reset(matcher_ssm_tool_call_grammar & tcg) {
    tcg.state = MATCHER_SSM_TCG_OUT_OF_THINKING;
    tcg.thinking_start_matcher.reset();
    tcg.thinking_end_matcher.reset();
    for (auto & trigger : tcg.triggers) {
        trigger.matcher.reset();
        trigger.replay_buffer.clear();
    }
    if (tcg.grammar_sampler) {
        llama_sampler_reset(tcg.grammar_sampler);
    }
}

static void tcg_free(matcher_ssm_tool_call_grammar & tcg) {
    if (tcg.grammar_sampler) {
        llama_sampler_free(tcg.grammar_sampler);
        tcg.grammar_sampler = nullptr;
    }
}

// --- Matcher sampler llama_sampler_i callbacks --------------------------------

static const char * common_matcher_sampler_name(const struct llama_sampler * /*smpl*/) {
    return "matcher-sampler";
}

static void common_matcher_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_matcher_sampler_ctx *) smpl->ctx;

    for (auto & ssm : ctx->ssms) {
        std::visit([token](auto & s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, matcher_ssm_reasoning_budget>) {
                rb_accept(s, token);
            } else if constexpr (std::is_same_v<T, matcher_ssm_tool_call_grammar>) {
                tcg_accept(s, token);
            }
        }, ssm);
    }
}

static void common_matcher_sampler_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_matcher_sampler_ctx *) smpl->ctx;

    for (auto & ssm : ctx->ssms) {
        std::visit([cur_p](auto & s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, matcher_ssm_reasoning_budget>) {
                rb_apply(s, cur_p);
            } else if constexpr (std::is_same_v<T, matcher_ssm_tool_call_grammar>) {
                tcg_apply(s, cur_p);
            }
        }, ssm);
    }
}

static void common_matcher_sampler_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_matcher_sampler_ctx *) smpl->ctx;

    for (auto & ssm : ctx->ssms) {
        std::visit([](auto & s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, matcher_ssm_reasoning_budget>) {
                rb_reset(s);
            } else if constexpr (std::is_same_v<T, matcher_ssm_tool_call_grammar>) {
                tcg_reset(s);
            }
        }, ssm);
    }
}

static struct llama_sampler * common_matcher_sampler_clone(const struct llama_sampler * smpl);

static void common_matcher_sampler_free(struct llama_sampler * smpl) {
    auto * ctx = (common_matcher_sampler_ctx *) smpl->ctx;

    for (auto & ssm : ctx->ssms) {
        std::visit([](auto & s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, matcher_ssm_tool_call_grammar>) {
                tcg_free(s);
            }
        }, ssm);
    }

    delete ctx;
}

static struct llama_sampler_i common_matcher_sampler_i = {
    /* .name              = */ common_matcher_sampler_name,
    /* .accept            = */ common_matcher_sampler_accept,
    /* .apply             = */ common_matcher_sampler_apply,
    /* .reset             = */ common_matcher_sampler_reset,
    /* .clone             = */ common_matcher_sampler_clone,
    /* .free              = */ common_matcher_sampler_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

// --- Clone helpers -----------------------------------------------------------

static matcher_ssm clone_ssm(const matcher_ssm & src) {
    return std::visit([](const auto & s) -> matcher_ssm {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, matcher_ssm_reasoning_budget>) {
            return s; // trivially copyable members
        } else if constexpr (std::is_same_v<T, matcher_ssm_tool_call_grammar>) {
            matcher_ssm_tool_call_grammar copy = s;
            copy.grammar_sampler = s.grammar_sampler ? llama_sampler_clone(s.grammar_sampler) : nullptr;
            return copy;
        }
    }, src);
}

static struct llama_sampler * common_matcher_sampler_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_matcher_sampler_ctx *) smpl->ctx;

    auto * new_ctx = new common_matcher_sampler_ctx;
    new_ctx->ssms.reserve(ctx->ssms.size());
    for (const auto & ssm : ctx->ssms) {
        new_ctx->ssms.push_back(clone_ssm(ssm));
    }

    return llama_sampler_init(&common_matcher_sampler_i, new_ctx);
}

// --- Internal init helper ----------------------------------------------------

static struct llama_sampler * create_matcher_sampler(std::vector<matcher_ssm> && ssms) {
    return llama_sampler_init(
        &common_matcher_sampler_i,
        new common_matcher_sampler_ctx { std::move(ssms) }
    );
}

// --- Reasoning budget init (prefill) -----------------------------------------

static matcher_ssm_rb_state rb_initial_state_from_prefill(
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & prefill_tokens) {
    matcher_ssm_rb_state initial_state = MATCHER_SSM_RB_IDLE;
    if (!prefill_tokens.empty() && !start_tokens.empty() &&
            prefill_tokens.size() >= start_tokens.size() &&
            std::equal(start_tokens.begin(), start_tokens.end(), prefill_tokens.begin())) {
        initial_state = MATCHER_SSM_RB_COUNTING;
        // If the end sequence also follows the start in the prefill, reasoning
        // was opened and immediately closed — stay IDLE.
        if (!end_tokens.empty() &&
                prefill_tokens.size() >= start_tokens.size() + end_tokens.size()) {
            auto end_start = prefill_tokens.end() - (ptrdiff_t) end_tokens.size();
            if (end_start >= prefill_tokens.begin() + (ptrdiff_t) start_tokens.size() &&
                    std::equal(end_tokens.begin(), end_tokens.end(), end_start)) {
                initial_state = MATCHER_SSM_RB_IDLE;
            }
        }
    }
    return initial_state;
}

// --- Public API --------------------------------------------------------------

struct llama_sampler * common_matcher_sampler_init_reasoning_budget(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        const std::vector<llama_token> & prefill_tokens) {
    matcher_ssm_rb_state initial_state = rb_initial_state_from_prefill(start_tokens, end_tokens, prefill_tokens);
    return common_matcher_sampler_init_reasoning_budget(vocab, start_tokens, end_tokens, forced_tokens, budget, initial_state);
}

struct llama_sampler * common_matcher_sampler_init_reasoning_budget(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        matcher_ssm_rb_state             initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == MATCHER_SSM_RB_COUNTING && budget <= 0) {
        initial_state = MATCHER_SSM_RB_FORCING;
    }

    std::vector<matcher_ssm> ssms;
    ssms.emplace_back(matcher_ssm_reasoning_budget {
        /* .vocab         = */ vocab,
        /* .start_matcher = */ { start_tokens, 0 },
        /* .end_matcher   = */ { end_tokens, 0 },
        /* .forced_tokens = */ forced_tokens,
        /* .budget        = */ budget,
        /* .remaining     = */ budget,
        /* .state         = */ initial_state,
        /* .force_pos     = */ 0,
    });

    return create_matcher_sampler(std::move(ssms));
}

static std::vector<tcg_trigger> make_tcg_triggers(
        const std::vector<std::vector<llama_token>> & tool_call_start_seqs) {
    std::vector<tcg_trigger> triggers;
    triggers.reserve(tool_call_start_seqs.size());
    for (const auto & seq : tool_call_start_seqs) {
        triggers.push_back({ /* .matcher = */ { seq, 0 }, /* .replay_buffer = */ {} });
    }
    return triggers;
}

void common_matcher_sampler_add_tool_call_grammar(
        struct llama_sampler                          * matcher_sampler,
        const std::vector<llama_token>                & thinking_start_tokens,
        const std::vector<llama_token>                & thinking_end_tokens,
        const std::vector<std::vector<llama_token>>   & tool_call_start_seqs,
        struct llama_sampler                          * grammar_sampler) {
    auto * ctx = (common_matcher_sampler_ctx *) matcher_sampler->ctx;

    ctx->ssms.emplace_back(matcher_ssm_tool_call_grammar {
        /* .thinking_start_matcher = */ { thinking_start_tokens, 0 },
        /* .thinking_end_matcher   = */ { thinking_end_tokens, 0 },
        /* .triggers               = */ make_tcg_triggers(tool_call_start_seqs),
        /* .state                  = */ MATCHER_SSM_TCG_OUT_OF_THINKING,
        /* .grammar_sampler        = */ grammar_sampler,
    });
}

struct llama_sampler * common_matcher_sampler_init_tool_call_grammar(
        const std::vector<llama_token>                & thinking_start_tokens,
        const std::vector<llama_token>                & thinking_end_tokens,
        const std::vector<std::vector<llama_token>>   & tool_call_start_seqs,
        struct llama_sampler                          * grammar_sampler) {
    std::vector<matcher_ssm> ssms;
    ssms.emplace_back(matcher_ssm_tool_call_grammar {
        /* .thinking_start_matcher = */ { thinking_start_tokens, 0 },
        /* .thinking_end_matcher   = */ { thinking_end_tokens, 0 },
        /* .triggers               = */ make_tcg_triggers(tool_call_start_seqs),
        /* .state                  = */ MATCHER_SSM_TCG_OUT_OF_THINKING,
        /* .grammar_sampler        = */ grammar_sampler,
    });

    return create_matcher_sampler(std::move(ssms));
}
