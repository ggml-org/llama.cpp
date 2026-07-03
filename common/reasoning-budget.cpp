#include "reasoning-budget.h"
#include "common.h"
#include "unicode.h"

#include "log.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_token> tokens;
    size_t pos = 0;

    bool advance(llama_token token) {
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

    void reset() { pos = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;
    std::vector<llama_token> forced_message_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget
    int32_t warn_offset;      // offset before budget exhaustion to show message

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    if (ctx->warn_offset > 0) {
                        ctx->state = REASONING_BUDGET_FORCING_MESSAGE;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "budget=0, forcing message immediately\n");
                    } else {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "budget=0, forcing immediately\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_CONCLUDING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->end_matcher.reset();
                    if (ctx->warn_offset > 0 && ctx->remaining > 0) {
                        ctx->state = REASONING_BUDGET_FORCING_MESSAGE;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "UTF-8 complete, now forcing warning message\n");
                    } else {
                        if (ctx->warn_offset > 0) {
                            ctx->state = REASONING_BUDGET_FORCING_END;
                        } else {
                            ctx->state = REASONING_BUDGET_FORCING;
                        }
                        ctx->force_pos = 0;
                        COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                    }
                }
            } else {
                ctx->remaining--;
                int32_t trigger_limit = (ctx->state == REASONING_BUDGET_COUNTING && ctx->warn_offset > 0) ? ctx->warn_offset : 0;
                if (ctx->remaining <= trigger_limit) {
                    if (utf8_complete) {
                        ctx->end_matcher.reset();
                        if (ctx->state == REASONING_BUDGET_COUNTING && ctx->warn_offset > 0) {
                            ctx->state = REASONING_BUDGET_FORCING_MESSAGE;
                            ctx->force_pos = 0;
                            COM_TRC("%s", "warn offset reached, forcing warning message\n");
                        } else {
                            if (ctx->warn_offset > 0) {
                                ctx->state = REASONING_BUDGET_FORCING_END;
                            } else {
                                ctx->state = REASONING_BUDGET_FORCING;
                            }
                            ctx->force_pos = 0;
                            COM_TRC("%s", "budget/concluding exhausted, forcing end sequence\n");
                        }
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "warn/exhausted limit reached, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        case REASONING_BUDGET_FORCING_MESSAGE:
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_message_tokens.size()) {
                ctx->state = REASONING_BUDGET_CONCLUDING;
                ctx->remaining = ctx->warn_offset;
                ctx->end_matcher.reset();
                COM_TRC("forced message complete, entering concluding phase with %d remaining tokens\n", ctx->remaining);
            }
            break;
        case REASONING_BUDGET_FORCING_END:
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->end_matcher.tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "forced end sequence complete, done\n");
            }
            break;
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    if (ctx->warn_offset > 0) {
                        ctx->state = REASONING_BUDGET_FORCING_MESSAGE;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "budget=0, forcing message immediately\n");
                    } else {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "budget=0, forcing immediately\n");
                    }
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    llama_token forced = -1;

    if (ctx->state == REASONING_BUDGET_FORCING) {
        if (ctx->force_pos < ctx->forced_tokens.size()) {
            forced = ctx->forced_tokens[ctx->force_pos];
        }
    } else if (ctx->state == REASONING_BUDGET_FORCING_MESSAGE) {
        if (ctx->force_pos < ctx->forced_message_tokens.size()) {
            forced = ctx->forced_message_tokens[ctx->force_pos];
        }
    } else if (ctx->state == REASONING_BUDGET_FORCING_END) {
        if (ctx->force_pos < ctx->end_matcher.tokens.size()) {
            forced = ctx->end_matcher.tokens[ctx->force_pos];
        }
    } else {
        // passthrough — don't modify logits
        return;
    }

    if (forced == -1) {
        return;
    }

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens, const std::vector<llama_token> & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state,
        int32_t warn_offset, const std::vector<llama_token> & forced_message_tokens);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab             * vocab,
        const std::vector<llama_token>       & start_tokens,
        const std::vector<llama_token>       & end_tokens,
        const std::vector<llama_token>       & forced_tokens,
        int32_t                                budget,
        common_reasoning_budget_state          initial_state,
        int32_t                                warn_offset,
        const std::vector<llama_token>       & forced_message_tokens) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        if (warn_offset > 0) {
            initial_state = REASONING_BUDGET_FORCING_MESSAGE;
        } else {
            initial_state = REASONING_BUDGET_FORCING;
        }
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab                 = */ vocab,
            /* .start_matcher         = */ { start_tokens, 0 },
            /* .end_matcher           = */ { end_tokens, 0 },
            /* .forced_tokens         = */ forced_tokens,
            /* .forced_message_tokens = */ forced_message_tokens,
            /* .budget                = */ budget,
            /* .remaining             = */ budget,
            /* .warn_offset           = */ warn_offset,
            /* .state                 = */ initial_state,
            /* .force_pos             = */ 0,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state,
        int32_t                          warn_offset,
        const std::vector<llama_token> & forced_message_tokens) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, budget, initial_state, warn_offset, forced_message_tokens);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING && ctx->state != REASONING_BUDGET_CONCLUDING) {
        return false;
    }

    if (ctx->warn_offset > 0) {
        ctx->state = REASONING_BUDGET_FORCING_END;
    } else {
        ctx->state = REASONING_BUDGET_FORCING;
    }
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
