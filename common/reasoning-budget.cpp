#include "reasoning-budget.h"
#include "common.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
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

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    // soft warning
    std::vector<llama_token> soft_forced_tokens;
    bool    soft_enabled;     // soft_ratio > 0 and soft_forced_tokens non-empty
    int32_t soft_threshold;   // trigger soft warning once remaining <= this
    bool    soft_triggered;   // soft warning already fired for this reasoning block
    size_t  soft_force_pos;   // next position in soft_forced_tokens to force

    // intro announcement
    std::vector<llama_token> intro_forced_tokens;
    size_t  intro_force_pos;  // next position in intro_forced_tokens to force

    // graceful hard stop
    int32_t grace_tokens;         // max tokens to wait for a paragraph boundary once exhausted (<= 0 = disabled)
    int32_t grace_remaining;      // tokens left in the current grace wait
    bool    hard_pending_prev_nl; // whether the previous token in HARD_PENDING ended with a newline
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static bool token_utf8_complete(const common_reasoning_budget_ctx * ctx, llama_token token) {
    if (ctx->vocab == nullptr) {
        return true;
    }
    const std::string piece = common_token_to_piece(ctx->vocab, token, false);
    return common_utf8_is_complete(piece);
}

// Transitions into FORCING/WAITING_UTF8 depending on whether this token completes
// a UTF-8 sequence. Shared by every path that decides "start forcing the hard
// cutoff sequence right now".
static void common_reasoning_budget_begin_forcing(common_reasoning_budget_ctx * ctx, llama_token token) {
    ctx->end_matcher.reset();
    if (token_utf8_complete(ctx, token)) {
        ctx->state = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
    } else {
        ctx->state = REASONING_BUDGET_WAITING_UTF8;
    }
}

// Called when the budget hits zero (from COUNTING or SOFT_PENDING): either waits
// (bounded by grace_tokens) for a paragraph boundary, or forces immediately if no
// grace period is configured.
static void common_reasoning_budget_enter_hard_exhausted(common_reasoning_budget_ctx * ctx, llama_token token) {
    if (ctx->grace_tokens > 0) {
        ctx->state = REASONING_BUDGET_HARD_PENDING;
        ctx->grace_remaining = ctx->grace_tokens;
        ctx->hard_pending_prev_nl = false;
        ctx->end_matcher.reset();
        COM_TRC("budget exhausted, waiting up to %d tokens for a paragraph break\n", ctx->grace_tokens);
        return;
    }

    common_reasoning_budget_begin_forcing(ctx, token);
    COM_TRC("%s", "budget exhausted, forcing end sequence\n");
}

// Called whenever a start tag is (re-)matched, to (re-)activate the reasoning
// block: resets the budget countdown, then routes to the intro message (if
// configured), straight to the hard cutoff (budget <= 0), or normal counting.
static void common_reasoning_budget_activate(common_reasoning_budget_ctx * ctx) {
    ctx->remaining = ctx->budget;
    ctx->soft_triggered = false;

    if (!ctx->intro_forced_tokens.empty()) {
        ctx->state = REASONING_BUDGET_INTRO_FORCING;
        ctx->intro_force_pos = 0;
        COM_TRC("activated, budget=%d tokens, forcing intro message\n", ctx->budget);
    } else if (ctx->remaining <= 0) {
        ctx->state = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
        COM_TRC("%s", "budget=0, forcing immediately\n");
    } else {
        ctx->state = REASONING_BUDGET_COUNTING;
        COM_TRC("activated, budget=%d tokens\n", ctx->budget);
    }
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token)) {
                common_reasoning_budget_activate(ctx);
            }
            break;
        }
        case REASONING_BUDGET_INTRO_FORCING:
            ctx->intro_force_pos++;
            if (ctx->intro_force_pos >= ctx->intro_forced_tokens.size()) {
                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "intro complete, budget=0, forcing immediately\n");
                } else {
                    ctx->state = REASONING_BUDGET_COUNTING;
                    COM_TRC("%s", "intro complete, resuming countdown\n");
                }
            }
            break;
        case REASONING_BUDGET_COUNTING:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            ctx->remaining--;
            if (ctx->remaining <= 0) {
                common_reasoning_budget_enter_hard_exhausted(ctx, token);
                break;
            }

            if (ctx->soft_enabled && !ctx->soft_triggered && ctx->remaining <= ctx->soft_threshold) {
                ctx->state = REASONING_BUDGET_SOFT_PENDING;
                COM_TRC("soft threshold reached, remaining=%d, waiting for newline\n", ctx->remaining);
            }
            break;
        }
        case REASONING_BUDGET_SOFT_PENDING:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            ctx->remaining--;
            if (ctx->remaining <= 0) {
                // hard budget wins: abandon the soft warning, no newline is forced
                COM_TRC("%s", "budget exhausted before newline, soft warning skipped\n");
                common_reasoning_budget_enter_hard_exhausted(ctx, token);
                break;
            }

            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                if (piece.find('\n') != std::string::npos) {
                    ctx->state = REASONING_BUDGET_SOFT_FORCING;
                    ctx->soft_force_pos = 0;
                    ctx->soft_triggered = true;
                    COM_TRC("%s", "newline boundary found, forcing soft warning\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_SOFT_FORCING:
            ctx->soft_force_pos++;
            if (ctx->soft_force_pos >= ctx->soft_forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_COUNTING;
                COM_TRC("%s", "soft warning complete, resuming countdown\n");
            }
            break;
        case REASONING_BUDGET_HARD_PENDING:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            ctx->grace_remaining--;

            const std::string piece = ctx->vocab != nullptr ? common_token_to_piece(ctx->vocab, token, false) : std::string();
            const bool paragraph_boundary = piece.find("\n\n") != std::string::npos ||
                (ctx->hard_pending_prev_nl && !piece.empty() && piece[0] == '\n');
            ctx->hard_pending_prev_nl = !piece.empty() && piece.back() == '\n';

            if (paragraph_boundary) {
                common_reasoning_budget_begin_forcing(ctx, token);
                COM_TRC("%s", "paragraph boundary found, forcing end sequence\n");
            } else if (ctx->grace_remaining <= 0) {
                common_reasoning_budget_begin_forcing(ctx, token);
                COM_TRC("%s", "grace period expired, forcing end sequence\n");
            }
            break;
        }
        case REASONING_BUDGET_WAITING_UTF8:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            if (token_utf8_complete(ctx, token)) {
                common_reasoning_budget_begin_forcing(ctx, token);
                COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
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
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window (including
            // its own intro message, if configured).
            if (ctx->start_matcher.advance(token)) {
                ctx->end_matcher.reset();
                common_reasoning_budget_activate(ctx);
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    llama_token forced;

    if (ctx->state == REASONING_BUDGET_FORCING) {
        if (ctx->force_pos >= ctx->forced_tokens.size()) {
            return;
        }
        forced = ctx->forced_tokens[ctx->force_pos];
    } else if (ctx->state == REASONING_BUDGET_SOFT_FORCING) {
        if (ctx->soft_force_pos >= ctx->soft_forced_tokens.size()) {
            return;
        }
        forced = ctx->soft_forced_tokens[ctx->soft_force_pos];
    } else if (ctx->state == REASONING_BUDGET_INTRO_FORCING) {
        if (ctx->intro_force_pos >= ctx->intro_forced_tokens.size()) {
            return;
        }
        forced = ctx->intro_forced_tokens[ctx->intro_force_pos];
    } else {
        // passthrough — don't modify logits
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
    ctx->soft_triggered = false;
    ctx->soft_force_pos = 0;
    ctx->intro_force_pos = 0;
    ctx->grace_remaining = ctx->grace_tokens;
    ctx->hard_pending_prev_nl = false;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens, const std::vector<llama_token> & forced_tokens,
        const std::vector<llama_token> & soft_forced_tokens, const std::vector<llama_token> & intro_forced_tokens,
        int32_t budget, float soft_ratio, int32_t grace_tokens, common_reasoning_budget_state initial_state);

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
        const std::vector<llama_token>       & soft_forced_tokens,
        const std::vector<llama_token>       & intro_forced_tokens,
        int32_t                                budget,
        float                                   soft_ratio,
        int32_t                                grace_tokens,
        common_reasoning_budget_state          initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    const bool soft_enabled = soft_ratio > 0.0f && !soft_forced_tokens.empty();
    int32_t soft_threshold = 0;
    if (soft_enabled) {
        const float ratio = std::min(soft_ratio, 1.0f);
        soft_threshold = std::max(0, budget - (int32_t) std::ceil(budget * ratio));
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab                = */ vocab,
            /* .start_matcher        = */ { start_tokens, 0 },
            /* .end_matcher          = */ { end_tokens, 0 },
            /* .forced_tokens        = */ forced_tokens,
            /* .budget               = */ budget,
            /* .remaining            = */ budget,
            /* .state                = */ initial_state,
            /* .force_pos            = */ 0,
            /* .soft_forced_tokens   = */ soft_forced_tokens,
            /* .soft_enabled         = */ soft_enabled,
            /* .soft_threshold       = */ soft_threshold,
            /* .soft_triggered       = */ false,
            /* .soft_force_pos       = */ 0,
            /* .intro_forced_tokens  = */ intro_forced_tokens,
            /* .intro_force_pos      = */ 0,
            /* .grace_tokens         = */ grace_tokens,
            /* .grace_remaining      = */ grace_tokens,
            /* .hard_pending_prev_nl = */ false,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        const std::vector<llama_token> & soft_forced_tokens,
        const std::vector<llama_token> & intro_forced_tokens,
        int32_t                          budget,
        float                             soft_ratio,
        int32_t                           grace_tokens,
        common_reasoning_budget_state    initial_state) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, soft_forced_tokens, intro_forced_tokens, budget, soft_ratio, grace_tokens, initial_state);
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

    // only a sampler that is actively counting down the budget (or emitting the
    // intro/soft messages, or waiting out the post-exhaustion grace period) may
    // be forced; any other state (idle, already hard-forcing/waiting, or done)
    // is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING &&
        ctx->state != REASONING_BUDGET_INTRO_FORCING &&
        ctx->state != REASONING_BUDGET_SOFT_PENDING &&
        ctx->state != REASONING_BUDGET_SOFT_FORCING &&
        ctx->state != REASONING_BUDGET_HARD_PENDING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
