#include "reasoning-budget.h"

#include "llama.h"
#include "log.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Check if a string ends with an incomplete UTF-8 multi-byte sequence.
// Returns true if cutting after this string would split a multi-byte character.
static bool common_utf8_is_incomplete(const std::string & s) {
    if (s.empty()) {
        return false;
    }

    // Scan backwards to count trailing continuation bytes (10xxxxxx)
    int i = (int)s.size() - 1;
    int n_cont = 0;
    while (i >= 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) {
        n_cont++;
        i--;
    }

    if (i < 0) {
        // Only continuation bytes, no leading byte — malformed
        return true;
    }

    const unsigned char lead = static_cast<unsigned char>(s[i]);

    if ((lead & 0x80) == 0x00) {
        // ASCII byte — complete on its own, trailing continuations would be malformed
        return n_cont > 0;
    }

    // Determine expected continuation bytes from leading byte
    int expected;
    if      ((lead & 0xE0) == 0xC0) { expected = 1; }  // 110xxxxx: 2-byte
    else if ((lead & 0xF0) == 0xE0) { expected = 2; }  // 1110xxxx: 3-byte
    else if ((lead & 0xF8) == 0xF0) { expected = 3; }  // 11110xxx: 4-byte
    else { return true; }                               // invalid leading byte

    return n_cont < expected;
}

// Helper to convert a token to a string piece using the public API
static std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token) {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) {
        // buffer too small (shouldn't happen for single tokens), return empty
        return {};
    }
    return std::string(buf, n);
}

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    std::vector<llama_token> start_tokens;   // sequence that starts counting (e.g. "<think>")
    std::vector<llama_token> end_tokens;     // sequence that deactivates naturally (e.g. "</think>")
    std::vector<llama_token> forced_tokens;  // sequence forced when budget expires (e.g. "(budget exceeded)</think>")

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for multi-token sequence matching
    size_t start_match_pos;   // how many tokens of start_tokens we've matched so far
    size_t end_match_pos;     // how many tokens of end_tokens we've matched so far

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
            // watch for start sequence
            if (!ctx->start_tokens.empty() && token == ctx->start_tokens[ctx->start_match_pos]) {
                ctx->start_match_pos++;
                if (ctx->start_match_pos >= ctx->start_tokens.size()) {
                    // full start sequence matched
                    ctx->state = REASONING_BUDGET_COUNTING;
                    ctx->remaining = ctx->budget;
                    ctx->start_match_pos = 0;
                    LOG_INF("reasoning-budget: activated, budget=%d tokens\n", ctx->budget);

                    if (ctx->remaining <= 0) {
                        // budget is 0 — go straight to forcing
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        LOG_INF("reasoning-budget: budget=0, forcing immediately\n");
                    }
                }
            } else {
                ctx->start_match_pos = 0;
                // check if current token starts a new match
                if (!ctx->start_tokens.empty() && token == ctx->start_tokens[0]) {
                    ctx->start_match_pos = 1;
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            // check for natural end sequence (deactivate)
            if (!ctx->end_tokens.empty() && token == ctx->end_tokens[ctx->end_match_pos]) {
                ctx->end_match_pos++;
                if (ctx->end_match_pos >= ctx->end_tokens.size()) {
                    // natural end — stop constraining
                    ctx->state = REASONING_BUDGET_DONE;
                    ctx->end_match_pos = 0;
                    LOG_INF("reasoning-budget: deactivated (natural end)\n");
                }
            } else {
                ctx->end_match_pos = 0;
                if (!ctx->end_tokens.empty() && token == ctx->end_tokens[0]) {
                    ctx->end_match_pos = 1;
                }
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                // Check if the token completes the UTF-8 sequence
                bool still_incomplete = false;  // default: assume complete (safe fallback for null vocab)
                if (ctx->vocab != nullptr) {
                    const std::string piece = common_token_to_piece(ctx->vocab, token);
                    still_incomplete = common_utf8_is_incomplete(piece);
                }

                if (!still_incomplete) {
                    // UTF-8 sequence complete, now start forcing
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_match_pos = 0;
                    LOG_INF("reasoning-budget: UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    // Budget exhausted — check if we need to wait for UTF-8 completion
                    bool wait_for_utf8 = false;
                    if (ctx->vocab != nullptr) {
                        const std::string piece = common_token_to_piece(ctx->vocab, token);
                        wait_for_utf8 = common_utf8_is_incomplete(piece);
                    }

                    if (wait_for_utf8) {
                        // Incomplete UTF-8 sequence, wait for completion
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->force_pos = 0;
                        ctx->end_match_pos = 0;
                        LOG_INF("reasoning-budget: budget exhausted, waiting for UTF-8 completion\n");
                    } else {
                        // Complete UTF-8, go straight to forcing
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_match_pos = 0;
                        LOG_INF("reasoning-budget: budget exhausted, forcing end sequence\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // force_pos is advanced in apply(), not here
            // This ensures the first forced token isn't skipped when the sampler
            // is initialized directly in FORCING state (e.g. activate_immediately + budget=0)
            break;
        }
        case REASONING_BUDGET_DONE:
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }

    // advance to next forced token (done here rather than in accept so that
    // the first forced token isn't skipped when starting in FORCING state)
    ctx->force_pos++;
    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        ctx->state = REASONING_BUDGET_DONE;
        LOG_INF("reasoning-budget: forced sequence complete, done\n");
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_match_pos = 0;
    ctx->end_match_pos = 0;
    ctx->force_pos = 0;
}

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    return common_reasoning_budget_init(
        ctx->vocab,
        ctx->start_tokens,
        ctx->end_tokens,
        ctx->forced_tokens,
        ctx->budget,
        ctx->state == REASONING_BUDGET_COUNTING || ctx->state == REASONING_BUDGET_FORCING || ctx->state == REASONING_BUDGET_WAITING_UTF8);
}

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

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        bool                             activate_immediately) {
    auto initial_state = activate_immediately ? REASONING_BUDGET_COUNTING : REASONING_BUDGET_IDLE;

    // if activated immediately with budget <= 0, go straight to forcing
    if (activate_immediately && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab           = */ vocab,
            /* .start_tokens    = */ start_tokens,
            /* .end_tokens      = */ end_tokens,
            /* .forced_tokens   = */ forced_tokens,
            /* .budget          = */ budget,
            /* .remaining       = */ budget,
            /* .state           = */ initial_state,
            /* .start_match_pos = */ 0,
            /* .end_match_pos   = */ 0,
            /* .force_pos       = */ 0,
        }
    );
}
