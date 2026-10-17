#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // local soft-budget extension
    int32_t    soft_threshold; // consumed-token count at which the soft message may fire; <= 0 disables
    llama_tokens soft_tokens;  // tokenized soft wrap-up message
    bool       soft_fired;     // the soft warning fires at most once per reasoning block
    size_t     soft_force_pos; // next position in soft_tokens to force
    int32_t    grace_total;    // grace tokens allowed past the budget; 0 disables grace
    int32_t    grace_used;     // tokens consumed inside the grace region
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->soft_fired = false;
                ctx->grace_used = 0;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_SOFT_PENDING:
        case REASONING_BUDGET_HARD_PENDING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            // a natural reasoning end always takes precedence over any
            // soft/hard budget intervention
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            std::string piece;
            if (ctx->vocab != nullptr) {
                piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
                break;
            }

            if (ctx->state == REASONING_BUDGET_HARD_PENDING) {
                // bounded grace region: accept a natural close (checked above),
                // close safely at a paragraph boundary, or force once the
                // grace tokens are exhausted; grace never grows unbounded
                ctx->grace_used++;
                const bool paragraph = piece.find("\n\n") != std::string::npos;
                if (utf8_complete && paragraph) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("grace paragraph boundary at token %d, forcing end sequence\n", ctx->grace_used);
                } else if (ctx->grace_used >= ctx->grace_total) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "grace exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "grace exhausted, waiting for UTF-8 completion\n");
                    }
                }
                break;
            }

            // COUNTING / SOFT_PENDING still consume the nominal budget
            ctx->remaining--;
            if (ctx->remaining <= 0) {
                if (ctx->grace_total > 0) {
                    ctx->state = REASONING_BUDGET_HARD_PENDING;
                    ctx->grace_used = 0;
                    COM_TRC("budget exhausted, entering bounded grace region (%d tokens)\n", ctx->grace_total);
                } else if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                } else {
                    ctx->state = REASONING_BUDGET_WAITING_UTF8;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                }
                break;
            }

            // one-shot soft warning once the configured fraction of the
            // budget has been consumed; injected at a line boundary so the
            // message never splits an arbitrary token sequence
            if (ctx->state == REASONING_BUDGET_COUNTING
                    && !ctx->soft_fired
                    && ctx->soft_threshold > 0
                    && (ctx->budget - ctx->remaining) >= ctx->soft_threshold) {
                ctx->soft_fired = true;
                const bool boundary = (ctx->vocab == nullptr)
                    || piece.find('\n') != std::string::npos;
                if (!ctx->soft_tokens.empty() && boundary && utf8_complete) {
                    ctx->state = REASONING_BUDGET_SOFT_FORCING;
                    ctx->soft_force_pos = 0;
                    COM_TRC("%s", "soft threshold reached at a line boundary, injecting wrap-up message\n");
                } else if (!ctx->soft_tokens.empty()) {
                    ctx->state = REASONING_BUDGET_SOFT_PENDING;
                    COM_TRC("%s", "soft threshold reached, waiting for a line boundary\n");
                } else {
                    COM_TRC("%s", "soft threshold reached, no soft message configured\n");
                }
                break;
            }

            if (ctx->state == REASONING_BUDGET_SOFT_PENDING) {
                const bool boundary = (ctx->vocab == nullptr)
                    || piece.find('\n') != std::string::npos;
                if (boundary && utf8_complete) {
                    ctx->state = REASONING_BUDGET_SOFT_FORCING;
                    ctx->soft_force_pos = 0;
                    COM_TRC("%s", "line boundary after soft threshold, injecting wrap-up message\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_SOFT_FORCING:
        {
            // force the soft wrap-up message token-by-token, then resume
            // normal counting; the natural end sequence stays armed
            ctx->soft_force_pos++;
            if (ctx->soft_force_pos >= ctx->soft_tokens.size()) {
                ctx->state = REASONING_BUDGET_COUNTING;
                COM_TRC("%s", "soft message complete, resuming counting\n");
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->soft_fired = false;
                ctx->grace_used = 0;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_SOFT_FORCING) {
        if (ctx->soft_force_pos >= ctx->soft_tokens.size()) {
            return;
        }

        const llama_token forced = ctx->soft_tokens[ctx->soft_force_pos];

        // set all logits to -inf except the forced token
        for (size_t i = 0; i < cur_p->size; i++) {
            if (cur_p->data[i].id != forced) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return;
    }

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
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->soft_fired = false;
    ctx->grace_used = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state,
        float soft_ratio, const llama_tokens & soft_tokens, int32_t grace_tokens);

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
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        float                             soft_ratio,
        const llama_tokens              & soft_tokens,
        int32_t                           grace_tokens) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    // soft guidance only applies to a finite positive budget; invalid ratios
    // disable the feature instead of producing surprising behavior
    int32_t soft_threshold = 0;
    if (budget > 0 && budget != INT_MAX
            && soft_ratio > 0.0f && soft_ratio <= 1.0f) {
        soft_threshold = (int32_t) std::ceil(budget * (double) soft_ratio);
        if (soft_threshold >= budget) {
            soft_threshold = 0; // would never fire before the hard budget
        }
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab          = */ vocab,
            /* .start_matcher  = */ token_matcher(start_seqs),
            /* .end_matcher    = */ token_matcher(end_seqs),
            /* .forced_tokens  = */ forced_tokens,
            /* .budget         = */ budget,
            /* .remaining      = */ budget,
            /* .state          = */ initial_state,
            /* .force_pos      = */ 0,
            /* .end_match      = */ -1,
            /* .soft_threshold = */ soft_threshold,
            /* .soft_tokens    = */ soft_tokens,
            /* .soft_fired     = */ false,
            /* .soft_force_pos = */ 0,
            /* .grace_total    = */ grace_tokens > 0 ? grace_tokens : 0,
            /* .grace_used     = */ 0,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        float                             soft_ratio,
        const llama_tokens              & soft_tokens,
        int32_t                           grace_tokens) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state,
            soft_ratio, soft_tokens, grace_tokens);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
