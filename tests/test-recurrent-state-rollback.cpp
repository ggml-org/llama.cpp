#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

//
// deep-rollback cases (DSV4-architecture fixtures only)
//
// The per-token rollback tier above covers depths up to n_rs_seq. These cases
// pin the contract for depths beyond it and for the rollback lifecycle:
//   a) a 128-aligned rollback deeper than n_rs_seq replays reference-equal
//   b) a rollback the raw SWA window cannot cover is refused cleanly or
//      falls back checkpoint-equivalent - never silently wrong output
//   c) a second seq_rm before any decode is rejected or cumulative-correct
//   d) a failed decode after a rollback is armed does not lose or corrupt the
//      pending rollback state - the next decode replays reference-equal
//

using pos_logits = std::map<llama_pos, std::vector<float>>;

static llama_context * make_deep_ctx(
        const common_params & params, llama_model * model, uint32_t n_rs_seq, uint32_t n_ubatch, uint32_t n_ctx) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = std::max(n_ctx, n_rs_seq + 1);
    cparams.n_ubatch  = std::max(n_ubatch, n_rs_seq + 1);
    return llama_init_from_model(model, cparams);
}

static bool decode_singles(
        llama_context * ctx, const std::vector<llama_token> & tokens,
        llama_pos p0, llama_pos p1, pos_logits * logits_out, uint32_t n_vocab) {
    for (llama_pos pos = p0; pos < p1; ++pos) {
        if (!decode_one(ctx, tokens[pos], pos)) {
            return false;
        }
        if (logits_out) {
            const float * logits = llama_get_logits_ith(ctx, 0);
            if (logits == nullptr) {
                return false;
            }
            (*logits_out)[pos].assign(logits, logits + n_vocab);
        }
    }
    return true;
}

// every replayed position must match its reference row
static bool logits_equal(const pos_logits & ref, const pos_logits & got, const char * label) {
    constexpr float eps = 1e-5f;
    for (const auto & [pos, got_row] : got) {
        const auto it = ref.find(pos);
        if (it == ref.end()) {
            fprintf(stderr, "%s: no reference logits at position %d\n", label, pos);
            return false;
        }
        for (size_t tok = 0; tok < got_row.size(); ++tok) {
            if (std::fabs(it->second[tok] - got_row[tok]) > eps) {
                fprintf(stderr, "%s: logits mismatch at position %d, token %zu (%g != %g)\n",
                        label, pos, tok, (double) it->second[tok], (double) got_row[tok]);
                return false;
            }
        }
    }
    return !got.empty();
}

// reference logits: an untouched context decoding the same stream the same way
static bool build_reference(
        const common_params & params, llama_model * model, const std::vector<llama_token> & tokens,
        uint32_t n_rs_seq, uint32_t n_ubatch, uint32_t n_ctx,
        llama_pos batch_end, llama_pos ref_from, llama_pos ref_to, pos_logits & out, uint32_t n_vocab) {
    llama_context * ctx = make_deep_ctx(params, model, n_rs_seq, n_ubatch, n_ctx);
    if (ctx == nullptr) {
        return false;
    }
    bool ok = decode_tokens(ctx, tokens, batch_end);
    ok = ok && decode_singles(ctx, tokens, batch_end, ref_from, nullptr, n_vocab);
    ok = ok && decode_singles(ctx, tokens, ref_from, ref_to, &out, n_vocab);
    llama_free(ctx);
    return ok;
}

// a) aligned deep rollback: from position 306 back to the 128-aligned 256,
//    depth 50 > n_rs_seq, then replay must be reference-equal
static bool case_aligned_deep_rollback(
        const common_params & params, llama_model * model, const std::vector<llama_token> & tokens, uint32_t n_vocab) {
    const char * label = "case_aligned_deep_rollback";

    pos_logits ref;
    if (!build_reference(params, model, tokens, 8, 512, 1024, 256, 256, 300, ref, n_vocab)) {
        fprintf(stderr, "%s: failed to build reference\n", label);
        return false;
    }

    llama_context * ctx = make_deep_ctx(params, model, 8, 512, 1024);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", label);
        return false;
    }

    bool ok = decode_tokens(ctx, tokens, 256) && decode_singles(ctx, tokens, 256, 306, nullptr, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: initial decode failed\n", label);
        llama_free(ctx);
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 256, -1)) {
        fprintf(stderr, "%s: aligned deep rollback to position 256 (depth 50) was refused\n", label);
        llama_free(ctx);
        return false;
    }

    pos_logits got;
    ok = decode_singles(ctx, tokens, 256, 300, &got, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: replay decode failed\n", label);
        llama_free(ctx);
        return false;
    }

    ok = logits_equal(ref, got, label);
    llama_free(ctx);
    return ok;
}

// b) uncovered depth: stacked rollbacks walk the sequence back beyond what the
//    raw SWA window still holds; the memory must refuse at some point or
//    replay reference-equal - silently wrong output fails the case
static bool case_uncovered_depth(
        const common_params & params, llama_model * model, const std::vector<llama_token> & tokens, uint32_t n_vocab) {
    const char * label = "case_uncovered_depth";

    pos_logits ref;
    if (!build_reference(params, model, tokens, 8, 16, 1024, 240, 263, 279, ref, n_vocab)) {
        fprintf(stderr, "%s: failed to build reference\n", label);
        return false;
    }

    llama_context * ctx = make_deep_ctx(params, model, 8, 16, 1024);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", label);
        return false;
    }

    bool ok = decode_tokens(ctx, tokens, 240) && decode_singles(ctx, tokens, 240, 900, nullptr, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: initial decode failed\n", label);
        llama_free(ctx);
        return false;
    }

    // stack depth-7 rollbacks from 899 down to 263: the cumulative depth walks
    // far behind anything the raw window ring still holds
    bool refused = false;
    for (llama_pos target = 893; target >= 263; target -= 7) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, target, -1)) {
            refused = true;
            break;
        }
    }

    if (refused) {
        fprintf(stderr, "%s: uncovered rollback cleanly refused\n", label);
        llama_free(ctx);
        return true;
    }

    pos_logits got;
    ok = decode_singles(ctx, tokens, 263, 279, &got, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: replay decode failed after unrefused uncovered rollback\n", label);
        llama_free(ctx);
        return false;
    }


    ok = logits_equal(ref, got, label);
    if (!ok) {
        fprintf(stderr, "%s: uncovered rollback was silently applied with wrong output\n", label);
    }
    llama_free(ctx);
    return ok;
}

// c) stacked rollback: a second seq_rm before any decode must be rejected or
//    behave cumulative-correct (replay from the deeper target reference-equal)
static bool case_stacked_rollback(
        const common_params & params, llama_model * model, const std::vector<llama_token> & tokens, uint32_t n_vocab) {
    const char * label = "case_stacked_rollback";

    // low positions on purpose: with only a handful of compressed blocks in
    // view, per-position state rows corrupted by the stacked rollback dominate
    // the attention output instead of vanishing into a long context
    pos_logits ref;
    if (!build_reference(params, model, tokens, 8, 512, 1024, 4, 10, 20, ref, n_vocab)) {
        fprintf(stderr, "%s: failed to build reference\n", label);
        return false;
    }

    llama_context * ctx = make_deep_ctx(params, model, 8, 512, 1024);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", label);
        return false;
    }

    bool ok = decode_tokens(ctx, tokens, 4) && decode_singles(ctx, tokens, 4, 20, nullptr, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: initial decode failed\n", label);
        llama_free(ctx);
        return false;
    }

    // two stacked rollbacks with cumulative depth 10 > n_rs_seq, both targets
    // mid-block so partial compressor state is live at each restore point
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 14, -1)) {
        fprintf(stderr, "%s: first rollback (depth 6) was refused\n", label);
        llama_free(ctx);
        return false;
    }

    const bool second = llama_memory_seq_rm(llama_get_memory(ctx), 0, 10, -1);
    if (!second) {
        fprintf(stderr, "%s: stacked rollback cleanly rejected\n", label);
        // the pending first rollback must still replay correctly from 14
        pos_logits got;
        ok = decode_singles(ctx, tokens, 14, 20, &got, n_vocab);
        ok = ok && logits_equal(ref, got, label);
        llama_free(ctx);
        return ok;
    }

    pos_logits got;
    ok = decode_singles(ctx, tokens, 10, 20, &got, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: replay decode failed after stacked rollback\n", label);
        llama_free(ctx);
        return false;
    }

    ok = logits_equal(ref, got, label);
    if (!ok) {
        fprintf(stderr, "%s: stacked rollback accepted but replay is not cumulative-correct\n", label);
    }
    llama_free(ctx);
    return ok;
}

// d) failed decode with a deep rollback armed: the failure must not lose or
//    corrupt the pending rollback - the subsequent replay must be
//    reference-equal
static bool case_failed_decode_lifecycle(
        const common_params & params, llama_model * model, const std::vector<llama_token> & tokens, uint32_t n_vocab) {
    const char * label = "case_failed_decode_lifecycle";

    pos_logits ref;
    if (!build_reference(params, model, tokens, 8, 512, 1024, 256, 256, 300, ref, n_vocab)) {
        fprintf(stderr, "%s: failed to build reference\n", label);
        return false;
    }

    llama_context * ctx = make_deep_ctx(params, model, 8, 512, 1024);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", label);
        return false;
    }

    bool ok = decode_tokens(ctx, tokens, 256) && decode_singles(ctx, tokens, 256, 306, nullptr, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: initial decode failed\n", label);
        llama_free(ctx);
        return false;
    }

    // arm a deep aligned rollback, then fail a decode before the replay
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 256, -1)) {
        fprintf(stderr, "%s: deep rollback to position 256 (depth 50) was refused\n", label);
        llama_free(ctx);
        return false;
    }

    // a decode that must fail: position far beyond n_ctx
    if (decode_one(ctx, tokens[0], 4000)) {
        fprintf(stderr, "%s: decode at position 4000 unexpectedly succeeded, cannot exercise the hazard\n", label);
        llama_free(ctx);
        return false;
    }

    pos_logits got;
    ok = decode_singles(ctx, tokens, 256, 300, &got, n_vocab);
    if (!ok) {
        fprintf(stderr, "%s: replay decode failed after the failed decode\n", label);
        llama_free(ctx);
        return false;
    }

    ok = logits_equal(ref, got, label);
    if (!ok) {
        fprintf(stderr, "%s: pending rollback state corrupted by the failed decode\n", label);
    }
    llama_free(ctx);
    return ok;
}

static int run_deep_rollback_cases(const common_params & params, llama_model * model, uint32_t n_vocab) {
    char arch[64] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 ||
            strcmp(arch, "deepseek4") != 0) {
        fprintf(stderr, "%s : skipping deep-rollback cases for non-DSV4 arch\n", __func__);
        return 0;
    }

    std::vector<llama_token> tokens;
    tokens.reserve(1024);
    for (uint32_t i = 0; i < 1024; ++i) {
        tokens.push_back(1 + (llama_token) ((i*7) % (n_vocab - 1)));
    }

    struct named_case {
        const char * name;
        bool (*fn)(const common_params &, llama_model *, const std::vector<llama_token> &, uint32_t);
    };
    const named_case cases[] = {
        { "aligned-deep-rollback",   case_aligned_deep_rollback },
        { "stacked-rollback",        case_stacked_rollback },
        { "failed-decode-lifecycle", case_failed_decode_lifecycle },
        { "uncovered-depth",         case_uncovered_depth },
    };

    int n_failed = 0;
    for (const auto & c : cases) {
        const bool ok = c.fn(params, model, tokens, n_vocab);
        fprintf(stderr, "%s : deep-rollback case %-24s : %s\n", __func__, c.name, ok ? "PASS" : "FAIL");
        if (!ok) {
            n_failed++;
        }
    }
    return n_failed;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model);
    llama_context * ctx_dst = make_ctx(params, model);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(logits_src[token] - logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (std::fabs(logits_src_replay[i][token] - logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    const int n_deep_failed = run_deep_rollback_cases(params, model, n_vocab);
    if (n_deep_failed > 0) {
        fprintf(stderr, "%s : %d deep-rollback case(s) failed\n", __func__, n_deep_failed);
        return 1;
    }
    return 0;
}
