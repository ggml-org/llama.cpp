#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

// n_ubatch == 0 -> size it to fit the whole prompt in one ubatch (single-ubatch reference);
// a small explicit value forces the prompt to span ubatches so build_rs_rollback_carry runs.
static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_ubatch = 0) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    if (n_ubatch == 0) {
        cparams.n_ubatch = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    } else {
        cparams.n_ubatch = n_ubatch;
    }
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, false);
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

    std::vector<llama_token> tokens = common_tokenize(ctx_src, "The quick brown fox jumps", true);
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    if (tokens.size() > n_rs_seq + 1) {
        tokens.resize(n_rs_seq + 1);
    }
    if (tokens.size() < 2) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    const uint32_t    n_tokens = tokens.size();
    const llama_token last_tok = tokens.back();
    const llama_pos   last_pos = (llama_pos) n_tokens - 2;

    // Decode the full prompt on the source, then roll back the last position.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, last_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    // Replay the rolled-back token on both contexts and compare logits.
    if (!decode_one(ctx_src, last_tok, last_pos) ||
        !decode_one(ctx_dst, last_tok, last_pos)) {
        fprintf(stderr, "%s : replay failed\n", __func__);
        return 1;
    }

    const float * logits_src = llama_get_logits_ith(ctx_src, 0);
    const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
    if (logits_src == nullptr || logits_dst == nullptr) {
        fprintf(stderr, "%s : missing logits\n", __func__);
        return 1;
    }

    constexpr float eps = 1e-5f;
    for (int i = 0; i < n_vocab; ++i) {
        if (std::fabs(logits_src[i] - logits_dst[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %d (%g != %g)\n",
                    __func__, i, (double) logits_src[i], (double) logits_dst[i]);
            return 1;
        }
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a *different* prompt's history, and rs_idx[0] is
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
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, last_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    if (!decode_one(ctx_dirty, last_tok, last_pos)) {
        fprintf(stderr, "%s : dirty replay failed\n", __func__);
        return 1;
    }

    const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
    if (logits_dirty == nullptr) {
        fprintf(stderr, "%s : missing dirty logits\n", __func__);
        return 1;
    }

    for (int i = 0; i < n_vocab; ++i) {
        if (std::fabs(logits_src[i] - logits_dirty[i]) > eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at token %d (%g != %g)\n",
                    __func__, i, (double) logits_src[i], (double) logits_dirty[i]);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);

    // capture the single-seq reference logits before freeing ctx_src
    std::vector<float> ref_logits(logits_src, logits_src + n_vocab);

    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    // ---- Multi-sequence parallel rollback ----
    // decode two identical sequences together; rollback must not mix per-seq state. split_equal runs
    // one graph pass for both seqs, exercising the batched snapshot writers and cross-seq carry.
    {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 2;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (2 * n_tokens));
        cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (2 * n_tokens));
        llama_context * ctx_multi = llama_init_from_model(model, cparams);
        if (ctx_multi == nullptr) {
            fprintf(stderr, "%s : failed to init multi-seq ctx\n", __func__);
            return 1;
        }

        // decode the identical prompt on seq 0 and seq 1 in a single batched decode
        llama_batch batch = llama_batch_init(2 * n_tokens, 0, 1);
        for (uint32_t pos = 0; pos < n_tokens; ++pos) {
            common_batch_add(batch, tokens[pos], pos, { 0 }, false);
        }
        for (uint32_t pos = 0; pos < n_tokens; ++pos) {
            common_batch_add(batch, tokens[pos], pos, { 1 }, false);
        }
        const bool decoded = llama_decode(ctx_multi, batch) == 0;
        llama_batch_free(batch);
        if (!decoded) {
            fprintf(stderr, "%s : multi-seq prompt decode failed\n", __func__);
            return 1;
        }

        // roll back the last position on both sequences (rs_idx != 0 on both)
        if (!llama_memory_seq_rm(llama_get_memory(ctx_multi), 0, last_pos, -1) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_multi), 1, last_pos, -1)) {
            fprintf(stderr, "%s : multi-seq rollback failed\n", __func__);
            return 1;
        }

        // replay the rolled-back token on both sequences in a single batched decode
        llama_batch rbatch = llama_batch_init(2, 0, 1);
        common_batch_add(rbatch, last_tok, last_pos, { 0 }, true);
        common_batch_add(rbatch, last_tok, last_pos, { 1 }, true);
        const bool replayed = llama_decode(ctx_multi, rbatch) == 0;
        const float * logits_s0 = replayed ? llama_get_logits_ith(ctx_multi, 0) : nullptr;
        const float * logits_s1 = replayed ? llama_get_logits_ith(ctx_multi, 1) : nullptr;
        if (logits_s0 == nullptr || logits_s1 == nullptr) {
            fprintf(stderr, "%s : multi-seq replay/logits failed\n", __func__);
            llama_batch_free(rbatch);
            llama_free(ctx_multi);
            return 1;
        }

        // cross-sequence isolation: identical inputs must yield byte-identical logits
        for (int i = 0; i < n_vocab; ++i) {
            if (std::fabs(logits_s0[i] - logits_s1[i]) > eps) {
                fprintf(stderr, "%s : multi-seq cross-sequence mismatch at token %d (%g != %g)\n",
                        __func__, i, (double) logits_s0[i], (double) logits_s1[i]);
                llama_batch_free(rbatch);
                llama_free(ctx_multi);
                return 1;
            }
        }

        // sanity vs the single-seq reference: the greedy argmax must agree (the batched shape can
        // perturb the logits below eps, so compare the selected token rather than every logit)
        int argmax_multi = 0, argmax_ref = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits_s0[i]   > logits_s0[argmax_multi]) argmax_multi = i;
            if (ref_logits[i]  > ref_logits[argmax_ref])  argmax_ref   = i;
        }
        llama_batch_free(rbatch);
        llama_free(ctx_multi);
        if (argmax_multi != argmax_ref) {
            fprintf(stderr, "%s : multi-seq argmax %d != single-seq argmax %d\n",
                    __func__, argmax_multi, argmax_ref);
            return 1;
        }

        fprintf(stderr, "%s : multi-sequence parallel rollback matches single-seq reference\n", __func__);
    }

    // ---- Active-gap rollback carry alias regression ----
    // co-decode the consecutive pair {1,2} while seq3 idles between them physically: find_slot gathers
    // seq2 by swapping cell metadata only (tensor rows stay put), so seq2.src0 aliases the extra-cell
    // write row. rolling seq2 back then replaying must match a clean, no-gap reference. the prefixes are
    // decoded in arrival order 1,3,2,0 so physical cells (seq1->0, seq3->1, seq2->2) differ from seq id
    // order; split_equal only co-batches consecutive ids, so the gathered pair must be consecutive.
    {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 4;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (4 * n_tokens));
        cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (4 * n_tokens));

        llama_context * ctx_gap   = llama_init_from_model(model, cparams);
        llama_context * ctx_clean = llama_init_from_model(model, cparams);
        if (ctx_gap == nullptr || ctx_clean == nullptr) {
            fprintf(stderr, "%s : failed to init active-gap contexts\n", __func__);
            return 1;
        }

        auto shifted = [&](int delta) {
            std::vector<llama_token> out = tokens;
            for (auto & t : out) {
                t = (llama_token) (((int64_t) t + delta) % n_vocab);
                if (t < 0) {
                    t = 0;
                }
            }
            return out;
        };

        const std::vector<llama_token> seq0 = shifted(0);
        const std::vector<llama_token> seq1 = shifted(1);
        const std::vector<llama_token> seq2 = shifted(2);
        const std::vector<llama_token> seq3 = shifted(3);
        const std::vector<llama_token> * seqs[4] = { &seq0, &seq1, &seq2, &seq3 };

        const uint32_t prefix = std::max<uint32_t>(1, n_tokens - 1);
        const llama_pos gap_pos = (llama_pos) prefix;

        // decode 4 prefixes in the given arrival order so physical cells fill in that order
        auto decode_prefix_4 = [&](llama_context * ctx, const llama_seq_id order[4]) {
            llama_batch batch = llama_batch_init(4 * prefix, 0, 1);
            for (uint32_t k = 0; k < 4; ++k) {
                const llama_seq_id sid = order[k];
                for (uint32_t pos = 0; pos < prefix; ++pos) {
                    common_batch_add(batch, (*seqs[sid])[pos], pos, { sid }, false);
                }
            }
            const bool ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
            return ok;
        };

        auto decode_pair = [&](llama_context * ctx,
                               llama_seq_id s0, llama_token t0,
                               llama_seq_id s1, llama_token t1) {
            llama_batch batch = llama_batch_init(2, 0, 1);
            common_batch_add(batch, t0, gap_pos, { s0 }, false);
            common_batch_add(batch, t1, gap_pos, { s1 }, false);
            const bool ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
            return ok;
        };

        auto rollback_and_replay_seq2 = [&](llama_context * ctx) -> const float * {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 2, gap_pos, -1)) {
                return nullptr;
            }
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, seq2[prefix], gap_pos, { 2 }, true);
            const bool ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
            return ok ? llama_get_logits_ith(ctx, 0) : nullptr;
        };

        // gap: arrival order 1,3,2,0 -> seq1@cell0, seq3@cell1, seq2@cell2; {1,2} brackets idle seq3
        const llama_seq_id gap_order[4]   = { 1, 3, 2, 0 };
        // clean: natural order -> seq1 and seq2 land on adjacent cells, no gather swap, no extra cell
        const llama_seq_id clean_order[4] = { 0, 1, 2, 3 };
        if (!decode_prefix_4(ctx_gap, gap_order) || !decode_prefix_4(ctx_clean, clean_order)) {
            fprintf(stderr, "%s : active-gap prefix decode failed\n", __func__);
            return 1;
        }

        if (!decode_pair(ctx_gap, 1, seq1[prefix], 2, seq2[prefix])) {
            fprintf(stderr, "%s : active-gap pair decode failed\n", __func__);
            return 1;
        }
        if (!decode_pair(ctx_clean, 1, seq1[prefix], 2, seq2[prefix])) {
            fprintf(stderr, "%s : clean pair decode failed\n", __func__);
            return 1;
        }

        const float * logits_gap   = rollback_and_replay_seq2(ctx_gap);
        const float * logits_clean = rollback_and_replay_seq2(ctx_clean);
        if (logits_gap == nullptr || logits_clean == nullptr) {
            fprintf(stderr, "%s : active-gap rollback/replay failed\n", __func__);
            return 1;
        }

        for (int i = 0; i < n_vocab; ++i) {
            if (std::fabs(logits_gap[i] - logits_clean[i]) > eps) {
                fprintf(stderr, "%s : active-gap alias mismatch at token %d (%g != %g)\n",
                        __func__, i, (double) logits_gap[i], (double) logits_clean[i]);
                llama_free(ctx_gap);
                llama_free(ctx_clean);
                return 1;
            }
        }

        llama_free(ctx_gap);
        llama_free(ctx_clean);

        fprintf(stderr, "%s : active-gap rollback carry is isolated\n", __func__);
    }

    // ---- Cross-ubatch rollback carry ----
    // a small n_ubatch forces the L = n_rs_seq+1 prompt to span >= 2 ubatches, so build_rs_rollback_carry
    // runs across an ubatch boundary. rolling back and replaying must match the single-ubatch reference.
    // a second back-to-back rollback+replay on the same ctx then guards prepare()'s dry-run from
    // double-advancing the rollback depth (rs_idx > 0 on the second pass).
    {
        llama_context * ctx_small = make_ctx(params, model, /*n_ubatch=*/2);
        if (ctx_small == nullptr) {
            fprintf(stderr, "%s : failed to init small-ubatch ctx\n", __func__);
            return 1;
        }

        auto rollback_and_replay = [&]() -> const float * {
            if (!llama_memory_seq_rm(llama_get_memory(ctx_small), 0, last_pos, -1)) {
                return nullptr;
            }
            return decode_one(ctx_small, last_tok, last_pos)
                ? llama_get_logits_ith(ctx_small, 0) : nullptr;
        };

        if (!decode_tokens(ctx_small, tokens, n_tokens)) {
            fprintf(stderr, "%s : small-ubatch prompt decode failed\n", __func__);
            return 1;
        }

        for (int pass = 0; pass < 2; ++pass) {
            const float * logits_small = rollback_and_replay();
            if (logits_small == nullptr) {
                fprintf(stderr, "%s : small-ubatch rollback/replay failed (pass %d)\n", __func__, pass);
                return 1;
            }
            for (int i = 0; i < n_vocab; ++i) {
                if (std::fabs(logits_small[i] - ref_logits[i]) > eps) {
                    fprintf(stderr, "%s : cross-ubatch logits mismatch at token %d on pass %d (%g != %g)\n",
                            __func__, i, pass, (double) logits_small[i], (double) ref_logits[i]);
                    llama_free(ctx_small);
                    return 1;
                }
            }
        }

        llama_free(ctx_small);

        fprintf(stderr, "%s : cross-ubatch rollback carry matches single-ubatch reference\n", __func__);
    }

    // rollback-depth bookkeeping: assert the bool return of seq_rm, the only observable effect of
    // rs_valid_depth (a rollback within the kept history is accepted, one beyond it refused). the logit
    // sections above can't catch an undercounted depth, since an over-strict refusal still leaves the
    // surviving state correct. a decode entered with a pending rollback runs find_slot twice (prepare's
    // dry-run + apply); if prepare does not restore rs_valid_depth the depth shrinks twice and a later
    // valid rollback is wrongly refused. a prompt longer than n_rs_seq makes the depth, not cell.pos,
    // the binding limit.
    {
        const uint32_t R = n_rs_seq;
        if (R < 4) {
            fprintf(stderr, "%s : skipping rollback-depth section (n_rs_seq=%u too small)\n", __func__, R);
        } else {
            const uint32_t L = R + 2; // longer than R so depth saturates at R while cell.pos > depth

            auto cparams = common_context_params_to_llama(params);
            cparams.n_seq_max = 1;
            cparams.n_rs_seq  = R;
            cparams.n_batch   = std::max(cparams.n_batch,  L + 1);
            cparams.n_ubatch  = std::max(cparams.n_ubatch, L + 1); // single ubatch: depth set in one pass

            llama_context * ctx_depth = llama_init_from_model(model, cparams);
            if (ctx_depth == nullptr) {
                fprintf(stderr, "%s : failed to init rollback-depth ctx\n", __func__);
                return 1;
            }
            llama_memory_t mem = llama_get_memory(ctx_depth);

            // build an L-token prompt by cycling the base tokens (content is irrelevant here; only the
            // recurrent-state bookkeeping is under test, and the asserts are on seq_rm's bool return)
            std::vector<llama_token> longp(L);
            for (uint32_t i = 0; i < L; ++i) {
                longp[i] = tokens[i % n_tokens];
            }

            if (!decode_tokens(ctx_depth, longp, L)) {
                fprintf(stderr, "%s : rollback-depth prompt decode failed\n", __func__);
                llama_free(ctx_depth);
                return 1;
            }
            // after this decode: rs_valid_depth = R, cell.pos = L-1 = R+1, rs_idx = 0

            // 1) partial rollback that sets rs_idx = 2 (m = 2). p0 picks rollback = cell.pos-(p0-1):
            //    cell.pos = R+1, want rollback = 2 -> p0 = cell.pos - 1 = R. index 2 <= depth R: must pass.
            const uint32_t m       = 2;
            const llama_pos cur_pos = (llama_pos) (L - 1); // R+1
            const llama_pos p0_set  = cur_pos - (llama_pos) (m - 1); // -> rollback = m = 2
            if (!llama_memory_seq_rm(mem, 0, p0_set, -1)) {
                fprintf(stderr, "%s : rollback-depth idx-setting rollback unexpectedly refused\n", __func__);
                llama_free(ctx_depth);
                return 1;
            }
            // now rs_idx = 2, cell.pos = p0_set - 1 = R-1

            // decode one fresh token at pos R, entered with rs_idx = 2 > n_written = 1: the step whose
            // prepare/apply set the new depth (R-1 when prepare restores it, R-2 if it double-shrinks).
            const llama_pos decode_pos = p0_set; // == (R-1)+1 == R
            if (!decode_one(ctx_depth, longp[decode_pos % L], decode_pos)) {
                fprintf(stderr, "%s : rollback-depth replay decode failed\n", __func__);
                llama_free(ctx_depth);
                return 1;
            }

            // roll back to index R-1 (p0 = 2, reachable since R-1 <= cell.pos = R): gated purely by the
            // depth, so it succeeds only if the depth wasn't double-shrunk.
            const llama_pos deep_pos = (llama_pos) 2; // rollback == R-1
            const bool deep_ok = llama_memory_seq_rm(mem, 0, deep_pos, -1);
            if (!deep_ok) {
                fprintf(stderr, "%s : valid rollback within kept depth was refused "
                                "(rs_valid_depth not restored across prepare's dry-run)\n", __func__);
                llama_free(ctx_depth);
                return 1;
            }

            // negative edge: a rollback one step past the correct depth (index R) must still be refused,
            // confirming the depth is genuinely bounded at R-1. the accepted rollback above consumed this
            // context's state, so reproduce the edge on a fresh, identically-prepared context.
            llama_free(ctx_depth);

            llama_context * ctx_neg = llama_init_from_model(model, cparams);
            if (ctx_neg == nullptr) {
                fprintf(stderr, "%s : failed to init rollback-depth negative ctx\n", __func__);
                return 1;
            }
            llama_memory_t mem_neg = llama_get_memory(ctx_neg);
            if (!decode_tokens(ctx_neg, longp, L)) {
                fprintf(stderr, "%s : rollback-depth negative prompt decode failed\n", __func__);
                llama_free(ctx_neg);
                return 1;
            }
            if (!llama_memory_seq_rm(mem_neg, 0, p0_set, -1) ||
                !decode_one(ctx_neg, longp[decode_pos % L], decode_pos)) {
                fprintf(stderr, "%s : rollback-depth negative setup failed\n", __func__);
                llama_free(ctx_neg);
                return 1;
            }
            // a rollback to index R (p0 = 1) is beyond the kept history and must be refused either way.
            const bool too_deep_ok = llama_memory_seq_rm(mem_neg, 0, (llama_pos) 1, -1);
            llama_free(ctx_neg);
            if (too_deep_ok) {
                fprintf(stderr, "%s : rollback beyond kept history was wrongly accepted\n", __func__);
                return 1;
            }

            fprintf(stderr, "%s : rollback-depth bookkeeping survives a post-rollback decode "
                            "(deepest valid rollback accepted, over-deep rollback refused)\n", __func__);
        }
    }

    // note: the clear()/seq_keep() depth resets aren't unit-guarded here -- their only seq_rm-observable
    // symptom is a too-large stale depth, but after a wipe cell.pos restarts low and a single-seq seq_rm
    // can't request an index above cell.pos, so the distinguishing window is unreachable via the public
    // API (it would need direct rs_valid_depth inspection).

    return 0;
}
