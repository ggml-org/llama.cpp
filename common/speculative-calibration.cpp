//
// speculative-calibration.cpp — manual drafter-forward + KV rollback
// + per-step telemetry capture for spec-decoding calibration.
//
// HISTORY
// -------
// This code was originally inlined in tools/imatrix/imatrix.cpp as
// `compute_imatrix_spec()`.  It was extracted here in 2026-07 so the
// calibration path can be reused from other tools (server telemetry,
// dspark-realign, etc.) without copy-pasting the 600+ lines of manual
// KV bookkeeping again.  The inlined `compute_imatrix_spec()` wrapper
// in imatrix.cpp was removed on 2026-08-03 (it was dead code after
// this extraction landed); the imatrix binary now calls
// common_speculative_calibration_run() directly.
//
// WHY IT BYPASSES common_speculative_*
// ------------------------------------
// The public common_speculative_draft() / common_speculative_accept() loop
// in common/speculative.cpp has off-by-one KV bugs.  Commit 9e9f275 said
// "bypasses common_speculative_* due to off-by-one KV bugs" and shipped
// the manual forward inline in imatrix.cpp as the workaround.  The
// off-by-one is deep (it involves the spec API's pos = n_past + i + 1
// bookkeeping colliding with the verifier's pos tracking), and fixing
// it risks destabilizing the production speculative-decoding path
// (server / interactive decode) that the public API serves.
//
// The calibration path has different needs: it must capture per-position
// verifier AND drafter top-k distributions (not just accept/reject),
// which forces the n_dft+1 per-prefix forward pattern.  That pattern
// is incompatible with the public spec API's "single batched forward"
// implementation, so even if the off-by-one were fixed in the public
// API, the calibration path would still need its own driver.  Hence
// the separate API.
//
// DO NOT CHANGE THE LOGIC HERE WITHOUT UPDATING THE TEST
// ------------------------------------------------------
// The test in tests/test-spec-calibration.cpp runs a single spec step
// against a known input and checks the JSONL output is byte-identical
// to what the inlined code produced.  Changing the ordering of
// llama_decode calls, the K/V trim boundaries, the JSONL field
// order, or the per-position distribution math will break the test.
// The test pins the *observable* behavior; if the per-step JSONL is
// the same, the refactor is correct.
//

#include "speculative-calibration.h"

#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

//
// Serialize one llama.tessera.spec.v1 JSONL record for a single spec
// step.  See speculative-calibration.h for the contract.
//
// This is the per-step record serialization moved out of
// common_speculative_calibration_run(); the additions are the
// provenance / sid trailing fields (emitted only when non-NULL) so the
// calibration output stays byte-identical, and the off-by-one fix
// described in the run() history below (confidence[i] and the bonus
// now use row i / row n_acc instead of row i+1 / row n_dft).
//
std::string common_spec_telemetry_record(
        int32_t step,
        llama_token id_last,
        const llama_tokens & draft,
        size_t n_acc,
        const std::vector<const float *> & v_logits_rows,
        const std::vector<const float *> & dft_logits_rows,
        int32_t n_vocab,
        int32_t topk,
        const char * provenance,
        const char * sid) {

    const int n_dft = (int) draft.size();
    topk = std::min(topk > 0 ? topk : 0, n_vocab);

    // Per-row top-k helper. Builds two parallel vectors
    // (tokens, logit-values) sorted high-to-low.
    auto topk_row = [&](const float * row,
                        int32_t & argmax_out) {
        std::vector<int32_t> tok;
        std::vector<float>   val;
        if (row == nullptr) {
            argmax_out = 0;
            return std::pair{std::move(tok), std::move(val)};
        }
        int32_t am = 0;
        float   am_val = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > am_val) { am_val = row[v]; am = v; }
        }
        argmax_out = am;
        if (topk == 0) {
            return std::pair{std::move(tok), std::move(val)};
        }
        std::vector<std::pair<float, int32_t>> heap;
        heap.reserve(topk);
        for (int v = 0; v < n_vocab; ++v) {
            if ((int) heap.size() < topk) {
                heap.emplace_back(row[v], v);
                std::push_heap(heap.begin(), heap.end(),
                    std::greater<std::pair<float, int32_t>>());
            } else if (row[v] > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(),
                    std::greater<std::pair<float, int32_t>>());
                heap.back() = {row[v], v};
                std::push_heap(heap.begin(), heap.end(),
                    std::greater<std::pair<float, int32_t>>());
            }
        }
        tok.reserve(topk);
        val.reserve(topk);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(),
                std::greater<std::pair<float, int32_t>>());
            tok.push_back(heap.back().second);
            val.push_back(heap.back().first);
            heap.pop_back();
        }
        return std::pair{std::move(tok), std::move(val)};
    };

    // Verifier: n_dft+1 per-position distributions from the per-prefix
    // forward rows.
    std::vector<std::vector<int32_t>> v_topk_tokens(n_dft + 1);
    std::vector<std::vector<float>>   v_topk_probs(n_dft + 1);
    std::vector<int32_t>              v_argmax_explicit(n_dft + 1, 0);
    std::vector<float>                confidence(n_dft, 0.0f);
    for (int i = 0; i <= n_dft; ++i) {
        int32_t am = 0;
        auto tk = topk_row(
            i < (int) v_logits_rows.size() ? v_logits_rows[i] : nullptr,
            am);
        v_topk_tokens[i] = std::move(tk.first);
        v_topk_probs[i]  = std::move(tk.second);
        v_argmax_explicit[i] = am;
    }
    // confidence[] is the softmax prob of draft[i] under the verifier
    // row that judges it: row i is conditioned on prefix +
    // draft[0..i-1] (the same conditioning the accept check uses).
    // Use the precomputed row (avoids re-scanning the vocab).
    for (int i = 0; i < n_dft; ++i) {
        const float * row = (i < (int) v_logits_rows.size())
                            ? v_logits_rows[i] : nullptr;
        if (row == nullptr) continue;
        float max_logit = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > max_logit) max_logit = row[v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            sum_exp += std::exp((double) row[v] - (double) max_logit);
        }
        const double prob = sum_exp > 0.0
            ? std::exp((double) row[draft[i]] - (double) max_logit) / sum_exp
            : 0.0;
        confidence[i] = (float) prob;
    }

    // Drafter: n_dft+1 per-position distributions (priming + drafts).
    std::vector<std::vector<int32_t>> d_topk_tokens(n_dft + 1);
    std::vector<std::vector<float>>   d_topk_probs(n_dft + 1);
    std::vector<int32_t>              d_argmax_explicit(n_dft + 1, 0);
    for (int i = 0; i <= n_dft; ++i) {
        int32_t am = 0;
        auto tk = topk_row(
            i < (int) dft_logits_rows.size() ? dft_logits_rows[i] : nullptr,
            am);
        d_topk_tokens[i] = std::move(tk.first);
        d_topk_probs[i]  = std::move(tk.second);
        d_argmax_explicit[i] = am;
    }

    // accepted_tokens: drafts[0..n_acc-1] + bonus
    // Bonus = verifier argmax of row n_acc, the first row conditioned
    // only on accepted drafts (prefix + draft[0..n_acc-1]). Rows past
    // n_acc also saw rejected drafts, so they must not contribute.
    const llama_token bonus_local = v_argmax_explicit[n_acc];
    std::vector<int32_t> accepted_tokens;
    accepted_tokens.reserve(n_acc + 1);
    for (size_t k = 0; k < n_acc; ++k) {
        accepted_tokens.push_back(draft[k]);
    }
    if (n_acc <= (size_t) n_dft) {
        accepted_tokens.push_back(bonus_local);
    }

    // Hand-build the JSONL record. Single schema
    // (llama.tessera.spec.v1). The cheap per-step fields are
    // always emitted; the top-k fields are added only when
    // topk > 0. We keep arrays parallel (tokens[] and probs[])
    // to keep the encoding compact; the training pipeline
    // re-zips them per position.
    std::string line;
    line  = "{\"schema\":\"llama.tessera.spec.v1\"";
    line += ",\"seq_id\":0";
    line += ",\"step_idx\":" + std::to_string(step);
    line += ",\"prime_token\":" + std::to_string(id_last);
    line += ",\"drafted\":" + std::to_string(n_dft);
    line += ",\"accepted\":" + std::to_string(n_acc);

    // drafted_tokens and accepted_tokens are part of the cheap
    // payload (always emitted) so that downstream consumers
    // (LK training, DFlash dataset prep) don't need topk > 0
    // to recover the draft trajectory.
    line += ",\"drafted_tokens\":[";
    for (int i = 0; i < n_dft; ++i) {
        if (i > 0) line += ",";
        line += std::to_string(draft[i]);
    }
    line += "]";

    line += ",\"accepted_tokens\":[";
    for (size_t i = 0; i < accepted_tokens.size(); ++i) {
        if (i > 0) line += ",";
        line += std::to_string(accepted_tokens[i]);
    }
    line += "]";

    // confidence = verifier softmax prob of each draft token.
    // Always emitted as part of the unified record.
    line += ",\"confidence\":[";
    for (size_t i = 0; i < confidence.size(); ++i) {
        if (i > 0) line += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8g",
                      (double) confidence[i]);
        line += buf;
    }
    line += "]";

    if (topk > 0) {
        // Per-position top-k distributions: argmaxes plus the
        // top-k token/prob arrays for both verifier and drafter.
        line += ",\"topk\":" + std::to_string(topk);

        line += ",\"verifier_argmax\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += std::to_string(v_argmax_explicit[i]);
        }
        line += "]";

        line += ",\"drafter_argmax\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += std::to_string(d_argmax_explicit[i]);
        }
        line += "]";

        // Verifier top-k: parallel arrays.
        line += ",\"verifier_topk_tokens\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < v_topk_tokens[i].size(); ++k) {
                if (k > 0) line += ",";
                line += std::to_string(v_topk_tokens[i][k]);
            }
            line += "]";
        }
        line += "]";
        line += ",\"verifier_topk_probs\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < v_topk_probs[i].size(); ++k) {
                if (k > 0) line += ",";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g",
                              (double) v_topk_probs[i][k]);
                line += buf;
            }
            line += "]";
        }
        line += "]";

        // Drafter top-k: parallel arrays.
        line += ",\"drafter_topk_tokens\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < d_topk_tokens[i].size(); ++k) {
                if (k > 0) line += ",";
                line += std::to_string(d_topk_tokens[i][k]);
            }
            line += "]";
        }
        line += "]";
        line += ",\"drafter_topk_probs\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < d_topk_probs[i].size(); ++k) {
                if (k > 0) line += ",";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g",
                              (double) d_topk_probs[i][k]);
                line += buf;
            }
            line += "]";
        }
        line += "]";
    }

    // Additive runtime fields: present only when the caller stamps
    // them; calibration passes NULL and the record is unchanged.
    if (provenance != nullptr) {
        line += ",\"provenance\":\"";
        line += provenance;
        line += "\"";
    }
    if (sid != nullptr) {
        line += ",\"sid\":\"";
        line += sid;
        line += "\"";
    }

    line += "}\n";
    return line;
}

//
// Run the spec-decoding calibration loop.  See speculative-calibration.h
// for the API contract.  Logic was moved verbatim from the inlined
// `compute_imatrix_spec()` that used to live in
// tools/imatrix/imatrix.cpp (since removed) — the only changes are:
//   - g_collector.begin_graph_observers() / .flush_graph_observers()
//     are replaced with opts.observer_hooks.begin() / .flush().
//   - A final observer_hooks.flush() at the end replaces the
//     g_collector.flush_graph_observers() that the original
//     function called in its tail `return`.
//   - `int` -> `int32_t` for the public-option-derived locals
//     (n_draft_max, n_steps) so the option type and the local type
//     match.  No behavior change.
//   - The per-step record serialization was moved verbatim into
//     common_spec_telemetry_record() below (byte-identical output;
//     pinned by tests/test-telemetry-golden.cpp).
//   - Off-by-one accept fix: the accept loop compared v_argmax[i] to
//     draft[i-1] and took the bonus from row n_dft, while the
//     per-prefix forwards document that row i (conditioned on prefix +
//     draft[0..i-1]) judges draft[i]. The loop now compares v_argmax[i]
//     to draft[i] and takes the bonus from row n_acc, matching the
//     production accept path. The shared emitter's confidence[] and
//     accepted_tokens bonus were shifted the same way and were fixed
//     in lockstep (this also corrects runtime records).
//
bool common_speculative_calibration_run(
    llama_context * ctx_tgt,
    llama_model * model_tgt,
    common_speculative * spec,
    common_params & params,
    const int32_t n_ctx,
    const common_speculative_calibration_options & opts) {

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int32_t n_draft_max = opts.n_draft_max_override > 0
        ? opts.n_draft_max_override
        : common_speculative_n_max(&params.speculative);
    const int32_t n_steps_opt = opts.n_steps_override != 0
        ? opts.n_steps_override
        : params.n_spec_steps;  // 0 means "until context limit"

    auto obs_begin = [&]() {
        if (opts.observer_hooks.begin) {
            opts.observer_hooks.begin(opts.observer_hooks.user_data);
        }
    };
    auto obs_flush = [&]() -> bool {
        if (opts.observer_hooks.flush) {
            return opts.observer_hooks.flush(opts.observer_hooks.user_data);
        }
        return true;
    };

    if (opts.verbosity > 0) {
        LOG_INF("%s: tokenizing the input ..\n", __func__);
    }
    std::vector<llama_token> tokens = common_tokenize(
        ctx_tgt, params.prompt, true, params.parse_special);
    if (opts.verbosity > 0) {
        LOG_INF("%s: tokenized %zu tokens\n", __func__, tokens.size());
    }

    if (int(tokens.size()) < n_ctx + 4) {
        LOG_ERR("%s: need at least %d tokens for spec calibration (have %zu)\n",
                __func__, n_ctx + 4, tokens.size());
        return false;
    }

    // Prime the verifier with most of the context window, leaving a tail
    // for the spec loop to use (1 verifier-token + n_draft_max draft-tokens
    // per step). Without this headroom the slot allocator refuses the
    // first spec batch with "failed to find a memory slot for batch of
    // size 4".
    //
    // We also prime the drafter with the same prompt: DRAFT_SIMPLE's
    // `begin()` is a no-op, so the drafter's KV cache is empty after
    // common_speculative_begin; the drafter needs to see the prompt
    // tokens before it can draft at position n_past.
    //
    // If `opts.telemetry_out` is set, we open a JSONL file and write
    // one record per spec step carrying the drafter's per-position
    // confidence (the verifier's softmax probability of the drafter's
    // pick), the draft / accepted token sequences, and - when
    // telemetry_topk > 0 - the full per-position top-k verifier and
    // drafter distributions. The schema is llama.tessera.spec.v1. This
    // is the input for downstream drafter fine-tuning (LK loss, D-PACE
    // weighted CE, rejection-sampling on dspark to bring it back into
    // alignment with the QAT target - see the dspark-realign pipeline).
    FILE * telemetry_fp = nullptr;
    if (!opts.telemetry_out.empty()) {
        telemetry_fp = std::fopen(opts.telemetry_out.c_str(), "w");
        if (telemetry_fp == nullptr) {
            LOG_ERR("%s: failed to open telemetry output '%s'\n",
                    __func__, opts.telemetry_out.c_str());
            return false;
        }
        if (opts.verbosity > 0) {
            LOG_INF("%s: writing per-step accept/reject telemetry to '%s'\n",
                    __func__, opts.telemetry_out.c_str());
        }
    }

    const int prime_size = std::max(8, n_ctx - n_draft_max - 4);
    std::vector<llama_token> prompt_tokens(
        tokens.begin(), tokens.begin() + prime_size);
    if (llama_decode(ctx_tgt, llama_batch_get_one(
            prompt_tokens.data(), prompt_tokens.size() - 1)) != 0) {
        LOG_ERR("%s: failed to prime verifier\n", __func__);
        if (telemetry_fp) std::fclose(telemetry_fp);
        return false;
    }
    if (params.speculative.draft.ctx_dft != nullptr) {
        // Prime the drafter with the same (N-1)-length prefix. After this,
        // the drafter's KV has positions 0..prime_size-2 filled and
        // n_past_dft == prime_size - 1.
        if (llama_decode(params.speculative.draft.ctx_dft,
                llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size() - 1)) != 0) {
            LOG_ERR("%s: failed to prime drafter\n", __func__);
            if (telemetry_fp) std::fclose(telemetry_fp);
            return false;
        }
    }
    {
        auto * mem_tgt = llama_get_memory(ctx_tgt);
        auto * mem_dft = params.speculative.draft.ctx_dft != nullptr
                              ? llama_get_memory(params.speculative.draft.ctx_dft)
                              : nullptr;
        const llama_pos pos_tgt = llama_memory_seq_pos_max(mem_tgt, 0);
        const llama_pos pos_dft = mem_dft ? llama_memory_seq_pos_max(mem_dft, 0) : -1;
        if (opts.verbosity > 0) {
            LOG_INF("%s: post-prime KV: tgt n_past=%lld, dft n_past=%lld, prime_size=%d\n",
                    __func__, (long long) pos_tgt + 1, (long long) pos_dft + 1, prime_size);
        }
    }

    // Begin the spec context with the prompt. The spec implementation
    // primes the drafter's KV cache with the same prompt here.
    common_speculative_begin(spec, 0, prompt_tokens);

    // Sampler for the verifier (greedy by default for determinism).
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    llama_batch batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    llama_token id_last = prompt_tokens.back();
    int n_past = (int) prompt_tokens.size() - 1;
    int n_drafted = 0;
    int n_accepted = 0;

    if (opts.verbosity > 0) {
        LOG_INF("%s: starting spec loop: n_ctx=%d, n_draft_max=%d, n_steps=%s\n",
                __func__, n_ctx, n_draft_max,
                n_steps_opt > 0 ? std::to_string(n_steps_opt).c_str() : "until-limit");
    }

    int step = 0;
    // Get the drafter context once. We bypass common_speculative_draft()
    // and run the drafter forward manually; the spec API's bookkeeping
    // (pos = n_past + i + 1) collides with our verifier's pos tracking in
    // ways that produce off-by-one KV errors.
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    if (ctx_dft == nullptr) {
        LOG_ERR("%s: drafter context is null\n", __func__);
        llama_batch_free(batch);
        if (telemetry_fp) std::fclose(telemetry_fp);
        return false;
    }
    // Separate sampler for the drafter (greedy). We don't accept its
    // samples; we just need its top-1 token at each step.
    common_sampler_ptr dft_smpl(
        common_sampler_init(model_tgt, params.sampling));

    while (n_steps_opt <= 0 || step < n_steps_opt) {
        // 1. Drafter forward: at position n_past, sample one token.
        //    Save the drafter's logits after each forward so the
        //    telemetry has the full per-position distribution (the
        //    drafter's graph exposes logits only at position 0 of the
        //    last batch, so we need to save it before the next forward
        //    overwrites it). When telemetry_topk > 0 we serialize these
        //    into the drafter_topk_* arrays; when topk == 0 we still
        //    need the verifier's softmax probability of the drafter's
        //    pick (confidence[]) which is computed below from the
        //    verifier's per-position row.
        std::vector<const float *> dft_logits_ptrs;
        std::vector<std::vector<float>> dft_logits_storage;
        dft_logits_ptrs.reserve(n_draft_max + 1);
        dft_logits_storage.reserve(n_draft_max + 1);

        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past, {0}, true);
        if (llama_decode(ctx_dft, batch) != 0) {
            LOG_ERR("%s: drafter forward failed at step %d\n", __func__, step);
            llama_batch_free(batch);
            if (telemetry_fp) std::fclose(telemetry_fp);
            return false;
        }
        // The drafter's logits are now at i_batch=0; sample top-1.
        llama_token dft_id = common_sampler_sample(dft_smpl.get(), ctx_dft, 0);
        common_sampler_accept(dft_smpl.get(), dft_id, true);
        if (telemetry_fp != nullptr) {
            const float * row = llama_get_logits_ith(ctx_dft, 0);
            if (row != nullptr) {
                const int n_vocab = llama_vocab_n_tokens(vocab);
                dft_logits_storage.emplace_back(row, row + n_vocab);
                dft_logits_ptrs.push_back(dft_logits_storage.back().data());
            } else {
                dft_logits_ptrs.push_back(nullptr);
            }
        }

        // Build the draft sequence by stepping the drafter forward
        // n_draft_max times. Each step: add the previous sample at the
        // next position, decode, sample. Save logits at each step.
        std::vector<llama_token> draft;
        draft.reserve(n_draft_max);
        for (int k = 0; k < n_draft_max; k++) {
            draft.push_back(dft_id);
            common_batch_clear(batch);
            common_batch_add(batch, dft_id, n_past + 1 + (llama_pos) k, {0}, true);
            if (llama_decode(ctx_dft, batch) != 0) {
                LOG_WRN("%s: drafter draft-step %d failed; using %d drafts\n",
                        __func__, k, (int) draft.size() - 1);
                draft.pop_back();
                break;
            }
            const llama_token next_id = common_sampler_sample(dft_smpl.get(), ctx_dft, 0);
            common_sampler_accept(dft_smpl.get(), next_id, true);
            dft_id = next_id;
            if (telemetry_fp != nullptr) {
                const float * row = llama_get_logits_ith(ctx_dft, 0);
                if (row != nullptr) {
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    dft_logits_storage.emplace_back(row, row + n_vocab);
                    dft_logits_ptrs.push_back(dft_logits_storage.back().data());
                } else {
                    dft_logits_ptrs.push_back(nullptr);
                }
            }
        }
        n_drafted += (int) draft.size();
        if (draft.empty()) {
            if (opts.verbosity > 0) {
                LOG_INF("%s: drafter produced empty draft at step %d, stopping\n",
                        __func__, step);
            }
            break;
        }

        const int n_dft = (int) draft.size();

        // 2. Verifier: do n_dft+1 per-prefix forwards. Each forward
        //    adds ONE new token (the latest in the prefix) — earlier
        //    tokens are already in the KV from the previous forward.
        //    After the priming, the KV has id_last at n_past. After
        //    forward i, the KV has id_last at n_past and draft[0..i-1]
        //    at n_past+1..n_past+i. The i-th forward's logits at
        //    position 0 are the verifier's prediction for the next
        //    token (i.e., for draft[i] or for the bonus if i==n_dft).
        //
        //    Note: the verifier's causal-attention graph only computes
        //    logits at the last position of a batched forward, so the
        //    old "single batched 4-token forward" path returned 0.0f
        //    confidence for every non-last position. Doing 1-token
        //    per-prefix forwards is the only way to get per-position
        //    verifier distributions without disabling causal attention.
        std::vector<const float *> v_logits_ptrs;
        std::vector<std::vector<float>> v_logits_storage;
        std::vector<int32_t> v_argmax(n_dft + 1, 0);
        if (telemetry_fp != nullptr) {
            v_logits_ptrs.reserve(n_dft + 1);
            v_logits_storage.reserve(n_dft + 1);
        }
        {
            llama_batch ver_batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
            for (int i = 0; i <= n_dft; ++i) {
                common_batch_clear(ver_batch);
                // The first forward (i==0) primes id_last; subsequent
                // forwards add the next draft token at n_past+i.
                if (i == 0) {
                    common_batch_add(ver_batch, id_last,
                                      n_past, {0}, true);
                } else {
                    common_batch_add(ver_batch, draft[i - 1],
                                      n_past + (llama_pos) i, {0}, true);
                }
                if (i == n_dft) {
                    // The "main" forward (last) runs the graph
                    // observers; earlier per-prefix forwards are
                    // observer-silent to keep imatrix attribution
                    // clean (each chunk = one spec step).
                    obs_begin();
                }
                if (llama_decode(ctx_tgt, ver_batch) != 0) {
                    LOG_ERR("%s: verifier per-prefix forward %d failed at step %d\n",
                            __func__, i, step);
                    llama_batch_free(ver_batch);
                    llama_batch_free(batch);
                    if (telemetry_fp) std::fclose(telemetry_fp);
                    return false;
                }
                if (i == n_dft) {
                    if (!obs_flush()) {
                        llama_batch_free(ver_batch);
                        llama_batch_free(batch);
                        if (telemetry_fp) std::fclose(telemetry_fp);
                        return false;
                    }
                }
                if (telemetry_fp != nullptr) {
                    const float * row = llama_get_logits_ith(ctx_tgt, 0);
                    if (row != nullptr) {
                        const int n_vocab = llama_vocab_n_tokens(vocab);
                        int32_t am = 0;
                        float am_val = row[0];
                        for (int v = 1; v < n_vocab; ++v) {
                            if (row[v] > am_val) { am_val = row[v]; am = v; }
                        }
                        v_argmax[i] = am;
                        v_logits_storage.emplace_back(row, row + n_vocab);
                        v_logits_ptrs.push_back(v_logits_storage.back().data());
                    } else {
                        v_logits_ptrs.push_back(nullptr);
                    }
                }
            }
            llama_batch_free(ver_batch);
        }

        // 3. Compute the accepted count from the per-position argmaxes.
        //    Row i is conditioned on prefix + draft[0..i-1], so it
        //    judges draft[i] - the same pairing as the production
        //    accept path (common_sampler_sample_and_accept_n over the
        //    batched verify forward; the adaptive muxer's trial_verify
        //    compares the logits of the entry that fed draft[i-1] to
        //    draft[i]). The bonus is the argmax of row n_acc, the
        //    first row conditioned only on accepted drafts.
        int n_acc = 0;
        for (int i = 0; i < n_dft; ++i) {
            if (v_argmax[i] == draft[i]) {
                n_acc = i + 1;
            } else {
                break;
            }
        }
        const llama_token bonus = v_argmax[n_acc];
        std::vector<llama_token> ids;
        ids.reserve(n_acc + 1);
        for (int k = 0; k < n_acc; ++k) {
            ids.push_back(draft[k]);
        }
        ids.push_back(bonus);
        n_accepted += (int) (ids.size() - 1);

        // 4. Roll back the verifier's KV for the rejected tokens. The
        //    verifier's KV now has positions n_past..n_past+n_dft
        //    (n_dft+1 tokens). Keep id_last + the accepted drafts
        //    (positions n_past..n_past+n_acc); the bonus becomes the
        //    next step's id_last at position n_past+n_acc+1. Trim
        //    positions n_past+n_acc+1..n_past+n_dft.
        if (n_acc < n_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0,
                                n_past + n_acc + 1,
                                n_past + n_dft + 1);
        }

        // 4b. If telemetry is enabled, serialize the per-step record
        //     with the shared emitter and write it. Single schema:
        //     llama.tessera.spec.v1. The cheap per-step fields
        //     (schema, seq_id, step_idx, prime_token, drafted,
        //     accepted, drafted_tokens, accepted_tokens, confidence)
        //     are always emitted; the per-position top-k distributions
        //     (verifier_topk_*, drafter_topk_*, *_argmax) are added only
        //     when telemetry_topk > 0.
        //
        //     The verifier's per-position logits are in v_logits_ptrs
        //     (n_dft+1 entries, one per prefix [id_last] up through
        //     [id_last, draft[0], ..., draft[n_dft-1]]). Row i is
        //     conditioned on prefix + draft[0..i-1]: it judges draft[i],
        //     and row n_dft is the bonus row. The drafter's rows in
        //     dft_logits_ptrs follow the same convention (priming row
        //     + one row per draft forward).
        if (telemetry_fp != nullptr) {
            const std::string line = common_spec_telemetry_record(
                step, id_last, draft, (size_t) n_acc,
                v_logits_ptrs, dft_logits_ptrs,
                llama_vocab_n_tokens(vocab), opts.telemetry_topk,
                /*provenance=*/nullptr, /*sid=*/nullptr);
            if (std::fwrite(line.data(), 1, line.size(), telemetry_fp) != line.size()) {
                LOG_WRN("%s: failed to write telemetry record at step %d\n",
                        __func__, step);
            }
        }

        // 5. Roll back the verifier's KV for the rejected tokens. The
        //    verifier saw n_dft+1 tokens (id_last at n_past + drafts at
        //    n_past+1..n_past+n_dft). The kept tail is n_acc drafts
        //    (positions n_past+1..n_past+n_acc) plus the bonus token
        //    (position n_past+n_acc+1). Rejected tokens are at
        //    positions n_past+n_acc+1..n_past+n_dft (the bonus + the
        //    remaining rejected drafts).
        if ((int) n_acc < n_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0,
                                n_past + (int) n_acc + 1,
                                n_past + n_dft + 1);
        }
        // Same roll-back on the drafter.
        if ((int) n_acc < n_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0,
                                n_past + (int) n_acc + 1,
                                n_past + n_dft + 1);
        }

        // 6. Advance past the kept (accepted + bonus) tokens.
        n_past += (int) n_acc + 1;
        id_last = ids.back();
        step++;

        if ((size_t) n_past + n_draft_max + 1 >= (size_t) llama_n_ctx(ctx_tgt)) {
            if (opts.verbosity > 0) {
                LOG_INF("%s: approaching context limit at step %d (n_past=%d), stopping\n",
                        __func__, step, n_past);
            }
            break;
        }
    }

    if (opts.verbosity > 0) {
        LOG_INF("%s: spec loop done: n_steps=%d, n_drafted=%d, n_accepted=%d, accept_rate=%.3f\n",
                __func__, step, n_drafted, n_accepted,
                n_drafted > 0 ? double(n_accepted) / double(n_drafted) : 0.0);
    }
    common_speculative_print_stats(spec);

    llama_batch_free(batch);
    if (telemetry_fp) {
        std::fclose(telemetry_fp);
        if (opts.verbosity > 0) {
            LOG_INF("%s: closed telemetry output\n", __func__);
        }
    }

    // Final observer flush (replaces g_collector.flush_graph_observers()
    // that the inlined code returned in the original imatrix.cpp).
    // The hook returns bool; false here is non-fatal (we've already
    // done the decode) but we propagate it for the caller to log.
    return obs_flush();
}

//
// Release any per-run state held by the calibration API.
// Currently a no-op.  See speculative-calibration.h for rationale.
//
void common_speculative_calibration_free() {
    // Intentionally empty: common_speculative_calibration_run() owns
    // no heap state across the run boundary.  Reserved for future
    // expansion (e.g. a stats buffer that survives multiple runs).
}
