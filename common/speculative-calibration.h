#pragma once

//
// Calibration API for spec-decoding (tessera add-on)
//
// PURPOSE
// -------
// This is a *calibration* / *telemetry* path for drafter / verifier pairs.
// It is NOT a production speculative-decoding driver.  See the
// `common_speculative_*` family in common/speculative.h for the production
// path.
//
// WHY THIS EXISTS (the bypass is intentional, not a bug)
// ------------------------------------------------------
// The public common_speculative_draft() / common_speculative_accept() loop
// in common/speculative.cpp has known off-by-one KV bugs that the upstream
// maintainers have not been able to fix without destabilizing the production
// spec-decoding path.  Commit 9e9f275 documented this and shipped a manual
// drafter forward + KV rollback directly inside tools/imatrix/imatrix.cpp so
// the calibration path could produce correct telemetry while the production
// path is fixed independently.
//
// This header (and speculative-calibration.cpp) lifts that manual forward
// out of the imatrix tool into a first-class API so other tools (server
// telemetry, dspark-realign, etc.) can reuse it without copy-pasting the
// 600+ lines of bookkeeping again.
//
// WHAT THIS API IS GOOD FOR
// --------------------------
//   - per-step accept/reject counts
//   - per-position verifier & drafter top-k distributions
//   - per-position verifier softmax probability of the drafter's pick
//     (the "confidence" the dspark-realign pipeline uses)
//
// WHAT THIS API IS NOT GOOD FOR
// -----------------------------
//   - production token generation.  The per-prefix forward scheme is
//     1-token-per-decode (n_dft+1 forwards per spec step) so we can
//     capture per-position verifier distributions without disabling
//     causal attention.  This is O(n_draft_max) forwards per step
//     instead of the 1 forward that production spec decoding uses.
//     For calibration that's fine; for inference it's 4-16x slower.
//
//   - any speculative type other than draft-model.  The manual forward
//     here is a draft-model forward; ngram / lookup / EAGLE / MTP
//     paths are out of scope.
//
//
// CALL SEQUENCE
// -------------
//   1. Caller loads verifier model + ctx_tgt, and (via common_speculative_init)
//      the drafter model + ctx_dft (which the caller attaches to
//      params.speculative.draft.ctx_dft as the imatrix path does).
//   2. Caller optionally attaches a graph observer filter to ctx_tgt
//      (for imatrix capture during the verifier's "main" forward).
//   3. Caller calls common_speculative_calibration_run().  It runs
//      until either:
//        - n_steps options are exhausted, OR
//        - the verifier's KV approaches the context limit, OR
//        - the drafter returns an empty draft.
//   4. Caller can call common_speculative_calibration_free() between runs
//      to release any per-run state.  Currently a no-op; reserved for
//      future expansion (e.g. a stats buffer).
//
// JSONL SCHEMA
// ------------
// Single canonical schema: `llama.tessera.spec.v1`.  Per-step cheap
// fields are always emitted; per-position top-k distributions are added
// only when telemetry_topk > 0.
//
// Always-emitted fields (cheap per-step payload, present for every
// telemetry_topk value):
//   schema, seq_id, step_idx, prime_token, drafted, accepted,
//   drafted_tokens[], accepted_tokens[], confidence[]
//
// Top-k fields (present only when telemetry_topk > 0):
//   topk, verifier_argmax[], drafter_argmax[],
//   verifier_topk_tokens[][], verifier_topk_probs[][],
//   drafter_topk_tokens[][], drafter_topk_probs[][]
//
// Confidence is the per-draft-token verifier softmax probability of the
// drafter's pick; consumers that only need the accept/reject signal can
// ignore the top-k fields entirely.  Consumers that need full per-position
// distributions set telemetry_topk > 0 and read the topk arrays.
//

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <cstdint>
#include <string>
#include <vector>

//
// Serialize one llama.tessera.spec.v1 JSONL record (including the
// trailing newline) for a single spec step.
//
// Factored out of common_speculative_calibration_run() so other drivers
// (the runtime spec engine in common/tessera-runtime.cpp) can emit
// schema-identical records.  Inputs:
//
//   - step, id_last, draft, n_acc: the step bookkeeping (n_acc is the
//     number of accepted drafts, longest accepted prefix).
//   - v_logits_rows / dft_logits_rows: the per-position verifier and
//     drafter logits rows, n_dft + 1 entries each (one per prefix).
//     Entries may be nullptr when a row was unavailable; such positions
//     serialize as argmax 0 and empty top-k arrays, matching the
//     historical inline behavior.
//   - n_vocab: vocabulary size used for the softmax / top-k scans.
//   - topk: clamped to [0, n_vocab].  0 emits only the cheap per-step
//     payload; > 0 adds the per-position top-k fields.
//   - provenance / sid: additive trailing fields, emitted only when
//     non-NULL (runtime records).  Calibration passes NULL for both so
//     its output stays byte-identical to the historical inline emitter.
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
    const char * provenance = nullptr,
    const char * sid = nullptr);

//
// Observer hooks (called around the verifier's "main" forward).
//
// The calibration path runs n_dft+1 verifier forwards per spec step
// to capture per-position distributions.  The LAST of those forwards
// (i == n_dft) is the "main" forward that should be observed by the
// caller (e.g. for imatrix capture).  The earlier forwards are
// observer-silent to keep imatrix attribution clean: one spec step
// == one observer chunk.
//
// begin()  is called immediately before the main forward.
// flush()  is called immediately after the main forward.
//
// The hooks are also called for the initial priming forward so the
// caller can bracket the entire imatrix session if desired.
//
struct common_speculative_calibration_observer_hooks {
    void (*begin)(void * user_data)            = nullptr;
    bool (*flush)(void * user_data)            = nullptr;
    void  * user_data                          = nullptr;
};

//
// Options for the spec-decoding calibration loop.
//
// Defaults match the imatrix path.  Callers usually set telemetry_out
// (and optionally telemetry_topk) and leave the rest alone.
//
struct common_speculative_calibration_options {
    // Override the number of draft tokens per spec step.
    // 0 = use params.speculative.n_max.
    int32_t n_draft_max_override = 0;

    // Override the maximum number of spec steps.
    // 0 = use params.n_spec_steps.  <0 = "until context limit".
    int32_t n_steps_override = 0;

    // Path to a JSONL file to emit per-step telemetry records to.
    // Empty = no telemetry.
    std::string telemetry_out;

    // Number of top-k entries to include in the per-position verifier
    // and drafter distributions.  0 = emit a record without the top-k
    // fields (cheap per-step payload only).  > 0 = additionally emit the
    // top-k token/prob arrays for the drafter fine-tuning pipeline.
    int32_t telemetry_topk = 0;

    // Observer hooks for the verifier's main forward.
    common_speculative_calibration_observer_hooks observer_hooks;

    // Verbosity knob: 0 = silent, 1 = per-step LOG_INF, 2 = debug.
    int32_t verbosity = 1;
};

//
// Run the spec-decoding calibration loop.
//
// Pre-conditions:
//   - ctx_tgt and model_tgt are the verifier context/model.
//   - params.speculative.draft.ctx_dft is the drafter context
//     (set this from common_speculative_init_result::context() before
//     calling; the imatrix path does this at line ~2664 of
//     tools/imatrix/imatrix.cpp).
//   - spec is the common_speculative handle returned by
//     common_speculative_init().  The handle does NOT need to be
//     used for drafting — the calibration path drives the drafter
//     manually.  It is kept so the spec stats (printed via
//     common_speculative_print_stats) still work.
//
// Post-conditions on success:
//   - One JSONL line per spec step was appended to opts.telemetry_out
//     (if non-empty).
//   - ctx_tgt's KV cache is left at the end of the last accepted
//     spec step, i.e. at position n_past + n_accepted + 1, with
//     id_last set to the bonus token of the last step.  This is the
//     same state a successful common_speculative_accept() would leave.
//   - The observer_hooks.begin / .flush bracketing has been called
//     once per spec step (plus once for the priming forward).
//
// Returns true on success, false on any error (tokenization failure,
// drafter / verifier forward failure, telemetry file open failure,
// etc.).
//
bool common_speculative_calibration_run(
    llama_context * ctx_tgt,
    llama_model   * model_tgt,
    common_speculative * spec,
    common_params & params,
    const int32_t n_ctx,
    const common_speculative_calibration_options & opts = {});

//
// Release any per-run state held by the calibration API.
//
// Currently a no-op (common_speculative_calibration_run() owns no
// heap state).  Provided for symmetry with the rest of the
// common_speculative_* API surface and to give future expansions
// (e.g. a stats buffer that needs to survive multiple runs) a place
// to plug in without breaking callers.
//
// Safe to call when no run is in progress.
//
void common_speculative_calibration_free();
