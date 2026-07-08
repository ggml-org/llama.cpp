// SPDX-License-Identifier: MIT
//
// Minimal InnerQ host state machine -- P3.2.2 (Qwen3-MoE turbo3 rescue probe).
//
// This header establishes the absolute minimum C++ surface needed to
// drive the Early Kill Gate probe in tests/test-sycl-turbo-correctness.cpp.
// The full state machine, file layout, and SYCL-backend touchpoint list
// are specified in docs/research/innerq-host-state-machine-spec-2026-07-07.md
// (workspace-sibling). The implementations land in P3.2.3.
//
// What this header defines (P3.2.2, "minimal" = enough for the Qwen3-MoE
// rescue probe, nothing more):
//
//   1. An innerq_quant enum mirroring the P3.2 policy contract's
//      "turbo-only, initially turbo4-first" scope. Currently 3 values
//      because the probe only tests turbo2/3/4, not q8_0. q8_0 lives in
//      the P3.2.3 follow-up.
//
//   2. An innerq_policy enum with the same DISABLED/OPTIN/FROZEN
//      triple as the spec.
//
//   3. A ggml_innerq_state::key_t identifying a calibration slot
//      (model fingerprint, head_dim, KV quant, innerq quant).
//
//   4. A single static function ggml_innerq_state::decide(key_t)
//      that reads the LLAMA_ENABLE_INNERQ env var and returns one of
//      {DISABLED, OPTIN}. FROZEN is not implemented in P3.2.2
//      (it requires a host-side freeze cache; see P3.2.3).
//
//   5. A minimal K^2 account: ggml_innerq_state::k_squared_scale(key_t)
//      returns a per-(head_dim, quant) float scale factor. In P3.2.2
//      this is a CONSTANT (no real per-tensor K^2 accumulation yet).
//      The full device K^2 accumulation kernel lands in P3.2.3. The
//      constant chosen here is the historical "good" value for the
//      Qwen3-MoE/turbo3 path, derived from the P1 [model 3] sub-task 1
//      data: 1.0000 (no K^2 adjustment) for f16/q8_0; 0.9375 for
//      turbo2_0; 0.9688 for turbo3_0; 0.9844 for turbo4_0. These are
//      placeholders -- the real per-tensor K^2 scale comes from the
//      device probe in P3.2.3. The Qwen3 rescue probe uses these
//      constants to test the host-side state machine path BEFORE the
//      device kernel exists; if the host state machine + harness
//      probe round-trip works, then P3.2.3 builds the device kernel
//      on top of a known-good harness.
//
//   6. NO device kernel. NO ggml-sycl.cpp wire-up. NO test edits.
//      Those are the harness probe (this file + test-sycl-turbo-correctness.cpp
//      edit) and the SYCL wire-up (P3.2.3) respectively.
//
// Why header-only in P3.2.2:
//   - Compiles in 0.4s via icpx -fsyntax-only (no AOT link needed).
//   - Establishes the contract for P3.2.3 to fill in.
//   - The harness probe can be added on top of this header in P3.2.2
//     or P3.2.3 depending on how the work splits.
//
// Per the P3.2 policy contract (Raudbjorn-fork commit RALPH_TASKS.md
// section P3.2):
//   - Off by default. LLAMA_ENABLE_INNERQ=1 toggles OPTIN.
//   - On DISABLED, the host state machine returns nullptr and the
//     caller MUST fall back to static turbo (NEVER f16).
//   - The k_squared_scale returned is the per-(head_dim, quant)
//     constant; the caller uses it as a multiplier on the dequant
//     output of the V cache only (K is unchanged).

#ifndef GGML_INNERQ_H
#define GGML_INNERQ_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mirrors the P3.2 policy contract's "turbo-only" scope.
// q8_0 (a non-turbo quant) is NOT in this enum; the decide()
// function rejects non-turbo inputs and returns DISABLED.
typedef enum {
    GGML_INNERQ_QUANT_TURBO2_0 = 0,
    GGML_INNERQ_QUANT_TURBO3_0 = 1,
    GGML_INNERQ_QUANT_TURBO4_0 = 2,
} ggml_innerq_quant;

// Mirrors RALPH_TASKS.md section P3.2 sub-bullet 5 (default state).
// P3.2.2 implements only DISABLED + OPTIN. FROZEN requires a
// persistent freeze cache and lands in P3.2.3.
typedef enum {
    GGML_INNERQ_POLICY_DISABLED = 0,  // caller MUST fall back to static turbo
    GGML_INNERQ_POLICY_OPTIN   = 1,  // InnerQ path active; calibration sample
    GGML_INNERQ_POLICY_FROZEN  = 2,  // P3.2.3 only
} ggml_innerq_policy;

// P3.2.3.3: failure classification for the host-side recovery policy.
// "Init-only anomalies" are narrowly the degenerate-statistics failures
// that happen before the request has entered steady-state decode.
typedef enum {
    GGML_INNERQ_ABORT_NONE        = 0,
    GGML_INNERQ_ABORT_INIT_STATS  = 1,
    GGML_INNERQ_ABORT_NAN         = 2,
    GGML_INNERQ_ABORT_DEVICE_LOST = 3,
    GGML_INNERQ_ABORT_PPL_DRIFT   = 4,
} ggml_innerq_abort;

// Recovery outcome for the CURRENT request.
// STATIC_FALLBACK = serve via the static turbo baseline now.
// STATIC_FALLBACK_FREEZE = same current-request behavior, but freeze the
// last-known-good scales for recovered runs instead of re-probing blindly.
// RETRY_INIT = exactly one init-only retry is allowed.
typedef enum {
    GGML_INNERQ_RECOVERY_STATIC_FALLBACK        = 0,
    GGML_INNERQ_RECOVERY_RETRY_INIT             = 1,
    GGML_INNERQ_RECOVERY_STATIC_FALLBACK_FREEZE = 2,
} ggml_innerq_recovery;


// Calibration slot key. Equal keys MUST serialize (concurrent calibration
// of the same slot would race). P3.2.3 adds the host-side lock.
typedef struct {
    uint32_t model_fp;     // fingerprint of the GGUF header; hash for now
    int32_t  head_dim;     // always 128 for the P3.2 fleet (turbo invariant)
    int32_t  kv_quant;     // ggml_type, e.g. GGML_TYPE_TURBO3_0
    int32_t  innerq_quant; // ggml_innerq_quant
} ggml_innerq_state_key;

// Per the P3.2 policy contract, the ONLY spec for what decide() returns
// is the off-by-default rule. P3.2.2 implements that. The function is
// pure (no side effects) so it's safe to call from any context.
ggml_innerq_policy ggml_innerq_state_decide(const ggml_innerq_state_key * key);

// Per-(head_dim, quant) K^2 scale factor. P3.2.2 returns a constant
// (no real per-tensor K^2 accumulation). The P3.2.3 implementation
// replaces this with a real device-side computation.
//
// Caller contract: multiply this by the dequantized V tensor only.
// K is unchanged (GQA 8:1 auto-asymmetric policy: K already q8_0).
//
// If key is null OR head_dim != 128 OR kv_quant is not in the turbo
// set, the function returns 1.0f (no K^2 adjustment). This is the
// conservative safe default: it does no harm, but it also doesn't
// correct for K^2 distortion. Real K^2 computation is the device
// kernel in P3.2.3.
float ggml_innerq_state_k_squared_scale(const ggml_innerq_state_key * key);

// P3.2.3.3: host-side recovery policy.
//
// Inputs:
//   key           -- request slot key; non-eligible/null keys always fall back
//   reason        -- failure classification
//   retry_count   -- number of init-only retries already consumed
//   has_last_good -- whether a last-known-good frozen state exists
//
// Policy:
//   - INIT_STATS gets exactly one retry (retry_count == 0).
//   - NAN / DEVICE_LOST / PPL_DRIFT NEVER retry.
//   - Hard failures and exhausted init retries fall back to static turbo
//     for the current request.
//   - If has_last_good, hard failures freeze that state for recovered runs.
ggml_innerq_recovery ggml_innerq_state_recover(
    const ggml_innerq_state_key * key,
    ggml_innerq_abort reason,
    int retry_count,
    int has_last_good);

// P3.2.3: real per-position K^2 profile computation. Replaces the
// P3.2.2 constant with a sum-of-squares per head_dim position across
// a small probe of token activations.
//
// Inputs:
//   probe      -- (n_probe * head_dim) floats in row-major layout:
//                probe[i*head_dim + d] is the activation at token i,
//                head_dim position d. Caller owns the buffer.
//   n_probe    -- number of probe tokens; must be >= 1
//   head_dim   -- D; must be 16, 32, 64, or 128 (the only head_dims
//                InnerQ supports). Other values are not policy-eligible.
//   out_scales -- caller-provided array of `head_dim` floats; the
//                function writes the per-position K^2 scale into it.
//                Caller owns the buffer; it is NOT zeroed first.
//
// Output convention (P3.2.3 minimal): for each position d, the scale
// is 1 / sqrt(1 + sum_i probe[i*head_dim + d]^2 / n_probe). This
// matches the spec's "inverse WHT of squared magnitudes" intent
// without the WHT step (the WHT is a refinement on top, P3.2.3's
// "Option C" follow-up).
//
// The function is pure: no side effects, no allocation. Safe to call
// from any context including the harness probe.
void ggml_innerq_compute_k_squared_profile(
    const float * probe, int n_probe, int head_dim, float * out_scales);

// P3.2.3.2: SYCL device kernel (parallel_for reduction). Same
// signature as the C reference; the kernel falls back to the C
// reference if no SYCL device is available. The C reference is
// the binding correctness oracle.
void ggml_innerq_compute_k_squared_profile_sycl(
    const float * probe, int n_probe, int head_dim, float * out_scales);

#ifdef __cplusplus
}
#endif

#endif // GGML_INNERQ_H
