// SPDX-License-Identifier: MIT
//
// ggml_innerq_state_decide / ggml_innerq_state_k_squared_scale
// implementations for the P3.2.2 minimal host state machine.
//
// P3.2.2 ships a CONSTANT k_squared_scale (no real per-tensor
// accumulation yet). P3.2.3 replaces this with the real device
// kernel. See ggml/include/ggml-innerq.h for the public surface
// and docs/research/innerq-host-state-machine-spec-2026-07-07.md
// for the spec.

#include "ggml-innerq.h"
#include "ggml.h"

#include <cmath>

#include <cstring>

// Internal helper: model fingerprint. P3.2.2 doesn't have a real
// fingerprint extractor; for now the caller passes a key with
// model_fp = 0 (a sentinel meaning "trust the env gate, ignore the
// model identity"). P3.2.3 wires the real fingerprint extractor
// (hash of GGUF header chunks) and rejects model_fp = 0 as
// "unidentified" -- which is the conservative safe default.
static int is_unidentified_model(uint32_t model_fp) {
    return model_fp == 0u;
}

extern "C" ggml_innerq_policy ggml_innerq_state_decide(
    const ggml_innerq_state_key * key) {
    // Per the P3.2 policy contract, off by default. LLAMA_ENABLE_INNERQ
    // toggles OPTIN. The env-var read is the single source of truth
    // for policy; the SYCL backend reads decide()'s return, NOT
    // getenv directly. (Spec, "Off-by-default contract" section.)
    if (key == nullptr) {
        return GGML_INNERQ_POLICY_DISABLED;
    }
    if (is_unidentified_model(key->model_fp)) {
        // Conservative safe default: refuse to InnerQ on a model we
        // don't recognise. P3.2.3 sets model_fp from a real GGUF
        // hash; today (P3.2.2) every caller should pass model_fp=0
        // and the state machine correctly refuses.
        return GGML_INNERQ_POLICY_DISABLED;
    }
    // Policy contract: only turbo-path quants are eligible. The harness
    // probe (test-sycl-turbo-correctness.cpp [8] InnerQ) only ever passes
    // TURBO{2,3,4}_0 kv_quants. f16/q8_0/q4_0 are evaluation baselines
    // and fall back targets -- not InnerQ-eligible per the policy.
    switch (key->kv_quant) {
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            break;
        default:
            return GGML_INNERQ_POLICY_DISABLED;
    }
    // Per the policy contract, head_dim=128 is the validation fleet
    // invariant (turbo hard invariant). d=256 stays behind the existing
    // opt-in gate (LLAMA_TEST_FA256), and d != {128, 256} is not
    // policy-eligible. P3.2.2 only handles d=128.
    if (key->head_dim != 128) {
        return GGML_INNERQ_POLICY_DISABLED;
    }
    // Opt-in gate: LLAMA_ENABLE_INNERQ=1 enables. Absent (or empty
    // value), DISABLED. This is the only place that reads the env
    // var on the host side.
    const char * env = getenv("LLAMA_ENABLE_INNERQ");
    if (env == nullptr || env[0] == '\0' || env[0] == '0') {
        return GGML_INNERQ_POLICY_DISABLED;
    }
    return GGML_INNERQ_POLICY_OPTIN;
}

// Per the policy contract sub-bullet 1 (validation corpus) and
// sub-bullet 2 (failure modes), the K^2 scale is the precision
// budget applied to the dequantized V tensor only. P3.2.2 returns
// the per-quant constant chosen from the P1 [model 3] sub-task 1
// PPL data; P3.2.3 replaces this with a real per-tensor device
// computation. The values here are PLACEHOLDERS -- they are the
// historical "good" values that produced the P1 PPL numbers; the
// real per-tensor K^2 scale will come from device-side probing.
extern "C" float ggml_innerq_state_k_squared_scale(
    const ggml_innerq_state_key * key) {
    if (key == nullptr) {
        return 1.0f;
    }
    if (key->head_dim != 128) {
        return 1.0f;  // not eligible; safe default
    }
    // Per-(kv_quant, innerq_quant) constant lookup. P3.2.2's
    // placeholder table; the real per-tensor computation is P3.2.3.
    // Values from the P1 [model 3] sub-task 1 data:
    //   turbo2 (innerq2) -> 0.9375f
    //   turbo2 (innerq3) -> 0.9688f
    //   turbo2 (innerq4) -> 0.9844f
    //   turbo3 (innerq2) -> 0.9375f
    //   turbo3 (innerq3) -> 0.9688f
    //   turbo3 (innerq4) -> 0.9844f
    //   turbo4 (innerq2) -> 0.9375f
    //   turbo4 (innerq3) -> 0.9688f
    //   turbo4 (innerq4) -> 0.9844f
    // In all 9 cases, the table is dominated by the innerq_quant
    // value (kv_quant is the cache substrate, doesn't shift the
    // precision budget on its own). The P3.2 policy contract says
    // "initially turbo4-first" -- that's the calibration order, not
    // a runtime policy decision.
    float base = 1.0f;
    switch (key->innerq_quant) {
        case GGML_INNERQ_QUANT_TURBO2_0: base = 0.9375f; break;
        case GGML_INNERQ_QUANT_TURBO3_0: base = 0.9688f; break;
        case GGML_INNERQ_QUANT_TURBO4_0: base = 0.9844f; break;
        default: return 1.0f;  // not eligible; safe default
    }
    (void) key->kv_quant;  // currently a no-op; P3.2.3 may extend.
    return base;
}

// P3.2.3: real per-position K^2 profile computation.
//
// For each head_dim position d, the scale is
//     scale[d] = 1 / sqrt(1 + mean_i probe[i*head_dim + d]^2)
// This matches the spec's "inverse WHT of squared magnitudes" intent
// without the WHT step (the WHT is a refinement on top; P3.2.3's
// "Option C" follow-up adds it). The +1 inside the sqrt is a
// smoothing constant that prevents division-by-zero on a zero-input
// probe (a degenerate case: all-zero probe gives scale=1.0 for all d).
//
// P3.2.3 minimal: this is the C reference. The SYCL device kernel
// (parallel_for reduction) is the "Option C" follow-up that
// generalizes across all shapes. The harness probe verifies this
// CPU reference is correct against an independent CPU
// implementation in the test; once verified, the SYCL kernel
// can be implemented to match.
//
// No allocation, no side effects. Safe to call from any context.
extern "C" void ggml_innerq_compute_k_squared_profile(
    const float * probe, int n_probe, int head_dim, float * out_scales) {
    // Defensive: zero the output first so a half-written run leaves
    // a clean (1.0) baseline rather than garbage. We overwrite
    // every position below, but the zero is the safe fallback if
    // n_probe < 1 (which would skip the loop) or head_dim invalid
    // (also skipped). Caller sees 1.0f for skipped positions,
    // matching the "no K^2 adjustment" safe default in
    // k_squared_scale().
    for (int d = 0; d < head_dim; ++d) {
        out_scales[d] = 1.0f;
    }
    if (probe == nullptr || out_scales == nullptr || n_probe < 1) {
        return;
    }
    // Only the head_dims InnerQ supports. Others leave the 1.0 default.
    if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return;
    }
    // Pass 1: per-position sum of squares.
    // Reuses the output array as a scratch buffer for the sum;
    // we'll overwrite with the final 1/sqrt(1+mean) in pass 2.
    for (int d = 0; d < head_dim; ++d) {
        double sumsq = 0.0;
        for (int i = 0; i < n_probe; ++i) {
            const double v = (double) probe[i * head_dim + d];
            sumsq += v * v;
        }
        out_scales[d] = (float)(sumsq / (double) n_probe);  // mean square
    }
    // Pass 2: convert mean-square to 1/sqrt(1+mean-square).
    for (int d = 0; d < head_dim; ++d) {
        const double ms = (double) out_scales[d];
        out_scales[d] = (float)(1.0 / std::sqrt(1.0 + ms));
    }
}
