// SPDX-License-Identifier: MIT
//
// InnerQ host-side reference implementation. Always compiled into ggml-base
// so callers (including src/llama-context.cpp) can link the host helpers
// without depending on the SYCL backend. The SYCL backend's
// ggml-sycl/innerq.cpp adds the device kernel implementation on top.

#include "ggml-innerq.h"
#include "ggml.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int is_unidentified_model(uint32_t model_fp) {
    return model_fp == 0u;
}

static int is_turbo_kv_quant(int kv_quant) {
    switch (kv_quant) {
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            return 1;
        default:
            return 0;
    }
}

static int is_policy_eligible(const ggml_innerq_state_key * key) {
    return key != NULL &&
           !is_unidentified_model(key->model_fp) &&
           key->head_dim == 128 &&
           is_turbo_kv_quant(key->kv_quant);
}

ggml_innerq_policy ggml_innerq_state_decide(const ggml_innerq_state_key * key) {
    // Per the P3.2 policy contract, off by default. LLAMA_ENABLE_INNERQ
    // toggles OPTIN. The env-var read is the single source of truth
    // for policy; the SYCL backend reads decide()'s return, NOT
    // getenv directly. (Spec, "Off-by-default contract" section.)
    if (key == NULL) {
        return GGML_INNERQ_POLICY_DISABLED;
    }
    if (is_unidentified_model(key->model_fp)) {
        // Conservative safe default: refuse to InnerQ on a model we
        // don't recognise. P3.2.3 sets model_fp from a real GGUF
        // hash; today (P3.2.2) every caller should pass model_fp=0
        // and the state machine correctly refuses.
        return GGML_INNERQ_POLICY_DISABLED;
    }
    // Policy contract: only turbo-path quants at d=128 are eligible.
    // f16/q8_0/q4_0 are evaluation baselines and fallback targets.
    if (!is_policy_eligible(key)) {
        return GGML_INNERQ_POLICY_DISABLED;
    }
    // Opt-in gate: LLAMA_ENABLE_INNERQ=1 enables. Absent (or empty
    // value, or "0"), DISABLED. This is the only place that reads
    // the env var on the host side. The runtime capture path in
    // src/llama-context.cpp also calls this function to gate the
    // capture path; callers MUST route their enable check through
    // ggml_innerq_state_decide to keep the contract in one place.
    const char * env = getenv("LLAMA_ENABLE_INNERQ");
    if (env == NULL || env[0] == '\0' || env[0] == '0') {
        return GGML_INNERQ_POLICY_DISABLED;
    }
    return GGML_INNERQ_POLICY_OPTIN;
}

float ggml_innerq_state_k_squared_scale(const ggml_innerq_state_key * key) {
    if (key == NULL) {
        return 1.0f;
    }
    // Caller contract (ggml-innerq.h): 1.0f when key is null,
    // head_dim != 128, OR kv_quant is not in the turbo set. The
    // turbo-set check matches is_policy_eligible() so a non-turbo
    // kv_quant never gets a K^2 correction even with a turbo
    // innerq_quant requested.
    if (key->head_dim != 128 || !is_turbo_kv_quant(key->kv_quant)) {
        return 1.0f;
    }
    // Per-innerq_quant constant at d=128 + turbo kv_quant. P3.2.2's
    // placeholder table; the real per-tensor computation is P3.2.3.
    // Values from the P1 [model 3] sub-task 1 data:
    //   innerq2 -> 0.9375f
    //   innerq3 -> 0.9688f
    //   innerq4 -> 0.9844f
    switch (key->innerq_quant) {
        case GGML_INNERQ_QUANT_TURBO2_0: return 0.9375f;
        case GGML_INNERQ_QUANT_TURBO3_0: return 0.9688f;
        case GGML_INNERQ_QUANT_TURBO4_0: return 0.9844f;
        default: return 1.0f;  // not eligible; safe default
    }
}

void ggml_innerq_compute_k_squared_profile(
    const float * probe, int n_probe, int head_dim, float * out_scales) {
    // Reject dimensions that could overrun or underflow the caller-owned
    // buffer. In-range unsupported dimensions still receive the documented
    // identity fallback before returning (the harness exercises head_dim=100).
    if (out_scales == NULL || head_dim <= 0 || head_dim > 128) {
        return;
    }
    for (int d = 0; d < head_dim; ++d) {
        out_scales[d] = 1.0f;
    }
    if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return;
    }
    if (probe == NULL || n_probe < 1) {
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
        out_scales[d] = (float)(1.0 / sqrt(1.0 + ms));
    }
}

ggml_innerq_recovery ggml_innerq_state_recover(
    const ggml_innerq_state_key * key,
    ggml_innerq_abort reason,
    int retry_count,
    int has_last_good) {
    if (!is_policy_eligible(key)) {
        return GGML_INNERQ_RECOVERY_STATIC_FALLBACK;
    }
    switch (reason) {
        case GGML_INNERQ_ABORT_INIT_STATS:
            if (retry_count <= 0) {
                return GGML_INNERQ_RECOVERY_RETRY_INIT;
            }
            return has_last_good
                ? GGML_INNERQ_RECOVERY_STATIC_FALLBACK_FREEZE
                : GGML_INNERQ_RECOVERY_STATIC_FALLBACK;
        case GGML_INNERQ_ABORT_NAN:
        case GGML_INNERQ_ABORT_DEVICE_LOST:
        case GGML_INNERQ_ABORT_PPL_DRIFT:
            return has_last_good
                ? GGML_INNERQ_RECOVERY_STATIC_FALLBACK_FREEZE
                : GGML_INNERQ_RECOVERY_STATIC_FALLBACK;
        case GGML_INNERQ_ABORT_NONE:
        default:
            return GGML_INNERQ_RECOVERY_STATIC_FALLBACK;
    }
}
