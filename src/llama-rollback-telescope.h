// Copyright (c) 2026 Song Wei
// SPDX-License-Identifier: MIT
//
// Forward-telescoping rollback for recurrent (Gated DeltaNet) models.
//
// A partial rollback (`seq_rm(p0)`) on a recurrent model normally forces a
// full prompt re-evaluation, because the recurrent state depends on the whole
// prefix. The forward-telescoping approach replaces "recompute everything"
// with "replay the exact linear state map from the nearest anchor":
//
//     S_t = (gamma_t*I - beta_t*k_t*k_t^T) * S_{t-1} + beta_t * k_t * v_t^T
//
// which is the exact per-step recurrence (code-verified, see
// delta-net-base.cpp) and requires no division, no gating scans and no layer
// re-evaluation: k/v/beta/gamma are captured in a metadata ring during the
// normal forward pass, and the state S is replayed forward from an fp32
// anchor snapshot. Window error saturates at u/(1-gamma_bar) and is
// independent of the total rollback depth (the heavy-tailed gate that would
// amplify division-based inversion instead damps the forward product).
//
// ---------------------------------------------------------------------------
// INTEGRATION GUIDE (this PR ships the engine + tests; wiring is a follow-up)
// ---------------------------------------------------------------------------
// The class is self-contained and CPU-first; the GPU kernel is a drop-in
// replacement for `fwd_telescope_execute`. To wire it into llama.cpp:
//
// 1. CAPTURE HOOK (required): in `build_recurrent_attn`
//    (src/models/qwen35.cpp / delta-net-base.cpp), after the conv projection
//    and before the state update, capture (k, v, beta, gate_logit) per
//    (layer, head) into the metadata ring. Gate semantics to verify first
//    (P0 gate): `gamma = exp(gate_logit)` with gate_logit = softplus(x*W_a +
//    dt)*A_log, gamma in (0,1]. Assert the operator identity with fp64
//    (rel-err <= 1e-6) before enabling.
//
// 2. SEQ_RM BRANCH: in `llama_memory_recurrent::seq_rm`
//    (src/llama-memory-recurrent.cpp), the partial-rollback path currently
//    returns false when `rollback > n_rs_seq`. Add a telescope branch:
//    if `rollback` is within `cfg.window` and the engine is enabled, call
//    `rollback(p0, S_out)` and write S_out back into the s_l tensors for the
//    sequence's tail cell; set cell.pos = p0-1. Otherwise keep the existing
//    fallback (full recompute).
//
// 3. CLI: add `--rollback-window` / `--rollback-coverage` to llama-server /
//    llama-cli (defaults 40 / 600), mapping to the telescope_config fields.
//
// 4. ACCEPTANCE: replay error <= 5e-4 on synthetic gates (const 0.9876,
//    heavy-tailed, real 30-layer sample) for depths 64..118000; latency
//    >= 90x vs full recompute on a real Gated DeltaNet model.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <vector>

namespace llama {

struct telescope_config {
    uint32_t window   = 40;    // B: anchor spacing = max replay window (tokens)
    uint32_t coverage = 600;   // R: anchor history horizon = slots * window
    uint32_t slots    = 15;    // = ceil(R / B)
    uint32_t L        = 0;     // total layers (GGUF-calibrated at runtime)
    uint32_t L_meta   = 0;     // replay layers (GGUF-calibrated)
    uint32_t H        = 0;     // value heads
    uint32_t d        = 0;     // ssm_d_state
    uint32_t ring_cap = 41;    // metadata ring = B + 1 (correction C1)
    float    posterior_tol = 1e-3f;
};

struct rollback_result {
    bool     ok = false;
    uint32_t anchor = 0;
    float    posterior_err = 0.0f;
    uint64_t fallback_count = 0;
};

// Forward-telescoping rollback engine: metadata ring + anchor ring + replay.
//
// The class is deliberately self-contained and CPU-first (OpenMP-parallel
// across layers); the GPU kernel is a drop-in replacement for
// `fwd_telescope_execute` in a follow-up.
class telescope_rollback {
public:
    explicit telescope_rollback(const telescope_config & cfg);

    // Capture the per-token per-layer metadata (k/v/beta/gamma) for the
    // metadata ring. `k`/`v` are [L_meta][H][d]; `beta`/`gamma` are [L_meta][H]
    // with gamma being the per-step multiplier (already exp(gate logit)).
    void capture(uint32_t pos, const float * k, const float * v,
                 const float * beta, const float * gamma);

    // Snapshot the full state S[L][H][d][d] at `pos` (every `window` tokens).
    void place_anchor(uint32_t pos, const float * S);

    // Roll back the recurrent state to position `p0` (exclusive). Writes the
    // restored state into `S_out` (caller-owned [L][H][d][d]).
    rollback_result rollback(uint32_t p0, float * S_out);

    // True if the replay window (a, p0] is fully covered by the metadata ring.
    bool resident(uint32_t a, uint32_t p0) const;

    uint64_t fallback_count() const { return fallback_count_; }
    size_t   resident_bytes() const;
    const telescope_config & config() const { return cfg_; }

private:
    uint32_t locate_anchor(uint32_t p0) const;   // largest anchor pos <= p0
    void fwd_telescope_execute(uint32_t a, uint32_t p0, float * S_out) const;

    telescope_config cfg_;

    // metadata ring: [ring_cap][L_meta][H][2d+2] fp32, absolute-position indexed
    std::vector<float> meta_ring_;
    std::vector<uint32_t> meta_pos_;   // absolute pos per ring slot (valid flag via sentinel)
    // anchor ring: [slots][L][H][d][d] fp32
    std::vector<float> anchor_ring_;
    std::vector<uint32_t> anchor_pos_;
    std::vector<bool>    anchor_valid_;

    uint64_t fallback_count_ = 0;
};

} // namespace llama
