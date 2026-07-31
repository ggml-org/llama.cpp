#pragma once

//
// tessera-acceptance.h
//
// G6 acceptance gate: the falsifiable novelty test. Determines whether
// the regime-routed kernel-fidelity composite actually beats single-proxy
// baselines (Test 1), and whether the offline proxy ranks candidates
// differently from kernel-direct fitness (Test 2). If both pass, the
// novelty claim survives; if ranking disagreement is near zero, the
// kernel-fidelity contribution is null and the novelty reduces to routing.
//
// See docs/research-alignment-2026-07-30.md Section 5.
//

#include <cstdint>

struct ts_acceptance_config {
    int   n_heldout_tensors;     // number of held-out tensors for evaluation
    float heldout_fraction;      // default 0.2 (20% of tensors held out)
    bool  verbose;
    char  output_path[1024];     // JSON report output (empty = no file)
};

// Per-tensor input scores for the acceptance gate. The caller computes
// these by quantizing each tensor under the composite (regime-routed)
// pipeline and under each single proxy forced for all tensors.
struct ts_acceptance_tensor {
    char  name[256];
    float composite_t2;          // regime-routed composite t_l^2
    float awq_t2;                // AWQ-only t_l^2
    float rotation_t2;           // rotation-only (DartQuant) t_l^2
    float lowrank_t2;            // low-rank-only (LRQ/FLRQ) t_l^2
    float hessian_t2;            // Hessian-mask-only (SEPTQ) t_l^2
    float offline_proxy_mse;     // offline proxy score (for ranking)
    float kernel_direct_t2;      // kernel-direct score (for ranking)
    bool  held_out;              // true = not used during GA evolution
};

struct ts_acceptance_result {
    // Test 1: composite beats single-proxy
    float composite_t2;          // mean regime-routed composite t_l^2 (held-out)
    float best_single_t2;        // best single-proxy mean t_l^2 (held-out)
    float improvement_pct;       // (best_single - composite) / best_single * 100
    bool  composite_wins;        // composite_t2 < best_single_t2

    // Test 2: ranking disagreement
    float kendall_tau;           // offline vs kernel-direct ranking agreement
    float ranking_disagreement;  // 1 - |kendall_tau| (0 = identical rankings)
    bool  novelty_survives;      // ranking_disagreement > 0.05 (non-trivial)

    // Per-proxy breakdown (mean t_l^2 over held-out tensors)
    float awq_t2;
    float rotation_t2;
    float lowrank_t2;
    float hessian_t2;

    // Overall verdict
    bool  acceptance_passed;     // composite_wins AND novelty_survives
    char  verdict[512];          // human-readable summary

    // Provenance
    int64_t n_tensors_total;
    int64_t n_tensors_heldout;
};

void ts_acceptance_default_config(ts_acceptance_config * cfg);

// Run the acceptance gate on pre-computed per-tensor scores.
// Returns 0 on success, -1 on invalid args.
int ts_acceptance_run(const ts_acceptance_config * config,
                      const ts_acceptance_tensor * tensors,
                      int64_t n_tensors,
                      ts_acceptance_result * result);

// Write the JSON report (schema llama.tessera.acceptance.v1).
// Returns 0 on success, -1 on error.
int ts_acceptance_write_json(const char * path,
                             const ts_acceptance_config * config,
                             const ts_acceptance_result * result,
                             const ts_acceptance_tensor * tensors,
                             int64_t n_tensors);
