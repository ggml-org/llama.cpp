#pragma once

//
// tessera-ab-harness.h
//
// G6 acceptance harness. Compares the offline ternary-MSE proxy against
// the kernel-direct fitness (L1.5 t_l^2) on held-out tensors, forms the
// alpha-weighted composites, and measures ranking disagreement between
// the cheap proxy and the expensive kernel-direct signal. This is the
// falsifiable novelty gate: the regime-routed composite must beat the
// best single proxy, and the proxy/kernel ranking divergence is reported
// honestly (near-zero divergence means the kernel loop buys nothing
// beyond routing).
//

#include <string>
#include <vector>
#include <cstdint>

struct ts_ab_tensor_scores {
    std::string name;
    float offline_proxy_mse;    // ternary MSE (cheap, no kernel)
    float kernel_direct_t2;     // t_l^2 from L1.5 reference (expensive, real kernel)
    float alpha_l;              // HIGGS weight (uniform = 1.0 for now)
};

struct ts_ab_harness_params {
    int64_t n_heldout;          // number of held-out tensors for evaluation
    bool    measure_ranking;    // compute Kendall tau between proxy and kernel-direct
    bool    verbose;
};

struct ts_ab_harness_result {
    std::vector<ts_ab_tensor_scores> scores;
    float composite_offline;        // Sum alpha_l * offline_proxy_mse
    float composite_kernel;         // Sum alpha_l * kernel_direct_t2
    float kendall_tau;              // ranking correlation (-1 to 1)
    float ranking_disagreement;     // fraction of pairs with different ordering
    bool  composite_beats_single;   // does composite beat best single proxy?
    std::string report;             // human-readable summary
};

// Run the A/B comparison.
// scores_in: pre-computed per-tensor scores (offline proxy + kernel-direct).
// The harness analyzes them, computes composites, measures ranking agreement.
int ts_ab_run(const std::vector<ts_ab_tensor_scores> * scores_in,
              const ts_ab_harness_params * params,
              ts_ab_harness_result * result);

// Compute Kendall tau-b rank correlation between two score arrays.
float ts_ab_kendall_tau(const float * a, const float * b, int64_t n);

// Generate a receipt JSON summarizing the A/B result (for provenance).
std::string ts_ab_receipt_json(const ts_ab_harness_result * result);
