#include "tessera-lk-loss.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

double ts_lk_token_contribution(float p_x, float q_x) {
    return (double)std::min(p_x, q_x);
}

double ts_lk_acceptance_rate(const float * p, const float * q, int n_vocab) {
    double alpha = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        alpha += std::min((double)p[i], (double)q[i]);
    }
    return alpha;
}

double ts_lk_loss(const float * p, const float * q, int n_vocab) {
    return -ts_lk_acceptance_rate(p, q, n_vocab);
}

double ts_lk_acceptance_rate_topk(const int32_t * p_tokens, const float * p_probs, int p_k,
                                  double p_mass,
                                  const int32_t * q_tokens, const float * q_probs, int q_k,
                                  double q_mass,
                                  int n_vocab) {
    // Build a map from token id -> (p_prob, q_prob) over the union of both
    // top-k sets. Tokens outside both sets share the residual mass.
    std::unordered_map<int32_t, std::pair<double, double>> entries;
    entries.reserve((size_t)(p_k + q_k));

    for (int i = 0; i < p_k; ++i) {
        entries[p_tokens[i]].first += (double)p_probs[i];
    }
    for (int i = 0; i < q_k; ++i) {
        entries[q_tokens[i]].second += (double)q_probs[i];
    }

    double alpha = 0.0;
    for (const auto & [tok, probs] : entries) {
        (void)tok;
        alpha += std::min(probs.first, probs.second);
    }

    // Residual bucket: probability mass outside both top-k sets.
    // The residual is spread uniformly over the remaining vocabulary, so
    // the per-token residual probability is tiny; the min() over the
    // residual bucket is bounded by the smaller residual mass.
    const double p_residual = std::max(0.0, 1.0 - p_mass);
    const double q_residual = std::max(0.0, 1.0 - q_mass);
    // Number of tokens NOT in either top-k set.
    const int n_in_union = (int)entries.size();
    const int n_residual  = std::max(0, n_vocab - n_in_union);
    if (n_residual > 0) {
        const double p_per = p_residual / n_residual;
        const double q_per = q_residual / n_residual;
        alpha += n_residual * std::min(p_per, q_per);
    }

    return alpha;
}

double ts_lk_acceptance_rate_batch(const int32_t * const * p_tokens,
                                   const float * const * p_probs,
                                   const int * p_k,
                                   const double * p_mass,
                                   const int32_t * const * q_tokens,
                                   const float * const * q_probs,
                                   const int * q_k,
                                   const double * q_mass,
                                   int n_positions,
                                   int n_vocab) {
    if (n_positions <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n_positions; ++i) {
        sum += ts_lk_acceptance_rate_topk(
            p_tokens[i], p_probs[i], p_k[i], p_mass[i],
            q_tokens[i], q_probs[i], q_k[i], q_mass[i],
            n_vocab);
    }
    return sum / n_positions;
}
