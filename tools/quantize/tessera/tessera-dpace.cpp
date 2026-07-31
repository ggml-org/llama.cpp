#include "tessera-dpace.h"

#include <algorithm>
#include <cmath>
#include <vector>

void ts_dpace_continuation_values(const float * acceptance_probs,
                                  int block_size,
                                  double * out) {
    if (block_size <= 0) {
        return;
    }
    // f_{B-1} = 1 (no positions after the last)
    // f_j = 1 + a_{j+1} * f_{j+1}
    out[block_size - 1] = 1.0;
    for (int j = block_size - 2; j >= 0; --j) {
        out[j] = 1.0 + (double)acceptance_probs[j + 1] * out[j + 1];
    }
}

double ts_dpace_accepted_length_surrogate(const float * acceptance_probs,
                                          int block_size) {
    if (block_size <= 0) {
        return 0.0;
    }
    // S = sum_{k=0}^{B-1} prod_{i=0}^{k} a_i
    double cumulative = 1.0;
    double surrogate  = 0.0;
    for (int k = 0; k < block_size; ++k) {
        cumulative *= (double)acceptance_probs[k];
        surrogate  += cumulative;
    }
    return surrogate;
}

void ts_dpace_weights(const float * acceptance_probs,
                      int block_size,
                      double * out) {
    if (block_size <= 0) {
        return;
    }
    // Continuation values via reverse recurrence
    std::vector<double> f(block_size);
    ts_dpace_continuation_values(acceptance_probs, block_size, f.data());

    // Cumulative product (forward pass) and weight assembly
    double cumulative = 1.0;
    for (int j = 0; j < block_size; ++j) {
        cumulative *= (double)acceptance_probs[j];
        out[j] = cumulative * f[j];
    }
}

void ts_dpace_weights_smoothed(const float * acceptance_probs,
                               int block_size,
                               float alpha,
                               double * out) {
    if (block_size <= 0) {
        return;
    }
    if (alpha <= 0.0f) {
        ts_dpace_weights(acceptance_probs, block_size, out);
        return;
    }
    // Apply asymmetric smoothing: a_tilde = (1 - alpha) * a + alpha
    std::vector<float> smoothed(block_size);
    for (int i = 0; i < block_size; ++i) {
        smoothed[i] = (1.0f - alpha) * acceptance_probs[i] + alpha;
    }
    ts_dpace_weights(smoothed.data(), block_size, out);
}

void ts_dflash_decay_weights(int block_size,
                             float gamma,
                             double * out) {
    if (block_size <= 0) {
        return;
    }
    if (gamma <= 0.0f) {
        // Degenerate: only the anchor gets weight
        for (int j = 0; j < block_size; ++j) {
            out[j] = (j == 0) ? 1.0 : 0.0;
        }
        return;
    }
    for (int j = 0; j < block_size; ++j) {
        out[j] = std::exp(-(double)j / (double)gamma);
    }
}

void ts_dpace_normalize_weights(double * weights, int block_size) {
    if (block_size <= 0) {
        return;
    }
    double sum = 0.0;
    for (int j = 0; j < block_size; ++j) {
        sum += weights[j];
    }
    if (sum <= 0.0) {
        return;
    }
    // Scale so weights sum to block_size (mean = 1.0)
    const double scale = (double)block_size / sum;
    for (int j = 0; j < block_size; ++j) {
        weights[j] *= scale;
    }
}

double ts_dpace_loss_from_probs(const float * probs,
                                const double * weights,
                                int block_size) {
    double loss = 0.0;
    for (int j = 0; j < block_size; ++j) {
        // Clamp to avoid log(0)
        double p = std::max(1e-10, (double)probs[j]);
        loss += weights[j] * (-std::log(p));
    }
    return loss;
}

// Extract softmax probability of token_id from a logit row.
static double softmax_prob_at(const float * logits, int n_vocab, int token_id) {
    if (token_id < 0 || token_id >= n_vocab) {
        return 1e-10;
    }
    // Numerically stable softmax: subtract max
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    double target_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        double e = std::exp((double)logits[i] - (double)max_logit);
        sum_exp += e;
        if (i == token_id) {
            target_exp = e;
        }
    }
    if (sum_exp <= 0.0) {
        return 1e-10;
    }
    return std::max(1e-10, target_exp / sum_exp);
}

double ts_dpace_loss(const float * logits,
                     const int32_t * target_tokens,
                     const double * weights,
                     int block_size,
                     int n_vocab) {
    double loss = 0.0;
    for (int j = 0; j < block_size; ++j) {
        double q = softmax_prob_at(logits + (size_t)j * n_vocab,
                                   n_vocab, target_tokens[j]);
        loss += weights[j] * (-std::log(q));
    }
    return loss;
}

double ts_dpace_loss_end_to_end(const float * logits,
                                const int32_t * target_tokens,
                                const float * acceptance_probs,
                                int block_size,
                                int n_vocab,
                                float alpha) {
    // Compute smoothed D-PACE weights (detached from gradient in the
    // training loop; here we just compute them as constants)
    std::vector<double> weights(block_size);
    ts_dpace_weights_smoothed(acceptance_probs, block_size, alpha, weights.data());
    ts_dpace_normalize_weights(weights.data(), block_size);

    return ts_dpace_loss(logits, target_tokens, weights.data(),
                         block_size, n_vocab);
}

ts_dpace_ab_result ts_dpace_ab_compare(const float * logits,
                                       const int32_t * target_tokens,
                                       const float * acceptance_probs,
                                       int block_size,
                                       int n_vocab,
                                       float alpha,
                                       float gamma) {
    ts_dpace_ab_result result = {};

    // D-PACE weights
    std::vector<double> dpace_w(block_size);
    ts_dpace_weights_smoothed(acceptance_probs, block_size, alpha, dpace_w.data());
    ts_dpace_normalize_weights(dpace_w.data(), block_size);

    // DFlash decay weights
    std::vector<double> decay_w(block_size);
    ts_dflash_decay_weights(block_size, gamma, decay_w.data());
    ts_dpace_normalize_weights(decay_w.data(), block_size);

    result.dpace_loss      = ts_dpace_loss(logits, target_tokens, dpace_w.data(),
                                           block_size, n_vocab);
    result.decay_loss      = ts_dpace_loss(logits, target_tokens, decay_w.data(),
                                           block_size, n_vocab);
    result.dpace_surrogate = ts_dpace_accepted_length_surrogate(acceptance_probs,
                                                                block_size);

    double dpace_sum = 0.0, decay_sum = 0.0;
    for (int j = 0; j < block_size; ++j) {
        dpace_sum += dpace_w[j];
        decay_sum += decay_w[j];
    }
    result.mean_dpace_weight = block_size > 0 ? dpace_sum / block_size : 0.0;
    result.mean_decay_weight = block_size > 0 ? decay_sum / block_size : 0.0;

    return result;
}
