#include "tessera-flrq.h"
#include "tessera-linalg.h"

#include <cmath>
#include <algorithm>

// --- FLRQ ---

static float ts_spectral_compactness(const float * S, int64_t k) {
    // rho = 1 - H/H_max, H = -sum(p * log(p)), p_i = s_i / sum(s)
    float sum = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        sum += S[i];
    }
    if (sum <= 0.0f) {
        return 0.0f;
    }
    float H = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        float p = S[i] / sum;
        if (p > 1e-12f) {
            H -= p * logf(p);
        }
    }
    float H_max = logf((float)k);
    if (H_max <= 0.0f) {
        return 1.0f;
    }
    return 1.0f - H / H_max;
}

int ts_train_flrq(const float * weights, int64_t out_dim, int64_t in_dim,
                  const ts_flrq_params * params, ts_flrq_result * result) {
    int64_t max_rank = params->max_rank > 0 ? params->max_rank : 16;
    float rho_thresh = params->rho_thresh > 0.0f ? params->rho_thresh : 0.8f;
    int64_t oversample = params->sketch_oversample > 0 ? params->sketch_oversample : 10;
    uint32_t seed = params->seed ? params->seed : 42;

    max_rank = std::min(max_rank, std::min(out_dim, in_dim));
    if (max_rank < 1) {
        return -1;
    }

    // compute top singular values via sketch + power iteration
    int64_t sketch_k = std::min(max_rank + oversample, in_dim);
    std::vector<float> sketch(out_dim * sketch_k);
    ts_linalg_sketch(weights, sketch.data(), out_dim, in_dim, sketch_k, seed);

    std::vector<float> U_sketch(out_dim * sketch_k);
    std::vector<float> S_vals(sketch_k);
    std::vector<float> V_sketch(in_dim * sketch_k);
    ts_linalg_svd_topk(sketch.data(), U_sketch.data(), S_vals.data(),
                       V_sketch.data(), out_dim, sketch_k, sketch_k, 20, seed);

    // spectral compactness gate
    float rho = ts_spectral_compactness(S_vals.data(), max_rank);
    if (rho < rho_thresh) {
        // spectrum too flat for low-rank; still return best effort
    }

    // select rank: smallest r where reconstruction error is acceptable
    int64_t chosen_rank = max_rank;
    float w_fro2 = 0.0f;
    for (int64_t i = 0; i < out_dim * in_dim; i++) {
        w_fro2 += weights[i] * weights[i];
    }
    w_fro2 += 1e-12f;

    for (int64_t r = 1; r <= max_rank; r++) {
        // reconstruction error from truncated SVD: sum of remaining singular values^2
        float tail = 0.0f;
        for (int64_t i = r; i < sketch_k; i++) {
            tail += S_vals[i] * S_vals[i];
        }
        float rel_mse = tail / w_fro2;
        if (rel_mse < 1e-3f) {
            chosen_rank = r;
            break;
        }
    }

    // extract U (out_dim x chosen_rank) and V (chosen_rank x in_dim)
    // U = U_sketch[:, :chosen_rank]
    // V = diag(S) @ V_sketch[:, :chosen_rank]^T
    std::vector<float> U(out_dim * chosen_rank);
    std::vector<float> V(chosen_rank * in_dim);

    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < chosen_rank; j++) {
            U[i * chosen_rank + j] = U_sketch[i * sketch_k + j];
        }
    }
    // V[j, k] = S[j] * V_sketch[k, j]
    for (int64_t j = 0; j < chosen_rank; j++) {
        for (int64_t k = 0; k < in_dim; k++) {
            V[j * in_dim + k] = S_vals[j] * V_sketch[k * sketch_k + j];
        }
    }

    // compute final MSE
    float mse = 0.0f;
    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < in_dim; j++) {
            float recon = 0.0f;
            for (int64_t r = 0; r < chosen_rank; r++) {
                recon += U[i * chosen_rank + r] * V[r * in_dim + j];
            }
            float d = weights[i * in_dim + j] - recon;
            mse += d * d;
        }
    }
    mse /= (float)(out_dim * in_dim);

    result->U = std::move(U);
    result->V = std::move(V);
    result->rank = chosen_rank;
    result->mse = mse;
    result->spectral_compactness = rho;
    return 0;
}
