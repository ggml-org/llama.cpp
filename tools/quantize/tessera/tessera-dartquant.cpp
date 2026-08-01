#include "tessera-dartquant.h"
#include "tessera-linalg.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// --- helpers ---

static float ts_mean_abs(const float * x, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        s += fabsf(x[i]);
    }
    return s / (float)n;
}

// ternarize_recon: sign(x) * (|x| >= threshold) * mean(|x|)
// (shared reconstruction primitive; also used by tessera-lrq)
static void ts_ternarize_recon(const float * x, float * out, int64_t n) {
    float thresh = ts_mean_abs(x, n);
    float scale = thresh;
    if (thresh <= 0.0f) {
        memset(out, 0, n * sizeof(float));
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        if (fabsf(v) >= thresh) {
            out[i] = (v > 0.0f ? 1.0f : -1.0f) * scale;
        } else {
            out[i] = 0.0f;
        }
    }
}

// --- DartQuant ---

static float ts_whip_loss(const float * W_rot, int64_t out_dim, int64_t in_dim) {
    // coefficient of variation of per-row L2 norms
    std::vector<float> norms(out_dim);
    for (int64_t i = 0; i < out_dim; i++) {
        float s = 0.0f;
        for (int64_t j = 0; j < in_dim; j++) {
            float v = W_rot[i * in_dim + j];
            s += v * v;
        }
        norms[i] = sqrtf(std::max(s, 1e-24f));
    }
    float mean = 0.0f;
    for (int64_t i = 0; i < out_dim; i++) {
        mean += norms[i];
    }
    mean /= (float)out_dim;
    if (mean < 1e-12f) {
        return 0.0f;
    }
    float var = 0.0f;
    for (int64_t i = 0; i < out_dim; i++) {
        float d = norms[i] - mean;
        var += d * d;
    }
    var /= (float)out_dim;
    return sqrtf(var) / mean;
}

// gradient of whip loss w.r.t. R (K x K)
static void ts_whip_grad(const float * W, const float * R, float * grad,
                         int64_t out_dim, int64_t K) {
    // W_rot = W @ R^T (out_dim x K)
    std::vector<float> W_rot(out_dim * K);
    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < K; j++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += W[i * K + k] * R[j * K + k];
            }
            W_rot[i * K + j] = s;
        }
    }

    std::vector<float> norms(out_dim);
    for (int64_t i = 0; i < out_dim; i++) {
        float s = 0.0f;
        for (int64_t j = 0; j < K; j++) {
            float v = W_rot[i * K + j];
            s += v * v;
        }
        norms[i] = sqrtf(std::max(s, 1e-24f));
    }
    float mean_n = 0.0f;
    for (int64_t i = 0; i < out_dim; i++) {
        mean_n += norms[i];
    }
    mean_n /= (float)out_dim;
    if (mean_n < 1e-12f) {
        memset(grad, 0, K * K * sizeof(float));
        return;
    }

    // diff_o = (norm_o - mean) / norm_o
    // dWp[o, j] = diff_o * W_rot[o, j]
    // grad = W^T @ dWp / (out_dim * mean_n)
    std::vector<float> dWp(out_dim * K);
    for (int64_t o = 0; o < out_dim; o++) {
        float diff = (norms[o] - mean_n) / std::max(norms[o], 1e-12f);
        for (int64_t j = 0; j < K; j++) {
            dWp[o * K + j] = diff * W_rot[o * K + j];
        }
    }

    // grad[i, j] = sum_o W[o, i] * dWp[o, j] / (out_dim * mean_n)
    float inv = 1.0f / ((float)out_dim * mean_n);
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = 0; j < K; j++) {
            float s = 0.0f;
            for (int64_t o = 0; o < out_dim; o++) {
                s += W[o * K + i] * dWp[o * K + j];
            }
            grad[i * K + j] = s * inv;
        }
    }
}

int ts_dartquant_qr_orth(const float * weights, int64_t out_dim, int64_t in_dim,
                         const ts_dartquant_params * params,
                         ts_dartquant_result * result) {
    int64_t K = params->block_size > 0 ? params->block_size : 64;
    int64_t max_iters = params->max_iters > 0 ? params->max_iters : 50;
    float lr = params->lr > 0.0f ? params->lr : 0.1f;

    if (K > in_dim) {
        K = in_dim;
    }

    int64_t n_blocks = in_dim / K;
    if (n_blocks < 1) {
        return -1;
    }

    // optimize one block rotation (all blocks share the same R for simplicity)
    std::vector<float> R(K * K);
    if (params->seed) {
        ts_linalg_random_orthogonal(R.data(), K, params->seed);
    } else {
        // identity
        memset(R.data(), 0, K * K * sizeof(float));
        for (int64_t i = 0; i < K; i++) {
            R[i * K + i] = 1.0f;
        }
    }

    // use first block of weights for optimization
    std::vector<float> W_block(out_dim * K);
    for (int64_t i = 0; i < out_dim; i++) {
        memcpy(&W_block[i * K], &weights[i * in_dim], K * sizeof(float));
    }

    std::vector<float> W_rot(out_dim * K);
    std::vector<float> grad(K * K);

    float best_whip = 1e30f;
    std::vector<float> best_R(R);

    for (int64_t it = 0; it < max_iters; it++) {
        // W_rot = W_block @ R^T
        for (int64_t i = 0; i < out_dim; i++) {
            for (int64_t j = 0; j < K; j++) {
                float s = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    s += W_block[i * K + k] * R[j * K + k];
                }
                W_rot[i * K + j] = s;
            }
        }

        float whip = ts_whip_loss(W_rot.data(), out_dim, K);
        if (whip < best_whip) {
            best_whip = whip;
            best_R = R;
        }

        ts_whip_grad(W_block.data(), R.data(), grad.data(), out_dim, K);

        // negate gradient for descent (qr_orth_step adds lr * G)
        for (int64_t i = 0; i < K * K; i++) {
            grad[i] = -grad[i];
        }

        ts_linalg_qr_orth_step(R.data(), grad.data(), lr, K, K);
    }

    R = best_R;

    // compute final metrics
    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < K; j++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += W_block[i * K + k] * R[j * K + k];
            }
            W_rot[i * K + j] = s;
        }
    }
    float final_whip = ts_whip_loss(W_rot.data(), out_dim, K);

    // MSE: ||W_rot - ternarize_recon(W_rot)||^2
    std::vector<float> w_q(out_dim * K);
    ts_ternarize_recon(W_rot.data(), w_q.data(), out_dim * K);
    float mse = 0.0f;
    for (int64_t i = 0; i < out_dim * K; i++) {
        float d = W_rot[i] - w_q[i];
        mse += d * d;
    }
    mse /= (float)(out_dim * K);

    result->R = std::move(R);
    result->whip_loss = final_whip;
    result->mse = mse;
    result->n_iters = max_iters;
    return 0;
}

void ts_dartquant_apply(const float * W, const float * R,
                        float * W_rot, int64_t out_dim, int64_t in_dim,
                        int64_t block_size) {
    int64_t n_blocks = in_dim / block_size;
    for (int64_t b = 0; b < n_blocks; b++) {
        int64_t col_off = b * block_size;
        for (int64_t i = 0; i < out_dim; i++) {
            for (int64_t j = 0; j < block_size; j++) {
                float s = 0.0f;
                for (int64_t k = 0; k < block_size; k++) {
                    // W_rot = W @ R^T
                    s += W[i * in_dim + col_off + k] * R[j * block_size + k];
                }
                W_rot[i * in_dim + col_off + j] = s;
            }
        }
    }
    // handle remainder
    int64_t rem = in_dim - n_blocks * block_size;
    if (rem > 0) {
        int64_t col_off = n_blocks * block_size;
        for (int64_t i = 0; i < out_dim; i++) {
            for (int64_t j = 0; j < rem; j++) {
                W_rot[i * in_dim + col_off + j] = W[i * in_dim + col_off + j];
            }
        }
    }
}
