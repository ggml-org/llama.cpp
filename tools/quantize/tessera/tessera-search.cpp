#include "tessera-search.h"
#include "tessera-linalg.h"
#include "tessera-lbfgs.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

// --- helpers ---

static float ts_mean_abs(const float * x, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        s += fabsf(x[i]);
    }
    return s / (float)n;
}

// ternarize_recon: sign(x) * (|x| >= threshold) * mean(|x|)
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

// C = A @ B, A is (M x K), B is (K x N), C is (M x N)
static void ts_matmul(const float * A, const float * B, float * C,
                      int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

// C = A^T @ B, A is (K x M) stored row-major, B is (K x N), C is (M x N)
static void ts_matmul_at(const float * A, const float * B, float * C,
                         int64_t K, int64_t M, int64_t N) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

// simple xorshift RNG
static uint32_t ts_rng_next(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float ts_rng_gaussian(uint32_t * state) {
    // Box-Muller
    uint32_t u1 = ts_rng_next(state) | 1;
    uint32_t u2 = ts_rng_next(state);
    float r1 = (float)u1 / 4294967296.0f;
    float r2 = (float)u2 / 4294967296.0f;
    return sqrtf(-2.0f * logf(r1)) * cosf(6.2831853f * r2);
}

// --- LRQ ---

int ts_train_lrq(const float * weights, int64_t out_dim, int64_t in_dim,
                 const ts_lrq_params * params, ts_lrq_result * result) {
    int64_t rank = params->rank;
    int64_t max_iters = params->max_iters > 0 ? params->max_iters : 200;
    float lr = params->lr > 0.0f ? params->lr : 1e-3f;
    float tol = params->tol > 0.0f ? params->tol : 1e-7f;
    uint32_t seed = params->seed ? params->seed : 42;

    if (rank < 1 || rank > std::min(out_dim, in_dim)) {
        return -1;
    }

    int64_t n_w = out_dim * in_dim;
    int64_t n_u = out_dim * rank;
    int64_t n_v = rank * in_dim;

    std::vector<float> U(n_u), V(n_v);
    std::vector<float> mU(n_u, 0.0f), vU(n_u, 0.0f);
    std::vector<float> mV(n_v, 0.0f), vV(n_v, 0.0f);
    std::vector<float> S(n_w), scaled(n_w), w_q(n_w);
    std::vector<float> d_scaled(n_w), d_s(n_w);
    std::vector<float> dU(n_u), dV(n_v);

    // init U, V so that S = U@V starts near 1 (identity scaling)
    uint32_t rng = seed;
    float noise = 0.01f / sqrtf((float)rank);
    for (int64_t i = 0; i < n_u; i++) {
        U[i] = ts_rng_gaussian(&rng) * noise;
    }
    for (int64_t i = 0; i < n_v; i++) {
        V[i] = ts_rng_gaussian(&rng) * noise;
    }
    // set first rank-1 component to produce S ≈ 1
    for (int64_t i = 0; i < out_dim; i++) {
        U[i * rank + 0] = 1.0f;
    }
    for (int64_t j = 0; j < in_dim; j++) {
        V[0 * in_dim + j] = 1.0f;
    }

    float beta1 = 0.9f, beta2 = 0.999f, eps_adam = 1e-8f;
    float prev_loss = 1e30f;

    for (int64_t it = 0; it < max_iters; it++) {
        // forward: S = U @ V
        ts_matmul(U.data(), V.data(), S.data(), out_dim, rank, in_dim);

        // scaled = W * S (element-wise)
        for (int64_t i = 0; i < n_w; i++) {
            scaled[i] = weights[i] * S[i];
        }

        // w_q = ternarize_recon(scaled)
        ts_ternarize_recon(scaled.data(), w_q.data(), n_w);

        // loss = mean((w_q - W)^2)
        float loss = 0.0f;
        for (int64_t i = 0; i < n_w; i++) {
            float d = w_q[i] - weights[i];
            loss += d * d;
        }
        loss /= (float)n_w;

        if (fabsf(prev_loss - loss) < tol) {
            prev_loss = loss;
            break;
        }
        prev_loss = loss;

        // backward (STE): d_loss/d_scaled = 2*(w_q - W) / n_w
        for (int64_t i = 0; i < n_w; i++) {
            d_scaled[i] = 2.0f * (w_q[i] - weights[i]) / (float)n_w;
        }

        // d_loss/d_S = d_scaled * W
        for (int64_t i = 0; i < n_w; i++) {
            d_s[i] = d_scaled[i] * weights[i];
        }

        // d_loss/d_U = d_s @ V^T: (out_dim x in_dim) @ (in_dim x rank)
        ts_matmul(d_s.data(), V.data(), dU.data(), out_dim, in_dim, rank);
        // note: V is (rank x in_dim), so V^T is (in_dim x rank)
        // ts_matmul with B=V treats V as (in_dim x rank) which is wrong
        // need: dU[i,j] = sum_k d_s[i,k] * V[j,k]
        for (int64_t i = 0; i < out_dim; i++) {
            for (int64_t j = 0; j < rank; j++) {
                float s = 0.0f;
                for (int64_t k = 0; k < in_dim; k++) {
                    s += d_s[i * in_dim + k] * V[j * in_dim + k];
                }
                dU[i * rank + j] = s;
            }
        }

        // d_loss/d_V = U^T @ d_s: (rank x out_dim) @ (out_dim x in_dim)
        for (int64_t i = 0; i < rank; i++) {
            for (int64_t j = 0; j < in_dim; j++) {
                float s = 0.0f;
                for (int64_t k = 0; k < out_dim; k++) {
                    s += U[k * rank + i] * d_s[k * in_dim + j];
                }
                dV[i * in_dim + j] = s;
            }
        }

        // Adam step
        float t = (float)(it + 1);
        float bc1 = 1.0f - powf(beta1, t);
        float bc2 = 1.0f - powf(beta2, t);
        for (int64_t i = 0; i < n_u; i++) {
            mU[i] = beta1 * mU[i] + (1.0f - beta1) * dU[i];
            vU[i] = beta2 * vU[i] + (1.0f - beta2) * dU[i] * dU[i];
            U[i] -= lr * (mU[i] / bc1) / (sqrtf(vU[i] / bc2) + eps_adam);
        }
        for (int64_t i = 0; i < n_v; i++) {
            mV[i] = beta1 * mV[i] + (1.0f - beta1) * dV[i];
            vV[i] = beta2 * vV[i] + (1.0f - beta2) * dV[i] * dV[i];
            V[i] -= lr * (mV[i] / bc1) / (sqrtf(vV[i] / bc2) + eps_adam);
        }
    }

    result->U = std::move(U);
    result->V = std::move(V);
    result->mse = prev_loss;
    result->rank = rank;
    result->n_iters = max_iters;
    return 0;
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

// --- CHAMP-Q ---

void ts_champq_sinkhorn(float * M, int64_t n, int64_t n_iters, float eps) {
    for (int64_t i = 0; i < n * n; i++) {
        M[i] = std::max(M[i], eps);
    }
    for (int64_t iter = 0; iter < n_iters; iter++) {
        // row normalization
        for (int64_t i = 0; i < n; i++) {
            float s = 0.0f;
            for (int64_t j = 0; j < n; j++) {
                s += M[i * n + j];
            }
            if (s > eps) {
                float inv = 1.0f / s;
                for (int64_t j = 0; j < n; j++) {
                    M[i * n + j] *= inv;
                }
            }
        }
        // column normalization
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                s += M[i * n + j];
            }
            if (s > eps) {
                float inv = 1.0f / s;
                for (int64_t i = 0; i < n; i++) {
                    M[i * n + j] *= inv;
                }
            }
        }
    }
}

// Hungarian algorithm (Kuhn-Munkres), O(n^3)
static void ts_hungarian(const float * cost, int32_t * assignment, int64_t n) {
    if (n == 0) {
        return;
    }
    const float INF = 1e30f;
    std::vector<float> u(n + 1, 0.0f), v(n + 1, 0.0f), minv(n + 1);
    std::vector<int64_t> p(n + 1, 0), way(n + 1, 0);
    std::vector<bool> used(n + 1);

    for (int64_t i = 1; i <= n; i++) {
        p[0] = i;
        int64_t j0 = 0;
        std::fill(minv.begin(), minv.end(), INF);
        std::fill(used.begin(), used.end(), false);

        while (true) {
            used[j0] = true;
            int64_t i0 = p[j0];
            float delta = INF;
            int64_t j1 = 0;
            for (int64_t j = 1; j <= n; j++) {
                if (used[j]) continue;
                float cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int64_t j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        while (true) {
            int64_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) break;
        }
    }

    for (int64_t j = 1; j <= n; j++) {
        if (p[j] != 0) {
            assignment[p[j] - 1] = (int32_t)(j - 1);
        }
    }
}

// greedy assignment: sort all (row, col) by descending M value
static void ts_greedy_assignment(const float * M, int32_t * assignment, int64_t n) {
    std::vector<int64_t> order(n * n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return M[a] > M[b];
    });

    std::vector<bool> col_used(n, false), row_used(n, false);
    std::fill(assignment, assignment + n, -1);

    for (int64_t idx : order) {
        int64_t r = idx / n;
        int64_t c = idx % n;
        if (!col_used[c] && !row_used[r]) {
            assignment[r] = (int32_t)c;
            col_used[c] = true;
            row_used[r] = true;
        }
    }
    // fill leftovers
    int64_t next_col = 0;
    for (int64_t i = 0; i < n; i++) {
        if (assignment[i] == -1) {
            while (next_col < n && col_used[next_col]) next_col++;
            assignment[i] = (int32_t)next_col;
            if (next_col < n) col_used[next_col] = true;
        }
    }
}

struct ts_champq_ctx {
    const float * W;
    int64_t       out_dim;
    int64_t       in_dim;
    float         binariness;
};

static float ts_champq_eval(const float * x, float * grad, int64_t n, void * ctx) {
    auto * c = (ts_champq_ctx *)ctx;
    int64_t K = c->in_dim;
    int64_t out_dim = c->out_dim;
    const float * W = c->W;
    float binariness = c->binariness;

    // M is (K x K), x is flattened M
    // W_perm = W @ M: (out_dim x K)
    std::vector<float> W_perm(out_dim * K);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t col = 0; col < K; col++) {
            float s = 0.0f;
            for (int64_t i = 0; i < K; i++) {
                s += W[r * K + i] * x[i * K + col];
            }
            W_perm[r * K + col] = s;
        }
    }

    float loss = 0.0f;
    if (K < 3) {
        // trivial: no interior columns
        if (binariness > 0.0f) {
            for (int64_t i = 0; i < n; i++) {
                loss += binariness * x[i] * (1.0f - x[i]);
                grad[i] = binariness * (1.0f - 2.0f * x[i]);
            }
        } else {
            memset(grad, 0, n * sizeof(float));
        }
        return loss;
    }

    // d2[r, c] = W_perm[r, c] - 2*W_perm[r, c+1] + W_perm[r, c+2], c in [0, K-3]
    // loss = sum d2^2
    std::vector<float> h(out_dim * K, 0.0f);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 0; c < K - 2; c++) {
            float d2 = W_perm[r * K + c] - 2.0f * W_perm[r * K + c + 1] + W_perm[r * K + c + 2];
            loss += d2 * d2;
            // h[r, c+1] += d2 (the center of the stencil)
            h[r * K + c + 1] = d2;
        }
    }

    // gradient: g[r, c] = 2 * (h[r, c-1] - 2*h[r, c] + h[r, c+1]) for interior
    // grad_M = W^T @ g
    std::vector<float> g(out_dim * K, 0.0f);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < K - 1; c++) {
            g[r * K + c] = 2.0f * (h[r * K + c - 1] - 2.0f * h[r * K + c] + h[r * K + c + 1]);
        }
    }

    // grad_M[i, k] = sum_r W[r, i] * g[r, k]
    for (int64_t i = 0; i < K; i++) {
        for (int64_t k = 0; k < K; k++) {
            float s = 0.0f;
            for (int64_t r = 0; r < out_dim; r++) {
                s += W[r * K + i] * g[r * K + k];
            }
            grad[i * K + k] = s;
        }
    }

    if (binariness > 0.0f) {
        for (int64_t i = 0; i < n; i++) {
            loss += binariness * x[i] * (1.0f - x[i]);
            grad[i] += binariness * (1.0f - 2.0f * x[i]);
        }
    }

    return loss;
}

struct ts_champq_project_ctx {
    int64_t K;
    int64_t sinkhorn_iters;
};

static void ts_champq_project(float * x, int64_t n, void * ctx) {
    auto * c = (ts_champq_project_ctx *)ctx;
    // clamp to non-negative
    for (int64_t i = 0; i < n; i++) {
        x[i] = std::max(x[i], 1e-12f);
    }
    ts_champq_sinkhorn(x, c->K, c->sinkhorn_iters, 1e-12f);
}

int ts_champq_compute(const float * weights, int64_t out_dim, int64_t in_dim,
                      const ts_champq_params * params, ts_champq_result * result) {
    int64_t max_iters = params->max_iters > 0 ? params->max_iters : 100;
    int64_t sinkhorn_iters = params->sinkhorn_iters > 0 ? params->sinkhorn_iters : 25;
    uint32_t seed = params->seed ? params->seed : 42;

    int64_t K = in_dim;
    int64_t n = K * K;

    // init M from L2-rank permutation
    std::vector<float> col_norms(K);
    for (int64_t j = 0; j < K; j++) {
        float s = 0.0f;
        for (int64_t i = 0; i < out_dim; i++) {
            float v = weights[i * in_dim + j];
            s += v * v;
        }
        col_norms[j] = sqrtf(s);
    }
    // argsort descending
    std::vector<int64_t> rank_perm(K);
    std::iota(rank_perm.begin(), rank_perm.end(), 0);
    std::sort(rank_perm.begin(), rank_perm.end(), [&](int64_t a, int64_t b) {
        return col_norms[a] > col_norms[b];
    });

    std::vector<float> M(n, 0.0f);
    for (int64_t i = 0; i < K; i++) {
        M[i * K + rank_perm[i]] = 1.0f;
    }

    // baseline smoothness (identity perm)
    float baseline_smooth = 0.0f;
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < K - 1; c++) {
            float d2 = weights[r * K + c - 1] - 2.0f * weights[r * K + c] + weights[r * K + c + 1];
            baseline_smooth += d2 * d2;
        }
    }

    if (params->use_lbfgs) {
        ts_champq_ctx eval_ctx = { weights, out_dim, in_dim, 1.0f };
        ts_champq_project_ctx proj_ctx = { K, sinkhorn_iters };

        ts_pgd_minimize(M.data(), n, ts_champq_eval, &eval_ctx,
                        ts_champq_project, &proj_ctx,
                        max_iters, 0.1f, 1e-6f);
    }

    // project to permutation
    std::vector<int32_t> perm(K);
    if (K <= 512) {
        // Hungarian on negative M (minimize cost = maximize M)
        std::vector<float> cost(n);
        for (int64_t i = 0; i < n; i++) {
            cost[i] = -M[i];
        }
        ts_hungarian(cost.data(), perm.data(), K);
    } else {
        ts_greedy_assignment(M.data(), perm.data(), K);
    }

    // compute final smoothness
    float final_smooth = 0.0f;
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 1; c < K - 1; c++) {
            float v0 = weights[r * K + perm[c - 1]];
            float v1 = weights[r * K + perm[c]];
            float v2 = weights[r * K + perm[c + 1]];
            float d2 = v0 - 2.0f * v1 + v2;
            final_smooth += d2 * d2;
        }
    }

    result->perm = std::move(perm);
    result->smoothness = final_smooth;
    result->mse_improvement = baseline_smooth > 0.0f
        ? (baseline_smooth - final_smooth) / baseline_smooth
        : 0.0f;
    return 0;
}

void ts_champq_apply(const float * W, const int32_t * perm,
                     float * W_perm, int64_t out_dim, int64_t in_dim) {
    for (int64_t i = 0; i < out_dim; i++) {
        for (int64_t j = 0; j < in_dim; j++) {
            W_perm[i * in_dim + j] = W[i * in_dim + perm[j]];
        }
    }
}

void ts_champq_invert(const int32_t * perm, int32_t * inv, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        inv[perm[i]] = (int32_t)i;
    }
}

// --- Regime router ---

ts_expert_id ts_route_expert(float kurtosis, float eff_rank,
                             const char * family) {
    (void)family;
    if (kurtosis > 10.0f) {
        return TS_EXPERT_DARTQUANT;
    }
    if (eff_rank < 0.3f) {
        return TS_EXPERT_FLRQ;
    }
    return TS_EXPERT_AWQ;
}
