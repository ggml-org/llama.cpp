#include "tessera-lrq.h"

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
// (shared reconstruction primitive; also used by tessera-dartquant)
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
