//
// tessera-peqat.cpp
//
// PE-QAT trainer. Port of tools/tessera/pe_qat.py: SmoothQuant smoothing,
// LoRA adapters on a frozen ternary fake-quantized weight, STE backward,
// AdamW. The base weight is ternarized once and frozen; the LoRA delta is
// applied in full precision on top (Y = (W_quant + B@A) @ x_s), so the
// adapters receive a clean gradient and the ternary quantizer is never
// differentiated (STE = identity on a frozen constant). Only A and B are
// trained; the SmoothQuant scales are fixed after initialization.
//

#include "tessera-peqat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// PRNG - xorshift32
// ---------------------------------------------------------------------------

struct ts_peqat_rng {
    uint32_t s;

    void init(uint32_t seed) {
        s = seed ? seed : 1;
    }

    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }

    float uniform() {
        return (float)(next() >> 8) * (1.0f / 16777216.0f);
    }

    float gauss(float sigma) {
        float u1 = std::max(uniform(), 1e-10f);
        float u2 = uniform();
        return sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    }
};

// ---------------------------------------------------------------------------
// SmoothQuant per-channel scales
// ---------------------------------------------------------------------------

// s_j = max|X[:, j]|^alpha / max|W[:, j]|^(1 - alpha), clipped to [1e-4, 1e4].
static void ts_peqat_smooth_init(const float * weights, const float * calib_X,
                                 float alpha, float * s,
                                 int64_t out_dim, int64_t in_dim, int64_t n_tokens) {
    const float eps = 1e-8f;
    for (int64_t j = 0; j < in_dim; j++) {
        float x_max = 0.0f;
        for (int64_t t = 0; t < n_tokens; t++) {
            x_max = std::max(x_max, std::fabs(calib_X[t * in_dim + j]));
        }
        float w_max = 0.0f;
        for (int64_t o = 0; o < out_dim; o++) {
            w_max = std::max(w_max, std::fabs(weights[o * in_dim + j]));
        }
        float num = powf(std::max(x_max, eps), alpha);
        float den = powf(std::max(w_max, eps), 1.0f - alpha);
        float v = num / den;
        s[j] = std::min(std::max(v, 1e-4f), 1e4f);
    }
}

// ---------------------------------------------------------------------------
// Ternary fake-quant (per output channel), forward only; STE in backward
// ---------------------------------------------------------------------------

// threshold = mean|row|, ternary = sign if |w| >= threshold else 0,
// reconstruction scale = mean of |w| over the non-zero set (TWN).
static void ts_peqat_ternarize(const float * W_s, float * W_q,
                               int64_t out_dim, int64_t in_dim) {
    for (int64_t o = 0; o < out_dim; o++) {
        const float * row = W_s + o * in_dim;
        float * out = W_q + o * in_dim;

        // float32 accumulation to match Python reference bit-exactly
        float abs_sum = 0.0f;
        for (int64_t j = 0; j < in_dim; j++) {
            abs_sum += std::fabs(row[j]);
        }
        float threshold = (in_dim > 0) ? (abs_sum / (float)in_dim) : 0.0f;

        double mag_sum = 0.0;
        int64_t nz = 0;
        for (int64_t j = 0; j < in_dim; j++) {
            if (std::fabs(row[j]) >= threshold) {
                mag_sum += (double)std::fabs(row[j]);
                nz++;
            }
        }
        float scale = (nz > 0) ? (float)(mag_sum / (double)nz) : 0.0f;

        for (int64_t j = 0; j < in_dim; j++) {
            float t = 0.0f;
            if (std::fabs(row[j]) >= threshold) {
                t = (row[j] > 0.0f) ? 1.0f : -1.0f;
            }
            out[j] = scale * t;
        }
    }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

struct ts_peqat_adamw {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float wd;
    int64_t step;
};

static void ts_peqat_adamw_step(std::vector<float> & p, const std::vector<float> & g,
                                std::vector<float> & m, std::vector<float> & v,
                                const ts_peqat_adamw & opt) {
    float b1c = 1.0f - powf(opt.beta1, (float)opt.step);
    float b2c = 1.0f - powf(opt.beta2, (float)opt.step);
    for (size_t i = 0; i < p.size(); i++) {
        m[i] = opt.beta1 * m[i] + (1.0f - opt.beta1) * g[i];
        v[i] = opt.beta2 * v[i] + (1.0f - opt.beta2) * g[i] * g[i];
        float mh = m[i] / b1c;
        float vh = v[i] / b2c;
        float upd = mh / (sqrtf(vh) + opt.eps) + opt.wd * p[i];
        p[i] -= opt.lr * upd;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int ts_peqat_train(const float * weights,
                   const float * calib_X,
                   const float * ref_output,
                   int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                   const ts_peqat_params * params,
                   ts_peqat_result * result) {
    if (!weights || !calib_X || !ref_output || !result ||
        out_dim <= 0 || in_dim <= 0 || n_tokens <= 0) {
        return -1;
    }

    ts_peqat_params defs;
    defs.lora_rank    = 8;
    defs.smooth_alpha = 0.5f;
    defs.lr           = 1e-4f;
    defs.weight_decay = 0.01f;
    defs.beta1        = 0.9f;
    defs.beta2        = 0.999f;
    defs.eps          = 1e-8f;
    defs.max_epochs   = 10;
    defs.tol          = 1e-6f;
    defs.seed         = 0;
    defs.verbose      = false;
    if (params) {
        defs = *params;
    }
    const int64_t rank = std::max(defs.lora_rank, (int64_t)1);
    const int64_t max_epochs = std::max(defs.max_epochs, (int64_t)0);

    ts_peqat_rng rng;
    rng.init(defs.seed);

    // SmoothQuant scales (fixed after init).
    std::vector<float> s((size_t)in_dim);
    ts_peqat_smooth_init(weights, calib_X, defs.smooth_alpha, s.data(),
                         out_dim, in_dim, n_tokens);

    // Smoothed activations x_s = X / s (constant across epochs).
    std::vector<float> x_s((size_t)(n_tokens * in_dim));
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t j = 0; j < in_dim; j++) {
            x_s[(size_t)(t * in_dim + j)] = calib_X[t * in_dim + j] / s[(size_t)j];
        }
    }

    // Frozen ternary fake-quant of the smoothed base weight (computed once).
    std::vector<float> W_s((size_t)(out_dim * in_dim));
    for (int64_t o = 0; o < out_dim; o++) {
        for (int64_t j = 0; j < in_dim; j++) {
            W_s[(size_t)(o * in_dim + j)] = weights[o * in_dim + j] * s[(size_t)j];
        }
    }
    std::vector<float> W_q((size_t)(out_dim * in_dim));
    ts_peqat_ternarize(W_s.data(), W_q.data(), out_dim, in_dim);

    // LoRA init: A ~ N(0, 1/rank), B = 0 (initial delta is zero).
    const float a_sigma = sqrtf(1.0f / (float)rank);
    std::vector<float> A((size_t)(in_dim * rank));
    std::vector<float> B((size_t)(rank * out_dim), 0.0f);
    for (int64_t j = 0; j < in_dim; j++) {
        for (int64_t r = 0; r < rank; r++) {
            A[(size_t)(j * rank + r)] = rng.gauss(a_sigma);
        }
    }

    ts_peqat_adamw opt;
    opt.lr    = defs.lr;
    opt.beta1 = defs.beta1;
    opt.beta2 = defs.beta2;
    opt.eps   = defs.eps;
    opt.wd    = defs.weight_decay;
    opt.step  = 0;

    std::vector<float> mA(A.size(), 0.0f), vA(A.size(), 0.0f);
    std::vector<float> mB(B.size(), 0.0f), vB(B.size(), 0.0f);
    std::vector<float> gA(A.size(), 0.0f), gB(B.size(), 0.0f);

    std::vector<float> y((size_t)(n_tokens * out_dim));
    std::vector<float> dW((size_t)(out_dim * in_dim));

    const float inv_n = 1.0f / (float)(n_tokens * out_dim);
    float loss = 0.0f;
    float prev_loss = 0.0f;
    int64_t epoch = 0;

    for (epoch = 0; epoch < max_epochs; epoch++) {
        // ---- forward: y[t, o] = sum_j (W_q[o, j] + delta[o, j]) * x_s[t, j] ----
        // delta[o, j] = sum_r B[r, o] * A[j, r]
        double loss_acc = 0.0;
        for (int64_t t = 0; t < n_tokens; t++) {
            for (int64_t o = 0; o < out_dim; o++) {
                float acc = 0.0f;
                for (int64_t j = 0; j < in_dim; j++) {
                    float d = 0.0f;
                    for (int64_t r = 0; r < rank; r++) {
                        d += B[(size_t)(r * out_dim + o)] * A[(size_t)(j * rank + r)];
                    }
                    acc += (W_q[(size_t)(o * in_dim + j)] + d) * x_s[(size_t)(t * in_dim + j)];
                }
                y[(size_t)(t * out_dim + o)] = acc;
                float diff = acc - ref_output[t * out_dim + o];
                loss_acc += (double)diff * (double)diff;
            }
        }
        prev_loss = loss;
        loss = (float)(loss_acc / (double)(n_tokens * out_dim));

        if (defs.verbose) {
            printf("  epoch %4lld  loss=%.6e\n", (long long)epoch, loss);
        }
        if (epoch > 0 && std::fabs(prev_loss - loss) < defs.tol) {
            epoch++;
            break;
        }

        // ---- backward ----
        // d_delta[o, j] = sum_t 2*diff[t, o]/N * x_s[t, j]
        for (int64_t o = 0; o < out_dim; o++) {
            for (int64_t j = 0; j < in_dim; j++) {
                float acc = 0.0f;
                for (int64_t t = 0; t < n_tokens; t++) {
                    float diff = y[(size_t)(t * out_dim + o)] - ref_output[t * out_dim + o];
                    acc += diff * x_s[(size_t)(t * in_dim + j)];
                }
                dW[(size_t)(o * in_dim + j)] = 2.0f * inv_n * acc;
            }
        }

        // grad_B[r, o] = sum_j d_delta[o, j] * A[j, r]
        // grad_A[j, r] = sum_o d_delta[o, j] * B[r, o]
        std::fill(gA.begin(), gA.end(), 0.0f);
        std::fill(gB.begin(), gB.end(), 0.0f);
        for (int64_t o = 0; o < out_dim; o++) {
            for (int64_t j = 0; j < in_dim; j++) {
                float dw = dW[(size_t)(o * in_dim + j)];
                for (int64_t r = 0; r < rank; r++) {
                    gB[(size_t)(r * out_dim + o)] += dw * A[(size_t)(j * rank + r)];
                    gA[(size_t)(j * rank + r)]   += dw * B[(size_t)(r * out_dim + o)];
                }
            }
        }

        // ---- AdamW ----
        opt.step++;
        ts_peqat_adamw_step(A, gA, mA, vA, opt);
        ts_peqat_adamw_step(B, gB, mB, vB, opt);
    }

    result->lora_A    = std::move(A);
    result->lora_B    = std::move(B);
    result->smooth_s  = std::move(s);
    result->final_loss = loss;
    result->epochs_run = epoch;

    return 0;
}
