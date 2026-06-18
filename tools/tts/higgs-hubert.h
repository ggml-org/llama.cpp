#pragma once

#include "higgs-gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// NOTE: This file is included from within namespace higgs_audio in higgs-codec.h
// Do NOT open a new namespace block here.

// ============================================================================
// Higgs Audio v3 – native C++ HuBERT / semantic encoder
// ============================================================================

constexpr int  HUBERT_HIDDEN_SIZE       = 768;
constexpr int  HUBERT_NUM_LAYERS        = 12;
constexpr int  HUBERT_NUM_HEADS         = 12;
constexpr int  HUBERT_HEAD_DIM          = 64;
constexpr int  HUBERT_INTERMEDIATE_SIZE = 3072;
constexpr int  HUBERT_FEAT_CHANNELS     = 512;
constexpr int  HUBERT_NUM_FEAT_LAYERS   = 7;
constexpr int  HUBERT_FEAT_STRIDES[7]   = {5, 2, 2, 2, 2, 2, 2};
constexpr int  HUBERT_FEAT_KERNELS[7]   = {10, 3, 3, 3, 3, 2, 2};
constexpr int  HUBERT_POS_CONV_KERNEL   = 128;
constexpr int  HUBERT_POS_CONV_GROUPS   = 16;
constexpr float HUBERT_LN_EPS           = 1e-5f;
constexpr int  HUBERT_SEMANTIC_DOWNSAMPLE = 2;

// ============================================================================
// Activation helpers
// ============================================================================

static inline float gelu(float x) {
    float c = 0.044715f;
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + c * x3); // sqrt(2/π) ≈ 0.79788456
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float elu(float x) {
    return (x > 0.0f) ? x : (expf(x) - 1.0f);
}

static inline int int_gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static inline std::vector<float> resample_kaiser(
        const std::vector<float>& input,
        double src_rate, double dst_rate)
{
    if (input.empty()) return input;

    int orig_freq = (int)src_rate;
    int new_freq  = (int)dst_rate;
    if (orig_freq == new_freq) return input;

    int g = int_gcd(orig_freq, new_freq);
    int P = orig_freq / g;
    int Q = new_freq  / g;

    const double rolloff = 0.99;
    const int L = 6;
    double base_freq = (double)std::min(P, Q) * rolloff;
    int width = (int)std::ceil((double)L * (double)P / base_freq);

    // Build kernel [Q, 1, 2*width + P]
    int kernel_len = 2 * width + P;
    std::vector<float> kernel((size_t)Q * (size_t)kernel_len, 0.0f);

    double scale = (double)base_freq / (double)P;
    const double pi = 3.14159265358979323846;
    const double pi_div_L2 = pi / (double)L / 2.0;

    for (int iq = 0; iq < Q; ++iq) {
        for (int ik = 0; ik < kernel_len; ++ik) {
            int idx_val = ik - width;
            // t = (iq / new_freq - idx_val / orig_freq) * base_freq ... wait
            // Actually: t = (iq / Q - idx / P) * base_freq ... hmm
            // Let me re-derive:
            // idx = arange(-width, width+P) / P
            // t = arange(0, -Q, -1)[:, None, None] / Q + idx
            //   = (starting_pos / Q) + (idx_range / P)
            // For position iq (0..Q-1), t = (iq * -1)/Q? No...
            // The arange(0, -Q, -1) gives positions [0, -1, -2, ..., -(Q-1)]
            // So for output channel q, t = (-q) / Q + idx_k / P = idx_k/P - q/Q
            // Wait: t = arange(0, -Q, -1) / Q + idx
            // arange(0, -Q, -1) = [0, -1, -2, ..., -(Q-1)]
            // So t[q] = -q/Q + idx[k]/P

            double t = (double)(-iq) / (double)Q + (double)idx_val / (double)P;
            t *= (double)base_freq;
            t = std::max(-(double)L, std::min((double)L, t));

            // Hann window: cos(t * pi / L / 2)^2
            double window = std::cos(t * pi_div_L2);
            window *= window;

            // sinc
            double sinc_val;
            double t_pi = t * pi;
            if (std::abs(t_pi) < 1e-9)
                sinc_val = 1.0;
            else
                sinc_val = std::sin(t_pi) / t_pi;

            size_t k_idx = (size_t)iq * (size_t)kernel_len + (size_t)ik;
            kernel[k_idx] = (float)(sinc_val * window * scale);
        }
    }

    // Pad input: (width, width + P) on each side
    size_t padded_len = input.size() + (size_t)width + (size_t)(width + P);
    std::vector<float> padded(padded_len, 0.0f);
    std::memcpy(padded.data() + (size_t)width, input.data(), input.size() * sizeof(float));

    // conv1d: in_channels=1, out_channels=Q, kernel_size=kernel_len, stride=P
    size_t conv_out_len = 0;
    if (padded_len >= (size_t)kernel_len)
        conv_out_len = (padded_len - (size_t)kernel_len) / (size_t)P + 1;
    else
        conv_out_len = 0;

    size_t target_len = std::max<size_t>(1, (size_t)std::ceil((double)input.size() * (double)Q / (double)P));
    std::vector<float> output(conv_out_len * (size_t)Q, 0.0f);

    for (size_t n = 0; n < conv_out_len; ++n) {
        for (int q = 0; q < Q; ++q) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_len; ++k) {
                size_t in_pos = n * (size_t)P + (size_t)k;
                size_t k_idx = (size_t)q * (size_t)kernel_len + (size_t)k;
                sum += padded[in_pos] * kernel[k_idx];
            }
            output[n * (size_t)Q + (size_t)q] = sum;
        }
    }

    output.resize(target_len);
    return output;
}

// ============================================================================
// Layer norm (tokens-first: [tokens, dim])
// ============================================================================

static inline std::vector<float> layer_norm_tokens(
        const std::vector<float>& x,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int dim)
{
    int n = (int)x.size() / dim;
    std::vector<float> out(x.size());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < n; ++t) {
        size_t off = (size_t)t * (size_t)dim;
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < dim; ++d) mean += x[off + (size_t)d];
        mean /= (float)dim;
        for (int d = 0; d < dim; ++d) {
            float diff = x[off + (size_t)d] - mean;
            var += diff * diff;
        }
        var = var / (float)dim + HUBERT_LN_EPS;
        float rsqrt = 1.0f / sqrtf(var);
        for (int d = 0; d < dim; ++d)
            out[off + (size_t)d] = (x[off + (size_t)d] - mean) * rsqrt * weight[(size_t)d] + bias[(size_t)d];
    }
    return out;
}

// ============================================================================
// GroupNorm 1D (channels-first: [channels, length])
// When groups==1: normalize globally (one mean/var over all elements)
// ============================================================================

static inline std::vector<float> group_norm_1d(
        const std::vector<float>& x,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int channels, int length, int num_groups)
{
    size_t total = (size_t)channels * (size_t)length;
    int group_size = channels / num_groups;
    std::vector<float> out(total);

    for (int g = 0; g < num_groups; ++g) {
        int c_start = g * group_size;
        int c_end   = c_start + group_size;
        size_t g_elems = (size_t)group_size * (size_t)length;

        float mean = 0.0f;
        for (int c = c_start; c < c_end; ++c)
            for (int t = 0; t < length; ++t)
                mean += x[(size_t)c * (size_t)length + (size_t)t];
        mean /= (float)g_elems;

        float var = 0.0f;
        for (int c = c_start; c < c_end; ++c) {
            for (int t = 0; t < length; ++t) {
                float d = x[(size_t)c * (size_t)length + (size_t)t] - mean;
                var += d * d;
            }
        }
        var = var / (float)g_elems + HUBERT_LN_EPS;
        float rsqrt = 1.0f / sqrtf(var);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int c = c_start; c < c_end; ++c) {
            for (int t = 0; t < length; ++t) {
                size_t idx = (size_t)c * (size_t)length + (size_t)t;
                out[idx] = (x[idx] - mean) * rsqrt * weight[(size_t)c] + bias[(size_t)c];
            }
        }
    }
    return out;
}

// ============================================================================
// 1D Convolution (channels-first, no padding – valid only)
// ============================================================================

static inline std::vector<float> conv1d_valid(
        const std::vector<float>& x,     // [in_c, in_len]
        const std::vector<float>& weight, // [out_c, in_c, kernel]
        const std::vector<float>& bias,   // [out_c] (may be empty for no bias)
        int in_c, int in_len, int out_c, int kernel, int stride)
{
    int out_len = (in_len - kernel) / stride + 1;
    std::vector<float> out((size_t)out_c * (size_t)out_len, 0.0f);
    bool has_bias = !bias.empty();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < out_c; ++oc) {
        for (int ot = 0; ot < out_len; ++ot) {
            float v = has_bias ? bias[(size_t)oc] : 0.0f;
            int it = ot * stride;
            for (int ic = 0; ic < in_c; ++ic) {
                size_t w_base = ((size_t)oc * (size_t)in_c + (size_t)ic) * (size_t)kernel;
                for (int k = 0; k < kernel; ++k)
                    v += weight[w_base + (size_t)k] * x[(size_t)ic * (size_t)in_len + (size_t)(it + k)];
            }
            out[(size_t)oc * (size_t)out_len + (size_t)ot] = v;
        }
    }
    return out;
}

// ============================================================================
// Linear layer (tokens-first: [tokens, in_dim] → [tokens, out_dim])
// ============================================================================

static inline std::vector<float> linear(
        const std::vector<float>& x,
        const std::vector<float>& weight, // [out_dim, in_dim]
        const std::vector<float>& bias,   // [out_dim]
        int in_dim, int out_dim, bool add_bias)
{
    int n_tokens = (int)x.size() / in_dim;
    std::vector<float> out((size_t)n_tokens * (size_t)out_dim, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < n_tokens; ++t) {
        for (int o = 0; o < out_dim; ++o) {
            size_t x_off = (size_t)t * (size_t)in_dim;
            size_t o_off = (size_t)t * (size_t)out_dim;
            float v = add_bias ? bias[(size_t)o] : 0.0f;
            size_t w_off = (size_t)o * (size_t)in_dim;
            for (int i = 0; i < in_dim; ++i)
                v += weight[w_off + (size_t)i] * x[x_off + (size_t)i];
            out[o_off + (size_t)o] = v;
        }
    }
    return out;
}

// ============================================================================
// Self-attention: multi-head, no mask
// ============================================================================

static inline std::vector<float> self_attention(
        const std::vector<float>& q_w, const std::vector<float>& q_b,
        const std::vector<float>& k_w, const std::vector<float>& k_b,
        const std::vector<float>& v_w, const std::vector<float>& v_b,
        const std::vector<float>& o_w, const std::vector<float>& o_b,
        const std::vector<float>& hidden, int n_tokens)
{
    auto q = linear(hidden, q_w, q_b, HUBERT_HIDDEN_SIZE, HUBERT_HIDDEN_SIZE, true);
    auto k = linear(hidden, k_w, k_b, HUBERT_HIDDEN_SIZE, HUBERT_HIDDEN_SIZE, true);
    auto v = linear(hidden, v_w, v_b, HUBERT_HIDDEN_SIZE, HUBERT_HIDDEN_SIZE, true);

    float scale = 1.0f / sqrtf((float)HUBERT_HEAD_DIM);
    std::vector<float> attn_out((size_t)n_tokens * (size_t)HUBERT_HIDDEN_SIZE, 0.0f);

    for (int h = 0; h < HUBERT_NUM_HEADS; ++h) {
        size_t h_off = (size_t)h * (size_t)HUBERT_HEAD_DIM;

        std::vector<float> scores((size_t)n_tokens * (size_t)n_tokens);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int i = 0; i < n_tokens; ++i) {
            size_t qi_off = (size_t)i * (size_t)HUBERT_HIDDEN_SIZE + h_off;
            for (int j = 0; j < n_tokens; ++j) {
                float s = 0.0f;
                size_t kj_off = (size_t)j * (size_t)HUBERT_HIDDEN_SIZE + h_off;
                for (int d = 0; d < HUBERT_HEAD_DIM; ++d)
                    s += q[qi_off + (size_t)d] * k[kj_off + (size_t)d];
                scores[(size_t)i * (size_t)n_tokens + (size_t)j] = s * scale;
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int i = 0; i < n_tokens; ++i) {
            float mx = scores[(size_t)i * (size_t)n_tokens];
            for (int j = 1; j < n_tokens; ++j)
                mx = std::max(mx, scores[(size_t)i * (size_t)n_tokens + (size_t)j]);
            float sum_exp = 0.0f;
            for (int j = 0; j < n_tokens; ++j) {
                float e = expf(scores[(size_t)i * (size_t)n_tokens + (size_t)j] - mx);
                scores[(size_t)i * (size_t)n_tokens + (size_t)j] = e;
                sum_exp += e;
            }
            for (int j = 0; j < n_tokens; ++j)
                scores[(size_t)i * (size_t)n_tokens + (size_t)j] /= sum_exp;
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int i = 0; i < n_tokens; ++i) {
            for (int d = 0; d < HUBERT_HEAD_DIM; ++d) {
                float val = 0.0f;
                for (int j = 0; j < n_tokens; ++j)
                    val += scores[(size_t)i * (size_t)n_tokens + (size_t)j] *
                           v[(size_t)j * (size_t)HUBERT_HIDDEN_SIZE + h_off + (size_t)d];
                attn_out[(size_t)i * (size_t)HUBERT_HIDDEN_SIZE + h_off + (size_t)d] = val;
            }
        }
    }

    return linear(attn_out, o_w, o_b, HUBERT_HIDDEN_SIZE, HUBERT_HIDDEN_SIZE, true);
}

// ============================================================================
// Feed-forward: Linear → GELU → Linear
// ============================================================================

static inline std::vector<float> ffn(
        const std::vector<float>& iw, const std::vector<float>& ib,
        const std::vector<float>& ow, const std::vector<float>& ob,
        const std::vector<float>& hidden)
{
    auto x = linear(hidden, iw, ib, HUBERT_HIDDEN_SIZE, HUBERT_INTERMEDIATE_SIZE, true);
    for (auto& v : x) v = gelu(v);
    return linear(x, ow, ob, HUBERT_INTERMEDIATE_SIZE, HUBERT_HIDDEN_SIZE, true);
}

// ============================================================================
// Positional convolution embedding (grouped Conv1d, causal: pad left)
// ============================================================================

static inline std::vector<float> pos_conv_embed(
        const std::vector<float>& input,          // [tokens, 768]
        const std::vector<float>& weight,          // [768, 48, 128]
        const std::vector<float>& bias,            // [768]
        int n_tokens)
{
    int kernel = HUBERT_POS_CONV_KERNEL;
    int groups = HUBERT_POS_CONV_GROUPS;
    int group_in_dim = HUBERT_HIDDEN_SIZE / groups; // 48

    // Pad symmetrically: pad_left = pad_right = kernel/2 = 64
    int pad = kernel / 2; // 64
    std::vector<float> padded((size_t)(n_tokens + 2 * pad) * (size_t)HUBERT_HIDDEN_SIZE, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        size_t src = (size_t)t * (size_t)HUBERT_HIDDEN_SIZE;
        size_t dst = ((size_t)t + (size_t)pad) * (size_t)HUBERT_HIDDEN_SIZE;
        memcpy(&padded[dst], &input[src], (size_t)HUBERT_HIDDEN_SIZE * sizeof(float));
    }

    // Conv1d(padding=64) produces n_tokens+1 outputs, then HubertSamePadLayer
    // removes the last one (kernel_size=128 is even → num_pad_remove=1).
    // We compute all n_tokens+1 outputs, apply GELU, then drop the last one.
    int conv_out = n_tokens + 1;
    std::vector<float> tmp((size_t)conv_out * (size_t)HUBERT_HIDDEN_SIZE, 0.0f);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < conv_out; ++t) {
        for (int g = 0; g < groups; ++g) {
            int g_start = g * group_in_dim;
            int g_end   = g_start + group_in_dim;
            for (int oc = g_start; oc < g_end; ++oc) {
                float val = bias[(size_t)oc];
                for (int ic = g_start; ic < g_end; ++ic) {
                    for (int k = 0; k < kernel; ++k) {
                        // PyTorch Conv1d(padding=64): output t accesses padded positions
                        // input[padded][t + k - pad] where padded is 0-padded on both sides.
                        // With our symmetrical padding, padded[pt] where pt = t + k
                        // gives input[t + k - pad] with zero-padding for out-of-range.
                        int pt = t + k;
                        if (pt >= 0 && pt < n_tokens + 2 * pad) {
                            size_t w_off = ((size_t)oc * (size_t)group_in_dim + (size_t)(ic - g_start)) * (size_t)kernel + (size_t)k;
                            val += weight[w_off] * padded[(size_t)pt * (size_t)HUBERT_HIDDEN_SIZE + (size_t)ic];
                        }
                    }
                }
                size_t idx = (size_t)t * (size_t)HUBERT_HIDDEN_SIZE + (size_t)oc;
                tmp[idx] = val;
            }
        }
    }

    // Apply GELU activation and trim last output (HubertSamePadLayer)
    std::vector<float> out((size_t)n_tokens * (size_t)HUBERT_HIDDEN_SIZE);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < n_tokens; ++t) {
        for (int c = 0; c < HUBERT_HIDDEN_SIZE; ++c) {
            size_t src = (size_t)t * (size_t)HUBERT_HIDDEN_SIZE;
            out[(size_t)t * (size_t)HUBERT_HIDDEN_SIZE + (size_t)c] =
                gelu(tmp[src + (size_t)c]);
        }
    }
    return out;
}

// ============================================================================
// Semantic encoder: residual units + conv blocks
// ============================================================================

static inline std::vector<float> residual_unit(
        const std::vector<float>& x,       // [channels, frames]
        const std::vector<float>& c1_w,    // [in_c, in_c, 3]
        const std::vector<float>& c2_w,    // [in_c, in_c, 1]
        int channels, int frames)
{
    // ELU → conv1 (kernel=3, pad=1) → ELU → conv2 (kernel=1) → +residual

    auto h = x;
    for (auto& v : h) v = elu(v);

    std::vector<float> h1((size_t)channels * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < channels; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = 0.0f;
            for (int ic = 0; ic < channels; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int pt = t + k - 1; // pad=1
                    if (pt >= 0 && pt < frames)
                        val += c1_w[((size_t)oc * (size_t)channels + (size_t)ic) * 3 + (size_t)k] *
                               h[(size_t)ic * (size_t)frames + (size_t)pt];
                }
            }
            h1[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }

    for (auto& v : h1) v = elu(v);

    std::vector<float> h2((size_t)channels * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < channels; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = 0.0f;
            for (int ic = 0; ic < channels; ++ic)
                val += c2_w[(size_t)oc * (size_t)channels + (size_t)ic] * h1[(size_t)ic * (size_t)frames + (size_t)t];
            h2[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }

    std::vector<float> out((size_t)channels * (size_t)frames);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = x[i] + h2[i];
    return out;
}

static inline std::vector<float> semantic_encoder_block(
        const std::vector<float>& x,       // [channels, frames]
        const std::vector<float>& c1_w,    // res_unit 0 conv1
        const std::vector<float>& c2_w,    // res_unit 0 conv2
        const std::vector<float>& r1_w,    // res_unit 1 conv1
        const std::vector<float>& r2_w,    // res_unit 1 conv2
        const std::vector<float>& conv_w,  // block conv weight [out, in, 3]
        const std::vector<float>& conv_b,  // block conv bias
        int channels, int frames)
{
    auto h = x;
    h = residual_unit(h, c1_w, c2_w, channels, frames);
    h = residual_unit(h, r1_w, r2_w, channels, frames);

    // Final conv: kernel=3, pad=1, bias
    std::vector<float> out((size_t)channels * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < channels; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = conv_b[(size_t)oc];
            for (int ic = 0; ic < channels; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int pt = t + k - 1;
                    if (pt >= 0 && pt < frames)
                        val += conv_w[((size_t)oc * (size_t)channels + (size_t)ic) * 3 + (size_t)k] *
                               h[(size_t)ic * (size_t)frames + (size_t)pt];
                }
            }
            out[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }
    return out;
}

// ============================================================================
// Weight structures and loaders
// ============================================================================

struct feat_extractor_w {
    std::vector<float> conv_w[7];   // [out_c, in_c, kernel]
    std::vector<float> norm_w;      // [512]
    std::vector<float> norm_b;      // [512]
    int in_c[7], out_c, kernel[7], stride[7];
};

struct feat_proj_w {
    std::vector<float> proj_w;      // [768, 512]
    std::vector<float> proj_b;      // [768]
    std::vector<float> norm_w;      // [512]
    std::vector<float> norm_b;      // [512]
};

struct encoder_layer_w {
    // attention
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    // ffn
    std::vector<float> ffn_iw, ffn_ib, ffn_ow, ffn_ob;
    // norms
    std::vector<float> attn_ln_w, attn_ln_b;
    std::vector<float> final_ln_w, final_ln_b;
};

struct hubert_encoder_w {
    encoder_layer_w  layers[12];
    std::vector<float> pos_conv_w; // [768, 48, 128] — raw weight (original1)
    std::vector<float> pos_conv_g; // [128] — weightnorm scale (original0)
    std::vector<float> pos_conv_b; // [768]
    std::vector<float> init_ln_w; // [768]
    std::vector<float> init_ln_b; // [768]
};

struct encoder_semantic_w {
    std::vector<float> conv_w;     // initial conv [768, 768, 3]
    // block 0
    std::vector<float> b0_ru0_c1, b0_ru0_c2, b0_ru1_c1, b0_ru1_c2;
    std::vector<float> b0_conv_w, b0_conv_b;
    // block 1
    std::vector<float> b1_ru0_c1, b1_ru0_c2, b1_ru1_c1, b1_ru1_c2;
    std::vector<float> b1_conv_w, b1_conv_b;
};

struct decoder_semantic_w {
    std::vector<float> conv1_w;    // initial conv [768, 768, 3]
    // block 0
    std::vector<float> b0_conv_w, b0_conv_b;
    std::vector<float> b0_ru0_c1, b0_ru0_c2, b0_ru1_c1, b0_ru1_c2;
    // block 1
    std::vector<float> b1_conv_w, b1_conv_b;
    std::vector<float> b1_ru0_c1, b1_ru0_c2, b1_ru1_c1, b1_ru1_c2;
    std::vector<float> conv2_w;    // final conv [768, 768, 3]
};

// --------------------------------------------------------------------------
// Tensor loader (names already stripped of CODEC_PREFIX by GGUF converter)
// --------------------------------------------------------------------------

static void load_tensor(const companion_file& file, const std::string& name,
                         size_t expected, std::vector<float>& buf) {
    buf = load_codec_tensor(file, name, expected);
}

static feat_extractor_w load_feat_extractor(const companion_file& file) {
    feat_extractor_w w;
    for (int i = 0; i < 7; ++i) {
        w.stride[i] = HUBERT_FEAT_STRIDES[i];
        w.kernel[i] = HUBERT_FEAT_KERNELS[i];
        w.in_c[i]   = (i == 0) ? 1 : HUBERT_FEAT_CHANNELS;
        w.out_c     = HUBERT_FEAT_CHANNELS;
        load_tensor(file,
            "semantic_model.feature_extractor.conv_layers." + std::to_string(i) + ".conv.weight",
            (size_t)w.out_c * (size_t)w.in_c[i] * (size_t)w.kernel[i], w.conv_w[i]);
    }
    load_tensor(file, "semantic_model.feature_extractor.conv_layers.0.layer_norm.weight",
                (size_t)HUBERT_FEAT_CHANNELS, w.norm_w);
    load_tensor(file, "semantic_model.feature_extractor.conv_layers.0.layer_norm.bias",
                (size_t)HUBERT_FEAT_CHANNELS, w.norm_b);
    return w;
}

static feat_proj_w load_feat_proj(const companion_file& file) {
    feat_proj_w w;
    load_tensor(file, "semantic_model.feature_projection.projection.weight",
                (size_t)HUBERT_HIDDEN_SIZE * (size_t)HUBERT_FEAT_CHANNELS, w.proj_w);
    load_tensor(file, "semantic_model.feature_projection.projection.bias",
                (size_t)HUBERT_HIDDEN_SIZE, w.proj_b);
    load_tensor(file, "semantic_model.feature_projection.layer_norm.weight",
                (size_t)HUBERT_FEAT_CHANNELS, w.norm_w);
    load_tensor(file, "semantic_model.feature_projection.layer_norm.bias",
                (size_t)HUBERT_FEAT_CHANNELS, w.norm_b);
    return w;
}

static hubert_encoder_w load_encoder(const companion_file& file) {
    hubert_encoder_w w;
    for (int i = 0; i < 12; ++i) {
        std::string base = "semantic_model.encoder.layers." + std::to_string(i) + ".";
        auto& l = w.layers[i];
        size_t hs = (size_t)HUBERT_HIDDEN_SIZE;
        size_t im = (size_t)HUBERT_INTERMEDIATE_SIZE;

        load_tensor(file, base + "attention.q_proj.weight", hs * hs, l.q_w);
        load_tensor(file, base + "attention.q_proj.bias",   hs,      l.q_b);
        load_tensor(file, base + "attention.k_proj.weight", hs * hs, l.k_w);
        load_tensor(file, base + "attention.k_proj.bias",   hs,      l.k_b);
        load_tensor(file, base + "attention.v_proj.weight", hs * hs, l.v_w);
        load_tensor(file, base + "attention.v_proj.bias",   hs,      l.v_b);
        load_tensor(file, base + "attention.out_proj.weight", hs * hs, l.o_w);
        load_tensor(file, base + "attention.out_proj.bias",   hs,      l.o_b);
        load_tensor(file, base + "feed_forward.intermediate_dense.weight", im * hs, l.ffn_iw);
        load_tensor(file, base + "feed_forward.intermediate_dense.bias",   im,      l.ffn_ib);
        load_tensor(file, base + "feed_forward.output_dense.weight",       hs * im, l.ffn_ow);
        load_tensor(file, base + "feed_forward.output_dense.bias",         hs,      l.ffn_ob);
        load_tensor(file, base + "layer_norm.weight",       hs, l.attn_ln_w);
        load_tensor(file, base + "layer_norm.bias",         hs, l.attn_ln_b);
        load_tensor(file, base + "final_layer_norm.weight", hs, l.final_ln_w);
        load_tensor(file, base + "final_layer_norm.bias",   hs, l.final_ln_b);
    }

    load_tensor(file, "semantic_model.encoder.layer_norm.weight", (size_t)HUBERT_HIDDEN_SIZE, w.init_ln_w);
    load_tensor(file, "semantic_model.encoder.layer_norm.bias",   (size_t)HUBERT_HIDDEN_SIZE, w.init_ln_b);

    load_tensor(file, "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
                (size_t)HUBERT_HIDDEN_SIZE * (size_t)(HUBERT_HIDDEN_SIZE / HUBERT_POS_CONV_GROUPS) * (size_t)HUBERT_POS_CONV_KERNEL,
                w.pos_conv_w);
    load_tensor(file, "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
                (size_t)HUBERT_POS_CONV_KERNEL, w.pos_conv_g);
    load_tensor(file, "semantic_model.encoder.pos_conv_embed.conv.bias",
                (size_t)HUBERT_HIDDEN_SIZE, w.pos_conv_b);

    // Apply weightnorm: w_eff = raw_w * g[k] / sqrt(sum(raw_w[:,:,k]^2))
    // Weight layout: [out_channels=768, in_per_group=48, kernel=128]
    // original0: g[k] = per-kernel-position scale [128]
    // original1: raw weight [768, 48, 128]
    {
        int K = HUBERT_POS_CONV_KERNEL;
        int out_c = HUBERT_HIDDEN_SIZE;
        int in_per_group = HUBERT_HIDDEN_SIZE / HUBERT_POS_CONV_GROUPS; // 48
        for (int k = 0; k < K; ++k) {
            double norm_sq = 0.0;
            for (int oc = 0; oc < out_c; ++oc) {
                for (int ic = 0; ic < in_per_group; ++ic) {
                    size_t idx = ((size_t)oc * (size_t)in_per_group + (size_t)ic) * (size_t)K + (size_t)k;
                    double v = (double)w.pos_conv_w[idx];
                    norm_sq += v * v;
                }
            }
            float scale = w.pos_conv_g[(size_t)k] / (float)std::sqrt(std::max(norm_sq, 1e-30));
            for (int oc = 0; oc < out_c; ++oc) {
                for (int ic = 0; ic < in_per_group; ++ic) {
                    size_t idx = ((size_t)oc * (size_t)in_per_group + (size_t)ic) * (size_t)K + (size_t)k;
                    w.pos_conv_w[idx] *= scale;
                }
            }
        }
    }

    return w;
}

static encoder_semantic_w load_encoder_semantic_w(const companion_file& file) {
    encoder_semantic_w w;
    size_t hs = (size_t)HUBERT_HIDDEN_SIZE;
    load_tensor(file, "encoder_semantic.conv.weight", hs * hs * 3, w.conv_w);

    // Block 0
    load_tensor(file, "encoder_semantic.conv_blocks.0.res_units.0.conv1.weight", hs * hs * 3, w.b0_ru0_c1);
    load_tensor(file, "encoder_semantic.conv_blocks.0.res_units.0.conv2.weight", hs * hs * 1, w.b0_ru0_c2);
    load_tensor(file, "encoder_semantic.conv_blocks.0.res_units.1.conv1.weight", hs * hs * 3, w.b0_ru1_c1);
    load_tensor(file, "encoder_semantic.conv_blocks.0.res_units.1.conv2.weight", hs * hs * 1, w.b0_ru1_c2);
    load_tensor(file, "encoder_semantic.conv_blocks.0.conv.weight", hs * hs * 3, w.b0_conv_w);
    load_tensor(file, "encoder_semantic.conv_blocks.0.conv.bias",   hs,          w.b0_conv_b);

    // Block 1
    load_tensor(file, "encoder_semantic.conv_blocks.1.res_units.0.conv1.weight", hs * hs * 3, w.b1_ru0_c1);
    load_tensor(file, "encoder_semantic.conv_blocks.1.res_units.0.conv2.weight", hs * hs * 1, w.b1_ru0_c2);
    load_tensor(file, "encoder_semantic.conv_blocks.1.res_units.1.conv1.weight", hs * hs * 3, w.b1_ru1_c1);
    load_tensor(file, "encoder_semantic.conv_blocks.1.res_units.1.conv2.weight", hs * hs * 1, w.b1_ru1_c2);
    load_tensor(file, "encoder_semantic.conv_blocks.1.conv.weight", hs * hs * 3, w.b1_conv_w);
    load_tensor(file, "encoder_semantic.conv_blocks.1.conv.bias",   hs,          w.b1_conv_b);

    return w;
}

static decoder_semantic_w load_decoder_semantic_w(const companion_file& file) {
    decoder_semantic_w w;
    size_t hs = (size_t)HUBERT_HIDDEN_SIZE;
    load_tensor(file, "decoder_semantic.conv1.weight", hs * hs * 3, w.conv1_w);

    // Block 0
    load_tensor(file, "decoder_semantic.conv_blocks.0.conv.weight", hs * hs * 3, w.b0_conv_w);
    load_tensor(file, "decoder_semantic.conv_blocks.0.conv.bias",   hs,          w.b0_conv_b);
    load_tensor(file, "decoder_semantic.conv_blocks.0.res_units.0.conv1.weight", hs * hs * 3, w.b0_ru0_c1);
    load_tensor(file, "decoder_semantic.conv_blocks.0.res_units.0.conv2.weight", hs * hs * 1, w.b0_ru0_c2);
    load_tensor(file, "decoder_semantic.conv_blocks.0.res_units.1.conv1.weight", hs * hs * 3, w.b0_ru1_c1);
    load_tensor(file, "decoder_semantic.conv_blocks.0.res_units.1.conv2.weight", hs * hs * 1, w.b0_ru1_c2);

    // Block 1
    load_tensor(file, "decoder_semantic.conv_blocks.1.conv.weight", hs * hs * 3, w.b1_conv_w);
    load_tensor(file, "decoder_semantic.conv_blocks.1.conv.bias",   hs,          w.b1_conv_b);
    load_tensor(file, "decoder_semantic.conv_blocks.1.res_units.0.conv1.weight", hs * hs * 3, w.b1_ru0_c1);
    load_tensor(file, "decoder_semantic.conv_blocks.1.res_units.0.conv2.weight", hs * hs * 1, w.b1_ru0_c2);
    load_tensor(file, "decoder_semantic.conv_blocks.1.res_units.1.conv1.weight", hs * hs * 3, w.b1_ru1_c1);
    load_tensor(file, "decoder_semantic.conv_blocks.1.res_units.1.conv2.weight", hs * hs * 1, w.b1_ru1_c2);

    load_tensor(file, "decoder_semantic.conv2.weight", hs * hs * 3, w.conv2_w);

    return w;
}

// ============================================================================
// Forward: HuBERT feature extractor ([T] → [512, frames])
// ============================================================================

static inline std::vector<float> feat_extract_forward(
        const feat_extractor_w& w, const std::vector<float>& pcm_16k)
{
    // Input: [1, T] (channels-first, 1 channel)
    std::vector<float> x = pcm_16k;
    int in_c = 1, in_len = (int)pcm_16k.size();

    for (int i = 0; i < 7; ++i) {
        std::vector<float> empty_bias; // no bias in feature extractor convs
        x = conv1d_valid(x, w.conv_w[i], empty_bias,
                         in_c, in_len, HUBERT_FEAT_CHANNELS, w.kernel[i], w.stride[i]);
        in_len = (in_len - w.kernel[i]) / w.stride[i] + 1;
        in_c = HUBERT_FEAT_CHANNELS;

        if (i == 0) {
            // GroupNorm(num_groups=HUBERT_FEAT_CHANNELS, num_channels=HUBERT_FEAT_CHANNELS)
            // = 512 groups of 1 channel each = per-channel normalization over time
            x = group_norm_1d(x, w.norm_w, w.norm_b, HUBERT_FEAT_CHANNELS, in_len, HUBERT_FEAT_CHANNELS);
        }

        // GELU
        for (auto& v : x) v = gelu(v);
    }
    return x; // [512, frames]
}

// ============================================================================
// Forward: Feature projection ([512, frames] → [frames, 768])
// ============================================================================

static inline std::vector<float> feat_proj_forward(
        const feat_proj_w& w, const std::vector<float>& feat, int frames)
{
    // Transpose: [512, frames] → [frames, 512]
    std::vector<float> tokens((size_t)frames * (size_t)HUBERT_FEAT_CHANNELS);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < frames; ++t)
        for (int c = 0; c < HUBERT_FEAT_CHANNELS; ++c)
            tokens[(size_t)t * (size_t)HUBERT_FEAT_CHANNELS + (size_t)c] =
                feat[(size_t)c * (size_t)frames + (size_t)t];

    // LayerNorm
    tokens = layer_norm_tokens(tokens, w.norm_w, w.norm_b, HUBERT_FEAT_CHANNELS);

    // Linear 512 → 768
    return linear(tokens, w.proj_w, w.proj_b, HUBERT_FEAT_CHANNELS, HUBERT_HIDDEN_SIZE, true);
}

// ============================================================================
// Forward: Encoder semantic
// ============================================================================

static inline std::vector<float> encoder_semantic_forward(
        const encoder_semantic_w& w,
        const std::vector<float>& x,   // [768, frames] channels-first
        int frames)
{
    int ch = HUBERT_HIDDEN_SIZE;
    // Initial conv: kernel=3, pad=1, no bias
    std::vector<float> h((size_t)ch * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < ch; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = 0.0f;
            for (int ic = 0; ic < ch; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int pt = t + k - 1;
                    if (pt >= 0 && pt < frames)
                        val += w.conv_w[((size_t)oc * (size_t)ch + (size_t)ic) * 3 + (size_t)k] *
                               x[(size_t)ic * (size_t)frames + (size_t)pt];
                }
            }
            h[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }

    // Block 0
    h = semantic_encoder_block(h, w.b0_ru0_c1, w.b0_ru0_c2, w.b0_ru1_c1, w.b0_ru1_c2,
                               w.b0_conv_w, w.b0_conv_b, ch, frames);
    // Block 1
    h = semantic_encoder_block(h, w.b1_ru0_c1, w.b1_ru0_c2, w.b1_ru1_c1, w.b1_ru1_c2,
                               w.b1_conv_w, w.b1_conv_b, ch, frames);

    return h; // [768, frames]
}

// ============================================================================
// Decoder semantic: conv1 → block0(conv→res×2) → block1(conv→res×2) → conv2
// Decoder block runs conv FIRST, then res_units (opposite of encoder block)
// ============================================================================

static inline std::vector<float> decoder_block(
        const std::vector<float>& x,
        const std::vector<float>& conv_w, const std::vector<float>& conv_b,
        const std::vector<float>& c1_w,  const std::vector<float>& c2_w,
        const std::vector<float>& r1_w,  const std::vector<float>& r2_w,
        int channels, int frames)
{
    // Conv first (kernel=3, pad=1, bias=True)
    std::vector<float> h((size_t)channels * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < channels; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = conv_b[(size_t)oc];
            for (int ic = 0; ic < channels; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int pt = t + k - 1;
                    if (pt >= 0 && pt < frames)
                        val += conv_w[((size_t)oc * (size_t)channels + (size_t)ic) * 3 + (size_t)k] *
                               x[(size_t)ic * (size_t)frames + (size_t)pt];
                }
            }
            h[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }

    // Residual units
    h = residual_unit(h, c1_w, c2_w, channels, frames);
    h = residual_unit(h, r1_w, r2_w, channels, frames);

    return h;
}

static inline std::vector<float> decoder_semantic_forward(
        const decoder_semantic_w& w,
        const std::vector<float>& x,   // [768, frames] channels-first
        int frames)
{
    int ch = HUBERT_HIDDEN_SIZE;
    // conv1: kernel=3, pad=1, no bias
    std::vector<float> h((size_t)ch * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < ch; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = 0.0f;
            for (int ic = 0; ic < ch; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int pt = t + k - 1;
                    if (pt >= 0 && pt < frames)
                        val += w.conv1_w[((size_t)oc * (size_t)ch + (size_t)ic) * 3 + (size_t)k] *
                               x[(size_t)ic * (size_t)frames + (size_t)pt];
                }
            }
            h[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }

    // Block 0
    h = decoder_block(h, w.b0_conv_w, w.b0_conv_b, w.b0_ru0_c1, w.b0_ru0_c2,
                      w.b0_ru1_c1, w.b0_ru1_c2, ch, frames);
    // Block 1
    h = decoder_block(h, w.b1_conv_w, w.b1_conv_b, w.b1_ru0_c1, w.b1_ru0_c2,
                      w.b1_ru1_c1, w.b1_ru1_c2, ch, frames);

    // conv2: kernel=3, pad=1, no bias
    std::vector<float> out((size_t)ch * (size_t)frames, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < ch; ++oc) {
        for (int t = 0; t < frames; ++t) {
            float val = 0.0f;
            for (int ic = 0; ic < ch; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int pt = t + k - 1;
                    if (pt >= 0 && pt < frames)
                        val += w.conv2_w[((size_t)oc * (size_t)ch + (size_t)ic) * 3 + (size_t)k] *
                               h[(size_t)ic * (size_t)frames + (size_t)pt];
                }
            }
            out[(size_t)oc * (size_t)frames + (size_t)t] = val;
        }
    }
    return out;
}

// ============================================================================
// fc1: linear 1024 → 768 (used in LLM multimodal decode path)
// ============================================================================

static inline std::vector<float> fc1_forward(
        const std::vector<float>& fc1_w, const std::vector<float>& fc1_b,
        const std::vector<float>& hidden)  // [tokens, 1024]
{
    return linear(hidden, fc1_w, fc1_b, HUBERT_HIDDEN_SIZE + 256, HUBERT_HIDDEN_SIZE, true);
}

// ============================================================================
// Full encode pipeline: WAV@24kHz → codes
// ============================================================================

inline std::vector<std::vector<int>> encode_reference_codes(
        const companion_file& file,
        const std::vector<float>& pcm_24k)
{
    if (pcm_24k.empty()) return {};

    // ---- Load weights ----
    auto feat_w   = load_feat_extractor(file);
    auto proj_w   = load_feat_proj(file);
    auto enc_w    = load_encoder(file);
    auto sem_w    = load_encoder_semantic_w(file);
    auto rvq_w    = load_rvq_decoder_weights(file);
    auto dac_w    = load_dac_encoder_weights(file);

    std::vector<float> fc_w, fc_b;
    load_tensor(file, "fc.weight", (size_t)1024 * 1024, fc_w);
    load_tensor(file, "fc.bias",   (size_t)1024,        fc_b);

    const int T_24k = (int)pcm_24k.size();

    // ========== Semantic path ==========
    // 1. Resample 24k → 16k (kaiser-window sinc, matching torchaudio)
    auto pcm_16k = resample_kaiser(pcm_24k, 24000.0, 16000.0);

    // 2. Pad 160 each side
    size_t pad16 = 160;
    std::vector<float> pcm_16k_padded(pcm_16k.size() + 2 * pad16, 0.0f);
    memcpy(pcm_16k_padded.data() + pad16, pcm_16k.data(), pcm_16k.size() * sizeof(float));

    // 3. Feature extractor → [512, feat_frames]
    auto feat = feat_extract_forward(feat_w, pcm_16k_padded);
    int feat_frames = (int)feat.size() / HUBERT_FEAT_CHANNELS;
    if (feat_frames <= 0) return {};

    // 4. Feature projection → [feat_frames, 768]
    auto hidden = feat_proj_forward(proj_w, feat, feat_frames);

    // 5. Positional conv embedding: conv1d + GELU (inside pos_conv_embed)
    auto pos_emb = pos_conv_embed(hidden, enc_w.pos_conv_w, enc_w.pos_conv_b, feat_frames);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += pos_emb[i];
    // HubertEncoder applies LayerNorm after pos_conv + residual
    hidden = layer_norm_tokens(hidden, enc_w.init_ln_w, enc_w.init_ln_b, HUBERT_HIDDEN_SIZE);

    // 6. Run encoder, capturing all 13 hidden states
    //    HubertEncoder stores: hs[0] = LN(pos_conv+residual) BEFORE layer0,
    //    then captures current hidden_states BEFORE each layer (hs[1..12] = after layers 0..11).
    //    No LN on the final state; HF does NOT use use_weighted_layer_sum.
    std::vector<float> mean_hs(hidden.size());
    const float mean_scale = 1.0f / 13.0f;
    for (size_t i = 0; i < hidden.size(); ++i) {
        mean_hs[i] = hidden[i] * mean_scale;
    }

    auto x = hidden;
    for (int l = 0; l < 12; ++l) {
        auto& L = enc_w.layers[l];

        // HubertEncoderLayer order:
        // 1. Self-attention on raw input (NO pre-LN)
        // 2. Residual: x = x + dropout(attention(x))
        // 3. LN: x = layer_norm(x)
        // 4. FFN on LN'd x: x = x + FFN(x)
        // 5. final_LN: x = final_layer_norm(x)
        auto attn_residual = x;
        auto x_attn = self_attention(L.q_w, L.q_b, L.k_w, L.k_b, L.v_w, L.v_b, L.o_w, L.o_b, x, feat_frames);
        for (size_t i = 0; i < x.size(); ++i) x[i] = attn_residual[i] + x_attn[i];

        x = layer_norm_tokens(x, L.attn_ln_w, L.attn_ln_b, HUBERT_HIDDEN_SIZE);

        auto ln_x = x;
        auto x_ffn = ffn(L.ffn_iw, L.ffn_ib, L.ffn_ow, L.ffn_ob, x);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ln_x[i] + x_ffn[i];

        x = layer_norm_tokens(x, L.final_ln_w, L.final_ln_b, HUBERT_HIDDEN_SIZE);

        for (size_t i = 0; i < mean_hs.size(); ++i) {
            mean_hs[i] += x[i] * mean_scale;
        }
    }

    // 7. Stack 13 states and mean → [feat_frames, 768]
    {
        hidden = std::move(mean_hs);
    }

    // 8. Semantic downsample ×2
    {
        int ds_frames = feat_frames / HUBERT_SEMANTIC_DOWNSAMPLE;
        std::vector<float> ds((size_t)ds_frames * (size_t)HUBERT_HIDDEN_SIZE);
        for (int t = 0; t < ds_frames; ++t) {
            size_t src = (size_t)(t * HUBERT_SEMANTIC_DOWNSAMPLE) * (size_t)HUBERT_HIDDEN_SIZE;
            size_t dst = (size_t)t * (size_t)HUBERT_HIDDEN_SIZE;
            memcpy(&ds[dst], &hidden[src], (size_t)HUBERT_HIDDEN_SIZE * sizeof(float));
        }
        hidden = std::move(ds);
    }
    int sem_frames = (int)hidden.size() / HUBERT_HIDDEN_SIZE;

    // 10. Transpose to channels-first and run encoder_semantic
    {
        std::vector<float> sem_ch((size_t)HUBERT_HIDDEN_SIZE * (size_t)sem_frames);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int t = 0; t < sem_frames; ++t)
            for (int d = 0; d < HUBERT_HIDDEN_SIZE; ++d)
                sem_ch[(size_t)d * (size_t)sem_frames + (size_t)t] =
                    hidden[(size_t)t * (size_t)HUBERT_HIDDEN_SIZE + (size_t)d];
        hidden = encoder_semantic_forward(sem_w, sem_ch, sem_frames);
        // hidden is now [768, sem_frames] channels-first
    }

    // ========== Acoustic path ==========
    // Python encode: only pad 480 when acoustic output frames != semantic output frames.
    // acoustic_model.hop_length = 960, so frames = T_24k / 960
    // For 5s audio: 120000/960 = 125, which equals semantic frames (125), so NO pad.
    const int dac_hop = 960; // product of DAC downsampling ratios
    int ac_no_pad_frames = T_24k / dac_hop;
    int pad_ac = (ac_no_pad_frames != feat_frames / HUBERT_SEMANTIC_DOWNSAMPLE) ? 480 : 0;
    std::vector<float> pcm_for_dac;
    if (pad_ac > 0) {
        pcm_for_dac.resize((size_t)T_24k + (size_t)2 * pad_ac, 0.0f);
        memcpy(pcm_for_dac.data() + pad_ac, pcm_24k.data(), (size_t)T_24k * sizeof(float));
    } else {
        pcm_for_dac = pcm_24k;
    }
    auto acoustic = dac_encode_acoustic_latents(dac_w, pcm_for_dac);
    if (acoustic.empty()) return {};
    int ac_frames = (int)acoustic.size() / dac_w.acoustic_size;

    // ========== Align and concat ==========
    int frames = std::min(sem_frames, ac_frames);
    if (frames <= 0) return {};

    const int ac_dim = dac_w.acoustic_size; // 256
    const int concat_dim = HUBERT_HIDDEN_SIZE + ac_dim; // 1024

    // Semantic: [768, sem_frames] → trim to [768, frames]
    // Acoustic: [ac_frames, 256] → trim to [frames, 256]
    // Concat: [frames, 256] + transpose([768, frames]) → [frames, 1024]

    std::vector<float> concat((size_t)frames * (size_t)concat_dim);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < frames; ++t) {
        size_t off = (size_t)t * (size_t)concat_dim;
        // Acoustic (first 256)
        for (int c = 0; c < ac_dim; ++c)
            concat[off + (size_t)c] = acoustic[(size_t)t * (size_t)ac_dim + (size_t)c];
        // Semantic (next 768), from channels-first
        for (int c = 0; c < HUBERT_HIDDEN_SIZE; ++c)
            concat[off + (size_t)ac_dim + (size_t)c] =
                hidden[(size_t)c * (size_t)frames + (size_t)t];
    }

    // fc: [frames, 1024] -> [frames, 1024]
    auto emb = linear(concat, fc_w, fc_b, concat_dim, concat_dim, true);

    // RVQ encode expects [frames, hidden_size] layout
    return encode_rvq_codes(rvq_w, emb);
}
