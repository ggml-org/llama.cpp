#pragma once

// shared graph building blocks for the LTX transformer and connectors

#include "mediagen-ltx.h"

#include <cmath>

constexpr double MG_PI = 3.14159265358979323846;

// rotary tables for the LTX "split" rope
// positions: n_tokens x n_axes fractional coordinates in [0, 1]; out: [n_tokens][dim/2]
inline void ltx_rope_tables(const std::vector<double> & positions, int n_tokens, int n_axes, int dim, double theta,
                            std::vector<float> & cos_out, std::vector<float> & sin_out) {
    const int half = dim / 2;
    const int n_freq = dim / (2 * n_axes);
    const int pad = half - n_freq * n_axes;
    std::vector<double> indices(n_freq);
    for (int f = 0; f < n_freq; f++) {
        const double e = n_freq > 1 ? (double) f / (double) (n_freq - 1) : 0.0;
        indices[f] = std::pow(theta, e) * MG_PI / 2.0;
    }
    cos_out.assign((size_t) n_tokens * half, 1.0f);
    sin_out.assign((size_t) n_tokens * half, 0.0f);
    for (int t = 0; t < n_tokens; t++) {
        float * c = cos_out.data() + (size_t) t * half + pad;
        float * s = sin_out.data() + (size_t) t * half + pad;
        for (int f = 0; f < n_freq; f++) {
            for (int a = 0; a < n_axes; a++) {
                const double frac  = positions[(size_t) t * n_axes + a];
                const double angle = indices[f] * (frac * 2.0 - 1.0);
                c[f * n_axes + a] = (float) std::cos(angle);
                s[f * n_axes + a] = (float) std::sin(angle);
            }
        }
    }
}

struct ltx_rope_in {
    ggml_tensor * cos_t = nullptr;
    ggml_tensor * sin_t = nullptr;
};

inline ltx_rope_in ltx_make_rope(mg_graph & g, const std::vector<double> & pos, int n_tokens, int n_axes, int dim, float theta, const char * name) {
    std::vector<float> c, s;
    ltx_rope_tables(pos, n_tokens, n_axes, dim, theta, c, s);
    ltx_rope_in r;
    r.cos_t = g.new_input_f32({ dim / 2, n_tokens }, c, (std::string(name) + "_cos").c_str());
    r.sin_t = g.new_input_f32({ dim / 2, n_tokens }, s, (std::string(name) + "_sin").c_str());
    return r;
}

// rms norm over ne[0], optional weight
inline ggml_tensor * ltx_rms(ggml_context * ctx, ggml_tensor * x, float eps, ggml_tensor * w = nullptr) {
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    if (w) {
        y = ggml_mul(ctx, y, w);
    }
    return y;
}

// rms norm over the channel dim ne[2] of a [W, H, C, T] activation
inline ggml_tensor * ltx_pixel_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    ggml_tensor * p = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C, W, H, T]
    p = ggml_rms_norm(ctx, p, eps);
    return ggml_cont(ctx, ggml_permute(ctx, p, 2, 0, 1, 3));
}

// x: [dim, T], cos/sin: [dim/2, T]; per head, the first half is rotated with the second half
inline ggml_tensor * ltx_rope_split(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t, int n_heads) {
    const int64_t dim = x->ne[0];
    const int64_t T   = x->ne[1];
    const int64_t hd  = dim / n_heads;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, ggml_cont(ctx, x), hd / 2, 2, n_heads, T);
    ggml_tensor * x1 = ggml_view_4d(ctx, x4, hd / 2, 1, n_heads, T, x4->nb[1], x4->nb[2], x4->nb[3], 0);
    ggml_tensor * x2 = ggml_view_4d(ctx, x4, hd / 2, 1, n_heads, T, x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[1]);
    x1 = ggml_cont(ctx, x1);
    x2 = ggml_cont(ctx, x2);
    ggml_tensor * c = ggml_reshape_4d(ctx, cos_t, hd / 2, 1, n_heads, T);
    ggml_tensor * s = ggml_reshape_4d(ctx, sin_t, hd / 2, 1, n_heads, T);
    ggml_tensor * o1 = ggml_sub(ctx, ggml_mul(ctx, x1, c), ggml_mul(ctx, x2, s));
    ggml_tensor * o2 = ggml_add(ctx, ggml_mul(ctx, x1, s), ggml_mul(ctx, x2, c));
    ggml_tensor * o  = ggml_concat(ctx, o1, o2, 1);
    return ggml_reshape_2d(ctx, o, dim, T);
}

// q: [dim_q, Tq], k, v: [dim_kv, Tk] -> [dim_v, Tq]
inline ggml_tensor * ltx_sdpa(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int n_heads, bool flash_attn) {
    const int64_t Tq = q->ne[1];
    const int64_t Tk = k->ne[1];
    const int64_t hd = q->ne[0] / n_heads;
    const int64_t hv = v->ne[0] / n_heads;
    const float scale = 1.0f / std::sqrt((float) hd);

    ggml_tensor * q3 = ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, n_heads, Tq), 0, 2, 1, 3); // [hd, Tq, H]
    ggml_tensor * k3 = ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, n_heads, Tk), 0, 2, 1, 3); // [hd, Tk, H]

    ggml_tensor * cur;
    if (flash_attn) {
        ggml_tensor * v3 = ggml_permute(ctx, ggml_reshape_3d(ctx, v, hv, n_heads, Tk), 0, 2, 1, 3); // [hv, Tk, H]
        k3 = ggml_cast(ctx, k3, GGML_TYPE_F16);
        v3 = ggml_cast(ctx, v3, GGML_TYPE_F16);
        cur = ggml_flash_attn_ext(ctx, q3, k3, v3, nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, hv * n_heads, Tq);
    } else {
        ggml_tensor * v3 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hv, n_heads, Tk), 1, 2, 0, 3)); // [Tk, hv, H]
        ggml_tensor * kq = ggml_mul_mat(ctx, k3, q3); // [Tk, Tq, H]
        kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v3, kq); // [hv, Tq, H]
        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);      // [hv, H, Tq]
        cur = ggml_cont_2d(ctx, cur, hv * n_heads, Tq);
    }
    return cur;
}

// LTX CrossAttention weights
struct ltx_attn_weights {
    ggml_tensor * to_q = nullptr, * to_q_b = nullptr;
    ggml_tensor * to_k = nullptr, * to_k_b = nullptr;
    ggml_tensor * to_v = nullptr, * to_v_b = nullptr;
    ggml_tensor * to_out = nullptr, * to_out_b = nullptr;
    ggml_tensor * q_norm = nullptr, * k_norm = nullptr;
    ggml_tensor * gate = nullptr, * gate_b = nullptr;

    static ltx_attn_weights load(const mg_weights & w, const std::string & prefix) {
        ltx_attn_weights a;
        a.to_q     = w.get(prefix + ".to_q.weight");
        a.to_q_b   = w.get(prefix + ".to_q.bias", false);
        a.to_k     = w.get(prefix + ".to_k.weight");
        a.to_k_b   = w.get(prefix + ".to_k.bias", false);
        a.to_v     = w.get(prefix + ".to_v.weight");
        a.to_v_b   = w.get(prefix + ".to_v.bias", false);
        a.to_out   = w.get(prefix + ".to_out.0.weight");
        a.to_out_b = w.get(prefix + ".to_out.0.bias", false);
        a.q_norm   = w.get(prefix + ".q_norm.weight");
        a.k_norm   = w.get(prefix + ".k_norm.weight");
        a.gate     = w.get(prefix + ".to_gate_logits.weight", false);
        a.gate_b   = w.get(prefix + ".to_gate_logits.bias", false);
        return a;
    }
};

// x: [dim_q, Tq] normalized input, context: [dim_kv, Tk] or nullptr for self attention
inline ggml_tensor * ltx_attention(ggml_context * ctx, const ltx_attn_weights & w, ggml_tensor * x, ggml_tensor * context,
                                   ggml_tensor * cos_q, ggml_tensor * sin_q, ggml_tensor * cos_k, ggml_tensor * sin_k,
                                   int n_heads, bool flash_attn) {
    ggml_tensor * kv_in = context ? context : x;
    ggml_tensor * q = mg_linear(ctx, x, w.to_q, w.to_q_b);
    ggml_tensor * k = mg_linear(ctx, kv_in, w.to_k, w.to_k_b);
    ggml_tensor * v = mg_linear(ctx, kv_in, w.to_v, w.to_v_b);
    q = ltx_rms(ctx, q, 1e-5f, w.q_norm);
    k = ltx_rms(ctx, k, 1e-5f, w.k_norm);
    if (cos_q) {
        q = ltx_rope_split(ctx, q, cos_q, sin_q, n_heads);
    }
    if (cos_k) {
        k = ltx_rope_split(ctx, k, cos_k, sin_k, n_heads);
    }
    ggml_tensor * out = ltx_sdpa(ctx, q, k, v, n_heads, flash_attn);
    if (w.gate) {
        const int64_t Tq = x->ne[1];
        const int64_t hv = out->ne[0] / n_heads;
        ggml_tensor * g = mg_linear(ctx, x, w.gate, w.gate_b);        // [H, Tq]
        g = ggml_scale(ctx, ggml_sigmoid(ctx, g), 2.0f);
        ggml_tensor * o3 = ggml_reshape_3d(ctx, out, hv, n_heads, Tq);
        o3 = ggml_mul(ctx, o3, ggml_reshape_3d(ctx, g, 1, n_heads, Tq));
        out = ggml_reshape_2d(ctx, o3, hv * n_heads, Tq);
    }
    return mg_linear(ctx, out, w.to_out, w.to_out_b);
}

// gelu(tanh) feed forward: net.0.proj -> gelu -> net.2
inline ggml_tensor * ltx_ff(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x) {
    ggml_tensor * h = mg_linear(ctx, x, w.get(prefix + ".net.0.proj.weight"), w.get(prefix + ".net.0.proj.bias", false));
    h = ggml_gelu(ctx, h);
    return mg_linear(ctx, h, w.get(prefix + ".net.2.weight"), w.get(prefix + ".net.2.bias", false));
}

// AdaLayerNormSingle: t [n] -> mod [coef*dim, n], emb [dim, n]
inline void ltx_adaln_single(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * t,
                             ggml_tensor ** mod, ggml_tensor ** emb) {
    ggml_tensor * te = ggml_timestep_embedding(ctx, t, 256, 10000);       // [256, n]
    te = mg_linear(ctx, te, w.get(prefix + ".emb.timestep_embedder.linear_1.weight"), w.get(prefix + ".emb.timestep_embedder.linear_1.bias"));
    te = ggml_silu(ctx, te);
    te = mg_linear(ctx, te, w.get(prefix + ".emb.timestep_embedder.linear_2.weight"), w.get(prefix + ".emb.timestep_embedder.linear_2.bias"));
    if (emb) {
        *emb = te;
    }
    ggml_tensor * m = mg_linear(ctx, ggml_silu(ctx, te), w.get(prefix + ".linear.weight"), w.get(prefix + ".linear.bias"));
    if (mod) {
        *mod = m;
    }
}

// row i of a [dim, n] table
inline ggml_tensor * ltx_row(ggml_context * ctx, ggml_tensor * table, int i, int64_t dim) {
    return ggml_view_2d(ctx, table, dim, 1, table->nb[1], (size_t) i * table->nb[1]);
}

// table[i] + mod[i*dim : (i+1)*dim]
inline ggml_tensor * ltx_ada(ggml_context * ctx, ggml_tensor * table, ggml_tensor * mod, int i, int64_t dim) {
    ggml_tensor * m = ggml_view_2d(ctx, mod, dim, 1, mod->nb[1], (size_t) i * dim * ggml_element_size(mod));
    return ggml_add(ctx, m, ltx_row(ctx, table, i, dim));
}
