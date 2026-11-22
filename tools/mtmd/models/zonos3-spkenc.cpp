#include "models.h"

// zonos3 speaker encoder: MECT-B2 + ASTP pooling (ownmect_s2), then the speaker projection of the backbone
// ported from the reference ggml runtime (spk-encoder.cpp)

namespace {

constexpr int kHidden   = 64;
constexpr int kFeat     = 2560;
constexpr int kHeads    = 4;
constexpr int kExperts  = 4;
constexpr int kPosK     = 113;

// per block: cnn depth, stride f, stride t
struct mect_block { int cnn, s_f, s_t; };
const std::vector<std::vector<mect_block>> & mect_b2() {
    static const std::vector<std::vector<mect_block>> cfg = {
        { {3, 1, 1}, {3, 1, 1} },
        { {3, 2, 1}, {3, 1, 2}, {3, 1, 2}, {3, 1, 2} },
        { {3, 2, 1}, {3, 1, 2}, {3, 1, 2}, {3, 1, 2}, {3, 1, 2} },
        { {3, 2, 1}, {2, 1, 2}, {2, 1, 2} },
    };
    return cfg;
}

} // namespace

ggml_tensor * mtmd_graph_zonos3_spkenc::W(const std::string & name) const {
    return model.zonos3_spk.at("a.spk." + name);
}

bool mtmd_graph_zonos3_spkenc::has(const std::string & name) const {
    return model.zonos3_spk.count("a.spk." + name) > 0;
}

// [C] as a per-channel broadcast over a [T, F, C] activation
ggml_tensor * mtmd_graph_zonos3_spkenc::chan(ggml_tensor * v) const {
    return ggml_reshape_4d(ctx0, v, 1, 1, v->ne[0], 1);
}

ggml_tensor * mtmd_graph_zonos3_spkenc::bn(ggml_tensor * x, const std::string & p) const {
    return ggml_add(ctx0, ggml_mul(ctx0, x, chan(W(p + ".scale"))), chan(W(p + ".shift")));
}

ggml_tensor * mtmd_graph_zonos3_spkenc::conv(ggml_tensor * x, const std::string & p, int s_t, int s_f, bool bias, int pad) const {
    ggml_tensor * y = ggml_conv_2d_direct(ctx0, W(p + ".weight"), x, s_t, s_f, pad, pad, 1, 1);
    return bias ? ggml_add(ctx0, y, chan(W(p + ".bias"))) : y;
}

// grouped 3x3 conv as one depthwise conv per input slot of a group
ggml_tensor * mtmd_graph_zonos3_spkenc::grouped3x3(ggml_tensor * x, const std::string & p, int s_t, int s_f) const {
    int n_slot = 0;
    while (has(p + ".w" + std::to_string(n_slot))) {
        n_slot++;
    }
    const int64_t T  = x->ne[0];
    const int64_t F  = x->ne[1];
    const int64_t IC = x->ne[2];
    const int64_t OC = W(p + ".w0")->ne[3];
    const int64_t G  = IC / n_slot;
    const int64_t B  = OC / G;
    ggml_tensor * x4 = ggml_reshape_4d(ctx0, x, T, F, n_slot, G);
    ggml_tensor * y  = nullptr;
    for (int j = 0; j < n_slot; j++) {
        ggml_tensor * xj = x;
        if (n_slot > 1 || B > 1) {
            ggml_tensor * slot = ggml_view_4d(ctx0, x4, T, F, 1, G, x4->nb[1], x4->nb[2], x4->nb[3], (size_t) j * x4->nb[2]);
            xj = ggml_reshape_3d(ctx0, ggml_repeat_4d(ctx0, slot, T, F, B, G), T, F, OC);
        }
        ggml_tensor * t = ggml_conv_2d_dw_direct(ctx0, W(p + ".w" + std::to_string(j)), xj, s_t, s_f, 1, 1, 1, 1);
        y = y ? ggml_add(ctx0, y, t) : t;
    }
    return y;
}

ggml_tensor * mtmd_graph_zonos3_spkenc::res_block(ggml_tensor * x, const std::string & q, int s_f, int s_t) const {
    ggml_tensor * t = grouped3x3(x, q + "conv1", s_t, s_f);
    t = ggml_relu(ctx0, bn(conv(t, q + "pw1", 1, 1, true, 0), q + "bn1"));
    t = grouped3x3(t, q + "conv2", 1, 1);
    t = ggml_relu(ctx0, bn(conv(t, q + "pw2", 1, 1, true, 0), q + "bn2"));
    ggml_tensor * sc = x;
    if (has(q + "sc.weight")) {
        sc = bn(conv(x, q + "sc", s_t, s_f, false, 0), q + "sc.bn");
    }
    return ggml_add(ctx0, t, sc);
}

ggml_tensor * mtmd_graph_zonos3_spkenc::linear(ggml_tensor * x, const std::string & p) const {
    return ggml_add(ctx0, ggml_mul_mat(ctx0, W(p + ".weight"), x), W(p + ".bias"));
}

ggml_tensor * mtmd_graph_zonos3_spkenc::layer_norm(ggml_tensor * x, const std::string & p) const {
    return ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, x, 1e-6f), W(p + ".weight")), W(p + ".bias"));
}

// [T, F, C] -> [F*C, T], k = f*C + c
ggml_tensor * mtmd_graph_zonos3_spkenc::flatten(ggml_tensor * x) const {
    const int64_t T = x->ne[0], F = x->ne[1], C = x->ne[2];
    return ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, x, 2, 1, 0, 3)), F * C, T);
}

ggml_tensor * mtmd_graph_zonos3_spkenc::unflatten(ggml_tensor * x, int64_t F, int64_t C) const {
    return ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, x, C, F, x->ne[1]), 2, 1, 0, 3));
}

ggml_tensor * mtmd_graph_zonos3_spkenc::attention(ggml_tensor * h, const std::string & p) const {
    const int64_t T  = h->ne[1];
    const int     hd = kHidden / kHeads;
    ggml_tensor * q = ggml_scale(ctx0, linear(h, p + "attn.q"), 1.0f / sqrtf((float) hd));
    ggml_tensor * k = linear(h, p + "attn.k");
    ggml_tensor * v = linear(h, p + "attn.v");
    auto heads = [&](ggml_tensor * z) { // [64, T] -> [hd, T, n_head]
        return ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, z, hd, kHeads, T), 0, 2, 1, 3));
    };
    ggml_tensor * qp = heads(q);
    ggml_tensor * kp = heads(k);
    ggml_tensor * vt = ggml_cont(ctx0, ggml_permute(ctx0, heads(v), 1, 0, 2, 3)); // [T, hd, n_head]
    // queries in blocks, so long clips do not need a T x T score matrix per head
    ggml_tensor * o = nullptr;
    for (int64_t a = 0; a < T; a += 512) {
        const int64_t nq = std::min<int64_t>(512, T - a);
        ggml_tensor * qb    = ggml_view_3d(ctx0, qp, hd, nq, kHeads, qp->nb[1], qp->nb[2], (size_t) a * qp->nb[1]);
        ggml_tensor * score = ggml_soft_max(ctx0, ggml_mul_mat(ctx0, kp, qb)); // [T, nq, n_head]
        ggml_tensor * ob    = ggml_mul_mat(ctx0, vt, score);                   // [hd, nq, n_head]
        ob = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, ob, 0, 2, 1, 3)), kHidden, nq);
        o = o ? ggml_concat(ctx0, o, ob, 1) : ob;
    }
    return linear(o, p + "attn.o");
}

// soft MoE: all experts run, weighted by the router
ggml_tensor * mtmd_graph_zonos3_spkenc::moe_ffn(ggml_tensor * h, const std::string & p) const {
    ggml_tensor * r = ggml_soft_max(ctx0, ggml_mul_mat(ctx0, W(p + "moe.gate.weight"), h)); // [n_expert, T]
    ggml_tensor * e = ggml_gelu(ctx0, ggml_mul_mat(ctx0, W(p + "moe.experts.weight"), h));  // [64 * n_expert, T]
    ggml_tensor * mix = nullptr;
    for (int i = 0; i < kExperts; i++) {
        ggml_tensor * ei = ggml_view_2d(ctx0, e, kHidden, e->ne[1], e->nb[1], (size_t) i * kHidden * ggml_element_size(e));
        ggml_tensor * ri = ggml_view_2d(ctx0, r, 1, r->ne[1], r->nb[1], (size_t) i * ggml_element_size(r));
        ggml_tensor * t  = ggml_mul(ctx0, ei, ri);
        mix = mix ? ggml_add(ctx0, mix, t) : t;
    }
    return linear(ggml_gelu(ctx0, mix), p + "ffn_out");
}

ggml_tensor * mtmd_graph_zonos3_spkenc::pos_conv(ggml_tensor * h, const std::string & p) const {
    const int64_t T = h->ne[1];
    ggml_tensor * ht = ggml_cont(ctx0, ggml_transpose(ctx0, h));               // [T, 64]
    ht = ggml_pad_ext(ctx0, ht, kPosK / 2, kPosK / 2, 0, 0, 0, 0, 0, 0);        // [T + K - 1, 64]
    ggml_tensor * acc = ggml_ssm_conv(ctx0, ggml_reshape_3d(ctx0, ht, ht->ne[0], kHidden, 1), W(p + "pos.dw.weight")); // [64, T, 1]
    acc = ggml_add(ctx0, ggml_reshape_2d(ctx0, acc, kHidden, T), W(p + "pos.dw.bias"));
    acc = ggml_add(ctx0, ggml_mul(ctx0, acc, W(p + "pos.bn.scale")), W(p + "pos.bn.shift"));
    acc = ggml_gelu_erf(ctx0, acc);
    return linear(acc, p + "pos.pw");
}

ggml_cgraph * mtmd_graph_zonos3_spkenc::build() {
    const int64_t T = img.nx();
    const int64_t n_mel = img.ny();

    ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T, n_mel, 1, 1);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    ggml_tensor * x  = bn(conv(inp, "fe.head.conv", 1, 1, true, 1), "fe.head.bn"); // [T, F, C]
    ggml_tensor * xf = flatten(x);
    std::vector<ggml_tensor *> feats;
    const auto & stages = mect_b2();
    for (size_t s = 0; s < stages.size(); s++) {
        for (size_t k = 0; k < stages[s].size(); k++) {
            const auto & b = stages[s][k];
            const std::string p = "fe.b" + std::to_string(s + 1) + "." + std::to_string(k) + ".";
            ggml_tensor * y = x;
            for (int r = 0; r < b.cnn; r++) {
                y = res_block(y, p + "cnn." + std::to_string(r) + ".", r == 0 ? b.s_f : 1, r == 0 ? b.s_t : 1);
            }
            if (has(p + "reduce.weight")) {
                y = bn(conv(y, p + "reduce", 1, 1, true, 0), p + "reduce.bn");
            }
            const int64_t F2 = y->ne[1], C2 = y->ne[2];
            ggml_tensor * x2 = flatten(y); // [2560, T']
            ggml_tensor * h  = layer_norm(linear(x2, p + "rproj"), p + "rproj.ln");
            h = ggml_add(ctx0, h, pos_conv(h, p));
            h = layer_norm(ggml_add(ctx0, h, attention(h, p)), p + "ln1");
            h = layer_norm(ggml_add(ctx0, h, moe_ffn(h, p)), p + "ln2");
            ggml_tensor * e = ggml_add(ctx0, linear(h, p + "eproj"), x2);
            if (b.s_t > 1) {
                // nearest upsampling back to the input frames
                const int64_t Tp = e->ne[1];
                e = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e, kFeat, 1, Tp), kFeat, b.s_t, Tp, 1);
                e = ggml_view_2d(ctx0, ggml_reshape_2d(ctx0, e, kFeat, (int64_t) b.s_t * Tp), kFeat, T, e->nb[1], 0);
            }
            xf = ggml_add(ctx0, e, xf);
            x  = unflatten(xf, F2, C2);
        }
        feats.push_back(xf);
    }

    ggml_tensor * dense = W("fe.dense_w"); // [2560, n_stage]
    ggml_tensor * fused = nullptr;
    for (size_t s = 0; s < feats.size(); s++) {
        ggml_tensor * t = ggml_mul(ctx0, feats[s], ggml_view_2d(ctx0, dense, kFeat, 1, dense->nb[1], s * dense->nb[1]));
        fused = fused ? ggml_add(ctx0, fused, t) : t;
    }

    // ASTP: attention-weighted mean and std over time
    ggml_tensor * att   = linear(ggml_tanh(ctx0, linear(fused, "pool.l1")), "pool.l2");   // [2560, T]
    ggml_tensor * alpha = ggml_soft_max(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, att))); // [T, 2560]
    ggml_tensor * xt    = ggml_cont(ctx0, ggml_transpose(ctx0, fused));                    // [T, 2560]
    ggml_tensor * mean  = ggml_sum_rows(ctx0, ggml_mul(ctx0, alpha, xt));                  // [1, 2560]
    ggml_tensor * sq    = ggml_sum_rows(ctx0, ggml_mul(ctx0, alpha, ggml_mul(ctx0, xt, xt)));
    ggml_tensor * var   = ggml_sub(ctx0, sq, ggml_mul(ctx0, mean, mean));
    ggml_tensor * stdv  = ggml_sqrt(ctx0, ggml_clamp(ctx0, var, 1e-7f, 3.0e38f));
    ggml_tensor * pooled = ggml_concat(ctx0, ggml_reshape_2d(ctx0, mean, kFeat, 1), ggml_reshape_2d(ctx0, stdv, kFeat, 1), 0);
    pooled = ggml_add(ctx0, ggml_mul(ctx0, pooled, W("bn.scale")), W("bn.shift"));
    ggml_tensor * spk = ggml_mul_mat(ctx0, W("linear.weight"), pooled); // [192, 1], not normalized
    cb(spk, "spk_embd", -1);

    ggml_tensor * out = linear(spk, "proj"); // speaker row of the prompt
    ggml_build_forward_expand(gf, out);
    return gf;
}
