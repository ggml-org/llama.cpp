#include "models.h"

ggml_cgraph * clip_graph_kani_spkenc::build() {
    const auto & m = model.kani_speaker;
    auto linear = [&](ggml_tensor * x, const clip_kani_speaker::linear & l) {
        return ggml_add(ctx0, build_mm(l.w, x), l.b);
    };
    auto norm = [&](ggml_tensor * x, const clip_kani_speaker::linear & l) {
        return ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, x, 1e-5f), l.w), l.b);
    };
    auto transpose = [&](ggml_tensor * x) { return ggml_cont(ctx0, ggml_transpose(ctx0, x)); };
    auto * raw = build_inp_raw(1);
    auto * x = ggml_reshape_1d(ctx0, raw, img.nx());
    // PyTorch std() uses the sample variance (correction=1).
    auto * centered = ggml_sub(ctx0, x, ggml_mean(ctx0, x));
    auto * variance = ggml_scale(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, centered)), (float) img.nx() / (img.nx() - 1));
    x = ggml_div(ctx0, centered, ggml_scale_bias(ctx0, ggml_sqrt(ctx0, variance), 1.0f, 1e-10f));
    x = ggml_reshape_2d(ctx0, x, img.nx(), 1);
    const int strides[] = {5, 2, 2, 2, 2, 2, 2};
    for (int i = 0; i < 7; ++i) {
        x = ggml_conv_1d(ctx0, m.convs[i].w, x, strides[i], 0, 1);
        x = transpose(ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1]));
        x = ggml_gelu_erf(ctx0, norm(x, m.convs[i].norm));
        cb(x, "kani_conv", i);
        x = transpose(x);
    }
    x = linear(norm(transpose(x), m.feature_norm), m.feature_proj);
    cb(x, "kani_feature", -1);
    const int64_t nt = x->ne[1];
    auto * xt = transpose(x);
    ggml_tensor * pos = nullptr;
    for (int group = 0; group < 16; ++group) {
        auto * w = ggml_view_3d(ctx0, m.pos.w, 128, 64, 64, m.pos.w->nb[1], m.pos.w->nb[2], group * m.pos.w->nb[3]);
        auto * chunk = ggml_cont(ctx0, ggml_view_2d(ctx0, xt, nt, 64, xt->nb[1], group * 64 * xt->nb[1]));
        chunk = ggml_conv_1d(ctx0, w, chunk, 1, 64, 1);
        chunk = ggml_cont(ctx0, ggml_view_2d(ctx0, chunk, nt, 64, chunk->nb[1], 0));
        pos = pos ? ggml_concat(ctx0, pos, chunk, 1) : chunk;
    }
    pos = ggml_gelu_erf(ctx0, ggml_add(ctx0, transpose(pos), m.pos.b));
    x = ggml_add(ctx0, x, pos);
    cb(x, "kani_position", -1);

    auto * buckets = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nt * nt);
    ggml_set_name(buckets, "kani_buckets");
    ggml_set_input(buckets);
    auto * bias = transpose(ggml_get_rows(ctx0, m.relative, buckets));
    bias = ggml_reshape_3d(ctx0, bias, nt, nt, 16);
    for (int i = 0; i < 24; ++i) {
        const auto & l = m.layers[i];
        auto * h = norm(x, l.norm);
        auto * gh = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, h, 64, 16, nt), 0, 2, 1, 3));
        auto * gates = ggml_sigmoid(ctx0, linear(gh, l.gate));
        auto * ga = ggml_view_3d(ctx0, gates, 1, nt, 16, gates->nb[1], gates->nb[2], 0);
        auto * gb = ggml_view_3d(ctx0, gates, 1, nt, 16, gates->nb[1], gates->nb[2], sizeof(float));
        auto * gate = ggml_scale_bias(ctx0, ggml_mul(ctx0, ga,
                ggml_scale_bias(ctx0, ggml_mul(ctx0, gb, l.gate_const), 1.0f, -1.0f)), 1.0f, 2.0f);
        auto * gated_bias = ggml_mul(ctx0, bias, gate);
        auto heads = [&](const clip_kani_speaker::linear & proj) {
            return ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, linear(h, proj), 64, 16, nt), 0, 2, 1, 3));
        };
        auto * q = heads(l.q);
        auto * k = heads(l.k);
        auto * v = heads(l.v);
        auto * scores = ggml_mul_mat(ctx0, k, q);
        scores = ggml_soft_max_ext(ctx0, scores, gated_bias, 0.125f, 0.0f);
        auto * attn = ggml_mul_mat(ctx0, transpose(v), scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        x = ggml_add(ctx0, x, linear(ggml_reshape_2d(ctx0, attn, 1024, nt), l.o));
        h = ggml_gelu_erf(ctx0, linear(norm(x, l.ffn_norm), l.up));
        x = ggml_add(ctx0, x, linear(h, l.down));
        cb(x, "kani_layer", i);
    }
    x = transpose(norm(x, m.output_norm));
    auto * mean = ggml_mean(ctx0, x);
    auto * delta = ggml_sub(ctx0, x, mean);
    auto * var = ggml_scale(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, delta)), (float) nt / (nt - 1));
    auto * std = ggml_sqrt(ctx0, ggml_clamp(ctx0, var, 1e-10f, INFINITY));
    x = ggml_concat(ctx0, transpose(mean), transpose(std), 0);
    for (int i = 0; i < 2; ++i) {
        x = ggml_relu(ctx0, linear(x, m.top[i]));
        x = ggml_add(ctx0, ggml_mul(ctx0, x, m.top_norm[i].w), m.top_norm[i].b);
    }
    x = ggml_l2_norm(ctx0, x, 1e-12f);
    cb(x, "kani_speaker_embedding", -1);
    x = build_mm(m.projection, x);
    cb(x, "kani_speaker_projection", -1);
    ggml_build_forward_expand(gf, x);
    return gf;
}
