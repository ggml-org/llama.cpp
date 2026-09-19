// Copyright (c) 2026 codec.cpp contributors
// SPDX-License-Identifier: MIT

#include "models.h"

ggml_tensor * clip_graph_nemo_nano_codec::half_snake(ggml_tensor * x, ggml_tensor * alpha) const {
    const int64_t half = x->ne[1] / 2;
    ggml_tensor * left = ggml_view_2d(ctx0, x, x->ne[0], half, x->nb[1], 0);
    ggml_tensor * right = ggml_view_2d(ctx0, x, x->ne[0], x->ne[1] - half, x->nb[1], half * x->nb[1]);
    ggml_tensor * a = ggml_reshape_2d(ctx0, alpha, 1, half);
    ggml_tensor * sine = ggml_sin(ctx0, ggml_mul(ctx0, left, a));
    left = ggml_add(ctx0, left, ggml_div(ctx0, ggml_sqr(ctx0, sine), ggml_scale_bias(ctx0, a, 1.0f, 1e-9f)));
    right = ggml_leaky_relu(ctx0, right, 0.01f, false);
    return ggml_concat(ctx0, left, right, 1);
}

ggml_tensor * clip_graph_nemo_nano_codec::conv1d(ggml_tensor * x, const clip_nemo_nano_codec::conv & c, int dilation) const {
    const int64_t padding = (c.w->ne[0] - 1) * dilation;
    x = ggml_pad_ext(ctx0, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx0, c.w, x, 1, 0, dilation);
    x = ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1]);
    return ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c.b, 1, c.b->ne[0]));
}

ggml_tensor * clip_graph_nemo_nano_codec::upsample(ggml_tensor * x, const clip_nemo_nano_codec::conv & c, int stride) const {
    const int64_t frames = x->ne[0];
    const int64_t channels = x->ne[1] / 2;
    const int64_t kernel = c.w->ne[1];
    // Each output channel sums a pair of input channels: [2, K, OC] x [2, T, OC].
    x = ggml_reshape_3d(ctx0, x, frames, 2, channels);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x = ggml_mul_mat(ctx0, c.w, x);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
    x = ggml_reshape_2d(ctx0, x, kernel * channels, frames);
    x = ggml_col2im_1d(ctx0, x, stride, channels, 0);
    // Causal transposed convolution discards the right overlap tail.
    x = ggml_cont(ctx0, ggml_view_2d(ctx0, x, frames * stride, channels, x->nb[1], 0));
    return ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c.b, 1, channels));
}

ggml_cgraph * clip_graph_nemo_nano_codec::build() {
    const auto & m = model.nemo;
    ggml_tensor * codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, m.n_groups);
    ggml_set_name(codes, "inp_codes");
    ggml_set_input(codes);
    ggml_tensor * cur = nullptr;
    for (int group = 0; group < m.n_groups; ++group) {
        auto * indices = ggml_view_1d(ctx0, codes, n_frames, group * codes->nb[1]);
        auto * embd = ggml_get_rows(ctx0, m.codebook, indices);
        cur = cur ? ggml_concat(ctx0, cur, embd, 0) : embd;
    }
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = conv1d(cur, m.pre);
    const int rates[] = {7, 7, 6, 3, 2};
    const int dilations[] = {1, 3, 5};
    for (int stage = 0; stage < 5; ++stage) {
        const auto & s = m.stages[stage];
        cur = upsample(half_snake(cur, s.up.alpha), s.up, rates[stage]);
        ggml_tensor * sum = nullptr;
        for (int block = 0; block < 3; ++block) {
            ggml_tensor * residual = cur;
            for (int layer = 0; layer < 3; ++layer) {
                const auto & first = s.res[block][layer][0];
                const auto & second = s.res[block][layer][1];
                auto * h = conv1d(half_snake(residual, first.alpha), first, dilations[layer]);
                h = conv1d(half_snake(h, second.alpha), second);
                residual = ggml_add(ctx0, residual, h);
            }
            sum = sum ? ggml_add(ctx0, sum, residual) : residual;
        }
        cur = ggml_scale(ctx0, sum, 1.0f / 3.0f);
        cb(cur, "nemo_stage", stage);
    }
    cur = conv1d(half_snake(cur, m.post.alpha), m.post);
    cur = ggml_clamp(ctx0, cur, -1.0f, 1.0f);
    ggml_set_name(cur, "out_audio");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
