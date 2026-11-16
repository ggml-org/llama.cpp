#include "models.h"

// SnakeBeta, alpha and beta folded at conversion: x + beta * sin(alpha * x)^2
// x: [T, C], alpha/beta: [C]
ggml_tensor * mtmd_graph_yue2_gen::snake(ggml_tensor * x, const mtmd_yue2::snake & s) const {
    ggml_tensor * a = ggml_reshape_2d(ctx0, s.alpha, 1, s.alpha->ne[0]);
    ggml_tensor * b = ggml_reshape_2d(ctx0, s.beta,  1, s.beta->ne[0]);

    // expand reshapes first so mul/sin/sqr/mul/add lands as consecutive nodes, letting backends fuse them
    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);

    ggml_tensor * y = ggml_sqr(ctx0, ggml_sin(ctx0, ggml_mul(ctx0, x, a)));
    return ggml_add(ctx0, x, ggml_mul(ctx0, y, b));
}

// same-length conv, stride 1; x: [T, IC], w: [K, IC, OC], returns [T, OC]
// im2col in F32: the reference decoder runs in FP32
ggml_tensor * mtmd_graph_yue2_gen::conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation) const {
    const int pad = (int) (w->ne[0] - 1) / 2 * dilation;
    ggml_tensor * col = ggml_im2col(ctx0, w, x, 1, 0, pad, 0, dilation, 0, false, GGML_TYPE_F32); // [1, T, IC * K]
    ggml_tensor * y = ggml_mul_mat(ctx0,
            ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]),
            ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2]));
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

ggml_cgraph * mtmd_graph_yue2_gen::build() {
    const auto & y = model.yue2;

    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_IN || gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_OUT) {
        const bool is_in = gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_IN;
        ggml_tensor * w = is_in ? y.inp_w : y.out_w;
        ggml_tensor * b = is_in ? y.inp_b : y.out_b;
        ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, w->ne[0], n_rows);
        ggml_set_name(cur, "inp_feats");
        ggml_set_input(cur);
        cur = ggml_add(ctx0, build_mm(w, cur), b);
        ggml_set_name(cur, "out_feats");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    GGML_ASSERT(gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV);
    static constexpr int dilations[3] = { 1, 3, 9 };

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [T, n_latent]
    x = conv1d(x, y.vae_in_w, y.vae_in_b, 1);
    for (size_t ib = 0; ib < y.vae.size(); ib++) {
        const auto & blk = y.vae[ib];
        const int stride = (int) (blk.up_w->ne[0] / 2);
        const int pad    = (stride + 1) / 2; // ceil(stride / 2) of the reference
        x = snake(x, blk.act);
        {
            // ConvTranspose1d as GEMM + col2im, same as qwen3tts; up_w: [K, OC, IC]
            const int64_t K  = blk.up_w->ne[0];
            const int64_t OC = blk.up_w->ne[1];
            ggml_tensor * w2  = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, blk.up_w, K * OC, blk.up_w->ne[2])));
            ggml_tensor * col = ggml_mul_mat(ctx0, w2, ggml_cont(ctx0, ggml_transpose(ctx0, x))); // [K * OC, T]
            x = ggml_col2im_1d(ctx0, col, stride, (int) OC, pad);
        }
        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, blk.up_b, 1, blk.up_b->ne[0]));
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & ru = blk.res[ir];
            ggml_tensor * h = snake(x, ru.act1);
            h = conv1d(h, ru.conv1_w, ru.conv1_b, dilations[ir]);
            h = snake(h, ru.act2);
            h = conv1d(h, ru.conv2_w, ru.conv2_b, 1);
            x = ggml_add(ctx0, x, h);
        }
        cb(x, "vae_block", (int) ib);
    }
    x = snake(x, y.vae_post);
    x = conv1d(x, y.vae_out_w, nullptr, 1); // [T, n_channels]

    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // interleaved
    ggml_set_name(x, "out_audio");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}
