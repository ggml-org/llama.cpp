#include "models.h"

ggml_tensor * mtmd_graph_voxcpm2_spkenc::conv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation, int pad, bool depthwise) const {
    if (pad > 0) {
        x = ggml_pad_ext(ctx0, x, pad, 0, 0, 0, 0, 0, 0, 0);
    }
    // im2col in F32: the reference AudioVAE runs in FP32
    ggml_tensor * y = nullptr;
    if (depthwise) {
        ggml_tensor * col = ggml_im2col(ctx0, w, ggml_reshape_4d(ctx0, x, x->ne[0], 1, x->ne[1], 1), stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
        y = ggml_mul_mat(ctx0, col, w); // [T, 1, C]
        y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[2]);
    } else {
        ggml_tensor * col = ggml_im2col(ctx0, w, x, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32); // [IC * K, T]
        y = ggml_mul_mat(ctx0,
                ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]),
                ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2])); // [T, OC]
    }
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

// causal AudioVAE encoder: waveform -> mu of the latents, 4 latents per output row
ggml_cgraph * mtmd_graph_voxcpm2_spkenc::build() {
    using g = mtmd_graph_voxcpm2_gen;
    const auto & m = model.voxcpm2;
    mtmd_graph_qwen3tts_gen::code2wav act(*this);

    // the preprocessor hands over the waveform as a single-row "mel"
    ggml_tensor * inp_raw = build_inp_raw(1);
    ggml_tensor * x = ggml_reshape_2d(ctx0, inp_raw, inp_raw->ne[0], 1); // [T, 1]

    x = conv(x, m.enc_conv_in_w, m.enc_conv_in_b, 1, 1, (int) m.enc_conv_in_w->ne[0] - 1, false);
    for (size_t ib = 0; ib < m.enc.size(); ib++) {
        const auto & blk = m.enc[ib];
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & ru  = blk.res[ir];
            const int    dil = g::res_dilations[ir];
            ggml_tensor * h = act.snake(x, ru.act1_alpha, ru.act1_beta);
            h = conv(h, ru.conv1_w, ru.conv1_b, 1, dil, (int) (ru.conv1_w->ne[0] - 1) * dil, true);
            h = act.snake(h, ru.act2_alpha, ru.act2_beta);
            h = conv(h, ru.conv2_w, ru.conv2_b, 1, 1, 0, false);
            x = ggml_add(ctx0, x, h);
        }
        x = act.snake(x, blk.snake_alpha, blk.snake_beta);
        // kernel 2 * stride, the causal padding is one stride
        const int stride = (int) (blk.conv_w->ne[0] / 2);
        x = conv(x, blk.conv_w, blk.conv_b, stride, 1, stride, false);
        cb(x, "vae_enc_block", (int) ib);
    }
    x = conv(x, m.enc_mu_w, m.enc_mu_b, 1, 1, (int) m.enc_mu_w->ne[0] - 1, false); // [T / hop, n_latent]

    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    x = ggml_reshape_2d(ctx0, x, g::n_latent * g::n_patch, x->ne[1] / g::n_patch);
    ggml_build_forward_expand(gf, x);
    return gf;
}
