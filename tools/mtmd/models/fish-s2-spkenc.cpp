#include "models.h"

#include <cmath>
#include <string>
#include <vector>

// reference clip -> codec frames -> backbone input rows, see DAC.encode() and DownsampleResidualVectorQuantize.encode()
// activations are [C, T]; a conv is a sum of matmuls over tap-shifted views: exact in F32, no im2col buffer

// causal conv, w is taps-major [stride * IC, OC, K / stride], see conversion/fish_s2.py
ggml_tensor * mtmd_graph_fish_s2_spkenc::conv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation) const {
    const int64_t n_tap = w->ne[2];
    const int     pad   = (int) (n_tap - 1) * stride * dilation;
    if (pad > 0) {
        x = ggml_pad_ext(ctx0, x, 0, 0, pad, 0, 0, 0, 0, 0);
    }
    if (stride > 1) {
        // stride frames per column, then each tap group is one matmul
        x = ggml_reshape_2d(ctx0, x, x->ne[0] * stride, x->ne[1] / stride);
    }
    const int64_t n_out = x->ne[1] - (n_tap - 1) * dilation;
    ggml_tensor * y = nullptr;
    for (int64_t k = 0; k < n_tap; k++) {
        ggml_tensor * wk = ggml_view_2d(ctx0, w, w->ne[0], w->ne[1], w->nb[1], (size_t) k * w->nb[2]);
        ggml_tensor * xk = ggml_view_2d(ctx0, x, x->ne[0], n_out, x->nb[1], (size_t) k * dilation * x->nb[1]);
        ggml_tensor * yk = ggml_mul_mat(ctx0, wk, xk);
        y = y ? ggml_add(ctx0, y, yk) : yk;
    }
    return ggml_add(ctx0, y, b);
}

// x + sin(alpha * x)^2 / alpha, beta = 1 / alpha is folded at conversion
// TODO @ngxson: on Metal, residual codes match the reference less often than on CPU (89-98% vs 99%), maybe the fast-math sin
ggml_tensor * mtmd_graph_fish_s2_spkenc::snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const {
    ggml_tensor * s = ggml_sqr(ctx0, ggml_sin(ctx0, ggml_mul(ctx0, x, alpha)));
    return ggml_add(ctx0, x, ggml_mul(ctx0, s, beta));
}

// causal depthwise conv (w [C, K]) -> LayerNorm -> pwconv1 -> GELU -> pwconv2 -> layer scale -> residual
ggml_tensor * mtmd_graph_fish_s2_spkenc::convnext(ggml_tensor * x, const mtmd_code2wav::upsample_block & blk) const {
    const int64_t n_tap = blk.dwconv_w->ne[1];
    const int64_t n_pos = x->ne[1];
    ggml_tensor * xp = ggml_pad_ext(ctx0, x, 0, 0, (int) n_tap - 1, 0, 0, 0, 0, 0);
    ggml_tensor * h  = nullptr;
    for (int64_t k = 0; k < n_tap; k++) {
        ggml_tensor * wk = ggml_view_1d(ctx0, blk.dwconv_w, blk.dwconv_w->ne[0], (size_t) k * blk.dwconv_w->nb[1]);
        ggml_tensor * xk = ggml_view_2d(ctx0, xp, xp->ne[0], n_pos, xp->nb[1], (size_t) k * xp->nb[1]);
        ggml_tensor * hk = ggml_mul(ctx0, xk, wk);
        h = h ? ggml_add(ctx0, h, hk) : hk;
    }
    h = ggml_add(ctx0, h, blk.dwconv_b);
    h = ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, h, 1e-6f), blk.norm_w), blk.norm_b);
    h = ggml_gelu_erf(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, blk.pw1_w, h), blk.pw1_b));
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.pw2_w, h), blk.pw2_b);
    return ggml_add(ctx0, x, ggml_mul(ctx0, h, blk.gamma));
}

ggml_cgraph * mtmd_graph_fish_s2_spkenc::build() {
    const auto & c2w = model.c2w;

    // the preprocessor hands over the waveform as a single-row "mel", [n_samples, 1] is also [1, n_samples]
    ggml_tensor * inp_raw = build_inp_raw(1);
    ggml_tensor * x = ggml_reshape_2d(ctx0, inp_raw, 1, inp_raw->ne[0]);

    // windowed causal transformer on a cold start: zero state, positions from 0
    mtmd_graph_qwen3tts_gen::code2wav dec(*this);
    auto zero_state = [&](const std::string & name, int64_t n0, int64_t n1) {
        dec.state_in[name] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n0, n1), 0.0f);
    };
    auto transformer = [&](ggml_tensor * cur, int il0, int il1, int64_t window, ggml_tensor * norm_w) {
        const int64_t n_kv = model.layers[il0].k_w->ne[1];
        zero_state("tfm_pos", 1, 1);
        for (int il = il0; il < il1; il++) {
            zero_state("tfm_k_" + std::to_string(il), n_kv, window - 1);
            zero_state("tfm_v_" + std::to_string(il), n_kv, window - 1);
            cur = dec.tfm_layer_forward(cur, model.layers[il], il, window);
        }
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps), norm_w);
    };

    // encoder: entry conv, 4x (3 residual units, Snake, strided conv), transformer, Snake, post conv
    x = conv(x, c2w.dac_entry_w, c2w.dac_entry_b, 1, 1);
    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };
    for (const auto & blk : c2w.dac) {
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & res = blk.res[ir];
            ggml_tensor * h = snake(x, res.act1_alpha, res.act1_beta);
            h = conv(h, res.conv1_w, res.conv1_b, 1, DAC_DILATIONS[ir]);
            h = snake(h, res.act2_alpha, res.act2_beta);
            h = conv(h, res.conv2_w, res.conv2_b, 1, 1);
            x = ggml_add(ctx0, x, h);
        }
        x = snake(x, blk.snake_alpha, blk.snake_beta);
        x = conv(x, blk.conv_w, blk.conv_b, (int) (blk.conv_w->ne[0] / x->ne[0]), 1);
    }
    x = transformer(x, 0, n_enc_tfm_layer, 512, model.post_ln_w);
    x = snake(x, c2w.dac_post_snake_alpha, c2w.dac_post_snake_beta);
    x = conv(x, c2w.dac_post_conv_w, c2w.dac_post_conv_b, 1, 1);
    cb(x, "codec_enc", -1);

    // quantizer: 2x (strided conv + ConvNeXt), transformer
    for (const auto & down : c2w.upsample) {
        x = conv(x, down.conv_w, down.conv_b, (int) (down.conv_w->ne[0] / x->ne[0]), 1);
        x = convnext(x, down);
    }
    ggml_tensor * z = transformer(x, n_enc_tfm_layer, (int) model.layers.size(), 128, c2w.tfm_output_norm_w);
    cb(z, "codec_latent", -1);

    // nearest entry by cosine similarity, then the entry through out_proj is removed from the residual
    auto vq = [&](ggml_tensor * in_w, ggml_tensor * in_b, ggml_tensor * cb_norm, ggml_tensor * table) {
        ggml_tensor * e = ggml_add(ctx0, ggml_mul_mat(ctx0, in_w, z), in_b);
        e = ggml_l2_norm(ctx0, e, 1e-12f);
        ggml_tensor * idx = ggml_argmax(ctx0, ggml_mul_mat(ctx0, cb_norm, e));
        z = ggml_sub(ctx0, z, ggml_get_rows(ctx0, table, idx));
        return idx;
    };
    auto page = [&](ggml_tensor * t, int64_t g) {
        return ggml_view_2d(ctx0, t, t->ne[0], t->ne[1], t->nb[1], (size_t) g * t->nb[2]);
    };

    const int64_t n_cb = model.gen_code_embd_w->ne[2];
    std::vector<ggml_tensor *> idx(n_cb);
    idx[0] = vq(c2w.quant_first_in_w, c2w.quant_first_in_b, c2w.quant_first_cb_norm_w, c2w.quant_first_cb_w);
    for (int64_t g = 1; g < n_cb; g++) {
        ggml_tensor * in_b = ggml_view_1d(ctx0, c2w.quant_rest_in_b, c2w.quant_rest_in_b->ne[0], (size_t) (g - 1) * c2w.quant_rest_in_b->nb[1]);
        idx[g] = vq(page(c2w.quant_rest_in_w, g - 1), in_b, page(c2w.quant_rest_cb_norm_w, g - 1), page(c2w.quant_rest_cb_w, g - 1));
    }
    // codes of the clip, read back as the gen-side reference
    for (int64_t g = 0; g < n_cb; g++) {
        ggml_set_name(idx[g], ("out_ref_code_" + std::to_string(g)).c_str());
        ggml_set_output(idx[g]);
        ggml_build_forward_expand(gf, idx[g]);
    }

    ggml_tensor * out = nullptr;
    if (hparams.spkenc_drama3_rows) {
        // drama3 backbone row: (E_lm[semantic] + E_cb[1][code 1]) / sqrt(2), see mtmd_graph_drama3_gen::build_tower()
        out = ggml_add(ctx0, ggml_get_rows(ctx0, model.gen_code_out_embd_w, idx[0]),
                             ggml_get_rows(ctx0, page(model.gen_code_embd_w, 1), idx[1]));
        out = ggml_scale(ctx0, out, 1.0f / sqrtf(2.0f));
    } else {
        // backbone row: (E_lm[semantic] + sum of E_cb[g][code g]) / sqrt(n_cb + 1), same as fish_s2_gen
        out = ggml_add(ctx0, ggml_get_rows(ctx0, model.gen_code_out_embd_w, idx[0]),
                             ggml_get_rows(ctx0, page(model.gen_code_embd_w, 0), idx[0]));
        for (int64_t g = 1; g < n_cb; g++) {
            out = ggml_add(ctx0, out, ggml_get_rows(ctx0, page(model.gen_code_embd_w, g), idx[g]));
        }
        out = ggml_scale(ctx0, out, 1.0f / sqrtf((float) (n_cb + 1)));
    }
    cb(out, "spk_rows", -1);
    ggml_build_forward_expand(gf, out);

    return gf;
}
