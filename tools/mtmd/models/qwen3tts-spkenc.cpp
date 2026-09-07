#include "models.h"

static constexpr int SPK_RES2NET_SCALE = 8; // enc_res2net_scale
static constexpr int SPK_DILATIONS[3]  = { 2, 3, 4 }; // enc_dilations[1..3]

// conv1d, kernel K, padding "same" (reflect), dilation d
// x: [C, T] (ne[0]=C, ne[1]=T) -> [out_c, T]
ggml_tensor * clip_graph_qwen3tts_spkenc::conv1d_same(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation) const {
    const int K   = (int) w->ne[0];
    const int IC  = (int) w->ne[1];
    const int OC  = (int) w->ne[2];
    const int pad = ((K - 1) * dilation) / 2;

    // ggml_pad_reflect_1d pads ne[0], so bring T onto ne[0] first, same layout as im2col wants
    ggml_tensor * x_t = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [T, IC]
    if (pad > 0) {
        x_t = ggml_pad_reflect_1d(ctx0, x_t, pad, pad); // [T + 2*pad, IC]
    }
    ggml_tensor * x4d = ggml_reshape_4d(ctx0, x_t, x_t->ne[0], IC, 1, 1);

    // dummy F32 kernel, im2col only reads its shape, so a quantized w does not assert
    ggml_tensor * dummy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, K, IC, 1, 1);

    ggml_tensor * col = ggml_im2col(ctx0, dummy, x4d, 1, 1, 0, 0, dilation, 1, false, GGML_TYPE_F32);
    const int64_t T_out = col->ne[1];
    col = ggml_reshape_2d(ctx0, col, (int64_t) K * IC, T_out);

    ggml_tensor * w2d = ggml_reshape_2d(ctx0, w, (int64_t) K * IC, OC);
    ggml_tensor * y   = ggml_mul_mat(ctx0, w2d, col); // [OC, T_out]
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);

    ggml_tensor * b2d = ggml_reshape_2d(ctx0, b, OC, 1);
    y = ggml_add(ctx0, y, b2d);
    return y;
}

// Res2Net: split channel axis into `scale` chunks, chain dilated conv1d branches
// x: [C, T] -> [C, T]
ggml_tensor * clip_graph_qwen3tts_spkenc::res2net(ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const {
    const int64_t C  = x->ne[0];
    const int64_t T  = x->ne[1];
    const int64_t Cs = C / scale;

    std::vector<ggml_tensor *> outs;
    outs.reserve(scale);

    auto chunk = [&](int i) -> ggml_tensor * {
        return ggml_view_2d(ctx0, x, Cs, T, x->nb[1], (size_t) i * Cs * x->nb[0]);
    };

    ggml_tensor * prev = nullptr;
    for (int i = 0; i < scale; i++) {
        ggml_tensor * c = ggml_cont(ctx0, chunk(i));
        if (i == 0) {
            outs.push_back(c);
            continue;
        }
        ggml_tensor * inp = (i >= 2) ? ggml_add(ctx0, c, prev) : c;
        ggml_tensor * y   = conv1d_same(inp, layer.res2_conv_w[i - 1], layer.res2_conv_b[i - 1], dilation);
        y                 = ggml_relu(ctx0, y);
        outs.push_back(y);
        prev = y;
    }

    ggml_tensor * acc = outs[0];
    for (int i = 1; i < scale; i++) {
        acc = ggml_concat(ctx0, acc, outs[i], 0);
    }
    return acc;
}

// squeeze-and-excitation gate. x: [C, T] -> [C, T]
ggml_tensor * clip_graph_qwen3tts_spkenc::se_block(ggml_tensor * x, const clip_layer & layer) const {
    // temporal mean, keepdim: transpose so T is on ne[0], reduce, transpose back
    ggml_tensor * x_t  = ggml_cont(ctx0, ggml_transpose(ctx0, x));    // [T, C]
    ggml_tensor * mean = ggml_mean(ctx0, x_t);                        // [1, C]
    mean               = ggml_cont(ctx0, ggml_transpose(ctx0, mean)); // [C, 1]

    ggml_tensor * h = conv1d_same(mean, layer.se_conv1_w, layer.se_conv1_b, 1);
    h = ggml_relu(ctx0, h);
    h = conv1d_same(h, layer.se_conv2_w, layer.se_conv2_b, 1);
    h = ggml_sigmoid(ctx0, h); // [C, 1]

    return ggml_mul(ctx0, x, h); // broadcast gate over T
}

// tdnn1 -> res2net -> tdnn2 -> se, plus residual. x: [C, T] -> [C, T]
ggml_tensor * clip_graph_qwen3tts_spkenc::se_res2net_block(ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const {
    ggml_tensor * residual = x;
    ggml_tensor * h  = conv1d_same(x, layer.conv_pw1_w, layer.conv_pw1_b, 1); // tdnn1
    h = ggml_relu(ctx0, h);
    h = res2net(h, layer, dilation, scale);
    h = conv1d_same(h, layer.conv_pw2_w, layer.conv_pw2_b, 1); // tdnn2
    h = ggml_relu(ctx0, h);
    h = se_block(h, layer);
    return ggml_add(ctx0, h, residual);
}

// attentive statistics pooling. x: [C, T] -> [2*C, 1]
ggml_tensor * clip_graph_qwen3tts_spkenc::attentive_stats_pool(ggml_tensor * x) const {
    const int64_t T = x->ne[1];

    // mean over T: [C, 1]
    ggml_tensor * x_t  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * mean = ggml_mean(ctx0, x_t);
    mean               = ggml_cont(ctx0, ggml_transpose(ctx0, mean));

    // std over T: sqrt(clamp(mean((x - mean)^2), eps))
    ggml_tensor * mean_rep = ggml_repeat(ctx0, mean, x);
    ggml_tensor * centered = ggml_sub(ctx0, x, mean_rep);
    ggml_tensor * var_t    = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_sqr(ctx0, centered)));
    ggml_tensor * var      = ggml_mean(ctx0, var_t);
    var                    = ggml_cont(ctx0, ggml_transpose(ctx0, var));
    var                    = ggml_scale_bias(ctx0, var, 1.0f, 1e-12f);
    ggml_tensor * std      = ggml_sqrt(ctx0, var);

    // attention input: cat([x, mean, std]) along channel axis -> [3C, T]
    ggml_tensor * std_rep = ggml_repeat(ctx0, std, x);
    ggml_tensor * cat     = ggml_concat(ctx0, x, mean_rep, 0);
    cat                   = ggml_concat(ctx0, cat, std_rep, 0);

    // attention TDNN (3C -> attn_c) + ReLU, tanh, then 1x1 conv (attn_c -> C)
    ggml_tensor * a = conv1d_same(cat, model.spk_asp_tdnn_w, model.spk_asp_tdnn_b, 1);
    a = ggml_relu(ctx0, a);
    a = ggml_tanh(ctx0, a);
    a = conv1d_same(a, model.spk_asp_attn_w, model.spk_asp_attn_b, 1);

    // softmax over T
    ggml_tensor * a_t = ggml_cont(ctx0, ggml_transpose(ctx0, a));   // [T, C]
    ggml_tensor * w_t = ggml_soft_max(ctx0, a_t);
    ggml_tensor * w   = ggml_cont(ctx0, ggml_transpose(ctx0, w_t)); // [C, T]

    // weighted mean: sum(w * x) over T, multiply by T to undo ggml_mean's 1/T scaling
    ggml_tensor * wx     = ggml_mul(ctx0, w, x);
    ggml_tensor * wx_t   = ggml_cont(ctx0, ggml_transpose(ctx0, wx));
    ggml_tensor * w_mean = ggml_mean(ctx0, wx_t);
    w_mean = ggml_scale(ctx0, w_mean, (float) T);
    w_mean = ggml_cont(ctx0, ggml_transpose(ctx0, w_mean)); // [C, 1]

    // weighted std: sum(w * (x - w_mean)^2) over T
    ggml_tensor * w_mean_rep = ggml_repeat(ctx0, w_mean, x);
    ggml_tensor * dev        = ggml_sub(ctx0, x, w_mean_rep);
    ggml_tensor * w_var_in   = ggml_mul(ctx0, w, ggml_sqr(ctx0, dev));
    ggml_tensor * w_var_t    = ggml_cont(ctx0, ggml_transpose(ctx0, w_var_in));
    ggml_tensor * w_var      = ggml_mean(ctx0, w_var_t);
    w_var                    = ggml_scale(ctx0, w_var, (float) T);
    w_var                    = ggml_cont(ctx0, ggml_transpose(ctx0, w_var));
    w_var                    = ggml_scale_bias(ctx0, w_var, 1.0f, 1e-12f);
    ggml_tensor * w_std      = ggml_sqrt(ctx0, w_var);

    return ggml_concat(ctx0, w_mean, w_std, 0); // [2C, 1]
}

ggml_cgraph * clip_graph_qwen3tts_spkenc::build() {
    // inp_raw: [T, n_mel, 1, 1], from mtmd_audio_preprocessor_qwen3tts_spk
    ggml_tensor * inp = build_inp_raw(1);
    inp = ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1]);

    // this file's convention is [C, T]; the preprocessor delivers [T, C]
    ggml_tensor * mel = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [n_mel, T]
    cb(mel, "mel", -1);

    // frontend conv0 TDNN k=5, dilation=1: 128 -> 512
    ggml_tensor * cur = conv1d_same(mel, model.conv1d_1_w, model.conv1d_1_b, 1);
    cur = ggml_relu(ctx0, cur);
    cb(cur, "frontend", -1);

    // 3 SE-Res2Net blocks at dilations 2, 3, 4
    GGML_ASSERT((int) model.layers.size() == 3);
    std::vector<ggml_tensor *> blk_out(3);
    for (int il = 0; il < 3; il++) {
        cur = se_res2net_block(cur, model.layers[il], SPK_DILATIONS[il], SPK_RES2NET_SCALE);
        blk_out[il] = cur;
        cb(cur, "block_out", il);
    }

    // multi-layer feature aggregation: cat blk[0..2] then TDNN k=1 + ReLU
    ggml_tensor * cat = ggml_concat(ctx0, blk_out[0], blk_out[1], 0);
    cat = ggml_concat(ctx0, cat, blk_out[2], 0); // [1536, T]
    ggml_tensor * mfa = conv1d_same(cat, model.conv_out_w, model.conv_out_b, 1);
    mfa = ggml_relu(ctx0, mfa);
    cb(mfa, "mfa", -1);

    // attentive statistics pooling: [1536, T] -> [3072, 1]
    ggml_tensor * stats = attentive_stats_pool(mfa);
    cb(stats, "asp", -1);

    // final FC k=1: [3072, 1] -> [enc_dim, 1]
    ggml_tensor * emb = conv1d_same(stats, model.mm_fc_w, model.mm_fc_b, 1);

    emb = ggml_reshape_1d(ctx0, emb, emb->ne[0]);
    emb = ggml_cont(ctx0, emb);
    cb(emb, "spk_embedding", -1);

    ggml_build_forward_expand(gf, emb);
    return gf;
}
