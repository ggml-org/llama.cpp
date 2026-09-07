// ABOUTME: Yasa2 vision encoder graph builder for ConvNeXt-based architecture.
// ABOUTME: Implements patch embedding, ConvNeXt stages with GRN, and adaptive pooling.

#include "models.h"

static ggml_tensor * add_channel_bias(
        ggml_context * ctx0,
        ggml_tensor * x_whcb,
        ggml_tensor * b_c) {
    if (!b_c) {
        return x_whcb;
    }
    ggml_tensor * b4 = ggml_reshape_4d(ctx0, b_c, 1, 1, b_c->ne[0], 1);
    return ggml_add(ctx0, x_whcb, b4);
}

static ggml_tensor * mul_channel_weight(
        ggml_context * ctx0,
        ggml_tensor * x_whcb,
        ggml_tensor * w_c) {
    if (!w_c) {
        return x_whcb;
    }
    ggml_tensor * w4 = ggml_reshape_4d(ctx0, w_c, 1, 1, w_c->ne[0], 1);
    return ggml_mul(ctx0, x_whcb, w4);
}

ggml_tensor * clip_graph_yasa2::layer_norm_channels(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b, float eps) {
    // Match HF ConvNextLayerNorm(channels_first):
    // u = mean_c(x), s = mean_c((x-u)^2), x = (x-u)/sqrt(s+eps)
    // cast back to input dtype before affine.
    ggml_tensor * cur = ggml_permute(ctx0, inp, 2, 1, 0, 3); // [W,H,C,B] -> [C,H,W,B]
    cur = ggml_cont(ctx0, cur);

    ggml_tensor * u = ggml_mean(ctx0, cur);                 // [1,H,W,B]
    ggml_tensor * xm = ggml_sub(ctx0, cur, u);              // [C,H,W,B]

    ggml_tensor * s = ggml_mul(ctx0, xm, xm);               // [C,H,W,B]
    s = ggml_mean(ctx0, s);                                 // [1,H,W,B]
    s = ggml_clamp(ctx0, s, eps, 1e30f);                    // avoid div-by-zero in no-alloc warmup
    s = ggml_sqrt(ctx0, s);                                 // [1,H,W,B]

    ggml_tensor * xhat = ggml_div(ctx0, xm, s);             // [C,H,W,B]
    xhat = ggml_permute(ctx0, xhat, 2, 1, 0, 3);            // [W,H,C,B]
    xhat = ggml_cont(ctx0, xhat);
    xhat = mul_channel_weight(ctx0, xhat, w);
    xhat = add_channel_bias(ctx0, xhat, b);
    return xhat;
}

ggml_tensor * clip_graph_yasa2::convnext_grn(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b) {
    // Exact ConvNeXtV2 GRN:
    // Gx = ||x||_2 over spatial dims (W,H), Nx = Gx / (mean_c(Gx) + eps)
    // y  = w * (x * Nx) + b + x
    const int64_t wdim = inp->ne[0];
    const int64_t hdim = inp->ne[1];
    const int64_t cdim = inp->ne[2];
    const int64_t bdim = inp->ne[3];

    // Keep GRN math in fp32 for stability; fp16/bf16 accumulation can drift.
    ggml_tensor * sq = ggml_mul(ctx0, inp, inp);
    ggml_tensor * sq_flat = ggml_reshape_4d(ctx0, sq, wdim * hdim, cdim, 1, bdim);   // [WH,C,1,B]
    ggml_tensor * gx = ggml_sum_rows(ctx0, sq_flat);                                   // [1,C,1,B]
    gx = ggml_sqrt(ctx0, gx);                                                           // [1,C,1,B]

    ggml_tensor * gx_ch_first = ggml_permute(ctx0, gx, 1, 0, 2, 3);                    // [C,1,1,B]
    gx_ch_first = ggml_cont(ctx0, gx_ch_first);
    ggml_tensor * gx_mean = ggml_mean(ctx0, gx_ch_first);                              // [1,1,1,B]

    gx_mean = ggml_clamp(ctx0, gx_mean, 1e-6f, 1e30f);                                  // approx +eps, warmup-safe
    ggml_tensor * nx = ggml_div(ctx0, gx, gx_mean);                                    // [1,C,1,B]
    nx = ggml_permute(ctx0, nx, 0, 2, 1, 3);                                            // [1,1,C,B]
    nx = ggml_cont(ctx0, nx);

    ggml_tensor * xnx = ggml_mul(ctx0, inp, nx);
    xnx = mul_channel_weight(ctx0, xnx, w);
    xnx = add_channel_bias(ctx0, xnx, b);
    return ggml_add(ctx0, inp, xnx);
}

ggml_cgraph * clip_graph_yasa2::build() {
    ggml_tensor * cur = build_inp_raw();

    // Patch embedding Conv2d(kernel=4, stride=4)
    cur = ggml_conv_2d(ctx0, model.yasa_patch_w, cur, patch_size, patch_size, 0, 0, 1, 1);
    cur = add_channel_bias(ctx0, cur, model.yasa_patch_b);
    ggml_set_name(cur, "yasa2_patch_conv_out");
    cb(cur, "yasa2_patch_conv_out", -1);
    cur = layer_norm_channels(cur, model.yasa_patch_ln_w, model.yasa_patch_ln_b, eps);
    ggml_set_name(cur, "yasa2_patch_ln_out");
    cb(cur, "yasa2_patch_ln_out", -1);

    // ConvNeXt stages
    for (size_t s = 0; s < model.yasa_stages.size(); ++s) {
        const auto & stage = model.yasa_stages[s];

        if (stage.down_conv_w) {
            cur = layer_norm_channels(cur, stage.down_ln_w, stage.down_ln_b, eps);
            cur = ggml_conv_2d(ctx0, stage.down_conv_w, cur, 2, 2, 0, 0, 1, 1);
            cur = add_channel_bias(ctx0, cur, stage.down_conv_b);
            ggml_format_name(cur, "yasa2_stage%zu_down_out", s);
        }

        for (size_t bi = 0; bi < stage.blocks.size(); ++bi) {
            const auto & blk = stage.blocks[bi];
            ggml_tensor * res = cur;

            ggml_tensor * x = ggml_conv_2d_dw(ctx0, blk.dw_w, cur, 1, 1, 3, 3, 1, 1);
            x = add_channel_bias(ctx0, x, blk.dw_b);
            x = layer_norm_channels(x, blk.ln_w, blk.ln_b, eps);

            // pwconv1/pwconv2 are HF Linear layers over channels; implement via matmul on tokens.
            const int64_t w = x->ne[0];
            const int64_t h = x->ne[1];
            const int64_t b = x->ne[3];

            ggml_tensor * tok = ggml_reshape_3d(ctx0, x, w * h, x->ne[2], b); // [T,C,B]
            tok = ggml_permute(ctx0, tok, 1, 0, 2, 3);                        // [C,T,B]
            tok = ggml_cont(ctx0, tok);

            tok = ggml_mul_mat(ctx0, blk.pw1_w, tok);                         // [4C,T,B]
            if (blk.pw1_b) {
                ggml_tensor * b1 = ggml_reshape_3d(ctx0, blk.pw1_b, blk.pw1_b->ne[0], 1, 1); // [4C,1,1]
                tok = ggml_add(ctx0, tok, b1);
            }
            x = ggml_permute(ctx0, tok, 1, 0, 2, 3);                         // [T,4C,B]
            x = ggml_cont(ctx0, x);
            x = ggml_reshape_4d(ctx0, x, w, h, tok->ne[0], b);               // [W,H,4C,B]
            x = ggml_gelu_erf(ctx0, x);
            x = convnext_grn(x, blk.grn_w, blk.grn_b);

            tok = ggml_reshape_3d(ctx0, x, w * h, x->ne[2], b);              // [T,4C,B]
            tok = ggml_permute(ctx0, tok, 1, 0, 2, 3);                       // [4C,T,B]
            tok = ggml_cont(ctx0, tok);

            tok = ggml_mul_mat(ctx0, blk.pw2_w, tok);                        // [C,T,B]
            if (blk.pw2_b) {
                ggml_tensor * b2 = ggml_reshape_3d(ctx0, blk.pw2_b, blk.pw2_b->ne[0], 1, 1); // [C,1,1]
                tok = ggml_add(ctx0, tok, b2);
            }
            x = ggml_permute(ctx0, tok, 1, 0, 2, 3);                         // [T,C,B]
            x = ggml_cont(ctx0, x);
            x = ggml_reshape_4d(ctx0, x, w, h, tok->ne[0], b);               // [W,H,C,B]

            cur = ggml_add(ctx0, res, x);
            ggml_format_name(cur, "yasa2_stage%zu_blk%zu_out", s, bi);
        }
    }

    // HF path adds vision position embeddings BEFORE adaptive pooling.
    const int64_t pre_w = cur->ne[0];
    const int64_t pre_h = cur->ne[1];
    ggml_tensor * tokens_pre = ggml_reshape_3d(ctx0, cur, pre_w * pre_h, cur->ne[2], cur->ne[3]); // [T,C,B]
    tokens_pre = ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [C,T,B]
    tokens_pre = ggml_cont(ctx0, tokens_pre);
    if (model.yasa_vision_pos_embed && tokens_pre->ne[1] == model.yasa_vision_pos_embed->ne[1]) {
        const int64_t n_ch = model.yasa_vision_pos_embed->ne[0];
        const int64_t n_tokens = model.yasa_vision_pos_embed->ne[1];
        ggml_tensor * pos = ggml_reshape_3d(ctx0, model.yasa_vision_pos_embed, (int) n_ch, (int) n_tokens, 1);
        tokens_pre = ggml_add(ctx0, tokens_pre, pos);
    }
    cur = ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [T,C,B]
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_4d(ctx0, cur, pre_w, pre_h, cur->ne[1], cur->ne[2]); // [W,H,C,B]

    // AdaptiveAvgPool2d target is 8x8 for real inputs, but warmup can use tiny images.
    const int pooled_w = std::min(8, (int) cur->ne[0]);
    const int pooled_h = std::min(8, (int) cur->ne[1]);
    const int kw = std::max(1, (int) cur->ne[0] / pooled_w);
    const int kh = std::max(1, (int) cur->ne[1] / pooled_h);
    cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kw, kh, kw, kh, 0, 0);

    // [W,H,C,B] -> [C,T,B]
    ggml_tensor * tokens = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2], cur->ne[3]);
    tokens = ggml_permute(ctx0, tokens, 1, 0, 2, 3);
    tokens = ggml_cont(ctx0, tokens);
    cb(tokens, "yasa2_tokens", -1);

    GGML_ASSERT(model.mm_0_w && model.mm_2_w);
    ggml_tensor * embeddings = build_ffn(
        tokens,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF,
        -1);
    cb(embeddings, "yasa2_emb", -1);

    ggml_build_forward_expand(gf, embeddings);
    return gf;
}
