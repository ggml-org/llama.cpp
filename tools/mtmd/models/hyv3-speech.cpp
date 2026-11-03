#include "models.h"

#include <algorithm>

// HYV3Speech audio tower.
//
// Reference: simulst_mm.model.audio_encoder.AudioEncoder + audio_tower.AudioTower.
// The encoder runs on a packed mel tensor [n_mel, F] that is re-chunked on a
// grid of n_window*2 = 100 frames; each chunk goes through the three stride-2
// convolutions on its own. Zero-padding a short chunk to the grid width before
// the convolutions is a no-op here: with k=3, s=2, p=1 the output row j only
// reads input [2j-1, 2j+1], which never reaches past the chunk's own length.
// Running each chunk at its true length is therefore bit-identical to padding
// it and dropping the padded rows afterwards.

int clip_graph_hyv3_speech::conv_out_len(int n) {
    if (n <= 0) {
        return 0;
    }
    const int rem = n % 100;
    if (rem == 0) {
        return (n / 100) * 13;
    }
    int a = (rem - 1) / 2 + 1;
    int b = (a - 1) / 2 + 1;
    int c = (b - 1) / 2 + 1;
    return c + (n / 100) * 13;
}

int clip_graph_hyv3_speech::n_tokens_from_mel(int n_mel_frames, int fsk) {
    const int n_enc = conv_out_len(n_mel_frames);
    return (n_enc + fsk - 1) / fsk;
}

ggml_cgraph * clip_graph_hyv3_speech::build() {
    const int n_mel = hparams.n_mel_bins;
    const int F     = img.nx();
    const int fsk   = hparams.audio_proj_downsample_rate > 0 ? hparams.audio_proj_downsample_rate : 2;
    const int grid  = 100;  // n_window * 2

    GGML_ASSERT(F > 0);
    GGML_ASSERT(img.ny() == n_mel);

    // inp_raw: [F, n_mel, 1, 1]
    ggml_tensor * inp_raw = build_inp_raw(1);
    GGML_ASSERT(inp_raw->type == GGML_TYPE_F32);

    int n_sub = (F + grid - 1) / grid;
    if (n_sub < 1) {
        n_sub = 1;
    }

    // The reference zero-pads a short tail chunk to the grid width, runs the
    // three convolutions over the padded block and only then drops the padded
    // rows. The padded region is not inert: it still receives the conv bias and
    // goes through gelu, so a chunk run at its true length is NOT the same
    // thing -- the last encoder frame of a short tail chunk comes out different,
    // and the transformer's self-attention then spreads it over every frame.
    const int     n_pad = n_sub * grid - F;
    ggml_tensor * mel   = inp_raw;
    if (n_pad > 0) {
        // one zero row, repeated to the pad width
        ggml_tensor * one = ggml_cont(
            ctx0, ggml_view_4d(ctx0, inp_raw, 1, n_mel, 1, 1, inp_raw->nb[1], inp_raw->nb[2], inp_raw->nb[3], 0));
        ggml_tensor * pad =
            ggml_repeat(ctx0, ggml_scale(ctx0, one, 0.0f), ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_pad, n_mel, 1, 1));
        mel = ggml_cont(ctx0, ggml_concat(ctx0, inp_raw, pad, 0));
    }

    auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
        x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
        return ggml_gelu_erf(ctx0, x);
    };

    ggml_tensor * enc_inp = nullptr;  // [n_enc, n_embd], frames as rows
    int64_t       n_enc   = 0;
    int           off     = 0;

    for (int i = 0; i < n_sub; ++i) {
        const int l_i = std::min(grid, F - off);  // frames that really exist
        if (l_i <= 0) {
            break;
        }

        // the convolutions always see a full grid block, padded with zeros
        ggml_tensor * x = ggml_cont(
            ctx0, ggml_view_4d(ctx0, mel, grid, n_mel, 1, 1, mel->nb[1], mel->nb[2], mel->nb[3], off * mel->nb[0]));
        off += grid;

        x = conv_block(x, model.conv2d_1_w, model.conv2d_1_b);
        x = conv_block(x, model.conv2d_2_w, model.conv2d_2_b);
        x = conv_block(x, model.conv2d_3_w, model.conv2d_3_b);
        // x: [OW, OH, OC, 1]

        // conv out is [n_mel/8, OW, OC, 1]; -> [n_mel/8, OC, OW] so that the
        // flattened feature index is f + (n_mel/8)*c, matching the reference
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 2, 0, 1, 3));
        x = ggml_reshape_2d(ctx0, x, x->ne[0] * x->ne[1], x->ne[2]);

        x = ggml_mul_mat(ctx0, model.conv_out_w, x);  // [n_embd, OW]

        // drop the rows the padding produced
        if (l_i < grid) {
            const int64_t keep = clip_graph_hyv3_speech::conv_out_len(l_i);
            GGML_ASSERT(keep <= x->ne[1]);
            x = ggml_cont(ctx0, ggml_view_2d(ctx0, x, x->ne[0], keep, x->nb[1], 0));
        }

        // sinusoidal positions: the table is [n_embd, max_pos_emb], take the leading columns
        const int64_t t_i = x->ne[1];
        ggml_tensor * pos = ggml_cont(
            ctx0, ggml_view_2d(ctx0, model.a_pos_embd, model.a_pos_embd->ne[0], t_i, model.a_pos_embd->nb[1], 0));
        x = ggml_add(ctx0, x, pos);

        // keep frames on ne1 so the stack sees [n_embd, n_enc]
        enc_inp = enc_inp ? ggml_concat(ctx0, enc_inp, x, 1) : x;
        n_enc += t_i;
    }
    enc_inp = ggml_cont(ctx0, enc_inp);
    cb(enc_inp, "enc_inp", -1);

    // transformer stack; the trailing ln_post is applied below, not by build_vit
    build_vit_opts opts;
    opts.skip_post_ln = true;
    ggml_tensor * cur = build_vit(enc_inp, n_enc, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr, nullptr, opts);
    cb(cur, "after_transformer", -1);

    cur = build_norm(cur, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.a_proj1_w, cur), model.a_proj1_b);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.a_proj2_w, cur), model.a_proj2_b);
    cb(cur, "after_proj2", -1);
    // cur: [n_embd_out, n_enc]

    // FSK adapter: stack fsk consecutive frames into one row
    const int64_t n_tok = (n_enc + fsk - 1) / fsk;
    ggml_tensor * t     = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // [n_enc, n_embd_out]
    if (n_enc != n_tok * fsk) {
        // one zero row, appended along the frame axis
        ggml_tensor * z = ggml_cont(ctx0, ggml_view_2d(ctx0, t, 1, t->ne[1], t->nb[1], 0));
        t               = ggml_concat(ctx0, t, ggml_scale(ctx0, z, 0.0f), 0);
    }

    // [n_tok*fsk, n_embd_out] -> [n_embd_out, fsk, n_tok] -> [n_embd_out*fsk, n_tok]
    ggml_tensor * s = ggml_cont(ctx0, ggml_transpose(ctx0, t));
    s               = ggml_reshape_3d(ctx0, s, s->ne[0], fsk, n_tok);
    s               = ggml_reshape_2d(ctx0, s, s->ne[0] * s->ne[1], s->ne[2]);

    ggml_tensor * out = ggml_add(ctx0, ggml_mul_mat(ctx0, model.a_mm_0_w, s), model.a_mm_0_b);
    out               = ggml_silu(ctx0, out);
    out               = ggml_add(ctx0, ggml_mul_mat(ctx0, model.a_mm_3_w, out), model.a_mm_3_b);
    out               = build_norm(out, model.a_mm_ln_w, model.a_mm_ln_b, NORM_TYPE_NORMAL, eps, -1);
    cb(out, "projected", -1);

    ggml_build_forward_expand(gf, out);
    return gf;
}
