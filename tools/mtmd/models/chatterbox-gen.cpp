#define _USE_MATH_DEFINES // for M_PI
#include "models.h"

// Chatterbox generation graphs: flow encoder, cfm estimator, s3 tokenizer and hift vocoder

// x [C, T]: y = W x + b with torch Linear weights stored as [in, out]
ggml_tensor * clip_graph_chatterbox_base::cbx_linear(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) const {
    ggml_tensor * y = ggml_mul_mat(ctx0, w, x);
    if (b) {
        y = ggml_add(ctx0, y, b);
    }
    return y;
}

// conv1d over time with asymmetric zero padding: x [C, T] -> [OC, T_out], kernel [K, IC, OC]
ggml_tensor * clip_graph_chatterbox_base::cbx_conv1d(ggml_tensor * k, ggml_tensor * b, ggml_tensor * x,
                                                     int stride, int pad_l, int pad_r) const {
    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [T, C]
    if (pad_l > 0) {
        ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pad_l, xt->ne[1]);
        z = ggml_scale(ctx0, z, 0.0f);
        xt = ggml_concat(ctx0, z, xt, 0);
    }
    if (pad_r > 0) {
        ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pad_r, xt->ne[1]);
        z = ggml_scale(ctx0, z, 0.0f);
        xt = ggml_concat(ctx0, xt, z, 0);
    }
    ggml_tensor * y = ggml_conv_1d(ctx0, k, xt, stride, 0, 1); // [T_out, OC]
    y = ggml_cont(ctx0, ggml_transpose(ctx0, y));              // [OC, T_out]
    if (b) {
        y = ggml_add(ctx0, y, b);
    }
    return y;
}

// x [C, T] -> symmetric-padded dilated conv -> [OC, T]
ggml_tensor * clip_graph_chatterbox_base::cbx_conv1d_dil(ggml_tensor * k, ggml_tensor * b, ggml_tensor * x,
                                                         int pad, int dil) const {
    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * y = ggml_conv_1d(ctx0, k, xt, 1, pad, dil);
    y = ggml_cont(ctx0, ggml_transpose(ctx0, y));
    if (b) {
        y = ggml_add(ctx0, y, b);
    }
    return y;
}

// Transformer-XL relative shift: bd [2T-1, T, H] -> [T, T, H] with out[j, i, h] = bd[(T-1) - i + j, i, h]
// same buffer walk as the espnet rel_shift: pad one column, drop the first row, keep the first T columns
static ggml_tensor * cbx_rel_shift(ggml_context * ctx0, ggml_tensor * bd, int T) {
    const int H = (int) bd->ne[2];
    ggml_tensor * z = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, T, H);
    z = ggml_scale(ctx0, z, 0.0f);
    ggml_tensor * p = ggml_cont(ctx0, ggml_concat(ctx0, z, bd, 0));           // [2T, T, H]
    p = ggml_reshape_3d(ctx0, p, T, 2 * T, H);                                // [T, 2T, H]
    p = ggml_view_3d(ctx0, p, T, 2 * T - 1, H, p->nb[1], p->nb[2], p->nb[1]); // drop first row
    p = ggml_cont(ctx0, p);
    p = ggml_reshape_3d(ctx0, p, 2 * T - 1, T, H);                            // [2T-1, T, H]
    p = ggml_view_3d(ctx0, p, T, T, H, p->nb[1], p->nb[2], 0);                // first T columns
    return ggml_cont(ctx0, p);
}

// espnet rel-pos self attention block, pre-norm, x [512, T], pos [512, 2T-1]
ggml_tensor * clip_graph_chatterbox::enc_layer(const clip_chatterbox::enc_layer & l, ggml_tensor * x,
                                               ggml_tensor * pos, int T) {
    const int n_head = 8;
    const int d_head = 64;
    const float scale = 1.0f / sqrtf((float) d_head);

    ggml_tensor * res = x;
    ggml_tensor * cur = build_norm(x, l.norm_mha_w, l.norm_mha_b, NORM_TYPE_NORMAL, 1e-5f, -1);

    ggml_tensor * q = cbx_linear(l.attn_q_w, l.attn_q_b, cur);
    ggml_tensor * k = cbx_linear(l.attn_k_w, l.attn_k_b, cur);
    ggml_tensor * v = cbx_linear(l.attn_v_w, l.attn_v_b, cur);
    ggml_tensor * pe = cbx_linear(l.attn_pos_w, nullptr, pos); // [512, 2T-1]

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, T);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head, T);
    v = ggml_reshape_3d(ctx0, v, d_head, n_head, T);
    pe = ggml_reshape_3d(ctx0, pe, d_head, n_head, 2 * T - 1);

    ggml_tensor * u = l.attn_pos_bias_u; // [64, 8]
    ggml_tensor * w = l.attn_pos_bias_v;

    ggml_tensor * qu = ggml_add(ctx0, q, ggml_reshape_3d(ctx0, u, d_head, n_head, 1));
    ggml_tensor * qv = ggml_add(ctx0, q, ggml_reshape_3d(ctx0, w, d_head, n_head, 1));

    // per head: [64, T] tensors, scores [T(k), T(q)]
    qu = ggml_cont(ctx0, ggml_permute(ctx0, qu, 0, 2, 1, 3)); // [64, T, 8]
    qv = ggml_cont(ctx0, ggml_permute(ctx0, qv, 0, 2, 1, 3));
    k  = ggml_cont(ctx0, ggml_permute(ctx0, k,  0, 2, 1, 3));
    v  = ggml_cont(ctx0, ggml_permute(ctx0, v,  0, 2, 1, 3));
    pe = ggml_cont(ctx0, ggml_permute(ctx0, pe, 0, 2, 1, 3)); // [64, 2T-1, 8]

    ggml_tensor * ac = ggml_mul_mat(ctx0, k, qu);   // [T(k), T(q), 8]
    ggml_tensor * bd = ggml_mul_mat(ctx0, pe, qv);  // [2T-1, T(q), 8]
    bd = cbx_rel_shift(ctx0, bd, T);                // [T(k), T(q), 8]

    ggml_tensor * scores = ggml_scale(ctx0, ggml_add(ctx0, ac, bd), scale);
    ggml_tensor * probs = ggml_soft_max(ctx0, scores);

    ggml_tensor * o = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), probs); // [64, T, 8]
    o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));                                 // [64, 8, T]
    o = ggml_reshape_2d(ctx0, o, n_head * d_head, T);
    o = cbx_linear(l.attn_out_w, l.attn_out_b, o);
    x = ggml_add(ctx0, res, o);

    res = x;
    cur = build_norm(x, l.norm_ff_w, l.norm_ff_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    cur = cbx_linear(l.ffn_1_w, l.ffn_1_b, cur);
    cur = ggml_silu(ctx0, cur); // swish
    cur = cbx_linear(l.ffn_2_w, l.ffn_2_b, cur);
    x = ggml_add(ctx0, res, cur);
    return x;
}

// mish = x * tanh(softplus(x))
static ggml_tensor * cbx_mish(ggml_context * ctx0, ggml_tensor * x) {
    return ggml_mul(ctx0, x, ggml_tanh(ctx0, ggml_softplus(ctx0, x)));
}

// causal block: conv k3 left-padded, layer norm over channels, mish; x [C, T]
ggml_tensor * clip_graph_chatterbox::causal_block(const clip_chatterbox::causal_block & b, ggml_tensor * x) {
    x = cbx_conv1d(b.conv_w, b.conv_b, x, 1, (int) b.conv_w->ne[0] - 1, 0);
    x = build_norm(x, b.norm_w, b.norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    return cbx_mish(ctx0, x);
}

// resnet block with time conditioning; x [C, T], temb [1024]
ggml_tensor * clip_graph_chatterbox::resnet(const clip_chatterbox::resnet & r, ggml_tensor * x, ggml_tensor * temb) {
    ggml_tensor * h = causal_block(r.block1, x);
    ggml_tensor * tproj = cbx_linear(r.mlp_w, r.mlp_b, cbx_mish(ctx0, temb));
    h = ggml_add(ctx0, h, tproj); // broadcast [256, 1] over T
    h = causal_block(r.block2, h);
    ggml_tensor * res = cbx_conv1d(r.res_conv_w, r.res_conv_b, x, 1, 0, 0);
    return ggml_add(ctx0, h, res);
}

// diffusers-style transformer block, full attention; x [256, T]
ggml_tensor * clip_graph_chatterbox::tfm_block(const clip_chatterbox::tfm_block & b, ggml_tensor * x) {
    const int n_head = 8;
    const int d_head = 64;
    const int T = (int) x->ne[1];

    ggml_tensor * cur = build_norm(x, b.norm1_w, b.norm1_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, cur), d_head, n_head, T);
    ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, cur), d_head, n_head, T);
    ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, cur), d_head, n_head, T);
    x = ggml_add(ctx0, x, build_attn(b.attn_out_w, b.attn_out_b, q, k, v, nullptr, 1.0f / sqrtf((float) d_head), -1));

    cur = build_norm(x, b.norm3_w, b.norm3_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    cur = cbx_linear(b.ff_in_w, b.ff_in_b, cur);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = cbx_linear(b.ff_out_w, b.ff_out_b, cur);
    return ggml_add(ctx0, x, cur);
}

// one estimator evaluation; x_noise, mu, cond [80, T], spks [80], temb [1024]
ggml_tensor * clip_graph_chatterbox::estimator(ggml_tensor * x_noise, ggml_tensor * mu, ggml_tensor * spks,
                                               ggml_tensor * cond, ggml_tensor * temb, int T) {
    const auto & c = model.cbx;

    // channels live on ne0, time on ne1: pack along ne0
    ggml_tensor * x = ggml_concat(ctx0, x_noise, mu, 0);                   // [160, T]
    ggml_tensor * spks_b = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, spks, 80, 1), ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, T));
    x = ggml_concat(ctx0, x, spks_b, 0);                                   // [240, T]
    x = ggml_concat(ctx0, x, cond, 0);                                     // [320, T]
    x = ggml_cont(ctx0, x);

    // down
    ggml_tensor * skip;
    x = resnet(c.est_down.res, x, temb);
    for (const auto & b : c.est_down.tfm) {
        x = tfm_block(b, x);
    }
    skip = x;
    x = cbx_conv1d(c.est_down.conv_w, c.est_down.conv_b, x, 1, (int) c.est_down.conv_w->ne[0] - 1, 0);

    // mid
    for (const auto & m : c.est_mid) {
        x = resnet(m.res, x, temb);
        for (const auto & b : m.tfm) {
            x = tfm_block(b, x);
        }
    }

    // up with skip
    x = ggml_concat(ctx0, x, skip, 0);                                     // [512, T]
    x = resnet(c.est_up.res, x, temb);
    for (const auto & b : c.est_up.tfm) {
        x = tfm_block(b, x);
    }
    x = cbx_conv1d(c.est_up.conv_w, c.est_up.conv_b, x, 1, (int) c.est_up.conv_w->ne[0] - 1, 0);

    x = causal_block(c.est_final_block, x);
    x = cbx_conv1d(c.est_final_proj_w, c.est_final_proj_b, x, 1, 0, 0);    // [80, T]
    return x;
}

// s3 tokenizer encoder: log-mel [T, 128] -> fsq code [8, T / 4], rounded to base 3 tokens on the host
ggml_cgraph * clip_graph_chatterbox::build_s3tok(int T) {
    const auto & c = model.cbx;
    const int n_head = 20;
    const int d_head = 64;
    const int T1 = (T - 1) / 2 + 1;
    const int T2 = (T1 - 1) / 2 + 1;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, 128);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T2);
    ggml_set_name(pos, "inp_pos");
    ggml_set_input(pos);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [128, T]
    x = cbx_conv1d(c.s3tok_conv1_w, c.s3tok_conv1_b, x, 2, 1, 1);
    x = ggml_gelu_erf(ctx0, x);
    x = cbx_conv1d(c.s3tok_conv2_w, c.s3tok_conv2_b, x, 2, 1, 1);
    x = ggml_gelu_erf(ctx0, x); // [1280, T2]

    for (const auto & blk : c.s3tok_blocks) {
        ggml_tensor * res = x;
        ggml_tensor * cur = build_norm(x, blk.attn_ln_w, blk.attn_ln_b, NORM_TYPE_NORMAL, 1e-5f, -1);
        ggml_tensor * q = cbx_linear(blk.attn_q_w, blk.attn_q_b, cur);
        ggml_tensor * k = ggml_mul_mat(ctx0, blk.attn_k_w, cur);
        ggml_tensor * v = cbx_linear(blk.attn_v_w, blk.attn_v_b, cur);

        // fsmn memory: depthwise conv over time on v plus residual, added to the attention output
        ggml_tensor * fsm = ggml_cont(ctx0, ggml_transpose(ctx0, v)); // [T2, 1280]
        {
            ggml_tensor * w = blk.fsmn_w;
            ggml_tensor * m = ggml_conv_1d_dw(ctx0, w, fsm, 1, ((int) w->ne[0] - 1) / 2, 1);
            fsm = ggml_add(ctx0, ggml_reshape_2d(ctx0, m, fsm->ne[0], fsm->ne[1]), fsm);
        }
        fsm = ggml_cont(ctx0, ggml_transpose(ctx0, fsm)); // [1280, T2]

        q = ggml_reshape_3d(ctx0, q, d_head, n_head, T2);
        k = ggml_reshape_3d(ctx0, k, d_head, n_head, T2);
        v = ggml_reshape_3d(ctx0, v, d_head, n_head, T2);
        q = ggml_rope_ext(ctx0, q, pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * o = build_attn(blk.attn_out_w, blk.attn_out_b, q, k, v, nullptr, 1.0f / sqrtf((float) d_head), -1);
        x = ggml_add(ctx0, res, ggml_add(ctx0, o, fsm));

        res = x;
        cur = build_norm(x, blk.mlp_ln_w, blk.mlp_ln_b, NORM_TYPE_NORMAL, 1e-5f, -1);
        cur = cbx_linear(blk.mlp_in_w, blk.mlp_in_b, cur);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = cbx_linear(blk.mlp_out_w, blk.mlp_out_b, cur);
        x = ggml_add(ctx0, res, cur);
    }

    x = cbx_linear(c.s3tok_down_w, c.s3tok_down_b, x); // [8, T2]
    x = ggml_tanh(ctx0, x);

    ggml_set_name(x, "out_fsq");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

ggml_cgraph * clip_graph_chatterbox::build() {
    if (gen_process == CLIP_GEN_PROCESS_TTS_VOCODE) {
        return build_vocoder(vocode_n_mel, vocode_n_stft);
    }
    if (gen_process == CLIP_GEN_PROCESS_TOKENIZE) {
        return build_s3tok(img.nx());
    }
    if (gen_process != CLIP_GEN_PROCESS_TTS) {
        // load-time buffer sizing path
        ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_name(inp, "inp_stub");
        ggml_set_input(inp);
        ggml_tensor * cur = ggml_dup(ctx0, inp);
        ggml_set_name(cur, "out_stub");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    const auto & c = model.cbx;

    const int T1 = n_tokens;      // token-rate length (prompt + generated)
    const int T2 = 2 * n_tokens;  // mel-rate length after the x2 upsample

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T1);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * pos1 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T1 - 1);
    ggml_set_name(pos1, "inp_pos1");
    ggml_set_input(pos1);

    ggml_tensor * pos2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T2 - 1);
    ggml_set_name(pos2, "inp_pos2");
    ggml_set_input(pos2);

    // token embedding
    ggml_tensor * x = ggml_get_rows(ctx0, c.input_embedding_w, inp_tokens); // [512, T1]

    // embed: linear + layer norm, then the espnet xscale
    x = cbx_linear(c.embed_linear_w, c.embed_linear_b, x);
    x = build_norm(x, c.embed_norm_w, c.embed_norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    x = ggml_scale(ctx0, x, sqrtf(512.0f));
    cb(x, "fenc_embd", -1);

    // pre-lookahead: conv k=4 right-padded 3, leaky 0.01, conv k=3 left-padded 2, residual
    {
        ggml_tensor * res = x;
        ggml_tensor * cur = cbx_conv1d(c.pre_conv1_w, c.pre_conv1_b, x, 1, 0, 3);
        cur = ggml_leaky_relu(ctx0, cur, 0.01f, false);
        cur = cbx_conv1d(c.pre_conv2_w, c.pre_conv2_b, cur, 1, 2, 0);
        x = ggml_add(ctx0, res, cur);
        cb(x, "fenc_pre_lookahead", -1);
    }

    for (size_t i = 0; i < c.enc.size(); i++) {
        x = enc_layer(c.enc[i], x, pos1, T1);
        cb(x, "fenc_enc", (int) i);
    }

    // upsample x2: nearest repeat, left pad 4, conv k=5
    {
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));            // [T1, 512]
        xt = ggml_interpolate(ctx0, xt, 2 * T1, 512, 1, 1, GGML_SCALE_MODE_NEAREST);
        ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, 512);
        z = ggml_scale(ctx0, z, 0.0f);
        xt = ggml_concat(ctx0, z, xt, 0);
        ggml_tensor * y = ggml_conv_1d(ctx0, c.up_conv_w, xt, 1, 0, 1);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, y));                            // [512, T2]
        x = ggml_add(ctx0, x, c.up_conv_b);
        cb(x, "fenc_upsample", -1);
    }

    // up embed: linear + layer norm + xscale
    x = cbx_linear(c.up_embed_linear_w, c.up_embed_linear_b, x);
    x = build_norm(x, c.up_embed_norm_w, c.up_embed_norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    x = ggml_scale(ctx0, x, sqrtf(512.0f));

    for (size_t i = 0; i < c.up_enc.size(); i++) {
        x = enc_layer(c.up_enc[i], x, pos2, T2);
        cb(x, "fenc_up_enc", (int) i);
    }

    x = build_norm(x, c.after_norm_w, c.after_norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);

    // encoder projection to the mel channel count
    ggml_tensor * mu = cbx_linear(c.encoder_proj_w, c.encoder_proj_b, x); // [80, T2]
    cb(mu, "flow_mu", -1);

    // cfm solver unrolled in the graph: meanflow takes 2 euler steps without cfg, its time embeds mix t and r
    // the classic solver takes 10 euler steps on the cosine schedule with cfg 0.7
    const bool meanflow = c.time_embed_mixer_w != nullptr;
    const int n_steps = meanflow ? 2 : 10;

    // span points, same schedule as the host side sinusoid fill in clip.cpp
    float span[11];
    for (int i = 0; i <= n_steps; i++) {
        const float u = (float) i / n_steps;
        span[i] = meanflow ? u : 1.0f - cosf(u * (float) M_PI / 2.0f);
    }

    ggml_tensor * noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, T2);
    ggml_set_name(noise, "inp_noise");
    ggml_set_input(noise);
    // sinusoidal time embeddings, one row per span point
    ggml_tensor * temb_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 320, n_steps + 1);
    ggml_set_name(temb_sin, "inp_temb");
    ggml_set_input(temb_sin);

    auto time_mlp = [&](ggml_tensor * e) {
        e = cbx_linear(c.time_mlp_1_w, c.time_mlp_1_b, e);
        e = ggml_silu(ctx0, e);
        e = cbx_linear(c.time_mlp_2_w, c.time_mlp_2_b, e);
        return e;
    };
    auto span_emb = [&](int i) {
        return ggml_view_2d(ctx0, temb_sin, 320, 1, temb_sin->nb[1], (size_t) i * temb_sin->nb[1]);
    };
    auto step_temb = [&](int i) {
        if (!meanflow) {
            return time_mlp(span_emb(i));
        }
        ggml_tensor * e = ggml_concat(ctx0, time_mlp(span_emb(i)), time_mlp(span_emb(i + 1)), 0); // [2048, 1]
        return ggml_mul_mat(ctx0, c.time_embed_mixer_w, e);                                       // [1024, 1]
    };

    // mel-rate conditions of the voice: prompt features then zeros, and the speaker vector
    ggml_tensor * pf = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, n_prompt_mel);
    ggml_set_name(pf, "inp_prompt_feat");
    ggml_set_input(pf);
    ggml_tensor * zc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, T2 - n_prompt_mel);
    zc = ggml_scale(ctx0, zc, 0.0f);
    ggml_tensor * cond = ggml_concat(ctx0, pf, zc, 1); // [80, T2]
    // raw 192-dim campplus x-vector, normalized then through the flow speaker affine
    ggml_tensor * xvec = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 192);
    ggml_set_name(xvec, "inp_xvec");
    ggml_set_input(xvec);
    ggml_tensor * n2 = ggml_sqrt(ctx0, ggml_sum(ctx0, ggml_mul(ctx0, xvec, xvec)));
    ggml_tensor * unit = ggml_div(ctx0, xvec, n2);
    ggml_tensor * spks = cbx_linear(c.spk_affine_w, c.spk_affine_b,
                                    ggml_reshape_2d(ctx0, unit, 192, 1));
    spks = ggml_cont(ctx0, ggml_reshape_1d(ctx0, spks, 80));

    const float cfg = 0.7f;
    ggml_tensor * mu_zero   = meanflow ? nullptr : ggml_scale(ctx0, mu, 0.0f);
    ggml_tensor * cond_zero = meanflow ? nullptr : ggml_scale(ctx0, cond, 0.0f);
    ggml_tensor * spks_zero = meanflow ? nullptr : ggml_scale(ctx0, spks, 0.0f);

    ggml_tensor * mx = noise;
    for (int i = 0; i < n_steps; i++) {
        ggml_tensor * temb = step_temb(i);
        ggml_tensor * d = estimator(mx, mu, spks, cond, temb, T2);
        if (!meanflow) {
            ggml_tensor * du = estimator(mx, mu_zero, spks_zero, cond_zero, temb, T2);
            d = ggml_add(ctx0, ggml_scale(ctx0, d, 1.0f + cfg), ggml_scale(ctx0, du, -cfg));
        }
        mx = ggml_add(ctx0, mx, ggml_scale(ctx0, d, span[i + 1] - span[i]));
        cb(mx, "cfm_step", i);
    }
    ggml_tensor * mel = mx;

    // trim the prompt frames at mel rate
    mel = ggml_view_2d(ctx0, mel, 80, T2 - n_prompt_mel, mel->nb[1], (size_t) n_prompt_mel * mel->nb[1]);
    mel = ggml_cont(ctx0, mel);
    ggml_set_name(mel, "out_mel");
    ggml_set_output(mel);
    ggml_build_forward_expand(gf, mel);

    // f0 predictor on the trimmed mel: 5x (conv k3 same-pad + elu), abs(linear)
    {
        ggml_tensor * fx = mel;
        for (const auto & fc : c.f0_condnet) {
            fx = cbx_conv1d(fc.w, fc.b, fx, 1, 1, 1);
            fx = ggml_elu(ctx0, fx);
        }
        fx = cbx_linear(c.f0_classifier_w, c.f0_classifier_b, fx);
        fx = ggml_abs(ctx0, fx); // [1, T]
        ggml_set_name(fx, "out_f0");
        ggml_set_output(fx);
        ggml_build_forward_expand(gf, fx);
    }
    return gf;
}

// snake activation with per-channel alpha: x + sin^2(alpha x) / alpha
static ggml_tensor * cbx_snake(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * alpha) {
    ggml_tensor * sx = ggml_sin(ctx0, ggml_mul(ctx0, x, alpha));
    sx = ggml_mul(ctx0, sx, sx);
    sx = ggml_div(ctx0, sx, alpha);
    return ggml_add(ctx0, x, sx);
}

// hifigan-snake resblock; kernels with dilations 1/3/5 on convs1, 1 on convs2
ggml_tensor * clip_graph_chatterbox::hift_resblock(const clip_chatterbox::hift_res & r, ggml_tensor * x) {
    static const int dil[3] = {1, 3, 5};
    for (size_t j = 0; j < r.units.size(); j++) {
        const auto & u = r.units[j];
        const int d = dil[j % 3];
        const int p1 = (int) (u.conv1_w->ne[0] - 1) / 2 * d;
        const int p2 = (int) (u.conv2_w->ne[0] - 1) / 2;
        ggml_tensor * xt = cbx_snake(ctx0, x, u.act1_alpha);
        xt = cbx_conv1d_dil(u.conv1_w, u.conv1_b, xt, p1, d);
        xt = cbx_snake(ctx0, xt, u.act2_alpha);
        xt = cbx_conv1d_dil(u.conv2_w, u.conv2_b, xt, p2, 1);
        x = ggml_add(ctx0, x, xt);
    }
    return x;
}

// mel [80, T] + source stft [18, T_stft] -> conv_post output [18, T_stft2]
ggml_cgraph * clip_graph_chatterbox::build_vocoder(int n_mel, int n_stft) {
    const auto & c = model.cbx;

    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, n_mel);
    ggml_set_name(mel, "inp_mel");
    ggml_set_input(mel);
    ggml_tensor * sstft = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 18, n_stft);
    ggml_set_name(sstft, "inp_sstft");
    ggml_set_input(sstft);

    ggml_tensor * x = cbx_conv1d_dil(c.hift_pre_w, c.hift_pre_b, mel, 3, 1);

    for (size_t i = 0; i < c.hift_ups.size(); i++) {
        const auto & up = c.hift_ups[i];
        ggml_tensor * uk = up.up_w;
        const int K = (int) uk->ne[0];
        const int S = K / 2;
        const int P = (K - S) / 2;

        x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        // conv transpose then trim the torch padding P on both sides
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        xt = ggml_conv_transpose_1d(ctx0, uk, xt, S, 0, 1);
        xt = ggml_cont(ctx0, ggml_view_2d(ctx0, xt, xt->ne[0] - 2 * P, xt->ne[1], xt->nb[1], (size_t) P * ggml_element_size(xt)));
        x = ggml_cont(ctx0, ggml_transpose(ctx0, xt));
        x = ggml_add(ctx0, x, up.up_b);

        const bool is_last = i + 1 == c.hift_ups.size();
        if (is_last) {
            ggml_tensor * xr = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            xr = ggml_pad_reflect_1d(ctx0, xr, 1, 0);
            x = ggml_cont(ctx0, ggml_transpose(ctx0, xr));
        }

        // source injection: strided conv on the source stft, one resblock
        ggml_tensor * sk = up.source_down_w;
        const int SK = (int) sk->ne[0];
        const int SS = SK > 1 ? SK / 2 : 1;
        const int SP = SK > 1 ? SS / 2 : 0;
        ggml_tensor * si;
        {
            ggml_tensor * st = ggml_cont(ctx0, ggml_transpose(ctx0, sstft));
            st = ggml_conv_1d(ctx0, sk, st, SS, SP, 1);
            si = ggml_cont(ctx0, ggml_transpose(ctx0, st));
            si = ggml_add(ctx0, si, up.source_down_b);
        }
        si = hift_resblock(up.source_res, si);
        // align lengths: the reflection pad on the last stage adds one step
        if ((int) si->ne[1] != (int) x->ne[1]) {
            const int n = (int) std::min(si->ne[1], x->ne[1]);
            si = ggml_cont(ctx0, ggml_view_2d(ctx0, si, si->ne[0], n, si->nb[1], 0));
            x  = ggml_cont(ctx0, ggml_view_2d(ctx0, x,  x->ne[0],  n, x->nb[1],  0));
        }
        x = ggml_add(ctx0, x, si);

        ggml_tensor * acc = hift_resblock(up.res_0, x);
        acc = ggml_add(ctx0, acc, hift_resblock(up.res_1, x));
        acc = ggml_add(ctx0, acc, hift_resblock(up.res_2, x));
        x = ggml_scale(ctx0, acc, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx0, x, 0.01f, false);
    x = cbx_conv1d_dil(c.hift_post_w, c.hift_post_b, x, 3, 1);
    ggml_set_name(x, "out_spec");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}
