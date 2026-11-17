#include "models.h"

#include <cmath>

// linspace(1, 0) with sway sampling (coef 1), see UnifiedCFM.forward()
std::vector<float> mtmd_graph_voxcpm2_gen::t_span(int n_steps) {
    std::vector<float> t(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        const double x = 1.0 - (double) i / n_steps;
        t[i] = (float) (x + std::cos(M_PI / 2 * x) - 1.0 + x);
    }
    return t;
}

// SinusoidalPosEmb with scale 1000: [sin, cos], the frequency step divides by half - 1
std::vector<float> mtmd_graph_voxcpm2_gen::time_embd(int n_steps, int n_embd) {
    const std::vector<float> t = t_span(n_steps);
    const int half = n_embd / 2;
    std::vector<float> out((size_t) n_embd * n_steps);
    for (int s = 0; s < n_steps; s++) {
        for (int j = 0; j < half; j++) {
            const double a = 1000.0 * t[s] * std::exp(-std::log(10000.0) * j / (half - 1));
            out[(size_t) s * n_embd + j]        = (float) std::sin(a);
            out[(size_t) s * n_embd + half + j] = (float) std::cos(a);
        }
    }
    return out;
}

ggml_tensor * mtmd_graph_voxcpm2_gen::inp_pos(const char * name, int n) const {
    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(pos, name);
    ggml_set_input(pos);
    return pos;
}

ggml_tensor * mtmd_graph_voxcpm2_gen::tfm(ggml_tensor * x, const std::vector<mtmd_layer> & layers, ggml_tensor * norm_w, ggml_tensor * pos) const {
    const int64_t n_tok     = x->ne[1];
    const int64_t n_seq     = x->ne[2];
    const int     n_head    = hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;
    const float   kq_scale  = 1.0f / sqrtf((float) head_dim);
    ggml_tensor * factors   = model.voxcpm2.rope_factors;

    for (size_t il = 0; il < layers.size(); il++) {
        const auto & layer = layers[il];
        ggml_tensor * cur = build_norm(x, layer.ln_1_w, nullptr, NORM_TYPE_RMS, hparams.eps, (int) il);

        ggml_tensor * q = ggml_reshape_4d(ctx0, build_mm(layer.q_w, cur), head_dim, n_head,    n_tok, n_seq);
        ggml_tensor * k = ggml_reshape_4d(ctx0, build_mm(layer.k_w, cur), head_dim, n_head_kv, n_tok, n_seq);
        ggml_tensor * v = ggml_reshape_4d(ctx0, build_mm(layer.v_w, cur), head_dim, n_head_kv, n_tok, n_seq);
        q = ggml_rope_ext(ctx0, q, pos, factors, head_dim, GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, factors, head_dim, GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        cur = build_attn(layer.o_w, nullptr, q, k, v, nullptr, kq_scale, (int) il);
        x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, cur, cur->ne[0], n_tok, n_seq));

        cur = build_norm(x, layer.ln_2_w, nullptr, NORM_TYPE_RMS, hparams.eps, (int) il);
        cur = build_ffn(cur, layer.ff_up_w, nullptr, layer.ff_gate_w, nullptr, layer.ff_down_w, nullptr, FFN_SILU, (int) il);
        x = ggml_add(ctx0, x, cur);
    }
    return build_norm(x, norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);
}

// each patch gets the special token in front, its output is the summary of the patch
ggml_tensor * mtmd_graph_voxcpm2_gen::loc_enc(ggml_tensor * latents, int n) const {
    const auto &  m      = model.voxcpm2;
    const int64_t n_embd = hparams.n_embd;

    ggml_tensor * x = ggml_add(ctx0, build_mm(m.enc_in_w, latents), m.enc_in_b);
    x = ggml_reshape_3d(ctx0, x, n_embd, n_patch, n);
    ggml_tensor * special = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, m.enc_special, n_embd, 1, 1),
                                        ggml_view_3d(ctx0, x, n_embd, 1, n, x->nb[1], x->nb[2], 0));
    x = ggml_concat(ctx0, special, x, 1); // [n_embd, 1 + n_patch, n]
    x = tfm(x, m.enc_layers, m.enc_norm_w, inp_pos("inp_pos_enc", 1 + n_patch));
    x = ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_embd, n, x->nb[2], 0));
    cb(x, "loc_enc_out", -1);
    return ggml_add(ctx0, build_mm(m.enc_to_lm_w, x), m.enc_to_lm_b);
}

void mtmd_graph_voxcpm2_gen::build_res_inp() {
    const auto &  m      = model.voxcpm2;
    const int64_t n_text = m.fusion_b->ne[0];

    ggml_tensor * h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_text, n_rows);
    ggml_set_name(h, "inp_raw");
    ggml_set_input(h);
    ggml_tensor * x = nullptr;
    if (audio_row) {
        x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_text, n_rows);
        ggml_set_name(x, "inp_feats");
        ggml_set_input(x);
        // FSQ: tanh, then round to the grid of 1/9
        ggml_tensor * q = ggml_tanh(ctx0, ggml_add(ctx0, build_mm(m.fsq_in_w, h), m.fsq_in_b));
        q = ggml_scale(ctx0, ggml_round(ctx0, ggml_scale(ctx0, q, fsq_scale)), 1.0f / fsq_scale);
        h = ggml_add(ctx0, build_mm(m.fsq_out_w, q), m.fsq_out_b);
    } else {
        x = ggml_scale(ctx0, h, 0.0f); // a text row has no input row
    }

    // logit(stop) - logit(continue), only the row of the next frame needs it
    if (n_rows == 1) {
        ggml_tensor * s = ggml_silu(ctx0, ggml_add(ctx0, build_mm(m.stop_proj_w, h), m.stop_proj_b));
        s = build_mm(m.stop_head_w, s);
        s = ggml_sub(ctx0, ggml_view_1d(ctx0, s, 1, ggml_element_size(s)), ggml_view_1d(ctx0, s, 1, 0));
        ggml_set_name(s, "out_eos_score");
        ggml_set_output(s);
        ggml_build_forward_expand(gf, s);
    }

    // residual LM input rows, then the (FSQ'd) hidden states
    ggml_tensor * r = ggml_add(ctx0, build_mm(m.fusion_w, ggml_concat(ctx0, h, x, 0)), m.fusion_b);
    ggml_tensor * out = ggml_concat(ctx0, r, h, 1);
    ggml_set_name(out, "out_feats");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
}

// Euler solver from noise (t = 1) to the patch (t = 0), see UnifiedCFM.solve_euler()
void mtmd_graph_voxcpm2_gen::build_flow() {
    const auto &  m      = model.voxcpm2;
    const int64_t n_text = m.fusion_b->ne[0];
    const int64_t n_embd = hparams.n_embd;

    // FSQ'd hidden state, then the residual LM output
    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_text, 2);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);
    ggml_tensor * cond = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_patch);
    ggml_set_name(cond, "inp_cond");
    ggml_set_input(cond);
    ggml_tensor * noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_patch);
    ggml_set_name(noise, "inp_noise");
    ggml_set_input(noise);
    ggml_tensor * temb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_steps);
    ggml_set_name(temb, "inp_temb");
    ggml_set_input(temb);
    ggml_tensor * pos = inp_pos("inp_pos_dit", n_dit_tok);

    ggml_tensor * mu_lm  = ggml_add(ctx0, build_mm(m.lm_to_dit_w,  ggml_view_1d(ctx0, inp, n_text, 0)),          m.lm_to_dit_b);
    ggml_tensor * mu_res = ggml_add(ctx0, build_mm(m.res_to_dit_w, ggml_view_1d(ctx0, inp, n_text, inp->nb[1])), m.res_to_dit_b);
    ggml_tensor * mu   = ggml_concat(ctx0, ggml_reshape_2d(ctx0, mu_lm, n_embd, 1), ggml_reshape_2d(ctx0, mu_res, n_embd, 1), 1);
    ggml_tensor * mu_0 = ggml_scale(ctx0, mu, 0.0f); // the unconditional branch has mu = 0

    temb = ggml_silu(ctx0, ggml_add(ctx0, build_mm(m.time0_w, temb), m.time0_b));
    temb = ggml_add(ctx0, build_mm(m.time1_w, temb), m.time1_b);
    ggml_tensor * cond_tok = ggml_add(ctx0, build_mm(m.dit_cond_w, cond), m.dit_cond_b);

    const std::vector<float> t = t_span(n_steps);
    // CFG-Zero*: the first steps keep x as is
    const int n_zero = std::max(1, (int) ((n_steps + 1) * 0.04));

    ggml_tensor * x = noise;
    for (int s = n_zero; s < n_steps; s++) {
        ggml_tensor * t_tok = ggml_view_2d(ctx0, temb, n_embd, 1, temb->nb[1], (size_t) s * temb->nb[1]);
        ggml_tensor * x_tok = ggml_add(ctx0, build_mm(m.dit_in_w, x), m.dit_in_b);
        ggml_tensor * tail  = ggml_concat(ctx0, ggml_concat(ctx0, t_tok, cond_tok, 1), x_tok, 1);
        ggml_tensor * h = ggml_concat(ctx0,
            ggml_reshape_3d(ctx0, ggml_concat(ctx0, mu,   tail, 1), n_embd, n_dit_tok, 1),
            ggml_reshape_3d(ctx0, ggml_concat(ctx0, mu_0, tail, 1), n_embd, n_dit_tok, 1), 2);
        h = tfm(h, model.layers, model.post_ln_w, pos);
        h = ggml_cont(ctx0, ggml_view_3d(ctx0, h, n_embd, n_patch, 2, h->nb[1], h->nb[2], (size_t) (n_dit_tok - n_patch) * h->nb[1]));
        h = ggml_add(ctx0, build_mm(m.dit_out_w, h), m.dit_out_b); // [n_latent, n_patch, 2]

        // guided velocity, the unconditional one is scaled by its projection on the conditional one
        ggml_tensor * v_c = ggml_view_2d(ctx0, h, n_latent, n_patch, h->nb[1], 0);
        ggml_tensor * v_u = ggml_view_2d(ctx0, h, n_latent, n_patch, h->nb[1], h->nb[2]);
        ggml_tensor * st  = ggml_div(ctx0, ggml_sum(ctx0, ggml_mul(ctx0, v_c, v_u)),
                                           ggml_scale_bias(ctx0, ggml_sum(ctx0, ggml_sqr(ctx0, v_u)), 1.0f, 1e-8f));
        ggml_tensor * v_s = ggml_mul(ctx0, v_u, st);
        ggml_tensor * v   = ggml_add(ctx0, v_s, ggml_scale(ctx0, ggml_sub(ctx0, v_c, v_s), cfg_scale));
        x = ggml_add(ctx0, x, ggml_scale(ctx0, v, t[s + 1] - t[s]));
    }
    GGML_ASSERT(x != noise && "voxcpm2: not enough flow steps");
    ggml_set_name(x, "out_feats");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // next backbone input row, the last node
    ggml_build_forward_expand(gf, loc_enc(x, 1));
}

void mtmd_graph_voxcpm2_gen::build_latent_in() {
    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_patch * n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);
    ggml_tensor * out = loc_enc(inp, n_rows);
    ggml_set_name(out, "out_feats");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
}

std::vector<c2w_state_slot> list_voxcpm2_state_slots(const mtmd_hparams & /*hparams*/, const mtmd_model & model) {
    const auto & m = model.voxcpm2;
    std::vector<c2w_state_slot> slots;
    slots.push_back({"in_dw", m.dec_in_dw_w->ne[0] - 1, m.dec_in_dw_w->ne[2]});
    for (size_t ib = 0; ib < m.dec.size(); ib++) {
        const auto & blk = m.dec[ib];
        const std::string name = "up" + std::to_string(ib);
        slots.push_back({name, blk.conv_w->ne[0] / 2, blk.conv_w->ne[1]});
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & ru = blk.res[ir];
            slots.push_back({name + "_res" + std::to_string(ir),
                             (ru.conv1_w->ne[0] - 1) * mtmd_graph_voxcpm2_gen::res_dilations[ir], ru.conv1_w->ne[2]});
        }
    }
    slots.push_back({"out", m.dec_out_w->ne[0] - 1, m.dec_out_w->ne[1]});
    return slots;
}

// causal AudioVAE decoder, the conv states carry the left context between calls
void mtmd_graph_voxcpm2_gen::build_dec() {
    const auto & m = model.voxcpm2;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_patch * n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    mtmd_graph_qwen3tts_gen::code2wav dec(*this);
    for (const auto & slot : list_voxcpm2_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        dec.state_in[slot.name] = t;
    }

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [T, n_latent]
    x = dec.causal_conv1d_dw(x, m.dec_in_dw_w, m.dec_in_dw_b, "in_dw");
    x = dec.causal_conv1d(x, m.dec_in_w, m.dec_in_b, 1, "");
    for (size_t ib = 0; ib < m.dec.size(); ib++) {
        const auto & blk  = m.dec[ib];
        const std::string name = "up" + std::to_string(ib);
        const int64_t n_ch = m.dec_sr_scale[ib]->ne[0];

        // sample-rate conditioning, the row of the output rate is picked at conversion
        x = ggml_add(ctx0, ggml_mul(ctx0, x, ggml_reshape_2d(ctx0, m.dec_sr_scale[ib], 1, n_ch)),
                           ggml_reshape_2d(ctx0, m.dec_sr_bias[ib], 1, n_ch));
        x = dec.snake(x, blk.snake_alpha, blk.snake_beta);
        x = dec.causal_conv_transpose1d(x, blk.conv_w, blk.conv_b, (int) (blk.conv_w->ne[0] / 2), name);
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & ru = blk.res[ir];
            ggml_tensor * h = dec.snake(x, ru.act1_alpha, ru.act1_beta);
            h = dec.causal_conv1d_dw(h, ru.conv1_w, ru.conv1_b, name + "_res" + std::to_string(ir), res_dilations[ir]);
            h = dec.snake(h, ru.act2_alpha, ru.act2_beta);
            h = dec.causal_conv1d(h, ru.conv2_w, ru.conv2_b, 1, "");
            x = ggml_add(ctx0, x, h);
        }
        cb(x, "vae_block", (int) ib);
    }
    x = dec.snake(x, m.dec_post_alpha, m.dec_post_beta);
    x = dec.causal_conv1d(x, m.dec_out_w, m.dec_out_b, 1, "out");
    x = ggml_tanh(ctx0, ggml_reshape_1d(ctx0, x, x->ne[0]));
    ggml_set_name(x, "out_audio");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    for (auto & slot : dec.state_out) {
        ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
        ggml_set_output(slot.second);
        ggml_build_forward_expand(gf, slot.second);
    }
}

ggml_cgraph * mtmd_graph_voxcpm2_gen::build() {
    switch (gen_process) {
        case MTMD_SUBCTX_GEN_PROCESS_RES_INP:   build_res_inp();   break;
        case MTMD_SUBCTX_GEN_PROCESS_GEN_CODE:  build_flow();      break;
        case MTMD_SUBCTX_GEN_PROCESS_LATENT_IN: build_latent_in(); break;
        case MTMD_SUBCTX_GEN_PROCESS_GEN_WAV:   build_dec();       break;
        default: GGML_ABORT("voxcpm2: unknown gen_process");
    }
    return gf;
}
