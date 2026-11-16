#include "models.h"

#include <cmath>

// latent frames of a window of n_frames, see MiniMaxMusic3ConditionEncoder of the reference
int64_t mtmd_graph_music3_gen::latent_frames(int64_t n_frames) {
    const double n = (double) n_frames * 44100.0 / 24000.0 * 960.0 / 512.0;
    return std::max<int64_t>(1, (int64_t) n);
}

// one frame of the depth decoder (RVQDepthDecoder): row 0 is the backbone hidden, row p > 0 the embedding of code p - 1
// each step reruns the 4 layers over the rows so far (at most 8, no KV cache), like the reference
ggml_cgraph * mtmd_graph_music3_gen::build_depth() {
    const int64_t n_cb  = model.gen_code_head_w->ne[2]; // residual codebooks
    const int64_t n_voc = model.gen_code_head_w->ne[1];
    const int64_t B     = n_rows; // cond, then uncond with guidance

    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, B);
    ggml_set_name(inp_h, "inp_feats");
    ggml_set_input(inp_h);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_cb);
    ggml_set_name(inp_rand, "inp_rand");
    ggml_set_input(inp_rand);

    mtmd_graph_qwen3tts_gen::code_gen cg(*this, top_k, 1.0f);

    // [n_embd, 1, B], the codes are the same in all rows
    auto project = [&](ggml_tensor * x) {
        ggml_tensor * y = build_mm(model.gen_code_proj_in_w, x);
        y = ggml_reshape_3d(ctx0, y, n_embd, 1, y->ne[1]);
        return y->ne[2] == B ? y : ggml_repeat_4d(ctx0, y, n_embd, 1, B, 1);
    };

    ggml_tensor * e0  = ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
    ggml_tensor * seq = ggml_concat(ctx0, project(inp_h), project(e0), 1);

    ggml_tensor * out_codes = code0;
    ggml_tensor * hiddens   = ggml_view_2d(ctx0, inp_h, n_embd, 1, inp_h->nb[1], 0); // cond row of each step
    ggml_tensor * embd_sum  = e0;

    for (int64_t g = 0; g < n_cb; g++) {
        const int64_t n_pos = g + 2;

        ggml_tensor * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos), 1.0f);
        build_vit_opts opts;
        opts.attn_mask = ggml_log(ctx0, ggml_tri(ctx0, ones, GGML_TRI_TYPE_LOWER_DIAG)); // 0 = keep, -inf = masked
        if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
            opts.attn_mask = ggml_cast(ctx0, opts.attn_mask, GGML_TYPE_F16);
        }
        ggml_tensor * pos = ggml_view_2d(ctx0, model.position_embeddings, n_embd, n_pos, model.position_embeddings->nb[1], 0);

        ggml_tensor * out  = build_vit(seq, n_pos, NORM_TYPE_RMS, FFN_SILU, pos, nullptr, opts); // [n_embd, n_pos, B]
        ggml_tensor * last = ggml_cont(ctx0, ggml_view_2d(ctx0, out, n_embd, B, out->nb[2], (n_pos - 1) * out->nb[1]));
        hiddens = ggml_concat(ctx0, hiddens, ggml_view_2d(ctx0, last, n_embd, 1, last->nb[1], 0), 1);

        ggml_tensor * head = ggml_view_2d(ctx0, model.gen_code_head_w, n_embd, n_voc, model.gen_code_head_w->nb[1], g * model.gen_code_head_w->nb[2]);
        ggml_tensor * logits = ggml_mul_mat(ctx0, head, last); // [n_voc, B]
        if (B > 1) {
            ggml_tensor * cond   = ggml_view_1d(ctx0, logits, n_voc, 0);
            ggml_tensor * uncond = ggml_view_1d(ctx0, logits, n_voc, logits->nb[1]);
            logits = ggml_add(ctx0, uncond, ggml_scale(ctx0, ggml_sub(ctx0, cond, uncond), cfg_scale));
        }

        ggml_tensor * code;
        if (temp > 0.0f) {
            ggml_tensor * rnd = ggml_view_1d(ctx0, inp_rand, 1, g * inp_rand->nb[0]);
            code = cg.do_sampling(ggml_scale(ctx0, logits, 1.0f / temp), rnd);
        } else {
            code = ggml_argmax(ctx0, logits);
        }
        code = ggml_reshape_1d(ctx0, code, 1);
        out_codes = ggml_concat(ctx0, out_codes, code, 0);

        ggml_tensor * table = ggml_view_2d(ctx0, model.gen_code_embd_w, n_embd, n_voc, model.gen_code_embd_w->nb[1], g * model.gen_code_embd_w->nb[2]);
        ggml_tensor * e = ggml_get_rows(ctx0, table, code);
        embd_sum = ggml_add(ctx0, embd_sum, e);
        if (g + 1 < n_cb) {
            seq = ggml_concat(ctx0, seq, project(e), 1);
        }
    }

    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    // condition row of the frame: the 8 hidden states mixed by the condition encoder weights
    ggml_tensor * cond_row = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, hiddens)), model.music3.cond_layer_w);
    ggml_set_name(cond_row, "out_feats");
    ggml_set_output(cond_row);
    ggml_build_forward_expand(gf, cond_row);

    // backbone input of the next frame: (E_lm[semantic] + sum of E_cb[g][code g]) / sqrt(n_cb + 1), must be the last node
    ggml_tensor * out_embd = ggml_scale(ctx0, embd_sum, 1.0f / sqrtf((float) (n_cb + 1)));
    cb(out_embd, "gen_audio_out", -1);
    ggml_build_forward_expand(gf, out_embd);
    return gf;
}

// frame rows [n_embd, n_rows] -> conv k = 3 -> nearest resampling to n_out latent frames
ggml_cgraph * mtmd_graph_music3_gen::build_cond() {
    const auto & m = model.music3;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_out);
    ggml_set_name(idx, "inp_idx");
    ggml_set_input(idx);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [T, n_embd]
    x = conv1d(x, m.cond_w, m.cond_b, 1);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [n_cond, T]
    x = ggml_get_rows(ctx0, x, idx);
    ggml_set_name(x, "out_feats");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// one velocity pass of the DiT over a window, the cond and uncond (zero condition) inputs as a batch of 2
// the input rows are [latent, zeros, condition], the time embedding is prepended as one extra token
ggml_cgraph * mtmd_graph_music3_gen::build_dit() {
    const auto & m = model.music3;
    const int64_t n_in    = m.pre_w->ne[0];
    const int64_t n_dit   = m.proj_in_w->ne[1];
    const int64_t n_head  = n_dit / dit_d_head;
    const int64_t n_pos   = n_rows + 1;
    const int64_t B       = 2;

    ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_in, n_rows, B);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * inp_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(inp_t, "inp_t");
    ggml_set_input(inp_t);

    ggml_tensor * x = ggml_add(ctx0, inp, build_mm(m.pre_w, inp));
    x = build_mm(m.proj_in_w, x); // [n_dit, n_rows, B]

    // time token: Fourier features -> linear -> silu -> linear
    ggml_tensor * ang  = ggml_mul(ctx0, m.time_freq, inp_t);
    ggml_tensor * temb = ggml_concat(ctx0, ggml_cos(ctx0, ang), ggml_sin(ctx0, ang), 0);
    temb = ggml_silu(ctx0, ggml_add(ctx0, build_mm(m.time_in_w, temb), m.time_in_b));
    temb = ggml_add(ctx0, build_mm(m.time_out_w, temb), m.time_out_b);
    temb = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, temb, n_dit, 1, 1), n_dit, 1, B, 1);
    x = ggml_concat(ctx0, temb, x, 1); // [n_dit, n_pos, B]
    x = ggml_reshape_2d(ctx0, x, n_dit, n_pos * B);

    ggml_tensor * pos = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) n_pos, 1.0f), GGML_TYPE_I32);
    const float kq = 1.0f / sqrtf((float) dit_d_head);

    for (size_t il = 0; il < m.dit.size(); il++) {
        const auto & layer = m.dit[il];

        ggml_tensor * h = build_norm(x, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, dit_eps, (int) il);
        ggml_tensor * q = ggml_reshape_4d(ctx0, build_mm(layer.q_w, h), dit_d_head, n_head, n_pos, B);
        ggml_tensor * k = ggml_reshape_4d(ctx0, build_mm(layer.k_w, h), dit_d_head, n_head, n_pos, B);
        ggml_tensor * v = ggml_reshape_4d(ctx0, build_mm(layer.v_w, h), dit_d_head, n_head, n_pos, B);

        // only the first dit_n_rot dims of each head rotate
        q = ggml_rope_ext(ctx0, q, pos, nullptr, dit_n_rot, GGML_ROPE_TYPE_NEOX, 0, dit_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, dit_n_rot, GGML_ROPE_TYPE_NEOX, 0, dit_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        x = ggml_add(ctx0, x, build_attn(layer.o_w, nullptr, q, k, v, nullptr, kq, (int) il));

        h = build_norm(x, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, dit_eps, (int) il);
        h = build_ffn(h, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, FFN_SILU, (int) il);
        x = ggml_add(ctx0, x, h);
    }

    // drop the time token
    x = ggml_reshape_3d(ctx0, x, n_dit, n_pos, B);
    x = ggml_view_3d(ctx0, x, n_dit, n_rows, B, x->nb[1], x->nb[2], x->nb[1]);
    x = build_mm(m.proj_out_w, ggml_cont(ctx0, x));
    x = ggml_add(ctx0, x, build_mm(m.post_w, x)); // [n_latent, n_rows, B]

    // classifier-free guidance
    ggml_tensor * cond   = ggml_view_2d(ctx0, x, n_latent, n_rows, x->nb[1], 0);
    ggml_tensor * uncond = ggml_view_2d(ctx0, x, n_latent, n_rows, x->nb[1], x->nb[2]);
    x = ggml_add(ctx0, uncond, ggml_scale(ctx0, ggml_sub(ctx0, cond, uncond), cfg_scale));
    ggml_set_name(x, "out_feats");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// the two channels are decoded as two mono streams of n_latent / 2 channels, then tanh
ggml_cgraph * mtmd_graph_music3_gen::build_vocoder() {
    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [T, n_latent]
    const int64_t n_half = n_latent / n_channels;
    ggml_tensor * out = nullptr;
    for (int c = 0; c < n_channels; c++) {
        ggml_tensor * xc = ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_rows, n_half, x->nb[1], c * n_half * x->nb[1]));
        ggml_tensor * y  = ggml_tanh(ctx0, build_snake_vae(xc, model.snake_vae)); // [T * ratio, 1]
        out = out ? ggml_concat(ctx0, out, y, 1) : y;
    }

    out = ggml_cont(ctx0, ggml_transpose(ctx0, out)); // interleaved
    ggml_set_name(out, "out_audio");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    return gf;
}

ggml_cgraph * mtmd_graph_music3_gen::build() {
    switch (gen_process) {
        case MTMD_SUBCTX_GEN_PROCESS_GEN_CODE:  return build_depth();
        case MTMD_SUBCTX_GEN_PROCESS_FLOW_COND: return build_cond();
        case MTMD_SUBCTX_GEN_PROCESS_FLOW_STEP: return build_dit();
        case MTMD_SUBCTX_GEN_PROCESS_GEN_WAV:   return build_vocoder();
        default: GGML_ABORT("minimax_music3: unsupported gen process");
    }
}
