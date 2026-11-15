#include "models.h"

// GEN_CODE: hidden state -> logits of the 10 codebooks (out_feats), or codes -> backbone row (graph output)
ggml_cgraph * mtmd_graph_zonos3_gen::build_gen_code() {
    ggml_tensor * embd_w = model.gen_code_out_embd_w; // [n_embd, n_vocab * n_codebooks]
    const int64_t n_vocab = embd_w->ne[1] / n_codebooks;

    if (!all_known) {
        ggml_tensor * h = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_mmproj_embd);
        ggml_set_name(h, "inp_raw");
        ggml_set_input(h);

        ggml_tensor * logits = ggml_mul_mat(ctx0, model.gen_code_head_w, h);
        logits = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, logits, 1.0f / softcap)), softcap);
        logits = ggml_reshape_2d(ctx0, logits, n_vocab, n_codebooks);
        ggml_set_name(logits, "out_feats");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    ggml_tensor * inp_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_codebooks);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);

    ggml_tensor * embd = ggml_get_rows(ctx0, embd_w, inp_codes); // [n_embd, n_codebooks]
    embd = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, embd)));
    embd = ggml_reshape_2d(ctx0, embd, n_mmproj_embd, 1);
    cb(embd, "gen_embd", -1);
    ggml_build_forward_expand(gf, embd);

    return gf;
}

std::vector<c2w_state_slot> list_zonos3_gen_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    GGML_UNUSED(hparams);
    const auto & z = model.zonos3;
    std::vector<c2w_state_slot> slots;
    slots.push_back({"pos", 1, 1});
    for (size_t i = 0; i < z.sem.size(); i++) {
        slots.push_back({"sem_conv_" + std::to_string(i), z.sem[i].dw_w->ne[0] - 1, z.sem[i].dw_w->ne[1]});
    }
    for (size_t s = 0; s < z.stages.size(); s++) {
        const int64_t d = z.stages[s].layers.empty() ? 0 : z.stages[s].layers[0].wo_w->ne[1];
        for (size_t il = 0; il < z.stages[s].layers.size(); il++) {
            const std::string sfx = std::to_string(s) + "_" + std::to_string(il);
            slots.push_back({"tfm_k_" + sfx, d, mtmd_graph_zonos3_gen::stage_ctx[s] - 1});
            slots.push_back({"tfm_v_" + sfx, d, mtmd_graph_zonos3_gen::stage_ctx[s] - 1});
        }
    }
    for (size_t i = 0; i < z.head.size(); i++) {
        slots.push_back({"head_conv_" + std::to_string(i), z.head[i].dw_w->ne[0] - 1, z.head[i].dw_w->ne[1]});
    }
    return slots;
}

static ggml_tensor * zonos3_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
}

// causal ConvNeXt block, the conv history is kept in the state
ggml_tensor * mtmd_graph_zonos3_gen::convnext(ggml_tensor * x, const mtmd_zonos3_wav::convnext & b, const std::string & state_name) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t K = b.dw_w->ne[0];

    ggml_tensor * hist = state_in.at(state_name); // [K - 1, C]
    ggml_tensor * xt   = ggml_concat(ctx0, hist, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 0); // [K - 1 + T, C]
    state_out.push_back({state_name, ggml_cont(ctx0, ggml_view_2d(ctx0, xt, K - 1, C, xt->nb[1], (size_t) T * xt->nb[0]))});

    ggml_tensor * h = ggml_ssm_conv(ctx0, ggml_reshape_3d(ctx0, xt, K - 1 + T, C, 1), b.dw_w); // [C, T, 1]
    h = ggml_add(ctx0, ggml_reshape_2d(ctx0, h, C, T), b.dw_b);
    h = zonos3_layer_norm(ctx0, h, b.norm_w, b.norm_b, convnext_eps);
    h = ggml_gelu_erf(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, b.pw1_w, h), b.pw1_b));
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.pw2_w, h), b.pw2_b);
    return ggml_add(ctx0, x, h);
}

// windowed causal MOSS layer, adapted from qwen3tts code2wav tfm_layer_forward()
ggml_tensor * mtmd_graph_zonos3_gen::moss_layer(ggml_tensor * x, const mtmd_zonos3_wav::layer & l, int s, int il, ggml_tensor * base) {
    const int64_t d      = x->ne[0];
    const int64_t N      = x->ne[1];
    const int64_t n_head = d / head_dim;
    const int64_t W      = stage_ctx[s];
    const int64_t prefix = W - 1;
    const int64_t n_kv   = prefix + N;
    const std::string sfx = std::to_string(s) + "_" + std::to_string(il);

    ggml_tensor * h   = zonos3_layer_norm(ctx0, x, l.norm1_w, l.norm1_b, ln_eps);
    ggml_tensor * qkv = ggml_mul_mat(ctx0, l.qkv_w, h); // [3d, N]
    auto section = [&](int i) {
        return ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, head_dim, n_head, N, ggml_row_size(qkv->type, head_dim), qkv->nb[1],
                                            ggml_row_size(qkv->type, i * d)));
    };

    ggml_tensor * pos = ggml_cast(ctx0, ggml_add(ctx0, ggml_arange(ctx0, 0.0f, (float) N, 1.0f), base), GGML_TYPE_I32);
    ggml_tensor * q = ggml_rope_ext(ctx0, section(0), pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k = ggml_rope_ext(ctx0, section(1), pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * v = section(2);

    ggml_tensor * k_full = ggml_concat(ctx0, state_in.at("tfm_k_" + sfx), ggml_reshape_2d(ctx0, k, d, N), 1); // [d, n_kv]
    ggml_tensor * v_full = ggml_concat(ctx0, state_in.at("tfm_v_" + sfx), ggml_reshape_2d(ctx0, v, d, N), 1);
    state_out.push_back({"tfm_k_" + sfx, ggml_cont(ctx0, ggml_view_2d(ctx0, k_full, d, prefix, k_full->nb[1], (size_t) N * k_full->nb[1]))});
    state_out.push_back({"tfm_v_" + sfx, ggml_cont(ctx0, ggml_view_2d(ctx0, v_full, d, prefix, v_full->nb[1], (size_t) N * v_full->nb[1]))});

    // key j is visible to query i iff 0 <= (prefix + i) - j < W and j >= prefix - base (cold start)
    ggml_tensor * pos_k = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_kv, 1.0f), n_kv, 1);
    ggml_tensor * pos_q = ggml_reshape_2d(ctx0, ggml_arange(ctx0, (float) prefix, (float) (prefix + N), 1.0f), 1, N);
    ggml_tensor * diff  = ggml_sub(ctx0, ggml_repeat_4d(ctx0, pos_q, n_kv, N, 1, 1), pos_k); // [n_kv, N]
    ggml_tensor * keep  = ggml_mul(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, diff, 1.0f, 0.5f)),
                                         ggml_step(ctx0, ggml_scale_bias(ctx0, diff, -1.0f, (float) W - 0.5f)));
    keep = ggml_mul(ctx0, keep, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, pos_k, base), 1.0f, 0.5f - (float) prefix)));
    ggml_tensor * mask = ggml_reshape_4d(ctx0, ggml_log(ctx0, keep), n_kv, N, 1, 1); // 0 = keep, -inf = masked
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
    }

    ggml_tensor * attn = build_attn(l.wo_w, nullptr,
                                    ggml_reshape_4d(ctx0, q, head_dim, n_head, N, 1),
                                    ggml_reshape_4d(ctx0, k_full, head_dim, n_head, n_kv, 1),
                                    ggml_reshape_4d(ctx0, v_full, head_dim, n_head, n_kv, 1),
                                    mask, 1.0f / sqrtf((float) head_dim), il);
    x = ggml_add(ctx0, x, ggml_mul(ctx0, attn, l.attn_scale));

    h = zonos3_layer_norm(ctx0, x, l.norm2_w, l.norm2_b, ln_eps);
    h = ggml_mul_mat(ctx0, l.ffn2_w, ggml_gelu_erf(ctx0, ggml_mul_mat(ctx0, l.ffn1_w, h)));
    return ggml_add(ctx0, x, ggml_mul(ctx0, h, l.ffn_scale));
}

ggml_cgraph * mtmd_graph_zonos3_gen::build_gen_wav() {
    const auto & z = model.zonos3;
    const int64_t T = n_frames;

    for (const auto & slot : list_zonos3_gen_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        state_in[slot.name] = t;
    }

    // codebook-major, residual codes are rows of the merged table
    ggml_tensor * inp_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T * n_codebooks);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);
    auto codes_of = [&](int c) {
        return ggml_view_1d(ctx0, inp_codes, T, (size_t) c * T * ggml_element_size(inp_codes));
    };

    // 1 for a known residual code, 0 for a missing one (early decode of a frame), codebook-major
    ggml_tensor * inp_keep = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, T * (n_codebooks - 1));
    ggml_set_name(inp_keep, "inp_keep");
    ggml_set_input(inp_keep);

    ggml_tensor * x = ggml_add(ctx0, ggml_get_rows(ctx0, z.sem_table, codes_of(0)), z.sem_bias); // [semantic_dim, T]
    for (size_t i = 0; i < z.sem.size(); i++) {
        x = convnext(x, z.sem[i], "sem_conv_" + std::to_string(i));
    }
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, z.sem_out_w, x), z.sem_out_b); // [latent, T]
    for (int c = 1; c < n_codebooks; c++) {
        ggml_tensor * bias = ggml_view_1d(ctx0, z.res_bias, z.res_bias->ne[0], (size_t) (c - 1) * z.res_bias->nb[1]);
        ggml_tensor * keep = ggml_view_2d(ctx0, inp_keep, 1, T, ggml_element_size(inp_keep), (size_t) (c - 1) * T * ggml_element_size(inp_keep));
        x = ggml_add(ctx0, x, ggml_mul(ctx0, ggml_add(ctx0, ggml_get_rows(ctx0, z.res_table, codes_of(c)), bias), keep));
    }
    cb(x, "zac_latent", -1);

    ggml_tensor * frames = ggml_reshape_1d(ctx0, state_in.at("pos"), 1);
    int cpf = 1; // positions per frame
    for (int s = 0; s < n_stages; s++) {
        const auto & st = z.stages[s];
        x = ggml_mul_mat(ctx0, st.in_w, x);
        ggml_tensor * base = ggml_scale(ctx0, frames, (float) cpf);
        for (size_t il = 0; il < st.layers.size(); il++) {
            x = moss_layer(x, st.layers[il], s, (int) il, base);
        }
        if (st.out_w) {
            x = ggml_mul_mat(ctx0, st.out_w, x);
        }
        if (s + 1 < n_stages) {
            // patch up x2: out[ch, 2t + j] = in[2ch + j, t]
            const int64_t C = x->ne[0], N = x->ne[1];
            x = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, x, 2, C / 2, N), 1, 0, 2, 3));
            x = ggml_reshape_2d(ctx0, x, C / 2, 2 * N);
            cpf *= 2;
        }
        cb(x, "zac_stage_out", s);
    }
    state_out.push_back({"pos", ggml_scale_bias(ctx0, state_in.at("pos"), 1.0f, (float) T)});

    for (size_t i = 0; i < z.head.size(); i++) {
        x = convnext(x, z.head[i], "head_conv_" + std::to_string(i));
    }
    x = zonos3_layer_norm(ctx0, x, z.head_norm_w, z.head_norm_b, convnext_eps);
    ggml_tensor * spec = ggml_add(ctx0, ggml_mul_mat(ctx0, z.head_out_w, x), z.head_out_b); // [n_fft0 + 2 + n_fft1 + 2, 16T]
    ggml_set_name(spec, "out_feats");
    ggml_set_output(spec);
    ggml_build_forward_expand(gf, spec);

    for (auto & slot : state_out) {
        ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
        ggml_set_output(slot.second);
        ggml_build_forward_expand(gf, slot.second);
    }
    return gf;
}

ggml_cgraph * mtmd_graph_zonos3_gen::build() {
    return gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV ? build_gen_wav() : build_gen_code();
}
