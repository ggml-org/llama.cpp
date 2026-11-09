#include "models.h"

#include <cmath>
#include <string>

// tower RoPE: interleaved, first 64 dims of each head
static constexpr int   DRAMA3_N_ROT      = 64;
static constexpr float DRAMA3_ROPE_THETA = 1e7f;

std::vector<c2w_state_slot> list_drama3_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    std::vector<c2w_state_slot> slots;
    slots.push_back({"tower_pos", 1, 1, true});
    for (size_t il = 0; il < model.layers.size(); il++) {
        const int64_t n_kv = model.layers[il].k_w->ne[1];
        slots.push_back({"tower_k_" + std::to_string(il), n_kv, mtmd_graph_drama3_gen::n_pos_max, true});
        slots.push_back({"tower_v_" + std::to_string(il), n_kv, mtmd_graph_drama3_gen::n_pos_max, true});
    }
    for (auto & slot : list_c2w_state_slots(hparams, model)) {
        slots.push_back(slot);
    }
    return slots;
}

// h_out = out_norm(Block(eh_proj([enorm(mask * E[code] + extra); hnorm(h_in)])))
ggml_tensor * mtmd_graph_drama3_gen::tower_layer(ggml_tensor * x, int il, ggml_tensor * code, ggml_tensor * mask, ggml_tensor * extra,
                                                 ggml_tensor * inp_pos, ggml_tensor * k_slot, ggml_tensor * v_slot, int pos) {
    const auto & layer = model.layers[il];
    const auto & tl    = model.drama3_layers[il];
    const int     n_head    = hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;
    const int64_t d_head    = layer.q_w->ne[1] / n_head;
    const int64_t n_kv      = d_head * n_head_kv;
    const float   eps       = hparams.eps;

    ggml_tensor * e = ggml_mul(ctx0, ggml_get_rows(ctx0, tl.embd_w, code), mask);
    if (extra) {
        e = ggml_add(ctx0, e, extra);
    }
    e = ggml_mul(ctx0, ggml_rms_norm(ctx0, e, eps), tl.enorm_w);
    ggml_tensor * h = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, eps), tl.hnorm_w);
    ggml_tensor * mixed = ggml_mul_mat(ctx0, tl.eh_proj_w, ggml_concat(ctx0, e, h, 0));

    ggml_tensor * cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, mixed, eps), layer.ln_1_w);
    ggml_tensor * q    = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.q_w, cur), d_head, n_head, 1);
    ggml_tensor * k    = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.k_w, cur), d_head, n_head_kv, 1);
    ggml_tensor * v    = ggml_mul_mat(ctx0, layer.v_w, cur);
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.attn_gate_w, cur);

    q = ggml_mul(ctx0, ggml_rms_norm(ctx0, q, eps), layer.q_norm);
    k = ggml_mul(ctx0, ggml_rms_norm(ctx0, k, eps), layer.k_norm);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, DRAMA3_N_ROT, GGML_ROPE_TYPE_NORMAL, 0, DRAMA3_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, DRAMA3_N_ROT, GGML_ROPE_TYPE_NORMAL, 0, DRAMA3_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // write K/V of this row to the slot, attention uses the cached rows + this row
    ggml_tensor * k_new = ggml_reshape_2d(ctx0, k, n_kv, 1);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_new, ggml_view_2d(ctx0, k_slot, n_kv, 1, k_slot->nb[1], (size_t) pos * k_slot->nb[1])));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, v,     ggml_view_2d(ctx0, v_slot, n_kv, 1, v_slot->nb[1], (size_t) pos * v_slot->nb[1])));
    ggml_tensor * k_all = k_new;
    ggml_tensor * v_all = v;
    if (pos > 0) {
        k_all = ggml_concat(ctx0, ggml_view_2d(ctx0, k_slot, n_kv, pos, k_slot->nb[1], 0), k_new, 1);
        v_all = ggml_concat(ctx0, ggml_view_2d(ctx0, v_slot, n_kv, pos, v_slot->nb[1], 0), v,     1);
    }

    ggml_tensor * attn = build_attn(nullptr, nullptr,
                                    ggml_reshape_4d(ctx0, q,     d_head, n_head,    1,       1),
                                    ggml_reshape_4d(ctx0, k_all, d_head, n_head_kv, pos + 1, 1),
                                    ggml_reshape_4d(ctx0, v_all, d_head, n_head_kv, pos + 1, 1),
                                    nullptr, 1.0f / sqrtf((float) d_head), il);
    attn = ggml_mul(ctx0, attn, ggml_sigmoid(ctx0, gate));
    cur = ggml_add(ctx0, mixed, ggml_mul_mat(ctx0, layer.o_w, attn));

    ggml_tensor * ff = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), layer.ln_2_w);
    ff = ggml_swiglu_split(ctx0, ggml_mul_mat(ctx0, layer.ff_gate_w, ff), ggml_mul_mat(ctx0, layer.ff_up_w, ff));
    cur = ggml_add(ctx0, cur, ggml_mul_mat(ctx0, layer.ff_down_w, ff));

    cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), tl.out_norm_w);
    cb(cur, "tower_out", il);
    return cur;
}

// one row: backbone hidden + next main code -> other codes of that frame + next backbone input
ggml_cgraph * mtmd_graph_drama3_gen::build_tower() {
    const int     n_layer   = (int) model.layers.size();
    const int     n_heads   = (int) model.gen_code_head_w->ne[2]; // one per residual codebook
    const int     n_shared  = n_layer - n_heads;
    const int64_t n_cb      = model.gen_code_embd_w->ne[2];
    const int64_t n_embd_cb = model.gen_code_embd_w->ne[0];

    int pos = 0;
    std::vector<ggml_tensor *> slots;
    const auto slot_list = list_drama3_state_slots(hparams, model);
    if (gen_state) {
        float pos_f = 0.0f;
        ggml_backend_tensor_get(gen_state->slots[0], &pos_f, 0, sizeof(float));
        pos = (int) pos_f;
        slots = gen_state->slots;
    } else {
        // buffer reserve: no gen state, use stand-in slots
        for (const auto & slot : slot_list) {
            ggml_tensor * t = nullptr;
            if (slot.in_place) {
                t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
                ggml_set_input(t);
            }
            slots.push_back(t);
        }
    }
    GGML_ASSERT(pos < n_pos_max);

    ggml_tensor * h = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_mmproj_embd);
    ggml_set_name(h, "inp_raw");
    ggml_set_input(h);

    ggml_tensor * inp_code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_code0, "inp_code0");
    ggml_set_input(inp_code0);

    // codes 2.. of the input frame, as row ids of the merged codebook table
    ggml_tensor * inp_extra = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cb - 2);
    ggml_set_name(inp_extra, "inp_extra");
    ggml_set_input(inp_extra);

    // [next frame is speech, input frame is speech]
    ggml_tensor * inp_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 2);
    ggml_set_name(inp_scale, "inp_scale");
    ggml_set_input(inp_scale);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_heads);
    ggml_set_name(inp_rand, "inp_rand");
    ggml_set_input(inp_rand);

    ggml_tensor * mask     = ggml_view_1d(ctx0, inp_scale, 1, 0);
    ggml_tensor * extra_on = ggml_view_1d(ctx0, inp_scale, 1, sizeof(float));

    ggml_tensor * cb_all = ggml_reshape_2d(ctx0, model.gen_code_embd_w, n_embd_cb, model.gen_code_embd_w->ne[1] * n_cb);
    ggml_tensor * extra  = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_get_rows(ctx0, cb_all, inp_extra)));
    extra = ggml_mul(ctx0, ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, extra), n_embd_cb, 1), extra_on);

    mtmd_graph_qwen3tts_gen::code_gen cg(*this, top_k, top_p);
    ggml_tensor * out_codes = cg.cache_set(ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_cb), 0, inp_code0);

    ggml_tensor * code = inp_code0;
    ggml_tensor * x    = ggml_reshape_2d(ctx0, h, n_mmproj_embd, 1);
    for (int il = 0; il < n_layer; il++) {
        x = tower_layer(x, il, code, mask, il == 0 ? extra : nullptr, inp_pos, slots[1 + 2 * il], slots[2 + 2 * il], pos);
        if (il < n_shared) {
            continue;
        }
        const int ih = il - n_shared;
        ggml_tensor * head   = ggml_view_2d(ctx0, model.gen_code_head_w, model.gen_code_head_w->ne[0], model.gen_code_head_w->ne[1],
                                            model.gen_code_head_w->nb[1], (size_t) ih * model.gen_code_head_w->nb[2]);
        ggml_tensor * logits = ggml_mul_mat(ctx0, head, x);
        if (temp > 0.0f && temp != 1.0f) {
            logits = ggml_scale(ctx0, logits, 1.0f / temp);
        }
        code = cg.do_sampling(logits, ggml_view_1d(ctx0, inp_rand, 1, (size_t) ih * sizeof(float)));
        out_codes = cg.cache_set(out_codes, ih + 1, code);
    }

    ggml_tensor * pos_slot = slots[0];
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_scale_bias(ctx0, pos_slot, 1.0f, 1.0f), pos_slot));

    out_codes = ggml_cont(ctx0, out_codes);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    // backbone input of the next row: (E_lm[main] + E_cb1[code 1]) / sqrt(2)
    ggml_tensor * code1 = ggml_view_1d(ctx0, out_codes, 1, out_codes->nb[1]);
    ggml_tensor * cb1   = ggml_view_2d(ctx0, model.gen_code_embd_w, n_embd_cb, model.gen_code_embd_w->ne[1],
                                       model.gen_code_embd_w->nb[1], model.gen_code_embd_w->nb[2]);
    ggml_tensor * out_embd = ggml_add(ctx0, ggml_get_rows(ctx0, model.gen_code_out_embd_w, inp_code0), ggml_get_rows(ctx0, cb1, code1));
    out_embd = ggml_scale(ctx0, out_embd, 1.0f / sqrtf(2.0f));
    cb(out_embd, "gen_audio_out", -1);
    ggml_build_forward_expand(gf, out_embd);

    return gf;
}

// frames -> PCM, qwen3tts code2wav blocks without pre_conv and tfm projections
ggml_cgraph * mtmd_graph_drama3_gen::build_codec() {
    const auto & c2w = model.c2w;
    const int64_t n_cb = model.gen_code_embd_w->ne[2];

    ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_cb);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);

    mtmd_graph_qwen3tts_gen::code2wav dec(*this);
    dec.gelu_erf = true;
    for (const auto & slot : list_c2w_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        dec.state_in[slot.name] = t;
    }

    // out_proj is folded into the codebook tables at conversion
    auto group_ids = [&](int g) {
        return ggml_view_1d(ctx0, inp_codes, n_frames, (size_t) g * inp_codes->nb[1]);
    };
    ggml_tensor * cur = ggml_get_rows(ctx0, c2w.quant_first_cb_w, group_ids(0)); // [dim, T]
    for (int g = 1; g < n_cb; g++) {
        ggml_tensor * table = ggml_view_2d(ctx0, c2w.quant_rest_cb_w, c2w.quant_rest_cb_w->ne[0], c2w.quant_rest_cb_w->ne[1],
                                           c2w.quant_rest_cb_w->nb[1], (size_t) (g - 1) * c2w.quant_rest_cb_w->nb[2]);
        cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, table, group_ids(g)));
    }

    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        cur = dec.tfm_layer_forward(cur, c2w.tfm_layers[il], il);
    }
    cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps), c2w.tfm_output_norm_w);
    cb(cur, "wav_tfm_out", -1);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [T, dim]
    for (size_t il = 0; il < c2w.upsample.size(); il++) {
        const auto & up = c2w.upsample[il];
        x = dec.causal_conv_transpose1d(x, up.conv_w, up.conv_b, 2, "");
        x = dec.convnext_block(x, up, "up" + std::to_string(il));
    }

    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };
    x = dec.causal_conv1d(x, c2w.dac_entry_w, c2w.dac_entry_b, 1, "dac_entry");
    for (size_t il = 0; il < c2w.dac.size(); il++) {
        const auto & blk = c2w.dac[il];
        const int stride = (int) (blk.conv_w->ne[0] / 2);
        const std::string blk_name = "dac" + std::to_string(il);
        x = dec.snake(x, blk.snake_alpha, blk.snake_beta);
        x = dec.causal_conv_transpose1d(x, blk.conv_w, blk.conv_b, stride, blk_name + "_tail");
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            x = dec.dac_res_unit(x, blk.res[ir], DAC_DILATIONS[ir], blk_name + "_res" + std::to_string(ir));
        }
    }
    x = dec.snake(x, c2w.dac_post_snake_alpha, c2w.dac_post_snake_beta);
    x = dec.causal_conv1d(x, c2w.dac_post_conv_w, c2w.dac_post_conv_b, 1, "dac_post_conv");
    x = ggml_tanh(ctx0, ggml_reshape_1d(ctx0, x, x->ne[0]));
    ggml_set_name(x, "out_audio");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    for (auto & slot : dec.state_out) {
        ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
        ggml_set_output(slot.second);
        ggml_build_forward_expand(gf, slot.second);
    }
    return gf;
}

ggml_cgraph * mtmd_graph_drama3_gen::build() {
    return gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV ? build_codec() : build_tower();
}
