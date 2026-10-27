#include "models.h"

// streaming encoder state: base position, conv left context, short-conv taps and attention window
// a zero state gives the same result as encoding the whole clip
std::vector<c2w_state_slot> list_lfm3a_state_slots(const clip_hparams & hparams, const clip_model & model) {
    std::vector<c2w_state_slot> slots;

    const int64_t n_mel  = hparams.n_mel_bins;
    const int64_t n_ch   = model.pre_encode_conv_X_w[0]->ne[3];
    const int64_t prefix = hparams.attn_window_size - 1;

    slots.push_back({"pos", 1, 1});
    slots.push_back({"ds0", 1, n_mel});
    slots.push_back({"ds1", 1, (n_mel / 2) * n_ch});
    slots.push_back({"ds2", 1, (n_mel / 4) * n_ch});
    for (size_t il = 0; il < model.layers.size(); il++) {
        const auto & layer = model.layers[il];
        if (layer.conv_dw_w) {
            slots.push_back({"conv_" + std::to_string(il), layer.conv_dw_w->ne[0] - 1, hparams.n_embd});
        } else {
            slots.push_back({"k_" + std::to_string(il), layer.k_w->ne[1], prefix});
            slots.push_back({"v_" + std::to_string(il), layer.v_w->ne[1], prefix});
        }
    }

    return slots;
}

// LFM3-Audio encoder: conv downsampler (8x in time) -> causal LFM2 stack -> MLP projector
ggml_cgraph * clip_graph_lfm3a::build() {
    const norm_type norm_t = NORM_TYPE_RMS;
    const float rope_theta = hparams.rope_theta;
    const int64_t prefix   = hparams.attn_window_size - 1;

    // the graph always streams, a zero state encodes a clip from its start
    std::map<std::string, ggml_tensor *> state_in;
    std::vector<std::pair<std::string, ggml_tensor *>> state_out;
    for (const auto & slot : list_lfm3a_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        state_in[slot.name] = t;
    }

    // stride-2 conv over [time, freq, ch]; the left column comes from the state and the right edge is zero-padded
    auto conv_down = [&](ggml_tensor * x, const std::string & name, ggml_tensor * w, ggml_tensor * b, bool depthwise) {
        ggml_tensor * left = ggml_reshape_4d(ctx0, state_in.at(name), 1, x->ne[1], x->ne[2], 1);
        x = ggml_concat(ctx0, left, x, 0);
        ggml_tensor * last = ggml_view_4d(ctx0, x, 1, x->ne[1], x->ne[2], 1, x->nb[1], x->nb[2], x->nb[3], (x->ne[0] - 1) * x->nb[0]);
        state_out.push_back({name, ggml_reshape_2d(ctx0, ggml_cont(ctx0, last), 1, x->ne[1] * x->ne[2])});
        x = ggml_pad_ext(ctx0, x, 0, 1, 0, 0, 0, 0, 0, 0);
        x = depthwise ? ggml_conv_2d_dw_direct(ctx0, w, x, 2, 2, 0, 1, 1, 1)
                      : ggml_conv_2d(ctx0, w, x, 2, 2, 0, 1, 1, 1);
        return ggml_add(ctx0, x, b);
    };

    // inp: [n_frames, n_mel, 1]
    ggml_tensor * cur = build_inp_raw(1);

    // downsampler, same layout as the lfm2a pre-encoder but with SiLU
    cur = conv_down(cur, "ds0", model.pre_encode_conv_X_w[0], model.pre_encode_conv_X_b[0], false);
    cur = ggml_silu_inplace(ctx0, cur);

    cur = conv_down(cur, "ds1", model.pre_encode_conv_X_w[2], model.pre_encode_conv_X_b[2], true);
    cur = ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
    cur = ggml_silu_inplace(ctx0, cur);

    cur = conv_down(cur, "ds2", model.pre_encode_conv_X_w[5], model.pre_encode_conv_X_b[5], true);
    cur = ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
    cur = ggml_silu_inplace(ctx0, cur);
    cb(cur, "downsample.conv", -1);

    // [n_pos, n_mel/8, n_ch] -> [n_ch * n_mel/8, n_pos]
    const int n_pos = cur->ne[0];
    cur = ggml_reshape_2d(ctx0, cur, n_pos, cur->ne[1] * cur->ne[2]);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = build_mm(model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
    cb(cur, "downsample.out", -1);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    state_out.push_back({"pos", ggml_scale_bias(ctx0, state_in.at("pos"), 1.0f, (float) n_pos)});

    // causal mask with a sliding window, shared by all attention layers
    ggml_tensor * kq_mask = build_inp_attn_mask(prefix + n_pos, n_pos);

    ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];

        cur = build_norm(inpL, layer.ln_1_w, nullptr, norm_t, eps, il);
        cb(cur, "operator_norm", il);

        if (layer.conv_dw_w) {
            // short conv: in_proj -> b * x -> causal depthwise conv -> c * conv -> out_proj
            ggml_tensor * bcx = build_mm(layer.conv_pw1_w, cur);
            cb(bcx, "conv_in_proj", il);

            ggml_tensor * b = ggml_view_2d(ctx0, bcx, n_embd, n_pos, bcx->nb[1], 0);
            ggml_tensor * c = ggml_view_2d(ctx0, bcx, n_embd, n_pos, bcx->nb[1], ggml_row_size(bcx->type, n_embd));
            ggml_tensor * x = ggml_view_2d(ctx0, bcx, n_embd, n_pos, bcx->nb[1], ggml_row_size(bcx->type, 2 * n_embd));

            ggml_tensor * bx = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mul(ctx0, b, x)));
            const int d_conv = layer.conv_dw_w->ne[0] - 1;
            const std::string name = "conv_" + std::to_string(il);
            bx = ggml_concat(ctx0, state_in.at(name), bx, 0);
            state_out.push_back({name, ggml_cont(ctx0, ggml_view_2d(ctx0, bx, d_conv, n_embd, bx->nb[1], (size_t) n_pos * bx->nb[0]))});
            bx = ggml_reshape_3d(ctx0, bx, bx->ne[0], n_embd, 1);

            cur = ggml_ssm_conv(ctx0, bx, layer.conv_dw_w);
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_pos);
            cb(cur, "conv_conv", il);

            cur = ggml_mul(ctx0, c, cur);
            cur = build_mm(layer.conv_pw2_w, cur);
            cb(cur, "conv_out_proj", il);
        } else {
            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head,    n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head_kv, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head_kv, n_pos);

            Qcur = build_norm(Qcur, layer.q_norm, nullptr, norm_t, eps, il);
            Kcur = build_norm(Kcur, layer.k_norm, nullptr, norm_t, eps, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, positions, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx0, Kcur, positions, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // the cached window goes in front, this chunk's tail is kept for the next call
            const std::string k_name = "k_" + std::to_string(il);
            const std::string v_name = "v_" + std::to_string(il);
            const int64_t n_embd_kv = d_head * n_head_kv;
            ggml_tensor * K = ggml_concat(ctx0, state_in.at(k_name), ggml_reshape_2d(ctx0, Kcur, n_embd_kv, n_pos), 1);
            ggml_tensor * V = ggml_concat(ctx0, state_in.at(v_name), ggml_reshape_2d(ctx0, Vcur, n_embd_kv, n_pos), 1);
            state_out.push_back({k_name, ggml_cont(ctx0, ggml_view_2d(ctx0, K, n_embd_kv, prefix, K->nb[1], (size_t) n_pos * K->nb[1]))});
            state_out.push_back({v_name, ggml_cont(ctx0, ggml_view_2d(ctx0, V, n_embd_kv, prefix, V->nb[1], (size_t) n_pos * V->nb[1]))});
            Kcur = ggml_reshape_3d(ctx0, K, d_head, n_head_kv, prefix + n_pos);
            Vcur = ggml_reshape_3d(ctx0, V, d_head, n_head_kv, prefix + n_pos);

            cur = build_attn(layer.o_w, nullptr, Qcur, Kcur, Vcur, kq_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, nullptr, norm_t, eps, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
            layer.ff_up_w, nullptr,
            layer.ff_gate_w, nullptr,
            layer.ff_down_w, nullptr,
            FFN_SILU, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, inpL, cur);
        inpL = cur;
    }

    cur = build_norm(inpL, model.post_ln_w, nullptr, norm_t, eps, -1);
    cb(cur, "embedding_norm", -1);

    // projector: LayerNorm -> Linear -> GELU -> Linear
    cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_3_w, model.mm_3_b, FFN_GELU_ERF, -1);
    cb(cur, "projected", -1);

    // the state outputs go first so that the embeddings stay the last node of the graph
    for (const auto & s : state_out) {
        ggml_set_name(s.second, ("state_out_" + s.first).c_str());
        ggml_set_output(s.second);
        ggml_build_forward_expand(gf, s.second);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}
