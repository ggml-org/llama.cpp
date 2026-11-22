#include "models.h"

// SEANet convolution stack of the mimi codec, see pocket_tts/modules/seanet.py
//
// tensors are T-first here: [T, C]
// the convs are causal: left context comes from a state slot, or from padding on a cold start

static int64_t div_ceil(int64_t a, int64_t b) {
    return a / b + (a % b ? 1 : 0);
}

// x: [T, IC], w: [K, IC, OC] -> [T / stride, OC]
// the convs are causal, so the whole K - stride padding goes on the left
ggml_tensor * mtmd_graph_pockettts_seanet::conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation,
                                                  bool pad_replicate, const std::string & state_name) const {
    const int64_t k_size  = (w->ne[0] - 1) * dilation + 1;
    const int64_t p_total = k_size - stride;

    // trailing padding so the last frame is not dropped, see pad_for_conv1d() in conv.py
    const int64_t n_frames  = div_ceil(x->ne[0] - k_size + p_total, stride);
    const int64_t ideal_len = n_frames * stride + k_size - p_total;
    const int64_t p_extra   = ideal_len - x->ne[0];

    if (!state_name.empty() && p_total > 0) {
        // streaming: the left context is the tail of the previous call
        ggml_tensor * left = state_in.at(state_name); // [p_total, IC]
        x = ggml_concat(ctx0, left, x, 0);
        state_out.push_back({state_name,
            ggml_cont(ctx0, ggml_view_2d(ctx0, x, p_total, x->ne[1], x->nb[1],
                                         (size_t) (x->ne[0] - p_total) * x->nb[0]))});
    } else if (pad_replicate && p_total > 0) {
        // the resamplers repeat the first frame instead of zero-padding
        ggml_tensor * first = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0);
        ggml_tensor * left  = ggml_repeat_4d(ctx0, first, p_total, x->ne[1], 1, 1);
        x = ggml_concat(ctx0, left, x, 0);
        if (pad_replicate_end && p_extra > 0) {
            ggml_tensor * last  = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], (size_t) (x->ne[0] - 1) * x->nb[0]);
            ggml_tensor * right = ggml_repeat_4d(ctx0, last, p_extra, x->ne[1], 1, 1);
            x = ggml_concat(ctx0, x, right, 0);
        } else {
            x = ggml_pad_ext(ctx0, x, 0, p_extra, 0, 0, 0, 0, 0, 0);
        }
    } else {
        x = ggml_pad_ext(ctx0, x, p_total, p_extra, 0, 0, 0, 0, 0, 0);
    }

    ggml_tensor * y;
    if (f32_im2col) {
        // ggml_conv_1d() with an F32 im2col, the kernel must be F32
        ggml_tensor * col = ggml_im2col(ctx0, w, x, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
        y = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[2] * col->ne[1]),
                         ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2]));
    } else {
        y = ggml_conv_1d(ctx0, w, x, stride, 0, dilation);
    }
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

// x: [T, IC], w: [K, OC/groups, IC] -> [T * stride, OC]
// the K - stride overlap tail belongs to the next call: added to its head when streaming, else dropped
ggml_tensor * mtmd_graph_pockettts_seanet::conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride,
                                                            const std::string & state_name) const {
    const int64_t K         = w->ne[0];
    const int64_t T         = x->ne[0];
    const int64_t p_total   = K - stride;
    const bool    depthwise = w->ne[1] == 1 && w->ne[2] > 1;
    const int64_t OC        = depthwise ? w->ne[2] : w->ne[1];
    const int64_t emit_len  = T * stride;

    // one column per input step, holding the [K, OC] window that col2im scatter-adds at t * stride
    ggml_tensor * col;
    if (depthwise) {
        // one group per channel: a batched matmul over the channels scales the kernel by each step
        ggml_tensor * krn = ggml_reshape_3d(ctx0, w, 1, K, OC);             // [1, K, OC]
        ggml_tensor * xs  = ggml_reshape_3d(ctx0, x, 1, T, OC);             // [1, T, OC]
        col = ggml_mul_mat(ctx0, krn, xs);                                  // [K, T, OC]
        col = ggml_cont(ctx0, ggml_permute(ctx0, col, 0, 2, 1, 3));         // [K, OC, T]
        col = ggml_reshape_2d(ctx0, col, K * OC, T);
    } else {
        ggml_tensor * w2 = ggml_reshape_2d(ctx0, w, K * OC, w->ne[2]);
        w2               = ggml_cont(ctx0, ggml_transpose(ctx0, w2));       // [IC, K * OC]
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));        // [IC, T]
        col = ggml_mul_mat(ctx0, w2, xt);
    }
    ggml_tensor * full = ggml_col2im_1d(ctx0, col, stride, OC, 0);          // [emit_len + p_total, OC]

    ggml_tensor * out;
    if (state_name.empty() || p_total == 0) {
        out = ggml_cont(ctx0, ggml_view_2d(ctx0, full, emit_len, full->ne[1], full->nb[1], 0));
    } else {
        // overlap-add the tail the previous call held back
        ggml_tensor * prev = state_in.at(state_name); // [p_total, OC]
        ggml_tensor * head = ggml_add(ctx0, ggml_view_2d(ctx0, full, p_total, full->ne[1], full->nb[1], 0), prev);
        if (emit_len > p_total) {
            ggml_tensor * rest = ggml_view_2d(ctx0, full, emit_len - p_total, full->ne[1], full->nb[1],
                                              (size_t) p_total * full->nb[0]);
            out = ggml_concat(ctx0, head, rest, 0);
        } else {
            out = head;
        }
        state_out.push_back({state_name,
            ggml_cont(ctx0, ggml_view_2d(ctx0, full, p_total, full->ne[1], full->nb[1],
                                         (size_t) emit_len * full->nb[0]))});
    }

    if (b) {
        out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return out;
}

ggml_tensor * mtmd_graph_pockettts_seanet::res_unit(ggml_tensor * x, const mtmd_seanet::stage & stage, int dilation,
                                                    const std::string & state_prefix) const {
    ggml_tensor * h = ggml_elu(ctx0, x);
    h = conv1d(h, stage.res_conv1_w, stage.res_conv1_b, 1, dilation, false, state_prefix);
    h = ggml_elu(ctx0, h);
    // the second conv is pointwise, it needs no left context
    h = conv1d(h, stage.res_conv2_w, stage.res_conv2_b, 1, 1);
    return ggml_add(ctx0, x, h);
}

ggml_tensor * mtmd_graph_pockettts_seanet::encode(ggml_tensor * x) const {
    const auto & seanet = model.seanet;
    const bool   stream = !state_in.empty();

    ggml_tensor * cur = conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1, false,
                               stream ? "enc_in" : "");
    cb(cur, "seanet_enc_in", -1);

    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage  = seanet.stages[i];
        const int    stride = hparams.seanet_ratios[i];
        const std::string id = std::to_string(i);

        cur = res_unit(cur, stage, 1, stream ? "enc_res_" + id : "");
        cur = ggml_elu(ctx0, cur);
        cur = conv1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride, 1, false,
                     stream ? "enc_down_" + id : "");
        cb(cur, "seanet_enc_stage", i);
    }

    cur = ggml_elu(ctx0, cur);
    cur = conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1, false,
                 stream ? "enc_out" : "");
    cb(cur, "seanet_enc_out", -1);

    return cur;
}

ggml_tensor * mtmd_graph_pockettts_seanet::decode(ggml_tensor * x) const {
    const auto & seanet = model.seanet;
    const bool   stream = !state_in.empty();

    ggml_tensor * cur = conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1, false,
                               stream ? "dec_in" : "");
    cb(cur, "seanet_dec_in", -1);

    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage = seanet.stages[i];
        // the decoder mirrors the encoder, so the ratios are walked backwards
        const int    stride = hparams.seanet_ratios[hparams.seanet_n_stage - 1 - i];
        const std::string id = std::to_string(i);

        cur = ggml_elu(ctx0, cur);
        cur = conv_transpose1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride,
                               stream ? "dec_up_" + id : "");
        cur = res_unit(cur, stage, 1, stream ? "dec_res_" + id : "");
        cb(cur, "seanet_dec_stage", i);
    }

    cur = ggml_elu(ctx0, cur);
    cur = conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1, false,
                 stream ? "dec_out" : "");
    cb(cur, "seanet_dec_out", -1);

    return cur;
}

// pre-norm blocks with layer scale on both residual paths, see mimi_transformer.py
// causal over a window of mimi_tfm_context frames, streams through the slots pos, tfm_k_<il>, tfm_v_<il>
// x: [dim, n_pos] -> [dim, n_pos]
ggml_tensor * mtmd_graph_pockettts_seanet::transformer(ggml_tensor * cur, const std::vector<mtmd_layer> & layers, int n_head, float eps) const {
    const int64_t n_pos  = cur->ne[1];
    const int64_t d_head = cur->ne[0] / n_head;
    const int64_t prefix = hparams.mimi_tfm_context - 1;
    const int64_t n_kv   = prefix + n_pos;
    const float   scale  = 1.0f / sqrtf((float) d_head);

    // positions continue across calls, the counter lives in the state
    ggml_tensor * base    = ggml_reshape_1d(ctx0, state_in.at("pos"), 1);
    ggml_tensor * inp_pos = ggml_cast(ctx0, ggml_add(ctx0, ggml_arange(ctx0, 0.0f, (float) n_pos, 1.0f), base),
                                      GGML_TYPE_I32);
    state_out.push_back({"pos", ggml_scale_bias(ctx0, state_in.at("pos"), 1.0f, (float) n_pos)});

    // banded causal mask over [cached prefix | this chunk]
    // the last factor masks out cache rows that hold no real frame yet
    ggml_tensor * pos_k = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_kv, 1.0f), n_kv, 1);
    ggml_tensor * pos_q = ggml_reshape_2d(ctx0, ggml_arange(ctx0, (float) prefix, (float) (prefix + n_pos), 1.0f), 1, n_pos);
    ggml_tensor * diff  = ggml_sub(ctx0, ggml_repeat_4d(ctx0, pos_q, n_kv, n_pos, 1, 1), pos_k);

    ggml_tensor * keep = ggml_mul(ctx0,
        ggml_step(ctx0, ggml_scale_bias(ctx0, diff, 1.0f, 0.5f)),                                     // delta >= 0
        ggml_step(ctx0, ggml_scale_bias(ctx0, diff, -1.0f, (float) hparams.mimi_tfm_context - 0.5f))); // delta < context
    keep = ggml_mul(ctx0, keep,
        ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, pos_k, base), 1.0f, 0.5f - (float) prefix)));
    ggml_tensor * kq_mask = ggml_reshape_4d(ctx0, ggml_log(ctx0, keep), n_kv, n_pos, 1, 1);
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
    }

    for (size_t il = 0; il < layers.size(); il++) {
        const auto & layer = layers[il];
        ggml_tensor * inp = cur;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        ggml_tensor * Qcur = build_mm(layer.q_w, cur);
        ggml_tensor * Kcur = build_mm(layer.k_w, cur);
        ggml_tensor * Vcur = build_mm(layer.v_w, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
        Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, d_head, GGML_ROPE_TYPE_NORMAL, 0,
                             hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, d_head, GGML_ROPE_TYPE_NORMAL, 0,
                             hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // prepend the cached window, then keep this chunk's tail for the next call
        const std::string k_name = "tfm_k_" + std::to_string(il);
        const std::string v_name = "tfm_v_" + std::to_string(il);
        ggml_tensor * k_full = ggml_concat(ctx0, state_in.at(k_name),
                                           ggml_reshape_2d(ctx0, Kcur, d_head * n_head, n_pos), 1);
        ggml_tensor * v_full = ggml_concat(ctx0, state_in.at(v_name), Vcur, 1);
        state_out.push_back({k_name, ggml_cont(ctx0, ggml_view_2d(ctx0, k_full, k_full->ne[0], prefix,
                                                                  k_full->nb[1], (size_t) n_pos * k_full->nb[1]))});
        state_out.push_back({v_name, ggml_cont(ctx0, ggml_view_2d(ctx0, v_full, v_full->ne[0], prefix,
                                                                  v_full->nb[1], (size_t) n_pos * v_full->nb[1]))});

        ggml_tensor * q_cur = ggml_reshape_4d(ctx0, Qcur,   d_head, n_head, n_pos, 1);
        ggml_tensor * k_cur = ggml_reshape_4d(ctx0, k_full, d_head, n_head, n_kv,  1);
        ggml_tensor * v_cur = ggml_reshape_4d(ctx0, v_full, d_head, n_head, n_kv,  1);

        cur = build_attn(layer.o_w, nullptr, q_cur, k_cur, v_cur, kq_mask, scale, il);
        cur = ggml_mul(ctx0, cur, layer.ls_1_w);
        cur = ggml_add(ctx0, cur, inp);

        inp = cur;
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_GELU, il);
        cur = ggml_mul(ctx0, cur, layer.ls_2_w);
        cur = ggml_add(ctx0, cur, inp);
    }
    return cur;
}

ggml_tensor * mtmd_graph_pockettts_seanet::mimi_decode(ggml_tensor * x, int n_head, float eps) const {
    // model frame rate -> encoder frame rate, depthwise transposed conv
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    cur = conv_transpose1d(cur, model.gen_upsample_w, nullptr, hparams.mimi_downsample, "up");
    cb(cur, "mimi_upsample", -1);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = transformer(cur, model.gen_tfm_layers, n_head, eps);
    cb(cur, "mimi_dec_tfm", -1);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    return decode(cur);
}

void mtmd_graph_pockettts_seanet::state_inputs(const std::vector<c2w_state_slot> & slots) {
    for (const auto & slot : slots) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        state_in[slot.name] = t;
    }
}

void mtmd_graph_pockettts_seanet::state_outputs(ggml_cgraph * graph) const {
    for (const auto & s : state_out) {
        ggml_set_name(s.second, ("state_out_" + s.first).c_str());
        ggml_set_output(s.second);
        ggml_build_forward_expand(graph, s.second);
    }
}
