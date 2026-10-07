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
ggml_tensor * clip_graph_pockettts_seanet::conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation,
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
        x = ggml_pad_ext(ctx0, x, 0, p_extra, 0, 0, 0, 0, 0, 0);
    } else {
        x = ggml_pad_ext(ctx0, x, p_total, p_extra, 0, 0, 0, 0, 0, 0);
    }

    ggml_tensor * y = ggml_conv_1d(ctx0, w, x, stride, 0, dilation);
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

// x: [T, IC], w: [K, OC/groups, IC] -> [T * stride, OC]
// the K - stride overlap tail belongs to the next call: added to its head when streaming, else dropped
ggml_tensor * clip_graph_pockettts_seanet::conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride,
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

ggml_tensor * clip_graph_pockettts_seanet::res_unit(ggml_tensor * x, const clip_seanet::stage & stage, int dilation,
                                                    const std::string & state_prefix) const {
    ggml_tensor * h = ggml_elu(ctx0, x);
    h = conv1d(h, stage.res_conv1_w, stage.res_conv1_b, 1, dilation, false, state_prefix);
    h = ggml_elu(ctx0, h);
    // the second conv is pointwise, it needs no left context
    h = conv1d(h, stage.res_conv2_w, stage.res_conv2_b, 1, 1);
    return ggml_add(ctx0, x, h);
}

ggml_tensor * clip_graph_pockettts_seanet::encode(ggml_tensor * x) const {
    const auto & seanet = model.seanet;

    ggml_tensor * cur = conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1);
    cb(cur, "seanet_enc_in", -1);

    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage  = seanet.stages[i];
        const int    stride = hparams.seanet_ratios[i];

        cur = res_unit(cur, stage, 1);
        cur = ggml_elu(ctx0, cur);
        cur = conv1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride, 1);
        cb(cur, "seanet_enc_stage", i);
    }

    cur = ggml_elu(ctx0, cur);
    cur = conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1);
    cb(cur, "seanet_enc_out", -1);

    return cur;
}

ggml_tensor * clip_graph_pockettts_seanet::decode(ggml_tensor * x) const {
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
