#include "models.h"

// Voxtral Realtime 4B - Causal Audio Encoder
//
// Key differences from standard Whisper encoder:
// - Causal conv1d (left-padded only)
// - RoPE (interleaved, mode=0) instead of learned positional embeddings
// - Causal attention with sliding window (750)
// - SwiGLU FFN
// - RMSNorm (not LayerNorm)
// - No AvgPool
// - Bias on Q, V, O but NOT on K
// - Stack factor = 4 (downsample)

ggml_cgraph * clip_graph_voxtral_realtime_enc::build() {
    const int n_frames = img.nx;

    ggml_tensor * inp = build_inp_raw(1);

    // ========================================================================
    // Causal conv1d block
    //
    // Python reference (causal_conv1d):
    //   padding_total = kernel_size - stride  (left-pad only)
    //   extra_padding for alignment when stride > 1
    //
    // Conv0: kernel=3, stride=1 -> left-pad by 2
    // Conv1: kernel=3, stride=2 -> left-pad by 1, plus alignment padding
    // ========================================================================
    {
        // Conv0: kernel_size=3, stride=1
        // Causal left-pad by (kernel_size - stride) = 3 - 1 = 2
        // inp shape: [n_frames, n_mel, 1] in ggml notation
        // Pad dimension 0 (time/n_frames): left=2, right=0
        ggml_tensor * padded = ggml_pad_ext(ctx0, inp,
            2, 0,   // dim0 (time): left=2, right=0
            0, 0,   // dim1 (channels): no padding
            0, 0,   // dim2: no padding
            0, 0);  // dim3: no padding
        ggml_tensor * cur = ggml_conv_1d(ctx0, model.conv1d_1_w, padded, 1, 0, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cb(cur, "after_conv0", -1);

        // Conv1: kernel_size=3, stride=2
        // Causal left-pad by (kernel_size - stride) = 3 - 2 = 1
        // Also need to compute extra_padding for alignment:
        //   n_frames_after_conv0 = n_frames (stride=1, same length)
        //   effective_ks = 3
        //   padding_total = 1
        //   n_frames_out = ceil((n_frames_after_conv0 + padding_total - effective_ks + 1) / stride)
        //   target_length = (n_frames_out - 1) * stride + effective_ks - padding_total
        //   extra_padding = target_length - n_frames_after_conv0
        //
        // For simplicity, compute the extra padding needed:
        int64_t conv0_out_len = n_frames;  // stride=1, same length
        int64_t ks = 3, stride = 2;
        int64_t pad_total = ks - stride;  // 1
        double n_out_f = (double)(conv0_out_len + pad_total - ks) / stride + 1.0;
        int64_t n_out = (int64_t)ceil(n_out_f);
        int64_t target_len = (n_out - 1) * stride + ks - pad_total;
        int64_t extra_pad = target_len - conv0_out_len;
        if (extra_pad < 0) extra_pad = 0;

        padded = ggml_pad_ext(ctx0, cur,
            (int)pad_total, (int)extra_pad,  // dim0 (time): left=1, right=extra
            0, 0,   // dim1 (channels): no padding
            0, 0,   // dim2: no padding
            0, 0);  // dim3: no padding
        cur = ggml_conv_1d(ctx0, model.conv1d_2_w, padded, 2, 0, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cb(cur, "after_conv1", -1);

        // Transpose to [n_embd, seq_len] (same as whisper-enc.cpp)
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(inp, "after_conv1d", -1);
    }

    const int64_t n_pos = inp->ne[1]; // sequence length after conv

    // Left-truncate to multiple of stack_factor
    const int stack_factor = hparams.proj_stack_factor;
    const int64_t trunc = n_pos % stack_factor;
    if (trunc > 0) {
        // Skip first 'trunc' positions to align with stack_factor
        inp = ggml_view_2d(ctx0, inp,
            inp->ne[0], n_pos - trunc,
            inp->nb[1],
            trunc * inp->nb[1]);
        inp = ggml_cont(ctx0, inp);
    }

    const int64_t seq_len = inp->ne[1];
    cb(inp, "after_truncate", -1);

    // Create position IDs for RoPE: [0, 1, 2, ..., seq_len-1]
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ========================================================================
    // Build sliding window + causal attention mask
    //
    // mask[i][j] = 0.0   if j <= i AND j >= i - (window - 1)
    // mask[i][j] = -inf   otherwise
    //
    // The sliding window size is 750 (from encoder config).
    // We build this as an input tensor of shape [seq_len, seq_len].
    // ========================================================================
    const int sliding_window = 750;

    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // Causal transformer with RoPE and sliding window attention
    ggml_tensor * inpL = inp;

    const float rope_theta = 1000000.0f;

    // Compute actual d_head from Q weight tensor shape
    // In Voxtral Realtime, head_dim (64) != n_embd / n_head (1280/32 = 40)
    // Q weight shape is [n_embd, n_head * d_head_actual]
    const int d_head_actual = (int)(model.layers[0].q_w->ne[1] / n_head);
    const float kq_scale_actual = 1.0f / sqrtf((float)d_head_actual);

    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        // Pre-attention RMSNorm
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_RMS, eps, il);
        cb(cur, "layer_inp_normed", il);

        // Self-attention with RoPE
        {
            // Q, K, V projections
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            // K has NO bias in Voxtral Realtime encoder

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            // Reshape to [d_head_actual, n_head, seq_len]
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head_actual, n_head, seq_len);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head_actual, n_head, seq_len);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head_actual, n_head, seq_len);

            // Apply RoPE (interleaved style, mode=0)
            Qcur = ggml_rope_ext(ctx0, Qcur, positions, nullptr,
                d_head_actual, 0, 0,
                rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx0, Kcur, positions, nullptr,
                d_head_actual, 0, 0,
                rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Attention with sliding window causal mask
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx0, Vcur, 1, 2, 0, 3);
            v = ggml_cont(ctx0, v);

            // Q*K^T -> [seq_len, seq_len, n_head, 1]
            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

            // Apply sliding window + causal mask via ggml_soft_max_ext
            // The mask tensor kq_mask has shape [seq_len, seq_len]
            // It will be broadcast across the n_head dimension
            // mask values: 0.0 for attend, -inf for masked
            kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale_actual, 0.0f);

            // KQV
            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*n_head, seq_len);

            cb(cur, "kqv_out", il);

            // Output projection
            cur = ggml_mul_mat(ctx0, layer.o_w, cur);
            if (layer.o_b) {
                cur = ggml_add(ctx0, cur, layer.o_b);
            }
            cb(cur, "attn_out", il);
        }

        // Residual
        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cb(cur, "ffn_inp", il);

        // Pre-FFN RMSNorm
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_RMS, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // SwiGLU FFN (FFN_SILU with gate tensor = SwiGLU)
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            FFN_SILU, il);

        cb(cur, "ffn_out", il);

        // Residual
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // Post-layernorm (RMSNorm)
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_RMS, eps, -1);
    }

    // Stack frames (4x downsample)
    if (model.audio_has_stack_frames()) {
        inpL = build_stack(inpL, hparams.proj_stack_factor, n_embd);
        cb(inpL, "after_stacked", -1);
    }

    // Adapter: Linear -> GELU -> Linear (no bias)
    {
        ggml_tensor * cur = inpL;
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);
        inpL = cur;
    }

    cb(inpL, "projected", -1);

    ggml_build_forward_expand(gf, inpL);

    return gf;
}
