#include "models.h"

#include <cstdlib>

ggml_cgraph * clip_graph_qwen3a::build() {
    // Ref implementation: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

    // inp_raw: [n_frames, n_mel, 1]  (nx=n_frames, ny=n_mel)
    ggml_tensor * inp = build_inp_raw(1);

    const int64_t n_frames   = inp->ne[0]; // total frames, padded to multiple of chunk_size
    const int64_t n_mel      = inp->ne[1]; // 128
    const int64_t chunk_size = 100;        // n_window * 2 (n_window=50 from model config)
    const int64_t n_chunks   = n_frames / chunk_size;
    const int64_t n_valid_frames = hparams.gt_asr_enabled ? img.audio_n_frames() : n_frames;
    const int64_t n_tail_frames = n_valid_frames % chunk_size;
    const int64_t n_valid_tokens =
        (n_valid_frames / chunk_size) * 13 + (n_tail_frames > 0 ? (n_tail_frames + 7) / 8 : 0);

    GGML_ASSERT(n_frames % chunk_size == 0); // preprocessor should already pad the input
    GGML_ASSERT(n_valid_frames > 0 && n_valid_frames <= n_frames);
    GGML_ASSERT(inp->type == GGML_TYPE_F32);

    // View mel spectrogram as batched 100-frame chunks: [chunk_size, n_mel, 1, n_chunks]
    inp = ggml_view_4d(ctx0, inp,
        chunk_size, n_mel, 1, n_chunks,
        n_frames   * (int64_t)sizeof(float), // nb[1]: stride over mel bins
        chunk_size * (int64_t)sizeof(float), // nb[2]: stride for C=1 (unused)
        chunk_size * (int64_t)sizeof(float), // nb[3]: stride over chunks
        0);
    inp = ggml_cont(ctx0, inp);
    cb(inp, "inp_chunks", -1);

    // 3 x conv2d + gelu
    {
        // conv output [OW, OH, C_out, n_chunks]
        auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
            if (b) {
                x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
            }
            return ggml_gelu_erf(ctx0, x);
        };

        inp = conv_block(inp, model.conv2d_1_w, model.conv2d_1_b);
        inp = conv_block(inp, model.conv2d_2_w, model.conv2d_2_b);
        inp = conv_block(inp, model.conv2d_3_w, model.conv2d_3_b);
        // inp: [OW=13, OH=16, OC=480, n_chunks]
        cb(inp, "after_conv_blocks", -1);
    }

    // permute [OW=25, OH=16, OC=480, n_chunks] -> [OH=16, OC=480, OW=25, n_chunks]
    // reshape to [OH*OC=7680, OW*n_chunks]
    // feature index h+16*c = c*16+f (matches python code)
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2] * inp->ne[3]);

    // Project to d_model: [d_model, 25*n_chunks]
    inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
    if (model.conv_out_b) {
        inp = ggml_add(ctx0, inp, model.conv_out_b);
    }
    cb(inp, "after_conv_out", -1);

    const int64_t n_pos_padded = inp->ne[1];

    // Per-chunk positional embeddings: repeat pos[0:13] for each chunk
    // (position indices reset 0..12 per chunk, not sequential across chunks)
    {
        const int64_t tokens_per_chunk = n_pos_padded / n_chunks; // 13
        ggml_tensor * pos_tmp = ggml_view_2d(ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], tokens_per_chunk,
            model.position_embeddings->nb[1], 0);
        ggml_tensor * tgt = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
            model.position_embeddings->ne[0], n_pos_padded);
        inp = ggml_add(ctx0, inp, ggml_repeat(ctx0, pos_tmp, tgt));
    }

    int64_t n_pos = n_pos_padded;
    if (hparams.gt_asr_enabled) {
        GGML_ASSERT(n_valid_tokens > 0 && n_valid_tokens <= n_pos_padded);
        n_pos = n_valid_tokens;
        if (n_pos != n_pos_padded) {
            inp = ggml_cont(ctx0, ggml_view_2d(ctx0, inp, inp->ne[0], n_pos, inp->nb[1], 0));
        }
    }

    const bool gt_asr_debug = hparams.gt_asr_enabled &&
        std::getenv("MTMD_GT_ASR_DEBUG_DIR") != nullptr;
    auto keep_gt_asr_debug_output = [&](ggml_tensor * tensor) {
        if (gt_asr_debug) {
            ggml_set_output(tensor);
        }
    };
    if (gt_asr_debug) {
        ggml_tensor * debug_transformer_input = ggml_dup(ctx0, inp);
        cb(debug_transformer_input, "audio_transformer_input", -1);
        ggml_set_output(debug_transformer_input);
        ggml_build_forward_expand(gf, debug_transformer_input);
    }

    ggml_tensor * cur = build_vit(inp, n_pos,
        NORM_TYPE_NORMAL, hparams.ffn_op,
        nullptr,  // pos embd already added above
        nullptr);
    if (gt_asr_debug) {
        ggml_tensor * debug_encoder_states = ggml_dup(ctx0, cur);
        cb(debug_encoder_states, "after_transformer", -1);
        ggml_set_output(debug_encoder_states);
        ggml_build_forward_expand(gf, debug_encoder_states);
    } else {
        cb(cur, "after_transformer", -1);
    }

    ggml_tensor * encoder_states = cur;

    // MLP projector
    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF, -1);
    cb(cur, "projected", -1);
    keep_gt_asr_debug_output(cur);

    if (hparams.gt_asr_enabled) {
        const int64_t n_tokens = encoder_states->ne[1];
        const int64_t hidden_dim = hparams.gt_asr_hidden_dim;
        const int64_t head_dim = hidden_dim / hparams.gt_asr_head_count;
        const auto & layer = model.gt_asr_context;

        auto linear = [&](ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
            x = build_mm(weight, x);
            return bias ? ggml_add(ctx0, x, bias) : x;
        };

        ggml_tensor * features = ggml_reshape_2d(ctx0, encoder_states, hparams.gt_asr_encoder_dim, n_tokens);
        features = build_norm(
            features,
            model.gt_asr_input_norm_w,
            model.gt_asr_input_norm_b,
            NORM_TYPE_NORMAL,
            hparams.gt_asr_norm_eps,
            -1);

        features = ggml_cont(ctx0, ggml_transpose(ctx0, features));
        features = ggml_reshape_3d(ctx0, features, n_tokens, hparams.gt_asr_encoder_dim, 1);
        for (int i = 0; i < 2; ++i) {
            features = ggml_conv_1d(
                ctx0,
                model.gt_asr_temporal_w[i],
                features,
                1,
                hparams.gt_asr_conv_kernel_size / 2,
                1);
            features = ggml_add(
                ctx0,
                features,
                ggml_reshape_3d(ctx0, model.gt_asr_temporal_b[i], 1, hidden_dim, 1));
            features = ggml_gelu_erf(ctx0, features);
        }
        features = ggml_reshape_2d(ctx0, features, n_tokens, hidden_dim);
        features = ggml_cont(ctx0, ggml_transpose(ctx0, features));
        cb(features, "gt_asr_after_temporal", -1);
        keep_gt_asr_debug_output(features);

        ggml_tensor * residual = features;
        ggml_tensor * normalized = build_norm(
            features,
            layer.ln_1_w,
            layer.ln_1_b,
            NORM_TYPE_NORMAL,
            hparams.gt_asr_norm_eps,
            -1);
        ggml_tensor * q = linear(normalized, layer.q_w, layer.q_b);
        ggml_tensor * k = linear(normalized, layer.k_w, layer.k_b);
        ggml_tensor * v = linear(normalized, layer.v_w, layer.v_b);
        q = ggml_reshape_4d(ctx0, q, head_dim, hparams.gt_asr_head_count, n_tokens, 1);
        k = ggml_reshape_4d(ctx0, k, head_dim, hparams.gt_asr_head_count, n_tokens, 1);
        v = ggml_reshape_4d(ctx0, v, head_dim, hparams.gt_asr_head_count, n_tokens, 1);
        features = build_attn(
            layer.o_w,
            layer.o_b,
            q,
            k,
            v,
            nullptr,
            1.0f / std::sqrt((float) head_dim),
            -1);
        features = ggml_add(ctx0, features, residual);

        residual = features;
        features = build_norm(
            features,
            layer.ln_2_w,
            layer.ln_2_b,
            NORM_TYPE_NORMAL,
            hparams.gt_asr_norm_eps,
            -1);
        features = build_ffn(
            features,
            layer.ff_up_w,
            layer.ff_up_b,
            nullptr,
            nullptr,
            layer.ff_down_w,
            layer.ff_down_b,
            FFN_GELU_ERF,
            -1);
        features = ggml_add(ctx0, features, residual);
        features = build_norm(
            features,
            model.gt_asr_output_norm_w,
            model.gt_asr_output_norm_b,
            NORM_TYPE_NORMAL,
            hparams.gt_asr_norm_eps,
            -1);
        cb(features, "gt_asr_features", -1);
        keep_gt_asr_debug_output(features);

        ggml_tensor * frame_confidence = linear(
            features,
            model.gt_asr_frame_confidence_w,
            model.gt_asr_frame_confidence_b);
        frame_confidence = ggml_sigmoid(ctx0, frame_confidence);
        cb(frame_confidence, "gt_asr_frame_confidence", -1);
        keep_gt_asr_debug_output(frame_confidence);

        ggml_tensor * pooled = ggml_cont(ctx0, ggml_transpose(ctx0, features));
        pooled = ggml_mean(ctx0, pooled);
        pooled = ggml_cont(ctx0, ggml_transpose(ctx0, pooled));
        ggml_tensor * global_uncertainty = linear(
            pooled,
            model.gt_asr_global_uncertainty_w,
            model.gt_asr_global_uncertainty_b);
        global_uncertainty = ggml_sigmoid(ctx0, global_uncertainty);
        cb(global_uncertainty, "gt_asr_global_uncertainty", -1);
        keep_gt_asr_debug_output(global_uncertainty);

        ggml_tensor * frame_evidence = linear(
            features,
            model.gt_asr_frame_evidence_w[0],
            model.gt_asr_frame_evidence_b[0]);
        frame_evidence = ggml_gelu_erf(ctx0, frame_evidence);
        frame_evidence = linear(
            frame_evidence,
            model.gt_asr_frame_evidence_w[1],
            model.gt_asr_frame_evidence_b[1]);
        cb(frame_evidence, "gt_asr_frame_evidence", -1);
        keep_gt_asr_debug_output(frame_evidence);
        frame_evidence = ggml_cont(ctx0, ggml_transpose(ctx0, frame_evidence));
        ggml_tensor * speech_probability = ggml_sigmoid(ctx0, ggml_mean(ctx0, frame_evidence));
        ggml_set_name(speech_probability, "gt_asr_speech_probability");
        ggml_set_output(speech_probability);
        ggml_build_forward_expand(gf, speech_probability);

        auto global_projection = [&](ggml_tensor * uncertainty) {
            ggml_tensor * projected = linear(
                uncertainty,
                model.gt_asr_global_projection_w[0],
                model.gt_asr_global_projection_b[0]);
            projected = ggml_gelu_erf(ctx0, projected);
            return linear(
                projected,
                model.gt_asr_global_projection_w[1],
                model.gt_asr_global_projection_b[1]);
        };
        ggml_tensor * global_delta = global_projection(global_uncertainty);
        ggml_tensor * global_zero = global_projection(ggml_scale(ctx0, global_uncertainty, 0.0f));
        global_delta = ggml_sub(ctx0, global_delta, global_zero);
        ggml_tensor * global_scale = ggml_scale(
            ctx0,
            ggml_tanh(ctx0, model.gt_asr_global_scale_raw),
            hparams.gt_asr_max_residual_scale);
        global_delta = ggml_mul(ctx0, global_delta, global_scale);
        cb(global_delta, "gt_asr_global_delta", -1);
        keep_gt_asr_debug_output(global_delta);
        global_delta = ggml_repeat(ctx0, global_delta, cur);

        ggml_tensor * local_gate = ggml_scale(ctx0, frame_confidence, -1.0f);
        ggml_tensor * local_gate_one = ggml_fill(
            ctx0,
            ggml_dup(ctx0, frame_confidence),
            1.0f);
        local_gate = ggml_add(ctx0, local_gate, local_gate_one);
        ggml_tensor * local_delta = linear(
            features,
            model.gt_asr_local_projection_w,
            model.gt_asr_local_projection_b);
        local_gate = ggml_repeat(ctx0, local_gate, local_delta);
        local_delta = ggml_mul(ctx0, local_delta, local_gate);
        ggml_tensor * local_scale = ggml_scale(
            ctx0,
            ggml_tanh(ctx0, model.gt_asr_local_scale_raw),
            hparams.gt_asr_max_residual_scale);
        local_delta = ggml_mul(ctx0, local_delta, local_scale);
        cb(local_delta, "gt_asr_local_delta", -1);
        keep_gt_asr_debug_output(local_delta);

        cur = ggml_add(ctx0, cur, ggml_add(ctx0, global_delta, local_delta));
        cb(cur, "gt_asr_fused", -1);
        keep_gt_asr_debug_output(cur);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
