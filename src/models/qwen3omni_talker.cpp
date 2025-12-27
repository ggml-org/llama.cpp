#include "models.h"

// SnakeBeta activation: f(x) = x + (1/β) * sin²(x * α)
// Used by HiFi-GAN decoder in Code2Wav vocoder
// alpha and beta are per-channel learnable parameters stored in LOG-SPACE
// Must apply exp() before using them (matching HuggingFace transformers)
static ggml_tensor * build_snake_activation(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * alpha,
        ggml_tensor * beta) {
    // CRITICAL: Apply exp() to stored parameters (they're in log-space)
    // Reference: transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:3676-3680
    ggml_tensor * exp_alpha = ggml_exp(ctx, alpha);
    ggml_tensor * exp_beta = beta ? ggml_exp(ctx, beta) : nullptr;

    // sin(x * exp(alpha))
    ggml_tensor * ax = ggml_mul(ctx, x, exp_alpha);
    ggml_tensor * sin_ax = ggml_sin(ctx, ax);

    // sin²(x * exp(alpha))
    ggml_tensor * sin2 = ggml_sqr(ctx, sin_ax);

    // (1/(exp(beta) + eps)) * sin²(x * exp(alpha))
    // Divide by exp(beta) + epsilon to prevent division by zero
    // Reference: self.no_div_by_zero = 0.000000001 in HuggingFace
    ggml_tensor * scaled;
    if (exp_beta) {
        // Add small epsilon to prevent division by zero
        ggml_tensor * eps = ggml_new_f32(ctx, 1e-9f);
        ggml_tensor * beta_safe = ggml_add(ctx, exp_beta, eps);
        scaled = ggml_div(ctx, sin2, beta_safe);
    } else {
        // Fallback if no beta provided (shouldn't happen for SnakeBeta)
        scaled = ggml_div(ctx, sin2, exp_alpha);
    }

    // x + (1/β) * sin²(x * α)
    return ggml_add(ctx, x, scaled);
}

// Qwen3-Omni Talker model builder
// This is a 20-layer MoE transformer with shared experts for text-to-speech
// It generates codec tokens that are then processed by Code2Wav for audio synthesis

llm_build_qwen3omni_talker::llm_build_qwen3omni_talker(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
        cb(moe_out, "ffn_moe_out", il);

        // Shared expert (if present)
        if (model.layers[il].ffn_gate_shexp) {
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // Apply sigmoid gating: gate = sigmoid(ffn_gate_inp_shexp @ cur)
            // Reference: HF Qwen3OmniMoeTalkerTextSparseMoeBlock.forward
            if (model.layers[il].ffn_gate_inp_shexp) {
                ggml_tensor * shexp_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
                cb(shexp_gate, "ffn_shexp_gate_inp", il);
                shexp_gate = ggml_sigmoid(ctx0, shexp_gate);
                cb(shexp_gate, "ffn_shexp_gate", il);
                ffn_shexp = ggml_mul(ctx0, ffn_shexp, shexp_gate);
                cb(ffn_shexp, "ffn_shexp_gated", il);
            }

            moe_out = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(moe_out, "ffn_moe_shexp_out", il);
        }

        cur = ggml_add(ctx0, moe_out, ffn_inp);
        // NOTE: build_cvec returns cur unchanged if no cvec, so l_out == residual output
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // Fix: Expose pre-norm hidden state for Code Predictor
    // HuggingFace uses hidden_states[0][-1] which is BEFORE output_norm
    // Mark this tensor as output so it can be extracted
    cb(cur, "pre_norm_hidden", -1);
    ggml_set_output(cur);

    // CRITICAL FIX: Filter to only output tokens marked for output
    // This is required for llama_get_logits to work correctly
    // Other models (dream, falcon-h1, smallthinker, etc.) all do this
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        cb(cur, "filtered_output", -1);
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head (codec_head)
    // Note: The Talker uses talker_codec_head, not the standard output tensor
    cur = build_lora_mm(model.talker_codec_head, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Code Predictor: 5-layer transformer for generating codebook tokens 2-16
// Architecture: n_embd=1024, n_head=16, n_head_kv=8, head_dim=128, n_ff=3072
llm_build_qwen3omni_code_predictor::llm_build_qwen3omni_code_predictor(
        const llama_model & model,
        const llm_graph_params & params,
        int codebook_idx)
    : llm_graph_context(params), m_codebook_idx(codebook_idx) {

    // Code Predictor hyperparameters (fixed, not from hparams)
    const int64_t cp_n_head = 16;
    const int64_t cp_n_head_kv = 8;
    const int64_t cp_head_dim = 128;
    const int64_t cp_n_layer = 5;

    GGML_ASSERT(codebook_idx >= 0 && codebook_idx < 15);
    GGML_ASSERT(!model.talker_cp_layers.empty());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Input: embed the codec token using the appropriate embedding
    // codebook_idx 0-14 corresponds to predicting codebooks 2-16
    // We use cp_codec_embd[codebook_idx] for the input embedding
    inpL = build_inp_embd(model.talker_cp_codec_embd[m_codebook_idx]);

    // Position input
    ggml_tensor * inp_pos = build_inp_pos();

    // Attention input (KV cache)
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // 5-layer transformer
    for (int il = 0; il < cp_n_layer; ++il) {
        const auto & layer = model.talker_cp_layers[il];
        ggml_tensor * inpSA = inpL;

        // Input LayerNorm (RMSNorm)
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "cp_attn_norm", il);

        // Self-attention with QK normalization
        {
            // Q/K/V projections
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(Qcur, "cp_Qcur", il);

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, cur);
            cb(Kcur, "cp_Kcur", il);

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);
            cb(Vcur, "cp_Vcur", il);

            // Reshape for multi-head attention
            Qcur = ggml_reshape_3d(ctx0, Qcur, cp_head_dim, cp_n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, cp_head_dim, cp_n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, cp_head_dim, cp_n_head_kv, n_tokens);

            // QK normalization (RMSNorm per head)
            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "cp_Qcur_normed", il);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "cp_Kcur_normed", il);

            // RoPE (Code Predictor uses rope_theta=1000000)
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    cp_head_dim, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    cp_head_dim, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "cp_Qcur_rope", il);
            cb(Kcur, "cp_Kcur_rope", il);
            cb(Vcur, "cp_Vcur", il);

            // Attention
            cur = build_attn(inp_attn,
                    layer.wo, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(cp_head_dim)), il);
        }

        if (il == cp_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "cp_ffn_inp", il);

        // FFN with SwiGLU
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "cp_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up, NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "cp_ffn_out", il);

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "cp_l_out", il);

        inpL = cur;
    }

    cur = inpL;

    // Output norm
    cur = build_norm(cur, model.talker_cp_output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "cp_result_norm", -1);
    res->t_embd = cur;

    // LM head (select based on codebook index)
    cur = ggml_mul_mat(ctx0, model.talker_cp_lm_head[m_codebook_idx], cur);
    cb(cur, "cp_result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Code2Wav: HiFi-GAN vocoder for converting codec tokens to audio waveform
// Architecture:
//   1. Code embedding: sum of 16 codebook embeddings
//   2. Pre-transformer: 8 layers with sliding window attention (window=72) + LayerScale
//   3. Upsample: 2 ConvNeXt blocks (upsampling factor per block)
//   4. Decoder: HiFi-GAN with Snake activations
llm_build_qwen3omni_code2wav::llm_build_qwen3omni_code2wav(
        const llama_model & model,
        const llm_graph_params & params)
    : llm_graph_context(params) {

    // Code2Wav hyperparameters (fixed, not from hparams)
    const int64_t c2w_n_embd = 1024;
    const int     c2w_n_head = 16;
    const int     c2w_head_dim = 64;  // 1024 / 16
    const int     c2w_n_layer = 8;
    const int     c2w_n_ff = 3072;
    const int     c2w_up_n_block = 2;
    const int     c2w_sliding_window = 72;  // sliding window attention

    (void)c2w_n_embd;  // used in build_inp_embd dimensions
    (void)c2w_n_ff;    // used in build_ffn dimensions
    (void)c2w_sliding_window;  // TODO: use for attention mask

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Input: embed codec tokens using code2wav embedding
    // In inference, we'd receive 16 codec token IDs and look them up
    // For now, start with standard token embedding
    inpL = build_inp_embd(model.c2w_code_embd);

    // Position input
    ggml_tensor * inp_pos = build_inp_pos();

    // Attention input (KV cache)
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Pre-transformer: 8 layers with LayerScale
    for (int il = 0; il < c2w_n_layer; ++il) {
        GGML_ASSERT(il < (int)model.c2w_pre_layers.size());
        const auto & layer = model.c2w_pre_layers[il];
        ggml_tensor * inpSA = inpL;

        // Input LayerNorm (RMSNorm)
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "c2w_attn_norm", il);

        // Self-attention (no QK norm in Code2Wav pre-transformer)
        {
            // Q/K/V projections
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(Qcur, "c2w_Qcur", il);

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, cur);
            cb(Kcur, "c2w_Kcur", il);

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);
            cb(Vcur, "c2w_Vcur", il);

            // Reshape for multi-head attention
            Qcur = ggml_reshape_3d(ctx0, Qcur, c2w_head_dim, c2w_n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, c2w_head_dim, c2w_n_head, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, c2w_head_dim, c2w_n_head, n_tokens);

            // RoPE (Code2Wav uses rope_theta=1000000)
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    c2w_head_dim, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    c2w_head_dim, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "c2w_Qcur_rope", il);
            cb(Kcur, "c2w_Kcur_rope", il);
            cb(Vcur, "c2w_Vcur", il);

            // Attention
            cur = build_attn(inp_attn,
                    layer.wo, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(c2w_head_dim)), il);
        }

        // LayerScale for attention output
        if (layer.attn_scale) {
            cur = ggml_mul(ctx0, cur, layer.attn_scale);
            cb(cur, "c2w_attn_scale", il);
        }

        if (il == c2w_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "c2w_ffn_inp", il);

        // FFN norm
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "c2w_ffn_norm", il);

        // FFN with SwiGLU
        cur = build_ffn(cur,
                layer.ffn_up, NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "c2w_ffn_out", il);

        // LayerScale for FFN output
        if (layer.ffn_scale) {
            cur = ggml_mul(ctx0, cur, layer.ffn_scale);
            cb(cur, "c2w_ffn_scale", il);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "c2w_l_out", il);

        inpL = cur;
    }

    cur = inpL;

    // Output norm
    cur = build_norm(cur, model.c2w_pre_output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "c2w_pre_result_norm", -1);

    // Upsample: 2 ConvNeXt blocks
    // Each block: transpose_conv -> depthwise_conv -> norm -> pwconv1 -> gelu -> pwconv2 -> gamma
    for (int ib = 0; ib < c2w_up_n_block; ++ib) {
        GGML_ASSERT(ib < (int)model.c2w_up_blocks.size());
        const auto & block = model.c2w_up_blocks[ib];

        // Transpose convolution for 2x upsampling
        // Kernel shape: {kernel_size=2, n_embd, n_embd}
        // stride=2, padding=0, dilation=1
        if (block.conv) {
            cur = ggml_conv_transpose_1d(ctx0, block.conv, cur, 2, 0, 1);
            cb(cur, "c2w_up_conv", ib);
        }

        // Depthwise convolution with causal padding
        // Kernel shape: {kernel_size=7, 1, n_embd}
        // Using dw_ph for same-length output with symmetric padding
        if (block.dwconv) {
            cur = ggml_conv_1d_dw_ph(ctx0, block.dwconv, cur, 1, 1);
            cb(cur, "c2w_up_dwconv", ib);
        }

        // Layer normalization
        if (block.norm) {
            // Note: This is LayerNorm, not RMSNorm
            cur = build_norm(cur, block.norm, NULL, LLM_NORM, ib);
            cb(cur, "c2w_up_norm", ib);
        }

        // Pointwise conv 1 (expand)
        if (block.pwconv1) {
            cur = ggml_mul_mat(ctx0, block.pwconv1, cur);
            cur = ggml_gelu(ctx0, cur);
            cb(cur, "c2w_up_pwconv1", ib);
        }

        // Pointwise conv 2 (contract)
        if (block.pwconv2) {
            cur = ggml_mul_mat(ctx0, block.pwconv2, cur);
            cb(cur, "c2w_up_pwconv2", ib);
        }

        // Layer scale (gamma)
        if (block.gamma) {
            cur = ggml_mul(ctx0, cur, block.gamma);
            cb(cur, "c2w_up_gamma", ib);
        }
    }

    cb(cur, "c2w_upsample_out", -1);

    // HiFi-GAN decoder with Snake activations
    // Converts pre-transformer output to audio waveform
    // Architecture: conv_in -> 4 stages (snake + upsample + 3 resblocks) -> snake + conv_out -> tanh

    // Upsample rates for each stage: [8, 5, 4, 3] = 480× total
    const int upsample_rates[] = {8, 5, 4, 3};
    const int c2w_dec_n_stage = 4;
    const int c2w_dec_n_resblk = 3;

    // 1. Initial conv: 1024 -> decoder_dim (1536)
    if (model.c2w_dec_conv_in) {
        cur = ggml_conv_1d_ph(ctx0, model.c2w_dec_conv_in, cur, 1, 1);
        cb(cur, "c2w_dec_conv_in", -1);
    }

    // 2. Four upsample stages
    for (int stage = 0; stage < c2w_dec_n_stage && stage < (int)model.c2w_dec_blocks.size(); ++stage) {
        const auto & dec_block = model.c2w_dec_blocks[stage];

        // 2a. Outer Snake activation
        if (dec_block.snake_alpha) {
            cur = build_snake_activation(ctx0, cur, dec_block.snake_alpha, dec_block.snake_beta);
            cb(cur, "c2w_dec_snake", stage);
        }

        // 2b. Transpose conv for upsampling
        if (dec_block.upsample) {
            cur = ggml_conv_transpose_1d(ctx0, dec_block.upsample, cur, upsample_rates[stage], 0, 1);
            // Add bias - reshape to broadcast over sequence dimension
            if (dec_block.upsample_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx0, dec_block.upsample_bias,
                                                         dec_block.upsample_bias->ne[0], 1);
                cur = ggml_add(ctx0, cur, bias_2d);
            }
            cb(cur, "c2w_dec_upsample", stage);
        }

        // 2c. Three residual blocks per stage
        for (int rb = 0; rb < c2w_dec_n_resblk; ++rb) {
            int flat_idx = stage * c2w_dec_n_resblk + rb;
            if (flat_idx >= (int)model.c2w_dec_res_blks.size()) break;

            const auto & res_blk = model.c2w_dec_res_blks[flat_idx];
            ggml_tensor * residual = cur;

            // ResBlock: Snake1 -> Conv1 -> Snake2 -> Conv2 -> + residual
            // First activation
            if (res_blk.act1_alpha) {
                cur = build_snake_activation(ctx0, cur, res_blk.act1_alpha, res_blk.act1_beta);
            }

            // First conv
            if (res_blk.conv1) {
                cur = ggml_conv_1d_ph(ctx0, res_blk.conv1, cur, 1, 1);
            } else if (res_blk.conv) {
                cur = ggml_conv_1d_ph(ctx0, res_blk.conv, cur, 1, 1);
            }

            // Second activation
            if (res_blk.act2_alpha) {
                cur = build_snake_activation(ctx0, cur, res_blk.act2_alpha, res_blk.act2_beta);
            }

            // Second conv
            if (res_blk.conv2) {
                cur = ggml_conv_1d_ph(ctx0, res_blk.conv2, cur, 1, 1);
            }

            // Residual connection
            cur = ggml_add(ctx0, cur, residual);
            cb(cur, "c2w_dec_resblk", flat_idx);
        }
    }

    // 3. Final Snake activation (uses last stage's snake params or separate final snake)
    // Note: Some HiFi-GAN variants have a separate final snake - if not present, skip
    // For Qwen3-Omni, we'll check if decoder.5 exists (final snake before conv_out)
    // Since we don't have it mapped yet, we'll skip this step

    // 4. Final conv: decoder_dim -> 1 (mono audio)
    if (model.c2w_dec_conv_out) {
        cur = ggml_conv_1d_ph(ctx0, model.c2w_dec_conv_out, cur, 1, 1);
        cb(cur, "c2w_dec_conv_out", -1);
    }

    // 5. Tanh to constrain output to [-1, 1]
    cur = ggml_tanh(ctx0, cur);
    cb(cur, "c2w_audio_out", -1);

    res->t_embd = cur;
    // No logits for vocoder - output is audio samples
    res->t_logits = nullptr;

    ggml_build_forward_expand(gf, cur);
}
