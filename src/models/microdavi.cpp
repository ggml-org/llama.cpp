#include "models.h"

// MicroDavi: 24-layer hybrid Mamba-2 / Transformer.
// Attention layers at indices 5, 11, 17, 23 (n_head_kv(il) != 0), full causal MHA with RoPE.
// Mamba2 layers everywhere else (n_head_kv(il) == 0) -- NO ffn sublayer on these,
// unlike Jamba's Mamba layers which DO carry one (Mamba2Block.forward is x + mamba(norm(x)), full stop).
// LM head is TIED to tok_embd -- checkpoint has no separate output.weight tensor at all,
// matches Jamba's TENSOR_NOT_REQUIRED + TENSOR_DUPLICATED fallback pattern below.
// RoPE: GGML_ROPE_TYPE_NORMAL (interleaved pairs) -- confirmed against training code's
// apply_rope (torch.view_as_complex on adjacent pairs), NOT neox/split-half.

void llama_model_microdavi::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE,              hparams.rope_freq_base_train, false);

    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    hparams.rope_type = LLAMA_ROPE_TYPE_NORM;

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_UNKNOWN; break; // 24-layer microdavi, no named size bucket yet
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_microdavi::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv    = hparams.ssm_d_conv;     // 4
    const int64_t d_inner   = hparams.ssm_d_inner;    // 640
    const int64_t d_state   = hparams.ssm_d_state;    // 64
    const int64_t n_group   = hparams.ssm_n_group;    // 1
    const int64_t n_ssm_head= hparams.ssm_dt_rank;    // 10 -- reusing dt_rank as the Mamba2 head count, confirmed earlier add_ssm_time_step_rank(10) == n_heads_mamba in the converter

    // Mamba2 in_proj output split: z(d_inner) + x(d_inner) + B,C(2*n_group*d_state) + dt(n_ssm_head)
    const int64_t d_in_proj  = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;
    const int64_t d_conv_dim = d_inner + 2 * n_group * d_state; // conv applied to x,B,C concat

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output -- tied to tok_embd, no separate tensor exists in the checkpoint
    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        const int64_t n_head_kv  = hparams.n_head_kv(i);
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa(i);

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (n_head_kv == 0) {
            // Mamba2 layer
            layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

            layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_conv_dim}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), {d_conv_dim}, 0);

            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_ssm_head}, 0);

            // no "weight" suffix for these, matching Jamba's convention
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_ssm_head}, 0);
            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_ssm_head}, 0);

            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);

            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);

            // NOTE: no ffn_norm / ffn_gate / ffn_up / ffn_down created for Mamba2 layers --
            // Mamba2Block has no FFN sublayer at all, unlike Jamba's Mamba layers.
        } else {
            // Attention layer
            create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_microdavi::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_microdavi::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v(); // 40, used only on attention layers

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp_hybrid = build_inp_mem_hybrid();
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head_kv = hparams.n_head_kv(il);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (n_head_kv == 0) {
            // ---- Mamba2 layer ----
            cur = build_mamba2_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "l_out", il);

            cur = build_cvec(cur, il);
            inpL = cur;
            continue; // no FFN sublayer at all on Mamba2 layers -- skip the block below
        }

        // ---- Attention layer ----
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_embd_head, GGML_ROPE_TYPE_NORMAL,
                              n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_embd_head, GGML_ROPE_TYPE_NORMAL,
                              n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        cur = build_attn(inp_hybrid->get_attn(),
                model.layers[il].wo, NULL, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, NULL, NULL, NULL, 1.0f/sqrtf(float(n_embd_head)), il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, inpL, cur);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_inp, cur);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}