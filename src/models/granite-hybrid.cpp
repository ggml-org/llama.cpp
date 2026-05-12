#include "models.h"

void llama_model_granite_hybrid::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, /* required */ false);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale, /* required */ false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, /* required */ false);
    ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale, /* required */ false);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Granite uses rope_finetuned as a switch for rope, so default to true
    bool rope_finetuned = true;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    // A layer is recurrent IFF the n_head_kv value is set to 0
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_embd) {
        case 768: type = LLM_TYPE_350M; break;
        case 1536: type = (hparams.n_ff() == 512 ? LLM_TYPE_7B_A1B : LLM_TYPE_1B); break;
        case 2048: case 2560: type = LLM_TYPE_3B; break;
        case 4096: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // For Granite MoE Shared
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, /* required */ false);
}

void llama_model_granite_hybrid::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // mamba2 Mixer SSM params
    // NOTE: int64_t for tensor dimensions
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t d_state    = hparams.ssm_d_state;
    const int64_t n_ssm_head = hparams.ssm_dt_rank;
    const int64_t n_group    = hparams.ssm_n_group;
    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;

    // only an expansion factor of 2 is supported for now
    GGML_ASSERT(2 * n_embd == d_inner);

    // embeddings
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed, duplicated to allow offloading
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.is_recurrent(i)) {
            // ssm layers
            layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner + 2*n_group*d_state}, TENSOR_NOT_REQUIRED);

            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_ssm_head}, 0);

            // no "weight" suffix for these
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_ssm_head}, 0);
            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_ssm_head}, 0);

            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);

            // out_proj
            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            // attention layers (with optional bias)
            const int64_t n_head_i = hparams.n_head(i);
            const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
            const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
            layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        }

        // feed forward (w/ optional biases)
        if (n_expert > 0) {
            // MoE FFN
            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
        } else {
            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_granite_hybrid::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_granite_hybrid::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Positional embeddings populated if rope enabled
    ggml_tensor * inp_pos = nullptr;
    if (hparams.rope_finetuned) {
        inp_pos = build_inp_pos();
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            // ssm layer //
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else {
            // attention layer //
            cur = build_attention_layer(cur, inp_pos, inp->get_attn(), model, n_embd_head, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // ffn
        cur = build_layer_ffn(cur, inpSA, model, il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    // For Granite architectures - scale logits
    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_granite_hybrid::graph::build_attention_layer(ggml_tensor *             cur,
                                                              ggml_tensor *             inp_pos,
                                                              llm_graph_input_attn_kv * inp_attn,
                                                              const llama_model &       model,
                                                              const int64_t             n_embd_head,
                                                              const int                 il) {
    auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, hparams.n_head(il), hparams.n_head_kv(il), il);

    const bool use_rope = hparams.rope_finetuned;
    if (use_rope) {
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llama_model_granite_hybrid::graph::build_layer_ffn(ggml_tensor *       cur,
                                                        ggml_tensor *       inpSA,
                                                        const llama_model & model,
                                                        const int           il) {
    // For Granite architectures - scale residual
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    // feed-forward network (non-MoE)
    if (model.layers[il].ffn_gate_inp == nullptr) {
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

    } else {
        // MoE branch
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
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
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(moe_out, "ffn_moe_out", il);

        // For Granite MoE Shared
        if (hparams.n_ff_shexp > 0) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    }

    // For Granite architectures - scale residual
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "ffn_out", il);

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}
