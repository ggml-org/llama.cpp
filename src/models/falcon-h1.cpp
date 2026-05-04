#include "models.h"

void llama_model_falcon_h1::load_arch_hparams(llama_model_loader & ml) {
    // Common parameters
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // SSM parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    std::fill(hparams.recurrent_layer_arr.begin(), hparams.recurrent_layer_arr.end(), true);

    switch (hparams.n_layer) {
        case 36:
            type = LLM_TYPE_0_5B; break;
        case 24:
            type = LLM_TYPE_1_5B; break;
        case 66:
            type = LLM_TYPE_1B; break;
        case 32:
            type = LLM_TYPE_3B; break;
        case 44:
            type = LLM_TYPE_7B; break;
        case 72:
            type = LLM_TYPE_34B; break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_falcon_h1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // Common
    const int64_t hidden_size = hparams.n_embd; // hidden_size

    // mamba2 Mixer SSM params
    const int64_t ssm_conv_kernel_size  = hparams.ssm_d_conv; // ssm_conv_kernel_size
    const int64_t ssm_n_groups          = hparams.ssm_n_group; // ssm_n_groups
    const int64_t ssm_state_size        = hparams.ssm_d_state; // ssm_state_size
    const int64_t ssm_intermediate_size = hparams.ssm_d_inner; // TODO expand
    const int64_t ssm_num_heads         = hparams.ssm_dt_rank; // ssm_num_heads
    const int64_t ssm_conv_dim          = ssm_intermediate_size + 2 * ssm_n_groups * ssm_state_size;
    const int64_t ssm_projection_size   = ssm_intermediate_size + ssm_conv_dim + ssm_num_heads;

    // attn params
    const int64_t attn_num_attention_head = hparams.n_head(0); // rename to: attn_num_attention_head
    const int64_t attn_num_key_value_head = hparams.n_head_kv(0);

    // ffn params
    const int64_t ffn_intermediate_size = hparams.n_ff(0);

    // embeddings
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hidden_size, n_vocab}, 0);

    // output
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {hidden_size, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {hidden_size}, 0);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hidden_size, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        /*SSM LAYERS*/
        // ssm in
        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {hidden_size, ssm_projection_size}, 0);
        // ssm 1d conv
        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {ssm_conv_kernel_size, ssm_conv_dim}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {ssm_conv_dim}, TENSOR_NOT_REQUIRED);
        // ssm_dt
        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {ssm_num_heads}, 0);
        // no "weight" suffix for these
        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, ssm_num_heads}, 0);
        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, ssm_num_heads}, 0);
        // ssm_norm
        layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {ssm_intermediate_size / ssm_n_groups, ssm_n_groups}, TENSOR_NOT_REQUIRED);
        // out_proj
        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {ssm_intermediate_size, hidden_size}, 0);

        /*ATTENTION LAYERS*/
        // attention layers (with optional bias)
        create_tensor_qkv(layer, i, hidden_size, n_embd_head_k * attn_num_attention_head, attn_num_key_value_head * n_embd_head_k, attn_num_key_value_head * n_embd_head_v, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * attn_num_attention_head, hidden_size}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {hidden_size}, TENSOR_NOT_REQUIRED);
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {hidden_size}, 0);


        // feed forward (w/ optional biases)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, i), {hidden_size}, 0);
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {hidden_size,   ffn_intermediate_size}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  ffn_intermediate_size, hidden_size}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {hidden_size,   ffn_intermediate_size}, 0);

        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {ffn_intermediate_size}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {hidden_size}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {ffn_intermediate_size}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_falcon_h1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_falcon_h1::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // Build the inputs in the recurrent & kv cache
    auto * inp = build_inp_mem_hybrid();

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, hparams.rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, hparams.rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        cb(Qcur, "Qcur-post-rope", il);
        cb(Kcur, "Kcur-post-rope", il);
        cb(Vcur, "Vcur-post-rope", il);

        ggml_tensor * attn_out = build_attn(inp->get_attn(),
                                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(attn_out, "attn_out", il);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        // Mamba2 layer
        cb(cur, "ssm_in", il);

        ggml_tensor * ssm_out = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        cb(ssm_out, "ssm_out", il);

        // // Aggregation
        cur   = ggml_add(ctx0, attn_out, ssm_out);
        inpSA = ggml_add(ctx0, cur, inpSA);
        cb(cur, "layer_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = inpSA;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, inpSA);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
