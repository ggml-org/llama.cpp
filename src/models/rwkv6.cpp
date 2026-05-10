#include "models.h"

void llama_model_rwkv6::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    ml.get_key(LLM_KV_WKV_HEAD_SIZE,               hparams.wkv_head_size);
    ml.get_key(LLM_KV_TIME_MIX_EXTRA_DIM,          hparams.time_mix_extra_dim);
    ml.get_key(LLM_KV_TIME_DECAY_EXTRA_DIM,        hparams.time_decay_extra_dim);
    ml.get_key(LLM_KV_RESCALE_EVERY_N_LAYERS,      hparams.rescale_every_n_layers, false);
    ml.get_key(LLM_KV_TOKEN_SHIFT_COUNT,           hparams.token_shift_count, false);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1_6B; break;
        case 32:
            switch (hparams.n_embd) {
                case 2560: type = LLM_TYPE_3B; break;
                case 4096: type = LLM_TYPE_7B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 61: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_rwkv6::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // Block 0, LN0
    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    const int time_mix_extra_dim = hparams.time_mix_extra_dim;
    const int time_decay_extra_dim = hparams.time_decay_extra_dim;
    const int head_size = hparams.wkv_head_size;
    const int attn_hidden_size = n_embd;
    const int ffn_size = hparams.n_ff_arr[0];

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);
        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, 0);

        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, time_mix_extra_dim * 5}, 0);
        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {time_mix_extra_dim, n_embd, 5}, 0);

        layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), {n_embd, 1, 1}, 0);
        layer.time_mix_lerp_w = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_W, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
        layer.time_mix_lerp_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
        layer.time_mix_lerp_v = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_V, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
        layer.time_mix_lerp_r = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_R, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
        layer.time_mix_lerp_g = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_G, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, TENSOR_NOT_REQUIRED);
        GGML_ASSERT(!(layer.time_mix_lerp_fused == NULL && layer.time_mix_lerp_w == NULL));

        layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), {head_size, n_embd / head_size}, 0);
        layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), {n_embd}, 0);
        layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), {n_embd, time_decay_extra_dim}, 0);
        layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), {time_decay_extra_dim, attn_hidden_size}, 0);
        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd}, 0);
        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd}, 0);
        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);
        layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), {attn_hidden_size, n_embd}, 0);

        layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd}, 0);
        layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd}, 0);
        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

        layer.channel_mix_lerp_k = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, 0);
        layer.channel_mix_lerp_r = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_R, "weight", i), {n_embd, 1, 1}, 0);

        layer.channel_mix_key = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_KEY, "weight", i), {n_embd, ffn_size}, 0);
        layer.channel_mix_value = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_VALUE, "weight", i), {ffn_size, n_embd}, 0);
        layer.channel_mix_receptance = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_RECEPTANCE, "weight", i), {n_embd, n_embd}, 0);
    }

}

std::unique_ptr<llm_graph_context> llama_model_rwkv6::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_rwkv6::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_rwkv6_base(model, params) {
    GGML_ASSERT(hparams.token_shift_count == 2);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);

    auto * rs_inp = build_rs_inp();

    const auto n_embd       = hparams.n_embd;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_seqs       = ubatch.n_seqs;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer * layer = &model.layers[il];
        inpL                      = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

        ggml_tensor * att_shift =
            ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
        ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1],
                                               token_shift->nb[2], n_embd * ggml_element_size(token_shift));

        ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
        cb(att_norm, "attn_norm", il);

        ggml_tensor * x_prev = ggml_concat(
            ctx0, att_shift,
            ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0), 1);

        cur = build_rwkv6_time_mix(rs_inp, att_norm, x_prev, ubatch, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        ggml_tensor * ffn_norm = build_norm(ffn_inp, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
        cb(ffn_norm, "ffn_norm", il);

        x_prev = ggml_concat(
            ctx0, ffn_shift,
            ggml_view_3d(ctx0, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], 0), 1);

        token_shift = ggml_concat(ctx0,
                                  ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2],
                                               (n_seq_tokens - 1) * n_embd * ggml_element_size(att_norm)),
                                  ggml_view_3d(ctx0, ffn_norm, n_embd, 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2],
                                               (n_seq_tokens - 1) * n_embd * ggml_element_size(ffn_norm)),
                                  1);
        ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

        ffn_inp  = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);
        ffn_norm = ggml_reshape_2d(ctx0, ffn_norm, n_embd, n_tokens);
        x_prev   = ggml_reshape_2d(ctx0, x_prev, n_embd, n_tokens);
        cur      = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

        if (il == n_layer - 1 && inp_out_ids) {
            ffn_inp  = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            ffn_norm = ggml_get_rows(ctx0, ffn_norm, inp_out_ids);
            x_prev   = ggml_get_rows(ctx0, x_prev, inp_out_ids);
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        cur = build_rwkv6_channel_mix(layer, ffn_norm, x_prev, LLM_ARCH_RWKV6);
        cur = ggml_add(ctx0, cur, ffn_inp);

        if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
            cur = ggml_scale(ctx0, cur, 0.5F);
        }
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
