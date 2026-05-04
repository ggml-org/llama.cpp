#include "models.h"

void llama_model_rwkv6qwen2::load_arch_hparams(llama_model_loader & ml) {
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

void llama_model_rwkv6qwen2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, TENSOR_NOT_REQUIRED);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    const int time_mix_extra_dim = hparams.time_mix_extra_dim;
    const int time_decay_extra_dim = hparams.time_decay_extra_dim;
    const int head_size = hparams.wkv_head_size;
    const int attn_hidden_size = n_embd;
    int attn_key_value_size;
    if (n_head_kv == 0 || attn_hidden_size / head_size == n_head_kv) {
        attn_key_value_size = attn_hidden_size;
    } else {
        attn_key_value_size = n_head_kv * head_size;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, time_mix_extra_dim * 5}, 0);
        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {time_mix_extra_dim, n_embd, 5}, 0);

        layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), {n_embd, 1, 1}, 0);
        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, 0);

        layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), {head_size, n_embd / head_size}, TENSOR_NOT_REQUIRED);
        layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), {n_embd}, 0);
        layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), {n_embd, time_decay_extra_dim}, 0);
        layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), {time_decay_extra_dim, attn_hidden_size}, 0);
        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {n_embd, attn_key_value_size}, 0);
        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {n_embd, attn_key_value_size}, 0);
        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);
        layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), {attn_hidden_size, n_embd}, 0);
        // optional bias tensors
        layer.time_mix_key_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "bias", i), {attn_key_value_size}, TENSOR_NOT_REQUIRED);
        layer.time_mix_value_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "bias", i), {attn_key_value_size}, TENSOR_NOT_REQUIRED);
        layer.time_mix_receptance_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "bias", i), {attn_hidden_size}, TENSOR_NOT_REQUIRED);

        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_rwkv6qwen2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_rwkv6qwen2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv6_base(model, params) {
    GGML_ASSERT(n_embd == hparams.n_embd_r());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    const auto n_embd = hparams.n_embd;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_seqs = ubatch.n_seqs;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer * layer = &model.layers[il];
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

        ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM_RMS, il);
        cb(att_norm, "attn_norm", il);

        ggml_tensor * x_prev = ggml_concat(
                ctx0,
                token_shift,
                ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                1
                );

        cur = build_rwkv6_time_mix(rs_inp, att_norm, x_prev, ubatch, il);

        token_shift = ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm));
        ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur     = ggml_reshape_2d(ctx0, cur,     n_embd, n_tokens);
        ffn_inp = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);

        if (il == n_layer - 1 && inp_out_ids) {
            cur     = ggml_get_rows(ctx0, cur,     inp_out_ids);
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
