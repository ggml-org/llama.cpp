#include "models.h"

void llama_model_gemma4_mtp::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_embd_assist = 0;
    ml.get_key(LLM_KV_EMBEDDING_LENGTH, n_embd_assist);

    uint32_t n_embd_backbone = n_embd_assist;
    ml.get_key("gemma4.assistant.backbone_hidden_size", n_embd_backbone, false);

    hparams.n_embd = n_embd_backbone;
    hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = false;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_gemma4_mtp::load_arch_tensors(llama_model_loader & ml) {
    uint32_t n_embd_assist = 0;
    ml.get_key(LLM_KV_EMBEDDING_LENGTH, n_embd_assist);

    const int64_t n_assist = n_embd_assist;
    const int64_t n_backbone = hparams.n_embd;
    const int64_t n_ff = hparams.n_ff();
    const int64_t n_vocab = vocab.n_tokens();

    tok_embd     = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), { n_assist, n_vocab }, 0);
    output_norm  = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_assist },          0);
    mtp_pre_proj = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,  "weight"), { 2 * n_backbone, n_assist }, 0);
    mtp_post_proj = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ, "weight"), { n_assist, n_backbone },     0);

    for (int i = 0; i < (int) hparams.n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_attn_out = i + 1 == (int) hparams.n_layer ? 8 * n_assist : 4 * n_assist;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_assist }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_assist }, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), { n_assist }, 0);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_assist }, 0);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, i),           { 1 },       0);

        layer.wq          = create_tensor(tn(LLM_TENSOR_ATTN_Q,      "weight", i), { n_assist, n_attn_out }, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", i), { n_attn_out, n_assist }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_attn_out / 4 },       0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_assist, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_assist }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_assist, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_mtp::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_backbone = hparams.n_embd;
    auto inp = std::make_unique<llm_graph_input_embd>(n_backbone);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_backbone, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "gemma4_mtp_h_input");

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "gemma4_mtp_tok_embd", -1);

    res->add_input(std::move(inp));

    ggml_tensor * concat = ggml_concat(ctx0, h_input, h_input, /*dim=*/ 0);
    cb(concat, "gemma4_mtp_backbone_concat", -1);

    ggml_tensor * cur = build_lora_mm(model.mtp_pre_proj, concat);
    cb(cur, "gemma4_mtp_pre_projection", -1);

    cur = ggml_add(ctx0, cur, tok_embd);
    cb(cur, "gemma4_mtp_input", -1);

    for (int il = 0; il < (int) hparams.n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = cur;
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_attn_norm", il);

        cur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(cur, "gemma4_mtp_attn_q_proj", il);

        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "gemma4_mtp_attn_out", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "gemma4_mtp_attn_residual", il);

        residual = cur;
        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_attn_post_norm", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "gemma4_mtp_ffn_out", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "gemma4_mtp_ffn_residual", il);

        cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_ffn_post_norm", il);
    }

    ggml_tensor * mtp_out = build_lora_mm(model.mtp_post_proj, cur);
    cb(mtp_out, "gemma4_mtp_post_projection", -1);
    res->t_mtp_out = mtp_out;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "gemma4_mtp_output_norm", -1);

    cur = build_lora_mm(model.tok_embd, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
