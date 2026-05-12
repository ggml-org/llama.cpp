#include "models.h"

void llama_model_jina_bert_v2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
    hparams.f_max_alibi_bias = 8.0f;

    switch (hparams.n_layer) {
        case 4:  type = LLM_TYPE_33M;  break; // jina-embeddings-small
        case 12: type = LLM_TYPE_137M; break; // jina-embeddings-base
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_jina_bert_v2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd  = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0); // word_embeddings
    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0); // token_type_embeddings

    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0); // LayerNorm
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0); // LayerNorm bias

    cls   = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, 1}, TENSOR_NOT_REQUIRED);
    cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"),   {1},         TENSOR_NOT_REQUIRED);
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i]; // JinaBertLayer

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);

        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0); //output_dens
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0); //output_dens

        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0); //output_norm
        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd}, 0);

        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);

        const auto tn_ffn_up_weight = tn(LLM_TENSOR_FFN_UP, "weight", i);
        ggml_tensor * t_ffn_up = ml.get_tensor_meta(tn_ffn_up_weight.str().c_str());
        const int64_t n_ffn_up = t_ffn_up ? t_ffn_up->ne[1] : n_ff;

        GGML_ASSERT(n_ffn_up == n_ff || n_ffn_up == n_ff * 2);
        layer.ffn_up   = create_tensor(tn_ffn_up_weight, {n_embd, n_ffn_up}, 0);
        layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), {n_ffn_up}, TENSOR_NOT_REQUIRED);

        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_jina_bert_v2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

