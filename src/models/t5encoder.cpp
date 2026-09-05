#include "models.h"

void llama_model_t5encoder::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_t5encoder::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm_enc  = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd}, 0);
        layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

        layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_t5encoder::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
