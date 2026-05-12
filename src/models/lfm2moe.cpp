#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid.h"

void llama_model_lfm2moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
    }

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_8B_A1B;  break;
        case 40: type = LLM_TYPE_24B_A2B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_lfm2moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM_LFM2, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool is_moe_layer = i >= static_cast<int>(hparams.n_layer_dense_lead);

        // ffn/moe is same for transformer and conv layers
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (is_moe_layer) {
            GGML_ASSERT(n_expert && n_expert_used);
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),  {n_embd, n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, hparams.n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp,   n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),   {n_embd, hparams.n_ff_exp, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        } else {  // dense
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        }

        // for operator_norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_recurrent(i)) {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
            GGML_ASSERT(n_embd_v_gqa == n_embd_k_gqa);

            create_tensor_qkv(layer, i, n_embd, n_embd, hparams.n_embd_k_gqa(i), hparams.n_embd_v_gqa(i), 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        } else {
            layer.shortconv.conv     = create_tensor(tn(LLM_TENSOR_SHORTCONV_CONV,    "weight", i), {hparams.n_shortconv_l_cache, n_embd}, 0);
            layer.shortconv.in_proj  = create_tensor(tn(LLM_TENSOR_SHORTCONV_INPROJ,  "weight", i), {n_embd, 3 * n_embd}, 0);
            layer.shortconv.out_proj = create_tensor(tn(LLM_TENSOR_SHORTCONV_OUTPROJ, "weight", i), {n_embd, n_embd}, 0);
        }
    }

    // for LFM2-ColBert-350M
    dense_2_out_layers   = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.n_embd_out()}, TENSOR_NOT_REQUIRED);
    dense_2_out_layers_b = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "bias"),   {hparams.n_embd_out()        }, TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_lfm2moe::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

