#include "models.h"

#include <algorithm>

void llama_model_kail::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    type = LLM_TYPE_1B;
}

void llama_model_kail::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // optional metadata for the controller/router inner dimensions;
    // reference implementation: inner_dim = max(hidden_size // reduction, 32)
    // missing keys default to the shipped checkpoint's reductions (4 and 8)
    uint32_t ctrl_reduction  = 4;
    uint32_t route_reduction = 8;
    ml.get_key(LLM_KV_KAIL_CONTROLLER_REDUCTION, ctrl_reduction,  false);
    ml.get_key(LLM_KV_KAIL_ROUTER_REDUCTION,     route_reduction, false);

    if (ctrl_reduction == 0 || route_reduction == 0) {
        throw std::runtime_error("KAIL: controller/router reduction must be >= 1");
    }

    const int64_t n_ctrl  = std::max<int64_t>(n_embd / (int64_t) ctrl_reduction,  32);
    const int64_t n_route = std::max<int64_t>(n_embd / (int64_t) route_reduction, 32);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_ctrl_fc1   = create_tensor(tn(LLM_TENSOR_ATTN_CTRL_FC1, "weight", i), {n_embd, n_ctrl}, 0);
        layer.attn_ctrl_fc1_b = create_tensor(tn(LLM_TENSOR_ATTN_CTRL_FC1, "bias",   i), {n_ctrl}, 0);
        layer.attn_ctrl_fc2   = create_tensor(tn(LLM_TENSOR_ATTN_CTRL_FC2, "weight", i), {n_ctrl, n_head}, 0);
        layer.attn_ctrl_fc2_b = create_tensor(tn(LLM_TENSOR_ATTN_CTRL_FC2, "bias",   i), {n_head}, 0);
        layer.attn_temp       = create_tensor(tn(LLM_TENSOR_ATTN_TEMP,     "weight", i), {n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);

        layer.ffn_router_fc1   = create_tensor(tn(LLM_TENSOR_FFN_ROUTER_FC1, "weight", i), {n_embd, n_route}, 0);
        layer.ffn_router_fc1_b = create_tensor(tn(LLM_TENSOR_FFN_ROUTER_FC1, "bias",   i), {n_route}, 0);
        layer.ffn_router_fc2   = create_tensor(tn(LLM_TENSOR_FFN_ROUTER_FC2, "weight", i), {n_route, n_ff}, 0);
        layer.ffn_router_fc2_b = create_tensor(tn(LLM_TENSOR_FFN_ROUTER_FC2, "bias",   i), {n_ff}, 0);
    }
}
