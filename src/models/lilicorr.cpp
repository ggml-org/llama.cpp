#include "models.h"

void llama_model_dflash::load_hparams_lilicorr(llama_model_loader & ml) {
    ml.get_key(LLM_KV_LILICORR_CANDIDATE_TOPK, hparams.lilicorr_candidate_topk);
    ml.get_key(LLM_KV_LILICORR_NUM_HEADS,      hparams.lilicorr_num_heads);
    ml.get_key(LLM_KV_LILICORR_NUM_LAYERS,     hparams.lilicorr_layers);
    ml.get_key(LLM_KV_LILICORR_VECTOR_EPS,     hparams.lilicorr_vector_eps);
    ml.get_key(LLM_KV_LILICORR_LOGIT_SCALE,    hparams.lilicorr_logit_scale);
    if (hparams.dflash_block_size < 2 || hparams.lilicorr_hidden_size == 0 ||
            hparams.lilicorr_num_heads == 0 || hparams.lilicorr_hidden_size % hparams.lilicorr_num_heads ||
            hparams.lilicorr_candidate_topk == 0 || hparams.lilicorr_layers == 0 ||
            hparams.dflash_selector_top_k > 0 ||
            (uint64_t) hparams.lilicorr_candidate_topk * (hparams.lilicorr_candidate_topk + 1) > hparams.n_embd) {
        throw std::runtime_error("invalid LiLiCorr DFlash metadata");
    }
}

void llama_model_dflash::load_tensors_lilicorr() {
    LLAMA_LOAD_LOCALS;

    const int64_t hidden = hparams.lilicorr_hidden_size;
    const int64_t top_k  = hparams.lilicorr_candidate_topk;
    const int64_t heads  = hparams.lilicorr_num_heads;
    uint32_t factor = 0, mlp = 0;
    ml->get_key(LLM_KV_LILICORR_FACTOR_DIM, factor);
    ml->get_key(LLM_KV_LILICORR_MLP_HIDDEN_SIZE, mlp);
    const int64_t slots = hparams.dflash_block_size - 1;
    const int feature_bid = n_layer_all + hparams.lilicorr_layers;
    if (factor == 0 || mlp == 0 || hparams.lilicorr_layers > layers.size()) {
        throw std::runtime_error("invalid LiLiCorr head dimensions");
    }
    if (hidden != n_embd) {
        lilicorr_token_proj         = create_tensor(tn(LLM_TENSOR_LILICORR_TOKEN_PROJ, "weight"), { n_embd, hidden }, 0);
        lilicorr_token_proj_b       = create_tensor(tn(LLM_TENSOR_LILICORR_TOKEN_PROJ, "bias"),   { hidden },         0);
        lilicorr_token_proj_s       = create_tensor(tn(LLM_TENSOR_LILICORR_TOKEN_PROJ, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    }
    lilicorr_pass_hidden_proj       = create_tensor(tn(LLM_TENSOR_LILICORR_PASS_HIDDEN_PROJ, "weight"), { hparams.n_embd, hidden }, 0);
    lilicorr_pass_hidden_proj_b     = create_tensor(tn(LLM_TENSOR_LILICORR_PASS_HIDDEN_PROJ, "bias"),   { hidden },                 0);
    lilicorr_pass_hidden_proj_s     = create_tensor(tn(LLM_TENSOR_LILICORR_PASS_HIDDEN_PROJ, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    lilicorr_feature_norm           = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", feature_bid), { 5 }, 0);
    lilicorr_feature_norm_b         = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", feature_bid),   { 5 }, 0);
    lilicorr_feature_mlp_1          = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", feature_bid), { 5, hidden }, 0);
    lilicorr_feature_mlp_1_b        = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", feature_bid),   { hidden },    0);
    lilicorr_feature_mlp_3          = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", feature_bid), { hidden, hidden }, 0);
    lilicorr_feature_mlp_3_b        = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", feature_bid),   { hidden },         0);
    lilicorr_slot_embedding         = create_tensor(tn(LLM_TENSOR_LILICORR_SLOT_EMBD, "weight"), { hidden, slots }, 0);
    lilicorr_rank_embedding         = create_tensor(tn(LLM_TENSOR_LILICORR_RANK_EMBD, "weight"), { hidden, top_k }, 0);
    lilicorr_attention_bias         = create_tensor(tn(LLM_TENSOR_LILICORR_ATTN_BIAS, "bias"), { slots * top_k, slots * top_k, heads }, 0);
    lilicorr_context_proj           = create_tensor(tn(LLM_TENSOR_LILICORR_CONTEXT_PROJ, "weight"), { n_embd, hidden }, 0);
    lilicorr_context_proj_b         = create_tensor(tn(LLM_TENSOR_LILICORR_CONTEXT_PROJ, "bias"),   { hidden },         0);
    lilicorr_context_proj_s         = create_tensor(tn(LLM_TENSOR_LILICORR_CONTEXT_PROJ, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    for (uint32_t i = 0; i < hparams.lilicorr_layers; ++i) {
        const int bid = n_layer_all + i;
        layers[i].lilicorr_attn_norm       = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", bid), { hidden }, 0);
        layers[i].lilicorr_mlp_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", bid), { hidden },  0);
        layers[i].lilicorr_attn_in_proj    = create_tensor(tn(LLM_TENSOR_LILICORR_ATTN_IN_PROJ, "weight", i), { hidden, 3 * hidden }, 0);
        layers[i].lilicorr_attn_in_proj_b  = create_tensor(tn(LLM_TENSOR_LILICORR_ATTN_IN_PROJ, "bias", i), { 3 * hidden },           0);
        layers[i].lilicorr_attn_out_proj   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", bid), { hidden, hidden }, 0);
        layers[i].lilicorr_attn_out_proj_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", bid), { hidden },           0);
        layers[i].lilicorr_attn_out_proj_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", bid), { 1 }, TENSOR_NOT_REQUIRED);
        layers[i].lilicorr_mlp_0           = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", bid), { hidden, mlp }, 0);
        layers[i].lilicorr_mlp_0_b         = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", bid), { mlp },           0);
        layers[i].lilicorr_mlp_0_s         = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", bid), { 1 }, TENSOR_NOT_REQUIRED);
        layers[i].lilicorr_mlp_2           = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", bid), { mlp, hidden }, 0);
        layers[i].lilicorr_mlp_2_b         = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", bid), { hidden },        0);
        layers[i].lilicorr_mlp_2_s         = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", bid), { 1 }, TENSOR_NOT_REQUIRED);
    }
    lilicorr_output_norm            = create_tensor(tn(LLM_TENSOR_LILICORR_OUTPUT_NORM, "weight"), { hidden }, 0);
    lilicorr_anchor_norm            = create_tensor(tn(LLM_TENSOR_LILICORR_ANCHOR_NORM, "weight"), { hidden }, 0);
    lilicorr_factor_input_proj      = create_tensor(tn(LLM_TENSOR_LILICORR_FACTOR_INPUT_PROJ, "weight"), { 3 * hidden, hidden }, 0);
    lilicorr_factor_input_proj_b    = create_tensor(tn(LLM_TENSOR_LILICORR_FACTOR_INPUT_PROJ, "bias"), { hidden },               0);
    lilicorr_out_head               = create_tensor(tn(LLM_TENSOR_LILICORR_OUT_HEAD, "weight"), { hidden, factor }, 0);
    lilicorr_out_head_b             = create_tensor(tn(LLM_TENSOR_LILICORR_OUT_HEAD, "bias"), { factor },           0);
    lilicorr_in_head                = create_tensor(tn(LLM_TENSOR_LILICORR_IN_HEAD, "weight"), { hidden, factor }, 0);
    lilicorr_in_head_b              = create_tensor(tn(LLM_TENSOR_LILICORR_IN_HEAD, "bias"), { factor },           0);
    lilicorr_anchor_out_head        = create_tensor(tn(LLM_TENSOR_LILICORR_ANCHOR_OUT_HEAD, "weight"), { hidden, factor }, 0);
    lilicorr_anchor_out_head_b      = create_tensor(tn(LLM_TENSOR_LILICORR_ANCHOR_OUT_HEAD, "bias"), { factor },           0);

    load_tensors_dflash2_conv();
}
