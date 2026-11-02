#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    ml.get_key(LLM_KV_LILICORR_CANDIDATE_TOPK, hparams.lilicorr_candidate_topk, has_lilicorr);
    ml.get_key(LLM_KV_LILICORR_NUM_HEADS,      hparams.lilicorr_num_heads,      has_lilicorr);
    ml.get_key(LLM_KV_LILICORR_NUM_LAYERS,     hparams.lilicorr_layers,         has_lilicorr);
    ml.get_key(LLM_KV_LILICORR_VECTOR_EPS,     hparams.lilicorr_vector_eps,     has_lilicorr);
    ml.get_key(LLM_KV_LILICORR_LOGIT_SCALE,    hparams.lilicorr_logit_scale,    has_lilicorr);
    if (has_lilicorr && (hparams.dflash_block_size < 2 || hparams.lilicorr_hidden_size == 0 ||
            hparams.lilicorr_num_heads == 0 || hparams.lilicorr_hidden_size % hparams.lilicorr_num_heads ||
            hparams.lilicorr_candidate_topk == 0 || hparams.lilicorr_layers == 0 ||
            hparams.dflash_selector_top_k > 0 ||
            (uint64_t) hparams.lilicorr_candidate_topk * (hparams.lilicorr_candidate_topk + 1) > hparams.n_embd)) {
        throw std::runtime_error("invalid LiLiCorr DFlash metadata");
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    // DeepSeek-V4 DSpark backbone: stages are full DSV4 blocks, uniform sliding window (the draft KV ring)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,                hparams.n_lora_q);
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,             hparams.n_swa);
        ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,    hparams.n_ff_exp_arr, hparams.n_layer_all);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                  hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,                 hparams.expert_weights_scale);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,                  hparams.expert_weights_norm);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                   hparams.expert_gating_func);
        ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,              hparams.swiglu_clamp_exp, hparams.n_layer_all);
        if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,       hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
            hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
        }
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
        ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
        ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS,            hparams.dsv4_compress_ratios, false);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINGLE_PASS,         hparams.dsv4_hc_single_pass, false);

        GGML_ASSERT(hparams.dsv4_o_group_count > 0); // avoid div by zero

        if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            throw std::runtime_error("DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (hparams.dsv4_compress_ratios[il] != 0) {
                throw std::runtime_error("DSpark DSV4 draft expects uncompressed attention on all stages");
            }
        }

        GGML_ASSERT(hparams.n_swa > 0);
        if (hparams.dsv4_hc_single_pass) {
            // DeepSeek-V4.1 DSpark: block queries see the main-KV ring plus the block via the mask row of the
            // first query, so that row's SWA window must reach window_size positions before the anchor
            hparams.n_swa += 1;
        }
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_swa_impl[il] = true;
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // Mistral Large 4 DSpark: dense MLA stages, one sliding window, plain YaRN
    if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl);
        ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,  hparams.rope_yarn_log_mul, false);
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);

        if (hparams.n_swa == 0) {
            throw std::runtime_error("DSpark Mistral Large 4 draft expects a sliding window");
        }
        if (hparams.n_embd_head_k_full != hparams.n_lora_kv + hparams.n_rot()) {
            throw std::runtime_error("DSpark Mistral Large 4 draft expects key_length == kv_lora_rank + rope dim");
        }

        // same as DSV4.1 DSpark: block uses the window of the anchor
        hparams.n_swa += 1;
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_swa_impl[il] = true;
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // optional interleaved sliding-window attention with per-layer pattern array.
    // DFlash has a single rope, so the SWA rope == main rope.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab_draft }, 0);
        dspark_markov_w2_s = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, TENSOR_NOT_REQUIRED);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        dspark_joint_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "weight"), { n_embd + 2*dspark_markov_rank, 3*dspark_markov_rank }, TENSOR_NOT_REQUIRED);
        dspark_joint_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "bias"),   { 3*dspark_markov_rank }, dspark_joint_proj ? 0 : TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
    }

    const struct ggml_tensor * selector_meta = ml->get_tensor_meta("selector_hidden.weight");
    const bool conv_meta = hparams.dflash_conv_kernel_size > 0 && hparams.dflash_conv_group_size > 0;
    if (conv_meta && n_embd % hparams.dflash_conv_group_size != 0) {
        throw std::runtime_error("DFlash hidden size must be divisible by conv_group_size");
    }
    if (selector_meta) {
        const int64_t rank = hparams.dflash_selector_rank;
        if (rank <= 0 || hparams.dflash_block_size <= 0 || hparams.dflash_selector_top_k <= 0 ||
                hparams.dflash_conv_kernel_size <= 0 || hparams.dflash_conv_group_size <= 0) {
            throw std::runtime_error("DFlash2 model is missing conv/selector metadata");
        }
        if (n_embd % hparams.dflash_conv_group_size != 0) {
            throw std::runtime_error("DFlash2 hidden size must be divisible by conv_group_size");
        }
        if (n_embd < hparams.dflash_selector_top_k * (hparams.dflash_selector_top_k + 1)) {
            throw std::runtime_error("DFlash2 hidden size is too small for the selector lattice");
        }

        dflash_selector_prev   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_PREV,   "weight"), { rank, n_vocab }, 0);
        dflash_selector_next   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_NEXT,   "weight"), { rank, n_vocab }, 0);
        dflash_selector_hidden = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_HIDDEN, "weight"), { n_embd, rank }, 0);

        LLAMA_LOG_INFO("%s: DFlash2 conv kernel = %u, group = %u, selector rank = %u, top-k = %u\n", __func__,
                hparams.dflash_conv_kernel_size, hparams.dflash_conv_group_size,
                hparams.dflash_selector_rank, hparams.dflash_selector_top_k);
    }

    if (hparams.lilicorr_layers > 0) {
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
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    if (hparams.n_lora_kv > 0) {
        const int64_t kv_lora_rank      = hparams.n_lora_kv;
        const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
        const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
        const int64_t n_embd_head_nope  = n_embd_head_k_mla - n_rot;

        dspark_mask_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_EMBD, "weight"), { n_embd, hparams.dflash_block_size - 1 }, 0);

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
            layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), { n_embd, n_head * n_embd_head_k_mla }, 0);
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), { n_embd, kv_lora_rank + n_rot }, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), { kv_lora_rank }, 0);
            layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", i), { n_embd_head_nope, kv_lora_rank, n_head }, 0);
            layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", i), { kv_lora_rank, n_embd_head_v_mla, n_head }, 0);
            layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), { n_head * n_embd_head_v_mla, n_embd }, 0);

            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        }
        return;
    }

    if (hparams.dsv4_hc_mult > 0) {
        const int64_t q_lora_rank     = hparams.n_lora_q;
        const int64_t n_ff_exp        = hparams.n_ff_exp();
        const int64_t n_expert_shared = hparams.n_expert_shared;
        const int64_t n_embd_head     = hparams.n_embd_head_k();
        const int64_t o_groups        = hparams.dsv4_o_group_count;
        const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
        const int64_t hc_mult         = hparams.dsv4_hc_mult;
        const int64_t hc_dim          = hc_mult * n_embd;
        const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

        // single-pass mHC: collapse with the last FFN pre-mix, no hc_head
        if (!hparams.dsv4_hc_single_pass) {
            hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
            hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
            hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);
        }

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
            layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
            layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
            layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
            layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        }
        return;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);

        if (conv_meta) {
            const int64_t kernel = hparams.dflash_conv_kernel_size;
            const int64_t groups = n_embd / hparams.dflash_conv_group_size;
            const int64_t projected = 2 * kernel * groups;
            layer.dflash_attn_conv_base = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
            layer.dflash_attn_conv_proj = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_PROJ, "weight", i), { n_embd, projected }, 0);
            layer.dflash_ffn_conv_base  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
            layer.dflash_ffn_conv_proj  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_PROJ,  "weight", i), { n_embd, projected }, 0);
        }
    }
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
static void build_dspark_markov_head(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens, ggml_tensor * hidden = nullptr) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && "DSpark markov weights not loaded");

    // confidence head is optional
    const bool has_conf = model.dspark_conf_proj != nullptr;

    // rnn head: gated state carried across block positions
    const bool is_rnn = model.dspark_joint_proj != nullptr;
    GGML_ASSERT(!is_rnn || hidden != nullptr);

    ggml_tensor * base = res->t_logits; // [n_vocab, n_tokens]
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const auto it = model.gguf_kv.find("dflash.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark draft requires 'dflash.block_size' in GGUF metadata");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    // bonus anchor (SpecForge exports): slot 0 is a bonus token, not a prediction slot
    const auto it_anchor          = model.gguf_kv.find("dflash.sample_from_anchor");
    const bool sample_from_anchor = it_anchor == model.gguf_kv.end() || it_anchor->second == "true";
    const int64_t i_draft_beg    = sample_from_anchor ? 0 : 1;

    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");
    // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    if (!sample_from_anchor) {
        // bonus anchor slot: pass the logits through unbiased, pad the (unread) confidence column
        cat = ggml_cont(ctx0, ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, 0));
        if (has_conf) {
            cat_conf = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, base, 1, n_blocks, base_stride, 0));
        }
    }

    ggml_tensor * state = nullptr; // [R, n_blocks]

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    for (int64_t i = i_draft_beg; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev); // [R, n_blocks]
        ggml_tensor * markov  = w1_prev;

        if (is_rnn) {
            const int64_t rank = w1_prev->ne[0];
            if (state == nullptr) {
                state = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, n_blocks), 0.0f);
            }

            ggml_tensor * hidden_i = ggml_view_2d(ctx0, hidden, hidden->ne[0], n_blocks,
                                                  (size_t) block_drafts * hidden->nb[1], i*hidden->nb[1]);
            ggml_tensor * joint = ggml_concat(ctx0, ggml_concat(ctx0, state, w1_prev, 0), ggml_cont(ctx0, hidden_i), 0);
            joint = ggml_add(ctx0, ggml_mul_mat(ctx0, model.dspark_joint_proj, joint), model.dspark_joint_proj_b); // [3R, n_blocks]

            const auto chunk = [&](int64_t k) {
                return ggml_cont(ctx0, ggml_view_2d(ctx0, joint, rank, n_blocks, joint->nb[1], k*rank*ggml_element_size(joint)));
            };
            ggml_tensor * gate = ggml_sigmoid(ctx0, chunk(0));
            ggml_tensor * cand = ggml_tanh(ctx0, chunk(1));

            // state = gate * state + (1 - gate) * cand
            state  = ggml_add(ctx0, cand, ggml_mul(ctx0, gate, ggml_sub(ctx0, state, cand)));
            markov = ggml_tanh(ctx0, chunk(2));
        }

        ggml_tensor * bias = g.build_lora_mm(w2, markov, model.dspark_markov_w2_s); // [n_vocab_draft, n_blocks]
        if (model.d2t) {
            // reduced draft vocab: scatter the bias to the target rows (base is -inf on the others)
            const int64_t n_draft_vocab = bias->ne[0];
            ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_blocks), 0.0f);
            bias = ggml_set_rows(ctx0, full,
                    ggml_reshape_3d(ctx0, bias,      1,             n_draft_vocab, n_blocks),
                    ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
            bias = ggml_reshape_2d(ctx0, bias, n_vocab, n_blocks);
        }

        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        if (has_conf) {
            // confidence head input: predicts per-position acceptance
            ggml_tensor * conf_inp   = res->t_embd; // [n_embd, n_tok]
            // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
            ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                    (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), markov, 0);
            ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
            if (model.dspark_conf_proj_b) {
                conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
            }
            conf = ggml_sigmoid(ctx0, conf);

            cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;
        }

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    if (has_conf) {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);

        // note: broadcast the [1, n_tok] confidences to n_embd-wide rows to be able to reuse `llama_get_embeddings_nextn`
        conf = ggml_repeat(ctx0, conf, res->t_embd);
        res->t_h_nextn = conf;
        ggml_build_forward_expand(g.gf, conf);
    }

    res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

static ggml_tensor * build_dflash2_conv(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * dynamic,
        ggml_tensor * base,
        int side) {
    const auto & hparams = g.hparams;
    const int64_t hidden_size = hidden->ne[0];
    const int64_t n_tokens    = hidden->ne[1];
    const int64_t n_blocks    = g.ubatch.n_seqs_unq;
    const int64_t kernel_size = hparams.dflash_conv_kernel_size;
    const int64_t group_size  = hparams.dflash_conv_group_size;
    const int64_t n_groups    = hidden_size / group_size;

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(dynamic && base && side >= 0 && side < 2);

    const int64_t block_size = n_tokens / n_blocks;
    ggml_context * ctx0 = g.ctx0;
    // ggml_cont copies even when the tensor is already contiguous
    if (!ggml_is_contiguous(hidden) || hidden->ne[1] != n_tokens) {
        hidden = ggml_cont_2d(ctx0, hidden, hidden_size, n_tokens);
    }
    if (!ggml_is_contiguous(dynamic) || dynamic->ne[1] != n_tokens) {
        dynamic = ggml_cont_2d(ctx0, dynamic, dynamic->ne[0], n_tokens);
    }
    ggml_tensor * blocks = ggml_reshape_3d(ctx0, hidden, hidden_size, block_size, n_blocks);
    ggml_tensor * coeffs = ggml_reshape_4d(ctx0, dynamic, n_groups, kernel_size, 2, n_tokens);
    ggml_tensor * coeffs_side = ggml_view_3d(ctx0, coeffs, n_groups, kernel_size, n_tokens,
            coeffs->nb[1], coeffs->nb[3], side * coeffs->nb[2]);

    ggml_tensor * coeff_all = ggml_cont(ctx0, coeffs_side);
    coeff_all = ggml_reshape_4d(ctx0, coeff_all, 1, n_groups, kernel_size, n_tokens);
    coeff_all = ggml_repeat_4d(ctx0, coeff_all, group_size, n_groups, kernel_size, n_tokens);

    ggml_tensor * base_side = ggml_reshape_4d(ctx0,
            ggml_view_1d(ctx0, base, hidden_size * kernel_size, side * base->nb[2]),
            group_size, n_groups, kernel_size, 1);

    ggml_tensor * weight_all = ggml_add(ctx0, coeff_all, base_side);

    // taps at or past block_size only read the left padding and add nothing
    const int64_t n_taps = std::min(kernel_size, block_size);

    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < n_taps; ++tap) {
        ggml_tensor * values = blocks;
        if (tap > 0) {
            ggml_tensor * previous = ggml_view_3d(ctx0, blocks, hidden_size, block_size - tap, n_blocks,
                    blocks->nb[1], blocks->nb[2], 0);
            values = ggml_pad_ext(ctx0, previous, 0, 0, tap, 0, 0, 0, 0, 0);
        }
        values = ggml_reshape_2d(ctx0, values, hidden_size, n_tokens);

        ggml_tensor * weight = ggml_reshape_2d(ctx0,
                ggml_cont(ctx0, ggml_view_4d(ctx0, weight_all, group_size, n_groups, 1, n_tokens,
                        weight_all->nb[1], weight_all->nb[2], weight_all->nb[3], tap * weight_all->nb[2])),
                hidden_size, n_tokens);

        ggml_tensor * term = ggml_mul(ctx0, weight, values);
        result = result ? ggml_add(ctx0, result, term) : term;
    }
    return result;
}

// DFlash2 selector: top-k candidates per block position plus the pairwise
// transition scores, packed into the nextn output slot for the CPU-side walk.
static void build_dflash2_selector(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const auto & hparams = g.hparams;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_embd   = g.n_embd;

    const int64_t top_k    = hparams.dflash_selector_top_k;
    const int64_t rank     = hparams.dflash_selector_rank;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(res->t_logits->ne[1] == n_tokens);
    if (!tokens) {
        return;
    }

    const int64_t tokens_per_block = n_tokens / n_blocks;
    const int64_t block_size = std::min<int64_t>(tokens_per_block, hparams.dflash_block_size);
    const int64_t row_used   = top_k + top_k * top_k;

    ggml_tensor * candidates  = ggml_top_k(ctx0, res->t_logits, top_k);
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, res->t_logits, 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * unary       = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, logits_rows, candidates), top_k, n_tokens);
    ggml_tensor * gate        = g.build_lora_mm(model.dflash_selector_hidden, res->t_embd);

    // Everything below indexes [.., tokens_per_block, n_blocks]: the block
    // position varies fastest, sequences are the outer dimension.
    ggml_tensor * cand_blk  = ggml_reshape_3d(ctx0, candidates, top_k, tokens_per_block, n_blocks);
    ggml_tensor * unary_blk = ggml_reshape_3d(ctx0, unary,      top_k, tokens_per_block, n_blocks);
    ggml_tensor * gate_blk  = ggml_reshape_3d(ctx0, gate,       rank,  tokens_per_block, n_blocks);

    // a position's score reads only the candidate sets at pos-1 and pos, so a run
    // of positions has no internal dependency and scores in one batched matmul
    auto score_run = [&](int64_t beg_pos, int64_t n_pos, ggml_tensor * pred_ids) {
        ggml_tensor * cand_run = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_blk, top_k, n_pos, n_blocks,
                    cand_blk->nb[1], cand_blk->nb[2], beg_pos * cand_blk->nb[1]));
        ggml_tensor * unary_run = ggml_cont(ctx0, ggml_view_3d(ctx0, unary_blk, top_k, n_pos, n_blocks,
                    unary_blk->nb[1], unary_blk->nb[2], beg_pos * unary_blk->nb[1]));
        ggml_tensor * gate_run = ggml_cont(ctx0, ggml_view_3d(ctx0, gate_blk, rank, n_pos, n_blocks,
                    gate_blk->nb[1], gate_blk->nb[2], beg_pos * gate_blk->nb[1]));

        const int64_t n_pred = pred_ids->ne[0] / (n_pos * n_blocks);

        ggml_tensor * successor = ggml_reshape_4d(ctx0,
                ggml_get_rows(ctx0, model.dflash_selector_next, ggml_reshape_1d(ctx0, cand_run, top_k * n_pos * n_blocks)),
                rank, top_k, n_pos, n_blocks);
        ggml_tensor * predecessor = ggml_reshape_4d(ctx0,
                ggml_get_rows(ctx0, model.dflash_selector_prev, pred_ids),
                rank, n_pred, n_pos, n_blocks);

        ggml_tensor * gate_bcast = ggml_reshape_4d(ctx0, gate_run, rank, 1, n_pos, n_blocks);
        ggml_tensor * cond  = ggml_mul(ctx0, predecessor, ggml_repeat(ctx0, gate_bcast, predecessor));
        ggml_tensor * score = ggml_mul_mat(ctx0, successor, cond);
        if (n_pred == 1) {
            score = ggml_repeat_4d(ctx0, score, top_k, top_k, n_pos, n_blocks);
        }
        ggml_tensor * unary_bcast = ggml_reshape_4d(ctx0, unary_run, top_k, 1, n_pos, n_blocks);
        score = ggml_add(ctx0, score, ggml_repeat(ctx0, unary_bcast, score));

        ggml_tensor * row = ggml_concat(ctx0,
                ggml_cast(ctx0, cand_run, GGML_TYPE_F32),
                ggml_reshape_3d(ctx0, score, top_k * top_k, n_pos, n_blocks), 0);
        return ggml_pad(ctx0, row, n_embd - row_used, 0, 0, 0);
    };

    ggml_tensor * packed = ggml_fill(ctx0,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, 1, n_blocks), 0.0f);

    if (block_size > 1) {
        // Position 1 alone: its predecessor is the anchor token, one id per
        // sequence rather than a candidate set.
        ggml_tensor * anchor_ids = ggml_cont_1d(ctx0,
                ggml_view_2d(ctx0, tokens, 1, n_blocks, tokens_per_block * tokens->nb[0], 0), n_blocks);
        packed = ggml_concat(ctx0, packed, score_run(1, 1, anchor_ids), 1);
    }
    if (block_size > 2) {
        ggml_tensor * prev_ids = ggml_reshape_1d(ctx0,
                ggml_cont(ctx0, ggml_view_3d(ctx0, cand_blk, top_k, block_size - 2, n_blocks,
                        cand_blk->nb[1], cand_blk->nb[2], cand_blk->nb[1])),
                top_k * (block_size - 2) * n_blocks);
        packed = ggml_concat(ctx0, packed, score_run(2, block_size - 2, prev_ids), 1);
    }

    packed = ggml_reshape_2d(ctx0, packed, n_embd, block_size * n_blocks);
    g.cb(packed, "dflash2_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

static void build_lilicorr_scores(llm_graph_context & g, const llama_model & model,
        ggml_tensor * tok_embd, ggml_tensor * anchors) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const auto & hparams = g.hparams;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_embd   = g.n_embd;

    const int64_t hidden   = hparams.lilicorr_hidden_size;
    const int64_t top_k    = hparams.lilicorr_candidate_topk;
    const int64_t heads    = hparams.lilicorr_num_heads;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(res->t_logits->ne[1] == n_tokens);
    GGML_ASSERT(anchors && tok_embd);

    const int64_t tokens_per_block = n_tokens / n_blocks;
    const int64_t block_size = hparams.dflash_block_size;
    const int64_t row_used   = top_k + top_k * top_k;
    const int64_t slots  = block_size - 1;
    const int64_t nodes  = slots * top_k;
    const int64_t factor_dim = model.lilicorr_out_head->ne[1];
    GGML_ASSERT(tokens_per_block == block_size);

    ggml_tensor * candidates = ggml_top_k(ctx0, res->t_logits, top_k);
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, res->t_logits, 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * candidate_logits = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, logits_rows, candidates), top_k, n_tokens);
    // Rank features and embeddings require candidates in descending logit order.
    ggml_tensor * order = ggml_argsort(ctx0, candidate_logits, GGML_SORT_ORDER_DESC);
    ggml_tensor * candidate_rows = ggml_reshape_3d(ctx0, candidates, 1, top_k, n_tokens);
    candidates = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, candidate_rows, order), top_k, n_tokens);
    ggml_tensor * prob_rows   = ggml_reshape_3d(ctx0, ggml_soft_max(ctx0, res->t_logits), 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * probs       = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, prob_rows, candidates), top_k, n_tokens);

    ggml_tensor * cand_run  = ggml_reshape_3d(ctx0, candidates, top_k, n_tokens / n_blocks, n_blocks);
    ggml_tensor * probs_run = ggml_reshape_3d(ctx0, probs,      top_k, n_tokens / n_blocks, n_blocks);
    cand_run = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_run, top_k, slots, n_blocks,
            cand_run->nb[1], cand_run->nb[2], cand_run->nb[1]));
    probs_run = ggml_cont(ctx0, ggml_view_3d(ctx0, probs_run, top_k, slots, n_blocks,
            probs_run->nb[1], probs_run->nb[2], probs_run->nb[1]));

    ggml_tensor * lp = ggml_log(ctx0, probs_run);
    ggml_tensor * lp_blk = ggml_reshape_3d(ctx0, lp, top_k, slots, n_blocks);
    ggml_tensor * top_lp = ggml_view_3d(ctx0, lp_blk, 1, slots, n_blocks,
            lp_blk->nb[1], lp_blk->nb[2], 0);
    ggml_tensor * relative = ggml_sub(ctx0, lp_blk, top_lp);
    ggml_tensor * feature = ggml_reshape_4d(ctx0, lp_blk, 1, top_k, slots, n_blocks);

    ggml_tensor * rank = ggml_arange(ctx0, 0, top_k, 1);
    rank = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, rank, top_k, 1, 1), lp_blk);
    ggml_tensor * first = ggml_scale_bias(ctx0, ggml_step(ctx0, rank), -1.0f, 1.0f);

    for (ggml_tensor * item : { probs_run, relative,
            ggml_scale(ctx0, rank, top_k == 1 ? 0.0f : 1.0f / (top_k - 1)), first }) {
        feature = ggml_concat(ctx0, feature, ggml_reshape_4d(ctx0, item, 1, top_k, slots, n_blocks), 0);
    }
    feature = ggml_reshape_2d(ctx0, feature, 5, nodes * n_blocks);

    ggml_tensor * embeds = ggml_get_rows(ctx0, tok_embd, ggml_reshape_1d(ctx0, cand_run, nodes * n_blocks));
    ggml_tensor * token_states = n_embd == hidden ? embeds : ggml_add(ctx0, g.build_lora_mm(model.lilicorr_token_proj, embeds, model.lilicorr_token_proj_s), model.lilicorr_token_proj_b);

    ggml_tensor * pass_blk = ggml_reshape_3d(ctx0, res->t_embd, n_embd, block_size, n_blocks);
    ggml_tensor * passes = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, pass_blk, n_embd, slots, n_blocks,
                pass_blk->nb[1], pass_blk->nb[2], pass_blk->nb[1])),
            n_embd, slots * n_blocks);

    ggml_tensor * pass_states = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_pass_hidden_proj, passes, model.lilicorr_pass_hidden_proj_s), model.lilicorr_pass_hidden_proj_b);

    pass_states = ggml_reshape_4d(ctx0, pass_states, hidden, 1, slots, n_blocks);
    pass_states = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, pass_states, hidden, top_k, slots, n_blocks), hidden, nodes * n_blocks);

    ggml_tensor * feature_states = ggml_norm(ctx0, feature, 1.0e-5f);
    feature_states = ggml_add(ctx0, ggml_mul(ctx0, feature_states, model.lilicorr_feature_norm), model.lilicorr_feature_norm_b);
    feature_states = ggml_silu(ctx0, ggml_add(ctx0, g.build_lora_mm(model.lilicorr_feature_mlp_1, feature_states), model.lilicorr_feature_mlp_1_b));
    feature_states = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_feature_mlp_3, feature_states), model.lilicorr_feature_mlp_3_b);

    ggml_tensor * slot = ggml_reshape_4d(ctx0, model.lilicorr_slot_embedding, hidden, 1, slots, 1);
    ggml_tensor * rank_embd = ggml_reshape_4d(ctx0, model.lilicorr_rank_embedding, hidden, top_k, 1, 1);
    ggml_tensor * pos = ggml_add(ctx0, ggml_repeat_4d(ctx0, slot, hidden, top_k, slots, n_blocks), ggml_repeat_4d(ctx0, rank_embd, hidden, top_k, slots, n_blocks));
    pos = ggml_reshape_2d(ctx0, pos, hidden, nodes * n_blocks);

    ggml_tensor * x = ggml_add(ctx0, ggml_add(ctx0, token_states, pass_states), feature_states);
    x = ggml_add(ctx0, x, pos);
    ggml_tensor * anchor_blk = ggml_reshape_3d(ctx0, anchors, n_embd, block_size, n_blocks);
    ggml_tensor * anchor = ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_view_3d(ctx0, anchor_blk, n_embd, 1, n_blocks,
                    anchor_blk->nb[1], anchor_blk->nb[2], 0)), n_embd, n_blocks);
    ggml_tensor * anchor_state = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_context_proj, anchor, model.lilicorr_context_proj_s), model.lilicorr_context_proj_b);

    const int64_t head_dim = hidden / heads;
    for (uint32_t layer = 0; layer < hparams.lilicorr_layers; ++layer) {
        const auto & weights = model.layers[layer];
        ggml_tensor * attn_input = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), weights.lilicorr_attn_norm);
        ggml_tensor * qkv = ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_attn_in_proj, attn_input), weights.lilicorr_attn_in_proj_b);
        qkv = ggml_reshape_3d(ctx0, qkv, 3 * hidden, nodes, n_blocks);
        ggml_tensor * q = ggml_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], 0));
        ggml_tensor * k = ggml_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], hidden * qkv->nb[0]));
        ggml_tensor * v = ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], 2 * hidden * qkv->nb[0]);
        ggml_tensor * attn = ggml_soft_max_ext(ctx0, ggml_mul_mat(ctx0, k, q), model.lilicorr_attention_bias,
                1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
        ggml_tensor * joined = ggml_out_prod(ctx0, v, ggml_transpose(ctx0, attn));
        joined = ggml_permute(ctx0, joined, 0, 2, 1, 3);
        ggml_tensor * attn_output = ggml_cont_2d(ctx0, joined, hidden, nodes * n_blocks);
        attn_output = ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_attn_out_proj, attn_output, weights.lilicorr_attn_out_proj_s), weights.lilicorr_attn_out_proj_b);
        x = ggml_add(ctx0, x, attn_output);
        ggml_tensor * mlp = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), weights.lilicorr_mlp_norm);
        mlp = ggml_silu(ctx0, ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_mlp_0, mlp, weights.lilicorr_mlp_0_s), weights.lilicorr_mlp_0_b));
        mlp = ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_mlp_2, mlp, weights.lilicorr_mlp_2_s), weights.lilicorr_mlp_2_b);
        x = ggml_add(ctx0, x, mlp);
    }

    x = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), model.lilicorr_output_norm);
    anchor_state = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchor_state, hparams.f_norm_rms_eps), model.lilicorr_anchor_norm);
    ggml_tensor * anchor_nodes = ggml_reshape_3d(ctx0, anchor_state, hidden, 1, n_blocks);
    anchor_nodes = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, anchor_nodes, hidden, nodes, n_blocks, 1), hidden, nodes * n_blocks);
    ggml_tensor * factor_in = ggml_concat(ctx0, x, anchor_nodes, 0);
    factor_in = ggml_concat(ctx0, factor_in, ggml_mul(ctx0, x, anchor_nodes), 0);
    ggml_tensor * factors = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_factor_input_proj, factor_in), model.lilicorr_factor_input_proj_b);
    factors = ggml_silu(ctx0, factors);

    ggml_tensor * out = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_out_head, factors), model.lilicorr_out_head_b);
    out = ggml_reshape_4d(ctx0, ggml_l2_norm(ctx0, out, hparams.lilicorr_vector_eps), factor_dim, top_k, slots, n_blocks);

    ggml_tensor * in = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_in_head, factors), model.lilicorr_in_head_b);
    in = ggml_reshape_4d(ctx0, ggml_l2_norm(ctx0, in, hparams.lilicorr_vector_eps), factor_dim, top_k, slots, n_blocks);

    ggml_tensor * anchor_out = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_anchor_out_head, anchor_state), model.lilicorr_anchor_out_head_b);
    anchor_out = ggml_l2_norm(ctx0, anchor_out, hparams.lilicorr_vector_eps);
    anchor_out = ggml_reshape_3d(ctx0, anchor_out, factor_dim, 1, n_blocks);

    ggml_tensor * first_in = ggml_cont(ctx0, ggml_view_4d(ctx0, in, factor_dim, top_k, 1, n_blocks,
            in->nb[1], in->nb[2], in->nb[3], 0));
    ggml_tensor * first_score = ggml_scale(ctx0,
            ggml_mul_mat(ctx0, first_in, ggml_reshape_4d(ctx0, anchor_out, factor_dim, 1, 1, n_blocks)),
            hparams.lilicorr_logit_scale);
    ggml_tensor * scores = ggml_pad(ctx0,
            ggml_reshape_3d(ctx0, first_score, top_k, 1, n_blocks), top_k * top_k - top_k, 0, 0, 0);
    if (slots > 1) {
        ggml_tensor * next = ggml_cont(ctx0, ggml_view_4d(ctx0, in, factor_dim, top_k, slots - 1, n_blocks,
                in->nb[1], in->nb[2], in->nb[3], in->nb[2]));
        ggml_tensor * previous = ggml_cont(ctx0, ggml_view_4d(ctx0, out, factor_dim, top_k, slots - 1, n_blocks,
                out->nb[1], out->nb[2], out->nb[3], 0));
        ggml_tensor * score = ggml_scale(ctx0, ggml_mul_mat(ctx0, next, previous), hparams.lilicorr_logit_scale);
        scores = ggml_concat(ctx0, scores,
                ggml_reshape_3d(ctx0, score, top_k * top_k, slots - 1, n_blocks), 1);
    }

    ggml_tensor * row = ggml_concat(ctx0, ggml_cast(ctx0, ggml_reshape_3d(ctx0, cand_run, top_k, slots, n_blocks),
            GGML_TYPE_F32), scores, 0);
    ggml_tensor * packed = ggml_pad_ext(ctx0, row, 0, n_embd - row_used, 1, 0, 0, 0, 0, 0);
    packed = ggml_reshape_2d(ctx0, packed, n_embd, n_tokens);
    g.cb(packed, "lilicorr_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    // optional iSWA: pick the matching attention input
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto build_rope = [&](ggml_tensor * cur, ggml_tensor * pos) {
        return rope_type == GGML_ROPE_TYPE_MROPE
            ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
    };

    // KV cache injection
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        cb(inp_target, "inp_target_features", -1);

        res->add_input(std::move(inp));

        // fuse the target features through the encoder
        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
        cb(inp_g, "inp_g_embeddings", -1);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g, layer.wk_s);
            const bool shared_kv = layer.wv == nullptr;
            ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, inp_g, layer.wv_s);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            if (shared_kv) {
                Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
            }
            Kcur = build_rope(Kcur, inp_pos);
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                // rotate K/V into the cache's rotated space
                ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
                ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
                if (k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                // rotate K/V into the cache's rotated space
                if (inp_attn->self_k_rot[il]) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
                }
                if (inp_attn->self_v_rot[il]) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
                }
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;
        if (model.lilicorr_pass_hidden_proj) {
            // Expose context features for retrieval by LiliCorr
            res->t_h_nextn = inp_g;
        }

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    ggml_tensor * inp_anchor = nullptr;
    if (model.lilicorr_pass_hidden_proj && ubatch.embd_state && n_tokens == (int64_t) hparams.dflash_block_size * ubatch.n_seqs_unq) {
        // LiliCorr sets inp_anchor to pass the context features to the lattice scorer
        auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd, n_embd);
        inp->tokens = inp_tokens;
        inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->h);
        inp_anchor = inp->h;
        res->add_input(std::move(inp));
    } else {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->tokens = inp_tokens;
        res->add_input(std::move(inp));
    }

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, noise_norm, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos);
        Kcur = build_rope(Kcur, inp_pos);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head from the target model (shared via ctx_other)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // DSpark: bias the draft logits with the Markov head
    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }

    if (model.dflash_selector_hidden) {
        build_dflash2_selector(*this, model, inp_tokens);
    } else if (model.lilicorr_pass_hidden_proj && ubatch.embd_state && n_tokens == (int64_t) hparams.dflash_block_size * ubatch.n_seqs_unq) {
        build_lilicorr_scores(*this, model, tok_embd, inp_anchor);
    }
}

// DSV4 DSpark decoder, dual-mode by batch type (see the DFlash decoder above):
//   * embd batch  -> project main_x through each stage's wkv and inject K into the ring cache
//   * token batch -> noise block through 3 full DSV4 stages (hc + MLA + MoE), markov + confidence heads
llama_model_dflash::graph_dsv4::graph_dsv4(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const int64_t n_embd_inp       = hparams.n_embd_inp_enc();
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // KV cache injection: fused target features from the encoder
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        cb(inp_target, "inp_target_features", -1);

        res->add_input(std::move(inp));

        // fuse the target features through the encoder
        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
        cb(inp_g, "inp_g_embeddings", -1);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            // main-track KV: kv_norm(wkv(main_x)) with rope on the trailing dims, same
            // rope parameters as the uncompressed layers in build_attention_impl
            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_rope_set_offset(kv, n_embd_head_nope);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa[il]) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    const int64_t hc = hparams.dsv4_hc_mult;
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        hc_pre attn_hc = build_hc_pre(inpL,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                il);
        ggml_tensor * cur = attn_hc.cur;
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_attn, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        hc_pre ffn_hc = build_hc_pre(inpL,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                il);
        cur = ffn_hc.cur;
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    // confidence head input: the reference scores the pre-norm collapsed hidden state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head from the target model (shared via ctx_other)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

// DeepSeek-V4.1 DSpark decoder (mtp.0..2), dual-mode by batch type like graph_dsv4:
//   * embd batch  -> writes every stage's kv_norm(wkv(main_proj(target features))) into its window ring
//   * token batch -> [anchor, noise...] block through the stages, single-pass mHC, no per-head q norm
// window of a block = window of its first query, see n_swa
static void build_dspark_block_mask(llm_graph_context & g, llm_graph_input_attn_k_iswa * inp_attn) {
    ggml_context * ctx0 = g.ctx0;

    ggml_tensor * m = inp_attn->self_kq_mask_swa_cnv;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    const int64_t bd       = n_blocks > 0 ? n_tokens / n_blocks : 0;
    if (m && bd > 1 && n_tokens % n_blocks == 0 && m->ne[2] == 1 && m->ne[1] % bd == 0 && ggml_is_contiguous(m)) {
        ggml_tensor * m4 = ggml_reshape_4d(ctx0, m, m->ne[0], bd, m->ne[1]/bd, m->ne[3]);
        ggml_tensor * r0 = ggml_view_4d(ctx0, m4, m4->ne[0], 1, m4->ne[2], m4->ne[3], m4->nb[1], m4->nb[2], m4->nb[3], 0);
        m4 = ggml_repeat(ctx0, r0, m4);
        inp_attn->self_kq_mask_swa_cnv = ggml_reshape_4d(ctx0, m4, m->ne[0], m->ne[1], 1, m->ne[3]);
        g.cb(inp_attn->self_kq_mask_swa_cnv, "dspark_block_mask", -1);
    }
}

llama_model_dflash::graph_dsv41::graph_dsv41(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek41::graph(params) {
    const int64_t n_embd_inp       = hparams.n_embd_inp_enc();
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    GGML_ASSERT(hparams.dsv4_hc_single_pass);

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // ratio-0 layers: plain rope at rope_theta, no YaRN (original_seq_len 0)
    const auto rope = [&](ggml_tensor * t) {
        t = ggml_rope_ext(ctx0, t, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_rope_set_offset(t, n_embd_head_nope);
    };

    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        cb(inp_target, "inp_target_features", -1);

        res->add_input(std::move(inp));

        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
        cb(inp_g, "main_x", -1);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = rope(kv);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa[il]) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    const int64_t hc            = hparams.dsv4_hc_mult;
    const int64_t n_groups      = hparams.dsv4_o_group_count;
    const int64_t n_heads_group = n_head / n_groups;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim   = n_heads_group*n_embd_head;
    const float   kq_scale      = 1.0f/sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    build_dspark_block_mask(*this, inp_attn);

    ggml_tensor * pre_prev = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);

        ggml_tensor * cur;
        if (pre_prev == nullptr) {
            // identity pre-mix of the first stage: copy 0
            cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        } else {
            cur = build_hc_pre(inpL, pre_prev, il);
        }
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // non-causal draft context: block queries see the main-KV ring plus every block key
        {
            ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
            qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr_norm", il);

            ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
            q = rope(q);
            cb(q, "q", il);

            ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = rope(kv);
            cb(kv, "kv", il);

            ggml_tensor * out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    q, kv, kv,
                    nullptr, layer.attn_sinks, nullptr,
                    kq_scale, il);
            cb(out, "attn_raw", il);

            out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, n_tokens);
            out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            out = ggml_rope_set_offset(out, n_embd_head_nope);
            cb(out, "attn_derope", il);

            out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, n_tokens);
            out = ggml_permute(ctx0, out, 0, 2, 1, 3);
            ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
            cb(oa, "attn_wo_a", il);
            oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, n_tokens);

            cur = build_lora_mm(layer.wo_b, oa);
            cb(cur, "attn_out", il);
        }

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        pre_prev = ffn_mix.pre;

        cur = build_hc_pre(inpL, attn_mix.pre, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);

        ggml_tensor * l_out = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        cb(l_out, "l_out", il);
        ggml_build_forward_expand(gf, l_out);
    }

    // no hc_head: collapse with the last FFN's pre-mix
    ggml_tensor * cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    // confidence head takes the un-normed collapsed state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

// Mistral Large 4 DSpark, dual-mode like graph_dsv41:
// - embd batch: write latent K of target_norm(fc(features)) to the cache, no attn norm
// - token batch: decode [anchor, mask_1, ...] block, non-causal
llama_model_dflash::graph_mistral_large4::graph_mistral_large4(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_inp        = hparams.n_embd_inp_enc();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_nope  = n_embd_head_k_mla - n_rot;

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head_k_mla));

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // no LeRoPE in the draft
    const auto rope = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx0, t, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    };

    const auto build_kv = [&](ggml_tensor * x, int il) -> std::pair<ggml_tensor *, ggml_tensor *> {
        const auto & layer = model.layers[il];

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, x);
        cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot), 0);
        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        cb(kv_cmpr, "kv_cmpr", il);

        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_rot, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
        k_pe = rope(k_pe);
        cb(k_pe, "k_pe", il);

        return { ggml_concat(ctx0, kv_cmpr, k_pe, 0), kv_cmpr };
    };

    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        cb(inp_target, "inp_target_features", -1);

        res->add_input(std::move(inp));

        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
        cb(inp_g, "main_x", -1);

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * kv = build_kv(inp_g, il).first;
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa[il]) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    res->add_input(std::move(inp));

    const int64_t n_blocks     = ubatch.n_seqs_unq;
    const int64_t block_drafts = n_blocks > 0 ? n_tokens / n_blocks : 0;

    ggml_tensor * inpL;
    if (block_drafts > 0 && n_tokens % n_blocks == 0 && block_drafts <= (int64_t) hparams.dflash_block_size) {
        // slot k > 0 uses mask row k - 1
        ggml_tensor * anchors = ggml_view_2d(ctx0, inp_tokens, 1, n_blocks, (size_t) block_drafts * inp_tokens->nb[0], 0);
        anchors = ggml_get_rows(ctx0, tok_embd, ggml_cont_1d(ctx0, anchors, n_blocks));
        inpL = ggml_reshape_3d(ctx0, anchors, n_embd, 1, n_blocks);
        if (block_drafts > 1) {
            ggml_tensor * masks = ggml_view_3d(ctx0, model.dspark_mask_embd, n_embd, block_drafts - 1, 1,
                    model.dspark_mask_embd->nb[1], model.dspark_mask_embd->nb[2], 0);
            masks = ggml_repeat_4d(ctx0, ggml_cast(ctx0, masks, GGML_TYPE_F32), n_embd, block_drafts - 1, n_blocks, 1);
            inpL = ggml_concat(ctx0, inpL, masks, 1);
        }
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_tokens);
    } else {
        // not a draft block (e.g. graph reserve)
        inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    }
    cb(inpL, "inp_noise_embd", -1);

    build_dspark_block_mask(*this, inp_attn);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(q, "q", il);

            // {n_embd_head_nope, n_tokens, n_head}
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k_mla),
                    ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

            // {kv_lora_rank, n_head, n_tokens}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_rot, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k_mla),
                    ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                    ggml_row_size(q->type, n_embd_head_nope));
            q_pe = rope(q_pe);
            cb(q_pe, "q_pe", il);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            auto [Kcur, Vcur] = build_kv(cur, il);
            cb(Kcur, "Kcur", il);

            // apply wv_b after attn, so the K rotation is already undone
            ggml_tensor * out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr,
                    kq_scale, il);
            cb(out, "attn_latent", il);

            // {kv_lora_rank, n_tokens, n_head} -> {n_embd_head_v_mla, n_tokens, n_head}
            out = ggml_reshape_3d(ctx0, out, kv_lora_rank, n_head, n_tokens);
            out = ggml_mul_mat(ctx0, layer.wv_b, ggml_permute(ctx0, out, 0, 2, 1, 3));
            out = ggml_cont_2d(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3), n_embd_head_v_mla*n_head, n_tokens);

            cur = build_lora_mm(layer.wo, out, layer.wo_s);
            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        inpL = ggml_add(ctx0, cur, ffn_inp);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    ggml_tensor * logits = build_lora_mm(output, cur, output_s);

    // softcap does not apply to the markov bias
    if (hparams.f_final_logit_softcapping) {
        logits = ggml_scale(ctx0, logits, 1.0f / hparams.f_final_logit_softcapping);
        logits = ggml_tanh(ctx0, logits);
        logits = ggml_scale(ctx0, logits, hparams.f_final_logit_softcapping);
    }
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens, cur);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.n_lora_kv > 0) {
                return std::make_unique<graph_mistral_large4>(*this, params);
            }
            if (hparams.dsv4_hc_mult > 0 && hparams.dsv4_hc_single_pass) {
                return std::make_unique<graph_dsv41>(*this, params);
            }
            if (hparams.dsv4_hc_mult > 0) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
