#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

// note: code adapted from deepseek32.cpp (DSA indexer + absorbed MLA) and step35.cpp (head-wise output gate)

void llama_model_dots3note::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    hparams.f_norm_eps = 1e-6;  // eps for the indexer k_norm layer norm

    // TODO: use MTP layer
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_all");

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func);

    // MLA parameters of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl);

    // MLA parameters of the sliding-window layers
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     hparams.n_lora_kv_swa);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   hparams.n_embd_head_k_mla_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, hparams.n_embd_head_v_mla_swa);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,       hparams.rope_freq_base_train_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    // the NextN/MTP block uses the sliding-attention MLA, so mark it as SWA
    // (the sliding_window_pattern key covers only the trunk layers)
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = 1;
    }

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl);

    switch (hparams.n_layer()) {
        case 46: type = LLM_TYPE_288B_A19B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_dots3note::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    GGML_UNUSED(ml);

    if (!hparams.is_mla()) {
        throw std::runtime_error("DOTS3NOTE architecture requires MLA");
    }

    const int64_t n_embd_head_qk_rope = hparams.n_rot();

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        const bool is_mtp = i >= n_layer;
        // the NextN/MTP block uses the sliding-attention geometry
        const bool is_swa = is_mtp || hparams.is_swa(i);

        // MTP tensors are loaded only when the MTP draft head is enabled
        const int flags = is_mtp ? (ml.load_mtp ? 0 : TENSOR_SKIP | TENSOR_NOT_REQUIRED) : 0;

        const int64_t n_head_l = hparams.n_head(i);

        const int64_t kv_lora_rank       = is_swa ? hparams.n_lora_kv_swa         : hparams.n_lora_kv;
        const int64_t n_embd_head_k_mla  = is_swa ? hparams.n_embd_head_k_mla_swa : hparams.n_embd_head_k_mla();
        const int64_t n_embd_head_v_mla  = is_swa ? hparams.n_embd_head_v_mla_swa : hparams.n_embd_head_v_mla();
        const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
        // norm applied on the shared rope key before rope
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head_qk_rope}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head_l * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head_l}, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head_l}, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head_l * n_embd_head_v_mla, n_embd}, flags);

        // head-wise sigmoid output gate
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_l}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // DSA indexer
        if (!is_mtp && hparams.is_indexer_full(i)) {
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, flags);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);
        }

        if (is_mtp || i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            if (n_expert == 0 || n_expert_used == 0) {
                throw std::runtime_error("n_expert and n_expert_used must be > 0");
            }

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        if (is_mtp) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd }, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dots3note::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_dots3note::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.is_mla());

    const int64_t n_embd_head_qk_rope = hparams.n_rot();

    const int64_t n_indexer_head      = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    // the indexer head layout is [rope | nope]
    GGML_ASSERT(hparams.n_rot() <= n_embd_indexer_head);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_dsa_iswa * inp_attn = build_attn_inp_k_dsa_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const bool is_swa = hparams.is_swa(il);

        const int64_t n_head_l = hparams.n_head(il);

        const int64_t kv_lora_rank        = is_swa ? hparams.n_lora_kv_swa         : hparams.n_lora_kv;
        const int64_t n_embd_head_k_mla   = is_swa ? hparams.n_embd_head_k_mla_swa : hparams.n_embd_head_k_mla();
        const int64_t n_embd_head_v_mla   = is_swa ? hparams.n_embd_head_v_mla_swa : hparams.n_embd_head_v_mla();
        const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

        const float kq_scale    = 1.0f/sqrtf(float(n_embd_head_k_mla));
        const float freq_base_l = model.get_rope_freq_base(cparams, il);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * attn_inp = cur;

            ggml_tensor * qr = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(qr, "qr", il);

            qr = build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr", il);

            ggml_tensor * top_k = nullptr;

            // lightning indexer (full-attention layers only)
            if (!is_swa) {
                ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
                cb(indexer_q, "indexer_q", il);

                // {n_embd_indexer_head, n_indexer_head, n_tokens}
                indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);
                indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_rot,
                                     LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(indexer_q, "indexer_q", il);

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                cb(indexer_k, "indexer_k", il);

                indexer_k = build_norm(indexer_k, model.layers[il].indexer_k_norm, model.layers[il].indexer_k_norm_b, LLM_NORM, il);
                cb(indexer_k, "indexer_k", il);

                // {n_embd_indexer_head, 1, n_tokens}
                indexer_k = ggml_reshape_3d(ctx0, indexer_k, n_embd_indexer_head, 1, n_tokens);
                indexer_k = ggml_rope_ext(ctx0, indexer_k, inp_pos, nullptr, n_rot,
                                     LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(indexer_k, "indexer_k", il);

                // perform Hadamard transform on indexer q and k
                indexer_q = ggml_mul_mat(ctx0, inp_attn->get_dsa()->self_k_rot_lid, indexer_q);
                cb(indexer_q, "indexer_q", il);
                indexer_k = ggml_mul_mat(ctx0, inp_attn->get_dsa()->self_k_rot_lid, indexer_k);
                cb(indexer_k, "indexer_k", il);

                // store indexer keys to KV cache
                const auto * mctx_lid = inp_attn->get_dsa()->mctx->get_lid();
                const auto & k_idxs_lid = inp_attn->get_dsa()->get_k_idxs_lid();
                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                cb(indexer_weights, "indexer_weights", il);

                indexer_k = mctx_lid->get_k(ctx0, il);

                // split the batch into streams if needed
                const auto n_stream = indexer_k->ne[3];
                indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream, indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
                indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream, indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

                // pre-scale weights to avoid scaling operations on huge indexer_score tensor
                indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                cb(indexer_weights, "indexer_weights", il);

                ggml_tensor * indexer_score = nullptr;
                if (cparams.fused_lid) {
                    indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_attn->get_dsa()->get_kq_mask_lid());
                    cb(indexer_score, "indexer_score", il);
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
                } else {
                    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                    cb(indexer_q, "indexer_q", il);
                    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
                    cb(indexer_k, "indexer_k", il);

                    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
                    cb(indexer_kq, "indexer_kq", il);

                    // ReLU requires contiguous tensors
                    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
                    cb(indexer_kq, "indexer_kq", il);

                    indexer_score = ggml_relu(ctx0, indexer_kq);
                    cb(indexer_score, "indexer_score", il);

                    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
                    cb(indexer_score, "indexer_score", il);

                    // sum by q n_indexer_head dimension
                    indexer_score = ggml_sum_rows(ctx0, indexer_score);
                    cb(indexer_score, "indexer_score", il);

                    // permute result to match KQ mask
                    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
                    cb(indexer_score, "indexer_score", il);

                    ggml_tensor * indexer_kq_mask = inp_attn->get_dsa()->get_kq_mask_lid();
                    indexer_score = ggml_add(ctx0, indexer_score, indexer_kq_mask);
                    cb(indexer_score, "indexer_score", il);
                }

                // get indices of top k indexer scores
                uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k ? indexer_score->ne[0] : n_indexer_top_k;
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                cb(top_k, "top_k", il);
            }

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_b, qr);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head_l, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head_l, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                             ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head_l, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head_l, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            // norm on the shared rope key, applied before rope
            k_pe = build_norm(k_pe, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // MLA attention with the absorption optimization
            {
                // {n_embd_head_qk_nope, n_tokens, n_head_l}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head_l} x {n_embd_head_qk_nope, n_tokens, n_head_l}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head_l, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head_l, n_tokens}
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                // apply the head-wise output gate before o_proj, so wo stays out of build_attn
                if (is_swa) {
                    cur = build_attn(inp_attn->get_swa(),
                            nullptr, nullptr, nullptr,
                            Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
                } else {
                    cur = build_attn(inp_attn->get_dsa(),
                            nullptr, nullptr, nullptr,
                            Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);
                }
                cb(cur, "attn_out", il);

                ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp);
                cb(gate, "attn_gate", il);

                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate_sigmoid", il);

                // broadcast the per-head gate over the head dimension
                ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v_mla, n_head_l, n_tokens);
                ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,                1, n_head_l, n_tokens);
                attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
                cb(attn_3d, "attn_gated", il);

                cur = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v_mla * n_head_l, n_tokens);

                cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
                cb(cur, "attn_output", il);
            }
        }

        if (il == n_layer - 1 && inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                    model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                    model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // post-norm hidden state feeds the NextN/MTP draft head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for dots3note
llama_model_dots3note::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DOTS3NOTE MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.is_mla() && "DOTS3NOTE MTP requires MLA");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    // the NextN block uses the sliding-attention MLA geometry (see load_arch_tensors)
    const int64_t n_head_l          = hparams.n_head(il);
    const int64_t kv_lora_rank      = hparams.n_lora_kv_swa;
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla_swa;
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla_swa;

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const float kq_scale    = 1.0f/sqrtf(float(n_embd_head_k_mla));
    const float freq_base_l = model.get_rope_freq_base(cparams, il);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA with the absorption optimization uses a K-only cache (V is a view of K)
    // the MTP cache is a sliding-window cache; build_attn_inp_k_impl rejects SWA
    // caches, so construct the input directly (same as the SWA input of the trunk)
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    inp_swa->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
    inp_swa->self_kq_mask_cnv = inp_swa->self_kq_mask;

    auto * inp_attn = (llm_graph_input_attn_k *) res->add_input(std::move(inp_swa));

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // self-attention: dense MLA, same construction as the sliding-window branch of the trunk graph
    {
        ggml_tensor * attn_inp = cur;

        ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(qr, "mtp_qr", il);

        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "mtp_qr", il);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        cb(q, "mtp_q", il);

        // split into {n_embd_head_qk_nope, n_head_l, n_tokens}
        ggml_tensor * q_nope =
            ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head_l, n_tokens,
                         ggml_row_size(q->type, n_embd_head_k_mla),
                         ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l, 0);
        cb(q_nope, "mtp_q_nope", il);

        // and {n_embd_head_qk_rope, n_head_l, n_tokens}
        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, q, n_embd_head_qk_rope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l,
            ggml_row_size(q->type, n_embd_head_qk_nope));
        cb(q_pe, "mtp_q_pe", il);

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

        // split into {kv_lora_rank, n_tokens}
        ggml_tensor * kv_cmpr =
            ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                         ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        cb(kv_cmpr, "mtp_kv_cmpr", il);

        // and {n_embd_head_qk_rope, 1, n_tokens}
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
        cb(k_pe, "mtp_k_pe", il);

        // norm on the shared rope key, applied before rope
        k_pe = build_norm(k_pe, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(k_pe, "mtp_k_pe", il);

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "mtp_q_pe", il);

        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_pe, "mtp_k_pe", il);

        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(kv_cmpr, "mtp_kv_cmpr", il);

        // MLA with the absorption optimization
        {
            // {n_embd_head_qk_nope, n_tokens, n_head_l}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "mtp_q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head_l} x {n_embd_head_qk_nope, n_tokens, n_head_l}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

            // {kv_lora_rank, n_head_l, n_tokens}
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

            // {n_embd_head_qk_rope + kv_lora_rank, n_head_l, n_tokens}
            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "mtp_Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

            // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "mtp_Kcur", il);

            // {kv_lora_rank, 1, n_tokens}
            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "mtp_Vcur", il);

            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
            cb(cur, "mtp_attn_out", il);

            // apply the head-wise output gate before o_proj, same as the trunk graph
            ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, attn_inp);
            cb(gate, "mtp_attn_gate", il);

            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "mtp_attn_gate_sigmoid", il);

            // broadcast the per-head gate over the head dimension
            ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v_mla, n_head_l, n_tokens);
            ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,                1, n_head_l, n_tokens);
            attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
            cb(attn_3d, "mtp_attn_gated", il);

            cur = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v_mla * n_head_l, n_tokens);

            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
            cb(cur, "mtp_attn_output", il);
        }
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    // dense FFN - the NextN block uses the dense branch (see load_arch_tensors)
    cur = build_ffn(cur,
        layer.ffn_up, NULL, layer.ffn_up_s,
        layer.ffn_gate, NULL, layer.ffn_gate_s,
        layer.ffn_down, NULL, layer.ffn_down_s,
        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    // shared_head_norm applied after the decoder block, before the shared LM head.
    // The post-norm hidden state seeds the next MTP step.
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "DOTS3NOTE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
