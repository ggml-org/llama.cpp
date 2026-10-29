#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

void llama_model_mistral_large4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,        hparams.f_norm_eps); // eps for the indexer layer norms

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    // MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q, false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl);

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 1);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);
    if (!hparams.is_indexer_full(0)) {
        throw std::runtime_error("the first layer must have a full indexer");
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN,     hparams.rope_pattern,   hparams.n_layer(), false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_KV_SOURCE_MAP,    hparams.kv_source_impl, hparams.n_layer(), false);
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const uint32_t il_src = hparams.kv_source(il);
        if (il_src > il || !hparams.has_kv(il_src) || hparams.has_rope(il_src) != hparams.has_rope(il)) {
            throw std::runtime_error(format("invalid KV source layer %u for layer %u", il_src, il));
        }
    }

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false);

    // SSMax uses log(pos + 1), computed via the attention temperature input as 1 + log(pos + 1)
    hparams.n_attn_temp_floor_scale = 1;
    hparams.f_attn_temp_scale       = 1.0f;
    hparams.f_attn_temp_offset      = 0.0f;

    switch (hparams.n_layer()) {
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_mistral_large4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (!hparams.is_mla()) {
        throw std::runtime_error("MISTRAL_LARGE4 architecture requires MLA");
    }

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_indexer_head      = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // NoPE layers use the full head for the non-rotary part
        const int64_t n_rot_l              = hparams.has_rope(i) ? n_rot : 0;
        const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_rot_l;

        // learned rope frequencies (LeRoPE), shared by all layers
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, i != 0 ? TENSOR_DUPLICATED : 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);

        if (hparams.has_kv(i)) {
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_rot_l}, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
        }

        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.attn_ssmax_s = create_tensor(tn(LLM_TENSOR_ATTN_SSMAX, "weight", i), {n_head}, 0);
        layer.attn_ssmax_b = create_tensor(tn(LLM_TENSOR_ATTN_SSMAX, "bias",   i), {n_head}, 0);

        // DSA indexer
        if (hparams.is_indexer_full(i)) {
            layer.indexer_q_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM,   "weight", i), {n_embd}, 0);
            layer.indexer_q_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM,   "bias",   i), {n_embd}, 0);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {n_embd, n_indexer_head * n_embd_indexer_head}, 0);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_embd_indexer_head}, 0);
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer_head}, 0);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_embd_indexer_head}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_indexer_head}, 0);
            layer.indexer_ssmax_s  = create_tensor(tn(LLM_TENSOR_INDEXER_SSMAX,    "weight", i), {n_indexer_head}, 0);
            layer.indexer_ssmax_b  = create_tensor(tn(LLM_TENSOR_INDEXER_SSMAX,    "bias",   i), {n_indexer_head}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_mistral_large4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mistral_large4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.is_mla());

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_indexer_head      = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const uint32_t n_indexer_top_k    = hparams.indexer_top_k;

    // the indexer head layout is [rope | nope]
    GGML_ASSERT(n_rot <= n_embd_indexer_head);

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // YaRN is used without magnitude scaling (apply_scale = false), so no mscale in kq_scale
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // SSMax position basis: log(pos + 1), {1, 1, n_tokens}
    ggml_tensor * ssmax_pos = ggml_scale_bias(ctx0, build_inp_attn_scale(), 1.0f, -1.0f);
    cb(ssmax_pos, "ssmax_pos", -1);

    // per-head SSMax scale s * log(pos + 1) + b, {1, n_head, n_tokens}
    auto build_ssmax = [&](ggml_tensor * s, ggml_tensor * b, int64_t n_head_l) {
        ggml_tensor * pos = ggml_repeat_4d(ctx0, ssmax_pos, 1, n_head_l, n_tokens, 1);
        ggml_tensor * res = ggml_mul(ctx0, pos, ggml_reshape_3d(ctx0, s, 1, n_head_l, 1));
        return ggml_add(ctx0, res, ggml_reshape_3d(ctx0, b, 1, n_head_l, 1));
    };

    // NoPE layers keep the cache layout of rope layers by zero-filling the rope part
    auto pad_rope = [&](ggml_tensor * x) {
        ggml_tensor * zeros = ggml_new_tensor_3d(ctx0, x->type, n_rot, x->ne[1], x->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);
        return ggml_concat(ctx0, x, zeros, 0);
    };

    llm_graph_input_attn_k_dsa * inp_attn_dsa = build_attn_inp_k_dsa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // shared indexer layers reuse the top_k from the previous full indexer layer
    ggml_tensor * prev_top_k = nullptr;

    // DSpark reads last hidden state of all tokens, select output rows after the last layer
    const bool extract_last = (size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer];

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        res->t_layer_inp[il] = inpL;

        const bool use_rope = hparams.has_rope(il);

        const int64_t n_rot_l             = use_rope ? n_rot : 0;
        const int64_t n_embd_head_qk_nope = n_embd_head_k - n_rot_l;

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * top_k = nullptr;

            // lightning indexer
            if (hparams.is_indexer_full(il)) {
                ggml_tensor * indexer_q = build_norm(cur, layer.indexer_q_norm, layer.indexer_q_norm_b, LLM_NORM, il);
                cb(indexer_q, "indexer_q_norm", il);

                indexer_q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, indexer_q);
                cb(indexer_q, "indexer_q", il);

                // {n_embd_indexer_head, n_indexer_head, n_tokens}
                indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);
                indexer_q = ggml_mul(ctx0, indexer_q, build_ssmax(layer.indexer_ssmax_s, layer.indexer_ssmax_b, n_indexer_head));
                cb(indexer_q, "indexer_q_ssmax", il);

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
                cb(indexer_k, "indexer_k", il);

                indexer_k = build_norm(indexer_k, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
                cb(indexer_k, "indexer_k", il);

                // {n_embd_indexer_head, 1, n_tokens}
                indexer_k = ggml_reshape_3d(ctx0, indexer_k, n_embd_indexer_head, 1, n_tokens);

                if (use_rope) {
                    indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, rope_factors, n_rot,
                                         rope_type, n_ctx_orig, freq_base, freq_scale,
                                         ext_factor, attn_factor, beta_fast, beta_slow);
                    cb(indexer_q, "indexer_q", il);

                    indexer_k = ggml_rope_ext(ctx0, indexer_k, inp_pos, rope_factors, n_rot,
                                         rope_type, n_ctx_orig, freq_base, freq_scale,
                                         ext_factor, attn_factor, beta_fast, beta_slow);
                    cb(indexer_k, "indexer_k", il);
                }

                // perform Hadamard transform on indexer q and k
                indexer_q = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q);
                cb(indexer_q, "indexer_q", il);
                indexer_k = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_k);
                cb(indexer_k, "indexer_k", il);

                // store indexer keys to KV cache
                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
                const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
                cb(indexer_weights, "indexer_weights", il);

                // get cached indexer keys
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
                    indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_attn_dsa->get_kq_mask_lid());
                    cb(indexer_score, "indexer_score", il);
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
                } else {
                    // calculate indexer kq
                    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                    cb(indexer_q, "indexer_q", il);
                    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
                    cb(indexer_k, "indexer_k", il);

                    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
                    cb(indexer_kq, "indexer_kq", il);

                    // ReLU requires contiguous tensors
                    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
                    cb(indexer_kq, "indexer_kq", il);

                    // apply ReLU
                    indexer_score = ggml_relu(ctx0, indexer_kq);
                    cb(indexer_score, "indexer_score", il);

                    // multiply scores by indexer weights
                    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
                    cb(indexer_score, "indexer_score", il);

                    // sum by q n_indexer_head dimension
                    indexer_score = ggml_sum_rows(ctx0, indexer_score);
                    cb(indexer_score, "indexer_score", il);

                    // permute result to match KQ mask
                    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
                    cb(indexer_score, "indexer_score", il);

                    // mask indexer scores
                    ggml_tensor * indexer_kq_mask = inp_attn_dsa->get_kq_mask_lid();
                    indexer_score = ggml_add(ctx0, indexer_score, indexer_kq_mask);
                    cb(indexer_score, "indexer_score", il);
                }

                // get indices of top k indexer scores
                uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k ? indexer_score->ne[0] : n_indexer_top_k;
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                prev_top_k = top_k;
                cb(top_k, "top_k", il);
            } else {
                GGML_ASSERT(prev_top_k != nullptr && "shared indexer layer must follow a previous full indexer layer");
                top_k = prev_top_k;
                cb(top_k, "top_k", il);
            }

            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(q, "q", il);

            // {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k),
                    ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            // {kv_lora_rank, n_head, n_tokens}
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            // {kv_lora_rank + n_rot, n_head, n_tokens}
            // note: rope must go first for in-place context shifting in build_rope_shift()
            ggml_tensor * Qcur = nullptr;
            if (use_rope) {
                ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_rot, n_head, n_tokens,
                        ggml_row_size(q->type, n_embd_head_k),
                        ggml_row_size(q->type, n_embd_head_k) * n_head,
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            } else {
                Qcur = pad_rope(q_nope_absorbed);
            }
            cb(Qcur, "Qcur", il);

            Qcur = ggml_mul(ctx0, Qcur, build_ssmax(layer.attn_ssmax_s, layer.attn_ssmax_b, n_head));
            cb(Qcur, "Qcur_ssmax", il);

            // layers without own KV read the cache written by their source layer
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;

            if (hparams.has_kv(il)) {
                ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
                cb(kv_cmpr_pe, "kv_cmpr_pe", il);

                // {kv_lora_rank, n_tokens}
                ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot_l), 0);
                cb(kv_cmpr, "kv_cmpr", il);

                kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(kv_cmpr, "kv_cmpr", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                if (use_rope) {
                    // {n_rot, 1, n_tokens}
                    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_rot, 1, n_tokens,
                            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
                    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                         ext_factor, attn_factor, beta_fast, beta_slow);
                    cb(k_pe, "k_pe", il);

                    Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                } else {
                    Kcur = pad_rope(kv_cmpr);
                }
                cb(Kcur, "Kcur", il);

                Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);
            }

            // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
            cur = build_attn(inp_attn_dsa,
                    layer.wo, NULL, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, top_k, kq_scale, il);
        }

        if (il == n_layer - 1 && inp_out_ids && !extract_last) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   NULL, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, NULL, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
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

    if (extract_last) {
        res->t_layer_inp[n_layer] = inpL;
        if (inp_out_ids) {
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
