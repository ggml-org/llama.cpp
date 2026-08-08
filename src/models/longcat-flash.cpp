#include "models.h"

void llama_model_longcat_flash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);

    // the router picks among the real experts plus this many identity ("zero-computation") ones
    ml.get_key(LLM_KV_N_ZERO_EXPERTS, hparams.n_zero_experts);

    // fixed MLA lora-rank scale factors, applied in the graph
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_SCALE,  hparams.f_attn_q_lora_scale,  false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_SCALE, hparams.f_attn_kv_lora_scale, false);

    // MTP head, loaded but not wired into the graph yet
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);

    if (hparams.n_layer() % 2 != 0) {
        throw std::runtime_error("longcat-flash requires an even block_count (2 blocks per HF layer)");
    }
    if (hparams.n_lora_q == 0) {
        throw std::runtime_error("q_lora_rank must be > 0");
    }
    if (hparams.n_lora_kv == 0) {
        throw std::runtime_error("kv_lora_rank must be > 0");
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_longcat_flash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla);

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_ff_exp     = hparams.n_ff_exp;

    // the router's actual output width includes the zero-computation experts
    const int64_t n_expert_full = n_expert + hparams.n_zero_experts;

    if (n_expert == 0) {
        throw std::runtime_error("n_expert must be > 0");
    }
    if (n_expert_used == 0) {
        throw std::runtime_error("n_expert_used must be > 0");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);

        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // every block has its own dense FFN
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // the shared MoE only attaches to the even block of each HF-layer pair (see conversion script)
        if (i % 2 == 0) {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert_full}, TENSOR_NOT_REQUIRED);
        }
        if (layer.ffn_gate_inp) {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert_full}, TENSOR_NOT_REQUIRED);

            // +1: dummy all-zero expert appended at conversion time, see build_moe_ffn_custom
            // create split gate/up tensors directly, build_moe_ffn_custom does not read the fused variant
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert + 1}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert + 1}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert + 1}, 0);
        }
    }

    // note: MTP head is unused for now
    if (hparams.n_layer_nextn > 0) {
        const int i = n_layer;
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, skip);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, skip);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, skip);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, skip);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, skip);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, skip);

        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, skip);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, skip);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, skip);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, skip);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, skip);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, skip);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, skip);

        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {2 * n_embd, n_embd}, skip);
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), {n_embd}, skip);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), {n_embd}, skip);

        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, skip);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},          skip);
    }
}

std::unique_ptr<llm_graph_context> llama_model_longcat_flash::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_longcat_flash::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();

    // note: this is the actual head size you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // MLA attention is copied from deepseek2, but the block structure is not.
    // Each HF layer packs 2 attn+dense-ffn sub-blocks, plus a shared MoE added a block later.
    GGML_ASSERT(is_mla);
    GGML_ASSERT(kv_lora_rank > 0);
    GGML_ASSERT(n_layer % 2 == 0);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = build_attn_inp_k(); // MLA-only

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int hl = 0; hl < n_layer / 2; ++hl) {
        ggml_tensor * moe_shortcut = nullptr;

        for (int sub = 0; sub < 2; ++sub) {
            const int il = hl * 2 + sub;

            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                ggml_tensor * q = NULL;

                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                if (hparams.f_attn_q_lora_scale != 0.0f) {
                    q = ggml_scale(ctx0, q, hparams.f_attn_q_lora_scale);
                    cb(q, "q_lora_scaled", il);
                }

                // split into {n_embd_head_qk_nope, n_head, n_tokens}
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                                 ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
                cb(q_nope, "q_nope", il);

                // and {n_embd_head_qk_rope, n_head, n_tokens}
                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                    ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
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

                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(k_pe, "k_pe", il);

                kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(kv_cmpr, "kv_cmpr", il);

                if (hparams.f_attn_kv_lora_scale != 0.0f) {
                    kv_cmpr = ggml_scale(ctx0, kv_cmpr, hparams.f_attn_kv_lora_scale);
                    cb(kv_cmpr, "kv_cmpr_lora_scaled", il);
                }

                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
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

                // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
                cur = build_attn(inp_attn_k,
                        model.layers[il].wo, NULL, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            const bool is_last_block = (il == n_layer - 1);

            if (is_last_block && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
                if (moe_shortcut) {
                    // moe_shortcut was computed over the full token set at sub==0.
                    // Prune it the same way here so it still lines up with cur/inpSA.
                    moe_shortcut = ggml_get_rows(ctx0, moe_shortcut, inp_out_ids);
                }
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if (sub == 0 && model.layers[il].ffn_gate_inp) {
                // shared MoE, its output is delayed and added in at the end of the next sub-block
                moe_shortcut = build_moe_ffn_custom(cur, model.layers[il], il);
            }

            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            if (sub == 1 && moe_shortcut) {
                cur = ggml_add(ctx0, cur, moe_shortcut);
                cb(cur, "ffn_out_moe_shortcut", il);
            }

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_longcat_flash::graph::build_moe_ffn_custom(ggml_tensor * cur, const llama_layer & layer, int il) const {
    const int64_t n_expert_full = layer.ffn_gate_inp->ne[1];

    ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, cur); // [n_expert_full, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    // softmax over the full set (real + zero experts), matching the reference router
    ggml_tensor * probs = ggml_soft_max(ctx0, logits); // [n_expert_full, n_tokens]
    cb(probs, "ffn_moe_probs", il);

    // the bias only steers which experts get picked below; the gathered weight always comes from the unbiased probs
    ggml_tensor * selection_probs = probs;
    if (layer.ffn_exp_probs_b) {
        selection_probs = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_reshape_3d(ctx0, probs, 1, n_expert_full, n_tokens);
    weights = ggml_get_rows(ctx0, weights, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (hparams.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_norm", il);
    }
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    ggml_tensor * ids_f32 = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);

    // TODO: PR #26631 makes mul_mat_id skip an expert on index -1
    //       once it lands, drop the dummy expert and map these slots to -1 with clamp(ids, -1, n_expert - 1) - n_expert*zero_mask
    //       ref: https://github.com/ggml-org/llama.cpp/pull/26631

    // clamp them onto the dummy all-zero expert at index n_expert, so their FFN output is 0
    ggml_tensor * ids = ggml_cast(ctx0, ggml_clamp(ctx0, ids_f32, 0.0f, float(n_expert)), GGML_TYPE_I32);
    cb(ids, "ffn_moe_topk_clamped", il);

    // 0/1 mask of those slots. example with n_expert = 4:
    //   ids     [  0,  1,  2, 3, 4, 5 ]
    //   shifted [ -3, -2, -1, 0, 1, 2 ]
    //   clamped [  0,  0,  0, 0, 1, 1 ]
    ggml_tensor * zero_mask = ggml_scale_bias(ctx0, ids_f32, 1.0f, -float(n_expert - 1));
    zero_mask = ggml_clamp(ctx0, zero_mask, 0.0f, 1.0f);
    cb(zero_mask, "ffn_moe_zero_mask", il);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    ggml_tensor * up   = ggml_mul_mat_id(ctx0, layer.ffn_up_exps,   cur_3d, ids); // [n_ff_exp, n_expert_used, n_tokens]
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, layer.ffn_gate_exps, cur_3d, ids); // [n_ff_exp, n_expert_used, n_tokens]
    cb(up,   "ffn_moe_up", il);
    cb(gate, "ffn_moe_gate", il);

    ggml_tensor * experts = ggml_swiglu_split(ctx0, gate, up);
    cb(experts, "ffn_moe_swiglu", il);

    experts = ggml_mul_mat_id(ctx0, layer.ffn_down_exps, experts, ids); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx0, experts, weights);
    cb(experts, "ffn_moe_weighted", il);

    ggml_tensor * moe_out = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], 0);
    for (int64_t i = 1; i < n_expert_used; ++i) {
        ggml_tensor * cur_expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
        moe_out = ggml_add(ctx0, moe_out, cur_expert);
    }

    // an identity zero expert contributes weight_i * cur
    ggml_tensor * w_zero = ggml_mul(ctx0, ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens), zero_mask);
    w_zero = ggml_sum_rows(ctx0, w_zero); // [1, n_tokens]
    cb(w_zero, "ffn_moe_weights_zero", il);

    moe_out = ggml_add(ctx0, moe_out, ggml_mul(ctx0, cur, w_zero));
    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}
