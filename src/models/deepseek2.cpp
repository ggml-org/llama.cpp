#include "models.h"

void llama_model_deepseek2::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    // lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B, Kanana-2-30B-A3B
    const bool is_lite = (hparams.n_layer() == 27 || hparams.n_layer() == 26 || (hparams.n_layer() == 48 && n_vocab == 128256));

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    if (!is_lite) {
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
    }
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        // for compatibility with existing DeepSeek V2 and V2.5 GGUFs
        // that have no expert_gating_func model parameter set
        if ((hparams.n_layer() == 47 || hparams.n_layer() == 48) && n_vocab == 154880) {
            // GLM 4.7 Lite
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
        } else {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
        }
    }

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    // NextN/MTP
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn == 0 ||
        hparams.n_layer() + hparams.n_layer_nextn == hparams.n_layer_all);

    // (optional) temperature tuning - used by mistral-large
    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE,  hparams.f_attn_temp_scale,       false);
    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.n_attn_temp_floor_scale, false); // FIXME why not use temperature_length?

    hparams.f_attn_temp_offset = 0.0f;

    switch (hparams.n_layer()) {
        case 27: type = LLM_TYPE_16B; break;
        case 47: type = LLM_TYPE_30B_A3B; break;
        case 60: type = LLM_TYPE_236B; break;
        case 61: type = LLM_TYPE_671B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const bool is_mla = hparams.is_mla();

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        if (q_lora_rank > 0) {
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        }

        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

        if (q_lora_rank > 0) {
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);
        }

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        if (is_mla) {
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
        } else {
            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, 0);
        }

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, 0);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }

        // NextN/MTP tensors
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, 0);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, 0);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, 0);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek2::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_deepseek2::graph_mtp::build_mtp_input_embd_h(
        const llama_model & model,
        const llama_layer & layer,
        ggml_tensor ** h_embd,
        int il) {
    GGML_ASSERT(h_embd != nullptr);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd = nullptr;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens
                ? layer.nextn.embed_tokens
                : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    *h_embd = inp->h;

    res->add_input(std::move(inp));

    return tok_embd;
}

ggml_tensor * llama_model_deepseek2::graph_mtp::build_mtp_fusion(
        ggml_tensor * tok_embd,
        ggml_tensor * h_embd,
        const llama_layer & layer,
        int il) {
    GGML_ASSERT(tok_embd != nullptr);
    GGML_ASSERT(h_embd   != nullptr);

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    return cur;
}

ggml_tensor * llama_model_deepseek2::graph_mtp::build_mtp_moe_ffn(
        const llama_layer & layer,
        ggml_tensor * ffn_inp,
        int il) {
    GGML_ASSERT(ffn_inp != nullptr);

    const bool has_split_exps =
            layer.ffn_up_exps   != nullptr &&
            layer.ffn_gate_exps != nullptr;

    const bool has_fused_exps =
            layer.ffn_gate_up_exps != nullptr;

    GGML_ASSERT(has_split_exps || has_fused_exps);

    GGML_ASSERT(layer.ffn_norm     != nullptr);
    GGML_ASSERT(layer.ffn_gate_inp != nullptr);
    GGML_ASSERT(layer.ffn_down_exps != nullptr);
    GGML_ASSERT(layer.ffn_gate_shexp != nullptr);
    GGML_ASSERT(layer.ffn_down_shexp != nullptr);
    GGML_ASSERT(layer.ffn_up_shexp != nullptr);

    ggml_tensor * cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            layer.ffn_gate_up_exps);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    return cur;
}

llama_model_deepseek2::graph_mtp::mtp_mla_q
llama_model_deepseek2::graph_mtp::build_mtp_mla_q(
        const llama_layer & layer,
        ggml_tensor * cur,
        int il) {
    GGML_ASSERT(cur != nullptr);

    GGML_ASSERT(hparams.n_lora_q > 0);
    GGML_ASSERT(layer.wq_a);
    GGML_ASSERT(layer.attn_q_a_norm);
    GGML_ASSERT(layer.wq_b);

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(q, "mtp_q_a", il);

    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q, "mtp_q_a_norm", il);

    q = ggml_mul_mat(ctx0, layer.wq_b, q);
    cb(q, "mtp_q_b", il);

    ggml_tensor * q_nope =
        ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
    cb(q_nope, "mtp_q_nope", il);

    ggml_tensor * q_pe =
        ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_pe, "mtp_q_pe", il);

    return { q_nope, q_pe };
}

llama_model_deepseek2::graph_mtp::mtp_mla_kv
llama_model_deepseek2::graph_mtp::build_mtp_mla_kv(
        const llama_layer & layer,
        ggml_tensor * cur,
        int il) {
    GGML_ASSERT(cur != nullptr);
    GGML_ASSERT(layer.wkv_a_mqa);
    GGML_ASSERT(layer.attn_kv_a_norm);

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t kv_lora_rank        = hparams.n_lora_kv;

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr =
        ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    cb(kv_cmpr, "mtp_kv_cmpr", il);

    ggml_tensor * k_pe =
        ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
    cb(k_pe, "mtp_k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "mtp_kv_cmpr_norm", il);

    return { kv_cmpr, k_pe };
}

ggml_tensor * llama_model_deepseek2::graph_mtp::build_mtp_mla_attn(
        const llama_layer & layer,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        llm_graph_input_attn_k * inp_attn_k,
        int il) {
    GGML_ASSERT(cur != nullptr);
    GGML_ASSERT(inp_pos != nullptr);
    GGML_ASSERT(inp_attn_k != nullptr);

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();

    GGML_ASSERT(ext_factor >= 0.0f);

    const float attn_factor_org =
            attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    const float mscale =
            attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));

    const float kq_scale =
            1.0f * mscale * mscale / sqrtf(float(n_embd_head_k_mla));

    mtp_mla_q q = build_mtp_mla_q(layer, cur, il);
    mtp_mla_kv kv = build_mtp_mla_kv(layer, cur, il);

    q.pe = ggml_rope_ext(ctx0, q.pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q.pe, "mtp_q_pe_rope", il);

    kv.pe = ggml_rope_ext(ctx0, kv.pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(kv.pe, "mtp_k_pe_rope", il);

    GGML_ASSERT(layer.wk_b);

    q.nope = ggml_permute(ctx0, q.nope, 0, 2, 1, 3);
    cb(q.nope, "mtp_q_nope_perm", il);

    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q.nope);
    cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q.pe, 0);
    cb(Qcur, "mtp_Qcur", il);

    kv.cmpr = ggml_reshape_3d(ctx0, kv.cmpr, hparams.n_lora_kv, 1, n_tokens);
    cb(kv.cmpr, "mtp_kv_cmpr_reshape", il);

    ggml_tensor * Kcur = ggml_concat(ctx0, kv.cmpr, kv.pe, 0);
    cb(Kcur, "mtp_Kcur", il);

    ggml_tensor * Vcur = kv.cmpr;
    cb(Vcur, "mtp_Vcur", il);

    cur = build_attn(inp_attn_k,
            layer.wo, nullptr, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(cur, "mtp_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_deepseek2::graph_mtp::build_mtp_logits_head(
        const llama_model & model,
        const llama_layer & layer,
        ggml_tensor * cur,
        ggml_tensor * inp_out_ids) {
    GGML_ASSERT(cur != nullptr);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "GLM4 MTP: missing both nextn.shared_head_norm and output_norm");

    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head
            ? layer.nextn.shared_head_head
            : model.output;

    ggml_tensor * head_s = layer.nextn.shared_head_head
            ? layer.nextn.shared_head_head_s
            : model.output_s;

    GGML_ASSERT(head_w && "GLM4 MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    return cur;
}

llama_model_deepseek2::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    GGML_ASSERT(hparams.n_layer_nextn > 0 && "GLM4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "GLM4 MTP currently only supports a single MTP block");
    GGML_ASSERT(hparams.is_mla() && "GLM4 MTP requires MLA");
    GGML_ASSERT(hparams.f_attn_temp_scale == 0.0f && "GLM4 MTP does not support attention temperature scaling");

    // The appended MTP block is stored immediately after the main decoder layers.
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    GGML_ASSERT((uint32_t) il >= hparams.n_layer_dense_lead && "GLM4 MTP block expected to use MoE FFN");

    ggml_tensor * h_embd = nullptr;
    ggml_tensor * tok_embd = build_mtp_input_embd_h(model, layer, &h_embd, il);

    ggml_tensor * cur = build_mtp_fusion(tok_embd, h_embd, layer, il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn_k = build_attn_inp_k();

    cur = build_mtp_mla_attn(layer, cur, inp_pos, inp_attn_k, il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_mtp_moe_ffn(layer, ffn_inp, il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    cur = build_mtp_logits_head(model, layer, cur, inp_out_ids);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

llama_model_deepseek2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    // lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B
    bool is_ocr = model.arch == LLM_ARCH_DEEPSEEK2OCR;

    const bool is_mla = hparams.is_mla();

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // (optional) temperature tuning - used by mistral-large
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_inp_attn_scale();
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_kv = !is_mla ? build_attn_inp_kv() : nullptr;
    auto * inp_attn_k  =  is_mla ? build_attn_inp_k()  : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        if (is_ocr) {
            const int n_embed_head = hparams.n_embd / hparams.n_head();
            const int ocr_rope_type = GGML_ROPE_TYPE_NEOX;
            GGML_ASSERT(n_embed_head == n_embd_head_k && n_embed_head == n_embd_head_v);

            ggml_tensor * Qcur = NULL;
            ggml_tensor * Kcur = NULL;
            ggml_tensor * Vcur = NULL;

            Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
            cb(Qcur, "q", il);
            cb(Kcur, "k", il);
            cb(Vcur, "v", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embed_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embed_head, n_head, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embed_head, n_head, n_tokens);

            GGML_ASSERT(fabs(freq_base - 10000.0) < 1e-4);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_embed_head, ocr_rope_type, 0, freq_base, 1, 0, 1, 0, 0);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_embed_head, ocr_rope_type, 0, freq_base, 1, 0, 1, 0, 0);
            cb(Qcur, "q_pe", il);
            cb(Kcur, "k_pe", il);

            cur = build_attn(inp_attn_kv,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }
        else {
            ggml_tensor * q = NULL;

            const bool is_lite = model.layers[il].wq;

            if (!is_lite) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
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

            if (is_mla) {
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

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
                cur = build_attn(inp_attn_k,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            } else {
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                cb(kv, "kv", il);

                // split into {n_embd_head_qk_nope, n_head, n_tokens}
                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
                cb(k_nope, "k_nope_view", il);

                // and {n_embd_head_v, n_head, n_tokens}
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope));
                cb(Vcur, "Vcur_view", il);

                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "Vcur_cont", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(Kcur, "Kcur", il);

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // note: MLA without the absorption optimization converts into MHA (ie: GQA with full n_head groups)
                cur = build_attn(inp_attn_kv,
                            model.layers[il].wo, NULL, model.layers[il].wo_s,
                            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
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
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
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
                model.layers[il].ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
