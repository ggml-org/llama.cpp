#include "models.h"
#include "llama-memory-recurrent.h"

#include <cmath>
#include <stdexcept>

void llama_model_gigachat35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_all");

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    ml.get_key(LLM_KV_SSM_DT_B_C_RMS,     hparams.ssm_dt_b_c_rms, false);

    // Explicit per-layer mask
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // Undo the converter-side 0.1 multiplier
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    type = LLM_TYPE_432B_A28B;
}

void llama_model_gigachat35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only  ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    auto create_tensor_from_meta = [&](const LLM_TN_IMPL & tname, int flags) -> ggml_tensor * {
        const ggml_tensor * meta = ml.get_tensor_meta(tname.str().c_str());
        if (meta == nullptr) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error("missing tensor '" + tname.str() + "'");
        }

        switch (ggml_n_dims(meta)) {
            case 1: return create_tensor(tname, { meta->ne[0] }, flags);
            case 2: return create_tensor(tname, { meta->ne[0], meta->ne[1] }, flags);
            case 3: return create_tensor(tname, { meta->ne[0], meta->ne[1], meta->ne[2] }, flags);
            case 4: return create_tensor(tname, { meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3] }, flags);
            default:
                throw std::runtime_error("GigaChat35: unsupported tensor rank for " + tname.str());
        }
    };

    auto load_norm = [&](ggml_tensor * & norm, ggml_tensor * & gate_up, ggml_tensor * & gate_down,
            llm_tensor norm_id, llm_tensor gate_up_id, llm_tensor gate_down_id, int il, int64_t dim, int flags) {
        norm      = create_tensor(tn(norm_id,      "weight", il), { dim }, flags);
        gate_up   = create_tensor_from_meta(tn(gate_up_id,   "weight", il), flags);
        gate_down = create_tensor_from_meta(tn(gate_down_id, "weight", il), flags);
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm           = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output_norm_gate_up   = create_tensor_from_meta(tn(LLM_TENSOR_OUTPUT_NORM_GATE_UP,   "weight"), 0);
    output_norm_gate_down = create_tensor_from_meta(tn(LLM_TENSOR_OUTPUT_NORM_GATE_DOWN, "weight"), 0);
    output                = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, trunk_flags);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp ? hparams.n_ff_exp : (n_expert_used > 0 ? n_ff / n_expert_used : 0);
    const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp * (hparams.n_expert_shared ? hparams.n_expert_shared : 1);
    GGML_ASSERT(n_ff_exp > 0);

    const int64_t n_embd_head_k_mla      = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla      = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope    = hparams.n_rot();
    const int64_t n_embd_head_qk_nope    = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t q_lora_rank            = hparams.n_lora_q;
    const int64_t kv_lora_rank           = hparams.n_lora_kv;

    GGML_ASSERT(n_embd_head_qk_nope >= 1);
    GGML_ASSERT(kv_lora_rank > 0);

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = 2 * key_dim + value_dim;

    auto load_common_norms = [&](int il, int flags) {
        auto & layer = layers[il];
        load_norm(layer.attn_norm, layer.attn_norm_gate_up, layer.attn_norm_gate_down,
                LLM_TENSOR_ATTN_NORM, LLM_TENSOR_ATTN_NORM_GATE_UP, LLM_TENSOR_ATTN_NORM_GATE_DOWN,
                il, n_embd, flags);
        load_norm(layer.attn_post_norm, layer.attn_post_norm_gate_up, layer.attn_post_norm_gate_down,
                LLM_TENSOR_ATTN_POST_NORM, LLM_TENSOR_ATTN_POST_NORM_GATE_UP, LLM_TENSOR_ATTN_POST_NORM_GATE_DOWN,
                il, n_embd, flags);
        load_norm(layer.ffn_norm, layer.ffn_norm_gate_up, layer.ffn_norm_gate_down,
                LLM_TENSOR_FFN_NORM, LLM_TENSOR_FFN_NORM_GATE_UP, LLM_TENSOR_FFN_NORM_GATE_DOWN,
                il, n_embd, flags);
        load_norm(layer.ffn_post_norm, layer.ffn_post_norm_gate_up, layer.ffn_post_norm_gate_down,
                LLM_TENSOR_FFN_POST_NORM, LLM_TENSOR_FFN_POST_NORM_GATE_UP, LLM_TENSOR_FFN_POST_NORM_GATE_DOWN,
                il, n_embd, flags);
    };

    auto load_mla_attention = [&](int il, int flags) {
        auto & layer = layers[il];
        const int64_t n_head_l = hparams.n_head(il);

        if (q_lora_rank > 0) {
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, flags);
            load_norm(layer.attn_q_a_norm, layer.attn_q_a_norm_gate_up, layer.attn_q_a_norm_gate_down,
                    LLM_TENSOR_ATTN_Q_A_NORM, LLM_TENSOR_ATTN_Q_A_NORM_GATE_UP, LLM_TENSOR_ATTN_Q_A_NORM_GATE_DOWN,
                    il, q_lora_rank, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head_l * n_embd_head_k_mla }, flags);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head_l * n_embd_head_k_mla }, flags);
        }

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_lora_rank + n_embd_head_qk_rope }, flags);
        load_norm(layer.attn_kv_a_norm, layer.attn_kv_a_norm_gate_up, layer.attn_kv_a_norm_gate_down,
                LLM_TENSOR_ATTN_KV_A_NORM, LLM_TENSOR_ATTN_KV_A_NORM_GATE_UP, LLM_TENSOR_ATTN_KV_A_NORM_GATE_DOWN,
                il, kv_lora_rank, flags);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { n_embd_head_qk_nope, kv_lora_rank, n_head_l }, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { kv_lora_rank, n_embd_head_v_mla, n_head_l }, flags);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head_l * n_embd_head_v_mla, n_embd }, flags);

        layer.wqkv_gate = create_tensor_from_meta(tn(LLM_TENSOR_ATTN_GATE, "weight", il), flags);
    };

    auto load_linear_attention = [&](int il, int flags) {
        auto & layer = layers[il];
        layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, conv_dim }, flags);
        layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
        layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
        layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
        layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
        layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
        layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
        layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
    };

    auto load_dense_ffn = [&](int il, int flags) {
        auto & layer = layers[il];
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff,   n_embd }, flags);
    };

    auto load_moe_ffn = [&](int il, int flags) {
        auto & layer = layers[il];
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   il), { n_expert }, flags);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd,    n_ff_shexp }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", il), { n_embd,    n_ff_shexp }, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    auto load_block = [&](int il, int flags, bool is_mtp) {
        load_common_norms(il, flags);

        if (is_mtp || !hparams.is_recr(il)) {
            load_mla_attention(il, flags);
        } else {
            load_linear_attention(il, flags);
        }

        if (il < (int) hparams.n_layer_dense_lead || is_mtp) {
            load_dense_ffn(il, flags);
        } else {
            load_moe_ffn(il, flags);
        }
    };

    for (int il = 0; il < n_layer; ++il) {
        load_block(il, trunk_flags, false);
    }

    for (int il = n_layer; il < n_layer_all; ++il) {
        const int eff_mtp_flags = mtp_flags;
        auto & layer = layers[il];

        load_block(il, eff_mtp_flags, true);

        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, eff_mtp_flags);
        load_norm(layer.nextn.enorm, layer.nextn.enorm_gate_up, layer.nextn.enorm_gate_down,
                LLM_TENSOR_NEXTN_ENORM, LLM_TENSOR_NEXTN_ENORM_GATE_UP, LLM_TENSOR_NEXTN_ENORM_GATE_DOWN,
                il, n_embd, eff_mtp_flags);
        load_norm(layer.nextn.hnorm, layer.nextn.hnorm_gate_up, layer.nextn.hnorm_gate_down,
                LLM_TENSOR_NEXTN_HNORM, LLM_TENSOR_NEXTN_HNORM_GATE_UP, LLM_TENSOR_NEXTN_HNORM_GATE_DOWN,
                il, n_embd, eff_mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
        load_norm(layer.nextn.shared_head_norm, layer.nextn.shared_head_norm_gate_up, layer.nextn.shared_head_norm_gate_down,
                LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,
                LLM_TENSOR_NEXTN_SHARED_HEAD_NORM_GATE_UP,
                LLM_TENSOR_NEXTN_SHARED_HEAD_NORM_GATE_DOWN,
                il, n_embd, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gigachat35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }

    return std::make_unique<graph>(*this, params);
}

llama_model_gigachat35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    graph_common(model, params) {
    GGML_ASSERT(hparams.is_mla());

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    GGML_ASSERT(n_embd_head_k_mla > hparams.n_rot());

    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale          = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale        = mscale * mscale / sqrtf(float(n_embd_head_k_mla));

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid_k();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_zero_centered_gated_norm(
                inpL, layer.attn_norm, layer.attn_norm_gate_up, layer.attn_norm_gate_down, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn_mla(inp->get_attn(), cur, inp_pos, kq_scale, il);
        }
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = build_zero_centered_gated_norm(
                cur, layer.attn_post_norm, layer.attn_post_norm_gate_up, layer.attn_post_norm_gate_down, il);
        cb(cur, "attn_post_norm", il);

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "attn_residual", il);

        residual = cur;
        cur = build_zero_centered_gated_norm(
                cur, layer.ffn_norm, layer.ffn_norm_gate_up, layer.ffn_norm_gate_down, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = build_zero_centered_gated_norm(
                cur, layer.ffn_post_norm, layer.ffn_post_norm_gate_up, layer.ffn_post_norm_gate_down, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "ffn_residual", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    // the MTP head reads the hidden state before the final norm
    res->t_h_nextn = inpL;
    cb(inpL, "h_nextn", -1);

    ggml_tensor * cur = build_zero_centered_gated_norm(
            inpL, model.output_norm, model.output_norm_gate_up, model.output_norm_gate_down, -1);

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_gigachat35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph_common(model, params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "GigaChat35 MTP requires n_layer_nextn > 0");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "GigaChat35 MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "GigaChat35 MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "GigaChat35 MTP block missing nextn.hnorm");

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    GGML_ASSERT(n_embd_head_k_mla > hparams.n_rot());

    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale          = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale        = mscale * mscale / sqrtf(float(n_embd_head_k_mla));

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
    auto        * inp_attn    = build_attn_inp_k();

    ggml_tensor * h_norm = build_zero_centered_gated_norm(
            h_embd, layer.nextn.hnorm, layer.nextn.hnorm_gate_up, layer.nextn.hnorm_gate_down, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_zero_centered_gated_norm(
            tok_embd, layer.nextn.enorm, layer.nextn.enorm_gate_up, layer.nextn.enorm_gate_down, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * residual = cur;
    cur = build_zero_centered_gated_norm(
            cur, layer.attn_norm, layer.attn_norm_gate_up, layer.attn_norm_gate_down, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_layer_attn_mla(inp_attn, cur, inp_pos, kq_scale, il);
    cb(cur, "mtp_attn_out", il);

    cur = build_zero_centered_gated_norm(
            cur, layer.attn_post_norm, layer.attn_post_norm_gate_up, layer.attn_post_norm_gate_down, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "mtp_attn_residual", il);

    residual = cur;
    cur = build_zero_centered_gated_norm(
            cur, layer.ffn_norm, layer.ffn_norm_gate_up, layer.ffn_norm_gate_down, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = build_zero_centered_gated_norm(
            cur, layer.ffn_post_norm, layer.ffn_post_norm_gate_up, layer.ffn_post_norm_gate_down, il);
    cb(cur, "mtp_ffn_post_norm", il);

    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "mtp_ffn_residual", il);

    // pre-norm hidden state: seeds the next MTP draft step
    res->t_h_nextn = cur;
    cb(cur, "h_nextn", -1);

    if (layer.nextn.shared_head_norm) {
        cur = build_zero_centered_gated_norm(
                cur,
                layer.nextn.shared_head_norm,
                layer.nextn.shared_head_norm_gate_up,
                layer.nextn.shared_head_norm_gate_down,
                -1);
    } else {
        cur = build_zero_centered_gated_norm(
                cur,
                model.output_norm,
                model.output_norm_gate_up,
                model.output_norm_gate_down,
                -1);
    }

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "GigaChat35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_gigachat35::graph_common::build_zero_centered_norm(
        ggml_tensor * input,
        ggml_tensor * weight,
        int           il) {
    GGML_ASSERT(weight != nullptr);

    ggml_tensor * cur = build_norm(input, nullptr, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * scale = ggml_scale_bias(ctx0, weight, 1.0f, 1.0f);

    return ggml_mul(ctx0, cur, scale);
}

ggml_tensor * llama_model_gigachat35::graph_common::build_zero_centered_gated_norm(
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * gate_up,
        ggml_tensor * gate_down,
        int           il) {
    GGML_ASSERT(gate_up   != nullptr);
    GGML_ASSERT(gate_down != nullptr);

    ggml_tensor * cur = build_zero_centered_norm(input, weight, il);
    ggml_tensor * gate = build_lora_mm(gate_up, cur);
    gate = ggml_silu(ctx0, gate);
    gate = build_lora_mm(gate_down, gate);
    gate = ggml_sigmoid(ctx0, gate);

    cur = ggml_mul(ctx0, cur, gate);
    cur = ggml_scale(ctx0, cur, 2.0f);

    return cur;
}

ggml_tensor * llama_model_gigachat35::graph::build_linear_output_norm(
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * gate,
        int           il) {
    ggml_tensor * cur = build_zero_centered_norm(input, weight, il);
    gate = ggml_sigmoid(ctx0, gate);
    cur = ggml_mul(ctx0, cur, gate);
    cur = ggml_scale(ctx0, cur, 2.0f);

    return cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_gigachat35::graph::build_qkvz(
        ggml_tensor * input,
        int           il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv = ggml_reshape_3d(ctx0, qkv, qkv->ne[0], n_seq_tokens, n_seqs);
    cb(qkv, "linear_attn_qkv", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "linear_attn_z", il);

    return { qkv, z };
}

ggml_tensor * llama_model_gigachat35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(d_inner == head_v_dim * num_v_heads);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv = qkvz.first;
    ggml_tensor * z   = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "linear_attn_beta", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    alpha = ggml_softplus(ctx0, alpha);
    cb(alpha, "linear_attn_alpha", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha, model.layers[il].ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(gate, "linear_attn_gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    const int64_t conv_kernel_size = model.layers[il].ssm_conv1d->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;
    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "linear_attn_state", il);

    ggml_tensor * conv = ggml_ssm_conv(ctx0, conv_input, model.layers[il].ssm_conv1d);
    conv = ggml_silu(ctx0, conv);
    cb(conv, "linear_attn_conv", il);

    const int64_t key_dim = head_k_dim * num_k_heads;
    const int64_t qkv_dim = 2 * key_dim + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv->type, qkv_dim);

    ggml_tensor * q = ggml_view_4d(ctx0, conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv));
    ggml_tensor * v = ggml_view_4d(ctx0, conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv->type, 2 * key_dim));

    q = ggml_l2_norm(ctx0, q, hparams.f_norm_rms_eps);
    k = ggml_l2_norm(ctx0, k, hparams.f_norm_rms_eps);

    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        const int64_t repeat_factor = num_v_heads / num_k_heads;

        ggml_tensor * q_reshaped = ggml_reshape_4d(ctx0, q,
                head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);
        ggml_tensor * k_reshaped = ggml_reshape_4d(ctx0, k,
                head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);

        ggml_tensor * q_repeated = ggml_repeat_4d(ctx0, q_reshaped,
                head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);
        ggml_tensor * k_repeated = ggml_repeat_4d(ctx0, k_reshaped,
                head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);

        q = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q, "linear_attn_q", il);
    cb(k, "linear_attn_k", il);
    cb(v, "linear_attn_v", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q, k, v, gate, beta, state, il);

    z = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    output = build_linear_output_norm(output, model.layers[il].ssm_norm, z, il);
    cb(output, "linear_attn_norm", il);

    output = ggml_reshape_3d(ctx0, output, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cur = build_lora_mm(model.layers[il].ssm_out, output, model.layers[il].ssm_out_s);
    cur = ggml_reshape_2d(ctx0, cur, hparams.n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_gigachat35::graph_common::build_layer_attn_mla(
        llm_graph_input_attn_k * inp_attn,
        ggml_tensor *           cur,
        ggml_tensor *           inp_pos,
        float                   kq_scale,
        int                     il) {
    const auto & layer = model.layers[il];

    const int64_t n_head_l              = hparams.n_head(il);
    const int64_t n_embd_head_k_mla     = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope   = hparams.n_rot();
    const int64_t n_embd_head_qk_nope   = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t q_lora_rank           = hparams.n_lora_q;
    const int64_t kv_lora_rank          = hparams.n_lora_kv;

    GGML_ASSERT(hparams.n_head_kv(il) == 1);
    GGML_ASSERT(n_embd_head_qk_nope > 0);
    GGML_ASSERT(kv_lora_rank > 0);

    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur, layer.wqkv_gate_s);
    cb(gate, "attn_gate", il);

    ggml_tensor * q = nullptr;
    if (q_lora_rank > 0) {
        q = build_lora_mm(layer.wq_a, cur);
        q = build_zero_centered_gated_norm(q, layer.attn_q_a_norm, layer.attn_q_a_norm_gate_up, layer.attn_q_a_norm_gate_down, il);
        q = ggml_scale(ctx0, q, sqrtf(float(hparams.n_embd) / float(q_lora_rank)));
        q = build_lora_mm(layer.wq_b, q);
    } else {
        q = build_lora_mm(layer.wq, cur);
    }
    cb(q, "q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q,
            n_embd_head_qk_nope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q,
            n_embd_head_qk_rope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv = build_lora_mm(layer.wkv_a_mqa, cur);
    cb(kv, "kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv,
            kv_lora_rank, n_tokens,
            ggml_row_size(kv->type, kv_lora_rank + n_embd_head_qk_rope),
            0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv,
            n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv->type, kv_lora_rank));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe", il);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_zero_centered_gated_norm(kv_cmpr,
            layer.attn_kv_a_norm, layer.attn_kv_a_norm_gate_up, layer.attn_kv_a_norm_gate_down, il);
    kv_cmpr = ggml_scale(ctx0, kv_cmpr, sqrtf(float(hparams.n_embd) / float(kv_lora_rank)));
    cb(kv_cmpr, "kv_cmpr", il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "q_nope_absorbed", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Qcur, "Qcur", il);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(cur, "attn_pregate", il);

    gate = ggml_sigmoid(ctx0, gate);
    cur = ggml_mul(ctx0, cur, gate);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    return cur;
}

ggml_tensor * llama_model_gigachat35::graph::build_layer_ffn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    if (il < int(hparams.n_layer_dense_lead)) {
        return build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU,
            hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_up_shexp == nullptr) {
        return moe_out;
    }

    ggml_tensor * shared = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(shared, "ffn_shared", il);

    return ggml_add(ctx0, moe_out, shared);
}
