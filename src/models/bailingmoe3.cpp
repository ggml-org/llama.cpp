#include "models.h"
#include "llama-memory-recurrent.h"

#include <algorithm>

void llama_model_bailingmoe3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,         hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,       hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,           hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,            hparams.n_lora_q, false);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,                  hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                     hparams.n_embd_head_kda);
    if (!ml.get_key(LLM_KV_KDA_SAFE_GATE, hparams.kda_safe_gate, false)) {
        hparams.kda_safe_gate = true;
    }
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,             hparams.kda_gate_lower_bound);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,       hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,              hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,        hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,             hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,              hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,               hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,           hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,         hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp * std::max(1u, hparams.n_expert_shared);
    }

    GGML_ASSERT(hparams.kda_safe_gate);
    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1536 && hparams.n_expert == 128 ? LLM_TYPE_7_9B_A1_3B : LLM_TYPE_UNKNOWN; break;
        case 42: type = hparams.n_embd == 2560 && hparams.n_expert == 512 ? LLM_TYPE_124B_A5_1B : LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bailingmoe3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner = head_dim * n_head;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_head_dim = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim = hparams.n_embd_head_v_mla();

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int       mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, trunk_flags);

        if (hparams.is_recr(il)) {
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", il), { d_conv, 1, d_inner, 1 }, trunk_flags);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", il), { d_conv, 1, d_inner, 1 }, trunk_flags);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", il), { d_conv, 1, d_inner, 1 }, trunk_flags);

            create_tensor_qkv(layer, il, n_embd, d_inner, d_inner, d_inner, trunk_flags);
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", il), { n_embd, d_inner }, trunk_flags);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), { n_embd, n_head }, trunk_flags);
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, il), { 1, n_head }, trunk_flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", il), { d_inner }, trunk_flags);
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", il), { n_embd, d_inner }, trunk_flags);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), { head_dim }, trunk_flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { d_inner, n_embd }, trunk_flags);
        } else {
            if (q_lora_rank > 0) {
                layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, trunk_flags);
                layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, trunk_flags);
                layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head * qk_head_dim }, trunk_flags);
            } else {
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head * qk_head_dim }, trunk_flags);
            }
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_lora_rank + qk_rope_head_dim }, trunk_flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { kv_lora_rank }, trunk_flags);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { qk_head_dim - qk_rope_head_dim, kv_lora_rank, n_head }, trunk_flags);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { kv_lora_rank, v_head_dim, n_head }, trunk_flags);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, n_head }, trunk_flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head * v_head_dim, n_embd }, trunk_flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, trunk_flags);
        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, trunk_flags);
            layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", il), { n_embd, n_ff }, trunk_flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, trunk_flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, trunk_flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, trunk_flags);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, hparams.n_ff_exp, n_expert }, trunk_flags);
            layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, hparams.n_ff_exp, n_expert }, trunk_flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { hparams.n_ff_exp, n_embd, n_expert }, trunk_flags);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, trunk_flags);
            layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, trunk_flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { hparams.n_ff_shexp, n_embd }, trunk_flags);
        }
    }

    for (int il = n_layer; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        const int flags = mtp_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, flags);
        if (q_lora_rank > 0) {
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, flags);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head * qk_head_dim }, flags);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head * qk_head_dim }, flags);
        }
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_lora_rank + qk_rope_head_dim }, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { kv_lora_rank }, flags);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { qk_head_dim - qk_rope_head_dim, kv_lora_rank, n_head }, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { kv_lora_rank, v_head_dim, n_head }, flags);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, n_head }, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head * v_head_dim, n_embd }, flags);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, flags);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, flags);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, hparams.n_ff_exp, n_expert }, flags);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, hparams.n_ff_exp, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { hparams.n_ff_exp, n_embd, n_expert }, flags);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, flags);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { hparams.n_ff_shexp, n_embd }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);
        layer.nextn.enorm = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", il), { n_embd }, flags);
        layer.nextn.hnorm = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", il), { n_embd }, flags);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", il), { n_embd }, flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_bailingmoe3::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static ggml_tensor * bailingmoe3_causal_conv1d(
        ggml_cgraph * gf,
        ggml_context * ctx0,
        ggml_tensor * conv_states_all,
        ggml_tensor * conv_state_all,
        int64_t qkv,
        ggml_tensor * x,
        ggml_tensor * proj_w,
        ggml_tensor * conv_w,
        int64_t d_conv,
        int64_t head_dim,
        int64_t n_head,
        int64_t n_seq_tokens,
        int64_t n_seqs,
        int64_t n_tokens,
        int64_t cache_head,
        uint32_t mem_size,
        uint32_t n_rs_seq) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t total_state_size = 3 * conv_state_size;

    ggml_tensor * conv_state = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
            (d_conv - 1) * ggml_element_size(conv_state_all),
            total_state_size * ggml_element_size(conv_state_all),
            qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    x_proj = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state, ggml_transpose(ctx0, x_proj), 0);

    const int64_t K = (int64_t) n_rs_seq + 1;
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    for (int64_t slot = 0; slot < n_written; ++slot) {
        ggml_tensor * conv_snap = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
                conv_x->nb[1], conv_x->nb[2], (conv_x->ne[0] - (d_conv - 1) - slot) * conv_x->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_snap,
                ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1) * ggml_element_size(conv_states_all),
                    total_state_size * ggml_element_size(conv_states_all),
                    ((slot * mem_size + cache_head) * total_state_size + qkv * conv_state_size) * ggml_element_size(conv_states_all))));
    }

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    out = ggml_silu(ctx0, ggml_reshape_2d(ctx0, out, d_inner, n_tokens));
    return ggml_reshape_4d(ctx0, out, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_bailingmoe3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid_k();
    auto * inp_rs = inp->get_recr();
    auto * inp_attn = inp->get_attn();

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner = n_head * head_dim;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t qk_head_dim = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim = hparams.n_embd_head_v_mla();
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const float kq_scale = 1.0f / sqrtf((float) qk_head_dim);

    GGML_ASSERT(n_seqs > 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            const auto * mctx_cur = inp_rs->mctx;
            const auto cache_head = mctx_cur->get_head();
            const auto mem_size = mctx_cur->get_size();
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            ggml_tensor * q = bailingmoe3_causal_conv1d(
                    gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv,
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, cache_head, mem_size, cparams.n_rs_seq);
            ggml_tensor * k = bailingmoe3_causal_conv1d(
                    gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv,
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, cache_head, mem_size, cparams.n_rs_seq);
            ggml_tensor * v = bailingmoe3_causal_conv1d(
                    gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv,
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, cache_head, mem_size, cparams.n_rs_seq);

            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            gate = ggml_add(ctx0, gate, layer.ssm_dt_b);
            gate = ggml_reshape_3d(ctx0, gate, head_dim, n_head, n_tokens);
            ggml_tensor * a = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            gate = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, a)), hparams.kda_gate_lower_bound);
            gate = ggml_reshape_4d(ctx0, gate, head_dim, n_head, n_seq_tokens, n_seqs);
            cb(gate, "kda_gate", il);

            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs));

            q = ggml_l2_norm(ctx0, q, hparams.f_norm_rms_eps);
            k = ggml_l2_norm(ctx0, k, hparams.f_norm_rms_eps);

            ggml_tensor * states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            ggml_tensor * out = ggml_cont(ctx0, build_recurrent_attn(
                    inp_rs, states_all, q, k, v, gate, beta, state, il));

            ggml_tensor * out_gate = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
            out_gate = ggml_reshape_3d(ctx0, out_gate, head_dim, n_head, n_tokens);
            out = ggml_reshape_3d(ctx0, out, head_dim, n_head, n_tokens);
            out = build_norm(out, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, out_gate));
            cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, out, d_inner, n_tokens));
            cb(cur, "kda_out", il);
        } else {
            ggml_tensor * attn_input = cur;
            ggml_tensor * q_all;
            if (layer.wq_a) {
                q_all = ggml_mul_mat(ctx0, layer.wq_a, cur);
                cb(q_all, "q_a", il);
                q_all = build_norm(q_all, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q_all, "q_a_norm", il);
                q_all = ggml_mul_mat(ctx0, layer.wq_b, q_all);
                cb(q_all, "q_b", il);
            } else {
                q_all = ggml_mul_mat(ctx0, layer.wq, cur);
            }
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q_all, qk_nope_head_dim, n_head, n_tokens,
                    ggml_row_size(q_all->type, qk_head_dim),
                    ggml_row_size(q_all->type, qk_head_dim) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q_all, qk_rope_head_dim, n_head, n_tokens,
                    ggml_row_size(q_all->type, qk_head_dim),
                    ggml_row_size(q_all->type, qk_head_dim) * n_head,
                    ggml_row_size(q_all->type, qk_nope_head_dim));

            ggml_tensor * kv_all = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            ggml_tensor * kv = ggml_view_2d(ctx0, kv_all, kv_lora_rank, n_tokens,
                    ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_all, qk_rope_head_dim, 1, n_tokens,
                    ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
                    ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
                    ggml_row_size(kv_all->type, kv_lora_rank));

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            q_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

            ggml_tensor * q = ggml_concat(ctx0, q_nope, q_pe, 0);
            kv = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
            ggml_tensor * k = ggml_concat(ctx0, kv, k_pe, 0);

            cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                    q, k, kv, nullptr, nullptr, layer.wv_b, kq_scale, il);

            ggml_tensor * attn_gate = ggml_mul_mat(ctx0, layer.wqkv_gate, attn_input);
            attn_gate = ggml_sigmoid(ctx0, ggml_reshape_3d(ctx0, attn_gate, 1, n_head, n_tokens));
            cur = ggml_reshape_3d(ctx0, cur, v_head_dim, n_head, n_tokens);
            cur = ggml_mul(ctx0, cur, attn_gate);
            cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, cur, v_head_dim * n_head, n_tokens));
            cb(cur, "mla_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up, nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            ggml_tensor * moe = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            ggml_tensor * shared = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cur = ggml_add(ctx0, moe, shared);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

llama_model_bailingmoe3::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "BailingMoE3 MTP requires one NextN layer");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.shared_head_norm && "MTP block missing final norm");

    const int64_t n_head = hparams.n_head();
    const int64_t qk_head_dim = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim = hparams.n_embd_head_v_mla();
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const float kq_scale = 1.0f / sqrtf((float) qk_head_dim);

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    ggml_tensor * h_norm = build_norm(inp->embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
    cb(cur, "mtp_eh_proj", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * inpSA = cur;
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * attn_input = cur;

    ggml_tensor * q_all;
    if (layer.wq_a) {
        q_all = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q_all, "q_a", il);
        q_all = build_norm(q_all, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q_all, "q_a_norm", il);
        q_all = ggml_mul_mat(ctx0, layer.wq_b, q_all);
        cb(q_all, "q_b", il);
    } else {
        q_all = ggml_mul_mat(ctx0, layer.wq, cur);
    }
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q_all, qk_nope_head_dim, n_head, n_tokens,
            ggml_row_size(q_all->type, qk_head_dim),
            ggml_row_size(q_all->type, qk_head_dim) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q_all, qk_rope_head_dim, n_head, n_tokens,
            ggml_row_size(q_all->type, qk_head_dim),
            ggml_row_size(q_all->type, qk_head_dim) * n_head,
            ggml_row_size(q_all->type, qk_nope_head_dim));

    ggml_tensor * kv_all = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    ggml_tensor * kv = ggml_view_2d(ctx0, kv_all, kv_lora_rank, n_tokens,
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_all, qk_rope_head_dim, 1, n_tokens,
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
            ggml_row_size(kv_all->type, kv_lora_rank));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

    ggml_tensor * q = ggml_concat(ctx0, q_nope, q_pe, 0);
    kv = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
    ggml_tensor * k = ggml_concat(ctx0, kv, k_pe, 0);

    cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
            q, k, kv, nullptr, nullptr, layer.wv_b, kq_scale, il);

    ggml_tensor * attn_gate = ggml_mul_mat(ctx0, layer.wqkv_gate, attn_input);
    attn_gate = ggml_sigmoid(ctx0, ggml_reshape_3d(ctx0, attn_gate, 1, n_head, n_tokens));
    cur = ggml_reshape_3d(ctx0, cur, v_head_dim, n_head, n_tokens);
    cur = ggml_mul(ctx0, cur, attn_gate);
    cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, cur, v_head_dim * n_head, n_tokens));

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * moe = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU,
            hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    ggml_tensor * shared = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cur = ggml_add(ctx0, moe, shared);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = build_norm(cur, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
