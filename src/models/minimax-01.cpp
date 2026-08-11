#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_minimax_01::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale);

    // we use n_embd_head_la to set recurrent memory n_embd_s
    hparams.n_embd_head_la = hparams.n_embd_head_k_full;

    // Mark recurrent layers (lightning attention layers).
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 8;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 80: type = LLM_TYPE_456B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_minimax_01::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_recr(i)) {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        } else {
            layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd_head_k * n_head}, 0);
            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd_head_k * n_head}, 0);
            layer.wg = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_01::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

class llm_graph_input_la : public llm_graph_input_i {
public:
    llm_graph_input_la(const llama_hparams & hparams, const llama_vocab & vocab) : hparams(hparams), vocab(vocab) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (inp_slopes) {
            const int64_t n_head = hparams.n_head();

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_slopes->buffer));

            float * data = (float *) inp_slopes->data;

            float start = powf(2, -powf(2, -(log2f(n_head) - 3)));
            float ratio = start;

            for (int h = 0; h < n_head; ++h) {
                data[h] = start * powf(ratio, h);
            }
        }

        if (inp_q_decay) {
            const int64_t n_head = hparams.n_head();
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_q_decay->buffer));

            float * slopes = (float *) inp_slopes->data;
            float * data = (float *) inp_q_decay->data;

            for (int i = 0; i < n_seq_tokens; ++i) {
                for (int h = 0; h < n_head; ++h) {
                    data[i * n_head + h] = -slopes[h] * (i + 1);
                }
            }
        }

        if (inp_k_decay) {
            const int64_t n_head = hparams.n_head();
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_k_decay->buffer));

            float * slopes = (float *) inp_slopes->data;
            float * data = (float *) inp_k_decay->data;

            for (int i = 0; i < n_seq_tokens; ++i) {
                for (int h = 0; h < n_head; ++h) {
                    data[i * n_head + h] = -slopes[h] * (n_seq_tokens - i - 1);
                }
            }
        }

        if (inp_diag_decay) {
            const int64_t n_head = hparams.n_head();
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_diag_decay->buffer));

            float * slopes = (float *) inp_slopes->data;
            float * data = (float *) inp_diag_decay->data;

            for (int j = 0; j < n_seq_tokens; ++j) {
                for (int i = 0; i < n_seq_tokens; ++i) {
                    int index = j - i;
                    for (int h = 0; h < n_head; ++h) {
                        float s_index = index >= 0 ? -slopes[h] * index : -INFINITY;
                        data[j * n_head * n_seq_tokens + i * n_head + h] = s_index;
                    }
                }
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return false;
    }

    const llama_hparams & hparams;
    const llama_vocab   & vocab;

    ggml_tensor * inp_slopes     = nullptr; // F32 [n_head]
    ggml_tensor * inp_q_decay    = nullptr; // F32 [n_batch, n_head]
    ggml_tensor * inp_k_decay    = nullptr; // F32 [n_batch, n_head]
    ggml_tensor * inp_diag_decay = nullptr; // F32 [n_batch, n_batch, n_head]
};

llama_model_minimax_01::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // GGML_ASSERT(n_embd_head == n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

    const int64_t n_seqs  = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp_hybrid = build_inp_mem_hybrid();
    auto * inp_rs = inp_hybrid->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_la * la = nullptr;

    auto inp = std::make_unique<llm_graph_input_la>(hparams, model.vocab);

    inp->inp_slopes = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head);
    ggml_set_input(inp->inp_slopes);
    cb(inp->inp_slopes, "slopes", -1);

    if (n_seq_tokens != 1) {
        inp->inp_q_decay = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_head, n_seq_tokens);
        ggml_set_input(inp->inp_q_decay);
        cb(inp->inp_q_decay, "q_decay_exp", -1);

        inp->inp_k_decay = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_head, n_seq_tokens);
        ggml_set_input(inp->inp_k_decay);
        cb(inp->inp_k_decay, "k_decay_exp", -1);

        inp->inp_diag_decay = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_head, n_seq_tokens, n_seq_tokens);
        ggml_set_input(inp->inp_diag_decay);
        cb(inp->inp_diag_decay, "diag_decay_exp", -1);
    }

    la = (llm_graph_input_la *) res->add_input(std::move(inp));

    ggml_tensor * slopes = la->inp_slopes;
    ggml_tensor * q_decay_exp = (n_seq_tokens != 1 ? la->inp_q_decay : nullptr);
    ggml_tensor * k_decay_exp = (n_seq_tokens != 1 ? la->inp_k_decay : nullptr);
    ggml_tensor * diag_decay_exp = (n_seq_tokens != 1 ? la->inp_diag_decay : nullptr);

    ggml_tensor * logits_mask = build_inp_logits_mask(model, 32);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * residual = cur;

        // self_attention
        if (!hparams.is_recr(il)) {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_hybrid->get_attn(),
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        } else {
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // TODO any way to make conv states optional in recurrent memory?
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_build_forward_expand(gf, conv_state_all);

            float slope_scale = 1.0 - 1.0 * il / (n_layer - 1) + 1e-5;
            ggml_tensor * slope_rate = ggml_scale(ctx0, slopes, slope_scale);
            cb(slope_rate, "slope_rate", il);

            cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], n_seq_tokens, 1, n_seqs);

            ggml_tensor * QKVcur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(QKVcur, "QKVcur", il);

            QKVcur = ggml_silu(ctx0, QKVcur);
            cb(QKVcur, "QKVcur_silu", il);

            QKVcur = ggml_reshape_4d(ctx0, QKVcur, n_embd_head * 3, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_4d(ctx0, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur->nb[1], QKVcur->nb[2], QKVcur->nb[3], 0*sizeof(float)*n_embd_head));
            ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_4d(ctx0, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur->nb[1], QKVcur->nb[2], QKVcur->nb[3], 1*sizeof(float)*n_embd_head));
            ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_4d(ctx0, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur->nb[1], QKVcur->nb[2], QKVcur->nb[3], 2*sizeof(float)*n_embd_head));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            ggml_tensor * la_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, la_states_all, hparams.n_embd_s(), n_seqs);

            ggml_tensor * kv_old = ggml_reshape_4d(ctx0, state, n_embd_head, n_embd_head, n_head, n_seqs);
            cb(kv_old, "kv_old", il);

            ggml_tensor * qkv = nullptr;
            ggml_tensor * kv_new = nullptr;

            if (n_seq_tokens == 1) {
                ggml_tensor * slopes_neg = ggml_scale(ctx0, slope_rate, -1.0);
                cb(slopes_neg, "slopes_neg", il);

                ggml_tensor * ratio = ggml_exp(ctx0, slopes_neg);
                cb(ratio, "ratio", il);

                ggml_tensor * ratio_3d = ggml_reshape_3d(ctx0, ratio, 1, 1, n_head);
                cb(ratio_3d, "ratio3d", il);

                ggml_tensor * v_trans = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));
                cb(v_trans, "v_trans", il);

                ggml_tensor * k_trans = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 1, 2, 0, 3));
                cb(k_trans, "k_trans", il);

                ggml_tensor * kv_cur = ggml_mul_mat(ctx0, k_trans, v_trans);
                cb(kv_cur, "kv_cur", il);

                ggml_tensor * kv_old_s = ggml_mul(ctx0, kv_old, ratio_3d);
                cb(kv_old_s, "kv_old_s", il);

                kv_new = ggml_add(ctx0, kv_old_s, kv_cur);
                cb(kv_new, "kv_new", il);

                ggml_tensor * q_trans = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q_trans, "q_trans", il);

                qkv = ggml_mul_mat(ctx0, kv_new, q_trans);
                cb(qkv, "qkv", il);
            } else if(n_seq_tokens > 1) {
                ggml_tensor * q_decay = ggml_exp(ctx0, ggml_scale(ctx0, q_decay_exp, slope_scale));
                cb(q_decay, "q_decay", il);
                ggml_tensor * k_decay = ggml_exp(ctx0, ggml_scale(ctx0, k_decay_exp, slope_scale));
                cb(k_decay, "k_decay", il);
                ggml_tensor * diag_decay = ggml_exp(ctx0, ggml_scale(ctx0, diag_decay_exp, slope_scale));
                cb(diag_decay, "diag_decay", il);

                ggml_tensor * q_s = ggml_mul(ctx0, Qcur, q_decay);
                cb(q_s, "q_s", il);

                ggml_tensor * q_s_trans = ggml_permute(ctx0, q_s, 0, 2, 1, 3);
                cb(q_s_trans, "q_s_trans", il);

                ggml_tensor * qkv_none_diag = ggml_mul_mat(ctx0, kv_old, q_s_trans);
                cb(qkv_none_diag, "qkv_none_diag", il);

                ggml_tensor * q_trans = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q_trans, "q_trans", il);

                ggml_tensor * k_trans = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
                cb(k_trans, "k_trans", il);

                ggml_tensor * qk = ggml_mul_mat(ctx0, k_trans, q_trans);
                cb(qk, "qk", il);

                ggml_tensor * diag_decay_trans = ggml_cont(ctx0, ggml_permute(ctx0, diag_decay, 2, 0, 1, 3));

                qk = ggml_mul(ctx0, qk, diag_decay_trans);
                cb(qk, "qk_s", il);

                ggml_tensor * v_trans = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));
                cb(v_trans, "v_trans", il);

                ggml_tensor * qkv_diag = ggml_mul_mat(ctx0, v_trans, qk);
                cb(qkv_diag, "qkv_diag", il);

                qkv = ggml_add(ctx0, qkv_none_diag, qkv_diag);
                cb(qkv, "qkv", il);

                ggml_build_forward_expand(gf, qkv);

                ggml_tensor * slopes_neg = ggml_scale(ctx0, slope_rate, -1.0*n_seq_tokens);
                cb(slopes_neg, "slopes_neg", il);

                ggml_tensor * block_decay = ggml_exp(ctx0, slopes_neg);
                cb(block_decay, "block_decay", il);

                ggml_tensor * block_decay_3d = ggml_reshape_3d(ctx0, block_decay, 1, 1, n_head);
                cb(block_decay_3d, "block_decay_3d", il);

                ggml_tensor * kv_old_s = ggml_mul(ctx0, kv_old, block_decay_3d);
                cb(kv_old_s, "kv_old_s", il);

                ggml_tensor * k_after_decay = ggml_mul(ctx0, Kcur, k_decay);
                cb(k_after_decay, "k_after_decay", il);

                ggml_tensor * k_after_decay_trans = ggml_cont(ctx0, ggml_permute(ctx0, k_after_decay, 1, 2, 0, 3));
                cb(k_after_decay_trans, "k_after_decay_trans", il);

                ggml_tensor * kv_cur = ggml_mul_mat(ctx0, k_after_decay_trans, v_trans);
                cb(kv_cur, "kv_cur", il);

                kv_new = ggml_add(ctx0, kv_old_s, kv_cur);
                cb(kv_new, "kv_new", il);
            }

            // update the recurrent states
            ggml_build_forward_expand(gf,
                                     ggml_cpy(ctx0, kv_new,
                                              ggml_view_1d(ctx0, la_states_all, hparams.n_embd_s() * n_seqs,
                                                           kv_head * hparams.n_embd_s() * ggml_element_size(la_states_all))));

            qkv = ggml_cont(ctx0, ggml_permute(ctx0, qkv, 0, 2, 1, 3));
            cb(qkv, "qkv_permuted", il);

            qkv = ggml_reshape_4d(ctx0, qkv, qkv->ne[0]*qkv->ne[1], qkv->ne[2], 1, qkv->ne[3]);

            // norm
            ggml_tensor * qkv_norm = build_norm(qkv,
                    model.layers[il].attn_norm_2, NULL,
                    LLM_NORM_RMS, il);
            cb(qkv_norm, "qkv_norm", il);

            ggml_tensor * g = build_lora_mm(model.layers[il].wg, cur);
            cb(g, "g", il);

            g = ggml_sigmoid(ctx0, g);
            cb(g, "g_sigm", il);

            cur = ggml_mul(ctx0, g, qkv_norm);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_out", il);

            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens*n_seqs);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        residual = ggml_scale(ctx0, residual, hparams.f_residual_scale);
        cb(residual, "residual_scaled_attn", il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        residual = cur;

        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(cur, "ffn_moe_out", il);

        residual = ggml_scale(ctx0, residual, hparams.f_residual_scale);
        cb(residual, "residual_scaled_ffn", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);
    cur = ggml_add(ctx0, cur, logits_mask);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
