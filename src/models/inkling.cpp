#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE, after the leading dense layers
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,       hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,      hparams.expert_weights_scale);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        hparams.n_layer_dense_lead > hparams.n_layer() || !hparams.n_expert_shared || !hparams.n_log_scale_floor ||
        hparams.n_shortconv_l_cache < 2 || hparams.n_shortconv_l_cache > 64 ||
        hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.expert_weights_scale, hparams.f_logit_scale}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling: normalization and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling: invalid logarithmic scale");
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff(il) || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded  = hparams.n_vocab_padded;
    const int64_t n_rel           = hparams.n_relative;
    const int64_t n_conv          = hparams.n_shortconv_l_cache;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head*n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        // feed-forward
        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_out_scale = create_tensor(tn(LLM_TENSOR_FFN_OUT_SCALE, "weight", i), {1}, 0);

        if (i < int(hparams.n_layer_dense_lead)) {
            const int64_t n_ff = hparams.n_ff(i);

            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp(i);

            // the router scores the routed and the shared experts together
            layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert + n_expert_shared}, 0);
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, 0);
            layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp, n_expert_shared}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp, n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp, n_embd, n_expert_shared}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// causal short convolution with a residual
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);

    return ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, out, n_channels, n_tokens));
}

llama_model_inkling::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    auto * inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    auto * attn = inp->get_attn();
    auto * recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    ggml_tensor * tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // do not normalize raw embeddings input (i.e. encoded image and audio embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const bool    is_swa       = hparams.is_swa(il);
        const int64_t n_head       = hparams.n_head(il);
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_head  = hparams.n_embd_head_k(il);
        const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

        ggml_tensor * state   = recr->mctx->get_r_l(il);
        ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

        ggml_tensor * inpSA = cur;

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(layer.wq, cur);
            ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, cur), layer.attn_k_conv, state, history, recr, 0);
            ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, cur), layer.attn_v_conv, state, history, recr, n_embd_k_gqa);
            ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, cur);

            Qcur = build_norm(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            Rcur = ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens);

            if (!is_swa) {
                Qcur = ggml_mul(ctx0, Qcur, tau);
                Rcur = ggml_mul(ctx0, Rcur, tau);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            cb(Rcur, "Rcur", il);

            ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

            const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
            const int64_t n_kv     = rel_pos->ne[0];

            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
            cb(kq_b, "kq_b", il);

            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, nullptr, 1.0f/n_embd_head, il);

            cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, state, history, recr, 2*n_embd_k_gqa);
            cb(cur, "attn_out", il);
        }

        cur = ggml_add(ctx0, inpSA, cur);

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (il < int(hparams.n_layer_dense_lead)) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            // MoE: one softmax over the selected routed experts and all shared experts
            const int64_t n_expert        = hparams.n_expert;
            const int64_t n_expert_shared = hparams.n_expert_shared;
            const int64_t n_expert_used   = hparams.n_expert_used(il);

            ggml_tensor * ffn_norm = cur;

            ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, ffn_norm);
            ggml_prec_set_acc(logits, GGML_PREC_F32);

            // the routed experts are selected by sigmoid score plus bias
            ggml_tensor * routed = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, n_expert, n_tokens, logits->nb[1], 0));
            ggml_tensor * choice = ggml_add(ctx0, ggml_sigmoid(ctx0, routed), layer.ffn_exp_probs_b);
            ggml_tensor * ids    = ggml_argsort_top_k(ctx0, choice, n_expert_used);

            // log(sigmoid(x)) = -softplus(-x) also preserves very small router scores
            ggml_tensor * log_probs = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, logits)));

            routed = ggml_cont(ctx0, ggml_view_2d(ctx0, log_probs, n_expert, n_tokens, log_probs->nb[1], 0));

            ggml_tensor * shared   = ggml_cont(ctx0, ggml_view_2d(ctx0, log_probs, n_expert_shared, n_tokens, log_probs->nb[1], n_expert*log_probs->nb[0]));
            ggml_tensor * selected = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, routed, 1, n_expert, n_tokens), ids);
            selected = ggml_reshape_2d(ctx0, selected, n_expert_used, n_tokens);

            ggml_tensor * weights = ggml_soft_max(ctx0, ggml_concat(ctx0, selected, shared, 0));
            ggml_tensor * mass    = ggml_sum_rows(ctx0, ggml_view_2d(ctx0, weights, n_expert_used, n_tokens, weights->nb[1], 0));

            cur = build_moe_ffn(ffn_norm,
                    nullptr,
                    nullptr,
                    nullptr,
                    layer.ffn_down_exps,
                    nullptr,
                    n_expert,
                    n_expert_used,
                    LLM_FFN_SILU, false,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                    il,
                    routed,
                    layer.ffn_gate_up_exps,
                    nullptr,
                    nullptr,
                    nullptr,
                    ids);

            // build_moe_ffn normalizes over the selected experts, restore their share of the joint routed/shared softmax
            cur = ggml_mul(ctx0, cur, mass);
            cb(cur, "ffn_moe_out", il);

            // shared experts, each weighted by its own softmax share
            ggml_tensor * gammas = ggml_view_2d(ctx0, weights, n_expert_shared, n_tokens, weights->nb[1], n_expert_used*weights->nb[0]);
            gammas = ggml_cont(ctx0, ggml_transpose(ctx0, gammas));
            gammas = ggml_reshape_3d(ctx0, gammas, 1, n_tokens, n_expert_shared);

            ggml_tensor * shared_inp = ggml_repeat_4d(ctx0, ffn_norm, n_embd, n_tokens, n_expert_shared, 1);

            ggml_tensor * gate = build_lora_mm(layer.ffn_gate_shexp, shared_inp);
            ggml_tensor * up   = build_lora_mm(layer.ffn_up_shexp,   shared_inp);

            shared = ggml_mul(ctx0, ggml_swiglu_split(ctx0, gate, up), gammas);
            shared = build_lora_mm(layer.ffn_down_shexp, shared);
            shared = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, shared, 1, 2, 0, 3)));
            shared = ggml_reshape_2d(ctx0, shared, n_embd, n_tokens);
            cb(shared, "ffn_shexp", il);

            cur = ggml_scale(ctx0, ggml_add(ctx0, cur, shared), hparams.expert_weights_scale);
        }

        cur = ggml_mul(ctx0, cur, layer.ffn_out_scale);
        cur = convolve(cur, layer.ffn_post_conv, state, history, recr, 2*n_embd_k_gqa + n_embd);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_inp, cur);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head, muP logit scale
    cur = build_lora_mm(model.output, ggml_scale(ctx0, cur, hparams.f_logit_scale));
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
