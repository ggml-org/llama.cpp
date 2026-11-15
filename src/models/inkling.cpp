#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
    ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE, hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        hparams.n_layer_dense_lead > hparams.n_layer() || !hparams.n_expert_shared || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
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
    hparams.rope_pattern.fill(false);
    uint64_t max_channels = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff(il) || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling: invalid attention or expert dimensions");
        }
        max_channels = std::max<uint64_t>(max_channels, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }
    if (max_channels * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling: convolution state is too large");
    }
    hparams.n_sconv_channels = max_channels;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling: tokenizer size must equal the unpadded vocabulary size");
    }
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    for (int i = 0; i < n_layer; ++i) {
        auto & l = layers[i];
        const int64_t nh = hparams.n_head(i);
        const int64_t nk = hparams.n_head_kv(i);
        const int64_t hd = hparams.n_embd_head_k(i);
        const int64_t nr = hparams.n_relative;
        const int64_t kernel = hparams.n_shortconv_l_cache;
        const int64_t extent = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;
        l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, nh*hd}, 0);
        l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, nk*hd}, 0);
        l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, nk*hd}, 0);
        l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {nh*hd, n_embd}, 0);
        l.attn_rel = create_tensor(tn(LLM_TENSOR_ATTN_REL, "weight", i), {n_embd, nh*nr}, 0);
        l.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {nr, extent}, 0);
        l.attn_k_conv = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_v_conv = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        l.ffn_post_conv = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        l.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {hd}, 0);
        l.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {hd}, 0);
        l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        l.ffn_out_scale = create_tensor(tn(LLM_TENSOR_FFN_OUT_SCALE, "weight", i), {1}, 0);
        if (i < int(hparams.n_layer_dense_lead)) {
            l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
            l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
            l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {hparams.n_ff(i), n_embd}, 0);
        } else {
            const int64_t nf = hparams.n_ff_exp(i);
            const int64_t ns = hparams.n_expert_shared;
            l.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert + ns}, 0);
            l.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
            l.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*nf, n_expert}, 0);
            l.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {nf, n_embd, n_expert}, 0);
            l.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, nf, ns}, 0);
            l.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), {n_embd, nf, ns}, 0);
            l.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {nf, n_embd, ns}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

class llm_graph_input_inkling_pos : public llm_graph_input_i {
public:
    const llama_kv_cache_context * cache[2];
    ggml_tensor * distance[2];
    uint32_t extent[2];

    bool can_reuse(const llm_graph_params & params) override {
        const auto * memory = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        cache[0] = memory->get_attn()->get_base();
        cache[1] = memory->get_attn()->get_swa();
        for (int i = 0; i < 2; ++i) {
            if (distance[i]->ne[0] != cache[i]->get_n_kv() || distance[i]->ne[1] != params.ubatch.n_tokens) {
                return false;
            }
        }
        return true;
    }

    void set_input(const llama_ubatch * ubatch) override {
        for (int i = 0; i < 2; ++i) {
            if (distance[i]->buffer) {
                cache[i]->set_input_relative_pos(distance[i], ubatch, extent[i]);
            }
        }
    }
};

ggml_tensor * llama_model_inkling::graph::convolve(ggml_tensor * cur, ggml_tensor * weight,
        ggml_tensor * state, ggml_tensor * history, llm_graph_input_rs * inp, int64_t offset) {
    const int64_t kernel = hparams.n_shortconv_l_cache;
    const int64_t channels = cur->ne[0];
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
    auto * past = ggml_view_3d(ctx0, history, kernel - 1, channels, ns, history->nb[1], history->nb[2], offset*history->nb[1]);
    auto * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, channels, nt, ns));
    x = ggml_concat(ctx0, past, x, 0);
    build_conv_state_store(inp, state, x, kernel - 1, offset*(kernel - 1)*ggml_element_size(state));
    auto * out = ggml_ssm_conv(ctx0, x, weight);
    return ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, out, channels, n_tokens));
}

llama_model_inkling::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    auto * inp = build_inp_mem_hybrid_iswa();
    auto * attn = inp->get_attn();
    auto * recr = inp->get_recr();
    auto * positions = build_inp_pos();
    auto * out_ids = build_inp_out_ids();
    auto rel = std::make_unique<llm_graph_input_inkling_pos>();
    rel->cache[0] = attn->mctx->get_base();
    rel->cache[1] = attn->mctx->get_swa();
    rel->extent[0] = hparams.n_relative_extent;
    rel->extent[1] = hparams.n_swa;
    ggml_tensor * distances[2];
    for (int i = 0; i < 2; ++i) {
        distances[i] = rel->distance[i] = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, rel->cache[i]->get_n_kv(), n_tokens);
        ggml_set_input(distances[i]);
    }
    res->add_input(std::move(rel));
    auto * tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, positions, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);
    auto * cur = build_inp_embd(model.tok_embd);
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }
    for (int il = 0; il < n_layer; ++il) {
        const auto & l = model.layers[il];
        const bool swa = hparams.is_swa(il);
        const int64_t nh = hparams.n_head(il);
        const int64_t nk = hparams.n_head_kv(il);
        const int64_t hd = hparams.n_embd_head_k(il);
        const int64_t kv = nk*hd;
        auto * state = recr->mctx->get_r_l(il);
        auto * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);
        auto * residual = cur;
        auto * input = build_norm(cur, l.attn_norm, nullptr, LLM_NORM_RMS, il);
        auto * q = build_lora_mm(l.wq, input);
        auto * k = convolve(build_lora_mm(l.wk, input), l.attn_k_conv, state, history, recr, 0);
        auto * v = convolve(build_lora_mm(l.wv, input), l.attn_v_conv, state, history, recr, kv);
        auto * r = build_lora_mm(l.attn_rel, input);
        q = build_norm(ggml_reshape_3d(ctx0, q, hd, nh, n_tokens), l.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        k = build_norm(ggml_reshape_3d(ctx0, k, hd, nk, n_tokens), l.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        v = ggml_reshape_3d(ctx0, v, hd, nk, n_tokens);
        r = ggml_reshape_3d(ctx0, r, hparams.n_relative, nh, n_tokens);
        if (!swa) {
            q = ggml_mul(ctx0, q, tau);
            r = ggml_mul(ctx0, r, tau);
        }
        auto * mask = swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
        const int64_t streams = mask->ne[3];
        const int64_t n_kv = distances[swa]->ne[0];
        // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x nh
        auto * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, l.attn_rel_bias, 0, 1, 0, 0), r);
        if (!cparams.flash_attn) {
            // build_attn scales the bias along with QK; FA adds the mask after scaling QK.
            table = ggml_scale(ctx0, table, float(hd));
        }
        table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
        auto * bias = ggml_get_rows(ctx0, table, distances[swa]);
        bias = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, bias, nh, n_kv, n_tokens/streams, streams), 2, 0, 1, 3));
        ggml_tensor * fa_mask = nullptr;
        if (cparams.flash_attn) {
            fa_mask = ggml_cast(ctx0, ggml_add(ctx0, bias, ggml_cast(ctx0, mask, GGML_TYPE_F32)), GGML_TYPE_F16);
            bias = nullptr;
        }
        cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, bias, nullptr, nullptr, 1.0f/hd, il, fa_mask);
        cur = convolve(build_lora_mm(l.wo, cur), l.attn_post_conv, state, history, recr, 2*kv);
        cur = ggml_add(ctx0, residual, cur);
        residual = cur;
        input = build_norm(cur, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
        if (il < int(hparams.n_layer_dense_lead)) {
            cur = build_ffn(input, l.ffn_up, nullptr, nullptr, l.ffn_gate, nullptr, nullptr,
                    l.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            const int64_t ne = hparams.n_expert;
            const int64_t ns = hparams.n_expert_shared;
            const int64_t ntop = hparams.n_expert_used(il);
            auto * logits = build_lora_mm(l.ffn_gate_inp, input);
            ggml_prec_set_acc(logits, GGML_PREC_F32);
            auto * routed = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, ne, n_tokens, logits->nb[1], 0));
            auto * choice = ggml_add(ctx0, ggml_sigmoid(ctx0, routed), l.ffn_exp_probs_b);
            auto * ids = ggml_argsort_top_k(ctx0, choice, ntop);
            // log(sigmoid(x)) = -softplus(-x) also preserves very small router scores.
            auto * log_probs = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, logits)));
            routed = ggml_cont(ctx0, ggml_view_2d(ctx0, log_probs, ne, n_tokens, log_probs->nb[1], 0));
            auto * shared = ggml_cont(ctx0, ggml_view_2d(ctx0, log_probs, ns, n_tokens, log_probs->nb[1], ne*log_probs->nb[0]));
            auto * selected = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, routed, 1, ne, n_tokens), ids), ntop, n_tokens);
            auto * weights = ggml_soft_max(ctx0, ggml_concat(ctx0, selected, shared, 0));
            auto * mass = ggml_sum_rows(ctx0, ggml_view_2d(ctx0, weights, ntop, n_tokens, weights->nb[1], 0));
            cur = build_moe_ffn(input, nullptr, nullptr, nullptr, l.ffn_down_exps, nullptr,
                    ne, ntop, LLM_FFN_SILU, false, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT, il,
                    routed, l.ffn_gate_up_exps, nullptr, nullptr, nullptr, ids);
            // The routed helper normalizes over selected experts; restore their mass from the joint routed/shared distribution.
            cur = ggml_mul(ctx0, cur, mass);
            auto * gammas = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_view_2d(ctx0, weights, ns, n_tokens, weights->nb[1], ntop*weights->nb[0])));
            gammas = ggml_reshape_3d(ctx0, gammas, 1, n_tokens, ns);
            auto * shared_input = ggml_repeat_4d(ctx0, input, n_embd, n_tokens, ns, 1);
            auto * gate = build_lora_mm(l.ffn_gate_shexp, shared_input);
            auto * up = build_lora_mm(l.ffn_up_shexp, shared_input);
            shared = ggml_mul(ctx0, ggml_swiglu_split(ctx0, gate, up), gammas);
            shared = build_lora_mm(l.ffn_down_shexp, shared);
            shared = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, shared, 1, 2, 0, 3)));
            shared = ggml_reshape_2d(ctx0, shared, n_embd, n_tokens);
            cur = ggml_scale(ctx0, ggml_add(ctx0, cur, shared), hparams.expert_weights_scale);
        }
        cur = ggml_mul(ctx0, cur, l.ffn_out_scale);
        cur = convolve(cur, l.ffn_post_conv, state, history, recr, 2*kv + n_embd);
        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "l_out", il);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    res->t_embd = cur;
    cur = build_lora_mm(model.output, ggml_scale(ctx0, cur, hparams.f_logit_scale));
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    res->t_logits = cur;
    cb(cur, "result_output", -1);
}
