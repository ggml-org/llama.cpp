#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP, hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP, hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE, hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE, hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA, hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }
    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta,
                       hparams.situ_linear_beta, hparams.expert_weights_scale, hparams.f_final_logit_softcapping}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }
    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_pattern.fill(false);
    uint64_t max_channels = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }
        max_channels = std::max<uint64_t>(max_channels, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }
    if (max_channels * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }
    hparams.n_sconv_channels = max_channels;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, hparams.n_vocab_padded}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);
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
        l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, nh*hd}, 0);
        l.attn_rel = create_tensor(tn(LLM_TENSOR_ATTN_REL, "weight", i), {n_embd, nh*nr}, 0);
        l.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {nr, extent}, 0);
        l.attn_q_clip = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP, "weight", i), {hd, nh}, 0);
        l.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {nr, nh}, 0);
        l.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, nh}, 0);
        l.attn_k_conv = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_v_conv = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV, "weight", i), {kernel, nk*hd}, 0);
        l.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        l.ffn_post_conv = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV, "weight", i), {kernel, n_embd}, 0);
        if (hparams.is_swa(i)) {
            l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            l.attn_q_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_k_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_v_inp_norm = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM, "weight", i), {n_embd}, 0);
            l.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, kernel*nk}, 0);
            l.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, kernel*nk}, 0);
        }
        if (i > 0) {
            l.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        l.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);
        l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        l.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        l.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        l.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*hparams.n_ff(i), n_expert}, 0);
        l.ffn_gate_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        l.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff(i), n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

class llm_graph_input_inkling2_pos : public llm_graph_input_i {
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

ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }
    auto * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);
    auto * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);
    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    auto * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (auto * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);
    auto * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));
    auto * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

ggml_tensor * llama_model_inkling2::graph::convolve(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dynamic,
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
    out = ggml_reshape_2d(ctx0, out, channels, n_tokens);
    if (dynamic) {
        const int64_t heads = dynamic->ne[0]/kernel;
        dynamic = ggml_reshape_3d(ctx0, dynamic, kernel, heads, n_tokens);
        for (int64_t lag = 0; lag < kernel; ++lag) {
            auto * tap = ggml_view_3d(ctx0, dynamic, 1, heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            auto * window = ggml_view_3d(ctx0, x, nt, channels, ns, x->nb[1], x->nb[2], (kernel - 1 - lag)*x->nb[0]);
            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, channels/heads, heads, n_tokens);
            auto * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), channels, n_tokens);
            out = ggml_add(ctx0, out, term);
        }
    }
    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*kernel + 1));
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    auto * inp = build_inp_mem_hybrid_iswa();
    auto * attn = inp->get_attn();
    auto * recr = inp->get_recr();
    auto * positions = build_inp_pos();
    auto * out_ids = build_inp_out_ids();
    auto rel = std::make_unique<llm_graph_input_inkling2_pos>();
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
    std::vector<ggml_tensor *> blocks;
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
        ggml_tensor * inputs[3] = {cur, cur, cur};
        if (l.attn_res_query) {
            auto * mixed = mix(blocks, cur, l.attn_res_query);
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], (swa ? 0 : j)*mixed->nb[1]));
            }
        }
        if (swa) {
            inputs[0] = build_norm(inputs[0], l.attn_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            inputs[0] = build_norm(inputs[0], l.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = build_norm(inputs[1], l.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[2] = build_norm(inputs[2], l.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
        }
        auto * q = build_lora_mm(l.wq, inputs[0]);
        auto * k = convolve(build_lora_mm(l.wk, inputs[1]), l.attn_k_conv,
                swa ? nullptr : build_lora_mm(l.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        auto * v = convolve(build_lora_mm(l.wv, inputs[2]), l.attn_v_conv,
                swa ? nullptr : build_lora_mm(l.attn_v_conv_proj, inputs[2]), state, history, recr, kv);
        auto * r = build_lora_mm(l.attn_rel, inputs[0]);
        q = clip(ggml_reshape_3d(ctx0, q, hd, nh, n_tokens), l.attn_q_clip, hparams.f_qk_clip);
        k = clip(ggml_reshape_3d(ctx0, k, hd, nk, n_tokens), nullptr, hparams.f_qk_clip);
        v = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, v, hd, nk, n_tokens), hparams.f_norm_rms_eps);
        r = clip(ggml_reshape_3d(ctx0, r, hparams.n_relative, nh, n_tokens), l.attn_rel_clip, hparams.f_relative_clip);
        if (!swa) {
            q = ggml_mul(ctx0, q, tau);
            r = ggml_mul(ctx0, r, tau);
        }
        auto * bank = ggml_pad(ctx0, l.attn_rel_bias, 0, 1, 0, 0);
        auto * index = ggml_reshape_1d(ctx0, distances[swa], ggml_nelements(distances[swa]));
        auto * profiles = ggml_get_rows(ctx0, bank, index);
        const int64_t n_kv = distances[swa]->ne[0];
        profiles = ggml_reshape_3d(ctx0, profiles, hparams.n_relative, n_kv, n_tokens);
        auto * bias = ggml_mul_mat(ctx0, profiles, r);
        bias = ggml_cont(ctx0, ggml_permute(ctx0, bias, 0, 2, 1, 3));
        auto * mask = swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
        const int64_t streams = mask->ne[3];
        bias = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_4d(ctx0, bias, n_kv, n_tokens/streams, streams, nh), 0, 1, 3, 2)), n_kv, n_tokens/streams, nh, streams);
        auto * sinks = build_lora_mm(l.attn_sink_proj, inputs[0]);
        sinks = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, sinks, nh, n_tokens/streams, streams), 1, 0, 2, 3)), 1, n_tokens/streams, nh, streams);
        ggml_tensor * fa_mask = nullptr;
        if (cparams.flash_attn) {
            // exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))).
            fa_mask = ggml_add(ctx0, ggml_sub(ctx0, bias, sinks), ggml_cast(ctx0, mask, GGML_TYPE_F32));
            fa_mask = ggml_cast(ctx0, fa_mask, GGML_TYPE_F16);
            sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, nh), 0.0f);
            bias = nullptr;
        } else {
            // build_attn scales the bias along with QK; FA adds the mask after scaling QK.
            bias = ggml_scale(ctx0, bias, sqrtf(hd));
        }
        cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, bias, sinks, nullptr, 1.0f/sqrtf(hd), il, fa_mask);
        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(l.wqkv_gate, inputs[0])));
        cur = convolve(build_lora_mm(l.wo, cur), l.attn_post_conv, nullptr, state, history, recr, 2*kv);
        if (il % hparams.attn_res_block_size == 0) {
            blocks.push_back(residual);
        } else {
            cur = ggml_add(ctx0, residual, cur);
        }
        residual = cur;
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, l.ffn_res_query), n_embd, n_tokens);
        cur = build_norm(cur, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_moe_ffn(cur, l.ffn_gate_inp, nullptr, nullptr, l.ffn_down_exps, l.ffn_exp_probs_b,
                hparams.n_expert, hparams.n_expert_used(il), LLM_FFN_SITU, true, hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, nullptr, l.ffn_gate_up_exps,
                l.ffn_gate_up_exps_s, nullptr, l.ffn_down_exps_s);
        cur = convolve(cur, l.ffn_post_conv, nullptr, state, history, recr, 2*kv + n_embd);
        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "l_out", il);
    }
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    const float cap = hparams.f_final_logit_softcapping;
    cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    res->t_logits = cur;
    cb(cur, "result_output", -1);
}
