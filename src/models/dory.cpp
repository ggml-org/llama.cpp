#include "models.h"

#include <stdexcept>

void llama_model_dory::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key("dory.input_layer_count", n_input);
    ml.get_key("dory.recurrent_layer_count", n_recurrent);
    ml.get_key("dory.output_layer_count", n_output);
    ml.get_key("dory.recurrent_loop_count", n_loops);
    ml.get_key("dory.recurrent_kv_cache_mode", recurrent_kv_cache_mode, false);
    if (recurrent_kv_cache_mode != "per_loop" && recurrent_kv_cache_mode != "last_loop") {
        throw std::runtime_error("Dory recurrent_kv_cache_mode must be per_loop or last_loop");
    }
    ml.get_key("dory.layer_pattern", pattern);
    uint32_t n_profile = 0;
    ml.get_arr_n("dory.rope.profile", n_profile);
    if (n_profile > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("Dory RoPE profile is too large");
    }
    ml.get_arr("dory.rope.profile", profile);
    ml.get_key("dory.init_std", init_std);
    ml.get_key("dory.alpha_init", alpha_init);
    ml.get_key("dory.gate_init", gate_init);
    ml.get_key("dory.rope.freq_base", theta_1);
    ml.get_key("dory.rope.freq_base_2", theta_2);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    const uint64_t n_physical = uint64_t(n_input) + n_recurrent + n_output;
    const uint64_t n_virtual = uint64_t(n_input) + uint64_t(n_recurrent)*n_loops + n_output;
    if (!n_input || !n_recurrent || !n_output || !n_loops || n_loops > 4 ||
            n_virtual > LLAMA_MAX_LAYERS || n_physical != hparams.n_layer_all ||
            pattern.size() != n_physical || profile.size() != n_physical ||
            pattern.find_first_not_of("S*-") != std::string::npos ||
            !(init_std > 0 && std::isfinite(init_std)) ||
            !(alpha_init > 0 && std::isfinite(alpha_init)) ||
            !(gate_init > 0 && std::isfinite(gate_init)) ||
            !(theta_1 > 0 && std::isfinite(theta_1)) || !(theta_2 > 0 && std::isfinite(theta_2)) ||
            !hparams.n_swa || pattern.find('S') == std::string::npos || pattern.find('*') == std::string::npos) {
        throw std::runtime_error("unsupported or invalid Dory recurrence metadata");
    }
    const auto nh = hparams.n_head(0);
    const auto nk = hparams.n_head_kv(0);
    if (!nh || !nk || nh % nk || !hparams.n_embd_head_k() ||
            hparams.n_embd_head_k() != hparams.n_embd_head_v() || hparams.n_rot() != hparams.n_embd_head_k()) {
        throw std::runtime_error("invalid Dory attention dimensions");
    }
    const auto nf = hparams.n_ff(0);
    if (!nf) {
        throw std::runtime_error("invalid Dory feed-forward dimension");
    }
    for (auto p : profile) {
        if (p < 1 || p > 2) {
            throw std::runtime_error("invalid Dory RoPE profile");
        }
    }
    for (uint32_t i = 0; i < n_input; ++i) {
        physical.push_back(i);
    }
    for (uint32_t k = 0; k < n_loops; ++k) {
        for (uint32_t i = 0; i < n_recurrent; ++i) {
            physical.push_back(n_input + i);
        }
    }
    for (uint32_t i = 0; i < n_output; ++i) {
        physical.push_back(n_input + n_recurrent + i);
    }
    hparams.n_layer_all = n_virtual;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_freq_base_train_swa = theta_2;
    for (uint32_t il = 0; il < n_virtual; ++il) {
        const uint32_t p = physical[il];
        const bool recurrent = p >= n_input && p < n_input + n_recurrent;
        cache_layer.push_back(recurrent && recurrent_kv_cache_mode == "last_loop" ? p : il);
        hparams.n_head_arr[il] = nh;
        hparams.n_head_kv_arr[il] = pattern[p] == '-' ? 0 : nk;
        hparams.n_ff_arr[il] = nf;
        hparams.is_swa_impl[il] = pattern[p] == 'S';
        hparams.rope_pattern[il] = profile[p] != 0 && pattern[p] != '-';
    }
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dory::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
    logit_scale = create_tensor(tn(LLM_TENSOR_DORY_LOGIT_SCALE, "weight"), {n_vocab}, 0);
    scale.resize(n_layer);
    std::vector<int> first(pattern.size(), -1);
    for (int il = 0; il < n_layer; ++il) {
        const int p = physical[il];
        if (first[p] >= 0) {
            layers[il] = layers[first[p]];
            scale[il] = scale[first[p]];
            continue;
        }
        first[p] = il;
        auto & l = layers[il];
        auto & s = scale[il];
        s.alpha = create_tensor(tn(LLM_TENSOR_DORY_ALPHA, "weight", p), {n_embd}, 0);
        if (pattern[p] == '-') {
            l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", p), {n_embd, n_ff}, 0);
            l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", p), {n_embd, n_ff}, 0);
            l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", p), {n_ff, n_embd}, 0);
            s.suv_gate = create_tensor(tn(LLM_TENSOR_DORY_SUV_GATE, "weight", p), {n_ff}, 0);
            s.suv_up = create_tensor(tn(LLM_TENSOR_DORY_SUV_UP, "weight", p), {n_ff}, 0);
        } else {
            const int64_t nq = n_head * n_embd_head_k;
            const int64_t nkv = hparams.n_head_kv(il) * n_embd_head_k;
            l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", p), {n_embd, nq}, 0);
            l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", p), {n_embd, nkv}, 0);
            l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", p), {n_embd, nkv}, 0);
            l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", p), {nq, n_embd}, 0);
            l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", p), {n_embd, nq}, 0);
            s.sqk = create_tensor(tn(LLM_TENSOR_DORY_SQK, "weight", p), {nq}, 0);
            s.gate = create_tensor(tn(LLM_TENSOR_DORY_S_GATE, "weight", p), {nq}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dory::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_dory::graph::graph(const llama_model_dory & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ggml_tensor * pos = build_inp_pos();
    ggml_tensor * out_ids = build_inp_out_ids();
    auto * attn = build_attn_inp_kv_iswa();
    ggml_tensor * input_h = nullptr;
    const int recurrent_end = model.n_input + model.n_recurrent * model.n_loops;
    const auto norm = [&](ggml_tensor * x) { return build_gdn_l2_norm(ctx0, x, 1e-6f); };
    const auto f32 = [&](ggml_tensor * x) { return x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx0, x, GGML_TYPE_F32); };

    for (int il = 0; il < n_layer; ++il) {
        const int p = model.physical[il];
        const auto & l = model.layers[il];
        const auto & s = model.scale[il];
        if (il == (int) model.n_input) {
            input_h = cur;
        }
        if (il >= (int) model.n_input && il < recurrent_end && (il - model.n_input) % model.n_recurrent == 0) {
            cur = norm(ggml_add(ctx0, cur, input_h));
        }
        ggml_tensor * residual = cur;
        if (model.pattern[p] == '-') {
            auto * g = build_lora_mm(l.ffn_gate, cur);
            auto * u = build_lora_mm(l.ffn_up, cur);
            g = ggml_mul(ctx0, g, ggml_scale(ctx0, f32(s.suv_gate), sqrtf(n_embd)));
            u = ggml_mul(ctx0, u, ggml_scale(ctx0, f32(s.suv_up), sqrtf(n_embd)));
            cur = norm(ggml_mul(ctx0, ggml_silu(ctx0, g), u));
            cur = build_lora_mm(l.ffn_down, cur);
        } else {
            const auto hd = hparams.n_embd_head_k(il);
            const auto nh = hparams.n_head(il);
            const auto nk = hparams.n_head_kv(il);
            auto [q, k, v] = build_qkv(l, cur, hd, nh, nk, il);
            if (model.profile[p] != 0) {
                const float theta = model.profile[p] == 2 ? model.theta_2 : model.theta_1;
                q = ggml_rope_ext(ctx0, q, pos, nullptr, hd, rope_type, n_ctx_orig, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(ctx0, k, pos, nullptr, hd, rope_type, n_ctx_orig, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            auto * sq = ggml_sqr(ctx0, ggml_scale(ctx0, f32(s.sqk), sqrtf(hd) / model.init_std));
            q = ggml_mul(ctx0, norm(q), ggml_reshape_3d(ctx0, sq, hd, nh, 1));
            k = norm(k);
            auto * gate = build_lora_mm(l.wqkv_gate, cur);
            auto * sg = ggml_sqr(ctx0, ggml_scale(ctx0, f32(s.gate), sqrtf(model.gate_init) / model.init_std));
            gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, sg));
            // Repeated writes replace only the current chunk; older tokens retain final-loop KV.
            cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(hd), model.cache_layer[il]);
            cur = norm(ggml_mul(ctx0, cur, gate));
            cur = build_lora_mm(l.wo, cur);
        }
        auto * a = norm(residual);
        auto * b = norm(cur);
        auto * alpha = ggml_abs(ctx0, ggml_scale(ctx0, f32(s.alpha), model.alpha_init / model.init_std));
        cur = norm(ggml_add(ctx0, a, ggml_mul(ctx0, ggml_sub(ctx0, b, a), alpha)));
        if (il >= (int) model.n_input && il < recurrent_end && (il + 1 - model.n_input) % model.n_recurrent == 0) {
            cur = norm(cur);
        }
        cb(cur, "l_out", il);
    }
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    res->t_embd = cur;
    cb(cur, "result_norm", -1);
    cur = build_lora_mm(model.output, cur);
    cur = ggml_mul(ctx0, cur, ggml_scale(ctx0, f32(model.logit_scale), 1.0f / model.init_std));
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    // Preserve F32 inputs through recurrent normalization; BF16 and quantized weights keep their normal kernels.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F32) {
            ggml_prec_set_src(node, GGML_PREC_F32, 1);
        }
    }
}
