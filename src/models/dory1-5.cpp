#include "models.h"
#include "llama-memory-recurrent.h"

#include <limits>
#include <stdexcept>

void llama_model_dory1_5::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_INPUT_LAYER_COUNT, n_input);
    ml.get_key(LLM_KV_RECURRENT_LAYER_COUNT, n_recurrent);
    ml.get_key(LLM_KV_OUTPUT_LAYER_COUNT, n_output);
    ml.get_key(LLM_KV_NUM_LOOPS, n_loops);
    ml.get_key(LLM_KV_LAYER_PATTERN, pattern);
    ml.get_key(LLM_KV_LOOP_CONDITIONING_HIDDEN_SIZE, n_cond);
    ml.get_key(LLM_KV_LOOP_CONDITIONING_MAX_LOOPS, max_loops);
    ml.get_key(LLM_KV_LOOP_ADAPTER_RANK, adapter_rank);
    ml.get_key(LLM_KV_LOOP_INPUT_MIXING, input_mixing);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE, hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE, hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT, hparams.ssm_n_group);
    ml.get_key(LLM_KV_SSM_HOUSEHOLDER_COUNT, hparams.ssm_n_householder);
    std::string cache_mode;
    ml.get_key(LLM_KV_RECURRENT_KV_CACHE_MODE, cache_mode);
    if (cache_mode != "per_loop" && cache_mode != "last_loop") {
        throw std::runtime_error("Dory 1.5 recurrent_kv_cache_mode must be per_loop or last_loop");
    }

    const uint64_t n_physical = uint64_t(n_input) + n_recurrent + n_output;
    const uint64_t n_virtual = uint64_t(n_input) + uint64_t(n_recurrent)*n_loops + n_output;
    if (!n_recurrent || !n_loops || !max_loops || n_virtual > LLAMA_MAX_LAYERS ||
            n_physical != hparams.n_layer_all || pattern.size() != n_physical ||
            pattern.find_first_not_of("M*-") != std::string::npos ||
            !n_cond || n_cond % 2 || n_cond > hparams.n_embd || adapter_rank > hparams.n_embd) {
        throw std::runtime_error("invalid Dory 1.5 recurrence metadata");
    }
    const uint32_t nh = hparams.n_head(0);
    const uint32_t nk = hparams.n_head_kv(0);
    const uint32_t nf = hparams.n_ff(0);
    const uint32_t hv = hparams.ssm_dt_rank;
    const uint32_t hk = hparams.ssm_n_group;
    const uint32_t sk = hparams.ssm_d_state;
    const uint32_t inner = hparams.ssm_d_inner;
    const uint32_t m = hparams.ssm_n_householder;
    if (!nh || !nk || nh % nk || !nf || !hparams.n_embd_head_k() ||
            hparams.n_embd_head_k() != hparams.n_embd_head_v() || !hv || !hk || hv % hk ||
            !inner || inner % hv || !sk || !m || m > 16 || hparams.ssm_d_conv < 2 ||
            !(hparams.f_norm_rms_eps > 0 && std::isfinite(hparams.f_norm_rms_eps))) {
        throw std::runtime_error("invalid Dory 1.5 attention dimensions");
    }
    const uint64_t conv_dim = uint64_t(m)*inner + uint64_t(m + 1)*hk*sk;
    if (conv_dim > std::numeric_limits<uint32_t>::max() / (hparams.ssm_d_conv - 1) ||
            uint64_t(sk)*inner > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Dory 1.5 recurrent state dimensions overflow");
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
    for (uint32_t il = 0; il < n_virtual; ++il) {
        const uint32_t p = physical[il];
        const char kind = pattern[p];
        const bool recurrent_attn = kind == '*' && p >= n_input && p < n_input + n_recurrent;
        // Only attention KV is shared across loops; GDP and convolution states remain per-loop.
        cache_layer.push_back(recurrent_attn && cache_mode == "last_loop" ? p : il);
        hparams.n_head_arr[il] = nh;
        hparams.n_head_kv_arr[il] = kind == '*' ? nk : 0;
        hparams.n_ff_arr[il] = kind == '-' ? nf : 0;
        hparams.is_recr_impl[il] = kind == 'M';
    }
    type = LLM_TYPE_4B;
}

void llama_model_dory1_5::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const auto load_linear = [&](linear & l, llm_tensor role, int64_t in, int64_t out, int p = -1) {
        l.weight = create_tensor(tn(role, "weight", p), {in, out}, 0);
        l.bias = create_tensor(tn(role, "bias", p), {out}, 0);
    };
    load_linear(cond_up, LLM_TENSOR_LOOP_COND_UP, 2*n_cond, n_cond);
    load_linear(cond_down, LLM_TENSOR_LOOP_COND_DOWN, n_cond, n_cond);
    load_linear(entry_gamma, LLM_TENSOR_LOOP_ENTRY_GAMMA, n_cond, n_embd);
    load_linear(entry_beta, LLM_TENSOR_LOOP_ENTRY_BETA, n_cond, n_embd);
    load_linear(exit_gate, LLM_TENSOR_LOOP_EXIT_GATE, n_cond, n_embd);
    if (input_mixing) {
        load_linear(input_mix, LLM_TENSOR_LOOP_INPUT_MIX, n_cond, n_embd);
    }
    load_linear(state_update, LLM_TENSOR_STATE_UPDATE, n_embd, n_embd);
    entry_norm = create_tensor(tn(LLM_TENSOR_LOOP_ENTRY_NORM, "weight"), {n_embd}, 0);
    state_alpha = create_tensor(tn(LLM_TENSOR_STATE_ALPHA, "weight"), {n_embd}, 0);

    const int64_t hv = hparams.ssm_dt_rank;
    const int64_t key_dim = int64_t(hparams.ssm_n_group)*hparams.ssm_d_state;
    const int64_t value_dim = hparams.ssm_d_inner;
    const int64_t m = hparams.ssm_n_householder;
    const int64_t conv_dim = m*value_dim + (m + 1)*key_dim;
    cond.resize(n_layer);
    std::vector<int> first(pattern.size(), -1);
    for (int il = 0; il < n_layer; ++il) {
        const int p = physical[il];
        if (first[p] >= 0) {
            layers[il] = layers[first[p]];
            cond[il] = cond[first[p]];
            continue;
        }
        first[p] = il;
        auto & l = layers[il];
        auto & c = cond[il];
        l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", p), {n_embd}, 0);
        if (p >= (int) n_input) {
            load_linear(c.gamma, LLM_TENSOR_LOOP_GAMMA, n_cond, n_embd, p);
            load_linear(c.beta, LLM_TENSOR_LOOP_BETA, n_cond, n_embd, p);
            load_linear(c.gate, LLM_TENSOR_LOOP_GATE, n_cond, n_embd, p);
            if (adapter_rank) {
                load_linear(c.adapter_gate, LLM_TENSOR_LOOP_ADAPTER_GATE, n_cond, adapter_rank, p);
                c.adapter_up = create_tensor(tn(LLM_TENSOR_LOOP_ADAPTER_UP, "weight", p), {adapter_rank, n_embd}, 0);
                c.adapter_down = create_tensor(tn(LLM_TENSOR_LOOP_ADAPTER_DOWN, "weight", p), {n_embd, adapter_rank}, 0);
            }
        }
        if (pattern[p] == '-') {
            l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", p), {n_embd, hparams.n_ff(il)}, 0);
            l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", p), {hparams.n_ff(il), n_embd}, 0);
        } else if (pattern[p] == '*') {
            create_tensor_qkv(l, p, n_embd, n_head*n_embd_head_k, hparams.n_embd_k_gqa(il), hparams.n_embd_v_gqa(il), 0);
            l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", p), {n_head*n_embd_head_k, n_embd}, 0);
        } else {
            l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", p), {n_embd, key_dim}, 0);
            l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", p), {n_embd, m*key_dim}, 0);
            l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", p), {n_embd, m*value_dim}, 0);
            l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", p), {n_embd, value_dim}, 0);
            l.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", p), {n_embd, hv}, 0);
            l.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", p), {n_embd, m*hv}, 0);
            l.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, p), {hv}, 0);
            l.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", p), {hv}, 0);
            l.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", p), {hparams.ssm_d_conv, conv_dim}, 0);
            l.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", p), {value_dim}, 0);
            l.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", p), {value_dim, n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dory1_5::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_dory1_5::graph::build_linear(const linear & layer, ggml_tensor * cur) {
    return ggml_add(ctx0, build_lora_mm(layer.weight, cur), layer.bias);
}

ggml_tensor * llama_model_dory1_5::graph::build_condition(float progress, float budget) {
    const int half = model.n_cond / 2;
    auto * freq = ggml_exp(ctx0, ggml_scale(ctx0, ggml_arange(ctx0, 0, half, 1), -logf(100.0f)/half));
    const auto embed = [&](float value) {
        auto * args = ggml_scale(ctx0, freq, 100.0f*value);
        return ggml_concat(ctx0, ggml_cos(ctx0, args), ggml_sin(ctx0, args), 0);
    };
    auto * cur = ggml_concat(ctx0, embed(progress), embed(budget), 0);
    cur = build_linear(model.cond_up, cur);
    return build_linear(model.cond_down, ggml_silu(ctx0, cur));
}

llama_model_dory1_5::graph::graph(const llama_model_dory1_5 & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    auto * cur = build_inp_embd(model.tok_embd);
    auto * out_ids = build_inp_out_ids();
    auto * inp = build_inp_mem_hybrid();
    const float budget = float(std::min(model.n_loops, model.max_loops))/model.max_loops;
    std::vector<ggml_tensor *> conditions;
    for (uint32_t k = 0; k < model.n_loops; ++k) {
        conditions.push_back(build_condition(float(k + 1)/model.n_loops, budget));
    }
    auto * alpha = ggml_softplus(ctx0, model.state_alpha);
    int il = 0;
    for (uint32_t i = 0; i < model.n_input; ++i) {
        cur = build_layer(inp, cur, nullptr, il++);
    }
    auto * input_h = cur;
    auto * state = cur;
    for (uint32_t k = 0; k < model.n_loops; ++k) {
        auto * c = conditions[k];
        auto * loop_input = ggml_add(ctx0, state, input_h);
        cur = loop_input;
        if (k > 0) {
            auto * scale = ggml_scale_bias(ctx0, build_linear(model.entry_gamma, c), 1.0f, 1.0f);
            cur = build_norm(cur, model.entry_norm, nullptr, LLM_NORM_RMS, il);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, scale), build_linear(model.entry_beta, c));
            if (model.input_mixing) {
                cur = ggml_add(ctx0, cur, ggml_mul(ctx0, input_h, build_linear(model.input_mix, c)));
            }
        }
        for (uint32_t i = 0; i < model.n_recurrent; ++i) {
            cur = build_layer(inp, cur, c, il++);
        }
        auto * beta = ggml_mul(ctx0, ggml_sigmoid(ctx0, build_linear(model.state_update, loop_input)), alpha);
        auto * new_state = ggml_add(ctx0, state, ggml_mul(ctx0, beta, ggml_sub(ctx0, cur, state)));
        auto * gate = ggml_scale_bias(ctx0, build_linear(model.exit_gate, c), 1.0f, k == 0 ? 1.0f : 0.0f);
        state = ggml_add(ctx0, state, ggml_mul(ctx0, ggml_sub(ctx0, new_state, state), gate));
    }
    cur = state;
    for (uint32_t i = 0; i < model.n_output; ++i) {
        cur = build_layer(inp, cur, conditions.back(), il++);
    }
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;
    cb(cur, "result_norm", -1);
    res->t_logits = build_lora_mm(model.output, cur);
    cb(res->t_logits, "result_output", -1);
}

ggml_tensor * llama_model_dory1_5::graph::build_layer(llm_graph_input_mem_hybrid * inp, ggml_tensor * cur, ggml_tensor * condition, int il) {
    const auto & l = model.layers[il];
    const auto & c = model.cond[il];
    auto * residual = cur;
    cur = build_norm(cur, l.attn_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gate = nullptr;
    if (condition) {
        auto * scale = ggml_scale_bias(ctx0, build_linear(c.gamma, condition), 1.0f, 1.0f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, scale), build_linear(c.beta, condition));
        gate = ggml_scale_bias(ctx0, build_linear(c.gate, condition), 1.0f, 1.0f);
        if (c.adapter_down) {
            auto * adapter = build_lora_mm(c.adapter_down, cur);
            adapter = ggml_mul(ctx0, adapter, build_linear(c.adapter_gate, condition));
            cur = ggml_add(ctx0, cur, build_lora_mm(c.adapter_up, adapter));
        }
    }
    switch (model.pattern[model.physical[il]]) {
        case '-':
            cur = build_lora_mm(l.ffn_down, ggml_sqr(ctx0, ggml_relu(ctx0, build_lora_mm(l.ffn_up, cur))));
            break;
        case '*': {
            auto [q, k, v] = build_qkv(l, cur, hparams.n_embd_head_k(), hparams.n_head(il), hparams.n_head_kv(il), il);
            cur = build_attn(inp->get_attn(), l.wo, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f/sqrtf(hparams.n_embd_head_k()), model.cache_layer[il]);
        } break;
        case 'M':
            cur = build_gdp(inp->get_recr(), cur, il);
            break;
    }
    if (gate) {
        cur = ggml_mul(ctx0, cur, gate);
    }
    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "l_out", il);
    return cur;
}

ggml_tensor * llama_model_dory1_5::graph::build_gdp(llm_graph_input_rs * inp, ggml_tensor * cur, int il) {
    const auto & l = model.layers[il];
    const int64_t sk = hparams.ssm_d_state;
    const int64_t hk = hparams.ssm_n_group;
    const int64_t hv = hparams.ssm_dt_rank;
    const int64_t sv = hparams.ssm_d_inner / hv;
    const int64_t m = hparams.ssm_n_householder;
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
    const int64_t kd = sk*hk;
    const int64_t vd = sv*hv;
    const int64_t cd = m*vd + (m + 1)*kd;
    GGML_ASSERT(ubatch.equal_seqs() && nt*ns == n_tokens);

    auto * q = build_lora_mm(l.wq, cur);
    auto * k = build_lora_mm(l.wk, cur);
    auto * v = build_lora_mm(l.wv, cur);
    auto * z = build_lora_mm(l.wqkv_gate, cur);
    auto * mixed = ggml_concat(ctx0, ggml_concat(ctx0, v, k, 0), q, 0);
    mixed = ggml_reshape_3d(ctx0, mixed, cd, nt, ns);
    auto * conv_all = inp->mctx->get_r_l(il);
    auto * state_all = inp->mctx->get_s_l(il);
    auto * conv = build_conv_state(inp, conv_all, mixed, hparams.ssm_d_conv, cd, il);
    conv = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv, l.ssm_conv1d));

    const auto section = [&](int64_t dim, int64_t count, int64_t offset) {
        return ggml_cont(ctx0, ggml_view_4d(ctx0, conv, dim, count, nt, ns,
                dim*ggml_element_size(conv), conv->nb[1], conv->nb[2], offset*ggml_element_size(conv)));
    };
    v = ggml_reshape_4d(ctx0, section(vd, m, 0), sv, hv, m*nt, ns);
    k = ggml_reshape_4d(ctx0, section(kd, m, m*vd), sk, hk, m*nt, ns);
    q = ggml_reshape_4d(ctx0, section(kd, 1, m*(vd + kd)), sk, hk, nt, ns);
    k = build_gdn_l2_norm(ctx0, k, 1e-6f);
    q = build_gdn_l2_norm(ctx0, q, 1e-6f);
    q = ggml_reshape_4d(ctx0, q, kd, 1, nt, ns);
    q = ggml_reshape_4d(ctx0, ggml_repeat_4d(ctx0, q, kd, m, nt, ns), sk, hk, m*nt, ns);
    const auto repeat_heads = [&](ggml_tensor * x) {
        x = ggml_reshape_4d(ctx0, x, sk, 1, hk, m*nt*ns);
        return ggml_reshape_4d(ctx0, ggml_repeat_4d(ctx0, x, sk, hv/hk, hk, m*nt*ns), sk, hv, m*nt, ns);
    };
    q = repeat_heads(q);
    k = repeat_heads(k);

    auto * a = ggml_add(ctx0, build_lora_mm(l.ssm_alpha, cur), l.ssm_dt);
    auto * g = ggml_mul(ctx0, ggml_softplus(ctx0, a), l.ssm_a);
    g = ggml_reshape_4d(ctx0, g, hv, 1, nt, ns);
    g = ggml_reshape_4d(ctx0, ggml_pad(ctx0, g, 0, m - 1, 0, 0), 1, hv, m*nt, ns);
    auto * b = ggml_reshape_4d(ctx0, ggml_sigmoid(ctx0, build_lora_mm(l.ssm_beta, cur)), 1, hv, m*nt, ns);
    auto * state = ggml_reshape_4d(ctx0, build_rs(inp, state_all, hparams.n_embd_s(), ns), sk, sv, hv, ns);
    auto result = build_delta_net(q, k, v, g, b, state);
    auto * out = result.first;
    state = result.second;
    out = ggml_view_4d(ctx0, out, sv, hv, nt, ns, out->nb[1], m*out->nb[2], out->nb[3], (m - 1)*out->nb[2]);
    const size_t offset = inp->mctx->get_head()*hparams.n_embd_s()*ggml_element_size(state_all);
    auto * dst = ggml_view_2d(ctx0, state_all, hparams.n_embd_s(), ns, state_all->nb[1], offset);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, state, dst));
    out = ggml_cont_2d(ctx0, out, vd, n_tokens);
    out = ggml_mul(ctx0, out, ggml_silu(ctx0, z));
    out = ggml_reshape_3d(ctx0, out, vd/hk, hk, n_tokens);
    out = ggml_rms_norm(ctx0, out, hparams.f_norm_rms_eps);
    out = ggml_mul(ctx0, ggml_reshape_2d(ctx0, out, vd, n_tokens), l.ssm_norm);
    return build_lora_mm(l.ssm_out, out);
}
