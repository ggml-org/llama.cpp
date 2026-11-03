#include "models.h"

#include <cmath>

void llama_model_metis16::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    ml.get_key(LLM_KV_SSM_NORM_EPS,       hparams.ssm_norm_eps);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                   hparams.moe_latent_size);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,            hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_METIS_PASS_COUNT,                  hparams.metis_n_pass);
    ml.get_key(LLM_KV_METIS_MEMORY_LENGTH,               hparams.metis_n_mem);
    ml.get_key(LLM_KV_METIS_ROUTE_LENGTH,                hparams.metis_n_route);
    ml.get_key(LLM_KV_ATTENTION_PASS_LORA_RANK,          hparams.metis_pass_lora_rank);
    ml.get_key(LLM_KV_METIS_CONTINUATION_LOGIT_LIMIT,    hparams.metis_cont_limit);

    n_pass     = hparams.metis_n_pass;
    n_stream   = hparams.dsv4_hc_mult;
    n_mem      = hparams.metis_n_mem;
    n_route    = hparams.metis_n_route;
    cont_limit = hparams.metis_cont_limit;

    if (n_pass < 1 || hparams.n_layer() % n_pass != 0) {
        throw std::runtime_error(format("metis16: %u layers do not split into %u passes", hparams.n_layer(), n_pass));
    }
    n_layer_phys = hparams.n_layer() / n_pass;

    bool has_attn = false;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.n_head_kv(il) != hparams.n_head_kv(il % n_layer_phys)) {
            throw std::runtime_error(format("metis16: layer %u does not repeat the layout of layer %u", il, il % n_layer_phys));
        }
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
        has_attn = has_attn || !hparams.is_recr_impl[il];
    }
    if (!has_attn) {
        throw std::runtime_error("metis16: at least one attention layer is required");
    }

    switch (n_layer_phys) {
        case 12: type = LLM_TYPE_3B;  break; // Praxis
        case 20: type = LLM_TYPE_10B; break; // Logos
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_metis16::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t d_state    = hparams.ssm_d_state;
    const int64_t n_ssm_head = hparams.ssm_dt_rank;
    const int64_t n_group    = hparams.ssm_n_group;
    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;
    const int64_t n_latent   = hparams.moe_latent_size;
    const int64_t n_ff_shexp = hparams.n_ff_shexp;
    const int64_t n_s = n_stream;
    const int64_t n_p = n_pass;
    const int64_t n_m = n_mem;
    const int64_t n_r = n_route;

    int64_t n_attn = 0;
    for (uint32_t il = 0; il < n_layer_phys; ++il) {
        n_attn += !hparams.is_recr(il);
    }

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    stream_embd = create_tensor(tn(LLM_TENSOR_STREAM_EMBD, "weight"), {n_s, n_embd}, 0);

    depth_mem_state_write   = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_STATE_WRITE, "weight"), {n_embd, n_m}, 0);
    depth_mem_meta_write    = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_META_WRITE,  "weight"), {n_r + 3, n_m}, 0);
    depth_mem_entry_bias    = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_ENTRY_BIAS,  "weight"), {n_m, n_p*(n_attn + 1)}, 0);
    depth_mem_query         = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_QUERY,       "weight"), {n_embd, n_m}, 0);
    depth_mem_query_b       = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_QUERY,       "bias"),   {n_m}, 0);
    depth_mem_key           = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_KEY,         "weight"), {n_m, n_m}, 0);
    depth_mem_key_b         = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_KEY,         "bias"),   {n_m}, 0);
    depth_mem_value         = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_VALUE,       "weight"), {n_m, n_m}, 0);
    depth_mem_value_b       = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_VALUE,       "bias"),   {n_m}, 0);
    depth_mem_out           = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_OUT,         "weight"), {n_m, n_embd}, 0);
    depth_mem_out_b         = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_OUT,         "bias"),   {n_embd}, 0);
    depth_mem_route_state   = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_ROUTE_STATE, "weight"), {n_embd, n_r}, 0);
    depth_mem_route_retr    = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_ROUTE_RETR,  "weight"), {n_embd, n_r}, 0);
    depth_mem_route_diff    = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_ROUTE_DIFF,  "weight"), {n_embd, n_r}, 0);
    depth_mem_route_hist    = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_ROUTE_HIST,  "weight"), {n_r, n_r}, 0);
    depth_mem_route_b       = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_ROUTE,       "bias"),   {n_r}, 0);
    depth_mem_stream_gate   = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_STREAM_GATE, "weight"), {n_embd, n_s}, 0);
    depth_mem_stream_gate_b = create_tensor(tn(LLM_TENSOR_DEPTH_MEM_STREAM_GATE, "bias"),   {n_s}, 0);

    cont_hidden_state = create_tensor(tn(LLM_TENSOR_CONT_HIDDEN_STATE, "weight"), {n_embd, n_r}, 0);
    cont_hidden_retr  = create_tensor(tn(LLM_TENSOR_CONT_HIDDEN_RETR,  "weight"), {n_embd, n_r}, 0);
    cont_hidden_diff  = create_tensor(tn(LLM_TENSOR_CONT_HIDDEN_DIFF,  "weight"), {n_embd, n_r}, 0);
    cont_hidden_route = create_tensor(tn(LLM_TENSOR_CONT_HIDDEN_ROUTE, "weight"), {n_r, n_r}, 0);
    cont_hidden_b     = create_tensor(tn(LLM_TENSOR_CONT_HIDDEN,       "bias"),   {n_r}, 0);
    cont_output       = create_tensor(tn(LLM_TENSOR_CONT_OUTPUT,       "weight"), {n_r, 1}, 0);
    cont_output_b     = create_tensor(tn(LLM_TENSOR_CONT_OUTPUT,       "bias"),   {1}, 0);

    for (int il = 0; il < (int) n_layer_phys; ++il) {
        auto & layer = layers[il];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", il), {n_embd}, 0);
        layer.hc_attn_mix   = create_tensor(tn(LLM_TENSOR_HC_ATTN_MIX,   "weight", il), {n_s, n_s, n_p}, 0);
        layer.hc_attn_read  = create_tensor(tn(LLM_TENSOR_HC_ATTN_READ,  "weight", il), {n_s, n_p}, 0);
        layer.hc_attn_write = create_tensor(tn(LLM_TENSOR_HC_ATTN_WRITE, "weight", il), {n_s, n_p}, 0);
        layer.hc_ffn_mix    = create_tensor(tn(LLM_TENSOR_HC_FFN_MIX,    "weight", il), {n_s, n_s, n_p}, 0);
        layer.hc_ffn_read   = create_tensor(tn(LLM_TENSOR_HC_FFN_READ,   "weight", il), {n_s, n_p}, 0);
        layer.hc_ffn_write  = create_tensor(tn(LLM_TENSOR_HC_FFN_WRITE,  "weight", il), {n_s, n_p}, 0);
        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", il), {n_embd}, 0);

        if (hparams.is_recr(il)) {
            layer.ssm_in       = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", il), {n_embd, d_in_proj}, 0);
            layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), {d_conv, d_inner + 2*n_group*d_state}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   il), {d_inner + 2*n_group*d_state}, 0);
            layer.ssm_dt_b     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), {n_ssm_head}, 0);
            layer.ssm_norm     = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), {d_inner / n_group, n_group}, 0);
            layer.ssm_out      = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), {d_inner, n_embd}, 0);

            // no "weight" suffix for these
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, il), {1, n_ssm_head}, 0);
            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, il), {1, n_ssm_head}, 0);
        } else {
            const int64_t n_qkv = n_embd_head_k * (hparams.n_head(il) + 2*hparams.n_head_kv(il));
            const int64_t rank  = hparams.metis_pass_lora_rank;

            layer.wqkv                = create_tensor(tn(LLM_TENSOR_ATTN_QKV,            "weight", il), {n_embd, n_qkv}, 0);
            layer.wo                  = create_tensor(tn(LLM_TENSOR_ATTN_OUT,            "weight", il), {n_embd_head_k * hparams.n_head(il), n_embd}, 0);
            layer.attn_pass_lora_down = create_tensor(tn(LLM_TENSOR_ATTN_PASS_LORA_DOWN, "weight", il), {n_embd, rank}, 0);
            layer.attn_pass_lora_up   = create_tensor(tn(LLM_TENSOR_ATTN_PASS_LORA_UP,   "weight", il), {rank, n_qkv}, 0);
            layer.attn_pass_gate      = create_tensor(tn(LLM_TENSOR_ATTN_PASS_GATE,      "weight", il), {n_p}, 0);
        }

        const int64_t n_ff_exp = hparams.n_ff_exp(il);

        layer.ffn_latent_down  = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN,  "weight", il), {n_embd, n_latent}, 0);
        layer.ffn_latent_up    = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,    "weight", il), {n_latent, n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", il), {n_latent + n_r, n_expert}, 0);
        layer.ffn_gate_inp_b   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "bias",   il), {n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   il), {n_expert}, 0);
        layer.ffn_exp_embd     = create_tensor(tn(LLM_TENSOR_FFN_EXP_EMBD,     "weight", il), {n_r, n_expert}, 0);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", il), {n_latent, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", il), {n_ff_exp, n_latent, n_expert}, 0);
        layer.ffn_gate_shexp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,   "weight", il), {n_latent, n_ff_shexp}, 0);
        layer.ffn_up_shexp     = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,     "weight", il), {n_latent, n_ff_shexp}, 0);
        layer.ffn_down_shexp   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,   "weight", il), {n_ff_shexp, n_latent}, 0);
    }

    // the virtual layers of later passes share the physical layer's weights
    for (int il = n_layer_phys; il < n_layer; ++il) {
        layers[il] = layers[il % n_layer_phys];
    }
}

std::unique_ptr<llm_graph_context> llama_model_metis16::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// streams are kept stream-innermost: [n_stream, n_embd, n_tokens]
ggml_tensor * llama_model_metis16::graph::stream_mean(ggml_tensor * x) const {
    ggml_tensor * mean = ggml_mean(ctx0, ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1]*x->ne[2]));
    return ggml_reshape_2d(ctx0, mean, x->ne[1], x->ne[2]);
}

// returns the normalized sublayer input
ggml_tensor * llama_model_metis16::graph::hc_read(ggml_tensor * x, ggml_tensor * mix, ggml_tensor * read, ggml_tensor * norm,
        int p, ggml_tensor ** mixed, int il) const {
    const int64_t n_s = x->ne[0];
    ggml_tensor * x2 = ggml_reshape_2d(ctx0, x, n_s, x->ne[1]*x->ne[2]);

    ggml_tensor * m = ggml_view_2d(ctx0, mix, n_s, n_s, mix->nb[1], p*mix->nb[2]);
    *mixed = ggml_mul_mat(ctx0, m, x2);

    ggml_tensor * r   = ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, read, n_s, p*read->nb[1]), n_s, 1);
    ggml_tensor * src = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, r, x2), x->ne[1], x->ne[2]);
    return build_norm(src, norm, NULL, LLM_NORM_RMS, il);
}

ggml_tensor * llama_model_metis16::graph::hc_write(ggml_tensor * mixed, ggml_tensor * write, ggml_tensor * update, int p) const {
    const int64_t n_s = mixed->ne[0];
    ggml_tensor * w   = ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, write, n_s, p*write->nb[1]), 1, n_s);
    ggml_tensor * u   = ggml_reshape_2d(ctx0, update, 1, update->ne[0]*update->ne[1]);
    ggml_tensor * out = ggml_add(ctx0, mixed, ggml_mul_mat(ctx0, w, u));
    return ggml_reshape_3d(ctx0, out, n_s, update->ne[0], update->ne[1]);
}

// blends the retrieved entries into the streams; returns their summary
ggml_tensor * llama_model_metis16::graph::mem_read(const llama_model_metis16 & model, ggml_tensor ** x) const {
    const int64_t n_s = (*x)->ne[0];
    const int64_t n_m = model.n_mem;

    ggml_tensor * x_e = ggml_cont(ctx0, ggml_permute(ctx0, *x, 1, 0, 2, 3));

    ggml_tensor * q = ggml_mul_mat(ctx0, model.depth_mem_query, ggml_reshape_2d(ctx0, x_e, n_embd, n_s*n_tokens));
    q = ggml_reshape_3d(ctx0, ggml_add(ctx0, q, model.depth_mem_query_b), n_m, n_s, n_tokens);

    ggml_tensor * kq = ggml_mul_mat(ctx0, bank_k, q);
    kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f/sqrtf(float(n_m)), 0.0f);

    ggml_tensor * ret  = ggml_mul_mat(ctx0, bank_vt, kq);
    ggml_tensor * proj = ggml_mul_mat(ctx0, model.depth_mem_out, ggml_reshape_2d(ctx0, ret, n_m, n_s*n_tokens));
    proj = ggml_reshape_3d(ctx0, ggml_add(ctx0, proj, model.depth_mem_out_b), n_embd, n_s, n_tokens);

    ggml_tensor * gate = ggml_sum_rows(ctx0, ggml_mul(ctx0, x_e, model.depth_mem_stream_gate));
    gate = ggml_sigmoid(ctx0, ggml_add(ctx0, gate, ggml_reshape_2d(ctx0, model.depth_mem_stream_gate_b, 1, n_s)));

    ggml_tensor * proj_s = ggml_cont(ctx0, ggml_permute(ctx0, proj, 1, 0, 2, 3));
    *x = ggml_add(ctx0, *x, ggml_mul(ctx0, proj_s, ggml_reshape_3d(ctx0, gate, n_s, 1, n_tokens)));

    return stream_mean(proj_s);
}

void llama_model_metis16::graph::mem_write(const llama_model_metis16 & model, ggml_tensor * x, ggml_tensor * summary,
        ggml_tensor * entropy, ggml_tensor * confidence, ggml_tensor * cont, int entry) {
    const int64_t n_m = model.n_mem;

    if (cont == nullptr) {
        cont = ggml_scale(ctx0, confidence, 0.0f);
    }
    ggml_tensor * meta = ggml_concat(ctx0, ggml_concat(ctx0, summary, entropy, 0), ggml_concat(ctx0, confidence, cont, 0), 0);

    ggml_tensor * e = ggml_add(ctx0, ggml_mul_mat(ctx0, model.depth_mem_state_write, stream_mean(x)), ggml_mul_mat(ctx0, model.depth_mem_meta_write, meta));
    e = ggml_add(ctx0, e, ggml_view_1d(ctx0, model.depth_mem_entry_bias, n_m, entry*model.depth_mem_entry_bias->nb[1]));

    ggml_tensor * k = ggml_add(ctx0, ggml_mul_mat(ctx0, model.depth_mem_key, e), model.depth_mem_key_b);
    ggml_tensor * v = ggml_add(ctx0, ggml_mul_mat(ctx0, model.depth_mem_value, e), model.depth_mem_value_b);
    k = ggml_reshape_3d(ctx0, k, n_m, 1, n_tokens);
    v = ggml_reshape_3d(ctx0, v, 1, n_m, n_tokens);
    bank_k  = bank_k  ? ggml_concat(ctx0, bank_k,  k, 1) : k;
    bank_vt = bank_vt ? ggml_concat(ctx0, bank_vt, v, 0) : v;
}

ggml_tensor * llama_model_metis16::graph::route_features(const llama_model_metis16 & model, ggml_tensor * state,
        ggml_tensor * summary, ggml_tensor * diff, ggml_tensor * history) const {
    ggml_tensor * f = ggml_mul_mat(ctx0, model.depth_mem_route_state, state);
    if (summary) {
        f = ggml_add(ctx0, f, ggml_mul_mat(ctx0, model.depth_mem_route_retr, summary));
    }
    if (diff) {
        f = ggml_add(ctx0, f, ggml_mul_mat(ctx0, model.depth_mem_route_diff, diff));
    }
    if (history) {
        f = ggml_add(ctx0, f, ggml_mul_mat(ctx0, model.depth_mem_route_hist, history));
    }
    return ggml_tanh(ctx0, ggml_add(ctx0, f, model.depth_mem_route_b));
}

llama_model_metis16::graph::graph(const llama_model_metis16 & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_s      = model.n_stream;
    const int     n_p      = model.n_pass;
    const int     n_phys   = model.n_layer_phys;
    const int64_t n_r      = model.n_route;
    const int64_t n_hd     = hparams.n_embd_head_k();
    const int64_t n_exp    = hparams.n_expert;
    const int64_t n_used   = hparams.n_expert_used();

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * x = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, 1, n_embd, n_tokens), n_s, n_embd, n_tokens, 1);
    x = ggml_add(ctx0, x, model.stream_embd);
    cb(x, "streams_in", -1);

    ggml_tensor * prev_state = stream_mean(x);
    ggml_tensor * history    = nullptr; // routing history, [n_route, n_tokens]
    ggml_tensor * cont       = nullptr; // depth head reading of the last pass, [1, n_tokens]
    int n_entry = 0;

    for (int p = 0; p < n_p; ++p) {
        ggml_tensor * summ = nullptr;
        ggml_tensor * ent  = nullptr;
        ggml_tensor * conf = nullptr;

        for (int il_p = 0; il_p < n_phys; ++il_p) {
            const int il = p*n_phys + il_p;
            const auto & layer = model.layers[il];
            const bool is_attn  = !hparams.is_recr(il);
            const bool writes   = is_attn || il_p == n_phys - 1;

            ggml_tensor * retrieved = bank_k ? mem_read(model, &x) : nullptr;
            ggml_tensor * feats = route_features(model, stream_mean(x), retrieved, nullptr, history);

            ggml_tensor * mixed = nullptr;
            ggml_tensor * cur = hc_read(x, layer.hc_attn_mix, layer.hc_attn_read, layer.attn_norm, p, &mixed, il);
            if (is_attn) {
                ggml_tensor * qkv  = build_lora_mm(layer.wqkv, cur);
                ggml_tensor * lora = ggml_mul_mat(ctx0, layer.attn_pass_lora_up, ggml_mul_mat(ctx0, layer.attn_pass_lora_down, cur));
                qkv = ggml_add(ctx0, qkv, ggml_mul(ctx0, lora, ggml_view_1d(ctx0, layer.attn_pass_gate, 1, p*layer.attn_pass_gate->nb[0])));

                const int64_t n_head    = hparams.n_head(il);
                const int64_t n_head_kv = hparams.n_head_kv(il);
                const size_t  es        = ggml_element_size(qkv);
                ggml_tensor * Q = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, n_hd, n_head,    n_tokens, n_hd*es, qkv->nb[1], 0));
                ggml_tensor * K = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, n_hd, n_head_kv, n_tokens, n_hd*es, qkv->nb[1], n_hd*n_head*es));
                ggml_tensor * V = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, n_hd, n_head_kv, n_tokens, n_hd*es, qkv->nb[1], n_hd*(n_head + n_head_kv)*es));

                cur = build_attn(inp->get_attn(), layer.wo, nullptr, nullptr, Q, K, V, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_hd)), il);
            } else {
                cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            }
            cb(cur, "mixer_out", il);
            x = hc_write(mixed, layer.hc_attn_write, cur, p);

            cur = hc_read(x, layer.hc_ffn_mix, layer.hc_ffn_read, layer.ffn_norm, p, &mixed, il);
            {
                ggml_tensor * latent = ggml_mul_mat(ctx0, layer.ffn_latent_down, cur);
                ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, ggml_concat(ctx0, latent, feats, 0));
                logits = ggml_add(ctx0, logits, layer.ffn_gate_inp_b);

                // the balance bias picks the experts; the weights come from the unbiased logits
                ggml_tensor * selected = ggml_argsort_top_k(ctx0, ggml_add(ctx0, logits, layer.ffn_exp_probs_b), n_used);

                ggml_tensor * moe_out = build_moe_ffn(latent, nullptr, nullptr, nullptr, layer.ffn_down_exps, nullptr,
                        n_exp, n_used, LLM_FFN_SILU, false, 0.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT, il_p,
                        logits, layer.ffn_gate_up_exps, nullptr, nullptr, nullptr, selected);
                ggml_tensor * shexp = build_ffn(latent,
                        layer.ffn_up_shexp,   NULL, NULL,
                        layer.ffn_gate_shexp, NULL, NULL,
                        layer.ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il_p);
                cur = ggml_mul_mat(ctx0, layer.ffn_latent_up, ggml_add(ctx0, moe_out, shexp));
                cb(cur, "ffn_out", il);

                // routing summary: the chosen experts' identity vectors blended by their weights
                ggml_tensor * logits3 = ggml_reshape_3d(ctx0, logits, 1, n_exp, n_tokens);
                ggml_tensor * weights = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, logits3, selected), n_used, n_tokens));
                ggml_tensor * rows    = ggml_get_rows(ctx0, layer.ffn_exp_embd, ggml_reshape_1d(ctx0, ggml_cont(ctx0, selected), n_used*n_tokens));
                rows = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, rows, n_r, n_used, n_tokens), 1, 0, 2, 3));
                summ = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, rows, ggml_reshape_3d(ctx0, weights, n_used, 1, n_tokens)), n_r, n_tokens);

                if (writes) {
                    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
                    ent  = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, ggml_log(ctx0, ggml_clamp(ctx0, probs, 1.0e-9f, 1.0f))));
                    ent  = ggml_scale(ctx0, ent, -1.0f);
                    conf = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_exp, n_tokens),
                            ggml_reshape_2d(ctx0, ggml_argmax(ctx0, probs), 1, n_tokens));
                    conf = ggml_reshape_2d(ctx0, conf, 1, n_tokens);
                }
            }
            x = hc_write(mixed, layer.hc_ffn_write, cur, p);
            cb(x, "streams", il);

            history = history ? ggml_add(ctx0, ggml_scale(ctx0, history, 0.5f), ggml_scale(ctx0, summ, 0.5f)) : ggml_scale(ctx0, summ, 0.5f);

            if (is_attn) {
                mem_write(model, x, summ, ent, conf, cont, n_entry++);
            }
        }

        // end of the pass: one more memory read, the depth head, and an entry recording where the pass left the token
        ggml_tensor * retrieved = mem_read(model, &x);
        ggml_tensor * state = stream_mean(x);
        ggml_tensor * diff  = ggml_sub(ctx0, state, prev_state);
        ggml_tensor * feats = route_features(model, state, retrieved, diff, history);

        ggml_tensor * h = ggml_mul_mat(ctx0, model.cont_hidden_state, state);
        h = ggml_add(ctx0, h, ggml_mul_mat(ctx0, model.cont_hidden_retr,  retrieved));
        h = ggml_add(ctx0, h, ggml_mul_mat(ctx0, model.cont_hidden_diff,  diff));
        h = ggml_add(ctx0, h, ggml_mul_mat(ctx0, model.cont_hidden_route, feats));
        h = ggml_silu(ctx0, ggml_add(ctx0, h, model.cont_hidden_b));

        ggml_tensor * logit = ggml_add(ctx0, ggml_mul_mat(ctx0, model.cont_output, h), model.cont_output_b);
        ggml_tensor * scale = ggml_clamp(ctx0, ggml_sqrt(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, state))), 1.0e-6f, INFINITY);
        ggml_tensor * innov = ggml_sqrt(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, diff)));
        logit = ggml_add(ctx0, logit, ggml_scale_bias(ctx0, ggml_tanh(ctx0, ggml_div(ctx0, innov, scale)), 1.0f, -0.5f));
        cont  = ggml_sigmoid(ctx0, ggml_clamp(ctx0, logit, -model.cont_limit, model.cont_limit));
        cb(cont, "continuation", p);

        mem_write(model, x, summ, ent, conf, cont, n_entry++);
        prev_state = state;
    }

    ggml_tensor * cur = stream_mean(x);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

}
