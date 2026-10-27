#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_lfm3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
    }

    // LFM3 always mixes full and sliding window attention layers
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    if (!hparams.is_swa_any()) {
        throw std::runtime_error("LFM3 requires sliding window attention layers");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa);

    switch (hparams.n_ff()) {
        case 7680: type = LLM_TYPE_1_7B; break;
        default:   type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_lfm3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        // absent when use_sandwich_norm is false
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        if ((layer.attn_post_norm == nullptr) != (layer.ffn_post_norm == nullptr)) {
            throw std::runtime_error("LFM3 requires both post norms or neither");
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        if (!hparams.is_recr(i)) {
            // note: n_embd_head_k * n_head may differ from n_embd
            const int64_t n_embd_q = n_embd_head_k * n_head;
            const int64_t n_embd_k = hparams.n_embd_k_gqa(i);
            const int64_t n_embd_v = hparams.n_embd_v_gqa(i);
            if (n_embd_k != n_embd_v) {
                throw std::runtime_error("LFM3 requires equal key and value sizes");
            }

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

            create_tensor_qkv(layer, i, n_embd, n_embd_q, n_embd_k, n_embd_v, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head}, 0);

            if (!hparams.is_swa(i)) {
                layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head_k / 2}, rope_freqs_flag);
                rope_freqs_flag = TENSOR_DUPLICATED;
            }
        } else {
            layer.shortconv.conv     = create_tensor(tn(LLM_TENSOR_SHORTCONV_CONV,    "weight", i), {hparams.n_shortconv_l_cache, n_embd}, 0);
            layer.shortconv.in_proj  = create_tensor(tn(LLM_TENSOR_SHORTCONV_INPROJ,  "weight", i), {n_embd, 3 * n_embd}, 0);
            layer.shortconv.out_proj = create_tensor(tn(LLM_TENSOR_SHORTCONV_OUTPROJ, "weight", i), {n_embd, n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_lfm3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_lfm3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_lfm_base(params) {
    auto build_attn_block = [&model, this](ggml_tensor *                  cur_in,
                                           ggml_tensor *                  inp_pos,
                                           llm_graph_input_attn_kv_iswa * inp_attn,
                                           int                            il) -> ggml_tensor * {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head_il   = hparams.n_head(il);
        const int64_t n_head_kv   = hparams.n_head_kv(il);

        const int     n_rot_l      = hparams.n_rot(il);
        const float   freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float   freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        auto [q, k, v] = build_qkv(model.layers[il], cur_in, n_embd_head, n_head_il, n_head_kv, il);

        q = build_norm(q, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);

        q = ggml_rope_ext(ctx0, q, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig,
                          freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig,
                          freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);

        ggml_tensor * attn_out = build_attn(inp_attn,
                NULL, NULL, NULL,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        cb(attn_out, "attn_out", il);

        ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur_in);
        gate = ggml_sigmoid(ctx0, gate);
        cb(gate, "model.layers.{}.self_attn.gate", il);

        ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head, n_head_il, n_tokens);
        ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,     1,           n_head_il, n_tokens);
        attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
        attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head * n_head_il, n_tokens);

        ggml_tensor * out = build_lora_mm(model.layers[il].wo, attn_out, model.layers[il].wo_s);
        cb(out, "model.layers.{}.self_attn.out_proj", il);

        return out;
    };

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_build_forward_expand(gf, cur);

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        auto * prev_cur = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        if (hparams.is_recr(il)) {
            const auto * mctx_recr = static_cast<const llama_memory_hybrid_iswa_context *>(mctx)->get_recr();
            cur = build_lfm_shortconv(model, mctx_recr, inp_hybrid->get_recr(), cur, il);
        } else {
            cur = build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il);
        }

        if (model.layers[il].attn_post_norm) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "model.layers.{}.post_operator_norm", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }

        cur = ggml_add(ctx0, prev_cur, cur);

        auto * ffn_norm_out = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        ggml_tensor * ffn_out = build_lfm_dense_ffn(model, ffn_norm_out, il);

        if (model.layers[il].ffn_post_norm) {
            ffn_out = build_norm(ffn_out, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(ffn_out, "model.layers.{}.post_ffn_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
