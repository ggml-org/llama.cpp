#include "models.h"

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    hparams.n_embd_inp_impl = hparams.n_embd_out();

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.f_attention_scale = 1.0f;

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn == hparams.n_layer_all && "n_layer_nextn must be == n_layer_impl");

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,           hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,     hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,     hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,   hparams.n_embd_head_v_swa);
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 assistant requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 assistant requires n_embd_head_k_swa == n_embd_head_v_swa");
    }
    if (hparams.n_embd_out() == n_embd) {
        throw std::runtime_error("Gemma 4 assistant requires embedding_length_out to carry the target hidden size");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);

    const int64_t n_embd_backbone = hparams.n_embd_inp();
    nextn_proj_post = create_tensor(tn(LLM_TENSOR_NEXTN_PROJ_POST, "weight"), { n_embd, n_embd_backbone }, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer_nextn; ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff        = hparams.n_ff(i);

        if (i == 0) {
            nextn_proj_pre = create_tensor(tn(LLM_TENSOR_NEXTN_PROJ_PRE, "weight", i), { 2*n_embd_backbone, n_embd }, 0);
        }

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), { n_embd, n_embd_head*n_head }, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), { n_embd_head*n_head, n_embd }, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), { n_embd_head }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1u }, 0);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head/2 }, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), { n_embd }, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
    const int64_t n_embd_backbone = hparams.n_embd_inp();

    ggml_tensor * inp_tokens;
    ggml_tensor * inp_h;
    {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_backbone);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        inp_tokens = inp->tokens;
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, ubatch.n_tokens);
        cb(inp->embd, "inp_h", -1);
        ggml_set_input(inp->embd);
        inp_h = inp->embd;
        res->t_inp_embd = inp->embd;

        res->add_input(std::move(inp));
    }

    GGML_ASSERT(cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(cparams.ctx_other);

    ggml_tensor * x = ggml_get_rows(ctx0, model_other->tok_embd, inp_tokens);
    x = ggml_scale(ctx0, x, sqrtf((float) n_embd_backbone));
    cb(x, "inp_embd_target", -1);

    ggml_tensor * xh = ggml_concat(ctx0, x, inp_h, 0);
    cb(xh, "inp_xh", -1);

    ggml_tensor * cur = ggml_mul_mat(ctx0, model.nextn_proj_pre, xh);
    cb(cur, "pre_proj", -1);

    auto *        inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer_nextn; ++il) {
        const bool is_swa = hparams.is_swa(il);

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        ggml_tensor * cur_norm = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur_norm, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur_norm);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);

        if (il == n_layer_nextn - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        cb(cur, "out_scaled", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * logits = build_lora_mm(model.output, cur);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_tensor * h_next = ggml_mul_mat(ctx0, model.nextn_proj_post, cur);
    cb(h_next, "h_nextn", -1);
    res->t_h_nextn = h_next;

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, h_next);
}
