#include "models.h"

// diffusion_gemma4 reuses the gemma4 decoder block (tensor layout + per-layer math) but
// runs as a bidirectional (non-causal, no KV cache) denoiser over the canvas, and applies
// the self-conditioning transform to the input embeddings.
//
// This implements a single bidirectional denoising pass with no prompt context. The
// self-conditioning module, with soft-conditioning = 0 (first denoising step), reduces to
// a scale-less RMS norm of the scaled input embeddings. The soft-conditioning input path
// (later denoising steps) and the encoder-KV cross-attention (prompted generation) are
// layered on top in a later step.
//
// NOTE: this graph assumes the canvas length does not exceed the sliding window, so the
// sliding/full attention layers are all equivalent to full (bidirectional) attention.

void llama_model_diffusion_gemma4::load_arch_hparams(llama_model_loader & ml) {
    // reuse the gemma4 hparam loading (sliding window pattern, dual head dims, MoE, rope,
    // softcapping, layer types, ...)
    llama_model_gemma4::load_arch_hparams(ml);

    // the diffusion decoder attends bidirectionally
    hparams.causal_attn = false;
}

void llama_model_diffusion_gemma4::load_arch_tensors(llama_model_loader & ml) {
    // load the shared gemma4 tensors (token embd, attention, dual dense+MoE FFN, norms,
    // per-layer layer_scalar, output)
    llama_model_gemma4::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // self_conditioning is a gated MLP at hidden_size -> intermediate_size -> hidden_size
    const int64_t n_ff_sc = n_ff;

    self_cond_norm = create_tensor(tn(LLM_TENSOR_SELF_COND_NORM, "weight"), {n_embd},           0);
    self_cond_gate = create_tensor(tn(LLM_TENSOR_SELF_COND_GATE, "weight"), {n_embd,  n_ff_sc}, 0);
    self_cond_up   = create_tensor(tn(LLM_TENSOR_SELF_COND_UP,   "weight"), {n_embd,  n_ff_sc}, 0);
    self_cond_down = create_tensor(tn(LLM_TENSOR_SELF_COND_DOWN, "weight"), {n_ff_sc, n_embd},  0);
}

std::unique_ptr<llm_graph_context> llama_model_diffusion_gemma4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_diffusion_gemma4::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // scaled word embeddings (sqrt(hidden_size)); raw embeddings input is not scaled
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // self-conditioning: sc_input = post_norm(inputs_embeds + sc_mlp(pre_norm(soft_embeds)))
    // soft_embeds is the previous denoising step's soft-embeddings
    // (softmax(prev_logits) @ embed * scale), zero on the first step.
    // TODO(diffusion sampler): the block-diffusion sampler feeds the real soft-embeddings
    // here each step; until then it is zero, which (since rms_norm(0)=0 -> sc_mlp=0) makes
    // sc_input == scale-less post-norm of the scaled embeddings (the verified step-0 case).
    const auto & dmodel = static_cast<const llama_model_diffusion_gemma4 &>(model);
    {
        ggml_tensor * soft = ggml_scale(ctx0, inpL, 0.0f); // placeholder zero soft-embeddings
        ggml_tensor * scn  = build_norm(soft, dmodel.self_cond_norm, nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * sc   = build_ffn(scn,
                dmodel.self_cond_up,   nullptr, nullptr,
                dmodel.self_cond_gate, nullptr, nullptr,
                dmodel.self_cond_down, nullptr, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, -1);
        inpL = ggml_add(ctx0, inpL, sc);
    }
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps); // scale-less post_norm
    cb(inpL, "self_cond_input", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    // bidirectional attention, no KV cache
    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        // full_attention layers use rope_freqs for proportional rope
        ggml_tensor * freq_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        // attention norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention (QK norm + scale-less V norm, mirrors Gemma4Attention)
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        cb(Kcur, "Kcur", il);
        // global (full-attention) layers have no v_proj: V = K (before norms)
        ggml_tensor * Vcur = model.layers[il].wv
                                ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                                : Kcur;
        cb(Vcur, "Vcur", il);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps); // scale-less v_norm
        cb(Kcur, "Kcur_normed", il);
        cb(Vcur, "Vcur_normed", il);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur_pos", il);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, model.layers[il].wo_s,
                         Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward: dense MLP (shared expert) + routed MoE, summed (mirrors gemma4)
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            ggml_tensor * cur_mlp = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);
            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp, model.layers[il].ffn_post_norm_1, nullptr, LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            ggml_tensor * cur_moe = build_norm(attn_out, model.layers[il].ffn_pre_norm_2, nullptr, LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // router operates on attn_out (scale-less norm * 1/sqrt(n_embd) * router scale)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp);
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe, model.layers[il].ffn_post_norm_2, nullptr, LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
