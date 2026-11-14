#pragma once

#include "models.h"

// Shared Gemma-4 backbone helpers used by both gemma4.cpp and diffusion-gemma.cpp, so the attention
// projection and dense-MLP + 128-expert-MoE blocks have a single source of truth. Every op/arg mirrors
// the original gemma4 forward; DiffusionGemma's *_s LoRA-scale tensors resolve to nullptr.

// Q projection + per-head q-norm + partial/proportional rope. Mirrors gemma4.cpp's Q block.
static inline ggml_tensor * gemma4_build_q(
        const llm_graph_context & g, const llama_model & model,
        ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & hp = g.hparams;
    const int64_t n_embd_head = hp.n_embd_head_k(il);
    const int64_t n_head      = hp.n_head(il);
    const float   freq_base_l  = model.get_rope_freq_base (g.cparams, il);
    const float   freq_scale_l = model.get_rope_freq_scale(g.cparams, il);
    const int     n_rot_l      = hp.n_rot(il);
    // full_attention (non-SWA) layers use rope_freqs for proportional rope
    ggml_tensor * freq_factors = hp.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

    ggml_tensor * Qcur = g.build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    g.cb(Qcur, "Qcur", il);
    Qcur = ggml_reshape_3d(g.ctx0, Qcur, n_embd_head, n_head, g.n_tokens);
    Qcur = g.build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    g.cb(Qcur, "Qcur_normed", il);
    Qcur = ggml_rope_ext(g.ctx0, Qcur, inp_pos, freq_factors, n_rot_l, g.rope_type, g.n_ctx_orig,
                         freq_base_l, freq_scale_l, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    g.cb(Qcur, "Qcur_pos", il);
    return Qcur;
}

// K/V projection + k-norm + V no-scale rms-norm + K rope. V == K (raw k_proj) when v_proj is absent
// (Gemma-4 global layers). Mirrors gemma4.cpp's K/V block.
static inline llm_graph_qkv gemma4_build_kv(
        const llm_graph_context & g, const llama_model & model,
        ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & hp = g.hparams;
    const int64_t n_embd_head = hp.n_embd_head_k(il);
    const int64_t n_head_kv   = hp.n_head_kv(il);
    const float   freq_base_l  = model.get_rope_freq_base (g.cparams, il);
    const float   freq_scale_l = model.get_rope_freq_scale(g.cparams, il);
    const int     n_rot_l      = hp.n_rot(il);
    ggml_tensor * freq_factors = hp.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

    ggml_tensor * Kcur = g.build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    g.cb(Kcur, "Kcur", il);
    ggml_tensor * Vcur = model.layers[il].wv
                            ? g.build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                            : Kcur;
    g.cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(g.ctx0, Kcur, n_embd_head, n_head_kv, g.n_tokens);
    Vcur = ggml_reshape_3d(g.ctx0, Vcur, n_embd_head, n_head_kv, g.n_tokens);

    Kcur = g.build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    Vcur = ggml_rms_norm(g.ctx0, Vcur, hp.f_norm_rms_eps); // v-norm: no scale
    g.cb(Kcur, "Kcur_normed", il);
    g.cb(Vcur, "Vcur_normed", il);

    Kcur = ggml_rope_ext(g.ctx0, Kcur, inp_pos, freq_factors, n_rot_l, g.rope_type, g.n_ctx_orig,
                         freq_base_l, freq_scale_l, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    g.cb(Kcur, "Kcur_pos", il);

    return { nullptr, Kcur, Vcur };
}

// Dense-MLP (shared expert) + 128-expert MoE on the post-attention residual attn_out, then ffn_post_norm
// + residual. Mirrors gemma4.cpp's feed-forward block (custom router: rms_norm(no-scale) -> *1/sqrt(n_embd)
// -> *ffn_gate_inp_s -> gate proj -> softmax top-k).
static inline ggml_tensor * gemma4_build_ffn_moe(
        const llm_graph_context & g, const llama_model & model,
        ggml_tensor * attn_out, int il) {
    const auto & hp    = g.hparams;
    const auto & layer = model.layers[il];
    const int64_t n_embd = g.n_embd;
    ggml_tensor * cur;

    const bool is_moe_layer = layer.ffn_gate_inp != nullptr;
    if (is_moe_layer) {
        // dense MLP (shared expert)
        ggml_tensor * cur_mlp = g.build_norm(attn_out, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        g.cb(cur_mlp, "ffn_norm_1", il);
        cur_mlp = g.build_ffn(cur_mlp,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
        cur_mlp = g.build_norm(cur_mlp, layer.ffn_post_norm_1, nullptr, LLM_NORM_RMS, il);
        g.cb(cur_mlp, "ffn_mlp", il);

        // MoE (router operates on the UNNORMED post-attention residual attn_out)
        ggml_tensor * cur_moe = g.build_norm(attn_out, layer.ffn_pre_norm_2, nullptr, LLM_NORM_RMS, il);
        g.cb(cur_moe, "ffn_norm_2", il);
        ggml_tensor * tmp = ggml_rms_norm(g.ctx0, attn_out, hp.f_norm_rms_eps);
        tmp = ggml_scale(g.ctx0, tmp, 1.0f / sqrtf((float) n_embd));
        tmp = ggml_mul(g.ctx0, tmp, layer.ffn_gate_inp_s);
        ggml_tensor * logits = g.build_lora_mm(layer.ffn_gate_inp, tmp);
        g.cb(logits, "ffn_moe_logits", il);
        cur_moe = g.build_moe_ffn(cur_moe,
                nullptr,
                layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                nullptr, g.n_expert, g.n_expert_used,
                LLM_FFN_GELU, true, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il, logits,
                layer.ffn_gate_up_exps, layer.ffn_up_exps_s, layer.ffn_gate_exps_s, layer.ffn_down_exps_s);
        cur_moe = g.build_norm(cur_moe, layer.ffn_post_norm_2, nullptr, LLM_NORM_RMS, il);
        g.cb(cur_moe, "ffn_moe", il);

        cur = ggml_add(g.ctx0, cur_mlp, cur_moe);
        g.cb(cur, "ffn_moe_combined", il);
    } else {
        cur = g.build_norm(attn_out, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        g.cb(cur, "ffn_norm", il);
        cur = g.build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
        g.cb(cur, "ffn_out", il);
    }
    cur = g.build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
    g.cb(cur, "ffn_post_norm", il);
    cur = ggml_add(g.ctx0, cur, attn_out); // residual
    return cur;
}
