#include "models.h"

#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>

void llama_model_gear::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // Gear stores the actual attention head dimension in head_dim / rope.dimension_count.
    // This can differ from hidden_size / num_attention_heads.
    if (hparams.n_rot_full > 0) {
        hparams.n_embd_head_k_full = hparams.n_rot_full;
        hparams.n_embd_head_v_full = hparams.n_rot_full;
    }
    if (hparams.n_rot_swa > 0) {
        hparams.n_embd_head_k_swa = hparams.n_rot_swa;
        hparams.n_embd_head_v_swa = hparams.n_rot_swa;
    }

    uint32_t n_head_kv_mixer = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
        if (hparams.n_head_kv(il) > 0) {
            n_head_kv_mixer = hparams.n_head_kv(il);
        }
    }

    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        if (!ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer(), false)) {
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                hparams.is_swa_impl[il] = !hparams.is_recr(il);
            }
        }
    }

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    GGML_ASSERT(n_head_kv_mixer > 0);
    hparams.n_shortconv_state_size =
        2 * hparams.n_shortconv_l_cache * hparams.n_embd_head_k_full * n_head_kv_mixer;

    hparams.llm_ffn_op = LLM_FFN_GELU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        hparams.llm_ffn_op = llm_ffn_op_type_from_string(hidden_act, LLM_FFN_GELU);
    }

    switch (hparams.n_ff()) {
        case 6912: type = LLM_TYPE_700M; break;
        default:   type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gear::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    int64_t n_head_kv_mixer = 0;
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.n_head_kv(i) > 0) {
            n_head_kv_mixer = hparams.n_head_kv(i);
            break;
        }
    }
    GGML_ASSERT(n_head_kv_mixer > 0);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_embd_q    = n_embd_head * n_head;
    const int64_t n_embd_kv   = n_embd_head * n_head_kv_mixer;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

        if (hparams.is_recr(i)) {
            layer.gear_mix_wq         = create_tensor(tn(LLM_TENSOR_GEAR_MIX_Q,          "weight", i), { n_embd, n_embd_q  }, 0);
            layer.gear_mix_wk         = create_tensor(tn(LLM_TENSOR_GEAR_MIX_K,          "weight", i), { n_embd, n_embd_kv }, 0);
            layer.gear_mix_wv         = create_tensor(tn(LLM_TENSOR_GEAR_MIX_V,          "weight", i), { n_embd, n_embd_kv }, 0);
            layer.gear_mix_wo         = create_tensor(tn(LLM_TENSOR_GEAR_MIX_OUT,        "weight", i), { n_embd_q, n_embd  }, 0);
            layer.gear_mix_q_norm     = create_tensor(tn(LLM_TENSOR_GEAR_MIX_Q_NORM,     "weight", i), { n_embd_head }, 0);
            layer.gear_mix_k_norm     = create_tensor(tn(LLM_TENSOR_GEAR_MIX_K_NORM,     "weight", i), { n_embd_head }, 0);
            layer.gear_mix_key_conv   = create_tensor(tn(LLM_TENSOR_GEAR_MIX_KEY_CONV,   "weight", i), { hparams.n_shortconv_l_cache, n_embd_kv }, 0);
            layer.gear_mix_value_conv = create_tensor(tn(LLM_TENSOR_GEAR_MIX_VALUE_CONV, "weight", i), { hparams.n_shortconv_l_cache, n_embd_kv }, 0);
        } else {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head }, 0);

            create_tensor_qkv(layer, i, n_embd, n_embd_q, n_embd_kv, n_embd_kv, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_q, n_embd }, 0);
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), { n_embd }, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gear::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    }
    return std::make_unique<graph<false>>(*this, params);
}

template <bool iswa>
llama_model_gear::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    using inp_hybrid_type = std::conditional_t<iswa, llm_graph_input_mem_hybrid_iswa,  llm_graph_input_mem_hybrid>;
    using inp_attn_type   = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa,     llm_graph_input_attn_kv>;
    using mem_hybrid_ctx  = std::conditional_t<iswa, llama_memory_hybrid_iswa_context, llama_memory_hybrid_context>;

    int64_t n_head_kv_mixer = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (hparams.n_head_kv(il) > 0) {
            n_head_kv_mixer = hparams.n_head_kv(il);
            break;
        }
    }
    GGML_ASSERT(n_head_kv_mixer > 0);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_embd_q    = n_embd_head * n_head;
    const int64_t n_embd_kv   = n_embd_head * n_head_kv_mixer;
    const int64_t n_gqa       = n_head / n_head_kv_mixer;

    auto build_gear_norm = [this](ggml_tensor * cur, ggml_tensor * weight, int il) -> ggml_tensor * {
        ggml_tensor * normed = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
        cb(normed, "gear_norm", il);

        ggml_tensor * scaled = ggml_mul(ctx0, normed, weight);
        cb(scaled, "gear_norm_weight", il);

        return ggml_add(ctx0, normed, scaled);
    };

    auto expand_kv = [this, n_embd_head, n_head_kv_mixer, n_gqa](ggml_tensor * cur, int64_t n_seq_tokens, int64_t n_seqs, int il) -> ggml_tensor * {
        GGML_UNUSED(il);
        if (n_gqa == 1) {
            return ggml_reshape_3d(ctx0, cur, n_embd_head * n_head_kv_mixer, n_seq_tokens, n_seqs);
        }

        ggml_tensor * cur_4d = ggml_reshape_4d(ctx0, cur, n_embd_head, 1, n_head_kv_mixer, n_seq_tokens * n_seqs);
        cur_4d = ggml_repeat_4d(ctx0, cur_4d, n_embd_head, n_gqa, n_head_kv_mixer, n_seq_tokens * n_seqs);
        return ggml_reshape_3d(ctx0, cur_4d, n_embd_head * n_head_kv_mixer * n_gqa, n_seq_tokens, n_seqs);
    };

    auto build_attn_block = [&model, this, build_gear_norm](ggml_tensor *   cur,
                                                            ggml_tensor *   inp_pos,
                                                            inp_attn_type * inp_attn,
                                                            int             il) -> ggml_tensor * {
        const int64_t n_embd_head_l = hparams.n_embd_head_k(il);
        const int64_t n_head_kv_l   = hparams.n_head_kv(il);
        const int64_t n_rot_l       = hparams.n_rot(il);
        const float   freq_base_l   = model.get_rope_freq_base(cparams, il);
        const float   freq_scale_l  = model.get_rope_freq_scale(cparams, il);

        GGML_ASSERT(n_head_kv_l > 0);
        GGML_ASSERT(hparams.n_embd_head_k(il) == hparams.n_embd_head_v(il));

        auto [q, k, v] = build_qkv(model.layers[il], cur, n_embd_head_l, n_head, n_head_kv_l, il);

        q = build_gear_norm(q, model.layers[il].attn_q_norm, il);
        cb(q, "Qcur_normed", il);

        k = build_gear_norm(k, model.layers[il].attn_k_norm, il);
        cb(k, "Kcur_normed", il);

        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, nullptr,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q, "Qcur_rope", il);
        cb(k, "Kcur_rope", il);

        const float kq_scale = 1.0f / sqrtf(float(n_embd_head_l));
        cur = build_attn(inp_attn,
                model.layers[il].wo, nullptr, model.layers[il].wo_s,
                q, k, v, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        return cur;
    };

    auto build_local_mixer_block = [&model, this, build_gear_norm, expand_kv, n_embd_head, n_head_kv_mixer, n_embd_q, n_embd_kv](
                                            ggml_tensor *        cur,
                                            llm_graph_input_rs * inp_recr,
                                            int                  il) -> ggml_tensor * {
        const auto * mctx_cur = static_cast<const mem_hybrid_ctx *>(mctx)->get_recr();
        const uint32_t kv_head = mctx_cur->get_head();

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;
        const int64_t l_cache      = hparams.n_shortconv_l_cache;

        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
        GGML_ASSERT(l_cache > 1);

        ggml_tensor * q = build_lora_mm(model.layers[il].gear_mix_wq, cur);
        ggml_tensor * k = build_lora_mm(model.layers[il].gear_mix_wk, cur);
        ggml_tensor * v = build_lora_mm(model.layers[il].gear_mix_wv, cur);
        cb(q, "mix_q", il);
        cb(k, "mix_k", il);
        cb(v, "mix_v", il);

        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
        k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv_mixer, n_tokens);
        q = build_gear_norm(q, model.layers[il].gear_mix_q_norm, il);
        k = build_gear_norm(k, model.layers[il].gear_mix_k_norm, il);
        cb(q, "mix_q_normed", il);
        cb(k, "mix_k_normed", il);

        q = ggml_cont_3d(ctx0, q, n_embd_q,  n_seq_tokens, n_seqs);
        k = ggml_cont_3d(ctx0, k, n_embd_kv, n_seq_tokens, n_seqs);
        v = ggml_reshape_3d(ctx0, v, n_embd_kv, n_seq_tokens, n_seqs);

        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
        ggml_tensor * conv_rs = build_rs(inp_recr, conv_states_all, hparams.n_embd_r(), n_seqs);
        ggml_tensor * conv_state = ggml_reshape_3d(ctx0, conv_rs, l_cache, 2 * n_embd_kv, n_seqs);

        ggml_tensor * key_state = ggml_view_3d(ctx0, conv_state, l_cache, n_embd_kv, n_seqs,
                conv_state->nb[1], conv_state->nb[2], 0);
        ggml_tensor * value_state = ggml_view_3d(ctx0, conv_state, l_cache, n_embd_kv, n_seqs,
                conv_state->nb[1], conv_state->nb[2], l_cache * n_embd_kv * ggml_element_size(conv_state));

        ggml_tensor * key_seq   = ggml_transpose(ctx0, k);
        ggml_tensor * value_seq = ggml_transpose(ctx0, v);

        ggml_tensor * key_prev = ggml_view_3d(ctx0, key_state, l_cache - 1, n_embd_kv, n_seqs,
                key_state->nb[1], key_state->nb[2], key_state->nb[0]);
        ggml_tensor * value_prev = ggml_view_3d(ctx0, value_state, l_cache - 1, n_embd_kv, n_seqs,
                value_state->nb[1], value_state->nb[2], value_state->nb[0]);

        ggml_tensor * key_conv_in   = ggml_concat(ctx0, key_prev,   key_seq,   0);
        ggml_tensor * value_conv_in = ggml_concat(ctx0, value_prev, value_seq, 0);

        k = ggml_ssm_conv(ctx0, key_conv_in, model.layers[il].gear_mix_key_conv);
        v = ggml_ssm_conv(ctx0, value_conv_in, model.layers[il].gear_mix_value_conv);
        cb(k, "mix_key_conv", il);
        cb(v, "mix_value_conv", il);

        ggml_tensor * key_state_in   = ggml_concat(ctx0, key_state,   key_seq,   0);
        ggml_tensor * value_state_in = ggml_concat(ctx0, value_state, value_seq, 0);
        ggml_tensor * new_key_state = ggml_view_3d(ctx0, key_state_in, l_cache, n_embd_kv, n_seqs,
                key_state_in->nb[1], key_state_in->nb[2], (key_state_in->ne[0] - l_cache) * key_state_in->nb[0]);
        ggml_tensor * new_value_state = ggml_view_3d(ctx0, value_state_in, l_cache, n_embd_kv, n_seqs,
                value_state_in->nb[1], value_state_in->nb[2], (value_state_in->ne[0] - l_cache) * value_state_in->nb[0]);

        ggml_tensor * new_state = ggml_concat(ctx0, new_key_state, new_value_state, 1);
        new_state = ggml_cont_3d(ctx0, new_state, l_cache, 2 * n_embd_kv, n_seqs);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                ggml_view_1d(ctx0, new_state, hparams.n_embd_r() * n_seqs, 0),
                ggml_view_1d(ctx0, conv_states_all, hparams.n_embd_r() * n_seqs,
                    kv_head * hparams.n_embd_r() * ggml_element_size(conv_states_all))));

        k = expand_kv(k, n_seq_tokens, n_seqs, il);
        v = expand_kv(v, n_seq_tokens, n_seqs, il);
        cb(k, "mix_k_expanded", il);
        cb(v, "mix_v_expanded", il);

        ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, q, k));
        cb(gate, "mix_gate", il);

        cur = ggml_mul(ctx0, gate, v);
        cur = ggml_reshape_2d(ctx0, cur, n_embd_q, n_tokens);
        cur = build_lora_mm(model.layers[il].gear_mix_wo, cur);
        cb(cur, "mix_output", il);

        return cur;
    };

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cur = ggml_scale(ctx0, cur, ubatch.token ? sqrtf(float(n_embd)) : 1.0f);
    cb(cur, "embed_scaled", -1);

    inp_hybrid_type * inp_hybrid = nullptr;
    if constexpr (iswa) {
        inp_hybrid = build_inp_mem_hybrid_iswa();
    } else {
        inp_hybrid = build_inp_mem_hybrid();
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = cur;

        cur = build_gear_norm(cur, model.layers[il].attn_norm, il);
        cb(cur, "input_layernorm", il);

        if (hparams.is_recr(il)) {
            cur = build_local_mixer_block(cur, inp_hybrid->get_recr(), il);
        } else {
            cur = build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il);
        }

        cur = build_gear_norm(cur, model.layers[il].attn_post_norm, il);
        cb(cur, "post_attention_layernorm", il);

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "attn_residual", il);

        residual = cur;

        cur = build_gear_norm(cur, model.layers[il].ffn_norm, il);
        cb(cur, "pre_feedforward_layernorm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr, hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn", il);

        cur = build_gear_norm(cur, model.layers[il].ffn_post_norm, il);
        cb(cur, "post_feedforward_layernorm", il);

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "ffn_residual", il);
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_gear_norm(cur, model.output_norm, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (!cparams.embeddings) {
        cur = build_lora_mm(model.output, cur, model.output_s);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_gear::graph<true>;
template struct llama_model_gear::graph<false>;
