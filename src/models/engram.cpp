#include "llama-engram.h"
#include "llama-impl.h"
#include "llama.h"
#include "models.h"

#include <cfloat>
#include <cstring>

llm_build_engram::llm_build_engram(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    const auto &  engram_hash  = model.engram_hash;
    const int32_t n_hash_heads = engram_hash.n_hash_heads();
    const int32_t d_per_head   = engram_hash.d_per_head();

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    for (int il = 0; il < n_layer; ++il) {
        // Engram Block
        if (model.layers[il].engram_mhe && engram_hash.has_layer(il)) {
            // 0. Build hash input for this layer
            auto inp_hash      = std::make_unique<llm_graph_input_engram_hash>(engram_hash, il);
            inp_hash->hash_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_hash_heads, n_tokens);
            ggml_set_input(inp_hash->hash_ids);
            ggml_tensor * hash_ids = inp_hash->hash_ids;
            cb(hash_ids, "engram_hash_ids", il);
            res->add_input(std::move(inp_hash));

            // 1. Flatten hash_ids
            ggml_tensor * flat_ids = ggml_reshape_1d(ctx0, hash_ids, (int64_t) n_hash_heads * n_tokens);

            // 2. Embedding lookup from multi-head hash ids
            ggml_tensor * emb = ggml_get_rows(ctx0, model.layers[il].engram_mhe, flat_ids);

            // 3. Reshape to (d_per_head * n_hash_heads, n_tokens) = (engram_hidden, n_tokens)
            emb = ggml_reshape_2d(ctx0, emb, (int64_t) d_per_head * n_hash_heads, n_tokens);

            // 4. Key projection + norm
            ggml_tensor * keys_mult  = ggml_mul_mat(ctx0, model.layers[il].engram_key_proj, emb);
            ggml_tensor * keys_projd = ggml_add(ctx0, keys_mult, model.layers[il].engram_key_bias);
            ggml_tensor * keys_reshaped =
                ggml_reshape_3d(ctx0, keys_projd, hparams.n_embd, hparams.n_hc_mult, n_tokens);
            ggml_tensor * keys_rms = ggml_rms_norm(ctx0, keys_reshaped, 0);
            ggml_tensor * keys     = ggml_mul(ctx0, keys_rms, model.layers[il].engram_norm1);

            // 5. Query from hidden state
            ggml_tensor * query_view = ggml_reshape_3d(ctx0, inpL, hparams.n_embd, 1, n_tokens);
            ggml_tensor * query      = ggml_repeat(ctx0, query_view, keys);
            ggml_tensor * query_rms  = ggml_rms_norm(ctx0, query, 0);
            ggml_tensor * query_mul  = ggml_mul(ctx0, query_rms, model.layers[il].engram_norm2);

            // 6. Gates = sigmoid(sum(keys * query) / sqrt(D))
            ggml_tensor * gate       = ggml_mul(ctx0, keys, query_mul);
            gate                     = ggml_sum_rows(ctx0, gate);
            gate                     = ggml_scale(ctx0, gate, 1.0f / sqrtf((float) hparams.n_embd));
            ggml_tensor * gate_sqrt  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, gate), 1e-6f, FLT_MAX));
            ggml_tensor * gate_sgn   = ggml_sgn(ctx0, gate);
            gate                     = ggml_mul(ctx0, gate_sgn, gate_sqrt);
            ggml_tensor * gate_final = ggml_sigmoid(ctx0, gate);

            // 7. Values
            ggml_tensor * val          = ggml_mul_mat(ctx0, model.layers[il].engram_val_proj, emb);
            ggml_tensor * val_biased   = ggml_add(ctx0, val, model.layers[il].engram_val_bias);
            val                        = ggml_reshape_3d(ctx0, val_biased, hparams.n_embd, 1, n_tokens);
            ggml_tensor * val_expanded = ggml_repeat(ctx0, val, keys);
            ggml_tensor * weighted_val = ggml_mul(ctx0, val_expanded, gate_final);

            // 8. ShortConv + residual
            const int64_t total_ch = (int64_t) hparams.n_embd * hparams.n_hc_mult;

            // Flatten to 2D (total_ch, T), then transpose to (T, total_ch) for conv_1d_dw
            ggml_tensor * conv_rms = ggml_rms_norm(ctx0, weighted_val, hparams.f_conv_rms_norm_eps);
            conv_rms               = ggml_mul(ctx0, conv_rms, model.layers[il].engram_sc_norm);
            ggml_tensor * conv_in  = ggml_reshape_2d(ctx0, conv_rms, total_ch, n_tokens);
            conv_in                = ggml_cont(ctx0, ggml_transpose(ctx0, conv_in));

            ggml_tensor * conv_w_2d   = model.layers[il].engram_sc_conv;
            ggml_tensor * conv_w      = ggml_reshape_3d(ctx0, conv_w_2d, conv_w_2d->ne[0], 1, conv_w_2d->ne[1]);
            const int     kernel_size = (int) conv_w->ne[0];
            const int     dilation    = hparams.engram_max_ngram_size;
            const int     padding     = (kernel_size - 1) * dilation;

            ggml_tensor * conv_out = ggml_conv_1d_dw(ctx0, conv_w, conv_in, 1, padding, dilation);
            conv_out               = ggml_silu(ctx0, conv_out);

            conv_out = ggml_view_2d(ctx0, conv_out, n_tokens, total_ch, conv_out->nb[1], 0);
            conv_out = ggml_cont(ctx0, ggml_transpose(ctx0, conv_out));
            conv_out = ggml_reshape_3d(ctx0, conv_out, hparams.n_embd, hparams.n_hc_mult, n_tokens);

            ggml_tensor * engram_result_full = ggml_add(ctx0, weighted_val, conv_out);
            cb(engram_result_full, "engram_result_full", il);

            // 9. Take hc=0 slice and add to hidden state
            ggml_tensor * engram_res_0 = ggml_view_3d(ctx0, engram_result_full, hparams.n_embd, 1, n_tokens,
                                                      engram_result_full->nb[1], engram_result_full->nb[2], 0);
            engram_res_0               = ggml_cont(ctx0, engram_res_0);
            engram_res_0               = ggml_reshape_2d(ctx0, engram_res_0, hparams.n_embd, n_tokens);
            inpL                       = ggml_add(ctx0, inpL, engram_res_0);
        }

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // MLA Self-Attention
        {
            ggml_tensor * q = NULL;

            if (model.layers[il].wq_a) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);
                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
            cb(kv, "kv", il);

            ggml_tensor * k_nope =
                ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                             ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                             ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
            cb(k_nope, "k_nope", il);

            ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                              ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                              ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                              ggml_row_size(kv->type, n_embd_head_qk_nope));
            Vcur               = ggml_cont(ctx0, Vcur);
            cb(Vcur, "Vcur", il);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
            cb(Kcur, "Kcur", il);

            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale,
                             il);
        }
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Attention residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // Dense FFN or MoE
        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                            model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(
                cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps, model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU,
                hparams.expert_weights_norm, hparams.expert_weights_scale, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, il);
            cb(moe_out, "ffn_moe_out", il);

            if (model.layers[il].ffn_gate_shexp) {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur, model.layers[il].ffn_up_shexp, NULL, NULL, model.layers[il].ffn_gate_shexp, NULL,
                              NULL, model.layers[il].ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
            } else {
                cur = moe_out;
            }
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    if (model.output_norm) {
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    }
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
