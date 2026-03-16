#include "models.h"

ggml_tensor * llm_build_step35_iswa::build_layer(
        ggml_tensor * inpL,
                int   il,
        ggml_tensor * inp_pos,
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * inp_out_ids) {

    ggml_tensor * cur;
    ggml_tensor * inpSA = inpL;

    const uint32_t n_head_l    = hparams.n_head(il);
    const uint32_t n_head_kv_l = hparams.n_head_kv(il);

    const float freq_base_l  = model.get_rope_freq_base(cparams, il);
    const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

    cur = inpL;

    cb(cur, "attn_norm_in", il);

    // self-attention
    {
        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);

        if (model.layers[il].attn_q_norm) {
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
        }
        if (model.layers[il].attn_k_norm) {
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);
        }

        const bool is_swa = hparams.is_swa(il);
        ggml_tensor * rope_factors = is_swa ? nullptr : model.get_rope_factors(cparams, il);
        const int64_t n_rot_l = is_swa ? hparams.n_rot : (hparams.n_rot / 2);
        Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, rope_factors,
            n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
            ext_factor, attn_factor, beta_fast, beta_slow
        );
        Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, rope_factors,
            n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
            ext_factor, attn_factor, beta_fast, beta_slow
        );
        cb(Qcur, "Qcur_pos", il);
        cb(Kcur, "Kcur_pos", il);

        const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
        ggml_tensor * attn_out = build_attn(inp_attn,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(attn_out, "attn_out", il);

        if (model.layers[il].wqkv_gate) {
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur);
            cb(gate, "attn_gate", il);

            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sigmoid", il);

            ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head_l, n_tokens);
            ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,       1,          n_head_l, n_tokens);
            cb(gate_3d, "attn_gate_3d", il);

            attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
            cb(attn_3d, "attn_gated_3d", il);

            attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v * n_head_l, n_tokens);
            cb(attn_out, "attn_gated", il);
        }

        cur = build_lora_mm(model.layers[il].wo, attn_out);
        cb(cur, "attn_proj", il);
    }

    if (inp_out_ids) {
        cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "ffn_norm", il);

    if (model.layers[il].ffn_gate_inp == nullptr) {
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, nullptr,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    } else {
        const bool  norm_w  = hparams.expert_weights_norm;
        const float w_scale = hparams.expert_weights_scale;
        const bool  scale_w = w_scale != 0.0f;
        ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                norm_w, scale_w, w_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * sh_out = build_ffn(cur,
                model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                model.layers[il].ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(sh_out, "ffn_shared_out", il);

        cur = ggml_add(ctx0, moe_out, sh_out);
        cb(cur, "ffn_out", il);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}

llm_build_step35_iswa::llm_build_step35_iswa(const llama_model & model, const llm_graph_params & params)
        : llm_graph_context(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    const int n_layer_main = n_layer - (int) hparams.nextn_predict_layers;

    if (mtp_op_type != LLM_MTP_OP_NONE) {
        // === MTP graph branch ===
        GGML_ASSERT(mtp_hidden_state != nullptr);
        const int il = mtp_layer_idx;
        GGML_ASSERT(il >= n_layer_main && il < n_layer);
        const auto & layer = model.layers[il];

        // 1. token embedding
        ggml_tensor * inp_tokens = build_inp_embd(
                layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd);

        ggml_tensor * inp_pos  = build_inp_pos();
        auto        * inp_attn = build_attn_inp_kv_iswa();

        // 2. MTP hidden state input
        ggml_tensor * prev_hidden = build_inp_mtp_hidden_state();

        // 3. enorm / hnorm (Gemma-style, weights already contain +1 offset)
        ggml_tensor * inp_normed = build_norm(inp_tokens, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
        cb(inp_normed, "mtp_enorm", il);

        ggml_tensor * hid_normed = build_norm(prev_hidden, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
        cb(hid_normed, "mtp_hnorm", il);

        // 4. concat + projection: [2*n_embd, n_tokens] -> [n_embd, n_tokens]
        ggml_tensor * concat = ggml_concat(ctx0, inp_normed, hid_normed, 0);
        cb(concat, "mtp_concat", il);

        cur = build_lora_mm(layer.nextn.eh_proj, concat);
        cb(cur, "mtp_eh_proj", il);

        // 5. full decoder layer (attention + MoE FFN with KV cache R/W)
        cur = build_layer(cur, il, inp_pos, inp_attn, nullptr);

        // 6. save hidden state for next MTP step
        res->t_embd = cur;

        // 7. shared head -> logits
        if (layer.nextn.shared_head_norm) {
            cur = build_norm(cur, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, -1);
        } else {
            cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        }
        cb(cur, "mtp_head_norm", il);

        if (layer.nextn.shared_head_head) {
            cur = build_lora_mm(layer.nextn.shared_head_head, cur);
        } else {
            cur = build_lora_mm(model.output, cur);
        }
        cb(cur, "mtp_logits", il);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
        return;
    }

    // === Main model graph ===
    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer_main; ++il) {
        ggml_tensor * out_ids = (il == n_layer_main - 1) ? inp_out_ids : nullptr;
        inpL = build_layer(inpL, il, inp_pos, inp_attn, out_ids);
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
