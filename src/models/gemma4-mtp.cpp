#include "models.h"

#include "llama-kv-cache-iswa.h"
#include "llama-memory.h"

#include <cmath>

void llama_model_gemma4_mtp::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);

    if (hparams.n_embd_out_impl == 0) {
        throw std::runtime_error("Gemma 4 MTP requires embedding_length_out = target backbone hidden size");
    }

    hparams.n_embd_inp_impl       = hparams.n_embd_out_impl;
    hparams.n_layer_kv_from_start = 0;    // Gemma 4 assistant layers share the target KV cache.
    hparams.f_attention_scale     = 1.0f; // Gemma 4 attention uses scaling = 1.0.

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = false;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_gemma4_mtp::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_backbone = hparams.n_embd_inp();

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 MTP requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 MTP requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    output_norm    = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,  "weight"), { n_embd }, 0);
    mtp_input_embd = create_tensor(tn(LLM_TENSOR_MTP_INP_EMBD, "weight"), { n_embd_backbone, n_vocab }, 0);
    mtp_pre_proj   = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ, "weight"), { 2 * n_embd_backbone, n_embd }, 0);
    mtp_post_proj  = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ, "weight"), { n_embd, n_embd_backbone }, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), { n_embd_head }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head * n_head }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head * n_head, n_embd }, 0);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head / 2 }, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), { n_embd }, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
        layer.out_scale     = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

static int gemma4_mtp_find_target_layer(const llama_hparams & hparams, bool is_swa) {
    for (int il = (int) hparams.n_layer - 1; il >= 0; --il) {
        if (hparams.is_swa(il) == is_swa) {
            return il;
        }
    }

    return -1;
}

static ggml_tensor * gemma4_mtp_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor  * cur,
        ggml_tensor  * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat(ctx, rot, res);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

llama_model_gemma4_mtp::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const uint32_t n_embd_backbone = hparams.n_embd_inp();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_backbone);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_backbone_h_input");

    if (params.mtp_target_model == nullptr || params.mtp_target_memory == nullptr) {
        ggml_tensor * cur = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
        cb(cur, "mtp_reserve_tok_embd", -1);

        res->t_mtp_out = ggml_scale(ctx0, inp->embd, 0.0f);
        cb(res->t_mtp_out, "mtp_reserve_out", -1);

        res->add_input(std::move(inp));

        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "mtp_reserve_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        res->t_logits = cur;
        ggml_build_forward_expand(gf, res->t_mtp_out);
        ggml_build_forward_expand(gf, cur);
        return;
    }

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.mtp_input_embd, inp->tokens);
    tok_embd = ggml_scale(ctx0, tok_embd, sqrtf((float) n_embd_backbone));
    cb(tok_embd, "mtp_input_tok_embd", -1);

    ggml_tensor * h_input = inp->embd;
    res->add_input(std::move(inp));

    ggml_tensor * cur = ggml_concat(ctx0, tok_embd, h_input, /*dim=*/ 0);
    cb(cur, "mtp_concat", -1);

    cur = build_lora_mm(model.mtp_pre_proj, cur);
    cb(cur, "mtp_pre_proj", -1);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa(params.mtp_target_memory->init_full(), params.mtp_target_model->hparams);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const bool    is_swa      = hparams.is_swa(il);
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);
        const int     n_rot_l     = hparams.n_rot(il);

        const int il_target = gemma4_mtp_find_target_layer(params.mtp_target_model->hparams, is_swa);
        GGML_ASSERT(il_target >= 0 && "Gemma 4 MTP could not find matching target KV layer");

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * residual = cur;
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "mtp_Qcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : layer.rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "mtp_Qcur_pos", il);

        auto * mctx_cur = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
        ggml_tensor * k_rot = is_swa ? inp_attn->self_k_rot_swa : inp_attn->self_k_rot;
        ggml_tensor * v_rot = is_swa ? inp_attn->self_v_rot_swa : inp_attn->self_v_rot;

        if (k_rot) {
            Qcur = gemma4_mtp_mul_mat_aux(ctx0, Qcur, k_rot);
            cb(Qcur, "mtp_Qcur_attn_rot", il);
        }

        ggml_build_forward_expand(gf, Qcur);

        ggml_tensor * kq_mask = is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask();
        ggml_tensor * Kcur    = mctx_cur->get_k(ctx0, il_target);
        ggml_tensor * Vcur    = mctx_cur->get_v(ctx0, il_target);

        cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, kq_mask, nullptr, nullptr, 1.0f, il);
        cb(cur, "mtp_attn_pregate", il);

        if (v_rot) {
            cur = gemma4_mtp_mul_mat_aux(ctx0, cur, v_rot);
            cb(cur, "mtp_attn_v_rot", il);
        }

        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "mtp_attn_out", il);

        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_attn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "mtp_attn_residual", il);

        residual = cur;
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "mtp_ffn_out", il);

        cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);

        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
            cb(cur, "mtp_out_scaled", il);
        }
    }

    ggml_tensor * draft_hidden = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(draft_hidden, "mtp_result_norm", -1);

    ggml_tensor * backbone_hidden = build_lora_mm(model.mtp_post_proj, draft_hidden);
    cb(backbone_hidden, "mtp_post_proj", -1);
    res->t_mtp_out = backbone_hidden;
    ggml_build_forward_expand(gf, res->t_mtp_out);

    cur = build_lora_mm(model.output, draft_hidden);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
