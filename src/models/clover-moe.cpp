#include "ggml.h"
#include "models.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <numeric>

void llama_model_clover_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_CLOVER_CORESET_ENABLED, hparams.clover_coreset);
    ml.get_key(LLM_KV_CLOVER_CORESET_TOP_P,   hparams.clover_coreset_top_p);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_35B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_clover_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, 0);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);
        } else {
            // Clover stores raw A_log in ssm_a.
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,     "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,    "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,       "bias",   il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,           il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,     "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,    "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,     "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,      "weight", il), { value_dim, n_embd }, 0);

            layer.ssm_conv1d_right   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,         "weight_right", il), { hparams.ssm_d_conv - 1, conv_dim }, 0);
            layer.ssm_bd_logit_scale = create_tensor(tn(LLM_TENSOR_SSM_BD_LOGIT_SCALE, "weight",       il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_bd_history_b   = create_tensor(tn(LLM_TENSOR_SSM_BD_HISTORY,     "bias",         il), { hparams.ssm_dt_rank }, 0);
            layer.ssm_bd_history     = create_tensor(tn(LLM_TENSOR_SSM_BD_HISTORY,     "weight",       il), { hparams.ssm_dt_rank }, 0);
        }

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, 0);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, 0);
    }
}

static void clover_sort_rows(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * src = dst->src[0];
    for (int64_t row = ith; row < ggml_nrows(src); row += nth) {
        const int64_t i1 = row % src->ne[1];
        const int64_t i2 = row / src->ne[1] % src->ne[2];
        const int64_t i3 = row / (src->ne[1] * src->ne[2]);
        const auto * values = reinterpret_cast<const float *>(
                static_cast<const char *>(src->data) + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]);
        auto * ids = static_cast<int32_t *>(dst->data) + row * src->ne[0];
        std::iota(ids, ids + src->ne[0], 0);
        std::sort(ids, ids + src->ne[0], [&](int32_t a, int32_t b) {
            return values[a] == values[b] ? a < b : values[a] > values[b];
        });
    }
}

ggml_tensor * llama_model_clover_moe::graph::build_clover_top_k(ggml_tensor * logits, int64_t k) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32 && logits->nb[0] == sizeof(float));
    GGML_ASSERT(k > 0 && k <= logits->ne[0]);
    // Clover requires lower expert IDs first on ties; backend argsort does not guarantee this.
    auto * order = ggml_custom_4d(ctx0, GGML_TYPE_I32, logits->ne[0], logits->ne[1], logits->ne[2], logits->ne[3],
            &logits, 1, clover_sort_rows, GGML_N_TASKS_MAX, nullptr);
    return ggml_view_4d(ctx0, order, k, order->ne[1], order->ne[2], order->ne[3], order->nb[1], order->nb[2], order->nb[3], 0);
}

ggml_tensor * llama_model_clover_moe::graph::build_clover_l2_norm(ggml_tensor * cur, bool round) {
    cur = ggml_cont(ctx0, cur);
    auto * squared = ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur));
    auto * norm    = ggml_sqrt(ctx0, ggml_scale_bias(ctx0, squared, 1.0f, 1e-6f));
    cur = ggml_div(ctx0, cur, norm);
    return round ? build_bf16_round(cur) : cur;
}

ggml_tensor * llama_model_clover_moe::graph::build_clover(
        llm_graph_input_rs * inp,
               ggml_tensor * qkv,
               ggml_tensor * gate,
                       int   il) {
    const int64_t nseq     = ubatch.n_seqs;
    const auto &  layer    = model.layers[il];
    const int64_t width    = ubatch.n_seq_tokens;
    const int64_t nk       = hparams.ssm_n_group;
    const int64_t nv       = hparams.ssm_dt_rank;
    const int64_t dk       = hparams.ssm_d_state;
    const int64_t dv       = hparams.ssm_d_inner / nv;
    const int64_t channels = 2 * nk * dk + nv * dv;
    const int64_t taps     = hparams.ssm_d_conv;

    auto * r = build_rs(inp, inp->mctx->get_r_l(il), hparams.n_embd_r(), nseq);
    r = ggml_reshape_3d(ctx0, r, taps - 1, channels, nseq);
    auto * x       = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, qkv, channels, width, nseq), 1, 0, 2, 3));
    auto * history = ggml_concat(ctx0, r, x, 0);
    auto * left    = ggml_ssm_conv(ctx0, history, layer.ssm_conv1d);
    auto * padded  = ggml_pad(ctx0, x, taps - 1, 0, 0, 0);
    auto * future  = ggml_view_3d(ctx0, padded, width + taps - 2, channels, nseq, padded->nb[1], padded->nb[2], padded->nb[0]);
    future = ggml_cont(ctx0, future);
    auto * right = ggml_ssm_conv(ctx0, future, ggml_cast(ctx0, layer.ssm_conv1d_right, GGML_TYPE_F32));
    auto * mixed = build_bf16_round(ggml_silu(ctx0, ggml_add(ctx0, left, right)));

    auto * q = ggml_view_4d(ctx0, mixed, dk, nk, width, nseq, dk * mixed->nb[0], mixed->nb[1], mixed->nb[2], 0);
    auto * k = ggml_view_4d(ctx0, mixed, dk, nk, width, nseq, dk * mixed->nb[0], mixed->nb[1], mixed->nb[2], dk * nk * mixed->nb[0]);
    auto * v = ggml_view_4d(ctx0, mixed, dv, nv, width, nseq, dv * mixed->nb[0], mixed->nb[1], mixed->nb[2], 2 * dk * nk * mixed->nb[0]);
    q = build_clover_l2_norm(q, true);
    k = build_clover_l2_norm(k, true);
    if (nk != nv) {
        q = ggml_repeat_4d(ctx0, q, dk, nv, width, nseq);
        k = ggml_repeat_4d(ctx0, k, dk, nv, width, nseq);
    }
    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
    v = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));
    auto * s = build_rs(inp, inp->mctx->get_s_l(il), hparams.n_embd_s(), nseq);
    s = ggml_reshape_4d(ctx0, s, dk, dv, nv, nseq);
    s = build_bf16_round(s);
    auto * remembered = ggml_mul_mat(ctx0, ggml_cast(ctx0, s, GGML_TYPE_BF16), q);
    auto * scores     = ggml_mul_mat(ctx0, ggml_cast(ctx0, k, GGML_TYPE_BF16), q);
    scores = ggml_mul(ctx0, scores, ggml_reshape_3d(ctx0, layer.ssm_bd_logit_scale, 1, 1, nv));
    auto * history_bias = ggml_add(ctx0, layer.ssm_bd_history_b,
            ggml_scale(ctx0, layer.ssm_bd_history, std::log(float(width) / 32.0f)));
    history_bias = ggml_reshape_3d(ctx0, history_bias, 1, 1, nv);
    history_bias = ggml_repeat_4d(ctx0, history_bias, 1, width, nv, nseq);
    auto * probs         = ggml_soft_max(ctx0, ggml_concat(ctx0, scores, history_bias, 0));
    auto * local_weights = ggml_view_4d(ctx0, probs, width, width, nv, nseq, probs->nb[1], probs->nb[2], probs->nb[3], 0);
    local_weights = build_bf16_round(ggml_cont(ctx0, local_weights));
    auto * local          = ggml_mul_mat(ctx0, ggml_cast(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), GGML_TYPE_BF16), local_weights);
    auto * history_weight = ggml_view_4d(ctx0, probs, 1, width, nv, nseq, probs->nb[1], probs->nb[2], probs->nb[3], width * probs->nb[0]);
    auto * decay          = ggml_cont(ctx0, ggml_permute(ctx0, gate, 0, 2, 1, 3));
    decay = ggml_scale(ctx0, ggml_exp(ctx0, decay), 1.0f / std::sqrt(float(dk)));
    remembered = ggml_mul(ctx0, remembered, ggml_mul(ctx0, decay, history_weight));
    auto * output = build_bf16_round(ggml_add(ctx0, remembered, local));
    output = ggml_cont(ctx0, ggml_permute(ctx0, output, 0, 2, 1, 3));
    cb(output, "clover", il);
    return ggml_reshape_4d(ctx0, output, dv, nv, width, nseq);
}

ggml_tensor * llama_model_clover_moe::graph::build_clover_coreset(ggml_tensor * logits) {
    const int64_t width = logits->ne[1];

    auto * probs    = ggml_soft_max(ctx0, logits);
    auto * ids      = build_clover_top_k(logits, n_expert_used);
    auto * shaped   = ggml_reshape_3d(ctx0, probs, 1, n_expert, width);
    auto * mass     = ggml_get_rows(ctx0, shaped, ids);
    auto * zero     = ggml_fill(ctx0, shaped, 0.0f);
    auto * selected = ggml_set_rows(ctx0, zero, mass, ids);
    selected = ggml_reshape_2d(ctx0, selected, n_expert, width);
    auto * totals = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, selected)));
    totals = ggml_cont(ctx0, ggml_transpose(ctx0, totals));
    auto * normalized = ggml_div(ctx0, totals, ggml_sum_rows(ctx0, totals));
    auto * order      = build_clover_top_k(normalized, n_expert);
    auto * ordered    = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, normalized, 1, n_expert), order);
    ordered = ggml_reshape_1d(ctx0, ordered, n_expert);
    auto * previous = ggml_sub(ctx0, ggml_cumsum(ctx0, ordered), ordered);
    auto * keep     = ggml_step(ctx0, ggml_scale_bias(ctx0, previous, -1.0f, hparams.clover_coreset_top_p));
    auto * allowed  = ggml_set_rows(ctx0, ggml_fill(ctx0, ggml_reshape_2d(ctx0, keep, 1, n_expert), 0.0f),
            ggml_reshape_2d(ctx0, keep, 1, n_expert), order);
    allowed = ggml_reshape_2d(ctx0, allowed, n_expert, 1);
    auto * mask = ggml_scale_bias(ctx0, allowed, 1e30f, -1e30f);
    return ggml_add(ctx0, logits, mask);
}

std::unique_ptr<llm_graph_context> llama_model_clover_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_clover_moe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = build_bf16_round(inpL);

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = build_bf16_round(cur);

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_clover_moe::graph::build_qkvz(
        ggml_tensor * input,
                int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_clover_moe::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate) {
    input = build_bf16_round(input);
    auto * normalized = ggml_mul(ctx0, ggml_rms_norm(ctx0, input, hparams.f_norm_rms_eps), weights);
    return build_bf16_round(ggml_mul(ctx0, normalized, ggml_silu(ctx0, gate)));
}

ggml_tensor * llama_model_clover_moe::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    cb(Kcur, "Kcur", il);

    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply RoPE
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Qcur = build_bf16_round(Qcur);

    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = build_bf16_round(Kcur);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = build_bf16_round(ggml_mul(ctx0, build_bf16_round(cur), gate_sigmoid));
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_clover_moe::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
               ggml_tensor * cur,
                       int   il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);

    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;
    ggml_tensor * beta      = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    ggml_tensor * alpha     = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    if (n_tokens != n_outputs) {
        beta = build_bf16_round(beta);
    }
    cb(beta, "beta_sigmoid", il);

    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    auto *        decay_a = ggml_neg(ctx0, ggml_exp(ctx0, model.layers[il].ssm_a));
    ggml_tensor * gate    = ggml_mul(ctx0, alpha_softplus, decay_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * output;
    if (!cparams.causal_attn) {
        output = build_clover(inp, qkv_mixed, gate, il);
    } else {
        ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

        ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);

        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        ggml_tensor * conv_output_silu = build_bf16_round(ggml_silu(ctx0, conv_output_proper));
        cb(conv_output_silu, "conv_output_silu", il);

        ggml_tensor * conv_qkv_mix = conv_output_silu;

        // Calculate the total conv dimension
        int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

        // Extract the convolved Q, K, V from conv_output
        ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim),
                nb1_qkv,
                nb1_qkv * n_seq_tokens,
                0);

        ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim),
                nb1_qkv,
                nb1_qkv * n_seq_tokens,
                head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

        ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_v_dim),
                nb1_qkv,
                nb1_qkv * n_seq_tokens,
                ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

        cb(q_conv, "q_conv", il);
        cb(k_conv, "k_conv", il);
        cb(v_conv, "v_conv", il);

        const bool prefill = n_tokens != n_outputs;
        q_conv = build_clover_l2_norm(q_conv, prefill);
        k_conv = build_clover_l2_norm(k_conv, prefill);


        // if head keys and value keys are different, repeat to force tensors into matching shapes
        // note: need explicit repeat only if we are not using the fused GDN.
        if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch ||
                (n_tokens != n_outputs))) {
            GGML_ASSERT(num_v_heads % num_k_heads == 0);
            q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
            k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        }

        cb(q_conv, "q_conv_predelta", il);
        cb(k_conv, "k_conv_predelta", il);
        cb(v_conv, "v_conv_predelta", il);

        output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_clover_moe::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * router_logits    = nullptr;
    ggml_tensor * selected_experts = nullptr;
    router_logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);
    cb(router_logits, "ffn_moe_logits", il);
    if (!cparams.causal_attn && hparams.clover_coreset) {
        cb(router_logits, "clover_router_raw", il);
        router_logits = build_clover_coreset(router_logits);
        cb(router_logits, "clover_router_masked", il);
    }
    selected_experts = build_clover_top_k(router_logits, n_expert_used);
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            router_logits, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s, selected_experts);
    moe_out = build_bf16_round(moe_out);
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp = build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate;
        shared_gate = ggml_sum_rows(ctx0, ggml_mul(ctx0, cur, model.layers[il].ffn_gate_inp_shexp));
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        cur = build_bf16_round(ggml_add(ctx0, moe_out, ffn_shexp));
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

ggml_tensor * llama_model_clover_moe::graph::build_bf16_round(ggml_tensor * cur) const {
    return ggml_cast(ctx0, ggml_cast(ctx0, cur, GGML_TYPE_BF16), GGML_TYPE_F32);
}
