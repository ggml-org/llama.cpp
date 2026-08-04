#include "models.h"

#include <cmath>
#include <stdexcept>

// Motif-3
// Uses GDLA (Grouped Differential Latent Attention), sigmoid scoring, PolyNorm gated activations and mHC on every layer
static constexpr float MOTIF_MHC_NORM_EPS = 1e-6f;

void llama_model_motif3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_NOISE_HEAD_COUNT,  hparams.motif_n_noise_heads);
    ml.get_key(LLM_KV_ATTENTION_YARN_MSCALE,       hparams.motif_mscale, false);

    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    hparams.n_moe_layer_step = 1;
    ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step, false);

    if (hparams.n_expert > 0) {
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
        hparams.n_ff_shexp = hparams.n_ff_exp;
        ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,       hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,      hparams.expert_weights_scale, false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,       hparams.expert_weights_norm, false);
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC,        hparams.expert_gating_func, false);
    }

    // PolyNorm
    ml.get_key(LLM_KV_POLYNORM_EPS,            hparams.motif_poly_eps,          false);
    ml.get_key(LLM_KV_POLYNORM_OUTPUT_SCALE,   hparams.motif_poly_out_scale,    false);
    ml.get_key(LLM_KV_POLYNORM_BIAS_CLAMP,     hparams.motif_poly_bias_clamp,   false);
    ml.get_key(LLM_KV_POLYNORM_HIDDEN_CLAMP,   hparams.motif_poly_hidden_clamp, false);
    ml.get_key(LLM_KV_POLYNORM_SIGMOID_WEIGHT, hparams.motif_poly_sigmoid_w,    false);

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,                hparams.motif_mhc_mult,       false);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  hparams.motif_mhc_iters,      false);
    ml.get_key(LLM_KV_MHC_H_POST_COEFF,                      hparams.motif_mhc_post_coeff, false);

    // interleaved SWA full attention at il % period == 0
    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        uint32_t swa_period = 4;
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period, /*dense_first =*/ true);
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        hparams.rope_freq_scale_train_swa = 1.0f;
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    if (hparams.motif_n_noise_heads == 0) {
        throw std::runtime_error("Motif-3: attention.noise_head_count must be > 0");
    }
    if (hparams.n_head() % hparams.motif_n_noise_heads != 0 ||
        (hparams.n_head() - hparams.motif_n_noise_heads) % hparams.motif_n_noise_heads != 0) {
        throw std::runtime_error("Motif-3: head_count must be divisible into noise groups");
    }
    if (hparams.n_rot() == 0 || hparams.n_rot() % 2 != 0 || hparams.n_rot() >= hparams.n_embd_head_k()) {
        throw std::runtime_error("Motif-3: rope dims must be even and smaller than the K head size");
    }
    if (hparams.n_head_kv() == 0 || hparams.n_head() % hparams.n_head_kv() != 0) {
        throw std::runtime_error("Motif-3: head_count must be a multiple of head_count_kv");
    }

    // GDLA MLA latent cache for the full attention layers, SWA layers keep the full GQA representation
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE &&
            ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_K_B, "weight", 0).str().c_str()) != nullptr) {
        hparams.motif_mla_kv        = true;
        hparams.motif_n_embd_head_k = hparams.n_embd_head_k_full;
        hparams.motif_n_embd_head_v = hparams.n_embd_head_v_full;
        hparams.motif_n_head_kv     = hparams.n_head_kv();

        hparams.n_embd_head_k_full  = hparams.n_lora_kv + hparams.n_rot_full;
        hparams.n_embd_head_v_full  = hparams.n_lora_kv;

        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (!hparams.is_swa(il)) {
                hparams.n_head_kv_arr[il] = 1;
            }
        }

        // flag the model as MLA so the iswa base cache is allocated K only, while the SWA cache always keeps a real V regardless of this flag
        hparams.n_embd_head_k_mla_impl = hparams.motif_n_embd_head_k;
        hparams.n_embd_head_v_mla_impl = hparams.motif_n_embd_head_v;

        LLAMA_LOG_INFO("%s: Motif-3 GDLA: using MLA latent KV cache on full-attention layers "
                "(%u + %u per token instead of %u)\n", __func__,
                hparams.n_lora_kv, hparams.n_rot_full,
                hparams.motif_n_head_kv*(hparams.motif_n_embd_head_k + hparams.motif_n_embd_head_v));
    }


    switch (hparams.n_layer()) {
        case 53: type = LLM_TYPE_314B_A13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_motif3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // With motif_mla_kv, n_embd_head_k/v and n_head_kv describe the latent cache of the full attention layers
    const int64_t n_embd_head_k_att = hparams.motif_mla_kv ? hparams.motif_n_embd_head_k : n_embd_head_k;
    const int64_t n_embd_head_v_att = hparams.motif_mla_kv ? hparams.motif_n_embd_head_v : n_embd_head_v;
    const int64_t n_head_kv_att     = hparams.motif_mla_kv ? hparams.motif_n_head_kv     : n_head_kv;

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_att - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t n_noise  = hparams.motif_n_noise_heads;
    const int64_t n_signal = n_head - n_noise;

    const int64_t n_ff_exp   = hparams.n_ff_exp;
    const int64_t n_ff_shexp = hparams.n_ff_shexp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t hc      = hparams.motif_mhc_mult;
    const int64_t hc_dim  = hc * n_embd;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        // GDLA
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head_k_att}, 0);
        layer.wq_b_gate     = create_tensor(tn(LLM_TENSOR_ATTN_GATE,     "weight", i), {q_lora_rank, n_signal * n_embd_head_v_att}, TENSOR_NOT_REQUIRED);

        layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
        layer.wkv_b          = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,      "weight", i), {kv_lora_rank, n_head_kv_att * (n_embd_head_qk_nope + n_embd_head_v_att)}, 0);

        if (hparams.motif_mla_kv) {
            // per kv head slices of wkv_b, kb transposed
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head_kv_att}, 0);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_att, n_head_kv_att}, 0);
        }

        layer.attn_lambda = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA, "weight", i), {n_embd, n_signal}, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_signal * n_embd_head_v_att, n_embd}, 0);

        // mHC
        if (hc > 0) {
            layer.mhc_attn_norm   = create_tensor(tn(LLM_TENSOR_MHC_ATTN_NORM,  "weight", i), {hc_dim}, 0);
            layer.mhc_attn_pre    = create_tensor(tn(LLM_TENSOR_MHC_ATTN_PRE,   "weight", i), {hc_dim, hc}, 0);
            layer.mhc_attn_pre_b  = create_tensor(tn(LLM_TENSOR_MHC_ATTN_PRE,   "bias",   i), {hc}, 0);
            layer.mhc_attn_post   = create_tensor(tn(LLM_TENSOR_MHC_ATTN_POST,  "weight", i), {hc_dim, hc}, 0);
            layer.mhc_attn_post_b = create_tensor(tn(LLM_TENSOR_MHC_ATTN_POST,  "bias",   i), {hc}, 0);
            layer.mhc_attn_res    = create_tensor(tn(LLM_TENSOR_MHC_ATTN_RES,   "weight", i), {hc_dim, hc * hc}, 0);
            layer.mhc_attn_res_b  = create_tensor(tn(LLM_TENSOR_MHC_ATTN_RES,   "bias",   i), {hc, hc}, 0);
            layer.mhc_attn_alpha  = create_tensor(tn(LLM_TENSOR_MHC_ATTN_ALPHA, "weight", i), {3}, 0);

            layer.mhc_ffn_norm    = create_tensor(tn(LLM_TENSOR_MHC_FFN_NORM,  "weight", i), {hc_dim}, 0);
            layer.mhc_ffn_pre     = create_tensor(tn(LLM_TENSOR_MHC_FFN_PRE,   "weight", i), {hc_dim, hc}, 0);
            layer.mhc_ffn_pre_b   = create_tensor(tn(LLM_TENSOR_MHC_FFN_PRE,   "bias",   i), {hc}, 0);
            layer.mhc_ffn_post    = create_tensor(tn(LLM_TENSOR_MHC_FFN_POST,  "weight", i), {hc_dim, hc}, 0);
            layer.mhc_ffn_post_b  = create_tensor(tn(LLM_TENSOR_MHC_FFN_POST,  "bias",   i), {hc}, 0);
            layer.mhc_ffn_res     = create_tensor(tn(LLM_TENSOR_MHC_FFN_RES,   "weight", i), {hc_dim, hc * hc}, 0);
            layer.mhc_ffn_res_b   = create_tensor(tn(LLM_TENSOR_MHC_FFN_RES,   "bias",   i), {hc, hc}, 0);
            layer.mhc_ffn_alpha   = create_tensor(tn(LLM_TENSOR_MHC_FFN_ALPHA, "weight", i), {3}, 0);
        }

        // FFN: dense or MoE
        const bool is_moe = hparams.n_expert > 0 &&
                            (uint32_t) i >= hparams.n_layer_dense_lead &&
                            hparams.n_moe_layer_step > 0 &&
                            ((uint32_t) i + 1) % hparams.n_moe_layer_step == 0;

        if (!is_moe) {
            layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_poly_w = create_tensor(tn(LLM_TENSOR_FFN_POLY, "weight", i), {3}, 0);
            layer.ffn_poly_b = create_tensor(tn(LLM_TENSOR_FFN_POLY, "bias",   i), {1}, 0);
        } else {
            if (n_expert == 0 || n_expert_used == 0) {
                throw std::runtime_error("Motif-3: n_expert and n_expert_used must be > 0 for MoE layers");
            }
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);

            layer.ffn_poly_exps_w = create_tensor(tn(LLM_TENSOR_FFN_POLY_EXPS, "weight", i), {3, n_expert}, 0);
            layer.ffn_poly_exps_b = create_tensor(tn(LLM_TENSOR_FFN_POLY_EXPS, "bias",   i), {1, n_expert}, 0);

            if (n_expert_shared > 0) {
                layer.ffn_gate_shexp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp * n_expert_shared}, 0);
                layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp * n_expert_shared}, 0);
                layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp * n_expert_shared, n_embd}, 0);
                layer.ffn_poly_shexp_w = create_tensor(tn(LLM_TENSOR_FFN_POLY_SHEXP, "weight", i), {3}, 0);
                layer.ffn_poly_shexp_b = create_tensor(tn(LLM_TENSOR_FFN_POLY_SHEXP, "bias",   i), {1}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_motif3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// mHC


void llama_model_motif3::graph::build_mhc_gates(
        ggml_tensor  * x,
        ggml_tensor  * norm_w,
        ggml_tensor  * pre_w,  ggml_tensor * pre_b,
        ggml_tensor  * post_w, ggml_tensor * post_b,
        ggml_tensor  * res_w,  ggml_tensor * res_b,
        ggml_tensor  * alpha,
        ggml_tensor ** h_pre,
        ggml_tensor ** h_post,
        ggml_tensor ** h_res,
        int il) const {
    const int64_t hc = hparams.motif_mhc_mult;
    const int64_t nt = x->ne[2];

    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == hc);

    // rms-norm over the flattened (hc * n_embd) lanes, with weight
    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, MOTIF_MHC_NORM_EPS);
    flat_norm = ggml_mul(ctx0, flat_norm, norm_w);
    cb(flat_norm, "mhc_norm", il);

    ggml_tensor * alpha_pre  = ggml_view_1d(ctx0, alpha, 1, 0*ggml_row_size(alpha->type, 1));
    ggml_tensor * alpha_post = ggml_view_1d(ctx0, alpha, 1, 1*ggml_row_size(alpha->type, 1));
    ggml_tensor * alpha_res  = ggml_view_1d(ctx0, alpha, 1, 2*ggml_row_size(alpha->type, 1));

    const bool fuse_comb = cparams.fused_dsv4_hc_comb && hc == 4 && il >= 0;

    // raw gate projections (shared by the fused and unfused h_res paths)
    ggml_tensor * pre_lin  = ggml_mul_mat(ctx0, pre_w,  flat_norm); // [hc, nt]
    ggml_tensor * post_lin = ggml_mul_mat(ctx0, post_w, flat_norm); // [hc, nt]
    ggml_tensor * res_lin  = ggml_mul_mat(ctx0, res_w,  flat_norm); // [hc*hc, nt]

    // h_pre
    ggml_tensor * pre = ggml_mul(ctx0, pre_lin, alpha_pre);
    pre = ggml_add(ctx0, pre, pre_b);
    pre = ggml_clamp(ctx0, pre, -10.0f, 10.0f);
    pre = ggml_sigmoid(ctx0, pre);
    cb(pre, "mhc_h_pre", il);
    *h_pre = pre;

    // h_post
    ggml_tensor * post = ggml_mul(ctx0, post_lin, alpha_post);
    post = ggml_add(ctx0, post, post_b);
    post = ggml_clamp(ctx0, post, -10.0f, 10.0f);
    post = ggml_sigmoid(ctx0, post);
    if (hparams.motif_mhc_post_coeff != 1.0f) {
        post = ggml_scale(ctx0, post, hparams.motif_mhc_post_coeff);
    }
    cb(post, "mhc_h_post", il);
    *h_post = post;

    // h_res
    ggml_tensor * m = nullptr;
    if (fuse_comb) {
        ggml_tensor * mixes = ggml_concat(ctx0, pre_lin, post_lin, 0);
        mixes = ggml_concat(ctx0, mixes, res_lin, 0); // [(2 + hc)*hc, nt]

        ggml_tensor * base = ggml_concat(ctx0, pre_b, post_b, 0);
        base = ggml_concat(ctx0, base, ggml_reshape_1d(ctx0, res_b, hc*hc), 0); // [(2 + hc)*hc]

        m = ggml_dsv4_hc_comb(ctx0, mixes, alpha, base, /*eps =*/ 0.0f, (int32_t) hparams.motif_mhc_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, m, il});
    } else {
        ggml_tensor * res_aff = ggml_mul(ctx0, res_lin, alpha_res);
        // reshape to [hc(src=j), hc(dst=i), nt]
        res_aff = ggml_reshape_3d(ctx0, res_aff, hc, hc, nt);
        res_aff = ggml_add(ctx0, res_aff, res_b); // res_b: [hc, hc]
        res_aff = ggml_clamp(ctx0, res_aff, -20.0f, 20.0f);
        m = ggml_exp(ctx0, res_aff);

        m = build_mhc_sinkhorn(m, il);
    }
    cb(m, "mhc_h_res", il);
    *h_res = m;
}

ggml_tensor * llama_model_motif3::graph::build_mhc_sinkhorn(ggml_tensor * m, int il) const {
    GGML_UNUSED(il);
    
    for (uint32_t it = 0; it < hparams.motif_mhc_iters; ++it) {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, m); // [1, hc, nt]
        row_sum = ggml_clamp(ctx0, row_sum, 1e-8f, INFINITY);
        m = ggml_div(ctx0, m, row_sum);

        ggml_tensor * mt = ggml_cont(ctx0, ggml_permute(ctx0, m, 1, 0, 2, 3)); // [hc(dst), hc(src), nt]
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, mt); // [1, hc, nt]
        col_sum = ggml_clamp(ctx0, col_sum, 1e-8f, INFINITY);
        mt = ggml_div(ctx0, mt, col_sum);
        m = ggml_cont(ctx0, ggml_permute(ctx0, mt, 1, 0, 2, 3));
    }
    return m;
}

ggml_tensor * llama_model_motif3::graph::build_mhc_apply_pre(ggml_tensor * x, ggml_tensor * h_pre, int il) const {
    const int64_t hc = hparams.motif_mhc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // dst[i, t] = sum_h x[i, h, t]*h_pre[h, t], ascending h
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, h_pre);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, h_pre, 1, nt, h_pre->nb[1], ih*h_pre->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }
    return result;
}

ggml_tensor * llama_model_motif3::graph::build_mhc_combine(
        ggml_tensor * x,
        ggml_tensor * y,
        ggml_tensor * h_post,
        ggml_tensor * h_res,
        int il) const {
    const int64_t hc = hparams.motif_mhc_mult;
    const int64_t nt = y->ne[1];

    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // ggml_dsv4_hc_post expects comb as [dst, src, nt]; h_res is [src, dst, nt].
        ggml_tensor * comb = ggml_cont(ctx0, ggml_permute(ctx0, h_res, 1, 0, 2, 3));
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, y, x, h_post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    // h_res layout: [hc(src=j), hc(dst=i), nt]
    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, h_post, 1, nt, h_post->nb[1], dst*h_post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, y, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], src*x->nb[1]);
            ggml_tensor * comb    = ggml_view_2d(ctx0, h_res, 1, nt, h_res->nb[2], src*h_res->nb[0] + dst*h_res->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }
    return out;
}

// PolyNorm

ggml_tensor * llama_model_motif3::graph::build_polynorm_act(
        ggml_tensor * gate,
        ggml_tensor * up,
        ggml_tensor * poly_w,
        ggml_tensor * poly_b,
        ggml_tensor * selected,
        bool          clamp_bias,
        bool          clamp_result,
        int il) const {
    const float eps     = hparams.motif_poly_eps;
    const float h_clamp = hparams.motif_poly_hidden_clamp;
    const float b_clamp = hparams.motif_poly_bias_clamp;

    ggml_tensor * w = poly_w; // [3] or [3, n_expert]
    ggml_tensor * b = poly_b; // [1] or [1, n_expert]

    if (selected) {
        // gather per-expert coefficients for each
        const int64_t n_used = selected->ne[0];
        const int64_t nt     = selected->ne[1];
        ggml_tensor * ids_flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, selected), n_used * nt);
        w = ggml_get_rows(ctx0, poly_w, ids_flat);   // [3, n_used * nt]
        w = ggml_reshape_3d(ctx0, w, 3, n_used, nt); // [3, n_used, nt]
        b = ggml_get_rows(ctx0, poly_b, ids_flat);   // [1, n_used * nt]
        b = ggml_reshape_3d(ctx0, b, 1, n_used, nt); // [1, n_used, nt]
    }

    if (hparams.motif_poly_sigmoid_w) {
        w = ggml_sigmoid(ctx0, w);
    }
    if (clamp_bias && b_clamp > 0.0f) {
        b = ggml_clamp(ctx0, b, -b_clamp, b_clamp);
    }

    ggml_tensor * w0 = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * w2 = nullptr;
    if (selected) {
        // [1, n_expert_used, nt] views of w
        w0 = ggml_view_3d(ctx0, w, 1, w->ne[1], w->ne[2], w->nb[1], w->nb[2], 0*w->nb[0]);
        w1 = ggml_view_3d(ctx0, w, 1, w->ne[1], w->ne[2], w->nb[1], w->nb[2], 1*w->nb[0]);
        w2 = ggml_view_3d(ctx0, w, 1, w->ne[1], w->ne[2], w->nb[1], w->nb[2], 2*w->nb[0]);
    } else {
        w0 = ggml_view_1d(ctx0, w, 1, 0*w->nb[0]);
        w1 = ggml_view_1d(ctx0, w, 1, 1*w->nb[0]);
        w2 = ggml_view_1d(ctx0, w, 1, 2*w->nb[0]);
    }

    if (h_clamp > 0.0f) {
        gate = ggml_clamp(ctx0, gate, -h_clamp, h_clamp);
        up   = ggml_clamp(ctx0, up,   -h_clamp, h_clamp);
    }

    ggml_tensor * x2 = ggml_mul(ctx0, gate, gate);
    ggml_tensor * x3 = ggml_mul(ctx0, x2, gate);

    ggml_tensor * t3 = ggml_mul(ctx0, ggml_rms_norm(ctx0, x3,   eps), w0);
    ggml_tensor * t2 = ggml_mul(ctx0, ggml_rms_norm(ctx0, x2,   eps), w1);
    ggml_tensor * t1 = ggml_mul(ctx0, ggml_rms_norm(ctx0, gate, eps), w2);

    ggml_tensor * poly = ggml_add(ctx0, ggml_add(ctx0, t3, t2), t1);
    poly = ggml_add(ctx0, poly, b);
    cb(poly, "ffn_poly", il);

    ggml_tensor * act = ggml_mul(ctx0, poly, up);
    if (clamp_result && h_clamp > 0.0f) {
        act = ggml_clamp(ctx0, act, -h_clamp, h_clamp);
    }
    if (hparams.motif_poly_out_scale != 1.0f) {
        act = ggml_scale(ctx0, act, hparams.motif_poly_out_scale);
    }
    cb(act, "ffn_poly_act", il);
    return act;
}

ggml_tensor * llama_model_motif3::graph::build_polynorm_mlp(
        ggml_tensor * cur,
        ggml_tensor * gate_w,
        ggml_tensor * up_w,
        ggml_tensor * down_w,
        ggml_tensor * poly_w,
        ggml_tensor * poly_b,
        int il) const {
    ggml_tensor * gate = build_lora_mm(gate_w, cur);
    cb(gate, "ffn_gate", il);
    ggml_tensor * up = build_lora_mm(up_w, cur);
    cb(up, "ffn_up", il);

    // note: MotifMLP only clamps gate/up before the activation, not the product
    ggml_tensor * act = build_polynorm_act(gate, up, poly_w, poly_b, nullptr,
            /*clamp_bias =*/ false, /*clamp_result =*/ false, il);

    ggml_tensor * out = build_lora_mm(down_w, act);
    cb(out, "ffn_down", il);
    return out;
}

// MoE with grouped (per-expert) PolyNorm activation.
ggml_tensor * llama_model_motif3::graph::build_moe_polynorm(const llama_model & model, ggml_tensor * cur, int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_ = cur->ne[0];
    const int64_t nt      = cur->ne[1];

    ggml_tensor * inp = cur; // keep the 2-D input for the shared expert

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur); // [n_expert, nt]
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32); // reference router runs in fp32
    cb(logits, "ffn_moe_logits", il);

    GGML_ASSERT(hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID);
    ggml_tensor * probs = ggml_sigmoid(ctx0, logits); // [n_expert, nt]
    cb(probs, "ffn_moe_probs", il);

    // expert selection bias
    ggml_tensor * selection_probs = probs;
    if (layer.ffn_exp_probs_b) {
        selection_probs = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [k, nt]
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * probs3  = ggml_reshape_3d(ctx0, probs, 1, n_expert, nt);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs3, selected_experts); // [1, k, nt]
    cb(weights, "ffn_moe_weights", il);

    if (hparams.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, nt);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, nt]
        weights_sum = ggml_clamp(ctx0, weights_sum, 1e-20f, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, nt);
        cb(weights, "ffn_moe_weights_norm", il);
    }
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd_, 1, nt);

    ggml_tensor * gate = build_lora_mm_id(layer.ffn_gate_exps, cur, selected_experts); // [n_ff_exp, k, nt]
    cb(gate, "ffn_moe_gate", il);
    ggml_tensor * up = build_lora_mm_id(layer.ffn_up_exps, cur, selected_experts); // [n_ff_exp, k, nt]
    cb(up, "ffn_moe_up", il);

    ggml_tensor * act = build_polynorm_act(gate, up,
            layer.ffn_poly_exps_w, layer.ffn_poly_exps_b, selected_experts,
            /*clamp_bias =*/ true, /*clamp_result =*/ true, il);

    ggml_tensor * experts = build_lora_mm_id(layer.ffn_down_exps, act, selected_experts); // [n_embd, k, nt]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx0, experts, weights);
    cb(experts, "ffn_moe_weighted", il);

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    GGML_ASSERT(n_expert_used > 0);
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd_, nt, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    ggml_tensor * moe_out = cur_experts[0];
    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }
    if (hparams.n_expert_used == 1) {
        moe_out = ggml_cont(ctx0, moe_out);
    }
    cb(moe_out, "ffn_moe_out", il);

    // shared expert (regular PolyNorm MLP)
    if (layer.ffn_gate_shexp) {
        ggml_tensor * shexp = build_polynorm_mlp(inp,
                layer.ffn_gate_shexp, layer.ffn_up_shexp, layer.ffn_down_shexp,
                layer.ffn_poly_shexp_w, layer.ffn_poly_shexp_b, il);
        cb(shexp, "ffn_shexp", il);
        moe_out = ggml_add(ctx0, moe_out, shexp);
        cb(moe_out, "ffn_out", il);
    }

    return moe_out;
}

// GDLA attention

ggml_tensor * llama_model_motif3::graph::build_gdla_attn(
        const llama_model & model,
        llm_graph_input_attn_kv      * inp_kv,
        llm_graph_input_attn_kv_iswa * inp_iswa,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        float kq_scale_full,
        float kq_scale_swa,
        int il) const {
    const auto & layer = model.layers[il];

    // With motif_mla_kv, n_embd_head_k/v and n_head_kv describe the latent cache of the full-attention layers instead
    const int64_t n_embd_head_k_att = hparams.motif_mla_kv ? (int64_t) hparams.motif_n_embd_head_k : n_embd_head_k;
    const int64_t n_embd_head_v_att = hparams.motif_mla_kv ? (int64_t) hparams.motif_n_embd_head_v : n_embd_head_v;
    const int64_t n_head_kv_att     = hparams.motif_mla_kv ? (int64_t) hparams.motif_n_head_kv     : n_head_kv;

    const int64_t n_embd_head_qk_rope = n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_att - n_rot;

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_noise  = hparams.motif_n_noise_heads;
    const int64_t n_group  = n_noise;
    const int64_t ratio    = (n_head - n_noise) / n_noise;
    const int64_t gs       = ratio + 1;
    const int64_t n_signal = n_head - n_noise;

    const bool is_swa = hparams.is_swa(il);

    // per-layer rope parameters
    // SWA layers use plain RoPE with the SWA freq base and no YaRN
    // full-attention layers use the (YaRN) context values
    const float freq_base_l   = is_swa ? hparams.rope_freq_base_train_swa : freq_base;
    const float freq_scale_l  = is_swa ? 1.0f : freq_scale;
    const float ext_factor_l  = is_swa ? 0.0f : ext_factor;
    const float attn_factor_l = (is_swa || ext_factor == 0.0f)
        ? 1.0f
        : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    const float kq_scale = is_swa ? kq_scale_swa : kq_scale_full;

    ggml_tensor * q_lat = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(q_lat, "gdla_q_lat", il);
    q_lat = build_norm(q_lat, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q_lat, "gdla_q_lat_norm", il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, q_lat); // [n_head * n_embd_head_k, nt]
    cb(q, "gdla_q", il);

    // HF layout within each head: [nope | rope]
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_att),
            ggml_row_size(q->type, n_embd_head_k_att) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_att),
            ggml_row_size(q->type, n_embd_head_k_att) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_raw = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_raw, "gdla_kv_raw", il);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_raw, kv_lora_rank, n_tokens,
            ggml_row_size(kv_raw->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_raw, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_raw->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_raw->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_raw->type, kv_lora_rank));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
    cb(q_pe, "gdla_q_pe", il);

    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
    cb(k_pe, "gdla_k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "gdla_kv_cmpr", il);

    const bool use_mla = hparams.motif_mla_kv && !is_swa;

    ggml_tensor * Qcur  = nullptr;
    ggml_tensor * Kcur  = nullptr;
    ggml_tensor * Vcur  = nullptr;
    ggml_tensor * v_mla = nullptr;

    if (use_mla) {
        // {n_embd_head_qk_nope, n_tokens, n_head}
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        cb(q_nope, "gdla_q_nope_perm", il);

        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        cb(q_nope_absorbed, "gdla_q_nope_absorbed", il);

        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(Qcur, "gdla_Qcur", il);

        ggml_tensor * kv_cmpr_3d = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

        Kcur = ggml_concat(ctx0, kv_cmpr_3d, ggml_cont(ctx0, k_pe), 0);
        cb(Kcur, "gdla_Kcur", il);

        Vcur = kv_cmpr_3d;
        cb(Vcur, "gdla_Vcur", il);

        v_mla = layer.wv_b;
    } else {

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr); // [n_head_kv * (nope + v), nt]
    cb(kv, "gdla_kv", il);

    ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head_kv_att, n_tokens,
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_att),
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_att) * n_head_kv_att, 0);
    Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_att, n_head_kv_att, n_tokens,
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_att),
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_att) * n_head_kv_att,
            ggml_row_size(kv->type, n_embd_head_qk_nope));
    Vcur = ggml_cont(ctx0, Vcur);
    cb(Vcur, "gdla_v", il);

    // assemble Q/K in [rope | nope] order
    Qcur = ggml_concat(ctx0, ggml_cont(ctx0, q_pe), ggml_cont(ctx0, q_nope), 0);
    cb(Qcur, "gdla_Qcur", il);

    ggml_tensor * k_pe_rep = ggml_repeat_4d(ctx0, ggml_cont(ctx0, k_pe),
            n_embd_head_qk_rope, n_head_kv_att, n_tokens, 1);
    Kcur = ggml_concat(ctx0, k_pe_rep, ggml_cont(ctx0, k_nope), 0);
    cb(Kcur, "gdla_Kcur", il);

    } // !use_mla

    // differential-attention lambda
    ggml_tensor * lambda = ggml_mul_mat(ctx0, layer.attn_lambda, cur); // [n_signal, nt]
    lambda = ggml_sigmoid(ctx0, lambda);
    cb(lambda, "gdla_lambda", il);

    // optional elementwise output gate from the Q latent
    ggml_tensor * gate = nullptr;
    if (layer.wq_b_gate) {
        gate = ggml_mul_mat(ctx0, layer.wq_b_gate, q_lat); // [n_signal * n_embd_head_v, nt]
        gate = ggml_sigmoid(ctx0, gate);
        cb(gate, "gdla_gate", il);
    }

    ggml_tensor * attn = inp_iswa
        ? build_attn(inp_iswa,
              nullptr, nullptr, nullptr,
              Qcur, Kcur, Vcur, nullptr, nullptr, v_mla, kq_scale, il)
        : build_attn(inp_kv,
              nullptr, nullptr, nullptr,
              Qcur, Kcur, Vcur, nullptr, nullptr, v_mla, kq_scale, il);
    cb(attn, "gdla_attn_raw", il);

    // split heads into n_group groups of (ratio signal + 1 noise) heads
    ggml_tensor * a4 = ggml_reshape_4d(ctx0, attn, n_embd_head_v_att, gs, n_group, n_tokens);

    ggml_tensor * attn1 = ggml_view_4d(ctx0, a4, n_embd_head_v_att, ratio, n_group, n_tokens,
            a4->nb[1], a4->nb[2], a4->nb[3], 0);
    attn1 = ggml_cont(ctx0, attn1);

    ggml_tensor * attn2 = ggml_view_4d(ctx0, a4, n_embd_head_v_att, 1, n_group, n_tokens,
            a4->nb[1], a4->nb[2], a4->nb[3], ratio*a4->nb[1]);
    attn2 = ggml_cont(ctx0, attn2);
    ggml_tensor * attn2_rep = ggml_repeat_4d(ctx0, attn2, n_embd_head_v_att, ratio, n_group, n_tokens);

    ggml_tensor * lam = ggml_reshape_4d(ctx0, lambda, 1, ratio, n_group, n_tokens);
    ggml_tensor * diff = ggml_sub(ctx0, attn1, ggml_mul(ctx0, attn2_rep, lam));
    cb(diff, "gdla_diff", il);

    if (gate) {
        gate = ggml_reshape_4d(ctx0, gate, n_embd_head_v_att, ratio, n_group, n_tokens);
        diff = ggml_mul(ctx0, diff, gate);
        cb(diff, "gdla_gated", il);
    }

    ggml_tensor * out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, diff), n_signal * n_embd_head_v_att, n_tokens);
    out = build_lora_mm(layer.wo, out);
    cb(out, "gdla_out", il);

    return out;
}

// graph

llama_model_motif3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t hc = hparams.motif_mhc_mult;

    const float yarn_log   = (freq_scale < 1.0f) ? logf(1.0f / freq_scale) : 0.0f;
    const float mscale_val = 1.0f + 0.1f * hparams.motif_mscale * yarn_log;

    // with motif_mla_kv, n_embd_head_k describes the latent cache, and the softmax scale must use the true head size
    const int64_t n_embd_head_k_att = hparams.motif_mla_kv ? (int64_t) hparams.motif_n_embd_head_k : n_embd_head_k;

    const float kq_scale_full = mscale_val * mscale_val / sqrtf(float(n_embd_head_k_att));
    const float kq_scale_swa  = 1.0f / sqrtf(float(n_embd_head_k_att));

    ggml_tensor * cur;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd); // [n_embd, nt]

    ggml_tensor * inp_pos = build_inp_pos();

    // the memory type depends on whether any layer uses SWA
    llm_graph_input_attn_kv      * inp_kv   = nullptr;
    llm_graph_input_attn_kv_iswa * inp_iswa = nullptr;
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_kv = build_attn_inp_kv();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // mHC: expand the residual stream to [n_embd, hc, nt]
    ggml_tensor * x = nullptr;
    if (hc > 0) {
        ggml_tensor * in3 = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        x = ggml_repeat_4d(ctx0, in3, n_embd, hc, n_tokens, 1);
        cb(x, "mhc_inp", -1);
    } else {
        x = inpL;
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const bool is_moe = layer.ffn_gate_exps != nullptr;

        if (hc > 0) {
            ggml_tensor * h_pre  = nullptr;
            ggml_tensor * h_post = nullptr;
            ggml_tensor * h_res  = nullptr;
            build_mhc_gates(x,
                    layer.mhc_attn_norm,
                    layer.mhc_attn_pre,  layer.mhc_attn_pre_b,
                    layer.mhc_attn_post, layer.mhc_attn_post_b,
                    layer.mhc_attn_res,  layer.mhc_attn_res_b,
                    layer.mhc_attn_alpha,
                    &h_pre, &h_post, &h_res, il);

            ggml_tensor * x_red = build_mhc_apply_pre(x, h_pre, il); // [n_embd, nt]
            cb(x_red, "mhc_attn_red", il);

            cur = build_norm(x_red, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            cur = build_gdla_attn(model, inp_kv, inp_iswa, cur, inp_pos, kq_scale_full, kq_scale_swa, il);

            x = build_mhc_combine(x, cur, h_post, h_res, il);
            cb(x, "mhc_attn_out", il);

            build_mhc_gates(x,
                    layer.mhc_ffn_norm,
                    layer.mhc_ffn_pre,  layer.mhc_ffn_pre_b,
                    layer.mhc_ffn_post, layer.mhc_ffn_post_b,
                    layer.mhc_ffn_res,  layer.mhc_ffn_res_b,
                    layer.mhc_ffn_alpha,
                    &h_pre, &h_post, &h_res, il);

            ggml_tensor * h_red = build_mhc_apply_pre(x, h_pre, il);
            cb(h_red, "mhc_ffn_red", il);

            cur = build_norm(h_red, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if (is_moe) {
                cur = build_moe_polynorm(model, cur, il);
            } else {
                cur = build_polynorm_mlp(cur,
                        layer.ffn_gate, layer.ffn_up, layer.ffn_down,
                        layer.ffn_poly_w, layer.ffn_poly_b, il);
            }
            cb(cur, "ffn_out", il);

            x = build_mhc_combine(x, cur, h_post, h_res, il);
            cb(x, "mhc_ffn_out", il);
        } else {
            ggml_tensor * inpSA = x;

            cur = build_norm(x, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            cur = build_gdla_attn(model, inp_kv, inp_iswa, cur, inp_pos, kq_scale_full, kq_scale_swa, il);

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if (is_moe) {
                cur = build_moe_polynorm(model, cur, il);
            } else {
                cur = build_polynorm_mlp(cur,
                        layer.ffn_gate, layer.ffn_up, layer.ffn_down,
                        layer.ffn_poly_w, layer.ffn_poly_b, il);
            }
            cb(cur, "ffn_out", il);

            x = ggml_add(ctx0, cur, ffn_inp);
            cb(x, "l_out", il);
        }
    }

    // reduce the mHC lanes
    if (hc > 0) {
        ggml_tensor * acc = nullptr;
        for (int64_t ih = 0; ih < hc; ++ih) {
            ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[2], ih*x->nb[1]);
            acc = acc ? ggml_add(ctx0, acc, xh) : xh;
        }
        cur = ggml_scale(ctx0, acc, 1.0f / float(hc));
        cb(cur, "mhc_mean", -1);
    } else {
        cur = x;
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
