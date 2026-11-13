#include "models.h"

#include "../llama-kv-cache-iswa.h"
#include "../llama-kv-cache.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-recurrent.h"

#include <algorithm>
#include <cfloat>

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = 1;
    }

    ml.get_key(LLM_KV_INKLING_D_REL,            hparams.inkling_d_rel);
    ml.get_key(LLM_KV_INKLING_REL_EXTENT,       hparams.inkling_rel_extent);
    ml.get_key(LLM_KV_INKLING_REL_EXTENT_SWA,   hparams.inkling_rel_extent_swa);
    ml.get_key(LLM_KV_INKLING_SHORTCONV_KERNEL, hparams.n_shortconv_l_cache);

    ml.get_key(LLM_KV_INKLING_LOG_SCALING_N_FLOOR, hparams.inkling_log_n_floor,      false);
    ml.get_key(LLM_KV_INKLING_LOG_SCALING_ALPHA,   hparams.inkling_log_alpha,        false);
    ml.get_key(LLM_KV_INKLING_UNPADDED_VOCAB_SIZE, hparams.inkling_unpadded_n_vocab, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,                hparams.attn_res_block_size);
    ml.get_key(LLM_KV_INKLING2_QK_CLIP_THRESHOLD,         hparams.inkling2_qk_clip);
    ml.get_key(LLM_KV_INKLING2_REL_CLIP_THRESHOLD,        hparams.inkling2_rel_clip);
    ml.get_key(LLM_KV_INKLING2_ATTN_RES_NORMALIZE_VALUES, hparams.inkling2_attn_res_norm_vals, false);
    ml.get_key(LLM_KV_INKLING2_MLP_GATE_SOFTCAP,          hparams.inkling2_mlp_gate_softcap,   false);
    ml.get_key(LLM_KV_INKLING2_MLP_UP_SOFTCAP,            hparams.inkling2_mlp_up_softcap,     false);
    ml.get_key(LLM_KV_INKLING2_VALUE_NORM,                hparams.inkling2_value_norm,         false);

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    GGML_ASSERT(hparams.inkling_d_rel > 0);
    GGML_ASSERT(hparams.inkling_rel_extent > 0 && hparams.inkling_rel_extent_swa > 0);
    GGML_ASSERT(hparams.inkling2_qk_clip > 0.0f && hparams.inkling2_rel_clip > 0.0f);
    GGML_ASSERT(hparams.attn_res_block_size > 0);

    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;
    hparams.n_embd_r_impl = d_conv * (hparams.n_embd_k_gqa_max() + hparams.n_embd_v_gqa_max() + 2*hparams.n_embd);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t head_dim = hparams.n_embd_head_k();
    const int64_t d_rel    = hparams.inkling_d_rel;
    const int64_t K        = hparams.n_shortconv_l_cache;
    const int64_t n_ff_exp = hparams.n_ff_exp();

    tok_embd         = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), {n_embd, n_vocab}, 0);
    tok_norm         = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM,  "weight", 0), {n_embd}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab}, 0);
    output_res_score = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_SCORE, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool    is_swa     = hparams.is_swa(i);
        const int64_t kvw        = hparams.n_head_kv(i) * head_dim;
        const int64_t rel_extent = is_swa ? hparams.inkling_rel_extent_swa : hparams.inkling_rel_extent;

        if (i > 0) {
            layer.attn_res_score = create_tensor(tn(LLM_TENSOR_ATTN_RES_SCORE, "weight", i), {n_embd, is_swa ? 1 : 3}, 0);
        }
        layer.ffn_res_score = create_tensor(tn(LLM_TENSOR_FFN_RES_SCORE, "weight", i), {n_embd}, 0);

        if (is_swa) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            layer.attn_q_in_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_IN_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_in_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_IN_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_in_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_IN_NORM,  "weight", i), {n_embd}, 0);
            layer.shortconv_k_dyn = create_tensor(tn(LLM_TENSOR_SHORTCONV_K_DYN, "weight", i), {n_embd, hparams.n_head_kv(i)*K}, 0);
            layer.shortconv_v_dyn = create_tensor(tn(LLM_TENSOR_SHORTCONV_V_DYN, "weight", i), {n_embd, hparams.n_head_kv(i)*K}, 0);
        }

        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, n_head*head_dim}, 0);
        layer.wk             = create_tensor(tn(LLM_TENSOR_ATTN_K,         "weight", i), {n_embd, kvw}, 0);
        layer.wv             = create_tensor(tn(LLM_TENSOR_ATTN_V,         "weight", i), {n_embd, kvw}, 0);
        layer.wr             = create_tensor(tn(LLM_TENSOR_ATTN_R,         "weight", i), {n_embd, n_head*d_rel}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head*head_dim, n_embd}, 0);
        layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, n_head*head_dim}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);
        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {head_dim, n_head}, 0);
        layer.attn_r_clip    = create_tensor(tn(LLM_TENSOR_ATTN_R_CLIP,    "weight", i), {d_rel, n_head}, 0);
        layer.attn_rel_proj  = create_tensor(tn(LLM_TENSOR_ATTN_REL_PROJ,  "weight", i), {rel_extent, d_rel}, 0);

        layer.shortconv_k    = create_tensor(tn(LLM_TENSOR_SHORTCONV_K,    "weight", i), {K, kvw}, 0);
        layer.shortconv_v    = create_tensor(tn(LLM_TENSOR_SHORTCONV_V,    "weight", i), {K, kvw}, 0);
        layer.shortconv_attn = create_tensor(tn(LLM_TENSOR_SHORTCONV_ATTN, "weight", i), {K, n_embd}, 0);
        layer.shortconv_mlp  = create_tensor(tn(LLM_TENSOR_SHORTCONV_MLP,  "weight", i), {K, n_embd}, 0);

        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps_s   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
    }
}

class llm_graph_input_inkling2 : public llm_graph_input_i {
public:
    llm_graph_input_inkling2(const llama_hparams & hparams, const llama_memory_hybrid_iswa_context * mctx) :
        hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_inkling2() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (tau) {
            GGML_ASSERT(ggml_backend_buffer_is_host(tau->buffer));
            float * data = (float *) tau->data;
            for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
                const float eff = (float) (ubatch->pos[i] + 1) / (float) hparams.inkling_log_n_floor;
                data[i] = 1.0f + hparams.inkling_log_alpha*logf(std::max(eff, 1.0f));
            }
        }
        if (rel_idx) {
            mctx->get_attn()->get_base()->set_input_pos_rel_flat(rel_idx, ubatch, hparams.inkling_rel_extent);
        }
        if (rel_idx_swa) {
            mctx->get_attn()->get_swa()->set_input_pos_rel_flat(rel_idx_swa, ubatch, hparams.inkling_rel_extent_swa);
        }
        if (vocab_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(vocab_mask->buffer));
            float * data = (float *) vocab_mask->data;
            for (int64_t id = 0; id < vocab_mask->ne[0]; ++id) {
                data[id] = id < (int64_t) hparams.inkling_unpadded_n_vocab ? 0.0f : -INFINITY;
            }
        }
    }

    ggml_tensor * tau         = nullptr;
    ggml_tensor * rel_idx     = nullptr;
    ggml_tensor * rel_idx_swa = nullptr;
    ggml_tensor * vocab_mask  = nullptr;

    const llama_hparams hparams;
    const llama_memory_hybrid_iswa_context * mctx;
};

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t head_dim = hparams.n_embd_head_k();
    const int64_t d_rel    = hparams.inkling_d_rel;
    const int64_t K        = hparams.n_shortconv_l_cache;
    const int64_t d_conv   = K - 1;
    const int64_t n_embd_r = hparams.n_embd_r();
    const int64_t kw_max   = hparams.n_embd_k_gqa_max();
    const int64_t vw_max   = hparams.n_embd_v_gqa_max();
    const float   eps      = hparams.f_norm_rms_eps;
    const uint32_t res_bs  = hparams.attn_res_block_size;

    const int64_t off_k    = 0;
    const int64_t off_v    = d_conv*kw_max;
    const int64_t off_attn = d_conv*(kw_max + vw_max);
    const int64_t off_mlp  = d_conv*(kw_max + vw_max + n_embd);

    const auto * mctx_hyb  = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
    const auto * mctx_recr = mctx_hyb->get_recr();
    const auto * mctx_attn = mctx_hyb->get_attn();

    const uint32_t kv_head = mctx_recr->get_head();

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const uint32_t n_kv_flash_base = cparams.flash_attn ? mctx_attn->get_base()->get_n_kv_pos_contiguous() : 0;
    const uint32_t n_kv_flash_swa  = cparams.flash_attn ? mctx_attn->get_swa ()->get_n_kv_pos_contiguous() : 0;

    const auto banded_type_ok = [](ggml_type type) {
        return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
    };

    // the banded op needs cell index == position; otherwise the dense path gathers the rel logits per cell
    const auto use_banded = [&](int il) {
        const auto * cache = hparams.is_swa(il) ? mctx_attn->get_swa() : mctx_attn->get_base();
        return (hparams.is_swa(il) ? n_kv_flash_swa : n_kv_flash_base) > 0 &&
            (head_dim == 64 || head_dim == 128) &&
            hparams.n_embd_head_v(il) == head_dim &&
            banded_type_ok(cache->type_k()) && banded_type_ok(cache->type_v());
    };

    bool has_global = false;
    bool dense_swa  = false;
    bool dense_base = false;
    for (int il = 0; il < n_layer; ++il) {
        if (hparams.is_swa(il)) {
            dense_swa |= !use_banded(il);
        } else {
            has_global = true;
            dense_base |= !use_banded(il);
        }
    }

    auto inp = std::make_unique<llm_graph_input_inkling2>(hparams, mctx_hyb);

    if (hparams.inkling_log_n_floor > 0 && has_global) {
        inp->tau = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(inp->tau);
    }
    if (dense_base) {
        inp->rel_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_attn->get_base()->get_n_kv(), n_tokens);
        ggml_set_input(inp->rel_idx);
    }
    if (dense_swa) {
        inp->rel_idx_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_attn->get_swa()->get_n_kv(), n_tokens);
        ggml_set_input(inp->rel_idx_swa);
    }
    const int64_t n_vocab = model.vocab.n_tokens();
    if (!cparams.embeddings && hparams.inkling_unpadded_n_vocab > 0 && (int64_t) hparams.inkling_unpadded_n_vocab < n_vocab) {
        inp->vocab_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_vocab);
        ggml_set_input(inp->vocab_mask);
    }

    ggml_tensor * tau         = inp->tau;
    ggml_tensor * rel_idx     = inp->rel_idx;
    ggml_tensor * rel_idx_swa = inp->rel_idx_swa;
    ggml_tensor * vocab_mask  = inp->vocab_mask;

    res->add_input(std::move(inp));

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();

    ggml_tensor * conv_rs_cur = nullptr;

    auto build_sconv = [&](ggml_tensor * x2d, ggml_tensor * kernel, int64_t off, ggml_tensor * dyn, int il) -> ggml_tensor * {
        const int64_t w  = x2d->ne[0];
        const size_t  sz = ggml_element_size(conv_rs_cur);

        ggml_tensor * xt    = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, x2d, w, n_seq_tokens, n_seqs));
        ggml_tensor * state = ggml_view_3d(ctx0, conv_rs_cur, d_conv, w, n_seqs, d_conv*sz, conv_rs_cur->nb[1], off*sz);
        ggml_tensor * sx    = ggml_concat(ctx0, state, xt, 0);

        ggml_tensor * conv_state = mctx_recr->get_r_l(il);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                ggml_view_3d(ctx0, sx, d_conv, w, n_seqs, sx->nb[1], sx->nb[2], (sx->ne[0] - d_conv)*sx->nb[0]),
                ggml_view_3d(ctx0, conv_state, d_conv, w, n_seqs, d_conv*sz, n_embd_r*sz, (kv_head*n_embd_r + off)*sz)));

        ggml_tensor * y = ggml_ssm_conv(ctx0, sx, kernel);

        if (dyn) {
            const int64_t n_kvh = dyn->ne[1];
            ggml_tensor * sxc = ggml_cont(ctx0, sx);
            for (int64_t lag = 0; lag < K; ++lag) {
                ggml_tensor * xl = ggml_view_3d(ctx0, sxc, n_seq_tokens, w, n_seqs, sxc->nb[1], sxc->nb[2], (d_conv - lag)*sxc->nb[0]);
                xl = ggml_cont(ctx0, ggml_transpose(ctx0, xl));
                xl = ggml_reshape_3d(ctx0, xl, w / n_kvh, n_kvh, n_tokens);
                ggml_tensor * dl = ggml_cont(ctx0, ggml_view_3d(ctx0, dyn, 1, n_kvh, n_tokens, dyn->nb[1], dyn->nb[2], lag*dyn->nb[0]));
                y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, ggml_mul(ctx0, xl, dl), w, n_seq_tokens, n_seqs));
            }
        }

        return ggml_reshape_2d(ctx0, y, w, n_tokens);
    };

    auto build_l2_clip = [&](ggml_tensor * x, ggml_tensor * scale, float thr) -> ggml_tensor * {
        if (scale) {
            x = ggml_mul(ctx0, x, scale);
        }
        ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));
        ggml_tensor * den = ggml_clamp(ctx0, ggml_scale_bias(ctx0, nrm, 1.0f/thr, 1e-6f/thr), 1.0f, FLT_MAX);
        return ggml_div(ctx0, x, den);
    };

    auto build_attn_res = [&](ggml_tensor * srcs, ggml_tensor * qw, int il) -> ggml_tensor * {
        const int64_t n_src = srcs->ne[1];
        const int64_t n_q   = ggml_nelements(qw) / n_embd;

        ggml_tensor * keys = ggml_rms_norm(ctx0, srcs, 1e-6f);

        ggml_tensor * sc = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, qw, n_embd, n_q), ggml_reshape_2d(ctx0, keys, n_embd, n_src*n_tokens));
        ggml_prec_set_acc(sc, GGML_PREC_F32_PEDANTIC);
        sc = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, sc, n_q, n_src, n_tokens), 1, 0, 2, 3));
        ggml_tensor * p = ggml_soft_max_ext(ctx0, sc, nullptr, 1.0f/sqrtf((float) n_embd), 0.0f);
        cb(p, "attn_res_probs", il);

        ggml_tensor * vals = hparams.inkling2_attn_res_norm_vals ? keys : srcs;
        vals = ggml_cont(ctx0, ggml_permute(ctx0, vals, 1, 0, 2, 3));

        ggml_tensor * out = ggml_mul_mat(ctx0, vals, p);
        ggml_prec_set_acc(out, GGML_PREC_F32_PEDANTIC);
        return out;
    };

    auto as_src = [&](ggml_tensor * x) { return ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens); };

    auto build_attn_block = [&](ggml_tensor * xq, ggml_tensor * xk, ggml_tensor * xv, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        const bool    is_swa     = hparams.is_swa(il);
        const int64_t n_head_kv  = hparams.n_head_kv(il);
        const int64_t rel_extent = is_swa ? hparams.inkling_rel_extent_swa : hparams.inkling_rel_extent;

        ggml_tensor * q    = build_lora_mm(layer.wq,             xq);
        ggml_tensor * r    = build_lora_mm(layer.wr,             xq);
        ggml_tensor * gate = build_lora_mm(layer.wqkv_gate,      xq);
        ggml_tensor * sink = build_lora_mm(layer.attn_sink_proj, xq);
        ggml_tensor * k    = build_lora_mm(layer.wk,             xk);
        ggml_tensor * v    = build_lora_mm(layer.wv,             xv);

        ggml_tensor * dyn_k = nullptr;
        ggml_tensor * dyn_v = nullptr;
        if (!is_swa) {
            dyn_k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.shortconv_k_dyn, xk), K, n_head_kv, n_tokens);
            dyn_v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layer.shortconv_v_dyn, xv), K, n_head_kv, n_tokens);
        }
        k = build_sconv(k, layer.shortconv_k, off_k, dyn_k, il);
        v = build_sconv(v, layer.shortconv_v, off_v, dyn_v, il);

        q = ggml_reshape_3d(ctx0, q, head_dim, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, head_dim, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, head_dim, n_head_kv, n_tokens);
        r = ggml_reshape_3d(ctx0, r, d_rel,    n_head,    n_tokens);

        q = build_l2_clip(q, layer.attn_q_clip, hparams.inkling2_qk_clip);
        k = build_l2_clip(k, nullptr,           hparams.inkling2_qk_clip);
        r = build_l2_clip(r, layer.attn_r_clip, hparams.inkling2_rel_clip);

        if (hparams.inkling2_value_norm) {
            v = ggml_rms_norm(ctx0, v, eps);
        }
        if (tau && !is_swa) {
            q = ggml_mul(ctx0, q, tau);
            r = ggml_mul(ctx0, r, tau);
        }
        cb(q, "Qcur", il);
        cb(k, "Kcur", il);
        cb(v, "Vcur", il);

        ggml_tensor * proj = ggml_cont(ctx0, ggml_transpose(ctx0, layer.attn_rel_proj));
        ggml_tensor * rel  = ggml_mul_mat(ctx0, proj, ggml_reshape_2d(ctx0, r, d_rel, n_head*n_tokens));
        ggml_prec_set_acc(rel, GGML_PREC_F32_PEDANTIC);
        rel = ggml_reshape_3d(ctx0, rel, rel_extent, n_head, n_tokens);

        const float kq_scale = 1.0f/sqrtf((float) head_dim);

        auto * inp_attn = inp_hybrid->get_attn();
        const auto * cache = is_swa ? inp_attn->mctx->get_swa() : inp_attn->mctx->get_base();
        ggml_tensor * mask = is_swa ? inp_attn->self_attn_mask_swa.raw : inp_attn->self_attn_mask.raw;

        const int64_t n_stream = mask->ne[3];
        GGML_ASSERT(n_tokens % n_stream == 0);
        const int64_t n_tps = n_tokens/n_stream;

        auto * k_rot = is_swa ? inp_attn->self_k_rot_swa[il] : inp_attn->self_k_rot[il];
        auto * v_rot = is_swa ? inp_attn->self_v_rot_swa[il] : inp_attn->self_v_rot[il];
        if (k_rot) {
            q = llama_mul_mat_hadamard(ctx0, q, k_rot);
            k = llama_mul_mat_hadamard(ctx0, k, k_rot);
        }
        if (v_rot) {
            v = llama_mul_mat_hadamard(ctx0, v, v_rot);
        }

        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, k);
        ggml_build_forward_expand(gf, v);
        ggml_build_forward_expand(gf, cache->cpy_k(ctx0, k, is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, cache->cpy_v(ctx0, v, is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs(), il));

        ggml_tensor * k_all = cache->get_k(ctx0, il);
        ggml_tensor * v_all = cache->get_v(ctx0, il);
        const bool v_trans = v_all->nb[1] > v_all->nb[2];

        ggml_tensor * cur;
        if (use_banded(il)) {
            const int64_t n_kv = is_swa ? n_kv_flash_swa : n_kv_flash_base;
            GGML_ASSERT(n_stream == 1 && n_kv <= k_all->ne[2]);

            ggml_tensor * q_fa = ggml_permute(ctx0, ggml_reshape_4d(ctx0, q, head_dim, n_head, n_tokens, 1), 0, 2, 1, 3);
            ggml_tensor * k_fa = ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], n_kv, k_all->ne[3],
                    k_all->nb[1], k_all->nb[2], k_all->nb[3], 0);
            ggml_tensor * v_fa;
            if (v_trans) {
                v_fa = ggml_view_4d(ctx0, v_all, n_kv, v_all->ne[1], v_all->ne[2], v_all->ne[3],
                        v_all->nb[1], v_all->nb[2], v_all->nb[3], 0);
                v_fa = ggml_transpose(ctx0, ggml_permute(ctx0, v_fa, 0, 2, 1, 3));
            } else {
                v_fa = ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], n_kv, v_all->ne[3],
                        v_all->nb[1], v_all->nb[2], v_all->nb[3], 0);
                v_fa = ggml_permute(ctx0, v_fa, 0, 2, 1, 3);
            }
            k_fa = ggml_permute(ctx0, k_fa, 0, 2, 1, 3);

            ggml_tensor * m = ggml_cont(ctx0, ggml_view_4d(ctx0, mask, n_kv, mask->ne[1], mask->ne[2], mask->ne[3],
                    mask->nb[1], mask->nb[2], mask->nb[3], 0));

            cur = ggml_flash_attn_ext_banded(ctx0, q_fa, k_fa, v_fa, m, ggml_reshape_4d(ctx0, rel, rel_extent, n_head, n_tokens, 1),
                    kq_scale, rel_extent);
            ggml_flash_attn_ext_add_sinks(cur, ggml_reshape_3d(ctx0, sink, n_head, n_tokens, 1));
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        } else {
            ggml_tensor * idx = is_swa ? rel_idx_swa : rel_idx;
            const int64_t n_kv = idx->ne[0];
            GGML_ASSERT(n_kv == k_all->ne[2]);

            rel = ggml_pad(ctx0, rel, 1, 0, 0, 0);
            rel = ggml_cont(ctx0, ggml_permute(ctx0, rel, 1, 0, 2, 3));
            rel = ggml_reshape_2d(ctx0, rel, n_head, (rel_extent + 1)*n_tokens);
            ggml_tensor * kq_b = ggml_get_rows(ctx0, rel, ggml_reshape_1d(ctx0, idx, n_kv*n_tokens));
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, kq_b, n_head, n_kv, n_tokens), 2, 0, 1, 3));
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_kv, n_tps, n_stream, n_head);
            kq_b = ggml_permute(ctx0, kq_b, 0, 1, 3, 2);

            ggml_tensor * qs = ggml_scale(ctx0, q, kq_scale);
            qs = ggml_permute(ctx0, ggml_reshape_4d(ctx0, qs, head_dim, n_head, n_tps, n_stream), 0, 2, 1, 3);

            ggml_tensor * kq = ggml_mul_mat(ctx0, ggml_permute(ctx0, k_all, 0, 2, 1, 3), qs);
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_add(ctx0, kq, kq_b);

            ggml_tensor * m = ggml_view_4d(ctx0, mask, n_kv, n_tps, 1, n_stream, mask->nb[1], mask->nb[2], mask->nb[3], 0);
            kq = ggml_add(ctx0, kq, ggml_cast(ctx0, m, GGML_TYPE_F32));

            // the sink is one more logit with a zero value: softmax over [kq, sink], then drop the sink column
            ggml_tensor * s = ggml_reshape_4d(ctx0, sink, 1, n_head, n_tps, n_stream);
            s  = ggml_cont(ctx0, ggml_permute(ctx0, s, 0, 2, 1, 3));
            kq = ggml_soft_max(ctx0, ggml_concat(ctx0, kq, s, 0));
            kq = ggml_cont(ctx0, ggml_view_4d(ctx0, kq, n_kv, kq->ne[1], kq->ne[2], kq->ne[3], kq->nb[1], kq->nb[2], kq->nb[3], 0));

            ggml_tensor * vv = ggml_permute(ctx0, v_all, 0, 2, 1, 3);
            if (!v_trans) {
                vv = ggml_cont(ctx0, ggml_transpose(ctx0, vv));
            }
            cur = ggml_permute(ctx0, ggml_mul_mat(ctx0, vv, kq), 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        }
        cb(cur, "kqv_out", il);

        if (v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
        }

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        return build_lora_mm(layer.wo, cur);
    };

    auto softcap_swiglu = [&](ggml_tensor * g, ggml_tensor * u) -> ggml_tensor * {
        const float gc = hparams.inkling2_mlp_gate_softcap;
        const float uc = hparams.inkling2_mlp_up_softcap;

        ggml_tensor * a = gc > 0.0f ? ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, g, 1.0f/gc)), gc) : g;
        a = ggml_mul(ctx0, a, ggml_sigmoid(ctx0, g));
        if (uc > 0.0f) {
            u = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, u, 1.0f/uc)), uc);
        }
        return ggml_mul(ctx0, a, u);
    };

    auto expert_scale = [&](ggml_tensor * s, ggml_tensor * selected) -> ggml_tensor * {
        ggml_tensor * ids = ggml_reshape_1d(ctx0, ggml_cont(ctx0, selected), ggml_nelements(selected));
        return ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, s, 1, s->ne[0]), ids), 1, selected->ne[0], selected->ne[1]);
    };

    auto build_moe = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur, nullptr, GGML_PREC_F32_PEDANTIC);
        ggml_tensor * scores = ggml_add(ctx0, ggml_sigmoid(ctx0, logits), layer.ffn_exp_probs_b);
        ggml_tensor * selected = ggml_argsort_top_k(ctx0, scores, n_expert_used);
        cb(selected, "ffn_moe_topk", il);

        ggml_tensor * w = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, logits, 1, n_expert, n_tokens), selected);
        w = ggml_reshape_2d(ctx0, w, n_expert_used, n_tokens);
        w = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, w)));
        w = ggml_scale(ctx0, ggml_soft_max(ctx0, w), hparams.expert_weights_scale);
        cb(w, "ffn_moe_weights", il);

        ggml_tensor * xr   = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        ggml_tensor * gate = build_lora_mm_id(layer.ffn_gate_exps, xr, selected);
        ggml_tensor * up   = build_lora_mm_id(layer.ffn_up_exps,   xr, selected);

        // NVFP4 experts are unscaled (up to ~2.7e3): F32 accumulation avoids F16 overflow
        if (layer.ffn_gate_exps_s) {
            ggml_prec_set_acc(gate, GGML_PREC_F32);
            gate = ggml_mul(ctx0, gate, expert_scale(layer.ffn_gate_exps_s, selected));
        }
        if (layer.ffn_up_exps_s) {
            ggml_prec_set_acc(up, GGML_PREC_F32);
            up = ggml_mul(ctx0, up, expert_scale(layer.ffn_up_exps_s, selected));
        }
        ggml_tensor * h = ggml_mul(ctx0, softcap_swiglu(gate, up), ggml_reshape_3d(ctx0, w, 1, n_expert_used, n_tokens));

        ggml_tensor * experts = build_lora_mm_id(layer.ffn_down_exps, h, selected);
        if (layer.ffn_down_exps_s) {
            ggml_prec_set_acc(experts, GGML_PREC_F32);
            experts = ggml_mul(ctx0, experts, expert_scale(layer.ffn_down_exps_s, selected));
        }

        ggml_tensor * moe_out = nullptr;
        for (int64_t i = 0; i < n_expert_used; ++i) {
            ggml_tensor * e = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx0, moe_out, e) : e;
        }
        return moe_out;
    };

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, NULL, LLM_NORM_RMS, -1);
    }
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
    }
    ggml_build_forward_expand(gf, cur);

    ggml_tensor * blocks  = nullptr;
    ggml_tensor * partial = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const bool is_swa   = hparams.is_swa(il);
        const bool boundary = (uint32_t) il % res_bs == 0;

        conv_rs_cur = build_rs(inp_hybrid->get_recr(), mctx_recr->get_r_l(il), n_embd_r, n_seqs);

        ggml_tensor * xq = partial;
        ggml_tensor * xk = partial;
        ggml_tensor * xv = partial;
        if (blocks) {
            ggml_tensor * mix = build_attn_res(ggml_concat(ctx0, blocks, as_src(partial), 1), layer.attn_res_score, il);
            if (is_swa) {
                xq = xk = xv = ggml_reshape_2d(ctx0, mix, n_embd, n_tokens);
            } else {
                xq = ggml_view_2d(ctx0, mix, n_embd, n_tokens, mix->nb[2], 0*mix->nb[1]);
                xk = ggml_view_2d(ctx0, mix, n_embd, n_tokens, mix->nb[2], 1*mix->nb[1]);
                xv = ggml_view_2d(ctx0, mix, n_embd, n_tokens, mix->nb[2], 2*mix->nb[1]);
            }
        }

        if (is_swa) {
            xq = xk = xv = build_norm(xq, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        } else {
            xq = build_norm(xq, layer.attn_q_in_norm, NULL, LLM_NORM_RMS, il);
            xk = build_norm(xk, layer.attn_k_in_norm, NULL, LLM_NORM_RMS, il);
            xv = build_norm(xv, layer.attn_v_in_norm, NULL, LLM_NORM_RMS, il);
        }

        ggml_tensor * attn_out = build_attn_block(xq, xk, xv, il);
        attn_out = build_sconv(attn_out, layer.shortconv_attn, off_attn, nullptr, il);

        ggml_tensor * ffn_srcs;
        if (boundary) {
            ggml_tensor * closed = blocks ? ggml_concat(ctx0, blocks, as_src(partial), 1) : as_src(partial);
            ffn_srcs = ggml_concat(ctx0, closed, as_src(attn_out), 1);
        } else {
            ffn_srcs = ggml_concat(ctx0, blocks, as_src(ggml_add(ctx0, partial, attn_out)), 1);
        }
        ggml_tensor * ffn_in = ggml_reshape_2d(ctx0, build_attn_res(ffn_srcs, layer.ffn_res_score, il), n_embd, n_tokens);
        ffn_in = build_norm(ffn_in, layer.ffn_norm, NULL, LLM_NORM_RMS, il);

        ggml_tensor * ffn_out = build_moe(ffn_in, il);
        ffn_out = build_sconv(ffn_out, layer.shortconv_mlp, off_mlp, nullptr, il);

        if (boundary) {
            blocks  = blocks ? ggml_concat(ctx0, blocks, as_src(partial), 1) : as_src(partial);
            partial = ggml_add(ctx0, attn_out, ffn_out);
        } else {
            partial = ggml_add(ctx0, partial, ggml_add(ctx0, attn_out, ffn_out));
        }
        cb(partial, "l_out", il);
    }

    cur = build_attn_res(ggml_concat(ctx0, blocks, as_src(partial), 1), model.output_res_score, -1);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (!cparams.embeddings) {
        cur = build_lora_mm(model.output, cur, nullptr, GGML_PREC_F32_PEDANTIC);
        if (hparams.f_final_logit_softcapping > 0.0f) {
            const float c = hparams.f_final_logit_softcapping;
            cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/c)), c);
        }
        if (vocab_mask) {
            cur = ggml_add(ctx0, cur, vocab_mask);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}
