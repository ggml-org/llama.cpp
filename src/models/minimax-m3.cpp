#include "models.h"
#include "llama-kv-cache.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

// MiniMax-M3, text-only: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling) and
// swigluoai activation. Sparse attention falls back to dense; vision tower and MTP are dropped.

void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head,     false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size,  false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k,      false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks, false);
    msa_p = { (int) hparams.indexer_block_size, (int) hparams.indexer_top_k, (int) hparams.indexer_local_blocks };

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_minimax_m3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_ff_exp        = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        // per-head QK-norm: a single head_dim vector applied to every head
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            // leading dense layers
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            // routed experts
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            // shared expert
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            
            // indexer
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEX_Q_PROJ, "weight", i), {n_embd, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEX_K_PROJ, "weight", i), {n_embd, hparams.indexer_head_size}, 0);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEX_Q_NORM, "weight", i), {hparams.indexer_head_size}, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEX_K_NORM, "weight", i), {hparams.indexer_head_size}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_m3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// ---- MSA block-mask op --------------------------------------------------
// Emits the COMBINED attention mask: a copy of the causal/padding mask with
// every key in a NON-selected block forced to -inf. Selected blocks keep their
// causal value, so future/pad positions inside a selected block stay masked.
// Causality is taken FROM the input mask (not recomputed), so this is correct
// for arbitrary cell<->position layouts. Output dtype matches the input mask
// (f16 when flash-attn is on, f32 otherwise).
//
// Assumes M3's mask is {0 = attendable, negative = forbidden}, which holds
// because M3 uses no ALiBi/soft-cap-in-mask (soft-cap is applied to kq directly
// in build_attn_mha). If that ever changes, revisit msa_is_masked().

static inline bool        msa_is_masked(ggml_fp16_t x) { return ggml_fp16_to_fp32(x) < 0.0f; }
static inline bool        msa_is_masked(float       x) { return x < 0.0f; }
static inline ggml_fp16_t msa_neg_val (ggml_fp16_t)    { return ggml_fp32_to_fp16(-INFINITY); }
static inline float       msa_neg_val (float)          { return -INFINITY; }
//----------------------------------------------------
// helpers near the top, by msa_is_masked/msa_neg_val:
static inline float msa_to_f32(float x)       { return x; }
static inline float msa_to_f32(ggml_fp16_t x) { return ggml_fp16_to_fp32(x); }

template <typename MT>
static void msa_selfcheck_t(const MT* dst, const MT* msk,
                            int64_t S, int64_t n_kv, int64_t squery, int ith, int nth) {
    double maxd = 0; long xor0 = 0;
    for (int64_t i = ith; i < S; i += nth)
        for (int64_t j = 0; j < n_kv; ++j) {
            float a = msa_to_f32(dst[i*squery + j]);
            float b = msa_to_f32(msk[i*squery + j]);
            if ((a < 0.f) != (b < 0.f)) xor0++;          // attend/forbid flipped
            else if (a >= 0.f && a != b) maxd = std::max(maxd, (double)fabsf(a-b));
        }
    if (maxd > 0 || xor0)
        fprintf(stderr, "MSA SELFCHECK FAIL ith=%d maxd=%g xor0=%ld\n", ith, maxd, xor0);
}
//-----------------------------------------------
template <typename MT>
static inline void msa_fill_mask_t(
        MT * dst, const MT * src_mask,
        int64_t mask_skey, int64_t mask_squery,
        const float * iq, const float * ik,
        int Hd, int S, int D, int64_t n_kv,
        int blk, int topk_blocks, int local,
        int ith, int nth, int64_t key_len ) {

    const int nblk = (int)((key_len + blk - 1) / blk);
    const int topk = topk_blocks < nblk ? topk_blocks : nblk;

    std::vector<float> bs (nblk);
    std::vector<int>   ord(nblk);
    std::vector<char>  sel(nblk);

    for (int i = ith; i < S; i += nth) {
        const float * q   = iq + (size_t) i * Hd * D;
        const MT    * msk = src_mask + (int64_t) i * mask_squery;
        MT          * out = dst      + (int64_t) i * mask_squery;

        // 2) block scores; causality from the mask (skip keys it forbids).
        //    amax over keys-in-block and over the Hd index heads.
        for (int bk = 0; bk < nblk; ++bk) {
            float m = -INFINITY;
            const int j0 = bk * blk;
            if (j0 >= (int) key_len) { bs[bk] = -INFINITY; continue; }
            const int j1 = std::min((bk + 1) * blk, (int) key_len);
            for (int j = j0; j < j1; ++j) {
                if (msa_is_masked(msk[(int64_t) j * mask_skey])) continue;
                const float * k = ik + (size_t) j * D;
                for (int h = 0; h < Hd; ++h) {
                    const float * qh = q + (size_t) h * D;
                    float s = 0.f;
                    for (int d = 0; d < D; ++d) s += qh[d] * k[d];
                    if (s > m) m = s;
                }
            }
            bs[bk] = m;
        }

        // 3) force local block(s)
        int qb = -1;
        for (int bk = nblk - 1; bk >= 0; --bk) { if (bs[bk] != -INFINITY) { qb = bk; break; } }
        for (int l = 0; l < local && qb - l >= 0; ++l) {
            bs[qb - l] = INFINITY;
        }

        // 4) top-k blocks, descending
        for (int t = 0; t < nblk; ++t) ord[t] = t;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return bs[a] > bs[b]; });

        // 5) mark selected, then knock out every key in a NON-selected block.
        std::fill(sel.begin(), sel.end(), (char) 0);
        for (int t = 0; t < topk; ++t) {
            const int bk = ord[t];
            if (bs[bk] == -INFINITY) break;
            sel[bk] = 1;
        }
        // --- self-contained MSA selection logger (remove later) ---
        // Fires once per decode/prefill batch, first query, first thread, first
        // layer-at-this-n_kv (layers share n_kv within a step and run il-ascending,
        // so the first call at a given n_kv is the first sparse layer).
        if (i == 0 && ith == 0) {
            static int64_t last_n_kv = -1;
            if (n_kv < last_n_kv) last_n_kv = -1; 
            if (n_kv != last_n_kv) {
                last_n_kv = n_kv;
                FILE * f = fopen("/tmp/msa_sel.log", "a");
                if (f) {
                    fprintf(f, "n_kv=%lld qb=%d nblk=%d sel=", (long long) n_kv, qb, nblk);
                    for (int bk = 0; bk < nblk; ++bk) if (sel[bk]) fprintf(f, "%d,", bk);
                    fprintf(f, "\n");
                    fflush(f);
                }
            }
        }
        // --- end logger ---
        for (int bk = 0; bk < nblk; ++bk) {
            if (sel[bk]) continue;
            const int j0 = bk * blk;
            const int j1 = (int) std::min<int64_t>((int64_t)(bk + 1) * blk, n_kv);
            for (int j = j0; j < j1; ++j) {
                out[(int64_t) j * mask_skey] = msa_neg_val(MT(0));
            }
        }
    }
}

static void msa_block_mask_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ggml_tensor * iq   = dst->src[0];   // [D, Hd, S]       f32
    const struct ggml_tensor * ik   = dst->src[1];   // [D, 1,  n_kv]    f32 (idx cache)
    const struct ggml_tensor * mask = dst->src[2];   // [n_kv, S, 1, ns] f16/f32
    const msa_params * p = (const msa_params *) userdata;

    const int     D    = iq->ne[0];
    const int     Hd   = iq->ne[1];
    const int     S    = iq->ne[2];
    const int64_t n_kv = dst->ne[0];

    GGML_ASSERT(ik->ne[0] == D && ik->ne[1] == 1);
    GGML_ASSERT(ggml_is_contiguous(ik));
    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->ne[0] == mask->ne[0] && dst->ne[1] == mask->ne[1]);
    GGML_ASSERT(dst->nb[1] == mask->nb[1]);
    GGML_ASSERT(mask->ne[1] >= (int64_t) S);
    GGML_ASSERT(ik->type == GGML_TYPE_F32);                              // idx cache
    if (dst->src[3]) GGML_ASSERT(dst->src[3]->type == GGML_TYPE_F32);    // raw batch key
    memcpy(dst->data, mask->data, ggml_nbytes(dst));
    const int64_t key_len = ik->ne[2];
    
    if (getenv("MSA_POPCHK") && ith == 0 && dst->src[3]) {
        static int64_t last = -1;
        if (n_kv < last) last = -1;          // reset on new sequence
        if (n_kv != last) {
            last = n_kv;
            const struct ggml_tensor * ikb = dst->src[3];                // raw batch key [D,1,S]
            const int64_t Sb = ikb->ne[2];
            const float * a = (const float *) ikb->data;                 // this batch
            const float * b = (const float *) ik->data + (n_kv - Sb)*D;  // cache tail
            double maxd = 0;
            for (int64_t t = 0; t < Sb*(int64_t)D; ++t) maxd = std::max(maxd, (double)fabsf(a[t]-b[t]));
            const float * c = (const float *) ik->data;
            double cmin = 1e30, cmax = -1e30; bool nan = false;
            for (int64_t t = 0; t < n_kv*(int64_t)D; ++t) {
                float v = c[t]; if (std::isnan(v)) nan = true;
                cmin = std::min(cmin,(double)v); cmax = std::max(cmax,(double)v);
            }
            fprintf(stderr, "MSA POPCHK n_kv_mask=%lld n_kv_cache=%lld Sb=%lld tail_maxabs=%g cache[min=%g max=%g nan=%d]\n",
                    (long long) n_kv, (long long) ik->ne[2], (long long) Sb, maxd, cmin, cmax, (int)nan);
        }
    }

    const int64_t mask_skey   = 1;
    const int64_t mask_squery = mask->nb[1] / ggml_type_size(mask->type);
    static const int topk_ovr = getenv("MSA_TOPK_OVR") ? atoi(getenv("MSA_TOPK_OVR")) : -1;
    const int topk_use = topk_ovr > 0 ? topk_ovr : p->topk_blocks;
    
    if (dst->type == GGML_TYPE_F16) {
        msa_fill_mask_t<ggml_fp16_t>(
            (ggml_fp16_t *) dst->data, (const ggml_fp16_t *) mask->data,
            mask_skey, mask_squery,
            (const float *) iq->data, (const float *) ik->data,
            Hd, S, D, n_kv, p->blk, topk_use, p->local, ith, nth, key_len);
    } else {
        msa_fill_mask_t<float>(
            (float *) dst->data, (const float *) mask->data,
            mask_skey, mask_squery,
            (const float *) iq->data, (const float *) ik->data,
            Hd, S, D, n_kv, p->blk, topk_use, p->local, ith, nth, key_len);
    }
    if (getenv("MSA_SELFCHECK")) {
        if (dst->type == GGML_TYPE_F16)
            msa_selfcheck_t<ggml_fp16_t>((const ggml_fp16_t*)dst->data, (const ggml_fp16_t*)mask->data,
                                         iq->ne[2], dst->ne[0], mask_squery, ith, nth);
        else
            msa_selfcheck_t<float>((const float*)dst->data, (const float*)mask->data,
                                   iq->ne[2], dst->ne[0], mask_squery, ith, nth);
    }
}


llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const auto & mm = static_cast<const llama_model_minimax_m3 &>(model);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

    // swigluoai params, shared by dense and expert FFNs
    const float swiglu_alpha = 1.702f;
    const float swiglu_limit = 7.0f;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    auto inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // per-head QK RMSNorm (weights already include Gemma's +1)
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // partial rotary: only the first n_rot dims are rotated
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            ggml_tensor * msa_mask = nullptr;
            if (il >= (int) hparams.n_layer_dense_lead) {                 // sparse layers == MoE layers for M3
                const int64_t n_idx_dim  = hparams.indexer_head_size;     // 128
                const int64_t n_idx_head = hparams.indexer_n_head;        // 4

                ggml_tensor * iq = build_lora_mm(model.layers[il].index_q_proj, cur);  // [512, n_tokens]
                ggml_tensor * ik = build_lora_mm(model.layers[il].index_k_proj, cur);  // [128, n_tokens]
                iq = ggml_reshape_3d(ctx0, iq, n_idx_dim, n_idx_head, n_tokens);       // [128,4,T]
                ik = ggml_reshape_3d(ctx0, ik, n_idx_dim, 1,          n_tokens);       // [128,1,T]
                iq = build_norm(iq, model.layers[il].index_q_norm, NULL, LLM_NORM_RMS, il);  // +1 baked
                ik = build_norm(ik, model.layers[il].index_k_norm, NULL, LLM_NORM_RMS, il);
                iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ik = ggml_rope_ext(ctx0, ik, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                const auto * mctx_cur = inp_attn->mctx;
                const auto & k_idxs   = inp_attn->get_k_idxs();
                ggml_build_forward_expand(gf, mctx_cur->cpy_k_idx(ctx0, ik, k_idxs, il));
                ggml_tensor * ik_kv = mctx_cur->get_k_idx(ctx0, il);      // [128,1,n_kv,1]

                // fold MSA selection into the attention mask: clone the causal mask,
                // knock out non-selected blocks. FA can then run.
                ggml_tensor * kqm = inp_attn->get_kq_mask();   // f16 (FA on) / f32
                ggml_tensor * srcs[4] = { iq, ik_kv, kqm, ik };   // ik = rope'd batch key [128,1,T]
                msa_mask = ggml_custom_4d(ctx0, kqm->type, kqm->ne[0], kqm->ne[1], kqm->ne[2], kqm->ne[3],
                                          srcs, 4, msa_block_mask_op, GGML_N_TASKS_MAX,
                                          const_cast<msa_params *>(&mm.msa_p));
                if (getenv("MSA_BYPASS")) msa_mask = nullptr; // -> dense attention switch
                if (msa_mask) cb(msa_mask, "msa_mask", il);
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, /*kq_b=*/nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il,
                    /*kq_mask_override=*/msa_mask);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // leading dense FFN (swigluoai)
            ggml_tensor * g = build_lora_mm(model.layers[il].ffn_gate, cur);
            ggml_tensor * u = build_lora_mm(model.layers[il].ffn_up,   cur);
            g   = ggml_swiglu_oai(ctx0, g, u, swiglu_alpha, swiglu_limit);
            cur = build_lora_mm(model.layers[il].ffn_down, g);
            cb(cur, "ffn_out", il);
        } else {
            // routed experts (swigluoai MoE)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (swigluoai)
            ggml_tensor * sg = build_lora_mm(model.layers[il].ffn_gate_shexp, cur);
            ggml_tensor * su = build_lora_mm(model.layers[il].ffn_up_shexp,   cur);
            sg = ggml_swiglu_oai(ctx0, sg, su, swiglu_alpha, swiglu_limit);
            ggml_tensor * ffn_shexp = build_lora_mm(model.layers[il].ffn_down_shexp, sg);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
