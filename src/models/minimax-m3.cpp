#include "models.h"
#include "llama-kv-cache.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

// MiniMax-M3: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling) and
// swigluoai activation. MTP is dropped.

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

// ---- MSA block-mask op ------
// Emits the COMBINED attention mask: a copy of the causal/padding mask with
// every key in a NON-selected block forced to -inf. Selected blocks keep their
// causal value, so future/pad positions inside a selected block stay masked.
// Causality is taken FROM the input mask (not recomputed). Output dtype matches the input mask
// (f16 when flash-attn is on, f32 otherwise).
//
// Assumes M3's mask is {0 = attendable, negative = forbidden}, which in this case holds
// because M3 uses no ALiBi/soft-cap-in-mask (soft-cap is applied to kq directly
// in build_attn_mha).

static inline bool        msa_is_masked(ggml_fp16_t x) { return ggml_fp16_to_fp32(x) < 0.0f; }
static inline bool        msa_is_masked(float       x) { return x < 0.0f; }
static inline ggml_fp16_t msa_neg_val (ggml_fp16_t)    { return ggml_fp32_to_fp16(-INFINITY); }
static inline float       msa_neg_val (float)          { return -INFINITY; }
//----------------------------------------------------
// Correction checkers:
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

        //block scores; Causality from the mask (skip keys it forbids). Amax over keys-in-block and over the Hd index heads.    
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

        //force local block(s)
        int qb = -1;
        for (int bk = nblk - 1; bk >= 0; --bk) { if (bs[bk] != -INFINITY) { qb = bk; break; } }
        for (int l = 0; l < local && qb - l >= 0; ++l) {
            bs[qb - l] = INFINITY;
        }

        //top-k blocks, descending
        for (int t = 0; t < nblk; ++t) ord[t] = t;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return bs[a] > bs[b]; });

        //mark selected, then knock out every key in a NON-selected block.
        std::fill(sel.begin(), sel.end(), (char) 0);
        for (int t = 0; t < topk; ++t) {
            const int bk = ord[t];
            if (bs[bk] == -INFINITY) break;
            sel[bk] = 1;
        }
        // --- MSA Selection Logger---
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

// ---- Decomposed MSA ----

static inline bool msa_score_masked(float x) { return x <= -1e30f; }  // -inf or pool -FLT_MAX; not +inf

template <typename MT>
static inline void msa_select_from_scores_t(
        MT * dst,
        int64_t mask_skey, int64_t mask_squery,
        const float * bs_in,                 // [nblk, S], column i at bs_in + i*nblk
        int S, int64_t n_kv,
        int blk, int topk_blocks, int local,
        int ith, int nth, int nblk ) {

    const int topk = topk_blocks < nblk ? topk_blocks : nblk;

    std::vector<float> bs (nblk);
    std::vector<int>   ord(nblk);
    std::vector<char>  sel(nblk);

    for (int i = ith; i < S; i += nth) {
        MT          * out  = dst   + (int64_t) i * mask_squery;   // base already memcpy'd from mask
        const float * bcol = bs_in + (size_t)  i * nblk;

        for (int bk = 0; bk < nblk; ++bk) bs[bk] = bcol[bk];      // local copy; never mutate src

        // qb = highest block with a real (non-masked) score == local block of this query.
        // threshold, so the pool's -FLT_MAX on pad/future blocks counts as empty.
        int qb = -1;
        for (int bk = nblk - 1; bk >= 0; --bk) { if (!msa_score_masked(bs[bk])) { qb = bk; break; } }
        for (int l = 0; l < local && qb - l >= 0; ++l) bs[qb - l] = INFINITY;

        for (int t = 0; t < nblk; ++t) ord[t] = t;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return bs[a] > bs[b]; });

        std::fill(sel.begin(), sel.end(), (char) 0);
        for (int t = 0; t < topk; ++t) {
            const int bk = ord[t];
            if (msa_score_masked(bs[bk])) break;   // first empty block: real blocks < topk
            sel[bk] = 1;
        }

        for (int bk = 0; bk < nblk; ++bk) {
            if (sel[bk]) continue;
            const int j0 = bk * blk;
            const int j1 = (int) std::min<int64_t>((int64_t)(bk + 1) * blk, n_kv);
            for (int j = j0; j < j1; ++j) out[(int64_t) j * mask_skey] = msa_neg_val(MT(0));
        }
    }
}

// VERIFY ONLY:
template <typename MT>
static void msa_verify_bs(
        const float * bs_gpu, const MT * src_mask,
        int64_t mask_skey, int64_t mask_squery,
        const float * iq, const float * ik,
        int Hd, int S, int D, int64_t n_kv,
        int blk, int topk_blocks, int local, int nblk, int64_t key_len ) {

    const int topk = topk_blocks < nblk ? topk_blocks : nblk;
    std::vector<float> bsc(nblk), bsg(nblk);
    std::vector<int>   ord(nblk);
    std::vector<char>  selc(nblk), selg(nblk);
    double max_abs = 0.0;
    long   finite_pairs = 0, gross = 0, sel_diff_cols = 0;

    auto sel_of = [&](const std::vector<float>& in, std::vector<char>& out){
        std::vector<float> b(in);
        int qb = -1; for (int bk = nblk-1; bk >= 0; --bk) { if (!msa_score_masked(b[bk])) { qb = bk; break; } }
        for (int l = 0; l < local && qb-l >= 0; ++l) b[qb-l] = INFINITY;
        for (int t = 0; t < nblk; ++t) ord[t] = t;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b2){ return b[a] > b[b2]; });
        out.assign(nblk, 0);
        for (int t = 0; t < topk; ++t) { int bk = ord[t]; if (msa_score_masked(b[bk])) break; out[bk] = 1; }
    };

    for (int i = 0; i < S; ++i) {
        const float * q   = iq + (size_t) i * Hd * D;
        const MT    * msk = src_mask + (int64_t) i * mask_squery;

        // monolithic bs: joint head+key amax, masked keys skipped (-> -inf if empty)
        for (int bk = 0; bk < nblk; ++bk) {
            float m = -INFINITY; const int j0 = bk * blk;
            if (j0 >= (int) key_len) { bsc[bk] = -INFINITY; continue; }
            const int j1 = std::min((bk + 1) * blk, (int) key_len);
            for (int j = j0; j < j1; ++j) {
                if (msa_is_masked(msk[(int64_t) j * mask_skey])) continue;
                const float * k = ik + (size_t) j * D;
                for (int h = 0; h < Hd; ++h) {
                    const float * qh = q + (size_t) h * D; float s = 0.f;
                    for (int d = 0; d < D; ++d) s += qh[d] * k[d];
                    if (s > m) m = s;
                }
            }
            bsc[bk] = m;
        }
        for (int bk = 0; bk < nblk; ++bk) bsg[bk] = bs_gpu[(size_t) i * nblk + bk];

        //score delta (both-masked equal; one masked = gross layout error)
        for (int bk = 0; bk < nblk; ++bk) {
            bool mc = msa_score_masked(bsc[bk]), mg = msa_score_masked(bsg[bk]);
            if (mc && mg) continue;
            if (mc != mg) { gross++; max_abs = std::max(max_abs, 1e30); continue; }
            max_abs = std::max(max_abs, (double) fabsf(bsc[bk] - bsg[bk])); finite_pairs++;
        }
        //selection agreement
        sel_of(bsc, selc); sel_of(bsg, selg);
        if (selc != selg) sel_diff_cols++;
    }
    fprintf(stderr, "MSA VERIFY n_kv=%lld nblk=%d S=%d | StageA max|dbs|=%.3g finite=%ld gross=%ld | StageB sel_diff=%ld%s\n",
            (long long) n_kv, nblk, S, max_abs, finite_pairs, gross, sel_diff_cols,
            (gross || max_abs > 1e-2 || sel_diff_cols) ? "  <-- INVESTIGATE" : "");
}

static void msa_mask_from_scores_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ggml_tensor * bs   = dst->src[0];   // [nblk, S]        f32
    const struct ggml_tensor * mask = dst->src[1];   // [n_kv, S, 1, ns] f16/f32
    const msa_params * p = (const msa_params *) userdata;

    const int     S    = bs->ne[1];
    const int     nblk = bs->ne[0];
    const int64_t n_kv = dst->ne[0];

    GGML_ASSERT(bs->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(bs));
    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->ne[0] == mask->ne[0] && dst->ne[1] == mask->ne[1]);
    GGML_ASSERT(dst->nb[1] == mask->nb[1]);
    GGML_ASSERT(mask->ne[1] >= (int64_t) S);
    GGML_ASSERT(n_kv % p->blk == 0);                 // pool drops a partial tail block
    GGML_ASSERT((int64_t) nblk == n_kv / p->blk);    // GPU front block count must match

    memcpy(dst->data, mask->data, ggml_nbytes(dst));

    const int64_t mask_skey   = 1;
    const int64_t mask_squery = mask->nb[1] / ggml_type_size(mask->type);

    if (dst->type == GGML_TYPE_F16) {
        msa_select_from_scores_t<ggml_fp16_t>(
            (ggml_fp16_t *) dst->data, mask_skey, mask_squery, (const float *) bs->data,
            S, n_kv, p->blk, p->topk_blocks, p->local, ith, nth, nblk);
    } else {
        msa_select_from_scores_t<float>(
            (float *) dst->data, mask_skey, mask_squery, (const float *) bs->data,
            S, n_kv, p->blk, p->topk_blocks, p->local, ith, nth, nblk);
    }

    if (ith == 0 && getenv("MSA_VERIFY_BS") && dst->src[2] && dst->src[3]) {
        const struct ggml_tensor * iq = dst->src[2];
        const struct ggml_tensor * ik = dst->src[3];
        const int D  = iq->ne[0];
        const int Hd = iq->ne[1];
        if (dst->type == GGML_TYPE_F16)
            msa_verify_bs<ggml_fp16_t>((const float*) bs->data, (const ggml_fp16_t*) mask->data,
                mask_skey, mask_squery, (const float*) iq->data, (const float*) ik->data,
                Hd, S, D, n_kv, p->blk, p->topk_blocks, p->local, nblk, ik->ne[2]);
        else
            msa_verify_bs<float>((const float*) bs->data, (const float*) mask->data,
                mask_skey, mask_squery, (const float*) iq->data, (const float*) ik->data,
                Hd, S, D, n_kv, p->blk, p->topk_blocks, p->local, nblk, ik->ne[2]);
    }
}

// ---- MSA 4-way per-group selection -------------------------------------------------
// dst = [n_kv, S, Hd, ns] f16/f32. Channel h = base causal mask (from src[1]) with the
// blocks NOT in group h's top-k forced to -inf. bs = [nblk, Hd, S] f32 (per-group block
// scores from the decomposed front). Per (query i, group h) the nblk scores are contiguous
// at bs + i*nblk*Hd + h*nblk. No global memcpy: each (i,h) copies its base-mask column,
// so threads partitioned over i never race (channels within a column are disjoint).
template <typename MT>
static inline void msa_select_4way_t(
        MT * dst, const MT * base_mask,
        int64_t mask_skey, int64_t dst_squery, int64_t base_squery, int64_t chan_stride,
        const float * bs_in, int Hd, int S, int64_t n_kv,
        int blk, int topk_blocks, int local,
        int ith, int nth, int nblk ) {

    const int topk = topk_blocks < nblk ? topk_blocks : nblk;
    std::vector<float> bs (nblk);
    std::vector<int>   ord(nblk);
    std::vector<char>  sel(nblk);

    for (int i = ith; i < S; i += nth) {
        for (int h = 0; h < Hd; ++h) {
            MT       * out = dst       + (int64_t) h*chan_stride + (int64_t) i*dst_squery;
            const MT * src = base_mask + (int64_t) i*base_squery;
            for (int64_t j = 0; j < n_kv; ++j) out[j*mask_skey] = src[j*mask_skey];   // base col i

            const float * bcol = bs_in + (size_t) i*nblk*Hd + (size_t) h*nblk;
            for (int bk = 0; bk < nblk; ++bk) bs[bk] = bcol[bk];

            // local block = highest non-masked block for this query; force-keep it (+ local-1 neighbors)
            int qb = -1;
            for (int bk = nblk - 1; bk >= 0; --bk) { if (!msa_score_masked(bs[bk])) { qb = bk; break; } }
            for (int l = 0; l < local && qb - l >= 0; ++l) bs[qb - l] = INFINITY;

            for (int t = 0; t < nblk; ++t) ord[t] = t;
            // top-k SET only; partial_sort keeps the "break at first empty" logic and
            // at O(nblk log topk) instead of O(nblk log nblk).
            std::partial_sort(ord.begin(), ord.begin() + topk, ord.end(),
                              [&](int a, int b){ return bs[a] > bs[b]; });

            std::fill(sel.begin(), sel.end(), (char) 0);
            for (int t = 0; t < topk; ++t) {
                const int bk = ord[t];
                if (msa_score_masked(bs[bk])) break;   // sorted desc: first empty => fewer than topk real
                sel[bk] = 1;
            }
           // ===== TEMP reuse probe. abs_pos read from the mask
            // (highest causally-visible key), robust to KV padding, unlike (n_kv - S).
            if (const char * e = getenv("MSA_DUMP_SEL")) {
                int abs_pos = -1;
                for (int64_t j = n_kv - 1; j >= 0; --j)
                    if (!msa_is_masked(src[j*mask_skey])) { abs_pos = (int) j; break; }   // src = base mask col i
                if (abs_pos == atoi(e)) {
                    char b[8192]; int n = 0;
                    n += snprintf(b+n, sizeof(b)-n, "SELDUMP pos=%d g=%d nblk=%d sel:", abs_pos, h, nblk);
                    for (int bk = 0; bk < nblk && n < (int) sizeof(b) - 8; ++bk)
                        if (sel[bk]) n += snprintf(b+n, sizeof(b)-n, " %d", bk);
                    fprintf(stderr, "%s\n", b);
                }
            }
            // ===== end probe =====

            for (int bk = 0; bk < nblk; ++bk) {
                if (sel[bk]) continue;
                const int j0 = bk * blk;
                const int j1 = (int) std::min<int64_t>((int64_t)(bk + 1) * blk, n_kv);
                for (int j = j0; j < j1; ++j) out[(int64_t) j * mask_skey] = msa_neg_val(MT(0));
            }
        }
    }
}

static void msa_mask_from_scores_4way_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ggml_tensor * bs   = dst->src[0];   // [nblk, Hd, S]     f32
    const struct ggml_tensor * mask = dst->src[1];   // [n_kv, S, 1, ns]  base causal mask
    const msa_params * p = (const msa_params *) userdata;

    const int     nblk = bs->ne[0];
    const int     Hd   = bs->ne[1];
    const int     S    = bs->ne[2];
    const int64_t n_kv = dst->ne[0];

    GGML_ASSERT(bs->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(bs));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->ne[0] == mask->ne[0]);            // n_kv
    GGML_ASSERT(dst->ne[1] == mask->ne[1]);            // S
    GGML_ASSERT(dst->ne[2] == (int64_t) Hd);           // channels = index heads
    GGML_ASSERT(mask->ne[2] == 1);
    GGML_ASSERT(mask->ne[3] == 1 && "MSA 4-way assumes single stream (-np 1)");
    GGML_ASSERT(n_kv % p->blk == 0);
    GGML_ASSERT((int64_t) nblk == n_kv / p->blk);

    const int64_t ts          = ggml_type_size(dst->type);
    const int64_t mask_skey   = 1;
    const int64_t dst_squery  = dst->nb[1]  / ts;      // = n_kv
    const int64_t base_squery = mask->nb[1] / ts;
    const int64_t chan_stride = dst->nb[2]  / ts;      // = n_kv * S

    if (dst->type == GGML_TYPE_F16) {
        msa_select_4way_t<ggml_fp16_t>(
            (ggml_fp16_t *) dst->data, (const ggml_fp16_t *) mask->data,
            mask_skey, dst_squery, base_squery, chan_stride,
            (const float *) bs->data, Hd, S, n_kv,
            p->blk, p->topk_blocks, p->local, ith, nth, nblk);
    } else {
        msa_select_4way_t<float>(
            (float *) dst->data, (const float *) mask->data,
            mask_skey, dst_squery, base_squery, chan_stride,
            (const float *) bs->data, Hd, S, n_kv,
            p->blk, p->topk_blocks, p->local, ith, nth, nblk);
    }

    // per-group correctness check vs a MASKED monolithic recompute (slow, single-thread).
    if (ith == 0 && getenv("MSA_VERIFY_BS4") && dst->src[2] && dst->src[3]) {
        const struct ggml_tensor * iq    = dst->src[2];   // [D, Hd, S] f32
        const struct ggml_tensor * ik    = dst->src[3];   // [D, 1, n_kv] f32
        const struct ggml_tensor * bmask = dst->src[1];   // [n_kv, S, 1, ns] base causal mask
        const int     D       = iq->ne[0];
        const int64_t key_len = ik->ne[2];
        const int     topk    = p->topk_blocks < nblk ? p->topk_blocks : nblk;
        const int64_t bm_skey = 1;
        const int64_t bm_sq   = bmask->nb[1] / ggml_type_size(bmask->type);
        const float * IK = (const float *) ik->data;

        auto key_masked = [&](int64_t j, int i) -> bool {
            const int64_t off = j*bm_skey + (int64_t) i*bm_sq;
            return bmask->type == GGML_TYPE_F16
                 ? msa_is_masked(((const ggml_fp16_t *) bmask->data)[off])
                 : msa_is_masked(((const float       *) bmask->data)[off]);
        };

        std::vector<int> ord(nblk); std::vector<char> sc_(nblk), sg_(nblk);
        auto sel_of = [&](std::vector<float> & b, std::vector<char> & o){
            int qb=-1; for(int bk=nblk-1;bk>=0;--bk){ if(!msa_score_masked(b[bk])){qb=bk;break;} }
            for(int l=0;l<p->local&&qb-l>=0;++l) b[qb-l]=INFINITY;
            for(int t=0;t<nblk;++t) ord[t]=t;
            std::partial_sort(ord.begin(),ord.begin()+topk,ord.end(),[&](int a,int b2){return b[a]>b[b2];});
            o.assign(nblk,0);
            for(int t=0;t<topk;++t){ int bk=ord[t]; if(msa_score_masked(b[bk])) break; o[bk]=1; }
        };

        long sel_diff = 0; double max_abs = 0;
        for (int i = 0; i < S; ++i) {
            const float * Q = (const float *) iq->data + (size_t) i*Hd*D;
            for (int h = 0; h < Hd; ++h) {
                const float * qh = Q + (size_t) h*D;
                std::vector<float> bman(nblk), bgpu(nblk);
                for (int bk = 0; bk < nblk; ++bk) {
                    float m = -INFINITY; const int j0 = bk*p->blk;
                    const int j1 = std::min<int>((bk+1)*p->blk, (int) key_len);
                    for (int j = j0; j < j1; ++j) {
                        if (key_masked(j, i)) continue;          // <-- causal/padding mask (was missing)
                        const float * kk = IK + (size_t) j*D; float s = 0;
                        for (int d = 0; d < D; ++d) s += qh[d]*kk[d];
                        if (s > m) m = s;
                    }
                    bman[bk] = m;                                 // -inf if block fully masked
                    bgpu[bk] = ((const float *) bs->data)[(size_t) i*nblk*Hd + (size_t) h*nblk + bk];
                }
                for (int bk = 0; bk < nblk; ++bk) {
                    bool mc = msa_score_masked(bman[bk]), mg = msa_score_masked(bgpu[bk]);
                    if (mc && mg) continue;
                    if (mc != mg) { max_abs = 1e30; continue; }
                    max_abs = std::max(max_abs, (double) fabsf(bman[bk]-bgpu[bk]));
                }
                std::vector<float> a=bman,b=bgpu; sel_of(a,sc_); sel_of(b,sg_); if (sc_!=sg_) sel_diff++;
            }
        }
        // TF32 on the f32 indexer mul_mat gives max|dbs| ~0.1; only gross score mismatch
        // (layout bug -> 1e30) or any selection mismatch is a real problem.
        fprintf(stderr, "MSA4 VERIFY n_kv=%lld nblk=%d Hd=%d S=%d | max|dbs|=%.3g per-group sel_diff=%ld%s\n",
                (long long) n_kv, nblk, Hd, S, max_abs, sel_diff,
                (max_abs > 0.5 || sel_diff) ? "  <-- INVESTIGATE" : "");
    }
}

ggml_tensor * llama_model_minimax_m3::graph::build_attn_msa_4way(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo, ggml_tensor * wo_s,
        ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,
        ggml_tensor * msa_mask4, float kq_scale, int il) const {

    GGML_ASSERT(!inp->self_k_rot && !inp->self_v_rot && "MSA 4-way: attn-rot not supported");

    // --- store K/V to cache (mirror build_attn) ---
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);
    const auto * mctx_cur = inp->mctx;
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);   // [D, n_head_kv, n_kv, ns]
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);   // [D, n_head_kv, n_kv, ns]  (v_trans=false under FA)

    const int64_t D    = k->ne[0];
    const int64_t HKV  = k->ne[1];
    const int64_t n_kv = k->ne[2];
    const int64_t ns   = k->ne[3];
    const int64_t HQ   = q_cur->ne[1];
    const int64_t T    = q_cur->ne[2];
    const int64_t Gp   = HQ / HKV;                 // query heads per group (16)
    GGML_ASSERT(HQ % HKV == 0);
    GGML_ASSERT(ns == 1 && "MSA 4-way assumes single stream (-np 1)");
    GGML_ASSERT(msa_mask4->ne[2] == HKV);          // one channel per group

    const bool v_trans = v->nb[1] > v->nb[2];      // false under FA; handled defensively

    ggml_tensor * acc = nullptr;
    for (int g = 0; g < (int) HKV; ++g) {
        // Q heads [Gp*g, Gp*g+Gp)  -> [D, Gp, T, 1] -> permute -> [D, T, Gp, 1]
        ggml_tensor * qg = ggml_view_4d(ctx0, q_cur, D, Gp, T, 1,
                                        q_cur->nb[1], q_cur->nb[2], q_cur->nb[3],
                                        (size_t) g * Gp * q_cur->nb[1]);
        qg = ggml_permute(ctx0, qg, 0, 2, 1, 3);

        // K kv-head g -> [D, 1, n_kv, ns] -> permute -> [D, n_kv, 1, ns]
        ggml_tensor * kg = ggml_view_4d(ctx0, k, D, 1, n_kv, ns,
                                        k->nb[1], k->nb[2], k->nb[3], (size_t) g * k->nb[1]);
        kg = ggml_permute(ctx0, kg, 0, 2, 1, 3);

        // V kv-head g (same head=ne[1] slice; works for both v_trans layouts)
        ggml_tensor * vg = ggml_view_4d(ctx0, v, v->ne[0], 1, v->ne[2], ns,
                                        v->nb[1], v->nb[2], v->nb[3], (size_t) g * v->nb[1]);
        vg = ggml_permute(ctx0, vg, 0, 2, 1, 3);
        if (v_trans) vg = ggml_transpose(ctx0, vg);     // no-op under FA

        if (kg->type == GGML_TYPE_F32) kg = ggml_cast(ctx0, kg, GGML_TYPE_F16);
        if (vg->type == GGML_TYPE_F32) vg = ggml_cast(ctx0, vg, GGML_TYPE_F16);

        // mask channel g -> [n_kv, S, 1] contiguous slice (FA broadcasts over the Gp heads)
        ggml_tensor * mg = ggml_view_3d(ctx0, msa_mask4, msa_mask4->ne[0], msa_mask4->ne[1], 1,
                                        msa_mask4->nb[1], msa_mask4->nb[2],
                                        (size_t) g * msa_mask4->nb[2]);

        ggml_tensor * og = ggml_flash_attn_ext(ctx0, qg, kg, vg, mg, kq_scale,
                                               hparams.f_max_alibi_bias, 0.0f);
        ggml_flash_attn_ext_set_prec(og, GGML_PREC_F32);   // [D, Gp, T, 1]

        acc = acc ? ggml_concat(ctx0, acc, og, 1) : og;    // concat along HEAD axis (ne[1])
    }

    // [D, HQ, T, 1] -> [n_embd, T]
    ggml_tensor * cur = ggml_reshape_2d(ctx0, acc, acc->ne[0]*acc->ne[1], acc->ne[2]*acc->ne[3]);
    cb(cur, "kqv_out", il);

    if (wo) cur = build_lora_mm(wo, cur, wo_s);            // o_proj
    return cur;
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
            ggml_tensor * msa_mask4 = nullptr;   // [n_kv, S, Hd, ns] per-group (default)
            ggml_tensor * msa_mask  = nullptr;   // [n_kv, S, 1,  ns] shared    (legacy debug)
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

                ggml_tensor * kqm  = inp_attn->get_kq_mask();   // f16 (FA on) / f32
                const int64_t n_kv = kqm->ne[0];



                const bool want_bypass = getenv("MSA_BYPASS");          // -> dense
                const bool want_shared = getenv("MSA_SHARED");          // -> old monolithic single-mask

                if (!want_bypass && !want_shared) {
                    // --- DEFAULT: per-group decomposed front (no head-amax) ---
                    const int blk = mm.msa_p.blk;

                    ggml_tensor * ik2d = ggml_reshape_2d(ctx0, ik_kv, n_idx_dim, n_kv);              // [D, n_kv]
                    ggml_tensor * iq2d = ggml_reshape_2d(ctx0, iq, n_idx_dim, n_idx_head*n_tokens);  // [D, Hd*S]
                    ggml_tensor * sc   = ggml_mul_mat(ctx0, ik2d, iq2d);                             // [n_kv, Hd*S] f32
                    sc = ggml_reshape_3d(ctx0, sc, n_kv, n_idx_head, n_tokens);                      // [n_kv, Hd, S]

                    // + causal mask, broadcast over Hd (dim 1). Single-stream: mask is [n_kv,S,1,1].
                    ggml_tensor * mf = (kqm->type == GGML_TYPE_F32) ? kqm : ggml_cast(ctx0, kqm, GGML_TYPE_F32);
                    mf = ggml_reshape_3d(ctx0, mf, n_kv, 1, n_tokens);                               // [n_kv, 1, S]
                    sc = ggml_add(ctx0, sc, mf);                                                     // [n_kv, Hd, S]

                    // block-amax over n_kv (dim 0), keep Hd -> [nblk, Hd, S]
                    ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);

                    ggml_tensor * srcs[4] = { bs, kqm, nullptr, nullptr };
                    int nsrc = 2;
                    if (getenv("MSA_VERIFY_BS4")) { srcs[2] = iq; srcs[3] = ik_kv; nsrc = 4; }
                    msa_mask4 = ggml_custom_4d(ctx0, kqm->type,
                                               n_kv, n_tokens, n_idx_head, kqm->ne[3],
                                               srcs, nsrc, msa_mask_from_scores_4way_op, GGML_N_TASKS_MAX,
                                               const_cast<msa_params *>(&mm.msa_p));
                    cb(msa_mask4, "msa_mask4", il);
                } else if (want_shared) {
                    // --- legacy shared-mask monolithic op (single FA call via build_attn) ---
                    ggml_tensor * srcs[4] = { iq, ik_kv, kqm, ik };
                    msa_mask = ggml_custom_4d(ctx0, kqm->type, kqm->ne[0], kqm->ne[1], kqm->ne[2], kqm->ne[3],
                                              srcs, 4, msa_block_mask_op, GGML_N_TASKS_MAX,
                                              const_cast<msa_params *>(&mm.msa_p));
                    cb(msa_mask, "msa_mask", il);
                }
                // want_bypass: both null -> dense attention
                if (getenv("MSA_BYPASS")) msa_mask = nullptr; // -> dense attention switch
                if (msa_mask) cb(msa_mask, "msa_mask", il);
            }

            if (il >= (int) hparams.n_layer_dense_lead && msa_mask4) {
                // sparse layer, per-group path
                cur = build_attn_msa_4way(inp_attn,
                        model.layers[il].wo, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, msa_mask4,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                // dense layers, MSA_BYPASS (msa_mask == nullptr), or legacy shared (msa_mask set)
                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, /*kq_b=*/nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il,
                        /*kq_mask_override=*/msa_mask);
            }
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
