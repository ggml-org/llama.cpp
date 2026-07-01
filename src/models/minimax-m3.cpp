#include "models.h"
#include "llama-kv-cache.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

// MiniMax-M3: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling) and
// swigluoai activation + Minimax Sparse Attention. MTP is dropped.

void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head,     false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size,  false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k,      false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks, false);
    msa_p = { (int) hparams.indexer_block_size, (int) hparams.indexer_top_k, (int) hparams.indexer_local_blocks };

    switch (hparams.n_layer()) {
        case 60: type = LLM_TYPE_428B_A23B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
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
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, hparams.indexer_head_size}, 0);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {hparams.indexer_head_size}, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {hparams.indexer_head_size}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_m3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Per-step local-block force for the MSA decode path. bias[b, i] = +BIG iff block b is one of
// query i's local blocks (L = pos/blk, plus local-1 neighbours). Added to block scores before
// top_k so the local window is always selected, matching msa_select_*'s bs[qb-l]=INF.
class llm_graph_input_msa_local : public llm_graph_input_i {
public:
    llm_graph_input_msa_local(int blk, int local, int64_t nblk) : blk(blk), local(local), nblk(nblk) {}
    void set_input(const llama_ubatch * ubatch) override {
        if (!bias || !ubatch->pos) return;
        const int64_t n_tokens = ubatch->n_tokens;
        std::vector<float> data((size_t) nblk * n_tokens, 0.0f);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int L = (int) (ubatch->pos[i] / blk);
            for (int l = 0; l < local && L - l >= 0; ++l) {
                if (L - l < nblk) data[(size_t) i * nblk + (L - l)] = 1e30f;
            }
        }
        ggml_backend_tensor_set(bias, data.data(), 0, data.size() * sizeof(float));
    }
    ggml_tensor * bias = nullptr;   // [nblk, n_tokens] f32
    int     blk;
    int     local;
    int64_t nblk;
};

static inline ggml_fp16_t msa_neg_val (ggml_fp16_t)    { return ggml_fp32_to_fp16(-INFINITY); }
static inline float       msa_neg_val (float)          { return -INFINITY; }

// ---- Decomposed MSA ----

static inline bool msa_score_masked(float x) { return x <= -1e30f; }  // -inf or pool -FLT_MAX; not +inf

// ---- MSA 4-way per-group selection -------------------------------------------------
// dst = [n_kv, S, Hd, ns] f16/f32. Channel h = base causal mask (from src[1]) with the
// blocks NOT in group h's top-k forced to -inf. bs = [nblk, Hd, S] f32 (per-group block
// scores from the decomposed OP). Per (query i, group h) the nblk scores are contiguous
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

    const bool v_trans = v->nb[1] > v->nb[2];      // false under FA

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
        cb(og, LLAMA_TENSOR_NAME_FATTN, il);

        acc = acc ? ggml_concat(ctx0, acc, og, 1) : og;    // concat along HEAD axis (ne[1])
    }

    // [D, HQ, T, 1] -> [n_embd, T]
    ggml_tensor * cur = ggml_reshape_2d(ctx0, acc, acc->ne[0]*acc->ne[1], acc->ne[2]*acc->ne[3]);
    cb(cur, "kqv_out", il);

    if (wo) cur = build_lora_mm(wo, cur, wo_s);            // o_proj
    return cur;
}

ggml_tensor * llama_model_minimax_m3::graph::build_attn_msa_decode(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo, ggml_tensor * wo_s,
        ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,
        ggml_tensor * bs,          // [nblk, Hd, 1] per-group block scores (mask-added) from the front
        ggml_tensor * local_bias,  // [nblk, 1] f32, +BIG at local block(s)
        ggml_tensor * kqm,         // [n_kv, 1, 1, 1] causal mask (f16/f32), contiguous
        int topk_blocks, float kq_scale, int il) const {

    GGML_ASSERT(!inp->self_k_rot && !inp->self_v_rot && "MSA decode: attn-rot not supported");
    GGML_ASSERT(q_cur->ne[2] == 1 && "MSA decode path is S==1 only");

    // --- store K/V to cache---
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
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);   // [D, HKV, n_kv, 1]
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);   // [D, HKV, n_kv, 1] (v_trans=false under FA)

    const int64_t D    = k->ne[0];
    const int64_t HKV  = k->ne[1];
    const int64_t n_kv = k->ne[2];
    const int64_t HQ   = q_cur->ne[1];
    const int64_t Gp   = HQ / HKV;
    const int64_t nblk = bs->ne[0];
    const int64_t Hd   = bs->ne[1];
    const int     blk  = (int) (n_kv / nblk);
    const int     K    = topk_blocks < (int) nblk ? topk_blocks : (int) nblk;
    GGML_ASSERT(Hd == HKV);
    GGML_ASSERT((int64_t) blk * nblk == n_kv);
    GGML_ASSERT(!(v->nb[1] > v->nb[2]) && "MSA decode assumes v_trans=false (FA on)");

    // --- force local block(s): bs += local_bias (broadcast over Hd) ---
    ggml_tensor * bsf = ggml_add(ctx0, bs, ggml_reshape_3d(ctx0, local_bias, nblk, 1, 1)); // [nblk,Hd,1]
    bsf = ggml_reshape_2d(ctx0, bsf, nblk, Hd);                                            // [nblk,Hd]

    // --- top-k block indices per group ---
    ggml_tensor * idx = ggml_top_k(ctx0, bsf, K);                                          // I32 [K, Hd]

    // --- expand block idx -> token idx: blk*idx + arange(blk) ---
    ggml_tensor * idxf = ggml_scale(ctx0, ggml_cast(ctx0, idx, GGML_TYPE_F32), (float) blk); // [K,Hd] f32
    idxf = ggml_reshape_3d(ctx0, idxf, 1, K, Hd);                                            // [1,K,Hd]
    ggml_tensor * tgt = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk, K, Hd);
    ggml_tensor * rep = ggml_repeat(ctx0, idxf, tgt);                                        // [blk,K,Hd]
    ggml_tensor * ar  = ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.f, (float) blk, 1.f), blk, 1, 1);
    ggml_tensor * tokf = ggml_add(ctx0, rep, ar);                                            // [blk,K,Hd]
    tokf = ggml_reshape_2d(ctx0, tokf, (int64_t) blk * K, Hd);                                // [blk*K,Hd]
    ggml_tensor * tok = ggml_cast(ctx0, tokf, GGML_TYPE_I32);                                 // [blk*K,Hd] I32

    ggml_tensor * km1 = ggml_reshape_2d(ctx0, kqm, 1, n_kv);   // [1, n_kv] for mask-gather (decode: kqm is [n_kv,1])

    ggml_tensor * acc = nullptr;
    for (int g = 0; g < (int) HKV; ++g) {
        ggml_tensor * tg = ggml_cont(ctx0, ggml_view_2d(ctx0, tok, (int64_t) blk * K, 1,
                                                        tok->nb[1], (size_t) g * tok->nb[1])); // [blk*K,1] I32

        // gather K/V for kv-head g from the strided [D,n_kv] head-slice (no cont of the cache)
        ggml_tensor * Kg2 = ggml_view_2d(ctx0, k, D, n_kv, k->nb[2], (size_t) g * k->nb[1]);
        ggml_tensor * Vg2 = ggml_view_2d(ctx0, v, D, n_kv, v->nb[2], (size_t) g * v->nb[1]);
        ggml_tensor * Kg  = ggml_get_rows(ctx0, Kg2, tg);   // [D, blk*K] f32
        ggml_tensor * Vg  = ggml_get_rows(ctx0, Vg2, tg);   // [D, blk*K] f32
        ggml_tensor * mg  = ggml_get_rows(ctx0, km1, tg);   // [1, blk*K] f32 (causal/padding per key)

        // shape for FA: q_g [D,1,Gp,1]; gathered K/V [D, blk*K, 1, 1] f16; mask [blk*K,1,1,1] f16
        ggml_tensor * qg = ggml_view_3d(ctx0, q_cur, D, Gp, 1,
                                        q_cur->nb[1], q_cur->nb[2], (size_t) g * Gp * q_cur->nb[1]);
        qg = ggml_permute(ctx0, ggml_reshape_4d(ctx0, ggml_cont(ctx0, qg), D, Gp, 1, 1), 0, 2, 1, 3);
        ggml_tensor * kgf = ggml_cast(ctx0, ggml_reshape_4d(ctx0, Kg, D, (int64_t) blk * K, 1, 1), GGML_TYPE_F16);
        ggml_tensor * vgf = ggml_cast(ctx0, ggml_reshape_4d(ctx0, Vg, D, (int64_t) blk * K, 1, 1), GGML_TYPE_F16);
        ggml_tensor * mgf = ggml_cast(ctx0, ggml_reshape_4d(ctx0, mg, (int64_t) blk * K, 1, 1, 1), GGML_TYPE_F16);

        ggml_tensor * og = ggml_flash_attn_ext(ctx0, qg, kgf, vgf, mgf, kq_scale,
                                               hparams.f_max_alibi_bias, 0.0f);
        ggml_flash_attn_ext_set_prec(og, GGML_PREC_F32);    // [D, Gp, 1, 1]
        cb(og, LLAMA_TENSOR_NAME_FATTN, il); 
        acc = acc ? ggml_concat(ctx0, acc, og, 1) : og;     // concat along head axis
    }

    ggml_tensor * cur = ggml_reshape_2d(ctx0, acc, acc->ne[0]*acc->ne[1], acc->ne[2]*acc->ne[3]); // [n_embd,1]
    cb(cur, "kqv_out", il);
    if (wo) cur = build_lora_mm(wo, cur, wo_s);
    return cur;
}

llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const auto & mm = static_cast<const llama_model_minimax_m3 &>(model);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    auto inp_attn = build_attn_inp_kv();

    // MSA decode-only gather path is active for single-token batches; build its local-force input once.
    // MSA's decode-gather AND 4-way paths both call ggml_flash_attn_ext directly and assume the
    // non-transposed V layout that llama.cpp only provides when flash attention is enabled
    // (v_trans = !cparams.flash_attn). So MSA as a whole requires FA; without it we fall back to
    // incorrect dense build_attn, which handles the transposed-V / explicit-softmax path.
    const bool fa_on       = cparams.flash_attn;     // resolved FA: supported AND enabled
    const bool single_stream = cparams.n_seq_max == 1; // MSA block selection is anchored to absolute
                                                       // KV slots, valid only for single-sequence decode
    const bool want_bypass  = getenv("MSA_BYPASS");
    const bool msa_enabled  = fa_on && single_stream && !want_bypass;

    static bool warned_no_fa = false;
    if (!fa_on && !want_bypass && !warned_no_fa) {
        LLAMA_LOG_WARN("%s: flash attention disabled; MSA requires it -> running DENSE attention "
                       "(no sparse selection). Enable flash attention for MSA.\n", __func__);
        warned_no_fa = true;
    }
    static bool warned_multi_seq = false;
    if (fa_on && !single_stream && !want_bypass && !warned_multi_seq) {
        LLAMA_LOG_WARN("%s: n_seq_max > 1; MSA is single-stream only (-np 1) -> running DENSE "
                       "attention (no sparse selection). Use -np 1 to enable MSA.\n", __func__);
        warned_multi_seq = true;
    }
    
    const bool msa_decode = msa_enabled && (n_tokens == 1); 
                                        
                                     
    llm_graph_input_msa_local * msa_loc = nullptr;
    if (msa_decode) {
        ggml_tensor * kqm0 = inp_attn->get_kq_mask();
        const int64_t n_kv0 = kqm0->ne[0];
        GGML_ASSERT(n_kv0 % mm.msa_p.blk == 0 &&
            "MSA: KV/mask n_kv must be a multiple of indexer.block_size (128); "
            "the flash-attention KV padding must be a multiple of the block size. "
            "A non-multiple would silently drop the partial tail block.");
        const int64_t nblk0 = n_kv0 / mm.msa_p.blk;
        auto loc = std::make_unique<llm_graph_input_msa_local>(mm.msa_p.blk, mm.msa_p.local, nblk0);
        loc->bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nblk0, n_tokens);   // [nblk,1] at decode
        ggml_set_input(loc->bias);
        msa_loc = (llm_graph_input_msa_local *) res->add_input(std::move(loc));
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

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
            bool          msa_decode_done = false; 
            if (msa_enabled && il >= (int) hparams.n_layer_dense_lead) {                 // sparse layers == MoE layers for M3
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
                
                // --- DEFAULT: per-group decomposed (no head-amax) ---
                GGML_ASSERT(n_kv % mm.msa_p.blk == 0 &&
                    "MSA: KV/mask n_kv must be a multiple of indexer.block_size (128); "
                    "the flash-attention KV padding must be a multiple of the block size. "
                    "A non-multiple would silently drop the partial tail block.");
                const int blk = mm.msa_p.blk;

                ggml_tensor * ik2d = ggml_reshape_2d(ctx0, ik_kv, n_idx_dim, n_kv);
                ik2d = ggml_cont(ctx0, ik2d);   // force contiguous
                ggml_tensor * iq2d = ggml_reshape_2d(ctx0, iq, n_idx_dim, n_idx_head*n_tokens);  // [D, Hd*S]
                ggml_tensor * sc   = ggml_mul_mat(ctx0, ik2d, iq2d);                             // [n_kv, Hd*S] f32
                sc = ggml_reshape_3d(ctx0, sc, n_kv, n_idx_head, n_tokens);                      // [n_kv, Hd, S]

                // + causal mask, broadcast over Hd (dim 1). Single-stream: mask is [n_kv,S,1,1].
                ggml_tensor * mf = (kqm->type == GGML_TYPE_F32) ? kqm : ggml_cast(ctx0, kqm, GGML_TYPE_F32);
                mf = ggml_reshape_3d(ctx0, mf, n_kv, 1, n_tokens);                               // [n_kv, 1, S]
                sc = ggml_add(ctx0, sc, mf);                                                     // [n_kv, Hd, S]

                // block-amax over n_kv (dim 0), keep Hd -> [nblk, Hd, S]
                ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);

                if (msa_decode) {
                    // decode gather skip the CPU tail op + msa_mask4 entirely
                    cur = build_attn_msa_decode(inp_attn,
                            model.layers[il].wo, model.layers[il].wo_s,
                            Qcur, Kcur, Vcur, bs, msa_loc->bias, kqm,
                            mm.msa_p.topk_blocks, 1.0f/sqrtf(float(n_embd_head)), il);
                    msa_decode_done = true;   // signal the dispatch below to skip
                } else {
                    ggml_tensor * srcs[4] = { bs, kqm, nullptr, nullptr };
                    int nsrc = 2;
                    msa_mask4 = ggml_custom_4d(ctx0, kqm->type,
                                               n_kv, n_tokens, n_idx_head, kqm->ne[3],
                                               srcs, nsrc, msa_mask_from_scores_4way_op, GGML_N_TASKS_MAX,
                                               const_cast<msa_params *>(&mm.msa_p));
                    cb(msa_mask4, "msa_mask4", il);
                }
                 
            }

            if (msa_decode_done) {
                // cur already computed by build_attn_msa_decode above
            } else if (il >= (int) hparams.n_layer_dense_lead && msa_mask4) {
                cur = build_attn_msa_4way(inp_attn, model.layers[il].wo, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, msa_mask4, 1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                cur = build_attn(inp_attn, model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il, nullptr);
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
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
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
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
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

