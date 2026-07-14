#include "models.h"
#include "llama-kv-cache.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

// MiniMax-M3: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling),
// swigluoai activation, and MiniMax Sparse Attention (MSA). MTP is not in released model weights.
// Notes: Blocks are anchored to absolute KV cache slots.

void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head,       false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size,    false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k,        false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size,   false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks, false);
    msa_p = { (int) hparams.indexer_block_size, (int) hparams.indexer_top_k, (int) hparams.indexer_local_blocks };
    hparams.indexer_kv = true;

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

// per-query local-force bias for MSA selection
// local window always wins a slot
class llm_graph_input_msa_local : public llm_graph_input_i {
public:
    llm_graph_input_msa_local(int blk, int local, int64_t nblk) : blk(blk), local(local), nblk(nblk) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (!bias || !ubatch->pos) {
            return;
        }
        const int64_t n_tokens = ubatch->n_tokens;
        std::vector<float> data((size_t) nblk * n_tokens, 0.0f);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int64_t L = ubatch->pos[i] / blk;
            for (int l = 0; l < local && L - l >= 0; ++l) {
                if (L - l < nblk) {
                    data[(size_t) i * nblk + (L - l)] = 1e30f;
                }
            }
        }
        ggml_backend_tensor_set(bias, data.data(), 0, data.size() * sizeof(float));
    }

    // valid as long as the bias tensor dims still match the new ubatch/cache window
    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

        bool res = true;
        res &= bias->ne[1] == params.ubatch.n_tokens;
        res &= bias->ne[0] * blk == (int64_t) mctx->get_n_kv();
        return res;
    }

    ggml_tensor * bias = nullptr;
    int     blk;
    int     local;
    int64_t nblk;
};

// pooled score of a block with no visible token: -inf from the mask, or -FLT_MAX from the
// max-pool identity when every element of the block is -inf
static inline bool msa_score_masked(float x) { return x <= -1e30f; }

// MSA block selection (batch regime)
// CPU custom op, the token-level expansion and the combination with the causal mask happen on the GPU.
static void msa_block_mask_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ggml_tensor * bs   = dst->src[0];
    const struct ggml_tensor * bias = dst->src[1];
    const msa_params * p = (const msa_params *) userdata;

    const int nblk = (int) bs->ne[0];
    const int Hd   = (int) bs->ne[1];
    const int S    = (int) bs->ne[2];

    GGML_ASSERT(bs->type   == GGML_TYPE_F32 && ggml_is_contiguous(bs));
    GGML_ASSERT(bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias));
    GGML_ASSERT(dst->type  == GGML_TYPE_F16 && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[0] == nblk && dst->ne[1] == S && dst->ne[2] == Hd);
    GGML_ASSERT(bias->ne[0] == nblk && bias->ne[1] == S);

    const int topk = p->topk_blocks < nblk ? p->topk_blocks : nblk;

    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);

    std::vector<float> rank(nblk);
    std::vector<char>  valid(nblk);
    std::vector<int>   ord(nblk);

    ggml_fp16_t * out = (ggml_fp16_t *) dst->data;

    for (int i = ith; i < S; i += nth) {
        const float * bias_col = (const float *) bias->data + (size_t) i * nblk;
        for (int h = 0; h < Hd; ++h) {
            const float * bs_col = (const float *) bs->data + ((size_t) i * Hd + h) * nblk;

            for (int bk = 0; bk < nblk; ++bk) {
                // a block is selectable if it has a visible token or is locally forced
                valid[bk] = !msa_score_masked(bs_col[bk]) || bias_col[bk] > 0.0f;
                rank [bk] = bs_col[bk] + bias_col[bk];
                ord  [bk] = bk;
            }

            std::partial_sort(ord.begin(), ord.begin() + topk, ord.end(),
                              [&](int a, int b) { return rank[a] > rank[b]; });

            ggml_fp16_t * dst_col = out + ((size_t) h * S + i) * nblk;
            for (int bk = 0; bk < nblk; ++bk) {
                dst_col[bk] = f16_ninf;
            }
            for (int t = 0; t < topk; ++t) {
                const int bk = ord[t];
                if (!valid[bk]) {
                    break;   // sorted desc: first invalid -> fewer than topk selectable blocks
                }
                dst_col[bk] = f16_zero;
            }
        }
    }
}

// One FA call for all GQA groups (and at multi-stream decode, all streams) by mapping them onto the FA sequence dim (ne[3])
ggml_tensor * llama_model_minimax_m3::graph::build_attn_msa_fa(
        ggml_tensor * q_cur,   // [D, HQ, T]
        ggml_tensor * k,       // [D, n_keys, 1, C]
        ggml_tensor * v,       // [D, n_keys, 1, C]
        ggml_tensor * mask,    // [n_keys, R, 1, C] f16, contiguous
        int64_t Gp, float kq_scale, int il) const {

    const int64_t D  = q_cur->ne[0];
    const int64_t HQ = q_cur->ne[1];
    const int64_t T  = q_cur->ne[2];
    const int64_t C  = k->ne[3];
    const int64_t R  = HQ*T/(Gp*C);
    GGML_ASSERT(Gp*C*R == HQ*T);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);

    // [D, HQ, T] -> [D, Gp, C, R] -> [D, R, Gp, C]
    // batch  (C=HKV,   R=T): channel = group
    // decode (C=HKV*ns, R=1): channel = (group, stream), group innermost
    ggml_tensor * q = ggml_reshape_4d(ctx0, q_cur, D, Gp, C, R);
    q = ggml_permute(ctx0, q, 0, 2, 3, 1);

    ggml_tensor * o = ggml_flash_attn_ext(ctx0, q, k, v, mask, kq_scale,
                                          hparams.f_max_alibi_bias, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    cb(o, "msa_fattn", il);

    // [D, Gp, R, C] -> [D, Gp, C, R] -> [n_embd, T]
    o = ggml_permute(ctx0, o, 0, 1, 3, 2);
    if (!ggml_is_contiguous(o)) {
        o = ggml_cont(ctx0, o);   // no-op layout at decode (R == 1), copy at batch
    }
    return ggml_reshape_2d(ctx0, o, D*HQ, T);
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

    // MSA calls ggml_flash_attn_ext directly and assumes the non-transposed V layout that
    // llama.cpp only provides when flash attention is enabled. Block selection is anchored
    // to absolute KV cache slots, which equal positions only for append-only per-stream
    // caches either a single sequence, or multiple sequences with kv_unified == false (each
    // stream then has its own slot space). A unified cache with multiple sequences
    // interleaves slots and would silently break block anchoring so it falls back to dense.
    const bool fa_on       = cparams.flash_attn;
    const bool streams_ok  = cparams.n_seq_max == 1 || !cparams.kv_unified;
    const bool msa_enabled = fa_on && streams_ok;

    static bool warned_no_fa = false;
    if (!fa_on && !warned_no_fa) {
        LLAMA_LOG_WARN("%s: flash attention disabled; MSA requires it -> running DENSE attention "
                       "(output may be degraded). Enable flash attention for MSA.\n", __func__);
        warned_no_fa = true;
    }
    static bool warned_unified = false;
    if (fa_on && !streams_ok && !warned_unified) {
        LLAMA_LOG_WARN("%s: unified KV cache with n_seq_max > 1; MSA needs per-sequence streams "
                       "-> running DENSE attention. Output may be degraded. Drop --kv-unified to enable MSA.\n", __func__);
        warned_unified = true;
    }

    // hoisted per-graph MSA state (shared by every sparse layer)
    llm_graph_input_msa_local * msa_loc = nullptr;
    ggml_tensor * msa_kqm = nullptr;
    ggml_tensor * msa_mf  = nullptr;
    int64_t n_kv = 0, nblk = 0, ns = 1, n_tps = 0;
    bool msa_decode = false;           // gather (1 token per stream) vs mask
    const int     blk = mm.msa_p.blk;
    const int64_t Hd  = hparams.indexer_n_head;   // one indexer head per GQA group

    if (msa_enabled) {
        msa_kqm = inp_attn->get_kq_mask();
        n_kv  = msa_kqm->ne[0];
        n_tps = msa_kqm->ne[1];        // tokens per stream
        ns    = msa_kqm->ne[3];        // streams in this ubatch
        GGML_ASSERT(msa_kqm->type == GGML_TYPE_F16 && "MSA requires the FA (f16) mask");
        GGML_ASSERT(n_tps*ns == n_tokens);
        GGML_ASSERT(n_kv % blk == 0 &&
            "MSA: KV/mask n_kv must be a multiple of indexer.block_size (128); "
            "the flash-attention KV padding must be a multiple of the block size. "
            "A non-multiple would silently drop the partial tail block.");
        nblk = n_kv / blk;
        msa_decode = n_tps == 1;

        msa_mf = ggml_cast(ctx0, msa_kqm, GGML_TYPE_F32);

        auto loc = std::make_unique<llm_graph_input_msa_local>(blk, mm.msa_p.local, nblk);
        loc->bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nblk, n_tokens);  // stream-grouped tokens
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
                ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            const bool is_sparse = msa_enabled && il >= (int) hparams.n_layer_dense_lead;

            if (!is_sparse) {
                cur = build_attn(inp_attn, model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                const int64_t n_idx_dim = hparams.indexer_head_size;   // 128

                GGML_ASSERT(!inp_attn->self_k_rot && !inp_attn->self_v_rot && "MSA: attn-rot not supported");

                // Index Branch, project, norm, partial RoPE, cache
                ggml_tensor * iq = build_lora_mm(model.layers[il].index_q_proj, cur);
                ggml_tensor * ik = build_lora_mm(model.layers[il].index_k_proj, cur);
                iq = ggml_reshape_3d(ctx0, iq, n_idx_dim, Hd, n_tokens);
                ik = ggml_reshape_3d(ctx0, ik, n_idx_dim, 1,  n_tokens);
                iq = build_norm(iq, model.layers[il].index_q_norm, NULL, LLM_NORM_RMS, il);  // +1 baked
                ik = build_norm(ik, model.layers[il].index_k_norm, NULL, LLM_NORM_RMS, il);
                iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ik = ggml_rope_ext(ctx0, ik, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                const auto * mctx_cur = inp_attn->mctx;
                ggml_build_forward_expand(gf, mctx_cur->cpy_k_idx(ctx0, ik, inp_attn->get_k_idxs(), il));
                ggml_tensor * ik_kv = mctx_cur->get_k_idx(ctx0, il);

                // Main branch: store K/V, take cache views
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
                ggml_tensor * k = mctx_cur->get_k(ctx0, il);
                ggml_tensor * v = mctx_cur->get_v(ctx0, il);
                GGML_ASSERT(!(v->nb[1] > v->nb[2]) && "MSA assumes v_trans=false (FA on)");

                const int64_t D   = k->ne[0];
                const int64_t HKV = k->ne[1];
                const int64_t Gp  = n_head/HKV;
                GGML_ASSERT(HKV == Hd && "MSA: one indexer head per GQA group");
                GGML_ASSERT(k->ne[3] == ns);
                const int K = mm.msa_p.topk_blocks < (int) nblk ? mm.msa_p.topk_blocks : (int) nblk;

                const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

                if (msa_decode) {
                    // decode: batched over streams top-k + gather, one grouped FA
                    // scores: per-stream batched matmul over the stream dim (ne[3]).
                    // the cache views are not contiguous across streams (stride = kv_size, not n_kv)
                    ggml_tensor * ikv4 = ggml_view_4d(ctx0, ik_kv, n_idx_dim, n_kv, 1, ns,
                            ik_kv->nb[2], ik_kv->nb[3], ik_kv->nb[3], 0);
                    ggml_tensor * iq4 = ggml_reshape_4d(ctx0, iq, n_idx_dim, Hd, 1, ns);
                    ggml_tensor * sc  = ggml_mul_mat(ctx0, ikv4, iq4);
                    ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
                    sc = ggml_add_inplace(ctx0, sc, msa_mf);
                    ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
                    cb(bs, "msa_bs", il);

                    ggml_tensor * bsf = ggml_add(ctx0, bs,
                            ggml_reshape_4d(ctx0, msa_loc->bias, nblk, 1, 1, ns));
                    ggml_tensor * idx = ggml_top_k(ctx0, bsf, K);

                    // token idx: tj[t,k,h,s] = blk*idx[k,h,s] + t   (for the mask gather)
                    // row   idx: tr[t,k,h,s] = tj*HKV + h           (for the per-stream K/V gather)
                    ggml_tensor * a = ggml_scale(ctx0, ggml_cast(ctx0, idx, GGML_TYPE_F32), (float) blk);
                    a = ggml_reshape_4d(ctx0, a, 1, K, Hd, ns);
                    ggml_tensor * tj = ggml_add(ctx0,
                            ggml_repeat_4d(ctx0, a, blk, K, Hd, ns),
                            ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.0f, (float) blk, 1.0f), blk, 1, 1));
                    ggml_tensor * tr = ggml_add(ctx0,
                            ggml_scale(ctx0, tj, (float) HKV),
                            ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.0f, (float) HKV, 1.0f), 1, 1, Hd));

                    ggml_tensor * tokj = ggml_cast(ctx0, ggml_reshape_2d(ctx0, tj, (int64_t) blk*K*Hd, ns), GGML_TYPE_I32);
                    ggml_tensor * tokr = ggml_cast(ctx0, ggml_reshape_2d(ctx0, tr, (int64_t) blk*K*Hd, ns), GGML_TYPE_I32);

                    ggml_tensor * k3 = ggml_view_3d(ctx0, k, D, HKV*n_kv, ns, k->nb[1], k->nb[3], 0);
                    ggml_tensor * v3 = ggml_view_3d(ctx0, v, D, HKV*n_kv, ns, v->nb[1], v->nb[3], 0);
                    ggml_tensor * m3 = ggml_reshape_3d(ctx0, msa_kqm, 1, n_kv, ns);

                    ggml_tensor * kg = ggml_get_rows(ctx0, k3, tokr);
                    ggml_tensor * vg = ggml_get_rows(ctx0, v3, tokr);
                    ggml_tensor * mg = ggml_get_rows(ctx0, m3, tokj);

                    // fold (group, stream) onto the FA channel dim
                    const ggml_type kt = ggml_is_quantized(k->type) ? GGML_TYPE_F16 : k->type;
                    const ggml_type vt = ggml_is_quantized(v->type) ? GGML_TYPE_F16 : v->type;
                    ggml_tensor * kfa = ggml_reshape_4d(ctx0, kg, D, (int64_t) blk*K, 1, Hd*ns);
                    ggml_tensor * vfa = ggml_reshape_4d(ctx0, vg, D, (int64_t) blk*K, 1, Hd*ns);
                    if (kfa->type != kt) { kfa = ggml_cast(ctx0, kfa, kt); }
                    if (vfa->type != vt) { vfa = ggml_cast(ctx0, vfa, vt); }
                    // the FA mask must be F16
                    ggml_tensor * mfa = ggml_cast(ctx0, ggml_reshape_4d(ctx0, mg, (int64_t) blk*K, 1, 1, Hd*ns), GGML_TYPE_F16);

                    cur = build_attn_msa_fa(Qcur, kfa, vfa, mfa, Gp, kq_scale, il);
                } else {
                    // batch: per-stream loop
                    std::vector<ggml_tensor *> outs(ns);
                    for (int64_t st = 0; st < ns; ++st) {
                        ggml_tensor * iq_s = ggml_view_3d(ctx0, iq, n_idx_dim, Hd, n_tps,
                                iq->nb[1], iq->nb[2], st*n_tps*iq->nb[2]);
                        ggml_tensor * ik_s = ggml_view_2d(ctx0, ik_kv, n_idx_dim, n_kv,
                                ik_kv->nb[2], st*ik_kv->nb[3]);
                        ggml_tensor * mf_s = ggml_view_3d(ctx0, msa_mf, n_kv, 1, n_tps,
                                msa_mf->nb[1], msa_mf->nb[1], st*msa_mf->nb[3]);
                        ggml_tensor * km_s = ggml_view_3d(ctx0, msa_kqm, n_kv, n_tps, 1,
                                msa_kqm->nb[1], msa_kqm->nb[3], st*msa_kqm->nb[3]);
                        ggml_tensor * bias_s = ggml_view_2d(ctx0, msa_loc->bias, nblk, n_tps,
                                msa_loc->bias->nb[1], st*n_tps*msa_loc->bias->nb[1]);
                        ggml_tensor * q_s = ggml_view_3d(ctx0, Qcur, D, n_head, n_tps,
                                Qcur->nb[1], Qcur->nb[2], st*n_tps*Qcur->nb[2]);
                        ggml_tensor * k_s = ggml_view_4d(ctx0, k, D, HKV, n_kv, 1,
                                k->nb[1], k->nb[2], k->nb[3], st*k->nb[3]);
                        ggml_tensor * v_s = ggml_view_4d(ctx0, v, D, HKV, n_kv, 1,
                                v->nb[1], v->nb[2], v->nb[3], st*v->nb[3]);

                        // block scores: bs = maxpool_blk(idx_q * idx_k^T + causal mask)
                        // scores are unscaled, only the top-k ordering matters
                        ggml_tensor * sc = ggml_mul_mat(ctx0, ik_s,
                                ggml_reshape_2d(ctx0, iq_s, n_idx_dim, Hd*n_tps));
                        // indexer scores run in F32
                        ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
                        sc = ggml_reshape_3d(ctx0, sc, n_kv, Hd, n_tps);
                        sc = ggml_add_inplace(ctx0, sc, mf_s);
                        ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
                        cb(bs, "msa_bs", il);

                        // block-level 0/-inf keep mask on the CPU, tiny transfer
                        ggml_tensor * srcs[2] = { bs, bias_s };
                        ggml_tensor * bm = ggml_custom_4d(ctx0, GGML_TYPE_F16,
                                nblk, n_tps, Hd, 1,
                                srcs, 2, msa_block_mask_op, GGML_N_TASKS_MAX,
                                const_cast<msa_params *>(&mm.msa_p));
                        cb(bm, "msa_block_mask", il);

                        // expand block -> token granularity on the GPU (j = bk*blk + t),
                        // then combine with the causal mask in place
                        ggml_tensor * bmx = ggml_repeat_4d(ctx0,
                                ggml_reshape_3d(ctx0, bm, 1, nblk, n_tps*Hd),
                                blk, nblk, n_tps*Hd, 1);
                        bmx = ggml_reshape_3d(ctx0, bmx, n_kv, n_tps, Hd);
                        ggml_tensor * mask4 = ggml_add_inplace(ctx0, bmx, km_s);
                        mask4 = ggml_reshape_4d(ctx0, mask4, n_kv, n_tps, 1, Hd);
                        cb(mask4, "msa_mask4", il);

                        // cache views with groups on ne[3];
                        ggml_tensor * kfa = ggml_permute(ctx0, k_s, 0, 3, 1, 2);
                        ggml_tensor * vfa = ggml_permute(ctx0, v_s, 0, 3, 1, 2);

                        outs[st] = build_attn_msa_fa(q_s, kfa, vfa, mask4, Gp, kq_scale, il);
                    }
                    cur = outs[0];
                    for (int64_t st = 1; st < ns; ++st) {
                        cur = ggml_concat(ctx0, cur, outs[st], 1);
                    }
                }

                cb(cur, "kqv_out", il);
                if (model.layers[il].wo) {
                    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
                }
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
