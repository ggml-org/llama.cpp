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

static inline void msa_fill_mask(
        float * dst, int64_t dst_skey, int64_t dst_squery,
        const float * iq, const float * ik, const int32_t * pos,
        int Hd, int S, int D, int64_t n_kv,
        int blk, int topk_blocks, int local,
        int ith, int nth, int64_t key_len ) {

    const float NEG = -INFINITY, POS = INFINITY;
    const int nblk = (n_kv + blk - 1)/blk;   
    const int topk = topk_blocks < nblk ? topk_blocks : nblk;

    std::vector<float> bs(nblk);
    std::vector<int>   ord(nblk);

    for (int i = ith; i < S; i += nth) {                 // split queries across threads
        const int pi = pos[i];
        const float * q = iq + (size_t) i * Hd * D;      // [Hd, D] for this query

        // block scores: max over heads and over keys-in-block, future keys excluded
        for (int bk = 0; bk < nblk; ++bk) {
            float m = NEG;
            const int j0 = bk * blk;
            if (j0 >= (int) key_len) { bs[bk] = NEG; continue; }
            const int j1 = std::min((bk + 1) * blk, (int) key_len);
            for (int j = j0; j < j1; ++j) {
                if (j > pi) break;                       // keys in order; rest of block is future
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

        // force the local block(s)
        const int qb = pi / blk;
        for (int l = 0; l < local; ++l) {
            int bk = qb - l; if (bk < 0) bk = 0;
            bs[bk] = POS;
        }

        // top-k blocks, descending
        for (int t = 0; t < nblk; ++t) ord[t] = t;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return bs[a] > bs[b]; });

        // default everything masked, then open the selected blocks
        float * col = dst + (int64_t) i * dst_squery;
        for (int64_t j = 0; j < n_kv; ++j) col[j * dst_skey] = NEG;
        for (int t = 0; t < topk; ++t) {
            const int bk = ord[t];
            if (bs[bk] == NEG) break;                    // fewer real blocks than topk
            const int j0 = bk * blk;
            if (j0 >= (int) key_len) continue;
            const int j1 = std::min((bk + 1) * blk, (int) key_len);
            for (int j = j0; j < j1; ++j)
                if (j <= pi) col[(int64_t) j * dst_skey] = 0.f;
        }
    }
}

static void msa_block_mask_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ggml_tensor * iq  = dst->src[0];   // [D, Hd, S]
    const struct ggml_tensor * ik  = dst->src[1];   // [D, 1,  S]
    const struct ggml_tensor * pos = dst->src[2];   // [S] i32
    const msa_params * p = (const msa_params *) userdata;

    const int     D    = iq->ne[0];
    const int     Hd   = iq->ne[1];
    const int     S    = iq->ne[2];
    const int64_t n_kv = dst->ne[0];

    GGML_ASSERT(ik->ne[0] == D && ik->ne[1] == 1);   // ne[2] is n_kv (history), not S
    GGML_ASSERT(ggml_is_contiguous(ik));             // raw j*D indexing needs contiguity
    GGML_ASSERT(dst->ne[1] == (int64_t) S);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[1] == iq->ne[2]);   // queries axis must match (this is the suspect)
    const int64_t key_len = n_kv;      // keys actually present this batch
    if (ith == 0) {
        fprintf(stderr, "msa: dst ne=[%lld,%lld,%lld,%lld] iq ne=[%lld,%lld,%lld] n_kv(dst.ne0)=%lld\n",
                (long long)dst->ne[0],(long long)dst->ne[1],(long long)dst->ne[2],(long long)dst->ne[3],
                (long long)iq->ne[0],(long long)iq->ne[1],(long long)iq->ne[2],
                (long long)dst->ne[0]);
    }

    msa_fill_mask((float *) dst->data, /*dst_skey=*/1, /*dst_squery=*/dst->ne[0],
                  (const float *) iq->data, (const float *) ik->data, (const int32_t *) pos->data,
                  Hd, S, D, n_kv, p->blk, p->topk_blocks, p->local, ith, nth, key_len);
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
            ggml_tensor * kq_b = nullptr;
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

                const int64_t n_kv = inp_attn->get_kq_mask()->ne[0];
                ggml_tensor * srcs[3] = { iq, ik_kv, inp_pos };
                kq_b = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1, 1,
                                      srcs, 3, msa_block_mask_op, GGML_N_TASKS_MAX, const_cast<msa_params *>(&mm.msa_p));
                cb(kq_b, "msa_kq_b", il);
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
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
