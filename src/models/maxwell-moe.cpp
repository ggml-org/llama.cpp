#include "models.h"
#include "hyper-connection.h"
#include "llama-kv-cache-msa.h"

// ---------------------------------------------------------------------------
// hyper-connection helpers (same math as DeepSeek-V4.1's mHC)
// ---------------------------------------------------------------------------

// per-graph MSA state: block selection happens in position space, K/V live in cell space
class llm_graph_input_maxwell_msa : public llm_graph_input_i {
public:
    llm_graph_input_maxwell_msa(const llama_kv_cache_msa_context * mctx, int blk, int local, int prefix) :
        mctx(mctx), blk(blk), local(local), prefix(prefix) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (pos_slot_i) { mctx->set_input_pos_slot(pos_slot_i, ubatch); }
        if (pos_slot_f) { mctx->set_input_pos_slot(pos_slot_f, ubatch); }
        if (cell_blk)   { mctx->set_input_cell_pos(cell_blk, ubatch, blk); }
        if (pos_mask)   { mctx->set_input_pos_mask(pos_mask, ubatch); }

        // force the fixed prefix blocks (positions 0..prefix*blk) and the local window into the top-k
        if (bias && ubatch->pos) {
            const int64_t n_tokens = ubatch->n_tokens;
            const int64_t nblk     = bias->ne[0];
            std::vector<float> data((size_t) nblk * n_tokens, 0.0f);
            for (int64_t i = 0; i < n_tokens; ++i) {
                const int64_t L = ubatch->pos[i] / blk;
                for (int b = 0; b < prefix && b <= L && b < nblk; ++b) {
                    data[(size_t) i * nblk + b] = 1e30f;
                }
                for (int l = 0; l < local && L - l >= 0; ++l) {
                    if (L - l < nblk) {
                        data[(size_t) i * nblk + (L - l)] = 1e30f;
                    }
                }
            }
            ggml_backend_tensor_set(bias, data.data(), 0, data.size() * sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_new = static_cast<const llama_kv_cache_msa_context *>(params.mctx);
        this->mctx = mctx_new;

        const int64_t n_ps = GGML_PAD((int64_t) mctx_new->get_n_pos(), blk);
        const int64_t ns   = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
        const bool decode  = params.ubatch.n_tokens == ns;

        bool res = true;
        res &= bias->ne[0] * blk == n_ps;
        res &= bias->ne[1]       == params.ubatch.n_tokens;
        res &= pos_mask->ne[0] == n_ps;
        res &= pos_mask->ne[1] == params.ubatch.n_tokens;
        res &= pos_slot_i->ne[0] == n_ps;
        res &= pos_slot_i->ne[1] == ns;
        res &= decode == (pos_slot_f != nullptr);
        res &= decode == (cell_blk   == nullptr);
        if (pos_slot_f) {
            res &= pos_slot_f->ne[0] == n_ps;
            res &= pos_slot_f->ne[1] == ns;
        }
        if (cell_blk) {
            res &= cell_blk->ne[0] == (int64_t) mctx_new->get_base()->get_n_kv();
            res &= cell_blk->ne[1] == ns;
        }
        return res;
    }

    ggml_tensor * bias       = nullptr; // F32 [nblk, n_tokens] forced-block bias
    ggml_tensor * pos_mask   = nullptr; // F32 [n_ps, n_tokens] 0/-inf visibility
    ggml_tensor * pos_slot_i = nullptr; // I32 [n_ps, ns]       pos -> cell
    ggml_tensor * pos_slot_f = nullptr; // F32 [n_ps, ns]       pos -> cell (decode)
    ggml_tensor * cell_blk   = nullptr; // I32 [n_kv, ns]       cell -> position block

    const llama_kv_cache_msa_context * mctx;

    int blk;
    int local;
    int prefix;
};

void llama_model_maxwell_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared,    false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    // the reference renormalizes the selected sigmoid scores; older GGUFs do not carry the key
    hparams.expert_weights_norm = true;
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_PREFIX_BLOCKS, hparams.indexer_prefix_blocks);
    // query positions below the dense prefix attend densely; older files do not carry the key
    hparams.indexer_dense_prefix_tokens = 16384;
    ml.get_key(LLM_KV_ATTENTION_INDEXER_DENSE_PREFIX_TOKENS, hparams.indexer_dense_prefix_tokens, false);

    // every backbone layer carries an indexer in this architecture; the metadata can override it per layer
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_indexer_full_impl[il] = 1;
    }
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);
    // the MTP layer(s) run dense attention and carry no indexer
    for (uint32_t il = (uint32_t) hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_indexer_full_impl[il] = 0;
    }

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters, false);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps,            false);

    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID) {
        throw std::runtime_error("maxwell-moe: only sigmoid expert gating is supported");
    }
    if (hparams.dsv4_hc_mult != 4) {
        throw std::runtime_error(format("maxwell-moe: expected 4 residual streams, got %u", hparams.dsv4_hc_mult));
    }

    // the MTP layer consumes the multi-stream backbone state
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_maxwell_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_q     = n_embd_head_k * n_head;
    const int64_t n_ff_exp     = hparams.n_ff_exp();
    const int64_t n_ff_shexp   = hparams.n_ff_shexp > 0 ? (int64_t) hparams.n_ff_shexp : n_ff_exp * hparams.n_expert_shared;
    const int64_t hc           = hparams.dsv4_hc_mult;
    const int64_t hc_dim       = hc*n_embd;
    const int64_t hc_mix_dim   = (2 + hc)*hc;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc},          0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1},           0);

    for (int i = 0; i < (int) hparams.n_layer_all; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        if (i < (int) n_layer && hparams.is_indexer_full(i)) {
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i),
                    {n_embd, (int64_t) hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i),
                    {n_embd, (int64_t) hparams.indexer_head_size}, 0);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i),
                    {(int64_t) hparams.indexer_head_size}, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i),
                    {(int64_t) hparams.indexer_head_size}, 0);
        }

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim},         0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3},                  0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim},         0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3},                  0);

        if (i >= (int) n_layer) {
            layer.nextn.eh_proj         = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,      "weight", i), {2*n_embd, n_embd}, 0);
            layer.nextn.enorm           = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,        "weight", i), {n_embd},           0);
            layer.nextn.hnorm           = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,        "weight", i), {n_embd},           0);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM,    "weight", i), {n_embd},           0);
            layer.nextn.hc_head_fn      = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_FN,   "weight", i), {hc_dim, hc},       0);
            layer.nextn.hc_head_base    = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_BASE, "weight", i), {hc},               0);
            layer.nextn.hc_head_scale   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_SCALE,"weight", i), {1},                0);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert},         0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_maxwell_moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------


ggml_tensor * llama_model_maxwell_moe::graph::build_attn_msa_fa(
        ggml_tensor * q_cur,   // [D, HQ, T]
        ggml_tensor * k,       // [D, n_keys, 1, C]
        ggml_tensor * v,       // [D, n_keys, 1, C]
        ggml_tensor * mask,    // [n_keys, R, 1, C] f16
        int64_t Gp, float kq_scale, int il) const {
    const int64_t D  = q_cur->ne[0];
    const int64_t HQ = q_cur->ne[1];
    const int64_t T  = q_cur->ne[2];
    const int64_t C  = k->ne[3];
    const int64_t R  = HQ*T/(Gp*C);
    GGML_ASSERT(Gp*C*R == HQ*T);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);

    ggml_tensor * q = ggml_reshape_4d(ctx0, q_cur, D, Gp, C, R);
    q = ggml_permute(ctx0, q, 0, 2, 3, 1);

    ggml_tensor * o = ggml_flash_attn_ext(ctx0, q, k, v, mask, kq_scale,
                                          hparams.f_max_alibi_bias, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    cb(o, "msa_fattn", il);

    o = ggml_permute(ctx0, o, 0, 1, 3, 2);
    if (!ggml_is_contiguous(o)) {
        o = ggml_cont(ctx0, o);
    }
    return ggml_reshape_2d(ctx0, o, D*HQ, T);
}

llama_model_maxwell_moe::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t hc          = hparams.dsv4_hc_mult;

    ggml_tensor * cur;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MSA needs flash attention and per-sequence cache streams
    const bool fa_on       = cparams.flash_attn;
    const bool streams_ok  = cparams.n_seq_max == 1 || !cparams.kv_unified;
    // the model attends dense up to the query-position prefix, block sparse past it
    const int64_t dense_prefix = hparams.indexer_dense_prefix_tokens;
    int64_t pos_min = INT64_MAX;
    if (params.ubatch.pos) {
        for (uint32_t i = 0; i < params.ubatch.n_tokens; ++i) {
            pos_min = std::min(pos_min, (int64_t) params.ubatch.pos[i]);
        }
    }
    const bool in_dense_prefix = pos_min < dense_prefix;
    // flash attention and per-sequence cache streams are prerequisites for the indexer; without them
    // the model attends dense and the indexer key cache does not exist
    const bool msa_available = fa_on && streams_ok;
    const bool sparse_mode   = msa_available && !in_dense_prefix;

    static bool warned_no_fa = false;
    if (!fa_on && !warned_no_fa) {
        LLAMA_LOG_WARN("%s: flash attention disabled; block sparse attention requires it -> DENSE attention\n", __func__);
        warned_no_fa = true;
    }
    static bool warned_unified = false;
    if (fa_on && !streams_ok && !warned_unified) {
        LLAMA_LOG_WARN("%s: unified KV cache with n_seq_max > 1 -> DENSE attention; use -np 1\n", __func__);
        warned_unified = true;
    }

    auto * inp_attn = build_attn_inp_kv_msa(msa_available);

    // hoisted per-graph MSA state
    llm_graph_input_maxwell_msa * msa = nullptr;
    ggml_tensor * msa_kqm = nullptr;
    ggml_tensor * msa_mf  = nullptr;
    int64_t n_kv = 0, n_ps = 0, nblk = 0, ns = 1, n_tps = 0;
    bool msa_decode = false;
    const int     blk    = (int) hparams.indexer_block_size;
    const int     local  = (int) hparams.indexer_local_blocks;
    const int     prefix = (int) hparams.indexer_prefix_blocks;
    const int64_t Hd     = hparams.indexer_n_head;

    if (sparse_mode) {
        const auto * mctx_msa = static_cast<const llama_kv_cache_msa_context *>(mctx);

        msa_kqm = inp_attn->self_attn_mask.raw;
        n_kv  = msa_kqm->ne[0];
        n_tps = msa_kqm->ne[1];
        ns    = msa_kqm->ne[3];
        GGML_ASSERT(msa_kqm->type == GGML_TYPE_F16);
        GGML_ASSERT(n_tps*ns == n_tokens);

        n_ps = GGML_PAD((int64_t) mctx_msa->get_n_pos(), blk);
        nblk = n_ps / blk;
        msa_decode = n_tps == 1;

        auto inp = std::make_unique<llm_graph_input_maxwell_msa>(mctx_msa, blk, local, prefix);

        inp->bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nblk, n_tokens);
        ggml_set_input(inp->bias);
        inp->pos_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ps, n_tokens);
        ggml_set_input(inp->pos_mask);
        inp->pos_slot_i = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_ps, ns);
        ggml_set_input(inp->pos_slot_i);

        if (msa_decode) {
            inp->pos_slot_f = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ps, ns);
            ggml_set_input(inp->pos_slot_f);
        } else {
            inp->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, ns);
            ggml_set_input(inp->cell_blk);
            msa_mf = ggml_cast(ctx0, msa_kqm, GGML_TYPE_F32);
        }

        msa = (llm_graph_input_maxwell_msa *) res->add_input(std::move(inp));
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // attention
        ggml_tensor * residual = inpL;

        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        cur = build_hc_pre(inpL, attn_mix.pre, il);
        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
            ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (!msa_available || !hparams.is_indexer_full(il)) {
                cur = build_attn(inp_attn, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur,
                        nullptr, nullptr, nullptr, kq_scale, il);
                cb(cur, "kqv_out", il);
            } else {
                const int64_t n_idx_dim = hparams.indexer_head_size;

                const auto * mctx_msa_l = static_cast<const llama_kv_cache_msa_context *>(mctx);
                const auto * mctx_cur = mctx_msa_l->get_base();
                const auto * mctx_idx = mctx_msa_l->get_idx();

                // the indexer key cache is written for every FA batch, dense or sparse; a later
                // sparse batch scores every cell, so dense batches must not leave holes
                ggml_tensor * ik = build_lora_mm(layer.index_k_proj, cur);
                ik = ggml_reshape_3d(ctx0, ik, n_idx_dim, 1, n_tokens);
                ik = build_norm(ik, layer.index_k_norm, NULL, LLM_NORM_RMS, il);
                ik = build_rope(ik, inp_pos, il);
                ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, ik, inp_attn->get_k_idxs_idx(), il));

                if (!sparse_mode) {
                    cur = build_attn(inp_attn, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur,
                            nullptr, nullptr, nullptr, kq_scale, il);
                    cb(cur, "kqv_out", il);
                } else {
                    ggml_tensor * iq = build_lora_mm(layer.index_q_proj, cur);
                    iq = ggml_reshape_3d(ctx0, iq, n_idx_dim, Hd, n_tokens);
                    iq = build_norm(iq, layer.index_q_norm, NULL, LLM_NORM_RMS, il);
                    iq = build_rope(iq, inp_pos, il);

                    ggml_tensor * ik_kv = mctx_idx->get_k(ctx0, il);

                    ggml_build_forward_expand(gf, Qcur);
                    ggml_build_forward_expand(gf, Kcur);
                    ggml_build_forward_expand(gf, Vcur);
                    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
                    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
                    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

                    const int64_t D   = k->ne[0];
                    const int64_t HKV = k->ne[1];
                    const int64_t Gp  = n_head/HKV;
                    GGML_ASSERT(HKV == Hd && "MSA: one indexer head per GQA group");
                    GGML_ASSERT(k->ne[3] == ns);
                    // fixed prefix + local blocks are forced into the top-k, the rest is remote
                    const int K = std::min<int>((int) (prefix + local + hparams.indexer_top_k), (int) nblk);

                    if (msa_decode) {
                        ggml_tensor * ik3 = ggml_view_3d(ctx0, ik_kv, n_idx_dim, n_kv, ns,
                                ik_kv->nb[2], ik_kv->nb[3], 0);
                        ggml_tensor * ikp = ggml_get_rows(ctx0, ik3, msa->pos_slot_i);
                        ggml_tensor * iq4 = ggml_reshape_4d(ctx0, iq, n_idx_dim, Hd, 1, ns);
                        ggml_tensor * sc  = ggml_mul_mat(ctx0,
                                ggml_reshape_4d(ctx0, ikp, n_idx_dim, n_ps, 1, ns), iq4);
                        ggml_prec_set_acc(sc, GGML_PREC_F32);
                        sc = ggml_add_inplace(ctx0, sc,
                                ggml_reshape_4d(ctx0, msa->pos_mask, n_ps, 1, 1, ns));
                        ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
                        ggml_tensor * bsf = ggml_add(ctx0, bs,
                                ggml_reshape_4d(ctx0, msa->bias, nblk, 1, 1, ns));
                        ggml_tensor * idx = ggml_top_k(ctx0, bsf, K);

                        ggml_tensor * a = ggml_scale(ctx0, ggml_cast(ctx0, idx, GGML_TYPE_F32), (float) blk);
                        a = ggml_reshape_4d(ctx0, a, 1, K, Hd, ns);
                        ggml_tensor * tj = ggml_add(ctx0,
                                ggml_repeat_4d(ctx0, a, blk, K, Hd, ns),
                                ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.0f, (float) blk, 1.0f), blk, 1, 1));
                        ggml_tensor * tokj = ggml_cast(ctx0,
                                ggml_reshape_2d(ctx0, tj, (int64_t) blk*K*Hd, ns), GGML_TYPE_I32);

                        ggml_tensor * cs = ggml_get_rows(ctx0,
                                ggml_reshape_3d(ctx0, msa->pos_slot_f, 1, n_ps, ns), tokj);
                        cs = ggml_reshape_4d(ctx0, cs, blk, K, Hd, ns);

                        ggml_tensor * tr = ggml_add(ctx0,
                                ggml_scale(ctx0, cs, (float) HKV),
                                ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.0f, (float) HKV, 1.0f), 1, 1, Hd));
                        ggml_tensor * tokr = ggml_cast(ctx0,
                                ggml_reshape_2d(ctx0, tr, (int64_t) blk*K*Hd, ns), GGML_TYPE_I32);

                        ggml_tensor * k3 = ggml_view_3d(ctx0, k, D, HKV*n_kv, ns, k->nb[1], k->nb[3], 0);
                        ggml_tensor * v3 = ggml_view_3d(ctx0, v, D, HKV*n_kv, ns, v->nb[1], v->nb[3], 0);
                        ggml_tensor * mp = ggml_reshape_3d(ctx0, msa->pos_mask, 1, n_ps, ns);

                        ggml_tensor * kg = ggml_get_rows(ctx0, k3, tokr);
                        ggml_tensor * vg = ggml_get_rows(ctx0, v3, tokr);
                        ggml_tensor * mg = ggml_get_rows(ctx0, mp, tokj);

                        const ggml_type kt = ggml_is_quantized(k->type) ? GGML_TYPE_F16 : k->type;
                        const ggml_type vt = ggml_is_quantized(v->type) ? GGML_TYPE_F16 : v->type;
                        ggml_tensor * kfa = ggml_reshape_4d(ctx0, kg, D, (int64_t) blk*K, 1, Hd*ns);
                        ggml_tensor * vfa = ggml_reshape_4d(ctx0, vg, D, (int64_t) blk*K, 1, Hd*ns);
                        if (kfa->type != kt) { kfa = ggml_cast(ctx0, kfa, kt); }
                        if (vfa->type != vt) { vfa = ggml_cast(ctx0, vfa, vt); }
                        ggml_tensor * mfa = ggml_cast(ctx0,
                                ggml_reshape_4d(ctx0, mg, (int64_t) blk*K, 1, 1, Hd*ns), GGML_TYPE_F16);

                        cur = build_attn_msa_fa(Qcur, kfa, vfa, mfa, Gp, kq_scale, il);
                    } else {
                        // batch: per-stream loop (mask based selection)
                        std::vector<ggml_tensor *> outs(ns);
                        for (int64_t st = 0; st < ns; ++st) {
                            ggml_tensor * iq_s = ggml_view_3d(ctx0, iq, n_idx_dim, Hd, n_tps,
                                    iq->nb[1], iq->nb[2], st*n_tps*iq->nb[2]);
                            ggml_tensor * ik_s = ggml_view_2d(ctx0, ik_kv, n_idx_dim, n_kv,
                                    ik_kv->nb[2], st*ik_kv->nb[3]);
                            ggml_tensor * psl_s = ggml_view_1d(ctx0, msa->pos_slot_i, n_ps,
                                    st*msa->pos_slot_i->nb[1]);
                            ggml_tensor * pm_s = ggml_view_3d(ctx0, msa->pos_mask, n_ps, 1, n_tps,
                                    msa->pos_mask->nb[1], msa->pos_mask->nb[1], st*n_tps*msa->pos_mask->nb[1]);
                            ggml_tensor * cb_s = ggml_view_1d(ctx0, msa->cell_blk, n_kv,
                                    st*msa->cell_blk->nb[1]);
                            ggml_tensor * mf_s = ggml_view_3d(ctx0, msa_mf, n_kv, n_tps, 1,
                                    msa_mf->nb[1], msa_mf->nb[3], st*msa_mf->nb[3]);
                            ggml_tensor * bias_s = ggml_view_3d(ctx0, msa->bias, nblk, 1, n_tps,
                                    msa->bias->nb[1], msa->bias->nb[1], st*n_tps*msa->bias->nb[1]);
                            ggml_tensor * q_s = ggml_view_3d(ctx0, Qcur, D, n_head, n_tps,
                                    Qcur->nb[1], Qcur->nb[2], st*n_tps*Qcur->nb[2]);

                            ggml_tensor * ikp = ggml_get_rows(ctx0, ik_s, psl_s);   // [n_idx_dim, n_ps]
                            ggml_tensor * sc = ggml_mul_mat(ctx0,
                                    ggml_reshape_3d(ctx0, ikp, n_idx_dim, n_ps, 1),
                                    ggml_reshape_3d(ctx0, iq_s, n_idx_dim, Hd*n_tps, 1));
                            ggml_prec_set_acc(sc, GGML_PREC_F32);
                            sc = ggml_reshape_3d(ctx0, sc, n_ps, Hd, n_tps);
                            sc = ggml_add_inplace(ctx0, sc, pm_s);
                            ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
                            ggml_tensor * bsf = ggml_add(ctx0, bs, bias_s);   // [nblk, Hd, n_tps]
                            ggml_tensor * idx = ggml_top_k(ctx0, bsf, K);

                            ggml_tensor * ninf = ggml_cast(ctx0,
                                    ggml_scale_bias(ctx0, bias_s, 0.0f, -1e30f), GGML_TYPE_F16);
                            ninf = ggml_repeat_4d(ctx0, ninf, nblk, Hd, n_tps, 1);
                            ggml_tensor * zero = ggml_scale(ctx0,
                                    ggml_cast(ctx0, idx, GGML_TYPE_F32), 0.0f);
                            ggml_tensor * bm = ggml_set_rows(ctx0,
                                    ggml_reshape_3d(ctx0, ninf, 1, nblk, Hd*n_tps),
                                    ggml_reshape_3d(ctx0, zero, 1, K,    Hd*n_tps),
                                    ggml_reshape_2d(ctx0, idx,     K,    Hd*n_tps));
                            bm = ggml_reshape_3d(ctx0, bm, nblk, Hd, n_tps);
                            bm = ggml_cont(ctx0, ggml_permute(ctx0, bm, 0, 2, 1, 3)); // [nblk, n_tps, Hd]

                            ggml_tensor * bm2 = ggml_cont(ctx0, ggml_transpose(ctx0,
                                    ggml_reshape_2d(ctx0, bm, nblk, n_tps*Hd)));
                            ggml_tensor * bmc = ggml_get_rows(ctx0, bm2, cb_s);
                            ggml_tensor * bmx = ggml_cont(ctx0, ggml_transpose(ctx0, bmc));
                            bmx = ggml_reshape_3d(ctx0, bmx, n_kv, n_tps, Hd);
                            ggml_tensor * mask4 = ggml_add_inplace(ctx0, bmx, mf_s);
                            mask4 = ggml_cast(ctx0,
                                    ggml_reshape_4d(ctx0, mask4, n_kv, n_tps, 1, Hd), GGML_TYPE_F16);

                            ggml_tensor * k_s = ggml_view_4d(ctx0, k, D, HKV, n_kv, 1,
                                    k->nb[1], k->nb[2], k->nb[3], st*k->nb[3]);
                            ggml_tensor * v_s = ggml_view_4d(ctx0, v, D, HKV, n_kv, 1,
                                    v->nb[1], v->nb[2], v->nb[3], st*v->nb[3]);
                            ggml_tensor * kfa = ggml_permute(ctx0, k_s, 0, 3, 1, 2);
                            ggml_tensor * vfa = ggml_permute(ctx0, v_s, 0, 3, 1, 2);

                            outs[st] = build_attn_msa_fa(q_s, kfa, vfa, mask4, Gp, kq_scale, il);
                        }
                        cur = outs[0];
                        for (int64_t st = 1; st < ns; ++st) {
                            cur = ggml_concat(ctx0, cur, outs[st], 1);
                        }
                    }
                    cb(cur, "kqv_msa_out", il);

                    // output projection (the dense path does this inside build_attn)
                    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
                    cb(cur, "kqv_out", il);
                }
            }
        }

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        // feed-forward
        residual = inpL;

        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        cur = build_hc_pre(inpL, ffn_mix.pre, il);
        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (il < (int) hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il);

            ggml_tensor * shexp_out = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);

            cur = ggml_add(ctx0, moe_out, shexp_out);
            cb(cur, "ffn_moe_out", il);
        }

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "hc_ffn_post", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    {
        ggml_tensor * flat     = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
        res->t_h_nextn = cparams.embeddings_nextn_masked ? flat_out : flat;
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        cb(cur, "result_norm_rows", -1);
    }

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// ---------------------------------------------------------------------------
// MTP (NextN) graph
// ---------------------------------------------------------------------------

llama_model_maxwell_moe::graph_mtp::graph_mtp(const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "maxwell-moe MTP requires one NextN layer");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn);

    const int     il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto  & layer = model.layers[il];
    const int64_t hc = hparams.dsv4_hc_mult;

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd_out());
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);
    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd_out());

    ggml_tensor * tok_embd = ubatch.token ? ggml_get_rows(ctx0, model.tok_embd, inp->tokens) : inp->embd;
    ggml_tensor * e_norm   = build_norm(tok_embd, layer.nextn.enorm, NULL, LLM_NORM_RMS, il);
    ggml_tensor * h_norm   = build_norm(ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens),
                                        layer.nextn.hnorm, NULL, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_h_norm", il);

    // the reference adds e_proj(embed) to h_proj of every residual stream: one [2*n_embd, n_embd] matmul
    ggml_tensor * e_rep = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    ggml_tensor * inpL  = ggml_mul_mat(ctx0, layer.nextn.eh_proj, ggml_concat(ctx0, e_rep, h_norm, 0));
    cb(inpL, "mtp_eh_proj", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto       * inp_attn     = build_attn_inp_kv();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_v));

    ggml_tensor * residual = inpL;

    const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
    ggml_tensor * cur = build_hc_pre(inpL, attn_mix.pre, il);
    cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);

    {
        ggml_tensor * Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_v, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_v, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);

        // the MTP block runs dense attention; no indexer
        cur = build_attn(inp_attn, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur,
                nullptr, nullptr, nullptr, kq_scale, il);
    }

    inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);

    residual = inpL;
    const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
    cur = build_hc_pre(inpL, ffn_mix.pre, il);
    cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);

    {
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il);

        ggml_tensor * shexp_out = build_ffn(cur,
                layer.ffn_up_shexp,   NULL, NULL,
                layer.ffn_gate_shexp, NULL, NULL,
                layer.ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);

        cur = ggml_add(ctx0, moe_out, shexp_out);
    }

    inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);

    cur = build_hc_head(inpL, layer.nextn.hc_head_fn, layer.nextn.hc_head_scale, layer.nextn.hc_head_base);
    cur = build_norm(cur, layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, il);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
