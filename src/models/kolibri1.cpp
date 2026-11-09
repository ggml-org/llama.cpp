#include "models.h"

#include "llama-kv-cache-iswa.h"

// Kolibri-1.1 replaces the full attention of Kolibri-1 by sparse attention. A query routes over the
// first KV head with the 12 queries of its group. The score of a key is the mean over these 12 heads:
// ReLU(q*k)/sqrt(n_embd_head), with q and k rounded to NVFP4. It then attends to the top_k best keys,
// its first N_SINK and its last N_RECENT keys. Indexers choose the pool of keys to select from:
// the whole context, or for contexts longer than the pool, the best blocks of consecutive keys,
// scored by the mean of their keys. Reindexers select from the pool of the preceding indexer.
// The sparse layers attend over K/V rounded to NVFP4.

// the leading and the most recent keys every sparse query attends to
static constexpr int32_t KOLIBRI_N_SINK   = 8;
static constexpr int32_t KOLIBRI_N_RECENT = 128;

void llama_model_kolibri1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);

    if (hparams.n_swa == 0 || hparams.n_expert_shared != 1 || hparams.n_ff_shexp == 0 ||
            hparams.n_layer_dense_lead >= hparams.n_layer() || (hparams.n_layer_dense_lead > 0 && hparams.n_ff(0) == 0)) {
        throw std::runtime_error("Kolibri-1: invalid sliding window, shared expert or dense prefix configuration");
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    hparams.rope_pattern = hparams.is_swa_impl;

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    // Kolibri-1.1: sparse attention in the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k, false);
    if (hparams.indexer_top_k > 0) {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   n_pool_block);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, n_pool_blocks);
        ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES,  hparams.is_indexer_full_impl, hparams.n_layer());
        ml.get_key_or_arr(LLM_KV_ATTENTION_KV_SOURCE_MAP,  hparams.kv_source_impl,       hparams.n_layer());

        if (n_pool_block == 0 || n_pool_blocks == 0 || hparams.n_embd_head_k() % 16 != 0 ||
                hparams.n_embd_head_k() != hparams.n_embd_head_v()) {
            throw std::runtime_error("Kolibri-1.1: invalid sparse attention configuration");
        }

        bool has_indexer = false;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            const uint32_t il_src = hparams.kv_source(il);
            const bool     swa    = hparams.is_swa(il);
            // reindexers need a preceding indexer, K/V come from an earlier full-attention layer
            if ((swa && (hparams.is_indexer_full(il) || il_src != il)) ||
                (!swa && !hparams.is_indexer_full(il) && !has_indexer) ||
                (!swa && (il_src > il || hparams.is_swa(il_src) || !hparams.has_kv(il_src)))) {
                throw std::runtime_error(format("Kolibri-1.1: invalid indexer or KV source for layer %u", il));
            }
            has_indexer |= !swa && hparams.is_indexer_full(il);
        }
    }

    switch (hparams.n_layer()) {
        case 50: type = LLM_TYPE_78B_A3_5B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kolibri1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        // layers reading the K/V of another layer have no K/V projections
        if (hparams.has_kv(i)) {
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // the leading layers of a dense prefix have a dense FFN instead of the MoE
        if ((uint32_t) i < hparams.n_layer_dense_lead) {
            const int64_t n_ff = hparams.n_ff(i);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            continue;
        }

        // Routing is sigmoid, the selection bias is added to the logits before top-k.
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
    }
}

// per-ubatch selection inputs of the sparse layers, from the positions of the cells of the base cache
class llama_model_kolibri1::llm_graph_input_sparse : public llm_graph_input_i {
public:
    llm_graph_input_sparse(const llama_kv_cache_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_sparse() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * cand      = nullptr; // F16 [n_kv, n_tokens]   0 for the keys a query selects from, else -inf
    ggml_tensor * always    = nullptr; // I32 [N_SINK + N_RECENT, n_tokens] cells every query attends to, -1 if none
    ggml_tensor * blk_cells = nullptr; // I32 [n_pool_block, n_blk] cells of each block of positions, nullptr without pool
    ggml_tensor * blk_mask  = nullptr; // F16 [n_blk, n_tokens]  0 for the complete blocks of a query, else -inf

    const llama_kv_cache_context * mctx;
};

void llama_model_kolibri1::llm_graph_input_sparse::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = cand->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);

    // indexer rules are stated in sequence positions, so we need a mapping to 
    // translate between cells and positions within the sequences of the ubatch
    std::map<llama_seq_id, std::vector<int32_t>> cell_to_pos;
    std::map<llama_seq_id, std::vector<int32_t>> pos_to_cell;
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        auto & pc = pos_to_cell[s];
        pc.resize(std::max<size_t>(pc.size(), ubatch->pos[i] + 1), -1);
    }
    for (auto & [s, pc] : pos_to_cell) {
        const auto & cells = mctx->get_cells(s);
        auto & cp = cell_to_pos[s];
        cp.assign(n_kv, -1);
        for (int64_t c = 0; c < n_kv; ++c) {
            if (!cells.is_empty(c) && cells.seq_has(c, s)) {
                const llama_pos p = cells.pos_get(c);
                cp[c] = p;
                if ((size_t) p < pc.size()) {
                    pc[p] = c;
                }
            }
        }
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(cand->buffer) && ggml_backend_buffer_is_host(always->buffer));
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & cp = cell_to_pos[s];
        const auto & pc = pos_to_cell[s];
        ggml_fp16_t * row = (ggml_fp16_t *) cand->data + i*n_kv;
        // the candidates of a query at position p are the positions [N_SINK, p - N_RECENT] of its sequence;
        // the sink and recent tokens are always attended to (see `always`), so the indexer never selects them
        if (i > 0 && ubatch->seq_id[i - 1][0] == s && ubatch->pos[i - 1] == p - 1) {
            // fast path: copy the row of the previous position and add the one position that left its recent window
            std::copy(row - n_kv, row, row);
            const llama_pos pk = p - KOLIBRI_N_RECENT;
            if (pk >= KOLIBRI_N_SINK && pc[pk] >= 0) {
                row[pc[pk]] = keep;
            }
        } else {
            // slow path: initialize the row based on the cell-to-position mapping
            for (int64_t c = 0; c < n_kv; ++c) {
                row[c] = cp[c] >= KOLIBRI_N_SINK && cp[c] <= p - KOLIBRI_N_RECENT ? keep : drop;
            }
        }
        int32_t * dst = (int32_t *) always->data + i*always->ne[0];
        for (int32_t j = 0; j < KOLIBRI_N_SINK; ++j) {
            dst[j] = j <= p ? pc[j] : -1;
        }
        for (int32_t j = 0; j < KOLIBRI_N_RECENT; ++j) {
            const llama_pos pj = p - KOLIBRI_N_RECENT + 1 + j;
            dst[KOLIBRI_N_SINK + j] = pj >= 0 ? pc[pj] : -1;
        }
    }

    if (blk_cells && blk_cells->buffer) {
        const int64_t bs    = blk_cells->ne[0];
        const int64_t n_blk = blk_cells->ne[1];

        // the complete blocks of each sequence, the padding blocks and the blocks with a missing position are never live
        GGML_ASSERT(ggml_backend_buffer_is_host(blk_cells->buffer) && ggml_backend_buffer_is_host(blk_mask->buffer));
        int32_t * cells = (int32_t *) blk_cells->data;
        std::fill(cells, cells + bs*n_blk, 0);
        std::vector<llama_seq_id> blk_seq(n_blk, -1);
        std::vector<llama_pos>    blk_end(n_blk, 0);
        int64_t b = 0;
        for (const auto & [s, pc] : pos_to_cell) {
            // block j of a sequence covers positions [j*bs, (j+1)*bs - 1] of the sequence
            // n_blk bounds the blocks of the sequences that do not share cells
            for (int64_t j = 0; j < (int64_t) pc.size()/bs && b < n_blk; ++j, ++b) {
                bool complete = true;
                for (int64_t k = 0; k < bs; ++k) {
                    const int32_t c = pc[j*bs + k];
                    complete &= c >= 0;
                    cells[b*bs + k] = std::max(c, 0);
                }
                blk_seq[b] = complete ? s : -1;
                blk_end[b] = (j + 1)*bs - 1;
            }
        }

        // update the block mask based on the sequence and block end positions
        ggml_fp16_t * data = (ggml_fp16_t *) blk_mask->data;
        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];
            for (int64_t j = 0; j < n_blk; ++j) {
                data[i*n_blk + j] = blk_seq[j] == s && blk_end[j] <= p ? keep : drop;
            }
        }
    }
}

bool llama_model_kolibri1::llm_graph_input_sparse::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx)->get_base();

    // the pool inputs exist for a fixed n_kv only
    return cand->ne[0] == mctx->get_n_kv() && cand->ne[1] == params.ubatch.n_tokens;
}

ggml_tensor * llama_model_kolibri1::graph::build_attn_sparse(
        llm_graph_input_attn_kv_iswa * inp_attn,
        llm_graph_input_sparse * inp_sparse,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * & pool,
                float   kq_scale,
                  int   il) const {
    const auto & mk = static_cast<const llama_model_kolibri1 &>(model);
    const auto * mctx_cur = inp_attn->mctx->get_base();

    const int64_t n_embd_head = q_cur->ne[0];
    const int64_t n_route     = n_head/n_head_kv; // the query heads of the first KV head

    if (k_cur) {
        k_cur = ggml_round_nvfp4(ctx0, k_cur);
        v_cur = ggml_round_nvfp4(ctx0, v_cur);
        cb(k_cur, "Kcur_nvfp4", il);
        cb(v_cur, "Vcur_nvfp4", il);

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs(), il));
    }

    ggml_tensor * k = mctx_cur->get_k(ctx0, il); // [n_embd_head, n_head_kv, n_kv, 1]
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(k->ne[3] == 1 && "Kolibri-1.1 sparse attention requires a single KV stream");

    const int64_t n_kv = k->ne[2];

    // the cache holds the K/V rounded to NVFP4, so the first KV head is the routing key
    ggml_tensor * k_route = ggml_view_4d(ctx0, k, n_embd_head, 1, n_kv, 1, k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * q_route = ggml_view_3d(ctx0, q_cur, n_embd_head, n_route, n_tokens, q_cur->nb[1], q_cur->nb[2], 0);
    q_route = ggml_round_nvfp4(ctx0, q_route);
    cb(q_route, "q_route", il);

    // the mean of the scores of the routing heads
    ggml_tensor * w_route = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_route, n_tokens);
    w_route = ggml_fill(ctx0, w_route, 1.0f/(n_route*sqrtf((float) n_embd_head)));

    if (hparams.is_indexer_full(il)) {
        pool = inp_sparse->cand;

        // the pool of a long context: the keys of the best blocks
        if (ggml_tensor * blk_cells = inp_sparse->blk_cells) {
            const int64_t bs    = blk_cells->ne[0];
            const int64_t n_blk = blk_cells->ne[1];

            // XXX: training averages the routing keys before their NVFP4 rounding, the cache holds the rounded keys only
            //      evals of inference with round(mean(round(keys))) show negligible differences though

            // the keys of the blocks are gathered from the cache and averaged to form the block representation
            // gather the keys member by member, so that the members of the blocks form contiguous slabs
            ggml_tensor * k_route_2d = ggml_view_2d(ctx0, k, n_embd_head, n_kv, k->nb[2], 0);
            ggml_tensor * members = ggml_cont(ctx0, ggml_transpose(ctx0, blk_cells)); // [n_blk, bs]
            ggml_tensor * k_mem = ggml_get_rows(ctx0, k_route_2d, ggml_reshape_1d(ctx0, members, bs*n_blk));
            ggml_tensor * k_blk = nullptr;
            for (int64_t m = 0; m < bs; ++m) {
                ggml_tensor * slab = ggml_view_2d(ctx0, k_mem, n_embd_head, n_blk, k_mem->nb[1], m*n_blk*k_mem->nb[1]);
                k_blk = k_blk ? ggml_add(ctx0, k_blk, slab) : slab;
            }
            k_blk = ggml_round_nvfp4(ctx0, ggml_scale(ctx0, k_blk, 1.0f/bs));
            k_blk = ggml_reshape_4d(ctx0, k_blk, n_embd_head, 1, n_blk, 1);
            cb(k_blk, "k_pool_blocks", il);

            ggml_tensor * blk_score = ggml_lightning_indexer(ctx0, q_route, k_blk, w_route, inp_sparse->blk_mask);
            cb(blk_score, "pool_block_score", il);

            ggml_tensor * blk_top = ggml_top_k(ctx0, blk_score, std::min<int64_t>(mk.n_pool_blocks, n_blk));
            pool = ggml_top_k_mask(ctx0, inp_sparse->cand, blk_top, blk_score, blk_cells, nullptr);
            cb(pool, "pool", il);
        }
    }
    GGML_ASSERT(pool != nullptr);

    ggml_tensor * score = ggml_lightning_indexer(ctx0, q_route, k_route, w_route, pool); // [n_kv, n_tokens]
    cb(score, "route_score", il);

    ggml_tensor * top_k = ggml_top_k(ctx0, score, std::min<int64_t>(hparams.indexer_top_k, n_kv));
    cb(top_k, "route_top_k", il);

    // the selected keys and the keys every query attends to, the selection and the inputs are causal
    ggml_tensor * keep = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, 1), 0.0f);
    ggml_tensor * kq_mask = ggml_top_k_mask(ctx0, keep, top_k, score, nullptr, inp_sparse->always);
    kq_mask = ggml_reshape_4d(ctx0, kq_mask, n_kv, n_tokens, 1, 1);
    cb(kq_mask, "kq_mask_sparse", il);

    const int64_t n_attended = hparams.indexer_top_k + KOLIBRI_N_SINK + KOLIBRI_N_RECENT;

    ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, kq_mask, nullptr, nullptr, nullptr, n_attended, kq_scale, il);
    cb(cur, "kqv_out", il);

    return build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
}

llama_model_kolibri1::graph::graph(const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    // Kolibri 1.1 with sparse attention
    llm_graph_input_sparse * inp_sparse = nullptr;
    if (hparams.indexer_top_k > 0) {
        const auto & mk = static_cast<const llama_model_kolibri1 &>(model);
        const auto * mctx_base = inp_attn->mctx->get_base();
        const int64_t n_kv = mctx_base->get_n_kv();

        auto inp = std::make_unique<llm_graph_input_sparse>(mctx_base);

        inp->cand = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, n_tokens);
        ggml_set_input(inp->cand);

        inp->always = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KOLIBRI_N_SINK + KOLIBRI_N_RECENT, n_tokens);
        ggml_set_input(inp->always);

        // the pool is the whole context unless a query can see more keys than the pool holds
        if (n_kv > (int64_t) mk.n_pool_block*mk.n_pool_blocks) {
            const int64_t n_blk = n_kv/mk.n_pool_block;

            inp->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mk.n_pool_block, n_blk);
            ggml_set_input(inp->blk_cells);

            inp->blk_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_blk, n_tokens);
            ggml_set_input(inp->blk_mask);
        }

        inp_sparse = (llm_graph_input_sparse *) res->add_input(std::move(inp));
    }

    // key pool of the closest preceding indexer
    ggml_tensor * pool = nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // QK-norm, then RoPE on the sliding layers only: the global layers are NoPE.
        // Layers reading the K/V of another layer project only their queries.
        ggml_tensor * Qcur;
        ggml_tensor * Kcur = nullptr;
        ggml_tensor * Vcur = nullptr;
        if (hparams.has_kv(il)) {
            auto qkv = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);
            Qcur = qkv.q;
            Vcur = qkv.v;

            Kcur = build_norm(qkv.k, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = build_rope(Kcur, inp_pos, il);
            cb(Kcur, "Kcur_rope", il);
        } else {
            Qcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, cur, layer.wq_s), n_embd_head, n_head, n_tokens);
        }

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Qcur = build_rope(Qcur, inp_pos, il);
        cb(Qcur, "Qcur_rope", il);

        if (inp_sparse && !hparams.is_swa(il)) {
            cur = build_attn_sparse(inp_attn, inp_sparse, Qcur, Kcur, Vcur, pool, kq_scale, il);
        } else {
            cur = build_attn(inp_attn,
                    layer.wo, NULL, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
        cb(cur, "attn_out", il);

        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr, nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // The selection adds the bias to the logits, the weights are the sigmoid of the unbiased logits, so the same logits feed both.
            ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
            cb(logits, "ffn_moe_logits", il);

            ggml_tensor * selection = ggml_add(ctx0, logits, layer.ffn_exp_probs_b);
            cb(selection, "ffn_moe_logits_biased", il);

            ggml_tensor * selected = ggml_argsort_top_k(ctx0, selection, n_expert_used);
            cb(selected, "ffn_moe_topk", il);

            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, 1.0f,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    logits,
                    nullptr, nullptr, nullptr, nullptr, selected);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * shared = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr, nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(shared, "ffn_shexp_out", il);

            cur = ggml_add(ctx0, moe_out, shared);
        }

        cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        inpL = build_cvec(cur, il);
        cb(inpL, "l_out", il);
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_kolibri1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}
