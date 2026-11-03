#include "models.h"

#include "llama-impl.h"

void llama_model_dflash::load_tensors_dflash2() {
    LLAMA_LOAD_LOCALS;

    const int64_t rank = hparams.dflash_selector_rank;
    if (rank <= 0 || hparams.dflash_block_size <= 0 || hparams.dflash_selector_top_k <= 0 ||
            hparams.dflash_conv_kernel_size <= 0 || hparams.dflash_conv_group_size <= 0) {
        throw std::runtime_error("DFlash2 model is missing conv/selector metadata");
    }
    if (n_embd < hparams.dflash_selector_top_k * (hparams.dflash_selector_top_k + 1)) {
        throw std::runtime_error("DFlash2 hidden size is too small for the selector lattice");
    }

    dflash_selector_prev   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_PREV,   "weight"), { rank, n_vocab }, 0);
    dflash_selector_next   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_NEXT,   "weight"), { rank, n_vocab }, 0);
    dflash_selector_hidden = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_HIDDEN, "weight"), { n_embd, rank }, 0);

    load_tensors_dflash2_conv();

    LLAMA_LOG_INFO("%s: DFlash2 conv kernel = %u, group = %u, selector rank = %u, top-k = %u\n", __func__,
            hparams.dflash_conv_kernel_size, hparams.dflash_conv_group_size,
            hparams.dflash_selector_rank, hparams.dflash_selector_top_k);
}

void llama_model_dflash::load_tensors_dflash2_conv() {
    LLAMA_LOAD_LOCALS;

    const bool conv_meta = hparams.dflash_conv_kernel_size > 0 && hparams.dflash_conv_group_size > 0;
    if (!conv_meta) {
        return;
    }
    if (n_embd % hparams.dflash_conv_group_size != 0) {
        throw std::runtime_error("DFlash hidden size must be divisible by conv_group_size");
    }

    const int64_t kernel = hparams.dflash_conv_kernel_size;
    const int64_t groups = n_embd / hparams.dflash_conv_group_size;
    const int64_t projected = 2 * kernel * groups;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.dflash_attn_conv_base = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
        layer.dflash_attn_conv_proj = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_PROJ, "weight", i), { n_embd, projected }, 0);
        layer.dflash_ffn_conv_base  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
        layer.dflash_ffn_conv_proj  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_PROJ,  "weight", i), { n_embd, projected }, 0);
    }
}

ggml_tensor * llama_model_dflash::build_dflash2_conv(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * dynamic,
        ggml_tensor * base,
        int side) {
    const auto & hparams = g.hparams;
    const int64_t hidden_size = hidden->ne[0];
    const int64_t n_tokens    = hidden->ne[1];
    const int64_t n_blocks    = g.ubatch.n_seqs_unq;
    const int64_t kernel_size = hparams.dflash_conv_kernel_size;
    const int64_t group_size  = hparams.dflash_conv_group_size;
    const int64_t n_groups    = hidden_size / group_size;

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(dynamic && base && side >= 0 && side < 2);

    const int64_t block_size = n_tokens / n_blocks;
    ggml_context * ctx0 = g.ctx0;
    // ggml_cont copies even when the tensor is already contiguous
    if (!ggml_is_contiguous(hidden) || hidden->ne[1] != n_tokens) {
        hidden = ggml_cont_2d(ctx0, hidden, hidden_size, n_tokens);
    }
    if (!ggml_is_contiguous(dynamic) || dynamic->ne[1] != n_tokens) {
        dynamic = ggml_cont_2d(ctx0, dynamic, dynamic->ne[0], n_tokens);
    }
    ggml_tensor * blocks = ggml_reshape_3d(ctx0, hidden, hidden_size, block_size, n_blocks);
    ggml_tensor * coeffs = ggml_reshape_4d(ctx0, dynamic, n_groups, kernel_size, 2, n_tokens);
    ggml_tensor * coeffs_side = ggml_view_3d(ctx0, coeffs, n_groups, kernel_size, n_tokens,
            coeffs->nb[1], coeffs->nb[3], side * coeffs->nb[2]);

    ggml_tensor * coeff_all = ggml_cont(ctx0, coeffs_side);
    coeff_all = ggml_reshape_4d(ctx0, coeff_all, 1, n_groups, kernel_size, n_tokens);
    coeff_all = ggml_repeat_4d(ctx0, coeff_all, group_size, n_groups, kernel_size, n_tokens);

    ggml_tensor * base_side = ggml_reshape_4d(ctx0,
            ggml_view_1d(ctx0, base, hidden_size * kernel_size, side * base->nb[2]),
            group_size, n_groups, kernel_size, 1);

    ggml_tensor * weight_all = ggml_add(ctx0, coeff_all, base_side);

    // taps at or past block_size only read the left padding and add nothing
    const int64_t n_taps = std::min(kernel_size, block_size);

    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < n_taps; ++tap) {
        ggml_tensor * values = blocks;
        if (tap > 0) {
            ggml_tensor * previous = ggml_view_3d(ctx0, blocks, hidden_size, block_size - tap, n_blocks,
                    blocks->nb[1], blocks->nb[2], 0);
            values = ggml_pad_ext(ctx0, previous, 0, 0, tap, 0, 0, 0, 0, 0);
        }
        values = ggml_reshape_2d(ctx0, values, hidden_size, n_tokens);

        ggml_tensor * weight = ggml_reshape_2d(ctx0,
                ggml_cont(ctx0, ggml_view_4d(ctx0, weight_all, group_size, n_groups, 1, n_tokens,
                        weight_all->nb[1], weight_all->nb[2], weight_all->nb[3], tap * weight_all->nb[2])),
                hidden_size, n_tokens);

        ggml_tensor * term = ggml_mul(ctx0, weight, values);
        result = result ? ggml_add(ctx0, result, term) : term;
    }
    return result;
}

// DFlash2 selector: top-k candidates per block position plus the pairwise
// transition scores, packed into the nextn output slot for the CPU-side walk.
static void build_dflash2_selector(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const auto & hparams = g.hparams;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_embd   = g.n_embd;

    const int64_t top_k    = hparams.dflash_selector_top_k;
    const int64_t rank     = hparams.dflash_selector_rank;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(res->t_logits->ne[1] == n_tokens);
    if (!tokens) {
        return;
    }

    const int64_t tokens_per_block = n_tokens / n_blocks;
    const int64_t block_size = std::min<int64_t>(tokens_per_block, hparams.dflash_block_size);
    const int64_t row_used   = top_k + top_k * top_k;

    ggml_tensor * candidates  = ggml_top_k(ctx0, res->t_logits, top_k);
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, res->t_logits, 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * unary       = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, logits_rows, candidates), top_k, n_tokens);
    ggml_tensor * gate        = g.build_lora_mm(model.dflash_selector_hidden, res->t_embd);

    // Everything below indexes [.., tokens_per_block, n_blocks]: the block
    // position varies fastest, sequences are the outer dimension.
    ggml_tensor * cand_blk  = ggml_reshape_3d(ctx0, candidates, top_k, tokens_per_block, n_blocks);
    ggml_tensor * unary_blk = ggml_reshape_3d(ctx0, unary,      top_k, tokens_per_block, n_blocks);
    ggml_tensor * gate_blk  = ggml_reshape_3d(ctx0, gate,       rank,  tokens_per_block, n_blocks);

    // a position's score reads only the candidate sets at pos-1 and pos, so a run
    // of positions has no internal dependency and scores in one batched matmul
    auto score_run = [&](int64_t beg_pos, int64_t n_pos, ggml_tensor * pred_ids) {
        ggml_tensor * cand_run = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_blk, top_k, n_pos, n_blocks,
                    cand_blk->nb[1], cand_blk->nb[2], beg_pos * cand_blk->nb[1]));
        ggml_tensor * unary_run = ggml_cont(ctx0, ggml_view_3d(ctx0, unary_blk, top_k, n_pos, n_blocks,
                    unary_blk->nb[1], unary_blk->nb[2], beg_pos * unary_blk->nb[1]));
        ggml_tensor * gate_run = ggml_cont(ctx0, ggml_view_3d(ctx0, gate_blk, rank, n_pos, n_blocks,
                    gate_blk->nb[1], gate_blk->nb[2], beg_pos * gate_blk->nb[1]));

        const int64_t n_pred = pred_ids->ne[0] / (n_pos * n_blocks);

        ggml_tensor * successor = ggml_reshape_4d(ctx0,
                ggml_get_rows(ctx0, model.dflash_selector_next, ggml_reshape_1d(ctx0, cand_run, top_k * n_pos * n_blocks)),
                rank, top_k, n_pos, n_blocks);
        ggml_tensor * predecessor = ggml_reshape_4d(ctx0,
                ggml_get_rows(ctx0, model.dflash_selector_prev, pred_ids),
                rank, n_pred, n_pos, n_blocks);

        ggml_tensor * gate_bcast = ggml_reshape_4d(ctx0, gate_run, rank, 1, n_pos, n_blocks);
        ggml_tensor * cond  = ggml_mul(ctx0, predecessor, ggml_repeat(ctx0, gate_bcast, predecessor));
        ggml_tensor * score = ggml_mul_mat(ctx0, successor, cond);
        if (n_pred == 1) {
            score = ggml_repeat_4d(ctx0, score, top_k, top_k, n_pos, n_blocks);
        }
        ggml_tensor * unary_bcast = ggml_reshape_4d(ctx0, unary_run, top_k, 1, n_pos, n_blocks);
        score = ggml_add(ctx0, score, ggml_repeat(ctx0, unary_bcast, score));

        ggml_tensor * row = ggml_concat(ctx0,
                ggml_cast(ctx0, cand_run, GGML_TYPE_F32),
                ggml_reshape_3d(ctx0, score, top_k * top_k, n_pos, n_blocks), 0);
        return ggml_pad(ctx0, row, n_embd - row_used, 0, 0, 0);
    };

    ggml_tensor * packed = ggml_fill(ctx0,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, 1, n_blocks), 0.0f);

    if (block_size > 1) {
        // Position 1 alone: its predecessor is the anchor token, one id per
        // sequence rather than a candidate set.
        ggml_tensor * anchor_ids = ggml_cont_1d(ctx0,
                ggml_view_2d(ctx0, tokens, 1, n_blocks, tokens_per_block * tokens->nb[0], 0), n_blocks);
        packed = ggml_concat(ctx0, packed, score_run(1, 1, anchor_ids), 1);
    }
    if (block_size > 2) {
        ggml_tensor * prev_ids = ggml_reshape_1d(ctx0,
                ggml_cont(ctx0, ggml_view_3d(ctx0, cand_blk, top_k, block_size - 2, n_blocks,
                        cand_blk->nb[1], cand_blk->nb[2], cand_blk->nb[1])),
                top_k * (block_size - 2) * n_blocks);
        packed = ggml_concat(ctx0, packed, score_run(2, block_size - 2, prev_ids), 1);
    }

    packed = ggml_reshape_2d(ctx0, packed, n_embd, block_size * n_blocks);
    g.cb(packed, "dflash2_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

llama_model_dflash::graph_dflash2::graph_dflash2(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);

    build_dflash2_selector(*this, model, inp_tokens);
}
