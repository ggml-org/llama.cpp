#include "models.h"

void llama_model_dflash::load_hparams_lilicorr(llama_model_loader & ml) {
    ml.get_key(LLM_KV_LILICORR_CANDIDATE_TOPK, hparams.lilicorr_candidate_topk);
    ml.get_key(LLM_KV_LILICORR_NUM_HEADS,      hparams.lilicorr_num_heads);
    ml.get_key(LLM_KV_LILICORR_NUM_LAYERS,     hparams.lilicorr_layers);
    ml.get_key(LLM_KV_LILICORR_VECTOR_EPS,     hparams.lilicorr_vector_eps);
    ml.get_key(LLM_KV_LILICORR_LOGIT_SCALE,    hparams.lilicorr_logit_scale);
    if (hparams.dflash_block_size < 2 || hparams.lilicorr_hidden_size == 0 ||
            hparams.lilicorr_num_heads == 0 || hparams.lilicorr_hidden_size % hparams.lilicorr_num_heads ||
            hparams.lilicorr_candidate_topk == 0 || hparams.lilicorr_layers == 0 ||
            hparams.dflash_selector_top_k > 0 ||
            (uint64_t) hparams.lilicorr_candidate_topk * (hparams.lilicorr_candidate_topk + 1) > hparams.n_embd) {
        throw std::runtime_error("invalid LiLiCorr DFlash metadata");
    }
}

void llama_model_dflash::load_tensors_lilicorr() {
    LLAMA_LOAD_LOCALS;

    const int64_t hidden = hparams.lilicorr_hidden_size;
    const int64_t top_k  = hparams.lilicorr_candidate_topk;
    const int64_t heads  = hparams.lilicorr_num_heads;
    uint32_t factor = 0, mlp = 0;
    ml->get_key(LLM_KV_LILICORR_FACTOR_DIM, factor);
    ml->get_key(LLM_KV_LILICORR_MLP_HIDDEN_SIZE, mlp);
    const int64_t slots = hparams.dflash_block_size - 1;
    const int feature_bid = n_layer_all + hparams.lilicorr_layers;
    if (factor == 0 || mlp == 0 || hparams.lilicorr_layers > layers.size()) {
        throw std::runtime_error("invalid LiLiCorr head dimensions");
    }
    if (hidden != n_embd) {
        lilicorr_token_proj         = create_tensor(tn(LLM_TENSOR_LILICORR_TOKEN_PROJ, "weight"), { n_embd, hidden }, 0);
        lilicorr_token_proj_b       = create_tensor(tn(LLM_TENSOR_LILICORR_TOKEN_PROJ, "bias"),   { hidden },         0);
        lilicorr_token_proj_s       = create_tensor(tn(LLM_TENSOR_LILICORR_TOKEN_PROJ, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    }
    lilicorr_pass_hidden_proj       = create_tensor(tn(LLM_TENSOR_LILICORR_PASS_HIDDEN_PROJ, "weight"), { hparams.n_embd, hidden }, 0);
    lilicorr_pass_hidden_proj_b     = create_tensor(tn(LLM_TENSOR_LILICORR_PASS_HIDDEN_PROJ, "bias"),   { hidden },                 0);
    lilicorr_pass_hidden_proj_s     = create_tensor(tn(LLM_TENSOR_LILICORR_PASS_HIDDEN_PROJ, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    lilicorr_feature_norm           = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", feature_bid), { 5 }, 0);
    lilicorr_feature_norm_b         = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", feature_bid),   { 5 }, 0);
    lilicorr_feature_mlp_1          = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", feature_bid), { 5, hidden }, 0);
    lilicorr_feature_mlp_1_b        = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", feature_bid),   { hidden },    0);
    lilicorr_feature_mlp_3          = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", feature_bid), { hidden, hidden }, 0);
    lilicorr_feature_mlp_3_b        = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", feature_bid),   { hidden },         0);
    lilicorr_slot_embedding         = create_tensor(tn(LLM_TENSOR_LILICORR_SLOT_EMBD, "weight"), { hidden, slots }, 0);
    lilicorr_rank_embedding         = create_tensor(tn(LLM_TENSOR_LILICORR_RANK_EMBD, "weight"), { hidden, top_k }, 0);
    lilicorr_attention_bias         = create_tensor(tn(LLM_TENSOR_LILICORR_ATTN_BIAS, "bias"), { slots * top_k, slots * top_k, heads }, 0);
    lilicorr_context_proj           = create_tensor(tn(LLM_TENSOR_LILICORR_CONTEXT_PROJ, "weight"), { n_embd, hidden }, 0);
    lilicorr_context_proj_b         = create_tensor(tn(LLM_TENSOR_LILICORR_CONTEXT_PROJ, "bias"),   { hidden },         0);
    lilicorr_context_proj_s         = create_tensor(tn(LLM_TENSOR_LILICORR_CONTEXT_PROJ, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    for (uint32_t i = 0; i < hparams.lilicorr_layers; ++i) {
        const int bid = n_layer_all + i;
        layers[i].lilicorr_attn_norm       = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", bid), { hidden }, 0);
        layers[i].lilicorr_mlp_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", bid), { hidden },  0);
        layers[i].lilicorr_attn_in_proj    = create_tensor(tn(LLM_TENSOR_LILICORR_ATTN_IN_PROJ, "weight", i), { hidden, 3 * hidden }, 0);
        layers[i].lilicorr_attn_in_proj_b  = create_tensor(tn(LLM_TENSOR_LILICORR_ATTN_IN_PROJ, "bias", i), { 3 * hidden },           0);
        layers[i].lilicorr_attn_out_proj   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", bid), { hidden, hidden }, 0);
        layers[i].lilicorr_attn_out_proj_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", bid), { hidden },           0);
        layers[i].lilicorr_attn_out_proj_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", bid), { 1 }, TENSOR_NOT_REQUIRED);
        layers[i].lilicorr_mlp_0           = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", bid), { hidden, mlp }, 0);
        layers[i].lilicorr_mlp_0_b         = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", bid), { mlp },           0);
        layers[i].lilicorr_mlp_0_s         = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", bid), { 1 }, TENSOR_NOT_REQUIRED);
        layers[i].lilicorr_mlp_2           = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", bid), { mlp, hidden }, 0);
        layers[i].lilicorr_mlp_2_b         = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", bid), { hidden },        0);
        layers[i].lilicorr_mlp_2_s         = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", bid), { 1 }, TENSOR_NOT_REQUIRED);
    }
    lilicorr_output_norm            = create_tensor(tn(LLM_TENSOR_LILICORR_OUTPUT_NORM, "weight"), { hidden }, 0);
    lilicorr_anchor_norm            = create_tensor(tn(LLM_TENSOR_LILICORR_ANCHOR_NORM, "weight"), { hidden }, 0);
    lilicorr_factor_input_proj      = create_tensor(tn(LLM_TENSOR_LILICORR_FACTOR_INPUT_PROJ, "weight"), { 3 * hidden, hidden }, 0);
    lilicorr_factor_input_proj_b    = create_tensor(tn(LLM_TENSOR_LILICORR_FACTOR_INPUT_PROJ, "bias"), { hidden },               0);
    lilicorr_out_head               = create_tensor(tn(LLM_TENSOR_LILICORR_OUT_HEAD, "weight"), { hidden, factor }, 0);
    lilicorr_out_head_b             = create_tensor(tn(LLM_TENSOR_LILICORR_OUT_HEAD, "bias"), { factor },           0);
    lilicorr_in_head                = create_tensor(tn(LLM_TENSOR_LILICORR_IN_HEAD, "weight"), { hidden, factor }, 0);
    lilicorr_in_head_b              = create_tensor(tn(LLM_TENSOR_LILICORR_IN_HEAD, "bias"), { factor },           0);
    lilicorr_anchor_out_head        = create_tensor(tn(LLM_TENSOR_LILICORR_ANCHOR_OUT_HEAD, "weight"), { hidden, factor }, 0);
    lilicorr_anchor_out_head_b      = create_tensor(tn(LLM_TENSOR_LILICORR_ANCHOR_OUT_HEAD, "bias"), { factor },           0);

    load_tensors_dflash2_conv();
}

static void build_lilicorr_scores(llm_graph_context & g, const llama_model & model,
        ggml_tensor * tok_embd, ggml_tensor * anchors) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const auto & hparams = g.hparams;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_embd   = g.n_embd;

    const int64_t hidden   = hparams.lilicorr_hidden_size;
    const int64_t top_k    = hparams.lilicorr_candidate_topk;
    const int64_t heads    = hparams.lilicorr_num_heads;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(res->t_logits->ne[1] == n_tokens);
    GGML_ASSERT(anchors && tok_embd);

    const int64_t tokens_per_block = n_tokens / n_blocks;
    const int64_t block_size = hparams.dflash_block_size;
    const int64_t row_used   = top_k + top_k * top_k;
    const int64_t slots  = block_size - 1;
    const int64_t nodes  = slots * top_k;
    const int64_t factor_dim = model.lilicorr_out_head->ne[1];
    GGML_ASSERT(tokens_per_block == block_size);

    ggml_tensor * candidates = ggml_top_k(ctx0, res->t_logits, top_k);
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, res->t_logits, 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * candidate_logits = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, logits_rows, candidates), top_k, n_tokens);
    // Rank features and embeddings require candidates in descending logit order.
    ggml_tensor * order = ggml_argsort(ctx0, candidate_logits, GGML_SORT_ORDER_DESC);
    ggml_tensor * candidate_rows = ggml_reshape_3d(ctx0, candidates, 1, top_k, n_tokens);
    candidates = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, candidate_rows, order), top_k, n_tokens);
    ggml_tensor * prob_rows   = ggml_reshape_3d(ctx0, ggml_soft_max(ctx0, res->t_logits), 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * probs       = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, prob_rows, candidates), top_k, n_tokens);

    ggml_tensor * cand_run  = ggml_reshape_3d(ctx0, candidates, top_k, n_tokens / n_blocks, n_blocks);
    ggml_tensor * probs_run = ggml_reshape_3d(ctx0, probs,      top_k, n_tokens / n_blocks, n_blocks);
    cand_run = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_run, top_k, slots, n_blocks,
            cand_run->nb[1], cand_run->nb[2], cand_run->nb[1]));
    probs_run = ggml_cont(ctx0, ggml_view_3d(ctx0, probs_run, top_k, slots, n_blocks,
            probs_run->nb[1], probs_run->nb[2], probs_run->nb[1]));

    ggml_tensor * lp = ggml_log(ctx0, probs_run);
    ggml_tensor * lp_blk = ggml_reshape_3d(ctx0, lp, top_k, slots, n_blocks);
    ggml_tensor * top_lp = ggml_view_3d(ctx0, lp_blk, 1, slots, n_blocks,
            lp_blk->nb[1], lp_blk->nb[2], 0);
    ggml_tensor * relative = ggml_sub(ctx0, lp_blk, top_lp);
    ggml_tensor * feature = ggml_reshape_4d(ctx0, lp_blk, 1, top_k, slots, n_blocks);

    ggml_tensor * rank = ggml_arange(ctx0, 0, top_k, 1);
    rank = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, rank, top_k, 1, 1), lp_blk);
    ggml_tensor * first = ggml_scale_bias(ctx0, ggml_step(ctx0, rank), -1.0f, 1.0f);

    for (ggml_tensor * item : { probs_run, relative,
            ggml_scale(ctx0, rank, top_k == 1 ? 0.0f : 1.0f / (top_k - 1)), first }) {
        feature = ggml_concat(ctx0, feature, ggml_reshape_4d(ctx0, item, 1, top_k, slots, n_blocks), 0);
    }
    feature = ggml_reshape_2d(ctx0, feature, 5, nodes * n_blocks);

    ggml_tensor * embeds = ggml_get_rows(ctx0, tok_embd, ggml_reshape_1d(ctx0, cand_run, nodes * n_blocks));
    ggml_tensor * token_states = n_embd == hidden ? embeds : ggml_add(ctx0, g.build_lora_mm(model.lilicorr_token_proj, embeds, model.lilicorr_token_proj_s), model.lilicorr_token_proj_b);

    ggml_tensor * pass_blk = ggml_reshape_3d(ctx0, res->t_embd, n_embd, block_size, n_blocks);
    ggml_tensor * passes = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, pass_blk, n_embd, slots, n_blocks,
                pass_blk->nb[1], pass_blk->nb[2], pass_blk->nb[1])),
            n_embd, slots * n_blocks);

    ggml_tensor * pass_states = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_pass_hidden_proj, passes, model.lilicorr_pass_hidden_proj_s), model.lilicorr_pass_hidden_proj_b);

    pass_states = ggml_reshape_4d(ctx0, pass_states, hidden, 1, slots, n_blocks);
    pass_states = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, pass_states, hidden, top_k, slots, n_blocks), hidden, nodes * n_blocks);

    ggml_tensor * feature_states = ggml_norm(ctx0, feature, 1.0e-5f);
    feature_states = ggml_add(ctx0, ggml_mul(ctx0, feature_states, model.lilicorr_feature_norm), model.lilicorr_feature_norm_b);
    feature_states = ggml_silu(ctx0, ggml_add(ctx0, g.build_lora_mm(model.lilicorr_feature_mlp_1, feature_states), model.lilicorr_feature_mlp_1_b));
    feature_states = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_feature_mlp_3, feature_states), model.lilicorr_feature_mlp_3_b);

    ggml_tensor * slot = ggml_reshape_4d(ctx0, model.lilicorr_slot_embedding, hidden, 1, slots, 1);
    ggml_tensor * rank_embd = ggml_reshape_4d(ctx0, model.lilicorr_rank_embedding, hidden, top_k, 1, 1);
    ggml_tensor * pos = ggml_add(ctx0, ggml_repeat_4d(ctx0, slot, hidden, top_k, slots, n_blocks), ggml_repeat_4d(ctx0, rank_embd, hidden, top_k, slots, n_blocks));
    pos = ggml_reshape_2d(ctx0, pos, hidden, nodes * n_blocks);

    ggml_tensor * x = ggml_add(ctx0, ggml_add(ctx0, token_states, pass_states), feature_states);
    x = ggml_add(ctx0, x, pos);
    ggml_tensor * anchor_blk = ggml_reshape_3d(ctx0, anchors, n_embd, block_size, n_blocks);
    ggml_tensor * anchor = ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_view_3d(ctx0, anchor_blk, n_embd, 1, n_blocks,
                    anchor_blk->nb[1], anchor_blk->nb[2], 0)), n_embd, n_blocks);
    ggml_tensor * anchor_state = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_context_proj, anchor, model.lilicorr_context_proj_s), model.lilicorr_context_proj_b);

    const int64_t head_dim = hidden / heads;
    for (uint32_t layer = 0; layer < hparams.lilicorr_layers; ++layer) {
        const auto & weights = model.layers[layer];
        ggml_tensor * attn_input = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), weights.lilicorr_attn_norm);
        ggml_tensor * qkv = ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_attn_in_proj, attn_input), weights.lilicorr_attn_in_proj_b);
        qkv = ggml_reshape_3d(ctx0, qkv, 3 * hidden, nodes, n_blocks);
        ggml_tensor * q = ggml_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], 0));
        ggml_tensor * k = ggml_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], hidden * qkv->nb[0]));
        ggml_tensor * v = ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], 2 * hidden * qkv->nb[0]);
        ggml_tensor * attn = ggml_soft_max_ext(ctx0, ggml_mul_mat(ctx0, k, q), model.lilicorr_attention_bias,
                1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
        ggml_tensor * joined = ggml_out_prod(ctx0, v, ggml_transpose(ctx0, attn));
        joined = ggml_permute(ctx0, joined, 0, 2, 1, 3);
        ggml_tensor * attn_output = ggml_cont_2d(ctx0, joined, hidden, nodes * n_blocks);
        attn_output = ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_attn_out_proj, attn_output, weights.lilicorr_attn_out_proj_s), weights.lilicorr_attn_out_proj_b);
        x = ggml_add(ctx0, x, attn_output);
        ggml_tensor * mlp = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), weights.lilicorr_mlp_norm);
        mlp = ggml_silu(ctx0, ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_mlp_0, mlp, weights.lilicorr_mlp_0_s), weights.lilicorr_mlp_0_b));
        mlp = ggml_add(ctx0, g.build_lora_mm(weights.lilicorr_mlp_2, mlp, weights.lilicorr_mlp_2_s), weights.lilicorr_mlp_2_b);
        x = ggml_add(ctx0, x, mlp);
    }

    x = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), model.lilicorr_output_norm);
    anchor_state = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchor_state, hparams.f_norm_rms_eps), model.lilicorr_anchor_norm);
    ggml_tensor * anchor_nodes = ggml_reshape_3d(ctx0, anchor_state, hidden, 1, n_blocks);
    anchor_nodes = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, anchor_nodes, hidden, nodes, n_blocks, 1), hidden, nodes * n_blocks);
    ggml_tensor * factor_in = ggml_concat(ctx0, x, anchor_nodes, 0);
    factor_in = ggml_concat(ctx0, factor_in, ggml_mul(ctx0, x, anchor_nodes), 0);
    ggml_tensor * factors = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_factor_input_proj, factor_in), model.lilicorr_factor_input_proj_b);
    factors = ggml_silu(ctx0, factors);

    ggml_tensor * out = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_out_head, factors), model.lilicorr_out_head_b);
    out = ggml_reshape_4d(ctx0, ggml_l2_norm(ctx0, out, hparams.lilicorr_vector_eps), factor_dim, top_k, slots, n_blocks);

    ggml_tensor * in = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_in_head, factors), model.lilicorr_in_head_b);
    in = ggml_reshape_4d(ctx0, ggml_l2_norm(ctx0, in, hparams.lilicorr_vector_eps), factor_dim, top_k, slots, n_blocks);

    ggml_tensor * anchor_out = ggml_add(ctx0, g.build_lora_mm(model.lilicorr_anchor_out_head, anchor_state), model.lilicorr_anchor_out_head_b);
    anchor_out = ggml_l2_norm(ctx0, anchor_out, hparams.lilicorr_vector_eps);
    anchor_out = ggml_reshape_3d(ctx0, anchor_out, factor_dim, 1, n_blocks);

    ggml_tensor * first_in = ggml_cont(ctx0, ggml_view_4d(ctx0, in, factor_dim, top_k, 1, n_blocks,
            in->nb[1], in->nb[2], in->nb[3], 0));
    ggml_tensor * first_score = ggml_scale(ctx0,
            ggml_mul_mat(ctx0, first_in, ggml_reshape_4d(ctx0, anchor_out, factor_dim, 1, 1, n_blocks)),
            hparams.lilicorr_logit_scale);
    ggml_tensor * scores = ggml_pad(ctx0,
            ggml_reshape_3d(ctx0, first_score, top_k, 1, n_blocks), top_k * top_k - top_k, 0, 0, 0);
    if (slots > 1) {
        ggml_tensor * next = ggml_cont(ctx0, ggml_view_4d(ctx0, in, factor_dim, top_k, slots - 1, n_blocks,
                in->nb[1], in->nb[2], in->nb[3], in->nb[2]));
        ggml_tensor * previous = ggml_cont(ctx0, ggml_view_4d(ctx0, out, factor_dim, top_k, slots - 1, n_blocks,
                out->nb[1], out->nb[2], out->nb[3], 0));
        ggml_tensor * score = ggml_scale(ctx0, ggml_mul_mat(ctx0, next, previous), hparams.lilicorr_logit_scale);
        scores = ggml_concat(ctx0, scores,
                ggml_reshape_3d(ctx0, score, top_k * top_k, slots - 1, n_blocks), 1);
    }

    ggml_tensor * row = ggml_concat(ctx0, ggml_cast(ctx0, ggml_reshape_3d(ctx0, cand_run, top_k, slots, n_blocks),
            GGML_TYPE_F32), scores, 0);
    ggml_tensor * packed = ggml_pad_ext(ctx0, row, 0, n_embd - row_used, 1, 0, 0, 0, 0, 0);
    packed = ggml_reshape_2d(ctx0, packed, n_embd, n_tokens);
    g.cb(packed, "lilicorr_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

llama_model_dflash::graph_lilicorr::graph_lilicorr(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        // Expose context features for retrieval by LiliCorr
        res->t_h_nextn = build_inject();
        return;
    }

    // only a full block per sequence has the anchors the scorer needs
    const bool is_block = ubatch.embd_state && n_tokens == (int64_t) hparams.dflash_block_size * ubatch.n_seqs_unq;

    ggml_tensor * inp_tokens = nullptr;
    ggml_tensor * inp_anchor = nullptr;
    if (is_block) {
        inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_tokens);
        res->t_inp_tokens = inp_tokens;

        // inp_anchor passes the context features to the lattice scorer
        auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd, n_embd);
        inp->tokens = inp_tokens;
        inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->h);
        inp_anchor = inp->h;
        res->add_input(std::move(inp));
    } else {
        inp_tokens = build_inp_tokens();
    }

    ggml_tensor * cur = build_layers(inp_tokens);

    build_logits(cur);

    if (is_block) {
        build_lilicorr_scores(*this, model, get_tok_embd(*this, model), inp_anchor);
    }
}
