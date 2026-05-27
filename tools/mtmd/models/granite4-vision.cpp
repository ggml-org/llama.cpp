#include "models.h"
#include "../clip-impl.h"
#include "../clip-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

/*
 * Granite Vision 4.1 clip graph
 *
 *   Stage 1a: SigLIP vision tower (N layers, post-norm)
 *   Stage 1b: WindowQFormer blocks (deepstack + spatial)
 *   Stage 1c: Concatenate and pack outputs
 *   Stage 1d: Append newline tokens if append_token is set
 */

// ---------------------------------------------------------------------------
// Permutation helpers
// ---------------------------------------------------------------------------

static ggml_tensor * g4v_gather(ggml_context * ctx, ggml_cgraph * gf,
                                ggml_tensor * src,
                                const std::string & name,
                                int idx_len) {
    (void) gf;
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, idx_len);
    ggml_set_name(idx, name.c_str());
    ggml_set_input(idx);
    return ggml_get_rows(ctx, src, idx);
}

// Area downsampling with average pooling (eg 24x24 -> 12x12 via 2x2)
static ggml_tensor * g4v_interp_down(ggml_context * ctx, ggml_tensor * src, int side, int new_side) {
    const int n_embd = src->ne[0];
    ggml_tensor * t = ggml_reshape_4d(ctx, src, n_embd, side, side, 1);
    t = ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3));
    const int kernel = side / new_side;
    t = ggml_pool_2d(ctx, t, GGML_OP_POOL_AVG, kernel, kernel, kernel, kernel, 0, 0);
    t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));
    return ggml_reshape_2d(ctx, t, n_embd, new_side * new_side);
}

// Build one WindowQFormer block's forward pass
static ggml_tensor * g4v_build_block(
        const clip_graph * g,
        ggml_context * ctx,
        ggml_cgraph * gf,
        const qf_block & blk,
        ggml_tensor * h,
        int bid,
        int spatial_offset,
        int image_side,
        int window_side,
        int query_side,
        float qformer_eps) {

    const int n_embd = h->ne[0];
    GGML_ASSERT(h->ne[1] == image_side * image_side);
    const int n = image_side / window_side;
    const int new_side = n * query_side;
    const int n_windows = n * n;
    const int enc_len = window_side * window_side;
    const int query_len = query_side * query_side;

    auto cbx = [&](ggml_tensor * & t, const char * step) {
        const std::string name = "g4v_blk" + std::to_string(bid) + "_" + step;
        ggml_set_name(t, name.c_str());
    };

    // 1. Top-level LN
    ggml_tensor * x = g->build_norm(h, blk.qf_proj_norm_w, blk.qf_proj_norm_b, NORM_TYPE_NORMAL, g->eps, bid);
    cbx(x, "norm");

    // 2. enc = _win(x, image_side, window_side)
    ggml_tensor * enc;
    {
        ggml_tensor * enc_flat = g4v_gather(ctx, gf, x,
            "g4v_blk" + std::to_string(bid) + "_win_idx",
            image_side * image_side);
        enc = ggml_reshape_3d(ctx, enc_flat, n_embd, enc_len, n_windows);
    }
    cbx(enc, "enc");

    // 3. downsampled = downsampler(x)
    ggml_tensor * d;
    (void) spatial_offset;
    if (spatial_offset >= 0) {
        d = g4v_gather(ctx, gf, x,
            "g4v_blk" + std::to_string(bid) + "_spatial_idx",
            new_side * new_side);
    } else {
        d = g4v_interp_down(ctx, x, image_side, new_side);
    }
    cbx(d, "downsampled");

    // 4. query_embeds = query + _win(d, new_side, query_side)
    ggml_tensor * q_in;
    {
        ggml_tensor * dw_flat = g4v_gather(ctx, gf, d,
            "g4v_blk" + std::to_string(bid) + "_qwin_idx",
            new_side * new_side);
        ggml_tensor * dw = ggml_reshape_3d(ctx, dw_flat, n_embd, query_len, n_windows);
        q_in = ggml_add(ctx, dw, blk.qf_proj_query);
    }
    cbx(q_in, "query_embeds");

    // 5. encoder_embeds = enc + image_positions → (C, enc_len, n_windows)
    ggml_tensor * e_in = ggml_add(ctx, enc, blk.qf_proj_img_pos);
    cbx(e_in, "encoder_embeds");

    // 6. Qformer forward.
    ggml_tensor * q = g->build_norm(q_in, blk.qf_proj_post_norm_w, blk.qf_proj_post_norm_b, NORM_TYPE_NORMAL, qformer_eps, bid);

    // Helper: one post-norm BERT attention block.
    //   attn(q,k,v) via dense W^Q/W^K/W^V + scaled dot-product
    //   residual: LN_out(dropout(dense(attn_out)) + residual_input)
    // For self-attention q == k == v == residual. For cross-attn q != k=v.
    auto run_postnorm_attn = [&](
            ggml_tensor * q_stream,
            ggml_tensor * kv_stream,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            ggml_tensor * wo, ggml_tensor * bo,
            ggml_tensor * ln_w, ggml_tensor * ln_b) -> ggml_tensor * {
        const int d_h    = 64; // hard coded in downsampling.py:77
        const int n_head = n_embd / d_h;
        const int nq     = q_stream->ne[1];
        const int nkv    = kv_stream->ne[1];

        // We loop over the n_windows dimension by treating it as batch via
        // the ne[2] dim in mul_mat.  Collapse (C, n, n_win) → (C, n * n_win)
        // for the linear projections (they don't care about the window dim),
        // then reshape back.
        auto linear = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) -> ggml_tensor * {
            ggml_tensor * t = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
            t = g->build_mm(w, t);
            if (b) t = ggml_add(ctx, t, b);
            return t;
        };

        ggml_tensor * Q = linear(q_stream, wq, bq);
        ggml_tensor * K = linear(kv_stream, wk, bk);
        ggml_tensor * V = linear(kv_stream, wv, bv);

        Q = ggml_reshape_4d(ctx, Q, d_h, n_head, nq,  n_windows);
        K = ggml_reshape_4d(ctx, K, d_h, n_head, nkv, n_windows);
        V = ggml_reshape_4d(ctx, V, d_h, n_head, nkv, n_windows);

        ggml_tensor * q_p = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor * k_p = ggml_permute(ctx, K, 0, 2, 1, 3);
        ggml_tensor * v_p = ggml_permute(ctx, V, 1, 2, 0, 3);
        v_p = ggml_cont(ctx, v_p);

        ggml_tensor * kq = ggml_mul_mat(ctx, k_p, q_p);
        const float scale = 1.0f / std::sqrt((float) d_h);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);

        ggml_tensor * kqv = ggml_mul_mat(ctx, v_p, kq);
        ggml_tensor * out = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        out = ggml_cont_3d(ctx, out, d_h * n_head, nq, n_windows);

        ggml_tensor * o = ggml_reshape_2d(ctx, out, n_embd, nq * n_windows);
        o = g->build_mm(wo, o);
        if (bo) o = ggml_add(ctx, o, bo);
        o = ggml_reshape_3d(ctx, o, n_embd, nq, n_windows);
        o = ggml_add(ctx, o, q_stream);
        o = g->build_norm(o, ln_w, ln_b, NORM_TYPE_NORMAL, qformer_eps, bid);
        return o;
    };

    // Get the single QFormer layer
    GGML_ASSERT(blk.qf_proj_layers.size() == 1);
    const auto & pl = blk.qf_proj_layers[0];

    // 6a. Self-attention
    ggml_tensor * sa_out = run_postnorm_attn(
        q, q,
        pl.q_w,  pl.q_b,
        pl.k_w,  pl.k_b,
        pl.v_w,  pl.v_b,
        pl.o_w, pl.o_b,
        pl.ln_1_w, pl.ln_1_b);
    cbx(sa_out, "sa_out");

    // 6b. Cross-attention
    ggml_tensor * ca_out = run_postnorm_attn(
        sa_out, e_in,
        pl.cross_attn_q_w,  pl.cross_attn_q_b,
        pl.cross_attn_k_w,  pl.cross_attn_k_b,
        pl.cross_attn_v_w,  pl.cross_attn_v_b,
        pl.cross_attn_o_w, pl.cross_attn_o_b,
        pl.cross_attn_norm_w, pl.cross_attn_norm_b);
    cbx(ca_out, "ca_out");

    // 6c. FFN
    ggml_tensor * ffn;
    {
        ggml_tensor * t = ggml_reshape_2d(ctx, ca_out, n_embd, query_len * n_windows);
        t = g->build_mm(pl.ff_up_w, t);
        if (pl.ff_up_b) t = ggml_add(ctx, t, pl.ff_up_b);
        t = ggml_gelu_erf(ctx, t);
        t = g->build_mm(pl.ff_down_w, t);
        if (pl.ff_down_b) t = ggml_add(ctx, t, pl.ff_down_b);
        t = ggml_reshape_3d(ctx, t, n_embd, query_len, n_windows);
        ffn = ggml_add(ctx, t, ca_out);
        ffn = g->build_norm(ffn, pl.ln_2_w, pl.ln_2_b, NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(ffn, "qformer_out");

    // 7. _unwin back to raster
    ggml_tensor * unwinned;
    {
        ggml_tensor * flat = ggml_reshape_2d(ctx, ffn, n_embd, query_len * n_windows);
        unwinned = g4v_gather(ctx, gf, flat,
            "g4v_blk" + std::to_string(bid) + "_unwin_idx",
            new_side * new_side);
    }
    cbx(unwinned, "unwin");

    // 8. out_linear
    ggml_tensor * out = g->build_mm(blk.qf_proj_linear_w, unwinned);
    if (blk.qf_proj_linear_b) out = ggml_add(ctx, out, blk.qf_proj_linear_b);
    cbx(out, "out");

    return out;
}

// ---------------------------------------------------------------------------
// build() - top-level graph
// ---------------------------------------------------------------------------

// Build the K-tiled, base-scaled newline row tensor.
// Shape: (n_mmproj_embd, 1)
ggml_tensor * clip_graph_granite4_vision::build_newline_row(ggml_context * ctx0) {
    const int K = (int) model.qf_proj_blocks.size();
    GGML_ASSERT(K > 0);
    GGML_ASSERT(n_mmproj_embd % K == 0);
    const int projection_dim = n_mmproj_embd / K;
    GGML_ASSERT(model.image_newline != nullptr);
    GGML_ASSERT(ggml_nelements(model.image_newline) == projection_dim);

    // Build newline_row[k*projection_dim + d] = nl[d] * (k == 0 ? base : 1.0)
    ggml_tensor * nl = model.image_newline; // (projection_dim,)
    ggml_tensor * nl_first_2d = ggml_reshape_2d(ctx0, nl, projection_dim, 1);
    ggml_tensor * nl_row_2d;
    if (K == 1) {
        nl_row_2d = nl_first_2d;
    } else {
        ggml_tensor * nl_2d = ggml_reshape_2d(ctx0, nl, projection_dim, 1);
        ggml_tensor * rest_template = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_F32, projection_dim, K - 1);
        ggml_tensor * nl_rest = ggml_repeat(ctx0, nl_2d, rest_template);
        nl_row_2d = ggml_concat(ctx0, nl_first_2d, nl_rest, 1); // (projection_dim, K)
    }
    nl_row_2d = ggml_cont(ctx0, nl_row_2d);
    return ggml_reshape_2d(ctx0, nl_row_2d, n_mmproj_embd, 1);
}

// Append a single newline row at the end of the tile output.
ggml_tensor * clip_graph_granite4_vision::append_rowwise_newlines(ggml_context * ctx0, ggml_tensor * tile_output) {
    // For the single-tile case, append one newline row at the end.
    // For the multi-tile rowwise case, this will be called per-tile
    // (though currently only the single-tile path uses it).
    ggml_tensor * nl_row = build_newline_row(ctx0);
    return ggml_concat(ctx0, tile_output, nl_row, 1);
}

ggml_cgraph * clip_graph_granite4_vision::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(!model.qf_proj_blocks.empty());

    // --- Stage 1a: SigLIP encoder producing intermediate hidden states ---
    ggml_tensor * inp = build_inp();
    inp = ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "pos_embed", -1);

    ggml_tensor * inpL = inp;
    std::vector<ggml_tensor *> layer_outs(n_layer, nullptr);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        // Self-attention
        ggml_tensor * Qcur = build_mm(layer.q_w, cur);
        if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);
        ggml_tensor * Kcur = build_mm(layer.k_w, cur);
        if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);
        ggml_tensor * Vcur = build_mm(layer.v_w, cur);
        if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);

        Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
        Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
        Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

        cur = build_attn(layer.o_w, layer.o_b,
                         Qcur, Kcur, Vcur, nullptr, kq_scale, il);

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur,
                        layer.ff_up_w, layer.ff_up_b,
                        layer.ff_gate_w, layer.ff_gate_b,
                        layer.ff_down_w, layer.ff_down_b,
                        hparams.ffn_op, il);
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);
        layer_outs[il] = cur;
        inpL = cur;
    }

    // --- Stage 1b/1c: WindowQFormer blocks ---
    const int projector_count = hparams.vision_feature_layer.size();
    const float qformer_eps = 1e-12f;

    ggml_tensor * mmproj = nullptr;
    for (int bid = 0; bid < projector_count; ++bid) {
        const auto & blk = model.qf_proj_blocks[bid];

        int vlayer = hparams.vision_feature_layer[bid];
        if (vlayer < 0) vlayer = (n_layer + 1) + vlayer;
        GGML_ASSERT(vlayer >= 1 && vlayer <= n_layer);
        ggml_tensor * h = layer_outs[vlayer - 1];

        ggml_tensor * stream = g4v_build_block(
            this, ctx0, gf, blk,
            h, bid,
            hparams.proj_spatial_offsets[bid],
            n_patches_x,
            hparams.downsample_window_side,
            hparams.downsample_query_side,
            qformer_eps);
        mmproj = mmproj ? ggml_concat(ctx0, mmproj, stream, 0) : stream;
    }

    // --- Stage 1d: Append newline tokens if append_token is set ---
    if (append_token_type == clip_image_f32::CLIP_APPEND_TOKEN_NEWLINE_ROWWISE) {
        mmproj = append_rowwise_newlines(ctx0, mmproj);
        ggml_set_name(mmproj, "g4v_mmproj_out_nl");
    } else {
        ggml_set_name(mmproj, "g4v_mmproj_out");
    }
    ggml_build_forward_expand(gf, mmproj);

    return gf;
}
