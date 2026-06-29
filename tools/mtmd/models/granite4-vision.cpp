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
 *   Stage 1d: Append newline tokens if add_newline is set
 */

// ---------------------------------------------------------------------------
// Member method implementations
// ---------------------------------------------------------------------------

ggml_tensor * clip_graph_granite4_vision::gather(
        ggml_tensor * src,
        const std::string & name,
        int idx_len) {
    ggml_tensor * idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, idx_len);
    ggml_set_name(idx, name.c_str());
    ggml_set_input(idx);
    return ggml_get_rows(ctx0, src, idx);
}

ggml_tensor * clip_graph_granite4_vision::interp_down(
        ggml_tensor * src,
        int side,
        int new_side) {
    const int n_embd = src->ne[0];
    ggml_tensor * t = ggml_reshape_4d(ctx0, src, n_embd, side, side, 1);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 2, 0, 1, 3));
    const int kernel = side / new_side;
    t = ggml_pool_2d(ctx0, t, GGML_OP_POOL_AVG, kernel, kernel, kernel, kernel, 0, 0);
    t = ggml_cont(ctx0, ggml_permute(ctx0, t, 1, 2, 0, 3));
    return ggml_reshape_2d(ctx0, t, n_embd, new_side * new_side);
}

// ---------------------------------------------------------------------------
// build_block - WindowQFormer block implementation
// ---------------------------------------------------------------------------

ggml_tensor * clip_graph_granite4_vision::build_block(
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
    cbx(h, "inp");
    ggml_tensor * x = build_norm(h, blk.qf_proj_norm_w, blk.qf_proj_norm_b, NORM_TYPE_NORMAL, eps, bid);
    cbx(x, "norm");

    // 2. enc = _win(x, image_side, window_side)
    ggml_tensor * enc;
    {
        ggml_tensor * enc_flat = gather(x,
            "g4v_blk" + std::to_string(bid) + "_win_idx",
            image_side * image_side);
        enc = ggml_reshape_3d(ctx0, enc_flat, n_embd, enc_len, n_windows);
    }
    cbx(enc, "enc");

    // 3. downsampled = downsampler(x)
    ggml_tensor * d;
    (void) spatial_offset;
    if (spatial_offset >= 0) {
        d = gather(x,
            "g4v_blk" + std::to_string(bid) + "_spatial_idx",
            new_side * new_side);
    } else {
        d = interp_down(x, image_side, new_side);
    }
    cbx(d, "downsampled");

    // 4. query_embeds = query + _win(d, new_side, query_side)
    ggml_tensor * q_in;
    {
        ggml_tensor * dw_flat = gather(d,
            "g4v_blk" + std::to_string(bid) + "_qwin_idx",
            new_side * new_side);
        ggml_tensor * dw = ggml_reshape_3d(ctx0, dw_flat, n_embd, query_len, n_windows);
        q_in = ggml_add(ctx0, dw, blk.qf_proj_query);
    }
    cbx(q_in, "query_embeds");

    // 5. encoder_embeds = enc + image_positions → (C, enc_len, n_windows)
    ggml_tensor * e_in = ggml_add(ctx0, enc, blk.qf_proj_img_pos);
    cbx(e_in, "encoder_embeds");

    // 6. Qformer forward.
    ggml_tensor * q = build_norm(q_in, blk.qf_proj_post_norm_w, blk.qf_proj_post_norm_b, NORM_TYPE_NORMAL, qformer_eps, bid);

    // Helper for linear projections with window batching
    auto linear = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) -> ggml_tensor * {
        ggml_tensor * t = ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1] * x->ne[2]);
        t = build_mm(w, t);
        if (b) t = ggml_add(ctx0, t, b);
        return t;
    };

    // Get the single QFormer layer
    GGML_ASSERT(blk.qf_proj_layers.size() == 1);
    const auto & pl = blk.qf_proj_layers[0];

    // 6a. Self-attention
    ggml_tensor * sa_out;
    {
        const int d_h = 64;
        const int n_head = n_embd / d_h;
        const int nq = q->ne[1];
        const float scale = 1.0f / std::sqrt((float) d_h);

        ggml_tensor * Q = linear(q, pl.q_w, pl.q_b);
        ggml_tensor * K = linear(q, pl.k_w, pl.k_b);
        ggml_tensor * V = linear(q, pl.v_w, pl.v_b);

        Q = ggml_reshape_4d(ctx0, Q, d_h, n_head, nq, n_windows);
        K = ggml_reshape_4d(ctx0, K, d_h, n_head, nq, n_windows);
        V = ggml_reshape_4d(ctx0, V, d_h, n_head, nq, n_windows);

        sa_out = build_attn(pl.o_w, pl.o_b, Q, K, V, nullptr, scale, bid);
        sa_out = ggml_reshape_3d(ctx0, sa_out, n_embd, nq, n_windows);

        sa_out = ggml_add(ctx0, sa_out, q);
        sa_out = build_norm(sa_out, pl.ln_1_w, pl.ln_1_b,
                            NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(sa_out, "sa_out");

    // 6b. Cross-attention
    ggml_tensor * ca_out;
    {
        const int d_h = 64;
        const int n_head = n_embd / d_h;
        const int nq = sa_out->ne[1];
        const int nkv = e_in->ne[1];
        const float scale = 1.0f / std::sqrt((float) d_h);

        ggml_tensor * Q = linear(sa_out, pl.cross_attn_q_w, pl.cross_attn_q_b);
        ggml_tensor * K = linear(e_in, pl.cross_attn_k_w, pl.cross_attn_k_b);
        ggml_tensor * V = linear(e_in, pl.cross_attn_v_w, pl.cross_attn_v_b);

        Q = ggml_reshape_4d(ctx0, Q, d_h, n_head, nq, n_windows);
        K = ggml_reshape_4d(ctx0, K, d_h, n_head, nkv, n_windows);
        V = ggml_reshape_4d(ctx0, V, d_h, n_head, nkv, n_windows);

        ca_out = build_attn(pl.cross_attn_o_w, pl.cross_attn_o_b,
                            Q, K, V, nullptr, scale, bid);
        ca_out = ggml_reshape_3d(ctx0, ca_out, n_embd, nq, n_windows);

        ca_out = ggml_add(ctx0, ca_out, sa_out);
        ca_out = build_norm(ca_out, pl.cross_attn_norm_w, pl.cross_attn_norm_b,
                            NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(ca_out, "ca_out");

    // 6c. FFN
    ggml_tensor * ffn;
    {
        ggml_tensor * t = ggml_reshape_2d(ctx0, ca_out, n_embd, query_len * n_windows);
        t = build_mm(pl.ff_up_w, t);
        if (pl.ff_up_b) t = ggml_add(ctx0, t, pl.ff_up_b);
        t = ggml_gelu_erf(ctx0, t);
        t = build_mm(pl.ff_down_w, t);
        if (pl.ff_down_b) t = ggml_add(ctx0, t, pl.ff_down_b);
        t = ggml_reshape_3d(ctx0, t, n_embd, query_len, n_windows);
        ffn = ggml_add(ctx0, t, ca_out);
        ffn = build_norm(ffn, pl.ln_2_w, pl.ln_2_b, NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(ffn, "qformer_out");

    // 7. _unwin back to raster
    ggml_tensor * unwinned;
    {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, ffn, n_embd, query_len * n_windows);
        unwinned = gather(flat,
            "g4v_blk" + std::to_string(bid) + "_unwin_idx",
            new_side * new_side);
    }
    cbx(unwinned, "unwin");

    // 8. out_linear
    ggml_tensor * out = build_mm(blk.qf_proj_linear_w, unwinned);
    if (blk.qf_proj_linear_b) out = ggml_add(ctx0, out, blk.qf_proj_linear_b);
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
    const int projector_count = hparams.feature_layers.size();
    const float qformer_eps = 1e-12f;

    ggml_tensor * mmproj = nullptr;
    for (int bid = 0; bid < projector_count; ++bid) {
        const auto & blk = model.qf_proj_blocks[bid];

        int vlayer = hparams.feature_layers[bid];
        GGML_ASSERT(vlayer >= 0 && vlayer < n_layer);
        ggml_tensor * h = layer_outs[vlayer];

        ggml_tensor * stream = build_block(
            blk, h, bid,
            hparams.proj_spatial_offsets[bid],
            n_patches_x,
            hparams.downsample_window_side,
            hparams.downsample_query_side,
            qformer_eps);
        cb(stream, (std::string("proj_") + std::to_string(bid) + std::string("_v_out")).c_str(), vlayer);
        mmproj = mmproj ? ggml_concat(ctx0, mmproj, stream, 0) : stream;
    }

    // --- Stage 1d: Append newline tokens if add_newline is set ---
    if (add_newline) {
        mmproj = append_rowwise_newlines(ctx0, mmproj);
        ggml_set_name(mmproj, "g4v_mmproj_out_nl");
    } else {
        ggml_set_name(mmproj, "g4v_mmproj_out");
    }
    ggml_build_forward_expand(gf, mmproj);

    return gf;
}
