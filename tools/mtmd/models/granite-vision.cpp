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
 *   Stage 1a: SigLIP vision tower (27 layers, post-norm)
 *   Stage 1b: WindowQFormer blocks (8 blocks total)
 *   Stage 1c: Concatenate and pack outputs
 *
 *   A single WindowQFormer block maps a (1, 576, 1152) vision-layer
 *   hidden state h to a (1, 144, 2560) deepstack feature.
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

// Area-interpolate 24x24 -> 12x12 via 2x2 avg pool
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
        ggml_tensor * tagged = ggml_cont(ctx, t);
        ggml_set_name(tagged, name.c_str());
        ggml_build_forward_expand(gf, tagged);
        t = tagged;
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
        const int n_head = 18; // QFormer: 1152 / 64
        const int d_h   = n_embd / n_head;
        const int nq   = q_stream->ne[1];
        const int nkv  = kv_stream->ne[1];

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
// build() — top-level graph
// ---------------------------------------------------------------------------

ggml_cgraph * clip_graph_granite_vision::build() {
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

    std::vector<ggml_tensor *> streams(projector_count, nullptr);
    for (int bid = 0; bid < projector_count; ++bid) {
        const auto & blk = model.qf_proj_blocks[bid];

        int vlayer = hparams.vision_feature_layer[bid];
        if (vlayer < 0) vlayer = (n_layer + 1) + vlayer;
        GGML_ASSERT(vlayer >= 1 && vlayer <= n_layer);
        ggml_tensor * h = layer_outs[vlayer - 1];

        streams[bid] = g4v_build_block(
            this, ctx0, gf, blk,
            h, bid,
            hparams.proj_spatial_offsets[bid],
            n_patches_x,
            hparams.downsample_window_side,
            hparams.downsample_query_side,
            qformer_eps);
    }

    // Sort by target LLM layer (implicit from vision_feature_layer ordering in conversion)
    // For Granite Vision, the blocks are already in the correct order
    ggml_tensor * mmproj = nullptr;
    for (int k = 0; k < projector_count; ++k) {
        ggml_tensor * s = ggml_cont(ctx0, streams[k]);
        if (k == 0 && hparams.base_stream_scale != 1.0f) {
            s = ggml_scale_inplace(ctx0, s, hparams.base_stream_scale);
        }
        mmproj = (k == 0) ? s : ggml_concat(ctx0, mmproj, s, 0);
    }
    ggml_set_name(mmproj, "g4v_mmproj_out");
    ggml_build_forward_expand(gf, mmproj);

    return gf;
}

// ---------------------------------------------------------------------------
// Assembler — pack-and-unpad + scaled image_newline injection
// ---------------------------------------------------------------------------
//
// Granite Vision 4.1 follows the llava-next layout: the per-tile encoder
// outputs are reshaped so the assembled super-image is read in spatial
// scan order with a learned newline embedding appended to every super-row.
// Single-tile inputs (overview only) get a single trailing newline row.
//
// All knowledge of image_newline and base_stream_scale lives here. The
// graph itself is a single ggml_get_rows over a virtual buffer formed by
// concatenating per-tile encoder outputs with the K-tiled, base-scaled
// newline row. The gather index is built host-side.

size_t granite_vision_n_assembled_output_tokens(
        const clip_ctx * ctx,
        const clip_image_f32_batch * batch) {
    GGML_ASSERT(batch != nullptr && !batch->entries.empty());
    const int n_per_tile = clip_n_output_tokens(
        const_cast<clip_ctx *>(ctx), batch->entries[0].get());
    if (batch->entries.size() == 1) {
        // overview tile + 1 newline row
        return (size_t)n_per_tile + 1;
    }
    const int per_tile_side = (int) std::lround(std::sqrt((double)n_per_tile));
    GGML_ASSERT(per_tile_side * per_tile_side == n_per_tile);
    // overview + (per_tile_side*grid_y) super-rows, each (per_tile_side*grid_x + 1) wide
    const size_t super_h = (size_t)per_tile_side * (size_t)batch->grid_y;
    const size_t super_w_plus_nl = (size_t)per_tile_side * (size_t)batch->grid_x + 1;
    return (size_t)n_per_tile + super_h * super_w_plus_nl;
}

// Build the host-side gather index. Each entry indexes into a virtual buffer
// of length (n_tiles*n_per_tile + 1) whose last row is the K-tiled newline.
static std::vector<int32_t> g4v_build_gather_idx(
        int n_tiles, int n_per_tile, int grid_x, int grid_y) {
    const int32_t nl_idx = n_tiles * n_per_tile;

    std::vector<int32_t> idx;
    idx.reserve((size_t)n_per_tile + 1);
    // overview comes first
    for (int i = 0; i < n_per_tile; ++i) {
        idx.push_back(i);
    }

    if (n_tiles == 1) {
        idx.push_back(nl_idx);
        return idx;
    }

    GGML_ASSERT(grid_x > 0 && grid_y > 0);
    GGML_ASSERT(n_tiles == grid_x * grid_y + 1);
    const int per_tile_side = (int) std::lround(std::sqrt((double)n_per_tile));
    GGML_ASSERT(per_tile_side * per_tile_side == n_per_tile);

    const size_t total = (size_t)n_per_tile +
        (size_t)per_tile_side * (size_t)grid_y *
            ((size_t)per_tile_side * (size_t)grid_x + 1);
    idx.reserve(total);
    for (int gy = 0; gy < grid_y; ++gy) {
        for (int ty = 0; ty < per_tile_side; ++ty) {
            for (int gx = 0; gx < grid_x; ++gx) {
                const int tile_idx = 1 + gy * grid_x + gx; // +1: skip overview
                const int row_off = tile_idx * n_per_tile + ty * per_tile_side;
                for (int tx = 0; tx < per_tile_side; ++tx) {
                    idx.push_back(row_off + tx);
                }
            }
            idx.push_back(nl_idx);
        }
    }
    GGML_ASSERT(idx.size() == total);
    return idx;
}

clip_assembler_granite_vision::clip_assembler_granite_vision(
        const clip_ctx * ctx,
        const float * per_tile_embd,
        int n_tiles, int grid_x, int grid_y)
    : model(*clip_get_model(ctx)),
      per_tile_embd(per_tile_embd),
      n_tiles(n_tiles),
      grid_x(grid_x),
      grid_y(grid_y) {
    GGML_ASSERT(per_tile_embd != nullptr && n_tiles > 0);

    n_mmproj_embd = clip_n_mmproj_embd(ctx);

    // Recover n_per_tile from the model's fixed tile size. All G4V tiles
    // are produced at hparams.image_size × hparams.image_size, so a dummy
    // tile image at that size yields the same n_per_tile as any real tile.
    clip_image_f32 tile_img;
    tile_img.nx = model.hparams.image_size;
    tile_img.ny = model.hparams.image_size;
    n_per_tile = clip_n_output_tokens(const_cast<clip_ctx *>(ctx), &tile_img);

    gather_idx = g4v_build_gather_idx(n_tiles, n_per_tile, grid_x, grid_y);
}

ggml_tensor * clip_assembler_granite_vision::build(
        ggml_context * ctx0, ggml_cgraph * gf) {
    const int K = (int) model.qf_proj_blocks.size();
    GGML_ASSERT(K > 0);
    GGML_ASSERT(n_mmproj_embd % K == 0);
    const int projection_dim = n_mmproj_embd / K;
    GGML_ASSERT(model.image_newline != nullptr);
    GGML_ASSERT(ggml_nelements(model.image_newline) == projection_dim);

    // 1) Inputs.
    ggml_tensor * per_tile_in = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, n_mmproj_embd, n_tiles * n_per_tile);
    ggml_set_name(per_tile_in, "g4v_assembler_per_tile_in");
    ggml_set_input(per_tile_in);

    ggml_tensor * idx_in = ggml_new_tensor_1d(
        ctx0, GGML_TYPE_I32, (int64_t) gather_idx.size());
    ggml_set_name(idx_in, "g4v_assembler_gather_idx");
    ggml_set_input(idx_in);

    // 2) Build K-tiled, base-scaled newline row.
    //      newline_row[k*projection_dim + d] = nl[d] * (k == 0 ? base : 1.0)
    //    i.e. with ne[0]=projection_dim, ne[1]=K stacking, then reshape to
    //    a 1-row vector of length K*projection_dim.
    ggml_tensor * nl = model.image_newline; // (projection_dim,)
    ggml_tensor * nl_first = ggml_scale(ctx0, nl, model.hparams.base_stream_scale);
    ggml_tensor * nl_first_2d = ggml_reshape_2d(ctx0, nl_first, projection_dim, 1);
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
    ggml_tensor * nl_row = ggml_reshape_2d(ctx0, nl_row_2d, n_mmproj_embd, 1);
    ggml_set_name(nl_row, "g4v_assembler_nl_row");

    // 3) Concat per-tile outputs with the newline row to form the virtual
    //    source. The newline becomes the row at index n_tiles*n_per_tile.
    ggml_tensor * src = ggml_concat(ctx0, per_tile_in, nl_row, 1);
    ggml_set_name(src, "g4v_assembler_src");

    // 4) Gather. ggml_get_rows selects along ne[1], producing a tensor of
    //    shape (n_mmproj_embd, gather_len) in spatial scan order.
    ggml_tensor * out = ggml_get_rows(ctx0, src, idx_in);
    ggml_set_name(out, "g4v_assembler_out");

    ggml_build_forward_expand(gf, out);
    return out;
}

void clip_assembler_granite_vision::set_inputs(ggml_cgraph * gf) {
    ggml_tensor * in_per_tile = ggml_graph_get_tensor(gf, "g4v_assembler_per_tile_in");
    ggml_tensor * in_gather   = ggml_graph_get_tensor(gf, "g4v_assembler_gather_idx");
    GGML_ASSERT(in_per_tile != nullptr && in_gather != nullptr);
    ggml_backend_tensor_set(in_per_tile, per_tile_embd, 0,
                            (size_t) n_tiles * n_per_tile * n_mmproj_embd * sizeof(float));
    ggml_backend_tensor_set(in_gather, gather_idx.data(), 0,
                            gather_idx.size() * sizeof(int32_t));
}
