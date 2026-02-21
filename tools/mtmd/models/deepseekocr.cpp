#include "models.h"

// Implementation based on approach suggested by Acly
// See: https://github.com/ggml-org/llama.cpp/pull/17383#issuecomment-3554227091
static ggml_tensor * window_partition(ggml_context * ctx0, ggml_tensor * x, const int window) {
    auto [c, w, h, b] = x->ne;
    // same as
    // x = ggml_win_part(m, x, window);
    // x = ggml_reshape_3d(m, x, c, window * window, x->ne[3]);

    const int64_t px  = (window - w % window) % window;
    const int64_t py  = (window - h % window) % window;
    const int64_t npw = (w + px) / window;
    const int64_t nph = (h + py) / window;

    ggml_tensor * cur = x;
    if (px > 0 || py > 0) {
        cur = ggml_pad(ctx0, cur, 0, static_cast<int>(px), static_cast<int>(py), 0);
    }
    cur = ggml_reshape_4d(ctx0, cur, c * window, npw, window, nph * b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, c, window, window, npw * nph * b);
    return cur;
}

// Implementation based on approach suggested by Acly
// See: https://github.com/ggml-org/llama.cpp/pull/17383#issuecomment-3554227091
static ggml_tensor * window_unpartition(ggml_context * ctx0,
                                        ggml_tensor *  x,
                                        const int      w,
                                        const int      h,
                                        const int      window) {
    const int64_t c = x->ne[0];
    // same as
    // x = ggml_reshape_4d(m, x, c, window, window, x->ne[2]);
    // x = ggml_win_unpart(m, x, w, h, window);

    const int64_t px  = (window - w % window) % window;
    const int64_t py  = (window - h % window) % window;
    const int64_t npw = (w + px) / window;
    const int64_t nph = (h + py) / window;

    const int64_t b = x->ne[3] / (npw * nph);
    ggml_tensor * cur = x;
    cur = ggml_reshape_4d(ctx0, cur, c * window, window, npw, nph * b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, c, w + px, h + py, b);
    cur = ggml_view_4d(ctx0, cur, cur->ne[0], w, h, cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0);
    cur = ggml_cont(ctx0, cur);
    return cur;
}

static ggml_tensor * get_rel_pos(ggml_context * ctx0,
                                 ggml_tensor *  rel_pos,  // [L, C]
                                 ggml_tensor *  indices,  // [q_size, k_size]
                                 const int      q_size,
                                 const int      k_size) {
    const int64_t C = rel_pos->ne[0];  // channels
    const int64_t L = rel_pos->ne[1];  // length

    GGML_ASSERT(indices != nullptr);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(indices->ne[0] == k_size);
    GGML_ASSERT(indices->ne[1] == q_size);

    const auto    max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    ggml_tensor * cur          = rel_pos;

    if (max_rel_dist != L) {
        // Linear interpolation
        const int64_t ne0 = cur->ne[0];
        const int64_t ne1 = cur->ne[1];
        const int64_t ne2 = cur->ne[2];
        const int64_t ne3 = cur->ne[3];

        cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3)), ne1, 1, ne0 * ne2 * ne3);
        cur = ggml_reshape_4d(
            ctx0, ggml_interpolate(ctx0, cur, max_rel_dist, 1, ne0 * ne2 * ne3, 1, GGML_SCALE_MODE_BILINEAR),
            max_rel_dist, ne0, ne2, ne3);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }

    // Flatten indices to 1D for ggml_get_rows
    const int qk = q_size * k_size;

    cur = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, cur, ggml_reshape_1d(ctx0, indices, qk)), C, k_size, q_size);

    return cur;  // [C, k_size, q_size]
}

ggml_cgraph * clip_graph_deepseekocr::build() {
    // patch embedding
    ggml_tensor * inp_raw = build_inp_raw();

    ggml_tensor * sam_out;
    // Building SAM
    {
        const int n_embd  = hparams.sam_n_embd;
        const int n_layer = hparams.sam_n_layer;
        const int n_heads = hparams.sam_n_head;
        const int d_heads = n_embd / n_heads;
        const int window  = hparams.attn_window_size;

        ggml_tensor * inpL;

        inpL = ggml_conv_2d_sk_p0(ctx0, model.patch_embed_proj_w, inp_raw);
        inpL = ggml_add(ctx0, inpL, ggml_reshape_3d(ctx0, model.patch_embed_proj_b, 1, 1, n_embd));
        inpL = ggml_cont(ctx0, ggml_permute(ctx0, inpL, 1, 2, 0, 3));

        ggml_tensor * rel_pos_indices_local;
        ggml_tensor * rel_pos_indices_global;

        rel_pos_indices_local  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, window, window);
        rel_pos_indices_global = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, inpL->ne[1], inpL->ne[2]);
        ggml_set_name(rel_pos_indices_local, "rel_pos_indices_local");
        ggml_set_name(rel_pos_indices_global, "rel_pos_indices_global");
        ggml_set_input(rel_pos_indices_local);
        ggml_set_input(rel_pos_indices_global);

        ggml_tensor * cur;
        const auto    tgt_size = inpL->ne[1];
        const auto    str_size = model.pos_embed->ne[1];

        if (str_size != tgt_size) {
            ggml_tensor * old_pos_embed = nullptr;
            old_pos_embed               = ggml_cont(ctx0, ggml_permute(ctx0, model.pos_embed, 2, 0, 1, 3));
            ggml_tensor * new_pos_embed =
                ggml_interpolate(ctx0, old_pos_embed, tgt_size, tgt_size, n_embd, 1, GGML_SCALE_MODE_BICUBIC);
            new_pos_embed = ggml_cont(ctx0, ggml_permute(ctx0, new_pos_embed, 1, 2, 0, 3));
            cur           = ggml_add(ctx0, inpL, new_pos_embed);
        } else {
            cur = ggml_add(ctx0, inpL, model.pos_embed);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto &        layer    = model.sam_layers[il];
            ggml_tensor * shortcut = cur;

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

            const int64_t w0 = cur->ne[1];
            const int64_t h0 = cur->ne[2];

            ggml_tensor * indices;

            if (hparams.is_global_attn(il)) {
                indices = rel_pos_indices_global;
            } else {
                // local attention layer - apply window partition
                cur     = window_partition(ctx0, cur, window);
                indices = rel_pos_indices_local;
            }

            const int64_t W = cur->ne[1];
            const int64_t H = cur->ne[2];
            // self-attention
            {
                const int B = cur->ne[3];

                cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                cur = ggml_add(ctx0, cur, layer.qkv_b);
                cur = ggml_cont(ctx0, cur);  // Ensure tensor is contiguous before reshape
                cur = ggml_reshape_4d(ctx0, cur, n_embd, 3, W * H, B);

                ggml_tensor * Q;
                ggml_tensor * K;
                ggml_tensor * V;

                Q = ggml_view_3d(ctx0, cur, n_embd, W * H, B, cur->nb[2], cur->nb[3], 0 * cur->nb[1]);
                Q = ggml_reshape_4d(ctx0, ggml_cont(ctx0, Q), d_heads, n_heads, W * H, B);

                K = ggml_view_3d(ctx0, cur, n_embd, W * H, B, cur->nb[2], cur->nb[3], 1 * cur->nb[1]);
                K = ggml_reshape_4d(ctx0, ggml_cont(ctx0, K), d_heads, n_heads, W * H, B);

                V = ggml_view_3d(ctx0, cur, n_embd, W * H, B, cur->nb[2], cur->nb[3], 2 * cur->nb[1]);
                V = ggml_reshape_4d(ctx0, ggml_cont(ctx0, V), d_heads, n_heads, W * H, B);

                ggml_tensor * mask;
                ggml_tensor * rw;
                ggml_tensor * rh;
                ggml_tensor * qr;

                rw = get_rel_pos(ctx0, layer.rel_pos_w, indices, W, W);  // [W, W, C]
                rh = get_rel_pos(ctx0, layer.rel_pos_h, indices, H, H);  // [H, H, C]
                qr = ggml_permute(ctx0, Q, 0, 2, 1, 3);
                qr = ggml_reshape_4d(ctx0, ggml_cont(ctx0, qr), d_heads, W, H, B * n_heads);

                rw   = ggml_mul_mat(ctx0, rw,
                                    ggml_cont(ctx0, ggml_permute(ctx0, qr, 0, 2, 1, 3)));  // [B*n_heads, W, H, W]
                rw   = ggml_cont(ctx0, ggml_permute(ctx0, rw, 0, 2, 1, 3));                // [B*n_heads, H, W, W]
                rw   = ggml_reshape_4d(ctx0, rw, W, 1, W * H, n_heads * B);
                rw   = ggml_repeat_4d(ctx0, rw, W, H, W * H, n_heads * B);
                rh   = ggml_mul_mat(ctx0, rh, qr);  // [B*n_heads, H, W, H]
                rh   = ggml_reshape_4d(ctx0, rh, 1, H, W * H, n_heads * B);
                mask = ggml_add(ctx0, rw, rh);      // [B*n_heads, H*W, H, W]
                mask = ggml_reshape_4d(ctx0, mask, W * H, W * H, n_heads, B);
                mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);

                const float scale = 1.0f / sqrtf(static_cast<float>(d_heads));

                cur = build_attn(layer.o_w, layer.o_b, Q, K, V, mask, scale,
                                 il);  // [B, H*W, n_embd]
                cur = ggml_reshape_4d(ctx0, ggml_cont(ctx0, cur), n_embd, W, H, B);
            }

            if (hparams.is_global_attn(il) == false) {
                // local attention layer - reverse window partition
                cur = window_unpartition(ctx0, cur, w0, h0, window);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, shortcut);

            ggml_tensor * inpFF = cur;

            // layernorm2
            cur = build_norm(inpFF, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);

            // ffn
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, nullptr, nullptr, layer.ff_down_w, layer.ff_down_b,
                            hparams.ffn_op, il);

            // residual 2
            cur = ggml_add(ctx0, cur, inpFF);
            cb(cur, "sam_layer_out", il);
        }

        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

        cur = ggml_conv_2d(ctx0, model.neck_0_w, cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = build_norm(cur, model.neck_1_w, model.neck_1_b, NORM_TYPE_NORMAL, hparams.eps, -1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

        cur = ggml_conv_2d(ctx0, model.neck_2_w, cur, 1, 1, 1, 1, 1, 1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = build_norm(cur, model.neck_3_w, model.neck_3_b, NORM_TYPE_NORMAL, hparams.eps, -1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

        cur = ggml_conv_2d(ctx0, model.net_2, cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_conv_2d(ctx0, model.net_3, cur, 2, 2, 1, 1, 1, 1);
        cb(cur, "sam_output", -1);

        ggml_build_forward_expand(gf, cur);
        sam_out = cur;
    }

    ggml_tensor * clip_out;
    // Building DS-OCR CLIP
    {
        ggml_tensor * inp;

        inp = ggml_cpy(ctx0, sam_out, ggml_dup_tensor(ctx0, sam_out));
        inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2]);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

        ggml_tensor * new_pos_embd =
            ggml_cpy(ctx0, model.position_embeddings, ggml_dup_tensor(ctx0, model.position_embeddings));

        int        n_pos    = new_pos_embd->ne[1];  // +1 for [CLS]
        const auto tgt_size = static_cast<int>(std::sqrt(inp->ne[1]));
        const auto src_size = static_cast<int>(std::sqrt(n_pos - 1));

        if (tgt_size != src_size) {
            ggml_tensor * old_pos_embd;
            ggml_tensor * cls_tok;

            old_pos_embd = ggml_view_2d(ctx0, new_pos_embd, new_pos_embd->ne[0], src_size * src_size,
                                        ggml_row_size(new_pos_embd->type, new_pos_embd->ne[0]), 0);
            cls_tok      = ggml_view_2d(ctx0, new_pos_embd, new_pos_embd->ne[0], 1,
                                        ggml_row_size(new_pos_embd->type, new_pos_embd->ne[0]), src_size * src_size);
            new_pos_embd = ggml_interpolate(ctx0, old_pos_embd, tgt_size, tgt_size, new_pos_embd->ne[0], 1,
                                            GGML_SCALE_MODE_BICUBIC);
            new_pos_embd = ggml_reshape_3d(ctx0, new_pos_embd, n_embd, tgt_size * tgt_size, 1);
            new_pos_embd = ggml_concat(ctx0, new_pos_embd, cls_tok, 1);
            n_pos        = tgt_size * tgt_size + 1;
        }

        // add CLS token
        inp = ggml_concat(ctx0, model.class_embedding, inp, 1);

        // for selecting learned pos embd, used by ViT
        ggml_tensor * positions        = ggml_cast(ctx0, ggml_arange(ctx0, 0, n_pos, 1), GGML_TYPE_I32);
        ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, new_pos_embd, positions);

        ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, FFN_GELU_QUICK, learned_pos_embd, nullptr);

        ggml_build_forward_expand(gf, cur);
        clip_out = cur;
    }

    const int clip_n_patches = sam_out->ne[0] * sam_out->ne[1];

    sam_out  = ggml_cont(ctx0, ggml_permute(ctx0, sam_out, 1, 2, 0, 3));
    sam_out  = ggml_reshape_2d(ctx0, sam_out, sam_out->ne[0], clip_n_patches);
    clip_out = ggml_view_2d(ctx0, clip_out, n_embd, clip_n_patches, clip_out->nb[1], clip_out->nb[1]);

    ggml_tensor * cur;
    cur = ggml_concat(ctx0, clip_out, sam_out, 0);
    cur = ggml_reshape_2d(ctx0, cur, 2 * n_embd, clip_n_patches);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model.fc_w, cur);
    cur = ggml_add(ctx0, cur, model.fc_b);

    const auto h     = static_cast<int>(std::sqrt(static_cast<float>(cur->ne[1])));
    const auto w     = h;
    const auto n_dim = cur->ne[0];

    ggml_tensor * imgnl;
    ggml_tensor * vs;

    imgnl = ggml_repeat_4d(ctx0, model.image_newline, n_dim, 1, h, 1);
    vs    = ggml_reshape_2d(ctx0, model.view_seperator, n_dim, 1);  // (n_dim, 1)
    cur   = ggml_reshape_3d(ctx0, cur, n_dim, w, h);
    cur   = ggml_reshape_2d(ctx0, ggml_concat(ctx0, cur, imgnl, 1), n_dim, (w + 1) * h);
    cur   = ggml_concat(ctx0, cur, vs, 1);  // (n_dim, h*(w+1) + 1)

    cb(cur, "dsocr_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
