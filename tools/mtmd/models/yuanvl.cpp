#include "models.h"

ggml_cgraph * clip_graph_yuanvl::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1;
    ggml_tensor * inp = build_inp();

    // add CLS token
    inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

    // InternViT 300M uses layer norm
    norm_type norm_t = NORM_TYPE_NORMAL;

    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            norm_t,
                            hparams.ffn_op,
                            model.position_embeddings,
                            nullptr);

    // remove CLS token
    cur = ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        ggml_row_size(cur->type, n_embd), 0);

    // PixelUnshuffle (torch.nn.PixelUnshuffle with downscale_factor=2)
    //
    // Yuan uses PixelUnshuffle which differs from InternVL's pixel_shuffle
    // in channel ordering. Both merge the same 2x2 spatial blocks, but:
    //   InternVL groups:  [all_C_block0, all_C_block1, all_C_block2, all_C_block3]
    //   PixelUnshuffle:   [b0_c0, b1_c0, b2_c0, b3_c0, b0_c1, b1_c1, ...]
    //
    // We first do InternVL-style merging, then fix the channel ordering with
    // a reshape+permute: (r*r, C, tokens) -> permute(1,0,2) -> (C, r*r, tokens)
    {
        const int scale_factor = model.hparams.n_merge;
        const int bsz    = 1;
        const int height = n_patches_y;
        const int width  = n_patches_x;
        GGML_ASSERT(scale_factor > 0);

        // InternVL-style spatial merging
        cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            height / scale_factor,
            width / scale_factor,
            bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D: (n_embd * r * r, n_output_patches)
        cur = ggml_cont_2d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            cur->ne[1] * cur->ne[2]);

        // Fix channel ordering: InternVL -> PixelUnshuffle
        // reshape (r*r * C, tokens) -> (r*r, C, tokens)
        const int n_out_patches = cur->ne[1];
        const int r2 = scale_factor * scale_factor;
        cur = ggml_reshape_3d(ctx0, cur, n_embd, r2, n_out_patches);
        // permute to (C, r*r, tokens) -- ggml dim order, so permute(1, 0, 2)
        cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
        cur = ggml_cont_2d(ctx0, cur, n_embd * r2, n_out_patches);
    }

    // projector: SwiGLU MLP + RMSNorm
    // ref: Yuan3.0-Flash modeling_yuanvl_chat.py YuanImageMLP
    //   x1 = up_proj(x)       # 4096 -> 8192
    //   x2 = gate_proj(x)     # 4096 -> 8192
    //   x3 = silu(x1) * x2    # SwiGLU
    //   x  = down_proj(x3)    # 8192 -> 2048
    //   x  = rms_norm(x)
    //
    // Note: Yuan applies silu to up_proj (not gate_proj like standard LLaMA).
    // In build_ffn, silu is applied to the 'gate' parameter, so we swap:
    //   gate param = up_proj weight (gets silu)
    //   up param   = gate_proj weight (multiplied)
    {
        cur = build_ffn(cur,
            model.mm_ffn_gate_w, nullptr,  // 'up' param = gate_proj (no silu)
            model.mm_ffn_up_w, nullptr,    // 'gate' param = up_proj (gets silu)
            model.mm_ffn_down_w, nullptr,
            FFN_SILU,
            -1);

        // RMSNorm (imagemlp_layernorm)
        cur = build_norm(cur, model.mm_post_norm_w, nullptr, NORM_TYPE_RMS, 1e-6, -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
