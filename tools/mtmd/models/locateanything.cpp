#include "models.h"

// LocateAnything-3B vision graph.
//
// The encoder is MoonViT-SO-400M, identical to the one Kimi-K2.5 uses, so we inherit
// clip_graph_kimik25 and reuse its 3D position-embedding resize and the interleaved->split
// RoPE convention (Q/K are permuted at conversion time, so build_rope_2d runs in split mode).
//
// The connector is the "Eagle MLP": LayerNorm(4608) -> Linear(4608, 2048) -> GELU -> Linear(2048, 2048).
// Unlike Kimi-K2.5 (whose projection norm is applied per 1152-dim sub-token before the 2x2 merge),
// the LayerNorm here normalises the full merged 4608-dim feature, so the build() below drops the
// sub-token view trick and normalises the merged axis directly.
ggml_cgraph * clip_graph_locateanything::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings_3d(GGML_SCALE_MODE_BICUBIC);

    // MoonViT uses interleaved 2D RoPE natively; Q/K are permuted to split format during
    // conversion, so build_rope_2d is called in split mode (interleave = false).
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        cur = build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
        return cur;
    };

    ggml_tensor * inp = build_inp();

    // add the learned position embedding before the encoder (mirrors clip_graph_kimik25::build)
    inp = ggml_add(ctx0, inp, learned_pos_embd);

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // 2x2 patch merge -> [4608, n_merged_patches]
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // Eagle-MLP projection norm: LayerNorm over the merged 4608 axis (no per-sub-token view).
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        cb(cur, "proj_inp_normed", -1);

        // projection mlp: Linear -> GELU -> Linear
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

        cb(cur, "proj_out", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}
