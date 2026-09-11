#include "models.h"

// MuseGlimmer vision encoder: 50-layer ViT with 2D RoPE, sparse block-diagonal
// window attention (every 4th + last layer global), pixel-shuffle downsample, then
// adapter MLP + LLM's vision_projection.
//
// Several quantities are precomputed on host and fed as named graph inputs (filled in
// clip.cpp set_input, PROJECTOR_TYPE_MUSE_GLIMMER branch):
//   muse_glimmer_pos_w/_h [n_tok] i32         : 1-indexed RoPE positions (sparse-permuted order)
//   muse_glimmer_sp_perm  [n_tok] i32         : window grouping permutation (applied after ln_pre)
//   muse_glimmer_inv_perm [n_tok] i32         : inverse of sp_perm (applied after blocks)
//   muse_glimmer_ds_perm  [n_tok] i32         : pixel-shuffle gather (original order)
//   muse_glimmer_sp_mask  [n_tok, n_tok] f32  : block-diagonal window mask (sparse layers)
ggml_cgraph * clip_graph_muse_glimmer::build() {
    const int ds = hparams.n_merge;              // downsample factor (2)
    const int sf = hparams.muse_glimmer_sparse_factor;   // 4
    const int n_tok     = n_patches;
    const int n_out     = (n_patches_x / ds) * (n_patches_y / ds);
    const float rope_base = hparams.rope_theta;  // 10000

    auto inp_i32 = [&](const char * name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    ggml_tensor * pos_w    = inp_i32("muse_glimmer_pos_w",    n_tok);
    ggml_tensor * pos_h    = inp_i32("muse_glimmer_pos_h",    n_tok);
    ggml_tensor * sp_perm  = inp_i32("muse_glimmer_sp_perm",  n_tok);
    ggml_tensor * inv_perm = inp_i32("muse_glimmer_inv_perm", n_tok);
    ggml_tensor * ds_perm  = inp_i32("muse_glimmer_ds_perm",  n_tok);

    ggml_tensor * sp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tok, n_tok);
    ggml_set_name(sp_mask, "muse_glimmer_sp_mask");
    ggml_set_input(sp_mask);

    // patchify via build_inp (conv2d over raw pixels) + bilinear-resized learned pos-emb
    ggml_tensor * x = build_inp();                                                     // [n_embd, n_tok, 1]
    x = ggml_add(ctx0, x, resize_position_embeddings(GGML_SCALE_MODE_BILINEAR));
    cb(x, "after_posemb", -1);

    // group patches into pgrid x pgrid windows (sparse attention order)
    x = ggml_get_rows(ctx0, x, sp_perm);
    cb(x, "after_sp_perm", -1);

    // per-layer mask: sparse layers get sp_mask, global layers (every sf-th and last) get none
    std::vector<ggml_tensor *> attn_mask_layers(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const bool is_global = (il == n_layer - 1) || ((il + 1) % sf == 0);
        attn_mask_layers[il] = is_global ? nullptr : sp_mask;
    }

    // 2D RoPE: first half of head_dim uses width pos, second half uses height pos
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, rope_base, false);
    };

    build_vit_opts opts;
    opts.attn_mask_layers = std::move(attn_mask_layers);

    // pre_ln, per-layer transformer, post_ln (all inside build_vit); reference uses exact (erf) GELU
    x = build_vit(x, n_tok, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr, add_pos, opts);

    // un-permute back to original grid order
    x = ggml_get_rows(ctx0, x, inv_perm);
    cb(x, "after_inv_perm", -1);

    // pixel-shuffle downsample: gather f*f spatial neighbors then concat channel-outer.
    // out[c*(ds*ds)+s, o] = x[ds_perm gathered][o*(ds*ds)+s, c]
    x = ggml_get_rows(ctx0, x, ds_perm);                 // [n_embd, n_tok], grouped
    x = ggml_reshape_3d(ctx0, x, n_embd, ds * ds, n_out);// [c, s, o]
    x = ggml_permute(ctx0, x, 1, 0, 2, 3);               // [s, c, o]
    x = ggml_cont(ctx0, x);
    x = ggml_reshape_2d(ctx0, x, n_embd * ds * ds, n_out); // [6144, n_out]
    cb(x, "encoder_out", -1);

    // adapter (6144->4096->4096, exact GELU each) + LLM vision_projection (4096->6656)
    x = build_mm(model.mm_0_w, x);
    x = ggml_gelu_erf(ctx0, x);
    x = build_mm(model.mm_1_w, x);
    x = ggml_gelu_erf(ctx0, x);
    x = build_mm(model.mm_2_w, x);                       // [6656, n_out]
    cb(x, "projected", -1);

    ggml_build_forward_expand(gf, x);
    return gf;
}
