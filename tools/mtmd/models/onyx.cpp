#include "models.h"

// Onyx vision encoder: 50-layer ViT with 2D RoPE, sparse block-diagonal
// window attention (every 4th + last layer global), pixel-shuffle downsample, then
// adapter MLP + LLM's vision_projection. Output dim = 6656 (onyx n_embd),
// injected via llama_batch.embd; the onyx LLM graph applies the scaleless rms_norm
// (== reference perception_emb_norm).
//
// Several quantities are precomputed on host and fed as named graph inputs (filled in
// clip.cpp set_input, PROJECTOR_TYPE_ONYX branch):
//   onyx_patches  [patch_dim, n_tok]  : patchified pixels ([pt,c,ps,ps] layout)
//   onyx_pos_emb  [n_embd,   n_tok]   : bilinear-interpolated learned pos-emb (orig order)
//   onyx_pos_w/_h [n_tok] i32         : 1-indexed RoPE positions (sparse-permuted order)
//   onyx_sp_perm  [n_tok] i32         : window grouping permutation (applied after ln_pre)
//   onyx_inv_perm [n_tok] i32         : inverse of sp_perm (applied after blocks)
//   onyx_ds_perm  [n_tok] i32         : pixel-shuffle gather (original order)
//   onyx_sp_mask  [n_tok, n_tok] f32  : block-diagonal window mask (sparse layers)
ggml_cgraph * clip_graph_onyx::build() {
    const int ds = hparams.n_merge;              // downsample factor (2)
    const int pt = hparams.onyx_patch_temporal;  // 2
    const int sf = hparams.onyx_sparse_factor;   // 4
    const int n_tok     = n_patches;
    const int patch_dim = pt * 3 * patch_size * patch_size;       // 1176
    const int n_out     = (n_patches_x / ds) * (n_patches_y / ds);
    const float rope_base  = hparams.rope_theta;                  // 10000
    const float attn_scale = 1.0f / sqrtf((float) d_head);        // SDPA default

    auto inp_i32 = [&](const char * name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
        ggml_set_name(t, name); ggml_set_input(t);
        return t;
    };

    ggml_tensor * patches = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, patch_dim, n_tok);
    ggml_set_name(patches, "onyx_patches"); ggml_set_input(patches);

    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tok);
    ggml_set_name(pos_emb, "onyx_pos_emb"); ggml_set_input(pos_emb);

    ggml_tensor * pos_w   = inp_i32("onyx_pos_w",   n_tok);
    ggml_tensor * pos_h   = inp_i32("onyx_pos_h",   n_tok);
    ggml_tensor * sp_perm = inp_i32("onyx_sp_perm", n_tok);
    ggml_tensor * inv_perm = inp_i32("onyx_inv_perm", n_tok);
    ggml_tensor * ds_perm = inp_i32("onyx_ds_perm", n_tok);

    ggml_tensor * sp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tok, n_tok);
    ggml_set_name(sp_mask, "onyx_sp_mask"); ggml_set_input(sp_mask);

    // patchify (conv1_linear as a matmul, no bias) + learned pos-emb
    ggml_tensor * x = build_mm(model.patch_embeddings_0, patches); // [n_embd, n_tok]
    x = ggml_add(ctx0, x, pos_emb);
    cb(x, "after_posemb", -1);

    // ln_pre (LayerNorm)
    x = build_norm(x, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);

    // group patches into 32x32 windows (sparse attention order)
    x = ggml_get_rows(ctx0, x, sp_perm);
    cb(x, "after_ln_pre", -1);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool is_global = (il == n_layer - 1) || ((il + 1) % sf == 0);

        ggml_tensor * inpL = x;

        ggml_tensor * cur = build_norm(x, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        ggml_tensor * Q = ggml_add(ctx0, build_mm(layer.q_w, cur), layer.q_b);
        ggml_tensor * K = ggml_add(ctx0, build_mm(layer.k_w, cur), layer.k_b);
        ggml_tensor * V = ggml_add(ctx0, build_mm(layer.v_w, cur), layer.v_b);

        Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, n_tok);
        K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_tok);
        V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_tok);

        // 2D RoPE: first half of head_dim uses width pos, second half uses height pos
        Q = build_rope_2d(ctx0, Q, pos_w, pos_h, rope_base, false);
        K = build_rope_2d(ctx0, K, pos_w, pos_h, rope_base, false);

        ggml_tensor * mask = is_global ? nullptr : sp_mask;
        cur = build_attn(layer.o_w, layer.o_b, Q, K, V, mask, attn_scale, il);

        x = ggml_add(ctx0, inpL, cur);   // residual 1
        inpL = x;

        cur = build_norm(x, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur,
            layer.ff_up_w,   layer.ff_up_b,
            nullptr,         nullptr,
            layer.ff_down_w, layer.ff_down_b,
            FFN_GELU_ERF, il);           // reference uses exact (erf) GELU
        x = ggml_add(ctx0, inpL, cur);   // residual 2
        cb(x, "layer_out", il);
    }

    // un-permute back to original grid order, then ln_post
    x = ggml_get_rows(ctx0, x, inv_perm);
    x = build_norm(x, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    cb(x, "after_ln_post", -1);

    // pixel-shuffle downsample: gather f*f spatial neighbors then concat channel-outer.
    // out[c*(ds*ds)+s, o] = x[ds_perm gathered][o*(ds*ds)+s, c]
    x = ggml_get_rows(ctx0, x, ds_perm);                 // [n_embd, n_tok], grouped
    x = ggml_reshape_3d(ctx0, x, n_embd, ds * ds, n_out);// [c, s, o]
    x = ggml_permute(ctx0, x, 1, 0, 2, 3);               // [s, c, o]
    x = ggml_cont(ctx0, x);
    x = ggml_reshape_2d(ctx0, x, n_embd * ds * ds, n_out); // [6144, n_out]
    cb(x, "encoder_out", -1);

    // adapter (6144->4096->4096, exact GELU each) + LLM vision_projection (4096->6656)
    x = build_mm(model.mm_adapter_fc, x);
    x = ggml_gelu_erf(ctx0, x);
    x = build_mm(model.mm_adapter_proj, x);
    x = ggml_gelu_erf(ctx0, x);
    x = build_mm(model.mm_vision_proj, x);               // [6656, n_out]
    cb(x, "projected", -1);

    ggml_build_forward_expand(gf, x);
    return gf;
}
