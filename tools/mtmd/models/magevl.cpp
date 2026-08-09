#include "models.h"

// Mage-VL vision tower (projector_type = "magevl")
//
// Architecture (from modeling_mage_vl.py):
//   - patch embed: Conv2d(patch=16, stride=16, no bias), embed_dim=1024
//   - layernorm_pre (LayerNorm with bias, eps=1e-6)
//   - 24 x encoder layer: pre-LN -> MHA(fused qkv, with bias) -> residual
//                         pre-LN -> SiglipMLP(fc1 gelu-erf fc2) -> residual
//   - patch merger: LayerNorm(ln_q) -> view(-1, 4096) -> Linear(4096,4096)
//                   -> GELU(erf) -> Linear(4096,2560)
//
// RoPE (MUST replicate the reference implementation bit-exactly):
//   - 3D (t,h,w) freqs with 4:6:6 split of half=head_dim/2=32 -> 8/12/12 sections
//   - inv_freq_sec[k] = rope_theta^(-k/size_sec), k = 0..size-1
//   - per-patch f[32] = [t*inv_t(8), h*inv_h(12), w*inv_w(12)]
//   - cos/sin vectors are cat([f, f]) -> 64 dims
//   - rotation is the *interleaved* rotate_half: (x1,x2,x3,x4) -> (-x2,x1,-x4,x3)
//     Note: interleaved rotation + cat([f,f]) is an unusual combination, but it is
//     exactly what the reference model does, so we reproduce it as-is:
//       q_embed = q * cos + (R @ q) * sin
//     where R is the constant 64x64 rotation matrix stored in the mmproj GGUF
//     as "v.rope_rotmat.weight" (ggml_mul_mat(rotmat, q) computes R @ q).
//
// Patch ordering: the HF processor emits patches in 2x2-block order
// (hb, wb, dy, dx). Here the conv output is row-major, so we reorder with a
// precomputed permutation index ("patch_perm" input, filled on CPU). The same
// block order is used for rope cos/sin and the 4-patch merger grouping.

ggml_cgraph * clip_graph_magevl::build() {
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.rope_rotmat != nullptr);
    GGML_ASSERT(n_batch == 1); // one frame per encode call

    const int n_pos = n_patches;
    const int pw    = n_patches_x;
    const int ph    = n_patches_y;
    GGML_ASSERT(pw % 2 == 0 && ph % 2 == 0);

    // ---- patch embedding: conv2d 16x16/s16 ----
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * conv = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw,
                                      patch_size, patch_size, 0, 0, 1, 1);
    // conv: [pw, ph, n_embd] (x fastest) -> [n_embd, pw, ph] -> [n_embd, n_pos] row-major
    // (ggml_permute 语义: result->ne[axes[i]] = a->ne[i]，同 qwen2vl 的注释约定)
    ggml_tensor * patches = ggml_permute(ctx0, conv, 1, 2, 0, 3);
    patches = ggml_cont_2d(ctx0, patches, n_embd, n_pos);

    // ---- reorder to 2x2-block layout (matches HF processor) ----
    ggml_tensor * patch_perm = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(patch_perm, "patch_perm");
    ggml_set_input(patch_perm);
    ggml_tensor * inpL = ggml_get_rows(ctx0, patches, patch_perm); // [n_embd, n_pos] block order

    // ---- rope cos/sin inputs (computed on CPU, block order) ----
    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head, n_pos);
    ggml_set_name(rope_cos, "rope_cos");
    ggml_set_input(rope_cos);
    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head, n_pos);
    ggml_set_name(rope_sin, "rope_sin");
    ggml_set_input(rope_sin);

    // ---- layernorm_pre ----
    inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);

    // rope application helper: x is [d_head, n_head, n_pos]
    auto apply_rope = [&](ggml_tensor * x) -> ggml_tensor * {
        ggml_tensor * xp = ggml_permute(ctx0, x, 0, 2, 1, 3);       // [d_head, n_pos, n_head]
        xp = ggml_cont_3d(ctx0, xp, d_head, n_pos, n_head);
        ggml_tensor * cos3 = ggml_reshape_3d(ctx0, rope_cos, d_head, n_pos, 1);
        ggml_tensor * sin3 = ggml_reshape_3d(ctx0, rope_sin, d_head, n_pos, 1);
        ggml_tensor * xrot = ggml_mul_mat(ctx0, model.rope_rotmat, xp); // R @ x
        ggml_tensor * out  = ggml_add(ctx0,
                              ggml_mul(ctx0, xp,   cos3),
                              ggml_mul(ctx0, xrot, sin3));
        out = ggml_permute(ctx0, out, 0, 2, 1, 3);                  // back to [d_head, n_head, n_pos]
        return ggml_cont_3d(ctx0, out, d_head, n_head, n_pos);
    };

    // ---- encoder layers ----
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        // pre-attn norm
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        // fused qkv: [3*n_embd, n_pos]
        ggml_tensor * qkv = ggml_add(ctx0, build_mm(layer.qkv_w, cur), layer.qkv_b);

        // split into q/k/v: rows [0, n_embd), [n_embd, 2*n_embd), [2*n_embd, 3*n_embd)
        const size_t qkv_nb1 = qkv->nb[1];
        ggml_tensor * Q = ggml_cont_3d(ctx0,
            ggml_view_2d(ctx0, qkv, n_embd, n_pos, qkv_nb1, 0),
            d_head, n_head, n_pos);
        ggml_tensor * K = ggml_cont_3d(ctx0,
            ggml_view_2d(ctx0, qkv, n_embd, n_pos, qkv_nb1, n_embd * ggml_element_size(qkv)),
            d_head, n_head, n_pos);
        ggml_tensor * V = ggml_cont_3d(ctx0,
            ggml_view_2d(ctx0, qkv, n_embd, n_pos, qkv_nb1, 2 * n_embd * ggml_element_size(qkv)),
            d_head, n_head, n_pos);

        Q = apply_rope(Q);
        K = apply_rope(K);

        // full attention within the frame (no mask; per-frame encode == cu_seqlens path)
        cur = build_attn(layer.o_w, layer.o_b, Q, K, V, nullptr, kq_scale, il);

        // residual 1
        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        // pre-ffn norm + SiglipMLP (gelu erf)
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            FFN_GELU_ERF, il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        inpL = cur;
    }

    // ---- patch merger ----
    // ln_q over n_embd, then group 4 consecutive (block-order) patches
    ggml_tensor * embeddings = build_norm(inpL, model.mm_input_norm_w, model.mm_input_norm_b,
                                          NORM_TYPE_NORMAL, eps, -1);
    embeddings = ggml_reshape_2d(ctx0, embeddings, n_embd * 4, n_pos / 4);
    embeddings = build_ffn(embeddings,
                    model.mm_0_w, model.mm_0_b,
                    nullptr, nullptr,
                    model.mm_1_w, model.mm_1_b,
                    FFN_GELU_ERF, -1);

    ggml_build_forward_expand(gf, embeddings);
    return gf;
}
