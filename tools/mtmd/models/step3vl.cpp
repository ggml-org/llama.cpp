#include "models.h"

// Step3-VL: PE-lang 1.8B ViT encoder with 2D RoPE + abs pos embd
// Architecture: 47 transformer blocks, hidden_size=1536, 16 heads, patch_size=14, image_size=728
// Projector: Conv2d(1536→3072, 3×3, s2) → Conv2d(3072→6144, 3×3, s2) → Linear(6144→4096)
// NO activations between projector layers

ggml_cgraph * clip_graph_step3vl::build() {
    // Step3-VL uses learned positional embeddings + 2D RoPE
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);  // no CLS token

    // 2D input positions for RoPE
    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    // Step3-VL 2D RoPE: first d_head/2 dims get width positions, second half gets height
    // freqs = cat([freqs_w, freqs_h], dim=-1), interleave_freq=false, theta=10000
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    ggml_tensor * inp = build_inp();

    // Build ViT with:
    // - NORM_TYPE_NORMAL (standard LayerNorm, not RMS)
    // - FFN_GELU_QUICK (quick_gelu activation, default when neither use_gelu nor use_silu)
    // - Learned positional embeddings (2704 positions = 52×52 grid)
    // - 2D RoPE callback
    // - Layer scale (v.blk.N.ls1/ls2.weight) is handled automatically by build_vit
    // - Pre-norm (v.pre_ln) is handled automatically by build_vit
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            model.position_embeddings,
                            add_pos);

    // No post-LayerNorm in Step3-VL (use_ln_post=false)

    // Step3-VL projector: two stride-2 Conv2d downsamplers + linear projection
    // NO activations between layers
    {
        // Reshape ViT output from [n_embd, n_patches] to spatial [n_embd, W, H]
        // where W = n_patches_x, H = n_patches_y (both 52 for 728/14)
        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
        // ggml_conv_2d expects [W, H, C, N] layout, but cur is [C, W, H]
        // ggml_permute semantics: input axis i -> output axis perm[i]
        // axis 0 (C) -> pos 2, axis 1 (W) -> pos 0, axis 2 (H) -> pos 1
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
        cur = ggml_cont(ctx0, cur);

        // Downsampler 1: Conv2d(n_embd → n_embd*2, 3×3, stride=2, padding=1)
        // mm.0.weight shape: {3, 3, n_embd, n_embd*2} in GGUF
        cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, 2, 2, 1, 1, 1, 1);
        if (model.mm_0_b) {
            // Reshape bias for broadcasting: [1, 1, n_embd*2, 1]
            ggml_tensor * bias = ggml_reshape_4d(ctx0, model.mm_0_b, 1, 1, model.mm_0_b->ne[0], 1);
            cur = ggml_add(ctx0, cur, bias);
        }

        // Downsampler 2: Conv2d(n_embd*2 → n_embd*4, 3×3, stride=2, padding=1)
        // mm.1.weight shape: {3, 3, n_embd*2, n_embd*4} in GGUF
        cur = ggml_conv_2d(ctx0, model.mm_1_w, cur, 2, 2, 1, 1, 1, 1);
        if (model.mm_1_b) {
            ggml_tensor * bias = ggml_reshape_4d(ctx0, model.mm_1_b, 1, 1, model.mm_1_b->ne[0], 1);
            cur = ggml_add(ctx0, cur, bias);
        }

        // Reshape back to 2D: [n_embd*4, n_output_tokens]
        // After two stride-2 downsamples: spatial dims are n_patches_x/4 × n_patches_y/4
        const int out_x = n_patches_x / 4;
        const int out_y = n_patches_y / 4;
        // cur is [W_out, H_out, C_out, 1], need [C_out, W_out*H_out]
        // axis 0 (W) -> pos 1, axis 1 (H) -> pos 2, axis 2 (C) -> pos 0
        cur = ggml_permute(ctx0, cur, 1, 2, 0, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0], out_x * out_y);

        // Linear projection: mm.2.weight shape [n_embd*4, text_hidden_size]
        cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
