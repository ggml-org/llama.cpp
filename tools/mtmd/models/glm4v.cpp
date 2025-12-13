#include "models.h"

ggml_cgraph * clip_graph_glm4v::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.patch_embeddings_1 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    // M-RoPE input positions (same pattern as Qwen2VL)
    // Format: [h0,h1,...,hN, w0,w1,...,wN, h0,h1,...,hN, w0,w1,...,wN]
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // GLM4V Patch Embedding using Conv2D on raw image
    // Reference: modeling_glm4v.py Glm4vVisionPatchEmbed.forward()
    //
    // HF uses Conv3d with temporal_patch_size=2 for video support.
    // For single images, HF duplicates the frame to create [2, C, H, W] input.
    // The Conv3d kernel is split into two temporal slices and summed.
    //
    // Since frame0 == frame1 for single images, this is equivalent to:
    //   conv2d(kernel_t0, img) + conv2d(kernel_t1, img)
    // which is the same pattern as Qwen2VL dual conv2d.

    ggml_tensor * inp_raw = build_inp_raw();

    // Apply both temporal kernel slices to the same image
    ggml_tensor * out0 = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    ggml_tensor * out1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    cb(out0, "conv_out0", -1);
    cb(out1, "conv_out1", -1);

    // Sum temporal frames (simulates Conv3d temporal reduction)
    ggml_tensor * inp = ggml_add(ctx0, out0, out1);

    // Reshape from conv2d output [1, n_patches_y, n_patches_x, n_embd] to [n_embd, n_patches]
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // Add patch embedding bias (Conv3d bias)
    // ref: self.proj.bias in Glm4vVisionPatchEmbed
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    cb(inp, "patch_embed", -1);

    // post-convolution layernorm
    // ref: self.post_conv_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    inp = build_norm(inp, model.mm_post_conv_ln_w, model.mm_post_conv_ln_b, NORM_TYPE_RMS, eps, -1);
    cb(inp, "post_conv_ln", -1);

    // absolute position embeddings (interpolated)
    // ref: self.embeddings
    ggml_tensor * learned_pos_embd = resize_position_embeddings();
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "abs_pos_embed", -1);

    // RoPE to be applied inside ViT blocks
    // Uses M-RoPE (same as Qwen2VL) with [h, w, h, w] pattern at 32-dim chunks
    // ref: self.rotary_pos_emb, apply_rotary_pos_emb_vision (identical to Qwen2VL)
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(ctx0, cur, positions, nullptr,
            d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
            32768, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    };

    // ViT blocks
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            FFN_SILU, // hidden_act is "silu"
                            nullptr,  // absolute embeddings already added
                            add_pos);

    // post-ViT layernorm
    cur = build_norm(cur, model.post_ln_w, model.post_ln_b, NORM_TYPE_RMS, eps, -1);
    cb(cur, "post_vit_ln", -1);

    // Patch merger downsampling - EXACT HF implementation
    // HF: hidden_states [576, 1536] -> view(-1, 2, 2, 1536) -> [144, 2, 2, 1536]
    //     -> permute(0, 3, 1, 2) -> [144, 1536, 2, 2]
    //     -> Conv2d(1536, 4096, kernel=2, stride=2) -> [144, 4096, 1, 1]
    //     -> flatten -> [144, 4096]
    //
    // In ggml (dimensions reversed):
    // Input: [1536, 576]
    // reshape_4d: [1536, 2, 2, 144]  (reversed from HF [144, 2, 2, 1536])
    // Need permute to: [2, 2, 1536, 144] (reversed from HF [144, 1536, 2, 2])
    // Conv2d output: [1, 1, 4096, 144]
    // Final: [4096, 144]
    const int merge_size = 2;
    const int num_merge_blocks = n_patches / (merge_size * merge_size);  // 576 / 4 = 144

    // Reshape to 4D: [1536, 2, 2, 144]
    cur = ggml_reshape_4d(ctx0, cur, n_embd, merge_size, merge_size, num_merge_blocks);

    // Permute to [2, 2, 1536, 144] for conv2d
    // ggml_permute(a,b,c,d): axis0->a, axis1->b, axis2->c, axis3->d
    // From [1536, 2, 2, 144] to [2, 2, 1536, 144]:
    //   axis0(1536)->2, axis1(2)->0, axis2(2)->1, axis3(144)->3
    cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
    cur = ggml_cont(ctx0, cur);
    cb(cur, "pre_downsample_permute", -1);

    // downsample conv2d - each 2x2 block -> 1 token with 4096 features
    // Output: [1, 1, 4096, 144]
    cur = ggml_conv_2d(ctx0, model.mm_downsample_w, cur, merge_size, merge_size, 0, 0, 1, 1);
    cb(cur, "downsample_conv", -1);

    // Reshape to [4096, 144] for ggml_mul_mat
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[2], cur->ne[3]);
    cb(cur, "post_downsample_reshape", -1);

    // patch merger FFN
    // ref: class Glm4vVisionPatchMerger(nn.Module):
    {
        // input projection
        cur = ggml_mul_mat(ctx0, model.mm_merger_proj_w, cur);

        // apply norm + GELU
        cur = build_norm(cur, model.mm_merger_norm_w, model.mm_merger_norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);
        cur = ggml_gelu(ctx0, cur);
        ggml_tensor * ffn_input = cur;
        cb(cur, "merger_ffn_inp", -1);

        // gate projection
        ggml_tensor * gate = ggml_mul_mat(ctx0, model.mm_merger_gate_w, ffn_input);
        cb(gate, "merger_gate", -1);

        // up projection
        ggml_tensor * up = ggml_mul_mat(ctx0, model.mm_merger_up_w, ffn_input);
        cb(up, "merger_up", -1);

        // activation + down projection
        cur = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, cur, up);
        cur = ggml_mul_mat(ctx0, model.mm_merger_down_w, cur);
        cb(cur, "merger_ffn_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
