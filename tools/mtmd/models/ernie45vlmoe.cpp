#include "models.h"

ggml_cgraph * clip_graph_ernie45vlmoe::build() {
    // ERNIE-4.5-VL-MoE Vision + Resampler:
    // 1. ViT encoder with 2D position embeddings and M-RoPE support
    // 2. Resampler with spatial conv (2x2 grouping) + optional temporal + MLP + RMS norm

    const int n_pos             = n_patches;
    const int spatial_conv_size = hparams.spatial_conv_size;  // 2

    GGML_ASSERT(spatial_conv_size == 2 && "ERNIE-4.5-VL-MoE requires spatial_conv_size=2");

    // ERNIE-VL Vision uses 2D position lookup RoPE:
    // - Front half of frequencies use h_position
    // - Back half of frequencies use w_position
    // For d_head=80, n_dims=40, we need sections[0]=20 (for h) and sections[1]=20 (for w)
    // GGML_ROPE_TYPE_VISION uses only 2 sections: sect_0 for first pos slot, sect_1 for second
    int mrope_sections[4] = {d_head/4, d_head/4, 0, 0};  // [20, 20, 0, 0] for d_head=80

    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * inp = build_inp();

    // Build ViT encoder using the generic build_vit() with M-RoPE position encoding
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    ggml_tensor * embeddings = build_vit(
                                    inp, n_pos,
                                    NORM_TYPE_NORMAL,
                                    hparams.ffn_op,
                                    nullptr,  // no learned position embeddings, using RoPE
                                    add_pos);
    cb(embeddings, "vision_output", -1);

    // -------------------------------------------
    // Resampler projection
    // -------------------------------------------
    // Group 2x2 patches: 40x40 -> 20x20, output shape [n_embd*4, n_groups]
    embeddings = build_patch_merge_permute(embeddings, spatial_conv_size);
    cb(embeddings, "spatial_reshape", -1);

    // Spatial linear path: Linear -> GELU -> Linear -> LayerNorm
    // Note: weights were transposed (.t()) during GGUF conversion, so we must
    // undo that with ggml_transpose before ggml_mul_mat
    ggml_tensor * spatial_out = embeddings;

    // First linear
    ggml_tensor * spatial_0_w = ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_0_w));
    spatial_out = ggml_mul_mat(ctx0, spatial_0_w, spatial_out);
    spatial_out = ggml_add(ctx0, spatial_out, model.mm_0_b);
    cb(spatial_out, "spatial_linear_0", -1);

    // GELU
    spatial_out = ggml_gelu(ctx0, spatial_out);
    cb(spatial_out, "spatial_gelu", -1);

    // Second linear
    ggml_tensor * spatial_2_w = ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_2_w));
    spatial_out = ggml_mul_mat(ctx0, spatial_2_w, spatial_out);
    spatial_out = ggml_add(ctx0, spatial_out, model.mm_2_b);
    cb(spatial_out, "spatial_linear_2", -1);

    // LayerNorm
    spatial_out = build_norm(spatial_out, model.mm_post_norm_w, model.mm_post_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cb(spatial_out, "spatial_norm", -1);

    ggml_tensor * resampler_out = spatial_out;

    // Temporal processing for single images (t=1):
    // Following ERNIE-VL original: when t=1, slice_offsets and slice_offsets2 both point to the same frame
    // So we concat(x, x, dim=-1) which in GGML's [hidden, seq] layout is dim=0
    // This doubles the hidden dimension: [5120, 400] -> [10240, 400]
    resampler_out = ggml_concat(ctx0, resampler_out, resampler_out, 0);

    // Temporal linear path: Linear -> GELU -> Linear -> LayerNorm
    // Weights were transposed (.t()) during GGUF conversion, undo with ggml_transpose

    // First temporal linear
    ggml_tensor * temp_0_w = ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_1_w));
    resampler_out = ggml_mul_mat(ctx0, temp_0_w, resampler_out);
    resampler_out = ggml_add(ctx0, resampler_out, model.mm_1_b);
    cb(resampler_out, "temporal_linear_0", -1);

    // GELU
    resampler_out = ggml_gelu(ctx0, resampler_out);
    cb(resampler_out, "temporal_gelu", -1);

    // Second temporal linear
    ggml_tensor * temp_2_w = ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_3_w));
    resampler_out = ggml_mul_mat(ctx0, temp_2_w, resampler_out);
    resampler_out = ggml_add(ctx0, resampler_out, model.mm_3_b);
    cb(resampler_out, "temporal_linear_2", -1);

    // LayerNorm
    resampler_out = build_norm(resampler_out, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cb(resampler_out, "temporal_norm", -1);

    // Final MLP
    ggml_tensor * mlp_w = ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_fc_w));
    resampler_out = ggml_mul_mat(ctx0, mlp_w, resampler_out);
    resampler_out = ggml_add(ctx0, resampler_out, model.mm_fc_b);
    cb(resampler_out, "mlp", -1);

    // RMS norm (final output normalization)
    resampler_out = build_norm(resampler_out, model.mm_norm_mid_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cb(resampler_out, "after_norm", -1);

    // Build the graph
    ggml_build_forward_expand(gf, resampler_out);

    return gf;
}
