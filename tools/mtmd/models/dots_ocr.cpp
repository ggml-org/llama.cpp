#include "models.h"

/*
 * dots.ocr vision encoder graph builder.
 *
 * Architecture: Qwen2 text backbone + modified Qwen2-VL vision encoder.
 * Key differences from Qwen2-VL vision encoder:
 *   - RMSNorm (not LayerNorm) in vision blocks
 *   - SiLU gated MLP (gate/up/down_proj) instead of GELU (fc1/fc2)
 *   - No attention bias in vision encoder
 *   - Conv2D patch embedding (temporal_patch_size=1, not Conv3D)
 *   - Extra post_trunk_norm (RMSNorm) before merger
 *   - No window attention
 *   - 2D RoPE with sections [d_head/4, d_head/4, 0, 0]
 *
 * Reference: https://github.com/foldl/chatllm.cpp/blob/master/models/dots.cpp
 */

ggml_cgraph * clip_graph_dots_ocr::build() {
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size = 1;
    const int n_pos      = n_patches;

    // dots.ocr uses RMSNorm throughout (not LayerNorm)
    const norm_type norm_t = NORM_TYPE_RMS;

    // dots.ocr 2D RoPE: only uses 2 sections (h, w), sections 2+3 are zero
    GGML_ASSERT(d_head % 4 == 0);
    int mrope_sections[4] = {d_head/4, d_head/4, 0, 0};

    // --- Patch Embedding: single Conv2D (no Conv3D temporal split) ---
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw,
                                      patch_size, patch_size, 0, 0, 1, 1);

    GGML_ASSERT(model.patch_embeddings_1 == nullptr); // no second conv for dots.ocr

    // Add patch embedding bias if present
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }

    // Reshape: [c, w, h, b] -> [n_embd, n_patches, batch]
    inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);
    inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);

    // Patch embedding norm (RMSNorm, no bias)
    if (model.norm_embd_w) {
        inp = build_norm(inp, model.norm_embd_w, model.norm_embd_b, norm_t, eps, -1);
    }

    // --- Position IDs ---
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // --- 2D M-RoPE position encoding ---
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    // --- Vision Transformer Blocks (via build_vit) ---
    // build_vit handles: pre_ln (nullptr for dots.ocr), VIT loop, post_ln (post_trunk_norm)
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            norm_t,
                            hparams.ffn_op,
                            nullptr,   // no learned position embeddings
                            add_pos);

    cb(cur, "vit_out", -1);

    // --- Merger: LayerNorm + MLP (same structure as Qwen2-VL) ---
    // mm_input_norm_w holds the merger's LayerNorm (visual.merger.ln_q)
    if (model.mm_input_norm_w) {
        cur = build_norm(cur, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, eps, -1);
        cb(cur, "merger_normed", -1);
    }

    // Spatial merge: group 2x2 patches -> project to text hidden size
    GGML_ASSERT(n_pos % 4 == 0);
    ggml_tensor * embeddings = cur;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    embeddings = build_ffn(embeddings,
                        model.mm_0_w, model.mm_0_b,
                        nullptr, nullptr,
                        model.mm_1_w, model.mm_1_b,
                        FFN_GELU,
                        -1);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
