#include "models.h"

ggml_cgraph * clip_graph_ling3vl::build() {
    // same vision tower as qwen3vl, but the merger is norm-only (no fc1/fc2) and
    // the projector MLP lives at the top level (mm.0 / mm.2)
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.mm_input_norm_w != nullptr); // merger norm (pre spatial merge)

    const int batch_size = 1;
    const int n_pos      = n_patches;

    norm_type norm_t = NORM_TYPE_NORMAL;

    // vision M-RoPE, same layout as qwen3vl: [row, col, row, col] quarters
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp_with_temporal_merge();

    // spatial merge
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }

    // calculate absolute position embedding and apply
    ggml_tensor * learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    learned_pos_embd = ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, batch_size);

    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * inpL = build_vit(
        inp, n_pos, norm_t, hparams.ffn_op, learned_pos_embd,
        [&](ggml_tensor * c, const clip_layer &) {
            return ggml_rope_multi(
                ctx0, c, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
        });

    // multimodal projection (linear_proj MLP over the merged patches)
    ggml_tensor * embeddings = inpL;

    // per-patch merger norm, applied post-blocks before the 2x2 merge
    // (merger.norm, LayerNorm over n_embd)
    embeddings = build_norm(embeddings, model.mm_input_norm_w, model.mm_input_norm_b, norm_t, eps, -1);
    cb(embeddings, "merger_norm", -1);

    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        ffn_op_type::FFN_GELU, -1);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
