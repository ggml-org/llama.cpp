#include "models.h"

ggml_cgraph * mtmd_graph_magevl::build() {
    GGML_ASSERT(hparams.n_merge == 2);

    const int n_pair = d_head / 2;
    const int unit   = n_pair / 16;

    // pairs are grouped as t, h, w at conversion
    int sections[4] = {4 * unit, 6 * unit, 6 * unit, 0};

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // a pair uses 2 angles: even outputs come from one rope, odd outputs from the other
    auto add_pos = [&](ggml_tensor * cur, const mtmd_layer &) {
        ggml_tensor * freqs_a = ggml_view_1d(ctx0, model.rope_freqs, n_pair, 0);
        ggml_tensor * freqs_b = ggml_view_1d(ctx0, model.rope_freqs, n_pair, ggml_row_size(model.rope_freqs->type, n_pair));

        ggml_tensor * rope_a = ggml_rope_multi(ctx0, cur, positions, freqs_a, d_head, sections,
                GGML_ROPE_TYPE_MROPE, 32768, 1.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor * rope_b = ggml_rope_multi(ctx0, cur, positions, freqs_b, d_head, sections,
                GGML_ROPE_TYPE_MROPE, 32768, 1.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        ggml_tensor * even = ggml_view_3d(ctx0, rope_a, n_pair, n_head, n_patches,
                rope_a->nb[1], rope_a->nb[2], 0);
        ggml_tensor * odd  = ggml_view_3d(ctx0, rope_b, n_pair, n_head, n_patches,
                rope_b->nb[1], rope_b->nb[2], ggml_row_size(rope_b->type, n_pair));

        return ggml_concat(ctx0, even, odd, 0);
    };

    // ggml_conv_2d rounds patches to F16
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * patches = ggml_im2col(ctx0, model.patch_embeddings_0, inp_raw,
            patch_size, patch_size, 0, 0, 1, 1, true, GGML_TYPE_F32);
    const int64_t n_patch_dim = patches->ne[0];
    ggml_tensor * inp = ggml_mul_mat(ctx0,
            ggml_reshape_2d(ctx0, model.patch_embeddings_0, n_patch_dim, n_embd),
            ggml_reshape_2d(ctx0, patches, n_patch_dim, n_patches));

    // reorder into 2x2 blocks
    inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, n_patches_y / 2);
    inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
    inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches, 1);

    ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, add_pos);

    // multimodal projection
    cur = ggml_reshape_3d(ctx0, cur, n_embd * 4, n_patches / 4, 1);
    cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_1_w, model.mm_1_b,
            hparams.ffn_op, -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
