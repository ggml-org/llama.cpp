#include "models.h"

#include <cmath>

ggml_cgraph * clip_graph_jinaclip2::build() {
    const bool has_cls = model.class_embedding != nullptr;
    GGML_ASSERT(has_cls && "JinaCLIP2 requires a CLS token");

    const int n_pos = n_patches + (has_cls ? 1 : 0);

    GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");

    // input for learned position embeddings
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // inputs for 2D RoPE positions (includes CLS at index 0)
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    // frequency scaling factors for the 2D RoPE halves
    GGML_ASSERT(d_head % 2 == 0);
    ggml_tensor * rope_c_first = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_head / 2);
    ggml_set_name(rope_c_first, "rope_c_first");
    ggml_set_input(rope_c_first);

    ggml_tensor * rope_c_second = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_head / 2);
    ggml_set_name(rope_c_second, "rope_c_second");
    ggml_set_input(rope_c_second);

    ggml_tensor * inp = build_inp();
    if (has_cls) {
        inp = ggml_concat(ctx0, model.class_embedding, inp, 1);
    }
    inp = ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));

    auto apply_rope_2d = [&](ggml_tensor * cur) -> ggml_tensor * {
        // cur is [d_head, n_head, n_pos]; convert to [d_head, n_pos, n_head] for convenient slicing
        ggml_tensor * cur_in = ggml_permute(ctx0, cur, 0, 2, 1, 3);

        const int64_t n_dim = cur_in->ne[0];
        const int64_t seq   = cur_in->ne[1];
        const int64_t nhead = cur_in->ne[2];
        GGML_ASSERT(seq == n_pos);
        GGML_ASSERT(n_dim % 2 == 0);

        const int64_t half = n_dim / 2;

        ggml_tensor * cls = nullptr;
        ggml_tensor * patches = cur_in;
        int64_t n_pos_patches = seq;
        int64_t pos_offset = 0;

        if (has_cls) {
            cls = ggml_view_3d(ctx0, cur_in, n_dim, 1, nhead, cur_in->nb[1], cur_in->nb[2], 0);
            patches = ggml_view_3d(ctx0, cur_in, n_dim, seq - 1, nhead, cur_in->nb[1], cur_in->nb[2], cur_in->nb[1]);
            n_pos_patches = seq - 1;
            pos_offset = 1;
        }

        // select positions for patch tokens
        ggml_tensor * pos_a = ggml_view_1d(ctx0, pos_h, n_pos_patches, pos_offset * (int64_t) ggml_element_size(pos_h));
        ggml_tensor * pos_b = ggml_view_1d(ctx0, pos_w, n_pos_patches, pos_offset * (int64_t) ggml_element_size(pos_w));

        // first half (H)
        ggml_tensor * first = ggml_view_3d(ctx0, patches,
            half, nhead, n_pos_patches,
            patches->nb[2], patches->nb[1], 0);
        ggml_tensor * first_rot = ggml_rope_ext(
            ctx0,
            first,
            pos_a,
            rope_c_first,
            half,
            0, 0, hparams.rope_theta,
            1.0f,
            0.0f, 1.0f, 0.0f, 0.0f);
        first = ggml_view_3d(ctx0, first_rot,
            half, n_pos_patches, nhead,
            first_rot->nb[2], first_rot->nb[1], 0);

        // second half (W)
        ggml_tensor * second = ggml_view_3d(ctx0, patches,
            half, nhead, n_pos_patches,
            patches->nb[2], patches->nb[1],
            half * (int64_t) ggml_element_size(patches));
        ggml_tensor * second_rot = ggml_rope_ext(
            ctx0,
            second,
            pos_b,
            rope_c_second,
            half,
            0, 0, hparams.rope_theta,
            1.0f,
            0.0f, 1.0f, 0.0f, 0.0f);
        second = ggml_view_3d(ctx0, second_rot,
            half, n_pos_patches, nhead,
            second_rot->nb[2], second_rot->nb[1], 0);

        ggml_tensor * patches_out = ggml_concat(ctx0, first, second, 0);
        ggml_tensor * out_seq = has_cls ? ggml_concat(ctx0, cls, patches_out, 1) : patches_out;
        return ggml_permute(ctx0, out_seq, 0, 2, 1, 3);
    };

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return apply_rope_2d(cur);
    };

    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    // Output: CLS embedding only (1 token).
    ggml_tensor * cls = ggml_view_2d(ctx0, cur, cur->ne[0], /*rows=*/1, cur->nb[1], /*offset=*/0);
    ggml_set_name(cls, "cls_view");
    ggml_build_forward_expand(gf, cls);

    return gf;
}
