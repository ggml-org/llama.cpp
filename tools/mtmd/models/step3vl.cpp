#include "models.h"

ggml_cgraph * clip_graph_step3vl::build() {
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    norm_type norm_t = NORM_TYPE_NORMAL;

    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * inp = build_inp();
    ggml_tensor * learned_pos_embd = resize_position_embeddings();

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    auto add_spatial_bias = [&](ggml_tensor * cur, ggml_tensor * bias) {
        if (bias == nullptr) {
            return cur;
        }

        const int64_t width    = cur->ne[0];
        const int64_t height   = cur->ne[1];
        const int64_t channels = cur->ne[2];

        cur = ggml_reshape_2d(ctx0, cur, width * height, channels);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_add(ctx0, cur, bias);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_reshape_3d(ctx0, cur, width, height, channels);

        return cur;
    };

    ggml_tensor * cur = build_vit(
        inp,
        n_patches,
        norm_t,
        hparams.ffn_op,
        learned_pos_embd,
        add_pos);
    cb(cur, "vit_out", -1);

    // [n_embd, n_patches] -> [w, h, n_embd] for spatial downsampling convolutions.
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont_3d(ctx0, cur, n_patches_x, n_patches_y, n_embd);

    // First downsampler: Conv2d(1536 -> 3072, k=3, s=2, p=1)
    cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, 2, 2, 1, 1, 1, 1);
    cur = add_spatial_bias(cur, model.mm_0_b);
    cb(cur, "downsample_0", -1);

    // Second downsampler: Conv2d(3072 -> 6144, k=3, s=2, p=1)
    cur = ggml_conv_2d(ctx0, model.mm_1_w, cur, 2, 2, 1, 1, 1, 1);
    cur = add_spatial_bias(cur, model.mm_1_b);
    cb(cur, "downsample_1", -1);

    // [w, h, c] -> [c, w*h]
    {
        const int64_t w = cur->ne[0];
        const int64_t h = cur->ne[1];
        cur = ggml_reshape_3d(ctx0, cur, w * h, cur->ne[2], cur->ne[3]);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }
    cb(cur, "downsample_flatten", -1);

    // Final projector: Linear(6144 -> projection_dim)
    cur = ggml_mul_mat(ctx0, model.mm_model_proj, cur);
    cb(cur, "projector_out", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
