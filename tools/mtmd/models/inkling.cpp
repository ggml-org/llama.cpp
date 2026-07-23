#include "models.h"

ggml_tensor * clip_graph_inkling::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    ggml_tensor * cur = ggml_mul_mat(ctx0, w, x);
    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
    return cur;
}

// fold square neighborhoods from W/H into channels; folded order is [h_fold, w_fold, C]
static ggml_tensor * inkling_fold_spatial(
        ggml_context * ctx0,
        ggml_tensor  * cur,
        int            scale) {
    GGML_ASSERT(scale > 0);
    GGML_ASSERT(cur->ne[1] % scale == 0);
    GGML_ASSERT(cur->ne[2] % scale == 0);

    const int64_t c = cur->ne[0];
    const int64_t w = cur->ne[1];
    const int64_t h = cur->ne[2];
    const int64_t b = cur->ne[3];

    cur = ggml_reshape_4d(ctx0, cur, c * scale, w / scale, h, b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, c * scale * scale, h / scale, w / scale, b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    return cur;
}

ggml_cgraph * clip_graph_inkling::build() {
    return model.modality == CLIP_MODALITY_AUDIO ? build_audio() : build_vision();
}

ggml_cgraph * clip_graph_inkling::build_vision() {
    static constexpr int temporal_patch_size = 2;
    static constexpr int spatial_folds[] = {5, 2, 4};

    GGML_ASSERT(img.nx() == 40 && img.ny() == 40);
    GGML_ASSERT(n_batch > 0 && n_batch % temporal_patch_size == 0);
    GGML_ASSERT(model.inkling_hmlp_layers.size() == 4);
    GGML_ASSERT(model.inkling_hmlp_final_norm_w);

    // Raw input is [W,H,RGB,temporal*patch]. Put RGB on ne[0].
    ggml_tensor * cur = build_inp_raw(3);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));

    for (int il = 0; il < 3; ++il) {
        cur = inkling_fold_spatial(ctx0, cur, spatial_folds[il]);
        const int64_t w = cur->ne[1];
        const int64_t h = cur->ne[2];
        const int64_t b = cur->ne[3];

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], w * h * b);
        cur = build_mm(model.inkling_hmlp_layers[il].linear_w, cur);
        cur = build_norm(cur, model.inkling_hmlp_layers[il].norm_w,
                         nullptr, NORM_TYPE_RMS, eps, il);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], w, h, b);
    }

    GGML_ASSERT(cur->ne[1] == 1 && cur->ne[2] == 1);
    const int64_t n_patches = n_batch / temporal_patch_size;
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * temporal_patch_size, n_patches);
    cur = build_mm(model.inkling_hmlp_layers[3].linear_w, cur);
    cur = build_norm(cur, model.inkling_hmlp_final_norm_w,
                     nullptr, NORM_TYPE_RMS, eps, 3);

    // Batched mtmd convention: one token in ne[1], patch count in ne[2].
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, n_patches);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * clip_graph_inkling::build_audio() {
    static constexpr int n_mels = 80;
    static constexpr int mel_vocab_size = 16;
    static constexpr int n_embd = 6144;

    GGML_ASSERT(img.ny() == n_mels);
    GGML_ASSERT(model.inkling_dmel_embd_w);
    GGML_ASSERT(model.inkling_dmel_final_norm_w);
    GGML_ASSERT(model.inkling_dmel_embd_w->ne[0] == n_embd);
    GGML_ASSERT(model.inkling_dmel_embd_w->ne[1] == n_mels * mel_vocab_size);

    const int64_t n_tokens = img.nx();
    // mtmd audio storage is mel-major and represented as [token, mel].
    ggml_tensor * bins = build_inp_raw(1);
    bins = ggml_cont(ctx0, ggml_transpose(ctx0, bins));
    ggml_tensor * offsets = ggml_arange(ctx0, 0, n_mels, 1);
    offsets = ggml_scale(ctx0, offsets, mel_vocab_size);
    offsets = ggml_reshape_2d(ctx0, offsets, n_mels, 1);
    ggml_tensor * indices = ggml_cast(ctx0, ggml_add(ctx0, bins, offsets), GGML_TYPE_I32);

    indices = ggml_reshape_1d(ctx0, indices, n_mels * n_tokens);
    ggml_tensor * cur = ggml_get_rows(ctx0, model.inkling_dmel_embd_w, indices);
    cur = ggml_reshape_3d(ctx0, cur, n_embd, n_mels, n_tokens);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_sum_rows(ctx0, cur);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    cur = build_norm(cur, model.inkling_dmel_final_norm_w,
                     nullptr, NORM_TYPE_RMS, eps, -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
