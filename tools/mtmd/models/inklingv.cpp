#include "models.h"

ggml_cgraph * mtmd_graph_inklingv::build() {
    auto * inp = build_inp_raw();
    GGML_ASSERT(n_batch == 1);
    auto * cur = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 2, 0, 3));
    int width = img.nx();
    int height = img.ny();
    int channels = 3;
    const int folds[] = {5, 2, 4};
    for (int i = 0; i < 3; ++i) {
        const int fold = folds[i];
        // Fold in height-offset, width-offset, channel order, within each temporal slice.
        cur = ggml_reshape_4d(ctx0, cur, channels * fold, width / fold, height, n_batch);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
        cur = ggml_reshape_4d(ctx0, cur, channels * fold * fold, height / fold, width / fold, n_batch);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
        width /= fold;
        height /= fold;
        cur = ggml_reshape_2d(ctx0, cur, channels * fold * fold, width * height * n_batch);
        cur = build_mm(model.layers[i].ff_up_w, cur);
        cur = build_norm(cur, model.layers[i].ln_1_w, nullptr, NORM_TYPE_RMS, eps, i);
        cur = ggml_gelu_erf(ctx0, cur);
        channels = model.layers[i].ff_up_w->ne[1];
        cur = ggml_reshape_4d(ctx0, cur, channels, width, height, n_batch);
        cb(cur, "encoder_stage", i);
    }
    // The tower expects two temporal slices even for a still image.
    cur = ggml_concat(ctx0, cur, cur, 0);
    cur = ggml_reshape_2d(ctx0, cur, channels * 2, width * height);
    cb(cur, "temporal_fold", -1);
    cur = build_mm(model.mm_0_w, cur);
    cur = build_norm(cur, model.post_ln_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cb(cur, "projected", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
