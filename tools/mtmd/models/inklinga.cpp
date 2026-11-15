#include "models.h"

// dMel: each log-mel value is quantized to the nearest of n_bins levels, a frame is the sum of the embeddings of its bins
ggml_cgraph * mtmd_graph_inklinga::build() {
    const int   n_mel    = hparams.n_mel_bins;
    const int   n_frames = img.nx();
    const int   n_bins   = model.mm_fc_w->ne[0] / n_mel;
    const float v_min    = -7.0f; // log10
    const float v_max    = 2.0f;
    const float scale    = (n_bins - 1) / (v_max - v_min);

    ggml_tensor * cur = build_inp_raw(1);
    ggml_tensor * bins = ggml_arange(ctx0, 0, n_bins, 1);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [n_mel, n_frames]
    cur = ggml_clamp(ctx0, cur, v_min, v_max);
    cur = ggml_round(ctx0, ggml_scale_bias(ctx0, cur, scale, -v_min * scale));

    // one-hot over the bins, the embedding lookup is then a matmul
    cur = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, cur, 1, n_mel, n_frames), n_bins, n_mel, n_frames, 1);
    cur = ggml_abs(ctx0, ggml_sub(ctx0, cur, bins));
    cur = ggml_relu(ctx0, ggml_scale_bias(ctx0, cur, -1.0f, 1.0f));
    cur = ggml_reshape_2d(ctx0, cur, n_bins * n_mel, n_frames);

    cur = build_mm(model.mm_fc_w, cur);
    cur = build_norm(cur, model.post_ln_w, nullptr, NORM_TYPE_RMS, eps, -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
