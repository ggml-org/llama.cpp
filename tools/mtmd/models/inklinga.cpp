#include "models.h"

ggml_cgraph * mtmd_graph_inklinga::build() {
    const int n_mel = hparams.n_mel_bins;
    const int n_frames = img.nx();
    GGML_ASSERT(n_batch == 1 && img.ny() == n_mel);
    auto * ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_mel);
    ggml_set_name(ids, "inp_codes");
    ggml_set_input(ids);

    ggml_tensor * cur = nullptr;
    for (int m = 0; m < n_mel; ++m) {
        auto * bin_ids = ggml_view_1d(ctx0, ids, n_frames, m * ids->nb[1]);
        auto * bin = ggml_get_rows(ctx0, model.a_code_embd, bin_ids);
        cur = cur ? ggml_add(ctx0, cur, bin) : bin;
    }
    cur = build_norm(cur, model.post_ln_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cb(cur, "projected", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
