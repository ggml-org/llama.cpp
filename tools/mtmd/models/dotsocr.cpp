#include "models.h"

ggml_cgraph * clip_graph_dotsocr::build() {
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    // note: similar to PaddleOCR
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    cb(cur, "vit_out", -1);

    // dots.ocr patch merger + projector
    {
        GGML_ASSERT(hparams.n_merge > 0);
        cur = build_norm(cur, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, 1e-6, -1);
        cur = build_patch_merge_permute(cur, hparams.n_merge);
        cb(cur, "after_patch_merger", -1);
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr, // no gate
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF, -1); // nn.GELU() defaults to exact erf-based GELU
        cb(cur, "after_projector", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
