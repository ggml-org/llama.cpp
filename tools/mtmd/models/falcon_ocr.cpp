#include "models.h"

// Falcon OCR projector: no ViT encoder — raw image patches are directly
// projected via a single linear layer implemented as conv2d with stride=patch_size.

ggml_cgraph * clip_graph_falcon_ocr::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    const int ps = patch_size;
    const int pw = img.nx / ps;
    const int ph = img.ny / ps;
    const int n_patch = pw * ph;

    ggml_tensor * proj_w = ggml_reshape_4d(ctx0, model.mm_0_w, ps, ps, 3, n_embd);

    ggml_tensor * cur = ggml_conv_2d(ctx0, proj_w, inp_raw, ps, ps, 0, 0, 1, 1);

    // conv2d output [OW, OH, OC, 1] -> [n_embd, n_patch]
    cur = ggml_reshape_2d(ctx0, cur, n_patch, n_embd);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    if (model.prefix_embd) {
        cur = ggml_concat(ctx0, model.prefix_embd, cur, 1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
