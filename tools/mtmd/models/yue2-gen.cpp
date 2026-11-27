#include "models.h"

ggml_cgraph * mtmd_graph_yue2_gen::build() {
    const auto & y = model.yue2;

    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_IN || gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_OUT) {
        const bool is_in = gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_IN;
        ggml_tensor * w = is_in ? y.inp_w : y.out_w;
        ggml_tensor * b = is_in ? y.inp_b : y.out_b;
        ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, w->ne[0], n_rows);
        ggml_set_name(cur, "inp_feats");
        ggml_set_input(cur);
        cur = ggml_add(ctx0, build_mm(w, cur), b);
        ggml_set_name(cur, "out_feats");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    GGML_ASSERT(gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_latent, n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [T, n_latent]
    x = build_snake_vae(x, model.snake_vae); // [T, n_channels]

    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // interleaved
    ggml_set_name(x, "out_audio");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}
