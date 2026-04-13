#include "models.h"

ggml_cgraph * clip_graph_qwen3a::build() {
    ggml_tensor * inp = build_inp_raw(1);

    // conv2d block
    // TODO: do we need to split by chunks of n_window each like on transformers impl?
    {
        inp = ggml_conv_2d(ctx0, model.conv2d_1_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_1_b);
        inp = ggml_gelu_erf(ctx0, inp);

        inp = ggml_conv_2d(ctx0, model.conv2d_2_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_2_b);
        inp = ggml_gelu_erf(ctx0, inp);

        inp = ggml_conv_2d(ctx0, model.conv2d_3_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_3_b);
        inp = ggml_gelu_erf(ctx0, inp);

        // inp [n_pos, n_mels/8, channels, 1] (W, H, C, N)
        cb(inp, "after_conv_blocks", -1);

        const int64_t n_pos_after_conv = inp->ne[0];
        const int64_t n_mel_after_conv = inp->ne[1]; // 128/8 = 16

        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 0, 2, 3, 1));
        inp = ggml_reshape_2d(ctx0, inp, n_pos_after_conv, n_mel_after_conv * inp->ne[3]); // [n_pos, 7680]
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [7680, n_pos]

        // project to n_embd
        inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
        if (model.conv_out_b) {
            inp = ggml_add(ctx0, inp, model.conv_out_b);
        }
        cb(inp, "after_conv_out", -1);
    }

    auto n_pos = inp->ne[1];

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr);

    cb(cur, "after_transformer", -1);

    // projector
    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF,
        -1);

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
