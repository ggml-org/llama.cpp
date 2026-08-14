#include "models.h"

ggml_cgraph * clip_graph_dots3note_a::build() {
    // inp_raw: [n_frames, n_mel, 1], one 60s chunk, mel frames not padded
    // the reference impl zero-masks conv inputs beyond the valid length at each stage;
    // running on exactly the valid frames with the convs' zero padding is equivalent
    ggml_tensor * inp = build_inp_raw(1);
    GGML_ASSERT(inp->type == GGML_TYPE_F32);

    // 3x conv2d (k=3, s=2, p=1) + gelu
    {
        auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
            x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
            return ggml_gelu_erf(ctx0, x);
        };

        inp = conv_block(inp, model.conv2d_1_w, model.conv2d_1_b);
        inp = conv_block(inp, model.conv2d_2_w, model.conv2d_2_b);
        inp = conv_block(inp, model.conv2d_3_w, model.conv2d_3_b);
        // inp: [OW=n_frames/8, OH=n_mel/8, OC=480, 1]
        cb(inp, "after_conv_stem", -1);
    }

    // [OW, OH, OC, 1] -> [OH*OC, OW], feature index f + OH*c (matches the reference permute+reshape)
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2]);

    // project to d_model (no bias)
    inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
    cb(inp, "after_conv_out", -1);

    const int64_t n_pos = inp->ne[1];

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // partial rotary: first half of each head, NEOX style
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_ext(ctx0, cur, positions, nullptr, d_head/2,
                             GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };

    ggml_tensor * cur = build_vit(inp, n_pos,
        NORM_TYPE_RMS, hparams.ffn_op,
        nullptr, add_pos);
    cb(cur, "after_transformer", -1);

    // adapter: LayerNorm -> Linear -> GELU -> Linear
    cur = build_norm(cur, model.mm_norm_pre_w, model.mm_norm_pre_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF, -1);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
