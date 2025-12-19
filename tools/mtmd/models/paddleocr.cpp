#include "models.h"

ggml_cgraph * clip_graph_paddleocr::build() {
    // 2D input positions
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings();

    // build ViT with 2D position embeddings
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        // first half is X axis and second half is Y axis
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // mlp_AR
        float proj_norm_eps = 1e-5; // PaddleOCR uses hard-coded value eps=1e-5 for Projector
        cur = build_norm(cur,
                    model.mm_input_norm_w, model.mm_input_norm_b,
                    NORM_TYPE_NORMAL, proj_norm_eps, -1);
        //cur = build_patch_merge_permute(cur, hparams.proj_scale_factor);

        // stack and padding
        int64_t stride          = hparams.proj_scale_factor * hparams.proj_scale_factor;
        int64_t n_embd          = cur->ne[0];
        int64_t n_tokens        = cur->ne[1];
        int64_t n_tokens_padded = CLIP_ALIGN(n_tokens, stride);
        int64_t n_pad           = n_tokens_padded - n_tokens;
        if (n_pad > 0) {
            cur = ggml_view_1d(ctx0, cur, ggml_nelements(cur), 0);
            cur = ggml_pad(ctx0, cur, n_pad * n_embd, 0, 0, 0);
        }
        cur = ggml_view_2d(ctx0, cur,
            n_embd * stride,
            n_tokens_padded / stride,
            ggml_row_size(cur->type, n_embd * stride), 0);
        cb(cur, "after_stacked", -1);

        cur = build_ffn(cur,
                    model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b,
                    hparams.ffn_op, -1);
        cb(cur, "mlp_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
