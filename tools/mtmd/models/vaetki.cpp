#include "models.h"

ggml_cgraph * clip_graph_vaetki::build() {
    GGML_ASSERT(model.class_embedding != nullptr);

    const int batch_size = 1;
    const int n_pos = n_patches + 1;
    const int n_pos_patches = n_patches;
    const int num_position_ids = n_pos_patches * 4;

    norm_type norm_t = NORM_TYPE_NORMAL;
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp();

    // add CLS token
    inp = ggml_concat(ctx0, model.class_embedding, inp, 1);
    cb(inp, "inp_with_cls", -1);

    // position IDs for 2D RoPE (patch tokens only)
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // precompute CLS position embedding cos/sin
    ggml_tensor * cls_cos = nullptr;
    ggml_tensor * cls_sin = nullptr;
    if (model.class_pos_emb) {
        ggml_tensor * cls_pos = ggml_concat(ctx0, model.class_pos_emb, model.class_pos_emb, 0);
        cls_cos = ggml_cos(ctx0, cls_pos);
        cls_sin = ggml_sin(ctx0, cls_pos);
    }

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) -> ggml_tensor * {
        // split CLS and patch tokens
        // use cur->nb[2] to support both fused QKV (nb[2]=3*n_embd) and separate Q/K/V (nb[2]=n_embd)
        ggml_tensor * cur_cls = ggml_view_3d(ctx0, cur, d_head, n_head, 1,
            ggml_row_size(cur->type, d_head),
            cur->nb[2], 0);
        ggml_tensor * cur_patch = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos_patches,
            ggml_row_size(cur->type, d_head),
            cur->nb[2],
            cur->nb[2]);

        // apply RoPE to CLS token using class_pos_emb
        if (cls_cos && cls_sin) {
            ggml_tensor * cls_1 = ggml_view_3d(ctx0, cur_cls, d_head/2, n_head, 1,
                ggml_row_size(cur_cls->type, d_head),
                ggml_row_size(cur_cls->type, d_head * n_head), 0);
            ggml_tensor * cls_2 = ggml_view_3d(ctx0, cur_cls, d_head/2, n_head, 1,
                ggml_row_size(cur_cls->type, d_head),
                ggml_row_size(cur_cls->type, d_head * n_head),
                ggml_row_size(cur_cls->type, d_head/2));
            ggml_tensor * cls_rot = ggml_concat(ctx0, ggml_neg(ctx0, cls_2), cls_1, 0);

            cur_cls = ggml_add(ctx0,
                ggml_mul(ctx0, cur_cls, cls_cos),
                ggml_mul(ctx0, cls_rot, cls_sin));
        }

        // apply 2D RoPE to patch tokens
        cur_patch = ggml_rope_multi(ctx0, cur_patch, positions, nullptr,
            d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

        return ggml_concat(ctx0, cur_cls, cur_patch, 2);
    };

    ggml_tensor * cur = build_vit(
        inp, n_pos,
        norm_t,
        hparams.ffn_op,
        nullptr,
        add_pos);

    cb(cur, "vit_out", -1);

    // remove CLS token
    ggml_tensor * embeddings = ggml_view_2d(ctx0, cur,
        n_embd, n_pos_patches,
        ggml_row_size(cur->type, n_embd),
        ggml_row_size(cur->type, n_embd));
    cb(embeddings, "patches_only", -1);

    // merger
    embeddings = build_norm(embeddings, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cb(embeddings, "merger_normed", -1);

    // pixel shuffle
    const int scale_factor = hparams.n_merge;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * scale_factor * scale_factor, n_pos_patches / (scale_factor * scale_factor), batch_size);
    cb(embeddings, "merger_reshaped", -1);

    embeddings = build_ffn(embeddings,
        model.mm_ffn_up_w, model.mm_ffn_up_b,
        nullptr, nullptr,
        model.mm_ffn_down_w, model.mm_ffn_down_b,
        FFN_GELU,
        -1);
    cb(embeddings, "merger_out", -1);

    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
