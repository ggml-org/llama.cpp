#include "models.h"

ggml_cgraph * clip_graph_deepseekocr2::build() {
    GGML_ASSERT(hparams.n_head_kv > 0);
    GGML_ASSERT(n_head % hparams.n_head_kv == 0);

    const int64_t B = grid_x * grid_y;

    ggml_tensor * inp_raw = build_inp_raw(3, grid_x * grid_y);

    ggml_tensor * sam_out = build_sam(inp_raw);

    ggml_tensor * qwen2_out;
    // Building Qwen2 encoder
    {
        // [W, H, C, B] -> [H*W, C, B] -> [C, H*W, B]
        ggml_tensor * inp = ggml_reshape_3d(ctx0, sam_out, sam_out->ne[0] * sam_out->ne[1], sam_out->ne[2], B);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

        auto num_image_tokens = inp->ne[1]; // H*W
        GGML_ASSERT(num_image_tokens == 144 || num_image_tokens == 256);

        // query based on numbers of image tokens (in SAM output)
        // 16x16 -> query_1024 (1024x1024 images)
        // 12x12 -> query_768 (768x768 images)

        ggml_tensor * query_embed = model.resample_query_1024;
        int           num_queries = 256;

        if (num_image_tokens == 144) {
            query_embed = model.resample_query_768;
            num_queries = 144;
        }

        // query_embed [C, num_queries]; broadcast across the batch and append:
        ggml_tensor * query_b = ggml_repeat_4d(ctx0, ggml_cast(ctx0, query_embed, inp->type),
                                               inp->ne[0], num_queries, B, 1);
        inp = ggml_concat(ctx0, inp, query_b, 1);

        auto seq_len = inp->ne[1];

        // qwen2 encoder attention mask
        ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, seq_len);
        ggml_set_name(attn_mask, "qwen2_attn_mask");
        ggml_set_input(attn_mask);

        ggml_tensor * inp_pos = ggml_cast(ctx0, ggml_arange(ctx0, 0, seq_len, 1), GGML_TYPE_I32);

        auto add_rope = [&](ggml_tensor * x, const clip_layer &) {
            return ggml_rope_ext(ctx0, x, inp_pos, nullptr, d_head,
                                 GGML_ROPE_TYPE_NEOX, 131072, 1000000, 1, 0, 1, 0, 0);
        };

        build_vit_opts vit_opts;
        vit_opts.attn_mask = attn_mask;

        // build_vit applies model.post_ln_w internally; do not re-apply
        ggml_tensor * cur = build_vit(inp, seq_len, NORM_TYPE_RMS, FFN_SILU,
                                      /* learned_pos_embd */ nullptr, add_rope, vit_opts); // [C, seq_len, B]

        // only keep the query tokens; [C, num_queries, B]
        cur = ggml_cont(ctx0,
                        ggml_view_3d(ctx0, cur, cur->ne[0], num_queries, B,
                                     cur->nb[1], cur->nb[2], cur->nb[1] * (cur->ne[1] - num_queries)));

        ggml_build_forward_expand(gf, cur);
        qwen2_out = cur;
    }

    ggml_tensor * cur;

    cur = ggml_mul_mat(ctx0, model.mm_fc_w, qwen2_out); // [n_dim, num_queries, B]
    cur = ggml_add(ctx0, cur, model.mm_fc_b);

    // view_seperator only after the global view
    if (img.add_viewsep) {
        GGML_ASSERT(B == 1);
        cur = ggml_concat(ctx0, cur, model.view_seperator, 1); // (n_dim, 257)
    }

    // flatten the batch
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "dsocr2_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
