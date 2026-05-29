#include "models.h"

ggml_cgraph * clip_graph_deepseekocr2::build() {
    GGML_ASSERT(hparams.n_head_kv > 0);
    GGML_ASSERT(n_head % hparams.n_head_kv == 0);

    // patch embedding
    ggml_tensor * inp_raw = build_inp_raw();

    ggml_tensor * sam_out = build_sam(inp_raw);

    ggml_tensor * qwen2_out;
    // Building Qwen2 encoder
    {
        ggml_tensor * inp;

        inp = ggml_reshape_2d(ctx0, sam_out, sam_out->ne[0] * sam_out->ne[1], sam_out->ne[2]); // H*W, C
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

        // (B, num_image_tokens + num_queries, C)
        inp = ggml_concat(ctx0, inp, ggml_cast(ctx0, query_embed, inp->type), 1);

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
                                      /* learned_pos_embd */ nullptr, add_rope, vit_opts);

        cur = ggml_cont(ctx0,
                        ggml_view_2d(ctx0, cur, cur->ne[0], num_queries, cur->nb[1],
                                     cur->nb[1] * (cur->ne[1] - num_queries))); // only take query tokens for output

        ggml_build_forward_expand(gf, cur);
        qwen2_out = cur;
    }

    ggml_tensor * cur;

    cur = ggml_mul_mat(ctx0, model.mm_fc_w, qwen2_out);
    cur = ggml_add(ctx0, cur, model.mm_fc_b);

    // view_seperator only after the global view
    if (img.add_viewsep) {
        cur = ggml_concat(ctx0, cur, model.view_seperator, 1); // (n_dim, 257)
    }

    cb(cur, "dsocr2_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
