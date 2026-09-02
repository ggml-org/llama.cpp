#include "models.h"

// DeepSeek-V4-Flash-Vision encoder (deepseek4v)
//
// native-resolution ViT (RMSNorm, SwiGLU, 2D RoPE, no CLS / learned pos-embd)
// then the "aligner": 3x3 patch merge (torch.nn.functional.unfold) + 2-layer GELU MLP
//
// the graph outputs the complete LLM token block, built from the aligner output and 4 learned sentinel embeddings:
//
//   [PAD]*lead_pad [START] <interleaved rows> [PAD]*pad_last [END]
//
// each aligner row ends with a NEWLINE, an odd row count is padded with a full row of PADs
// pairs of adjacent rows are interleaved column-wise ("N-layout")
// the mapping is precomputed on CPU as the "layout_idx" input (see set_input in clip.cpp)
//
// ref: inference/vision.py and inference/image_processor.py in the HF repo

ggml_cgraph * clip_graph_deepseek4v::build() {
    const int n_merge = hparams.n_merge;

    // 2D input positions
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    int sections[4] = {d_head/4, d_head/4, 0, 0};
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(ctx0, cur, positions, nullptr,
            d_head/2, sections, GGML_ROPE_TYPE_VISION,
            0, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            hparams.ffn_op,
                            nullptr, // no learned pos embd
                            add_pos);
    cb(cur, "vit_out", -1);

    // aligner patch merge: zero-pad the patch grid to a multiple of n_merge
    // then F.unfold == im2col with a dummy kernel (same trick as pixtral)
    {
        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3); // [x, y, n_embd]
        cur = ggml_cont(ctx0, cur);

        const int pad_x = (n_merge - n_patches_x % n_merge) % n_merge;
        const int pad_y = (n_merge - n_patches_y % n_merge) % n_merge;
        if (pad_x || pad_y) {
            cur = ggml_pad(ctx0, cur, pad_x, pad_y, 0, 0);
        }

        ggml_tensor * kernel = ggml_view_3d(ctx0, cur, n_merge, n_merge, cur->ne[2], 0, 0, 0);
        cur = ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp->type);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);

        // aligner MLP (F.gelu in the reference == erf-based gelu)
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);
        cb(cur, "aligner_out", -1);
    }

    // assemble the token block: append the sentinel embeddings as extra rows
    // then reorder everything with the precomputed layout index
    {
        const int64_t n_embd_out = cur->ne[0];
        const int64_t n_grid     = cur->ne[1]; // n_llm_w * n_llm_h

        // rows n_grid + 0..3, keep in sync with the index computation in set_input
        ggml_tensor * sentinels[] = {
            model.token_embd_img_start,
            model.token_embd_img_end,
            model.image_newline,
            model.token_embd_img_pad,
        };
        for (ggml_tensor * tok : sentinels) {
            cur = ggml_concat(ctx0, cur, ggml_reshape_2d(ctx0, tok, n_embd_out, 1), 1);
        }

        const int n_llm_w = CLIP_ALIGN(n_patches_x, n_merge) / n_merge;
        const int n_llm_h = CLIP_ALIGN(n_patches_y, n_merge) / n_merge;
        const int n_out   = dsv4_get_block_layout(n_llm_w, n_llm_h, img.lead_pad).n_out;
        GGML_ASSERT(n_grid == n_llm_w * n_llm_h);

        ggml_tensor * layout_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_out);
        ggml_set_name(layout_idx, "layout_idx");
        ggml_set_input(layout_idx);

        cur = ggml_get_rows(ctx0, cur, layout_idx);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
