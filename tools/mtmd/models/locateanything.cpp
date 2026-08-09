#include "models.h"
#include <cstring>
#include <cmath>

// NVIDIA LocateAnything-3B: MoonViT-SO-400M vision tower + Eagle MLP projector.
// Same MoonViT lineage as Kimi-K2.5; this graph is a near-direct clone of
// clip_graph_kimik25 with the projector swapped for Eagle MLP (LayerNorm +
// Linear + GELU + Linear, no final norm).

ggml_tensor * clip_graph_locateanything::resize_position_embeddings_3d(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny() / patch_size;
    const int width        = img.nx() / patch_size;
    const uint32_t mode    = interpolation_mode;

    GGML_ASSERT(pos_embd);

    const int64_t stored_c = pos_embd->ne[0];  // C = 1152
    const int64_t orig_w   = pos_embd->ne[1];  // W = 64
    const int64_t orig_h   = pos_embd->ne[2];  // H = 64

    GGML_ASSERT(stored_c == n_embd);

    if (height == (int)orig_h && width == (int)orig_w) {
        return ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    }

    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_interpolate(ctx0, pos_embd, height, width, n_embd, 1, mode);
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    return pos_embd;
}

ggml_cgraph * clip_graph_locateanything::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings_3d(GGML_SCALE_MODE_BICUBIC);

    // MoonViT uses interleaved 2D RoPE natively; Q/K are permuted at conversion
    // time (conversion/locateanything.py::permute) so build_rope_2d (sectioned)
    // produces the same numerical output as the reference's interleaved rope.
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        cur = build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
        return cur;
    };

    ggml_tensor * inp = build_inp();

    // pos-emb is added before the encoder loop (matches reference patch_embed.forward)
    inp = ggml_add(ctx0, inp, learned_pos_embd);

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // patch_merger: 2x2 spatial groups merged -> ne[0] = scale_factor^2 * n_embd = 4608
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // Eagle MLP (modeling_locateanything.py:136-140):
        //   LayerNorm(4608) -> Linear(4608 -> 2048) -> GELU -> Linear(2048 -> 2048)
        // The LN is over the merged 4608-dim feature axis (NOT pre-merge per
        // sub-token like Kimi-K2.5). mm_input_norm_{w,b} are shape (4608,).
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        cb(cur, "proj_inp_normed", -1);

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

        cb(cur, "proj_out", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}
