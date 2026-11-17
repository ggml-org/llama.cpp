#include "models.h"

ggml_cgraph * clip_graph_qwen3a::build() {
    // Ref implementation: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

    // inp_raw: [n_frames, n_mel, 1]  (nx=n_frames, ny=n_mel)
    ggml_tensor * inp = build_inp_raw(1);

    const int64_t n_frames   = inp->ne[0]; // total frames, padded to multiple of chunk_size
    const int64_t n_mel      = inp->ne[1]; // 128
    const int64_t chunk_size = 100;        // n_window * 2 (n_window=50 from model config)
    const int64_t n_chunks   = n_frames / chunk_size;

    GGML_ASSERT(n_frames % chunk_size == 0); // preprocessor should already pad the input
    GGML_ASSERT(inp->type == GGML_TYPE_F32);

    // View mel spectrogram as batched 100-frame chunks: [chunk_size, n_mel, 1, n_chunks]
    inp = ggml_view_4d(ctx0, inp,
        chunk_size, n_mel, 1, n_chunks,
        n_frames   * (int64_t)sizeof(float), // nb[1]: stride over mel bins
        chunk_size * (int64_t)sizeof(float), // nb[2]: stride for C=1 (unused)
        chunk_size * (int64_t)sizeof(float), // nb[3]: stride over chunks
        0);
    inp = ggml_cont(ctx0, inp);
    cb(inp, "inp_chunks", -1);

    // 3 x conv2d + gelu
    {
        // conv output [OW, OH, C_out, n_chunks]
        auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
            if (b) {
                x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
            }
            return ggml_gelu_erf(ctx0, x);
        };

        inp = conv_block(inp, model.conv2d_1_w, model.conv2d_1_b);
        inp = conv_block(inp, model.conv2d_2_w, model.conv2d_2_b);
        inp = conv_block(inp, model.conv2d_3_w, model.conv2d_3_b);
        // inp: [OW=13, OH=16, OC=480, n_chunks]
        cb(inp, "after_conv_blocks", -1);
    }

    // permute [OW=25, OH=16, OC=480, n_chunks] -> [OH=16, OC=480, OW=25, n_chunks]
    // reshape to [OH*OC=7680, OW*n_chunks]
    // feature index h+16*c = c*16+f (matches python code)
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2] * inp->ne[3]);

    // Project to d_model: [d_model, 25*n_chunks]
    inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
    if (model.conv_out_b) {
        inp = ggml_add(ctx0, inp, model.conv_out_b);
    }
    cb(inp, "after_conv_out", -1);

    const int64_t n_pos = inp->ne[1]; // 25 * n_chunks

    // Per-chunk positional embeddings: repeat pos[0:13] for each chunk
    // (position indices reset 0..12 per chunk, not sequential across chunks)
    {
        const int64_t tokens_per_chunk = n_pos / n_chunks; // 13
        ggml_tensor * pos_tmp = ggml_view_2d(ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], tokens_per_chunk,
            model.position_embeddings->nb[1], 0);
        ggml_tensor * tgt = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
            model.position_embeddings->ne[0], n_pos);
        inp = ggml_add(ctx0, inp, ggml_repeat(ctx0, pos_tmp, tgt));
    }

    ggml_tensor * cur = build_vit(inp, n_pos,
        NORM_TYPE_NORMAL, hparams.ffn_op,
        nullptr,  // pos embd already added above
        nullptr);
    cb(cur, "after_transformer", -1);

    // MLP projector
    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF, -1);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
