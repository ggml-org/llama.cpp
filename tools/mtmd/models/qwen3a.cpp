#include "models.h"

ggml_cgraph * clip_graph_qwen3a::build() {
    // Ref implementation: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

    // inp_raw: [n_frames, n_mel, 1]  (nx=n_frames, ny=n_mel)
    ggml_tensor * inp = build_audio_conv2d_stem(build_inp_raw(1));

    const int64_t n_chunks = img.nx() / 100;
    const int64_t n_pos    = inp->ne[1]; // 13 * n_chunks

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
