/**
 * qwen3a audio encoder graph builder
 *
 * Architecture:
 *   1. Conv2d downsampling: 3x Conv2d(stride=2, padding=1) + GELU
 *      Input:  [1, n_mel=128, T]  (mel spectrogram, mono channel)
 *      After conv2d1: [480, 64, T/2]
 *      After conv2d2: [480, 32, T/4]
 *      After conv2d3: [480, 16, T/8]
 *   2. Flatten + linear: [T/8, 480*16=7680] → [T/8, 896]
 *   3. Sinusoidal positional embeddings: [max_pos, 896] (per conv chunk)
 *   4. 18x encoder layers: pre-LN → self-attn(14 heads, bias) → residual → pre-LN → FFN(GELU) → residual
 *   5. Post LayerNorm
 *   6. Output projector: Linear(896,896) + GELU + Linear(896,1024)
 *
 * Inference windows (n_window_infer): Multiple conv chunks (n_window*2 frames each)
 * are grouped into a single encoder call. Conv and positional embeddings are applied
 * per sub-chunk, then the transformer runs on the concatenated sequence with a
 * block-diagonal attention mask restricting attention within sub-chunk boundaries.
 * This matches the PyTorch cu_seqlens-based block-diagonal attention.
 */

#include "models.h"

// Helper: compute output time steps after 3x Conv2d(stride=2, padding=1)
static int conv2d_output_len(int n) {
    n = (n - 1) / 2 + 1;
    n = (n - 1) / 2 + 1;
    n = (n - 1) / 2 + 1;
    return n;
}

ggml_cgraph * clip_graph_qwen3a::build() {
    // Input: mel spectrogram packed as clip_image_f32
    //   img.nx = n_frames (time steps) — may span multiple conv chunks
    //   img.ny = n_mel_bins (128)
    //   img.buf = [n_mel * n_frames] float32, row-major (mel × time)
    const int n_frames = img.nx;
    const int n_mel    = img.ny;

    const int freq_after_conv = conv2d_output_len(n_mel);
    const int flatten_dim = freq_after_conv * hparams.downsample_hidden_size;

    // -----------------------------------------------------------------------
    // Sub-chunk parameters
    // Conv chunks are n_window*2 mel frames each (e.g. 100 for 0.6B).
    // The mel preprocessor sends inference-window-sized inputs (up to
    // n_window_infer frames = 8 conv chunks). We split here and recombine.
    // -----------------------------------------------------------------------
    const int conv_chunk_frames = hparams.n_window > 0
        ? hparams.n_window * 2
        : n_frames;  // fallback: treat entire input as one chunk
    const int n_sub_chunks = (n_frames + conv_chunk_frames - 1) / conv_chunk_frames;

    // Load raw mel input as flat [n_frames * n_mel] tensor
    ggml_tensor * inp_raw = build_inp_raw(1);

    // Helper: reshape 3D conv bias [1, 1, OC] → 4D [1, 1, OC, 1] for broadcast
    auto reshape_conv_bias = [&](ggml_tensor * bias) -> ggml_tensor * {
        return ggml_reshape_4d(ctx0, bias, 1, 1, bias->ne[2], 1);
    };

    GGML_ASSERT(model.position_embeddings != nullptr);

    // -----------------------------------------------------------------------
    // Per-sub-chunk: Conv2d → flatten → project → positional embeddings
    // Each sub-chunk shares the same conv weights and gets the same
    // positional embeddings (positions 0..tokens_per_chunk).
    // -----------------------------------------------------------------------
    std::vector<ggml_tensor *> sub_embeddings;
    std::vector<int> sub_token_counts;
    int total_tokens = 0;

    for (int ci = 0; ci < n_sub_chunks; ci++) {
        const int start     = ci * conv_chunk_frames;
        const int chunk_len = std::min(conv_chunk_frames, n_frames - start);
        const int tokens    = conv2d_output_len(chunk_len);

        sub_token_counts.push_back(tokens);
        total_tokens += tokens;

        // View into mel data for this sub-chunk: [chunk_len, n_mel] (strided)
        // inp_raw layout: mel_bin_0[frame_0..n-1], mel_bin_1[frame_0..n-1], ...
        ggml_tensor * sub_view = ggml_view_2d(ctx0, inp_raw,
            chunk_len, n_mel,
            (size_t)n_frames * ggml_type_size(inp_raw->type),
            (size_t)start * ggml_type_size(inp_raw->type));

        // Make contiguous and reshape to conv input [W=chunk_len, H=n_mel, C=1, N=1]
        ggml_tensor * sub_inp = ggml_reshape_4d(ctx0, ggml_cont(ctx0, sub_view),
            chunk_len, n_mel, 1, 1);

        // 3x Conv2d(stride=2, padding=1) + GELU(erf)
        ggml_tensor * sub_cur;

        sub_cur = ggml_conv_2d(ctx0, model.conv2d_1_w, sub_inp, 2, 2, 1, 1, 1, 1);
        sub_cur = ggml_add(ctx0, sub_cur, reshape_conv_bias(model.conv2d_1_b));
        sub_cur = ggml_gelu_erf(ctx0, sub_cur);

        sub_cur = ggml_conv_2d(ctx0, model.conv2d_2_w, sub_cur, 2, 2, 1, 1, 1, 1);
        sub_cur = ggml_add(ctx0, sub_cur, reshape_conv_bias(model.conv2d_2_b));
        sub_cur = ggml_gelu_erf(ctx0, sub_cur);

        sub_cur = ggml_conv_2d(ctx0, model.conv2d_3_w, sub_cur, 2, 2, 1, 1, 1, 1);
        sub_cur = ggml_add(ctx0, sub_cur, reshape_conv_bias(model.conv2d_3_b));
        sub_cur = ggml_gelu_erf(ctx0, sub_cur);

        // Permute + flatten: [T', F', 480, 1] → [F'*480, T'] → [7680, tokens]
        sub_cur = ggml_cont(ctx0, ggml_permute(ctx0, sub_cur, 2, 0, 1, 3));
        sub_cur = ggml_reshape_2d(ctx0, sub_cur, flatten_dim, tokens);

        // Linear projection: [7680, tokens] → [896, tokens]
        sub_cur = ggml_mul_mat(ctx0, model.conv_out_w, sub_cur);

        // Per-chunk positional embeddings (same positions 0..tokens for every chunk)
        GGML_ASSERT(model.position_embeddings->ne[1] >= tokens);
        ggml_tensor * pos_embd = ggml_view_2d(ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], tokens,
            model.position_embeddings->nb[1], 0);
        sub_cur = ggml_add(ctx0, sub_cur, pos_embd);

        cb(sub_cur, "sub_chunk_embd", ci);
        sub_embeddings.push_back(sub_cur);
    }

    // -----------------------------------------------------------------------
    // Concatenate sub-chunk embeddings → [d_model, total_tokens]
    // -----------------------------------------------------------------------
    ggml_tensor * cur = sub_embeddings[0];
    for (int i = 1; i < n_sub_chunks; i++) {
        cur = ggml_concat(ctx0, cur, sub_embeddings[i], 1);
    }
    cb(cur, "after_concat", -1);

    // -----------------------------------------------------------------------
    // Attention: FULL (unmasked) within each inference window.
    // The block-diagonal restriction between inference windows is handled
    // by the mel preprocessor (separate encoder calls per window).
    // Within a window, all conv sub-chunks attend to each other — this is
    // the key difference from per-chunk processing and matches the Python
    // reference's cu_seqlens-based inference-window attention.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Transformer encoder layers (18x) with full attention
    // Pre-norm: LN → self-attn → residual → LN → FFN(GELU_ERF) → residual
    // -----------------------------------------------------------------------
    const int n_pos = total_tokens;

    // Sanity checks
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_w && model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].k_w && model.layers[0].k_b);
    GGML_ASSERT(model.layers[0].v_w && model.layers[0].v_b);
    GGML_ASSERT(model.layers[0].o_w && model.layers[0].o_b);

    ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        cur = inpL;

        // LayerNorm 1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ln1", il);

        // Self-attention with block-diagonal mask
        {
            ggml_tensor * Qcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
            ggml_tensor * Kcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
            ggml_tensor * Vcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // Residual 1
        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cb(cur, "ffn_inp", il);

        // LayerNorm 2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // FFN (GELU_ERF)
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            FFN_GELU_ERF, il);
        cb(cur, "ffn_out", il);

        // Residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);
        inpL = cur;
    }

    // Post-LayerNorm
    if (model.post_ln_w) {
        cur = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    } else {
        cur = inpL;
    }
    cb(cur, "after_encoder", -1);

    // -----------------------------------------------------------------------
    // Output projector: Linear(896, 896) + GELU + Linear(896, 1024)
    // -----------------------------------------------------------------------

    // proj1: Linear(d_model, d_model)
    cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
    if (model.mm_1_b) {
        cur = ggml_add(ctx0, cur, model.mm_1_b);
    }
    cur = ggml_gelu_erf(ctx0, cur);
    cb(cur, "proj1", -1);

    // proj2: Linear(d_model, output_dim)
    cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
    if (model.mm_2_b) {
        cur = ggml_add(ctx0, cur, model.mm_2_b);
    }
    cb(cur, "proj2", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
