#include "models.h"

ggml_cgraph * clip_graph_qwen3omni_audio::build() {
    // Qwen3-Omni Audio Encoder:
    // 1. Mel spectrogram input: [time, 128, 1] in ggml format (W, H, C)
    // 2. 3x Conv2d (stride 2, padding 1) with GELU
    //    Each conv reduces both time and mel_bins by factor of 2
    //    After 3 convs: [time/8, 16, 480] in ggml format (W, H, C)
    // 3. Flatten mel + channels: [time/8, 16*480] = [time/8, 7680]
    // 4. conv_out linear projection: [time/8, 7680] -> [time/8, 1280]
    // 5. Add sinusoidal position embeddings
    // 6. 32 transformer layers via build_vit()
    // 7. post_ln -> proj1 -> GELU -> proj2: [time/8, 1280] -> [time/8, 2048]

    const int n_frames = img.nx;  // time dimension
    const int n_mels = img.ny;    // 128 mel bins

    // After 3x stride-2 convolutions
    const int n_pos = n_frames / 8;
    const int n_mel_after_conv = n_mels / 8;  // 16

    GGML_ASSERT(n_mels == 128);  // Qwen3-Omni expects 128 mel bins
    GGML_ASSERT(model.position_embeddings && "position_embeddings tensor not loaded");
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    // Input: mel spectrogram [n_frames, n_mels, 1] from build_inp_raw(1)
    // In ggml this is [W=time, H=mels, C=1]
    ggml_tensor * inp = build_inp_raw(1);
    // Add batch dimension: [time, 128, 1, 1] = [W, H, C, N]
    inp = ggml_reshape_4d(ctx0, inp, n_frames, n_mels, 1, 1);
    cb(inp, "mel_input", -1);

    // Conv2d block: 3 layers with stride 2, padding 1
    // ggml_conv_2d expects:
    //   kernel: [KW, KH, IC, OC]
    //   input:  [W, H, C, N]
    //   output: [W', H', OC, N] where W'=(W+2*pad-KW)/stride+1

    // conv2d1: [time, 128, 1, 1] -> [time/2, 64, 480, 1]
    ggml_tensor * cur = ggml_conv_2d(ctx0, model.conv1d_1_w, inp, 2, 2, 1, 1, 1, 1);
    // Add bias - need to broadcast [480] to [W', H', 480, 1]
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.conv1d_1_b, 1, 1, 480, 1));
    cur = ggml_gelu(ctx0, cur);
    cb(cur, "after_conv2d_1", -1);

    // conv2d2: [time/2, 64, 480, 1] -> [time/4, 32, 480, 1]
    cur = ggml_conv_2d(ctx0, model.conv1d_2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.conv1d_2_b, 1, 1, 480, 1));
    cur = ggml_gelu(ctx0, cur);
    cb(cur, "after_conv2d_2", -1);

    // conv2d3: [time/4, 32, 480, 1] -> [time/8, 16, 480, 1]
    cur = ggml_conv_2d(ctx0, model.conv1d_3_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.conv1d_3_b, 1, 1, 480, 1));
    cur = ggml_gelu(ctx0, cur);
    cb(cur, "after_conv2d_3", -1);

    // Current shape: [time/8, 16, 480, 1] = [n_pos, n_mel_after_conv, 480, 1] in ggml [W, H, C, N]
    //
    // HuggingFace: x.permute(0, 3, 1, 2).view(b, t, c * f)
    //   Input: [N, C, H, W] = [1, 480, 16, 32]
    //   After permute(0,3,1,2): [N, W, C, H] = [1, 32, 480, 16]
    //   After view: [1, 32, 7680] where j = c*16 + h (h varies fastest 0-15, then c 0-479)
    //
    // ggml input: [W=n_pos, H=16, C=480, N=1]
    // We need: after flatten, j = h + c*16 (h varies fastest)
    //
    // To match HuggingFace: permute to [W, N, H, C] so H is innermost when flattened

    // Permute: [W, H, C, N] -> [W, N, H, C] = [n_pos, 1, 16, 480]
    // ggml_permute(a, axis0, axis1, axis2, axis3) puts:
    //   - original dim 0 at position axis0
    //   - original dim 1 at position axis1
    //   - original dim 2 at position axis2
    //   - original dim 3 at position axis3
    // We want: W->0, H->2, C->3, N->1
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 3, 1));
    // Reshape: [n_pos, 1, 16, 480] -> [n_pos, 16*480] = [n_pos, 7680]
    // Flattening gives: j = h + c*16
    cur = ggml_reshape_2d(ctx0, cur, n_pos, n_mel_after_conv * 480);
    // Transpose for mul_mat: [n_pos, 7680] -> [7680, n_pos]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cb(cur, "after_flatten", -1);

    // conv_out linear projection: [7680, n_pos] -> [1280, n_pos]
    // conv_out_w shape: [7680, 1280]
    cur = ggml_mul_mat(ctx0, model.conv_out_w, cur);
    cb(cur, "after_conv_out", -1);

    // Add position embeddings
    // position_embeddings shape: [1280, max_positions] in ggml format
    // cur shape: [1280, n_pos] - same format, just select first n_pos positions
    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    cur = ggml_add(ctx0, cur, pos_embd_selected);
    cb(cur, "after_pos_embd", -1);

    // Sanity check for transformer layers
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(model.layers[0].k_b);   // Qwen3-Omni audio has all QKV biases

    // 32 transformer layers via build_vit()
    // Input shape: [n_embd, n_pos] = [1280, n_pos] (standard ViT format)
    cur = build_vit(
        cur, n_pos,
        NORM_TYPE_NORMAL,
        hparams.ffn_op,
        nullptr,  // position embeddings already added
        nullptr   // no additional position function
    );
    cb(cur, "after_transformer", -1);

    // Qwen3-Omni audio projector: proj1 -> GELU -> proj2
    // Shape: [1280, n_pos] -> [1280, n_pos] -> [2048, n_pos]
    // Note: post_ln is already applied inside build_vit() if model.post_ln_w exists

    // proj1: [1280, n_pos] -> [1280, n_pos]
    cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
    cur = ggml_add(ctx0, cur, model.mm_1_b);
    cur = ggml_gelu(ctx0, cur);

    // proj2: [1280, n_pos] -> [2048, n_pos]
    cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
    cur = ggml_add(ctx0, cur, model.mm_2_b);

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
