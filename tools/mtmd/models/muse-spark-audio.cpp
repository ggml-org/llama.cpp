#include "models.h"

// Muse Spark audio tower: a causal conv pyramid over the raw waveform, no attention and no spectrogram
// each stage folds `patch` samples into the channel dim, projects, then runs `depth` residual blocks (depthwise causal conv, RMS norm, GELU MLP)
ggml_cgraph * clip_graph_muse_spark_audio::build() {
    const int kernel = hparams.audio_conv_kernel_size;
    const int n_stage = (int) hparams.audio_patch_sizes.size();

    // shape [1, n_samples], so the first stage folds the same way as the others
    ggml_tensor * cur = build_inp_raw(1);
    cur = ggml_reshape_2d(ctx0, cur, 1, img.nx());
    cb(cur, "inp_raw", -1);

    int bid = 0;
    for (int stage = 0; stage < n_stage; ++stage) {
        const int patch = hparams.audio_patch_sizes[stage];
        const auto & pe = model.a_patches[stage];

        // fold `patch` positions into the channel dim
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * patch, cur->ne[1] / patch);
        cb(cur, "patch_folded", stage);

        GGML_ASSERT(pe.proj_w->ne[1] == hparams.audio_hidden_sizes[stage]);
        cur = ggml_mul_mat(ctx0, pe.proj_w, cur);
        cur = ggml_add(ctx0, cur, pe.proj_b);
        cur = build_norm(cur, pe.norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, stage);
        cb(cur, "patch_embd", stage);

        for (int i = 0; i < hparams.audio_depths[stage]; ++i, ++bid) {
            const auto & layer = model.layers[bid];
            ggml_tensor * residual = cur;

            // depthwise conv over time, pad on the left so it is causal
            ggml_tensor * conv = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
            conv = ggml_pad_ext(ctx0, conv, kernel - 1, 0, 0, 0, 0, 0, 0, 0);
            conv = ggml_conv_1d_dw(ctx0, layer.conv_w, conv, 1, 0, 1);
            conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));
            conv = ggml_add(ctx0, conv, layer.conv_b);
            cb(conv, "conv", bid);

            cur = build_norm(conv, layer.conv_norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, bid);

            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                nullptr, nullptr,
                layer.ff_down_w, layer.ff_down_b,
                FFN_GELU_ERF, bid);
            cb(cur, "ffn_out", bid);

            cur = ggml_add(ctx0, cur, residual);
            cb(cur, "block_out", bid);
        }
    }

    cur = build_norm(cur, model.pre_ln_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);
    cb(cur, "enc_norm", -1);

    // straight into the text hidden size, no adapter
    cur = ggml_mul_mat(ctx0, model.mm_fc_w, cur);
    cur = ggml_add(ctx0, cur, model.mm_fc_b);
    cur = build_norm(cur, model.post_ln_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
