// Copyright (c) 2026 codec.cpp contributors
// SPDX-License-Identifier: MIT

#include "models.h"
#include "../mtmd-audio.h"

// Soprano-1.1 Vocos: normalized Qwen3 hidden states -> STFT coefficients.
ggml_cgraph * clip_graph_soprano::build() {
    const auto & v = model.vocos;
    const int n_up = 4 * (n_frames - 1) + 1;
    ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, n_frames);
    ggml_set_name(cur, "inp_feats");
    ggml_set_input(cur);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_interpolate(ctx0, cur, n_up, 512, 1, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_add(ctx0, build_mm(v.input_w, cur), v.input_b);
    cur = build_norm(cur, v.norm_w, v.norm_b, NORM_TYPE_NORMAL, 1e-6f, -1);

    for (size_t il = 0; il < v.blocks.size(); ++il) {
        const auto & b = v.blocks[il];
        ggml_tensor * residual = cur;
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_conv_1d_dw(ctx0, b.dwconv_w, cur, 1, 1, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_add(ctx0, cur, b.dwconv_b);
        cur = build_norm(cur, b.norm_w, b.norm_b, NORM_TYPE_NORMAL, 1e-6f, il);
        cur = ggml_add(ctx0, build_mm(b.pw1_w, cur), b.pw1_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_add(ctx0, build_mm(b.pw2_w, cur), b.pw2_b);
        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, cur, b.gamma));
        cb(cur, "vocos_block", il);
    }
    cur = build_norm(cur, v.output_norm_w, v.output_norm_b, NORM_TYPE_NORMAL, 1e-6f, -1);
    cur = ggml_add(ctx0, build_mm(v.output_w, cur), v.output_b);
    cb(cur, "vocos_head", -1);

    const int n_bins = 1025;
    ggml_tensor * mag = ggml_view_2d(ctx0, cur, n_bins, n_up, cur->nb[1], 0);
    ggml_tensor * phase = ggml_view_2d(ctx0, cur, n_bins, n_up, cur->nb[1], n_bins * sizeof(float));
    mag = ggml_clamp(ctx0, ggml_exp(ctx0, ggml_cont(ctx0, mag)), 0.0f, 100.0f);
    phase = ggml_cont(ctx0, phase);
    ggml_tensor * re = ggml_mul(ctx0, mag, ggml_cos(ctx0, phase));
    ggml_tensor * im = ggml_mul(ctx0, mag, ggml_sin(ctx0, phase));
    re = ggml_reshape_3d(ctx0, re, 1, n_bins, n_up);
    im = ggml_reshape_3d(ctx0, im, 1, n_bins, n_up);
    cur = ggml_concat(ctx0, re, im, 0);
    ggml_set_name(cur, "out_spectrum");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

void clip_graph_soprano::decode_spectrum(const std::vector<float> & spectrum, std::vector<float> & pcm) {
    const size_t n_frames = spectrum.size() / 2050;
    mtmd_audio_streaming_istft istft(2048, 512);
    pcm.clear();
    for (size_t i = 0; i < n_frames; ++i) {
        auto frame = istft.process_frame(spectrum.data() + i * 2050);
        pcm.insert(pcm.end(), frame.begin(), frame.end());
    }
    auto tail = istft.flush();
    pcm.insert(pcm.end(), tail.begin(), tail.end());
    // The shared ISTFT removes 768 samples; torch.istft(center=True) removes 1024.
    pcm.erase(pcm.begin(), pcm.begin() + 256);
    pcm.resize((n_frames - 1) * 512);
}
