#include "models.h"
#include "../mtmd-audio.h"

// Soprano Vocos: normalized Qwen3 hidden states -> STFT coefficients.
ggml_cgraph * clip_graph_soprano::build() {
    const auto & v = model.vocos;
    const int n_inp = hparams.projection_dim;
    const int n_up = upscale * (n_frames - 1) + 1;
    ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_inp, n_frames);
    ggml_set_name(cur, "inp_feats");
    ggml_set_input(cur);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_interpolate(ctx0, cur, n_up, n_inp, 1, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    if (ggml_n_dims(v.input_w) == 3) {
        const int k = v.input_w->ne[0];
        cur = ggml_conv_1d(ctx0, v.input_w, cur, 1, k / 2, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    } else {
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = build_mm(v.input_w, cur);
    }
    cur = ggml_add(ctx0, cur, v.input_b);
    cur = build_norm(cur, v.norm_w, v.norm_b, NORM_TYPE_NORMAL, eps, -1);

    for (size_t il = 0; il < v.blocks.size(); ++il) {
        const auto & b = v.blocks[il];
        const int k = b.dwconv_w->ne[0];
        ggml_tensor * residual = cur;
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_conv_1d_dw(ctx0, b.dwconv_w, cur, 1, k / 2, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_add(ctx0, cur, b.dwconv_b);
        cur = build_norm(cur, b.norm_w, b.norm_b, NORM_TYPE_NORMAL, eps, il);
        cur = ggml_add(ctx0, build_mm(b.pw1_w, cur), b.pw1_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_add(ctx0, build_mm(b.pw2_w, cur), b.pw2_b);
        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, cur, b.gamma));
        cb(cur, "vocos_block", il);
    }
    cur = build_norm(cur, v.output_norm_w, v.output_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cur = ggml_add(ctx0, build_mm(v.output_w, cur), v.output_b);
    cb(cur, "vocos_head", -1);

    // the head holds n_fft/2+1 log-magnitudes followed by n_fft/2+1 phases
    const int n_bins = v.output_w->ne[1] / 2;
    ggml_tensor * mag = ggml_view_2d(ctx0, cur, n_bins, n_up, cur->nb[1], 0);
    ggml_tensor * phase = ggml_view_2d(ctx0, cur, n_bins, n_up, cur->nb[1], n_bins * sizeof(float));
    mag = ggml_clamp(ctx0, ggml_exp(ctx0, ggml_cont(ctx0, mag)), 0.0f, 100.0f);
    phase = ggml_cont(ctx0, phase);
    ggml_tensor * re = ggml_mul(ctx0, mag, ggml_cos(ctx0, phase));
    ggml_tensor * im = ggml_mul(ctx0, mag, ggml_sin(ctx0, phase));
    re = ggml_reshape_3d(ctx0, re, 1, n_bins, n_up);
    im = ggml_reshape_3d(ctx0, im, 1, n_bins, n_up);
    cur = ggml_concat(ctx0, re, im, 0);
    ggml_set_name(cur, "out_audio");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

void clip_graph_soprano::decode_spectrum(const clip_model & model, const std::vector<float> & spectrum, std::vector<float> & pcm) {
    const int n_frame_vals = model.vocos.output_w->ne[1];
    const int n_fft = n_frame_vals - 2;
    const int hop_length = model.hparams.audio_hop_len;
    const size_t n_frames = spectrum.size() / n_frame_vals;
    mtmd_audio_streaming_istft istft(n_fft, hop_length);
    pcm.clear();
    for (size_t i = 0; i < n_frames; ++i) {
        auto frame = istft.process_frame(spectrum.data() + i * n_frame_vals);
        pcm.insert(pcm.end(), frame.begin(), frame.end());
    }
    auto tail = istft.flush();
    pcm.insert(pcm.end(), tail.begin(), tail.end());
    // the shared ISTFT removes (n_fft - hop) / 2 samples, torch.istft(center=True) removes n_fft / 2
    pcm.erase(pcm.begin(), pcm.begin() + hop_length / 2);
    pcm.resize((n_frames - 1) * hop_length);
}
