#include "mediagen-ltx-graph.h"

#include <cstdlib>
#include <cmath>

// LTX-2 audio: latents -> causal 2-D VAE -> mel -> BigVGAN style vocoder (16 kHz) -> bandwidth extension (48 kHz)
// layouts: 2-D activations [freq, time, C, 1], 1-D signals [T, C, 1]

//
// audio vae decoder
//

// 3x3 conv, zero padded: causal along time (2 before), symmetric along frequency
static ggml_tensor * avae_conv(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, int k) {
    ggml_tensor * kw = w.get(prefix + ".weight");
    ggml_tensor * b  = w.get(prefix + ".bias", false);
    if (k == 3) {
        x = ggml_pad_ext(ctx, x, 1, 1, 2, 0, 0, 0, 0, 0);
    }
    ggml_tensor * y = ggml_conv_2d(ctx, kw, x, 1, 1, 0, 0, 1, 1);
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
    }
    return y;
}

static ggml_tensor * avae_pixel_norm(ggml_context * ctx, ggml_tensor * x) {
    return ltx_pixel_norm(ctx, x, 1e-6f);
}

static ggml_tensor * avae_resblock(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x) {
    ggml_tensor * h = ggml_silu(ctx, avae_pixel_norm(ctx, x));
    h = avae_conv(ctx, w, prefix + ".conv1.conv", h, 3);
    h = ggml_silu(ctx, avae_pixel_norm(ctx, h));
    h = avae_conv(ctx, w, prefix + ".conv2.conv", h, 3);
    if (w.has(prefix + ".nin_shortcut.conv.weight")) {
        x = avae_conv(ctx, w, prefix + ".nin_shortcut.conv", x, 1);
    }
    return ggml_add(ctx, x, h);
}

// nearest x2 upsample, conv, drop the first time row
static ggml_tensor * avae_upsample(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x) {
    x = ggml_interpolate(ctx, x, x->ne[0] * 2, x->ne[1] * 2, x->ne[2], x->ne[3], GGML_SCALE_MODE_NEAREST);
    x = avae_conv(ctx, w, prefix + ".conv.conv", x, 3);
    x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1] - 1, x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], x->nb[1]);
    return ggml_cont(ctx, x);
}

// latents [freq, time, 8, 1] -> mel [freq=64, time, 2, 1]
static ggml_tensor * avae_decode(ggml_context * ctx, const mg_weights & w, ggml_tensor * z, int n_levels, int n_res_blocks) {
    const std::string pre = "audio_vae.decoder";
    ggml_tensor * h = avae_conv(ctx, w, pre + ".conv_in.conv", z, 3);
    h = avae_resblock(ctx, w, pre + ".mid.block_1", h);
    h = avae_resblock(ctx, w, pre + ".mid.block_2", h);
    for (int level = n_levels - 1; level >= 0; level--) {
        for (int i = 0; i < n_res_blocks + 1; i++) {
            h = avae_resblock(ctx, w, mg_format("%s.up.%d.block.%d", pre.c_str(), level, i), h);
        }
        if (level != 0) {
            h = avae_upsample(ctx, w, mg_format("%s.up.%d.upsample", pre.c_str(), level), h);
        }
    }
    h = ggml_silu(ctx, avae_pixel_norm(ctx, h));
    return avae_conv(ctx, w, pre + ".conv_out.conv", h, 3);
}

//
// vocoder building blocks (1-D, [T, C, 1])
//

static ggml_tensor * conv1d(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, int pad, int dil) {
    ggml_tensor * kw = w.get(prefix + ".weight");
    ggml_tensor * b  = w.get(prefix + ".bias", false);
    ggml_tensor * y  = ggml_conv_1d(ctx, kw, x, 1, pad, dil);
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
    }
    return y;
}

// PyTorch ConvTranspose1d(k, stride, padding = (k - stride) / 2)
static ggml_tensor * conv_transpose1d(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, int stride) {
    ggml_tensor * kw = w.get(prefix + ".weight"); // [K, OC, IC]
    ggml_tensor * b  = w.get(prefix + ".bias", false);
    const int k   = (int) kw->ne[0];
    const int pad = (k - stride) / 2;
    ggml_tensor * x2 = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    ggml_tensor * y  = ggml_conv_transpose_1d(ctx, kw, x2, stride, 0, 1); // [T', OC]
    if (pad > 0) {
        y = ggml_view_2d(ctx, y, y->ne[0] - 2 * pad, y->ne[1], y->nb[1], pad * y->nb[0]);
        y = ggml_cont(ctx, y);
    }
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], 1);
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
    }
    return y;
}

// replicate padding along time
static ggml_tensor * replicate_pad1d(ggml_context * ctx, ggml_tensor * x, int left, int right) {
    const int64_t T = x->ne[0], C = x->ne[1];
    ggml_tensor * out = x;
    if (left > 0) {
        ggml_tensor * first = ggml_view_3d(ctx, x, 1, C, 1, x->nb[1], x->nb[2], 0);
        ggml_tensor * rep = ggml_repeat(ctx, first, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, left, C, 1));
        out = ggml_concat(ctx, rep, out, 0);
    }
    if (right > 0) {
        ggml_tensor * last = ggml_view_3d(ctx, x, 1, C, 1, x->nb[1], x->nb[2], (T - 1) * x->nb[0]);
        ggml_tensor * rep = ggml_repeat(ctx, last, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, right, C, 1));
        out = ggml_concat(ctx, out, rep, 0);
    }
    return out;
}

// one filter for every channel: channels become the batch of a 1-channel conv; x: [T, C, 1], filter: [K, 1, 1]
static ggml_tensor * filter_conv1d(ggml_context * ctx, ggml_tensor * filter, ggml_tensor * x, int stride, int pad) {
    const int64_t T = x->ne[0], C = x->ne[1];
    ggml_tensor * xb = ggml_reshape_3d(ctx, x, T, 1, C); // channels as batch
    ggml_tensor * y  = ggml_conv_1d(ctx, filter, xb, stride, pad, 1); // [T', 1, C]
    return ggml_reshape_3d(ctx, y, y->ne[0], C, 1);
}

// zero insertion: [T, C] -> [(T-1)*ratio+1, C]
// concat rather than pad: the CUDA pad kernel maps ne1 to gridDim.y (max 65535)
static ggml_tensor * zero_stuff(ggml_context * ctx, ggml_tensor * x, int ratio) {
    const int64_t T = x->ne[0], C = x->ne[1];
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_cont(ctx, x), 1, T, C);
    ggml_tensor * zeros = ggml_scale(ctx, v, 0.0f);
    for (int r = 1; r < ratio; r++) {
        v = ggml_concat(ctx, v, zeros, 0);
    }
    v = ggml_reshape_3d(ctx, v, ratio * T, C, 1);
    v = ggml_view_3d(ctx, v, (T - 1) * ratio + 1, C, 1, v->nb[1], v->nb[2], 0);
    return ggml_cont(ctx, v);
}

// BigVGAN UpSample1d; the filter is symmetric so the transposed conv is zero stuffing + full conv
static ggml_tensor * aa_upsample(ggml_context * ctx, ggml_tensor * filter, ggml_tensor * x, int ratio, int pad, int pad_left, int pad_right) {
    const int k = (int) filter->ne[0];
    x = replicate_pad1d(ctx, x, pad, pad);
    x = zero_stuff(ctx, x, ratio);
    ggml_tensor * y = filter_conv1d(ctx, filter, x, 1, k - 1);
    y = ggml_scale(ctx, y, (float) ratio);
    const int64_t T = y->ne[0];
    y = ggml_view_3d(ctx, y, T - pad_left - pad_right, y->ne[1], 1, y->nb[1], y->nb[2], pad_left * y->nb[0]);
    return ggml_cont(ctx, y);
}

// BigVGAN DownSample1d: replicate pad then strided lowpass
static ggml_tensor * aa_downsample(ggml_context * ctx, ggml_tensor * filter, ggml_tensor * x, int ratio) {
    const int k = (int) filter->ne[0];
    x = replicate_pad1d(ctx, x, k / 2 - 1, k / 2);
    return filter_conv1d(ctx, filter, x, ratio, 0);
}

// snake beta: x + inv_beta * sin(alpha * x)^2, parameters precomputed by ltx_audio_transform
static ggml_tensor * snake_beta(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x) {
    ggml_tensor * a  = w.get(prefix + ".alpha");
    ggml_tensor * ib = w.get(prefix + ".beta");
    ggml_tensor * s  = ggml_sin(ctx, ggml_mul(ctx, x, ggml_reshape_3d(ctx, a, 1, a->ne[0], 1)));
    s = ggml_mul(ctx, ggml_sqr(ctx, s), ggml_reshape_3d(ctx, ib, 1, ib->ne[0], 1));
    return ggml_add(ctx, x, s);
}

// Activation1d: anti-aliased snake (2x up, act, 2x down), kernel 12
static ggml_tensor * activation1d(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x) {
    ggml_tensor * up   = w.get(prefix + ".upsample.filter");
    ggml_tensor * down = w.get(prefix + ".downsample.lowpass.filter");
    const int k = (int) up->ne[0];
    const int ratio = 2;
    const int pad = k / ratio - 1;
    const int pad_left  = pad * ratio + (k - ratio) / 2;
    const int pad_right = pad * ratio + (k - ratio + 1) / 2;
    x = aa_upsample(ctx, up, x, ratio, pad, pad_left, pad_right);
    x = snake_beta(ctx, w, prefix + ".act", x);
    return aa_downsample(ctx, down, x, ratio);
}

static ggml_tensor * amp_block(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, int kernel, const int dil[3]) {
    for (int i = 0; i < 3; i++) {
        ggml_tensor * xt = activation1d(ctx, w, mg_format("%s.acts1.%d", prefix.c_str(), i), x);
        xt = conv1d(ctx, w, mg_format("%s.convs1.%d", prefix.c_str(), i), xt, (kernel * dil[i] - dil[i]) / 2, dil[i]);
        xt = activation1d(ctx, w, mg_format("%s.acts2.%d", prefix.c_str(), i), xt);
        xt = conv1d(ctx, w, mg_format("%s.convs2.%d", prefix.c_str(), i), xt, (kernel - 1) / 2, 1);
        x = ggml_add(ctx, x, xt);
    }
    return x;
}

struct vocoder_cfg {
    std::vector<int> up_rates;
    std::vector<int> resblock_kernels = { 3, 7, 11 };
    int  dilations[3] = { 1, 3, 5 };
    bool clamp_output = true;
};

// x: [T, 128, 1] stacked stereo mels -> [T * prod(up_rates), 2, 1]
static ggml_tensor * vocoder(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, const vocoder_cfg & cfg) {
    x = conv1d(ctx, w, prefix + ".conv_pre", x, 3, 1);
    const int nk = (int) cfg.resblock_kernels.size();
    for (size_t i = 0; i < cfg.up_rates.size(); i++) {
        x = conv_transpose1d(ctx, w, mg_format("%s.ups.%zu", prefix.c_str(), i), x, cfg.up_rates[i]);
        ggml_tensor * xs = nullptr;
        for (int j = 0; j < nk; j++) {
            ggml_tensor * r = amp_block(ctx, w, mg_format("%s.resblocks.%zu", prefix.c_str(), i * nk + j), x, cfg.resblock_kernels[j], cfg.dilations);
            xs = xs ? ggml_add(ctx, xs, r) : r;
        }
        x = ggml_scale(ctx, xs, 1.0f / nk);
    }
    x = activation1d(ctx, w, prefix + ".act_post", x);
    x = conv1d(ctx, w, prefix + ".conv_post", x, 3, 1);
    if (cfg.clamp_output) {
        x = ggml_clamp(ctx, x, -1.0f, 1.0f);
    }
    return x;
}

// causal log-mel of one channel [T, 1, 1] -> [frames, n_mels, 1]
// one channel at a time: ggml_conv_1d only lays out its output correctly for a batch of one
static ggml_tensor * mel_stft(ggml_context * ctx, const mg_weights & w, ggml_tensor * x, int n_fft, int hop) {
    ggml_tensor * basis = w.get("vocoder.mel_stft.stft_fn.forward_basis"); // [n_fft, 1, 2 * n_freqs]
    ggml_tensor * melb  = w.get("vocoder.mel_stft.mel_basis");             // [n_freqs, n_mels]
    x = ggml_pad_ext(ctx, x, n_fft - hop, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * spec = ggml_conv_1d(ctx, basis, x, hop, 0, 1);            // [frames, 2 * n_freqs, 1]
    const int64_t F = spec->ne[0], nf = spec->ne[1] / 2;
    ggml_tensor * re = ggml_view_2d(ctx, spec, F, nf, spec->nb[1], 0);
    ggml_tensor * im = ggml_view_2d(ctx, spec, F, nf, spec->nb[1], nf * spec->nb[1]);
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, ggml_cont(ctx, re)), ggml_sqr(ctx, ggml_cont(ctx, im)))); // [F, nf]
    mag = ggml_cont(ctx, ggml_transpose(ctx, mag));                          // [nf, F]
    ggml_tensor * mel = ggml_mul_mat(ctx, melb, mag);                        // [n_mels, F]
    mel = ggml_log(ctx, ggml_clamp(ctx, mel, 1e-5f, INFINITY));
    mel = ggml_cont(ctx, ggml_transpose(ctx, mel));                          // [F, n_mels]
    return ggml_reshape_3d(ctx, mel, mel->ne[0], mel->ne[1], 1);
}

// torchaudio-style hann windowed sinc resampler filter (rolloff 0.99, width 6)
static void hann_resample_filter(int ratio, std::vector<float> & f, int & pad, int & pad_left, int & pad_right) {
    const double rolloff = 0.99;
    const int lowpass_filter_width = 6;
    const int width = (int) std::ceil(lowpass_filter_width / rolloff);
    const int k = 2 * width * ratio + 1;
    pad = width;
    pad_left  = 2 * width * ratio;
    pad_right = k - ratio;
    f.resize(k);
    for (int i = 0; i < k; i++) {
        const double t  = ((double) i / ratio - width) * rolloff;
        const double tc = std::max(-(double) lowpass_filter_width, std::min((double) lowpass_filter_width, t));
        const double win = std::pow(std::cos(tc * MG_PI / lowpass_filter_width / 2.0), 2.0);
        const double sinc = t == 0.0 ? 1.0 : std::sin(MG_PI * t) / (MG_PI * t);
        f[i] = (float) (sinc * win * rolloff / ratio);
    }
}

//
// entry point
//

bool ltx_audio_decode(const ltx_model & model, mg_backend & be, const ltx_latent_audio & lat,
                      int & sample_rate, int & n_channels, std::vector<float> & pcm) {
    const auto & hp = model.hp;
    const auto & w  = model.audio_vae;
    const int T = lat.n_frames, C = lat.channels, Fq = lat.freq;

    // denormalize, lay out as [freq, time, channels]
    std::vector<float> mean, stdv;
    if (!mg_tensor_to_f32(w.get("audio_vae.per_channel_statistics.mean-of-means"), C * Fq, mean) ||
        !mg_tensor_to_f32(w.get("audio_vae.per_channel_statistics.std-of-means"),  C * Fq, stdv)) {
        return false;
    }
    std::vector<float> z((size_t) Fq * T * C);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < Fq; f++) {
                const int k = c * Fq + f;
                z[f + (size_t) Fq * (t + (size_t) T * c)] = lat.x[(size_t) t * C * Fq + k] * stdv[k] + mean[k];
            }
        }
    }

    mg_graph g(be);
    ggml_context * ctx = g.get();
    ggml_tensor * zt = g.new_input_f32({ Fq, T, C, 1 }, z, "audio_latent");

    // vae decoder: -> [64, T_mel, 2, 1]
    ggml_tensor * mel = avae_decode(ctx, w, zt, 3, 2);
    const int t_mel = T * hp.audio_latent_downsample - (hp.audio_latent_downsample - 1);
    if (mel->ne[1] < t_mel || mel->ne[0] * mel->ne[2] != 128) {
        MG_ERR("%s: unexpected mel shape [%lld, %lld, %lld]\n", __func__, (long long) mel->ne[0], (long long) mel->ne[1], (long long) mel->ne[2]);
        return false;
    }
    if (mel->ne[1] != t_mel) {
        mel = ggml_cont(ctx, ggml_view_4d(ctx, mel, mel->ne[0], t_mel, mel->ne[2], 1, mel->nb[1], mel->nb[2], mel->nb[3], 0));
    }
    // stereo mels stacked along channels: [T_mel, 128, 1]
    ggml_tensor * vin = ggml_cont(ctx, ggml_permute(ctx, mel, 1, 0, 2, 3)); // [T_mel, 64, 2]
    vin = ggml_reshape_3d(ctx, vin, vin->ne[0], vin->ne[1] * vin->ne[2], 1);

    // vocoder to 16 kHz stereo: [T16, 2, 1]
    vocoder_cfg vcfg;
    vcfg.up_rates = { 5, 2, 2, 2, 2, 2 };
    ggml_tensor * x16 = vocoder(ctx, w, "vocoder.vocoder", vin, vcfg);

    ggml_tensor * out = x16;
    ggml_tensor * dbg_mel = nullptr, * dbg_skip = nullptr, * dbg_res = nullptr;
    int out_rate = hp.audio_sample_rate;
    if (w.has("vocoder.bwe_generator.conv_pre.weight")) {
        const int in_rate = 16000, out_rate_bwe = 48000, hop = 80, n_fft = 512;
        const int ratio = out_rate_bwe / in_rate;
        const int64_t t_low = x16->ne[0];
        const int64_t t_out = t_low * ratio;
        ggml_tensor * xpad = x16;
        if (t_low % hop != 0) {
            xpad = ggml_pad(ctx, x16, (int) (hop - t_low % hop), 0, 0, 0);
        }
        // mel of each low rate channel, stacked along channels: [frames, 128, 1]
        ggml_tensor * m = nullptr;
        for (int64_t c = 0; c < xpad->ne[1]; c++) {
            ggml_tensor * xc = ggml_view_3d(ctx, xpad, xpad->ne[0], 1, 1, xpad->nb[1], xpad->nb[2], c * xpad->nb[1]);
            ggml_tensor * mc = mel_stft(ctx, w, ggml_cont(ctx, xc), n_fft, hop);
            m = m ? ggml_concat(ctx, m, mc, 1) : mc;
        }
        vocoder_cfg bcfg;
        bcfg.up_rates = { 6, 5, 2, 2, 2 };
        bcfg.clamp_output = false;
        ggml_tensor * residual = vocoder(ctx, w, "vocoder.bwe_generator", m, bcfg); // [frames * 480, 2]
        // resampled low rate audio
        std::vector<float> hf;
        int pad, pad_left, pad_right;
        hann_resample_filter(ratio, hf, pad, pad_left, pad_right);
        ggml_tensor * hfilt = g.new_input_f32({ (int64_t) hf.size(), 1, 1 }, hf, "resample_filter");
        ggml_tensor * skip = aa_upsample(ctx, hfilt, xpad, ratio, pad, pad_left, pad_right);
        if (skip->ne[0] != residual->ne[0]) {
            MG_ERR("%s: bandwidth extension length mismatch\n", __func__);
            return false;
        }
        dbg_mel = m; dbg_skip = skip; dbg_res = residual;
        out = ggml_clamp(ctx, ggml_add(ctx, residual, skip), -1.0f, 1.0f);
        out = ggml_cont(ctx, ggml_view_3d(ctx, out, t_out, out->ne[1], 1, out->nb[1], out->nb[2], 0));
        out_rate = out_rate_bwe;
    }

    g.mark_output(out);
    const bool debug = getenv("MEDIAGEN_DUMP") != nullptr;
    if (debug) {
        g.mark_output(mel);
        g.mark_output(x16);
        if (dbg_mel) { g.mark_output(dbg_mel); g.mark_output(dbg_skip); g.mark_output(dbg_res); }
    }
    if (!g.compute(be)) {
        return false;
    }
    if (debug) {
        std::vector<float> tmp;
        g.get_output(mel, tmp);
        mg_dump("audio_mel", tmp);
        g.get_output(x16, tmp);
        mg_dump("audio_x16", tmp);
        if (dbg_mel) {
            g.get_output(dbg_mel, tmp);  mg_dump("audio_bwe_mel", tmp);
            g.get_output(dbg_skip, tmp); mg_dump("audio_bwe_skip", tmp);
            g.get_output(dbg_res, tmp);  mg_dump("audio_bwe_res", tmp);
        }
    }
    std::vector<float> y;
    g.get_output(out, y);
    const int64_t n = out->ne[0];
    const int nch = (int) out->ne[1];
    pcm.resize((size_t) n * nch);
    for (int64_t t = 0; t < n; t++) {
        for (int c = 0; c < nch; c++) {
            pcm[t * nch + c] = y[t + n * c];
        }
    }
    sample_rate = out_rate;
    n_channels  = nch;
    return true;
}

// snake parameters: alpha -> exp(alpha), beta -> 1 / (exp(beta) + eps)
void ltx_audio_transform(const std::string & name, float * data, int64_t n) {
    const bool is_alpha = name.size() > 10 && name.compare(name.size() - 10, 10, ".act.alpha") == 0;
    const bool is_beta  = name.size() > 9  && name.compare(name.size() - 9, 9, ".act.beta") == 0;
    if (is_alpha) {
        for (int64_t i = 0; i < n; i++) data[i] = std::exp(data[i]);
    } else if (is_beta) {
        for (int64_t i = 0; i < n; i++) data[i] = 1.0f / (std::exp(data[i]) + 1e-9f);
    }
}
