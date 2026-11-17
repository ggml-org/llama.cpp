#include "models.h"
#include "../mtmd-audio.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>

// same-length conv with 3 taps; x: [T, IC], w: [3, IC, OC], returns [T, OC]
ggml_tensor * mtmd_graph_tiny_voca::conv3(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const {
    ggml_tensor * col = ggml_im2col(ctx0, w, x, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx0,
            ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]),
            ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2]));
    return ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
}

ggml_cgraph * mtmd_graph_tiny_voca::build() {
    const auto & tv = model.tiny_voca;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * cur;
    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_IN) {
        // duration predictor: (conv, gelu, layer norm) x 2, then a linear layer
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        cur = conv3(cur, tv.dur_conv1_w, tv.dur_conv1_b);
        cur = ggml_gelu_erf(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));
        cur = build_norm(cur, tv.dur_norm1_w, tv.dur_norm1_b, NORM_TYPE_NORMAL, eps, -1);
        cur = conv3(ggml_cont(ctx0, ggml_transpose(ctx0, cur)), tv.dur_conv2_w, tv.dur_conv2_b);
        cur = ggml_gelu_erf(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));
        cur = build_norm(cur, tv.dur_norm2_w, tv.dur_norm2_b, NORM_TYPE_NORMAL, eps, -1);
        cur = ggml_add(ctx0, build_mm(tv.dur_out_w, cur), tv.dur_out_b);
    } else {
        GGML_ASSERT(gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_OUT);
        ggml_tensor * h = ggml_add(ctx0, build_mm(tv.harm_w,  inp), tv.harm_b);
        ggml_tensor * n = ggml_add(ctx0, build_mm(tv.noise_w, inp), tv.noise_b);
        cur = ggml_concat(ctx0, h, n, 0);
    }
    ggml_set_name(cur, "out_feats");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

//
// host: harmonic plus noise synthesis, see SourceFilter in the reference (model.py)
//

// band-limited pulse train with unit power, F0 given per frame and interpolated in the log domain
static void tiny_voca_pulse_train(const std::vector<float> & f0, std::vector<float> & out) {
    using g = mtmd_graph_tiny_voca;
    const int64_t n_frames = (int64_t) f0.size();
    std::vector<double> knots(n_frames + 1);
    for (int64_t i = 0; i < n_frames; i++) {
        knots[i] = std::log(std::max((double) f0[i], 1.0));
    }
    knots[n_frames] = knots[n_frames - 1];

    out.resize((size_t) (n_frames * g::hop));
    double acc = 0.0;
    for (int64_t n = 0; n < n_frames * g::hop; n++) {
        const int64_t low = n / g::hop;
        const double  w   = (double) (n - low * g::hop) / g::hop;
        const double  freq = std::exp(knots[low] * (1.0 - w) + knots[low + 1] * w);
        acc += freq / g::sample_rate;
        const double theta = std::fmod(2.0 * M_PI * acc, 2.0 * M_PI);
        const double harmonics = std::max((double) g::harmonic_limit * g::sample_rate / freq, 1.0);
        const double whole = std::floor(harmonics);
        const double denom = 2.0 * std::sin(theta / 2.0);
        const double core  = std::fabs(denom) < 1e-7 ? whole : std::sin((whole + 0.5) * theta) / denom - 0.5;
        const double wave  = core + (harmonics - whole) * std::cos((whole + 1.0) * theta);
        out[n] = (float) (wave * std::sqrt(2.0 / harmonics));
    }
}

// time-varying filter: STFT of the source (center, zero padding), times exp(rfft(cepstrum)) of each frame, ISTFT
// cep(j, buf) writes the n_fft cepstrum of STFT frame j, the source gets n_fft / 2 trailing zeros first
static void tiny_voca_filter(const std::vector<float> & src, int n_fft, int hop,
        const std::function<void(int64_t, float *)> & cep, std::vector<float> & out) {
    const int     n_bins = n_fft / 2 + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;      // length given to the STFT
    const int64_t n_stft = 1 + n_pad / hop;        // frames, center = true
    const int64_t n_full = n_pad + n_fft;          // with the center padding

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    const float * win = cache.hann_window.data();

    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);

    std::vector<float> ola((size_t) n_full, 0.0f), env((size_t) n_full, 0.0f);
    std::vector<float> fin(8 * n_fft), fout(8 * n_fft), rin(8 * n_fft), rout(8 * n_fft), cbuf(n_fft);
    std::vector<float> resp(2 * n_bins);
    for (int64_t j = 0; j < n_stft; j++) {
        // response of this frame
        std::fill(cbuf.begin(), cbuf.end(), 0.0f);
        cep(j, cbuf.data());
        std::copy(cbuf.begin(), cbuf.end(), rin.begin());
        mtmd_audio_fft(cache, rin.data(), n_fft, rout.data());
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(rout[2 * k], -14.0f, 6.0f));
            resp[2 * k + 0] = mag * std::cos(rout[2 * k + 1]);
            resp[2 * k + 1] = mag * std::sin(rout[2 * k + 1]);
        }

        // spectrum of the source frame
        const float * seg = x.data() + j * hop;
        for (int i = 0; i < n_fft; i++) {
            fin[i] = seg[i] * win[i];
        }
        mtmd_audio_fft(cache, fin.data(), n_fft, fout.data());

        // filtered spectrum, mirrored for the complex inverse FFT; irfft ignores the imaginary parts of DC and Nyquist
        for (int k = 0; k < n_bins; k++) {
            const float a = fout[2 * k], b = fout[2 * k + 1];
            const float c = resp[2 * k], d = resp[2 * k + 1];
            rin[2 * k + 0] = a * c - b * d;
            rin[2 * k + 1] = (k == 0 || k == n_fft / 2) ? 0.0f : a * d + b * c;
        }
        for (int k = 1; k < n_fft - n_bins + 1; k++) {
            rin[2 * (n_fft - k) + 0] =  rin[2 * k + 0];
            rin[2 * (n_fft - k) + 1] = -rin[2 * k + 1];
        }
        mtmd_audio_ifft(cache, rin.data(), n_fft, rout.data());

        float * o = ola.data() + j * hop;
        float * e = env.data() + j * hop;
        for (int i = 0; i < n_fft; i++) {
            o[i] += rout[2 * i] * win[i];
            e[i] += win[i] * win[i];
        }
    }

    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) {
        const int64_t k = i + n_fft / 2; // drop the center padding
        out[i] = env[k] > 1e-11f ? ola[k] / env[k] : 0.0f;
    }
}

void mtmd_graph_tiny_voca::synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    GGML_ASSERT(n_frames > 0 && voc.size() == (size_t) (n_frames * n_voc));

    // liftering: coefficient i scaled by 1 / (1 + i / 4), the gain (coefficient 0) by 2
    auto lifter = [](int i, int first) { return i == 0 && first == 0 ? 2.0f : 1.0f / (1.0f + (float) (i + first) / 4.0f); };

    // harmonic branch: envelope interpolated between frames, one gain per sub-frame
    const int split = n_cep_causal + n_cep_anti;
    auto harm_cep = [&](int64_t j, float * buf) {
        const int64_t s    = std::min(j, harm_steps * n_frames - 1);
        const float * here = voc.data() + (s / harm_steps) * n_voc;
        const float * next = voc.data() + std::min((s + 1) / harm_steps, n_frames - 1) * n_voc;
        for (int i = 0; i < n_cep_causal; i++) {
            buf[i] = 0.5f * (here[i] + next[i]) * lifter(i, 0);
        }
        buf[0] += here[split + s % harm_steps];
        for (int i = 0; i < n_cep_anti; i++) {
            buf[harm_fft - 1 - i] = 0.5f * (here[n_cep_causal + i] + next[n_cep_causal + i]) * lifter(i, 1);
        }
    };
    // noise branch: one envelope per frame, one gain per sub-frame
    auto noise_cep = [&](int64_t j, float * buf) {
        const int64_t s   = std::min(j, noise_steps * n_frames - 1);
        const float * row = voc.data() + (s / noise_steps) * n_voc + split + harm_steps;
        for (int i = 0; i < n_noise_cep; i++) {
            buf[i] = row[i] * lifter(i, 0);
        }
        buf[0] += row[n_noise_cep + s % noise_steps];
    };

    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    tiny_voca_pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : noise) {
        v = dist(rng);
    }
    tiny_voca_filter(pulses, harm_fft,  harm_hop,  harm_cep,  harm_out);
    tiny_voca_filter(noise,  noise_fft, noise_hop, noise_cep, noise_out);

    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) {
        audio[i] = harm_out[i] + noise_out[i];
    }
}

//
// host: unknown-word model of the phonemizer, see g2p.py of dizione
//

static const char * const tiny_voca_g2p_letters = "abcdefghijklmnopqrstuvwxyz'";
static const char * const tiny_voca_g2p_symbols[mtmd_graph_tiny_voca::n_g2p_symbol] = {
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
};

// one direction of a GRU layer over x [n, n_in] -> y [n, n_hidden], written at column offset of rows of width stride
// the products are summed in double and rounded once, as close as can be to the reference (numpy, float32): two
// symbols are sometimes almost as likely, and the order of a float32 sum would decide between them
static void tiny_voca_gru(const mtmd_tiny_voca::g2p_gru & g, const std::vector<float> & x, int64_t n, int64_t n_in, int64_t n_hidden,
        bool reverse, std::vector<float> & y, int64_t offset, int64_t stride) {
    const int64_t h3 = 3 * n_hidden;
    std::vector<float> inp((size_t) (n * h3));
    for (int64_t t = 0; t < n; t++) {
        for (int64_t r = 0; r < h3; r++) {
            const float * w = g.w_ih.data() + r * n_in;
            const float * xt = x.data() + t * n_in;
            double acc = 0.0;
            for (int64_t c = 0; c < n_in; c++) {
                acc += (double) w[c] * xt[c];
            }
            inp[t * h3 + r] = (float) acc + g.b_ih[r];
        }
    }
    std::vector<float> state((size_t) n_hidden, 0.0f), rec((size_t) h3);
    for (int64_t s = 0; s < n; s++) {
        const int64_t t = reverse ? n - 1 - s : s;
        for (int64_t r = 0; r < h3; r++) {
            const float * w = g.w_hh.data() + r * n_hidden;
            double acc = 0.0;
            for (int64_t c = 0; c < n_hidden; c++) {
                acc += (double) w[c] * state[c];
            }
            rec[r] = (float) acc + g.b_hh[r];
        }
        const float * in = inp.data() + t * h3;
        for (int64_t k = 0; k < n_hidden; k++) {
            const float reset  = 1.0f / (1.0f + std::exp(-(in[k] + rec[k])));
            const float update = 1.0f / (1.0f + std::exp(-(in[n_hidden + k] + rec[n_hidden + k])));
            const float cand   = std::tanh(in[2 * n_hidden + k] + reset * rec[2 * n_hidden + k]);
            state[k] = (1.0f - update) * cand + update * state[k];
        }
        std::copy(state.begin(), state.end(), y.begin() + t * stride + offset);
    }
}

std::string mtmd_graph_tiny_voca::g2p(const mtmd_tiny_voca & tv, const std::string & word) {
    const int64_t w = tv.g2p_n_width;
    const int64_t h = tv.g2p_n_hidden;
    if (w == 0 || tv.g2p_layers.empty()) {
        return "";
    }
    std::vector<int> ids;
    for (char c : word) {
        const char * p = strchr(tiny_voca_g2p_letters, std::tolower((unsigned char) c));
        if (c != '\0' && p) {
            ids.push_back((int) (p - tiny_voca_g2p_letters) + 1);
        }
    }
    if (ids.empty()) {
        return "";
    }

    // each letter takes n_g2p_slot positions: its embedding plus the embedding of the slot
    const int64_t n = (int64_t) ids.size() * n_g2p_slot;
    std::vector<float> x((size_t) (n * w));
    for (size_t l = 0; l < ids.size(); l++) {
        for (int s = 0; s < n_g2p_slot; s++) {
            float * row = x.data() + ((int64_t) l * n_g2p_slot + s) * w;
            for (int64_t c = 0; c < w; c++) {
                row[c] = tv.g2p_embd[ids[l] * w + c] + tv.g2p_slot[s * w + c];
            }
        }
    }
    int64_t n_in = w;
    for (size_t il = 0; il + 1 < tv.g2p_layers.size(); il += 2) {
        std::vector<float> y((size_t) (n * 2 * h));
        tiny_voca_gru(tv.g2p_layers[il],     x, n, n_in, h, false, y, 0, 2 * h);
        tiny_voca_gru(tv.g2p_layers[il + 1], x, n, n_in, h, true,  y, h, 2 * h);
        x    = std::move(y);
        n_in = 2 * h;
    }

    // CTC, read greedily: a label repeated or the blank (0) is no new symbol
    std::string out;
    int prev = 0;
    for (int64_t t = 0; t < n; t++) {
        int best = 0;
        float best_v = -INFINITY;
        for (int k = 0; k <= n_g2p_symbol; k++) {
            const float * wr = tv.g2p_out_w.data() + k * n_in;
            double sum = 0.0;
            for (int64_t c = 0; c < n_in; c++) {
                sum += (double) wr[c] * x[t * n_in + c];
            }
            const float acc = (float) sum + tv.g2p_out_b[k];
            if (acc > best_v) {
                best_v = acc;
                best   = k;
            }
        }
        if (best != prev && best != 0) {
            out += (out.empty() ? "" : " ") + std::string(tiny_voca_g2p_symbols[best - 1]);
        }
        prev = best;
    }
    return out;
}
