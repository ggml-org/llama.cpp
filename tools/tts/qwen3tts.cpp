// tools/tts/qwen3tts.cpp
// Qwen3-TTS CLI: text + reference audio → speech via llama.cpp
//
// Pipeline:
//   1. Speaker encoder (ECAPA-TDNN) extracts speaker embedding from reference WAV
//   2. Talker model generates codec token IDs (codebook 0) with MRoPE
//   3. Code predictor generates remaining 15 codebook tokens per frame
//   4. Vocoder decodes all 16 codebooks into audio waveform
//
// Usage:
//   llama-qwen3tts \
//     --model-talker talker.gguf \
//     --model-cp code-predictor.gguf \
//     --model-vocoder tokenizer.gguf \
//     --ref-audio reference.wav \
//     --text "Hello world" \
//     --output output.wav

#include "llama.h"
#include "common.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <map>
#include <numeric>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════════════════
//  WAV I/O helpers
// ═══════════════════════════════════════════════════════════════════════════════

static bool read_wav(const char * path, std::vector<float> & out, int target_sr = 24000) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return false; }

    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); fprintf(stderr, "ERROR: not RIFF\n"); return false; }
    fseek(f, 4, SEEK_CUR);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); fprintf(stderr, "ERROR: not WAVE\n"); return false; }

    int sr = 0, n_ch = 0, bps = 0; bool have_fmt = false;
    while (true) {
        char id[4]; uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) break;
        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t afmt, ch; uint32_t srate; uint16_t bps16;
            fread(&afmt, 2, 1, f); fread(&ch, 2, 1, f); fread(&srate, 4, 1, f);
            fseek(f, 4, SEEK_CUR); fseek(f, 2, SEEK_CUR); fread(&bps16, 2, 1, f);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
            if (afmt != 1) { fclose(f); fprintf(stderr, "ERROR: not PCM\n"); return false; }
            sr = (int)srate; n_ch = (int)ch; bps = (int)bps16; have_fmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            if (!have_fmt) { fclose(f); return false; }
            int n = (int)(sz / (bps / 8) / n_ch);
            out.resize(n);
            for (int i = 0; i < n; i++) {
                int16_t s16 = 0;
                for (int c = 0; c < n_ch; c++) {
                    int16_t cs; fread(&cs, 2, 1, f);
                    if (c == 0) s16 = cs;
                }
                out[i] = s16 / 32767.0f;
            }
            break;
        } else { fseek(f, sz, SEEK_CUR); }
    }
    fclose(f);
    if (sr != target_sr) {
        fprintf(stderr, "WARN: %s sample rate %d != %d (no resampling)\n", path, sr, target_sr);
    }
    printf("Read %s  (%d samples, %.2fs)\n", path, (int)out.size(), (float)out.size() / sr);
    return !out.empty();
}

static void write_wav(const char * path, const float * samples, int n, int sr = 24000) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }
    const int32_t dsz = n * 2, csz = 36 + dsz, brate = sr * 2;
    const int16_t n_ch = 1, bps = 16, balign = 2, pcm = 1;
    fwrite("RIFF", 1, 4, f); fwrite(&csz, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    const int32_t fmt = 16; fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&n_ch, 2, 1, f); fwrite(&sr, 4, 1, f);
    fwrite(&brate, 4, 1, f); fwrite(&balign, 2, 1, f); fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dsz, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float v = fmaxf(-1.0f, fminf(1.0f, samples[i]));
        int16_t s16 = (int16_t)(v * 32767.0f);
        fwrite(&s16, 2, 1, f);
    }
    fclose(f);
    printf("Wrote %s  (%d samples, %.2fs)\n", path, n, (float)n / sr);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Speaker Encoder (ECAPA-TDNN)
//  Extracts a 1024-dim speaker embedding from reference audio.
//  Weights are read from the Talker GGUF under the spk_enc.* namespace.
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int SPK_N_MELS   = 128;
static constexpr int SPK_N_FFT    = 1024;
static constexpr int SPK_HOP      = 256;
static constexpr int SPK_WIN      = 1024;
static constexpr int SPK_EMB_DIM  = 1024;
static constexpr int SPK_HIDDEN   = 512;
static constexpr int SPK_SCALE    = 8;
static constexpr int SPK_BRANCH   = SPK_HIDDEN / SPK_SCALE; // 64
#define SPK_MAX_NODES 16384

struct spk_res2net_block {
    ggml_tensor * tdnn1_w = nullptr;
    ggml_tensor * tdnn1_b = nullptr;
    ggml_tensor * res2net_w[7] = {};
    ggml_tensor * res2net_b[7] = {};
    ggml_tensor * tdnn2_w = nullptr;
    ggml_tensor * tdnn2_b = nullptr;
    ggml_tensor * se_conv1_w = nullptr;
    ggml_tensor * se_conv1_b = nullptr;
    ggml_tensor * se_conv2_w = nullptr;
    ggml_tensor * se_conv2_b = nullptr;
};

struct spk_encoder_model {
    ggml_tensor * conv0_w = nullptr;
    ggml_tensor * conv0_b = nullptr;
    spk_res2net_block blocks[3];
    ggml_tensor * mfa_w   = nullptr;
    ggml_tensor * mfa_b   = nullptr;
    ggml_tensor * asp_conv_w = nullptr;
    ggml_tensor * asp_conv_b = nullptr;
    ggml_tensor * asp_tdnn_w = nullptr;
    ggml_tensor * asp_tdnn_b = nullptr;
    ggml_tensor * fc_w    = nullptr;
    ggml_tensor * fc_b    = nullptr;
};

// Slaney mel filterbank (matches librosa norm='slaney')
static void compute_mel_filterbank(float * fb, int n_mels, int n_fft, int sr, float fmin, float fmax) {
    auto hz2mel = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;
        if (hz < 1000.0f) return hz / f_sp;
        return 1000.0f / f_sp + logf(hz / 1000.0f) / (logf(6.4f) / 27.0f);
    };
    auto mel2hz = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_mel = 1000.0f / f_sp;
        if (mel < min_log_mel) return f_sp * mel;
        return 1000.0f * expf((logf(6.4f) / 27.0f) * (mel - min_log_mel));
    };
    int bins = n_fft / 2 + 1;
    float mel_min = hz2mel(fmin), mel_max = hz2mel(fmax);
    std::vector<float> mel_pts(n_mels + 2), hz_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_pts[i] = mel2hz(mel_pts[i]);
    }
    memset(fb, 0, n_mels * bins * sizeof(float));
    for (int m = 0; m < n_mels; m++) {
        float fl = hz_pts[m], fc = hz_pts[m + 1], fr = hz_pts[m + 2];
        float enorm = 2.0f / (fr - fl);
        for (int k = 0; k < bins; k++) {
            float freq = (float)k * sr / n_fft;
            if (freq >= fl && freq <= fc && fc > fl)
                fb[m * bins + k] = enorm * (freq - fl) / (fc - fl);
            else if (freq > fc && freq <= fr && fr > fc)
                fb[m * bins + k] = enorm * (fr - freq) / (fr - fc);
        }
    }
}

static void compute_dft(const float * input, float * real, float * imag, int n) {
    for (int k = 0; k < n; k++) {
        real[k] = 0.0f; imag[k] = 0.0f;
        for (int t = 0; t < n; t++) {
            float angle = -2.0f * (float)M_PI * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] += input[t] * sinf(angle);
        }
    }
}

static bool compute_mel_spectrogram(const float * samples, int n_samples,
                                     std::vector<float> & mel, int & n_frames) {
    int padding = (SPK_N_FFT - SPK_HOP) / 2;
    int padded_len = n_samples + 2 * padding;

    std::vector<float> padded(padded_len);
    for (int i = 0; i < padded_len; i++) {
        int src;
        if (i < padding)                   src = padding - i;
        else if (i >= padding + n_samples) src = 2 * n_samples - (i - padding) - 2;
        else                               src = i - padding;
        padded[i] = samples[std::max(0, std::min(n_samples - 1, src))];
    }

    n_frames = (padded_len - SPK_N_FFT) / SPK_HOP + 1;
    if (n_frames <= 0) return false;

    int bins = SPK_N_FFT / 2 + 1;
    std::vector<float> filterbank(SPK_N_MELS * bins);
    compute_mel_filterbank(filterbank.data(), SPK_N_MELS, SPK_N_FFT, 24000, 0.0f, 12000.0f);

    std::vector<float> window(SPK_N_FFT, 0.0f);
    int offset = (SPK_N_FFT - SPK_WIN) / 2;
    for (int i = 0; i < SPK_WIN; i++)
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / SPK_WIN));

    mel.resize(SPK_N_MELS * n_frames);
    std::vector<float> frame(SPK_N_FFT), fft_re(SPK_N_FFT), fft_im(SPK_N_FFT);

    for (int fr = 0; fr < n_frames; fr++) {
        int start = fr * SPK_HOP;
        for (int i = 0; i < SPK_N_FFT; i++) frame[i] = padded[start + i] * window[i];
        compute_dft(frame.data(), fft_re.data(), fft_im.data(), SPK_N_FFT);
        for (int m = 0; m < SPK_N_MELS; m++) {
            float sum = 0.0f;
            for (int k = 0; k < bins; k++) {
                float mag = sqrtf(fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k] + 1e-9f);
                sum += filterbank[m * bins + k] * mag;
            }
            mel[m * n_frames + fr] = logf(std::max(sum, 1e-5f));
        }
    }
    return true;
}

static ggml_tensor * spk_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int stride, int pad, int dilation) {
    if (w->type != GGML_TYPE_F16) {
        w = ggml_cast(ctx, w, GGML_TYPE_F16);
    }
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

struct gguf_tensor_loader {
    ggml_context * ctx = nullptr;
    struct gguf_context * guf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    ~gguf_tensor_loader() {
        if (guf) gguf_free(guf);
        if (ctx) ggml_free(ctx);
    }

    bool load(const char * path, const char * prefix) {
        struct gguf_init_params params;
        params.no_alloc = false;
        params.ctx = &ctx;
        guf = gguf_init_from_file(path, params);
        if (!guf) {
            fprintf(stderr, "ERROR: cannot open GGUF: %s\n", path);
            return false;
        }
        int64_t n = gguf_get_n_tensors(guf);
        int loaded = 0;
        size_t prefix_len = strlen(prefix);
        for (int64_t i = 0; i < n; i++) {
            const char * name = gguf_get_tensor_name(guf, i);
            if (strncmp(name, prefix, prefix_len) == 0) {
                ggml_tensor * t = ggml_get_tensor(ctx, name);
                if (t) {
                    tensors[name] = t;
                    loaded++;
                }
            }
        }
        printf("Loaded %d tensors with prefix '%s' from %s\n", loaded, prefix, path);
        return loaded > 0;
    }

    ggml_tensor * get(const char * name) const {
        auto it = tensors.find(name);
        return (it != tensors.end()) ? it->second : nullptr;
    }
};

static bool spk_load_weights(gguf_tensor_loader & loader, spk_encoder_model & spk) {
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = loader.get(name);
        if (!t) fprintf(stderr, "WARN: missing tensor: %s\n", name);
        return t;
    };

    spk.conv0_w = get("spk_enc.conv0.weight");
    spk.conv0_b = get("spk_enc.conv0.bias");

    for (int b = 0; b < 3; b++) {
        char buf[128];
        auto & blk = spk.blocks[b];
        int hf_idx = b + 1;
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn1.weight", hf_idx);     blk.tdnn1_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn1.bias", hf_idx);       blk.tdnn1_b = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn2.weight", hf_idx);     blk.tdnn2_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.tdnn2.bias", hf_idx);       blk.tdnn2_b = get(buf);
        for (int r = 0; r < 7; r++) {
            snprintf(buf, sizeof(buf), "spk_enc.blk.%d.res2net.%d.weight", hf_idx, r); blk.res2net_w[r] = get(buf);
            snprintf(buf, sizeof(buf), "spk_enc.blk.%d.res2net.%d.bias", hf_idx, r);   blk.res2net_b[r] = get(buf);
        }
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv1.weight", hf_idx);  blk.se_conv1_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv1.bias", hf_idx);    blk.se_conv1_b = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv2.weight", hf_idx);  blk.se_conv2_w = get(buf);
        snprintf(buf, sizeof(buf), "spk_enc.blk.%d.se.conv2.bias", hf_idx);    blk.se_conv2_b = get(buf);
    }

    spk.mfa_w     = get("spk_enc.mfa.weight");
    spk.mfa_b     = get("spk_enc.mfa.bias");
    spk.asp_conv_w = get("spk_enc.asp.conv.weight");
    spk.asp_conv_b = get("spk_enc.asp.conv.bias");
    spk.asp_tdnn_w = get("spk_enc.asp.tdnn.weight");
    spk.asp_tdnn_b = get("spk_enc.asp.tdnn.bias");
    spk.fc_w      = get("spk_enc.fc.weight");
    spk.fc_b      = get("spk_enc.fc.bias");

    bool ok = spk.conv0_w && spk.fc_w;
    if (!ok) fprintf(stderr, "ERROR: missing critical speaker encoder tensors\n");
    return ok;
}

static ggml_cgraph * spk_build_graph(ggml_context * ctx0, const spk_encoder_model & m, int n_frames) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, SPK_MAX_NODES, false);
    const int64_t seq_len_init = n_frames;

    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, SPK_N_MELS);
    ggml_set_name(mel, "mel_input"); ggml_set_input(mel);

    ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, SPK_N_MELS, 1);

    // Reflect pad left by 2
    // For the initial conv (kernel=5), we need pad=2 on each side
    // Using ggml_pad_ext for left padding
    cur = ggml_pad_ext(ctx0, cur, 2, 0, 0, 0, 0, 0, 0, 0);
    {
        ggml_tensor * conv0_w = m.conv0_w;
        if (conv0_w->type != GGML_TYPE_F16) {
            conv0_w = ggml_cast(ctx0, conv0_w, GGML_TYPE_F16);
        }
        cur = ggml_conv_1d(ctx0, conv0_w, cur, 1, 0, 1);
    }
    if (m.conv0_b) {
        int64_t oc = cur->ne[1];
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.conv0_b, 1, oc, 1));
    }
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "conv0_out");

    int64_t seq_len = cur->ne[0];

    ggml_tensor * block_outputs[4];
    block_outputs[0] = cur;

    int dilations[3] = {2, 3, 4};

    for (int blk = 0; blk < 3; blk++) {
        const auto & block = m.blocks[blk];
        int dilation = dilations[blk];

        ggml_tensor * residual = cur;

        cur = spk_conv1d(ctx0, block.tdnn1_w, block.tdnn1_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);

        // Res2Net: split into 8 branches of 64 channels
        ggml_tensor * branches[SPK_SCALE];
        for (int b = 0; b < SPK_SCALE; b++) {
            branches[b] = ggml_view_3d(ctx0, cur, seq_len, SPK_BRANCH, 1,
                                        cur->nb[1], cur->nb[2], b * SPK_BRANCH * cur->nb[1]);
            branches[b] = ggml_cont(ctx0, branches[b]);
        }

        ggml_tensor * outputs[SPK_SCALE];
        outputs[0] = branches[0];
        for (int b = 1; b < SPK_SCALE; b++) {
            ggml_tensor * input = (b == 1) ? branches[b] : ggml_add(ctx0, branches[b], outputs[b - 1]);
            if (block.res2net_w[b - 1]) {
                int pad_val = (3 - 1) * dilation;
                input = ggml_pad_ext(ctx0, input, pad_val, 0, 0, 0, 0, 0, 0, 0);
                outputs[b] = spk_conv1d(ctx0, block.res2net_w[b - 1], block.res2net_b[b - 1],
                                        input, 1, 0, dilation);
                outputs[b] = ggml_relu(ctx0, outputs[b]);
            } else {
                outputs[b] = input;
            }
        }

        cur = outputs[0];
        for (int b = 1; b < SPK_SCALE; b++)
            cur = ggml_concat(ctx0, cur, outputs[b], 1);

        cur = spk_conv1d(ctx0, block.tdnn2_w, block.tdnn2_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);

        // SE (Squeeze-Excitation)
        ggml_tensor * se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
        se = ggml_reshape_3d(ctx0, se, 1, SPK_HIDDEN, 1);
        se = spk_conv1d(ctx0, block.se_conv1_w, block.se_conv1_b, se, 1, 0, 1);
        se = ggml_relu(ctx0, se);
        se = spk_conv1d(ctx0, block.se_conv2_w, block.se_conv2_b, se, 1, 0, 1);
        se = ggml_sigmoid(ctx0, se);
        cur = ggml_mul(ctx0, cur, se);

        cur = ggml_add(ctx0, cur, residual);
        block_outputs[blk + 1] = cur;
    }

    // MFA: concatenate block outputs [1:]
    ggml_tensor * mfa_in = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
    mfa_in = ggml_concat(ctx0, mfa_in, block_outputs[3], 1);
    cur = spk_conv1d(ctx0, m.mfa_w, m.mfa_b, mfa_in, 1, 0, 1);
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "mfa_out");

    // ASP (Attentive Statistics Pooling)
    const int mfa_ch = 1536;
    ggml_tensor * global_mean = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    global_mean = ggml_reshape_3d(ctx0, global_mean, 1, mfa_ch, 1);

    ggml_tensor * sq = ggml_sqr(ctx0, cur);
    ggml_tensor * mean_sq = ggml_pool_1d(ctx0, sq, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    mean_sq = ggml_reshape_3d(ctx0, mean_sq, 1, mfa_ch, 1);
    ggml_tensor * var = ggml_sub(ctx0, mean_sq, ggml_sqr(ctx0, global_mean));
    var = ggml_clamp(ctx0, var, 1e-12f, 1e10f);
    ggml_tensor * global_std = ggml_sqrt(ctx0, var);

    ggml_tensor * ref_3d = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, mfa_ch, 1);
    ggml_tensor * mean_exp = ggml_repeat(ctx0, global_mean, ref_3d);
    ggml_tensor * std_exp  = ggml_repeat(ctx0, global_std, ref_3d);

    ggml_tensor * attn = ggml_concat(ctx0, cur, mean_exp, 1);
    attn = ggml_concat(ctx0, attn, std_exp, 1);

    attn = spk_conv1d(ctx0, m.asp_tdnn_w, m.asp_tdnn_b, attn, 1, 0, 1);
    attn = ggml_relu(ctx0, attn);
    attn = ggml_tanh(ctx0, attn);
    attn = spk_conv1d(ctx0, m.asp_conv_w, m.asp_conv_b, attn, 1, 0, 1);
    attn = ggml_soft_max(ctx0, attn);

    ggml_tensor * weighted = ggml_mul(ctx0, attn, cur);
    ggml_tensor * w_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    w_mean = ggml_scale(ctx0, w_mean, (float)seq_len);
    w_mean = ggml_reshape_3d(ctx0, w_mean, 1, mfa_ch, 1);

    ggml_tensor * mean_for_std = ggml_repeat(ctx0, w_mean, ref_3d);
    ggml_tensor * diff = ggml_sub(ctx0, cur, mean_for_std);
    ggml_tensor * diff_sq = ggml_sqr(ctx0, diff);
    ggml_tensor * w_var = ggml_mul(ctx0, attn, diff_sq);
    ggml_tensor * var_sum = ggml_pool_1d(ctx0, w_var, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    var_sum = ggml_scale(ctx0, var_sum, (float)seq_len);
    var_sum = ggml_reshape_3d(ctx0, var_sum, 1, mfa_ch, 1);
    var_sum = ggml_clamp(ctx0, var_sum, 1e-12f, 1e10f);
    ggml_tensor * w_std = ggml_sqrt(ctx0, var_sum);

    ggml_tensor * pooled = ggml_concat(ctx0, w_mean, w_std, 1);

    cur = spk_conv1d(ctx0, m.fc_w, m.fc_b, pooled, 1, 0, 1);
    cur = ggml_reshape_1d(ctx0, cur, SPK_EMB_DIM);
    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    (void)seq_len_init;
    return gf;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Speech Tokenizer Encoder (MimiModel-based)
//  Encodes raw audio → discrete codec tokens for ICL voice cloning.
//  Architecture: SEANet encoder → transformer (8 layers, RoPE) → SplitRVQ
//  Ported from llama.cpp mimi codec implementation.
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int ENC_DIM        = 512;
static constexpr int ENC_VQ_DIM     = 256;
static constexpr int ENC_VQ_BINS    = 2048;
static constexpr int ENC_N_Q        = 32;
static constexpr int ENC_N_HEADS    = 8;
static constexpr int ENC_HEAD_DIM   = ENC_DIM / ENC_N_HEADS;  // 64
static constexpr int ENC_FFN_DIM    = 2048;
static constexpr int ENC_N_TR       = 8;
static constexpr int ENC_N_SBLOCKS  = 4;
static constexpr float ENC_NORM_EPS = 1e-5f;
static constexpr float ENC_ROPE_THETA = 10000.0f;

static const int enc_ratios[4]  = {4, 5, 6, 8};
static const int enc_ch_in[4]   = { 64, 128, 256,  512};
static const int enc_ch_out[4]  = {128, 256, 512, 1024};
static const int enc_kernels[4] = {8, 10, 12, 16};

struct enc_seanet_block {
    ggml_tensor * res_conv1   = nullptr;  // residual conv1 [ch/2, ch, 3]
    ggml_tensor * res_conv1_b = nullptr;
    ggml_tensor * res_conv2   = nullptr;  // residual conv2 [ch, ch/2] (k=1, stored as 2D)
    ggml_tensor * res_conv2_b = nullptr;
    ggml_tensor * conv_stride   = nullptr;  // strided conv [ch_out, ch_in, 2*stride]
    ggml_tensor * conv_stride_b = nullptr;
};

struct enc_tr_layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_scale  = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_norm_b  = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
    ggml_tensor * ffn_scale   = nullptr;
};

struct enc_model {
    ggml_tensor * conv_in     = nullptr;  // [7, 1, 64]
    ggml_tensor * conv_in_b   = nullptr;
    enc_seanet_block blocks[ENC_N_SBLOCKS];
    ggml_tensor * conv_out    = nullptr;  // [3, 1024, 512]
    ggml_tensor * conv_out_b  = nullptr;
    ggml_tensor * downsample  = nullptr;  // [4, 512, 512]
    ggml_tensor * downsample_b = nullptr;
    enc_tr_layer tr[ENC_N_TR];
    ggml_tensor * vq_semantic_input_proj = nullptr;   // [512, 256]
    ggml_tensor * vq_acoustic_input_proj = nullptr;   // [512, 256]
    ggml_tensor * vq_semantic_codebook[1]  = {};      // codebook 0
    ggml_tensor * vq_acoustic_codebook[31] = {};      // codebooks 1..31
};

static bool enc_load_weights(gguf_tensor_loader & loader, enc_model & enc) {
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = loader.get(name);
        return t;
    };

    // SEANet conv_in: encoder.layers[0] = initial conv (k=7, 1→64)
    enc.conv_in   = get("tok_enc.conv.0.weight");
    enc.conv_in_b = get("tok_enc.conv.0.bias");

    // SEANet blocks: encoder.layers[1..4] each have residual + strided conv
    // HF encoder.layers layout: [conv_in, block0, block1, block2, block3, conv_out, ...]
    // Each block_i = encoder.layers[1+i*3 .. 3+i*3] containing residual sub-block + ELU + strided conv
    // In the GGUF: tok_enc.conv.{idx} and tok_enc.res.{idx}
    // Layout from MimiEncoder:
    //   layers[0] = conv_in
    //   layers[1] = ResidualUnit (block 0), layers[2] = ELU, layers[3] = strided conv (block 0)
    //   layers[4] = ResidualUnit (block 1), layers[5] = ELU, layers[6] = strided conv (block 1)
    //   layers[7] = ResidualUnit (block 2), layers[8] = ELU, layers[9] = strided conv (block 2)
    //   layers[10] = ResidualUnit (block 3), layers[11] = ELU, layers[12] = strided conv (block 3)
    //   layers[13] = ELU, layers[14] = conv_out
    // SEANet encoder layer layout:
    //   0=conv_in, 1=res0, 2=elu, 3=stride0, 4=res1, 5=elu, 6=stride1,
    //   7=res2, 8=elu, 9=stride2, 10=res3, 11=elu, 12=stride3, 13=elu, 14=conv_out
    // Residual sub-blocks: blk.1 = conv1 (k=3, ch→ch/2), blk.3 = conv2 (k=1, ch/2→ch)
    static const int res_indices[4]    = {1, 4, 7, 10};
    static const int stride_indices[4] = {3, 6, 9, 12};
    for (int i = 0; i < ENC_N_SBLOCKS; i++) {
        auto & blk = enc.blocks[i];
        char buf[128];
        int ri = res_indices[i];
        int si = stride_indices[i];
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.1.weight", ri);  blk.res_conv1   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.1.bias", ri);    blk.res_conv1_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.3.weight", ri);  blk.res_conv2   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.res.%d.blk.3.bias", ri);    blk.res_conv2_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.conv.%d.weight", si);       blk.conv_stride   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.conv.%d.bias", si);         blk.conv_stride_b = get(buf);
    }

    enc.conv_out   = get("tok_enc.conv.14.weight");
    enc.conv_out_b = get("tok_enc.conv.14.bias");

    enc.downsample   = get("tok_enc.downsample.weight");
    enc.downsample_b = nullptr;

    for (int i = 0; i < ENC_N_TR; i++) {
        char buf[128];
        auto & tr = enc.tr[i];
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_norm.weight", i);  tr.attn_norm   = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_norm.bias", i);    tr.attn_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_q.weight", i);     tr.attn_q      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_k.weight", i);     tr.attn_k      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_v.weight", i);     tr.attn_v      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_output.weight", i); tr.attn_output = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.attn_scale", i);        tr.attn_scale  = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_norm.weight", i);   tr.ffn_norm    = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_norm.bias", i);     tr.ffn_norm_b  = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_up.weight", i);     tr.ffn_up      = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_down.weight", i);   tr.ffn_down    = get(buf);
        snprintf(buf, sizeof(buf), "tok_enc.blk.%d.ffn_scale", i);         tr.ffn_scale   = get(buf);
    }

    enc.vq_semantic_input_proj = get("tok_enc.vq_semantic.input_proj.weight");
    enc.vq_acoustic_input_proj = get("tok_enc.vq_acoustic.input_proj.weight");

    // Codebooks: semantic has 1 codebook, acoustic has 31
    enc.vq_semantic_codebook[0] = get("tok_enc.vq_semantic.0.codebook");
    for (int i = 0; i < 31; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "tok_enc.vq_acoustic.%d.codebook", i);
        enc.vq_acoustic_codebook[i] = get(buf);
    }

    return enc.conv_in && enc.conv_out && enc.downsample &&
           enc.vq_semantic_input_proj && enc.vq_semantic_codebook[0];
}

// ELU activation
static inline float enc_elu(float x) { return x >= 0.f ? x : expf(x) - 1.f; }

// Causal Conv1d in C++. Weight layout: [OC, IC, K], data is channel-major [C*T].
// Causal padding = kernel - stride (matches HF MimiConv1d with use_causal_conv=True).
static void enc_causal_conv1d(
        const float * x,  int C_in,  int T_in,
        const float * w,  int C_out, int K,
        const float * b,  float * y, int stride) {
    const int pad   = K - stride;
    const int T_out = (T_in + pad - K) / stride + 1;
    memset(y, 0, (size_t)C_out * T_out * sizeof(float));
    for (int t_out = 0; t_out < T_out; t_out++) {
        for (int k = 0; k < K; k++) {
            int t_in = t_out * stride + k - pad;
            if (t_in < 0 || t_in >= T_in) continue;
            for (int co = 0; co < C_out; co++) {
                for (int ci = 0; ci < C_in; ci++) {
                    y[co * T_out + t_out] += x[ci * T_in + t_in] * w[co * C_in * K + ci * K + k];
                }
            }
        }
    }
    if (b) for (int co = 0; co < C_out; co++)
        for (int t = 0; t < T_out; t++) y[co * T_out + t] += b[co];
}

// Mat-mul for k=1 conv (weight is 2D [ch_out, ch_in])
static void enc_mul_mat(const float * w, int ci, int co,
                        const float * b, const float * x, int T, float * y) {
    memset(y, 0, (size_t)co * T * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < co; o++) {
            float s = 0.f;
            for (int i = 0; i < ci; i++) s += x[i * T + t] * w[i + o * ci];
            y[o * T + t] = s;
        }
    }
    if (b) for (int o = 0; o < co; o++)
        for (int t = 0; t < T; t++) y[o * T + t] += b[o];
}

// SEANet residual block (identity skip)
static void enc_seanet_residual(
        const float * x, int ch, int T,
        const float * w1, const float * b1,
        const float * w2, const float * b2,
        float * y) {
    const int ch2 = ch / 2;
    std::vector<float> eu(ch * T), tmp(ch2 * T), branch(ch * T);
    for (int i = 0; i < ch * T; i++) eu[i] = enc_elu(x[i]);
    enc_causal_conv1d(eu.data(), ch, T, w1, ch2, 3, b1, tmp.data(), 1);
    for (int i = 0; i < ch2 * T; i++) tmp[i] = enc_elu(tmp[i]);
    enc_mul_mat(w2, ch2, ch, b2, tmp.data(), T, branch.data());
    for (int i = 0; i < ch * T; i++) y[i] = x[i] + branch[i];
}

// Get float data from a tensor (handles BF16 by casting to F32)
static const float * enc_get_f32(ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return nullptr;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_BF16) {
        const uint16_t * src = (const uint16_t *)t->data;
        for (int64_t i = 0; i < n; i++) {
            uint32_t tmp = (uint32_t)src[i] << 16;
            float f; memcpy(&f, &tmp, sizeof(f));
            buf[i] = f;
        }
    }
    return buf.data();
}

// SEANet encoder: raw audio → 512-dim latent (channel-major)
static std::vector<float> enc_seanet_encode(enc_model & m, const float * audio, int T_audio, int * T_out) {
    std::vector<float> b_conv_in, b_conv_in_b, b_res1, b_res1b, b_res2, b_res2b;
    std::vector<float> b_stride, b_strideb, b_co, b_cob, b_ds, b_dsb;

    const float * w_in  = enc_get_f32(m.conv_in,   b_conv_in);
    const float * b_in  = enc_get_f32(m.conv_in_b, b_conv_in_b);

    int T_cur = T_audio;
    std::vector<float> buf(64 * T_cur);
    enc_causal_conv1d(audio, 1, T_cur, w_in, 64, 7, b_in, buf.data(), 1);
    {
        float rms = 0.f;
        for (int i = 0; i < 64 * T_cur; i++) rms += buf[i] * buf[i];
        rms = sqrtf(rms / (64 * T_cur));
        printf("  [dbg] conv_in: T=%d, rms=%.6f, ch0[:5]=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               T_cur, rms, buf[0], buf[T_cur], buf[2*T_cur], buf[3*T_cur], buf[4*T_cur]);
        printf("  [dbg] conv_in ch0 first5 samples: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               buf[0], buf[1], buf[2], buf[3], buf[4]);
    }

    for (int i = 0; i < ENC_N_SBLOCKS; i++) {
        auto & blk = m.blocks[i];
        const int Ci  = enc_ch_in[i];
        const int Co  = enc_ch_out[i];
        const int K   = enc_kernels[i];
        const int str = enc_ratios[i];

        const float * w1  = enc_get_f32(blk.res_conv1,   b_res1);
        const float * b1  = enc_get_f32(blk.res_conv1_b, b_res1b);
        const float * w2  = enc_get_f32(blk.res_conv2,   b_res2);
        const float * b2  = enc_get_f32(blk.res_conv2_b, b_res2b);

        std::vector<float> res(Ci * T_cur);
        enc_seanet_residual(buf.data(), Ci, T_cur, w1, b1, w2, b2, res.data());
        buf = std::move(res);
        {
            float rms = 0.f;
            for (int j = 0; j < Ci * T_cur; j++) rms += buf[j] * buf[j];
            rms = sqrtf(rms / (Ci * T_cur));
            printf("  [dbg] blk%d res: T=%d, rms=%.6f, ch0[0]=%.6f\n", i, T_cur, rms, buf[0]);
        }

        for (auto & v : buf) v = enc_elu(v);

        const float * ws = enc_get_f32(blk.conv_stride,   b_stride);
        const float * bs = enc_get_f32(blk.conv_stride_b, b_strideb);
        const int T_ds = (T_cur - 1) / str + 1;
        std::vector<float> ds(Co * T_ds);
        enc_causal_conv1d(buf.data(), Ci, T_cur, ws, Co, K, bs, ds.data(), str);
        buf = std::move(ds);
        T_cur = T_ds;
        {
            float rms = 0.f;
            for (int j = 0; j < Co * T_cur; j++) rms += buf[j] * buf[j];
            rms = sqrtf(rms / (Co * T_cur));
            printf("  [dbg] blk%d stride: T=%d, rms=%.6f, ch0[0]=%.6f\n", i, T_cur, rms, buf[0]);
        }
    }

    for (auto & v : buf) v = enc_elu(v);
    {
        const float * wc = enc_get_f32(m.conv_out,   b_co);
        const float * bc = enc_get_f32(m.conv_out_b, b_cob);
        std::vector<float> co(ENC_DIM * T_cur);
        enc_causal_conv1d(buf.data(), 1024, T_cur, wc, ENC_DIM, 3, bc, co.data(), 1);
        buf = std::move(co);
    }
    // NOTE: downsample happens AFTER the transformer in HF's _encode_frame,
    // but SEANet returns here. The caller handles transformer → downsample order.

    *T_out = T_cur;
    return buf;
}

// Encoder transformer (GGML graph, 8 layers)
static std::vector<float> enc_run_transformer(enc_model & m, const std::vector<float> & ch_in, int T) {
    const size_t GMEM = 512ull * 1024 * 1024;
    struct ggml_init_params gp = { GMEM, nullptr, false };
    struct ggml_context * comp = ggml_init(gp);
    if (!comp) { fprintf(stderr, "enc_transformer: ggml_init OOM\n"); return {}; }

    ggml_tensor * inp = ggml_new_tensor_2d(comp, GGML_TYPE_F32, ENC_DIM, T);
    {
        float * dst = (float *)inp->data;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < ENC_DIM; c++)
                dst[c + t * ENC_DIM] = ch_in[c * T + t];
    }
    ggml_tensor * pos = ggml_new_tensor_1d(comp, GGML_TYPE_I32, T);
    for (int i = 0; i < T; i++) ((int32_t *)pos->data)[i] = i;

    ggml_tensor * cur = inp;
    const float scale = 1.f / sqrtf((float)ENC_HEAD_DIM);

    for (int il = 0; il < ENC_N_TR; il++) {
        const auto & layer = m.tr[il];

        ggml_tensor * res1 = cur;
        ggml_tensor * x = ggml_norm(comp, cur, ENC_NORM_EPS);
        x = ggml_mul(comp, x, layer.attn_norm);
        x = ggml_add(comp, x, layer.attn_norm_b);

        ggml_tensor * Q = ggml_mul_mat(comp, layer.attn_q, x);
        ggml_tensor * K = ggml_mul_mat(comp, layer.attn_k, x);
        ggml_tensor * V = ggml_mul_mat(comp, layer.attn_v, x);

        Q = ggml_reshape_3d(comp, Q, ENC_HEAD_DIM, ENC_N_HEADS, T);
        K = ggml_reshape_3d(comp, K, ENC_HEAD_DIM, ENC_N_HEADS, T);
        V = ggml_reshape_3d(comp, V, ENC_HEAD_DIM, ENC_N_HEADS, T);

        Q = ggml_rope_ext(comp, Q, pos, nullptr, ENC_HEAD_DIM, 0, 250,
                          ENC_ROPE_THETA, 1.f, 0.f, 1.f, 0.f, 0.f);
        K = ggml_rope_ext(comp, K, pos, nullptr, ENC_HEAD_DIM, 0, 250,
                          ENC_ROPE_THETA, 1.f, 0.f, 1.f, 0.f, 0.f);

        Q = ggml_cont(comp, ggml_permute(comp, Q, 0, 2, 1, 3));
        K = ggml_cont(comp, ggml_permute(comp, K, 0, 2, 1, 3));
        V = ggml_cont(comp, ggml_permute(comp, V, 1, 2, 0, 3));

        ggml_tensor * attn = ggml_mul_mat(comp, K, Q);
        attn = ggml_scale(comp, attn, scale);
        attn = ggml_diag_mask_inf(comp, attn, 0);
        attn = ggml_soft_max(comp, attn);
        ggml_tensor * ao = ggml_mul_mat(comp, V, attn);

        ao = ggml_reshape_2d(comp, ggml_cont(comp, ggml_permute(comp, ao, 0, 2, 1, 3)), ENC_DIM, T);
        ao = ggml_mul_mat(comp, layer.attn_output, ao);
        ao = ggml_mul(comp, ao, layer.attn_scale);
        cur = ggml_add(comp, res1, ao);

        ggml_tensor * res2 = cur;
        cur = ggml_norm(comp, cur, ENC_NORM_EPS);
        cur = ggml_mul(comp, cur, layer.ffn_norm);
        cur = ggml_add(comp, cur, layer.ffn_norm_b);

        cur = ggml_mul_mat(comp, layer.ffn_up, cur);
        cur = ggml_gelu_erf(comp, cur);
        cur = ggml_mul_mat(comp, layer.ffn_down, cur);
        cur = ggml_mul(comp, cur, layer.ffn_scale);
        cur = ggml_add(comp, res2, cur);
    }

    ggml_cgraph * gf = ggml_new_graph(comp);
    ggml_build_forward_expand(gf, cur);
    struct ggml_cplan plan = ggml_graph_plan(gf, 4, nullptr);
    std::vector<uint8_t> work(plan.work_size > 0 ? plan.work_size : 1);
    plan.work_data = work.data();
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "enc_transformer: compute failed\n");
        ggml_free(comp); return {};
    }

    std::vector<float> out(ENC_DIM * T);
    const float * src = (const float *)cur->data;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < ENC_DIM; c++)
            out[c * T + t] = src[c + t * ENC_DIM];

    {
        float rms = 0.f;
        for (int i = 0; i < ENC_DIM * T; i++) rms += out[i] * out[i];
        rms = sqrtf(rms / (ENC_DIM * T));
        printf("  [dbg] transformer out: rms=%.6f, frame0[:5]=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               rms, src[0], src[1], src[2], src[3], src[4]);
        if (T > 1) {
            printf("  [dbg] transformer out: frame1[:5]=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   src[ENC_DIM], src[ENC_DIM+1], src[ENC_DIM+2], src[ENC_DIM+3], src[ENC_DIM+4]);
        }
    }

    ggml_free(comp);
    return out;
}

// RVQ encode: 512-dim transformer output → discrete codes
static void enc_vq_encode(enc_model & m, const float * enc_out_tm, int T, int n_q, int32_t * codes) {
    std::vector<float> b_ip0, b_ipr;
    const float * ip0 = enc_get_f32(m.vq_semantic_input_proj, b_ip0);
    const float * ipr = m.vq_acoustic_input_proj ? enc_get_f32(m.vq_acoustic_input_proj, b_ipr) : ip0;

    std::vector<std::vector<float>> cb_norms(n_q, std::vector<float>(ENC_VQ_BINS));
    std::vector<std::vector<float>> cb_data(n_q);
    for (int q = 0; q < n_q; q++) {
        ggml_tensor * cb_t = (q == 0) ? m.vq_semantic_codebook[0] : m.vq_acoustic_codebook[q - 1];
        if (!cb_t) continue;
        std::vector<float> tmp;
        const float * cb = enc_get_f32(cb_t, tmp);
        cb_data[q].assign(cb, cb + ENC_VQ_BINS * ENC_VQ_DIM);
        for (int k = 0; k < ENC_VQ_BINS; k++) {
            const float * e = cb_data[q].data() + k * ENC_VQ_DIM;
            float n2 = 0.f;
            for (int d = 0; d < ENC_VQ_DIM; d++) n2 += e[d] * e[d];
            cb_norms[q][k] = n2;
        }
    }

    auto vq_nearest = [&](const float * z, int q) -> int32_t {
        const float * cb = cb_data[q].data();
        const float * norms = cb_norms[q].data();
        int best_k = 0;
        float best = FLT_MAX;
        for (int k = 0; k < ENC_VQ_BINS; k++) {
            const float * e = cb + k * ENC_VQ_DIM;
            float dot = 0.f;
            for (int d = 0; d < ENC_VQ_DIM; d++) dot += z[d] * e[d];
            float score = norms[k] - 2.f * dot;
            if (score < best) { best = score; best_k = k; }
        }
        return (int32_t)best_k;
    };

    for (int t = 0; t < T; t++) {
        const float * x_t = enc_out_tm + t * ENC_DIM;

        float z_sem[ENC_VQ_DIM] = {};
        for (int co = 0; co < ENC_VQ_DIM; co++) {
            float s = 0.f;
            for (int ci = 0; ci < ENC_DIM; ci++) s += x_t[ci] * ip0[ci + co * ENC_DIM];
            z_sem[co] = s;
        }
        codes[t * n_q + 0] = vq_nearest(z_sem, 0);
        if (n_q <= 1) continue;

        float z_ac[ENC_VQ_DIM] = {};
        for (int co = 0; co < ENC_VQ_DIM; co++) {
            float s = 0.f;
            for (int ci = 0; ci < ENC_DIM; ci++) s += x_t[ci] * ipr[ci + co * ENC_DIM];
            z_ac[co] = s;
        }
        float residual[ENC_VQ_DIM];
        memcpy(residual, z_ac, ENC_VQ_DIM * sizeof(float));
        for (int q = 1; q < n_q; q++) {
            int32_t bk = vq_nearest(residual, q);
            codes[t * n_q + q] = bk;
            const float * e = cb_data[q].data() + bk * ENC_VQ_DIM;
            for (int d = 0; d < ENC_VQ_DIM; d++) residual[d] -= e[d];
        }
    }
}

// Full encode pipeline: SEANet → Transformer → Downsample → VQ
// (HF order: encoder → encoder_transformer → downsample → quantizer)
static int enc_encode_audio(enc_model & m, const float * audio, int n_samples, int n_q, std::vector<std::vector<int32_t>> & out_frames) {
    int T_seanet = 0;
    printf("Speech tokenizer: encoding %d samples (%.2fs)...\n", n_samples, (float)n_samples / 24000.f);

    // Step 1: SEANet encoder (raw audio → 512-dim, channel-major)
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> seanet_out = enc_seanet_encode(m, audio, n_samples, &T_seanet);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (seanet_out.empty()) { fprintf(stderr, "enc_encode: SEANet failed\n"); return -1; }
    printf("  SEANet encoder: %d frames (%.1f ms)\n", T_seanet,
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Step 2: Encoder transformer (channel-major in/out)
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<float> tr_out = enc_run_transformer(m, seanet_out, T_seanet);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (tr_out.empty()) { fprintf(stderr, "enc_encode: transformer failed\n"); return -1; }
    printf("  Encoder transformer: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t3 - t2).count());

    // Step 3: Downsample (stride=2, k=4) AFTER transformer
    std::vector<float> b_ds;
    const float * wd = enc_get_f32(m.downsample, b_ds);
    const int T_frames = (T_seanet - 1) / 2 + 1;
    std::vector<float> ds_out(ENC_DIM * T_frames);
    enc_causal_conv1d(tr_out.data(), ENC_DIM, T_seanet, wd, ENC_DIM, 4, nullptr, ds_out.data(), 2);
    printf("  Downsample: %d → %d frames\n", T_seanet, T_frames);

    // Convert channel-major → time-major for VQ
    std::vector<float> vq_in(ENC_DIM * T_frames);
    for (int t = 0; t < T_frames; t++)
        for (int c = 0; c < ENC_DIM; c++)
            vq_in[c + t * ENC_DIM] = ds_out[c * T_frames + t];

    {
        FILE * dbg = fopen("tools/tts/_enc_dump/vq_input_cpp.bin", "wb");
        if (dbg) {
            int32_t hdr[2] = { T_frames, ENC_DIM };
            fwrite(hdr, sizeof(int32_t), 2, dbg);
            fwrite(vq_in.data(), sizeof(float), ENC_DIM * T_frames, dbg);
            fclose(dbg);
            printf("  Dumped VQ input (%dx%d) to tools/tts/_enc_dump/vq_input_cpp.bin\n", T_frames, ENC_DIM);
        }
    }

    // Step 4: RVQ quantize
    auto t4 = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> codes(T_frames * n_q);
    enc_vq_encode(m, vq_in.data(), T_frames, n_q, codes.data());
    auto t5 = std::chrono::high_resolution_clock::now();
    printf("  RVQ quantization (%d codebooks): %.1f ms\n", n_q,
           std::chrono::duration<double, std::milli>(t5 - t4).count());

    out_frames.resize(T_frames);
    for (int t = 0; t < T_frames; t++) {
        out_frames[t].resize(n_q);
        for (int q = 0; q < n_q; q++)
            out_frames[t][q] = codes[t * n_q + q];
    }
    printf("  Encoded: %d frames x %d codebooks\n", T_frames, n_q);
    return T_frames;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Vocoder Decoder (WavTokenizer-style)
//  VQ lookup → pre-conv → pre-transformer (8 layers, RoPE) →
//  upsample (ConvNeXt) → decoder blocks (Snake + ConvTranspose + residual) →
//  output conv → tanh → audio
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int VOC_N_CODEBOOKS  = 16;
static constexpr int VOC_CB_SIZE      = 2048;
static constexpr int VOC_CB_DIM       = 256;
static constexpr int VOC_HIDDEN       = 512;
static constexpr int VOC_LATENT       = 1024;
static constexpr int VOC_N_PRE_TFM    = 8;
static constexpr int VOC_N_HEADS      = 16;
static constexpr int VOC_HEAD_DIM     = 64;
static constexpr int VOC_DEC_DIM      = 1536;
static constexpr float VOC_RMS_EPS    = 1e-5f;
static constexpr float VOC_ROPE_THETA = 10000.0f;
#define VOC_MAX_NODES 32768

struct voc_pre_tfm_layer {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_q_w   = nullptr;
    ggml_tensor * attn_k_w   = nullptr;
    ggml_tensor * attn_v_w   = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_scale = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_gate_w = nullptr;
    ggml_tensor * ffn_up_w   = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
    ggml_tensor * ffn_scale  = nullptr;
};

struct voc_residual_block {
    int dilation = 1;
    ggml_tensor * act1_alpha = nullptr;
    ggml_tensor * act1_beta  = nullptr;
    ggml_tensor * conv1_w    = nullptr;
    ggml_tensor * conv1_b    = nullptr;
    ggml_tensor * act2_alpha = nullptr;
    ggml_tensor * act2_beta  = nullptr;
    ggml_tensor * conv2_w    = nullptr;
    ggml_tensor * conv2_b    = nullptr;
};

struct voc_decoder_block {
    ggml_tensor * snake_alpha = nullptr;
    ggml_tensor * snake_beta  = nullptr;
    ggml_tensor * conv_t_w    = nullptr;
    ggml_tensor * conv_t_b    = nullptr;
    voc_residual_block res[3];
};

struct voc_upsample_block {
    ggml_tensor * conv_w    = nullptr;
    ggml_tensor * conv_b    = nullptr;
    ggml_tensor * dwconv_w  = nullptr;
    ggml_tensor * dwconv_b  = nullptr;
    ggml_tensor * norm_w    = nullptr;
    ggml_tensor * norm_b    = nullptr;
    ggml_tensor * pwconv1_w = nullptr;
    ggml_tensor * pwconv1_b = nullptr;
    ggml_tensor * pwconv2_w = nullptr;
    ggml_tensor * pwconv2_b = nullptr;
    ggml_tensor * gamma     = nullptr;
};

struct voc_model {
    ggml_tensor * vq_first_output_proj = nullptr;
    ggml_tensor * vq_first_codebook    = nullptr;
    ggml_tensor * vq_rest_output_proj  = nullptr;
    ggml_tensor * vq_rest_codebook[15] = {};

    voc_upsample_block upsample[2];

    ggml_tensor * pre_tfm_input_proj_w  = nullptr;
    ggml_tensor * pre_tfm_input_proj_b  = nullptr;
    voc_pre_tfm_layer pre_tfm_layers[VOC_N_PRE_TFM];
    ggml_tensor * pre_tfm_norm_w        = nullptr;
    ggml_tensor * pre_tfm_output_proj_w = nullptr;
    ggml_tensor * pre_tfm_output_proj_b = nullptr;

    ggml_tensor * pre_conv_w = nullptr;
    ggml_tensor * pre_conv_b = nullptr;

    ggml_tensor * dec0_conv_w = nullptr;
    ggml_tensor * dec0_conv_b = nullptr;
    voc_decoder_block dec_blocks[4];
    ggml_tensor * dec5_snake_alpha = nullptr;
    ggml_tensor * dec5_snake_beta  = nullptr;
    ggml_tensor * dec6_conv_w      = nullptr;
    ggml_tensor * dec6_conv_b      = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

static ggml_tensor * voc_ensure_f16(ggml_context * ctx, ggml_tensor * w) {
    return (w->type != GGML_TYPE_F16) ? ggml_cast(ctx, w, GGML_TYPE_F16) : w;
}

static ggml_tensor * voc_snake(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * alpha, ggml_tensor * beta) {
    int64_t T = x->ne[0], C = x->ne[1], B = x->ne[2];

    ggml_tensor * a_exp = ggml_exp(ctx, alpha);
    ggml_tensor * a_3d  = ggml_reshape_3d(ctx, a_exp, 1, C, 1);
    ggml_tensor * ref   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, C, B);
    ggml_tensor * a_br  = ggml_repeat(ctx, a_3d, ref);

    ggml_tensor * ax     = ggml_mul(ctx, x, a_br);
    ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);

    ggml_tensor * neg_b   = ggml_scale(ctx, beta, -1.0f);
    ggml_tensor * inv_b_e = ggml_exp(ctx, neg_b);
    ggml_tensor * inv_b_3 = ggml_reshape_3d(ctx, inv_b_e, 1, C, 1);
    ggml_tensor * inv_br  = ggml_repeat(ctx, inv_b_3, ref);

    return ggml_add(ctx, x, ggml_mul(ctx, sin_sq, inv_br));
}

static ggml_tensor * voc_rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, VOC_RMS_EPS), w);
}

static ggml_tensor * voc_pre_tfm_layer_fwd(ggml_context * ctx, ggml_tensor * x,
                                             const voc_pre_tfm_layer & l,
                                             int n_frames, ggml_tensor * pos) {
    if (!l.attn_norm_w || !l.attn_q_w) return x;

    ggml_tensor * res = x;
    ggml_tensor * n = voc_rms_norm(ctx, x, l.attn_norm_w);

    ggml_tensor * Q = ggml_mul_mat(ctx, l.attn_q_w, n);
    ggml_tensor * K = ggml_mul_mat(ctx, l.attn_k_w, n);
    ggml_tensor * V = ggml_mul_mat(ctx, l.attn_v_w, n);

    Q = ggml_reshape_3d(ctx, Q, VOC_HEAD_DIM, VOC_N_HEADS, n_frames);
    K = ggml_reshape_3d(ctx, K, VOC_HEAD_DIM, VOC_N_HEADS, n_frames);
    V = ggml_reshape_3d(ctx, V, VOC_HEAD_DIM, VOC_N_HEADS, n_frames);

    Q = ggml_rope_ext(ctx, Q, pos, nullptr, VOC_HEAD_DIM, GGML_ROPE_TYPE_NEOX, 0,
                      VOC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr, VOC_HEAD_DIM, GGML_ROPE_TYPE_NEOX, 0,
                      VOC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)VOC_HEAD_DIM));
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);

    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    ggml_tensor * out = ggml_mul_mat(ctx, V, KQ);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx, out, VOC_N_HEADS * VOC_HEAD_DIM, n_frames);
    out = ggml_mul_mat(ctx, l.attn_out_w, out);
    if (l.attn_scale) out = ggml_mul(ctx, out, l.attn_scale);

    x = ggml_add(ctx, res, out);
    res = x;

    n = voc_rms_norm(ctx, x, l.ffn_norm_w);
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, l.ffn_gate_w, n));
    ggml_tensor * up   = ggml_mul_mat(ctx, l.ffn_up_w, n);
    ggml_tensor * ffn  = ggml_mul_mat(ctx, l.ffn_down_w, ggml_mul(ctx, gate, up));
    if (l.ffn_scale) ffn = ggml_mul(ctx, ffn, l.ffn_scale);

    return ggml_add(ctx, res, ffn);
}

static ggml_tensor * voc_residual_fwd(ggml_context * ctx, ggml_tensor * x,
                                       const voc_residual_block & blk) {
    ggml_tensor * res = x;
    if (blk.act1_alpha) x = voc_snake(ctx, x, blk.act1_alpha, blk.act1_beta);

    int64_t oc = blk.conv1_w->ne[2];
    int padding = 6 * blk.dilation;
    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, voc_ensure_f16(ctx, blk.conv1_w), x, 1, 0, blk.dilation);
    if (blk.conv1_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv1_b, 1, oc, 1));

    if (blk.act2_alpha) x = voc_snake(ctx, x, blk.act2_alpha, blk.act2_beta);

    oc = blk.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, voc_ensure_f16(ctx, blk.conv2_w), x, 1, 0, 1);
    if (blk.conv2_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv2_b, 1, oc, 1));

    return ggml_add(ctx, res, x);
}

static ggml_tensor * voc_decoder_block_fwd(ggml_context * ctx, ggml_tensor * x,
                                             const voc_decoder_block & blk, int upsample_rate) {
    if (blk.snake_alpha) x = voc_snake(ctx, x, blk.snake_alpha, blk.snake_beta);

    int64_t T_in = x->ne[0], ch_in = x->ne[1];
    int64_t ch_out = blk.conv_t_w->ne[1];
    int kernel = (int)blk.conv_t_w->ne[0];

    ggml_tensor * x2d = ggml_reshape_2d(ctx, x, T_in, ch_in);
    x2d = ggml_conv_transpose_1d(ctx, voc_ensure_f16(ctx, blk.conv_t_w), x2d, upsample_rate, 0, 1);

    int64_t T_new = x2d->ne[0];
    x = ggml_reshape_3d(ctx, x2d, T_new, ch_out, 1);

    // CausalTransConvNet: remove right_pad = kernel - stride from the RIGHT only
    int right_pad = kernel - upsample_rate;
    int64_t T_out = T_new - right_pad;
    x = ggml_view_3d(ctx, x, T_out, ch_out, 1, x->nb[1], x->nb[2], 0);
    x = ggml_cont(ctx, x);

    if (blk.conv_t_b)
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv_t_b, 1, ch_out, 1));

    for (int i = 0; i < 3; i++)
        x = voc_residual_fwd(ctx, x, blk.res[i]);

    return x;
}

static ggml_tensor * voc_upsample_fwd(ggml_context * ctx, ggml_tensor * x,
                                        const voc_upsample_block & blk) {
    int64_t T = x->ne[0], C = x->ne[1];

    ggml_tensor * x2d = ggml_reshape_2d(ctx, x, T, C);
    x2d = ggml_conv_transpose_1d(ctx, voc_ensure_f16(ctx, blk.conv_w), x2d, 2, 0, 1);
    int64_t T_new = x2d->ne[0];
    x = ggml_reshape_3d(ctx, x2d, T_new, C, 1);
    if (blk.conv_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.conv_b, 1, C, 1));

    ggml_tensor * res = x;

    if (blk.dwconv_w) {
        x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_conv_1d_dw(ctx, voc_ensure_f16(ctx, blk.dwconv_w), x, 1, 0, 1);
        if (blk.dwconv_b) x = ggml_add(ctx, x, ggml_reshape_3d(ctx, blk.dwconv_b, 1, C, 1));
    }

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    if (blk.norm_w && blk.norm_b) {
        x = ggml_norm(ctx, x, 1e-6f);
        x = ggml_mul(ctx, x, blk.norm_w);
        x = ggml_add(ctx, x, blk.norm_b);
    }
    x = ggml_mul_mat(ctx, blk.pwconv1_w, x);
    if (blk.pwconv1_b) x = ggml_add(ctx, x, blk.pwconv1_b);
    x = ggml_gelu(ctx, x);
    x = ggml_mul_mat(ctx, blk.pwconv2_w, x);
    if (blk.pwconv2_b) x = ggml_add(ctx, x, blk.pwconv2_b);
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

    if (blk.gamma) {
        ggml_tensor * g3 = ggml_reshape_3d(ctx, blk.gamma, 1, C, 1);
        ggml_tensor * ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_new, C, 1);
        x = ggml_mul(ctx, x, ggml_repeat(ctx, g3, ref));
    }

    return ggml_add(ctx, res, x);
}

static bool voc_load_weights(gguf_tensor_loader & loader, voc_model & voc) {
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = loader.get(name);
        if (!t) fprintf(stderr, "WARN: missing vocoder tensor: %s\n", name);
        return t;
    };

    voc.vq_first_codebook    = get("tok_dec.vq_first.0.codebook");
    voc.vq_first_output_proj = get("tok_dec.vq_first.output_proj.weight");
    voc.vq_rest_output_proj  = get("tok_dec.vq_rest.output_proj.weight");
    for (int i = 0; i < 15; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "tok_dec.vq_rest.%d.codebook", i);
        voc.vq_rest_codebook[i] = get(buf);
    }

    voc.pre_conv_w = get("tok_dec.pre_conv.weight");
    voc.pre_conv_b = get("tok_dec.pre_conv.bias");

    voc.pre_tfm_input_proj_w  = get("tok_dec.pre_tfm.input_proj.weight");
    voc.pre_tfm_input_proj_b  = get("tok_dec.pre_tfm.input_proj.bias");
    for (int i = 0; i < VOC_N_PRE_TFM; i++) {
        char buf[128];
        voc_pre_tfm_layer & layer = voc.pre_tfm_layers[i];
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_norm.weight", i);   layer.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_q.weight", i);      layer.attn_q_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_k.weight", i);      layer.attn_k_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_v.weight", i);      layer.attn_v_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_output.weight", i); layer.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.attn_scale", i);         layer.attn_scale = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_norm.weight", i);   layer.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_gate.weight", i);   layer.ffn_gate_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_up.weight", i);     layer.ffn_up_w   = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_down.weight", i);   layer.ffn_down_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.pre_tfm.blk.%d.ffn_scale", i);         layer.ffn_scale  = get(buf);
    }
    voc.pre_tfm_norm_w         = get("tok_dec.pre_tfm.norm.weight");
    voc.pre_tfm_output_proj_w  = get("tok_dec.pre_tfm.output_proj.weight");
    voc.pre_tfm_output_proj_b  = get("tok_dec.pre_tfm.output_proj.bias");

    for (int i = 0; i < 2; i++) {
        char buf[128];
        voc_upsample_block & blk = voc.upsample[i];
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.conv.weight", i);       blk.conv_w    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.conv.bias", i);         blk.conv_b    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.dwconv.weight", i);    blk.dwconv_w  = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.dwconv.bias", i);       blk.dwconv_b  = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.norm.weight", i);      blk.norm_w    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.norm.bias", i);        blk.norm_b    = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv1.weight", i);   blk.pwconv1_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv1.bias", i);     blk.pwconv1_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv2.weight", i);   blk.pwconv2_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.pwconv2.bias", i);     blk.pwconv2_b = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.upsample.%d.gamma", i);            blk.gamma     = get(buf);
    }

    voc.dec0_conv_w = get("tok_dec.dec.0.conv.weight");
    voc.dec0_conv_b = get("tok_dec.dec.0.conv.bias");

    const int dec_dilations[3] = {1, 3, 9};
    for (int db = 0; db < 4; db++) {
        int dec_idx = db + 1;
        voc_decoder_block & blk = voc.dec_blocks[db];
        char buf[128];
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.snake.alpha", dec_idx); blk.snake_alpha = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.snake.beta", dec_idx);  blk.snake_beta  = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.conv_t.weight", dec_idx); blk.conv_t_w = get(buf);
        snprintf(buf, sizeof(buf), "tok_dec.dec.%d.conv_t.bias", dec_idx);   blk.conv_t_b = get(buf);
        for (int r = 0; r < 3; r++) {
            int res_idx = r + 2;
            voc_residual_block & res = blk.res[r];
            res.dilation = dec_dilations[r];
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act1.alpha", dec_idx, res_idx); res.act1_alpha = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act1.beta", dec_idx, res_idx);  res.act1_beta  = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv1.weight", dec_idx, res_idx); res.conv1_w = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv1.bias", dec_idx, res_idx);   res.conv1_b = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act2.alpha", dec_idx, res_idx); res.act2_alpha = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.act2.beta", dec_idx, res_idx);  res.act2_beta  = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv2.weight", dec_idx, res_idx); res.conv2_w = get(buf);
            snprintf(buf, sizeof(buf), "tok_dec.dec.%d.res.%d.conv2.bias", dec_idx, res_idx);   res.conv2_b = get(buf);
        }
    }

    voc.dec5_snake_alpha = get("tok_dec.dec.5.snake.alpha");
    voc.dec5_snake_beta  = get("tok_dec.dec.5.snake.beta");
    voc.dec6_conv_w      = get("tok_dec.dec.6.conv.weight");
    voc.dec6_conv_b      = get("tok_dec.dec.6.conv.bias");

    bool ok = voc.vq_first_codebook && voc.pre_conv_w && voc.dec0_conv_w && voc.dec6_conv_w;
    if (!ok) fprintf(stderr, "ERROR: missing critical vocoder tensors\n");
    return ok;
}

static ggml_cgraph * voc_build_graph(ggml_context * ctx0, const voc_model & voc,
                                      const std::vector<std::vector<int32_t>> & all_codes) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VOC_MAX_NODES, false);
    const int n_frames = (int)all_codes.size();
    if (n_frames == 0) return gf;

    // 1. VQ codebook lookup
    std::vector<int32_t> cb0_ids(n_frames), cb_rest_ids[15];
    for (int c = 0; c < 15; c++) cb_rest_ids[c].resize(n_frames);
    for (int f = 0; f < n_frames; f++) {
        cb0_ids[f] = all_codes[f][0];
        for (int c = 1; c < 16; c++) cb_rest_ids[c - 1][f] = all_codes[f][c];
    }

    // ggml_get_rows(codebook[cb_dim, vocab], ids[n]) → [cb_dim, n_frames]
    // We need [n_frames, cb_dim, 1] for conv_1d (sequence along dim0)
    const int cb_dim = VOC_CB_DIM; // 256

    ggml_tensor * cb0_ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(cb0_ids_t, "cb0_ids"); ggml_set_input(cb0_ids_t);
    ggml_tensor * vq0 = ggml_get_rows(ctx0, voc.vq_first_codebook, cb0_ids_t);
    // vq0: [cb_dim, n_frames] → permute to [n_frames, cb_dim] → reshape to [n_frames, cb_dim, 1]
    vq0 = ggml_cont(ctx0, ggml_transpose(ctx0, vq0));
    vq0 = ggml_reshape_3d(ctx0, vq0, n_frames, cb_dim, 1);
    vq0 = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.vq_first_output_proj), vq0, 1, 0, 1);
    ggml_tensor * vq_sum = vq0;

    char buf_name[64];
    for (int c = 0; c < 15; c++) {
        ggml_tensor * ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        snprintf(buf_name, sizeof(buf_name), "cb_rest_ids_%d", c);
        ggml_set_name(ids_t, buf_name);
        ggml_set_input(ids_t);
        ggml_tensor * vqc = ggml_get_rows(ctx0, voc.vq_rest_codebook[c], ids_t);
        vqc = ggml_cont(ctx0, ggml_transpose(ctx0, vqc));
        vqc = ggml_reshape_3d(ctx0, vqc, n_frames, cb_dim, 1);
        vqc = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.vq_rest_output_proj), vqc, 1, 0, 1);
        vq_sum = ggml_add(ctx0, vq_sum, vqc);
    }

    // VQ output: [n_frames, 512, 1]
    ggml_tensor * cur = vq_sum;

    // 2. Pre-conv: CausalConvNet(512, 1024, kernel=3)
    //    causal padding = kernel_size - stride = 3 - 1 = 2 on the LEFT
    cur = ggml_pad_ext(ctx0, cur, 2, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.pre_conv_w), cur, 1, 0, 1);
    if (voc.pre_conv_b) cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, voc.pre_conv_b, 1, (int)cur->ne[1], 1));

    // 3. Pre-transformer: project 1024→512, 8 transformer layers, project 512→1024
    int64_t T_seq = cur->ne[0];

    // Reshape to [T_seq, 1024] for linear projections (swap to channels-last)
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, (int)cur->ne[0], (int)cur->ne[1]);
    cur = ggml_mul_mat(ctx0, voc.pre_tfm_input_proj_w, cur);
    if (voc.pre_tfm_input_proj_b) cur = ggml_add(ctx0, cur, voc.pre_tfm_input_proj_b);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_seq);
    ggml_set_name(pos, "voc_pos");
    ggml_set_input(pos);
    for (int i = 0; i < VOC_N_PRE_TFM; i++)
        cur = voc_pre_tfm_layer_fwd(ctx0, cur, voc.pre_tfm_layers[i], (int)T_seq, pos);

    cur = voc_rms_norm(ctx0, cur, voc.pre_tfm_norm_w);
    cur = ggml_mul_mat(ctx0, voc.pre_tfm_output_proj_w, cur);
    if (voc.pre_tfm_output_proj_b) cur = ggml_add(ctx0, cur, voc.pre_tfm_output_proj_b);

    // Reshape back to [T_seq, 1024, 1] (channels-first for conv)
    cur = ggml_reshape_3d(ctx0, cur, (int)cur->ne[0], (int)cur->ne[1], 1);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);

    // No residual from pre-conv (HF does NOT add a skip connection here)
    // 4. Upsample (2 ConvNeXt blocks, each ×2)
    cur = voc_upsample_fwd(ctx0, cur, voc.upsample[0]);
    cur = voc_upsample_fwd(ctx0, cur, voc.upsample[1]);

    // 5. Decoder: dec0 is CausalConvNet(latent_dim, decoder_dim, kernel=7)
    //    causal padding = kernel_size - stride = 7 - 1 = 6 on the LEFT
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.dec0_conv_w), cur, 1, 0, 1);
    if (voc.dec0_conv_b) cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, voc.dec0_conv_b, 1, (int)cur->ne[1], 1));

    const int upsample_rates[4] = {8, 5, 4, 3};
    for (int i = 0; i < 4; i++)
        cur = voc_decoder_block_fwd(ctx0, cur, voc.dec_blocks[i], upsample_rates[i]);

    // dec5 = SnakeBeta, dec6 = CausalConvNet(output_dim, 1, kernel=7)
    cur = voc_snake(ctx0, cur, voc.dec5_snake_alpha, voc.dec5_snake_beta);
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, voc_ensure_f16(ctx0, voc.dec6_conv_w), cur, 1, 0, 1);
    if (voc.dec6_conv_b) cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, voc.dec6_conv_b, 1, (int)cur->ne[1], 1));

    // HF uses clamp(-1, 1), not tanh
    cur = ggml_clamp(ctx0, cur, -1.0f, 1.0f);
    ggml_set_name(cur, "audio");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Sampling (temperature, top-k, top-p, repetition penalty)
// ═══════════════════════════════════════════════════════════════════════════════

struct tts_sampler_params {
    float temp          = 0.9f;
    int   top_k         = 50;
    float top_p         = 1.0f;
    float rep_penalty   = 1.05f;
    int   rep_last_n    = 64;
    bool  greedy        = false;
    std::vector<int32_t> suppress_tokens;
};

static int32_t tts_sample(
        const float * logits, int n_vocab,
        const tts_sampler_params & params,
        const std::vector<int32_t> & recent_tokens,
        std::mt19937 & rng) {

    std::vector<std::pair<float, int32_t>> candidates(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        candidates[i] = {logits[i], (int32_t)i};
    }

    for (int32_t tok : params.suppress_tokens) {
        if (tok >= 0 && tok < n_vocab) {
            candidates[tok].first = -std::numeric_limits<float>::infinity();
        }
    }

    if (params.greedy) {
        int32_t best = 0;
        float best_v = candidates[0].first;
        for (int i = 1; i < n_vocab; i++) {
            if (candidates[i].first > best_v) { best_v = candidates[i].first; best = i; }
        }
        return best;
    }

    // Repetition penalty
    if (params.rep_penalty != 1.0f && !recent_tokens.empty()) {
        int lookback = std::min((int)recent_tokens.size(), params.rep_last_n);
        for (int k = (int)recent_tokens.size() - lookback; k < (int)recent_tokens.size(); k++) {
            int32_t tok = recent_tokens[k];
            if (tok >= 0 && tok < n_vocab) {
                if (candidates[tok].first > 0.0f) {
                    candidates[tok].first /= params.rep_penalty;
                } else {
                    candidates[tok].first *= params.rep_penalty;
                }
            }
        }
    }

    // Temperature
    float temp = std::max(params.temp, 1e-8f);
    for (auto & c : candidates) c.first /= temp;

    // Top-k: keep only top_k highest
    if (params.top_k > 0 && params.top_k < n_vocab) {
        std::partial_sort(candidates.begin(), candidates.begin() + params.top_k, candidates.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        candidates.resize(params.top_k);
    } else {
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
    }

    // Softmax
    float max_logit = candidates[0].first;
    float sum_exp = 0.0f;
    for (auto & c : candidates) {
        c.first = expf(c.first - max_logit);
        sum_exp += c.first;
    }
    for (auto & c : candidates) c.first /= sum_exp;

    // Top-p (nucleus)
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum = 0.0f;
        int cutoff = (int)candidates.size();
        for (int i = 0; i < (int)candidates.size(); i++) {
            cum += candidates[i].first;
            if (cum >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
        sum_exp = 0.0f;
        for (auto & c : candidates) sum_exp += c.first;
        for (auto & c : candidates) c.first /= sum_exp;
    }

    // Weighted random sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (auto & c : candidates) {
        cum += c.first;
        if (r <= cum) return c.second;
    }
    return candidates.back().second;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

static const std::map<std::string, uint32_t> LANGUAGE_IDS = {
    {"english",    2050}, {"chinese",    2055}, {"german",  2053},
    {"spanish",    2054}, {"french",     2061}, {"italian", 2070},
    {"japanese",   2058}, {"korean",     2064}, {"portuguese", 2071},
    {"russian",    2069}, {"auto",       0},
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Qwen3-TTS: text-to-speech via llama.cpp\n\n"
        "Usage:\n"
        "  %s --model-talker <talker.gguf> \\\n"
        "     --model-cp <code-predictor.gguf> \\\n"
        "     --model-vocoder <tokenizer.gguf> \\\n"
        "     --text \"Hello world\" \\\n"
        "     --output <output.wav>\n\n"
        "Required:\n"
        "  --model-talker        Talker GGUF (contains speaker encoder + LLM)\n"
        "  --model-cp            Code Predictor GGUF\n"
        "  --text                Input text to synthesize\n\n"
        "Voice cloning:\n"
        "  --ref-audio <wav>     Reference audio for speaker cloning\n"
        "  --ref-text <text>     Reference transcript for ICL cloning (requires --ref-audio)\n"
        "  --ref-codes <file>    Precomputed codec codes file (optional for ICL)\n"
        "                        If omitted, codes are auto-encoded from ref-audio using built-in encoder\n\n"
        "Language:\n"
        "  --language <lang>     Target language (default: english)\n"
        "                        Supported: english, chinese, german, spanish, french,\n"
        "                        italian, japanese, korean, portuguese, russian, auto\n\n"
        "Sampling:\n"
        "  --temp <float>        Talker temperature (default: 0.9, 0 = greedy)\n"
        "  --top-k <int>         Talker top-k (default: 50, 0 = disabled)\n"
        "  --top-p <float>       Talker top-p / nucleus (default: 1.0)\n"
        "  --rep-penalty <float> Repetition penalty (default: 1.05, 1.0 = off)\n"
        "  --cp-temp <float>     Code Predictor temperature (default: 0.9)\n"
        "  --cp-top-k <int>      Code Predictor top-k (default: 50)\n"
        "  --greedy              Force greedy decoding (overrides temp/top-k)\n"
        "  --seed <int>          Random seed (default: random)\n\n"
        "Other options:\n"
        "  --model-vocoder       Tokenizer GGUF (vocoder decoder)\n"
        "  --output              Output WAV path (default: output.wav)\n"
        "  --max-tokens N        Max decode frames (default: 2048)\n"
        "  --streaming-text      Enable streaming text mode (feed text progressively)\n"
        "  --n-gpu-layers N      Number of GPU layers (default: 0)\n"
        "  --dump-intermediates  Directory to dump codec codes for parity testing\n\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Parse arguments
    std::string talker_path, cp_path, vocoder_path;
    std::string ref_audio_path, ref_text, ref_codes_path;
    std::string text, output_path = "output.wav";
    std::string dump_dir;
    std::string language = "english";
    int max_tokens = 2048;
    int n_gpu = 0;
    bool streaming_text = false;

    tts_sampler_params talker_sparams;
    tts_sampler_params cp_sparams;
    int seed = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-talker") == 0 && i + 1 < argc) talker_path = argv[++i];
        else if (strcmp(argv[i], "--model-cp") == 0 && i + 1 < argc) cp_path = argv[++i];
        else if (strcmp(argv[i], "--model-vocoder") == 0 && i + 1 < argc) vocoder_path = argv[++i];
        else if (strcmp(argv[i], "--ref-audio") == 0 && i + 1 < argc) ref_audio_path = argv[++i];
        else if (strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) ref_text = argv[++i];
        else if (strcmp(argv[i], "--ref-codes") == 0 && i + 1 < argc) ref_codes_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) talker_sparams.temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) talker_sparams.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) talker_sparams.top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) talker_sparams.rep_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cp-temp") == 0 && i + 1 < argc) cp_sparams.temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cp-top-k") == 0 && i + 1 < argc) cp_sparams.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0) { talker_sparams.greedy = true; cp_sparams.greedy = true; }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc) n_gpu = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-intermediates") == 0 && i + 1 < argc) dump_dir = argv[++i];
        else if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) language = argv[++i];
        else if (strcmp(argv[i], "--streaming-text") == 0) streaming_text = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    // --temp 0 implies greedy
    if (talker_sparams.temp <= 0.0f) talker_sparams.greedy = true;
    if (cp_sparams.temp <= 0.0f) cp_sparams.greedy = true;

    if (talker_path.empty() || cp_path.empty()) {
        fprintf(stderr, "ERROR: --model-talker and --model-cp are required\n");
        return 1;
    }
    if (text.empty()) {
        fprintf(stderr, "ERROR: --text is required\n");
        return 1;
    }

    // ── Load Talker model ──────────────────────────────────────────────────
    printf("Loading Talker model: %s\n", talker_path.c_str());
    llama_model_params talker_mparams = llama_model_default_params();
    talker_mparams.n_gpu_layers = n_gpu;
    llama_model * talker_model = llama_model_load_from_file(talker_path.c_str(), talker_mparams);
    if (!talker_model) { fprintf(stderr, "ERROR: failed to load talker model\n"); return 1; }

    // ── Load Code Predictor model ──────────────────────────────────────────
    printf("Loading Code Predictor model: %s\n", cp_path.c_str());
    llama_model_params cp_mparams = llama_model_default_params();
    cp_mparams.n_gpu_layers = n_gpu;
    llama_model * cp_model = llama_model_load_from_file(cp_path.c_str(), cp_mparams);
    if (!cp_model) { fprintf(stderr, "ERROR: failed to load code predictor model\n"); return 1; }

    // ── Create Talker context ──────────────────────────────────────────────
    llama_context_params talker_cparams = llama_context_default_params();
    talker_cparams.n_ctx      = 4096;
    talker_cparams.n_batch    = 512;
    talker_cparams.no_perf    = true;
    talker_cparams.embeddings = true;
    llama_context * talker_ctx = llama_init_from_model(talker_model, talker_cparams);
    if (!talker_ctx) { fprintf(stderr, "ERROR: failed to create talker context\n"); return 1; }

    // ── Create Code Predictor context ──────────────────────────────────────
    llama_context_params cp_cparams = llama_context_default_params();
    cp_cparams.n_ctx      = 32;
    cp_cparams.n_batch    = 32;
    cp_cparams.no_perf    = true;
    cp_cparams.embeddings = true;
    cp_cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp_cparams.type_k     = GGML_TYPE_F32;
    cp_cparams.type_v     = GGML_TYPE_F32;
    llama_context * cp_ctx = llama_init_from_model(cp_model, cp_cparams);
    if (!cp_ctx) { fprintf(stderr, "ERROR: failed to create code predictor context\n"); return 1; }

    printf("Models loaded successfully.\n");

    // ── Initialize RNG ────────────────────────────────────────────────
    if (seed < 0) {
        seed = (int)std::random_device{}();
    }
    std::mt19937 rng((uint32_t)seed);
    printf("Sampling: temp=%.2f top_k=%d top_p=%.2f rep_penalty=%.2f %s (seed=%d)\n",
           talker_sparams.temp, talker_sparams.top_k, talker_sparams.top_p,
           talker_sparams.rep_penalty,
           talker_sparams.greedy ? "[greedy]" : "",
           seed);
    if (!cp_sparams.greedy) {
        printf("CP sampling: temp=%.2f top_k=%d\n", cp_sparams.temp, cp_sparams.top_k);
    }

    // ── Declare GGUF tensor loaders before any goto cleanup ────────────
    gguf_tensor_loader tts_loader;
    gguf_tensor_loader cp_loader;
    gguf_tensor_loader voc_loader;

    // ── Load reference audio (shared by speaker encoder + speech tokenizer) ──
    std::vector<float> ref_audio_samples;
    std::vector<float> speaker_embedding(SPK_EMB_DIM, 0.0f);
    if (!ref_audio_path.empty()) {
        if (!read_wav(ref_audio_path.c_str(), ref_audio_samples)) {
            fprintf(stderr, "ERROR: cannot read reference audio\n");
            goto cleanup;
        }
    }

    // ── Speaker encoding ───────────────────────────────────────────────────
    if (!ref_audio_path.empty()) {
        printf("Extracting speaker embedding from: %s\n", ref_audio_path.c_str());

        gguf_tensor_loader spk_loader;
        if (spk_loader.load(talker_path.c_str(), "spk_enc.")) {
            spk_encoder_model spk_model;
            if (spk_load_weights(spk_loader, spk_model)) {
                int n_frames = 0;
                std::vector<float> mel_data;
                if (compute_mel_spectrogram(ref_audio_samples.data(), (int)ref_audio_samples.size(), mel_data, n_frames)) {
                    size_t ctx_size = ggml_tensor_overhead() * SPK_MAX_NODES + 256 * 1024 * 1024;
                    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
                    ggml_context * spk_ctx = ggml_init(ctx_params);
                    ggml_cgraph * spk_gf = spk_build_graph(spk_ctx, spk_model, n_frames);

                    ggml_tensor * mel_in = ggml_graph_get_tensor(spk_gf, "mel_input");
                    ggml_tensor * emb_out = ggml_graph_get_tensor(spk_gf, "embedding");

                    ggml_backend_t backend = ggml_backend_cpu_init();
                    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    ggml_gallocr_alloc_graph(alloc, spk_gf);

                    ggml_backend_tensor_set(mel_in, mel_data.data(), 0, mel_data.size() * sizeof(float));
                    ggml_backend_graph_compute(backend, spk_gf);
                    ggml_backend_tensor_get(emb_out, speaker_embedding.data(), 0, SPK_EMB_DIM * sizeof(float));

                    printf("  Speaker embedding extracted (first 3: %.4f %.4f %.4f)\n",
                           speaker_embedding[0], speaker_embedding[1], speaker_embedding[2]);

                    ggml_gallocr_free(alloc);
                    ggml_backend_free(backend);
                    ggml_free(spk_ctx);
                }
            }
        } else {
            printf("  No speaker encoder weights found, using zero embedding.\n");
        }
    } else {
        printf("No reference audio provided, using default speaker embedding.\n");
    }

    // ── Load text projection + codec embedding weights from GGUF ─────────
    if (!tts_loader.load(talker_path.c_str(), "tts.")) {
        fprintf(stderr, "ERROR: cannot load TTS tensors from Talker GGUF\n");
        goto cleanup;
    }

    {
        ggml_tensor * w_text_embd    = tts_loader.get("tts.text_embd.weight");
        ggml_tensor * w_proj_up      = tts_loader.get("tts.text_proj_up.weight");
        ggml_tensor * w_proj_up_b    = tts_loader.get("tts.text_proj_up.bias");
        ggml_tensor * w_proj_down    = tts_loader.get("tts.text_proj_down.weight");
        ggml_tensor * w_proj_down_b  = tts_loader.get("tts.text_proj_down.bias");
        ggml_tensor * w_codec_embd   = tts_loader.get("tts.codec_embd.weight");
        ggml_tensor * w_codec_head   = tts_loader.get("tts.codec_head.weight");

        if (!w_text_embd || !w_proj_up || !w_proj_down || !w_codec_embd || !w_codec_head) {
            fprintf(stderr, "ERROR: missing required TTS tensors\n");
            goto cleanup;
        }

        const int n_embd      = (int)llama_model_n_embd(talker_model);
        const int n_text_embd = (int)w_text_embd->ne[0]; // 2048

        printf("Text embd dim: %d, model embd: %d\n", n_text_embd, n_embd);
        // Codec embd: [n_embd, vocab], Codec head: [n_embd, vocab]

        // ── Read special IDs from GGUF metadata ─────────────────────────
        // These were stored in the Talker GGUF during conversion
        auto get_meta_u32 = [&](const char * key, uint32_t def) -> uint32_t {
            int idx = gguf_find_key(tts_loader.guf, key);
            if (idx < 0) return def;
            return (uint32_t)gguf_get_val_u32(tts_loader.guf, idx);
        };

        const uint32_t tts_bos_id     = get_meta_u32("qwen3tts.tts_bos_token_id",  151672);
        const uint32_t tts_eos_id     = get_meta_u32("qwen3tts.tts_eos_token_id",  151673);
        const uint32_t tts_pad_id     = get_meta_u32("qwen3tts.tts_pad_token_id",  151671);
        const uint32_t codec_bos_id   = get_meta_u32("qwen3tts.codec.bos_id",      2149);
        const uint32_t codec_eos_id   = get_meta_u32("qwen3tts.codec.eos_id",      2150);
        const uint32_t codec_pad_id   = get_meta_u32("qwen3tts.codec.pad_id",      2148);
        const uint32_t codec_nothink  = get_meta_u32("qwen3tts.codec.nothink_id",  2155);
        const uint32_t codec_think_bos = get_meta_u32("qwen3tts.codec.think_bos_id", 2156);
        const uint32_t codec_think_eos = get_meta_u32("qwen3tts.codec.think_eos_id", 2157);
        const uint32_t codec_think_id = get_meta_u32("qwen3tts.codec.think_id",    2154);
        printf("Special IDs: tts_bos=%u tts_eos=%u tts_pad=%u codec_bos=%u codec_eos=%u codec_pad=%u\n",
               tts_bos_id, tts_eos_id, tts_pad_id, codec_bos_id, codec_eos_id, codec_pad_id);

        // Resolve language → codec language ID
        std::string lang_lower = language;
        std::transform(lang_lower.begin(), lang_lower.end(), lang_lower.begin(), ::tolower);
        auto lang_it = LANGUAGE_IDS.find(lang_lower);
        if (lang_it == LANGUAGE_IDS.end()) {
            fprintf(stderr, "ERROR: unsupported language '%s'\nSupported:", language.c_str());
            for (auto & kv : LANGUAGE_IDS) fprintf(stderr, " %s", kv.first.c_str());
            fprintf(stderr, "\n");
            goto cleanup;
        }
        uint32_t codec_lang_id = lang_it->second;
        printf("Language: %s (codec_id=%u)\n", language.c_str(), codec_lang_id);

        // ── Tokenize input text ─────────────────────────────────────────
        // Wrap in chat template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        const llama_vocab * vocab = llama_model_get_vocab(talker_model);
        printf("Talker vocab size: %d\n", llama_vocab_n_tokens(vocab));

        std::string tts_prompt = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<llama_token> tokens(tts_prompt.size() + 32);
        int n_tokens = llama_tokenize(vocab, tts_prompt.c_str(), (int)tts_prompt.size(),
                                       tokens.data(), (int)tokens.size(),
                                       false, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, tts_prompt.c_str(), (int)tts_prompt.size(),
                                       tokens.data(), (int)tokens.size(),
                                       false, true);
        }
        tokens.resize(n_tokens);
        printf("Input text: \"%s\" (%d tokens)\n", text.c_str(), n_tokens);
        printf("Token IDs:");
        for (int i = 0; i < n_tokens; i++) printf(" %d", tokens[i]);
        printf("\n");

        // ── CPU helpers for text projection and codec embedding ────────
        // The GGUF tensor loader reads weights with no_alloc=false, so
        // tensor->data points to CPU memory directly.

        auto read_row = [](const ggml_tensor * t, int64_t row, float * out, int64_t cols) {
            if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * src = (const ggml_fp16_t *)t->data + row * cols;
                for (int64_t i = 0; i < cols; i++) out[i] = ggml_fp16_to_fp32(src[i]);
            } else if (t->type == GGML_TYPE_BF16) {
                const ggml_bf16_t * src = (const ggml_bf16_t *)t->data + row * cols;
                ggml_bf16_to_fp32_row(src, out, cols);
            } else {
                const float * src = (const float *)t->data + row * cols;
                memcpy(out, src, cols * sizeof(float));
            }
        };

        auto read_bias = [](const ggml_tensor * t, float * out, int64_t n) {
            if (!t) return;
            if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
                for (int64_t i = 0; i < n; i++) out[i] += ggml_fp16_to_fp32(src[i]);
            } else if (t->type == GGML_TYPE_BF16) {
                const ggml_bf16_t * src = (const ggml_bf16_t *)t->data;
                for (int64_t i = 0; i < n; i++) { float v; ggml_bf16_to_fp32_row(src + i, &v, 1); out[i] += v; }
            } else {
                const float * src = (const float *)t->data;
                for (int64_t i = 0; i < n; i++) out[i] += src[i];
            }
        };

        // text_projection(token_ids): text_embd → fc1 → silu → fc2
        // text_embd: [n_text_embd, n_text_vocab]  (GGML stores row-major, ne[0]=n_text_embd)
        // proj_up:   [n_text_embd, n_text_embd]
        // proj_down: [n_text_embd, n_embd]
        auto text_project = [&](const std::vector<int32_t> & ids) -> std::vector<float> {
            int n = (int)ids.size();
            std::vector<float> result(n * n_embd, 0.0f);
            std::vector<float> emb_row(n_text_embd);
            std::vector<float> fc1_out(n_text_embd);
            std::vector<float> fc2_out(n_embd);

            for (int t = 0; t < n; t++) {
                int32_t id = ids[t];
                read_row(w_text_embd, id, emb_row.data(), n_text_embd);

                // fc1 = emb_row @ proj_up^T + bias
                // proj_up is [n_text_embd, n_text_embd] in GGML: ne[0]=n_text_embd, ne[1]=n_text_embd
                // Row i of proj_up = output neuron i's weights
                for (int i = 0; i < n_text_embd; i++) {
                    float sum = 0.0f;
                    std::vector<float> w_row(n_text_embd);
                    read_row(w_proj_up, i, w_row.data(), n_text_embd);
                    for (int j = 0; j < n_text_embd; j++) sum += emb_row[j] * w_row[j];
                    fc1_out[i] = sum;
                }
                read_bias(w_proj_up_b, fc1_out.data(), n_text_embd);

                // silu activation
                for (int i = 0; i < n_text_embd; i++) {
                    float x = fc1_out[i];
                    fc1_out[i] = x / (1.0f + expf(-x));
                }

                // fc2 = fc1_out @ proj_down^T + bias
                // proj_down: [n_text_embd, n_embd] → ne[0]=n_text_embd, ne[1]=n_embd
                for (int i = 0; i < n_embd; i++) {
                    float sum = 0.0f;
                    std::vector<float> w_row(n_text_embd);
                    read_row(w_proj_down, i, w_row.data(), n_text_embd);
                    for (int j = 0; j < n_text_embd; j++) sum += fc1_out[j] * w_row[j];
                    fc2_out[i] = sum;
                }
                read_bias(w_proj_down_b, fc2_out.data(), n_embd);

                memcpy(&result[t * n_embd], fc2_out.data(), n_embd * sizeof(float));
            }
            return result;
        };

        // Read a row from codec_embd: [n_embd, n_codec_vocab]
        auto codec_embed = [&](int32_t id) -> std::vector<float> {
            std::vector<float> row(n_embd);
            read_row(w_codec_embd, id, row.data(), n_embd);
            return row;
        };

        // ── Compute text projections ────────────────────────────────────
        // Project special TTS tokens
        std::vector<int32_t> special_ids = {(int32_t)tts_bos_id, (int32_t)tts_eos_id, (int32_t)tts_pad_id};
        std::vector<float> special_proj = text_project(special_ids);
        // special_proj layout: [tts_bos(n_embd), tts_eos(n_embd), tts_pad(n_embd)]
        const float * tts_bos_embed = special_proj.data();
        const float * tts_eos_embed = special_proj.data() + n_embd;
        const float * tts_pad_embed = special_proj.data() + 2 * n_embd;

        // tts_bos and tts_pad embeddings loaded

        // Project all text tokens (including role tokens at the front)
        std::vector<int32_t> text_ids(tokens.begin(), tokens.end());
        std::vector<float> text_proj = text_project(text_ids);

        // text projection computed

        // ── Determine cloning mode ────────────────────────────────────
        bool has_speaker   = !ref_audio_path.empty();
        bool icl_mode      = has_speaker && !ref_text.empty() && (!ref_codes_path.empty() || !vocoder_path.empty());

        // ── Load or compute ICL reference codes ──────────────────────
        std::vector<std::vector<int32_t>> ref_code_frames;
        if (!ref_codes_path.empty()) {
            std::ifstream rc_file(ref_codes_path);
            if (!rc_file.is_open()) {
                fprintf(stderr, "ERROR: cannot open ref-codes file: %s\n", ref_codes_path.c_str());
                goto cleanup;
            }
            std::string line;
            while (std::getline(rc_file, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                std::vector<int32_t> frame;
                int32_t v;
                while (iss >> v) frame.push_back(v);
                if (!frame.empty()) ref_code_frames.push_back(frame);
            }
            printf("Loaded %d ref_code frames from %s\n", (int)ref_code_frames.size(), ref_codes_path.c_str());
            if (ref_text.empty()) {
                fprintf(stderr, "WARNING: --ref-codes provided without --ref-text, ICL disabled\n");
                icl_mode = false;
            }
        } else if (icl_mode && !vocoder_path.empty() && !ref_audio_samples.empty()) {
            printf("Auto-encoding reference audio for ICL mode...\n");
            gguf_tensor_loader enc_loader;
            if (enc_loader.load(vocoder_path.c_str(), "tok_enc.")) {
                enc_model enc;
                if (enc_load_weights(enc_loader, enc)) {
                    int n_active_q = 16;
                    int result = enc_encode_audio(enc, ref_audio_samples.data(), (int)ref_audio_samples.size(), n_active_q, ref_code_frames);
                    if (result <= 0) {
                        fprintf(stderr, "WARNING: speech tokenizer encoding failed, falling back to x-vector mode\n");
                        icl_mode = false;
                    } else if (!dump_dir.empty()) {
#ifdef _WIN32
                        _mkdir(dump_dir.c_str());
#else
                        mkdir(dump_dir.c_str(), 0755);
#endif
                        std::string codes_path = dump_dir + "/ref_codes_cpp.txt";
                        FILE * fp = fopen(codes_path.c_str(), "w");
                        if (fp) {
                            for (const auto & frame : ref_code_frames) {
                                for (size_t q = 0; q < frame.size(); q++) {
                                    if (q > 0) fprintf(fp, " ");
                                    fprintf(fp, "%d", frame[q]);
                                }
                                fprintf(fp, "\n");
                            }
                            fclose(fp);
                            printf("  Dumped ref codes to %s\n", codes_path.c_str());
                        }
                    }
                } else {
                    fprintf(stderr, "WARNING: encoder weights not found in vocoder GGUF, falling back to x-vector mode\n");
                    icl_mode = false;
                }
            } else {
                fprintf(stderr, "WARNING: cannot load encoder from vocoder GGUF, falling back to x-vector mode\n");
                icl_mode = false;
            }
        }

        if (icl_mode) {
            printf("Voice cloning mode: ICL (in-context learning)\n");
        } else if (has_speaker) {
            printf("Voice cloning mode: x-vector only\n");
        } else {
            printf("Voice cloning mode: none (default voice)\n");
        }

        // ── Get codec embeddings for prefill ────────────────────────────
        std::vector<uint32_t> codec_prefill_ids;
        if (codec_lang_id != 0) {
            codec_prefill_ids = {codec_think_id, codec_think_bos, codec_lang_id, codec_think_eos};
        } else {
            codec_prefill_ids = {codec_nothink, codec_think_bos, codec_think_eos};
        }

        std::vector<std::vector<float>> codec_prefill_embeds;
        for (uint32_t id : codec_prefill_ids)
            codec_prefill_embeds.push_back(codec_embed((int32_t)id));

        if (has_speaker) {
            codec_prefill_embeds.push_back(speaker_embedding);
        }

        codec_prefill_embeds.push_back(codec_embed((int32_t)codec_pad_id));
        codec_prefill_embeds.push_back(codec_embed((int32_t)codec_bos_id));

        int n_codec_pre = (int)codec_prefill_embeds.size();

        // ── Build prefill embedding ──────────────────────────────────
        // tokens layout: [<|im_start|>, assistant, \n, text..., <|im_end|>, \n, <|im_start|>, assistant, \n]
        // tokens[:3] = role prefix, tokens[3:-5] = text content, tokens[-5:] = suffix
        std::vector<float> codec_pad_emb = codec_embed((int32_t)codec_pad_id);

        const int n_role = 3;
        const int n_text_content = std::max(0, n_tokens - 3 - 5);
        const int n_codec_preamble = n_codec_pre - 1;

        int n_prefill;
        std::vector<float> prefill_embed;

        if (icl_mode) {
            // ICL mode: generate_icl_prompt
            // Tokenize ref_text with the same chat template wrapper
            std::string ref_prompt = "<|im_start|>assistant\n" + ref_text + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<llama_token> ref_tokens(ref_prompt.size() + 32);
            int n_ref_tokens = llama_tokenize(vocab, ref_prompt.c_str(), (int)ref_prompt.size(),
                                               ref_tokens.data(), (int)ref_tokens.size(),
                                               false, true);
            if (n_ref_tokens < 0) {
                ref_tokens.resize(-n_ref_tokens);
                n_ref_tokens = llama_tokenize(vocab, ref_prompt.c_str(), (int)ref_prompt.size(),
                                               ref_tokens.data(), (int)ref_tokens.size(),
                                               false, true);
            }
            ref_tokens.resize(n_ref_tokens);
            int n_ref_content = std::max(0, n_ref_tokens - 3 - 5);

            // Project reference text tokens
            std::vector<int32_t> ref_ids(ref_tokens.begin(), ref_tokens.end());
            std::vector<float> ref_proj = text_project(ref_ids);

            // ICL text = ref_text[3:-5] + target_text[3:-5]
            int n_icl_text = n_ref_content + n_text_content;
            int n_icl_text_section = n_icl_text + 1; // +1 for eos

            // Build codec embed for each ref frame:
            // sum of codec_embed[codebook_i](code) over all 16 codebooks
            int n_ref_frames = (int)ref_code_frames.size();
            std::vector<std::vector<float>> ref_codec_embeds(n_ref_frames, std::vector<float>(n_embd, 0.0f));

            // Load CP codec embeddings for codebooks 1-15
            if (cp_loader.tensors.empty()) {
                cp_loader.load(cp_path.c_str(), "tts.cp.");
            }

            for (int f = 0; f < n_ref_frames; f++) {
                auto & frame = ref_code_frames[f];
                // cb0: from talker codec_embed
                std::vector<float> cb0_emb = codec_embed(frame[0]);
                for (int j = 0; j < n_embd; j++) ref_codec_embeds[f][j] += cb0_emb[j];
                // cb1..15: from code predictor's per-codebook embeddings
                for (int cb = 1; cb < 16 && cb < (int)frame.size(); cb++) {
                    char tname[64];
                    snprintf(tname, sizeof(tname), "tts.cp.codec_embd.%d.weight", cb - 1);
                    ggml_tensor * cp_embd = cp_loader.get(tname);
                    if (cp_embd) {
                        std::vector<float> cb_emb(n_embd);
                        read_row(cp_embd, frame[cb], cb_emb.data(), n_embd);
                        for (int j = 0; j < n_embd; j++) ref_codec_embeds[f][j] += cb_emb[j];
                    }
                }
            }

            // ICL prefill layout (non-streaming):
            //   [0..2]: role = text_proj(tokens[:3])
            //   [3..3+n_codec_preamble-1]: tts_pad*(n-2) + tts_bos overlaid on codec_embed_preamble[:-1]
            //   ICL prompt (from generate_icl_prompt):
            //     ref_text + target_text + eos overlaid on codec_pad
            //     codec_bos + ref_codec_embeds overlaid on tts_pad
            //   final: tts_pad + codec_bos

            // generate_icl_prompt returns:
            //   text_embed = text_proj(ref_tokens[3:-5] + target_tokens[3:-5]) + eos_embed
            //   codec_embed_icl = codec_bos + sum_of_all_codebook_embeds for each ref frame
            //   In non-streaming mode:
            //     icl_input = text_embed + codec_pad overlay
            //     codec part = codec_embed_icl + tts_pad overlay
            //     concat: [text_with_pad, codec_with_pad]
            // Then: [role, preamble, icl_input, final_tts_pad+codec_bos]

            int n_icl_input = n_icl_text_section + n_ref_frames + 1; // text+eos + bos + ref frames
            n_prefill = n_role + n_codec_preamble + n_icl_input + 1; // +1 for final

            prefill_embed.resize(n_prefill * n_embd, 0.0f);
            int pos = 0;

            // (a) Role tokens
            for (int i = 0; i < n_role && i < n_tokens; i++) {
                memcpy(&prefill_embed[pos * n_embd], &text_proj[i * n_embd], n_embd * sizeof(float));
                pos++;
            }

            // (b) Codec preamble (same as x-vector mode)
            for (int i = 0; i < n_codec_pre - 2; i++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[i][j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_bos_embed[j] + codec_prefill_embeds[n_codec_pre - 2][j];
            }
            pos++;

            // (c) ICL text section: (ref_text + target_text + eos) overlaid on codec_pad
            for (int i = 0; i < n_ref_content; i++) {
                int tok_idx = 3 + i;
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = ref_proj[tok_idx * n_embd + j] + codec_pad_emb[j];
                }
                pos++;
            }
            for (int i = 0; i < n_text_content; i++) {
                int tok_idx = 3 + i;
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = text_proj[tok_idx * n_embd + j] + codec_pad_emb[j];
                }
                pos++;
            }
            // tts_eos + codec_pad
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_eos_embed[j] + codec_pad_emb[j];
            }
            pos++;

            // (d) ICL codec section: codec_bos + ref_codec_embeds, overlaid with tts_pad
            std::vector<float> codec_bos_emb = codec_embed((int32_t)codec_bos_id);
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_bos_emb[j];
            }
            pos++;
            for (int f = 0; f < n_ref_frames; f++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + ref_codec_embeds[f][j];
                }
                pos++;
            }

            // (e) Final: tts_pad + codec_bos
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[n_codec_pre - 1][j];
            }
            pos++;

            GGML_ASSERT(pos == n_prefill);

        } else if (streaming_text && !icl_mode) {
            // Streaming text mode: prefill = role + preamble + first_text_token+codec_bos
            // Remaining text fed one-by-one during decode via trailing_text
            n_prefill = n_role + n_codec_preamble + 1; // +1 for first_text+codec_bos

            prefill_embed.resize(n_prefill * n_embd, 0.0f);
            int pos = 0;

            // (a) Role tokens
            for (int i = 0; i < n_role && i < n_tokens; i++) {
                memcpy(&prefill_embed[pos * n_embd], &text_proj[i * n_embd], n_embd * sizeof(float));
                pos++;
            }

            // (b) Codec preamble
            for (int i = 0; i < n_codec_pre - 2; i++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[i][j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_bos_embed[j] + codec_prefill_embeds[n_codec_pre - 2][j];
            }
            pos++;

            // (c) First text token + codec_bos
            if (n_text_content > 0) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = text_proj[3 * n_embd + j]
                                                      + codec_prefill_embeds[n_codec_pre - 1][j];
                }
            } else {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j]
                                                      + codec_prefill_embeds[n_codec_pre - 1][j];
                }
            }
            pos++;

            GGML_ASSERT(pos == n_prefill);

        } else {
            // Non-streaming mode (default)
            const int n_text_section = n_text_content + 1;
            n_prefill = n_role + n_codec_preamble + n_text_section + 1;

            prefill_embed.resize(n_prefill * n_embd, 0.0f);
            int pos = 0;

            // (a) Role tokens
            for (int i = 0; i < n_role && i < n_tokens; i++) {
                memcpy(&prefill_embed[pos * n_embd], &text_proj[i * n_embd], n_embd * sizeof(float));
                pos++;
            }

            // (b) Codec preamble
            for (int i = 0; i < n_codec_pre - 2; i++) {
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[i][j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_bos_embed[j] + codec_prefill_embeds[n_codec_pre - 2][j];
            }
            pos++;

            // (c) All text content + eos, overlaid on codec_pad
            for (int i = 0; i < n_text_content; i++) {
                int tok_idx = 3 + i;
                for (int j = 0; j < n_embd; j++) {
                    prefill_embed[pos * n_embd + j] = text_proj[tok_idx * n_embd + j] + codec_pad_emb[j];
                }
                pos++;
            }
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_eos_embed[j] + codec_pad_emb[j];
            }
            pos++;

            // (d) Final: tts_pad + codec_bos
            for (int j = 0; j < n_embd; j++) {
                prefill_embed[pos * n_embd + j] = tts_pad_embed[j] + codec_prefill_embeds[n_codec_pre - 1][j];
            }
            pos++;

            GGML_ASSERT(pos == n_prefill);
        }

        // ── Trailing text for streaming mode ─────────────────────────
        std::vector<float> trailing_text;
        int n_trailing_text = 0;
        std::vector<float> pad_embed_vec(tts_pad_embed, tts_pad_embed + n_embd);

        if (streaming_text && !icl_mode && n_text_content > 1) {
            // tokens[4:-5] = remaining text content (skip first text token already in prefill)
            // + tts_eos at the end
            int n_remaining = n_text_content - 1;
            n_trailing_text = n_remaining + 1; // +1 for eos
            trailing_text.resize(n_trailing_text * n_embd);
            for (int i = 0; i < n_remaining; i++) {
                int tok_idx = 4 + i; // tokens[4..n_tokens-6]
                memcpy(&trailing_text[i * n_embd], &text_proj[tok_idx * n_embd], n_embd * sizeof(float));
            }
            memcpy(&trailing_text[n_remaining * n_embd], tts_eos_embed, n_embd * sizeof(float));
        } else {
            trailing_text.resize(n_embd);
            memcpy(trailing_text.data(), tts_pad_embed, n_embd * sizeof(float));
            n_trailing_text = 0;
        }

        printf("Prefill: %d positions (%d trailing text)%s\n",
               n_prefill, n_trailing_text,
               streaming_text ? " [streaming]" : "");

        // ── Send prefill embeddings to Talker ───────────────────────────
        // The Talker uses MRoPE (n_pos_per_embd=4). When sending embeddings
        // (not tokens), llama_batch expects explicit multi-dim positions.
        // pos layout: [dim0*n, dim1*n, dim2*n, dim3*n] where each dim has
        // n_tokens entries. For TTS, all 3 active dims use the same position
        // and dim3 is 0.

        const int n_pos_per_embd = 4; // iMRoPE
        llama_batch batch = llama_batch_init(512, n_embd, 1);

        // Reallocate pos for multi-dim positions
        free(batch.pos);
        batch.pos = (llama_pos *)malloc(sizeof(llama_pos) * 512 * n_pos_per_embd);
        memset(batch.pos, 0, sizeof(llama_pos) * 512 * n_pos_per_embd);

        batch.n_tokens = n_prefill;

        for (int i = 0; i < n_prefill; i++) {
            memcpy(batch.embd + i * n_embd, &prefill_embed[i * n_embd], n_embd * sizeof(float));
            // MRoPE: dims 0,1,2 get sequential position; dim 3 = 0
            for (int d = 0; d < 3; d++) {
                batch.pos[d * n_prefill + i] = i;
            }
            batch.pos[3 * n_prefill + i] = 0;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n_prefill - 1) ? 1 : 0;
        }

        printf("\nRunning Talker prefill (%d embeddings)...\n", n_prefill);
        auto t_prefill_start = std::chrono::high_resolution_clock::now();
        int ret = llama_decode(talker_ctx, batch);
        auto t_prefill_end = std::chrono::high_resolution_clock::now();
        if (ret != 0) {
            fprintf(stderr, "ERROR: Talker prefill decode failed: %d\n", ret);
            llama_batch_free(batch);
            goto cleanup;
        }

        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

        // Get hidden state from last position (the one with logits=1)
        float * hidden = llama_get_embeddings_ith(talker_ctx, -1);
        if (!hidden) {
            fprintf(stderr, "ERROR: no embeddings from Talker prefill\n");
            llama_batch_free(batch);
            goto cleanup;
        }

        printf("Talker prefill OK.  (%d tokens in %.1f ms = %.1f tok/s)\n",
               n_prefill, prefill_ms, n_prefill / (prefill_ms / 1000.0));

        // ── Apply codec_head externally to get cb0 logits ───────────────
        // logits = hidden @ codec_head^T
        // codec_head: [n_embd, n_codec_vocab] → row i = output token i's weights
        int n_codec_vocab = (int)w_codec_head->ne[1]; // 3072

        // HF suppresses tokens [vocab_size-1024, vocab_size) except EOS during Talker generation.
        // These are reserved special tokens that should never appear in codec output.
        {
            int suppress_start = n_codec_vocab - 1024;
            for (int i = suppress_start; i < n_codec_vocab; i++) {
                if (i != (int)codec_eos_id) {
                    talker_sparams.suppress_tokens.push_back((int32_t)i);
                }
            }
            printf("Talker: suppressing %d tokens [%d..%d) except EOS=%d\n",
                   (int)talker_sparams.suppress_tokens.size(), suppress_start, n_codec_vocab, (int)codec_eos_id);
        }

        auto compute_logits = [&](const float * h, std::vector<float> & out_logits) {
            out_logits.resize(n_codec_vocab);
            std::vector<float> w_row(n_embd);
            for (int i = 0; i < n_codec_vocab; i++) {
                read_row(w_codec_head, i, w_row.data(), n_embd);
                float sum = 0.0f;
                for (int j = 0; j < n_embd; j++) sum += h[j] * w_row[j];
                out_logits[i] = sum;
            }
        };

        std::vector<float> cb0_logits;
        compute_logits(hidden, cb0_logits);

        std::vector<int32_t> talker_recent_tokens;
        int cb0_token = tts_sample(cb0_logits.data(), n_codec_vocab,
                                    talker_sparams, talker_recent_tokens, rng);
        talker_recent_tokens.push_back(cb0_token);

        printf("Prefill cb0 token: %d (logit=%.4f)\n", cb0_token, cb0_logits[cb0_token]);

        // ── Load CP per-codebook weights ────────────────────────────────
        if (!cp_loader.load(cp_path.c_str(), "tts.cp.")) {
            fprintf(stderr, "ERROR: cannot load CP tensors from Code Predictor GGUF\n");
            llama_batch_free(batch);
            goto cleanup;
        }

        const int n_codebooks = 15; // cb1..cb15
        const int cp_vocab = 2048;
        ggml_tensor * cp_lm_heads[15]   = {};
        ggml_tensor * cp_codec_embds[15] = {};

        for (int i = 0; i < n_codebooks; i++) {
            char buf[128];
            snprintf(buf, sizeof(buf), "tts.cp.lm_head.%d.weight", i);
            cp_lm_heads[i] = cp_loader.get(buf);
            snprintf(buf, sizeof(buf), "tts.cp.codec_embd.%d.weight", i);
            cp_codec_embds[i] = cp_loader.get(buf);
            if (!cp_lm_heads[i] || !cp_codec_embds[i]) {
                fprintf(stderr, "ERROR: missing CP tensor: %s\n", buf);
                llama_batch_free(batch);
                goto cleanup;
            }
        }
        printf("Code Predictor per-codebook tensors loaded (%d lm_heads, %d codec_embeds)\n",
               n_codebooks, n_codebooks);

        // CP helpers: read row from per-codebook embedding, apply lm_head
        auto cp_codec_embed = [&](int cb_idx, int32_t token_id) -> std::vector<float> {
            std::vector<float> row(n_embd);
            read_row(cp_codec_embds[cb_idx], token_id, row.data(), n_embd);
            return row;
        };

        auto cp_compute_logits = [&](int cb_idx, const float * h, std::vector<float> & out) {
            out.resize(cp_vocab);
            std::vector<float> w_row(n_embd);
            for (int i = 0; i < cp_vocab; i++) {
                read_row(cp_lm_heads[cb_idx], i, w_row.data(), n_embd);
                float sum = 0.0f;
                for (int j = 0; j < n_embd; j++) sum += h[j] * w_row[j];
                out[i] = sum;
            }
        };

        // ── Autoregressive decode loop ──────────────────────────────────
        std::vector<std::vector<int32_t>> all_codes; // [n_frames, 16]

        int gen_step = 0;
        int talker_pos = n_prefill;

        // Pre-allocate reusable Talker step batch (avoids alloc/free per frame)
        llama_batch talker_step_batch = llama_batch_init(1, n_embd, 1);
        free(talker_step_batch.pos);
        talker_step_batch.pos = (llama_pos *)malloc(sizeof(llama_pos) * n_pos_per_embd);

        printf("\n[Starting autoregressive decode loop]\n");
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        double talker_decode_ms = 0, cp_decode_ms = 0, head_ms = 0;

        for (int frame = 0; frame < max_tokens; frame++) {
            if (cb0_token == (int)codec_eos_id) {
                printf("  EOS detected at frame %d\n", frame);
                break;
            }

            // ── Step 1: Code Predictor → generate cb1..cb15 ─────────────
            auto t_cp_start = std::chrono::high_resolution_clock::now();
            llama_free(cp_ctx);
            cp_ctx = llama_init_from_model(cp_model, cp_cparams);
            if (!cp_ctx) {
                fprintf(stderr, "ERROR: failed to recreate CP context at frame %d\n", frame);
                llama_batch_free(talker_step_batch);
                llama_batch_free(batch);
                goto cleanup;
            }

            std::vector<int32_t> frame_codes(16, 0);
            frame_codes[0] = cb0_token;

            std::vector<float> talker_h(hidden, hidden + n_embd);

            // Prefill CP with 2 embeddings: [talker_hidden, cb0_embd]
            std::vector<float> cb0_emb = codec_embed(cb0_token);

            llama_batch cp_prefill_batch = llama_batch_init(2, n_embd, 1);
            cp_prefill_batch.n_tokens = 2;
            memcpy(cp_prefill_batch.embd,           talker_h.data(), n_embd * sizeof(float));
            memcpy(cp_prefill_batch.embd + n_embd,  cb0_emb.data(),  n_embd * sizeof(float));
            cp_prefill_batch.pos[0] = 0; cp_prefill_batch.pos[1] = 1;
            cp_prefill_batch.n_seq_id[0] = 1; cp_prefill_batch.n_seq_id[1] = 1;
            cp_prefill_batch.seq_id[0][0] = 0; cp_prefill_batch.seq_id[1][0] = 0;
            cp_prefill_batch.logits[0] = 0; cp_prefill_batch.logits[1] = 1;

            ret = llama_decode(cp_ctx, cp_prefill_batch);
            llama_batch_free(cp_prefill_batch);

            if (ret != 0) {
                fprintf(stderr, "ERROR: CP prefill failed at frame %d: %d\n", frame, ret);
                break;
            }

            float * cp_hidden = llama_get_embeddings_ith(cp_ctx, -1);
            if (!cp_hidden || !std::isfinite(cp_hidden[0])) {
                fprintf(stderr, "ERROR: CP prefill produced invalid output at frame %d\n", frame);
                break;
            }
            // Step 1: apply lm_head[0] to get cb1
            std::vector<float> cp_logits_buf;
            std::vector<int32_t> cp_recent;
            cp_compute_logits(0, cp_hidden, cp_logits_buf);
            frame_codes[1] = tts_sample(cp_logits_buf.data(), cp_vocab,
                                         cp_sparams, cp_recent, rng);
            cp_recent.push_back(frame_codes[1]);

            // Steps 2-15: autoregressive decode for cb2..cb15
            int32_t prev_token = frame_codes[1];
            for (int cb_step = 1; cb_step < n_codebooks; cb_step++) {
                // Embed previous token using codec_embeds[cb_step - 1]
                std::vector<float> step_emb = cp_codec_embed(cb_step - 1, prev_token);

                // Single-token decode
                llama_batch step_batch = llama_batch_init(1, n_embd, 1);
                step_batch.n_tokens = 1;
                memcpy(step_batch.embd, step_emb.data(), n_embd * sizeof(float));
                step_batch.pos[0]      = 2 + (cb_step - 1); // positions 2,3,4,...,15
                step_batch.n_seq_id[0] = 1;
                step_batch.seq_id[0][0] = 0;
                step_batch.logits[0]   = 1;

                ret = llama_decode(cp_ctx, step_batch);
                llama_batch_free(step_batch);

                if (ret != 0) {
                    fprintf(stderr, "ERROR: CP decode step %d failed at frame %d: %d\n", cb_step, frame, ret);
                    break;
                }

                float * step_hidden = llama_get_embeddings_ith(cp_ctx, -1);
                if (!step_hidden || !std::isfinite(step_hidden[0])) {
                    fprintf(stderr, "ERROR: CP step %d produced invalid output at frame %d\n", cb_step, frame);
                    break;
                }

                // Apply lm_head[cb_step] to get the next codebook token
                cp_compute_logits(cb_step, step_hidden, cp_logits_buf);
                frame_codes[cb_step + 1] = tts_sample(cp_logits_buf.data(), cp_vocab,
                                                       cp_sparams, cp_recent, rng);
                cp_recent.push_back(frame_codes[cb_step + 1]);
                prev_token = frame_codes[cb_step + 1];
            }

            auto t_cp_end = std::chrono::high_resolution_clock::now();
            cp_decode_ms += std::chrono::duration<double, std::milli>(t_cp_end - t_cp_start).count();

            all_codes.push_back(frame_codes);

            // Build next Talker input
            std::vector<float> next_embd(n_embd, 0.0f);
            std::vector<float> emb0 = codec_embed(cb0_token);
            for (int j = 0; j < n_embd; j++) next_embd[j] += emb0[j];
            for (int cb = 0; cb < n_codebooks; cb++) {
                std::vector<float> emb_cb = cp_codec_embed(cb, frame_codes[cb + 1]);
                for (int j = 0; j < n_embd; j++) next_embd[j] += emb_cb[j];
            }
            if (gen_step < n_trailing_text) {
                for (int j = 0; j < n_embd; j++)
                    next_embd[j] += trailing_text[gen_step * n_embd + j];
            } else {
                for (int j = 0; j < n_embd; j++)
                    next_embd[j] += pad_embed_vec[j];
            }

            // Talker decode (reuse pre-allocated batch)
            auto t_talker_step_start = std::chrono::high_resolution_clock::now();
            talker_step_batch.n_tokens = 1;
            memcpy(talker_step_batch.embd, next_embd.data(), n_embd * sizeof(float));
            for (int d = 0; d < 3; d++) talker_step_batch.pos[d] = talker_pos;
            talker_step_batch.pos[3] = 0;
            talker_step_batch.n_seq_id[0] = 1;
            talker_step_batch.seq_id[0][0] = 0;
            talker_step_batch.logits[0]   = 1;

            ret = llama_decode(talker_ctx, talker_step_batch);
            if (ret != 0) {
                fprintf(stderr, "ERROR: Talker decode failed at frame %d: %d\n", frame, ret);
                break;
            }

            hidden = llama_get_embeddings_ith(talker_ctx, -1);
            if (!hidden) {
                fprintf(stderr, "ERROR: no embeddings at frame %d\n", frame);
                break;
            }

            auto t_head_start = std::chrono::high_resolution_clock::now();
            compute_logits(hidden, cb0_logits);
            cb0_token = tts_sample(cb0_logits.data(), n_codec_vocab,
                                    talker_sparams, talker_recent_tokens, rng);
            talker_recent_tokens.push_back(cb0_token);
            auto t_head_end = std::chrono::high_resolution_clock::now();
            head_ms += std::chrono::duration<double, std::milli>(t_head_end - t_head_start).count();

            auto t_talker_step_end = std::chrono::high_resolution_clock::now();
            talker_decode_ms += std::chrono::duration<double, std::milli>(t_talker_step_end - t_talker_step_start).count();

            talker_pos++;
            gen_step++;

            printf("  frame %d: cb0=%d", frame, frame_codes[0]);
            for (int cb = 1; cb < 16; cb++) printf(" %d", frame_codes[cb]);
            printf("\n");
            fflush(stdout);
        }

        llama_batch_free(talker_step_batch);

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double total_decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        int n_gen = (int)all_codes.size();

        printf("\nGeneration complete: %d frames\n", n_gen);
        printf("\n=== Performance ===\n");
        printf("  Prefill:  %d tokens in %.1f ms  (%.1f tok/s)\n",
               n_prefill, prefill_ms, n_prefill / (prefill_ms / 1000.0));
        printf("  Decode:   %d frames in %.1f ms  (%.2f frames/s)\n",
               n_gen, total_decode_ms, n_gen / (total_decode_ms / 1000.0));
        if (n_gen > 0) {
            printf("    Talker:    %.1f ms total  (%.1f ms/frame)\n",
                   talker_decode_ms, talker_decode_ms / n_gen);
            printf("    CP:        %.1f ms total  (%.1f ms/frame)\n",
                   cp_decode_ms, cp_decode_ms / n_gen);
            printf("    Head:      %.1f ms total  (%.1f ms/frame)\n",
                   head_ms, head_ms / n_gen);
            double audio_s = n_gen / 12.0;
            double wall_s = (prefill_ms + total_decode_ms) / 1000.0;
            printf("  Real-time factor: %.2fx  (%.2fs audio in %.2fs)\n",
                   audio_s / wall_s, audio_s, wall_s);
        }

        // ── Dump intermediates (for parity testing) ───────────────────────────
        if (!dump_dir.empty() && !all_codes.empty()) {
#ifdef _WIN32
            _mkdir(dump_dir.c_str());
#else
            mkdir(dump_dir.c_str(), 0755);
#endif
            std::string codes_path = dump_dir + "/all_codes.txt";
            FILE * fp = fopen(codes_path.c_str(), "w");
            if (fp) {
                fprintf(fp, "# frame cb0 cb1 cb2 ... cb15 (16 codebooks per frame)\n");
                for (size_t f = 0; f < all_codes.size(); f++) {
                    for (int c = 0; c < 16; c++) {
                        if (c > 0) fprintf(fp, " ");
                        fprintf(fp, "%d", all_codes[f][c]);
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
                printf("Dumped codec codes to %s\n", codes_path.c_str());
            }
        }

        // ── Vocoder: decode codebooks → audio ────────────────────────────────────
        if (!vocoder_path.empty() && !all_codes.empty()) {
            printf("Loading vocoder: %s\n", vocoder_path.c_str());
            if (voc_loader.load(vocoder_path.c_str(), "tok_dec.")) {
                voc_model voc;
                if (voc_load_weights(voc_loader, voc)) {
                    size_t ctx_size = ggml_tensor_overhead() * VOC_MAX_NODES + 128 * 1024 * 1024;
                    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
                    ggml_context * voc_ctx = ggml_init(ctx_params);
                    ggml_cgraph * voc_gf = voc_build_graph(voc_ctx, voc, all_codes);

                    int n_frames = (int)all_codes.size();

                    ggml_tensor * cb0_in = ggml_graph_get_tensor(voc_gf, "cb0_ids");
                    ggml_tensor * audio_out = ggml_graph_get_tensor(voc_gf, "audio");
                    ggml_tensor * pos_in = ggml_graph_get_tensor(voc_gf, "voc_pos");

                    char voc_buf_name[64];
                    std::vector<int32_t> cb0_ids(n_frames);
                    for (int f = 0; f < n_frames; f++) cb0_ids[f] = all_codes[f][0];

                    ggml_backend_t backend = ggml_backend_cpu_init();
                    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    ggml_gallocr_alloc_graph(alloc, voc_gf);

                    ggml_backend_tensor_set(cb0_in, cb0_ids.data(), 0, n_frames * sizeof(int32_t));
                    for (int c = 0; c < 15; c++) {
                        snprintf(voc_buf_name, sizeof(voc_buf_name), "cb_rest_ids_%d", c);
                        ggml_tensor * ids_in = ggml_graph_get_tensor(voc_gf, voc_buf_name);
                        if (ids_in) {
                            std::vector<int32_t> ids(n_frames);
                            for (int f = 0; f < n_frames; f++) ids[f] = all_codes[f][c + 1];
                            ggml_backend_tensor_set(ids_in, ids.data(), 0, n_frames * sizeof(int32_t));
                        } else {
                            fprintf(stderr, "WARNING: vocoder tensor %s not found in graph\n", voc_buf_name);
                        }
                    }
                    // Pre-transformer position tensor has n_frames entries
                    // (pre-transformer runs before upsampling)
                    int64_t pos_len = n_frames;
                    std::vector<int32_t> pos_data(pos_len);
                    for (int64_t i = 0; i < pos_len; i++) pos_data[i] = (int32_t)i;
                    ggml_backend_tensor_set(pos_in, pos_data.data(), 0, pos_len * sizeof(int32_t));

                    ggml_backend_graph_compute(backend, voc_gf);

                    int n_samples = (int)(audio_out->ne[0] * audio_out->ne[1] * audio_out->ne[2]);
                    std::vector<float> audio_data(n_samples);
                    ggml_backend_tensor_get(audio_out, audio_data.data(), 0, n_samples * sizeof(float));

                    write_wav(output_path.c_str(), audio_data.data(), n_samples, 24000);

                    ggml_gallocr_free(alloc);
                    ggml_backend_free(backend);
                    ggml_free(voc_ctx);
                }
            } else {
                fprintf(stderr, "WARN: failed to load vocoder, skipping audio generation\n");
            }
        } else if (vocoder_path.empty()) {
            printf("No vocoder specified, skipping audio generation.\n");
        }

        llama_batch_free(batch);
    }

cleanup:
    llama_free(cp_ctx);
    llama_free(talker_ctx);
    llama_model_free(cp_model);
    llama_model_free(talker_model);

    return 0;
}
