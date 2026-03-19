// tools/tts/speaker-encoder/speaker_encoder.cpp
// Standalone ECAPA-TDNN speaker encoder
//
// Extracts 1024-dimensional speaker embeddings (x-vectors) from audio files.
// Uses the same weights as the Qwen3-TTS speaker encoder, stored in the
// Talker GGUF file under the spk_enc.* prefix.
//
// Usage:
//   speaker-encoder --model talker.gguf --audio input.wav [--output embedding.bin]
//   speaker-encoder --model talker.gguf --audio a.wav b.wav c.wav --cosine
//
// Output modes:
//   (default)  Print embedding values to stdout
//   --output   Write raw float32 embedding to binary file
//   --cosine   Compute pairwise cosine similarity between all input files

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════════════════
//  Constants
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

// ═══════════════════════════════════════════════════════════════════════════════
//  WAV I/O
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
    return !out.empty();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Mel Spectrogram
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  GGUF Tensor Loader
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
//  ECAPA-TDNN Model
// ═══════════════════════════════════════════════════════════════════════════════

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

    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, SPK_N_MELS);
    ggml_set_name(mel, "mel_input"); ggml_set_input(mel);

    ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, SPK_N_MELS, 1);

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

    ggml_tensor * mfa_in = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
    mfa_in = ggml_concat(ctx0, mfa_in, block_outputs[3], 1);
    cur = spk_conv1d(ctx0, m.mfa_w, m.mfa_b, mfa_in, 1, 0, 1);
    cur = ggml_relu(ctx0, cur);

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
    return gf;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Embedding extraction
// ═══════════════════════════════════════════════════════════════════════════════

static bool extract_embedding(const spk_encoder_model & model,
                               const float * samples, int n_samples,
                               float * out_embedding) {
    int n_frames = 0;
    std::vector<float> mel_data;
    if (!compute_mel_spectrogram(samples, n_samples, mel_data, n_frames)) {
        fprintf(stderr, "ERROR: mel spectrogram computation failed\n");
        return false;
    }

    size_t ctx_size = ggml_tensor_overhead() * SPK_MAX_NODES + 256 * 1024 * 1024;
    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(ctx_params);
    ggml_cgraph * gf = spk_build_graph(ctx, model, n_frames);

    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel_input");
    ggml_tensor * emb_out = ggml_graph_get_tensor(gf, "embedding");

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(mel_in, mel_data.data(), 0, mel_data.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(emb_out, out_embedding, 0, SPK_EMB_DIM * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

static void print_usage(const char * prog) {
    printf("Usage: %s --model <talker.gguf> --audio <file.wav> [file2.wav ...] [options]\n", prog);
    printf("\nExtracts 1024-dim speaker embeddings (x-vectors) from audio files.\n");
    printf("\nOptions:\n");
    printf("  --model <path>    Path to Talker GGUF containing spk_enc.* tensors\n");
    printf("  --audio <paths>   One or more WAV files (24kHz, 16-bit PCM)\n");
    printf("  --output <path>   Write embedding to binary file (raw float32, 1024 values)\n");
    printf("  --cosine          Compute pairwise cosine similarity between all inputs\n");
    printf("  --quiet           Suppress per-dimension output, only print summary\n");
    printf("\nExamples:\n");
    printf("  %s --model talker.gguf --audio speaker.wav\n", prog);
    printf("  %s --model talker.gguf --audio a.wav b.wav c.wav --cosine\n", prog);
    printf("  %s --model talker.gguf --audio speaker.wav --output emb.bin\n", prog);
}

static float cosine_similarity(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string output_path;
    std::vector<std::string> audio_paths;
    bool do_cosine = false;
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--audio") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                audio_paths.push_back(argv[++i]);
            }
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--cosine") {
            do_cosine = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || audio_paths.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    gguf_tensor_loader loader;
    if (!loader.load(model_path.c_str(), "spk_enc.")) {
        fprintf(stderr, "ERROR: no speaker encoder weights found in %s\n", model_path.c_str());
        return 1;
    }

    spk_encoder_model model;
    if (!spk_load_weights(loader, model)) {
        fprintf(stderr, "ERROR: failed to load speaker encoder weights\n");
        return 1;
    }

    std::vector<std::vector<float>> all_embeddings;

    for (const auto & path : audio_paths) {
        std::vector<float> samples;
        if (!read_wav(path.c_str(), samples)) {
            fprintf(stderr, "ERROR: failed to read %s\n", path.c_str());
            return 1;
        }
        printf("Processing: %s  (%d samples, %.2fs)\n", path.c_str(),
               (int)samples.size(), samples.size() / 24000.0f);

        std::vector<float> embedding(SPK_EMB_DIM);
        if (!extract_embedding(model, samples.data(), (int)samples.size(), embedding.data())) {
            fprintf(stderr, "ERROR: embedding extraction failed for %s\n", path.c_str());
            return 1;
        }

        float rms = 0;
        for (int i = 0; i < SPK_EMB_DIM; i++) rms += embedding[i] * embedding[i];
        rms = sqrtf(rms / SPK_EMB_DIM);

        printf("  Embedding: dim=%d, rms=%.6f, first=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               SPK_EMB_DIM, rms,
               embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]);

        if (!quiet && audio_paths.size() == 1 && !do_cosine) {
            printf("\nFull embedding (%d dimensions):\n", SPK_EMB_DIM);
            for (int i = 0; i < SPK_EMB_DIM; i++) {
                printf("  [%4d] %12.6f\n", i, embedding[i]);
            }
        }

        all_embeddings.push_back(std::move(embedding));
    }

    if (!output_path.empty() && all_embeddings.size() == 1) {
        FILE * f = fopen(output_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "ERROR: cannot write %s\n", output_path.c_str());
            return 1;
        }
        fwrite(all_embeddings[0].data(), sizeof(float), SPK_EMB_DIM, f);
        fclose(f);
        printf("Wrote embedding to %s (%d floats, %zu bytes)\n",
               output_path.c_str(), SPK_EMB_DIM, SPK_EMB_DIM * sizeof(float));
    }

    if (do_cosine && all_embeddings.size() >= 2) {
        printf("\nPairwise cosine similarity:\n");
        for (size_t i = 0; i < all_embeddings.size(); i++) {
            for (size_t j = i + 1; j < all_embeddings.size(); j++) {
                float sim = cosine_similarity(all_embeddings[i].data(),
                                               all_embeddings[j].data(), SPK_EMB_DIM);
                printf("  %s vs %s: %.6f\n",
                       audio_paths[i].c_str(), audio_paths[j].c_str(), sim);
            }
        }
    }

    return 0;
}
