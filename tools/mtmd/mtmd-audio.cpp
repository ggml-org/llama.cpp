#include "mtmd-audio.h"

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>

// some of the code here is copied from whisper.cpp

constexpr bool DEBUG = false;

void mtmd_audio_cache::fill_sin_cos_table(uint32_t n) {
    sin_vals.resize(n);
    cos_vals.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        double theta = (2 * M_PI * i) / n;
        sin_vals[i]  = sinf(theta);
        cos_vals[i]  = cosf(theta);
    }
}

void mtmd_audio_cache::fill_hann_window(uint32_t length, bool periodic) {
    hann_window.resize(length);
    int offset = periodic ? 0 : -1;
    for (uint32_t i = 0; i < length; i++) {
        hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

void mtmd_audio_cache::fill_mel_filterbank_matrix(int   n_mel,
                                                  int   n_fft,
                                                  int   sample_rate,
                                                  float fmin,
                                                  float fmax,
                                                  bool  slaney_area_norm,
                                                  float scale,
                                                  bool  use_htk) {
    GGML_ASSERT(n_mel > 0 && n_fft > 1);
    if (fmax <= 0.0f) {
        fmax = 0.5f * sample_rate;
    }

    std::function<double(double)> hz_to_mel;
    std::function<double(double)> mel_to_hz;

    if (use_htk) {
        hz_to_mel = [](const double f_hz) -> double {
            return 2595.0 * log10(1.0 + f_hz / 700.0);
        };
        mel_to_hz = [](const double m) -> double {
            return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
        };
    } else {
        // Slaney scale (matches librosa default)
        const double min_log_hz  = 1000.0;
        const double lin_slope   = 3 / 200.;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step    = log(6.4) / 27.0;
        hz_to_mel = [min_log_hz, lin_slope, log_step, min_log_mel](const double f_hz) -> double {
            return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + log(f_hz / min_log_hz) / log_step;
        };
        mel_to_hz = [min_log_hz, lin_slope, log_step, min_log_mel](const double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
        };
    }

    // infer N_fft from n_fft_bins
    const double bin_hz_step = double(sample_rate) / double(n_fft);

    // mel grid: n_mel + 2 edges
    const double        m_lo = hz_to_mel(fmin);
    const double        m_hi = hz_to_mel(fmax);
    std::vector<double> mel_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        mel_pts[i] = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
    }

    // convert to Hz
    std::vector<double> hz_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        hz_pts[i] = mel_to_hz(mel_pts[i]);
    }

    const int n_fft_bins = n_fft / 2 + 1;

    // filterbank
    std::vector<float> out(n_mel * n_fft_bins, 0);
    for (int m = 0; m < n_mel; ++m) {
        const double f_left   = hz_pts[m];
        const double f_center = hz_pts[m + 1];
        const double f_right  = hz_pts[m + 2];

        const double denom_l = std::max(1e-30, f_center - f_left);
        const double denom_r = std::max(1e-30, f_right - f_center);
        const double enorm   = slaney_area_norm ? (2.0 / std::max(1e-30, f_right - f_left)) : 1.0;

        for (int k = 0; k < n_fft_bins; ++k) {
            const double f = k * bin_hz_step;
            double       w = 0.0;
            if (f >= f_left && f <= f_center) {
                w = (f - f_left) / denom_l;
            } else if (f > f_center && f <= f_right) {
                w = (f_right - f) / denom_r;
            }
            out[size_t(m) * size_t(n_fft_bins) + size_t(k)] = float(w * enorm * scale);
        }
    }

    filters.n_mel = n_mel;
    filters.n_fft = n_fft;
    filters.data  = std::move(out);

    if (DEBUG) {  // debug
        for (size_t i = 0; i < filters.data.size(); ++i) {
            if (filters.data[i] != 0.0f) {
                printf("filters[%zu] = %f\n", i, filters.data[i] * 1000.0f);
            }
        }
    }
}

// Unified DFT implementation for both forward and inverse transforms
// Template parameters:
//   Inverse: false = DFT with exp(-2πi·k·n/N), no scaling
//            true  = IDFT with exp(+2πi·k·n/N), scales by 1/N
//   RealInput: true = input is real-valued (stride 1), avoids imaginary computations
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void dft_impl(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    const int n_sin_cos_vals = cache.sin_vals.size();
    const int sin_cos_step   = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    const float     scale = Inverse ? (1.0f / N) : 1.0f;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int   idx     = (k * n * sin_cos_step) % n_sin_cos_vals;
            float cos_val = cache.cos_vals[idx];
            float sin_val = cache.sin_vals[idx];

            if constexpr (RealInput) {
                // Real input: in_im = 0, simplifies to:
                // re += in_re * cos_val
                // im += sign * in_re * sin_val
                float in_re = in[n];
                re += in_re * cos_val;
                im += sign * in_re * sin_val;
            } else {
                float in_re = in[n * 2 + 0];
                float in_im = in[n * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += in_re * cos_val - sign * in_im * sin_val;
                im += sign * in_re * sin_val + in_im * cos_val;
            }
        }

        out[k * 2 + 0] = re * scale;
        out[k * 2 + 1] = im * scale;
    }
}

// Cooley-Tukey FFT/IFFT unified implementation
// Template parameters:
//   Inverse: false = FFT with exp(-2πi·k/N), no scaling
//            true  = IFFT with exp(+2πi·k/N), scales by 0.5 at each level
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    if (N == 1) {
        out[0] = in[0];
        if constexpr (RealInput) {
            out[1] = 0.0f;
        } else {
            out[1] = in[1];
        }
        return;
    }

    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        // Odd N: fall back to DFT
        dft_impl<Inverse, RealInput>(cache, in, N, out);
        return;
    }

    // Split into even and odd
    if constexpr (RealInput) {
        // Real input: stride is 1, copy only real values
        float * even = in + N;
        for (int i = 0; i < half_N; ++i) {
            even[i] = in[2 * i];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, true>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i] = in[2 * i + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, true>(cache, odd, half_N, odd_fft);
    } else {
        // Complex input: stride is 2, copy complex pairs
        float * even = in + N * 2;
        for (int i = 0; i < half_N; ++i) {
            even[i * 2 + 0] = in[2 * i * 2 + 0];
            even[i * 2 + 1] = in[2 * i * 2 + 1];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, false>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0];
            odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, false>(cache, odd, half_N, odd_fft);
    }

    float * even_fft = out + 2 * N;
    float * odd_fft  = even_fft + N;

    const int sin_cos_step = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;

    for (int k = 0; k < half_N; k++) {
        int   idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re  = cache.cos_vals[idx];
        float im  = sign * cache.sin_vals[idx];

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
        out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);

        out[2 * (k + half_N) + 0] = scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
        out[2 * (k + half_N) + 1] = scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
    }
}

// Forward FFT for real input (used by mel spectrogram)
static void fft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<false, true>(cache, in, N, out);
}

// Inverse FFT for complex input
static void ifft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<true, false>(cache, in, N, out);
}

struct filter_params {
    int32_t n_mel;
    int32_t n_fft_bins;
    int32_t hann_window_size;
    int32_t hop_length;
    int32_t sample_rate;
    bool    no_padding      = false;
    bool    center_padding  = false;
    float   preemph         = 0.f;
    bool    use_natural_log = false;
    bool    norm_per_feature = false;
    bool    use_magnitude   = false;  // |X| instead of |X|^2
    float   mel_floor       = 5.960464477539063e-08f;
};

static void log_mel_spectrogram_worker_thread(int                        ith,
                                              const float *              hann,
                                              const std::vector<float> & samples,
                                              int                        n_samples,
                                              int                        frame_size,
                                              int                        frame_step,
                                              int                        n_threads,
                                              const filter_params &      params,
                                              const mtmd_audio_cache &   cache,
                                              mtmd_audio_mel &           out) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft_bins = params.n_fft_bins;
    int i = ith;

    const auto & filters = cache.filters;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    GGML_ASSERT(n_fft_bins == 1 + (frame_size / 2));
    GGML_ASSERT(cache.sin_vals.size() == cache.cos_vals.size());
    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, out.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(cache, fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 (power) or modulus (magnitude)
        for (int j = 0; j < n_fft_bins; j++) {
            float power = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
            fft_out[j] = params.use_magnitude ? sqrtf(power) : power;
        }

        // mel spectrogram
        for (int j = 0; j < out.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft_bins - 3; k += 4) {
                size_t idx = size_t(j) * size_t(n_fft_bins) + size_t(k);
                sum +=
                        fft_out[k + 0] * filters.data[idx + 0] +
                        fft_out[k + 1] * filters.data[idx + 1] +
                        fft_out[k + 2] * filters.data[idx + 2] +
                        fft_out[k + 3] * filters.data[idx + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft_bins; k++) {
                sum += fft_out[k] * filters.data[j * n_fft_bins + k];
            }
            sum = std::max(sum, (double)params.mel_floor);
            sum = params.use_natural_log
                ? log(sum)
                : log10(sum);
            out.data[j * out.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = params.use_natural_log ? log(1e-10) : log10(1e-10);
    for (; i < out.n_len; i += n_threads) {
        for (int j = 0; j < out.n_mel; j++) {
            out.data[j * out.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int     n_samples_in,
        const int     n_threads,
        const filter_params & params,
        const mtmd_audio_cache & cache,
        mtmd_audio_mel & out) {
    //const int64_t t_start_us = ggml_time_us();

    out.n_len_org = n_samples_in;
    int n_samples = n_samples_in;

    // Hann window
    const float * hann       = cache.hann_window.data();
    const int     frame_size = (params.n_fft_bins - 1) * 2;
    const int     frame_step = params.hop_length;

    // Padding
    std::vector<float> samples_padded;
    if (params.no_padding) {
        // no padding, use samples as-is
        samples_padded = std::vector<float>(samples, samples + n_samples);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else if (params.center_padding) {
        const auto pad_amount = frame_size / 2;
        samples_padded = std::vector<float>(n_samples + 2 * pad_amount, 0);
        std::copy(samples, samples + n_samples, samples_padded.data() + pad_amount);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else {
        // existing padding logic
        int64_t stage_1_pad = params.sample_rate * 30;
        int64_t stage_2_pad = frame_size / 2;
        samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
        std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);
        // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
        std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);
        // reflective pad 200 samples at the beginning of audio
        if (n_samples < stage_2_pad + 1) {
            // TODO: Handle short audio differently or return error
            return false;
        }
        std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

        // expose the padded buffer to downstream FFT and to out.n_len computation
        // mirrors the no_padding and center_padding branches above
        samples   = samples_padded.data();
        n_samples = samples_padded.size();
    }

    // preemphasis
    if (params.preemph) {
        const int   pad_amount = frame_size / 2;
        const float preemph = 0.97f;
        float       prev = samples_padded[pad_amount];
        for (int i = pad_amount + 1; i + pad_amount < n_samples; ++i) {
            float cur = samples_padded[i];
            samples_padded[i] = cur - preemph * prev;
            prev = cur;
        }
    }

    // pad hann window if it's smaller than frame_size
    // TODO: probably unnecessary here? (or better doing it in g_cache?)
    std::vector<float> hann_window_padded;
    if (params.hann_window_size < frame_size) {
        hann_window_padded.resize(frame_size);
        const int padding = (frame_size - params.hann_window_size) / 2;
        std::copy(hann, hann + params.hann_window_size, &hann_window_padded[padding]);
        hann = hann_window_padded.data();
    }


    GGML_ASSERT(params.n_fft_bins > 0);
    GGML_ASSERT(params.hop_length > 0);
    out.n_mel = params.n_mel;
    out.n_len = (n_samples - frame_size) / frame_step + 1;
    // TODO: handle these checks better
    if (out.n_mel > 0 && (unsigned long)out.n_len > SIZE_MAX / out.n_mel) {
        LOG_ERR("%s: size overflow\n", __func__);
        return false;
    }
    if (n_samples < frame_size) {
        LOG_ERR("%s: not enough samples after padding\n", __func__);
        return false;
    }
    out.data.resize(out.n_mel * out.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] =
                std::thread(log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded), n_samples,
                            frame_size, frame_step, n_threads, std::cref(params), std::cref(cache), std::ref(out));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples, frame_size, frame_step, n_threads, params,
                                          cache, out);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    const int effective_n_len = n_samples_in / frame_step;
    if (params.norm_per_feature) {
        GGML_ASSERT(effective_n_len > 1);
        for (int i = 0; i < out.n_mel; i++) {
            double mean = 0;
            for (int j = 0; j < effective_n_len; ++j) {
                mean += out.data[i * out.n_len + j];
            }
            mean /= effective_n_len;

            double var = 0.0;
            for (int j = 0; j < effective_n_len; ++j) {
                const double value = out.data[i * out.n_len + j] - mean;
                var += value * value;
            }
            var /= effective_n_len - 1;  // unbiased
            const double mstd = std::sqrt(var + 1e-5);

            for (int j = 0; j < effective_n_len; ++j) {
                auto &value = out.data[i * out.n_len + j];
                value        = (value - mean) / mstd;
            }

            // pad the rest with zeros
            for (int j = effective_n_len; j < out.n_len; ++j) {
                out.data[i * out.n_len + j] = 0.0;
            }
        }
    } else if (!params.no_padding) {
        // Whisper-style clamping and normalization (NOT used by Gemma4)
        double mmax = -1e20;
        for (int i = 0; i < out.n_mel*out.n_len; i++) {
            if (out.data[i] > mmax) {
                mmax = out.data[i];
            }
        }

        mmax -= 8.0;

        for (int i = 0; i < out.n_mel*out.n_len; i++) {
            if (out.data[i] < mmax) {
                out.data[i] = mmax;
            }
            out.data[i] = (out.data[i] + 4.0)/4.0;
        }
    }

    // Dump log_mel_spectrogram
    if (DEBUG) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < out.data.size() - 1; i++) {
            outFile << out.data[i] << ", ";
        }
        outFile << out.data[out.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

//
// mtmd_audio_preprocessor_whisper
//

void mtmd_audio_preprocessor_whisper::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);
    cache.fill_hann_window(hparams.audio_window_len, true);
    cache.fill_mel_filterbank_matrix(hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate);
}

bool mtmd_audio_preprocessor_whisper::preprocess(const float *                 samples,
                                                 size_t                        n_samples,
                                                 std::vector<mtmd_audio_mel> & output) {
    if (n_samples == 0) {
        // empty audio
        return false;
    }

    std::vector<float> smpl;
    // if input is too short, pad with zeros
    // this is to avoid potential issues with stage1/2 padding in log_mel_spectrogram
    // TODO: maybe handle this better
    size_t min_samples = (size_t) hparams.audio_sample_rate * (hparams.audio_chunk_len + 1);  // +1 second margin
    if (n_samples < min_samples) {
        smpl.resize(min_samples, 0.0f);
        std::memcpy(smpl.data(), samples, n_samples * sizeof(float));
        samples   = smpl.data();
        n_samples = smpl.size();
    }

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_window_len;
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.center_padding   = false;
    params.preemph          = 0.0f;  // disabled
    params.use_natural_log  = false;
    params.norm_per_feature = false;

    // make sure the cache is initialized
    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    mtmd_audio_mel out_full;
    bool           ok = log_mel_spectrogram(samples, n_samples,
                                            4,  // n_threads
                                            params, cache, out_full);
    if (!ok) {
        return false;
    }

    // because the cgraph in clip.cpp only accepts 3000 frames each, we need to split the mel
    // we always expect the mel to have 3000 silent frames at the end
    if (DEBUG) {
        printf("output: n_mel = %d, n_len = %d\n", out_full.n_mel, out_full.n_len);
    }
    const size_t frames_per_chunk = 3000;
    GGML_ASSERT((size_t) out_full.n_len > frames_per_chunk);
    for (size_t off = 0; off < (size_t) out_full.n_len; off += frames_per_chunk) {
        int n_len = std::min(frames_per_chunk, (size_t) out_full.n_len - off);
        if ((size_t) n_len < frames_per_chunk) {
            break;  // last incomplete chunk will always be a padded chunk, safe to ignore
        }

        mtmd_audio_mel out_chunk;
        out_chunk.n_len     = n_len;
        out_chunk.n_mel     = out_full.n_mel;
        out_chunk.n_len_org = out_full.n_mel;  // unused
        out_chunk.data.reserve(out_chunk.n_mel * out_chunk.n_len);

        for (int i = 0; i < out_full.n_mel; i++) {
            auto src = out_full.data.begin() + i * out_full.n_len + off;
            out_chunk.data.insert(out_chunk.data.end(), src, src + frames_per_chunk);
        }

        output.push_back(std::move(out_chunk));
    }

    return true;
}

//
// mtmd_audio_preprocessor_conformer
//

void mtmd_audio_preprocessor_conformer::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);
    cache.fill_hann_window(hparams.audio_window_len, true);
    cache.fill_mel_filterbank_matrix(hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate);
}

bool mtmd_audio_preprocessor_conformer::preprocess(const float *                 samples,
                                                   size_t                        n_samples,
                                                   std::vector<mtmd_audio_mel> & output) {
    // empty audio
    if (n_samples == 0) {
        return false;
    }

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_window_len;
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.center_padding   = true;
    params.preemph          = 0.97f;
    params.use_natural_log  = true;
    params.norm_per_feature = true;

    // make sure the cache is initialized
    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    mtmd_audio_mel out_full;
    bool           ok = log_mel_spectrogram(samples, n_samples,
                                            4,  // n_threads
                                            params, cache, out_full);
    if (!ok) {
        return false;
    }

    output.push_back(std::move(out_full));
    return true;
}

//
// mtmd_audio_preprocessor_granite_speech
//

void mtmd_audio_preprocessor_granite_speech::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);
    cache.fill_hann_window(hparams.audio_window_len, true);
    cache.fill_mel_filterbank_matrix(
        hparams.n_mel_bins / 2, hparams.audio_n_fft, hparams.audio_sample_rate,
        0.0f, -1.0f, false, 1.0f, true);
}

bool mtmd_audio_preprocessor_granite_speech::preprocess(const float *                 samples,
                                                        size_t                        n_samples,
                                                        std::vector<mtmd_audio_mel> & output) {
    if (n_samples == 0) {
        return false;
    }

    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    const int n_fft = hparams.audio_n_fft;
    const int pad   = n_fft / 2;

    // reflect padding
    const int n_padded = (int)n_samples + 2 * pad;
    std::vector<float> padded(n_padded, 0.0f);
    std::copy(samples, samples + n_samples, padded.data() + pad);
    for (int i = 0; i < pad; i++) {
        int src = i + 1;
        if (src >= (int)n_samples) {
            src = (int)n_samples - 1;
        }
        padded[pad - 1 - i] = samples[src];
    }
    for (int i = 0; i < pad; i++) {
        int src = (int)n_samples - 2 - i;
        if (src < 0) {
            src = 0;
        }
        padded[pad + (int)n_samples + i] = samples[src];
    }

    filter_params params;
    params.n_mel            = hparams.n_mel_bins / 2;
    params.n_fft_bins       = 1 + (n_fft / 2);
    params.hann_window_size = hparams.audio_window_len;
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.no_padding       = true;
    params.center_padding   = false;
    params.preemph          = 0.0f;
    params.use_natural_log  = false;
    params.norm_per_feature = false;
    params.mel_floor        = 1e-10f;

    mtmd_audio_mel mel;
    if (!log_mel_spectrogram(padded.data(), n_padded, 4, params, cache, mel)) {
        return false;
    }

    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }
    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }
        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    int n_frames = mel.n_len;
    if (n_frames % 2 == 1) {
        n_frames--;
    }
    const int n_mel     = mel.n_mel;
    const int n_stacked = n_frames / 2;

    mtmd_audio_mel stacked;
    stacked.n_mel     = 2 * n_mel;
    stacked.n_len     = n_stacked;
    stacked.n_len_org = (int)n_samples;
    stacked.data.resize(2 * n_mel * n_stacked);

    for (int t = 0; t < n_stacked; t++) {
        for (int m = 0; m < n_mel; m++) {
            stacked.data[m * n_stacked + t] = mel.data[m * mel.n_len + 2 * t];
            stacked.data[(m + n_mel) * n_stacked + t] = mel.data[m * mel.n_len + 2 * t + 1];
        }
    }

    output.push_back(std::move(stacked));
    return true;
}

//
// mtmd_audio_preprocessor_gemma4a
//

void mtmd_audio_preprocessor_gemma4a::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);

    // Standard periodic Hann window, zero-padded to FFT size
    cache.hann_window.assign(hparams.audio_n_fft, 0.0f);
    for (uint32_t i = 0; i < (uint32_t)hparams.audio_window_len; i++) {
        cache.hann_window[i] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * i) / hparams.audio_window_len);
    }

    // HTK mel scale, no Slaney area normalization
    cache.fill_mel_filterbank_matrix(
        hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate,
        0.0f, hparams.audio_sample_rate / 2.0f,
        /*slaney_area_norm=*/ false,
        /*scale=*/ 1.0f,
        /*use_htk=*/ true
    );
}

bool mtmd_audio_preprocessor_gemma4a::preprocess(const float *                 samples,
                                                  size_t                        n_samples,
                                                  std::vector<mtmd_audio_mel> & output) {
    if (n_samples == 0) {
        return false;
    }

    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_n_fft; // window is zero-padded to FFT size
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.no_padding       = true;
    params.center_padding   = false;
    params.preemph          = 0.0f;
    params.use_natural_log  = true;
    params.use_magnitude    = true;
    params.mel_floor        = 0.001f;
    params.norm_per_feature = false;

    // Split into 30-second chunks (model context limit, ~750 tokens each)
    const size_t chunk_samples = 30 * hparams.audio_sample_rate;
    for (size_t off = 0; off < n_samples; off += chunk_samples) {
        const float * chunk_ptr = samples + off;
        size_t chunk_len = std::min(chunk_samples, n_samples - off);

        // Semicausal left-padding + right-padding to match PyTorch frame count
        const int pad_left = hparams.audio_window_len / 2;
        const int fft_size = hparams.audio_n_fft;
        const int hop = hparams.audio_hop_len;
        const int n_with_left = (int)chunk_len + pad_left;
        // PyTorch: unfold(size=frame_length+1, step=hop) on semicausal-padded waveform
        const int pt_frames = (n_with_left - (hparams.audio_window_len + 1)) / hop + 1;
        const int n_padded_needed = (pt_frames - 1) * hop + fft_size;
        const int total_pad = std::max((int)(n_padded_needed - (int)chunk_len), pad_left);
        std::vector<float> padded_samples(total_pad + chunk_len, 0.0f);
        std::copy(chunk_ptr, chunk_ptr + chunk_len, padded_samples.data() + pad_left);

        mtmd_audio_mel out_chunk;
        bool ok = log_mel_spectrogram(padded_samples.data(), padded_samples.size(), 4, params, cache, out_chunk);
        if (!ok) {
            return false;
        }

        // Trim to PyTorch frame count
        out_chunk.n_len = std::min(out_chunk.n_len, pt_frames);

        output.push_back(std::move(out_chunk));
    }

    return true;
}

//
// mtmd_audio_streaming_istft implementation
//

mtmd_audio_streaming_istft::mtmd_audio_streaming_istft(int n_fft, int hop_length) :
    n_fft(n_fft),
    hop_length(hop_length),
    n_fft_bins(n_fft / 2 + 1),
    overlap_buffer(n_fft, 0.0f),
    window_sum_buffer(n_fft, 0.0f),
    padding_to_remove((n_fft - hop_length) / 2),
    ifft_in(n_fft * 2 * 4, 0.0f),  // extra space for recursive IFFT
    ifft_out(n_fft * 2 * 4, 0.0f) {
    GGML_ASSERT(n_fft > 0 && hop_length > 0 && hop_length <= n_fft);
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
}

void mtmd_audio_streaming_istft::reset() {
    std::fill(overlap_buffer.begin(), overlap_buffer.end(), 0.0f);
    std::fill(window_sum_buffer.begin(), window_sum_buffer.end(), 0.0f);
    padding_to_remove = (n_fft - hop_length) / 2;
}

std::vector<float> mtmd_audio_streaming_istft::process_frame(const float * frame_spectrum) {
    std::vector<float> output(hop_length);

    // copy frequencies
    for (int j = 0; j < n_fft_bins; j++) {
        ifft_in[j * 2 + 0] = frame_spectrum[j * 2 + 0];
        ifft_in[j * 2 + 1] = frame_spectrum[j * 2 + 1];
    }

    // mirror negative frequencies
    for (int j = 1; j < n_fft_bins - 1; j++) {
        int mirror_idx              = n_fft - j;
        ifft_in[mirror_idx * 2 + 0] = ifft_in[j * 2 + 0];
        ifft_in[mirror_idx * 2 + 1] = -ifft_in[j * 2 + 1];  // conjugate
    }

    ifft(cache, ifft_in.data(), n_fft, ifft_out.data());

    // update window sum and overlap buffer
    for (int j = 0; j < n_fft; j++) {
        window_sum_buffer[j] += cache.hann_window[j] * cache.hann_window[j];
        overlap_buffer[j] += ifft_out[j * 2] * cache.hann_window[j];
    }

    // extract hop_length samples with normalization
    for (int i = 0; i < hop_length; i++) {
        if (window_sum_buffer[i] > 1e-8f) {
            output[i] = overlap_buffer[i] / window_sum_buffer[i];
        } else {
            output[i] = overlap_buffer[i];
        }
    }

    // shift buffers left by hop_length
    std::copy(overlap_buffer.begin() + hop_length, overlap_buffer.end(), overlap_buffer.begin());
    std::fill(overlap_buffer.end() - hop_length, overlap_buffer.end(), 0.0f);

    std::copy(window_sum_buffer.begin() + hop_length, window_sum_buffer.end(), window_sum_buffer.begin());
    std::fill(window_sum_buffer.end() - hop_length, window_sum_buffer.end(), 0.0f);

    // Remove padding if needed
    int to_remove = std::min(padding_to_remove, (int) output.size());
    padding_to_remove -= to_remove;
    output.erase(output.begin(), output.begin() + to_remove);

    return output;
}

std::vector<float> mtmd_audio_streaming_istft::flush() {
    std::vector<float> output;

    // Extract remaining samples from overlap buffer
    // Continue until we've extracted all meaningful samples
    int remaining = n_fft - hop_length;
    while (remaining > 0) {
        int chunk_size = std::min(remaining, hop_length);

        for (int i = 0; i < chunk_size; i++) {
            float sample;
            if (window_sum_buffer[i] > 1e-8f) {
                sample = overlap_buffer[i] / window_sum_buffer[i];
            } else {
                sample = overlap_buffer[i];
            }
            output.push_back(sample);
        }

        // Shift buffers
        std::copy(overlap_buffer.begin() + chunk_size, overlap_buffer.end(), overlap_buffer.begin());
        std::fill(overlap_buffer.end() - chunk_size, overlap_buffer.end(), 0.0f);

        std::copy(window_sum_buffer.begin() + chunk_size, window_sum_buffer.end(), window_sum_buffer.begin());
        std::fill(window_sum_buffer.end() - chunk_size, window_sum_buffer.end(), 0.0f);

        remaining -= chunk_size;
    }

    return output;
}
