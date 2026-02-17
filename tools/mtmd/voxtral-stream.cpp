/**
 * voxtral-stream.cpp - Streaming audio infrastructure for Voxtral Realtime
 *
 * Implements:
 * 1. Lock-free ring buffer for audio samples
 * 2. Cross-platform microphone capture via miniaudio
 * 3. Streaming mel spectrogram computation (incremental FFT + mel filterbank)
 * 4. Streaming session management
 *
 * This file is self-contained and does not depend on mtmd internal headers.
 * It uses miniaudio with device I/O enabled (separate from mtmd-helper.cpp
 * which uses MA_NO_DEVICE_IO).
 */

// Fix MSVC min/max macro conflicts
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// miniaudio: enable device I/O for microphone capture
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include "voxtral-stream.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Audio Ring Buffer Implementation
// ============================================================================

void voxtral_ring_buffer::init(size_t cap) {
    capacity = cap + 1;  // +1 to distinguish full from empty
    buffer.resize(capacity, 0.0f);
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
}

void voxtral_ring_buffer::reset() {
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
}

size_t voxtral_ring_buffer::write(const float * samples, size_t n_samples) {
    size_t written = 0;
    size_t h = head.load(std::memory_order_relaxed);
    size_t t = tail.load(std::memory_order_acquire);

    for (size_t i = 0; i < n_samples; i++) {
        size_t next = (h + 1) % capacity;
        if (next == t) {
            break;  // buffer full
        }
        buffer[h] = samples[i];
        h = next;
        written++;
    }

    head.store(h, std::memory_order_release);
    return written;
}

size_t voxtral_ring_buffer::read(float * out, size_t max_samples) {
    size_t count = 0;
    size_t h = head.load(std::memory_order_acquire);
    size_t t = tail.load(std::memory_order_relaxed);

    while (count < max_samples && t != h) {
        out[count++] = buffer[t];
        t = (t + 1) % capacity;
    }

    tail.store(t, std::memory_order_release);
    return count;
}

size_t voxtral_ring_buffer::available() const {
    size_t h = head.load(std::memory_order_acquire);
    size_t t = tail.load(std::memory_order_acquire);
    if (h >= t) {
        return h - t;
    }
    return capacity - t + h;
}

// ============================================================================
// Microphone Capture Implementation
// ============================================================================

static void mic_data_callback(ma_device * pDevice, void * pOutput, const void * pInput, ma_uint32 frameCount) {
    (void)pOutput;
    voxtral_ring_buffer * ring = (voxtral_ring_buffer *)pDevice->pUserData;
    const float * samples = (const float *)pInput;
    ring->write(samples, frameCount);
}

int voxtral_mic_capture::start(int target_sample_rate, float buffer_seconds) {
    if (is_running) {
        return 0;  // already running
    }

    sample_rate = target_sample_rate;
    size_t buffer_frames = (size_t)(target_sample_rate * buffer_seconds);
    ring.init(buffer_frames);

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format   = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate       = (ma_uint32)target_sample_rate;
    config.dataCallback     = mic_data_callback;
    config.pUserData        = &ring;
    config.periodSizeInFrames = 512;  // low latency (~32ms at 16kHz)

    ma_device * dev = new ma_device;
    if (ma_device_init(NULL, &config, dev) != MA_SUCCESS) {
        fprintf(stderr, "voxtral_mic_capture: failed to initialize audio device\n");
        delete dev;
        return -1;
    }

    if (ma_device_start(dev) != MA_SUCCESS) {
        fprintf(stderr, "voxtral_mic_capture: failed to start audio device\n");
        ma_device_uninit(dev);
        delete dev;
        return -1;
    }

    device = dev;
    is_running = true;

    return 0;
}

void voxtral_mic_capture::stop() {
    if (!is_running || device == nullptr) {
        return;
    }

    ma_device * dev = (ma_device *)device;
    ma_device_uninit(dev);
    delete dev;
    device = nullptr;
    is_running = false;

    // silent stop
}

size_t voxtral_mic_capture::read_samples(float * out, size_t max_samples) {
    return ring.read(out, max_samples);
}

size_t voxtral_mic_capture::available() const {
    return ring.available();
}

voxtral_mic_capture::~voxtral_mic_capture() {
    stop();
}

// ============================================================================
// Internal mel filterbank and FFT cache (self-contained)
// ============================================================================

struct mel_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    std::vector<float> hann_window;

    // Mel filterbank: [n_mel x n_fft_bins]
    int32_t filter_n_mel     = 0;
    int32_t filter_n_fft     = 0;
    int32_t filter_n_fft_bins = 0;
    std::vector<float> filter_data;

    bool initialized = false;

    void init(int n_fft, int n_mel, int sample_rate) {
        // Sin/cos table
        sin_vals.resize(n_fft);
        cos_vals.resize(n_fft);
        for (int i = 0; i < n_fft; i++) {
            double theta = (2.0 * M_PI * i) / n_fft;
            sin_vals[i] = (float)sin(theta);
            cos_vals[i] = (float)cos(theta);
        }

        // Periodic Hann window
        hann_window.resize(n_fft);
        for (int i = 0; i < n_fft; i++) {
            hann_window[i] = 0.5f * (1.0f - (float)cos((2.0 * M_PI * i) / n_fft));
        }

        // Build mel filterbank (Slaney scale, matching librosa/Whisper)
        filter_n_fft = n_fft;
        filter_n_mel = n_mel;
        filter_n_fft_bins = n_fft / 2 + 1;

        float fmin = 0.0f;
        float fmax = (float)sample_rate / 2.0f;

        // Slaney mel scale
        const double min_log_hz  = 1000.0;
        const double lin_slope   = 3.0 / 200.0;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step    = log(6.4) / 27.0;

        auto hz_to_mel = [&](double f) -> double {
            return (f < min_log_hz) ? f * lin_slope : min_log_mel + log(f / min_log_hz) / log_step;
        };
        auto mel_to_hz = [&](double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
        };

        double bin_hz_step = (double)sample_rate / (double)n_fft;

        // Mel grid: n_mel + 2 edges
        double m_lo = hz_to_mel(fmin);
        double m_hi = hz_to_mel(fmax);
        std::vector<double> mel_pts(n_mel + 2);
        for (int i = 0; i < n_mel + 2; i++) {
            mel_pts[i] = m_lo + (m_hi - m_lo) * ((double)i / (n_mel + 1));
        }

        std::vector<double> hz_pts(n_mel + 2);
        for (int i = 0; i < n_mel + 2; i++) {
            hz_pts[i] = mel_to_hz(mel_pts[i]);
        }

        filter_data.resize(n_mel * filter_n_fft_bins, 0.0f);
        for (int m = 0; m < n_mel; m++) {
            double f_left   = hz_pts[m];
            double f_center = hz_pts[m + 1];
            double f_right  = hz_pts[m + 2];

            double denom_l = std::max(1e-30, f_center - f_left);
            double denom_r = std::max(1e-30, f_right - f_center);
            double enorm   = 2.0 / std::max(1e-30, f_right - f_left);  // Slaney area norm

            for (int k = 0; k < filter_n_fft_bins; k++) {
                double f = k * bin_hz_step;
                double w = 0.0;
                if (f >= f_left && f <= f_center) {
                    w = (f - f_left) / denom_l;
                } else if (f > f_center && f <= f_right) {
                    w = (f_right - f) / denom_r;
                }
                filter_data[(size_t)m * (size_t)filter_n_fft_bins + (size_t)k] = (float)(w * enorm);
            }
        }

        initialized = true;
    }
};

// ============================================================================
// Streaming Mel Spectrogram Implementation
// ============================================================================

struct voxtral_streaming_mel::cache_data {
    mel_cache mel;
};

void voxtral_streaming_mel::init(int n_fft_val, int hop_length_val, int n_mel_val, int sample_rate_val) {
    n_fft       = n_fft_val;
    hop_length  = hop_length_val;
    n_mel       = n_mel_val;
    sample_rate = sample_rate_val;

    if (cache == nullptr) {
        cache = new cache_data;
    }

    cache->mel.init(n_fft, n_mel, sample_rate);
    reset();
}

void voxtral_streaming_mel::reset() {
    sample_buffer.clear();
    total_frames_produced = 0;
}

// Compute mel spectrogram for a contiguous block of samples
// Single-threaded, self-contained DFT + mel filterbank
static void compute_mel_frames(
        const float * samples,
        int n_samples,
        int frame_size,
        int frame_step,
        int n_mel_bins,
        const mel_cache & mc,
        std::vector<float> & mel_data,
        int & n_frames_out) {

    const float * hann = mc.hann_window.data();
    const int n_fft_bins = frame_size / 2 + 1;
    const int n_sin_cos = (int)mc.sin_vals.size();

    n_frames_out = (n_samples - frame_size) / frame_step + 1;
    if (n_frames_out <= 0) {
        n_frames_out = 0;
        return;
    }

    mel_data.resize(n_mel_bins * n_frames_out);

    std::vector<float> fft_in(frame_size, 0.0f);
    std::vector<float> mag_sq(n_fft_bins, 0.0f);

    for (int i = 0; i < n_frames_out; i++) {
        const int offset = i * frame_step;

        // Apply Hann window
        int valid = std::min(frame_size, n_samples - offset);
        for (int j = 0; j < valid; j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        for (int j = valid; j < frame_size; j++) {
            fft_in[j] = 0.0f;
        }

        // DFT -> magnitude squared (only positive frequencies)
        int sin_cos_step = n_sin_cos / frame_size;
        for (int k = 0; k < n_fft_bins; k++) {
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < frame_size; n++) {
                int idx = (int)(((int64_t)k * n * sin_cos_step) % n_sin_cos);
                re += fft_in[n] * mc.cos_vals[idx];
                im -= fft_in[n] * mc.sin_vals[idx];
            }
            mag_sq[k] = re * re + im * im;
        }

        // Mel filterbank -> log10
        for (int j = 0; j < n_mel_bins; j++) {
            double sum = 0.0;
            int k = 0;
            for (k = 0; k < n_fft_bins - 3; k += 4) {
                size_t idx = (size_t)j * (size_t)n_fft_bins + (size_t)k;
                sum += mag_sq[k + 0] * mc.filter_data[idx + 0]
                     + mag_sq[k + 1] * mc.filter_data[idx + 1]
                     + mag_sq[k + 2] * mc.filter_data[idx + 2]
                     + mag_sq[k + 3] * mc.filter_data[idx + 3];
            }
            for (; k < n_fft_bins; k++) {
                sum += mag_sq[k] * mc.filter_data[j * n_fft_bins + k];
            }

            // Log10 with floor (Whisper-style)
            sum = log10(std::max(sum, 1e-10));

            // Store in [mel_bin, frame] layout
            mel_data[j * n_frames_out + i] = (float)sum;
        }
    }
}

// Apply Whisper-style normalization to mel data
static void normalize_mel(std::vector<float> & data, int n_mel_bins, int n_len) {
    double mmax = -1e20;
    for (int i = 0; i < n_mel_bins * n_len; i++) {
        if (data[i] > mmax) {
            mmax = data[i];
        }
    }
    mmax -= 8.0;
    for (int i = 0; i < n_mel_bins * n_len; i++) {
        if (data[i] < mmax) {
            data[i] = (float)mmax;
        }
        data[i] = (data[i] + 4.0f) / 4.0f;
    }
}

bool voxtral_streaming_mel::process(const float * samples, size_t n_samples,
                                     std::vector<voxtral_mel_chunk> & output,
                                     size_t chunk_frames) {
    if (cache == nullptr || !cache->mel.initialized) {
        return false;
    }

    // Append new samples to internal buffer
    if (samples != nullptr && n_samples > 0) {
        size_t old_size = sample_buffer.size();
        sample_buffer.resize(old_size + n_samples);
        std::memcpy(sample_buffer.data() + old_size, samples, n_samples * sizeof(float));
    }

    // We need at least one full FFT frame to produce any mel frames
    int frame_size = n_fft;
    if ((int)sample_buffer.size() < frame_size) {
        return false;
    }

    if (chunk_frames == 0) {
        // Return all available frames as a single chunk
        int total_possible_frames = ((int)sample_buffer.size() - frame_size) / hop_length + 1;
        if (total_possible_frames <= 0) {
            return false;
        }

        std::vector<float> mel_data;
        int n_frames = 0;
        compute_mel_frames(sample_buffer.data(), (int)sample_buffer.size(),
                           frame_size, hop_length, n_mel,
                           cache->mel, mel_data, n_frames);

        if (n_frames <= 0) {
            return false;
        }

        normalize_mel(mel_data, n_mel, n_frames);

        voxtral_mel_chunk chunk;
        chunk.n_len     = n_frames;
        chunk.n_len_org = n_frames;
        chunk.n_mel     = n_mel;
        chunk.data      = std::move(mel_data);
        output.push_back(std::move(chunk));

        // Keep overlap for next call
        int consumed_samples = (n_frames - 1) * hop_length + frame_size;
        int remaining = (int)sample_buffer.size() - consumed_samples + (frame_size - hop_length);
        if (remaining > 0 && remaining < (int)sample_buffer.size()) {
            std::vector<float> leftover(sample_buffer.end() - remaining, sample_buffer.end());
            sample_buffer = std::move(leftover);
        } else {
            sample_buffer.clear();
        }

        total_frames_produced += n_frames;
        return true;
    }

    // Produce chunks of chunk_frames mel frames each
    bool produced = false;

    while (true) {
        int samples_needed = (int)chunk_frames * hop_length + (frame_size - hop_length);
        if ((int)sample_buffer.size() < samples_needed) {
            break;
        }

        std::vector<float> mel_data;
        int n_frames = 0;
        compute_mel_frames(sample_buffer.data(), samples_needed,
                           frame_size, hop_length, n_mel,
                           cache->mel, mel_data, n_frames);

        if (n_frames <= 0) {
            break;
        }

        normalize_mel(mel_data, n_mel, n_frames);

        voxtral_mel_chunk chunk;
        chunk.n_len     = n_frames;
        chunk.n_len_org = n_frames;
        chunk.n_mel     = n_mel;
        chunk.data      = std::move(mel_data);
        output.push_back(std::move(chunk));

        // Advance buffer
        int consumed = (int)chunk_frames * hop_length;
        if (consumed < (int)sample_buffer.size()) {
            sample_buffer.erase(sample_buffer.begin(), sample_buffer.begin() + consumed);
        } else {
            sample_buffer.clear();
        }

        total_frames_produced += n_frames;
        produced = true;
    }

    return produced;
}

bool voxtral_streaming_mel::flush(std::vector<voxtral_mel_chunk> & output,
                                   size_t chunk_frames) {
    if (cache == nullptr || !cache->mel.initialized) {
        return false;
    }

    if (sample_buffer.empty()) {
        return false;
    }

    // Pad with zeros to fill a complete chunk
    int frame_size = n_fft;
    size_t target_samples;
    if (chunk_frames > 0) {
        target_samples = chunk_frames * hop_length + (frame_size - hop_length);
    } else {
        target_samples = sample_buffer.size() + frame_size;
    }

    if (sample_buffer.size() < target_samples) {
        sample_buffer.resize(target_samples, 0.0f);
    }

    return process(nullptr, 0, output, chunk_frames > 0 ? chunk_frames : 0);
}

voxtral_streaming_mel::~voxtral_streaming_mel() {
    delete cache;
}

// ============================================================================
// Streaming Session Implementation
// ============================================================================

int voxtral_stream_session::init(int sr, int nfft, int hop, int nmel) {
    mel_proc.init(nfft, hop, nmel, sr);
    return 0;
}

int voxtral_stream_session::start_mic() {
    int ret = mic.start(mel_proc.sample_rate, 30.0f);
    if (ret != 0) {
        return ret;
    }
    is_running = true;
    return 0;
}

int voxtral_stream_session::process_file_chunk(const float * samples, size_t n_samples,
                                                std::vector<voxtral_mel_chunk> & mel_chunks) {
    size_t chunk_frames = (size_t)(chunk_duration_ms * mel_proc.sample_rate / (1000 * mel_proc.hop_length));
    mel_proc.process(samples, n_samples, mel_chunks, chunk_frames);
    total_audio_ms += (n_samples * 1000) / mel_proc.sample_rate;
    return (int)mel_chunks.size();
}

void voxtral_stream_session::stop() {
    mic.stop();
    is_running = false;
}
