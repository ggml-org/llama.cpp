#pragma once

/**
 * mtmd-mic.h - Microphone capture for mtmd CLI
 *
 * Provides cross-platform microphone capture using miniaudio.
 * Used by mtmd-cli for live audio transcription (e.g., Voxtral Realtime).
 */

#include <cstddef>
#include <vector>
#include <atomic>

struct mtmd_mic_ring_buffer {
    std::vector<float> buffer;
    size_t capacity = 0;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};

    void init(size_t cap);
    void reset();
    size_t write(const float * samples, size_t n_samples);
    size_t read(float * out, size_t max_samples);
    size_t available() const;
};

struct mtmd_mic_capture {
    mtmd_mic_ring_buffer ring;
    void * device = nullptr;  // opaque: ma_device*
    int sample_rate = 16000;
    bool is_running = false;

    int start(int sample_rate = 16000, float buffer_seconds = 30.0f);
    void stop();
    size_t read_samples(float * out, size_t max_samples);
    size_t available() const;
    ~mtmd_mic_capture();
};
