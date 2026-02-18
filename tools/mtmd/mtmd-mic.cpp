/**
 * mtmd-mic.cpp - Microphone capture for mtmd CLI
 *
 * Implements cross-platform mic capture via miniaudio with device I/O enabled.
 * This is a separate compilation unit from mtmd-helper.cpp (which uses
 * MA_NO_DEVICE_IO for file decoding only).
 */

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include "mtmd-mic.h"

#include <cstring>
#include <algorithm>
#include <cstdio>

// ============================================================================
// Ring Buffer
// ============================================================================

void mtmd_mic_ring_buffer::init(size_t cap) {
    capacity = cap + 1;
    buffer.resize(capacity, 0.0f);
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
}

void mtmd_mic_ring_buffer::reset() {
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
}

size_t mtmd_mic_ring_buffer::write(const float * samples, size_t n_samples) {
    size_t written = 0;
    size_t h = head.load(std::memory_order_relaxed);
    size_t t = tail.load(std::memory_order_acquire);

    for (size_t i = 0; i < n_samples; i++) {
        size_t next = (h + 1) % capacity;
        if (next == t) break;
        buffer[h] = samples[i];
        h = next;
        written++;
    }

    head.store(h, std::memory_order_release);
    return written;
}

size_t mtmd_mic_ring_buffer::read(float * out, size_t max_samples) {
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

size_t mtmd_mic_ring_buffer::available() const {
    size_t h = head.load(std::memory_order_acquire);
    size_t t = tail.load(std::memory_order_acquire);
    return (h >= t) ? (h - t) : (capacity - t + h);
}

// ============================================================================
// Microphone Capture
// ============================================================================

static void mic_data_callback(ma_device * pDevice, void * pOutput, const void * pInput, ma_uint32 frameCount) {
    (void)pOutput;
    auto * ring = (mtmd_mic_ring_buffer *)pDevice->pUserData;
    ring->write((const float *)pInput, frameCount);
}

int mtmd_mic_capture::start(int target_sample_rate, float buffer_seconds) {
    if (is_running) return 0;

    sample_rate = target_sample_rate;
    ring.init((size_t)(target_sample_rate * buffer_seconds));

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format   = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate       = (ma_uint32)target_sample_rate;
    config.dataCallback     = mic_data_callback;
    config.pUserData        = &ring;
    config.periodSizeInFrames = 512;

    auto * dev = new ma_device;
    if (ma_device_init(NULL, &config, dev) != MA_SUCCESS) {
        fprintf(stderr, "mtmd_mic_capture: failed to initialize audio device\n");
        delete dev;
        return -1;
    }

    if (ma_device_start(dev) != MA_SUCCESS) {
        fprintf(stderr, "mtmd_mic_capture: failed to start audio device\n");
        ma_device_uninit(dev);
        delete dev;
        return -1;
    }

    device = dev;
    is_running = true;
    return 0;
}

void mtmd_mic_capture::stop() {
    if (!is_running || device == nullptr) return;

    auto * dev = (ma_device *)device;
    ma_device_uninit(dev);
    delete dev;
    device = nullptr;
    is_running = false;
}

size_t mtmd_mic_capture::read_samples(float * out, size_t max_samples) {
    return ring.read(out, max_samples);
}

size_t mtmd_mic_capture::available() const {
    return ring.available();
}

mtmd_mic_capture::~mtmd_mic_capture() {
    stop();
}
