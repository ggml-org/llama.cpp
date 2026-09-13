#pragma once
#include "llama-kv-stream-policy.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include <cmath>

// Convert completed backend windows to cumulative policy counters without claiming hardware utilization.
class llama_kv_stream_feedback_window {
public:
    void reset() { reset(value.epoch+1); }
    // The runtime supplies cross-instance identity; pure tests can use local monotonic epochs.
    void reset(uint64_t epoch) { value = {}; value.epoch = epoch; }
    bool add(const ggml_kv_stream_copy_feedback & sample) {
        if (!sample.available || !sample.samples || sample.misses > sample.samples || !sample.bytes ||
                !sample.timed_bytes || sample.timed_bytes > sample.bytes || sample.peak_slots > UINT32_MAX ||
                !std::isfinite(sample.copy_ms) || sample.copy_ms < 0 || !std::isfinite(sample.elapsed_ms) || sample.elapsed_ms <= 0 ||
                sample.samples > UINT64_MAX-value.samples || sample.misses > UINT64_MAX-value.misses) return false;
        const double busy = (sample.copy_ms/sample.elapsed_ms)*(double(sample.bytes)/sample.timed_bytes);
        if (!std::isfinite(busy)) return false;
        value.available = true; value.samples += sample.samples; value.misses += sample.misses;
        value.copy_busy_ratio = std::min(1.0,busy); value.peak_slots = uint32_t(sample.peak_slots);
        return true;
    }
    llama_kv_stream_feedback snapshot() const { return value; }
private:
    llama_kv_stream_feedback value{false,1,0,0,0,0};
};

// Compare full-ring and bounded spans using completed run latency, with one warmup per candidate.
class llama_kv_stream_span_tuner {
public:
    explicit llama_kv_stream_span_tuner(uint32_t trials = 16) : trials(std::max(1u,trials)) {}
    void reset() { count[0] = count[1] = 0; total[0] = total[1] = 0; choice = done = false; }
    void observe(double ms, bool bounded) {
        if (done || bounded != choice || !std::isfinite(ms) || ms <= 0) return;
        const size_t i = bounded ? 1 : 0;
        if (count[i] == 0) { ++count[i]; return; }
        if (!std::isfinite(total[i]+ms)) return;
        ++count[i];
        total[i] += ms;
        if (count[i]-1 < trials) return;
        if (!bounded) { choice = true; return; }
        choice = total[1] < total[0]*.995; done = true;
    }
    bool bounded() const { return choice; }
    bool selected() const { return done; }
private:
    uint32_t trials;
    uint64_t count[2] = {};
    double total[2] = {};
    bool choice = false, done = false;
};
