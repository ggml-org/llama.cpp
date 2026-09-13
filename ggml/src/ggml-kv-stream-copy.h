#pragma once
#include "ggml-backend.h"
#include "ggml-kv-stream.h"
#include <vector>

// Owner-thread bookkeeping only; GPU completion is established by backend events, not these states.
class ggml_kv_stream_copy_state {
public:
    enum class phase { empty, queued, acquired, released };
    explicit ggml_kv_stream_copy_state(size_t slots) : slots(slots,phase::empty) {}
    // A new run starts only after the previous run has drained.
    bool begin() { if (active || slots.empty()) return false; active = true; return true; }
    bool can_queue(size_t i) const { return active && i < slots.size() && (slots[i] == phase::empty || slots[i] == phase::released); }
    bool recycled(size_t i) const { return active && i < slots.size() && slots[i] == phase::released; }
    bool waiting(size_t i) const { return active && i < slots.size() && slots[i] == phase::queued; }
    bool held(size_t i) const { return active && i < slots.size() && slots[i] == phase::acquired; }
    // Publish transitions only after the corresponding event/copy commands have been submitted.
    bool queue(size_t i) { if (!can_queue(i)) return false; slots[i] = phase::queued; return true; }
    bool acquire(size_t i) { if (!waiting(i)) return false; slots[i] = phase::acquired; return true; }
    bool release(size_t i) { if (!held(i)) return false; slots[i] = phase::released; return true; }
    // Call only after both streams drain, including on cancellation or failure.
    void drained() { active = false; for (auto & slot : slots) slot = phase::empty; }
private:
    std::vector<phase> slots;
    bool active = false;
};

// Optional registry "ggml_backend_kv_stream_copy_ops". Calls enqueue work except drain/free.
// Caller holds the device lease/pin and immutable host content until drain; backend outlives the handle.
// Owner-thread only and outside active CUDA capture; ready() is observation, not a host-content lifetime fence.
struct ggml_kv_stream_copy_ops {
    uint32_t version;
    // Retain buffers and create event resources, without allocating KV storage or submitting work.
    void * (*create)(ggml_backend_t, ggml_backend_buffer_t device, ggml_backend_buffer_t host,
                     const ggml_kv_stream_shape &, size_t slots);
    bool (*begin)(void *);
    bool (*enqueue)(void *, size_t slot, const void * k, const void * v, size_t live_tokens, size_t padded_tokens);
    bool (*ready)(void *, size_t slot);
    bool (*acquire)(void *, size_t slot);
    bool (*release)(void *, size_t slot);
    // Cancellation abandons unconsumed slots only after already-submitted work completes.
    void (*drain)(void *);
    void (*free)(void *);
};
using ggml_kv_stream_copy_ops_get = const ggml_kv_stream_copy_ops * (*)();
