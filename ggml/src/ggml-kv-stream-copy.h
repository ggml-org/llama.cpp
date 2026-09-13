#pragma once
#include "ggml-backend.h"
#include "ggml-kv-stream.h"
#include <vector>
#include <algorithm>

// Clamp a logical run to both the physical ring boundary and the caller's span ceiling.
inline size_t ggml_kv_stream_contiguous_pages(size_t block, size_t blocks, size_t slots, size_t limit) {
    if (block >= blocks || !slots || !limit) return 0;
    return std::min({blocks-block,slots-block%slots,limit});
}

// Owner-thread bookkeeping only; GPU completion is established by backend events, not these states.
class ggml_kv_stream_copy_state {
public:
    enum class phase { empty, queued, acquired, released };
    explicit ggml_kv_stream_copy_state(size_t slots) : slots(slots,phase::empty) {}
    // A new run starts only after the previous run has drained.
    bool begin() { if (active || slots.empty()) return false; active = true; return true; }
    bool running() const { return active; }
    bool can_queue(size_t i) const { return active && i < slots.size() && (slots[i] == phase::empty || slots[i] == phase::released); }
    // Validate the entire physical run before changing any slot's ownership.
    bool can_queue_span(size_t first, size_t count) const {
        if (!active || !count || first >= slots.size() || count > slots.size()-first) return false;
        for (size_t i = 0; i < count; ++i) if (!can_queue(first+i)) return false;
        return true;
    }
    bool queue_span(size_t first, size_t count) {
        if (!can_queue_span(first,count)) return false;
        for (size_t i = 0; i < count; ++i) slots[first+i] = phase::queued;
        return true;
    }
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

struct ggml_kv_stream_copy_stats { size_t bytes = 0, calls = 0; };

struct ggml_kv_stream_copy_feedback {
    bool available = false;
    uint64_t samples = 0, misses = 0;
    size_t bytes = 0, timed_bytes = 0, peak_slots = 0, instrumentation_bytes = 0;
    // Completed copy-stream intervals exclude dependency waits; elapsed_ms is the whole host run window.
    double copy_ms = 0, elapsed_ms = 0;
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
    // Version 2: one contiguous K transfer and one V transfer; never crosses the physical ring boundary.
    bool (*enqueue_span)(void *, size_t first_slot, const void * k, const void * v, size_t live_tokens, size_t padded_tokens);
    // Actual submitted payload bytes and memcpy calls, excluding padding fills; reset by begin().
    ggml_kv_stream_copy_stats (*stats)(void *);
    // Version 3: caller has already synchronized every encoded-slot reader; no queued consumer fence is needed.
    bool (*release_completed)(void *, size_t slot);
    // Version 4: opt-in measurement may allocate bounded diagnostic storage; change only while idle.
    bool (*measure)(void *, bool enable);
    // One GPU deadline sample for the whole consumed span, before its mandatory ready-event waits.
    bool (*acquire_span)(void *, size_t first_slot, size_t count);
    // Available only after drain; cancellation may expose diagnostics, but must not train runtime policy.
    ggml_kv_stream_copy_feedback (*feedback)(void *);
};
using ggml_kv_stream_copy_ops_get = const ggml_kv_stream_copy_ops * (*)();
