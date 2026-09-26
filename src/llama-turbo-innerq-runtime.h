#pragma once

#include <array>
#include <cstddef>
#include <mutex>

struct llama_turbo_innerq_runtime_snapshot {
    static constexpr size_t N_CHANNELS = 128;

    std::array<float, N_CHANNELS> scale_inv;
    bool   finalized        = false;
    bool   dirty            = false;
    int    abort_reason     = 0;
    int    retry_count      = 0;
    bool   freeze_last_good = false;
};

class llama_turbo_innerq_runtime_state {
public:
    llama_turbo_innerq_runtime_state();

    void publish_scale_inv(const float * scale_inv, size_t n, bool finalized);
    void publish_abort(int abort_reason, int retry_count, bool freeze_last_good);
    // Reset to the initial state (scale_inv = all 1.0f, not finalized,
    // no abort, no freeze). Called when the per-cache tensor is reset
    // so should_attach_scale_tensor() does not get stuck on stale
    // flags from a previous use of this cache.
    void reset();

    llama_turbo_innerq_runtime_snapshot peek() const;
    bool consume_if_dirty(llama_turbo_innerq_runtime_snapshot & out);
    bool should_attach_scale_tensor() const;

private:
    mutable std::mutex mutex;
    llama_turbo_innerq_runtime_snapshot state;
};
