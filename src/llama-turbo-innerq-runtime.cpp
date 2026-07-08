#include "llama-turbo-innerq-runtime.h"

#include <algorithm>

llama_turbo_innerq_runtime_state::llama_turbo_innerq_runtime_state() {
    state.scale_inv.fill(1.0f);
}

void llama_turbo_innerq_runtime_state::publish_scale_inv(const float * scale_inv, size_t n, bool finalized) {
    std::lock_guard<std::mutex> lock(mutex);

    state.scale_inv.fill(1.0f);
    if (scale_inv != nullptr) {
        const size_t n_copy = std::min(n, state.scale_inv.size());
        std::copy_n(scale_inv, n_copy, state.scale_inv.begin());
    }

    state.finalized = finalized;
    // A fresh successful scale publish supersedes any prior abort-only state.
    state.abort_reason = 0;
    state.retry_count = 0;
    state.freeze_last_good = false;
    state.dirty = true;
}

void llama_turbo_innerq_runtime_state::publish_abort(int abort_reason, int retry_count, bool freeze_last_good) {
    std::lock_guard<std::mutex> lock(mutex);

    state.abort_reason = abort_reason;
    state.retry_count = retry_count;
    state.freeze_last_good = freeze_last_good;
    if (!freeze_last_good) {
        state.scale_inv.fill(1.0f);
        state.finalized = false;
    }
    state.dirty = true;
}

llama_turbo_innerq_runtime_snapshot llama_turbo_innerq_runtime_state::peek() const {
    std::lock_guard<std::mutex> lock(mutex);
    return state;
}

bool llama_turbo_innerq_runtime_state::consume_if_dirty(llama_turbo_innerq_runtime_snapshot & out) {
    std::lock_guard<std::mutex> lock(mutex);

    out = state;
    if (!state.dirty) {
        return false;
    }

    state.dirty = false;
    return true;
}

bool llama_turbo_innerq_runtime_state::should_attach_scale_tensor() const {
    std::lock_guard<std::mutex> lock(mutex);

    if (state.abort_reason != 0) {
        return state.freeze_last_good;
    }

    return state.finalized;
}
