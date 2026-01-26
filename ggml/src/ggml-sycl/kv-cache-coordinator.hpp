#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace ggml_sycl {

// Memory model for KV cache
enum class KVCacheModel {
    STANDARD,        // Full context, grows linearly
    SLIDING_WINDOW,  // Fixed window, constant memory
    FLASH_ATTENTION  // Uses scratch memory for softmax
};

// KV cache memory requirements
struct KVCacheRequirements {
    size_t       bytes_per_token;      // Memory per token (2 * n_layer * n_embd * sizeof(half))
    size_t       max_context_tokens;   // Maximum context length
    size_t       current_tokens;       // Current token count
    size_t       sliding_window_size;  // For sliding window model (0 if not used)
    size_t       flash_scratch_size;   // For flash attention scratch (0 if not used)
    KVCacheModel model;
};

class KVCacheCoordinator {
  public:
    using MemoryCallback = std::function<void(size_t available_bytes)>;

    KVCacheCoordinator() = default;

    // Initialize with model parameters
    void init(int n_layer, int n_embd, int n_head, size_t max_context, KVCacheModel model = KVCacheModel::STANDARD) {
        n_layer_ = n_layer;
        n_embd_  = n_embd;
        n_head_  = n_head;

        reqs_.model              = model;
        reqs_.max_context_tokens = max_context;
        reqs_.current_tokens     = 0;

        // 2 (K+V) * n_layer * n_embd * sizeof(half)
        reqs_.bytes_per_token = 2 * n_layer * n_embd * sizeof(uint16_t);

        // Flash attention scratch: batch * n_head * seq_len * sizeof(float)
        // Estimated for typical batch size
        reqs_.flash_scratch_size = (model == KVCacheModel::FLASH_ATTENTION) ? n_head * max_context * sizeof(float) : 0;
    }

    // Set sliding window size
    void set_sliding_window(size_t window_size) {
        reqs_.sliding_window_size = window_size;
        if (window_size > 0) {
            reqs_.model = KVCacheModel::SLIDING_WINDOW;
        }
    }

    // Get current KV cache memory usage
    size_t get_current_kv_bytes() const {
        if (reqs_.model == KVCacheModel::SLIDING_WINDOW && reqs_.sliding_window_size > 0) {
            // Sliding window: capped at window size
            size_t effective_tokens = std::min(reqs_.current_tokens, reqs_.sliding_window_size);
            return effective_tokens * reqs_.bytes_per_token;
        }
        return reqs_.current_tokens * reqs_.bytes_per_token;
    }

    // Get maximum KV cache memory
    size_t get_max_kv_bytes() const {
        if (reqs_.model == KVCacheModel::SLIDING_WINDOW && reqs_.sliding_window_size > 0) {
            return reqs_.sliding_window_size * reqs_.bytes_per_token;
        }
        return reqs_.max_context_tokens * reqs_.bytes_per_token;
    }

    // Get flash attention scratch size
    size_t get_flash_scratch_bytes() const { return reqs_.flash_scratch_size; }

    // Called when a token is generated
    void on_token_generated(size_t total_memory_budget) {
        reqs_.current_tokens++;

        size_t kv_bytes = get_current_kv_bytes();
        size_t scratch  = get_flash_scratch_bytes();
        size_t reserved = kv_bytes + scratch;

        // Notify weight cache of available memory
        size_t available = (reserved < total_memory_budget) ? total_memory_budget - reserved : 0;

        if (weight_cache_callback_) {
            weight_cache_callback_(available);
        }
    }

    // Register callback to notify weight cache
    void set_weight_cache_callback(MemoryCallback callback) { weight_cache_callback_ = std::move(callback); }

    // Predict memory needed for N more tokens
    size_t predict_memory_for_tokens(size_t additional_tokens) const {
        if (reqs_.model == KVCacheModel::SLIDING_WINDOW && reqs_.sliding_window_size > 0) {
            // Sliding window doesn't grow beyond window size
            size_t future_tokens = reqs_.current_tokens + additional_tokens;
            size_t effective     = std::min(future_tokens, reqs_.sliding_window_size);
            return effective * reqs_.bytes_per_token;
        }
        return (reqs_.current_tokens + additional_tokens) * reqs_.bytes_per_token;
    }

    // Check if we can generate N more tokens within budget
    bool can_generate_tokens(size_t additional_tokens, size_t available_memory) const {
        size_t needed = predict_memory_for_tokens(additional_tokens);
        return needed <= available_memory;
    }

    // Get bytes per token
    size_t get_bytes_per_token() const { return reqs_.bytes_per_token; }

    // Get current token count
    size_t get_current_tokens() const { return reqs_.current_tokens; }

    // Reset for new inference
    void reset() { reqs_.current_tokens = 0; }

    // Get requirements
    const KVCacheRequirements & get_requirements() const { return reqs_; }

  private:
    int                 n_layer_ = 0;
    int                 n_embd_  = 0;
    int                 n_head_  = 0;
    KVCacheRequirements reqs_{};
    MemoryCallback      weight_cache_callback_;
};

}  // namespace ggml_sycl
