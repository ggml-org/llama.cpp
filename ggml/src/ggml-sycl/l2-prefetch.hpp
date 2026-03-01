//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_L2_PREFETCH_HPP
#define GGML_SYCL_L2_PREFETCH_HPP

#include <sycl/sycl.hpp>
#include <atomic>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <string>

#include "common.hpp"

#ifdef GGML_SYCL_ESIMD_AVAILABLE
#include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
#endif

namespace ggml_sycl {

// L2 cache prefetch for TG (token generation) optimization.
//
// For batch=1 inference where the model fits in VRAM, performance is limited
// by memory bandwidth. By prefetching the next tensor's data into L2 cache
// while the current tensor is being computed, we can hide memory latency.
//
// Arc B580 L2 cache: ~18 MB
// Prefetch chunk size: 16 MB (safe margin)
//
// The prefetch kernel uses ESIMD lsc_prefetch to bring data into L2 with
// appropriate cache hints for retention.

// Maximum prefetch size (fits in L2 with margin)
constexpr size_t L2_PREFETCH_MAX_BYTES = 16 * 1024 * 1024;  // 16 MB

// Prefetch granularity (cache line size on Intel GPUs)
constexpr size_t L2_PREFETCH_CACHELINE = 64;

// Work-group size for prefetch kernel
constexpr int L2_PREFETCH_WG_SIZE = 256;

// Check if L2 prefetch is enabled via environment
// Disabled by default - enable with GGML_SYCL_L2_PREFETCH=1
inline bool is_l2_prefetch_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_L2_PREFETCH");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

#ifdef GGML_SYCL_ESIMD_AVAILABLE

// ESIMD L2 prefetch kernel
// Each work-item prefetches one cache line (64 bytes)
class l2_prefetch_kernel_name;

inline void launch_l2_prefetch_kernel(sycl::queue & q,
                                       const void * ptr,
                                       size_t       bytes,
                                       sycl::event  dep_event = sycl::event()) {
    if (!ptr || bytes == 0) {
        return;
    }

    // Clamp to max prefetch size
    bytes = std::min(bytes, L2_PREFETCH_MAX_BYTES);

    // Calculate number of cache lines to prefetch
    const size_t num_cachelines = (bytes + L2_PREFETCH_CACHELINE - 1) / L2_PREFETCH_CACHELINE;
    const size_t global_size = ((num_cachelines + L2_PREFETCH_WG_SIZE - 1) / L2_PREFETCH_WG_SIZE) * L2_PREFETCH_WG_SIZE;

    const uint8_t * base_ptr = static_cast<const uint8_t *>(ptr);

    q.submit([&](sycl::handler & cgh) {
        if (dep_event.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) {
            cgh.depends_on(dep_event);
        }

        cgh.parallel_for<l2_prefetch_kernel_name>(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(L2_PREFETCH_WG_SIZE)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
                const size_t idx = item.get_global_id(0);
                const size_t offset = idx * L2_PREFETCH_CACHELINE;

                if (offset < bytes) {
                    const uint8_t * addr = base_ptr + offset;

                    // Use ESIMD prefetch with L2 caching hints
                    // L1: cached (keep in L1 for immediate reuse)
                    // L2: cached (this is our target - warm up L2)
                    constexpr auto props = esimd::properties{
                        esimd::cache_hint_L1<esimd::cache_hint::cached>,
                        esimd::cache_hint_L2<esimd::cache_hint::cached>
                    };

                    // Prefetch 64 bytes (one cache line) as 16 uint32_t values
                    // Alignment: ensure we're 4-byte aligned for uint32_t access
                    const uint32_t * aligned_addr = reinterpret_cast<const uint32_t *>(
                        reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t)3);

                    esimd::prefetch<uint32_t, 16>(aligned_addr, 0, esimd::simd_mask<1>(1), props);
                }
            }
        );
    });
    // Note: Don't wait - prefetch runs asynchronously
}

#else  // !GGML_SYCL_ESIMD_AVAILABLE

// Fallback: use sycl::handler::prefetch (less control but portable)
inline void launch_l2_prefetch_kernel(sycl::queue & q,
                                       const void * ptr,
                                       size_t       bytes,
                                       sycl::event  dep_event = sycl::event()) {
    if (!ptr || bytes == 0) {
        return;
    }

    bytes = std::min(bytes, L2_PREFETCH_MAX_BYTES);

    q.submit([&](sycl::handler & cgh) {
        if (dep_event.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) {
            cgh.depends_on(dep_event);
        }
        cgh.prefetch(ptr, bytes);
    });
}

#endif  // GGML_SYCL_ESIMD_AVAILABLE

// Tensor info for prefetch tracking
struct tensor_prefetch_info {
    const void * ptr;      // Device pointer to tensor data
    size_t       bytes;    // Size in bytes
    int          layer_id; // Layer index (-1 for non-layer tensors)
    int          tensor_idx; // Index within layer (0=attn_q, 1=attn_k, etc.)
};

// L2 Prefetch Manager
// Tracks tensor execution order and issues async prefetch for next tensor
class L2PrefetchManager {
  public:
    explicit L2PrefetchManager(sycl::queue * compute_queue) :
        // Create separate queue for async prefetch (same context/device)
        prefetch_queue_(compute_queue->get_context(), compute_queue->get_device()) {

        GGML_LOG_INFO("[L2-PREFETCH] Manager initialized (max_bytes=%zu MB)\n",
                      L2_PREFETCH_MAX_BYTES / (1024 * 1024));
    }

    ~L2PrefetchManager() {
        // Wait for any pending prefetch
        try {
            prefetch_queue_.wait();
        } catch (...) {
            // Ignore exceptions during destruction
        }

        GGML_LOG_INFO("[L2-PREFETCH] Stats: registered=%zu, prefetch=%zu, skipped=%zu\n",
                      tensor_registry_.size(), prefetch_count_, skip_count_);
    }

    // Register a tensor for prefetch tracking
    // Called during model loading or first inference pass
    void register_tensor(const char * name, const void * ptr, size_t bytes) {
        if (!name || !ptr || bytes == 0) {
            return;
        }

        int layer_id = extract_layer_id(name);
        int tensor_idx = get_tensor_index_in_layer(name);

        if (layer_id >= 0 && tensor_idx >= 0) {
            std::lock_guard<std::mutex> lock(mutex_);

            std::string key = std::string(name);
            tensor_registry_[key] = {ptr, bytes, layer_id, tensor_idx};

            // Track max layer
            if (layer_id > max_layer_id_) {
                max_layer_id_ = layer_id;
            }

            GGML_SYCL_DEBUG("[L2-PREFETCH] Registered: %s layer=%d idx=%d bytes=%zu\n",
                            name, layer_id, tensor_idx, bytes);
        }
    }

    // Called when a tensor computation starts
    // Triggers async prefetch for the next tensor in sequence
    void on_tensor_compute_start(const char * current_name, const void * current_ptr) {
        if (!is_l2_prefetch_enabled() || !current_name) {
            return;
        }

        // Find next tensor to prefetch
        auto next = get_next_tensor(current_name);
        if (!next.has_value()) {
            skip_count_++;
            return;
        }

        // Launch async prefetch
        launch_l2_prefetch_kernel(prefetch_queue_, next->ptr, next->bytes);
        prefetch_count_++;

        GGML_SYCL_DEBUG("[L2-PREFETCH] Prefetching next tensor: layer=%d idx=%d bytes=%zu\n",
                        next->layer_id, next->tensor_idx, next->bytes);
    }

    // Reset for new sequence (clears execution state, keeps registry)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        last_layer_id_ = -1;
        last_tensor_idx_ = -1;
    }

  private:
    sycl::queue   prefetch_queue_;
    std::mutex    mutex_;

    // Tensor registry: name -> info
    std::unordered_map<std::string, tensor_prefetch_info> tensor_registry_;

    // Execution tracking
    int max_layer_id_ = -1;
    int last_layer_id_ = -1;
    int last_tensor_idx_ = -1;

    // Stats
    size_t prefetch_count_ = 0;
    size_t skip_count_ = 0;

    // Get tensor index within a layer (execution order)
    // 0=attn_q, 1=attn_k, 2=attn_v, 3=attn_output, 4=ffn_gate, 5=ffn_up, 6=ffn_down
    static int get_tensor_index_in_layer(const char * name) {
        if (strstr(name, "attn_q.weight")) return 0;
        if (strstr(name, "attn_k.weight")) return 1;
        if (strstr(name, "attn_v.weight")) return 2;
        if (strstr(name, "attn_output.weight")) return 3;
        if (strstr(name, "ffn_gate.weight")) return 4;
        if (strstr(name, "ffn_up.weight")) return 5;
        if (strstr(name, "ffn_down.weight")) return 6;
        return -1;
    }

    // Build tensor name from layer_id and tensor_idx
    static std::string build_tensor_name(int layer_id, int tensor_idx) {
        static const char * tensor_suffixes[] = {
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight"
        };
        if (tensor_idx < 0 || tensor_idx > 6) {
            return "";
        }
        return "blk." + std::to_string(layer_id) + "." + tensor_suffixes[tensor_idx];
    }

    // Get the next tensor to prefetch based on current tensor
    std::optional<tensor_prefetch_info> get_next_tensor(const char * current_name) {
        std::lock_guard<std::mutex> lock(mutex_);

        int layer_id = extract_layer_id(current_name);
        int tensor_idx = get_tensor_index_in_layer(current_name);

        if (layer_id < 0 || tensor_idx < 0) {
            return std::nullopt;
        }

        // Calculate next tensor
        int next_layer = layer_id;
        int next_idx = tensor_idx + 1;

        if (next_idx > 6) {
            // Move to next layer
            next_layer = layer_id + 1;
            next_idx = 0;

            // Check if we've passed the last layer
            if (next_layer > max_layer_id_) {
                return std::nullopt;
            }
        }

        // Look up next tensor in registry
        std::string next_name = build_tensor_name(next_layer, next_idx);
        auto it = tensor_registry_.find(next_name);
        if (it == tensor_registry_.end()) {
            return std::nullopt;
        }

        return it->second;
    }
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_L2_PREFETCH_HPP
