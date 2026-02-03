//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_LAYER_PREFETCH_HPP
#define GGML_SYCL_LAYER_PREFETCH_HPP

#include <atomic>
#include <cstring>
#include <string>
#include <unordered_map>
#include <mutex>

#include "tensor-types.hpp"

namespace ggml_sycl {

// Inter-layer weight prefetch system for TG (token generation) workloads.
//
// For batch=1 inference (TG), the memory bandwidth is the bottleneck.
// By prefetching layer N+1's weights while layer N is computing, we can
// hide memory latency and improve throughput.
//
// The transformer layer structure is predictable:
// blk.N.attn_q.weight -> blk.N.attn_k.weight -> ... -> blk.N.ffn_down.weight -> blk.N+1.attn_q.weight
//
// This system tracks layer transitions and issues async prefetch for the next layer's
// first tensor (attn_q.weight) when we detect we're starting a new layer.

class layer_prefetch_tracker {
  public:
    // Tensor sequence within a layer (Mistral/LLaMA architecture)
    // These are processed in order during inference
    enum class tensor_in_layer {
        ATTN_NORM,      // attn_norm (not a matmul, but marks layer start)
        ATTN_Q,         // attn_q.weight
        ATTN_K,         // attn_k.weight
        ATTN_V,         // attn_v.weight
        ATTN_OUTPUT,    // attn_output.weight
        FFN_NORM,       // ffn_norm (not a matmul)
        FFN_GATE,       // ffn_gate.weight (SwiGLU)
        FFN_UP,         // ffn_up.weight
        FFN_DOWN,       // ffn_down.weight - last tensor in layer
        UNKNOWN
    };

    // Record a tensor access during TG (batch=1) inference
    // Returns true if this tensor triggered a layer transition (prefetch opportunity)
    // next_layer_id is set to the layer that should be prefetched (-1 if none)
    bool record_access(const char * tensor_name, int & next_layer_id) {
        if (!tensor_name) {
            return false;
        }

        int layer_id = extract_layer_id(tensor_name);
        if (layer_id < 0) {
            return false;  // Not a layer tensor (embedding, output head, etc.)
        }

        tensor_in_layer tensor_type = classify_tensor_in_layer(tensor_name);
        if (tensor_type == tensor_in_layer::UNKNOWN) {
            return false;
        }

        // Check for layer transition: seeing layer N when we were on layer N-1
        int prev_layer = current_layer_.load(std::memory_order_relaxed);

        if (layer_id > prev_layer) {
            // Layer transition detected!
            current_layer_.store(layer_id, std::memory_order_relaxed);

            // When we see attn_q of layer N, prefetch attn_q of layer N+1
            // This gives the maximum prefetch window (entire layer's compute)
            if (tensor_type == tensor_in_layer::ATTN_Q) {
                next_layer_id = layer_id + 1;
                return true;
            }
        } else if (layer_id < prev_layer) {
            // Went backwards (new sequence/generation step) - reset and track
            current_layer_.store(layer_id, std::memory_order_relaxed);
            // Prefetch next layer from the start of this new pass
            if (tensor_type == tensor_in_layer::ATTN_Q) {
                next_layer_id = layer_id + 1;
                return true;
            }
        } else if (layer_id == prev_layer) {
            // Same layer - track position for potential mid-layer prefetch
            // When we see ffn_gate, the layer is halfway done - good time to prefetch next
            if (tensor_type == tensor_in_layer::FFN_GATE) {
                next_layer_id = layer_id + 1;
                return true;
            }
        }

        return false;
    }

    // Reset tracker for new sequence/batch
    void reset() {
        current_layer_.store(-1, std::memory_order_relaxed);
    }

    // Get current layer being processed
    int current_layer() const {
        return current_layer_.load(std::memory_order_relaxed);
    }

    // Check if TG prefetch is enabled
    static bool is_enabled() {
        static int enabled = -1;
        if (enabled < 0) {
            const char * env = std::getenv("GGML_SYCL_TG_PREFETCH");
            enabled = env ? std::atoi(env) : 1;  // Enabled by default
        }
        return enabled != 0;
    }

  private:
    std::atomic<int> current_layer_{ -1 };

    static tensor_in_layer classify_tensor_in_layer(const char * name) {
        if (strstr(name, "attn_q.weight")) return tensor_in_layer::ATTN_Q;
        if (strstr(name, "attn_k.weight")) return tensor_in_layer::ATTN_K;
        if (strstr(name, "attn_v.weight")) return tensor_in_layer::ATTN_V;
        if (strstr(name, "attn_output.weight")) return tensor_in_layer::ATTN_OUTPUT;
        if (strstr(name, "ffn_gate.weight")) return tensor_in_layer::FFN_GATE;
        if (strstr(name, "ffn_up.weight")) return tensor_in_layer::FFN_UP;
        if (strstr(name, "ffn_down.weight")) return tensor_in_layer::FFN_DOWN;
        if (strstr(name, "attn_norm")) return tensor_in_layer::ATTN_NORM;
        if (strstr(name, "ffn_norm")) return tensor_in_layer::FFN_NORM;
        return tensor_in_layer::UNKNOWN;
    }
};

// Global prefetch tracker instance
inline layer_prefetch_tracker & get_prefetch_tracker() {
    static layer_prefetch_tracker tracker;
    return tracker;
}

// Build the tensor name for a specific layer's first weight (attn_q)
// Format: "blk.{layer_id}.attn_q.weight"
inline std::string build_layer_first_tensor_name(int layer_id) {
    return "blk." + std::to_string(layer_id) + ".attn_q.weight";
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_LAYER_PREFETCH_HPP
