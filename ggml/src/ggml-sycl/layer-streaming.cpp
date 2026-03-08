//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "layer-streaming.hpp"

#include "alloc-registry.hpp"
#include "common.hpp"
#include "ggml-impl.h"
#include "tensor-types.hpp"

#include <algorithm>

namespace ggml_sycl {

layer_stream_manager::~layer_stream_manager() {
    shutdown();
}

void layer_stream_manager::build_layer_map(
    const std::pair<std::string, size_t> * inventory, size_t count) {

    layers_.clear();
    name_to_location_.clear();
    max_layer_size_ = 0;

    // First pass: find max layer_id to size the vector
    int max_layer = -1;
    for (size_t i = 0; i < count; i++) {
        int layer_id = extract_layer_id(inventory[i].first.c_str());
        if (layer_id >= 0) {
            max_layer = std::max(max_layer, layer_id);
        }
    }

    if (max_layer < 0) {
        GGML_LOG_WARN("[LAYER-STREAM] No layer tensors found in inventory\n");
        return;
    }

    layers_.resize(max_layer + 1);

    // Second pass: populate layer entries
    for (size_t i = 0; i < count; i++) {
        const auto & [name, size] = inventory[i];
        int layer_id = extract_layer_id(name.c_str());
        if (layer_id < 0 || layer_id > max_layer) {
            continue;  // Non-layer tensor (embedding, output, etc.)
        }

        auto & layer = layers_[layer_id];
        weight_entry entry;
        entry.name            = name;
        entry.size            = size;
        entry.offset_in_layer = layer.total_size;  // Pack sequentially
        entry.host_ptr        = nullptr;  // Registered later

        size_t weight_idx = layer.weights.size();
        layer.weights.push_back(std::move(entry));
        layer.total_size += size;

        name_to_location_[name] = {layer_id, weight_idx};
    }

    // Calculate max layer size
    for (const auto & layer : layers_) {
        max_layer_size_ = std::max(max_layer_size_, layer.total_size);
    }

    GGML_LOG_INFO("[LAYER-STREAM] Built layer map: %d layers, max_layer_size=%.1f MB, total_entries=%zu\n",
                  static_cast<int>(layers_.size()),
                  max_layer_size_ / (1024.0 * 1024.0),
                  name_to_location_.size());
}

bool layer_stream_manager::allocate_buffers(sycl::queue & queue) {
    if (max_layer_size_ == 0) {
        GGML_LOG_ERROR("[LAYER-STREAM] Cannot allocate buffers: no layer map built\n");
        return false;
    }

    // Round up to 2MB alignment for efficient DMA
    const size_t align = 2 * 1024 * 1024;
    buffer_size_ = ((max_layer_size_ + align - 1) / align) * align;

    // Store context for sycl::free() in shutdown()
    ctx_ = queue.get_context();

    for (int i = 0; i < 2; i++) {
        buffers_[i] = sycl::malloc_device(buffer_size_, queue);
        if (buffers_[i]) {
            alloc_registry::instance().register_alloc(buffers_[i], buffer_size_,
                                                      ggml_sycl_get_device_id_from_queue(queue), alloc_type::DEVICE);
        }
        if (!buffers_[i]) {
            GGML_LOG_ERROR("[LAYER-STREAM] Failed to allocate buffer %d (%.1f MB)\n",
                           i, buffer_size_ / (1024.0 * 1024.0));
            // Clean up buffer 0 if buffer 1 failed
            if (i == 1 && buffers_[0]) {
                sycl::free(buffers_[0], queue);
                buffers_[0] = nullptr;
            }
            return false;
        }
        loaded_layers_[i] = -1;
    }

    GGML_LOG_INFO("[LAYER-STREAM] Allocated 2 x %.1f MB device buffers (%.1f MB total)\n",
                  buffer_size_ / (1024.0 * 1024.0),
                  (buffer_size_ * 2) / (1024.0 * 1024.0));
    return true;
}

void layer_stream_manager::shutdown() {
    // Wait for any pending prefetch
    if (prefetch_pending_) {
        try {
            prefetch_event_.wait();
        } catch (...) {}
        prefetch_pending_ = false;
    }

    for (int i = 0; i < 2; i++) {
        if (buffers_[i]) {
            sycl::free(buffers_[i], ctx_);
            buffers_[i] = nullptr;
        }
        loaded_layers_[i] = -1;
    }
    buffer_size_ = 0;
}

void layer_stream_manager::register_host_ptr(
    const char * tensor_name, const void * host_ptr, size_t size) {
    if (!tensor_name || !host_ptr) return;

    std::lock_guard<std::mutex> lock(host_ptr_mutex_);
    auto it = name_to_location_.find(tensor_name);
    if (it == name_to_location_.end()) return;

    auto [layer_id, weight_idx] = it->second;
    if (layer_id < 0 || layer_id >= static_cast<int>(layers_.size())) return;
    if (weight_idx >= layers_[layer_id].weights.size()) return;

    auto & entry = layers_[layer_id].weights[weight_idx];
    entry.host_ptr = host_ptr;
    (void) size;  // Size already known from inventory
}

int layer_stream_manager::pick_buffer_for_layer(int layer_id) const {
    // Check if already loaded
    for (int i = 0; i < 2; i++) {
        if (loaded_layers_[i] == layer_id) return i;
    }
    // Pick the buffer NOT currently holding the adjacent layer
    // Simple heuristic: alternate by layer parity
    return layer_id % 2;
}

bool layer_stream_manager::load_layer_sync(int layer_id, int buffer_idx, sycl::queue & queue) {
    if (layer_id < 0 || layer_id >= static_cast<int>(layers_.size())) return false;
    if (buffer_idx < 0 || buffer_idx > 1) return false;
    if (!buffers_[buffer_idx]) return false;

    const auto & layer = layers_[layer_id];
    if (layer.weights.empty()) return false;

    char * dst_base = static_cast<char *>(buffers_[buffer_idx]);
    int    copied   = 0;
    int    skipped  = 0;

    for (const auto & w : layer.weights) {
        if (!w.host_ptr) {
            GGML_LOG_WARN("[LAYER-STREAM] No host pointer for %s (layer %d), skipping\n",
                          w.name.c_str(), layer_id);
            skipped++;
            continue;
        }
        queue.memcpy(dst_base + w.offset_in_layer, w.host_ptr, w.size);
        copied++;
    }

    // Wait for all copies to complete (synchronous)
    queue.wait();

    loaded_layers_[buffer_idx] = layer_id;

    GGML_LOG_DEBUG("[LAYER-STREAM] Loaded layer %d into buffer %d: %d tensors (%.1f MB), %d skipped\n",
                   layer_id, buffer_idx, copied,
                   layer.total_size / (1024.0 * 1024.0), skipped);
    return true;
}

bool layer_stream_manager::ensure_layer(int layer_id, sycl::queue & queue) {
    if (!is_active()) return false;

    // Already loaded?
    if (is_layer_loaded(layer_id)) return true;

    // Check if there's a pending prefetch for this layer
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        if (prefetch_pending_ && prefetch_target_layer_ == layer_id) {
            // Wait for the prefetch to complete
            try {
                prefetch_event_.wait();
            } catch (const sycl::exception & e) {
                GGML_LOG_ERROR("[LAYER-STREAM] Prefetch wait failed: %s\n", e.what());
            }
            prefetch_pending_ = false;
            loaded_layers_[prefetch_buffer_] = layer_id;
            return true;
        }
    }

    // Need to load synchronously
    int buf = pick_buffer_for_layer(layer_id);
    return load_layer_sync(layer_id, buf, queue);
}

void layer_stream_manager::prefetch_next_layer(int layer_id, sycl::queue & queue) {
    if (!is_active()) return;
    if (layer_id < 0 || layer_id >= static_cast<int>(layers_.size())) return;
    if (is_layer_loaded(layer_id)) return;  // Already loaded

    std::lock_guard<std::mutex> lock(prefetch_mutex_);

    // Cancel any existing prefetch that hasn't been consumed
    if (prefetch_pending_ && prefetch_target_layer_ != layer_id) {
        try {
            prefetch_event_.wait();
        } catch (...) {}
        prefetch_pending_ = false;
    }

    if (prefetch_pending_ && prefetch_target_layer_ == layer_id) {
        return;  // Already prefetching this layer
    }

    // Pick the buffer not currently in use (prefer empty, then alternate by parity)
    int buf = (loaded_layers_[0] == -1) ? 0 : (loaded_layers_[1] == -1) ? 1 : (layer_id % 2);

    // Don't evict a layer that's currently being computed
    // The "other" buffer is the safe one to overwrite
    if (loaded_layers_[buf] >= 0 && loaded_layers_[buf] == layer_id - 1) {
        // Buffer holds the immediately preceding layer (might still be computing)
        // Use the other buffer instead
        buf = 1 - buf;
    }

    const auto & layer = layers_[layer_id];
    if (layer.weights.empty()) return;

    char * dst_base = static_cast<char *>(buffers_[buf]);

    // Submit async copies
    // In-order queue: last event implies all prior copies complete
    sycl::event last_event;
    for (const auto & w : layer.weights) {
        if (!w.host_ptr) continue;
        last_event = queue.memcpy(dst_base + w.offset_in_layer, w.host_ptr, w.size);
    }

    prefetch_target_layer_ = layer_id;
    prefetch_buffer_       = buf;
    prefetch_event_        = last_event;
    prefetch_pending_      = true;
}

void layer_stream_manager::await_prefetch() {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    if (!prefetch_pending_) return;

    try {
        prefetch_event_.wait();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[LAYER-STREAM] Prefetch await failed: %s\n", e.what());
    }

    loaded_layers_[prefetch_buffer_] = prefetch_target_layer_;
    prefetch_pending_ = false;
}

void * layer_stream_manager::get_weight_device_ptr(const char * tensor_name) const {
    if (!tensor_name || !is_active()) return nullptr;

    auto it = name_to_location_.find(tensor_name);
    if (it == name_to_location_.end()) return nullptr;

    auto [layer_id, weight_idx] = it->second;

    // Find which buffer holds this layer
    int buf = buffer_for_layer(layer_id);
    if (buf < 0) return nullptr;  // Layer not loaded

    const auto & entry = layers_[layer_id].weights[weight_idx];
    return static_cast<char *>(buffers_[buf]) + entry.offset_in_layer;
}

bool layer_stream_manager::is_layer_loaded(int layer_id) const {
    return loaded_layers_[0] == layer_id || loaded_layers_[1] == layer_id;
}

int layer_stream_manager::buffer_for_layer(int layer_id) const {
    if (loaded_layers_[0] == layer_id) return 0;
    if (loaded_layers_[1] == layer_id) return 1;
    return -1;
}

// --- Global accessors ---

static std::unordered_map<int, std::unique_ptr<layer_stream_manager>> g_layer_managers;
static std::mutex g_layer_managers_mutex;

layer_stream_manager & get_layer_stream_manager(int device_id) {
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto [it, inserted] = g_layer_managers.try_emplace(device_id);
    if (inserted) {
        it->second = std::make_unique<layer_stream_manager>();
    }
    return *it->second;
}

bool layer_streaming_active(int device_id) {
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto it = g_layer_managers.find(device_id);
    return it != g_layer_managers.end() && it->second->is_active();
}

void * layer_streaming_get_weight_ptr(int device_id, const char * name) {
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto it = g_layer_managers.find(device_id);
    if (it == g_layer_managers.end()) return nullptr;
    return it->second->get_weight_device_ptr(name);
}

void layer_streaming_ensure_layer(int device_id, int layer_id, sycl::queue & queue) {
    layer_stream_manager * mgr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
        auto it = g_layer_managers.find(device_id);
        if (it == g_layer_managers.end()) return;
        mgr = it->second.get();
    }
    mgr->ensure_layer(layer_id, queue);
}

void layer_streaming_prefetch_next(int device_id, int layer_id, sycl::queue & queue) {
    layer_stream_manager * mgr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
        auto it = g_layer_managers.find(device_id);
        if (it == g_layer_managers.end()) return;
        mgr = it->second.get();
    }
    mgr->prefetch_next_layer(layer_id, queue);
}

void layer_streaming_await_prefetch(int device_id) {
    layer_stream_manager * mgr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
        auto it = g_layer_managers.find(device_id);
        if (it == g_layer_managers.end()) return;
        mgr = it->second.get();
    }
    mgr->await_prefetch();
}

void layer_streaming_register_host_ptr(int device_id, const char * name, const void * ptr, size_t size) {
    layer_stream_manager * mgr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
        auto it = g_layer_managers.find(device_id);
        if (it == g_layer_managers.end()) return;
        mgr = it->second.get();
    }
    mgr->register_host_ptr(name, ptr, size);
}

}  // namespace ggml_sycl
