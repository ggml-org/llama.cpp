# Host Weight Streaming — Double-Buffered Layer Streaming

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Enable models larger than VRAM to run by streaming weight data from host memory to device using double-buffered layer-level DMA, overlapping compute and transfer for maximum throughput.

**Architecture:** Two device buffers (A and B), each sized to hold one transformer layer's weights. While GPU computes layer N from buffer A, DMA loads layer N+1 into buffer B asynchronously. Pointers are stable per-buffer, with layer transitions triggering buffer swaps. Graph replay and persistent TG are disabled when streaming is active since baked pointers are incompatible with buffer rotation.

**Tech Stack:** C++17, SYCL/Level Zero, llama.cpp unified cache, existing `layer_prefetch_tracker`

**Beads Issue:** llama.cpp-i7zx

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 3 | Layer stream manager module + async double-buffer prefetch |
| B | 2, 4 | Disable graphs/persistent TG + wire streaming into dispatch |

### Dependency Graph

```
Task 1 (Track A, no deps)       — Layer stream manager module
Task 2 (Track B, no deps)       — Disable graph replay + persistent TG for streaming
Task 3 (Track A, depends on 1)  — Wire async double-buffer prefetch into dispatch loop
Task 4 (depends on 1, 2)        — Wire streaming into pointer resolution + tiered stubs
Task 5 (depends on 3, 4)        — Integration test
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/layer-streaming.hpp` (NEW) | 1, 3 | Sequential (same track) |
| `ggml/src/ggml-sycl/layer-streaming.cpp` (NEW) | 1, 3 | Sequential (same track) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (~29465-29640, graph section) | 2 | None (isolated section) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (pointer resolution, stubs, dispatch) | 3, 4 | Sequential (3 before 4) |
| `ggml/src/ggml-sycl/common.hpp` (get_layout_ptr) | 4 | None |

---

## Background: What Exists vs. What's Missing

### What Exists (DO NOT re-implement)

1. **Unified cache with tiered storage**: `unified_cache` (device VRAM) + `host_cache` (pinned host memory). Weights evict from device → host when budget exceeded. (`unified-cache.hpp:508-968`)

2. **DMA infrastructure**: `copy_to_device()`, `copy_to_device_async()`, `stream_dma()` with staging buffers. (`unified-cache.hpp:826-741`)

3. **Layer transition detection**: `layer_prefetch_tracker` in `layer-prefetch.hpp:32-135`. Detects layer transitions by parsing tensor names (`blk.N.attn_q.weight`). Triggers at attn_q (layer start) and ffn_gate (layer midpoint). Already integrated into `ggml_backend_sycl_mul_mat` dispatch at line 19400.

4. **Tensor inventory**: `g_tensor_inventory` (`ggml-sycl.cpp:1995`) stores all weight tensor names and sizes. `extract_layer_id()` (`tensor-types.hpp:139`) parses layer IDs from tensor names. `g_model_exceeds_vram` flag set at line 2093 when model > VRAM budget.

5. **Layer prefetch system**: `queue_layer_prefetch()`, `await_layer()`, `release_layer()` with background worker thread and per-layer ready tracking (`unified-cache.hpp:612-958`). Currently pins cache entries — not suitable for host→device DMA.

6. **Graph replay**: `ggml_backend_sycl_graph_compute` (`ggml-sycl.cpp:29504`) records command graphs and replays them. `graph_refresh_input_tensors` (`ggml-sycl.cpp:25401`) updates input tensor data but NOT weight pointers (those are baked into kernel arguments).

7. **Persistent TG**: `should_use_persistent_tg()` (`ggml-sycl.cpp:29465`) gates persistent kernel dispatch. `extract_persistent_plan()` builds a plan with baked weight pointers.

### What's Missing (THIS plan implements)

1. **Layer stream manager**: A new module that manages two device buffers, builds a per-layer weight map from the tensor inventory, and performs synchronous/asynchronous DMA from host to device at layer granularity.

2. **Graph replay disable**: When `g_model_exceeds_vram` is true, graph replay must be disabled because weight pointers rotate between two buffers per token. Baked pointers in the graph would reference stale data.

3. **Persistent TG disable**: Same reason — the persistent plan records weight pointers at build time. With double-buffered streaming, these pointers change per token.

4. **Pointer resolution wiring**: The existing pointer resolution functions (`ggml_sycl_get_weight_layout_ptr`, `ggml_sycl_get_data_ptr_slow`, 5 tiered dispatch stubs) must query the layer stream manager for device pointers instead of returning host pointers.

5. **Async prefetch integration**: The existing `layer_prefetch_tracker` detects layer transitions but only logs them. It needs to trigger async DMA of the next layer into the alternate buffer.

---

## Performance Analysis

### Buffer Sizing (Mistral 7B Q4_0)

Per-layer weight tensors (32 layers):
- `attn_q.weight`: 4096×4096 Q4_0 = ~9.4 MB
- `attn_k.weight`: 4096×1024 Q4_0 = ~2.4 MB
- `attn_v.weight`: 4096×1024 Q4_0 = ~2.4 MB
- `attn_output.weight`: 4096×4096 Q4_0 = ~9.4 MB
- `ffn_gate.weight`: 4096×14336 Q4_0 = ~33.6 MB
- `ffn_up.weight`: 4096×14336 Q4_0 = ~33.6 MB
- `ffn_down.weight`: 14336×4096 Q4_0 = ~33.6 MB
- Norms (~32 KB, negligible)
- **Total per layer: ~125 MB**
- **Two buffers: ~250 MB**

Non-layer tensors (stay in VRAM permanently):
- `token_embd.weight`: ~77 MB
- `output.weight`: ~77 MB
- **Total: ~154 MB**

VRAM footprint for streaming: ~250 MB (buffers) + ~154 MB (embedding/output) + ~300 MB (KV cache, compute) ≈ 700 MB. Arc B580 has 12 GB, so this fits with large headroom.

### Throughput Estimates (Arc B580, PCIe Gen4 x16)

PCIe practical bandwidth: ~15-20 GB/s
Layer DMA time: 125 MB / 15 GB/s ≈ 8.3 ms
Layer compute time (TG batch=1): 13.2 ms / 32 layers ≈ 0.4 ms

- **Without streaming** (all VRAM): 13.2 ms/token → 76 tok/s
- **Single scratch (no overlap)**: 32 × (8.3 + 0.4) = 278 ms → ~3.6 tok/s
- **Double buffer (compute/DMA overlap)**: 8.3 + 31 × max(8.3, 0.4) = 266 ms → ~3.8 tok/s
- **PP512 with double buffer**: DMA hidden behind compute → much better overlap

Double-buffering benefits grow with larger batch sizes and PP mode where compute dominates.

---

## Task 1: Layer Stream Manager Module

**Track:** A
**Depends on:** None
**File scope:**
- Create: `ggml/src/ggml-sycl/layer-streaming.hpp`
- Create: `ggml/src/ggml-sycl/layer-streaming.cpp`
- (No CMakeLists.txt change needed — `file(GLOB GGML_SOURCES_SYCL "*.cpp")` at line 34 auto-includes new .cpp files, and `file(GLOB GGML_HEADERS_SYCL "*.hpp")` at line 26 auto-includes new .hpp files)

**Description:**

Self-contained module implementing double-buffered layer streaming. Manages two device buffers, builds a per-layer weight map from the tensor inventory, and provides synchronous and asynchronous DMA for loading layers from host to device. This is a new class that doesn't modify any existing files.

**Acceptance Criteria:**
- [ ] `layer_stream_manager` class compiles and links into ggml-sycl library
- [ ] `build_layer_map()` correctly groups tensors by layer ID from tensor inventory
- [ ] `allocate_buffers()` allocates two device buffers of `max_layer_size` bytes
- [ ] `ensure_layer(layer_id)` synchronously loads all host-resident weights for a layer into the correct buffer
- [ ] `get_weight_device_ptr(name)` returns the correct device pointer (buffer base + offset) for a loaded weight
- [ ] `prefetch_next_layer(layer_id)` starts async DMA on a copy queue
- [ ] `await_prefetch()` blocks until async DMA completes
- [ ] Non-layer tensors (embedding, output) are excluded from the layer map
- [ ] Buffer lifecycle (allocation, freeing) is correct — no leaks
- [ ] Build succeeds with zero warnings

**Implementation Guide:**

### Step 1: Create the header `layer-streaming.hpp`

```cpp
//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_LAYER_STREAMING_HPP
#define GGML_SYCL_LAYER_STREAMING_HPP

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"

namespace ggml_sycl {

// Double-buffered layer streaming manager.
// Manages two device buffers that alternate holding transformer layer weights.
// While GPU computes on buffer A (layer N), DMA loads layer N+1 into buffer B.
//
// Usage:
//   1. build_layer_map(inventory, count) — once after tensor inventory is set
//   2. allocate_buffers(queue) — once to create device buffers
//   3. ensure_layer(layer_id) — before each layer's first kernel
//   4. get_weight_device_ptr(name) — to get device pointer for a specific weight
//   5. prefetch_next_layer(layer_id+1) — after layer N's first kernel launches
//   6. shutdown() — cleanup

class layer_stream_manager {
  public:
    layer_stream_manager() = default;
    ~layer_stream_manager();

    // Non-copyable, non-movable (owns device memory)
    layer_stream_manager(const layer_stream_manager &)            = delete;
    layer_stream_manager & operator=(const layer_stream_manager &) = delete;

    // Build the layer weight map from tensor inventory.
    // inventory: array of (name, size) pairs
    // count: number of entries
    // Filters to only include "blk.N.*" tensors (layer tensors).
    void build_layer_map(const std::pair<std::string, size_t> * inventory, size_t count);

    // Allocate two device buffers sized to max_layer_size.
    // Returns false if allocation fails.
    // queue: the SYCL queue for device allocation.
    bool allocate_buffers(sycl::queue & queue);

    // Free device buffers and reset state.
    void shutdown();

    // Synchronously ensure a layer's weights are loaded into a device buffer.
    // If the layer is already loaded (from a previous prefetch or ensure call), this is a no-op.
    // If not loaded, performs synchronous DMA from host pointers to the buffer.
    // host_ptrs: map from tensor name to host pointer (from tensor->data or host cache).
    // Returns true if the layer is now loaded.
    bool ensure_layer(int layer_id, sycl::queue & queue);

    // Start async DMA of a layer into the alternate buffer.
    // Non-blocking — the DMA runs on copy_queue (if provided) or main queue.
    // Call await_prefetch() before accessing the layer's data.
    void prefetch_next_layer(int layer_id, sycl::queue & queue);

    // Block until the most recent async prefetch completes.
    void await_prefetch();

    // Get the device pointer for a specific weight tensor.
    // Returns nullptr if the weight's layer is not currently loaded.
    // The pointer is valid until the layer is evicted (overwritten by a new layer load).
    void * get_weight_device_ptr(const char * tensor_name) const;

    // Check if a layer is currently loaded in a buffer.
    bool is_layer_loaded(int layer_id) const;

    // Get which buffer index (0 or 1) a layer is in. Returns -1 if not loaded.
    int buffer_for_layer(int layer_id) const;

    // Check if the manager is active (has buffers allocated).
    bool is_active() const { return buffers_[0] != nullptr; }

    // Total bytes allocated for the two buffers.
    size_t allocated_bytes() const { return buffer_size_ * 2; }

    // Max layer size across all layers.
    size_t max_layer_size() const { return max_layer_size_; }

    // Number of layers in the model.
    int n_layers() const { return static_cast<int>(layers_.size()); }

    // Register the host pointer for a tensor (called during model load/cache setup).
    // This is needed because the tensor inventory only stores names and sizes,
    // not the actual host pointers. The pointers are registered as tensors are
    // encountered during inference.
    void register_host_ptr(const char * tensor_name, const void * host_ptr, size_t size);

  private:
    // Per-weight entry within a layer
    struct weight_entry {
        std::string  name;
        size_t       size            = 0;
        size_t       offset_in_layer = 0;  // Byte offset within layer buffer
        const void * host_ptr        = nullptr;  // Host-resident data pointer (registered later)
    };

    // Per-layer info
    struct layer_info {
        std::vector<weight_entry> weights;
        size_t                    total_size = 0;
    };

    // Layer data
    std::vector<layer_info>                       layers_;         // Indexed by layer_id
    size_t                                        max_layer_size_ = 0;
    std::unordered_map<std::string, std::pair<int, size_t>> name_to_location_;  // name -> (layer_id, weight_idx)

    // Double buffers
    void *  buffers_[2]       = {nullptr, nullptr};
    size_t  buffer_size_      = 0;
    int     loaded_layers_[2] = {-1, -1};  // Which layer is in each buffer (-1 = empty)

    // Async prefetch state
    int         prefetch_target_layer_ = -1;
    int         prefetch_buffer_       = -1;
    sycl::event prefetch_event_;
    bool        prefetch_pending_      = false;
    mutable std::mutex prefetch_mutex_;

    // Host pointer registration
    mutable std::mutex host_ptr_mutex_;

    // Internal helpers
    int  pick_buffer_for_layer(int layer_id) const;
    bool load_layer_sync(int layer_id, int buffer_idx, sycl::queue & queue);
};

// Global accessor — returns the singleton layer stream manager for a device.
// Creates on first access. Thread-safe.
layer_stream_manager & get_layer_stream_manager(int device_id);

// Free function API for use from ggml-sycl.cpp
bool layer_streaming_active(int device_id);
void * layer_streaming_get_weight_ptr(int device_id, const char * name);
void layer_streaming_ensure_layer(int device_id, int layer_id, sycl::queue & queue);
void layer_streaming_prefetch_next(int device_id, int layer_id, sycl::queue & queue);
void layer_streaming_await_prefetch(int device_id);
void layer_streaming_register_host_ptr(int device_id, const char * name, const void * ptr, size_t size);

}  // namespace ggml_sycl

#endif  // GGML_SYCL_LAYER_STREAMING_HPP
```

### Step 2: Create the implementation `layer-streaming.cpp`

```cpp
//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "layer-streaming.hpp"
#include "tensor-types.hpp"

#include "ggml-impl.h"

#include <algorithm>
#include <cstring>

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

    for (int i = 0; i < 2; i++) {
        buffers_[i] = sycl::malloc_device(buffer_size_, queue);
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

    GGML_LOG_INFO("[LAYER-STREAM] Allocated 2 × %.1f MB device buffers (%.1f MB total)\n",
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

    // Note: we can't free here because we don't have a queue reference.
    // The buffers are freed when the SYCL context is destroyed.
    // In practice, shutdown() is called from the backend destructor
    // which happens during process exit.
    buffers_[0] = nullptr;
    buffers_[1] = nullptr;
    buffer_size_ = 0;
    loaded_layers_[0] = -1;
    loaded_layers_[1] = -1;
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

    // Pick the buffer not currently in use
    int buf = -1;
    for (int i = 0; i < 2; i++) {
        if (loaded_layers_[i] != loaded_layers_[0] || i == 1) {
            // Pick the buffer with the older layer (or any free buffer)
            buf = (loaded_layers_[0] == -1) ? 0 : (loaded_layers_[1] == -1) ? 1 : (layer_id % 2);
            break;
        }
    }
    if (buf < 0) buf = layer_id % 2;

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
    auto it = g_layer_managers.find(device_id);
    if (it == g_layer_managers.end()) {
        g_layer_managers[device_id] = std::make_unique<layer_stream_manager>();
        return *g_layer_managers[device_id];
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
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto it = g_layer_managers.find(device_id);
    if (it != g_layer_managers.end()) {
        it->second->ensure_layer(layer_id, queue);
    }
}

void layer_streaming_prefetch_next(int device_id, int layer_id, sycl::queue & queue) {
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto it = g_layer_managers.find(device_id);
    if (it != g_layer_managers.end()) {
        it->second->prefetch_next_layer(layer_id, queue);
    }
}

void layer_streaming_await_prefetch(int device_id) {
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto it = g_layer_managers.find(device_id);
    if (it != g_layer_managers.end()) {
        it->second->await_prefetch();
    }
}

void layer_streaming_register_host_ptr(int device_id, const char * name, const void * ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_layer_managers_mutex);
    auto it = g_layer_managers.find(device_id);
    if (it != g_layer_managers.end()) {
        it->second->register_host_ptr(name, ptr, size);
    }
}

}  // namespace ggml_sycl
```

### Step 3: Build and verify compilation

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

The new files will be auto-included via the CMakeLists.txt glob. Verify zero warnings, zero errors.

Quick smoke test (no streaming expected — module is not wired in yet):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, ...` (correct output, no regression)

**Commit:**
```bash
git add ggml/src/ggml-sycl/layer-streaming.hpp ggml/src/ggml-sycl/layer-streaming.cpp
git commit -m "sycl: add layer stream manager for double-buffered weight streaming"
```

**Notes for implementer:**
- The `register_host_ptr()` mechanism is necessary because the tensor inventory (`g_tensor_inventory`) only stores names and sizes, not actual pointers. Host pointers are registered lazily as tensors are encountered during inference (from `tensor->data` or from the unified cache's host entries).
- The `shutdown()` method intentionally does NOT call `sycl::free()` because the queue reference isn't stored. The SYCL runtime will free device memory when the context is destroyed. If we need explicit free, store a `sycl::queue` reference at allocation time.
- The global accessor `get_layer_stream_manager()` uses a per-device map because multi-device setups have independent memory spaces.
- The `pick_buffer_for_layer()` uses simple parity (layer_id % 2). This works because consecutive layers alternate buffers: layer 0→buf 0, layer 1→buf 1, layer 2→buf 0, etc.
- The existing `file(GLOB ...)` in CMakeLists.txt at line 34 automatically picks up new .cpp files. No CMakeLists change needed.

---

## Task 2: Disable Graph Replay + Persistent TG for Streaming Mode

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines ~29465-29500 in `should_use_persistent_tg`, lines ~29635 persistent dispatch, lines ~29790-29860 graph decision)

**Description:**

When `g_model_exceeds_vram` is true, graph replay must be disabled because weight pointers rotate between two device buffers per token (the graph has baked pointers that reference the wrong buffer after rotation). Similarly, persistent TG must be disabled because its plan records weight pointers at build time that become stale when buffers rotate.

This is a small, isolated change to the graph_compute entry point. No overlap with Task 1's new files.

**Acceptance Criteria:**
- [ ] `should_use_persistent_tg()` returns false when `g_model_exceeds_vram` is true
- [ ] Graph replay is disabled (falls through to `compute_impl()`) when `g_model_exceeds_vram` is true
- [ ] No regression when `g_model_exceeds_vram` is false (normal models use graphs and persistent TG as before)
- [ ] Build succeeds
- [ ] Normal model perf: PP512 >= 1200, TG128 >= 68

**Implementation Guide:**

### Step 1: Gate persistent TG on streaming mode

In `ggml-sycl.cpp`, find `should_use_persistent_tg()` at line 29465. Add after the environment variable gate (line 29468):

```cpp
static bool should_use_persistent_tg(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // 1. Environment variable gate
    if (!ggml_sycl::env_persistent_tg_enabled()) {
        return false;
    }

    // 1b. Model exceeds VRAM — persistent TG records weight pointers that become
    // stale during double-buffered layer streaming. Disable to use per-op dispatch.
    if (g_model_exceeds_vram.load(std::memory_order_acquire)) {
        GGML_SYCL_DEBUG("[PERSISTENT-TG] Disabled: model exceeds VRAM, weight streaming active\n");
        return false;
    }

    if (!cgraph || cgraph->n_nodes == 0) {
        return false;
    }
    // ... rest of function unchanged
```

### Step 2: Gate graph replay on streaming mode

In `ggml_backend_sycl_graph_compute()`, find the graph support check around line 29790. Add before the existing `graph_support` check:

```cpp
    // Disable graph replay when model exceeds VRAM — weight streaming rotates
    // pointers between two device buffers, incompatible with baked graph pointers.
    if (g_model_exceeds_vram.load(std::memory_order_acquire)) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Disabled: model exceeds VRAM, weight streaming active\n");
        compute_impl();
        record_completion(false);
        return GGML_STATUS_SUCCESS;
    }
```

Add this BEFORE the existing `if (!graph_support)` check (around line 29790), so it short-circuits before any graph logic.

### Step 3: Build and test

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

Normal model (no streaming, graphs should still work):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68 (graphs and persistent TG active)

Low VRAM (streaming mode — graphs disabled, per-op dispatch):
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Still produces output (even if wrong — streaming isn't wired yet). Confirms graph disable doesn't crash.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: disable graph replay and persistent TG when model exceeds VRAM"
```

**Notes for implementer:**
- The `g_model_exceeds_vram` check is an `atomic<bool>` loaded with `memory_order_acquire`. This is consistent with how it's checked elsewhere (e.g., line 19400).
- The graph disable MUST come before any graph recording/replay logic. Place it at the very start of the graph section, before `if (!graph_support)`.
- This change alone won't fix the streaming bug — it just removes obstacles. The actual streaming is wired in Tasks 3 and 4.
- Test that PP512 perf is unchanged for normal models. The `g_model_exceeds_vram` flag is false when the model fits in VRAM, so the new checks are never hit.

---

## Task 3: Wire Async Double-Buffer Prefetch into Dispatch Loop

**Track:** A
**Depends on:** Task 1 (needs `layer_stream_manager` class)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines ~2030-2100 in `set_tensor_inventory`, lines ~19390-19410 in layer prefetch section)

**Description:**

Initialize the layer stream manager during model load (tensor inventory setup), and wire the layer transition detection into the dispatch loop to trigger async DMA of the next layer. This uses the existing `layer_prefetch_tracker` which already detects layer transitions at `ggml_backend_sycl_mul_mat` line 19400.

**Acceptance Criteria:**
- [ ] Layer stream manager is initialized from tensor inventory when `g_model_exceeds_vram` is true
- [ ] Two device buffers are allocated on the correct device
- [ ] Layer transitions trigger `prefetch_next_layer()` on the manager
- [ ] `ensure_layer()` is called before each layer's first weight access
- [ ] Host pointers are registered as weights are encountered during inference
- [ ] Build succeeds with zero warnings

**Implementation Guide:**

### Step 1: Initialize layer stream manager in `set_tensor_inventory`

In `ggml-sycl.cpp`, find `ggml_backend_sycl_set_tensor_inventory()` at line 2030. After the `g_model_exceeds_vram` is set (line 2093), add initialization of the layer stream manager:

```cpp
    g_model_exceeds_vram.store(model_exceeds_vram, std::memory_order_release);
    g_tiered_enabled.store(ggml_sycl::unified_cache_enabled(), std::memory_order_release);

    // Initialize double-buffered layer streaming when model exceeds VRAM
    if (model_exceeds_vram) {
        auto & mgr = ggml_sycl::get_layer_stream_manager(ctx->device);
        mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
        sycl::queue & q = ggml_sycl_get_device(ctx->device).default_queue();
        if (mgr.allocate_buffers(q)) {
            GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled: %d layers, 2 × %.1f MB buffers\n",
                          mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
        } else {
            GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed\n");
        }
    }
```

Add include at top of file (near line 118):
```cpp
#include "ggml-sycl/layer-streaming.hpp"
```

### Step 2: Wire layer transitions to trigger prefetch

In `ggml-sycl.cpp`, find the layer prefetch tracker integration at line 19400. Replace the existing log-only code:

```cpp
    if (src0->name && src1->ne[1] == 1 && g_model_exceeds_vram.load(std::memory_order_acquire) &&
        ggml_sycl::layer_prefetch_tracker::is_enabled()) {
        int next_layer_id = -1;
        if (ggml_sycl::get_prefetch_tracker().record_access(src0->name, next_layer_id)) {
            // Layer transition detected - prefetch opportunity identified
            // For now, just log the opportunity. Full prefetch requires cache_id lookup.
            GGML_SYCL_DEBUG("[PREFETCH] Layer transition at %s -> prefetch layer %d\n",
                            src0->name, next_layer_id);
        }
    }
```

With:

```cpp
    if (src0->name && src1->ne[1] == 1 && g_model_exceeds_vram.load(std::memory_order_acquire) &&
        ggml_sycl::layer_prefetch_tracker::is_enabled()) {
        int next_layer_id = -1;
        if (ggml_sycl::get_prefetch_tracker().record_access(src0->name, next_layer_id)) {
            if (ggml_sycl::layer_streaming_active(ctx.device) && next_layer_id >= 0) {
                // Layer transition — start async DMA of next layer
                sycl::queue & q = ggml_sycl_get_device(ctx.device).default_queue();
                ggml_sycl::layer_streaming_prefetch_next(ctx.device, next_layer_id, q);
                GGML_SYCL_DEBUG("[LAYER-STREAM] Prefetching layer %d (triggered by %s)\n",
                                next_layer_id, src0->name);
            }
        }

        // Ensure current layer is loaded before accessing weights
        int current_layer = ggml_sycl::extract_layer_id(src0->name);
        if (current_layer >= 0 && ggml_sycl::layer_streaming_active(ctx.device)) {
            sycl::queue & q = ggml_sycl_get_device(ctx.device).default_queue();
            ggml_sycl::layer_streaming_ensure_layer(ctx.device, current_layer, q);
        }
    }
```

### Step 3: Register host pointers during dispatch

In the `get_data_ptr_slow` function (line ~2339), add host pointer registration when a weight tensor's host pointer is resolved. Find the section where `tensor->data` is accessed for weight tensors and add:

```cpp
    // Register host pointer with layer stream manager for DMA
    if (is_weight && tensor->name && tensor->data &&
        g_model_exceeds_vram.load(std::memory_order_acquire)) {
        ggml_sycl::layer_streaming_register_host_ptr(
            device, tensor->name, tensor->data, ggml_nbytes(tensor));
    }
```

Also add registration in the cache lookup path when a host-resident cached pointer is found, so the manager knows where to DMA from.

### Step 4: Build and test

```bash
ninja -C build -j $(nproc)
```

Normal model (no streaming):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Correct output, no streaming messages in stderr.

Low VRAM (streaming active):
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0 2>&1 | head -50
```
Expected: Layer streaming messages in stderr. Output may still be wrong (pointer resolution not yet wired — that's Task 4).

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: wire layer stream manager into dispatch and prefetch"
```

**Notes for implementer:**
- The host pointer registration is lazy — pointers are registered as tensors are encountered during the first inference pass. The first token will be slower (sync DMA for all layers) but subsequent tokens benefit from prefetch.
- The `ensure_layer()` call before each weight access is the synchronization point. If the prefetch completed, it's a no-op. If not, it blocks until DMA finishes.
- Don't add the `ensure_layer()` call in the PP (prompt processing) path — PP uses batch sizes > 1 where `src1->ne[1] == 1` is false. The gate at line 19400 handles this.
- For PP mode, the weights go through the normal pointer resolution which will be updated in Task 4. PP with streaming will be slower (no prefetch) but correct.

---

## Task 4: Wire Streaming into Pointer Resolution + Tiered Dispatch Stubs

**Track:** B (converges with Track A)
**Depends on:** Task 1 (layer_stream_manager API), Task 2 (graphs disabled)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines ~2339-2430 get_data_ptr_slow, ~5078 get_weight_layout_ptr, ~11946 ~15142 ~19428 ~20855 ~21823 tiered stubs, ~19335 MMVQ fast-path, ~25965 get_tensor_ptr_fast)
- Modify: `ggml/src/ggml-sycl/common.hpp` (lines ~1810 get_layout_ptr host fallback)

**Description:**

The existing pointer resolution functions return host pointers for weights that were evicted to host memory. This task wires them to the layer stream manager instead: when a weight is host-resident and its layer is loaded by the stream manager, return the device pointer from the buffer. Also replaces all 5 NOP tiered dispatch stubs with actual streaming calls.

**Acceptance Criteria:**
- [ ] `ggml_sycl_get_weight_layout_ptr()` returns device pointer from layer buffer when host-resident
- [ ] `ggml_sycl_get_data_ptr_slow()` returns device pointer from layer buffer when host-resident
- [ ] All 5 tiered dispatch stubs replaced with streaming-aware helper
- [ ] MMVQ fast-path ensures weight is on device before layout check
- [ ] `get_tensor_ptr_fast` lambda (unified kernel path) checks layer stream manager
- [ ] `ggml_sycl_get_layout_ptr()` host fallback uses streaming
- [ ] Correct deterministic output with `GGML_SYCL_VRAM_BUDGET_PCT=40`
- [ ] No regression for models that fit in VRAM
- [ ] Build succeeds with zero warnings

**Implementation Guide:**

### Step 1: Create streaming-aware helper function

Add near line 11940 in `ggml-sycl.cpp` (before the first tiered dispatch stub):

```cpp
// Resolve weight pointer for streaming dispatch.
// When model exceeds VRAM, checks layer stream manager first,
// then falls back to tiered cache lookup + on-demand streaming.
static void ggml_sycl_ensure_weight_on_device(const ggml_tensor * src0, int device) {
    if (!src0 || !src0->name || !g_model_exceeds_vram.load(std::memory_order_acquire)) {
        return;
    }

    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra) {
        return;
    }

    // Fast path: check layer stream manager
    if (ggml_sycl::layer_streaming_active(device)) {
        void * streamed = ggml_sycl::layer_streaming_get_weight_ptr(device, src0->name);
        if (streamed) {
            extra->data_device[device] = streamed;
            return;
        }
        // Layer not loaded — ensure it's loaded now
        int layer_id = ggml_sycl::extract_layer_id(src0->name);
        if (layer_id >= 0) {
            sycl::queue & q = ggml_sycl_get_device(device).default_queue();
            ggml_sycl::layer_streaming_ensure_layer(device, layer_id, q);
            streamed = ggml_sycl::layer_streaming_get_weight_ptr(device, src0->name);
            if (streamed) {
                extra->data_device[device] = streamed;
                return;
            }
        }
    }

    // Fallback: tiered cache lookup (for non-layer tensors like embedding, output)
    ggml_sycl::memory_tier tier;
    bool                   in_inventory = false;
    void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
    if (!cached_ptr || !in_inventory) {
        return;
    }

    if (tier == ggml_sycl::memory_tier::VRAM) {
        extra->data_device[device] = cached_ptr;
    }
    // For non-layer host-resident tensors, the normal pointer resolution
    // path in get_data_ptr_slow handles the copy via unified cache.
}
```

### Step 2: Replace all 5 tiered dispatch stubs

Replace each NOP stub of the form:
```cpp
    if (src0->name && g_model_exceeds_vram.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool                   in_inventory = false;
        void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            // Future: use cached_ptr for tiered dispatch
            GGML_LOG_DEBUG("[SYCL] Tiered cache hit for %s (tier=%d)\n", ...);
        } else if (in_inventory) {
            GGML_LOG_DEBUG("[SYCL] Tiered: tensor %s in inventory, pending cache\n", ...);
        }
    }
```

With:
```cpp
    ggml_sycl_ensure_weight_on_device(src0, ctx.device);
```

At all 5 locations:
1. `ggml_sycl_get_rows` (~line 11946)
2. `ggml_sycl_mul_mat_batched_sycl` (~line 15142)
3. `ggml_backend_sycl_mul_mat` (~line 19428)
4. `ggml_sycl_op_mul_mat_moe_q` (~line 20855)
5. `ggml_sycl_op_mul_mat_moe_f16` (~line 21823)

### Step 3: Fix `ggml_sycl_get_weight_layout_ptr()` host-resident handling

At line ~5078, replace the host-resident block:

```cpp
    if (!had_exception && resolved != GGML_LAYOUT_AOS && result.host_resident) {
        GGML_LOG_WARN(...)
        ...
        return nullptr;
    }
```

With:

```cpp
    if (!had_exception && result.host_resident) {
        // Check layer stream manager for device pointer
        if (ggml_sycl::layer_streaming_active(device)) {
            void * streamed = ggml_sycl::layer_streaming_get_weight_ptr(device, tensor->name);
            if (streamed) {
                GGML_LOG_DEBUG("[LAYER-STREAM] Using streamed ptr for %s (layout=%d)\n",
                               tensor->name, (int) resolved);
                return streamed;
            }
        }
        // Not in layer buffer — fall back to AOS
        if (resolved != GGML_LAYOUT_AOS) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Host-resident layout=%d for %s not streamable, falling back to AoS\n",
                          (int) resolved, tensor->name);
            if (cache_key.valid) {
                ggml_sycl_force_layout_choice(cache_key, device, GGML_LAYOUT_AOS, tensor->name);
            }
            return nullptr;
        }
    }
```

### Step 4: Fix `ggml_sycl_get_data_ptr_slow()` for host-resident weights

At line ~2400, after the existing unified cache block, add:

```cpp
            // Check layer stream manager for host-resident weight
            if (is_weight && ggml_sycl::layer_streaming_active(device) && tensor->name) {
                void * streamed = ggml_sycl::layer_streaming_get_weight_ptr(device, tensor->name);
                if (streamed) {
                    GGML_LOG_DEBUG("get_data_ptr_slow: %s from layer stream buffer\n", tensor->name);
                    return streamed;
                }
            }
```

### Step 5: Fix MMVQ fast-path

At line ~19335, just before the layout checks, add:

```cpp
            // Ensure weight is on device before checking layout
            ggml_sycl_ensure_weight_on_device(src0, ctx.device);
```

### Step 6: Fix `get_tensor_ptr_fast` lambda

At line ~25965, add layer stream check:

```cpp
    auto get_tensor_ptr_fast = [&](const ggml_tensor * tensor) -> void * {
        if (!tensor) return nullptr;
        if (tensor->extra) {
            auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
            if (extra->data_device[ctx.device] != nullptr) {
                return extra->data_device[ctx.device];
            }
        }
        // Check layer stream manager for host-resident weights
        if (ggml_sycl_tensor_is_weight(tensor) &&
            g_model_exceeds_vram.load(std::memory_order_acquire) &&
            ggml_sycl::layer_streaming_active(ctx.device) && tensor->name) {
            void * streamed = ggml_sycl::layer_streaming_get_weight_ptr(ctx.device, tensor->name);
            if (streamed) return streamed;
        }
        if (tensor->data != nullptr) return tensor->data;
        return ggml_sycl_get_data_ptr(tensor, ctx.device);
    };
```

### Step 7: Fix `ggml_sycl_get_layout_ptr()` in common.hpp

At line ~1810, update the host_weights fallback:

```cpp
        if (host_weights) {
            // Check layer stream manager first
            if (ggml_sycl::layer_streaming_active(device) && tensor->name) {
                void * streamed = ggml_sycl::layer_streaming_get_weight_ptr(device, tensor->name);
                if (streamed) {
                    ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK);
                    return streamed;
                }
            }
            ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK);
            return ggml_sycl_get_data_ptr(tensor, device);
        }
```

Add include at top of common.hpp:
```cpp
#include "layer-streaming.hpp"
```

### Step 8: Build and test

```bash
ninja -C build -j $(nproc)
```

Correctness (normal VRAM):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

Low VRAM streaming:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Correct output (same sequence). Will be slow (~3-4 tok/s) but correct.

Performance regression:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/common.hpp
git commit -m "sycl: wire layer streaming into pointer resolution and tiered dispatch"
```

**Notes for implementer:**
- The `ggml_sycl_ensure_weight_on_device()` function checks layer streaming first (fast O(1) hash lookup), then falls back to tiered cache lookup. This is efficient because the hash lookup is ~nanoseconds while `get_pointer_type()` is ~1ms.
- For non-layer tensors (embedding, output), the layer stream manager returns nullptr (they're not in the layer map). These tensors go through the normal unified cache path which already handles host→device copies.
- The MMVQ fast-path forces AOS dispatch for streamed weights because SOA layout isn't available (the SOA reordering data is in the unified cache's host entry, not in the layer buffer). This means streamed weights use the slower AOS MMVQ path. This is acceptable for v1.
- For PP mode (batch > 1), `ensure_weight_on_device()` is called at each tiered stub. Since PP doesn't go through the TG prefetch path (gated by `src1->ne[1] == 1`), PP loads layers on-demand (synchronous). This is slower but correct. PP with streaming benefits from the double-buffer's stable pointers.
- Be careful with the `#include "layer-streaming.hpp"` in common.hpp — it must not create circular dependencies. The layer-streaming module only depends on SYCL and ggml basics, not on common.hpp.

---

## Task 5: Integration Test and Verification

**Track:** — (convergence point)
**Depends on:** Task 3, Task 4
**File scope:**
- No code changes (testing only)

**Description:**

Comprehensive verification that host weight streaming works correctly across all dispatch paths and doesn't regress performance for normal models.

**Acceptance Criteria:**
- [ ] Mistral Q4_0 at 40% VRAM produces correct deterministic output
- [ ] Mistral Q4_0 at 100% VRAM (default) produces correct output with no perf regression
- [ ] GPT-OSS 20B (exceeds VRAM) loads and produces output (may be slow)
- [ ] Layer streaming messages visible in debug output
- [ ] No crashes, no OOM, no hangs

**Implementation Guide:**

### Test 1: Low VRAM correctness

```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

### Test 2: No performance regression

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

Wait 30+ seconds between runs to avoid thermal throttling on Arc B580.

### Test 3: GPT-OSS 20B (model exceeding VRAM)

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 'Hello, my name is' -n 10 --seed 42 --temp 0
```
Expected: Coherent output (may take a minute+ per token due to streaming).
If it hangs, that may be a pre-existing MoE issue — document and move on.

### Test 4: Very low budget

```bash
GGML_SYCL_VRAM_BUDGET_PCT=10 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 'Hello' -n 1 --seed 42 --temp 0
```
Expected: Either produces output or fails gracefully (no crash, no hang).

### Test 5: Debug output verification

```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 'Hello' -n 1 --seed 42 --temp 0 2>/tmp/stream_debug.txt
```
Check `/tmp/stream_debug.txt` for:
- `[LAYER-STREAM] Built layer map:` — manager initialization
- `[LAYER-STREAM] Allocated 2 ×` — buffer allocation
- `[LAYER-STREAM] Loaded layer N into buffer` — DMA activity
- `[LAYER-STREAM] Prefetching layer N` — async prefetch
- `[PERSISTENT-TG] Disabled: model exceeds VRAM` — persistent TG disabled
- `[SYCL-GRAPH] Disabled: model exceeds VRAM` — graph replay disabled

**No code commit — verification only.**

**Notes for implementer:**
- Wait 30+ seconds between benchmark runs to avoid thermal throttling on Arc B580. A single TG128=2.41 result is NOT a regression — re-run after GPU cools.
- The GPT-OSS 20B test may fail due to pre-existing MoE issues (llama.cpp-a73u). Focus on the Mistral tests as the primary verification.
- Check stderr for streaming debug messages even without `GGML_SYCL_DEBUG=1` — the `GGML_LOG_INFO` messages from initialization are always visible.
- If Test 1 produces wrong output, the issue is likely in pointer resolution (Task 4). Check if `get_weight_device_ptr()` returns valid pointers by adding a temporary `fprintf(stderr, ...)` in the function.

---

## Verification After Each Task

```bash
# Build
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Quick correctness
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Performance (no regression)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

## Key Design Decisions

1. **Layer-granularity over tensor-granularity**: Loading all weights for a layer at once (~125 MB) into a buffer means each weight has a stable pointer until the buffer is overwritten two layers later. This is compatible with per-op dispatch.

2. **AOS-only for streamed weights**: SOA layout data exists in the unified cache's host entries, but the layer stream manager copies raw tensor data (AOS). SOA reordering during streaming is too expensive. MMVQ falls back to AOS dispatch for streamed weights.

3. **Disable graphs + persistent TG**: Both bake weight pointers at record/build time. With double-buffered streaming, pointers rotate per token. Disabling these is the simplest correct approach for v1.

4. **Lazy host pointer registration**: The tensor inventory stores names/sizes but not pointers. Host pointers are registered as tensors are first encountered during inference. First-token latency is higher (sync DMA for all layers) but subsequent tokens benefit from prefetch.

5. **Two equal-sized buffers**: Both buffers are sized to `max_layer_size`. This handles models with variable layer sizes (e.g., MoE where some layers have expert tensors). The parity-based buffer selection (layer_id % 2) ensures consecutive layers alternate buffers.
