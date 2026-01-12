# Tiered Memory Architecture for MoE Models

**Date:** 2026-01-12
**Status:** Design Complete
**Target:** SYCL backend, Intel Arc GPUs, MoE models (GPT-OSS-120B+)

## Problem Statement

Current limitations:
1. Intel Level Zero driver has ~11GB per-allocation limit for `malloc_host`
2. Large model buffers (12GB+) fail and fall back to unpinned CPU memory
3. Unpinned CPU memory requires explicit staging, adding latency
4. No intelligent tiering between VRAM, pinned host, and mmap

**Goal:** Use ALL available system RAM as GPU-accessible pinned memory via chunked allocation, with intelligent tiering for optimal MoE performance.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunking location | Unified cache pool | ggml expects contiguous buffers; cache already manages per-tensor |
| Chunk size | Fixed 8GB | Well under 11GB limit, simple, predictable |
| Chunk allocation | Lazy growth | Memory-efficient, fits existing budget system |
| Tensor placement | Fill VRAM first | Maximize fast memory utilization |
| Dense caching | Layer-wise prefetch | Matches execution flow, enables overlap |
| Expert caching | LRU + frequency | Adapts to routing patterns |
| Allocation within chunks | Bump allocator + grouping | Maximizes spatial locality |
| Host budget | Auto-detect + user % | Existing `--sycl-unified-cache-host-pct` |
| Graph support | Deprioritized | MoE routing is inherently dynamic |

## Architecture

### Memory Tier Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Device VRAM                                         │
│   Budget: GPU VRAM minus ~1GB reserve for activations       │
│   Contents:                                                 │
│     - Dense layers (attention Q/K/V/O, FFN if non-MoE)      │
│     - Hot MoE experts (LRU/frequency cached)                │
│   Eviction: LRU when VRAM pressure, evict to Tier 2         │
├─────────────────────────────────────────────────────────────┤
│ TIER 2: Pinned Host Memory (chunked pool)                   │
│   Budget: Configurable % of system RAM (default 90%)        │
│   Allocation: 8GB chunks via malloc_host, lazy growth       │
│   Contents:                                                 │
│     - All weights not in VRAM                               │
│     - GPU-accessible via PCIe (no copy needed for access)   │
│   Eviction: LRU when host budget exceeded, evict to Tier 3  │
├─────────────────────────────────────────────────────────────┤
│ TIER 3: mmap (file-backed, fallback only)                   │
│   Budget: Unlimited (disk-backed)                           │
│   Contents: Weights that don't fit in RAM                   │
│   Access: Page-fault on demand, very slow                   │
│   Used only for models larger than available RAM            │
└─────────────────────────────────────────────────────────────┘
```

### Chunked Pinned Memory Pool

Replaces direct `malloc_host` calls in `host_cache`:

```cpp
class pinned_chunk_pool {
    static constexpr size_t CHUNK_SIZE = 8ULL * 1024 * 1024 * 1024;  // 8GB

    struct chunk {
        void*  base;        // malloc_host result
        size_t size;        // CHUNK_SIZE
        size_t used;        // bump pointer offset
        std::vector<std::pair<size_t, size_t>> free_regions;  // offset, size
    };

    std::vector<chunk> chunks_;
    size_t budget_;
    size_t total_allocated_ = 0;
    sycl::queue& queue_;
    std::mutex mutex_;

public:
    pinned_chunk_pool(sycl::queue& queue, size_t budget)
        : queue_(queue), budget_(budget) {}

    ~pinned_chunk_pool() {
        for (auto& c : chunks_) {
            sycl::free(c.base, queue_);
        }
    }

    void* allocate(size_t size, size_t alignment = 64) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Round up size to alignment
        size = (size + alignment - 1) & ~(alignment - 1);

        // Try existing chunks
        for (auto& c : chunks_) {
            size_t aligned_offset = (c.used + alignment - 1) & ~(alignment - 1);
            if (aligned_offset + size <= c.size) {
                void* ptr = static_cast<char*>(c.base) + aligned_offset;
                c.used = aligned_offset + size;
                return ptr;
            }
        }

        // Need new chunk
        if (total_allocated_ + CHUNK_SIZE > budget_) {
            return nullptr;  // Over budget
        }

        if (!grow()) {
            return nullptr;  // Allocation failed
        }

        // Allocate from new chunk
        auto& c = chunks_.back();
        void* ptr = c.base;
        c.used = size;
        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        // Bump allocator - track free regions but don't compact
        // Chunk released when entirely free
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& c : chunks_) {
            char* base = static_cast<char*>(c.base);
            char* p = static_cast<char*>(ptr);
            if (p >= base && p < base + c.size) {
                c.free_regions.push_back({p - base, size});
                // Check if chunk entirely free
                size_t total_free = 0;
                for (auto& r : c.free_regions) total_free += r.second;
                if (total_free >= c.used) {
                    // Could release chunk here if desired
                }
                return;
            }
        }
    }

private:
    bool grow() {
        void* ptr = nullptr;
        try {
            ptr = sycl::malloc_host(CHUNK_SIZE, queue_);
        } catch (...) {
            return false;
        }
        if (!ptr) return false;

        chunks_.push_back({ptr, CHUNK_SIZE, 0, {}});
        total_allocated_ += CHUNK_SIZE;
        return true;
    }
};
```

### Tensor Loading & Placement

Route tensors to appropriate tiers during model load:

```cpp
// Placement decision
ggml_backend_buffer_type_t select_buffer_for_tensor(
    const char* tensor_name,
    size_t tensor_size,
    size_t vram_available,
    size_t pinned_available
) {
    // Try VRAM first for ANY tensor (dense or expert)
    if (tensor_size <= vram_available) {
        return ggml_backend_sycl_vram_buffer_type();
    }
    // Overflow to pinned host
    if (tensor_size <= pinned_available) {
        return ggml_backend_sycl_pinned_buffer_type();
    }
    // Last resort
    return ggml_backend_sycl_mmap_buffer_type();
}

// Loading sequence
void load_model_tiered(ggml_model& model, const char* path) {
    // 1. Parse GGUF, enumerate tensors
    auto tensors = enumerate_tensors(path);

    // 2. Sort by priority: dense first, then experts
    std::sort(tensors.begin(), tensors.end(), [](auto& a, auto& b) {
        bool a_dense = !is_moe_expert(a.name);
        bool b_dense = !is_moe_expert(b.name);
        if (a_dense != b_dense) return a_dense;  // Dense first
        return a.layer_id < b.layer_id;          // Then by layer
    });

    // 3. Allocate in priority order
    size_t vram_remaining = get_vram_budget();
    size_t pinned_remaining = get_pinned_budget();

    for (auto& t : tensors) {
        auto buft = select_buffer_for_tensor(
            t.name, t.size, vram_remaining, pinned_remaining);

        if (buft == vram_buffer_type()) {
            vram_remaining -= t.size;
        } else if (buft == pinned_buffer_type()) {
            pinned_remaining -= t.size;
        }

        allocate_tensor(t, buft);
    }
}
```

### Runtime Expert Caching

Dynamic VRAM ↔ Pinned Host swapping:

```cpp
class expert_cache {
    struct vram_entry {
        void* vram_ptr;
        size_t size;
        uint64_t last_access;
        uint32_t access_count;
    };

    std::unordered_map<expert_key, vram_entry> vram_entries_;
    pinned_chunk_pool& pinned_pool_;
    sycl::queue& compute_queue_;
    sycl::queue copy_queue_;  // Separate queue for async transfers

    size_t vram_budget_;
    size_t vram_used_ = 0;
    uint64_t time_ = 0;

public:
    void* get_expert(int layer, int expert_id, size_t size) {
        expert_key key{layer, expert_id};

        // Check VRAM cache
        auto it = vram_entries_.find(key);
        if (it != vram_entries_.end()) {
            it->second.last_access = time_++;
            it->second.access_count++;
            return it->second.vram_ptr;  // Fast path
        }

        // Cache miss - load from pinned host
        void* host_ptr = pinned_pool_.get(key);
        void* vram_ptr = allocate_vram(size);

        // Sync copy (or could batch with prefetch)
        compute_queue_.memcpy(vram_ptr, host_ptr, size).wait();

        vram_entries_[key] = {vram_ptr, size, time_++, 1};
        return vram_ptr;
    }

    // Prefetch experts predicted by router
    void prefetch(const std::vector<expert_key>& experts) {
        for (auto& key : experts) {
            if (vram_entries_.find(key) != vram_entries_.end()) {
                continue;  // Already cached
            }

            void* host_ptr = pinned_pool_.get(key);
            void* vram_ptr = allocate_vram(expert_size_);

            // Async copy - overlaps with compute
            copy_queue_.memcpy(vram_ptr, host_ptr, expert_size_);

            vram_entries_[key] = {vram_ptr, expert_size_, time_++, 0};
        }
        // Don't wait - let transfers overlap with compute
    }

private:
    void* allocate_vram(size_t size) {
        while (vram_used_ + size > vram_budget_) {
            evict_lowest_score();
        }
        void* ptr = sycl::malloc_device(size, compute_queue_);
        vram_used_ += size;
        return ptr;
    }

    void evict_lowest_score() {
        // Find entry with lowest score
        auto worst = vram_entries_.begin();
        float worst_score = compute_score(worst->second);

        for (auto it = vram_entries_.begin(); it != vram_entries_.end(); ++it) {
            float score = compute_score(it->second);
            if (score < worst_score) {
                worst = it;
                worst_score = score;
            }
        }

        // Free VRAM (pinned host still has data)
        sycl::free(worst->second.vram_ptr, compute_queue_);
        vram_used_ -= worst->second.size;
        vram_entries_.erase(worst);
    }

    float compute_score(const vram_entry& e) {
        // Combined LRU + frequency
        float recency = 1.0f / (time_ - e.last_access + 1);
        float frequency = static_cast<float>(e.access_count);
        return 0.3f * recency + 0.7f * frequency;
    }
};
```

### Dense Layer Scheduler (Large Model Support)

For models where dense layers exceed VRAM:

```cpp
class dense_layer_scheduler {
    void* vram_slot_[2];
    size_t slot_size_;
    int current_slot_ = 0;
    int layer_in_slot_[2] = {-1, -1};

    pinned_chunk_pool& pinned_pool_;
    sycl::queue& compute_queue_;
    sycl::queue copy_queue_;
    sycl::event pending_prefetch_;

public:
    dense_layer_scheduler(pinned_chunk_pool& pool, sycl::queue& q, size_t max_layer_size)
        : pinned_pool_(pool), compute_queue_(q), copy_queue_(q.get_context(), q.get_device())
    {
        slot_size_ = max_layer_size;
        vram_slot_[0] = sycl::malloc_device(slot_size_, compute_queue_);
        vram_slot_[1] = sycl::malloc_device(slot_size_, compute_queue_);
    }

    void* get_dense_layer(int layer_id, size_t size) {
        // Wait for any pending prefetch to this slot
        if (layer_in_slot_[current_slot_] != layer_id) {
            pending_prefetch_.wait();
        }

        // Verify correct layer is loaded
        if (layer_in_slot_[current_slot_] != layer_id) {
            void* host_ptr = pinned_pool_.get_dense(layer_id);
            compute_queue_.memcpy(vram_slot_[current_slot_], host_ptr, size).wait();
            layer_in_slot_[current_slot_] = layer_id;
        }

        return vram_slot_[current_slot_];
    }

    void prefetch_next(int next_layer_id, size_t size) {
        int prefetch_slot = 1 - current_slot_;

        if (layer_in_slot_[prefetch_slot] == next_layer_id) {
            return;  // Already there
        }

        void* host_ptr = pinned_pool_.get_dense(next_layer_id);
        pending_prefetch_ = copy_queue_.memcpy(
            vram_slot_[prefetch_slot], host_ptr, size);

        layer_in_slot_[prefetch_slot] = next_layer_id;
    }

    void advance_slot() {
        current_slot_ = 1 - current_slot_;
    }
};
```

## Implementation Plan

| Phase | Description | Files | Verification |
|-------|-------------|-------|--------------|
| **1** | Chunked pinned pool | `unified-cache.hpp/cpp` | Unit test allocation/deallocation |
| **2** | Replace host_cache malloc | `unified-cache.cpp` | Load GPT-OSS-120B, no CPU fallback warnings |
| **3** | Buffer type routing | `ggml-sycl.cpp`, `llama-model.cpp` | Verify tensor distribution |
| **4** | Expert cache with LRU | `unified-cache.hpp/cpp` | Monitor cache hit rates |
| **5** | Dense layer scheduler | `unified-cache.hpp/cpp` | Test with model > VRAM |

## Testing

### Phase 1-2 Verification
```bash
# Should see NO "using CPU memory" warnings
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-120b-Q4_0.gguf \
  -ngl 99 --lazy-moe -p "Hello" -n 10
```

### Cache Statistics
```bash
# Enable debug output
GGML_SYCL_DEBUG=1 ./build/bin/llama-cli ...
# Should show:
# - Chunk allocations (8GB each)
# - Expert cache hits/misses
# - VRAM utilization
```

## Future Work

1. **Frequency prediction**: Use router logits to predict upcoming experts
2. **Batch prefetch**: Prefetch all experts for a batch before execution
3. **Adaptive chunk sizing**: Adjust chunk size based on available RAM
4. **NUMA awareness**: Allocate chunks on GPU-local NUMA node
5. **Graph support**: Investigate static-address approaches for graph compatibility
