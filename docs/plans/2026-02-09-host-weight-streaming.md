# Host Weight Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Wire the existing DMA streaming infrastructure into kernel dispatch so that host-resident weights are automatically streamed to device before kernel execution, enabling models larger than VRAM to run (slowly but correctly).

**Architecture:** When a weight is needed by a GPU kernel but is host-resident (evicted from device cache or never loaded), the dispatch path must: (1) check if the weight is cached on device, (2) if not, use the unified cache's `copy_to_device_async()` or `ensure_cached_layout()` to bring it to device, (3) pass the device pointer to the kernel. This happens transparently in the existing pointer resolution functions.

**Tech Stack:** C++17, SYCL/Level Zero, llama.cpp unified cache

**Beads Issue:** llama.cpp-i7zx

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 3, 5 | Core pointer resolution + unified kernel path + integration test |
| B | 2, 4 | Legacy dispatch path + MMVQ fast-path |

### Dependency Graph

```
Task 1 (Track A, no deps)     — Core streaming in pointer resolution
Task 2 (Track B, no deps)     — Legacy dispatch hook
Task 3 (Track A, depends on 1) — Unified kernel plan-build streaming
Task 4 (Track B, depends on 2) — MMVQ fast-path streaming
Task 5 (Track A, depends on 1,2) — Integration test
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/common.hpp` (get_layout_ptr) | 1 | None (single task) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (get_data_ptr_slow + tiered stubs) | 1, 2 | **Sequential** — Task 2 depends on Task 1's changes |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (plan_build section ~27380) | 3 | None (depends on 1) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (TG fast-path ~19320) | 4 | None (depends on 2) |
| `ggml/src/ggml-sycl/unified-cache.hpp` | 1 | None |
| `ggml/src/ggml-sycl/unified-cache.cpp` | 1 | None |

---

## Background: What Exists vs. What's Missing

### What Exists (DO NOT re-implement)

1. **Unified cache with tiered storage**: `unified_cache` (device VRAM) + `host_cache` (pinned host memory). Weights evict from device → host when budget exceeded. (`unified-cache.hpp:508-968`)

2. **DMA streaming infrastructure**:
   - `unified_cache::copy_to_device(dst, src, size)` — sync host→device copy via staging buffer (`unified-cache.hpp:826`)
   - `unified_cache::copy_to_device_async(dst, src, size, deps)` — async copy with event dependencies (`unified-cache.hpp:827`)
   - `unified_cache::stream_dma(src, total_bytes, slice_bytes, buffer_count, slice_fn, ctx, deps)` — pipelined multi-buffer DMA streaming (`unified-cache.hpp:734-741`)
   - `dma_staging_buffers` — device-resident staging buffer pool (`unified-cache.hpp:715-722`)
   - Staging buffer in cache: `staging_` / `staging_size_` (default 64MB, `unified-cache.hpp:879-881`)

3. **Layer prefetch system**: Background worker thread (`prefetch_worker_loop`), `queue_layer_prefetch()`, `wait_layer_prefetch()`, per-layer pinning (`unified-cache.hpp:613-958`). This is for proactive prefetch, NOT for on-demand streaming.

4. **Tensor inventory + tier tracking**: `g_tensor_inventory`, `get_cached_tensor_ptr()`, `ggml_sycl_get_cached_tensor_ptr_for()` — these know which weights are VRAM/host/mmap and return the cached pointer + tier. (`ggml-sycl.cpp:2234-2330`)

5. **`ensure_cached_layout()` with host fallback**: When device allocation fails, creates host-resident entry and sets `result.host_resident = true`. (`unified-cache.cpp:2077, 2419, 2504`)

6. **`g_model_exceeds_vram` flag**: Set during tensor inventory when model size > VRAM budget. Controls whether tiered dispatch checks are active. (`ggml-sycl.cpp:1999, 2093`)

### What's Missing (THIS plan implements)

1. **On-demand device streaming in `ggml_sycl_get_weight_layout_ptr()`**: When `result.host_resident == true`, instead of returning nullptr and falling back to AOS, stream the data to a temporary device buffer and return the device pointer.

2. **On-demand device streaming in `ggml_sycl_get_data_ptr_slow()`**: The fallback path for AOS weights. When cache lookup returns a host pointer, copy to device staging and return device pointer.

3. **Tiered dispatch hookup in legacy mul_mat**: The `"// Future: use cached_ptr"` stubs at lines 11946, 15142, 19428, 20855, 21823 need to actually use the cached device pointer when available, and trigger on-demand streaming when not.

4. **Unified kernel plan-build streaming**: When building a persistent TG plan, `get_tensor_ptr_fast()` returns `extra->data_device[device]` which may be stale/null after eviction. Need to resolve through streaming-aware path.

5. **MMVQ fast-path awareness**: The TG fast-path at line 19320 checks `extra->optimized_feature.is_reordered()` but doesn't handle the case where the reordered layout was evicted to host.

---

## Task 1: Core Streaming in Pointer Resolution

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 5078-5090 in `ggml_sycl_get_weight_layout_ptr`, lines 2339-2430 in `ggml_sycl_get_data_ptr_slow`)
- Modify: `ggml/src/ggml-sycl/common.hpp` (lines 1790-1825 in `ggml_sycl_get_layout_ptr`)
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (add `stream_to_device()` public method)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (implement `stream_to_device()`)

**Description:**

The core problem: when `ensure_cached_layout()` returns `host_resident=true`, `ggml_sycl_get_weight_layout_ptr()` returns nullptr, and the entire fallback chain eventually returns a host pointer to the GPU kernel. The fix is to add an on-demand streaming path that copies host-resident data to a reusable device staging buffer.

**Acceptance Criteria:**
- [ ] When a weight is host-resident, `ggml_sycl_get_weight_layout_ptr()` streams it to device and returns a device pointer
- [ ] When `ggml_sycl_get_data_ptr_slow()` encounters a host-resident cached weight, it streams to device
- [ ] A reusable "streaming scratch" device buffer is managed by the unified cache (sized to max weight tensor)
- [ ] Streaming is synchronous (simple first implementation — async prefetch is WI-3 layer prefetch, separate work)
- [ ] No regression for models that fit in VRAM (streaming code is only reached when weight is actually host-resident)
- [ ] Build succeeds with zero warnings
- [ ] `llama-completion` with Mistral Q4_0 at `GGML_SYCL_VRAM_BUDGET_PCT=40` produces correct deterministic output

**Implementation Guide:**

### Step 1: Add `stream_weight_to_device()` to unified_cache

Add a new public method to `unified_cache` that:
1. Takes a host pointer, size, and optional layout info
2. Uses the existing staging buffer (`staging_` / `staging_size_`) for the transfer
3. Returns a device pointer to a reusable "streaming scratch" buffer

In `ggml/src/ggml-sycl/unified-cache.hpp`, add after `copy_to_device_async` (line 827):

```cpp
    // Stream a host-resident weight to a reusable device scratch buffer.
    // Returns device pointer to the scratch buffer containing the weight data.
    // The scratch buffer is reused across calls (only valid until next stream call).
    // Thread-safe: protected by staging_mutex_.
    void * stream_weight_to_device(const void * host_ptr, size_t size);
```

Add private member after `dma_staging_mutex_` (line 888):

```cpp
    // Reusable device scratch buffer for on-demand weight streaming.
    // Sized to hold the largest weight tensor that needs streaming.
    void *     stream_scratch_         = nullptr;
    size_t     stream_scratch_size_    = 0;
    std::mutex stream_scratch_mutex_;
```

In `unified-cache.cpp`, implement:

```cpp
void * unified_cache::stream_weight_to_device(const void * host_ptr, size_t size) {
    if (!host_ptr || size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(stream_scratch_mutex_);

    // Grow scratch buffer if needed
    if (size > stream_scratch_size_) {
        if (stream_scratch_) {
            sycl::free(stream_scratch_, queue_);
            saturating_sub_used(stream_scratch_size_);
        }
        // Round up to 16MB alignment for efficient reuse
        const size_t aligned_size = ((size + (16 * 1024 * 1024 - 1)) / (16 * 1024 * 1024)) * (16 * 1024 * 1024);
        stream_scratch_ = sycl::malloc_device(aligned_size, queue_);
        if (!stream_scratch_) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to allocate streaming scratch buffer (%zu MB)\n",
                           aligned_size / (1024 * 1024));
            stream_scratch_size_ = 0;
            return nullptr;
        }
        stream_scratch_size_ = aligned_size;
        used_.fetch_add(aligned_size, std::memory_order_relaxed);
        GGML_LOG_INFO("[UNIFIED-CACHE] Allocated streaming scratch buffer: %.1f MB\n",
                      aligned_size / (1024.0f * 1024.0f));
    }

    // Copy host data to device scratch via staging buffer
    copy_to_device(stream_scratch_, host_ptr, size);

    return stream_scratch_;
}
```

Also add cleanup in the destructor — find the existing destructor and add:
```cpp
    if (stream_scratch_) {
        sycl::free(stream_scratch_, queue_);
        stream_scratch_ = nullptr;
        stream_scratch_size_ = 0;
    }
```

Also add a free function for external use:

In `unified-cache.hpp` after `unified_cache_is_budget_exceeded` (~line 1052):
```cpp
// Stream a host-resident weight to a reusable device scratch buffer.
// Returns device pointer valid until the next stream call on the same device.
void * unified_cache_stream_weight_to_device(int device_id, const void * host_ptr, size_t size);
```

In `unified-cache.cpp`, implement:
```cpp
void * unified_cache_stream_weight_to_device(int device_id, const void * host_ptr, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->stream_weight_to_device(host_ptr, size);
}
```

### Step 2: Fix `ggml_sycl_get_weight_layout_ptr()` host-resident handling

In `ggml-sycl.cpp`, find the block at line 5078:

```cpp
    if (!had_exception && resolved != GGML_LAYOUT_AOS && result.host_resident) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Host-resident layout=%d for %s is not usable; falling back to AoS\n",
                      (int) resolved, tensor->name);
        if (cache_key.valid) {
            ggml_sycl_force_layout_choice(cache_key, device, GGML_LAYOUT_AOS, tensor->name);
        }
        return nullptr;
    }
```

Replace with:

```cpp
    if (!had_exception && resolved != GGML_LAYOUT_AOS && result.host_resident) {
        // Non-AOS layout is host-resident — stream to device scratch buffer
        void * device_ptr = ggml_sycl::unified_cache_stream_weight_to_device(
            device, result.device_ptr, result.size);
        if (device_ptr) {
            GGML_LOG_DEBUG("[UNIFIED-CACHE] Streamed host-resident layout=%d for %s to device (%zu bytes)\n",
                           (int) resolved, tensor->name, result.size);
            // Update the layout info to point to the device scratch
            if (auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra)) {
                ggml_sycl_update_layout_from_cache(extra, tensor, device, resolved, device_ptr, result.size,
                                                   result.xmx_info, result.onednn_pack_m);
            }
            return device_ptr;
        }
        // Streaming failed — fall back to AOS
        GGML_LOG_WARN("[UNIFIED-CACHE] Host-resident layout=%d for %s: streaming failed, falling back to AoS\n",
                      (int) resolved, tensor->name);
        if (cache_key.valid) {
            ggml_sycl_force_layout_choice(cache_key, device, GGML_LAYOUT_AOS, tensor->name);
        }
        return nullptr;
    }
```

### Step 3: Fix `ggml_sycl_get_layout_ptr()` host weight final fallback

In `common.hpp`, the function `ggml_sycl_get_layout_ptr()` at line 1810 has:

```cpp
        if (host_weights) {
            ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK);
            return ggml_sycl_get_data_ptr(tensor, device);
        }
```

This returns `ggml_sycl_get_data_ptr()` which may return a host pointer. The fix is to ensure this falls through to the streaming path. The data_ptr_slow function (Step 4 below) will handle streaming, so this line is actually OK — but we need to verify that `ggml_sycl_get_data_ptr_slow` handles the case correctly.

### Step 4: Fix `ggml_sycl_get_data_ptr_slow()` for host-resident weights

In `ggml-sycl.cpp` at line 2400, after the cache lookup:

```cpp
            if (ggml_sycl::unified_cache_enabled() && is_weight) {
                if (auto * cache = ggml_sycl::get_unified_cache_for_device(device)) {
                    ggml_sycl_cache_id cache_key = ggml_backend_sycl_get_weight_cache_key(tensor, device);
                    if (cache_key.valid) {
                        void * cached = cache->get_or_wait(cache_key, GGML_LAYOUT_AOS);
                        if (cached) {
                            ...
                            return cached;
                        }
                    }
                }
            }
```

After this block, if `cached` was null but the weight IS in the host cache, we need to stream it. Add after the closing `}` of the unified cache block:

```cpp
            // Weight not in device cache — try streaming from host
            if (ggml_sycl::unified_cache_enabled() && is_weight) {
                void * streamed = ggml_sycl::unified_cache_stream_weight_to_device(
                    device, tensor->data, ggml_nbytes(tensor));
                if (streamed) {
                    GGML_LOG_DEBUG("ggml_sycl_get_data_ptr_slow: tensor=%s, device=%d, streamed host->device %p -> %p (%zu bytes)\n",
                                    tensor->name, device, tensor->data, streamed, ggml_nbytes(tensor));
                    return streamed;
                }
            }
```

### Step 5: Build and test

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

Correctness test (normal VRAM):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

Low VRAM test (host streaming):
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Same correct output (may be slow due to per-tensor streaming)

Performance test (no regression for models that fit):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp \
        ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/common.hpp
git commit -m "sycl: add on-demand weight streaming from host to device"
```

**Notes for implementer:**
- The streaming scratch buffer is **reusable** — only one weight needs to be in device memory at a time during inference. Each kernel reads its weight, then the next kernel can reuse the same buffer.
- This is a **synchronous** implementation. Async double-buffered streaming (WI-3 layer prefetch) is future work.
- Be careful with `used_` accounting — the scratch buffer counts against the cache budget. Don't double-count.
- The `copy_to_device()` method already uses the staging buffer for efficient host→device copy. Don't reinvent this.
- Wait for 30+ seconds between benchmark runs to avoid thermal throttling on Arc B580.

---

## Task 2: Legacy Dispatch Tiered Streaming

**Track:** B
**Depends on:** None (uses same `unified_cache_stream_weight_to_device` but can start with basic streaming inline)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (tiered dispatch stubs at lines ~11946, ~15142, ~19428, ~20855, ~21823)

**Description:**

The legacy dispatch path (non-unified kernel) has 5 tiered dispatch stubs that currently log and do nothing. When `g_model_exceeds_vram` is true and `get_cached_tensor_ptr()` returns a tier other than VRAM, the weight data may be in host memory. The stubs need to actually use the cached pointer or trigger streaming.

**Important context:** These stubs run on EVERY mul_mat/get_rows when `g_model_exceeds_vram` is true. Most weights will be in the device cache already (only evicted weights need streaming). The common case is a cache hit returning a device pointer — so the hot path must be fast.

**Acceptance Criteria:**
- [ ] All 5 tiered dispatch stubs are implemented (not just commented)
- [ ] When tier is VRAM and alloc is device, the cached pointer replaces `tensor->data` via `extra->data_device`
- [ ] When tier is HOST_PINNED or MMAP, streaming to device happens automatically
- [ ] No performance regression when `g_model_exceeds_vram` is false (stubs are gated, not reached)
- [ ] Build succeeds
- [ ] Correct output with `GGML_SYCL_VRAM_BUDGET_PCT=40`

**Implementation Guide:**

### Step 1: Create a helper function for tiered dispatch

Add near line 11940 (before the first tiered dispatch stub in `ggml_sycl_get_rows`):

```cpp
// Resolve weight pointer for tiered dispatch.
// When model exceeds VRAM, weights may be in host memory.
// This function ensures we have a valid device pointer.
static void ggml_sycl_ensure_weight_on_device(const ggml_tensor * src0, int device) {
    if (!src0->name || !g_model_exceeds_vram.load(std::memory_order_acquire)) {
        return;
    }
    ggml_sycl::memory_tier tier;
    bool                   in_inventory = false;
    void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);

    if (!cached_ptr || !in_inventory) {
        return;  // Not a tracked tensor or not in cache
    }

    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra) {
        return;
    }

    if (tier == ggml_sycl::memory_tier::VRAM) {
        // Weight is on device — update cached pointer for fast access
        sycl::queue &    q     = ggml_sycl_get_device(device).default_queue();
        sycl::usm::alloc alloc = sycl::get_pointer_type(cached_ptr, q.get_context());
        if (alloc == sycl::usm::alloc::device) {
            extra->data_device[device] = cached_ptr;
        }
    } else {
        // Weight is in host memory — stream to device
        void * device_ptr = ggml_sycl::unified_cache_stream_weight_to_device(
            device, cached_ptr, ggml_nbytes(src0));
        if (device_ptr) {
            extra->data_device[device] = device_ptr;
            GGML_LOG_DEBUG("[SYCL] Streamed %s from host to device (%zu bytes)\n",
                           src0->name, ggml_nbytes(src0));
        }
    }
}
```

### Step 2: Replace all 5 tiered dispatch stubs

Replace each stub of the form:
```cpp
    if (src0->name && g_model_exceeds_vram.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool                   in_inventory = false;
        void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            // Future: use cached_ptr for tiered dispatch
            GGML_LOG_DEBUG("[SYCL] Tiered cache hit for %s (tier=%d)\n", src0->name, static_cast<int>(tier));
        } else if (in_inventory) {
            GGML_LOG_DEBUG("[SYCL] Tiered: tensor %s in inventory, pending cache\n", src0->name);
        }
    }
```

With a single call:
```cpp
    ggml_sycl_ensure_weight_on_device(src0, ctx.device);
```

Do this at all 5 locations:
1. `ggml_sycl_get_rows` (~line 11946)
2. `ggml_sycl_mul_mat_batched_sycl` (~line 15142)
3. `ggml_backend_sycl_mul_mat` — the main mul_mat dispatcher (~line 19428)
4. `ggml_sycl_op_mul_mat_moe_q` (~line 20855)
5. `ggml_sycl_op_mul_mat_moe_f16` (~line 21823)

**NOTE:** The `get_pointer_type()` call in the helper is expensive (~1ms). This is acceptable for host-streaming mode since streaming itself takes much longer. For models that fit in VRAM, the entire function is skipped due to the `g_model_exceeds_vram` gate.

### Step 3: Build and test

Same test commands as Task 1.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: implement tiered weight streaming in legacy dispatch path"
```

**Notes for implementer:**
- The helper uses `get_pointer_type()` which is known to be slow (~1ms). This is ONLY called when `g_model_exceeds_vram` is true, so it doesn't affect normal models.
- The `extra->data_device[device]` update is important — it caches the device pointer so that `get_tensor_ptr_fast()` in the unified kernel path picks it up.
- Don't worry about the streaming scratch being reused between layers — the synchronous copy completes before the kernel launches. Each weight is fully loaded before use.

---

## Task 3: Unified Kernel Plan-Build Streaming

**Track:** A
**Depends on:** Task 1 (needs `unified_cache_stream_weight_to_device`)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines ~27380-27420 in plan_build, lines ~25965-25980 in `get_tensor_ptr_fast`)

**Description:**

The unified kernel's persistent TG plan building uses `get_tensor_ptr_fast()` which checks `extra->data_device[device]` then `tensor->data`. When weights are evicted, `extra->data_device[device]` may be null and `tensor->data` may be a stale pointer. The plan build needs to go through the streaming-aware path.

**Acceptance Criteria:**
- [ ] `get_tensor_ptr_fast` lambda uses streaming-aware pointer resolution for weight tensors
- [ ] Plan build succeeds with `GGML_SYCL_VRAM_BUDGET_PCT=40`
- [ ] Persistent TG kernel produces correct output with host-resident weights
- [ ] No regression for models that fit in VRAM

**Implementation Guide:**

### Step 1: Make `get_tensor_ptr_fast` streaming-aware

In `ggml-sycl.cpp` at line 25965, the `get_tensor_ptr_fast` lambda currently does:

```cpp
    auto get_tensor_ptr_fast = [&](const ggml_tensor * tensor) -> void * {
        if (!tensor) {
            return nullptr;
        }
        if (tensor->extra) {
            auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
            if (extra->data_device[ctx.device] != nullptr) {
                return extra->data_device[ctx.device];
            }
        }
        if (tensor->data != nullptr) {
            return tensor->data;
        }
        return ggml_sycl_get_data_ptr(tensor, ctx.device);
    };
```

Replace with:

```cpp
    auto get_tensor_ptr_fast = [&](const ggml_tensor * tensor) -> void * {
        if (!tensor) {
            return nullptr;
        }
        if (tensor->extra) {
            auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
            if (extra->data_device[ctx.device] != nullptr) {
                return extra->data_device[ctx.device];
            }
        }
        // For weight tensors when model exceeds VRAM, ensure device-resident data
        if (ggml_sycl_tensor_is_weight(tensor) &&
            g_model_exceeds_vram.load(std::memory_order_acquire)) {
            ggml_sycl_ensure_weight_on_device(tensor, ctx.device);
            if (tensor->extra) {
                auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
                if (extra->data_device[ctx.device] != nullptr) {
                    return extra->data_device[ctx.device];
                }
            }
        }
        if (tensor->data != nullptr) {
            return tensor->data;
        }
        return ggml_sycl_get_data_ptr(tensor, ctx.device);
    };
```

### Step 2: Also fix the weight pointer resolution in plan build (line ~27395)

The weight pointer resolution for quantized weights at line 27414-27418:

```cpp
                    if (!weight) {
                        weight = get_tensor_ptr_view_fast(weight_tensor);
                    }
```

`get_tensor_ptr_view_fast` calls `get_tensor_ptr_fast` which now has streaming support. This should work automatically with the Step 1 change.

### Step 3: Build and test

```bash
ninja -C build -j $(nproc)
```

Test with low VRAM using both legacy and unified kernel:
```bash
# Default (unified kernel for PP, MMVQ for TG):
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Force legacy kernel:
GGML_SYCL_VRAM_BUDGET_PCT=40 GGML_SYCL_UNIFIED_FORCE_LEGACY=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Both produce correct output.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add streaming support to unified kernel plan building"
```

**Notes for implementer:**
- The persistent TG plan records weight pointers once during plan build. If the streaming scratch buffer is reused, the pointer recorded in the plan will be overwritten by the next weight. This means persistent TG can't work with host-streaming as-is. The plan-build code needs to either:
  a) Detect host-streaming mode and disable persistent TG (fall back to per-token dispatch), OR
  b) Record that weights need streaming and re-stream before each replay
- Option (a) is simpler and correct. Add a check: if any weight was streamed during plan build, set a flag that disables persistent TG for this session. This is a performance hit (TG will be slow) but correct.
- Implement option (a): add `bool weights_streamed = false;` before the plan-build loop. Set to true if `ggml_sycl_ensure_weight_on_device` was called. After plan build, if `weights_streamed`, cancel the plan and fall back to legacy dispatch.

---

## Task 4: MMVQ Fast-Path Streaming

**Track:** B
**Depends on:** Task 2 (needs `ggml_sycl_ensure_weight_on_device`)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines ~19320-19370, TG fast-path)

**Description:**

The MMVQ fast-path checks `extra->optimized_feature.is_reordered()` to decide SOA vs AOS dispatch. When weights are host-resident, the reordered (SOA) layout may not be on device. Need to ensure the weight is on device before entering the MMVQ path.

**Acceptance Criteria:**
- [ ] MMVQ fast-path calls `ggml_sycl_ensure_weight_on_device` before checking layout
- [ ] SOA layout correctly detected even after streaming
- [ ] Falls back to AOS MMVQ gracefully if SOA layout can't be streamed
- [ ] TG produces correct output with `GGML_SYCL_VRAM_BUDGET_PCT=40`

**Implementation Guide:**

### Step 1: Add streaming call to TG fast-path

In `ggml-sycl.cpp` at line ~19335, just before the existing layout checks, add:

```cpp
        if (!tg_fast_disabled && src1->ne[1] == 1 && !fast_split
            && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32
            && dst->type == GGML_TYPE_F32
            && !(g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1)) {

            // Ensure weight is on device before checking layout
            ggml_sycl_ensure_weight_on_device(src0, ctx.device);

            ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
            ...
```

That's it — `ggml_sycl_ensure_weight_on_device` already updates `extra->data_device[device]`, and the existing MMVQ code goes through `ggml_sycl_op_mul_mat` which uses `ggml_sycl_get_layout_ptr_for()` to get the actual weight pointer.

### Step 2: Build and test

```bash
ninja -C build -j $(nproc)
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add weight streaming to MMVQ fast-path"
```

**Notes for implementer:**
- The MMVQ fast-path's `has_soa_reorder` check looks at `extra->optimized_feature.is_reordered()`. This flag is set during `set_tensor` when the SOA layout was materialized. If SOA materialization failed (host-resident), this flag might not be set. In that case, the code falls through to AOS MMVQ or DMMV, which is correct behavior.
- The streaming only ensures the AOS data is on device. SOA re-ordering would require additional work (stream + re-order) which is beyond scope. AOS MMVQ is slower but correct.

---

## Task 5: Integration Test and Verification

**Track:** A
**Depends on:** Task 1, Task 2
**File scope:**
- No code changes (testing only)

**Description:**

Run comprehensive verification to ensure host weight streaming works correctly across all dispatch paths and doesn't regress performance for normal models.

**Acceptance Criteria:**
- [ ] Mistral Q4_0 at 40% VRAM produces correct deterministic output
- [ ] Mistral Q4_0 at 100% VRAM (default) produces correct output with no perf regression
- [ ] GPT-OSS 20B (exceeds VRAM) loads and produces output (may be slow)
- [ ] Budget summary shows per-category breakdown including streaming scratch
- [ ] No crashes, no OOM, no hangs

**Implementation Guide:**

### Test 1: Low VRAM correctness

```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, ...` (correct sequence)

### Test 2: No performance regression

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

### Test 3: GPT-OSS 20B (model exceeding VRAM)

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 'Hello, my name is' -n 20 --seed 42 --temp 0
```
Expected: Coherent output (may take minutes due to per-weight streaming)

### Test 4: Strict mode with very low budget

```bash
GGML_SYCL_VRAM_BUDGET_PCT=20 GGML_SYCL_MEMORY_STRICT=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 'Hello' -n 1 --seed 42 --temp 0
```
Expected: Either produces output (with host streaming) or fails gracefully (no crash)

**Commit:** No code commit — this is verification only.

**Notes for implementer:**
- Wait 30+ seconds between benchmark runs to avoid thermal throttling.
- The GPT-OSS 20B test may be very slow (minutes per token) due to per-weight streaming. This is expected — performance optimization is future work (WI-3 layer prefetch).
- Check stderr for streaming debug messages to verify the streaming path is being exercised.
- If GPT-OSS 20B hangs during loading, it may be the same pre-existing issue (not related to this work). Focus on the Mistral low-VRAM test as the primary verification.

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

## Critical Warning: Streaming Scratch Buffer Lifetime

The streaming scratch buffer is **reusable** — only valid until the next `stream_weight_to_device()` call. This means:

1. **Legacy dispatch (per-token):** Safe. Each mul_mat reads its weight from the scratch, completes, then the next mul_mat overwrites the scratch with the next weight.

2. **Unified kernel (persistent TG):** **UNSAFE.** The plan records pointers at build time. If weight A's pointer is the scratch buffer, and weight B is streamed next (overwriting the scratch), weight A's pointer is now invalid. Task 3 must handle this by disabling persistent TG when streaming is active.

3. **Graph replay:** The graph captures kernel arguments including pointers. If the scratch pointer is captured, replaying the graph after the scratch is overwritten produces garbage. The solution is to also disable graph replay when streaming is active, OR ensure the scratch pointer is stable (one per weight — expensive but correct).

The simplest correct approach for v1: **disable persistent TG and graph replay when `g_model_exceeds_vram` is true.** This makes inference slow but correct. Performance optimization is WI-3 layer prefetch work.
