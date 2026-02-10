# Host Weight Streaming Activation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Fix the host weight streaming path so models exceeding VRAM produce correct output instead of zero tokens / garbage. The DMA infrastructure exists (layer-streaming.hpp/cpp) but is never activated.

**Architecture:** Three bugs conspire to make host-resident weights unusable: (1) `g_model_exceeds_vram` ignores `GGML_SYCL_VRAM_BUDGET_PCT`, so streaming gates never trigger, (2) layer streaming is gated behind `GGML_SYCL_FORCE_STREAMING=1` so it never auto-activates, (3) graph replay uses baked pointers that go stale when the unified cache evicts weights under budget pressure.

**Tech Stack:** C++17, SYCL/Level Zero, llama.cpp unified cache

**Beads Issue:** llama.cpp-i7zx

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 3, 5 | Budget flag fix + auto-activate streaming + budget tracking |
| B | 2, 4 | Graph/TG disable under pressure + non-layer tensor staging |

### Dependency Graph

```
Task 1 (Track A, no deps)       — Fix g_model_exceeds_vram to use effective budget
Task 2 (Track B, no deps)       — Disable graph replay when cache under pressure
Task 3 (Track A, depends on 1)  — Auto-activate layer streaming
Task 4 (Track B, depends on 2)  — Non-layer tensor staging
Task 5 (Track A, depends on 1)  — Budget streaming buffers in unified cache
Task 6 (depends on 3, 4, 5)     — Integration test
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 2086-2165, set_tensor_inventory) | 1, 3 | Sequential (same track) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 2212-2268, recalc) | 1 | None |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 29920-29940, persistent TG gate) | 2 | None |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 30200-30210, graph replay gate) | 2 | None |
| `ggml/src/ggml-sycl/unified-cache.hpp` | 2, 5 | Sequential (2 before 5) |
| `ggml/src/ggml-sycl/unified-cache.cpp` | 2, 5 | Sequential |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 2452-2558, get_data_ptr_slow) | 4 | None |

---

## Background: What Exists vs. What's Missing

### What Already Exists (DO NOT re-implement)

All of these were implemented in commits 00b844cda through bec2d4730:

1. **`layer_stream_manager`** class (`layer-streaming.hpp:34-139`, `layer-streaming.cpp:14-377`): Full double-buffered layer streaming with `build_layer_map()`, `allocate_buffers()`, `ensure_layer()`, `prefetch_next_layer()`, `get_weight_device_ptr()`, `register_host_ptr()`. Per-device singleton via `get_layer_stream_manager()`.

2. **`ggml_sycl_ensure_weight_on_device()`** (`ggml-sycl.cpp:12372-12415`): Checks layer stream manager first, falls back to tiered cache lookup. Called at 6 dispatch sites (lines 12420, 15607, 19795, 19894, 21308, 22266).

3. **Graph replay disable gate** (`ggml-sycl.cpp:30204`): Disables graph replay when `g_model_exceeds_vram` is true. Already implemented.

4. **Persistent TG disable gate** (`ggml-sycl.cpp:29929`): Disables persistent TG when `g_model_exceeds_vram` is true. Already implemented.

5. **Prefetch wiring** (`ggml-sycl.cpp:19859-19877`): Layer transition triggers `prefetch_next_layer()`, and `ensure_layer()` called before weight access. Already integrated into `ggml_backend_sycl_mul_mat`.

6. **Host pointer registration** (`ggml-sycl.cpp:2516-2518`): Registers `tensor->data` with layer stream manager in `get_data_ptr_slow()`. Already implemented.

7. **Pointer resolution for streaming** (`ggml-sycl.cpp:2537-2543`, `26426-26431`): `get_data_ptr_slow()` and `get_tensor_ptr_fast()` check layer streaming. Already implemented.

### What's Missing (THIS plan fixes)

1. **`g_model_exceeds_vram` uses 90% of total VRAM, ignoring `GGML_SYCL_VRAM_BUDGET_PCT`** — When user sets `GGML_SYCL_VRAM_BUDGET_PCT=40`, the flag stays false even though the unified cache only has 40% budget. All streaming gates (graph disable, persistent TG disable, ensure_weight_on_device, layer streaming init) require `g_model_exceeds_vram=true`.

   - **Root cause**: `ggml-sycl.cpp:2109` calculates `vram_budget_base = base_mem * 0.9` using total VRAM, ignoring the env var. The env var only affects the unified cache constructor and `get_memory()`.
   - **Impact**: With `VRAM_BUDGET_PCT=40`, unified cache evicts weights to host, but `g_model_exceeds_vram=false` → graph replay active with stale baked pointers → garbage output.

2. **Layer streaming gated behind `GGML_SYCL_FORCE_STREAMING=1`** — Even when `g_model_exceeds_vram` is true, the layer stream manager is only initialized if the user explicitly sets `GGML_SYCL_FORCE_STREAMING=1`. Without this env var, the code logs "CPU offload via fit_params preferred" and doesn't init streaming.

   - **Root cause**: `ggml-sycl.cpp:2126-2144` checks `streaming_forced` before `model_exceeds_vram`. When streaming isn't forced, the `else if (model_exceeds_vram)` branch just logs info without initializing.
   - **Impact**: Models that genuinely exceed VRAM (after fit_params offloading) have no streaming → kernels get null/host pointers → fail.

3. **Graph replay not disabled when unified cache evicts under budget pressure** — The graph replay disable only checks `g_model_exceeds_vram`. When the model fits (per 90% calculation) but unified cache evicts due to reduced budget, graph replay is still active. Baked pointers in the graph reference evicted device memory.

   - **Root cause**: No mechanism to signal "cache is under eviction pressure" to the graph system.
   - **Impact**: Stale baked pointers → garbage output even for models that "fit" per the initial calculation.

4. **Non-layer tensors (embedding, output) not handled by streaming** — `layer_stream_manager` only covers `blk.N.*` tensors. `token_embd.weight` (~77 MB) and `output.weight` (~77 MB) can also be evicted to host.

5. **Streaming buffer allocation not tracked in budget** — The 2 × max_layer_size device allocation (~250 MB for Mistral 7B) bypasses unified cache budget tracking, potentially causing OOM.

---

## Root Cause Analysis: The Three Bugs

### Bug 1: g_model_exceeds_vram ignores budget override

```
set_tensor_inventory (ggml-sycl.cpp:2086-2122):
  base_mem = total_mem = 12 GB                           // Line 2093
  vram_budget_base = base_mem * 0.9 = 10.8 GB           // Line 2109
  vram_budget = 10.8 - headroom = 9.6 GB                // Line 2112
  model_size = 3.8 GB (Mistral Q4_0)
  g_model_exceeds_vram = 3.8 > 9.6 = FALSE              // Line 2122

Meanwhile in unified_cache constructor (unified-cache.cpp:4398):
  GGML_SYCL_VRAM_BUDGET_PCT=40 → budget = 4.8 GB       // Line 4409
  4.8 GB - headroom = ~3.6 GB effective for weights
  3.8 GB > 3.6 GB → cache EVICTS some weights to host

Result: g_model_exceeds_vram=false but cache is actively evicting → all gates wrong
```

### Bug 2: Graph replay with evicted pointers

```
1. Graph recorded with weight at device_ptr=0xABCD (VRAM)
2. Unified cache evicts weight → device_ptr freed
3. Graph replays with baked pointer 0xABCD → reads freed memory
4. Garbage logits → zero tokens
```

### Bug 3: Streaming never auto-activates

```
set_tensor_inventory (ggml-sycl.cpp:2126-2144):
  streaming_forced = (GGML_SYCL_FORCE_STREAMING == "1")  // FALSE by default
  if (streaming_forced) { ... }
  else if (model_exceeds_vram) {
      LOG("CPU offload via fit_params preferred");        // Just a log message!
      // No streaming initialization
  }
```

---

## Task 1: Fix g_model_exceeds_vram to Use Effective Budget

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 2086-2122 in `set_tensor_inventory`, lines 2212-2268 in `recalc`)

**Description:**

The `g_model_exceeds_vram` flag must respect `GGML_SYCL_VRAM_BUDGET_PCT` so that all streaming gates (graph disable, persistent TG disable, ensure_weight_on_device, layer streaming init) activate when the effective budget is exceeded. The fix reads the same env var used by the unified cache and the `get_memory()` function.

**Acceptance Criteria:**
- [ ] With `GGML_SYCL_VRAM_BUDGET_PCT=30`, Mistral 7B Q4_0 (3.8 GB) sets `g_model_exceeds_vram=true`
- [ ] Without the env var, Mistral 7B still sets `g_model_exceeds_vram=false` (no regression)
- [ ] The `recalc_model_exceeds_vram()` function allows false→true transitions when budget is reduced
- [ ] Build succeeds with zero warnings
- [ ] Diagnostic log shows the effective budget used

**Implementation Guide:**

### Step 1: Apply GGML_SYCL_VRAM_BUDGET_PCT to vram_budget calculation

In `ggml-sycl.cpp`, find `ggml_backend_sycl_set_tensor_inventory()`. After line 2093 (`const size_t base_mem = ...`), replace lines 2099-2112:

**Current code** (ggml-sycl.cpp:2099-2112):
```cpp
    // Budget for weight cache = 90% of total VRAM minus headroom
    // Headroom = max(256MB, 10% of total) for runtime scratch space
    // NOTE: We do NOT subtract already_allocated here. The unified cache tracks all
    // allocations and will stream from host if VRAM is exhausted. Pre-subtracting
    // already_allocated artificially reduces the budget, causing models that fit
    // to be incorrectly marked as "exceeds VRAM" and placed in host memory.
    const size_t min_headroom = 256ull * 1024ull * 1024ull;
    const size_t base_headroom = std::max<size_t>(min_headroom, base_mem / 10);

    // Tiered mode budget: total * 0.9 - headroom (no already_allocated subtraction)
    size_t vram_budget_base = static_cast<size_t>(base_mem * 0.9);
    size_t vram_budget = 0;
    if (vram_budget_base > base_headroom) {
        vram_budget = vram_budget_base - base_headroom;
    }
```

**Replacement:**
```cpp
    // Budget for weight cache = effective_pct% of total VRAM minus headroom
    // Headroom = max(256MB, 10% of total) for runtime scratch space
    // NOTE: We do NOT subtract already_allocated here. The unified cache tracks all
    // allocations and will stream from host if VRAM is exhausted. Pre-subtracting
    // already_allocated artificially reduces the budget, causing models that fit
    // to be incorrectly marked as "exceeds VRAM" and placed in host memory.
    const size_t min_headroom = 256ull * 1024ull * 1024ull;
    const size_t base_headroom = std::max<size_t>(min_headroom, base_mem / 10);

    // Apply GGML_SYCL_VRAM_BUDGET_PCT if set (same env var as unified cache and get_memory)
    int budget_pct = 90;  // Default: 90% of total VRAM
    const char * env_budget_pct = std::getenv("GGML_SYCL_VRAM_BUDGET_PCT");
    if (env_budget_pct) {
        int pct = std::atoi(env_budget_pct);
        if (pct >= 1 && pct <= 100) {
            budget_pct = pct;
        }
    }

    size_t vram_budget_base = static_cast<size_t>(base_mem * (static_cast<double>(budget_pct) / 100.0));
    size_t vram_budget = 0;
    if (vram_budget_base > base_headroom) {
        vram_budget = vram_budget_base - base_headroom;
    }
```

### Step 2: Update recalc to allow false→true for budget reduction

In `ggml_sycl_recalc_model_exceeds_vram()` (line 2224-2235), replace the guard that prevents false→true transitions:

**Current code** (ggml-sycl.cpp:2224-2235):
```cpp
    // Only allow recalc to flip true→false (model now fits after reservations decrease).
    // Never flip false→true: the initial set_tensor_inventory determination (using 90% budget)
    // is authoritative. When GGML_SYCL_VRAM_BUDGET_PCT reduces the budget further,
    // fit_params handles the overflow via ngl/context reduction — we should NOT
    // independently start streaming or offloading KV based on the reduced budget.
    if (!old_exceeds && new_exceeds) {
        GGML_LOG_DEBUG("[SYCL-BUDGET] Recalc: model exceeds effective budget "
                      "(model=%.1f MB, effective_budget=%.1f MB) but initial determination was false — "
                      "deferring to fit_params for ngl/context reduction\n",
                      g_tensor_inventory_total_size / (1024.0 * 1024.0),
                      effective_budget / (1024.0 * 1024.0));
        return;
    }
```

**Replacement:**
```cpp
    // Allow false→true transitions when effective budget has been reduced
    // (e.g., by GGML_SYCL_VRAM_BUDGET_PCT or runtime reservations).
    // Previously this was blocked, but it prevented streaming activation when
    // the cache is actively evicting weights to host.
    if (!old_exceeds && new_exceeds) {
        GGML_LOG_INFO("[SYCL-BUDGET] Recalc: model now exceeds effective budget "
                      "(model=%.1f MB, effective_budget=%.1f MB) — enabling streaming gates\n",
                      g_tensor_inventory_total_size / (1024.0 * 1024.0),
                      effective_budget / (1024.0 * 1024.0));
    }
```

(Remove the `return;` so the function continues to the state update.)

### Step 3: Update diagnostic log

In the log block at lines 2147-2153, update the message to show the effective budget_pct:

After `g_model_exceeds_vram.store(...)` (line 2123), add:
```cpp
    GGML_LOG_INFO("[SYCL-BUDGET] budget_pct=%d%%, vram_budget=%.1f MB, model_size=%.1f MB, exceeds=%s\n",
                  budget_pct, vram_budget / (1024.0 * 1024.0),
                  effective_model_size / (1024.0 * 1024.0),
                  model_exceeds_vram ? "true" : "false");
```

### Step 4: Build and test

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

Verify flag behavior:
```bash
# Default (no env var): model fits → g_model_exceeds_vram=false
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>&1 | grep SYCL-BUDGET

# 30% budget: model exceeds → g_model_exceeds_vram=true
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0 2>&1 | grep SYCL-BUDGET
```

Expected: First shows `exceeds=false`, second shows `exceeds=true`.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: fix g_model_exceeds_vram to respect GGML_SYCL_VRAM_BUDGET_PCT"
```

**Notes for implementer:**
- The `budget_pct` variable is read from the same env var (`GGML_SYCL_VRAM_BUDGET_PCT`) used by the unified cache (`unified-cache.cpp:4398`) and `get_memory()` (`ggml-sycl.cpp:30774`). This ensures consistency.
- When `budget_pct < 90`, the effective budget shrinks, making `model_exceeds_vram` more likely to be true. This is correct: if the cache only has 30% VRAM, a model using 32% will be partially evicted.
- The `recalc` change is safe because later code (lines 2247-2268) already handles the false→true transition by initializing streaming when FORCE_STREAMING=1. Task 3 extends this to auto-activate.
- DO NOT change the unified cache budget calculation — that's already correct. This task only fixes the flag that gates streaming activation.

---

## Task 2: Disable Graph Replay Under Cache Eviction Pressure

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (add eviction pressure flag)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (set flag on eviction)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 30200-30210 graph gate, lines 29920-29935 persistent TG gate)

**Description:**

Graph replay bakes tensor pointers at record time. When the unified cache evicts a weight from VRAM to host, the baked pointer becomes a dangling reference. This task adds a flag that's set when the cache has evicted ANY weight, and uses it to disable graph replay and persistent TG.

**Acceptance Criteria:**
- [ ] When unified cache evicts a weight, `g_cache_has_evictions` flag is set
- [ ] Graph replay is disabled when eviction flag is set (even if `g_model_exceeds_vram` is false)
- [ ] Persistent TG is disabled when eviction flag is set
- [ ] No regression when no evictions occur (flag stays false, graphs work normally)
- [ ] Build succeeds with zero warnings
- [ ] PP512 >= 1200, TG128 >= 68 for models that fit in VRAM

**Implementation Guide:**

### Step 1: Add eviction pressure flag to unified cache

In `unified-cache.hpp`, find the `unified_tensor_cache` class declaration. Add near other member variables:

```cpp
    // Set to true when any weight has been evicted from device to host.
    // Used by graph replay / persistent TG to know baked pointers may be stale.
    std::atomic<bool> has_evictions_{false};
public:
    bool has_evictions() const { return has_evictions_.load(std::memory_order_acquire); }
```

In `unified-cache.cpp`, find the eviction path in `update_reserved_bytes()` or wherever weights are evicted from device to host. After a successful eviction, add:
```cpp
    has_evictions_.store(true, std::memory_order_release);
```

Search for `memory_tier::HOST_PINNED` or `evict` in unified-cache.cpp to find the exact eviction site. The eviction happens when `used_ > budget_` and the cache moves entries from VRAM to host.

### Step 2: Add public query function

In `unified-cache.hpp`, add a free function declaration near other query functions:

```cpp
// Returns true if any unified cache instance has evicted weights from device to host.
// Thread-safe.
bool unified_cache_has_evictions();
```

In `unified-cache.cpp`, implement:

```cpp
bool unified_cache_has_evictions() {
    // Check the global cache instance(s)
    // If any per-device cache has evictions, return true
    for (auto & [device, cache] : g_device_caches) {
        if (cache && cache->has_evictions()) {
            return true;
        }
    }
    return false;
}
```

If the global cache is accessed differently (e.g., via `g_tensor_cache`), adjust to match the actual global accessor pattern. Check how `get_unified_cache_for_device()` works and use the same pattern.

### Step 3: Update graph replay gate

In `ggml-sycl.cpp`, find the graph replay disable at line 30204. Change:

**Current code** (ggml-sycl.cpp:30201-30209):
```cpp
#ifdef GGML_SYCL_GRAPH
    // Disable graph replay when model exceeds VRAM — weight streaming rotates
    // pointers between two device buffers, incompatible with baked graph pointers.
    if (g_model_exceeds_vram.load(std::memory_order_acquire)) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Disabled: model exceeds VRAM, weight streaming active\n");
        compute_impl();
        record_completion(false);
        return GGML_STATUS_SUCCESS;
    }
```

**Replacement:**
```cpp
#ifdef GGML_SYCL_GRAPH
    // Disable graph replay when weight pointers may be stale:
    // 1. Model exceeds VRAM → layer streaming rotates buffer pointers
    // 2. Cache has evictions → baked pointers reference freed device memory
    if (g_model_exceeds_vram.load(std::memory_order_acquire) ||
        ggml_sycl::unified_cache_has_evictions()) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Disabled: weight pointers may be stale (exceeds=%d, evictions=%d)\n",
                        (int)g_model_exceeds_vram.load(std::memory_order_acquire),
                        (int)ggml_sycl::unified_cache_has_evictions());
        compute_impl();
        record_completion(false);
        return GGML_STATUS_SUCCESS;
    }
```

### Step 4: Update persistent TG gate

In `ggml-sycl.cpp`, find `should_use_persistent_tg()` at line 29929. Change:

**Current code** (ggml-sycl.cpp:29927-29932):
```cpp
    // 1b. Model exceeds VRAM — persistent TG records weight pointers that become
    // stale during double-buffered layer streaming. Disable to use per-op dispatch.
    if (g_model_exceeds_vram.load(std::memory_order_acquire)) {
        GGML_SYCL_DEBUG("[PERSISTENT-TG] Disabled: model exceeds VRAM, weight streaming active\n");
        return false;
    }
```

**Replacement:**
```cpp
    // 1b. Model exceeds VRAM or cache has evictions — persistent TG records weight
    // pointers that become stale. Disable to use per-op dispatch.
    if (g_model_exceeds_vram.load(std::memory_order_acquire) ||
        ggml_sycl::unified_cache_has_evictions()) {
        GGML_SYCL_DEBUG("[PERSISTENT-TG] Disabled: weight pointers may be stale\n");
        return false;
    }
```

### Step 5: Build and test

```bash
ninja -C build -j $(nproc)
```

Performance (no evictions, graphs should still work):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68 (has_evictions is false → graphs active)

Low budget (evictions likely):
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0
```
Expected: No crash (but output may still be wrong until Tasks 3-4 are done).

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp \
        ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: disable graph replay when unified cache has evicted weights"
```

**Notes for implementer:**
- The `has_evictions_` flag is one-way (false → true, never reset). This is intentional — once the cache starts evicting, we can't trust baked graph pointers for the rest of the session.
- The flag is per-cache-instance, but the query function checks all instances. For single-GPU setups, there's only one instance.
- This is a correctness fix, not a performance fix. Models that fit in VRAM (no evictions) are unaffected.
- The `unified_cache_has_evictions()` function needs access to the global cache. Find how `get_unified_cache_for_device()` is implemented and use the same pattern. It likely accesses `g_tensor_cache` via a mutex.
- DO NOT add the eviction flag check to `ensure_weight_on_device()` — that's already guarded by `g_model_exceeds_vram` which Task 1 fixes.

---

## Task 3: Auto-Activate Layer Streaming

**Track:** A
**Depends on:** Task 1 (g_model_exceeds_vram now reflects effective budget)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 2126-2144 in `set_tensor_inventory`, lines 2247-2268 in `recalc`)

**Description:**

Layer streaming is currently gated behind `GGML_SYCL_FORCE_STREAMING=1`. After Task 1 fixes the budget flag, streaming should auto-activate when the model exceeds the effective budget. `GGML_SYCL_FORCE_STREAMING` remains available but becomes a testing/override tool rather than a required activation mechanism.

**Acceptance Criteria:**
- [ ] Layer streaming auto-activates when `g_model_exceeds_vram=true` (no env var needed)
- [ ] `GGML_SYCL_FORCE_STREAMING=1` still works as override for models that fit
- [ ] Buffer allocation messages visible in logs
- [ ] Host pointer registration active when streaming is on
- [ ] Build succeeds with zero warnings
- [ ] No regression when model fits (streaming not initialized)

**Implementation Guide:**

### Step 1: Change activation logic in set_tensor_inventory

In `ggml-sycl.cpp`, replace the streaming init block at lines 2126-2144:

**Current code** (ggml-sycl.cpp:2126-2144):
```cpp
    // Initialize double-buffered layer streaming when model exceeds VRAM
    // OR when user explicitly forces streaming via GGML_SYCL_FORCE_STREAMING=1
    const char * force_stream = std::getenv("GGML_SYCL_FORCE_STREAMING");
    const bool streaming_forced = force_stream && std::atoi(force_stream) == 1;
    if (streaming_forced) {
        // FORCE_STREAMING=1 enables streaming regardless of model size
        auto & mgr = ggml_sycl::get_layer_stream_manager(ctx->device);
        mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
        sycl::queue & q = ggml_sycl_get_device(ctx->device).default_queue();
        if (mgr.allocate_buffers(q)) {
            GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled (forced): %d layers, 2 x %.1f MB buffers\n",
                          mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
        } else {
            GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed\n");
        }
    } else if (model_exceeds_vram) {
        // Model exceeds VRAM but streaming not forced — CPU offload via fit_params is preferred
        GGML_LOG_INFO("[SYCL-BUDGET] Model exceeds VRAM by %.1f MB — CPU offload via fit_params preferred over streaming\n",
                      (g_tensor_inventory_total_size - vram_budget) / (1024.0 * 1024.0));
    }
```

**Replacement:**
```cpp
    // Initialize double-buffered layer streaming when:
    // 1. Model exceeds effective VRAM budget (auto-activation)
    // 2. User forces streaming via GGML_SYCL_FORCE_STREAMING=1 (testing/override)
    const char * force_stream = std::getenv("GGML_SYCL_FORCE_STREAMING");
    const bool streaming_forced = force_stream && std::atoi(force_stream) == 1;
    if (model_exceeds_vram || streaming_forced) {
        auto & mgr = ggml_sycl::get_layer_stream_manager(ctx->device);
        mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
        sycl::queue & q = ggml_sycl_get_device(ctx->device).default_queue();
        if (mgr.allocate_buffers(q)) {
            GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled%s: %d layers, 2 x %.1f MB buffers\n",
                          streaming_forced ? " (forced)" : "",
                          mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
        } else {
            GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed\n");
        }
    }
```

### Step 2: Update recalc streaming activation

In `ggml_sycl_recalc_model_exceeds_vram()`, replace lines 2247-2268:

**Current code** (ggml-sycl.cpp:2247-2268):
```cpp
        // Initialize layer streaming when transitioning to model_exceeds_vram
        // Only if GGML_SYCL_FORCE_STREAMING=1 (CPU offload via fit_params is preferred)
        if (!old_exceeds && new_exceeds && !g_tensor_inventory.empty()) {
            const char * force_stream = std::getenv("GGML_SYCL_FORCE_STREAMING");
            const bool streaming_forced = force_stream && std::atoi(force_stream) == 1;
            if (streaming_forced) {
                int device = g_tensor_inventory_device;
                auto & mgr = ggml_sycl::get_layer_stream_manager(device);
                if (!mgr.is_active()) {
                    mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
                    sycl::queue & q = ggml_sycl_get_device(device).default_queue();
                    if (mgr.allocate_buffers(q)) {
                        GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled (late init, forced): %d layers, 2 x %.1f MB buffers\n",
                                      mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
                    } else {
                        GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed (late init)\n");
                    }
                }
            } else {
                GGML_LOG_INFO("[SYCL-BUDGET] Model exceeds effective budget — CPU offload via fit_params preferred over streaming\n");
            }
        }
```

**Replacement:**
```cpp
        // Initialize layer streaming when transitioning to model_exceeds_vram
        if (!old_exceeds && new_exceeds && !g_tensor_inventory.empty()) {
            int device = g_tensor_inventory_device;
            auto & mgr = ggml_sycl::get_layer_stream_manager(device);
            if (!mgr.is_active()) {
                mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
                sycl::queue & q = ggml_sycl_get_device(device).default_queue();
                if (mgr.allocate_buffers(q)) {
                    GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled (late init): %d layers, 2 x %.1f MB buffers\n",
                                  mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
                } else {
                    GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed (late init)\n");
                }
            }
        }
```

### Step 3: Build and test

```bash
ninja -C build -j $(nproc)
```

Auto-activation with low budget:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>&1 | grep -E "LAYER-STREAM|SYCL-BUDGET"
```
Expected: "[SYCL-BUDGET] Layer streaming enabled" in logs. Output may still be wrong (Task 4 needed for non-layer tensors).

No regression:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: auto-activate layer streaming when model exceeds effective budget"
```

**Notes for implementer:**
- After this task + Task 1, `GGML_SYCL_VRAM_BUDGET_PCT=30` will activate streaming automatically. No `GGML_SYCL_FORCE_STREAMING` needed.
- fit_params still runs first and offloads what it can. Streaming is a safety net for weights that still need to be evicted.
- The streaming buffer allocation (2 × max_layer_size) happens immediately on activation. If this fails, the code falls back to no streaming (host USM pointers used directly, which may work slowly or crash depending on the memory type). Task 5 will track this allocation in the budget.

---

## Task 4: Non-Layer Tensor Staging for Host-Resident Embedding/Output

**Track:** B
**Depends on:** Task 2 (graph replay disabled when cache evicts, preventing stale pointer use)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 12372-12415 in `ensure_weight_on_device`, lines 2452-2558 in `get_data_ptr_slow`)

**Description:**

The `layer_stream_manager` only handles `blk.N.*` tensors. Non-layer tensors (`token_embd.weight` ~77 MB, `output.weight` ~77 MB, `output_norm.weight` tiny) can also be evicted to host. When these are accessed, `ensure_weight_on_device()` falls through to the tiered cache lookup which may return a host pointer. The kernel then receives a host USM pointer — if it's pinned memory this works slowly via PCIe, but if it's mmap it can't be accessed by the GPU.

This task adds explicit on-demand staging for non-layer host-resident tensors using the existing `ggml_sycl_get_staged_ptr_device()` function.

**Acceptance Criteria:**
- [ ] Non-layer tensors (embedding, output) that are host-resident get staged to device before kernel execution
- [ ] The staging buffer is persistent (not re-allocated per token)
- [ ] `ensure_weight_on_device()` handles both layer (via streaming) and non-layer (via staging) tensors
- [ ] Build succeeds with zero warnings
- [ ] No regression when all weights fit in VRAM

**Implementation Guide:**

### Step 1: Enhance ensure_weight_on_device for non-layer tensors

In `ggml-sycl.cpp`, modify `ggml_sycl_ensure_weight_on_device()` at line 12372. Replace the fallback section:

**Current code** (ggml-sycl.cpp:12402-12415):
```cpp
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
```

**Replacement:**
```cpp
    // Fallback: tiered cache lookup (for non-layer tensors like embedding, output)
    ggml_sycl::memory_tier tier;
    bool                   in_inventory = false;
    void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
    if (!cached_ptr || !in_inventory) {
        return;
    }

    if (tier == ggml_sycl::memory_tier::VRAM) {
        extra->data_device[device] = cached_ptr;
    } else {
        // Non-layer host-resident tensor (embedding, output) — stage to device
        // via the pinned memory staging path. This copies from host to a persistent
        // device-accessible pinned buffer so the kernel can read it.
        size_t nbytes = ggml_nbytes(src0);
        void * staged = ggml_sycl_get_staged_ptr_device(cached_ptr, nbytes, device);
        if (staged) {
            // Update the staging buffer contents (weight data may have been updated)
            sycl::queue & q = ggml_sycl_get_device(device).default_queue();
            q.memcpy(staged, cached_ptr, nbytes);
            q.wait();
            extra->data_device[device] = staged;
        }
        // If staging fails, fall through — get_data_ptr_slow will handle via USM host
    }
```

Wait — `ggml_sycl_get_staged_ptr_device` copies from src (mmap/malloc) to pinned host memory, not device memory. For GPU kernels, we need the data on DEVICE. The staging function creates a USM host pinned copy, which the GPU CAN access via PCIe.

Actually, looking at `ggml_sycl_get_staged_ptr_device()` more carefully (`common.cpp:264`), it allocates pinned host memory and memcpy's the source data into it. The returned pointer is USM host pinned — accessible by GPU but via PCIe bandwidth.

For non-layer tensors accessed every token (embedding at PP, output at every TG step), this is acceptable. The alternative (sycl::malloc_device + memcpy) would be better but requires managing the allocation lifecycle.

**Revised approach**: Use the staging function as-is. It returns a USM host pinned pointer that the GPU can access. This is slow but correct.

**Revised replacement:**
```cpp
    // Fallback: tiered cache lookup (for non-layer tensors like embedding, output)
    ggml_sycl::memory_tier tier;
    bool                   in_inventory = false;
    void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
    if (!cached_ptr || !in_inventory) {
        return;
    }

    if (tier == ggml_sycl::memory_tier::VRAM) {
        extra->data_device[device] = cached_ptr;
    } else if (tier == ggml_sycl::memory_tier::HOST_PINNED) {
        // Host pinned (USM) — GPU can access directly via PCIe. Slow but correct.
        extra->data_device[device] = cached_ptr;
    } else {
        // MMAP tier — not USM, GPU cannot access. Stage to pinned memory.
        size_t nbytes = ggml_nbytes(src0);
        void * staged = ggml_sycl_get_staged_ptr_device(cached_ptr, nbytes, device);
        if (staged) {
            extra->data_device[device] = staged;
        }
        // If staging fails, fall through — get_data_ptr_slow may handle it
    }
```

### Step 2: Build and test

```bash
ninja -C build -j $(nproc)
```

With Tasks 1-3 applied, low budget should now produce correct output:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, ...` (correct output, may be slow ~3-4 tok/s)

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: handle host-resident non-layer tensors in streaming dispatch"
```

**Notes for implementer:**
- `memory_tier::HOST_PINNED` indicates USM host pinned memory, which the GPU CAN access directly. It's slow (PCIe bandwidth) but correct. No staging needed.
- `memory_tier::MMAP` indicates mmap'd file memory, which is NOT USM. The GPU CANNOT access it. Must stage through pinned memory first.
- The `ggml_sycl_get_staged_ptr_device()` function caches the staging buffer by source pointer, so repeated accesses to the same tensor reuse the same pinned buffer.
- Non-layer tensors accessed during inference: `token_embd.weight` (during tokenization/PP), `output.weight` (every TG step), `output_norm.weight` (every step, tiny). The overhead from PCIe access for these is small compared to the overall streaming overhead.

---

## Task 5: Budget Streaming Buffers in Unified Cache

**Track:** A
**Depends on:** Task 1 (streaming buffers only allocated when g_model_exceeds_vram reflects effective budget)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 2130-2140 streaming init section)

**Description:**

The layer stream manager allocates 2 × max_layer_size device memory (~250 MB for Mistral 7B). This allocation bypasses the unified cache budget tracking, potentially causing OOM when the cache tries to use all available VRAM.

**Acceptance Criteria:**
- [ ] Streaming buffer allocation calls `unified_cache_add_runtime_bytes()` to register with budget
- [ ] Budget diagnostic log shows streaming buffer allocation
- [ ] No OOM from double-booking VRAM
- [ ] Build succeeds with zero warnings

**Implementation Guide:**

### Step 1: Track streaming buffer allocation in budget

In the streaming init block (modified in Task 3), after `allocate_buffers()` succeeds, add budget registration:

```cpp
    if (model_exceeds_vram || streaming_forced) {
        auto & mgr = ggml_sycl::get_layer_stream_manager(ctx->device);
        mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
        sycl::queue & q = ggml_sycl_get_device(ctx->device).default_queue();
        if (mgr.allocate_buffers(q)) {
            // Register streaming buffer allocation with unified cache budget
            ggml_sycl::unified_cache_add_runtime_bytes(ctx->device, mgr.allocated_bytes());
            GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled%s: %d layers, 2 x %.1f MB buffers (%.1f MB budgeted)\n",
                          streaming_forced ? " (forced)" : "",
                          mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0),
                          mgr.allocated_bytes() / (1024.0 * 1024.0));
        } else {
            GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed\n");
        }
    }
```

Do the same in the `recalc` late-init path (Task 3's replacement code):

```cpp
            if (!mgr.is_active()) {
                mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
                sycl::queue & q = ggml_sycl_get_device(device).default_queue();
                if (mgr.allocate_buffers(q)) {
                    ggml_sycl::unified_cache_add_runtime_bytes(device, mgr.allocated_bytes());
                    GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled (late init): %d layers, 2 x %.1f MB buffers\n",
                                  mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
                }
            }
```

### Step 2: Build and test

```bash
ninja -C build -j $(nproc)
```

Verify budget shows streaming allocation:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0 2>&1 | grep -E "BUDGET|STREAM"
```
Expected: Budget log shows streaming buffer bytes.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: track layer streaming buffer allocation in unified cache budget"
```

**Notes for implementer:**
- `unified_cache_add_runtime_bytes(device, bytes)` is the existing API for registering device memory usage with the budget. It's used by KV cache, compute buffers, and expert staging.
- The streaming buffers are allocated once and never freed (lifetime of the process). No corresponding `sub_runtime_bytes` call needed.
- If budget registration causes the cache to evict more weights, that's correct — the streaming buffers consume the VRAM that the cache would have used for weights. The streaming manager can then load those evicted weights back on demand.

---

## Task 6: Integration Test

**Track:** — (convergence point)
**Depends on:** Tasks 3, 4, 5
**File scope:**
- No code changes (testing only)

**Description:**

Comprehensive verification that host weight streaming works end-to-end across different budget levels.

**Acceptance Criteria:**
- [ ] `GGML_SYCL_VRAM_BUDGET_PCT=30`: Mistral Q4_0 produces correct deterministic output
- [ ] `GGML_SYCL_VRAM_BUDGET_PCT=40`: Mistral Q4_0 produces correct output
- [ ] Default (no env var): PP512 >= 1200, TG128 >= 68 (no regression)
- [ ] `GGML_SYCL_FORCE_STREAMING=1`: Mistral Q4_0 with streaming forced, correct output
- [ ] Layer streaming log messages visible in debug output
- [ ] Budget diagnostic shows streaming buffer allocation
- [ ] No crashes, no OOM, no hangs

**Implementation Guide:**

### Test 1: Default (no budget override, no streaming)

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

### Test 2: Performance regression check

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

Wait 30+ seconds between runs (Arc B580 thermal throttling).

### Test 3: 30% budget (streaming auto-activates)

```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Correct output (same as default). Will be slow (~3-4 tok/s).

### Test 4: 40% budget

```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Correct output.

### Test 5: Forced streaming (override for testing)

```bash
GGML_SYCL_FORCE_STREAMING=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Correct output, streaming active even though model fits.

### Test 6: Debug verification

```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 'Hello' -n 1 --seed 42 --temp 0 2>/tmp/stream_debug.txt
```
Check `/tmp/stream_debug.txt` for:
- `[SYCL-BUDGET] budget_pct=30%` — Task 1 effective budget
- `[SYCL-BUDGET] Layer streaming enabled:` — Task 3 auto-activation
- `[LAYER-STREAM] Loaded layer N into buffer` — DMA activity
- `[SYCL-GRAPH] Disabled: weight pointers may be stale` — Task 2 graph disable
- `[PERSISTENT-TG] Disabled: weight pointers may be stale` — Task 2 TG disable

### Test 7: GPT-OSS 20B (model exceeding VRAM)

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 'Hello, my name is' -n 5 --seed 42 --temp 0
```
Expected: Coherent output or graceful failure message. May take a long time per token.
If it fails due to pre-existing MoE/MXFP4 issues, document and move on.

**No code commit — verification only.**

**Notes for implementer:**
- Wait 30+ seconds between benchmark runs to avoid thermal throttling on Arc B580.
- The GPT-OSS 20B test may fail due to pre-existing issues (MXFP4 kernel, temp buffer OOM). The primary validation targets are Tests 1-5 with Mistral 7B.
- If Test 3 produces wrong output, check that `g_model_exceeds_vram=true` in the logs. If false, Task 1 isn't applied correctly.
- If Test 3 shows correct `g_model_exceeds_vram=true` but wrong output, check for `[LAYER-STREAM]` messages. If absent, Task 3 isn't activating streaming.
- If streaming is active but output is wrong, add a temporary `fprintf(stderr, ...)` in `ensure_weight_on_device()` to verify layer loading.

---

## Verification After Each Task

```bash
# Build
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Quick correctness (no streaming)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Performance (no regression)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

## Key Design Decisions

1. **Fix the flag, not the infrastructure**: Tasks 1-4 of the old plan (layer-streaming.hpp/cpp, graph disable, prefetch, pointer resolution) are already implemented. The issue is that these are never activated. This plan fixes the activation path, not the streaming machinery.

2. **Eviction pressure flag**: Rather than only checking `g_model_exceeds_vram`, graph replay also checks `unified_cache_has_evictions()`. This covers the edge case where the model "fits" per initial calculation but the cache evicts under runtime pressure (KV cache growth, compute buffer allocation).

3. **Host pinned as fallback**: Non-layer tensors in `HOST_PINNED` tier are directly accessible by the GPU via USM. While slow (PCIe bandwidth), this is correct and doesn't require additional staging.

4. **MMAP staging**: Tensors in `MMAP` tier (original file data) are NOT USM-accessible. These go through `ggml_sycl_get_staged_ptr_device()` which copies to a persistent pinned buffer.

5. **`GGML_SYCL_FORCE_STREAMING` kept as override**: Users can still force streaming even for models that fit in VRAM (useful for testing). The env var is no longer REQUIRED — streaming auto-activates when needed.
