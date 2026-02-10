# CPU Offload via Unified Cache + llama_params_fit

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Enable models larger than VRAM to run by offloading overflow layers to CPU, using the existing `llama_params_fit` infrastructure with the unified cache as the authoritative memory budget source.

**Architecture:** The unified cache calculates the VRAM budget and exports it. `llama_params_fit` queries this budget to determine optimal `n_gpu_layers`. Layers that fit run on GPU (with graph replay, persistent TG, SOA — full perf). Overflow layers run on CPU at host memory bandwidth (~80 GB/s DDR5). Only activations (~8 KB per split) transfer over PCIe.

**Tech Stack:** C++17, existing `llama_params_fit`, existing `ggml_backend_sched`, unified cache API

**Beads Issue:** llama.cpp-i7zx (alternative approach — benchmark against GPU streaming)

---

## Team Topology

**Recommended implementers:** 1 (all tasks are sequential, touching related code)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Dependency Graph

```
Task 1 (no deps)       — Re-enable fit_params for SYCL backend
Task 2 (depends on 1)  — Align VRAM budget between unified cache and fit_params
Task 3 (depends on 2)  — Track CPU-resident weights in unified cache
Task 4 (depends on 3)  — Integration test + benchmark vs GPU streaming
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `common/common.h` | 1 | None (1 line) |
| `common/common.cpp` | 1, 2 | Sequential |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 2 | None (isolated section) |
| `ggml/src/ggml-sycl/unified-cache.hpp` | 2, 3 | Sequential |
| `ggml/src/ggml-sycl/unified-cache.cpp` | 2, 3 | Sequential |

---

## Background: Why CPU Offload Beats GPU Streaming

For TG (batch=1), the bottleneck comparison:

| Approach | Per-Layer Cost | 32 Layers | Throughput |
|----------|---------------|-----------|------------|
| GPU (all VRAM) | 0.4ms compute | 13ms | 76 tok/s |
| GPU streaming | 8.3ms PCIe DMA + 0.4ms compute | 266ms | 3.8 tok/s |
| CPU offload | 1.6ms (DDR5 bandwidth-bound) | 51ms | ~20 tok/s |

CPU is **5x faster** than GPU streaming because:
- Host memory bandwidth (80 GB/s DDR5) >> PCIe bandwidth (15 GB/s Gen4 x16)
- Only 8 KB activations transfer at CPU↔GPU split (not 125 MB weights)
- GPU layers keep graph replay + persistent TG (no perf degradation for cached layers)

### What Already Exists

1. **`llama_params_fit`** (`src/llama.cpp:143-689`): Sophisticated parameter fitting that:
   - Reduces context size first (e.g., 131K → 4K)
   - Then iteratively reduces `n_gpu_layers`
   - Supports partial-layer offload (attn only, FFN only, MoE only)
   - Supports multi-device with tensor_split
   - Uses `tensor_buft_overrides` for fine-grained tensor placement

2. **Layer assignment** (`src/llama-model.cpp:3310-3342`):
   ```cpp
   const int i_gpu_start = max(n_layer - n_gpu_layers, 0);
   // Layers before i_gpu_start → CPU, after → GPU
   ```

3. **Backend scheduler** (`ggml_backend_sched`): Automatically splits compute graph across CPU+GPU backends, handles activation transfers at split points.

4. **Our branch disables fit_params** (`common/common.h:335`):
   ```cpp
   bool fit_params = false;  // disabled by default - SYCL tiered memory handles large models
   ```

### What's Missing

1. `fit_params` is disabled — needs to be re-enabled
2. VRAM budget alignment — `llama_params_fit` queries raw `ggml_backend_dev_memory()` but the unified cache has its own budget formula (90% total - headroom). These must agree.
3. CPU-resident weight tracking — the unified cache should know about weights that went to CPU (for budget reporting and diagnostics).

---

## Task 1: Re-enable fit_params for SYCL Backend

**Depends on:** None
**File scope:**
- Modify: `common/common.h` (1 line)
- Modify: `common/common.cpp` (add conditional logic)

**Description:**

Change the default `fit_params` to `true` when SYCL backend is active. This allows `llama_params_fit` to correctly determine `n_gpu_layers` before model load, so overflow layers run on CPU.

**Acceptance Criteria:**
- [ ] `fit_params` defaults to true (was false)
- [ ] When model fits in VRAM: `llama_params_fit` returns early with no changes (no regression)
- [ ] When model exceeds VRAM: `n_gpu_layers` is reduced, overflow layers on CPU
- [ ] Build succeeds
- [ ] Mistral 7B Q4_0 (fits in VRAM): PP512 >= 1200, TG128 >= 68

**Implementation Guide:**

### Step 1: Change default fit_params

In `common/common.h`, change line 335:

```cpp
    bool    fit_params         = true;  // fit model parameters to device memory
```

### Step 2: Build and test

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

Normal model (fits in VRAM — should see "no changes needed" from params_fit):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68. Should see `llama_params_fit: no changes needed` in log.

Low VRAM test:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `llama_params_fit` should reduce n_gpu_layers. Output may or may not be correct yet (budget alignment not done).

**Commit:**
```bash
git add common/common.h
git commit -m "sycl: re-enable llama_params_fit for CPU offload of overflow layers"
```

**Notes for implementer:**
- This is a 1-line change. The infrastructure is already there — we just disabled it.
- `llama_params_fit` queries `ggml_backend_dev_memory()` which returns raw device free/total. The unified cache's budget may differ — that's Task 2.

---

## Task 2: Align VRAM Budget Between Unified Cache and fit_params

**Depends on:** Task 1
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (add budget export API)
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (add budget query API)
- Modify: `common/common.cpp` (set fit_params_target based on unified cache budget)

**Description:**

`llama_params_fit` uses raw device memory to decide what fits. The unified cache uses a different budget formula (90% of total - headroom, or `GGML_SYCL_VRAM_BUDGET_PCT`). These must agree or `llama_params_fit` might assign more layers to GPU than the unified cache can handle.

The approach: export the unified cache's headroom calculation as a free function, and use it to set `fit_params_target` (the margin parameter to `llama_params_fit`).

**Acceptance Criteria:**
- [ ] `llama_params_fit` margin matches unified cache headroom
- [ ] When `GGML_SYCL_VRAM_BUDGET_PCT=40` is set, `llama_params_fit` correctly computes n_gpu_layers for 40% VRAM
- [ ] GPU layers use graph replay and persistent TG (no degradation)
- [ ] CPU overflow layers produce correct output
- [ ] Build succeeds

**Implementation Guide:**

### Step 1: Export VRAM headroom from SYCL backend

Add a new API to the SYCL backend that returns the amount of VRAM headroom the unified cache reserves. This lets `common.cpp` set `fit_params_target` appropriately.

In `ggml-sycl.cpp`, add near the existing `ggml_backend_sycl_get_device_memory`:

```cpp
// Returns the VRAM headroom that the unified cache reserves for non-weight allocations.
// This is the margin that llama_params_fit should use to avoid overcommitting VRAM.
size_t ggml_backend_sycl_get_vram_headroom(int device) {
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(device, &free_mem, &total_mem);

    // Match the unified cache's budget formula
    size_t budget_pct = ggml_sycl::unified_cache_get_budget_pct();
    size_t headroom = total_mem - static_cast<size_t>(total_mem * budget_pct / 100.0);

    return headroom;
}
```

Also add the declaration to the public header (or a SYCL-specific header that common.cpp can include conditionally).

### Step 2: Set fit_params_target from unified cache headroom

In `common/common.cpp`, in `common_init_from_params()`, before calling `llama_params_fit`, set the target based on the SYCL backend's headroom:

```cpp
    if (params.fit_params) {
#ifdef GGML_USE_SYCL
        // Use unified cache's VRAM headroom as the fit_params margin
        size_t sycl_headroom = ggml_backend_sycl_get_vram_headroom(params.main_gpu);
        if (sycl_headroom > 0) {
            params.fit_params_target = sycl_headroom;
        }
#endif
        LOG_INF("%s: fitting params to device memory (margin=%.1f MB)\n", __func__,
                params.fit_params_target / (1024.0 * 1024.0));
        llama_params_fit(...);
    }
```

### Step 3: Handle GGML_SYCL_VRAM_BUDGET_PCT in budget calculation

The existing `GGML_SYCL_VRAM_BUDGET_PCT` env var (commit 7ef88c72a) overrides the budget in `unified-cache.cpp`. The headroom export must respect this:

In the unified cache, add:
```cpp
size_t unified_cache_get_budget_pct() {
    static size_t pct = 0;
    static bool initialized = false;
    if (!initialized) {
        const char * env = getenv("GGML_SYCL_VRAM_BUDGET_PCT");
        pct = env ? std::atoi(env) : 90;
        pct = std::min(pct, size_t(100));
        initialized = true;
    }
    return pct;
}
```

### Step 4: Build and test

```bash
ninja -C build -j $(nproc)
```

Normal model:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68 (all layers on GPU, graphs+persistent TG active)

Low VRAM (CPU offload):
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Correct output with reduced n_gpu_layers. Some layers on CPU.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/unified-cache.hpp \
        ggml/src/ggml-sycl/unified-cache.cpp common/common.cpp
git commit -m "sycl: align llama_params_fit margin with unified cache VRAM budget"
```

**Notes for implementer:**
- The `GGML_SYCL_VRAM_BUDGET_PCT` env var must be respected by both the unified cache AND the fit_params margin. A single source of truth (the new `unified_cache_get_budget_pct()` function) ensures they agree.
- `llama_params_fit` runs BEFORE model load, so it can't query the unified cache instance. It queries the budget FORMULA (percentage + headroom), not the cache state.
- When `GGML_SYCL_VRAM_BUDGET_PCT=40`, fit_params should set n_gpu_layers such that only 40% of VRAM is used for weights. The remaining layers go to CPU.

---

## Task 3: Track CPU-Resident Weights in Unified Cache

**Depends on:** Task 2
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (add CPU tracking category)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (implement tracking)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (report CPU weights in budget diagnostics)

**Description:**

The unified cache should be aware of ALL model weights, including those assigned to CPU by `llama_params_fit`. This doesn't mean the unified cache ALLOCATES CPU memory — the CPU backend handles that. Instead, the cache maintains a ledger of CPU-resident weights for budget reporting and diagnostics.

**Acceptance Criteria:**
- [ ] Unified cache tracks CPU-resident weight count and total size
- [ ] `unified_cache_log_budget_summary()` shows CPU vs GPU weight breakdown
- [ ] `set_tensor_inventory()` records which weights are CPU-resident based on n_gpu_layers
- [ ] No functional changes — this is purely accounting/diagnostics
- [ ] Build succeeds

**Implementation Guide:**

### Step 1: Add CPU weight tracking to unified cache

In `unified-cache.hpp`, add to the runtime tracking:

```cpp
// CPU-resident weight tracking (informational only — CPU backend owns the memory)
struct cpu_weight_info {
    size_t count       = 0;
    size_t total_bytes = 0;
};
```

Add a free function:
```cpp
void unified_cache_record_cpu_weights(size_t count, size_t total_bytes);
cpu_weight_info unified_cache_get_cpu_weights();
```

### Step 2: Record CPU weights in set_tensor_inventory

In `ggml-sycl.cpp`, in `set_tensor_inventory()` after computing `model_exceeds_vram`, also compute CPU weight stats based on the current `n_gpu_layers` (which has been set by `llama_params_fit` at this point):

```cpp
    // Count weights assigned to CPU (layers beyond n_gpu_layers)
    // n_gpu_layers is available from the backend context
    size_t cpu_weight_count = 0;
    size_t cpu_weight_bytes = 0;
    int n_gpu = ctx->n_gpu_layers;  // need to pass this through
    for (size_t i = 0; i < inventory->count; i++) {
        int layer_id = ggml_sycl::extract_layer_id(inventory->tensors[i].name);
        if (layer_id >= 0 && layer_id < (int(hparams_n_layer) - n_gpu)) {
            cpu_weight_count++;
            cpu_weight_bytes += inventory->tensors[i].size;
        }
    }
    ggml_sycl::unified_cache_record_cpu_weights(cpu_weight_count, cpu_weight_bytes);
```

### Step 3: Update budget diagnostics

In `unified_cache_log_budget_summary()`, add CPU weight info:

```cpp
    auto cpu_info = unified_cache_get_cpu_weights();
    if (cpu_info.count > 0) {
        GGML_LOG_INFO("[UNIFIED-CACHE] CPU-resident weights: %zu tensors, %.1f MB\n",
                      cpu_info.count, cpu_info.total_bytes / (1024.0 * 1024.0));
    }
```

### Step 4: Build and test

```bash
ninja -C build -j $(nproc)
```

Low VRAM test — should see CPU weight stats in output:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Log shows "CPU-resident weights: N tensors, X MB"

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp \
        ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: track CPU-resident weights in unified cache budget diagnostics"
```

---

## Task 4: Integration Test + Benchmark vs GPU Streaming

**Depends on:** Task 3
**File scope:**
- No code changes (testing only)

**Description:**

Comprehensive verification and benchmark comparison between CPU offload and GPU streaming approaches.

**Acceptance Criteria:**
- [ ] Mistral Q4_0 at 40% VRAM produces correct deterministic output via CPU offload
- [ ] Mistral Q4_0 at 100% VRAM (default) has no perf regression
- [ ] TG throughput measured for CPU offload at various VRAM budgets
- [ ] Side-by-side comparison with GPU streaming results (from other plan)
- [ ] GPT-OSS 20B test (naturally exceeds VRAM)

**Implementation Guide:**

### Test 1: Normal model (no offload)

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68. All layers on GPU.

### Test 2: Low VRAM correctness (CPU offload active)

```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, ...` (correct output)

### Test 3: CPU offload TG benchmark

```bash
GGML_SYCL_VRAM_BUDGET_PCT=40 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Record PP512 and TG128 numbers. Compare with GPU streaming results.

### Test 4: Various VRAM budgets

Test at 20%, 40%, 60%, 80% VRAM budget and record TG throughput for each.

### Test 5: GPT-OSS 20B (naturally exceeds VRAM)

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 'Hello, my name is' -n 10 --seed 42 --temp 0
```

### Comparison Table (fill in results)

| Metric | GPU Streaming | CPU Offload | Ratio |
|--------|-------------|-------------|-------|
| TG128 (40% VRAM) | ? tok/s | ? tok/s | ? |
| PP512 (40% VRAM) | ? tok/s | ? tok/s | ? |
| TG128 (100% VRAM) | ~70 tok/s | ~70 tok/s | ~1.0 |
| PP512 (100% VRAM) | ~1240 tok/s | ~1240 tok/s | ~1.0 |
| Implementation complexity | 5 tasks, 2 new files | 3 tasks, 0 new files | — |

---

## Key Design Decisions

1. **Unified cache as authority**: The unified cache's VRAM budget formula is THE source of truth. `llama_params_fit` uses the same formula (via exported headroom function) to determine layer assignment.

2. **CPU backend handles CPU layers**: The unified cache doesn't allocate CPU weight memory — the CPU backend does that through its normal buffer allocator. The unified cache tracks CPU weights for accounting only.

3. **No streaming needed**: CPU offload eliminates the need for host→device DMA streaming. GPU layers run at full speed with graph replay and persistent TG. CPU layers run at host memory bandwidth.

4. **Compatible with GPU streaming**: If GPU streaming is also implemented, the two approaches are complementary. CPU offload is the fast default. GPU streaming could be an optional mode for cases where CPU is overloaded (e.g., running multiple models).

5. **`GGML_SYCL_VRAM_BUDGET_PCT` controls both**: The single env var controls both the unified cache budget AND the `llama_params_fit` margin, ensuring they always agree.
