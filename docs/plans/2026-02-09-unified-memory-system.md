# Unified Memory System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Integrate all weight placement strategies (VRAM-only, CPU offload, MoE expert caching, sub-layer offload, GPU streaming) into one graduated system where the unified cache is the single authority for all memory decisions.

**Architecture:** The unified cache calculates a `unified_memory_plan` at model load time, choosing the least-invasive strategy from a 6-level pressure hierarchy. `llama_params_fit` receives the budget from the unified cache (not from raw device memory), so both systems agree on placement. MoE models benefit from expert-level caching via the existing `prestage_routed_experts` infrastructure. GPU weight streaming is the last resort, activated only when CPU offload is insufficient.

**Tech Stack:** C++17, existing `llama_params_fit`, `ggml_backend_sched`, unified cache API, `layer_stream_manager`, MoE profiling (`llama_load_moe_profile`)

**Beads Issue:** llama.cpp-i7zx (host weight streaming broken)

**Research Inspiration:**
- **OD-MoE** (Dec 2025): On-demand expert loading with look-ahead prediction — 99.94% accuracy, 75% of full-GPU speed at 1/3 memory. Key idea: predict which experts will be needed multiple layers ahead and start loading them asynchronously.
- **MoEpic** (Sep 2025): Vertical expert splitting — cache the "top segment" of hot experts to fit more experts partially in VRAM. Divide-and-conquer algorithm for adaptive cache configuration.
- **PIPO** (Apr 2025): Pipelined offloading — overlap compute and transfer to keep GPU utilization >90%. Achieved 3.1x throughput vs naive offloading on 6 GB GPU.
- **NEO** (MLSys 2025): CPU offloading for online LLM inference — optimized CPU kernel dispatch for bandwidth-bound layers.

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 3, 5 | Budget export API + MoE-aware budget + integration test |
| B | 2, 4 | Re-enable fit_params + wire GPU streaming dispatch stubs |

### Dependency Graph

```
Task 1 (Track A, no deps)       — Unified cache budget export API
Task 2 (Track B, no deps)       — Re-enable fit_params with unified cache budget
Task 3 (Track A, depends on 1)  — MoE-aware budget calculation
Task 4 (Track B, depends on 1, 2) — Wire GPU streaming into tiered dispatch stubs
Task 5 (depends on 2, 3, 4)    — Integration test: pressure hierarchy + benchmarks
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/unified-cache.hpp` | 1, 3 | Sequential (same track) |
| `ggml/src/ggml-sycl/unified-cache.cpp` | 1, 3 | Sequential (same track) |
| `common/common.h` | 2 | None (1 line) |
| `common/common.cpp` | 2 | None (isolated section) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (set_tensor_inventory ~2025-2125) | 1 | None (budget export) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (tiered dispatch stubs ~11967, ~15163, ~19421, ~19459, ~20886) | 4 | None (isolated stubs) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph gating ~29504, ~29779) | 4 | None (isolated section) |

---

## Background: Existing Infrastructure

### Unified Cache Budget System (`unified-cache.hpp:1007-1049`)

The unified cache already has:
- `alloc_hint` enum: WEIGHT, COMPUTE, EPHEMERAL, PERSISTENT, DEBUG
- `runtime_category` enum: KV_CACHE, COMPUTE, STAGING, GRAPH, OTHER
- `unified_cache_add_runtime_bytes(device, bytes, category)` — track non-weight VRAM
- `unified_cache_available_for_compute(device)` — query available VRAM after weights+runtime
- `unified_cache_total_managed(device)` — raw VRAM budget
- `unified_cache_weight_bytes(device)` — current weight bytes on device
- `unified_cache_log_budget_summary(device)` — diagnostic output
- Budget calculation: `base_mem * pct/100` minus `max(256MB, total/10)` headroom (`unified-cache.cpp:4386-4436`)
- `GGML_SYCL_VRAM_BUDGET_PCT` env var override

### llama_params_fit (`src/llama.cpp:143-689`)

A sophisticated 4-step algorithm:
1. **Step 1**: Check if model fits at all (query `llama_get_device_memory_data`)
2. **Step 2**: Reduce context size (`hp_nct` → `n_ctx_min`, lines 228-276)
3. **Step 3**: Iteratively reduce `n_gpu_layers` with binary search (lines 490-557)
4. **Step 4** (MoE): Convert dense-only layers to full layers (lines 559-689)

Key patterns used for sub-layer offload (`get_overflow_pattern`, lines 312-349):
- `LAYER_FRACTION_ATTN`: offloads `blk.N.ffn_(up|gate|down).*` to CPU (keep attention on GPU)
- `LAYER_FRACTION_UP`: offloads `blk.N.ffn_(gate|down).*`
- `LAYER_FRACTION_GATE`: offloads `blk.N.ffn_down.*`
- `LAYER_FRACTION_MOE`: offloads `blk.N.ffn_(up|down|gate)_(ch|)exps` (MoE experts to CPU)

### Current Branch State

`common/common.h:335`: `bool fit_params = false;` — SYCL branch disabled it because unified cache handles memory.

`ggml-sycl.cpp:2093-2094`: Sets `g_model_exceeds_vram` based on tensor inventory vs VRAM budget. This flag currently gates:
- Host placement for new weights
- 5 tiered dispatch stubs that are NOPs ("`Future: use cached_ptr`")
- Graph replay disable (`ggml-sycl.cpp:29504`)
- Persistent TG disable (`ggml-sycl.cpp:29779`)

### MoE Expert Infrastructure

- **Routing-aware pre-staging** (`unified-cache.hpp:1117-1174`): `prestage_routed_experts()` and `unpin_routed_experts()` — but DISABLED at line 22172: `const bool use_routing_prestage = false; // DISABLED: cache key mismatch causes DEVICE_LOST`
- **MoE profile system**: `moe_profile_path`, `moe_warmup_tokens`, `moe_gpu_fraction` in `common_params`. APIs: `llama_load_moe_profile`, `llama_analyze_moe_profile`, `llama_save_moe_profile` in `llama-context.cpp`
- **Expert staging buffers**: `g_expert_staging` map (`ggml-sycl.cpp:2506`), per-expert device/host staging in MUL_MAT_ID path

### Layer Streaming Infrastructure

- `layer-streaming.hpp/cpp` (NEW, commit 00b844cda): `layer_stream_manager` with double-buffered device memory, `build_layer_map`, `allocate_buffers`, `ensure_layer`, `prefetch_next_layer`
- `layer_prefetch_tracker` (`layer-prefetch.hpp:32-135`): Detects layer transitions by parsing tensor names
- Integration point: `set_tensor_inventory` already calls `get_layer_stream_manager` when model exceeds VRAM (`ggml-sycl.cpp:2098-2108`)

---

## The Pressure Hierarchy

When loading a model, the unified cache evaluates strategies in order from least to most invasive:

```
Level 0: Everything in VRAM           → full speed (~70 TG, ~1240 PP)
Level 1: Context reduction             → reduce n_ctx to free VRAM for weights
Level 2: MoE expert offload (MoE only) → keep attention + active experts, stage rest
Level 3: Full-layer CPU offload        → overflow layers on CPU backend (~20 tok/s)
Level 4: Sub-layer CPU offload         → partial layers on CPU (attn-only GPU, FFN on CPU)
Level 5: GPU weight streaming          → double-buffered DMA (~3.8 tok/s, last resort)
```

---

## Task 1: Unified Cache Budget Export API

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (add ~30 lines)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (add ~80 lines)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:2025-2125` (add ~20 lines, compute + export budget)

**Description:**

Add a public API that exports the unified cache's VRAM budget so external consumers (like `llama_params_fit`) can use it instead of independently querying raw device memory. This is the foundation for the unified memory system — one budget, one authority.

**Acceptance Criteria:**

- [ ] New struct `unified_budget_info` with fields: `total_vram`, `budget_bytes`, `weight_bytes`, `runtime_bytes`, `available_for_weights`, `budget_pct`, `device_id`
- [ ] New function `unified_cache_get_budget_info(device)` returns this struct
- [ ] New function `unified_cache_get_margin_bytes(device)` returns how many bytes are available as margin for `llama_params_fit` (= budget - weight_bytes - runtime_bytes)
- [ ] Budget info is populated during `set_tensor_inventory` and updated when runtime bytes change
- [ ] `unified_cache_log_budget_summary` enhanced to include the new fields
- [ ] Build succeeds with zero new warnings
- [ ] Existing benchmarks unchanged (PP512 >= 1200, TG128 >= 68)

**Implementation Guide:**

1. **Add the budget info struct to `unified-cache.hpp`** after line 1049 (after `unified_cache_log_budget_summary`):

```cpp
// Budget information exported for external consumers (e.g., llama_params_fit)
struct unified_budget_info {
    int    device_id;
    size_t total_vram;           // Total device memory
    size_t budget_bytes;         // Managed budget (total * pct - headroom)
    size_t weight_bytes;         // Current weight cache usage
    size_t runtime_bytes;        // KV + compute + staging + graph
    size_t available_for_weights; // budget - runtime (what can hold weights)
    int    budget_pct;           // GGML_SYCL_VRAM_BUDGET_PCT value used
    bool   model_exceeds_vram;   // True if model > available_for_weights
};

// Get budget info for a device (thread-safe snapshot)
unified_budget_info unified_cache_get_budget_info(int device);

// Get margin in bytes for llama_params_fit (how much free space after weights + runtime)
// Returns 0 if budget is exceeded
size_t unified_cache_get_margin_bytes(int device);
```

2. **Implement in `unified-cache.cpp`** after `unified_cache_log_budget_summary` (~line 4840):

```cpp
unified_budget_info unified_cache_get_budget_info(int device) {
    unified_budget_info info = {};
    info.device_id = device;

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(device, &free_mem, &total_mem);
    info.total_vram = ggml_sycl_info().devices[device].total_vram;
    if (info.total_vram == 0) {
        info.total_vram = total_mem > 0 ? total_mem : free_mem;
    }

    auto * cache = get_unified_cache_for_device(device);
    if (cache) {
        info.budget_bytes   = unified_cache_total_managed(device);
        info.weight_bytes   = unified_cache_weight_bytes(device);
        info.runtime_bytes  = unified_cache_get_runtime_bytes(device);
        info.available_for_weights = info.budget_bytes > info.runtime_bytes
                                       ? info.budget_bytes - info.runtime_bytes : 0;
    } else {
        // Cache not yet initialized — use raw calculation
        int pct = 90;
        const char * env_pct = getenv("GGML_SYCL_VRAM_BUDGET_PCT");
        if (env_pct) pct = std::atoi(env_pct);
        pct = std::max(1, std::min(100, pct));

        info.budget_bytes = static_cast<size_t>(info.total_vram * (static_cast<double>(pct) / 100.0));
        const size_t headroom = std::max(size_t(256) << 20, info.total_vram / 10);
        if (info.total_vram > headroom && info.budget_bytes > info.total_vram - headroom) {
            info.budget_bytes = info.total_vram - headroom;
        }
        info.available_for_weights = info.budget_bytes;
    }

    info.budget_pct = 90;
    const char * env_pct = getenv("GGML_SYCL_VRAM_BUDGET_PCT");
    if (env_pct) info.budget_pct = std::atoi(env_pct);

    info.model_exceeds_vram = g_model_exceeds_vram.load(std::memory_order_acquire);

    return info;
}

size_t unified_cache_get_margin_bytes(int device) {
    auto info = unified_cache_get_budget_info(device);
    if (info.available_for_weights > info.weight_bytes) {
        return info.available_for_weights - info.weight_bytes;
    }
    return 0;
}
```

Note: `g_model_exceeds_vram` is declared `extern` in `ggml-sycl.cpp:155`. Add an accessor or include the extern declaration.

3. **Export budget via ggml backend API** — Add a C-linkage accessor in `ggml-sycl.cpp` after `ggml_backend_sycl_model_exceeds_vram` (~line 2210):

```cpp
// Export unified cache budget for llama_params_fit integration
size_t ggml_backend_sycl_get_vram_budget(ggml_backend_t backend) {
    if (!backend) return 0;
    ggml_backend_sycl_context * ctx = (ggml_backend_sycl_context *) backend->context;
    if (!ctx) return 0;
    auto info = ggml_sycl::unified_cache_get_budget_info(ctx->device);
    return info.available_for_weights;
}

size_t ggml_backend_sycl_get_vram_margin(ggml_backend_t backend) {
    if (!backend) return 0;
    ggml_backend_sycl_context * ctx = (ggml_backend_sycl_context *) backend->context;
    if (!ctx) return 0;
    return ggml_sycl::unified_cache_get_margin_bytes(ctx->device);
}
```

Declare these in `ggml-sycl.h`.

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp \
        ggml/src/ggml-sycl/ggml-sycl.cpp ggml/include/ggml-sycl.h
git commit -m "sycl: add unified cache budget export API for memory system integration"
```

**Notes for implementer:**
- The budget calculation in `unified-cache.cpp:4386-4436` is the canonical formula. Match it exactly in the pre-init fallback path.
- `g_model_exceeds_vram` is an `std::atomic<bool>` declared in `ggml-sycl.cpp:2000`. Use `extern std::atomic<bool> g_model_exceeds_vram;` or route through the existing accessor `ggml_backend_sycl_model_exceeds_vram()`.
- Don't add `#include "unified-cache.hpp"` to `ggml-sycl.h` — keep the C header clean. The C functions forward to C++ internally.

---

## Task 2: Re-enable fit_params with Unified Cache Budget

**Track:** B
**Depends on:** None (can start immediately, but integration with Task 1's budget API improves it)
**File scope:**
- Modify: `common/common.h:335` (1 line change)
- Modify: `common/common.cpp:1348-1363` (add ~15 lines, budget alignment)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:2093-2108` (add ~10 lines, conditional streaming)

**Description:**

Re-enable `llama_params_fit` for the SYCL backend so models exceeding VRAM automatically get optimal `n_gpu_layers`. The key change: set `fit_params = true` but use the unified cache's budget as the margin source instead of raw `ggml_backend_dev_memory`. When fit_params determines the right n_gpu_layers, only the layers that genuinely overflow go to CPU — GPU layers keep full graph replay, persistent TG, and SOA layout.

**Acceptance Criteria:**

- [ ] `common/common.h:335`: `fit_params` changed from `false` to `true`
- [ ] Models that fit in VRAM: zero behavior change (n_gpu_layers = all, no CPU layers)
- [ ] Models exceeding VRAM: `llama_params_fit` reduces n_gpu_layers appropriately
- [ ] `g_model_exceeds_vram` remains `false` when fit_params already offloaded overflow (weights fit after offload)
- [ ] Layer streaming NOT initialized when fit_params handles the overflow
- [ ] Build succeeds with zero new warnings
- [ ] Mistral 7B Q4_0 (3.9 GB, fits in 12 GB VRAM): PP512 >= 1200, TG128 >= 68

**Implementation Guide:**

1. **Change `fit_params` default** in `common/common.h:335`:

```cpp
// OLD:
bool fit_params = false;  // disabled by default - SYCL tiered memory handles large models
// NEW:
bool fit_params = true;   // auto-fit model to VRAM via llama_params_fit
```

2. **Add budget-aware margin** in `common/common.cpp` near line 1355 (inside `common_init_from_params`, before the `llama_params_fit` call):

Find the section that calls `llama_params_fit` and ensure the margin (`fit_params_target`) accounts for the unified cache's overhead. The existing margin is 1 GB by default, which is reasonable. No change needed to the margin itself, but add a log:

```cpp
if (params.fit_params) {
    LOG_INF("%s: fit_params enabled, margin=%.1f MB, min_ctx=%d\n",
            __func__, params.fit_params_target / (1024.0 * 1024.0), params.fit_params_min_ctx);
}
```

3. **Conditional layer streaming** in `ggml-sycl.cpp:2097-2108`:

The current code initializes layer streaming whenever `model_exceeds_vram` is true. But if `fit_params` has already reduced `n_gpu_layers`, the overflow layers won't be sent to this device at all — so streaming is unnecessary. Add a guard:

```cpp
// Initialize double-buffered layer streaming when model exceeds VRAM
// BUT only if fit_params didn't already handle the overflow via CPU offload
if (model_exceeds_vram) {
    // Check if fit_params will handle this (n_gpu_layers < total layers)
    // If so, don't initialize streaming — CPU offload is faster
    // The caller can still force streaming via GGML_SYCL_FORCE_STREAMING=1
    const char * force_stream = getenv("GGML_SYCL_FORCE_STREAMING");
    const bool streaming_forced = force_stream && std::atoi(force_stream) == 1;
    if (streaming_forced) {
        auto & mgr = ggml_sycl::get_layer_stream_manager(ctx->device);
        mgr.build_layer_map(g_tensor_inventory.data(), g_tensor_inventory.size());
        sycl::queue & q = ggml_sycl_get_device(ctx->device).default_queue();
        if (mgr.allocate_buffers(q)) {
            GGML_LOG_INFO("[SYCL-BUDGET] Layer streaming enabled (forced): %d layers, 2 x %.1f MB buffers\n",
                          mgr.n_layers(), mgr.max_layer_size() / (1024.0 * 1024.0));
        } else {
            GGML_LOG_ERROR("[SYCL-BUDGET] Layer streaming buffer allocation failed\n");
        }
    } else {
        GGML_LOG_INFO("[SYCL-BUDGET] Model exceeds VRAM by %.1f MB — CPU offload via fit_params preferred over streaming\n",
                      (g_tensor_inventory_total_size - vram_budget) / (1024.0 * 1024.0));
    }
}
```

4. **Ensure g_model_exceeds_vram stays false for offloaded models**: After fit_params runs, only the layers assigned to this device will go through `set_tensor_inventory`. If fit_params offloads enough layers, the remaining weights fit in VRAM, so `g_tensor_inventory_total_size <= vram_budget` → `g_model_exceeds_vram = false` → graph replay and persistent TG stay active for the GPU layers. **No code change needed** — this is already the correct behavior because set_tensor_inventory only sees the weights that actually get sent to the device.

**Commit:**
```bash
git add common/common.h common/common.cpp ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: re-enable fit_params for CPU offload of overflow layers"
```

**Notes for implementer:**
- Verify by running with a model that fits (Mistral 7B Q4_0 at 3.9 GB): should see `fit_params` log "no changes needed" and performance identical to current.
- If we had a model that doesn't fit (GPT-OSS 20B at 11.3 GB on 12 GB VRAM), fit_params would reduce n_gpu_layers. We can test with `GGML_SYCL_VRAM_BUDGET_PCT=30` to simulate VRAM pressure on Mistral 7B.
- `GGML_SYCL_FORCE_STREAMING=1` env var lets users force GPU streaming even when CPU offload is available (for benchmarking/testing).

---

## Task 3: MoE-Aware Budget Calculation

**Track:** A
**Depends on:** Task 1
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (add ~20 lines)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (add ~100 lines)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:2031-2060` (add ~30 lines, MoE detection in inventory)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:22170-22200` (enable routing prestage, ~5 lines)

**Description:**

For MoE models, a large fraction of weights are expert FFN parameters that are only needed when the router activates them (typically 2 out of 8, or 8 out of 128). The unified cache should recognize MoE expert tensors in the tensor inventory and calculate a more accurate budget that only accounts for the *active* expert fraction, rather than treating all expert weights as mandatory.

This task also re-enables the routing-aware pre-staging that was disabled due to a cache key mismatch bug (line 22172). The fix involves ensuring cache keys use per-expert granularity rather than per-tensor.

**Acceptance Criteria:**

- [ ] New fields in `unified_budget_info`: `n_expert_total`, `n_expert_used`, `expert_weight_bytes`, `active_expert_bytes`
- [ ] New function `compute_moe_budget()` that estimates VRAM needed with only active experts loaded
- [ ] `g_model_exceeds_vram` accounts for MoE expert savings (model may fit if only active experts are in VRAM)
- [ ] Routing-aware pre-staging re-enabled (remove `const bool use_routing_prestage = false;` at line 22172)
- [ ] Cache key mismatch fixed — each expert gets a unique key incorporating expert index
- [ ] Build succeeds with zero new warnings
- [ ] Existing non-MoE benchmarks unchanged (PP512 >= 1200, TG128 >= 68)

**Implementation Guide:**

1. **Detect MoE tensors during `set_tensor_inventory`** — After building the inventory (line 2060), scan for expert patterns:

```cpp
// Detect MoE expert tensors and calculate expert memory fraction
size_t moe_expert_total_bytes = 0;
int    moe_n_experts_detected = 0;
int    moe_n_experts_used     = 0;  // From model hparams, set later
std::regex expert_pattern("blk\\.\\d+\\.ffn_(up|down|gate)_(ch|)exps");
for (const auto & [name, size] : g_tensor_inventory) {
    if (std::regex_search(name, expert_pattern)) {
        moe_expert_total_bytes += size;
    }
}
// Store for budget calculation
g_moe_expert_total_bytes = moe_expert_total_bytes;
```

Add global: `static size_t g_moe_expert_total_bytes = 0;`

2. **Add MoE fields to `unified_budget_info`** in `unified-cache.hpp`:

```cpp
// MoE expert breakdown (non-zero only for MoE models)
size_t expert_weight_bytes;      // Total bytes for ALL expert tensors
size_t active_expert_bytes;      // Estimated bytes for active experts only
int    n_expert_total;           // Total experts per layer (e.g., 8, 128)
int    n_expert_used;            // Experts per token (e.g., 2, 4)
```

3. **Compute MoE-adjusted budget** — Add a helper function in `unified-cache.cpp`:

```cpp
// Calculate effective weight bytes accounting for MoE expert sparsity
// For an 8-expert top-2 model, only 25% of expert weights are needed at any time
static size_t compute_moe_effective_weight_bytes(size_t total_weight_bytes,
                                                  size_t expert_total_bytes,
                                                  int n_expert, int n_expert_used) {
    if (n_expert <= 0 || n_expert_used <= 0 || expert_total_bytes == 0) {
        return total_weight_bytes;  // Dense model, no savings
    }
    // Active expert fraction + some headroom for expert cache churn
    // Use 1.5x active ratio to account for recently-used experts still in cache
    double active_ratio = static_cast<double>(n_expert_used) / n_expert;
    double effective_ratio = std::min(1.0, active_ratio * 1.5);
    size_t expert_savings = static_cast<size_t>(expert_total_bytes * (1.0 - effective_ratio));
    return total_weight_bytes - expert_savings;
}
```

4. **Update `g_model_exceeds_vram` to use MoE-adjusted size** — In `set_tensor_inventory` at line 2093:

```cpp
// For MoE models, use effective weight size (only active experts needed in VRAM)
const size_t effective_model_size = compute_moe_effective_weight_bytes(
    g_tensor_inventory_total_size, g_moe_expert_total_bytes,
    g_moe_n_experts_total, g_moe_n_experts_used);
const bool model_exceeds_vram = effective_model_size > vram_budget;
```

Note: `g_moe_n_experts_total` and `g_moe_n_experts_used` need to come from model hparams. They can be passed via the tensor inventory or added as fields to `ggml_sycl_tensor_inventory`. Check where `set_tensor_inventory` is called from — it's from `llama_model_load_from_file` after parsing hparams.

5. **Re-enable routing-aware pre-staging** at `ggml-sycl.cpp:22172`:

```cpp
// OLD:
const bool use_routing_prestage = false; // DISABLED: cache key mismatch causes DEVICE_LOST
// NEW:
const bool use_routing_prestage = (n_experts > 64);  // Enable for large MoE models
```

6. **Fix cache key mismatch** — The bug that caused DEVICE_LOST was in how cache keys are generated for individual experts. Each expert slice needs a unique key incorporating the expert index. Search for how `prestage_routed_experts` generates cache keys and ensure the expert ID is part of the key hash. Look in `unified-cache.cpp` for the prestage implementation.

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp \
        ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: MoE-aware budget calculation with routing-aware expert pre-staging"
```

**Notes for implementer:**
- The `n_expert` and `n_expert_used` hparams are available in `llama_hparams`. Check how `set_tensor_inventory` gets called — the caller may need to pass these values.
- The cache key mismatch bug needs careful investigation. Read `prestage_routed_experts` in `unified-cache.cpp` to understand the current key generation. The issue is likely that expert slices reuse the parent tensor's cache key without differentiating by expert index.
- MoEpic's "vertical expert splitting" idea — caching just the top segment of hot experts — is interesting but too complex for this task. Note it as future work.
- Test with `GGML_SYCL_VRAM_BUDGET_PCT=80` on GPT-OSS 20B if available.

---

## Task 4: Wire GPU Streaming into Tiered Dispatch Stubs

**Track:** B
**Depends on:** Task 1, Task 2
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (5 tiered dispatch stubs, ~20 lines each)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph compute, ~10 lines)

**Description:**

Complete the GPU weight streaming path by wiring the `layer_stream_manager` into the 5 tiered dispatch stubs that currently are NOPs. When `GGML_SYCL_FORCE_STREAMING=1` is set and the model exceeds VRAM, the dispatch stubs use the layer stream manager to ensure weights are in a device buffer before kernel execution. Graph replay and persistent TG are disabled in this mode.

This task completes the Layer 5 (streaming) fallback of the pressure hierarchy. It's the "last resort" path — CPU offload (Task 2) is preferred for normal operation.

**Acceptance Criteria:**

- [ ] All 5 tiered dispatch stubs wired to `layer_stream_manager::ensure_layer()` + `get_buffer()`
- [ ] Stubs call `prefetch_next_layer()` after launching compute for current layer
- [ ] Graph replay disabled when streaming active (`g_model_exceeds_vram && streaming buffers allocated`)
- [ ] Persistent TG disabled when streaming active
- [ ] `GGML_SYCL_FORCE_STREAMING=1` forces streaming path (for testing/benchmarking)
- [ ] Build succeeds with zero new warnings
- [ ] Models that fit: zero behavior change (streaming not activated)

**Implementation Guide:**

1. **Identify the 5 dispatch stubs** — These are the NOPs in `ggml-sycl.cpp` that check `g_model_exceeds_vram`:

   - Line ~11967: `ggml_sycl_op_mul_mat_sycl` (BLAS/oneDNN path)
   - Line ~15163: `ggml_sycl_op_mul_mat_vec_q` (MMVQ path)
   - Line ~19421: `ggml_sycl_mul_mat` (main dispatch, batch=1 check)
   - Line ~19459: `ggml_sycl_mul_mat` (main dispatch, model exceeds check)
   - Line ~20886: `ggml_sycl_mul_mat_id` (MoE MUL_MAT_ID path)

2. **Add streaming helper** — Create a helper function near the top of the file:

```cpp
// Resolve weight pointer through layer streaming when active
// Returns device pointer to weight data, or nullptr if streaming not active
static void * resolve_streaming_weight(ggml_backend_sycl_context & ctx,
                                        const ggml_tensor * weight,
                                        sycl::queue & stream) {
    if (!g_model_exceeds_vram.load(std::memory_order_acquire)) {
        return nullptr;  // Model fits, no streaming needed
    }

    auto & mgr = ggml_sycl::get_layer_stream_manager(ctx.device);
    if (!mgr.is_allocated()) {
        return nullptr;  // Streaming buffers not allocated
    }

    // Extract layer ID from tensor name
    int layer_id = ggml_sycl::extract_layer_id(weight->name);
    if (layer_id < 0) {
        return nullptr;  // Non-layer tensor (embedding, output)
    }

    // Ensure layer is loaded into a device buffer, blocking if DMA in progress
    mgr.ensure_layer(layer_id, stream);

    // Get device pointer for this tensor within the buffer
    return mgr.get_tensor_ptr(weight->name);
}
```

3. **Wire each stub** — For each of the 5 dispatch sites, replace the NOP comment with:

```cpp
// Try layer streaming first
void * streamed_ptr = resolve_streaming_weight(ctx, src0, *stream);
if (streamed_ptr) {
    // Use streamed pointer for weight data
    src0_dd = (const char *) streamed_ptr;

    // Prefetch next layer asynchronously
    int next_layer = ggml_sycl::extract_layer_id(src0->name) + 1;
    mgr.prefetch_next_layer(next_layer, *stream);
} else {
    // Fall back to unified cache pointer resolution
    // (existing code path)
}
```

The exact variable names (`src0_dd`, etc.) vary per stub — read each stub to match the right variable.

4. **Graph/persistent TG gating** — At `ggml-sycl.cpp:29504` (graph replay check) and `29779` (persistent TG check), add streaming check:

```cpp
// Disable graph replay when streaming is active (buffer pointers rotate per token)
auto & mgr = ggml_sycl::get_layer_stream_manager(sycl_ctx->device);
if (g_model_exceeds_vram.load(std::memory_order_acquire) && mgr.is_allocated()) {
    GGML_SYCL_DEBUG("[GRAPH] Skipping graph replay — layer streaming active\n");
    // Fall through to per-op dispatch
}
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: wire layer streaming into tiered dispatch stubs for GPU weight streaming"
```

**Notes for implementer:**
- The stubs have DIFFERENT variable names for the weight pointer in each context. Read each stub carefully.
- `extract_layer_id()` is in `tensor-types.hpp:139` and parses `blk.N.` prefix.
- `layer_stream_manager::get_tensor_ptr()` may not exist yet — if not, it needs to be added (compute offset within the buffer based on tensor's position in the layer map).
- The streaming path is intentionally behind `GGML_SYCL_FORCE_STREAMING=1` for now. In the default flow, fit_params handles overflow via CPU offload (Task 2).
- For testing, use `GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_FORCE_STREAMING=1` to force streaming on Mistral 7B.

---

## Task 5: Integration Test — Pressure Hierarchy + Benchmarks

**Track:** — (convergence point)
**Depends on:** Task 2, Task 3, Task 4
**File scope:**
- No code changes (test-only task)

**Description:**

Verify the complete pressure hierarchy works correctly and benchmark all paths:
1. Level 0: Model fits in VRAM — full speed, no changes
2. Level 1: Context reduction (simulated via `GGML_SYCL_VRAM_BUDGET_PCT`)
3. Level 3: CPU offload via fit_params
4. Level 5: GPU streaming via `GGML_SYCL_FORCE_STREAMING=1`

**Acceptance Criteria:**

- [ ] Level 0: Mistral 7B Q4_0 at default budget → PP512 >= 1200, TG128 >= 68
- [ ] Level 3: Mistral 7B Q4_0 at VRAM_BUDGET_PCT=30 → CPU offload, TG > 0, no crash
- [ ] Level 5: Mistral 7B Q4_0 at VRAM_BUDGET_PCT=30 + FORCE_STREAMING=1 → streaming, TG > 0
- [ ] Correctness: All paths produce identical output for `'1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0`
- [ ] Budget log output shows correct per-category breakdown
- [ ] No OOM, no DEVICE_LOST, no hangs across all test configurations

**Implementation Guide:**

1. **Test Level 0 (everything VRAM)**:
```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expected: PP512 >= 1200 tok/s, TG128 >= 68 tok/s
```

2. **Test correctness baseline**:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>/dev/null
# Record output
```

3. **Test Level 3 (CPU offload)**:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>&1 | tee /tmp/cpu_offload.log
# Verify: fit_params log shows reduced n_gpu_layers
# Verify: output matches baseline
# Verify: no crash/OOM
```

```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Record: PP and TG with CPU offload
```

4. **Test Level 5 (GPU streaming)**:
```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_FORCE_STREAMING=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>&1 | tee /tmp/streaming.log
# Verify: layer streaming log shows buffer allocation
# Verify: output matches baseline
# Verify: no crash/OOM
```

```bash
GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_FORCE_STREAMING=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Record: PP and TG with streaming
```

5. **Compare results** — Create summary table:

| Configuration | PP512 tok/s | TG128 tok/s | Notes |
|---------------|-------------|-------------|-------|
| Level 0 (100%) | ~1240 | ~70 | Baseline, all VRAM |
| Level 3 (30% CPU) | ??? | ??? | CPU offload via fit_params |
| Level 5 (30% stream) | ??? | ??? | GPU streaming via double-buffer |

6. **Verify budget diagnostics** — In each log file, check for:
- `[UNIFIED-CACHE] Device 0 (...)`: budget, free, total
- `[SYCL-BUDGET]`: model_exceeds_vram flag
- fit_params log: n_gpu_layers, context reduction
- Per-category breakdown from `unified_cache_log_budget_summary`

**Commit:** No code commit. Document results in a comment on the beads issue.

**Notes for implementer:**
- Wait 30-60 seconds between benchmark runs to avoid Arc B580 thermal throttling (can cause 29x slowdown).
- If CPU offload path crashes, check whether the CPU backend is included in the backend list. The backend scheduler needs both SYCL and CPU backends registered.
- If streaming path hangs, check that graph replay is properly disabled (look for `[GRAPH] Skipping graph replay` in logs).
- For MoE testing (if GPT-OSS 20B available): test with default budget to exercise MoE expert budget calculation from Task 3.

---

## Future Work (Not in This Plan)

These are documented in `docs/plans/2026-02-09-unified-memory-architecture.md`:

1. **Phase C: Unified Budget Authority** — Move the budget calculation entirely from `llama_params_fit` into the unified cache. The cache determines the optimal memory plan and exports it as `n_gpu_layers` + `tensor_buft_overrides`.

2. **Phase D: MoE Expert-Aware Budgeting** — Build on Task 3 with OD-MoE-inspired look-ahead prediction (predict experts needed 2-3 layers ahead, start async DMA). MoEpic's vertical expert splitting (cache top segment of hot experts).

3. **Phase E: Sub-Layer Granularity** — Generate `tensor_buft_overrides` patterns from unified cache analysis. Keep attention on GPU, offload FFN to CPU for partial-layer offload.

4. **Phase F: Adaptive Placement** — Runtime weight migration based on access patterns. Track per-tensor hotness, promote hot CPU tensors to GPU when headroom available, demote cold ones.

5. **Layer Prefetch Optimization** — OD-MoE-style async double-buffered prefetch for streaming path. Predict layer N+2 while computing N, DMA N+1 into alternate buffer.
