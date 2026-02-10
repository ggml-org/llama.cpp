# Unified Memory Architecture — Comprehensive Design

> **For Claude:** This is an architecture design document, not a direct implementation plan. Use it to inform team-driven-development plans for individual phases.

**Goal:** A single coherent system where the unified cache is the central authority for all weight memory management, integrating with `llama_params_fit`, CPU offloading, MoE expert caching, and sub-layer streaming into one graduated response to VRAM pressure.

**Principle:** The unified cache owns ALL memory decisions. `llama_params_fit` is a consumer of the cache's budget information, not an independent decision-maker.

---

## The Pressure Hierarchy

The unified cache responds to VRAM pressure through a graduated hierarchy of strategies. Each level is tried in order until the model fits.

```
Level 0: Everything in VRAM
  └─ All weights cached on device
  └─ Graph replay + persistent TG active
  └─ Full speed (~76 tok/s TG, ~1240 tok/s PP)

Level 1: KV cache offloading
  └─ Keep full context size, offload cold KV entries to pinned host memory
  └─ Hot window: recent N tokens' KV entries stay in VRAM
  └─ Cold tier: older entries evict to host, stream back via PCIe for attention
  └─ Prefetch: attention access is sequential, prefetch next cold chunk during compute
  └─ Preserves full context capability (no sequence length reduction)
  └─ New infrastructure: KV cache tier in unified cache
  └─ Inspired by: vLLM KV offloading connector (Jan 2026), InfiniGen

Level 2: MoE expert offload (MoE models only)
  └─ Move inactive expert FFN weights to CPU
  └─ Keep attention + routing + active experts on GPU
  └─ On-demand loading via staging pool when routing changes
  └─ For 8-expert top-2: saves ~75% of expert weight memory
  └─ GPU compute for all operations (loaded experts are in VRAM)
  └─ Existing infrastructure: staging_pool_ensure_loaded

Level 3: Full-layer CPU offload
  └─ Move bottom layers to CPU backend via n_gpu_layers
  └─ GPU layers: full speed (graphs, persistent TG)
  └─ CPU layers: host memory bandwidth (~20 tok/s for all-CPU)
  └─ Backend scheduler handles activation transfers at split
  └─ Existing infrastructure: llama_params_fit + ggml_backend_sched

Level 4: Sub-layer CPU offload
  └─ Move individual tensor groups to CPU (attention or FFN or both)
  └─ tensor_buft_overrides with LAYER_FRACTION_ATTN/UP/GATE/MOE
  └─ Finer control than full-layer offload
  └─ Existing infrastructure: llama_params_fit tensor_buft_overrides

Level 5: GPU weight streaming (last resort)
  └─ Double-buffered layer DMA from host to device
  └─ Disables graph replay + persistent TG
  └─ PCIe-bottlenecked (~3.8 tok/s for all-streamed)
  └─ New infrastructure: layer_stream_manager
  └─ Only useful when CPU is overloaded or for batch inference
```

### When Each Level is Used

| Model Type | Fits VRAM? | Strategy |
|-----------|-----------|----------|
| Mistral 7B Q4_0 (3.9 GB) | Yes (12 GB budget) | Level 0 |
| Mistral 7B Q8_0 (7.2 GB) | Yes (12 GB budget) | Level 0 |
| GPT-OSS 20B MXFP4 (11.3 GB) | Barely (KV to host frees VRAM) | Level 1 + Level 2 |
| Mixtral 8x7B Q4_0 (~26 GB) | No | Level 2 + Level 3 |
| Llama 70B Q4_0 (~40 GB) | No | Level 3 |
| GPT-OSS 120B MXFP4 (~60 GB) | No | Level 3 + Level 4 |

---

## Architecture

### Phase 1: Budget Calculation (Pre-Load)

The unified cache calculates the VRAM budget and determines the optimal strategy.

```
Input:
  - Tensor inventory (names, sizes, types)
  - Total VRAM (from ggml_backend_dev_memory)
  - GGML_SYCL_VRAM_BUDGET_PCT override
  - Model architecture info (n_layer, n_expert, expert_used_count)

Output:
  - n_gpu_layers (how many full layers on GPU)
  - tensor_buft_overrides (which sub-layer tensors go to CPU)
  - moe_policy (which experts to keep on GPU)
  - kv_offload_policy (hot window size, cold tier destination)
  - streaming_mode (enable layer streaming as last resort)
```

The budget calculation is a waterfall:

```
available_vram = total_vram * budget_pct / 100

1. Try Level 0: all weights in VRAM
   if total_weight_size <= available_vram → DONE (Level 0)

2. Try Level 1: KV cache offloading
   kv_size = compute_kv_size(n_ctx, n_layer, n_head, d_head)
   vram_after_kv_offload = available_vram + kv_size  // KV moves to host
   hot_window = min(n_ctx, available_vram_for_kv / kv_per_token)
   if total_weight_size <= vram_after_kv_offload → DONE (Level 1)
   // KV hot window stays in VRAM, cold entries stream from pinned host

3. Try Level 2: MoE expert offload (if n_expert > 0)
   active_expert_ratio = expert_used_count / n_expert
   moe_weight_savings = (1 - active_expert_ratio) * moe_weight_total
   if total_weight_size - moe_weight_savings <= available_vram → DONE (Level 2)

4. Try Level 3: full-layer CPU offload
   binary search for optimal n_gpu_layers where:
     embedding + output + sum(layer_size for top n_gpu_layers) <= available_vram
   if n_gpu_layers >= 1 → DONE (Level 3)

5. Try Level 4: sub-layer CPU offload
   for each layer in GPU set:
     try moving FFN to CPU (keep attention)
     recalculate fit
   DONE (Level 4) when it fits

6. Fallback: Level 5 streaming
   Set n_gpu_layers = 0, enable streaming
   All layers stream through double buffers
```

### Phase 2: Parameter Export (Pre-Load)

The unified cache exports its decisions to `llama_params_fit` format:

```cpp
struct unified_memory_plan {
    int32_t n_gpu_layers;
    uint32_t n_ctx;  // reduced context, or 0 for default
    std::vector<tensor_buft_override> overrides;  // sub-layer CPU assignments
    bool moe_expert_offload;  // enable on-demand expert loading
    bool streaming_enabled;   // Level 5 fallback
};
```

This plan is consumed by `common_init_from_params()` which feeds it into `mparams` and `cparams` before model load.

### Phase 3: Model Load

Based on the memory plan:
- GPU layers → SYCL backend buffer type → unified cache manages these
- CPU layers → CPU backend buffer type → CPU backend allocates, unified cache tracks
- MoE experts → initially on CPU, staging pool loads on-demand during inference

### Phase 4: Inference

```
For each token:
  For each layer L:
    if L is on GPU:
      → Normal dispatch (graph replay, persistent TG, SOA)
    elif L is on CPU:
      → CPU backend computes (no PCIe transfer for weights)
      → Backend scheduler handles activation transfer at split

    if L is MoE and expert_offload enabled:
      → Router determines which experts are needed
      → staging_pool_ensure_loaded(needed_experts)
      → Compute with loaded experts
      → Evict unused experts if VRAM pressure increases
```

### Phase 5: Adaptive (Future)

During inference, the unified cache can dynamically adjust:

1. **Expert popularity tracking**: Track which MoE experts are most frequently routed. Pre-load popular experts, evict rare ones.

2. **Layer promotion**: If KV cache shrinks (shorter sequences), promote CPU layers back to GPU using freed VRAM.

3. **Profile-guided placement**: Save expert usage profiles to disk (existing `moe_profile_path` in common.h). Use them to pre-load the right experts at model start.

---

## Integration Points

### Unified Cache ↔ llama_params_fit

Currently independent:
```
llama_params_fit → queries ggml_backend_dev_memory → decides n_gpu_layers
unified_cache → has its own budget formula → evicts when needed
```

Proposed:
```
unified_cache → calculates budget + memory plan
  ↓ exports
llama_params_fit → receives budget as margin → adjusts n_gpu_layers to match
  ↓ or
common_init_from_params → directly sets n_gpu_layers from unified cache plan
```

### Unified Cache ↔ MoE Expert Staging

Current: staging pool exists but works independently of the budget hierarchy.

Proposed: staging pool is part of the Level 2 strategy, managed by the unified cache:
- Cache budget reserves VRAM for active expert slots
- When expert needs loading, cache handles eviction of inactive experts
- Expert access patterns feed back into adaptive placement

### Unified Cache ↔ Layer Streaming

Current: layer_stream_manager is a separate module.

Proposed: streaming is the Level 5 fallback, activated only when n_gpu_layers = 0:
- Unified cache allocates the double buffers from its device budget
- Layer transitions trigger DMA via the existing prefetch tracker
- This is the SLOW path — CPU offload (Level 3/4) is preferred

### Unified Cache ↔ Graph Replay / Persistent TG

These are only disabled when streaming is active (Level 5):
- Level 0-4: graphs + persistent TG fully active for GPU layers
- Level 5: disabled because buffer pointers rotate

---

## Implementation Phases

### Phase A: CPU Offload (Quick Win) — 1-2 tasks

**Plan:** `docs/plans/2026-02-09-cpu-offload-approach.md`

1. Re-enable `fit_params = true`
2. Align VRAM budgets between unified cache and `llama_params_fit`
3. Track CPU weights in unified cache

**Result:** Models exceeding VRAM work at ~20 tok/s (CPU layers) instead of crashing.

### Phase B: GPU Streaming (Parallel Implementation) — 5 tasks

**Plan:** `docs/plans/2026-02-09-host-weight-streaming.md`

Already in progress. Provides Level 5 fallback.

**Result:** Last-resort streaming at ~3.8 tok/s when CPU offload isn't sufficient.

### Phase C: Unified Budget Authority — 2-3 tasks

Move the budget calculation from `llama_params_fit` into the unified cache. The cache determines the optimal memory plan and exports it.

1. Create `unified_memory_plan` computation in unified cache
2. Replace `llama_params_fit`'s independent budget with unified cache export
3. Single source of truth for `GGML_SYCL_VRAM_BUDGET_PCT`

### Phase D: MoE Expert-Aware Budgeting — 2-3 tasks

Integrate MoE expert offloading into the budget hierarchy (Level 2).

1. During budget calculation, compute active expert memory vs total expert memory
2. For MoE models: keep attention + top-K experts in VRAM, stage inactive experts
3. Wire expert staging pool into the unified cache's budget tracking
4. Profile-guided expert pre-loading (use existing `moe_profile_path`)

### Phase E: Sub-Layer Granularity — 1-2 tasks

Enable finer-grained CPU offload (Level 4).

1. During budget calculation, try sub-layer splits before falling to Level 5
2. Generate `tensor_buft_overrides` patterns for partial-layer CPU offload
3. Test with models that barely don't fit (GPT-OSS 20B)

### Phase F: Adaptive Placement (Future) — 3-4 tasks

Runtime weight migration based on access patterns.

1. Track per-tensor access frequency during inference
2. Promote hot CPU tensors to GPU when VRAM headroom available
3. Demote cold GPU tensors to CPU when VRAM pressure increases
4. MoE expert popularity tracking and pre-caching

---

## Performance Expectations

### TG128 (Mistral 7B Q4_0, Arc B580)

| VRAM Budget | Level | Strategy | Expected TG tok/s |
|-------------|-------|----------|-------------------|
| 100% | 0 | All VRAM | ~70 |
| 80% | 0 | All VRAM (model is 3.9 GB, fits) | ~70 |
| 40% | 3 | ~12 GPU layers + ~20 CPU layers | ~15-25 |
| 20% | 3 | ~6 GPU layers + ~26 CPU layers | ~10-15 |
| 10% | 5 | Streaming (if CPU offload insufficient) | ~3-5 |

### TG128 (GPT-OSS 20B MXFP4, Arc B580)

| Strategy | Expected TG tok/s |
|----------|-------------------|
| Level 2 (MoE expert offload) | ~5-10 (if attention fits) |
| Level 3 (layer CPU offload) | ~3-8 |
| Level 5 (streaming) | ~1-3 |

### PP512 (Mistral 7B Q4_0, Arc B580)

PP is compute-bound, not memory-bound. CPU offload is much worse for PP than TG.

| VRAM Budget | Level | Expected PP tok/s |
|-------------|-------|-------------------|
| 100% | 0 | ~1240 |
| 40% | 3 | ~200-400 (CPU layers are slow for large batches) |
| 10% | 5 | ~50-100 |

---

## Key Design Principles

1. **One budget, one authority**: The unified cache IS the memory manager. Everything else queries it.

2. **Graduated response**: Start with the least invasive strategy and escalate only as needed. Never stream when you can offload. Never offload when you can offload KV cache. Never reduce context — offload KV to host instead.

3. **Preserve GPU speed**: Strategies 0-4 keep graph replay and persistent TG active for GPU layers. Only Level 5 (streaming) disables them.

4. **MoE-aware**: MoE models waste massive VRAM on inactive experts. The cache should exploit this by only keeping active experts in VRAM.

5. **Profile-guided**: Expert access patterns and layer hotness can be profiled and saved, enabling smarter placement on subsequent runs.

6. **Single env var**: `GGML_SYCL_VRAM_BUDGET_PCT` controls the entire hierarchy. Set it low to test CPU offload, set it high for maximum GPU utilization.
