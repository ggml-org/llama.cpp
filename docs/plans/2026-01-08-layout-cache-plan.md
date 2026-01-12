# Layout-Aware Unified Cache Plan (SYCL)

## Goals
- Single source of truth for **device-resident** weight layouts via the unified cache.
- Per-tensor optimal layout selection (AOS/SOA/COALESCED/XMX_TILED) based on `layout_policy` + tensor usage + hardware capabilities.
- Eliminate device duplication: once a non-AOS layout is available, do not keep additional AoS/SoA copies on the device.
- All SYCL kernels access weight data via a **layout accessor** (not `tensor->data` directly).
- Default to eager layout finalization (`finalize_layouts()` ON) for deterministic memory + graph safety,
  **except** MoE expert weights which remain lazy and are finalized on-demand.
- Graph compatible: no blocking waits while recording; use event chaining and/or “pre-convert before capture”.
- No hardcoded hardware assumptions; use stored device capabilities.
- Kernel/layout compatibility: **strict 1:1** mapping (each kernel has exactly one optimal layout). Dispatch selects the
  kernel first (using a fixed priority order), then sets the layout based on that kernel; never launch a kernel on an
  unsupported layout.

## Key Invariants (Correctness + Memory)
- **Host AoS source**: for mmap’d weights, the AoS pointer remains the canonical source-of-truth for content (no mutation). Note: mmap is demand-paged; reads may fault pages in from disk, which is expected during one-time finalization.
- **Device canonical pointer**: kernels use only `ggml_sycl_get_layout_ptr(tensor, device)`; never assume `tensor->data` is the weight pointer.
- **Single device copy per (weight, layout, device)**: the cache stores only the selected device layout entry for that weight. If `layout_mode != AOS`, do not keep any additional device AoS copies for the same weight.
- **Graph stability**: any pointer used inside a recorded graph must remain valid for the graph lifetime (pin or otherwise guarantee non-eviction).
  For MoE experts, use a stable **pointer table** indirection so graphs can run without full expert preloads.
- **Kernel layout safety**: dispatch selects kernel → layout. If the preferred kernel/layout is unavailable (e.g., XMX
  tiled unsupported or OOM), fall back to a non‑XMX kernel+layout deterministically.
- **No layout mismatch fallback**: kernel/layout mismatches are bugs. Do not silently fall through to a different layout;
  dispatch must only launch kernels on their declared layout.
- **MoE pointer table**: always-on indirection (graph and non-graph) for expert weights; use a **compact per-dispatch list**
  of pointers in `ids` order (`n_ids × n_tokens`, row-major by `id`) so kernels can index without host sync.
- **Device-side pointer table update**: when `ids` are device-only (graph capture or async dispatch), build the compact
  list on-device (no host sync) using a kernel that reads `ids` + a device-resident expert pointer palette. If any
  required expert pointer is missing, record a miss and **fall back to a non‑XMX kernel** for that op (disable graphs
  for that op only in graph mode).
- **Non-graph eviction allowed**: if eviction is enabled, non-graph dispatch must re-resolve weight pointers per-dispatch (no stale pointers).
- **In-flight safety**: an entry must not be freed while any submitted kernel may still read it. Graph mode achieves this via pinning; non-graph mode needs either per-dispatch pin/refcount or event-based deferred free.

## Architecture Decisions
### Memory Residency (DECIDED)
- Dense weights must be **evictable**, so weights cannot be permanently resident in `tensor->data` device allocations.
- Device-resident weights live in the unified cache; kernels access them via the layout accessor.
- Keep the canonical source as host (mmap / host-loaded weights) and use the cache for device residency.
- MoE expert weights stay **lazy** (no full preload). They are converted on-demand into the cache and referenced via a
  graph-safe pointer table.

### Weight Identity / Cache Key
- Cache keys must uniquely identify a weight across:
  - tensor identity (prefer stable mmap pointer / file-backed pointer when available)
  - device id
  - target `layout_mode`
  - TP sharding (include shard offset/shape in key when TP is enabled)

### Layout Policy (Selection)
- Usage inference order (DECIDED):
  1) consume GGUF/model metadata when available (tensor dims + model hparams/arch; optional explicit tags if present)
  2) fallback: `infer_tensor_usage(tensor->name)`
- Use `layout_policy::get_with_override(qtype, usage)` with capability gating **as a hint**, but finalize layout based on
  the chosen kernel (kernel‑first selection). Overrides must be enforced by kernel selection (no layout/kernel mismatch).
- Kernel‑first selection:
  - For each op (MMQ/MMVQ/DMMV/XMX), define the single optimal layout per kernel and the eligibility rules.
  - Choose the highest‑performance eligible kernel (type, dims, hardware) using a fixed, easy‑to‑tune priority order.
  - If the preferred kernel is unavailable (e.g., XMX tiled unsupported or conversion fails), fall back to a non‑XMX
    kernel+layout deterministically.
- Capability gating:
  - XMX tiled requires `GGML_SYCL_XMX_MOE=1`, `GGML_SYCL_XMX_MOE_TILED=1`, and `ggml_sycl_info().devices[device].xmx_caps.supported`.
  - Coalesced requires `is_coalesced_supported(qtype)` and (optionally) a size/alignment check.
- Fallback policy must be explicit: if target layout conversion fails (OOM, unsupported dims), fall back to SOA (quantized) or AOS (float).

### Kernel Priority Order (Default)
- Kernel variants are layout-specific to preserve strict 1:1 kernel↔layout mapping (e.g., `MMVQ_COALESCED`).
- Priority order is centralized in a single table so it is easy to change later.
- `MUL_MAT` (quantized) default order:
  1) `DMMV_SOA` (batch==1 eligible)
  2) `MMVQ_COALESCED`
  3) `MMVQ_SOA`
  4) `MMVQ_AOS`
  5) `XMX_GEMM_AOS` (experimental)
  6) `MMQ_COALESCED`
  7) `MMQ_SOA`
  8) `MMQ_AOS`
  9) `ONEDNN_AOS`
- `MUL_MAT_ID` (MoE) default order:
  1) `XMX_TILED` (only XMX path; requires env + caps)
  2) `FUSED_ESIMD_AOS` (device-resident only)
  3) `MMVQ_COALESCED`
  4) `MMVQ_SOA`
  5) `MMVQ_AOS`
  6) `HOST_ROUTING_AOS` (graph-disabled)
  - XMX MoE uses the sorted-token path for expert weights and stays higher priority than coalesced MMVQ.

### Layout Sizing (No Hardcodes)
- XMX tiled sizing must use:
  - `moe_xmx_fused::MXFPXMXConfig::from_device(device_id)`
  - `moe_xmx_fused::MXFPXMXLayoutInfo::compute(out_dim, in_dim, cfg)`
  - (and must respect `ggml_sycl_info().devices[device_id].xmx_caps`)

### Memory Budgeting
- Unified cache budget must leave headroom for:
  - KV cache
  - activation buffers
  - per-op scratch
  - graph internal allocations
- Degrade strategy (DECIDED): best-effort, per-tensor fallback when the cache cannot fit all finalized layouts (never corrupt).
  - Preferred fallback order: `XMX_TILED → SOA`, then `COALESCED → SOA`, then `SOA → AOS` only when required (e.g., float).

### KV Cache Allocation (SYCL device allocation limit)
- The SYCL device allocator currently falls back to shared USM for single allocations > ~3.5 GB.
- Plan: chunk KV cache allocations into multiple **device** buffers to stay under the per-allocation limit.
  - Run a runtime probe at device init to determine a safe per-device max allocation size.
  - Store both the raw device limit and the probed safe limit in `ggml_sycl_info()`.
  - Use device-reported `max_mem_alloc_size` as an upper bound and fallback if probing fails; apply a safety margin
    (no hardcoded limit).
  - Add a small ggml-core helper for the probe so other backends can reuse it later.
  - Replace the hardcoded 3.5 GB fallback with the probed `safe_alloc_size`.
  - Prefer a KV-specific buffer type (or KV-only max-size override) so other allocations are unaffected.
  - Leverage `ggml_backend_alloc_ctx_tensors_from_buft()` splitting to produce multiple buffers; tensors are still valid
    because each tensor is allocated directly inside a sub-buffer.
  - Verify this avoids the “Large buffer ... using shared memory” fallback and keeps KV resident in device memory.

### Pinned Host Allocation Chunking (SYCL host buffer type)
- Large host-pinned allocations currently hard-stop at 4 GB, which forces a fallback to regular CPU memory.
- Plan: chunk SYCL host buffer allocations using the **same** probed `safe_alloc_size` from KV chunking.
  - Add a max-size override for the SYCL host buffer type so `ggml_backend_alloc_ctx_tensors_from_buft()` splits
    large pinned host allocations into multiple chunks below the shared safe cap.
  - Ensure the fallback to CPU memory is only used if chunked pinned allocations still fail.

## Graph Compatibility
- `finalize_layouts()` must run **before** graph capture begins and must not call `.wait()` while capture is active.
- If any conversion requires host sync (CPU reorder), it must be performed outside capture.
- Any layout pointers used in graphs must be pinned for the graph lifetime; unpinned on graph invalidation.
- `finalize_layouts()` must cover all model weights **except MoE expert weights** (lazy on-demand) and must run before capture.
- For MoE graphs, use a stable pointer table indirection instead of full expert preload/pinning, and update the table
  before each graph execution based on the current `ids`.
- Pointer table updates must avoid host waits during graph capture (device-only update path required), and a miss should
  disable graphs only for the affected op.
- If graph pinning cannot fit required weights/layouts in cache (non-MoE), graphs must be disabled (best-effort correctness > graph).

## Environment Variables
- `GGML_SYCL_XMX_MOE=1` (enable XMX MoE paths)
- `GGML_SYCL_XMX_MOE_TILED=1` (enable XMX tiled layout path)
- `GGML_SYCL_UNIFIED_CACHE=0` (disable unified cache entirely; default is enabled)
- `GGML_SYCL_UNIFIED_CACHE_MODE={global|per_device|auto}` (cache placement; default is `auto`)
- `GGML_SYCL_LAYOUT_OVERRIDE={aos|soa|coalesced|xmx_tiled}` (forced layout; debug)
- `GGML_SYCL_DISABLE_FINALIZE_LAYOUTS=1` (opt-out of eager finalization)
- (proposed) `GGML_SYCL_LAYOUT_CACHE_STRATEGY={inplace|cache|hybrid}` (locks down Task 0 strategy)
- (proposed) `GGML_SYCL_LAYOUT_CACHE_STRICT=1` (hard-fail on any layout fallback)
- (proposed) `GGML_SYCL_LAYOUT_CACHE_LOG=1` (one-time summary of layouts + fallbacks)
- `GGML_SYCL_WEIGHTS_EVICTABLE=0` (disable host-backed weights; default ON when unified cache is enabled)

## Command-line Arguments
- `--unified-cache-pct <0..100>`: percent of *free* VRAM used for SYCL unified cache budget at init (default `90`).

## Scope
- Extend unified cache key/entry to be layout-aware.
- Add layout-aware cache API that allocates + converts into the target layout.
- Route all weight availability through this system (quantized + non-quantized).
- Update kernel call sites to use the accessor + layout metadata.
- Eagerly finalize layouts before inference and before graph recording.
- Add tests (unit) + manual real-model validation.

## Non-goals
- No new layout types beyond AOS/SOA/COALESCED/XMX_TILED.
- No changes to quantization math.
- No changes to non-SYCL backends.

## Risks / Spinach
- **Memory double-residency risk**: if we keep `tensor->data` as a full device copy *and* allocate cache-owned layout buffers, we can easily OOM.
  - The implementation plan must explicitly choose a residency strategy that avoids duplicates.
- **Eviction safety**: if the cache can evict weights while kernels hold pointers, we can crash or silently corrupt.
  - Graph mode requires pinning; non-graph mode requires either pinning or resolving pointers per-dispatch.
- **Usage inference**: `infer_tensor_usage()` is name-based; unknown names must still behave correctly.

## Validation
- Unit tests for:
  - layout selection + fallback
  - AoS drop policy
  - XMX tiled conversion correctness (AoS→tiled)
- Graph recording smoke test (no waits during capture).
- End-to-end: `llama-completion` on `ONEAPI_DEVICE_SELECTOR=level_zero:1` with real models and deterministic prompt.

## Assumptions (current best-effort)
- GGUF does not provide explicit per-tensor layout/usage hint keys; rely on standard GGUF info (tensor name/type/dims + file offset/size) plus heuristics.
