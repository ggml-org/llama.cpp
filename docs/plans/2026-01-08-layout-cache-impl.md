# Layout-Aware Unified Cache Implementation Plan

This file is the trackable implementation checklist for `docs/plans/2026-01-08-layout-cache-plan.md`.
Update statuses inline as tasks complete.

---

## Task 0: Lock down the memory residency model (no device duplication)
- Status: DONE
- Why:
  - The plan requires “single device copy per (weight, layout, device)”.
  - Today, SYCL weights often have a device allocation at `tensor->data`; if we also allocate cache-owned layout buffers, we can OOM.
- Work:
  - Document and implement a single, explicit strategy (DECIDED: **Strategy B**).
  - Call out the interaction with current behavior:
    - `ggml/src/ggml-sycl/ggml-sycl.cpp:ggml_backend_sycl_buffer_set_tensor()` currently uploads into `tensor->data` and may perform in-place AoS→{SoA,Coalesced,XMX_TILED} conversions.
    - `ggml/src/ggml-sycl/ggml-sycl.cpp:get_or_cache_weight()` currently caches dense weights by keying on `tensor->data`, then does a D2D copy back into `tensor->data` (cache copy + weights buffer copy ⇒ double residency).
    - `ggml/src/ggml-sycl/unified-cache.cpp:cache_moe_expert_with_reorder()` currently caches AoS then reorders via callback; XMX tiled currently allocates an extra temporary buffer during reorder (peak VRAM risk).
    - Caching is skipped during model load via `ggml_backend_sycl_set_model_loading(true)` in `src/llama-model-loader.cpp` and must remain safe.
    - the chosen strategy must not regress existing “no duplicate SoA+tiled” OOM fixes.
  - Dense weights must be evictable, so the model must not permanently allocate device-resident weight storage at load time.
    - Implementation detail: ensure weights remain host-backed (mmap when enabled) and are made device-resident only via the unified cache.
  - Implementation (Strategy B):
    - Host-backed weight selection in `src/llama-model.cpp` when SYCL weights are evictable (default ON unless `GGML_SYCL_WEIGHTS_EVICTABLE=0`).
    - Host weight tensors are registered for layout metadata; `ggml_sycl_get_layout_ptr()` resolves device pointers via unified cache for host weights.
    - `ggml_backend_sycl_buffer_set_tensor()` skips dense-weight cache/D2D copy when weights are evictable (avoids duplication).
    - Offload + graph-preload treat “weights evictable” as streaming to keep GPU-side execution consistent.

  **Strategy A (in-place in `tensor->data`, minimal refactor)**
  - During `ggml_backend_sycl_buffer_set_tensor`, write the *target* layout directly into `tensor->data`.
  - `extra->layout.data_ptr == tensor->data` for all weights when target_bytes <= ggml allocation bytes.
  - If target layout requires more bytes than `ggml_nbytes(tensor)`, apply a deterministic fallback (disable that layout for that tensor/device).
  - Pros:
    - smallest code change; matches current SoA/coalesced behavior.
    - graph-safe by default (pointers stable, no eviction required).
    - avoids *additional* device allocations for layouts that fit in-place.
  - Cons:
    - cannot support layouts that require larger buffers than `ggml_nbytes(tensor)` (unless we change ggml allocation sizing).
    - unified cache cannot truly “evict” dense weights (memory is fixed for model lifetime).

  **Strategy B (cache-owned device pointers, full centralization)**
  - Device weight memory is allocated/owned by unified cache.
  - `tensor->data` becomes unused for weights (or becomes a small stub), and kernels always resolve through `extra->layout`/accessor.
  - Requires changes in buffer allocation/weight upload so we don’t pay double residency.
  - Pros:
    - enables true eviction for all weights/layouts; aligns with “per-dispatch resolve” model.
    - supports layouts that change size (e.g., padded tiled layouts).
  - Cons:
    - largest refactor: avoid allocating a full GPU weights buffer (or accept wasted VRAM).
    - must harden all weight access paths to handle CPU-backed tensors + cache streaming.

  **Strategy C (hybrid)**
  - Cache-owned pointers only for layouts that cannot fit in existing allocations (e.g., XMX tiled padding), in-place otherwise.
  - Pros:
    - preserves today’s dense-weight flow while enabling “oversize” layouts where needed.
    - can keep eviction focused on expensive layouts (e.g., tiled) and MoE experts first.
  - Cons:
    - if ggml still allocates the full weights buffer, cache-owned “oversize” layouts add VRAM on top (must be controlled via best-effort fallback).
    - eviction semantics become tiered (some pointers always resident, others evictable).

- Done when:
  - We have a written decision in this file and the subsequent tasks follow it (no “accidental” duplication).

---

## Task 1: Make unified cache layout-aware (keys + eviction correctness)
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/unified-cache.hpp`
  - `ggml/src/ggml-sycl/unified-cache.cpp`
- Required changes:
  - Extend `unified_cache_key` to include `layout_mode`.
  - Replace `ptr_to_key_` mapping so it is keyed by `{key_ptr, layout_mode}`.
  - Update pin/unpin APIs and call sites so pinning is **layout-specific** (graphs must pin the exact layout entry they will use).
  - Fix eviction/removal logic so evicting one layout does not break lookups for other layouts.
  - Extend `unified_cache_entry` to record its `layout_mode`.
  - Update `print_stats()` to report counts/bytes by layout.
- Edge cases:
  - Multiple layouts for the same weight must coexist.
  - `evict_one()` must not blindly erase a single `key_ptr` mapping if other layouts exist.

---

## Task 2: Define a stable weight identity key (beyond “pointer-only”)
- Status: DONE
- Why:
  - Pointer identity differs between mmap vs staged vs TP sharded buffers.
  - Graph capture requires pointer stability.
- Files:
  - `ggml/include/ggml-sycl.h` (new registration API)
  - `ggml/src/ggml-sycl/common.hpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - `src/llama-model-loader.cpp` (register identities during load)
- Implementation notes:
  - Added GGUF identity registration and stable key accessors in SYCL:
    - `ggml_backend_sycl_register_weight_identity(...)`
    - `ggml_backend_sycl_get_weight_cache_key(...)`
  - Unified cache lookups now use a stable key derived from:
    - GGUF `{file_idx, file_offs, tensor_nbytes}` when available.
    - Fallback `{tensor_name, type, dims}` when not.
    - `device_id` and TP shard fields `{rank, world_size, local_ne, offset_ne}`.
  - Layout remains a field on `unified_cache_key` (no duplication in the stable identity key).

---

## Task 3: Add a layout-aware cache API that is graph-safe
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/unified-cache.hpp`
  - `ggml/src/ggml-sycl/unified-cache.cpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
- API shape (minimum requirements):
  - `ensure_cached_layout(request, deps...) -> {ptr, size, layout_mode, xmx_meta, event, status}`
  - Must accept dependency events and return a `sycl::event` representing completion.
  - Must never call `.wait()` when `g_ggml_sycl_graph_recording` is true.
- Concurrency + in-flight conversions:
  - Cache must track “IN_PROGRESS vs READY” for each `{weight_id, layout}` so two callers don’t allocate/convert twice.
  - Subsequent callers should receive the same `{ptr, completion_event}` and either:
    - chain it as a dependency (graph-safe), or
    - wait outside capture if needed (non-graph).
- Temporary buffer lifetime (critical):
  - Any device temporaries used during conversion (e.g., AoS staging for AoS→XMX_TILED) must not be freed before the conversion event completes.
  - Prefer: persistent per-device staging buffers, or deferred-free lists keyed by `sycl::event`.
- Required internal refactors:
  - `unified_cache::copy_to_device()` is currently synchronous (`.wait()`); it must gain an async/event-returning path for graph-safe usage, or conversions must be guaranteed to run pre-capture only.
- Conversion sizing:
  - For XMX tiled: compute bytes via `MXFPXMXLayoutInfo::compute()` using `MXFPXMXConfig::from_device()` and `xmx_caps`.
  - For coalesced: size and alignment checks must be explicit; conversion may fall back to SOA if unsupported.
- Failure strategy:
  - On allocation failure, return a status that causes caller to fall back deterministically (and optionally disable that layout for the run).

- Implementation notes:
  - Added `ensure_cached_layout()` with request/result structs, layout metadata, status, and dependency event chaining.
  - Unified cache entries now track `READY` vs `IN_PROGRESS` plus completion events; eviction skips in-flight entries.
  - Added async copy path (no `.wait()`) for graph-safe cache fills.
  - Fixed AoS→XMX_TILED staging lifetime by persisting per-tensor staging until conversion completes.

---

## Task 4: Centralize layout conversions into a single conversion engine
- Status: Done
- Files:
  - `ggml/src/ggml-sycl/ggml-sycl.cpp` (conversion dispatch)
  - `ggml/src/ggml-sycl/moe-tile-convert.cpp` (XMX tiled kernels)
- Work:
  - Implement canonical conversions:
    - AoS → SoA (existing CPU/GPU reorder)
    - AoS → Coalesced (direct when supported; otherwise AoS→SoA→Coalesced)
    - SoA → Coalesced
    - AoS → XMX_TILED (must be **direct**, no SoA intermediate; AoS-only source)
    - XMX_TILED is not built from SoA/Coalesced; conversions happen at finalize time from AoS only.
  - Ensure temporary buffers are freed promptly and do not overlap peak memory.
  - Remove any fixed “max experts” limitations in conversion code paths:
    - Example today: `ggml/src/ggml-sycl/ggml-sycl.cpp` uses `constexpr size_t MAX_EXPERTS_CONVERSION = 32;` in the tiled MXFP4 path.
    - Target behavior: support any `n_experts` by looping/streaming (one expert at a time) without allocating `O(n_experts)` metadata buffers.
- Graph requirements:
  - Conversions used during capture must be pre-run (preferred) or implemented as pure device commands with event chaining.

---

## Task 5: Route *all* weights through the unified cache + layout selection
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/ggml-sycl.cpp` (upload + weight access)
  - `ggml/src/ggml-sycl/common.hpp` (accessor)
- Work:
  - Determine usage via GGUF/model metadata when possible; otherwise `infer_tensor_usage(tensor->name)`.
    - Edge case to handle explicitly: tied weights (`token_embd.weight` reused as `output.weight`) must choose a layout that is valid for *all* consumer ops (e.g., GET_ROWS + MUL_MAT) or maintain multiple layouts with explicit selection.
  - Determine target layout via kernel‑first selection:
    - Select the highest‑performance eligible kernel for each op (MMQ/MMVQ/DMMV/XMX) based on type, dims, and hardware.
    - Use a fixed, easy‑to‑tune kernel priority list (so we can reorder later if benchmarks change).
    - Implement priority as a centralized table of **kernel variants** (layout-specific), e.g., `MMVQ_COALESCED`.
    - Default priority order must match the plan:
      - `MUL_MAT`: `DMMV_SOA` → `MMVQ_COALESCED` → `MMVQ_SOA` → `MMVQ_AOS` → `XMX_GEMM_AOS` → `MMQ_COALESCED`
        → `MMQ_SOA` → `MMQ_AOS` → `ONEDNN_AOS`.
      - `MUL_MAT_ID`: `XMX_TILED` → `FUSED_ESIMD_AOS` → `MMVQ_COALESCED` → `MMVQ_SOA` → `MMVQ_AOS`
        → `HOST_ROUTING_AOS`.
    - Each kernel has a **single optimal layout**; set layout based on the chosen kernel.
    - If the preferred kernel is unavailable (XMX unsupported/OOM/invalid dims), fall back to a non‑XMX kernel+layout.
    - `layout_policy::get_with_override(qtype, usage)` remains as a hint/override but must be enforced by kernel selection
      (no layout/kernel mismatch).
  - Ensure weights (quantized + non-quantized) are made available in the target layout via the cache API.
  - MoE expert weights remain lazy:
    - Use on-demand cache conversion per expert (AoS → target layout).
    - Ensure conversions/reorders use event dependencies (no blocking waits during graph capture).
    - Update the MoE pointer table when cache entries are ready.
    - Add a device-side **compact list** update path:
      - Kernel reads `ids` on-device and writes a compact pointer list (`n_ids × n_tokens`, row-major by `id`) using a
        device-resident expert pointer palette.
      - No host sync during graph capture; chain events from cache conversions into the update kernel.
      - Detect missing expert pointers (null) and deterministically fall back to a **non‑XMX kernel** for that op;
        disable graphs for that op only when recording.
    - Keep XMX sorted MoE enabled for expert weights; ensure it uses the layout-selected pointer (no direct `tensor->data`),
      and keep it higher priority than coalesced MMVQ.
  - Enforce AoS-drop rule:
    - Once a non-AOS layout exists, remove any device AoS/SoA cache entries for that weight.
- Edge cases:
  - Unknown usage names must still select a safe layout (SOA for quantized, AOS for float).
  - For non-quantized weights (F16/F32), the “optimal” layout is likely AOS; do not invent reorder.
  - MoE experts can be evicted between runs; pointer table refresh must handle this.
  - Kernel/layout mismatch must never silently fall through to an unsupported kernel.

---

## Task 6: Make kernels use the layout accessor (no direct `tensor->data`)
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/common.hpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - `ggml/src/ggml-sycl/mmvq.cpp` (and other SYCL kernels that read weights)
- Work:
  - Add/standardize:
    - `ggml_sycl_get_layout_ptr(const ggml_tensor*, int device)`
    - `ggml_sycl_get_layout_info(const ggml_tensor*)`
    - A kernel layout capability registry (e.g., `ggml_sycl_kernel_layout_for(op, type, dims, device)` returning the
      single optimal layout or `NONE` if ineligible), backed by the centralized priority table.
  - Update dispatch sites to use these accessors for weights.
  - Use the registry to choose the kernel first, then set the layout; fallback to non‑XMX kernel+layout if needed.
  - Ensure XMX MoE paths consume `extra->layout.xmx_info` for tiled kernels.
  - Add MoE expert pointer table usage for graph-safe lazy experts (MMVQ / fused MoE paths read from table); make it
    always-on (graph and non‑graph) for consistent fast paths.
  - Device-side compact list update kernel must be callable from MoE paths (MMVQ + XMX sorted) to avoid host waits.
  - Update MoE kernels to consume the compact list directly (row-major by `id`, no per-expert lookup on host).
- Edge cases:
  - Accessor must be safe when `extra` is null (fallback to `tensor->data`).
  - In graph mode, accessor must not trigger allocations/conversions (must already be finalized).

---

## Task 7: Implement `finalize_layouts()` as the default pre-inference stage
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/ggml-sycl.cpp` (function: `finalize_layouts` and call sites)
- Requirements:
  - Runs by default; opt-out via `GGML_SYCL_DISABLE_FINALIZE_LAYOUTS=1`.
  - Must finalize layouts for **all weights except MoE expert weights**, not only `MUL_MAT`/`MUL_MAT_ID`.
    - Enumerate via backend tensor registry (e.g., `ggml_backend_sycl_buffer_context::tensor_extras`) or an explicit weight list captured at model load.
  - Must run **before** graph capture starts.
  - Must be idempotent and cheap on subsequent calls (add a “layouts_finalized” flag in the backend ctx keyed by device + model instance).
- Failure behavior:
  - If full finalization fails due to memory, choose a deterministic degrade path (see Task 8).

---

## Task 8: Make cache eviction/pinning compatible with inference + graphs
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/unified-cache.cpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
- Work:
  - Define lifetime rules:
    - Graph: pin all layout entries referenced by the graph until graph invalidation.
    - Non-graph (DECIDED): allow eviction; resolve pointers per-dispatch (no stale pointers).
    - MoE graphs: use a stable pointer table instead of full expert preload; update table entries on-demand and before each graph execution based on current `ids`.
  - In-flight safety (must-have):
    - Prevent eviction from freeing an entry while a previously-submitted kernel may still read it.
    - Implement: event-based deferred free (eviction queues `sycl::free` behind the last-use `sycl::event`), plus periodic cleanup at safe points.
  - Stale pointer handling:
    - Treat `extra->layout.data_ptr` as a fast-path cache only; it must be refreshed from the unified cache each dispatch (or validated via a generation counter).
    - If an entry was evicted, accessor/dispatch must recreate it via `ensure_cached_layout()` before launching kernels.
    - Pointer table updates must be chained after layout conversion events (no host waits during capture).
  - Update graph helpers:
    - `graph_preload_weights()` / `graph_unpin_weights()` must pin/unpin the correct **layout entries**, not just “by pointer”.
    - pin/unpin should include `{device_id, layout_mode}` in addition to the weight identity.
    - Replace MoE expert preloading with pointer table updates; avoid full cache pinning for experts.
- Edge cases:
  - If eviction is allowed in non-graph mode, accessor must re-resolve and update pointers before launch.

---

## Task 9: Memory budgeting + graceful degradation
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/unified-cache.cpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
- Work:
  - Budget sizing must reserve headroom for runtime allocations.
    - Note today: unified cache auto-budget uses ~70% of *free* memory at cache init (`ggml/src/ggml-sycl/unified-cache.cpp:create_cache_for_device()`).
    - Update policy:
      - Default: `--unified-cache-pct 90` (% of *free* VRAM at cache init).
      - User override: `--unified-cache-pct <0..100>`; clamp to sensible bounds; treat `set_unified_cache_budget(bytes)` as higher-precedence if set programmatically.
      - Plumbing:
        - Add a `common_params` field (e.g., `sycl_unified_cache_pct`, default 90).
        - Parse it in all `llama-*` CLIs via `common/arg.*` and call a SYCL-backend setter *before* backend init.
        - Add SYCL API (example): `ggml_backend_sycl_set_unified_cache_budget_pct(int pct)` which updates a global default used by `create_cache_for_device()`.
    - Still reserve runtime headroom (KV/scratch/graphs): if computed budget would starve runtime, degrade layouts early rather than thrash.
  - Precompute the total bytes required for the chosen target layouts (per device) before allocating; if it cannot fit, degrade early instead of partially allocating and thrashing.
  - Define degrade strategies when cache cannot fit all layouts:
    - Prefer dropping XMX_TILED first (fall back to SOA) if tiled would OOM.
    - Prefer SOA over COALESCED when coalesced conversion fails alignment.
    - (DECIDED) best-effort fallbacks are allowed; optional env var for “strict” can hard-fail if desired.
  - Observability:
  - Log a one-time summary of chosen layouts + any fallbacks.
  - Optional optimization (only if needed): mitigate USM fragmentation by adding an allocator/pool for cache allocations (many `sycl::malloc_device` calls can fail even with apparent free VRAM).

---

## Task 19: Chunk KV cache allocations to stay in device USM
- Status: DONE
- Files:
  - `src/llama-kv-cache.cpp`
  - `ggml/include/ggml-alloc.h`
  - `ggml/src/ggml-sycl/common.hpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - `ggml/src/ggml-alloc.c` (new helper implementation)
- Why:
  - KV cache buffers can exceed the single-allocation device limit, which triggers shared USM fallback and hurts performance.
- Work:
  - Add a small ggml-core helper to probe a safe max allocation size for a buffer type:
    - API in `ggml-alloc.h` (example): `ggml_backend_probe_max_alloc_size(buft, upper_bound, safety_margin)`.
    - Implementation in `ggml-alloc.c`: bounded binary search that allocates + frees a buffer at each step.
    - Use `buft->get_max_size` (or the provided `upper_bound`) as the probe ceiling; fall back to the raw limit if probing fails.
  - Record per-device allocation limits in `ggml_sycl_info()`:
    - Capture `sycl::info::device::max_mem_alloc_size` at device init and clamp to `total_vram`.
    - Run the ggml-core probe once per device to derive `safe_alloc_size` (no hardcoded limit).
    - Store both raw and safe limits in `ggml_sycl_info()` for logging and reuse.
  - Replace the hardcoded 3.5 GB threshold in `ggml_backend_sycl_buffer_type_alloc_buffer()` with the probed `safe_alloc_size`.
  - Add a SYCL KV buffer type (or a max-size override) whose `get_max_size` returns `safe_alloc_size` so
    `ggml_backend_alloc_ctx_tensors_from_buft()` splits KV allocations into multiple device buffers.
  - Select the KV buffer type in `src/llama-kv-cache.cpp` when `offload` is enabled (including TP mode),
    so KV allocations stay under the per-buffer limit without affecting other SYCL allocations.
  - Log a KV allocation summary (per device): buffer count, chunk size, and whether any shared fallback was used.
- Done when:
  - Large-context KV allocations no longer emit the “Large buffer ... using shared memory” log.
  - `ggml_backend_alloc_ctx_tensors_from_buft_size()` reports the sum of chunked allocations and inference succeeds.
  - KV allocation logs report chunking when multi-buffer splitting is used.

---

## Task 20: Chunk pinned host allocations for SYCL host buffers
- Status: DONE
- Files:
  - `ggml/src/ggml-sycl/common.cpp`
  - `ggml/src/ggml-sycl/ggml-sycl.cpp`
- Why:
  - Host-pinned allocations currently hard-stop at 4 GB and fall back to regular CPU memory even when chunking would work.
- Work:
  - Reuse the **same** probed `safe_alloc_size` from Task 19 (stored in `ggml_sycl_info()`).
  - Add a `get_max_size` override for the SYCL host buffer type so `ggml_backend_alloc_ctx_tensors_from_buft()`
    splits large pinned allocations into multiple chunks under the shared safe cap.
  - Replace the fixed 4 GB guard in `ggml_sycl_host_malloc()` with a “chunking first” policy:
    - allow larger total allocations if the buffer type can split,
    - fallback to CPU memory only if chunked pinned allocations still fail.
- Done when:
  - Large SYCL host buffers (e.g., ~5.5 GB weights) allocate as multiple pinned chunks without falling back to CPU memory.

---

## Task 10: Unit tests (single-GPU correctness)
- Status: DONE
- Files:
  - `tests/test-mxfp4-xmx-tiled.cpp`
  - `tests/test-layout-cache.cpp` (new)
  - `tests/CMakeLists.txt`
- Unit test requirements:
  - Layout selection: verifies `infer_tensor_usage()` + `layout_policy` picks expected targets.
  - AoS-drop: verifies that once a non-AOS layout is cached, AoS is not retained as a device-resident cache entry.
  - XMX tiled correctness: AoS→tiled conversion produces numerically equivalent output vs reference path.
- Commands:
  - `ONEAPI_DEVICE_SELECTOR=level_zero:1 ctest -R test-mxfp4-xmx-tiled -V`
  - `ONEAPI_DEVICE_SELECTOR=level_zero:1 ctest -R test-layout-cache -V`
- Results:
  - `test-layout-cache`: PASS
  - `test-mxfp4-xmx-tiled`: PASS

---

## Task 11: End-to-end validation with real models
- Status: IN PROGRESS
- Steps:
  - Build (SYCL):
    - `source /opt/intel/oneapi/setvars.sh --force`
    - `cmake --build build --config Release -j $(nproc)`
  - Run:
    - `ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_XMX_MOE=1 GGML_SYCL_XMX_MOE_TILED=1 ./build/bin/llama-completion -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0`
  - Verify:
    - output matches a known baseline
    - logs confirm XMX tiled path is active
    - no OOM / device-lost
  - Current result:
    - Mistral Q6_K run completed, output did not match expected sequence (saw `1, 2, 3, 4, 5,rassrac`).
    - gpt-oss-20b run still crashes with SIGSEGV; gdb backtrace points to `ggml_sycl::compute_content_hash` in `ggml/src/ggml-sycl/unified-cache.cpp:109` during `ensure_cached_layout` → `ggml_sycl_get_weight_layout_ptr` → `ggml_sycl_op_mul_mat` → `ggml_sycl_mul_mat_id`.

---

## Task 12: Review checkpoints + compliance audit
- Status: TODO
- Checkpoints:
  - After Task 0: confirm we truly avoid device duplication.
  - After Task 6: confirm all kernel dispatches use the accessor.
  - After Task 7/8: confirm graph capture never hits waits and pointers remain stable.
  - After Task 19: confirm KV buffers stay in device USM (no shared fallback) and chunking uses device caps.
  - After Task 20: confirm pinned host allocations chunk under the shared safe cap and avoid CPU fallback.
  - Final: run a “review agent” pass focused on plan compliance, memory safety, and graph safety.

---

## Assumptions / Resolved Decisions (current)
- Memory residency: Strategy B (cache-owned device pointers; dense weights evictable).
- GGUF metadata: consume standard GGUF tensor identity (`{file_idx, file_offset, nbytes}`) when available; assume no custom per-tensor layout/usage keys.
- Unknown tensor names: quantized → SOA, float → AOS (safe default).
- `finalize_layouts()`: runs by default and targets all model weights **except MoE expert weights** (best-effort fallback if cache cannot fit).
- Graph safety: no waits during capture; graph mode pins all referenced layouts; MoE graphs use a pointer table to avoid full expert preload.
- Non-graph mode: eviction allowed; pointers resolved per-dispatch; eviction must be in-flight safe (no free while kernels may read).
