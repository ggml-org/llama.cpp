Project 03C — Deterministic Attention on Other Backends + KV‑Cache Invariance
=============================================================================

Purpose
-------
Extend deterministic attention semantics beyond CUDA to Metal, Vulkan, OpenCL (and validate HIP via CUDA path), and add end‑to‑end KV‑cache invariance. Deterministic means: for covered shapes, batch‑invariant and run‑to‑run bitwise identical outputs under `GGML_DETERMINISTIC=1`.

Non‑Goals (03C)
---------------
- Cross‑device/driver bitwise parity (pin builds if needed).
- Full quantized K/V coverage parity with CUDA (limit to feasible vec instances).
- Multi‑GPU determinism (Project 04).

Definitions (recap)
-------------------
- Run‑to‑run determinism: same inputs, same binary → identical bytes.
- Batch invariance: per‑row result is independent of batch size (fixed reduction order per row/token).
- KV‑cache invariance: logits for a token at position P are bitwise identical whether computed via single‑shot prefill to P or via incremental decode that appends tokens up to P.

Global Policy (to mirror CUDA 03A/03B)
-------------------------------------
When `ggml_is_deterministic()`:
- Enforce single‑column kernels/workgroups and forbid multi‑workgroup combines (no split‑K, no post‑combine reductions).
- Prefer vec single‑column kernels when available; otherwise use a deterministic tile/single‑column kernel (F16 K/V only). Quantized K/V must use vec‑only deterministic instances.
- Keep `stream_k`‑like pipeline features off where applicable.
- Maintain KV length multiple of `FATTN_KQ_STRIDE` and pad mask to `GGML_KQ_MASK_PAD` (64).

Environment Toggles (reuse + extend)
------------------------------------
- `GGML_DETERMINISTIC=1` – enable deterministic mode.
- `GGML_DETERMINISTIC_ATTENTION_FORCE_VEC=1` – force vec path when available.
- `GGML_DETERMINISTIC_ATTENTION_FORCE_TILE=1` – force tile/single‑column (F16 K/V only).
- `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1` – allow tensor‑core/coop‑mat paths if single‑workgroup and batch‑invariant.
- `GGML_DET_ATTENTION_DISABLE_TILE_80_96_112=1` – matches CUDA special heads gating (for future parity).
- `GGML_DET_ATTENTION_DISABLE_MMA_80_96_112=1` – matches CUDA opt‑out of default MMA for special heads.

Backends: File Targets and Changes
----------------------------------

Metal (Apple GPUs)
- Files:
  - `ggml/src/ggml-metal/ggml-metal.m` — pipeline selection and dispatch.
  - `ggml/src/ggml-metal/ggml-metal.metal` — kernel code: `kernel_flash_attn_ext` and `kernel_flash_attn_ext_vec_*` plus `_vec_reduce`.
  - `ggml/src/ggml-metal/ggml-metal-impl.h` — kargs and function constants (FC_*).
- Goals:
  - Deterministic pipeline: single‑workgroup per query column (nwg=1) and no usage of `_vec_reduce` combine pass.
  - Honor FORCE_VEC/FORCE_TILE/ALLOW_MMA toggles.
  - For quantized K/V, allow only vec variants with known instances; otherwise abort with guidance.
- Concrete changes:
  1) In `ggml_metal_get_pipeline_flash_attn_ext_vec(...)` and dispatch sites, introduce a deterministic branch:
     - If `ggml_is_deterministic()` and not FORCE_VEC/FORCE_TILE overrides, set `nwg=1` and select vec kernel variants that operate per‑column without requiring `kernel_flash_attn_ext_vec_reduce`.
     - Avoid compiling/dispatching `_vec_reduce` pipeline when det mode is on.
  2) In `ggml_metal_get_pipeline_flash_attn_ext(...)` (non‑vec path), ensure `nsg` (simdgroup count) does not trigger multi‑workgroup accumulation; force scheduling so one workgroup performs complete per‑column reduction.
  3) In `.metal` shaders, audit that all per‑column reductions stay within a simdgroup/workgroup for the deterministic mode (guard via FC constants). Add static asserts/comments if needed; prefer code‑path selection via `FC_flash_attn_ext_nsg`, `FC_flash_attn_ext_vec_reduce_NWG`.
  4) Guard unsupported features (e.g., logit_softcap for special head sizes) consistently with CUDA error texts.

Vulkan
- File: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Observations:
  - Uses multiple pipelines including `pipeline_flash_attn_split_k_reduce` (split‑K post combine) which breaks determinism.
- Goals:
  - A deterministic fast path that never dispatches `*_split_k_reduce`.
  - Single‑workgroup/workgroup‑fixed reduction per query column.
- Concrete changes:
  1) In `ggml_vk_flash_attn(...)` add early deterministic branch:
     - If `ggml_is_deterministic()`: set split‑K to 1; choose a pipeline variant with no split‑K; set dispatch dimensions to 1 workgroup across the reduction axis; ensure subgroup reductions stay within the workgroup.
     - Respect FORCE_VEC/FORCE_TILE; ALLOW_MMA can map to coop‑mat/WMMA variants only when they run with a single workgroup and no post‑combine.
  2) Remove/skip the dispatch at 7412 for `pipeline_flash_attn_split_k_reduce` under deterministic mode.
  3) Validate mask/ALiBi/sinks handling matches CUDA ordering.

OpenCL
- Files:
  - `ggml/src/ggml-opencl/ggml-opencl.cpp` — kernel selection and launch parameters (block sizes).
  - `ggml/src/ggml-opencl/kernels/flash_attn_*.cl` — kernel code.
- Observations:
  - The runtime chooses block_m/block_n (bm/bn) and may do cross‑workgroup accumulation.
- Goals:
  - Deterministic scheduling: bm/bn set so that a single workgroup handles a column’s reduction (no split‑K, no multi‑group combine).
  - Keep existing kernels; specialize launch parameters only.
- Concrete changes:
  1) In `ggml_cl_flash_attn(...)` (around 5764), detect `ggml_is_deterministic()` and force bm/bn such that one workgroup processes the reduction for a column.
  2) Respect FORCE_VEC/FORCE_TILE toggles if vec/tile variations exist; otherwise pick the closest “single workgroup” kernels.
  3) Error on unsupported quantized shapes with actionable guidance (mirror CUDA texts).

HIP (ROCm)
- Source lives via CUDA templates (see `ggml-hip/CMakeLists.txt` includes CUDA template instances).
- Action: No additional dispatcher code is required if the CUDA deterministic branch compiles under HIP (it does via `GGML_USE_HIP` guards in `common.cuh`).
- Task: Add HIP CI job to run the determinism tests; gate kernel selection limits similarly to CUDA (env toggles honored the same way).

CPU fallback (softmax ext)
- File: `ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_soft_max_ext_*`) and `ggml/src/ggml.c` plumbing.
- Task: When FlashAttention is absent on a backend and det mode is ON, dispatch to a deterministic softmax‑ext path that reduces per row within one thread/workgroup (no planner‑driven split‑reductions). This provides a portable deterministic fallback at reduced throughput.

KV‑Cache Invariance
-------------------
API‑level behaviors (no core API changes needed):
- Normalize KV views and strides so attention sees the same contiguous K/V layout regardless of how many tokens are in cache vs current batch.
- Ensure mask is padded to `GGML_KQ_MASK_PAD` and corresponds 1:1 with the intended query columns; avoid separate reductions for cache vs current tokens (single pass over KV in fixed order).

Test Plan (new)
---------------
- New test: `tests/test-kvcache-invariance.cpp` (backend‑agnostic harness similar to `test-attention-determinism.cpp`).
  - Build two graphs over the same window [0, P):
    1) Single‑shot prefill to length P.
    2) Incremental: append tokens one by one (and in fixed chunk sizes, e.g., 8/33) until P; compute logits each step.
  - Assert bitwise equality of the final token’s logits between (1) and (2).
  - Grid:
    - Ds: {64, 128, 256}
    - KVs: {256, 1024}
    - Batches: {1, 8}
    - GQA: {1, 2}
    - Features: mask on/off, ALiBi on/off (`max_bias`), sinks off (and a small smoke with sinks on).
  - Backend selection: enumerate all registered backends; run for Metal/Vulkan/OpenCL/HIP when available; skip cleanly otherwise.
  - Optional envs:
    - `RUN_KVCACHE_CHUNK_TESTS=1` — adds chunked incremental variants (chunk=8/33) to ensure batch invariance under chunking.

Acceptance Criteria
-------------------
1) Deterministic attention on Metal/Vulkan/OpenCL/HIP:
   - Cross‑run determinism for covered shapes (bitwise equality across two runs).
   - Batch invariance for B∈{1,8,33} (first column matches across B).
2) KV‑cache invariance:
   - Single‑shot vs incremental decode bitwise equality for the final position across the grid above.
3) Feature gates:
   - Quantized K/V: vec‑only instances; unsupported pairs abort deterministically with guidance.
   - Special head sizes: mirror CUDA behavior where feasible; clearly error when unsupported.
4) Observability:
   - One‑time INFO logs when falling back to deterministic tile/single‑workgroup paths.

Docs & Runbooks
---------------
- Update `docs/DETERMINISM.md` with backend details (Metal/Vulkan/OpenCL/HIP) and KV‑cache invariance policy.
- Add `projects/03-deterministic-attention/runbook-03C.md`:
  - How to build and run tests for each backend:
    - Metal: Xcode/AppleClang settings; `./test-attention-determinism` and `./test-kvcache-invariance` on macOS; `GGML_METAL_PATH` if needed.
    - Vulkan: loader/device selection, environment variables, and required ICDs.
    - OpenCL: platform/device selection flags.
    - HIP: ROCm version, arch flags, container image example.
  - Debug envs (FORCE_VEC/FORCE_TILE/ALLOW_MMA, disable split‑K, etc.).

Milestones & Tasks
------------------
M1 — Scaffolding (0.5d)
- Add new test skeleton `tests/test-kvcache-invariance.cpp` (CUDA first via CPU buffers but backend‑agnostic).
- Wire backend enumeration and skip behaviors.

M2 — Metal deterministic path (1.5d)
- `ggml-metal.m`: deterministic branch in pipeline selection; set `nwg=1`, avoid `_vec_reduce` under det mode; honor toggles.
- `ggml-metal.metal`: verify single‑workgroup reductions for det path; guard unsupported head sizes/logit_softcap like CUDA.

M3 — Vulkan deterministic path (1.5d)
- `ggml-vulkan.cpp`: deterministic branch in `ggml_vk_flash_attn`; split‑K off; no post‑combine; dispatch dimensions = single workgroup along reduction axis; toggles honored.

M4 — OpenCL deterministic scheduling (1.0d)
- `ggml-opencl.cpp`: fix bm/bn selection for det mode; ensure no cross‑workgroup combine; toggles honored.

M5 — KV‑cache invariance tests (0.5d)
- Finish test coverage across backends; add chunked incremental variants under `RUN_KVCACHE_CHUNK_TESTS=1`.

M6 — Docs & Runbooks (0.5d)
- Update `docs/DETERMINISM.md` and author `runbook-03C.md`.

Risk & Mitigations
------------------
- Performance regressions in det mode: acceptable by design; document in runbook.
- Kernel coverage gaps (e.g., quantized on non‑CUDA): error with guidance; expand later as feasible.
- Vendor compiler differences (Metal/Vulkan/OpenCL): keep det path simple (single‑workgroup); avoid dynamic planner behavior.

Audit Points (exact code locations)
-----------------------------------
- Metal:
  - `ggml-metal.m`: `ggml_metal_get_pipeline_flash_attn_ext[_vec|_vec_reduce]`, dispatch sites for FlashAttention.
  - `ggml-metal.metal`: sections starting near FC constants for `flash_attn_ext` and `flash_attn_ext_vec[_reduce]`.
- Vulkan:
  - `ggml-vulkan.cpp`: pipelines around 2500–3200 (creation), `ggml_vk_flash_attn(...)` around 7100–7420, and split‑K reduce dispatch around 7412.
- OpenCL:
  - `ggml-opencl.cpp`: `ggml_cl_flash_attn(...)` around 5764 and kernel selection/tuning tables 1320–1400.
- CPU fallback:
  - `ggml-cpu/ops.cpp`: `ggml_compute_forward_soft_max_ext_*` single‑thread/workgroup deterministic path.

Success Criteria (Summary)
--------------------------
- All determinism tests pass on at least one device per backend (Metal/Vulkan/OpenCL/HIP) for the defined grid.
- KV‑cache invariance test shows bitwise equality single‑shot vs incremental across target shapes.
- Docs/runbooks updated; backend toggles consistent with CUDA semantics.

