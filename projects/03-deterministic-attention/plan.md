Project 03 — Deterministic Attention (CUDA, Phase 03A)
=====================================================

Goal
----

- When `ggml_is_deterministic()` is true, FlashAttention forward on CUDA is bitwise deterministic and batch‑invariant across runs and batch sizes for common LLaMA shapes.
- Deterministic mode remains opt‑in. Default builds keep current fast behavior.

Non‑Goals (03A)
----------------

- Backward pass; multi‑GPU tensor parallelism; other backends (Metal/Vulkan/OpenCL/HIP); quantized K/V correctness across all shapes; cross‑device parity.

Policy (Deterministic Mode)
--------------------------

- Dispatcher: bypass heuristic kernel chooser and route to deterministic path.
- Kernel selection: prefer vector kernels with `cols_per_block=1` (one query column per block). Use vec‑F16 when available; otherwise vec‑F32. As a last resort, use the tile kernel with `cols_per_block=1`.
- Reduction order: force `parallel_blocks=1` and `stream_k=false` so no cross‑block combine or stream‑k fixup runs.
- Softmax/ALiBi/sinks/GQA: supported; accumulation order remains fixed.

Acceptance Criteria
-------------------

1) Cross‑run determinism: identical bytes for the same inputs across two executions.
2) Batch invariance: for the same token column, `B=1` output equals `B∈{2,8,33}` outputs bitwise.
3) Shapes: D∈{64,128,256}, KV∈{256,1024,4096} (KV must be a multiple of 256), B∈{1,2,8,33}, GQA∈{1,2,4}; mask on/off; ALiBi on/off; sinks on/off.
4) Deterministic mode only; default fast path unchanged.

Implementation Tasks
--------------------

1) Deterministic Dispatcher (CUDA) — implemented in 03A
   - File: `ggml/src/ggml-cuda/fattn.cu`
   - `ggml_cuda_flash_attn_ext(...)` contains an early deterministic branch (no new function) that prefers vec‑F16 → vec‑F32 → tile; bypasses the heuristic picker.
   - All paths pass through `launch_fattn`, which enforces `parallel_blocks=1` and `stream_k=false` in deterministic mode.
   - Optional future: one‑time log when tile fallback is used.

2) Launch Policy: force single‑block accumulation
   - File: `ggml/src/ggml-cuda/fattn-common.cuh`
   - In `launch_fattn<...>(...)`:
     - Early in the function, detect `const bool det = ggml_is_deterministic();`
     - If `det == true`:
       - Force `parallel_blocks = 1` (skip occupancy/efficiency search and avoid allocating `dst_tmp`/`dst_tmp_meta`).
       - Enforce sequencing such that `flash_attn_combine_results` is never launched (it already keys off `parallel_blocks > 1`).
       - Keep `stream_k=false` for deterministic calls (the det dispatcher must only call variants that pass `stream_k=false`).
     - Rationale: guarantees fixed accumulation order and avoids cross‑block nondeterminism.

3) Deterministic vec/tile invocation (one column per block)
   - Files: `ggml/src/ggml-cuda/fattn-vec-f16.cuh`, `ggml/src/ggml-cuda/fattn-vec-f32.cuh`, `ggml/src/ggml-cuda/fattn-tile.cu`
   - The vec `..._case` helpers already pick `cols_per_block=1` for NVIDIA when `Q->ne[1] == 1` or generically on NVIDIA; verify this behavior remains and is used by the deterministic dispatcher.
   - For the tile kernel, invoke via existing helper but ensure the call chain passes `cols_per_block=1` (through the `launch_fattn` head‑size/ncols ladder) and `stream_k=false`.

4) Logging (optional, single‑shot)
   - File: `ggml/src/ggml-cuda/fattn.cu`
   - Add a static flag and a guarded log to note when tile fallback is used in deterministic mode:
     - Example: `GGML_LOG_INFO("[det] attention falling back to tile kernel; expect lower throughput.\n");`

5) Tests — Determinism and Batch Invariance
   - File: `tests/test-attention-determinism.cpp`
   - Harness:
     - Set `GGML_DETERMINISTIC=1` (Windows and POSIX branches as done in existing tests).
     - Build graphs with `ggml_flash_attn_ext(q,k,v,mask, scale, max_bias, logit_softcap)` and `ggml_flash_attn_ext_add_sinks` + `ggml_flash_attn_ext_set_prec` as needed.
     - Initialize tensors with reproducible RNG (reuse `init_tensor_uniform` pattern). Masks padded per kernel requirements.
   - Cases:
     - Head sizes: {64,128,256}
     - KV sizes: {256, 1024, 4096} (KV must be a multiple of 256)
     - Batch sizes: {1, 2, 8, 33}
     - GQA ratios: {1, 2, 4}
     - Toggles: mask on/off; ALiBi on/off (`max_bias`); sinks on/off; precision default (F16 path), and a small sweep forcing vec‑F32 when available.
   - Assertions:
     - Cross‑run determinism: run twice, compare full output buffers bitwise.
     - Batch invariance: compare output slices for a chosen token column at `B=1` vs each `B∈{2,8,33}`.
   - Skips:
     - If CUDA backend not present; keep runtime under a few minutes by selecting a subset grid for CI.

6) Docs — Deterministic Attention (CUDA)
   - File: `docs/DETERMINISM.md`
   - Add a new section “Attention (CUDA)” describing:
     - Deterministic dispatch policy (one‑column vec preferred; tile fallback), `parallel_blocks=1`, `stream_k=false`.
     - Supported shapes and features (D, GQA, masks, ALiBi, sinks) for 03A.
     - Caveats: performance trade‑offs; unsupported shapes may fall back to deterministic tile with lower throughput.
     - Usage examples with `--deterministic` and CUDA build flags.

7) Container: build + run
   - Script: `scripts/build-in-container.sh` (no code change required if already supports `--gpus all`).
   - Add README snippet to run `test-attention-determinism` inside the container with GPUs passed through.

Design Notes / Constraints
--------------------------

- We reuse existing kernels to minimize risk. Determinism arises from fixed dispatch and launch policy, not new math.
- We explicitly avoid `stream_k` and multi‑tile combine to keep reduction order fixed.
- We do not change KV‑cache layout in 03A; tests must validate batch invariance with realistic cache views.

Backlog (03B / 03C)
-------------------

- 03B Coverage & Fallbacks
  - Broaden support for quantized K/V in deterministic mode; ensure vec or tile fallback is deterministic and reasonably fast.
  - Add debug envs: `GGML_DETERMINISTIC_ATTENTION_FORCE_VEC=1`, `..._FORCE_TILE=1` for triage.
  - Expand tests to quantized KV types and more head sizes (80/96/112; Deepseek 576/512).

- 03C Other Backends & KV Cache Invariance
  - Mirror deterministic launch policy in Metal/Vulkan/OpenCL/HIP (single‑column, no cross‑block combine), where feasible.
  - Validate end‑to‑end determinism with incremental decode and cache growth.

Checklist Summary (for PR review)
---------------------------------

- [x] Deterministic dispatcher (inline early branch in `ggml_cuda_flash_attn_ext`) and wiring.
- [x] `launch_fattn` forces `parallel_blocks=1` when deterministic; `stream_k=false` used by deterministic path.
- [ ] (Optional) One‑time log if tile fallback is used.
- [x] Tests: `tests/test-attention-determinism.cpp` cover cross‑run and batch invariance; CUDA‑only skip otherwise.
- [x] Docs updated: `docs/DETERMINISM.md` attention section and quick test run.
- [x] Container instructions in runbook.

Status
------

- 03A implemented and validated:
  - Deterministic dispatcher and single‑block launch policy landed.
  - Tests `test-attention-determinism` pass on NVIDIA Ada (compute 8.9) with `CUDA_VISIBLE_DEVICES` scoping.
  - Docs updated with Attention (CUDA) section.

Next Phases
-----------

- 03B — Coverage & Fallbacks (CUDA)
  - Deterministic quantized KV:
    - Extend the deterministic dispatcher to attempt vector kernels for quantized K/V types (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) before falling back to tile.
    - Add tests for D=128 with the supported quantized pairs (per existing template instances); include masks, GQA.
  - Additional head sizes:
    - Validate D∈{80,96,112} and DeepSeek D=576/DV=512 shapes. Ensure launch policy (parallel_blocks=1, no stream_k) maintains determinism even when MMA paths are used. Add shape‑specific tests.
  - Soft features:
    - Add logit_softcap tests (D=128,256) in deterministic mode; verify vec/tile paths produce identical bits across runs and batches.
  - Diagnostics & controls:
    - Add optional envs `GGML_DETERMINISTIC_ATTENTION_FORCE_VEC=1` and `..._FORCE_TILE=1` to simplify triage and perf checks. One‑time info log of chosen path.
  - Performance note:
    - Document perf deltas for representative shapes (small/medium/long KV) vs non‑deterministic defaults.

- 03C — KV‑Cache Invariance + Other Backends
  - KV‑cache invariance:
    - Audit attention call sites (views into KV cache) to ensure position‑invariant layout regardless of prompt length. Pin KV stride to `FATTN_KQ_STRIDE` boundaries and unify view creation for incremental decode.
    - Add integration test that appends tokens across steps and asserts bitwise equality with an equivalent single‑shot decode for the same positions.
  - Backends:
    - Port deterministic attention policy to Metal/Vulkan/OpenCL/HIP: enforce one‑column per block/workgroup where feasible, disable multi‑block combines, and add backend‑gated tests.
  - Softmax fallback path:
    - For shapes where FlashAttention isn’t available, add a deterministic `soft_max_ext` path (single‑block per row reduction) and tests.

Open Questions / Risks
----------------------

- MMA/WMMA determinism: Validate that single‑block launches for MMA/WMMA variants remain batch‑invariant across devices; otherwise gate to vec/tile with clear messaging.
- Quantized combo coverage varies by `GGML_CUDA_FA_ALL_QUANTS`. Ensure deterministic dispatch respects compile‑time flags and fails over deterministically.
- Multi‑GPU (TP/Pipeline): out of scope here; deterministic reductions would require fixed all‑reduce ordering and chunking in NCCL/HIP — propose as Project 04.
