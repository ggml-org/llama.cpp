Project 03B — Deterministic Attention Coverage (CUDA, Ada-first)
================================================================

Objective
---------

Extend deterministic attention on CUDA to cover:
- Quantized K/V (selected, supported combos in vec kernels) with batch/run determinism.
- Additional head sizes used by modern LLMs (80/96/112; DeepSeek 576/512) under deterministic policy.
- Clear runbook to build and validate on NVIDIA Ada (compute 8.9) and Ampere (compute 8.6) via `scripts/build-in-container.sh`.

Constraints
-----------

- Deterministic mode remains opt-in via `GGML_DETERMINISTIC` and/or the CMake option.
- Maintain default performance when determinism is OFF; no regression in the non-deterministic dispatcher.
- Keep accumulation order fixed by enforcing `parallel_blocks=1` and `stream_k=false` through `launch_fattn()` (already implemented in 03A).
- KV length must be a multiple of 256 (`FATTN_KQ_STRIDE`).

Scope
-----

- CUDA backend only (Ada priority; Ampere used to cross-check).
- Forward pass only (no backward).
- FlashAttention kernels (vec/tile/mma/wmma) as available; prefer vec; allow MMA for shapes not covered by vec/tile.

Non-Goals
---------

- Other backends (Metal, Vulkan, HIP, OpenCL) — Project 03C.
- Multi-GPU determinism (NCCL/collectives) — separate project.

Design Decisions (Deterministic Dispatcher v2)
----------------------------------------------

1) Shape→Kernel selection in deterministic mode (building on 03A dispatcher):
   - Try to choose a vec kernel if supported for the (D, type_K, type_V) triple.
   - For quantized K/V (e.g., Q4_0/Q8_0 at D=128), prefer vec-f16 when `prec==default` else vec-f32.
   - For head sizes without vec/tile support (80/96/112/576), plan to allow MMA path while keeping `parallel_blocks=1` and `stream_k=false` (deterministic). If MMA is not compiled/supported, fail with a clear error. Note: MMA is not used by the current deterministic branch (03A); enabling it is part of 03B work.
   - For F16 K/V, keep current order: vec-f16 → vec-f32 → tile.
   - For quantized K/V, do not fall back to tile (tile expects F16 K/V). If vec support is missing, error out with clear message.

2) Support probing (internal):
   - Use `ggml_cuda_get_best_fattn_kernel(device, dst)` to probe vec-f16/vec-f32/MMA availability for a constructed `dst`. Do not use its result for non-deterministic dispatching — only to avoid calling unsupported vec variants that would abort.
   - New helper (internal to `fattn.cu`): `static bool det_vec_supported(ggml_tensor *dst, bool want_fp16)` to decide vec-f16 vs vec-f32, else false. Another helper `static bool det_mma_supported(ggml_tensor *dst)` for 80/96/112/576.

3) Logging (one-time INFO):
   - If quantized K/V in deterministic mode, log the chosen path (vec-f16/vec-f32). If unsupported, log a helpful error suggesting `K/V=F16` or `D=128` with specific quant pairs.

Implementation Tasks
--------------------

A) Dispatcher updates (ggml/src/ggml-cuda/fattn.cu)
   - [ ] Add support-probe helpers:
     - `static best_fattn_kernel best_kernel_for(const ggml_tensor *dst)` (wraps `ggml_cuda_get_best_fattn_kernel`).
     - `static bool det_vec_supported(ggml_tensor *dst, bool want_fp16)` – true if best kernel is vec-f16 or vec-f32 accordingly.
     - `static bool det_mma_supported(ggml_tensor *dst)` – true if best kernel is mma.
   - [ ] Extend existing deterministic branch in `ggml_cuda_flash_attn_ext(...)`:
     - If `K/V` are quantized:
       - If `prec==GGML_PREC_DEFAULT` and `det_vec_supported(dst, /*want_fp16=*/true)`: call `ggml_cuda_flash_attn_ext_vec_f16`.
       - Else if `det_vec_supported(dst, /*want_fp16=*/false)`: call `ggml_cuda_flash_attn_ext_vec_f32`.
       - Else: `GGML_ABORT` with message: quantized K/V not supported in deterministic mode for this shape; advise F16 K/V or D=128 with q4_0/q8_0.
     - Else if `K/V` are F16:
       - Keep current order vec-f16 → vec-f32 → tile.
     - Else (future types): fall back to existing logic (tile if possible; else error).
     - Head-size exception: if D∈{80,96,112,576} and `det_mma_supported(dst)`: call `ggml_cuda_flash_attn_ext_mma_f16`.
   - [ ] Ensure all calls flow through `launch_fattn`, which already enforces `parallel_blocks=1` and no `stream_k` in deterministic mode.

B) Tests (tests/test-attention-determinism.cpp)
   - Add 2 new groups and gate runtime to CUDA only.

   1. Quantized K/V deterministic tests (D=128):
      - Shapes: D=128, DV=128, H=8, GQA∈{1,2}, KV∈{256, 1024}.
      - Pairs:
        - K/V = Q4_0 / Q4_0; K/V = Q8_0 / Q8_0. (These pairs are supported by the vec kernels in default build.)
      - Data prep:
        - Generate FP32 K_f32/V_f32, then quantize to the target types using `ggml_quantize_chunk()` with `nrow = KV*H_kv` and `n_per_row = D` (or `DV` for V).
        - Create GGML tensors for K/V with the quantized types and set the bytes.
      - Assertions:
        - Batch invariance: B=1 vs B∈{2, 8, 33} on the first query column (`DV*H` floats).
        - Cross-run determinism: repeat with same inputs.
      - Skips:
        - If a quant pair causes a runtime error (unsupported), skip that pair with a console note, not a hard failure.

   2. Additional head sizes (F16 K/V):
      - Heads: D∈{80, 96, 112, 576}, DV matched (576→512 for DeepSeek).
      - GQA constraints per kernel: for 576, require GQA multiple of 16.
      - Assertions same as above.
      - The test expects deterministic success; if unsupported due to build flags, print [SKIP] with reason.

   - [ ] Add helpers:
     - `bool try_run_attention(...)` that returns success/failure without aborting the process (wraps run in a child process or pre-probes combos; if not feasible, guard inputs to only known-safe combos and mark others as skipped).
     - `void quantize_matrix(type, rows, cols, src_f32, dst_bytes)` using `ggml_quantize_chunk`.

C) Docs (docs/DETERMINISM.md)
   - [ ] Expand “Attention (CUDA)” with a “Quantized K/V” subsection: supported pairs, head sizes, and fallbacks (vec only; tile not applicable for quantized K/V).
   - [ ] Add “Special Head Sizes” note: allowing MMA for 80/96/112/576 in deterministic mode with single-block accumulation.

D) Runbook & CI Hooks (projects/03-deterministic-attention)
   - [x] Add `runbook-03B.md` with exact commands:
     - Ada: `ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89' scripts/build-in-container.sh`
     - Ampere: same with `-DCMAKE_CUDA_ARCHITECTURES=86`.
     - Test runs: `docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=<idx> -v "$PWD:/src" -w /src/build-container/bin "$IMAGE" bash -lc 'GGML_DETERMINISTIC=1 ./test-attention-determinism'`.
   - [ ] Optional: propose `RUN_TESTS=1` support in the script (build then run selected tests when CUDA is ON).
   - [ ] Mixed-arch note: build for both `86;89` or scope tests with `CUDA_VISIBLE_DEVICES`.

Acceptance Criteria
-------------------

- Deterministic mode produces bitwise-identical outputs for the following:
  - F16 K/V: D∈{64,128,256} (03A), plus D∈{80,96,112,576} (03B), with masks, GQA, and sinks/ALiBi toggles.
  - Quantized K/V: D=128 with K/V in {Q4_0/Q4_0, Q8_0/Q8_0} across KV∈{256,1024}, B∈{1,2,8,33}.
- Tests pass on Ada (compute 8.9) and Ampere (8.6) in the CUDA 12.4 container using `build-in-container.sh`.
- KV length always a multiple of 256.
- Documentation updated to reflect coverage and caveats.

Risk & Mitigations
------------------

- Vec support matrix is compile-time dependent: we mitigate by probing best kernel to avoid calling unsupported specializations; tests print [SKIP] per unsupported pair.
- MMA determinism: we rely on single-block accumulation to fix reduction order; add targeted tests; if any flakiness surfaces, gate D∈{80,96,112,576} to vec/tile where possible or document unsupported.
- Tile does not support quantized K/V (expects F16) — dispatcher avoids tile for quantized K/V.
- Deterministic mode will be slower (cols_per_block=1, no stream-k, parallel_blocks=1). Document expected slowdowns and how to restore performance (disable determinism).

Timeline
--------

1) Dispatcher support probing + path selection (1 day) — Ada first.
2) Quantized K/V tests & helpers (0.5–1 day), head-size tests (0.5 day).
3) Docs + runbook (0.5 day). Bench/notes (optional: 0.5 day).
