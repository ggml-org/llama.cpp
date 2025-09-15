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

Design Decisions (03B: tile‑first, then MMA)
-------------------------------------------

1) Deterministic dispatcher (landed) chooses vec when supported; F16 tiles as fallback; quantized is vec‑only with clear error otherwise.
2) Special head sizes (80/96/112/576):
   - 03B.1: extend tile to cover D∈{80,96,112}. This path is batch‑invariant and simple to validate. Enabled by default in det mode once it compiles on Ada/Ampere.
   - 03B.3: prototype MMA ncols=1 (single column per block) for 80/96/112 as an optional path, gated by `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1`. Keep tile as fallback.
   - 03B.5: add MMA ncols=1 for 576/512; no tile fallback targeted for 576.

2) Support probing (internal):
   - Use `ggml_cuda_get_best_fattn_kernel(device, dst)` to probe vec-f16/vec-f32/MMA availability for a constructed `dst`. Do not use its result for non-deterministic dispatching — only to avoid calling unsupported vec variants that would abort.
   - New helper (internal to `fattn.cu`): `static bool det_vec_supported(ggml_tensor *dst, bool want_fp16)` to decide vec-f16 vs vec-f32, else false. Another helper `static bool det_mma_supported(ggml_tensor *dst)` for 80/96/112/576.

3) Logging (one-time INFO):
   - If quantized K/V in deterministic mode, log the chosen path (vec-f16/vec-f32). If unsupported, log a helpful error suggesting `K/V=F16` or `D=128` with specific quant pairs.

Implementation Tasks
--------------------

A) 03B.1 — Tile coverage for D∈{80,96,112}
   - [x] Implement deterministic single‑column tile path for F16 K/V at D∈{80,96,112}.
   - [x] Add explicit head‑size mapping in `launch_fattn_tile_switch_head_size` for 80/96/112.
   - [x] Tests: batch invariance + cross‑run determinism integrated into the main grid.
   - [x] Docs: coverage/perf notes updated.

B) 03B.2 — Observability and toggles
   - [x] One‑time INFO when 80/96/112 use tile in det mode.
   - [x] Optional env `GGML_DET_ATTENTION_DISABLE_TILE_80_96_112=1` to disable tile at those head sizes.

C) 03B.3 — MMA ncols=1 prototype (80/96/112)
   - [x] Use existing MMA instances with `ncols2=1` path under deterministic launch policy (no stream‑k, single‑block), gate behind `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1`.
   - [x] Tests: shapes from 03B.1; compare vs tile with tolerance (1e‑3) and assert cross‑run determinism. Gated via `RUN_MMA_PROTO_TESTS=1`.
   - [ ] Validate on Ampere (A4000) in container; if issues, tune config for cc 8.6.

D) 03B.4 — Enable MMA by default for 80/96/112
   - [ ] Switch det default to MMA for these head sizes when available; keep tile fallback.
   - [ ] Perf note/update docs.

E) 03B.5 — 576/512 support (MMA ncols=1 only)
   - [ ] Add DKQ=576, DV=512 ncols=1 MMA; enforce GQA multiple‑of‑16; no tile fallback planned.
   - [ ] Tests: batch invariance + cross‑run, B∈{1,8}, KV∈{256,1024}.

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

- Tile coverage (03B.1): Bitwise‑identical outputs for D∈{80,96,112}, KV∈{256,1024}, B∈{1,8}, GQA∈{1,2}; masks/ALiBi/sinks toggles.
- Quantized K/V: D=128 with {q4_0/q4_0, q8_0/q8_0}; additional pairs when `GGML_CUDA_FA_ALL_QUANTS=ON`, all with determinism and batch invariance.
- MMA ncols=1 (opt‑in) matched numerics on covered shapes; no regressions; gate can remain OFF until soak completes.
- KV multiple of 256 enforced; mask padding per kernel requirements.
- Tests pass on Ada (8.9) and Ampere (8.6) in the CUDA 12.4 container.

Risk & Mitigations
------------------

- Vec support matrix varies with build: probe before dispatch; tests cover minimal and expanded sets; error with guidance.
- Tile compile‑time asserts for 80/96/112: pick safe `kq_stride`/`cols_per_block` combos; keep explicit mapping per D.
- MMA determinism: single‑column path only; keep opt‑in until burn‑in; tile fallback always available for F16 K/V.
- Deterministic mode slowdown: document and provide toggles to opt‑out (disable determinism) or switch paths (FORCE_*).

Timeline (targeted)
-------------------

1) 03B.1 tile coverage for 80/96/112 (1–2 days including compile/layout tuning) — Ada first, then Ampere.
2) 03B.2 observability + toggles (0.25 day).
3) 03B.3 MMA ncols=1 prototype for 80 (1 day), then 96/112 (0.5 day each); opt‑in.
4) 03B.4 flip default to MMA after soak (0.25 day).
5) 03B.5 576/512 MMA ncols=1 (1 day) + tests.
