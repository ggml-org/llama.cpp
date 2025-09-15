Chronological Commit Summary (after 40be511)
===========================================

Scope
-----
This document lists, in chronological order, the eight commits made in our fork after upstream commit `40be511`.
For each commit, we capture the date, author, title, key changes, and notable files touched.

1) 11232b7 — feat: Deterministic RMSNorm
----------------------------------------
- Date: 2025-09-14T00:28:40+05:30
- Author: Diwank Singh Tomer
- What we did:
  - Introduced deterministic mode plumbing hooks and CLI integration for early RMSNorm coverage.
  - Added a dedicated test for RMSNorm batch‑invariance and cross‑run determinism.
  - Seeded Project 01 docs (plan, critique, implementation notes).
- Notable files:
  - `common/arg.cpp` (CLI flag)
  - `ggml/include/ggml.h`, `ggml/src/ggml.c` (det toggle)
  - `tests/test-rmsnorm-determinism.cpp`
  - `projects/01-deterministic-rmsnorm/*`

2) a817d6a — Deterministic numerics: Project 01 (RMSNorm) + Project 02 (MatMul CUDA)
------------------------------------------------------------------------------------
- Date: 2025-09-14T12:30:43+05:30
- Author: Codex CLI
- What we did:
  - Documented deterministic mode in `docs/DETERMINISM.md` and wired CUDA matmul policy for det mode.
  - Implemented deterministic CUDA matmul: prefer `mmf` when eligible; otherwise fixed‑order `mmvf` fallback.
  - Added tests for CUDA matmul determinism (batch invariance, cross‑run) across F32/F16/BF16.
  - Added Project 02 planning/report docs.
- Notable files:
  - `docs/DETERMINISM.md`
  - `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/mmvf.{cu,cuh}`
  - `tests/test-matmul-determinism.cpp`
  - `projects/02-deterministic-matmul/{plan.md,report.md}`

3) cf483c9 — CUDA deterministic MoE (mul_mat_id) groundwork
-----------------------------------------------------------
- Date: 2025-09-14T12:36:24+05:30
- Author: Codex CLI
- What we did:
  - Ensured batch invariance for `mul_mat_id` in det mode by computing per token/slot sequentially when `src1,dst` are F32.
  - Added optional MoE invariance test gate (`TEST_MATMUL_ID=1`).
- Notable files:
  - `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `03-deterministic-attention/report.md` (placeholder planning note)

4) b094602 — Deterministic MoE: F16/BF16 support via F32 promotion
------------------------------------------------------------------
- Date: 2025-09-14T13:21:40+05:30
- Author: Codex CLI
- What we did:
  - Extended deterministic `mul_mat_id` to support F16/BF16 by promoting input columns to F32 before matmul; preserved sequential order.
  - Enabled MoE invariance test by default alongside main matmul checks.
- Notable files:
  - `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `tests/test-matmul-determinism.cpp`

5) 42386a5 — Deterministic Attention (03A): plan/docs/tests
-----------------------------------------------------------
- Date: 2025-09-14T15:43:34+05:30
- Author: Codex CLI
- What we did:
  - Implemented deterministic attention launch policy: `launch_fattn()` forces `parallel_blocks=1` and disables stream‑k in det mode.
  - Added deterministic dispatcher branch scaffolding in attention and an extensive test for batch invariance and cross‑run determinism (masks, ALiBi, sinks; softcap for D=128/256).
  - Added Project 03 plan, 03B phase plan, status, and runbook (Ada/Ampere). Updated docs with KV stride and mask padding constraints.
- Notable files:
  - `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-cuda/fattn.cu`
  - `tests/test-attention-determinism.cpp`
  - `projects/03-deterministic-attention/{plan.md,phase-03B-plan.md,status.md,runbook-03B.md}`
  - `docs/DETERMINISM.md`, `scripts/build-in-container.sh`

6) 9584351 — Deterministic Attention (03B): dispatcher probe + quant + MMA gate
-------------------------------------------------------------------------------
- Date: 2025-09-14T19:25:57+05:30
- Author: Codex CLI
- What we did:
  - Added deterministic dispatcher logic: probe vec availability and fall back deterministically (F16 tile); quantized K/V supported via vec for D=128 pairs (q4_0/q4_0, q8_0/q8_0).
  - Gated MMA path behind `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1`; added tests for FORCE_* toggles and quantized determinism.
  - Clarified docs and updated project status.
- Notable files:
  - `ggml/src/ggml-cuda/fattn.cu`
  - `tests/test-attention-determinism.cpp`
  - `docs/DETERMINISM.md`, `projects/03-deterministic-attention/status.md`

7) 49625c3 — 03B follow‑ups: docs/toggles, tile special sizes, dual‑arch runbook
---------------------------------------------------------------------------------
- Date: 2025-09-14T20:53:08+05:30
- Author: Codex CLI
- What we did:
  - Clarified docs: F16 tile fallback vs quantized no‑tile; noted special head sizes constraints; documented env flags.
  - Extended tile kernel path and observability for special head sizes (80/96/112 single‑column F16 tile) and added disable flag docs `GGML_DET_ATTENTION_DISABLE_TILE_80_96_112`.
  - Enabled mixed Ada+Ampere build note in runbook; tightened tests gating for toggles/MMA protos.
- Notable files:
  - `ggml/src/ggml-cuda/fattn-tile.cu`
  - `docs/DETERMINISM.md`
  - `projects/03-deterministic-attention/*`
  - `tests/test-attention-determinism.cpp`

8) ffe6666 — Project: Progress 03B.3 (MMA prototype + docs/tests/runbook)
----------------------------------------------------------------------------
- Date: 2025-09-14T21:41:47+05:30
- Author: Codex CLI
- What we did:
  - Landed 03B.3 prototype work for MMA ncols=1 on special head sizes (80/96/112), kept behind `GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA=1`.
  - Added opt-in test gating `RUN_MMA_PROTO_TESTS=1` to compare MMA vs deterministic tile (bitwise first, else tol=1e-3) and to verify cross-run determinism on MMA.
  - Refreshed docs (DETERMINISM.md) to call out special head sizes support and the opt-in MMA path; clarified that logit_softcap is unsupported for 80/96/112 in det mode.
  - Updated runbook with prototype toggles and kept Ada validation notes; Ampere soak pending.
  - Updated project status to mark 03B.3 prototype as landed, with 03B.4 (default-enable after soak) and 03B.5 (576/512) still open.
- Notable files:
  - `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-tile.cu`
  - `tests/test-attention-determinism.cpp`
  - `docs/DETERMINISM.md`
  - `projects/03-deterministic-attention/{phase-03B-plan.md,runbook-03B.md,status.md}`
