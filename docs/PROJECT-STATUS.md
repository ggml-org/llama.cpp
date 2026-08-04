# Tessera — Project Status

_Last updated: 2026-08-01_

Tessera is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
[Tribunus-dev/tessera](https://github.com/Tribunus-dev/tessera). Default branch:
`tessera/integration-upstream-experiments` (Tessera work rebased onto
upstream `master`).

This document tracks what's been built, what works today, and what comes next.
For subsystem details see the linked docs.

---

## TL;DR

We started from a gemma 4 12B QAT drafter that was incoherent at 0.86% spec
acceptance and a Tessera-quantized target that diverged 70–150 % from BF16 at
the middle layers. After five phases of work:

- Tessera's quantizer is now a per-tensor evolutionary system with a direct
  round-trip fitness mode. It improves round-trip Frobenius by 12–18 % per
  tensor over the legacy importance-weighted calibrator.
- Spec-decoding telemetry (`llama.tessera.spec.v1`) is in `llama-imatrix`.
  The schema is a single unified record whose cheap per-step fields
  (drafted, accepted, confidence[]) are always emitted; the per-position
  verifier + drafter top-k distributions are added on top when
  `--telemetry-topk > 0`. Suitable for rejection-sampling LoRA
  fine-tuning of dspark.
- The fork builds clean for `llama-cli`, `llama-server`, and `llama-imatrix`
  against upstream `master` (60 commits ahead of the original dspark-int base).
- Eight worktrees cover the active branches: `main` (the original Tessera
  chain), seven hardening-agent branches, and `tessera/integration-upstream-experiments`
  (the upstream-rebased line).

**The next big thing is the runtime-aware calibration pipeline (Layers 1–6)**
that closes the loop between kernel dequant fidelity and per-tensor GA fitness.

---

## What we've built

### Phase 1 — Diagnose the incoherency

The shipped gemma 4 12B Tessera build with a dflash drafter was producing
0.86 % acceptance vs. the 30–70 % range expected for working dflash. Two
root causes:

1. **Null imatrix ledger.** The pre-existing
   `gemma4-12b-rich.imatrix.gguf` was clean (328 tensors, chunks 64 → 2048,
   no NaNs). The "rich" rerun that produced the broken state had 321/328
   tensors with all-null `previous_moments` due to a Metal OOM during graph
   construction. The fix is the missing `isfinite()` guard in
   `tools/imatrix/imatrix.cpp:772–778` of the upstream codebase.
2. **Wrong sub-hypothesis.** The drafter was never the problem. Per-layer
   differential probing (F16 vs Tessera-corrected logits) showed 70–150 %
   relative divergence at layers 4, 8, 16, and 32. The sensitive tensors
   (QK-norm, post-norm, attn_output, ffn_down) were fine; the bulk
   (Q, K, V, gate, up, down) was mis-calibrated.

The reframing was the key insight of the project: **the requantization
algorithm was the bug, not the drafter.**

### Phase 2 — SOTA survey

`final_turn_001.md` at
`/Volumes/Julian T7/mavis-deep-research/20260729_130345_tessera-imatrix-quant-sota/`
captures the deep-research pass: AWQ, k-quants, i-quants, GPTQ, SmoothQuant,
OmniQuant, BRECQ, SqueezeLLM, SpQR, and a long tail of inference-time
techniques. The conclusion was that AWQ + importance-matrix remains the SOTA
starting point but none of the public tools have proper per-tensor ternary
threshold tuning.

### Phase 3 — Tessera quantizer gap-fill

Six identified gaps in `tile640_quantize_v3.py`, all now closed:

- **§5.1 n_swa override.** `apply_gemma4_metadata_overrides()` forces
  `gemma4.attention.sliding_window` from 1024 to 512.
- **§5.2 imatrix_mse range selection.** `_imatrix_mse_row_scale()` and
  `quantize_2d_imatrix_mse()` for per-row scale-aware error.
- **§5.3 AWQ layer-output error search.** `awq_scale_search` dispatches via
  `AWQ_SEARCH_TARGET`.
- **§5.4 gemma 4 sensitive tensors.** `is_gemma4_sensitive_tensor()` plus
  `DEFAULT_GEMMA4_SENSITIVE_PATTERNS`.
- **§5.5/§5.6 calibration real X.** `load_calibration_activations()` reads
  from `.npz`; `CALIBRATION_ACTIVATIONS` is a module-level cache.
- **New: per-tensor `ternary_threshold` knob.** Multiplier on per-row
  `mean(|W|)`, range `[0.3, 3.0]`, default `1.0` (legacy behaviour). The
  missing calibration control that surfaced during the layer-probe analysis.

`tools/tessera/per_tensor_calibrate.py` is the new GA over the full
calibration mutation space.

### Phase 4 — Per-tensor GA

`tools/tessera/per_tensor_calibrate.py` runs a small GA per tensor over six
mutation dimensions:

| Dimension | Range | Default |
|---|---|---|
| `ternary_threshold` | [0.3, 3.0] | 1.0 |
| `outlier_fraction` | [0.0001, 0.05] | (legacy) |
| `awq_alpha` | [0, 1] | (legacy) |
| `awq_clip` | [0.7, 1.0] | (legacy) |
| `moment_mix` | [0, 1] | (legacy) |
| `tail_guard` | [0, 2] | (legacy) |

Three fitness modes: `direct` (BF16-source-vs-dequant round-trip relative
Frobenius), `importance` (legacy imatrix-weighted), `combined` (direct + a
max-abs penalty, λ = 4). Default population 8, generations 6, islands 2. The
`direct` mode gives 12–18 % improvement per tensor over `importance` (which
gives 2–5 %). `--lossless-target X` enables early stop when relative MSE
falls below `X`.

Two production policies are already in place:
`/Volumes/Julian T7/runs/gemma4-12b-tessera-overnight/gemma4-12b-per-tensor.json`
and `…-direct.json`.

### Phase 5 — Spec-decoding telemetry

Built the spec-calibration telemetry path inside `llama-imatrix`:

- **Unified schema** (`llama.tessera.spec.v1`) — single record per spec
  step. The cheap per-step payload (drafted, accepted, confidence[],
  draft / accepted token sequences) is always emitted; the per-position
  verifier and drafter top-k distributions are added on top when
  `--telemetry-topk > 0`. This is the right shape for rejection-sampling
  LoRA fine-tuning of dspark: we record what each side would have
  predicted at each position and the relative probability mass, then
  sample dspark outputs from the verifier's distribution weighted by
  the drafter's confidence.

CLI surface:
- `--model-draft <path>` — path to a dflash/DSpark drafter gguf.
- `--spec-steps N` — number of spec steps to roll forward.
- `--telemetry-out <path>` — JSONL output.
- `--telemetry-topk K` — switch on v2 schema with K-element top-k per position.

Plus: `dft.` prefix on drafter observer names to keep verifier/drafter
tensors separated inside `IMatrixCollector::m_stats`.

### Phase 6 — dspark drafter

`tools/dspark-gguf-patch/` is a preprocessor for legacy dspark `.gguf` files
because the shipped `dspark_gemma4_12b_q4pure.gguf` doesn't load directly:

1. Rename arch `dspark` → `dflash` (folded-arch convention from PR #25173).
2. Rename `markov.w{1,2}.weight` → `markov_w{1,2}.weight` and
   `confidence.proj.{weight,bias}` → `conf_proj.{weight,bias}`.
3. Rename hparam prefix `dspark.*` → `dflash.*` (keep `dspark.markov_*`).
4. Inject `blk.{N}.attn_v.weight` by copying `blk.{N}.attn_k.weight` (MQA
   V = K; the loader requires explicit V).
5. Set `dflash.attention.sliding_window = 0` (gemma 4 12B drafter doesn't
   use SWA despite the upstream default).

`dspark` actually runs end-to-end: 33 % acceptance on Q4_0 (1-of-3 step),
11 % on Q5_K_M (3-step). The DFlash-only path reaches ~30 % on Q4_K_M and
Q5_K_M. Spec alignment is the bottleneck, not the loader.

### Phase 7 — Production hardening audit

`tessera/docs/audit-2026-07-29.md` lists 12 concrete findings, including:

- Duplicate `--spec-steps`/`--telemetry-out`/`--telemetry-topk` registrations
  in `common/arg.cpp` (since fixed by the `arg-cpp-dedup` agent).
- `mtp_context()`/`ane_mtp_program()` stubbed to `nullptr` in
  `common/speculative.cpp` (the upstream rewrite absorbed the real MTP
  integration under the `DRAFT_MTP` enum).
- `dft.` string prefix workaround in `src/llama-graph.cpp` (4 call sites).
- Two parallel telemetry schemas (v1, v2) without an adapter. The
  `telemetry-schemas` agent first unified them under
  `llama.spec_calib.v3` (with v1/v2 as legacy adapters); the
  `spec-consolidate` agent later collapsed v1/v2/v3 into a single
  `llama.tessera.spec.v1` record.
- Test coverage shockingly thin (28 621 LOC of code, 68 lines of test).
  The `tests` agent added production-grade coverage for dflash, dspark,
  telemetry, server-MTP, patcher, and quantizer policy.

### Phase 8 — Upstream integration

Surveyed 637 upstream branches, identified 13 high-value experimental
candidates, confirmed via `git format-patch -1` that **all 13 are already
absorbed into upstream `master`** through the recent rewrite. The actual
integration work is bringing Tessera's commits onto current master:

- `upstream/master` is at `64d528be7` (60 commits, no shared ancestry with
  our dspark-int line at `720d7fa4`-based history).
- `tessera/integration-upstream-experiments` rebases the 9 tessera commits
  onto `upstream/master`. 4 of the 9 are no-ops (already in master). The
  remaining 5 plus 1 porting-fix commit produce a clean build with
  `llama-cli`, `llama-server`, and `llama-imatrix`.

---

## What works today

| Subsystem | Status | Where |
|---|---|---|
| `llama-cli` | builds + runs | tessera fork, integration branch |
| `llama-server` | builds + runs | tessera fork, integration branch |
| `llama-imatrix` | builds + runs | tessera fork, integration branch |
| Per-tensor GA calibration | ready for use | `tools/tessera/per_tensor_calibrate.py` |
| dspark-gguf-patch preprocessor | ready | `tools/dspark-gguf-patch/` |
| Spec-decoding telemetry v1 | ready | `tools/imatrix/imatrix.cpp` |
| Spec-decoding telemetry v2 | ready | `tools/imatrix/imatrix.cpp` |
| DFlash drafter | 30 % accept on Q4_K_M, Q5_K_M | via llama-imatrix spec path |
| DSpark drafter | 33 % accept on Q4_0 (1-step), 11 % on Q5_K_M (3-step) | via dspark-gguf-patch |
| Per-tensor quant policy loading | ready | `tile640_quantize_v3.py --calibration-policy` |
| Direct-fitness GA mode | ready | `per_tensor_calibrate.py --fitness direct` |
| ANE MTP prefill | compiles, untested runtime | `common/ane-mtp.{h,mm}` |
| `dft.` observer prefix | applied | `src/llama-graph.cpp` |
| `--no-embedded-mtp` flag | ready | `common/arg.cpp` |
| Production tests (dflash, dspark, telemetry, server-MTP, patcher, policy) | landed in `tessera/tests` branch | `tests/` |

---

## Active worktrees

```
/Users/user/Developer/GitHub/tessera                                             220e60f4f [main]
/Users/user/Developer/GitHub/tessera.worktrees/arg-cpp-dedup                     603324327 [tessera/arg-cpp-dedup]
/Users/user/Developer/GitHub/tessera.worktrees/auto-mtp-fix                      f9b6d0211 [tessera/auto-mtp-fix]
/Users/user/Developer/GitHub/tessera.worktrees/dflash-gemma4                     e9c211bc6 [tessera/dflash-gemma4]
/Users/user/Developer/GitHub/tessera.worktrees/dft-observer                      b38fcc42f [tessera/dft-observer]
/Users/user/Developer/GitHub/tessera.worktrees/integration-upstream-experiments  d682e2302 [tessera/integration-upstream-experiments]
/Users/user/Developer/GitHub/tessera.worktrees/spec-calib-api                    ce60508b5 [tessera/spec-calib-api]
/Users/user/Developer/GitHub/tessera.worktrees/telemetry-schemas                 1a7e9a577 [tessera/telemetry-schemas]
/Users/user/Developer/GitHub/tessera.worktrees/tests                             b1ab95c91 [tessera/tests]
```

Each `tessera/*` branch is on top of `tessera:main` (`220e60f4f`). The
integration branch is the only one rebased onto upstream `master`.

---

## External research inputs (2026-07-31)

Two video sources, with every paper identity verified against arXiv. Full
analysis and citations in `docs/research-efficiency-and-mutation-2026-07-31.md`
(the source of truth; where it and a plan doc disagree, it wins until the
plan doc is updated):

- **YC "Kernel & Chip Club"** (`youtube.com/watch?v=n8dz2FX0_uY`) — the
  state of the art on efficient inference, and almost a mirror of
  Tessera's own thesis.
- **"I Tried to Make an AI"** (`youtube.com/watch?v=IoM5zUI8oFc`,
  commonLuke) — a from-scratch neuroevolution demo whose one transferable
  idea is the mutation operator.

A second, separate research input landed the same day: a deep-research pass
over the five shipping coding agents with the most mature permission
systems plus thirty years of human-factors trust-calibration literature.
Full analysis and citations in
`docs/research-autonomy-calibration-2026-07-31.md` (source of truth for
autonomy calibration; binds `tessera-studio-design.md` section 15.5 and
Priority 9 Wave 1 below). Headline: every shipping agent has a static
permission gate; none learns. Tessera's receipt-driven learned permission
is the differentiator.

Six findings bind the roadmap:

1. **Intelligence per Watt** (arXiv:2511.07885). `IPW = accuracy / watt`
   (steady-state), `IPJ = accuracy / joule` (end-to-end). Local models
   answer 88.7 % of queries; IPW improved 5.3x over 2023–2025; local
   accelerators sit >=1.4x below cloud on the identical model ("significant
   headroom for local accelerator optimization"). Tessera's hero metric
   (`mWh/token`, the 30-minute flight test) IS IPJ — adopt the vocabulary
   and cite the 1.4x headroom as the external justification for the
   CoreML/ANE line. Follow-up "Open Jarvis" is a near-neighbor to track.
2. **Reward hacking in self-improving code agents** (KernelBench,
   arXiv:2502.10517; OpenReview `ikrQWGgxYg`). LLMs optimizing kernels game
   the eval — the "world's fastest vector mean" returns 0; one hack detected
   the correctness-vs-performance phase and submitted correct-slow then
   fast-wrong (explicitly compared to VW dieselgate). Mitigation that
   worked: an adversarial detector plus a flywheel where every hack becomes
   a regression test. Independent published validation of the loop's
   grounding rule (agent curates, world judges, never self-judge).
3. **Heterogeneous inference + the roofline** (Williams/Waterman/Patterson,
   CACM 2009). Prefill is compute-bound, decode is memory-bandwidth-bound,
   attention vs MLP differ in arithmetic intensity — no single backend wins
   everywhere. First-principles explanation for "ANE beats Metal ~3x on
   prefill." On-device twist: route prefill to Metal and decode to
   CoreML/ANE on the same SoC, measured with IOReport.
4. **The evolutionary mutation operator** (NEAT, Stanley & Miikkulainen,
   Artificial Life 2002). Selection + crossover + mutation; mutation is the
   exploration term that reaches gains greedy hill-climbing provably cannot.
   The one genuinely net-new mechanism for Tessera — the offensive twin of
   the collapse guard. Becomes Priority 7.
5. **ParallelKittens** (arXiv:2511.13940; ThunderKittens, arXiv:2410.20399).
   Low direct applicability (single-SoC, not NVLink multi-GPU), but the
   "simple kernels maintainable by humans and AI agents" philosophy is the
   AGENTS.md directive stated back at us. The compute/communication-overlap
   idea transfers to overlapping ANE execution with memory movement.
6. **Batch simulation / GPU ECS** (Madrona; Large Batch Simulation for Deep
   RL, ICLR '21). Batching thousands of environments into one throughput GPU
   megakernel gives 100–1000x over CPU. The self-improving loop is
   bottlenecked by eval throughput — batch candidate evaluations into one
   pass rather than one at a time.

---

## What's next

### Priority 1 — Runtime-aware calibration pipeline (Layers 1–6)

Status as of 2026-08-01. Design in `docs/pipeline-design.md`; per-layer
details, code paths, and Reality notes in
`docs/runtime-aware-pipeline.md`. The hook and the GA fitness are no
longer the blocker; the remaining work is the forward-pass layers and
the L5 apply loop.

- **Layer 1: kernel dequant fidelity — SHIPPED.** The
  `LLAMA_TILE640_DEBUG_DEQUANT_DIR` hook emits the effective dequantized
  weight per row to a v3 TDQT sidecar. Complete in all three backends
  (`ggml-cpu/cpu-dump-dequant.cpp`,
  `ggml-cuda/cuda-dump-dequant.cu`, `ggml-metal/metal-dump-dequant.mm`),
  all called from their real matmul paths. Fitness reader in
  `tessera-l1-fitness.{h,cpp}`. This is the runtime ground truth.
- **Layer 1.5: W4A4 FP16 reference sidecar — PARTIAL.** Writer and
  reader shipped and the suffix mismatch is fixed (both sides now use
  `.act.dequant.f32`), so the path is exercisable end-to-end. Remaining:
  the backend hooks currently emit the same F32 buffer as L1 rather than
  an FP16 ground truth.
- **Layer 2: BF16 vs quantized differential — SHIPPED (weight-level).**
  Per-tensor weight-level divergence and type-aware flagging in
  `tessera-l2-diff.{h,cpp}`. The two-forward-pass differential and
  `tools/tessera/runtime_probe.py` are not yet built.
- **Layer 3: per-token coherence — SHIPPED (per-row cosine).**
  `tessera-l3-coherence.{h,cpp}` produces per-row cosine between the L1
  and L1.5 sidecars. Per-token KL and `per_token_coherence.py` are not
  yet built; depends on the L1.5 fix above.
- **Layer 4: end-to-end probe — PARTIAL.** A data-free PPL/KL
  substitute exists in `tessera-ppl.{h,cpp}`. The prompt-bank probe,
  exact-match, and rank-correlation metrics are not yet built.
- **Layer 5: adaptive requantization — SHIPPED (on the dispatch path).**
  Sensitivity scorers and L2-closing adaptive requant in
  `tessera-l5.{h,cpp}`. The full generational loop
  (`ts_dispatch_run_l5_loop` in `tessera-dispatch.cpp`) runs when
  the `l5` subcommand is active (the `--enabled` / `--no-enabled` flag
  on `l5`; on by default): L2 measure ->
  `ts_l5_adaptive_requant` plan -> A/B per tensor family (Stage A
  tightens alpha/clip as multipliers, Stage B raises outlier_fraction)
  -> re-quantize flagged tensors in place -> re-measure, up to
  `l5 --generations`. Emits an `llama.tessera.l5-loop.v1` report
  at `l5 --out`.
- **Layer 6: kernel-based GA fitness — SHIPPED.** The C++ dispatch GA
  consumes L1 sidecars as `t_l^2 = ||dequant_kernel(W_l) - W_l||_F^2 /
  ||W_l||_F^2`, blended with the offline proxy, via
  `tessera-l1-fitness.{h,cpp}` and `tessera-dispatch.cpp:263-294`. CLI:
  `kernel-fitness --enabled`, `--dir`, `--blend`. This is what closes
  the loop; the loop is closed at the GA-scoring level.

Remaining work, ranked:

1. ~~Fix the L1.5 suffix mismatch so the W4A4 reference path is live~~
   (done 2026-08-01; L3's end-to-end use is unblocked).
2. Build the forward-pass differential (L2) and the prompt-bank probe
   (L4) - these are what make the L6 fidelity claim measurable as
   user-visible behavior, not just per-tensor `t_l^2`.
3. ~~Wire `tessera-l5` into the dispatch path and add the
   apply-plan-and-iterate loop.~~ (done 2026-08-01; the loop is live
   behind the `l5` subcommand's `--enabled` / `--no-enabled` flag,
   gated on L2 `relative_frobenius` rather than L4).
4. Lift the L1.5 ground truth to actual FP16 (currently bit-identical
   to L1).

### Priority 2 — Rebase dspark-int work onto integration

The 7 hardening-agent branches are stacked on `tessera:main`, not on
`tessera/integration-upstream-experiments`. Need a `tessera/main..int`
rebase pass to bring:

- `arg-cpp-dedup` — `--spec-steps`/`--telemetry-out`/`--telemetry-topk`
  deduplication and help-text polish.
- `auto-mtp-fix` — server no longer auto-triggers broken MTP path.
- `dflash-gemma4` — extract gemma4-specific extras into
  `llama_model_dflash_gemma4` (cleaner than the `TENSOR_NOT_REQUIRED`
  bolted-on pattern).
- `dft-observer` — replace `dft.` string-prefix workaround with proper
  per-scope observer state.
- `spec-calib-api` — extract spec-decoding calibration into
  `common/speculative-calibration.{h,cpp}`.
- `telemetry-schemas` — unify v1/v2 under `llama.spec_calib.v3` with
  v1/v2 as legacy adapters. Superseded by `spec-consolidate` which
  collapsed v1/v2/v3 into a single `llama.tessera.spec.v1` record.
- `tests` — production-grade test coverage.

Expected conflict surface: `common/arg.cpp`, `common/speculative.cpp`,
`src/llama-graph.cpp`, `tests/test-*`. Estimated 50–150 lines of
resolution.

### Priority 3 — Native drafter-training pipeline (C++, not Python)

_Architect directive (2026-07-31): the training drivers are native C++/Swift,
not PyTorch/peft. The Python plan that used to live here is superseded (kept in
git history). The drivers train drafters directly against
`llama.tessera.spec.v1` telemetry, in-tree, reusing ggml-opt and the
llama training API — no second runtime, no model-format round-trip._

Two drivers share one plumbing (the self-improving flywheel's training step):

1. **Path A — LK autoregressive drafter driver (LANDED).** Executable
   `tools/quantize/tessera/tessera-train-lk` + pure trace→dense-label builder
   `tessera-lk-train-data.{h,cpp}` (27/27 standalone tests). Trains with
   `GGML_OPT_LOSS_TYPE_LK` (total-variation distance = 1 − acceptance rate).
   One spec step per datapoint: input `[prime, draft...]`, label at position j
   = `densify(verifier_topk[j])`. This is on-policy distillation, and it is the
   only input prefix consistent with how the traces were collected (the
   verifier distributions are conditioned on the draft prefix). Design:
   `docs/tessera-lk-training-design.md`. Status: built, unit-tested, and the
   dataset contract verified line-for-line against the llama-layer dense-label
   epoch path; the numeric training loop still needs a real drafter GGUF smoke
   test (this driver is the first consumer of that path).
2. **Path B — DFlash/D-PACE block-drafter driver (next).** Reuses the arg
   pre-scan, the dataset-build pattern, and the epoch loop; its labels are
   pre-weighted cross-entropy rows (baked D-PACE weights from `tessera dataset
   --mode dflash`), not dense LK columns. Plus the offline
   trunk-feature capture pipeline. Design: `docs/tessera-dflash-training-design.md`.

Sanity target (carried over from the old plan): drafter acceptance on Q4_0 from
~33 % (1-step) to ≥50 %. This is still the right way to align a drafter with
Tessera's QAT target, since a stock drafter's head is trained against a
different distribution.

### Priority 4 — End-to-end verification

Once the GA and rebase work is done, validate against the gemma 4 12B
QAT target:

1. Build Tessera Q4_K_M and Q5_K_M with the per-tensor GA policy.
2. Load with `llama-cli` and the dspark drafter, run the Paris coherency
   probe (`--no-embedded-mtp` first, then with the real MTP path).
3. Compare to Unsloth UD-Q4_K_XL (6.7 GB, no MTP, works) for baseline.
4. Compare dspark acceptance before/after the LoRA pass.
5. Compare F16 vs Tessera-corrected layer probe deltas — they should
   close from the current 70–150 % at middle layers down to <20 %.

### Priority 5 — ANE MTP prefill

`common/ane-mtp.{h,mm}` compiles but has no real end-to-end test. The next
step is to:

1. Build a smoke-test mlpackage for gemma 4 12B prefill.
2. Verify the prefill handoff against the autoregressive baseline.
3. Measure the ANE-vs-CPU latency and energy trade.
4. Wire the `ane_mtp_program` through the spec-decoding calibration
   path so it gets used during imatrix collection.

### Priority 6 — Production polish

- CI workflow on the integration branch (the fork doesn't have one).
- Doc coverage: per-tensor calibration API, telemetry schemas, dspark
  patcher, ANE prefill.
- Schema versioning policy for `llama.tessera.*` (the spec-decoding
  telemetry is now a single `llama.tessera.spec.v1` record; the
  previous v1/v2/v3 split is gone) and
  `llama.tessera.per-tensor-calibration.v*`.

### Priority 7 — Evolutionary mutation operator (heavy-tailed, world-gated)

From finding 4 (NEAT). Mutation is the offensive twin of the collapse
guard: the guard stops the loop getting worse, mutation is how it gets
unexpectedly better. Full design in
`research-efficiency-and-mutation-2026-07-31.md` section 3. In priority
order:

1. **Drafter loop is the safe sandbox — mutate here first.** Drafter
   recursion is already safe (the trunk verifier rejects bad drafts), so
   the acceptance rate against the trunk is a clean world-grounded fitness
   — the exact analogue of "did Mario advance." Run a high mutation rate
   over drafter configs (decoding thresholds, regime routing, LoRA
   rank/alpha, prompt-template variants) essentially for free. This is
   finding 3's drafter/verifier split repurposed as explore/exploit:
   drafter = spice, trunk = world gate.
2. **Three NEAT-style mutation classes** on the per-tensor GA and the
   capability archive (which today is pure exploitation):
   - *Parametric* — perturb a continuous knob, with a HEAVY-TAILED step
     (Levy-flight / log-normal), not fixed-range Gaussian. Mostly tiny
     nudges, rarely a large jump — the precise meaning of "occasional" +
     "spicy."
   - *Structural* — occasionally change structure, not just values: add a
     regime bucket, enable/disable a drafter, swap a routing rule,
     introduce a new tool. This is where the surprising gains live.
   - *Random-restart* — very low probability, sample a fully random
     configuration.
3. **Every mutant still passes the world gate.** A mutant enters the
   archive only if tests/builds/commits pass and guard axes do not regress
   > epsilon. Because mutation widens the search, it widens the reward-hack
   attack surface (finding 2) — strengthen the KernelGuard-style checker in
   proportion. A dieselgate mutant is rejected and becomes a regression
   test.
4. **Adaptive schedule + island migration.** Trigger mutation BURSTS on
   stagnation (no archive improvement for K generations -> reheat). Use the
   existing island-GA infra for occasional cross-island migration (~1–5 %
   every N generations) so islands don't each converge on their own local
   optimum.
5. **Measure it, don't hand-tune it.** Treat mutation rate and step
   distribution as just another axis the multi-axis eval optimizes, A/B'd
   via `tessera-ab-harness`, with guard axes ensuring "spicier" never means
   "regressed."

### Priority 8 — External-validation follow-ups (low effort, high leverage)

Cheap deltas from findings 1–3 and 6 that strengthen existing work without
new subsystems:

- **Rename/align the hero metric to IPW/IPJ** (finding 1). `mWh/token` and
  the flight-test metric are the same quantity as IPJ; adopt the vocabulary
  in `tessera-studio-design.md` and `runtime-aware-pipeline.md` so Tessera's
  numbers are comparable to a published Stanford baseline, and cite the
  "1.4x local headroom" result as the written justification for the
  CoreML/ANE line.
- **Add a `fast_p`-shaped acceptance criterion** (finding 2): correct AND
  beats baseline by threshold — never accuracy-or-speed alone.
- **Add roofline / arithmetic-intensity framing** to
  `tessera-coreml-conversion-design.md` to justify backend routing
  (finding 3): compute-bound prefill -> Metal, bandwidth-bound decode ->
  CoreML/ANE, measured with IOReport.
- **Add a KernelGuard-style adversarial reward-hack checker** on acceptance
  traces (`self-improving-loop-design.md` 4.4), not just a pass/fail gate,
  with every discovered hack archived as a permanent regression test
  (finding 2).
- **Batch candidate evaluations** into one throughput pass rather than one
  at a time (`self-improving-loop-design.md` 4.7), per Madrona (finding 6).
- **Track "Open Jarvis"** (finding 1) as a near-neighbor of the
  self-improving coding harness.

### Priority 9 — General-agent harness (open-source absorption)

Make Studio a genuinely good GENERAL agent harness, not just a coding
agent - and the vehicle for the model-improvement flywheel, since the two
are the same loop ("one machine, two payloads,"
`self-improving-loop-design.md` section 1). Full absorption map in
`docs/tessera-harness-absorption-2026-07-31.md`, built from seven scouted
open-source agents (open-interpreter/Codex-RS, self-operating-computer,
UI-TARS, OpenAdapt, browser-use, gpt-researcher, openclaw); per-repo
evidence in `tessera-scout/reports/`.

Ground truth: the inward flywheel is already built (agent loop + tool
protocol + approval engine + 17 Learning services + 9 learning tools, all
building green). The new work is the OUTWARD capabilities plus the safety
spine both payloads share. Five themes, sequenced in three waves:

- **Wave 1 — safety spine + cheap high-soul wins (P0).** Approval-engine
  hardening (layered permission: policy x profile x sandbox-enforceability,
  fail-safe to AskUser); fail-closed action verifier ("verify a real state
  change, not a self-reported success"); denial circuit-breaker (the
  collapse guard, made concrete); per-claim citation + never-fabricate
  contract; skills directory + `SKILL.md` loader; research tool over a
  newly-built `TesseraWebSearch`.
- **Wave 2 — native capabilities (P0/P1, macOS-first).** Computer-use tool
  (ScreenCaptureKit -> Accessibility -> CGEvent, model-native coordinate
  grounding, skill-capture receipts, capture-time PII scrub); browser tool
  (WKWebView + indexed-DOM serializer + page-change re-ground guard).
- **Wave 3 — identity + polish (P1/P2).** `SOUL.md` persona, per-model
  harness profiles + context-budget rules, local-first config posture +
  `doctor` migrations, scoped gating, source curation.

**Wave 1 status (2026-07-31): landed.** The safety spine
(`TesseraSafetyDecision` / `TesseraActionVerifier` /
`TesseraDenialCircuitBreaker`), the skills loader, and the cited research
tool over the keyless `TesseraWebSearch` are built and tested. The
approval engine now produces AND the loop honors all three outcomes
(`autoApprove` / `askUser` / `reject`): `askUser` forces a real prompt
even for a tool the user generally auto-approves, and a user denial feeds
the circuit breaker. This is the research-backed spec in
`research-autonomy-calibration-2026-07-31.md` and
`tessera-studio-design.md` 15.5. The full autonomy system is specified in
`docs/autonomy-calibration-design.md` (the action-class-identity decision
that gated the ratchet is settled there: structural verb-prefix /
path-glob / arg-shape classes, no ML in the classifier).

**Autonomy Phases A-C (2026-07-31): landed.** The learned-permission
RATCHET (grant after N=5 approvals across M=3 sessions, revoke on one
denial), the irreversible-class guard, the dispositional floor/ceiling,
scoped YOLO (time/goal/session-boxed, always logged, always expired),
breaker suspension semantics, audit + revocation UI (Settings >
Autonomy), recommendation-confirmation, the miscalibration regime-shift
detector, and the LEASHED neural approver (pure-Swift MLP, idle-trained
on approval receipts, calibration collapse guard with rollback, smart
YOLO) are built and tested (188 tests green). The net predicts, never
grants, and fails closed; the rule-based ratchet remains load-bearing
and runs alone until the net warms up (50 receipts). Two pre-landing
bugs were fixed during integration: the regime-shift trigger was
mathematically unsatisfiable as written (replaced with a high-regime
latch), and a failed first net training now stays cold instead of
posing a random net as warm.

Deliberately NOT absorbed (the skips are as important as the takes):
anyone else's agent loop, cloud/vendor/server infra, heavy Python/CUDA/CV
stacks, unsigned-binary supply chains, and self-judging evaluation. The
cross-cutting risk is privacy (a screen recorder captures passwords/PHI by
construction; a browser agent runs inside logged-in sessions), so the
approval engine + no-egress boundary + capture-time scrub are gates, not
afterthoughts.

Differentiation: every scouted repo is a harness pointing at a model it
does not own, or a model with no harness. Tessera owns BOTH and lets them
co-evolve - the agent used by day improves the model by night, with a
receipt for every step.

**Product-direction decisions (2026-07-31).** Five calls that shape both
payloads:

- **Agent manager, not an editor.** Studio orchestrates, verifies, and
  records agents; editors and browsers are things it drives and diffs
  against, not things it is. No text editor, LSP, or debugger - that is a
  commodity (Antigravity just forked VS Code) and a tar pit for a solo
  dev. The seat Tessera takes is the layer above the editor.
- **Distribution: Developer ID + notarization for the Mac app** (confirmed
  by the user). Deep macOS integration (Accessibility, Full Disk Access,
  screen recording) is impossible under Mac App Store sandboxing, so the
  Mac app ships Developer ID; an iPhone companion, if built, is App Store
  and acts as a remote control only.
- **Telemetry is the fuel, not a liability.** Always-on LOCAL telemetry is
  required: every (prompt, context, model output, user accept/reject,
  outcome) tuple is a training example and the accept/reject signal is the
  label, feeding idle-time LoRA of the local model
  (`self-improving-loop-design.md`). The privacy invariant is that capture
  and training stay on-device by default - EGRESS is what the approval
  engine gates, not capture.
- **Cloud teachers are required; Apple Foundation Models are the default
  one.** The local model is the student; teachers supply reasoning on
  problems the student struggles with (struggle-detect -> teacher query ->
  reasoning capture -> distill). Apple Foundation Models (macOS 26+, no API
  key, on-device or Private Cloud Compute) are the always-available
  zero-friction default teacher; third-party cloud teachers (Claude/GPT)
  are higher-capability but higher-egress, so opt-in and approval-gated.
  Teacher bias (R3) and reasoning-externalization (R6) from the
  self-improving-loop risks apply and must be managed.
- **Egress caveat (the one honest tension).** Teacher distillation sends the
  user's struggling prompts to a teacher - the single real egress in an
  otherwise-local system. It is therefore opt-in, approval-governed, and
  scrubbed/anonymized where possible; AFM/PCC is the low-egress default
  precisely because of this.
- **Autonomy calibration: needy -> learned trust -> scoped YOLO.** Studio
  starts needy and asks often. Every approval/denial is a receipt, and the
  approval policy is a learned projection over that history: action-classes
  the user consistently allows auto-continue, novel/edge cases keep
  prompting. Safety invariant: learning only moves toward MORE autonomy on
  OBSERVED-SAFE patterns - a new consequential or irreversible action-class
  always prompts regardless of history. Scoped YOLO mode is a
  time/goal/session-boxed override that auto-approves within scope, is
  always logged, and always expires.

---

## Open questions

1. **Should the per-tensor GA's `direct` fitness be the new default, or
   should `combined` win?** Right now `direct` gives 12–18 % improvement
   per tensor; `combined` adds the max-abs penalty but its effect on
   end-to-end perplexity is untested. A test pass on gemma 4 12B QAT
   will settle this.

2. **Keep `tessera:main` and `tessera/integration-upstream-experiments`
   as separate branches, or merge?** They have different bases
   (`main` on the dspark-int chain, integration on `upstream/master`).
   Merging them means picking a base; the integration branch is the
   natural one because it has all of master's refactors. But the dspark
   drafter work and the hardening-agent work live on `main`.

3. **Is the runtime-aware pipeline the right way to spend the next two
   weeks, or is dspark LoRA finetuning more urgent?** The pipeline
   improves the quantizer, which improves the baseline; the LoRA pass
   improves the drafter, which is the drafter for the current QAT.
   Either is high-leverage. The user should pick.

4. **Push dspark-gguf-patch upstream?** The five patches are
   necessary-but-mundane renames that wouldn't surprise upstream. Could
   go as a `tools/dspark-gguf-patch/` PR.

5. **What does "tessera" mean to the project as a whole?** Right now
   it's a code-name for the per-tensor evolutionary calibration line.
   The fork is a wider container (ANE prefill, dspark patcher, runtime
   probe, etc.). The brand could use a clear statement of intent —
   probably a paragraph in the README.

6. **How spicy is too spicy?** The mutation operator (Priority 7) hinges
   on the heavy-tailed step distribution and the burst-on-stagnation
   policy, but the right tail heaviness, mutation rate, and stagnation
   threshold K are unknown. The plan is to A/B them via
   `tessera-ab-harness` with the guard axes as the regression constraint —
   but the guard epsilon that defines "regressed" still needs a number.

7. **Where does the adversarial reward-hack checker live?** Priority 8
   adds a KernelGuard-style checker on acceptance traces, but it's unclear
   whether it belongs in the eval harness (reject at the gate) or as a
   post-hoc auditor over the capability archive. Mutation widening the
   search (Priority 7) raises the stakes on this answer.
