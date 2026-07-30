# Tessera — Project Status

_Last updated: 2026-07-29_

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
- Spec-decoding telemetry (`llama.dflash.acceptance.v1` and
  `llama.spec_calib.v2`) is in `llama-imatrix`. The v2 schema captures full
  per-position verifier + drafter distributions suitable for rejection-sampling
  LoRA fine-tuning of dspark.
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

- **v1 schema** (`llama.dflash.acceptance.v1`) — per-step accept/reject JSONL.
- **v2 schema** (`llama.spec_calib.v2`) — per-position verifier and drafter
  top-k distributions. This is the right shape for rejection-sampling LoRA
  fine-tuning of dspark: we record what each side would have predicted at each
  position and the relative probability mass, then sample dspark outputs from
  the verifier's distribution weighted by the drafter's confidence.

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
- Two parallel telemetry schemas (v1, v2) without an adapter (the
  `telemetry-schemas` agent unified this under `llama.spec_calib.v3` with
  v1/v2 as legacy adapters).
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

## What's next

### Priority 1 — Runtime-aware calibration pipeline (Layers 1–6)

The big unfinished work. Design in `tessera/docs/pipeline-design.md`:

- **Layer 1: kernel dequant fidelity.** `LLAMA_TILE640_DEBUG_DEQUANT=1`
  mode — kernel emits the effective dequantized weight per row to a sidecar
  file. This is the only ground truth of the runtime.
- **Layer 2: BF16 vs quantized differential forward.** Per-tensor capture
  of `max|Δ|`, relative Frobenius, top-1/top-5 divergence.
- **Layer 3: per-token coherence.** KL divergence, top-1 mismatch per
  generated token.
- **Layer 4: end-to-end probe.** 30–50-token runs, exact-match, perplexity
  delta, logit rank correlation.
- **Layer 5: adaptive requantization.** Re-run the per-tensor GA with
  Layer 1 output as the fitness target for tensors where divergence
  exceeds threshold.
- **Layer 6: kernel-based GA fitness.** Replace the synthetic
  `_ternary_reconstruct` reference inside the GA with the actual kernel
  output. This is what closes the loop.

The work lives on `tessera/integration-upstream-experiments` and needs:

1. The `LLAMA_TILE640_DEBUG_DEQUANT=1` env-var hook in `ggml-cuda` and
   `ggml-metal` kernels (currently partial).
2. Python orchestration around the kernel debug output
   (`tools/tessera/runtime_probe.py`).
3. A new GA fitness mode in `per_tensor_calibrate.py` that consumes the
   sidecar file.

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
  v1/v2 as legacy adapters.
- `tests` — production-grade test coverage.

Expected conflict surface: `common/arg.cpp`, `common/speculative.cpp`,
`src/llama-graph.cpp`, `tests/test-*`. Estimated 50–150 lines of
resolution.

### Priority 3 — dspark LoRA finetuning pipeline

Use `llama.spec_calib.v2` telemetry to rejection-sample dspark outputs:

1. **Phase 1: PyTorch dspark implementation** — ~400 lines mirroring
   `dflash.cpp` forward, plus the dspark-gguf-patch loading path. Standalone
   `tools/tessera/dspark_pytorch.py`.
2. **Phase 2: training harness** — `transformers` + `peft` + `accelerate`,
   rejection-sampling loss against the v2 verifier top-k distribution.
3. **Phase 3: first LoRA pass** — `α=β=γ=ε=0` (pure KL to verifier
   top-k) as a sanity check. Sanity target: dspark acceptance on Q4_0
   goes from 33 % (1-step) to ≥50 %.

This is the right way to get dspark aligned with Tessera's QAT target,
since dspark's DeepSeek-style markov head is trained against a different
distribution.

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
- Schema versioning policy for `llama.spec_calib.v*` and
  `llama.tessera.per-tensor-calibration.v*`.

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
