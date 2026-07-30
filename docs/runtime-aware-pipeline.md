# Runtime-Aware Calibration Pipeline — Implementation Plan

_Companion to [`pipeline-design.md`](pipeline-design.md). Where the design
doc describes what each layer is supposed to do, this document describes
what needs to change to build them._

> Roadmap alignment: the runtime-aware proxy-objective research
> (2026-07-30) validates this pipeline as the differentiating capability
> and promotes L1 to the critical path for the C++ port's GA
> (`c++-port-design.md` G4-done and G6 both depend on L1). It also
> sharpens the L6 fitness form and adds a QEP off-switch. See
> [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md);
> inline notes below mark the touched sections.

## Overview

The pipeline has six layers. Each one closes a different gap between the
offline calibration reference and the runtime:

| Layer | Question it answers | Status |
|---|---|---|
| 1 | What does the kernel actually dequant? | **Not started** |
| 2 | How does that dequant differ from the BF16 source? | **Not started** |
| 3 | What is the per-token coherence cost? | **Not started** |
| 4 | What is the end-to-end behavioural delta? | **Not started** |
| 5 | Where should we re-quantize? | **Not started** |
| 6 | Can the GA optimize for the kernel directly? | **Not started** |

Layer 1 must land first. Each later layer consumes the artifacts of the
previous one. The build order below mirrors that.

## Build order and dependencies

```
L1 (kernel hook) ──► L2 (BF16 vs quant differential) ──► L3 (per-token) ──► L4 (E2E)
                                                       │
                                                       └► L5 (adaptive requantize) ──► L6 (kernel fitness)
```

L1 is the critical path. L2, L3, L4 are progressive layers of the same
forward-pass analysis; they can be built in parallel. L5 and L6 are the
feedback loop; L6 closes the GA onto the kernel.

Estimated effort, person-weeks for a single engineer familiar with the
llama.cpp GGML kernel API:

- L1: 1 week (small C++ surface, but invasive across ggml-cpu/cuda/metal)
- L2: 1 week (Python orchestration + JSON schema)
- L3: 0.5 weeks (small Python addition on top of L2)
- L4: 0.5 weeks (CI smoke test harness)
- L5: 1 week (policy application logic + scheduling)
- L6: 1.5 weeks (kernel-based fitness + GA plumbing)

Total: ~5.5 weeks. The L1 surface is the most uncertain; everything else
follows the schema.

---

## Layer 1 — Kernel dequant fidelity (the ground truth)

### Goal

For every Tessera-quantized matmul kernel, when `LLAMA_TILE640_DEBUG_DEQUANT=1`
is set in the environment, the kernel writes the **effective dequantized
weight** for each invocation to a sidecar file. This is the only measurement
that tells us what the runtime is doing, not what the offline reference
thinks.

### What's needed

#### 1.1 Sidecar writer

Add a small C++ helper to `common/tessera-debug.h` (new file, ~80 lines):

```cpp
namespace tessera_debug {
    // Open the sidecar file keyed by tensor name. The first call to this
    // function opens the file; subsequent calls return the cached writer.
    // Sidecar path is $LLAMA_TILE640_DEBUG_DEQUANT_DIR/<tensor_name>.dequant.f32
    // The writer is append-only; one row per matmul call.
    void open_dequant_writer(const char * tensor_name, int64_t rows, int64_t cols);
    void write_dequant_row(int64_t row_idx, const float * data, int64_t n);
    void close_dequant_writer();
    bool dequant_debug_enabled();
}
```

The sidecar file format is a header + N rows of F32 data:

```
header:  magic="TDQT" (4 bytes)  version=1 (uint32)  rows (int64)  cols (int64)  dtype=F32
data:    row_0[0..cols-1], row_1[0..cols-1], ...
```

Header lets us validate the file even if `cols` is variable across kernels.
Dtype field is forward-compatible (F16, BF16, F32 all possible).

#### 1.2 Kernel instrumentation

For each backend, find the Tessera matmul kernels and add the sidecar
writes. The minimum is the Tile640 matmul, but Q4_K, Q5_K, Q6_K also have
runtime-sensitive dequants worth capturing.

Files to modify:
- `ggml/src/ggml-cpu/arch/arm/quants.c` — ARM NEON vec_dot for Q2–Q8
- `ggml/src/ggml-cpu/arch/x86/quants.c` — AVX2/AVX512 vec_dot
- `ggml/src/ggml-cuda/vecdotq.cuh` — CUDA vec_dot
- `ggml/src/ggml-cuda/mmq.cuh` — CUDA MMQ dequant
- `ggml/src/ggml-metal/ggml-metal-ops.m` — Metal matmul ops
- `ggml/src/ggml-metal/ggml-metal.metal` — Metal shader dequants

For each kernel, the pattern is:

```cpp
if (tessera_debug::dequant_debug_enabled()) {
    // The dequantized weight is the K-th operand (typically src0 in vec_dot).
    // Materialize it to F32 and write to the sidecar.
    ggml_compute_forward_..._dequant_to_f32(src0, scratch, ...);
    tessera_debug::open_dequant_writer(tensor_name, ne0, ne1);
    for (int64_t r = 0; r < ne0; r++) {
        tessera_debug::write_dequant_row(r, scratch + r * ne1, ne1);
    }
    tessera_debug::close_dequant_writer();
}
```

The exact materialization pattern depends on the kernel; the CPU kernels
have a row-by-row vec_dot where the dequant is per-row. CUDA and Metal
may have batched dequants.

#### 1.3 CLI gate

Add to `common/arg.cpp`:

```
--tessera-dequant-dir PATH    enable kernel dequant sidecar output to PATH
                              (env: LLAMA_TILE640_DEBUG_DEQUANT_DIR)
```

When set, `tessera_debug::dequant_debug_enabled()` returns true.

#### 1.4 What it should produce

For a single 12B model on a calibration pass of ~32 chunks of 2048 tokens,
the expected sidecar footprint is:

- ~320 tensors in a 12B dense model
- ~4096 rows × ~4096 cols per matmul (worst case: ffn_down)
- F32 = 4 bytes/value
- 4096 × 4096 × 4 = 64 MB per tensor, 320 tensors = 20 GB per chunk

That's too much. Mitigations:

1. **Sparse capture**: only capture every Nth row (`--tessera-dequant-stride 16`)
2. **Hash-only mode**: emit a hash of each row instead of the full data
   (used to detect "did the kernel change for this input", not for round-trip)
3. **Filtered capture**: only capture rows that exceed a divergence threshold
   (this requires L2 first; do it in L5)

For the initial L1, default to stride=16 and emit full F32.

### Acceptance criteria

- `LLAMA_TILE640_DEBUG_DEQUANT_DIR=path/to/dir llama-imatrix -m model.gguf ...`
  produces a directory of `.dequant.f32` files, one per quantized tensor.
- The F32 data round-trips through `np.fromfile()` and matches the BF16
  source under reasonable per-row tolerance (1e-3 to 1e-5 depending on dtype).
- A representative MatMul invocation (one BLK FFN row) emits one row of F32
  with the same values the offline `_ternary_reconstruct` would produce
  (modulo F16 precision).
- Files are written in a streaming fashion; the sidecar directory does
  not double the memory footprint of the model.

---

## Layer 2 — BF16 vs quantized differential forward

### Goal

For a calibration corpus, run two forward passes — one with the BF16 source,
one with the Tessera-quantized model. Per quantized tensor, capture the
divergence between the two passes' matmul outputs. Output: per-tensor JSON
report.

### What's needed

#### 2.1 Python orchestrator

New file `tools/tessera/runtime_probe.py`:

- Loads two models side by side: BF16 source, quantized target.
- Runs the same calibration corpus through both.
- For each quantized tensor, captures the BF16 output and the quantized
  output, computes the divergence metrics.
- Writes a JSON report to `output.json` with the schema below.

#### 2.2 Per-tensor capture

For each quantized tensor `T` in the quantized model:

```python
{
    "tensor": "blk.16.ffn_down.weight",
    "shape": [4096, 4096],
    "n_samples": 32,        # number of forward passes that touched this tensor
    "divergence": {
        "max_abs": 0.034,
        "mean_abs": 0.0012,
        "relative_frobenius": 0.018,
        "top1_mismatch": 0,  # out of 32 samples
        "top5_mismatch": 0,
        "per_layer_norm": 0.014
    }
}
```

Schema name: `llama.tessera.runtime-probe.v1`.

#### 2.3 Dtype-aware tolerances

Different quant types have different expected divergences. Define a per-type
baseline:

| Type | Expected relative Frobenius |
|---|---|
| F16 (sanity) | < 1e-5 |
| Q8_0 | < 1e-3 |
| Q4_K | < 5e-2 |
| Q4_0 | < 5e-2 |
| Tessera-T640 | < 2e-2 (target) |
| Tessera-T640 (per-tensor GA) | < 1e-2 (target) |

A tensor whose divergence exceeds 1.5x its type's expected value is flagged
for requantization (this is what feeds L5).

#### 2.4 Cost

The orchestrator runs two forwards per chunk. For 32 chunks, that's 64
forwards. On a 12B model with M-series Metal, each forward is ~10s; total
~10 minutes per calibration pass. This is acceptable for offline use.

### Acceptance criteria

- `runtime_probe.py --bf16 model-f16.gguf --quantized model-tessera.gguf --chunks 32 --output probe.json`
  produces a JSON report with the schema above.
- A "no-op" test (F16 vs F16 of the same model) returns all metrics near
  machine epsilon.
- A "regression" test (Q4_0 vs F16) returns the expected baseline ranges
  for the uncalibrated quantizer.

---

## Layer 3 — Per-token coherence

### Goal

Generate tokens with both BF16 and quantized models on a fixed prompt. For
each generated token, track the divergence between the two distributions.
This is the per-token behaviour the user actually cares about.

### What's needed

#### 3.1 Token-level KL

For each generated token `t` (positions 1..N):

```python
{
    "position": 7,
    "token": 1234,                    # generated by the quantized model
    "expected_token": 1235,           # generated by the BF16 model
    "verifier_top1": 1235,
    "verifier_top1_prob": 0.71,
    "drafter_top1": 1234,             # (== quantized model's pick)
    "drafter_top1_prob": 0.42,
    "kl_divergence": 0.18,            # D(P_BF16 || P_quantized) at this position
    "top1_mismatch": 1,
    "top5_overlap": 0.6
}
```

Output: `tools/tessera/per_token_coherence.csv`.

#### 3.2 Statistical aggregation

A coherence test should fail (suggesting requantization) if:
- Average KL > 0.1 over the first 50 tokens, OR
- Top-1 mismatch rate > 5 % in the first 50 tokens, OR
- The first 5 generated tokens include any mismatch (the "Paris" test).

This is the test that catches the original 0.86 % dflash acceptance symptom
at the per-token level.

#### 3.3 CLI surface

```
runtime_probe.py --bf16 ... --quantized ... --tokens 50 --per-token output.csv
```

### Acceptance criteria

- A "no-op" run (F16 vs F16) shows KL = 0 on every position.
- A "Q4_0" run shows first-token top-1 mismatch rate comparable to the
  measured dflash acceptance rate (~50 % for Q4_0).
- A "Tessera-corrected" run shows lower divergence than Q4_0 at the same
  perplexity budget.

---

## Layer 4 — End-to-end probe (the smoke test)

### Goal

A short, deterministic 30-50-token generation run that answers the
"is this model coherent?" question with a single number.

### What's needed

#### 4.1 Prompt set

A small bank of deterministic prompts that exercise the model in known
ways:

- `prompts/paris.txt` — "The capital of France is" → expected " Paris"
- `prompts/gsm8k-easy.txt` — a one-step arithmetic problem
- `prompts/multi-turn.txt` — a 4-message exchange with system + user
- `prompts/code.txt` — a small code-completion task

Each prompt has a known-good reference output from the BF16 model.

#### 4.2 Matchers

- `exact_match`: target token sequence equals the reference (greedy decoding).
- `perplexity_delta`: log-perplexity of the quantized model's predicted
  distribution vs the BF16 reference, per token.
- `logit_rank_correlation`: Spearman rank correlation of the BF16 vs
  quantized top-K logits (default K=100).

#### 4.3 CI integration

A single `tools/tessera/e2e_probe.py` that runs all four prompts with both
models and emits a single line of CI status: `PASS`, `WARN` (one mismatch),
`FAIL` (multiple mismatches or perplexity > 2x reference).

This is what the `tests` worktree branch should add as a CI smoke test.

### Acceptance criteria

- `e2e_probe.py --bf16 ... --quantized ...` exits 0 on Tessera-corrected
  builds and non-zero on Q4_0 baseline.
- The probe completes in < 5 minutes on a single 12B model.
- Probe output is JSON-shaped so it can be archived per-build.

---

## Layer 5 — Adaptive requantization

### Goal

Given the L2 report (per-tensor divergence) and the L4 outcome, identify
the tensors that exceed their type's expected divergence and re-run the
per-tensor GA on them with kernel-based fitness. Apply the new policy.

### What's needed

#### 5.1 Divergence → policy mapper

A new mode in `tools/tessera/per_tensor_calibrate.py` that:

1. Reads the L2 JSON report.
2. Identifies tensors with `relative_frobenius > 1.5 * type_expected`.
3. Re-runs the GA on those tensors only, with a kernel-based fitness
   (L6).
4. Writes a delta policy that the existing `--calibration-policy` flag
   can consume as an overlay.

#### 5.2 Scheduling

The full adaptive requantization loop:

```
L2 report ──► divergence_map ──► identify tensors
   │                                  │
   │                                  ▼
   │                            GA per-tensor (L6)
   │                                  │
   │                                  ▼
   └──────────────────────► delta_policy.json
                                       │
                                       ▼
                              apply via tile640_quantize_v3.py
                                       │
                                       ▼
                              re-run L4 (e2e probe)
```

The loop terminates when L4 passes or when no further improvement is
observed (diminishing returns on the per-tensor GA).

#### 5.3 CLI

```
tile640_calibrate_quantize.py \
  --input model-f16.gguf \
  --output model-tessera-v2.gguf \
  --calibration-policy base-policy.json \
  --adaptive-requantize-from probe.json \
  --l4-pass-metric paris-exact-match
```

The orchestrator runs L2 → identify → GA → apply → L4 in one command.

### Acceptance criteria

- A Q4_0 model with bad middle-layer quantization (the original gemma 4
  symptom) reaches L4 PASS after one adaptive-requantize iteration.
- The total wall-clock for the adaptive loop is < 1 hour per model.
- A Tessera-corrected model is stable under the loop (no further changes
  needed; the loop terminates after 0 iterations).

---

## Layer 6 — Kernel-based GA fitness

### Goal

Replace the offline `_ternary_reconstruct` reference in
`per_tensor_calibrate.py`'s fitness evaluation with the actual kernel
dequant captured in L1. The GA now optimizes for the deployed fidelity,
not the reference.

### What's needed

#### 6.1 New fitness mode

Add `fitness = "kernel-direct"` to `per_tensor_calibrate.py`. The mode:

1. Reads the L1 sidecar for the current tensor.
2. Loads the BF16 source tensor.
3. Computes `relative_frobenius(dequant_kernel, BF16_source)`.
4. Returns that as the fitness.

The existing modes (`direct`, `importance`, `combined`) remain for
sanity-check comparisons; `kernel-direct` is the production default once
L1 lands.

> Alignment (2026-07-30): `kernel-direct` is the ground-truth
> instantiation of the Linearity-Theorem term. Per tensor,
> `t_l^2 = ||dequant_kernel(W_l) - W_l||_F^2 / ||W_l||_F^2`, where
> `dequant_kernel(W_l)` is the L1 sidecar (what the kernel actually
> dequantizes), not the offline `_ternary_reconstruct`. The GA objective
> that aggregates these is `Sum_l alpha_l * t_l^2`, with `alpha_l` the
> method-independent layer coefficients estimated once per model (HIGGS
> calibration: perturb each layer, measure PPL response) and cached in
> the sidecar / policy. If `alpha_l` estimation proves noisy, fall back
> to uniform weights; the form still holds structurally.
>
> QEP off-switch: do NOT add cross-layer error propagation to this
> fitness for TESSERA_T640 v1. The Linearity Theorem holds in this
> regime; QEP (arXiv:2504.09629) shows the cross-layer correction only
> pays off sub-3-bit. Revisit only for a T640_3D sub-3-bit extension or
> the W4A4 activation boundary. See
> research-alignment-2026-07-30.md Sections 4.1 and 7.

#### 6.2 GA plumbing

The GA itself doesn't change — it just calls a different fitness function.
The 6D mutation space, population, generation, island count are all
unchanged. The output policy schema is the same.

#### 6.3 Cost

`kernel-direct` requires the L1 sidecar to exist. Generating the sidecar
takes ~1-2 minutes per tensor (kernel time on the calibration input). With
48 tensors in a 12B model and a 6-gen × 8-pop × 2-island GA, that's ~150
fitness evaluations per tensor → ~5 hours per calibration pass. This is
acceptable for offline use, comparable to the existing `direct` mode.

### Acceptance criteria

- `per_tensor_calibrate.py --fitness kernel-direct --calibration-input path/to/calib.txt`
  consumes L1 sidecars and produces a per-tensor policy.
- The resulting policy, when applied, produces a model that beats the
  `direct` mode on L4 (e2e probe) by at least 10 % on top-1 match rate.
- The GA's `importance` mode (legacy) is demonstrably worse on L4 than
  `kernel-direct` for the same model.

---

## File-level summary

| Layer | New C++ | New Python | Modified C++ | Modified Python |
|---|---|---|---|---|
| 1 | `common/tessera-debug.h` | — | 5 ggml kernel files | — |
| 2 | — | `tools/tessera/runtime_probe.py` | — | — |
| 3 | — | `tools/tessera/per_token_coherence.py` (or fold into runtime_probe.py) | — | — |
| 4 | — | `tools/tessera/e2e_probe.py` | — | `tests/test-*.cpp` (smoke) |
| 5 | — | — | — | `tools/tessera/per_tensor_calibrate.py` |
| 6 | — | — | — | `tools/tessera/per_tensor_calibrate.py` |

Total: 1 new C++ file, 4 new Python tools, 5 modified C++ kernel files, 2
modified Python tools, 1 modified test file.

---

## What this unblocks (and what it doesn't)

### Unblocks

- A defensible end-to-end correctness story for Tessera-quantized models:
  "the L4 probe passes on the per-tensor-GA policy" is a concrete
  acceptance criterion.
- Honest comparison to Q4_0 / Q4_K / Q5_K baselines on the same prompt
  set.
- Detection of regressions in the kernel — if L1 sidecar changes
  unexpectedly between releases, that's a red flag.
- A path to validating new quant types (Q2_K, ternary, NVFP4) against
  runtime fidelity instead of just offline round-trip.

### Doesn't unblock

- The 0.86 % dflash acceptance issue on the original dflash drafter. That
  needs the LoRA finetuning path (separate workstream, see
  `docs/PROJECT-STATUS.md` Priority 3).
- The auto-MTP path in `tools/server`. The MTP integration is upstream's
  work; we just have a `--no-embedded-mtp` workaround.
- The ANE prefill path. The `.mm` compiles but isn't end-to-end tested.

---

## Open questions

1. **Capture everything or just Tile640?** The L1 plan captures all
   quantized matmul kernels. A more focused version would only capture
   Tile640 (the Tessera-specific one) and let other types use the
   existing offline reference. Capturing all kernels gives better
   diagnostics but at higher build complexity. Recommend: capture all,
   with a `--tessera-dequant-filter` regex to limit.

2. **How to handle kernels that batch dequants?** Some CUDA MMQ kernels
   materialize dequant to a shared scratch buffer once and reuse across
   the matmul. The sidecar writer needs to dedupe by `(tensor, row)` to
   avoid emitting the same data multiple times.

3. **How does this interact with KV-cache quantization?** Once we add
   quantized KV cache (upstream work in progress), the L1 capture should
   also include `ggml_mul_mat` calls against quantized cache tensors.
   Probably a follow-up layer (L7?).

4. **Should L4 be a CI gate?** A 5-minute probe is expensive for fast CI
   but cheap for nightly. A two-tier approach: per-PR runs the probe
   against a small model (e.g., 1B), nightly runs against the 12B.

5. **Can the GA be made faster?** 5 hours per calibration pass is the
   bottleneck. Options: cache intermediate fitness evaluations, parallelize
   across cores, or use a surrogate model (a small NN that predicts
   kernel-direct fitness from the offline metrics).
