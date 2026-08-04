# Runtime-Aware Calibration Pipeline — Implementation Plan

_Last updated: 2026-08-01. Status table and per-layer "Reality" notes
reflect what is actually in the tree; the design prose below each layer
is the original spec and is preserved for context._

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

| Layer | Question it answers | Status | Code path |
|---|---|---|---|
| 1 | What does the kernel actually dequant? | **Shipped** (v3 superset) | `common/tessera-debug/`, backend hooks in `ggml-{cpu,cuda,metal}/*-dump-dequant.*`, fitness in `tessera-l1-fitness.{h,cpp}` |
| 1.5 | W4A4 FP16 reference sidecar | **Partial** (suffix-fixed; FP16 ground truth pending) | `tessera-debug.h` FP16-reference writer, `tessera-l15.{h,cpp}` reader |
| 2 | How does that dequant differ from the BF16 source? | **Shipped (weight-level)**; forward-pass differential is still design | `tessera-l2-diff.{h,cpp}` |
| 3 | What is the per-token coherence cost? | **Shipped (per-row cosine)**; per-token KL is still design | `tessera-l3-coherence.{h,cpp}` |
| 4 | What is the end-to-end behavioural delta? | **Partial** (data-free PPL/KL substitute); prompt-bank probe is still design | `tessera-ppl.{h,cpp}` |
| 5 | Where should we re-quantize? | **Shipped** (scorers + adaptive requant + dispatch loop) | `tessera-l5.{h,cpp}`, loop in `tessera-dispatch.cpp` |
| 6 | Can the GA optimize for the kernel directly? | **Shipped** as the C++ dispatch GA fitness (not the Python `per_tensor_calibrate.py` mode the spec described) | `tessera-l1-fitness.{h,cpp}` consumed by `tessera-dispatch.cpp:263-294` |

L1 was the critical path and has landed; the layers below consume its
sidecar. Where the shipped code does only part of what a layer's prose
describes, a **Reality** callout at the end of that layer's section
states the gap precisely.

> Note on L1.5 (status): the writer/reader suffix mismatch is fixed -
> both sides now use `.act.dequant.f32` (`tessera-debug.h:115`,
> `tessera-l15.{h,cpp}`), so `ts_l15_load_directory` consumes real
> writer output and the L1.5 reference path is exercisable end-to-end.
> Remaining gap: the backend hooks currently populate the L1.5 sidecar
> with the same F32 dequant buffer as L1 rather than an FP16 ground
> truth (acknowledged in the hook comments as a follow-up). Lifting the
> ground truth to actual FP16 is tracked as the next step for the W4A4
> path; the L1 contract above is unaffected.

## Build order and dependencies

```
L1 (kernel hook) ──► L2 (BF16 vs quant differential) ──► L3 (per-token) ──► L4 (E2E)
                                                       │
                                                       └► L5 (adaptive requantize) ──► L6 (kernel fitness)
```

L1 was the critical path and has landed; the layers above now consume
its sidecar. L2, L3, L4 are progressive layers of the same forward-pass
analysis; the shipped implementations cover the weight-level / per-row
forms (see each layer's Reality note). L5 and L6 are the feedback loop;
L6 has closed the GA onto the kernel via the C++ dispatch path.

Estimated effort, person-weeks for a single engineer familiar with the
llama.cpp GGML kernel API (original estimates, retained for context; L1
and the C++ GA wiring have landed):

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

### Reality (as shipped)

L1 has landed and exceeds the spec above. The v3 TDQT sidecar format
(`common/tessera-debug/tessera-debug.h`) adds per-row outlier counts
(LLM.int8()-style 6.0 threshold), per-row timing/kernel_id/dispatch_count
metadata, and a provenance JSON sidecar (kernel_version, main tip,
calibration corpus hash). The hook is called from the real matmul paths
in all three backends: `ggml-cpu.c`, `ggml-cuda.cu`, `ggml-metal-ops.cpp`
(linked via `llama-tessera-debug` in each backend's CMakeLists). CLI
wiring is in `common/arg.cpp` (`--tessera-dequant-dir`,
`--tessera-dequant-stride`, env `LLAMA_TILE640_DEBUG_DEQUANT_DIR`).
Tests: `test_l1_sidecar.cpp`, `common/tessera-debug/test_sidecar_v3.cpp`.
The acceptance criterion above is met.

The L1.5 FP16-reference sidecar (W4A4 mode) is partially shipped: the
writer and reader both exist, but see the open bug note in the Overview.

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

### Reality (as shipped)

The weight-level differential has landed in
`tools/quantize/tessera/tessera-l2-diff.{h,cpp}`: per-tensor
`max_abs` / `mean_abs` / `relative_frobenius` / `per_layer_norm`, the
type-aware tolerance table (2.3), the 1.5x flag decision, and a JSON
report reader/writer using schema `llama.tessera.runtime-probe.v1`.
Test: `test_l2l5.cpp::test_l2()`.

Gap versus the spec above: this is the offline weight-level equivalent.
The full two-forward-pass differential (with `top1_mismatch`,
`top5_mismatch`, `n_samples` per tensor) and the
`tools/tessera/runtime_probe.py` orchestrator are **not** shipped - the
header comment is explicit that "the quantize tool cannot run full
forwards." The acceptance criteria above (no-op F16-vs-F16, Q4_0 vs F16
baseline) are therefore not exercisable from shipped code.

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

### Reality (as shipped)

The per-row weight-level analogue has landed in
`tools/quantize/tessera/tessera-l3-coherence.{h,cpp}`:
`ts_l3_row_cosine`, `ts_l3_tensor_coherence`, and `ts_l3_run` compute
per-row cosine similarity between the L1 kernel sidecar and the L1.5
reference sidecar. Test: `test_l2l5.cpp::test_l3()`.

Gap versus the spec above: this is **per-row weight-level cosine**, not
the per-token KL divergence / top-1 mismatch / `top5_overlap` the spec
describes, and `tools/tessera/per_token_coherence.py` does not exist.
Because L3 reads the L1.5 reference sidecar and L1.5 has the open suffix
bug noted in the Overview, `ts_l3_run` would skip every tensor in
practice until that bug is fixed.

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

### Reality (as shipped)

A data-free PPL/KL substitute has landed in
`tools/quantize/tessera/tessera-ppl.{h,cpp}`: `ts_ppl_compare` computes
`delta_ppl`, `ppl_ratio`, `kl_divergence`, and a pass/fail verdict
against a 0.5 threshold, over a forward callback
(`ts_ppl_forward_fn`). Test: `test_l2l5.cpp::test_l4()` runs it on
synthetic uniform-vs-peaked logits.

Gap versus the spec above: there is no prompt bank (paris / gsm8k /
multi-turn / code), no `exact_match`, no `logit_rank_correlation`, and
no `tools/tessera/e2e_probe.py`. `test_e2e_pipeline.cpp` exists but is
unrelated - it is an 8-step corpus->imatrix->quantize integration test
on synthetic weights, not an L4 probe. The acceptance criteria above
(exits 0 on Tessera-corrected builds, <5 min on 12B) are therefore not
demonstrable from shipped code.

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

### Reality (as shipped)

Two pieces have landed in `tools/quantize/tessera/tessera-l5.{h,cpp}`:

1. Sensitivity scorers (`ts_l5_imatrix_magnitude`,
   `ts_l5_gradient_proxy`, `ts_l5_layer_position_prior`, `ts_l5_combine`,
   EMA, percentile rank, top-fraction picker, quantization ladder
   `ts_l5_step_up` / `ts_l5_step_down`) and a generational orchestrator
   (`ts_l5_orchestrate_step`).
2. L2-closing adaptive requant (`ts_l5_adaptive_requant`): reads an
   `ts_l2_report`, finds flagged tensors, tightens `alpha`/`clip`
   proportional to overshoot, emits an `ts_l5_adaptive_plan`.

L5 is **on the dispatch path**. `ts_dispatch_run_l5_loop` in
`tessera-dispatch.cpp` runs the full generational loop (L2 measure ->
`ts_l5_adaptive_requant` plan -> re-quantize flagged tensors in place ->
re-measure) when the `l5` subcommand is active (the `--enabled` /
`--no-enabled` flag on `l5`; on by default). The loop:

- captures each 2D tensor into a `ts_dispatch_refine_entry` map during
  step 7, so flagged tensors can be re-targeted without re-walking the
  GGUF,
- groups flagged tensors by `ts_regime_infer_family` (attn_q / attn_k /
  attn_v / attn_out / ffn_gate / ffn_up / ffn_down), runs an A/B on one
  representative per family - Stage A tightens alpha/clip as multipliers
  on the GA/policy values (mirroring `ts_expert_profile.alpha_scale` /
  `clip_scale`), Stage B raises `outlier_fraction` by overshoot - and
  applies the winning strategy to every flagged tensor in that family,
- re-quantizes flagged tensors in place into their existing deque
  elements (stable addresses) and refreshes the GGUF descriptors via
  `ts_gguf_repoint_tensor_cluster`,
- emits the loop receipt at `l5 --out` (default: beside
  `policy --out` as `<stem>.l5-loop.json`), schema
  `llama.tessera.l5-loop.v1`, recording per generation: n_flagged,
  n_requant, per-family winning stage + frob_A/frob_B, and per-tensor
  before/after `relative_frobenius`.

Source weights are re-read from the input GGUF per flagged tensor per
generation (no full-source retention), keeping the L5 memory budget flat
at the cost of one read per flagged tensor.

Tests: `test_l5.cpp` covers the sensitivity scorers and orchestrator;
`test_l2l5.cpp` exercises `ts_l5_adaptive_requant` end-to-end on a
loaded L2 report; `test_l5_dispatch.cpp` drives the full dispatch
pipeline with `adaptive_requantize` on a synthetic two-tensor GGUF and
asserts the loop runs, the report is well-formed, and the output GGUF
survives the in-place re-quantization.

Remaining gap versus the spec above: convergence is measured by L2
weight-level `relative_frobenius`, not the L4 prompt-bank probe. The
acceptance criteria above (L4 PASS after one iteration, <1h wall-clock)
are therefore not yet demonstrable - see Layer 4.

---

## Layer 6 — Kernel-based GA fitness

### Goal

Replace the offline `_ternary_reconstruct` reference in the GA's
fitness evaluation with the actual kernel dequant captured in L1. The
GA then optimizes for the deployed fidelity, not the offline reference.

### What's needed

#### 6.1 New fitness mode

The original plan was to add `fitness = "kernel-direct"` to
`per_tensor_calibrate.py`. As shipped, the kernel-direct mode was
implemented in the **C++ dispatch GA** instead of the Python tool (see
Reality below), so the Python `--fitness` choices remain `awq`, `lrq`,
`flrq`, `dartquant`, `compare`. The mode's semantics are as originally
specified:

1. Reads the L1 sidecar for the current tensor.
2. Loads the BF16 source tensor.
3. Computes `relative_frobenius(dequant_kernel, BF16_source)`.
4. Returns that as the fitness.

The offline modes remain as cheap sanity-check comparisons; the
kernel-direct `t_l^2` is the production fitness when a sidecar is
present, with `blend_factor` interpolating to the offline proxy when
sidecar coverage is partial.

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

- The C++ dispatch GA, with `kernel-fitness --enabled` and a sidecar
  directory, consumes L1 sidecars and produces a per-tensor policy
  whose per-tensor fitness is the kernel-direct `t_l^2`.
- The resulting policy, when applied, produces a model that beats the
  offline-proxy fitness on L4 (e2e probe) by at least 10 % on top-1
  match rate. *(Not yet demonstrable - L4 is the partial PPL
  substitute, see Layer 4 Reality.)*
- The offline-proxy fitness is demonstrably worse on L4 than
  kernel-direct for the same model. *(Same caveat.)*

### Reality (as shipped)

L6 has landed in the C++ dispatch GA, not in `per_tensor_calibrate.py`.
`tools/quantize/tessera/tessera-l1-fitness.{h,cpp}` implements
`ts_l1_load_sidecar`, `ts_l1_kernel_direct_t2`, `ts_l1_blended_t2`, and
`ts_l1_compute_all_t2`. `tessera-dispatch.cpp` consumes them at lines
263-294 (per-candidate kernel-direct `t_l^2`, blended with the offline
proxy via `blend_factor`), 725-742 (enable from `params->kernel_fitness`),
and 833-858 (A/B harness report). CLI subcommand: `kernel-fitness` with
`--enabled`, `--dir`, `--blend` flags (in `common/arg.cpp`).
Test: `test_l1_fitness.cpp`.

Two caveats from the audit:

- The QEP off-switch note below is honored - the fitness is purely
  per-tensor `t_l^2` aggregation, no cross-layer error propagation.
- In the standalone dispatch acceptance path (`dispatch.cpp:1346`),
  `at.kernel_direct_t2 = comp_t2;` hardcodes kernel-direct equal to
  offline when no sidecar is present. L6 is therefore effective in the
  GA scoring path but not in the dispatch acceptance verdict.

---

## File-level summary

The original plan called for four new Python tools; the work ultimately
landed primarily in C++ in `tools/quantize/tessera/`. The actual layout:

| Layer | New C++ (shipped) | New Python (shipped) | Modified C++ |
|---|---|---|---|
| 1 | `common/tessera-debug/tessera-debug.{h,cpp}`, `tessera-sidecar-v3.{h,cpp}`, `tessera-l1-fitness.{h,cpp}` | — | `ggml-{cpu,cuda,metal}/*-dump-dequant.*`, `common/arg.cpp` |
| 1.5 | (uses L1 writer) | — | — |
| 2 | `tessera-l2-diff.{h,cpp}` | — | — |
| 3 | `tessera-l3-coherence.{h,cpp}` | — | — |
| 4 | `tessera-ppl.{h,cpp}` | — | — |
| 5 | `tessera-l5.{h,cpp}` | — | — |
| 6 | `tessera-l1-fitness.{h,cpp}` (consumed by `tessera-dispatch.cpp`) | — | `tessera-dispatch.{h,cpp}`, `common/arg.cpp` |

The originally-planned `tools/tessera/runtime_probe.py`,
`per_token_coherence.py`, and `e2e_probe.py` were **not** created; the
forward-pass differential, per-token KL, and prompt-bank probe remain
unimplemented (see each layer's Reality note).

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
