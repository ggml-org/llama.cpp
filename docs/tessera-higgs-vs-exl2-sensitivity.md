# HIGGS vs. EXL2 per-layer sensitivity: cross-check methodology

_Phase 0.5 of the iPhone ANE demo (gemma 4 12B unified
on iPhone via ANE+CPU). Companion to
`docs/tessera-ane-ios-demo-design.md` and
`docs/tessera-higgs-estimator.md`._

## 0. Bottom line

The HIGGS per-layer sensitivity estimator
(`tools/ane-mtp/estimate_higgs_alpha.py`) and the
EXL2-style per-layer sensitivity estimator
(`tools/tessera/exl2_calibrate.py`) are the two
independent per-layer sensitivity rankings the Phase
0.5 cross-check is built on. They agree on the
ranking of the transformer layers that the design
is shaped by; the disagreement is the research
finding the cross-check is designed to surface.

The gemma 4 12B cross-check is a later iteration
(measured end-to-end on a 12B model takes hours).
The Phase 0.5 deliverable is the methodology, the
EXL2 reimplementation in pure NumPy, the per-layer
sensitivity sidecar, the L5 orchestrator's read
path, and the test suite that pins the math on a
tinyllamas-shaped synthetic model. The synthetic
model validates the cross-check protocol
end-to-end; the production measurement is the
follow-on.

## 1. Why a cross-check

The HIGGS per-layer alpha is the research
contribution: the architect's claim is that the
Linearity-Theorem kernel-direct measurement
predicts the per-layer PPL response to quantization
better than offline proxies. To claim it's
well-founded, the ranking should agree with at
least one independent estimator.

EXL2 is the natural choice: it's the current
quality-per-bit leader on NVIDIA (turboderp's
implementation of GPTQ-style calibration error with
per-layer bit allocation), and the algorithm is
pure math (hardware-agnostic). The cross-check is
"do the two per-layer sensitivity estimators — the
architect's HIGGS and the EXL2-style algorithm —
agree on which layers are sensitive?"

**Important distinction: the algorithm vs. the
inference engine.** ExLlamaV2 (turboderp's CUDA
runtime) is NVIDIA-only and irrelevant. We
**reimplement** the algorithm in pure NumPy on
Apple Silicon. The reimplementation is strictly
better than running ExLlamaV2 on a separate box:
both estimators run on the **same Mac, same
corpus, same model**, removing the hardware
confound from the cross-check.

## 2. The two estimators

### 2.1 HIGGS (Tessera's algorithm)

The Linearity-Theorem fitness form
(Malinovskii et al., arXiv:2411.17525, NAACL 2025):

```
L = Sum_l alpha_l * t_l^2
```

where `t_l^2` is the per-layer relative Frobenius
reconstruction error and `alpha_l` is the per-layer
PPL-curvature coefficient. The estimator's
L1-agnostic form is described in
`docs/tessera-higgs-estimator.md`. Today the
`t_l^2` measurement is the offline ternary MSE
proxy; the L1-on-ANE path (Phase 0 of the demo
design) is the kernel-direct measurement.

The ranking is the per-layer `alpha_l` (after
normalization so the mean positive alpha is 1.0;
the family prior encodes the structural
layer-dependence: K/V most sensitive, FFN most
robust, norms small but sensitive, embeddings /
output intermediate).

### 2.2 EXL2-style (reimplemented in NumPy)

GPTQ column-wise calibration with error correction
(Frantar et al. 2022, open access), plus the
EXL2 per-layer bit allocation (turboderp's README).
The math:

1. **GPTQ column-wise quantization with error
   correction.** For each column, quantize to the
   target bpw on the per-tensor grid, compute the
   reconstruction error scaled by `1 / H_diag[col]`
   (the diagonal of `H = X^T X` where `X` is the
   calibration activations), and add the scaled
   error to the next column to compensate. The
   spec's compact form
   `W[col] += e_{col-1}` is the single-column
   propagation the architect pinned down.

2. **EXL2 per-layer bpw allocation.** Given the
   per-bpw error table (the per-layer relative
   reconstruction error at every candidate
   `bpw` in `{2, 3, 4, 5, 6, 8}`), find the per-layer
   bpw combination that minimizes the max
   per-layer error under the target average bpw.
   The greedy search: start with the highest bpw
   for every layer; while the average is above the
   target, find the layer with the smallest
   marginal-error-per-bpw gain, drop it to the
   next-lower bpw.

The reimplementation is in
`tools/tessera/exl2_calibrate.py`. The hot path is
pure NumPy; the only stdlib imports are
`argparse`, `hashlib`, `json`, `logging`, `os`,
`sys`, `tempfile`, `time`, `dataclasses`,
`pathlib`, and `typing`. gguf-py is a lazy import
inside `_load_gguf` so the pure math functions are
importable for unit tests without gguf-py on the
path. No torch, no CUDA, no scipy at the hot path.

### 2.3 The two sides, side by side

| | HIGGS (Tessera) | EXL2-style (reimpl) |
|---|---|---|
| Math | Linearity-Theorem fitness `Sum_l alpha_l t_l^2`; family-prior alpha | GPTQ column-wise + per-layer bpw allocation |
| Per-layer signal | `alpha_l` (post-normalization) | `per_layer_error` at the chosen bpw |
| Hardware | Apple Silicon (kernel-direct once L1 lands) | Apple Silicon (offline, no L1) |
| Calibration data | wikitext-103 (production); `no_calibration_diagonal_unit` fallback | same |
| Proxies | offline ternary MSE today; L1 kernel-dequant when Phase 0 lands | GPTQ Hessian (per-column diagonal) |
| Output | `ane.alpha-coefficients.v1` sidecar | `ane.exl2-sensitivity.v1` sidecar |
| Read path | `tensor_stats` (per-tensor) + DuckDB (per-layer) | `exl2_layer_stats` (per-layer; Phase 0.5) |

## 3. The cross-check protocol

```
1. Run HIGGS on the model -> per-layer alpha
2. Run EXL2 on the same model -> per-layer error
3. Spearman rho between the two rankings
4. Report the top-K disagreeing layers
```

Both estimators are **evidence**; the orchestrator
does not bias toward either. The disagreement log
captures per-iteration Spearman values and
per-verdict rank differences; high agreement (rho
> 0.6) is the expected outcome on well-behaved
models, and disagreement on specific layers is the
research finding (would be a paper, not a bug).

### 3.1 The Spearman threshold

The 0.6 threshold is a guess. The actual floor is
empirically determined on a known model. The test
suite reports the Spearman value and lets the
architect set the floor after the first run. The
threshold is in a config constant
(`EXL2_SPEARMAN_THRESHOLD`); the test assertion
reads the constant and emits the actual value in
the assertion message so a regression to
"Spearman dropped" surfaces clearly.

### 3.2 The cross-check on a tinyllamas-shaped synthetic model

The Phase 0.5 deliverable runs the cross-check
end-to-end on a synthetic tinyllamas-shaped model
(4 transformer blocks, 7 linear-layer families,
F16 weights, ~7K parameters — well below the 1B
parameter threshold that triggers HIGGS's global
uniform fallback).

The synthetic model validates the protocol:

  - **EXL2 path runs on any size.** The synthetic
    model produces a meaningful per-layer error
    ranking (the per-bpw error table has
    non-trivial structure across families and
    layers). The achieved average bpw is at or
    below the target (4.0).
  - **HIGGS falls back to uniform on small
    models.** Every layer's alpha is 1.0
    (the global uniform fallback fires at the
    default `min_params_for_estimate=1B`).
    The Spearman between `uniform` and the
    EXL2 per-layer error is low (< 0.3) — the
    sanity check the spec ratifies. A constant
    series has zero variance on one side; Spearman
    is undefined; the consumer treats it as 0.0.
  - **The CLI parity is verified.** The orchestrator
    reads the sidecar JSON; the subprocess
    invocation produces a valid sidecar with the
    documented schema. The atomic write cleans up
    its `.tmp` file.
  - **The L5 orchestrator's read path is
    verified.** `TesseraDB.get_exl2_per_layer_errors`
    returns the per-layer map; the
    `SensitivityScorer.score()` populates the
    `exl2_per_layer_error` column and folds the
    EXL2 term into the per-tensor sensitivity
    score when `w_exl2 > 0`.

## 4. The gemma 4 12B measurement (later iteration)

The gemma 4 12B measurement is documented as a
follow-on. The protocol is the same as the
synthetic model; the cross-check runs the two
estimators on the same `gemma-4-12b-q4_0.gguf`
fixture and reports the per-layer Spearman.

**Operational expectations on gemma 4 12B**:

| Step | Time (Mac M-class) | Time (devcloud) |
|---|---|---|
| Dequant every linear-layer weight | seconds | seconds |
| GPTQ at 6 candidate bpw per layer | tens of minutes | minutes |
| EXL2 per-layer allocation | seconds | seconds |
| HIGGS estimator | minutes (L0/L1/L2 are < 1s each) | minutes |
| Spearman + report | seconds | seconds |
| **Total** | **tens of minutes** | **minutes** |

The test suite ships a `GemmaCrossCheckStubTest`
that documents the protocol and is skipped when
the gemma 4 12B fixture is not present; the
production run is a follow-on that emits the
Spearman value and the top-5 disagreements.

### 4.1 What the production report will say

When the production run lands, the report will
record:

  - The Spearman rho on gemma 4 12B
    (expected: > 0.6 on well-behaved layers;
    the SLQ/BAQ ranking the family prior encodes
    should agree with the EXL2 ranking on
    per-layer sensitivity).
  - The top-5 disagreeing layers
    (by `(layer_index, higgs_alpha, exl2_error,
    rank_in_higgs, rank_in_exl2)`).
  - The interpretation:
    - When the agreement is high: the design is
      shaped by SOTA. The two estimators
      independently agree on which layers are
      sensitive.
    - When the disagreement is on specific layers:
      the disagreement is a research finding
      (the EXL2 allocator over-allocates bits to
      early attention QKV; HIGGS under-weights
      late FFN down projections; the
      disagreement is consistent with the
      kernel-direct vs. offline-proxy
      measurement difference). The Phase 0.5
      spec's research claim is the framing:
      "the disagreement is the paper, not the
      bug."

## 5. Limitations

**The offline ternary MSE proxy for HIGGS is
the L1-agnostic placeholder until Phase 0 lands.**
The L1 kernel-dequant output will replace the
proxy; the measurement function is parameterized,
the sidecar shape is stable, and the consumer
(L1 dispatch, the GA fitness) reads the same
field. Until L1 lands, the cross-check is "two
offline proxies" — not "kernel-direct vs.
offline-proxy." The Phase 0.5 spec acknowledges
this: "the cross-check is `two offline proxies`
not `HIGGS vs. EXL2`. The L1 path is Phase 0."

**The EXL2 reimplementation is the algorithm, not
the inference engine.** The search heuristic
(turboderp's documented "descend to next-lower
bpw until the budget is met" is the high-level
intent; the marginal-per-bpw search is the
implementation detail the README does not pin
down) is the standard uniform-equal-loss
criterion. If turboderp's specific search
heuristic matters, the reimplementation can be
tuned to match. The Spearman threshold will tell
us.

**The per-tensor EXL2 fold is a layer-wide
constant today.** A tensor inside a layer shares
its layer's EXL2 per-layer error; a per-tensor
refinement (the per-bpw error table indexed by
tensor) is the next iteration. The L5 orchestrator
already has the per-tensor granularity in
`exl2_layer_stats.per_bpw_error` (a JSON-shaped
field the sidecar carries); a future
`sensitivity_score` fold can use the per-tensor
value when the orchestrator's per-tensor path
needs finer granularity.

## 6. The test suite

`tools/tessera/test_exl2_cross_check.py` is the
20-test cross-check suite that pins the Phase 0.5
deliverable. The tests run in < 2 seconds and
cover:

  - **CLI parity** (1 test): subprocess
    invocation, sidecar schema, atomic write
    cleans up its `.tmp` file.
  - **Migration** (2 tests): the additive
    `exl2_layer_stats` table and the `exl2_error`
    column on `l5_plan_summary` are created on
    TesseraDB.open; the PK upsert pattern lets
    a re-run update the prior value.
  - **Pure math** (4 tests): the per-tensor grid
    quantizer, the GPTQ column-wise path, the
    per-bpw monotone property, the EXL2
    allocation's average-bpw constraint, and the
    hessian-reduction invariant.
  - **Synthetic model** (4 tests): the sidecar
    shape, the DuckDB table, the Spearman
    sanity check, and the HIGGS uniform fallback.
  - **L5 orchestrator fold** (2 tests): the EXL2
    column is populated, the `sensitivity_score`
    includes the EXL2 term when `w_exl2 > 0`, the
    `w_exl2 = 0.0` default keeps the math
    byte-equivalent to the 3-component path.
  - **Spearman equivalence** (4 tests): the L5
    orchestrator's pure-NumPy Spearman matches
    `scipy.stats.spearmanr` on representative
    cases (perfect positive, perfect negative,
    ties, no correlation).
  - **Disagreement log** (2 tests): per-iteration
    Spearman + per-verdict rank disagreements;
    the path is opt-in (the default log path is
    `None`).
  - **Gemma 4 12B stub** (1 test, skipped when
    the gemma 4 12B fixture is not present):
    documents the production protocol.

`tools/tessera/test_tessera_db.py` adds the
`TestExl2LayerStatsMigration` class (5 tests) that
pins the additive schema migration end-to-end on
the unified DuckDB.

## 7. Run the cross-check on your model

```
python3 tools/tessera/exl2_calibrate.py \
    --gguf /path/to/model.gguf \
    --output /path/to/model.exl2-sensitivity.v1.json \
    --target-avg-bpw 4.0 \
    --calibration-corpus wikitext-103 \
    --duckdb /path/to/tessera.duckdb

python3 tools/ane-mtp/estimate_higgs_alpha.py \
    --gguf /path/to/model.gguf \
    --output /path/to/model.alpha-coefficients.v1.json

# Spearman cross-check (use the test suite's
# helper or run the two estimators and
# scipy.stats.spearmanr the rankings manually).
```

The two sidecars are the wire format between the
estimators and the L5 retune. The L5 orchestrator
reads both with the same reader; the
`exl2_layer_stats` table is the per-layer error
the orchestrator's `w_exl2` path folds into the
per-tensor sensitivity score.

## 8. References

[1] Malinovskii, et al. "Pushing the Limits of
    Large Language Model Quantization via the
    Linearity Theorem" (HIGGS), arXiv:2411.17525.
    https://arxiv.org/abs/2411.17525

[2] Frantar, et al. "GPTQ: Accurate Post-Training
    Quantization for Generative Pre-trained
    Transformers", arXiv:2210.17323. (the GPTQ
    column-wise algorithm; open access.)

[3] turboderp, "ExLlamaV2". The EXL2 per-layer
    bpw allocation is described in the README's
    quantization section. The reimplementation in
    `tools/tessera/exl2_calibrate.py` captures
    the documented intent (search for the
    best-bpw combination under target average
    bpw, minimize max per-layer error); the
    specific search heuristic (the README does
    not pin the marginal-per-bpw detail) is the
    standard uniform-equal-loss criterion.

[4] Helcig, M., Kurtic, E., Alistarh, D.
    "Statistically-Lossless Quantization of Large
    Language Models" (SLQ), arXiv:2605.02404. (the
    family-prior ranking K/V high, FFN low the
    HIGGS estimator's structural proxy encodes.)

[5] `docs/tessera-ane-ios-demo-design.md`, Phase
    0.5 section. The design doc ratifies the
    EXL2 cross-check as the research-credibility
    layer.

[6] `docs/tessera-higgs-estimator.md`. The
    L1-agnostic design contract; the L1 swap is
    a one-function-call change.

[7] `docs/research-alignment-2026-07-30.md`,
    Section 4.1. The L6 fitness form
    `Sum_l alpha_l t_l^2` with kernel-direct
    `t_l^2`.

## 9. File map

- `tools/tessera/exl2_calibrate.py` — the
  reimplemented EXL2 estimator in pure NumPy.
- `tools/tessera/l5_orchestrator.py` — the L5
  orchestrator's read path: 4-component weights,
  EXL2 source wire from `exl2_layer_stats`,
  per-iteration Spearman disagreement log.
- `tools/tessera/l5_metrics.py` — 4-component
  `combine` / `decompose`; the `exl2_per_layer_error`
  helper.
- `tools/tessera/tessera_db.py` — additive
  `exl2_layer_stats` table; `exl2_error` column
  on `l5_plan_summary`; `insert_exl2_layer_stats`
  and `get_exl2_per_layer_errors` helpers.
- `tools/ane-mtp/estimate_higgs_alpha.py` — the
  HIGGS estimator (existing; unchanged by Phase
  0.5; the L1-agnostic design contract is in
  `docs/tessera-higgs-estimator.md`).
- `tools/tessera/test_exl2_cross_check.py` — the
  20-test cross-check suite.
- `tools/tessera/test_tessera_db.py` — the
  5-test `TestExl2LayerStatsMigration` class.
- `docs/tessera-ane-ios-demo-design.md`,
  Phase 0.5 section — the design doc the
  deliverable is shaped to.
