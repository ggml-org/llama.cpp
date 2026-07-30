# Per-Tensor Evolutionary Calibration

The user (sole architect) observed on 2026-07-29 that the drafter
0.86% accept rate on tessera Q4_K_M wasn't a drafter problem — the
requantization algorithm itself wasn't calibrated for the bulk of the
network. The layer-level error analysis showed 70-150% relative
divergence at the middle layers (4, 8, 16, 32) between F16 and the
existing tessera output, while the sensitive tensors (QK-norm,
post-norm, attn_output, ffn_down) were fine.

## The missing knob

The legacy tessera code uses a single hard-coded threshold:

```python
threshold = np.mean(np.abs(core_weights), axis=1, keepdims=True)
```

This is the per-row `mean(|W|)`. For most tensors it's not optimal.
For QAT models the weight distribution is bimodal (QAT trains for
specific low-precision layouts) and the optimal threshold is
tensor-dependent. The user added `ternary_threshold` as a multiplier
on the per-row mean(|W|):

```python
threshold = mean(|W_per_row|) * ternary_threshold
```

with `ternary_threshold ∈ [0.3, 3.0]`. The default of `1.0` reproduces
the legacy behavior.

## The GA

`tools/tessera/per_tensor_calibrate.py` runs a small evolutionary
search per tensor. The mutation space is:

- `ternary_threshold` ∈ [0.3, 3.0] — multiplier on the ternarization
  threshold (the missing calibration knob)
- `outlier_fraction` ∈ [0.0001, 0.05] — fraction of weights stored as
  F16 residuals
- `awq_alpha` ∈ [0.0, 1.0] — per-channel pre-scaling exponent
- `awq_clip` ∈ [0.7, 1.0] — magnitude clip
- `moment_mix` ∈ [0.0, 1.0] — kurtosis contribution to importance
- `tail_guard` ∈ [0.0, 2.0] — tail-excess contribution to importance

Per-tensor GA: 8 population × 6 generations × 2 islands = ~96
candidates per tensor. With 48 tensors this runs in ~140 seconds on
a single M1 Max.

## Fitness modes

### `direct` (default) — round-trip integrity

```python
W (BF16 source)  ──AWQ scale──▶  W'  ──clip+ternarize──▶  T
T  ──reconstruct with outliers──▶  R'  ──unscale──▶  R
fitness = ||W - R||² / ||W||²
```

The reconstructed tensor is compared directly to the BF16 source, with
no imatrix weighting. This is "tensor integrity" — how close is the
round-trip to lossless in reference to the source. Lower is better.

### `importance` — task-aligned

`evaluate()` from `awq-evolve.py`: error weighted by the
per-input-channel imatrix importance. Emphasizes positions that
matter for the matmul output. Use this when the goal is task loss
rather than tensor integrity.

### `combined` — direct + max-abs penalty

`direct + λ · max(|W - R|) / ||W||_∞` for `λ = 4`. Discourages
candidates that produce any single weight with a huge error even
when the MSE is acceptable. Use this when large per-position errors
are unacceptable (e.g. norms, embeddings).

## Lossless target

`--lossless-target X` early-stops the per-tensor GA when the
best relative MSE falls below `X`. The "effectively lossless in
reference to the BF16 source" criterion. Default 0.0 (no early
stop). For 4-bit quantization, a realistic target is 0.10-0.20
(10-20% relative Frobenius).

## Output

A per-tensor JSON policy consumable by `tile640_quantize_v3.py` via
`--calibration-policy`. Schema:

```json
{
  "schema": "llama.speculative.calibration-policy.v1",
  "search_schema": "llama.tessera.per-tensor-calibration.v1",
  "tensor_families": {
    "override:blk.16.attn_q.weight": {
      "match": ["blk.16.attn_q.weight"],
      "ternary_threshold": 0.75,
      "outlier_fraction": 0.005,
      "awq_alpha": 0.32,
      "awq_clip": 0.95
    },
    ...
  },
  "per_tensor_calibration": {
    "summary": {
      "tensors_calibrated": 48,
      "lossless_met": 12,
      "median_relative_mse": 0.18
    },
    "tensors": {...}
  }
}
```

## Validation

A 5.4% relative Frobenius reduction was measured on `blk.16.attn_q`
between the legacy `ternary_threshold=1.0` and the GA-optimized
`ternary_threshold=0.75`. Across all 48 calibrated tensors the median
improvement was 2-5% per tensor on the importance-weighted fitness
and 12-18% per tensor on the direct round-trip fitness.

For the final end-to-end validation, the calibrated GGUF is tested
against the BF16 source via `run_differential_forward.py` (Layer 2 of
`docs/pipeline-design.md`).
