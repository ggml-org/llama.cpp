# SEPTQ retrospective addendum

Addendum to the v1 SEPTQ commit `6179dc753` and to the SEPTQ prod
work on branch `tessera/track-septq-prod` (commits `f80152db4` through
`fcb98e4a3`).

## v1 column-major storage bug

The v1 commit (`6179dc753`) reported `+44% MSE at ratio=0.5` for the
synthetic 4096x4096 regression. The actual SEPTQ improvement on the
v1 synthetic is much larger: the v1 figure was based on a
column-major storage bug in the ternarize-with-threshold output of
`quantize_2d` (`tools/tile640/quantize_v3.py`, the
`ternary_threshold != 1.0` branch of the legacy tessera 2D path).

The legacy path built the per-row ternarization threshold
`per_row_threshold` of shape `(out_dim, 1)`, flattened the weight to
`(out_dim * in_dim,)`, and compared:

```python
# v1 (buggy)
keep = abs_flat >= per_row_threshold.reshape(-1)
```

The right-hand side has shape `(out_dim,)` while the left-hand side
has shape `(out_dim * in_dim,)`. Numpy broadcasts the smaller array
against the larger as if it were column-major: row 0 sees the correct
threshold, but every subsequent row sees the threshold of the row
above it (off by one row's worth of positions). The ternarized output
was correct for row 0 and wrong for every row past it. The bug is
silent (no error, just wrong values) and was hidden in the v1
synthetic by the fact that the rank-8 structure makes the ternarized
output "look reasonable" at a glance.

The fix is in `f80152db4`:

```python
threshold_flat = np.repeat(per_row_threshold.reshape(-1), in_dim)
keep = abs_flat >= threshold_flat
```

`np.repeat(per_row_threshold.reshape(-1), in_dim)` tiles each row's
threshold `in_dim` times so the broadcast is row-major and correct.

## Corrected v1 numbers

With the column-major bug fixed, the v1 synthetic 4096x4096
regression gives:

| mode            | hessian   | improve%  | source commit |
|-----------------|-----------|-----------|---------------|
| `quant_error_h` | diagonal  | **+92.88%** | `f80152db4` |
| `quant_error_h` | banded b=32 | **+91.25%** | `696d3f301` |

The `+91.25%` banded number is on the harness's `--synthetic` path
(the only one that supplies raw calibration activations). The diagonal
`+92.88%` matches the v1 expectation.

## Heavy-tail failure on the realistic bundle

The v1 SEPTQ loses on a rank-32 + 0.1% Student-t(3) heavy-tailed
bundle. The previous SEPTQ prod agent reported `-247%`; on this
branch's realistic bundle (rank-32 + Gaussian + 0.1% Student-t(3)
outliers at 30x the bulk standard deviation) the v1 importance score
loses `-24.05%`. The qualitative finding is the same: the original
importance score `(W - Q(W))^2 * h_diag` is dominated by the
heavy-tail elements (largest |W| -> largest ternarization error
because `Q(W) = sign(W)` for `|W| > row_mean`), so the mask picks
the outliers, and ternarization then destroys their full-precision
values.

The exact `-247%` benchmark lands between `0.5% at 10x std (+28%)`
and `0.5% at 30x std (-409%)` in sweeps, with the `-24%` on this
branch's `0.1% at 30x std` construction sitting in the same regime.
The "10x" multiplier in the user's description is ambiguous
(median / std / max); the construction in
`tools/tessera/septq_build_bundles.py:make_realistic` uses
`outlier_scale=30.0` times the bulk standard deviation as a
calibrated middle ground.

## Weighted importance extension (this branch)

The `tessera/track-septq-prod` branch adds four importance score
modes to `quantize_2d_septq` (commit `e1fe1c4dc`):

| mode            | score                                                    |
|-----------------|----------------------------------------------------------|
| `quant_error_h` | `(W - Q(W))^2 * h_diag` (v1, default)                    |
| `inv_abs_w`     | `(W - Q(W))^2 * h_diag / (\|W\| + eps)`                  |
| `inv_cdf`       | `(W - Q(W))^2 * h_diag * (1 - per-row-CDF(\|W\|))`       |
| `hybrid`        | `base + lambda * h_diag / (\|W\| + eps)` (lambda config) |

A/B table on the v1 synthetic (no tail):

| mode            | diagonal  | banded b=32 |
|-----------------|-----------|-------------|
| `quant_error_h` | +92.88%   | +91.25%     |
| `inv_abs_w`     | +93.80%   | +92.11%     |
| `inv_cdf`       | +94.09%   | +92.26%     |
| `hybrid l=1`    | +97.63%   | +95.99%     |

All four modes win big on the no-tail synthetic. The weighted modes
are slightly better than the original because the `1/(|W|+eps)` and
`1-CDF` weights add a small bias toward the bulk, which is the right
behaviour when there is no heavy tail to protect.

A/B table on the realistic bundle (heavy tail):

| mode            | diagonal  | result      |
|-----------------|-----------|-------------|
| `quant_error_h` | -24.05%   | SEPTQ loses |
| `inv_abs_w`     | -23.67%   | still loses |
| `inv_cdf`       | **+69.03%** | **SEPTQ wins** |
| `hybrid l=1`    | -20.67%   | still loses |

`inv_cdf` is the clear winner on heavy-tailed data. `inv_abs_w` and
the additive `hybrid` are not aggressive enough on this bundle: the
outliers still dominate the importance score. The per-row CDF is the
most aggressive downweighting because the top of the row's magnitude
distribution gets weight ~0, so the mask never picks outliers.

## Honest assessment

The weighted importance extension **partially** fixes the heavy-tail
failure. `inv_cdf` recovers the loss on the realistic bundle (from
`-24%` to `+69%`), but `inv_abs_w` and `hybrid` (with `lambda=1`)
do not. The 1/(|W|+eps) and additive-hybrid weightings are not
aggressive enough to push the outliers out of the mask on this
bundle.

Three options for follow-up work, in order of expected return:

1. **Run on a real-data bundle.** The synthetic heavy-tail is a
   calibrated construction; a real-data bundle (e.g., a Llama-3 or
   Qwen2 attention-output weight at FP16 imatrix) may have a
   different tail shape. The `inv_cdf` result on the synthetic is
   encouraging, but the real-data behaviour could differ.
2. **Sweep `hybrid` lambda.** `lambda=1` normalises the additive
   `h_diag/(|W|+eps)` term to the median of the base importance, but
   the additive form may need a different scaling to push the
   outliers out. A small sweep (`lambda` in {0.1, 1, 10, 100}) on
   the realistic bundle would tell.
3. **Kurtosis-gating as a follow-up.** The user originally chose
   weighted importance (option 2) over the cheap kurtosis-gating
   option (option 1). If `inv_cdf` is not enough on real data, a
   per-row kurtosis gate that disables SEPTQ on rows with heavy
   tails (and falls back to the standard tessera flow for those
   rows) is a cheap next step.

The branch's current state is "weighted importance committed and
validated on the synthetic; `inv_cdf` recovers the heavy-tail failure
on the calibrated realistic bundle." The next step is a real-data
validation; the kurtosis-gating follow-up is conditional on the
real-data result.
