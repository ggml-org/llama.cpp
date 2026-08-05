# HIGGS per-layer alpha estimator (Phase 3 of the iPhone ANE demo)

_Companion to `docs/tessera-ane-ios-demo-design.md` (Phase 3
section) and `docs/research-higgs-alpha-2026-07-30.md` (the
architect's research spine). This document is the on-disk
specification for the estimator that ships in
`tools/ane-mtp/estimate_higgs_alpha.py` and the sidecar JSON
it produces._

## 0. Bottom line

The HIGGS per-layer alpha ``alpha_l`` is the per-tensor weight
in the Linearity-Theorem fitness

    L = Sum_l alpha_l * t_l^2

(Equation 4 in HIGGS, Malinovskii et al., arXiv:2411.17525).
The estimator in `tools/ane-mtp/estimate_higgs_alpha.py`
computes both ``t_l^2`` (the per-layer relative Frobenius
reconstruction error) and ``alpha_l`` (the per-layer PPL
curvature coefficient) and writes them to the
`ane.alpha-coefficients.v1` sidecar JSON. The iOS app's ANE
dispatch (Phase 2's GGUF->IOSurface streaming layer) reads
this sidecar at model load and uses the per-tensor alpha as
a per-tensor weight in the kernel-direct fitness.

The estimator is **L1-agnostic by design**. The
``t_l^2`` measurement is parameterized; today's
implementation is the offline ternary MSE proxy, but the
measurement function is a one-line swap. When Phase 0's L1
kernel-dequant path lands, the same code runs with a
different measurement function (reading the L1 sidecar
instead of the offline reference), and the sidecar shape
stays unchanged. This is the architect's "implement now
with the proxy, upgrade to L1 later" sequencing
(`docs/tessera-ane-ios-demo-design.md` Section "HIGGS
per-layer alpha story").

## 1. The math (the Linearity Theorem)

HIGGS Theorem 1, Eqn. 4 (Malinovskii et al., arXiv:2411.17525):

    E[PPL(W_hat)] ~= PPL(W*) + Sum_l alpha_l * t_l^2

where:

- ``W*`` is the pretrained model (the reference, BF16 or F32);
- ``W_hat`` is a possibly-randomized quantized model;
- ``t_l^2 = E[||W_hat_l - W_l||_F^2] / ||W_l||_F^2`` is the
  **relative per-layer Frobenius reconstruction error** for
  layer ``l`` (Eqn. 3). It is dimensionless (a normalized
  squared norm) and non-negative.
- ``alpha_l`` is the per-layer PPL-curvature coefficient.
  Units are PPL-per-unit-relative-MSE. The paper shows
  ``alpha_l >= 0`` in the valid regime (loss-curvature at a
  local minimum is positive semi-definite).

The theorem holds in the **medium-bitwidth regime** (roughly
``b > 3.0``). Below that, cross-layer error propagation
dominates and the additive linear model breaks down; the QEP
paper (Arai and Ichikawa, arXiv:2504.09629) quantifies the
breakdown. Tessera T640 v1 is in the valid regime; the QEP
off-switch at 3.0 bits is stamped in the sidecar so the
consumer can disable alpha-weighting if a future T640_3D
extension pushes the operating point below it.

The closed-form **through-origin least-squares fit** the
paper uses for Algorithm 3 is

    alpha_l = (Sum_j Delta_{l,j} * t_j^2) / (Sum_j (t_j^2)^2)

where ``Delta_{l,j}`` is the measured PPL (or KL) response
to a Gaussian-noise perturbation of layer ``l`` at noise
level ``t_j`` (with the Eqn. 9 normalization that gives
``E[||G_l - W_l||_F^2] = t^2 * ||W_l||_F^2`` exactly). The
intercept is fixed at zero: the ``t=0`` measurement has zero
error by construction, so an intercept would absorb
baseline noise and bias the slope.

The R^2 of the through-origin fit is

    R^2 = 1 - Sum_j (Delta_{l,j} - alpha_l * t_j^2)^2 / Sum_j Delta_{l,j}^2

with the through-origin reference (not the mean-centered
one). The estimator stamps the R^2 in the per-tensor
sidecar record; a low R^2 triggers the per-layer-uniform
fallback.

## 2. The L1-agnostic design

The estimator today uses an **offline ternary MSE proxy**
for ``t_l^2`` (the design doc's "until L1 lands" caveat).
The L1 path (Phase 0 of the iPhone demo plan) is the L1
kernel-dequant output: the actual reconstruction error
produced by the T640_3D dequant kernel on the ANE. The two
measurements agree when the kernel is correct (the L1
output is by definition a valid round-trip through the
kernel); they differ only when the kernel is buggy, which
is exactly the diagnostic signal the GA wants.

The architecture:

1. The pure math functions (`ternary_round`,
   `ternary_dequantize`, `relative_frobenius_error`,
   `through_origin_slope`, `structural_alpha`,
   `classify_family`) take a NumPy F32 reference and return
   pure numbers. They have no GGUF dependency and are
   unit-tested in isolation.
2. The measurement function (`measure_t_squared_offline`
   today; a future `measure_t_squared_l1` reads the L1
   sidecar) is **parameterized into the orchestrator**. The
   orchestrator's signature is

       estimate(tensors, kv_keys, config, *, measurement)

   so the L1 swap is a one-call change. The unit tests
   pin the parameterization with a synthetic
   constant-measurement function.
3. The model loading (`_load_gguf`, `_dequantize_to_f32`)
   is the only place that imports `gguf-py`. The lazy
   import keeps the pure math functions importable without
   gguf-py on the path.
4. The sidecar JSON shape, the family prior table, the
   through-origin fit, the regime gate, and the uniform
   fallback are all stable. Only the ``measurement``
   field on the sidecar changes when L1 lands (it goes
   from `"offline_ternary_mse"` to `"l1_kernel_dequant"`).

## 3. The structural Hessian-trace proxy (the L1-agnostic alpha)

The Linearity-Theorem theoretical form is

    alpha_l ~= (||W_l||_F^2 / 2) * Tr(H_l)

(``H_l`` is the Hessian of the loss with respect to layer
``l``'s weights at ``W*``). Full Hessian computation is
infeasible at 12B (the paper computed it only on OPT-125M
[1]). The production approach is a Hutchinson
stochastic-trace estimate or a HAWQ-V2 / BAQ
Hessian-proxy surrogate. The estimator's L1-agnostic
default is a **structural family-prior proxy** that
matches the layer-dependence shape the SLQ, BAQ, and
HAWQ-V2 lines of work all agree on:

- K/V projections are the most PPL-sensitive
  (SLQ allocates 8-bit to K/V).
- Output projection is slightly less sensitive
  (SLQ allocates 6-7 bit).
- Q projection is medium-sensitive.
- FFN (gate / up / down) is the most robust
  (SLQ allocates 4-5 bit).
- Norms are small but sensitive.
- Embeddings / output weights are intermediate.

The family-prior table is the **ranking**; the actual
alpha magnitudes are normalized so the mean positive
alpha is 1.0 (uniform alpha = 1.0 = no weighting). This
matches the GA's "sum-to-N" convention used by the D-PACE
loss (`tools/quantize/tessera/tessera-dpace.h:91`).

The proxy deliberately drops the ``(||W_l||_F^2 / 2)``
multiplier. Without a proper Hessian-trace estimate,
multiplying by the Frobenius norm would let large
embeddings dominate the normalization and wash out the
layer-dependence ranking the family prior is trying to
express. The Frobenius norm is reported in the sidecar
for diagnostic purposes; it is not used in the alpha
calculation. The proxy is the **ranking**; the
magnitudes are not meaningful until Algorithm 3 (the
perturbation-sweep fit) is wired in.

When Algorithm 3 becomes cheap (L1-on-ANE), the
orchestrator's measurement parameter switches from
`measure_t_squared_offline` to `measure_t_squared_l1`
(which reads the L1 sidecar's per-tensor relative
error), and the structural proxy is replaced by the
closed-form through-origin fit. The sidecar shape and
the consumer's read path are unchanged.

## 4. The fallback ladder

Per the research doc Section 4 (pitfall P2 and the
fallback ladder), the estimator applies three layered
fallbacks:

1. **Global uniform fallback** when the model's
   parameter count is below ``--min-params-for-estimate``
   (default 1B, the architect's design-doc gate). Every
   layer's alpha is 1.0. The ``t_l^2`` measurement is
   still emitted as the diagnostic; the consumer reads
   the sidecar, sees the global fallback flag, and uses
   uniform alpha.
2. **Per-layer uniform fallback** when a layer's fitted
   alpha is below the positive floor
   (``--alpha-floor-fraction`` of the post-normalization
   mean, default 1e-3). A negative or near-zero alpha
   is a noise artifact (a true alpha is non-negative at
   a local minimum) and must be replaced with the
   positive floor so the GA does not divide-by-zero on
   the fitness normalization.
3. **No fallback** (the production path): the structural
   family-prior alpha is used directly.

The sidecar stamps both the global and per-layer
fallback indicators so the consumer can detect the
degraded path at any granularity. The
``alpha_floor_applied`` field is per-tensor; the
``fallback_global`` and ``fallback_reason`` fields are
sidecar-level.

## 5. The sidecar JSON shape

The sidecar is the wire format between this estimator
and the iOS app's ANE dispatch (Phase 2's streaming
layer). It is designed to round-trip cleanly with the
existing `ane_state_layout.v1` sidecar (the L1 dispatch
reads both at load time).

Top-level:

```
{
  "schema": "ane.alpha-coefficients.v1",
  "version": 1,
  "bundle_name": "<gguf stem>",
  "gguf_path": "<absolute path to source GGUF>",
  "model_hash": "<16 hex chars; cache-invalidation key>",
  "fitness_form": "Sum_l alpha_l * t_l^2",
  "measurement": "offline_ternary_mse" | "l1_kernel_dequant" | <custom>,
  "probe": {
    "metric": "kl_proxy_via_hessian_trace" | "kl" | "ppl",
    "n_tokens": <int>,
    "data_free": <bool>,
    "J": <int>,
    "t2_grid": [<float> ...]
  },
  "regime_gate": {
    "min_operating_bits": 3.0,
    "qep_off_switch": <bool>
  },
  "total_params": <int>,
  "fallback_global": <bool>,
  "fallback_reason": "none" | "<explanation>",
  "layer_count": <int>,
  "layers": [
    <per-tensor record>, ...
  ]
}
```

Per-tensor record (one entry per quantized weight
tensor, in declaration order):

```
{
  "name": "blk.16.attn_v.weight",
  "family": "attn_v" | "attn_k" | "attn_q" | "attn_output"
          | "ffn_gate" | "ffn_up" | "ffn_down" | "norm"
          | "token_embd" | "output" | "other",
  "shape": [<int> ...],
  "n_elements": <int>,
  "frobenius_norm": <float>,
  "t_squared": <float>,                 // offline ternary MSE today
  "t_squared_source": "<measurement function id>",
  "dtype_source": "<gguf dtype name>",  // e.g. "Q4_0", "F16", "F32"
  "alpha": <float>,                     // post-normalization, post-floor
  "alpha_floor_applied": <bool>,
  "fit_r2": <float>,                    // 1.0 for the proxy; < 1.0 for Algorithm 3
  "n_samples": <int>,                   // J for Algorithm 3; 0 for the proxy
  "fallback": "none" | "per_layer_uniform" | "global_uniform"
}
```

The ``schema`` discriminator lets the consumer pick
the right reader. The ``fitness_form`` is the
architect's ratified string; a consumer that sees a
different value knows the sidecar is from a different
version. The ``measurement`` field documents which
source produced ``t_squared`` so a future audit can
trace the value back to its provenance.

## 6. The measurement function contract

The orchestrator's ``measurement`` parameter is the
L1-agnostic hook. The contract is

    Callable[[np.ndarray], tuple[float, str]]

(input is a dequantized F32 reference, output is
``(t_squared, source_label)``). The default
implementation is `measure_t_squared_offline`:

    def measure_t_squared_offline(reference):
        scale = mean(|reference|)
        q = ternary_round(reference)         # {-1, 0, +1}
        recon = ternary_dequantize(q, scale) # {0, scale}
        return relative_frobenius_error(reference, recon), "offline_ternary_mse"

The L1 swap is `measure_t_squared_l1`:

    def measure_t_squared_l1(reference, l1_sidecar, tensor_name):
        return l1_sidecar[tensor_name].t_squared, "l1_kernel_dequant"

A unit test (`EstimateOrchestratorTest::
test_estimate_with_synthetic_measurement`) pins the
parameterization with a synthetic constant-measurement
function. The orchestrator's contract is stable across
measurement implementations.

## 7. Operational notes

### 7.1 Sidecar filename convention

The conventional name is

    <bundle>.alpha-coefficients.v1.json

Mirroring the `ane_state_layout.v1` convention. The
report (markdown) is

    <bundle>.alpha-coefficients.v1.report.md

sibling to the JSON.

### 7.2 Cache invalidation

The `model_hash` is the cache-invalidation key. It is
the first 16 hex chars of SHA-256 over the GGUF header
(first 64KB) + the file's last 64KB. This catches
header changes (tensor count, kv count, tensor-info
block) and the tail of the weight data; the prefix is
the standard "content fingerprint" approach and is
cheap to compute on multi-GB models. Files smaller than
the 64KB window fall through to a single-block hash
(test fixture path).

### 7.3 Phase 2 integration

Phase 2's GGUF->IOSurface streaming layer reads two
sidecars at model load:

1. `ane_state_layout.v1.json` (the existing slot
   table) - tells the iOS app which IOSurface slot each
   function's input/output lives at.
2. `ane.alpha-coefficients.v1.json` (this sidecar) -
   tells the L1 dispatch the per-tensor alpha to use
   when aggregating the kernel-direct ``t_l^2``
   measurement.

The two sidecars are loaded together. The per-tensor
alpha in the alpha sidecar is referenced by the
``name`` field (e.g. ``blk.16.attn_v.weight``), which
matches the GGUF tensor name and the slot name in the
state layout sidecar.

### 7.4 EXL2 cross-check (Phase 0.5)

The architect's research design ratifies an EXL2
cross-check as the research-credibility layer (Phase 0.5
of `docs/tessera-ane-ios-demo-design.md`): compute the
per-layer sensitivity ranking on the same model with
EXL2's GPTQ-style calibration, and verify the
Spearman rank correlation with the HIGGS alpha
ranking. The expected outcome is ``rho > 0.6``; the
top-5 disagreements are a research finding (would be a
paper, not a test failure). The estimator's sidecar is
the per-layer ranking the Spearman correlation runs
against; the EXL2 cross-check is a separate tool
(`tools/tessera/exl2_calibrate.py`, planned for
Phase 0.5).

## 8. References

[1] Malinovskii, et al. "Pushing the Limits of Large
    Language Model Quantization via the Linearity
    Theorem" (HIGGS), arXiv:2411.17525.
    https://arxiv.org/abs/2411.17525

[2] Malinovskii, et al. NAACL 2025 proceedings
    (contains Algorithm 3 and the Linearity Theorem).
    https://aclanthology.org/2025.naacl-long.543/

[3] Arai, Y. and Ichikawa, Y. "Quantization Error
    Propagation: Revisiting Layer-Wise Post-Training
    Quantization" (QEP), arXiv:2504.09629.
    https://arxiv.org/abs/2504.09629
    (the off-switch at 3.0 bits)

[4] Helcig, M., Kurtic, E., Alistarh, D. "Statistically-
    Lossless Quantization of Large Language Models"
    (SLQ), arXiv:2605.02404. (the family-prior ranking
    K/V high, FFN low)

[5] "BAQ: Efficient Bit Allocation Quantization", arXiv:
    2506.05664. (the equal-loss principle)

[6] "HAWQ-V2: Hessian Aware Quantization of Neural
    Networks with Mixed-Precision", arXiv:1911.03852.
    (the Hessian-trace sensitivity predecessor)

[7] "GAMMA: mixed-precision search via stable
    sensitivity ranking across bit budgets", arXiv:
    2605.18475. (the cross-budget stability of the
    sensitivity *ranking*, supporting
    estimate-once-and-cache)

## 9. File map

- `tools/ane-mtp/estimate_higgs_alpha.py` - the
  estimator (CLI + orchestrator + sidecar writer).
- `tools/tessera/test_estimate_higgs_alpha.py` - the
  test suite (52 tests, runs in <2s with the fixture,
  <0.1s without).
- `docs/tessera-ane-ios-demo-design.md` - the design
  doc, Phase 3 section.
- `docs/research-higgs-alpha-2026-07-30.md` - the
  research spine, the math and the cross-check
  acceptance.
- `docs/research-alignment-2026-07-30.md` - the
  research alignment, Section 4.1 (L6 fitness form =
  ``Sum_l alpha_l * t_l^2`` with kernel-direct
  ``t_l^2``).
