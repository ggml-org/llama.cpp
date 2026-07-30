# HIGGS per-layer alpha_l estimation: implementation spec for Tessera stage 2

_Date: 2026-07-30. Source: deep-research run
`higgs-alpha-estimation`. Companion to
`research-alignment-2026-07-30.md` (which ratifies the staged plan:
ship uniform weights first, add per-layer alpha_l as a follow-on
refinement with a permanent uniform fallback) and to
`runtime-aware-pipeline.md` L6 (the `kernel-direct` fitness). This
document is the implementation spec for the alpha_l estimation layer.
Where this document and a plan doc disagree on the estimation
procedure, this document wins until the plan doc is updated._

## 0. Bottom line

The HIGGS paper gives an **explicit, paper-faithful fitting procedure**
for alpha_l. We do not have to infer one. It is Algorithm 3
("Error coefficient calibration") [1][2]: perturb exactly one layer at
a time with Gaussian weight noise at J noise levels, measure the
perplexity (or KL) response at each level, and fit alpha_l as the
through-origin least-squares slope of delta-PPL against the squared
relative perturbation norm t_j^2. The coefficients are a property of
the pretrained weights and the PPL/KL loss surface only; they are
independent of the quantization method and (within the valid regime)
of the bitwidth. That makes them estimate-once-and-cache.

Three decisions fall out of the primary sources and should be locked:

1. **Estimator:** Gaussian-noise perturbation sweep + through-origin
   linear fit (Algorithm 3), not direct Hessian computation. The
   Hessian-trace form is interpretation and a cheap sanity cross-check
   only; the paper states full Hessian computation was feasible only on
   OPT-125M [1].
2. **Probe metric:** the data-free KL variant is acceptable for
   production. HIGGS reports KL on 287k random tokens performs nearly
   identically to data-dependent PPL on 287k WikiText-2 train tokens
   [1]. KL is also independently argued to be a tighter PPL proxy than
   MSE/SQNR [8]. This removes the calibration-dataset dependency from
   the harness.
3. **Regime gate:** the additive linear model is valid in the
   medium-bitwidth regime (roughly b > 3.0) and breaks down sub-3-bit,
   where cross-layer error propagation dominates (QEP) [3]. Tessera
   T640 v1 lives in the valid regime, so alpha_l-weighting is
   justified; the QEP off-switch in `research-alignment-2026-07-30.md`
   Section 4.1 holds.

The recommended procedure, condensed:

- Perturb one layer at a time; keep all others at W*.
- Use the paper's Gaussian noise insertion (Eqn. 9), which gives
  E[||G_l - W_l||_F^2] = t^2 ||W_l||_F^2 exactly, so t_j^2 is the
  controlled abscissa.
- J = 15 noise levels uniformly sampled from the linear-theorem
  applicability range [1].
- Fit alpha_l = argmin Sum_j (Delta_{l,j} - alpha'_l t_j^2)^2, i.e. the
  closed-form through-origin slope Sum_j Delta_{l,j} t_j^2 / Sum_j t_j^4.
- Use KL on random tokens (data-free) for the harness; keep a
  WikiText-2 PPL path for validation only.
- Estimate once per model, cache in the sidecar/policy, reuse across
  bitwidths and quantization runs.
- Clamp alpha_l to a positive floor; fall back to uniform per-layer if
  the fit is noisy (low R^2) or the regime is sub-3-bit.

## 1. Definition, units, sign, magnitude, physical meaning

The Linearity Theorem (Theorem 1, Eqn. 4) [1][2] states that for any
possibly-randomized perturbation applied to the weights, as long as the
per-layer relative errors t_1, ..., t_L are small enough,

    E[PPL(W_hat)] ~= PPL(W*) + Sum_l alpha_l t_l^2          (Eqn. 4)

where the expectation is over the compression randomness and

    t_l^2 = E[||W_hat_l - W_l||_F^2] / ||W_l||_F^2           (Eqn. 3)

is the relative per-layer Frobenius reconstruction error. alpha_l is
the per-layer scaling coefficient; t_l is the per-layer error
coefficient. The paper is explicit that alpha_l is "universal": its
value depends only on the layer weights, not on the quantization
function [1]. In Tessera's fitness, t_l^2 is instantiated as the
kernel-direct relative Frobenius error
`||dequant_kernel(W_l) - W_l||_F^2 / ||W_l||_F^2` (the L1 sidecar, the
runtime ground truth), and the GA minimizes `Sum_l alpha_l t_l^2`
(`runtime-aware-pipeline.md` L6 alignment note).

**Units and sign.** t_l^2 is dimensionless (a normalized squared
norm). PPL is dimensionless. So alpha_l is dimensionless and is, to
first order, a PPL-per-unit-relative-MSE coefficient. Its sign is
non-negative in the valid regime: it is a local curvature of a loss
minimum. The theoretical reduction makes this precise. For a Gaussian
noise insertion the coefficient reduces approximately to

    alpha_l ~= (||W_l||_F^2 / 2) Tr(H_l)

where H_l is the Hessian of the loss with respect to layer l's weights
at W* [1] (more generally alpha_l = z_l M_{2,l} / 2, with z_l a
loss-curvature factor and M_{2,l} the second moment of the perturbation
distribution). At a local minimum the Hessian is positive
semi-definite, so Tr(H_l) >= 0 and alpha_l >= 0. A negative fitted
alpha_l is therefore not a physical result; it is a noise or
out-of-regime artifact and must be clamped (Section 4).

**Physical meaning.** alpha_l is the local PPL sensitivity (curvature)
of layer l. A large alpha_l means a unit of relative reconstruction
error in that layer buys a lot of perplexity; the GA should spend its
bit budget there. A small alpha_l means the layer is robust; the GA can
let its t_l^2 float. This is exactly the Hessian-trace sensitivity idea
from HAWQ-V2, made method-independent by the theorem [1][6].

**Magnitude and layer-dependence.** The paper does not publish a table
of raw alpha_l values, so we cannot quote absolute numbers faithfully;
treat any specific magnitude as inferred, not cited. What is
well-supported is the structure:

- alpha_l varies substantially across layers. The entire dynamic
  bitwidth application (Section 5 of [1]) only works because the
  alpha_l are non-uniform; uniform alpha collapses it to uniform
  bitwidth.
- The sensitivity ranking is stable and has a consistent shape across
  methods. SLQ's ILP allocation, driven by independently-estimated
  sensitivity coefficients, assigns the highest precision (8-bit) to
  K/V projections, 6-7 bit to Q and output projections, and the lowest
  (4-5 bit) to MLP layers [4]. BAQ's Hessian-proxy allocation reaches a
  similar non-uniform structure and an equal-loss principle [5]. This
  is the layer-dependence Tessera's alpha_l should reproduce: attention
  K/V and output projections sensitive, FFN/MLP robust.
- The ranking is reusable across budgets: GAMMA shows mixed-precision
  preferences encode a stable sensitivity ranking that transfers across
  bit budgets [7]. This is direct evidence for estimate-once-and-cache.

For Tessera the practical consequence: alpha_l is a per-tensor weight in
the fitness, expected to span a wide dynamic range, with attention
K/V and output projections at the high end and FFN tensors at the low
end. The regime descriptors already collected (tensor family,
kurtosis, effective rank) should correlate with alpha_l; that
correlation is itself a validation signal (Section 5.5).

## 2. The exact empirical procedure (Algorithm 3)

This is the paper-faithful estimator. Algorithm 3 [1][2]:

    ---------------------------------------------------------------
    ALGORITHM 3: Error coefficient calibration
    Input:  calibration constants t_1, ..., t_J;
            pretrained model W* = (W*_1, ..., W*_L)
    Output: linear coefficients alpha_1, ..., alpha_L

    for l = 1, ..., L do
        for j = 1, ..., J do
            Delta_{l,j} = PPL(W*(l, t_j)) - PPL(W*)
        end for
        alpha_l = argmin_{alpha'_l} Sum_{j=1..J} (Delta_{l,j}
                                              - alpha'_l t_j^2)^2
    end for
    ---------------------------------------------------------------

`W*(l, t_j)` is the model with all layers intact except layer l, which
is replaced by a Gaussian-noise-perturbed copy at noise level t_j [1].
The noise insertion (Eqn. 9, with the construction in Appendix B.2 and
Eqn. 12) is

    G_l(W_l, t) = W_l + t * ||W_l||_F / sqrt(d_in * d_out) * Sigma_l

where Sigma_l has i.i.d. N(0,1) entries. The normalization is chosen so
that E[||G_l - W_l||_F^2] = t^2 ||W_l||_F^2, i.e. the squared relative
perturbation norm is exactly t^2. That is why t_j^2 is the correct,
controlled abscissa for the fit, and why the same t_j^2 axis lines up
with Tessera's kernel-direct t_l^2.

Reading the loop:

- **Single-layer isolation.** Only layer l is perturbed per sweep; the
  other L-1 layers stay at W*. This is what makes alpha_l a partial
  derivative of PPL with respect to layer-l error, and it is the source
  of the additive model's independence assumption (Section 4, QEP).
- **Delta is relative to the clean model.** Delta_{l,j} subtracts
  PPL(W*), the unperturbed baseline. Fit the slope of delta-PPL, not
  absolute PPL.
- **The fit is through the origin.** There is no intercept term: at
  t_j = 0 the delta is 0 by construction. The closed-form solution is

      alpha_l = (Sum_j Delta_{l,j} t_j^2) / (Sum_j t_j^4)

  Do not add an intercept; an intercept absorbs baseline noise and
  biases the slope.

**Parameters the paper fixes.**

| Parameter | Value in [1] | Notes |
|---|---|---|
| J (noise levels per layer) | 15 | "enough to get accurate coefficients" [1] |
| t_j sampling | uniform from the linear-theorem applicability range | the range where Eqn. 4 holds; stay in the medium-bitwidth-equivalent error band |
| Data-dependent metric | PPL on 287k WikiText-2 train tokens | the calibration set used in the paper |
| Data-free metric | KL(pretrained || perturbed) on 287k random tokens | "performs nearly identically" to the PPL variant [1] |
| Perturbation | Gaussian, Eqn. 9 | gives E[||G_l - W_l||_F^2] = t^2 ||W_l||_F^2 exactly |

The t_j range is the one knob that needs engineering judgement. The
paper says "uniformly sampled from the linear-theorem applicability
range" and validates the error model in the medium-bitwidth regime
(roughly b > 3.0; Section 6.1 of [1]). Operationally: pick t_j^2 values
that bracket the relative errors Tessera's kernels actually produce
(the L1 sidecar gives this distribution directly), and stay below the
point where the delta-PPL-vs-t^2 curve visibly bends (the onset of
non-linearity). Fitting over a range that extends into the non-linear
region inflates alpha_l; fitting over a range far below the operating
point makes the estimate irrelevant to the actual quantization. Match
the calibration abscissa to the deployment operating point.

**Metric choice for Tessera.** Use the data-free KL variant as the
production probe. It removes the calibration-corpus dependency, matches
the paper's reported accuracy, and KL is independently shown to be a
tighter PPL proxy than MSE/SQNR [8]. Keep a WikiText-2 PPL path only as
a validation cross-check (Section 5.5). The KL form replaces
`PPL(W*(l,t_j)) - PPL(W*)` with the KL divergence between the clean and
perturbed model's next-token distributions, averaged over the random
token set; the through-origin slope fit is unchanged.

## 3. Cost, cheap approximations, caching and stability

**Cost model.** Algorithm 3 is O(L x J) loss evaluations: one forward
(or one KL evaluation) per (layer, noise-level) pair, plus the L
baseline evaluations of PPL(W*) / KL(W*). SLQ restates the cost as
O(T x M) forward passes, T = noise levels, M = groups [4]. For a ~12B
dense model with ~320 tensors (`runtime-aware-pipeline.md` L1.4) and
J = 15, that is ~4800 forward passes over the probe token set. On the
data-free KL probe the token set can be modest (the paper uses 287k
random tokens for its reported numbers; a smaller set is a legitimate
cost/accuracy tradeoff for a cached coefficient). This is an offline,
run-once cost, comparable in spirit to the existing ~5-hour per-pass
kernel-direct GA budget (`runtime-aware-pipeline.md` L6.3), and it
amortizes across every subsequent quantization run.

Two cost levers, in priority order:

1. **Cut J before cutting layers.** J = 15 is the paper's validated
   default, but the fit is a one-parameter through-origin regression;
   J = 5-8 well-placed noise levels give a stable slope if the t_j are
   chosen to bracket the operating point. Validate J-down by checking
   the fit R^2 and the stability of the alpha_l ranking under
   resampling.
2. **Cheap structural approximations for a first pass / cross-check.**
   The theoretical reduction alpha_l ~= (||W_l||_F^2 / 2) Tr(H_l) [1]
   gives a Hessian-trace proxy. Full Hessian is infeasible at 12B (the
   paper computed it only on OPT-125M [1]), but a Hutchinson-style
   stochastic trace estimate on the loss Hessian, or the cheaper
   HAWQ-V2 / BAQ Hessian-proxy surrogates [5][6], give a ranking-grade
   alpha_l at a fraction of the cost. Use this only to (a) sanity-check
   the perturbation-sweep ranking and (b) provide a warm start; do not
   ship it as the production coefficient, because the whole point of
   Algorithm 3 is that it measures the true loss curvature rather than
   a proxy.

**Caching and stability - the core economic argument.** alpha_l is
method-independent and, within the valid regime, bitwidth-independent
[1]. It depends only on W* and the loss surface. Therefore:

- **Estimate once per model, cache, reuse across quantization runs and
  across bitwidths.** This is explicit in the paper's dynamic-bitwidth
  application: the same alpha_l feed the LP for every target average
  bitwidth [1].
- **Stable across calibration sets:** the data-free KL variant on
  random tokens matches the data-dependent PPL variant [1], which is
  evidence that the coefficient is not calibration-set-sensitive. Treat
  this as strong-but-not-absolute; re-estimate if the model weights
  change (fine-tune, further training).
- **Stable across model sizes:** not directly established by the paper
  for alpha_l values themselves. What is established is that the
  sensitivity *ranking* is stable across budgets [7] and consistent in
  shape across methods [4][5]. Do not transfer alpha_l numerically
  across model sizes; transfer the ranking intuition only.
- **Cache key:** the coefficient set is keyed by the exact pretrained
  weight tensor identity (a content hash of W*). If the BF16 source
  changes, the cache is invalid. Store the fit diagnostics (R^2, the
  Delta_{l,j} samples, the t_j grid) alongside the coefficients so a
  cached alpha_l is auditable.

## 4. Numerical pitfalls and fallback rules

The additive linear model has known failure modes. Each maps to a
concrete guard in the harness.

**P1 - Sub-3-bit / cross-layer regime (the QEP off-switch).** The
theorem holds for sufficiently small relative per-layer errors; the
paper validates it in the medium-bitwidth regime (roughly b > 3.0) and
shows divergence below it [1]. QEP independently shows why: layer-wise
reconstruction error ignores cross-layer error propagation, which
accumulates near-exponentially at low bit widths, so a purely additive
layer-wise objective breaks down sub-3-bit [3]. QEP explicitly cites
HIGGS / Malinovskii et al. as part of the saturating layer-wise PTQ
baseline [3]. Guard: if any tensor's operating bitwidth is sub-3-bit
(a T640_3D extension, or the W4A4 activation boundary), disable
alpha_l-weighting and fall back to uniform, or switch to a
cross-layer-aware objective. This is exactly the off-switch ratified in
`research-alignment-2026-07-30.md` Section 7 call 5.

**P2 - Negative or near-zero alpha_l.** A true alpha_l is >= 0
(Section 1). A negative fit means noise dominated signal (low SNR at
small t_j), an outlier noise level sat in the non-linear region, or the
layer is genuinely flat. Guard: clamp alpha_l to a positive floor
alpha_min > 0 (e.g. a small fraction of the median positive alpha_l),
never let a negative weight enter the fitness - a negative weight would
reward the GA for *increasing* that layer's error. A near-zero alpha_l
is legitimate (a robust layer); leave it, but floor it so the GA does
not divide-by-zero or ignore the tensor entirely.

**P3 - Noisy fit / ill-conditioned layer.** If the Delta_{l,j} samples
do not lie on a line through the origin (low R^2), the alpha_l estimate
is unreliable. Causes: too few tokens in the probe, t_j range in the
non-linear region, or a layer whose loss surface is flat and
noise-dominated. Guard: compute the fit R^2 (or the residual of the
through-origin regression) per layer; below a threshold, fall back that
layer to uniform (alpha_l = alpha_uniform) and flag it. Do not let a
single noisy layer's coefficient drive the search.

**P4 - t_j range mismatch.** If the calibration t_j^2 grid does not
bracket the deployment operating point (the actual kernel-direct t_l^2
from L1), the fitted slope is extrapolated and can be badly wrong.
Guard: derive the t_j grid from the L1 sidecar's measured relative
error distribution, not from an arbitrary default. This is the single
most important practical detail and the one the paper leaves to the
implementer ("applicability range").

**P5 - Naming collision (process hazard, not numerical).** QEP also
uses a symbol alpha_l, but it is a propagation-strength regularizer in
[0,1], not the HIGGS PPL-sensitivity coefficient [3]. Keep the two
strictly separate in code and docs; do not import QEP's alpha_l into
the HIGGS fitness path.

**Fallback ladder (decisive).**

1. Full alpha_l-weighting when: regime is b > 3.0 AND fit R^2 is above
   threshold AND alpha_l > alpha_min.
2. Per-layer uniform fallback when: that layer's fit is noisy (P3) or
   its alpha_l is non-positive (P2). Set alpha_l = alpha_uniform for
   that layer only.
3. Global uniform fallback when: the regime is sub-3-bit (P1), or a
   majority of layers fail the fit, or the acceptance test (Section
   5.5) does not pass. Uniform is permanent and safe: the theorem still
   holds structurally with all alpha_l equal
   (`research-alignment-2026-07-30.md` Section 7 call 4).

## 5. Tessera integration design

### 5.1 Where alpha_l lives: sidecar / policy schema

alpha_l is a per-tensor coefficient estimated once per model. It belongs
in the calibration policy as an overlay that the existing
`--calibration-policy` flag can consume (`runtime-aware-pipeline.md`
L5.1), and/or alongside the L1 dequant sidecar. Proposed schema,
following the existing `llama.tessera.runtime-probe.v1` naming pattern:

    schema: llama.tessera.alpha-coefficients.v1
    {
      "model_hash": "<content hash of the BF16 source>",
      "probe": {
        "metric": "kl" | "ppl",
        "n_tokens": 287000,
        "data_free": true,
        "J": 15,
        "t2_grid": [ ... the J squared relative noise levels ... ]
      },
      "layers": [
        {
          "tensor": "blk.16.attn_v.weight",
          "alpha": 3.7e-3,
          "fit_r2": 0.984,
          "alpha_floor_applied": false,
          "fallback": "none" | "per_layer_uniform" | "global_uniform",
          "samples": [ {"t2": ..., "delta": ...}, ... ]   // audit trail
        },
        ...
      ],
      "regime_gate": {"min_operating_bits": 3.0, "qep_off_switch": true}
    }

The `samples` array stores the raw (t_j^2, Delta_{l,j}) pairs so a
cached coefficient is re-fittable and auditable without re-running the
forward passes. `model_hash` is the cache-invalidation key.

### 5.2 The estimation harness

A new offline tool, `tools/tessera/alpha_calibrate.py`, parallel to
`runtime_probe.py`:

1. Load the BF16 source model.
2. Compute the clean baseline KL(W*) (data-free) over the probe token
   set.
3. For each tensor l: for each of J noise levels t_j, build W*(l, t_j)
   by the Eqn. 9 Gaussian insertion (perturb layer l only), compute
   KL(W*(l, t_j)), set Delta_{l,j} = KL - KL(W*).
4. Fit alpha_l = (Sum_j Delta_{l,j} t_j^2) / (Sum_j t_j^4); record R^2
   and the samples.
5. Apply the fallback ladder (Section 4); write the schema in 5.1.

The harness is read-only with respect to Tessera source; it produces a
policy artifact. The GA in `per_tensor_calibrate.py` reads the artifact
and weights its `kernel-direct` fitness (`runtime-aware-pipeline.md`
L6) by alpha_l: fitness = Sum_l alpha_l t_l^2 with t_l^2 measured
against the L1 sidecar. The GA itself is unchanged (6D mutation space,
population, generations, islands); only the fitness aggregation gains
the per-layer weight, exactly as the L6 alignment note specifies.

### 5.3 When to trigger it

- **Trigger:** once per new BF16 source model, offline, before the
  per-tensor GA. It is a predecessor of G4-done, in the same offline
  phase as L1 sidecar generation.
- **Cache hit:** if `model_hash` matches a cached artifact and the
  regime gate passes, skip re-estimation entirely.
- **Re-estimate:** on model weight change (fine-tune), on probe-config
  change (J, metric, token set), or if a downstream acceptance test
  regresses.
- **Do not** re-estimate per bitwidth or per quantization run; that is
  the whole economic point (Section 3).

### 5.4 Relationship to the staged plan

Stage 1 (uniform) is alpha_l = constant for all l; the fitness reduces
to Sum_l t_l^2. Stage 2 (this spec) replaces the constant with the
estimated per-layer alpha_l. The two share the same fitness form and
the same harness plumbing; stage 2 is a strict refinement, and the
uniform artifact is just the degenerate coefficient set. This is the
staging ratified in `research-alignment-2026-07-30.md` Section 7 call
4, including the permanent uniform fallback.

### 5.5 Acceptance test (falsifiable)

The transition from uniform to alpha_l-weighting must earn its keep on
two independent axes, mirroring the G6 acceptance gate
(`research-alignment-2026-07-30.md` Section 5):

1. **Held-out kernel fidelity.** Hold out tensors not used to fit
   alpha_l. Run the GA with alpha_l-weighted kernel-direct fitness and
   with uniform fitness at the same bit budget. The alpha_l-weighted
   policy must produce a lower alpha_l-weighted Sum_l t_l^2 against the
   L1 sidecar than uniform. (Lower unweighted Sum_l t_l^2 is a bonus,
   not the criterion - the whole point is that not all t_l^2 are equal.)
2. **End-to-end PPL (the L4 probe).** The alpha_l-weighted model must
   beat the uniform model on the L4 end-to-end probe
   (`runtime-aware-pipeline.md` L4: Paris exact-match, top-1 match
   rate, perplexity delta). This is the test that matters; a win on the
   weighted objective that does not show up in PPL means the alpha_l are
   fitting noise.
3. **Validation cross-checks (diagnostics, not gates):**
   - The KL-probe alpha_l ranking correlates (Spearman / Kendall tau)
     with a WikiText-2-PPL-probe alpha_l ranking on a subset of layers.
     High correlation validates the data-free choice (and mirrors the
     KL-PPL rank-correlation methodology in [8]).
   - The alpha_l ranking correlates with the regime descriptors
     (tensor family, kurtosis, effective rank). In particular,
     attention K/V and output projections should rank high and MLP
     tensors low, matching SLQ's allocation [4] and the Hessian-trace
     literature [5][6]. If the ranking is uncorrelated with tensor
     family, suspect a harness bug before suspecting the theorem.
   - The Hessian-trace proxy ranking (cheap, Section 3) agrees with the
     perturbation-sweep ranking to ranking grade.

If (1) and (2) both pass, ship alpha_l-weighting. If (1) passes but (2)
does not, the coefficients are optimizing the proxy, not the model;
fall back to uniform and treat it as a research signal. If neither
passes, uniform stays permanently - and the theorem still holds
structurally, so nothing is lost.

## 6. Follow-on work that refines or challenges alpha_l

**QEP - the bound on where linearity holds [3].** "Quantization Error
Propagation: Revisiting Layer-Wise Post-Training Quantization" (Arai
and Ichikawa, Fujitsu) is the direct challenge. Its claim: layer-wise
PTQ objectives (HIGGS's additive Sum_l alpha_l t_l^2 included) ignore
cross-layer error propagation, which grows approximately exponentially
across layers and matters in the extremely low-bit regime (INT2/INT3),
especially for smaller models. Gains from modeling propagation are
concentrated sub-3-bit. QEP cites HIGGS / Malinovskii et al. 2024 as
part of the saturating layer-wise baseline. For Tessera this is the
off-switch, not a refutation: T640 v1 is in the medium-bitwidth regime
where the additive model holds, so alpha_l-weighting is valid; revisit
only for a sub-3-bit T640_3D extension or the W4A4 activation boundary.
Code: github.com/FujitsuResearch/qep. Process note: QEP's own alpha_l
is a propagation regularizer in [0,1], a different object from the
HIGGS coefficient (pitfall P5).

**SLQ - independent re-implementation and the cross-layer refinement
[4].** "Statistically-Lossless Quantization of Large Language Models"
(Helcig, Kurtic, Alistarh; IST-DASLab) is the strongest corroboration.
It explicitly adopts "linear estimation (Malinovskii et al., 2025)" -
the HIGGS paper - as its baseline sensitivity method: model degradation
as Delta_KL(b) ~= Sum_m alpha_m e_m^(b_m), with alpha_m estimated via
noise injection at O(T x M) forward passes [4]. That is an independent
statement and implementation of exactly the procedure in Section 2, and
it is the closest public code to a HIGGS Algorithm 3 (github.com/
IST-DASLab/SLQ). SLQ's contribution is the refinement: a multi-bitwidth
Shapley estimation that captures the cross-layer interactions the
linear/HIGGS model assumes away (the same physical effect QEP targets),
and an ILP allocation over the resulting sensitivity database. SLQ's
empirical allocation - 8-bit to K/V, 6-7 bit to Q and output
projections, 4-5 bit to MLP [4] - is the layer-dependence shape
Tessera's alpha_l should reproduce. SLQ also introduces EAR (Expected
Acceptance Rate) as a distribution-level fidelity metric, relevant to
Tessera's L3/L4 probes.

**BAQ - the Hessian-proxy / equal-loss view [5].** "BAQ: Efficient Bit
Allocation Quantization" derives a closed-form, Hessian-informed bit
allocation under a global bit budget and shows an equal-loss principle:
the optimal assignment equalizes each component's contribution to total
loss [5]. This is the allocation-side dual of alpha_l-weighting and a
cheap ranking-grade cross-check (Section 3): BAQ's Hessian proxy can
warm-start or sanity-check the perturbation-sweep alpha_l.

**GAMMA - stability of the ranking across budgets [7].** GAMMA shows
mixed-precision search preferences encode a stable sensitivity ranking
that transfers across bit budgets [7]. This is direct support for
estimate-once-and-cache (Section 3): the alpha_l *ranking*, which is
what drives the GA's budget allocation, is the stable object.

**KL-lens (CVPR 2026W) - the probe-metric justification [8].** "A KL
Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision
SSM-Transformer Hybrids" gives a formal and empirical argument that KL
divergence is a tighter PPL proxy than MSE/SQNR for language modeling,
using a forward-only, backprop-free per-layer sensitivity (quantize one
layer, measure KL to the teacher) with a Kendall-tau KL-PPL rank
correlation [8]. This independently validates both the data-free KL
probe choice (Section 2) and the acceptance-test cross-check that
correlates the KL-probe ranking with a PPL-probe ranking (Section 5.5).

**HAWQ-V2 - the predecessor [6].** HAWQ-V2 establishes the average
Hessian trace as a per-layer sensitivity metric, with a loss term of
the form Sum_i Tr(H_i) ||Q(W_i) - W_i||^2 [6]. This is the lineage the
HIGGS alpha_l ~= (||W_l||_F^2 / 2) Tr(H_l) reduction descends from, and
it is why alpha_l is interpretable as Hessian-trace sensitivity made
method-independent by the theorem.

**Public HIGGS code status.** Hugging Face Transformers ships HIGGS
support, but it exposes fixed-bit HIGGS grids via `HiggsConfig(bits=...)`
backed by the FLUTE runtime; it implements the grid quantization, not
the dynamic per-layer alpha_l calibration (Algorithm 3) [9]. No
standalone public repository implementing Algorithm 3 in isolation was
confirmed; SLQ's "Linear" sensitivity method [4] is the closest public
realization of the noise-injection alpha_l fit.

## 7. Open items and honest gaps

- **No published raw alpha_l table.** The paper validates the model and
  the dynamic-bitwidth application but does not tabulate alpha_l values
  [1]. Magnitudes in this spec are inferred from the theoretical form
  and the follow-on allocation literature, not cited. The harness must
  measure them, not hard-code them.
- **t_j "applicability range" is under-specified.** The paper leaves
  the noise-level range to the implementer [1]. Section 2 and pitfall
  P4 give the operational rule (bracket the L1-measured operating
  point, stay below the non-linearity onset), but the exact endpoints
  are an empirical choice the harness must make and record.
- **Cross-size stability is ranking-only.** alpha_l values should not
  be transferred across model sizes; only the ranking shape transfers
  [4][5][7]. Re-estimate per model.
- **Sub-3-bit is genuinely out of scope.** Both HIGGS's own validation
  [1] and QEP [3] say the additive model breaks there. Do not attempt
  alpha_l-weighting in that regime without a cross-layer objective.

## References

[1] Malinovskii, et al. "Pushing the Limits of Large Language Model
    Quantization via the Linearity Theorem" (HIGGS), arXiv:2411.17525.
    https://arxiv.org/abs/2411.17525

[2] Malinovskii, et al. "Pushing the Limits of Large Language Model
    Quantization via the Linearity Theorem", NAACL 2025 (proceedings
    PDF, contains Algorithm 3 and the Linearity Theorem).
    https://aclanthology.org/2025.naacl-long.543/

[3] Arai, Y. and Ichikawa, Y. "Quantization Error Propagation:
    Revisiting Layer-Wise Post-Training Quantization" (QEP),
    arXiv:2504.09629. Code: https://github.com/FujitsuResearch/qep
    https://arxiv.org/abs/2504.09629

[4] Helcig, M., Kurtic, E., Alistarh, D. "Statistically-Lossless
    Quantization of Large Language Models" (SLQ), arXiv:2605.02404.
    Code: https://github.com/IST-DASLab/SLQ
    https://arxiv.org/abs/2605.02404

[5] "BAQ: Efficient Bit Allocation Quantization for Large Language
    Models", arXiv:2506.05664.
    https://arxiv.org/abs/2506.05664

[6] "HAWQ-V2: Hessian Aware Quantization of Neural Networks with
    Mixed-Precision" (Hessian-trace sensitivity predecessor).
    https://arxiv.org/abs/1911.03852

[7] "GAMMA: mixed-precision search via stable sensitivity ranking
    across bit budgets", arXiv:2605.18475.
    https://arxiv.org/abs/2605.18475

[8] Kong, et al. "A KL Lens on Quantization: Fast, Forward-Only
    Sensitivity for Mixed-Precision SSM-Transformer Hybrids", CVPR
    2026 Workshop (EDGE).
    https://openaccess.thecvf.com/content/CVPR2026W/EDGE/papers/Kong_A_KL_Lens_on_Quantization_Fast_Forward-Only_Sensitivity_for_Mixed-Precision_CVPRW_2026_paper.pdf

[9] Hugging Face Transformers, "HIGGS" quantization documentation
    (fixed-bit HIGGS grids via HiggsConfig, FLUTE runtime).
    https://huggingface.co/docs/transformers/v4.48.0/en/quantization/higgs
