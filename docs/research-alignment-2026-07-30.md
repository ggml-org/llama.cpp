# Research Alignment: Runtime-Aware Proxy Objectives -> Tessera Roadmap

_Date: 2026-07-30. Source: deep-research run
`runtime-aware-quant-proxies` (final report
`final_turn_001.md`, 58 references). This document is the
source of truth for how that research reshapes the Tessera
implementation plans. Where this document and a plan doc disagree,
this document wins until the plan doc is updated._

## 0. Purpose

The research question was: how do published quantization proxy
objectives correlate with true execution-boundary fidelity, across
regimes, and can they be composed into an evolutionary-search-friendly
objective that scales across architectures? The answer is a design
input, not a literature survey. This document translates it into
concrete deltas against the six Tessera plan docs and re-derives a
single unified roadmap.

The plans it aligns:

| Doc | Phasing | Role |
|---|---|---|
| `runtime-aware-pipeline.md` | L1-L6 | the kernel-fidelity loop |
| `c++-port-design.md` | G0-G6 | the quantizer + GA in C++ |
| `multimodal-calibration-design.md` | G0-MM - G4-MM | modality extension |
| `w4a4-calibration-design.md` | (internal) | W4A4 activation path |
| `tessera-coreml-conversion-design.md` | (internal) | CoreML conversion + telemetry |
| `tessera-studio-design.md` | Phase 1-8 | the app shell |

## 1. Research conclusions that bind the roadmap

Seven results carry weight. The rest of the report is support.

1. **Linearity Theorem (HIGGS, arXiv:2411.17525).** In the
   medium-bitwidth, locally-smooth regime,
   `E[PPL(W_hat)] ~= PPL(W*) + Sum_l alpha_l * t_l^2`, where
   `t_l^2 = E[||W_hat_l - W_l||_F^2] / ||W_l||_F^2` is the relative
   per-layer Frobenius reconstruction error and `alpha_l` is a
   layer-specific, method-independent coefficient. This gives the
   composite objective a principled, falsifiable form: minimize
   `Sum_l alpha_l * t_l^2`.

2. **QEP (arXiv:2504.09629) is the off-switch.** Layer-wise
   reconstruction error breaks down at low bit widths (roughly sub-3-bit)
   because it ignores cross-layer error propagation, which accumulates
   near-exponentially there. Consequence: do NOT add cross-layer error
   propagation to the fitness for TESSERA_T640 v1. It is only justified
   if we push a sub-3-bit regime (a T640_3D extension, or W4A4 at the
   activation boundary).

3. **Regime axes select the method, not just the difficulty.**
   Kurtosis / heavy tails, outlier localization (DuQuant: `down_proj`),
   effective rank / spectral compactness, tensor family (attention vs
   FFN), and architecture paradigm (MoE expert FFNs are more robust than
   dense; routers and some attention projections need more precision)
   are all empirically predictive of which transform wins. Consequence:
   `tensor_families` and the imatrix regime statistics must become
   operative routing signals, not passive metadata.

4. **Rotation is the most GA-mature transform family.** Continuous
   Cayley / Stiefel knobs, smooth data-free proxies (GSR, OptRot), and a
   demonstrated structured-multimodal landscape (SpinQuant: random
   rotations vary accuracy by up to ~16 points, so the landscape is real,
   not noise). Permutation is NOT GA-friendly in its published form
   (DuQuant / PermuQuant are heuristic / closed-form); to make it
   evolvable it must be relaxed (Gumbel-Sinkhorn, differentiable sorting,
   ShuffleSoftSort). Low-rank is a regime-sensitive discrete knob (rank)
   plus cheap continuous factors, gated by spectral compactness.

5. **All search-based quantization prior art searches discrete
   bit-widths, not continuous reconstruction knobs.** HAQ, RAMP,
   FracBits, Q-Palette, QuantEA, EvoPress: every one assigns bit widths
   (EvoPress comes closest, evolving over a precomputed GPTQ bit-width
   database). Nobody runs an evolutionary search over continuous
   per-tensor reconstruction knobs (AWQ alpha, rotation angles, smoothing
   s, low-rank factors). This is the granularity boundary, and it is the
   white space.

6. **Nobody closes the loop on actual kernel-dequant fidelity.** QAT
   uses idealized fake-quant during training and converts to real kernels
   afterward; HAQ / OHQ close hardware-efficiency loops, not
   numerical-kernel-fidelity loops. A practitioner report of ~23%
   accuracy variance for the same INT8 model across five Snapdragon
   chipsets exists, but only as a QA regression gate, never as
   calibration fitness. Scoring candidate policies against the real
   TESSERA_T640 dequant kernel output is unoccupied.

7. **Quality-diversity is the natural GA architecture for a
   regime-conditioned objective.** MAP-Elites (and multi-objective MOME)
   keep the best configuration per region of a descriptor space. The
   descriptor space should be the regime axes. The archive then IS the
   regime map: one champion reconstruction-knob config per regime cell.

**The novelty is the composition, not any component.** Evolutionary
search over continuous reconstruction knobs, scored against actual
kernel-dequant fidelity, conditioned on regime descriptors. Each piece
is prior art; this exact composition is not.

## 2. The structural finding: there is no unified spine

The six plan docs phase independently. The kernel-fidelity loop
(`runtime-aware-pipeline.md` L1-L6) is the capability the research
identifies as both the differentiator and the ground-truth evaluator,
yet it sits marked "all Not started" (~5.5 person-weeks) with no
sequencing relationship to the C++ port's GA phases (G0-G6, 7-10 days),
whose GA fitness is offline (`ternary layer-output MSE`).

This is the primary misalignment. The research says the kernel hook (L1)
and kernel-direct fitness (L6) are the crown jewel and should drive
ordering. Today they are a parallel, disconnected workstream.

**Reframe.** The kernel-fidelity loop is the spine. Every other plan is
sequenced relative to it:

```
L1 kernel dequant hook (ground truth)
   |
   +--> G0-G3 quantizer + writer in C++ (produces TESSERA_T640 artifacts)
   |
   +--> G4 GA in C++, fitness = alpha-weighted t_l^2,
   |       evaluated against L1 kernel output (== L6),
   |       archive = MAP-Elites indexed by regime descriptors
   |
   +--> G6 acceptance: regime-routed kernel-fidelity composite
   |       beats best single proxy on held-out tensors vs L1
   |
   +--> L2-L5 differential / per-token / e2e / adaptive requantize
   |       (the verification + feedback layers, parallelizable)
   |
   +--> G0-MM - G4-MM (modality as a regime axis)
   +--> W4A4 (a lower-bit regime; QEP off-switch may need revisiting)
   +--> CoreML conversion + IOReport telemetry (runtime-boundary receipts)
   +--> Studio Phase 1-8 (the user-facing surface for L4/L6 + the archive)
```

The practical consequence: **L1 should land early, before G4 is
considered done.** A GA whose fitness is offline ternary MSE is exactly
the thing the research says is insufficient. G4's acceptance criterion
must be restated in terms of kernel-direct fitness, which requires L1.

## 3. Per-doc delta map

Verdicts: VALIDATED (research confirms the plan, no change needed),
REFINED (plan is right but its form / acceptance sharpens), ADDED (new
work the research introduces), DEFERRED (research says do not do yet).

| Doc | Verdict | What changes |
|---|---|---|
| `runtime-aware-pipeline.md` | VALIDATED + REFINED | L6 fitness form becomes `Sum_l alpha_l * t_l^2` with kernel-direct `t_l^2`; add QEP off-switch note; L5/L6 gain regime conditioning. L1 promoted to critical path. |
| `c++-port-design.md` | REFINED + ADDED | G2 reframed as "6 regime experts + a router"; G4 MAP-Elites descriptor = regime axes, fitness form stated; G6 acceptance becomes the falsifiable composite-beats-single-proxy test; CHAMP-Q permutation + FLRQ gating notes; sequencing cross-ref to L1/L6. |
| `multimodal-calibration-design.md` | VALIDATED + REFINED | Modality confirmed as a first-class regime axis; modality weights (0.5/0.3/0.2) are the alpha-weighting idea extended to modality; modality enters the MAP-Elites descriptor. |
| `w4a4-calibration-design.md` | VALIDATED | W4A16/W4A4 weighted fitness (0.5/0.5) and per-semantic-family breakdown already instantiate the composite + regime routing. Flag: W4A4 is the regime where the QEP off-switch is most likely to need revisiting. |
| `tessera-coreml-conversion-design.md` | VALIDATED (light) | The ~23% cross-chipset variance finding validates the IOReport telemetry + receipts thesis and the Metal auto-fallback (C7). `modality_scales` translation (2.5) carries the regime axis into the CoreML artifact. |
| `tessera-studio-design.md` | VALIDATED (light) | The `evaluate` tool + fitness chart + A/B compare are the surface for L4/L6. The fitness chart should expose the alpha-weighted composite and the regime-indexed archive, not a single fitness line. |

## 4. Concrete spec deltas

### 4.1 `runtime-aware-pipeline.md`

**L6 fitness form.** The current L6 mode computes
`relative_frobenius(dequant_kernel, BF16_source)` per tensor and returns
it as fitness. Restate this as the ground-truth instantiation of the
Linearity-Theorem term:

- Per tensor: `t_l^2 = ||dequant_kernel(W_l) - W_l||_F^2 / ||W_l||_F^2`,
  where `dequant_kernel(W_l)` is the L1 sidecar (what the kernel actually
  dequantizes), not the offline `_ternary_reconstruct`.
- Cross-tensor aggregation (the GA objective):
  `Sum_l alpha_l * t_l^2`. The `alpha_l` are the method-independent
  layer coefficients; estimate them once per model by the HIGGS
  calibration (perturb each layer, measure PPL response) and cache them
  in the sidecar / policy.
- The existing `direct` / `importance` / `combined` modes remain as
  cheap proxies; `kernel-direct` is the production fitness and the
  ground truth the proxies are validated against.

**QEP off-switch (new note).** Do not add cross-layer error propagation
to the L6 fitness for TESSERA_T640 v1. The Linearity Theorem holds in
this regime; QEP shows the correction only pays off sub-3-bit. Revisit
only for a T640_3D sub-3-bit extension or the W4A4 activation boundary.

**Regime conditioning (L5/L6).** The L5 requantize planner and the L6
fitness report should carry the regime descriptors per tensor (kurtosis,
effective rank, tensor family, modality) so the adaptive loop can route
experts, not just re-run one GA uniformly.

**Sequencing.** L1 is the critical path for G4, not a parallel
workstream. State this dependency explicitly.

### 4.2 `c++-port-design.md`

**G2 reframed: 6 regime experts + a router.** Today G2 ports
LRQ / SEPTQ / CHAMP-Q / DartQuant / FLRQ / PE-QAT as selectable modes.
The research says these are regime experts. Add a regime router that
picks the expert per tensor from the regime descriptors:

| Regime signal | Routes to |
|---|---|
| high kurtosis / massive outliers (esp. `down_proj`) | rotation (DartQuant) + permutation; NOT affine-only |
| high spectral compactness (low effective rank) | low-rank residual (FLRQ / LRQ) |
| attention Q/K projections | higher-precision / Hessian-mask expert (SEPTQ) |
| MoE expert FFNs | lighter expert (robust regime); router/gating higher precision |
| default / well-conditioned | AWQ diagonal scaling |

The router is cheap (it reads statistics already in the imatrix v2 /
`tensor_families`) and is the operative use of `tensor_families` that the
research demands. G2 LoC grows by the router (~150-250 LoC); the expert
ports are unchanged.

**G4: MAP-Elites descriptor = regime axes.** `ts_awq_archive_cell` is
currently a generic "3-axis bin index." Restate the 3 axes as the regime
descriptors: (kurtosis bucket, effective-rank bucket, tensor-family /
modality bucket). The archive then stores the best reconstruction-knob
config per regime cell, which is exactly the quality-diversity
architecture the research recommends. This is a small, concrete change
to an already-planned function.

**G4 / G6 fitness form.** State the GA objective as
`Sum_l alpha_l * t_l^2`, with `t_l^2` evaluated against the L1 kernel
output once L1 lands (i.e. G4's production fitness == L6). Until L1
lands, the offline ternary MSE is the stand-in proxy, and G6 must measure
how far the proxy ranking diverges from the kernel-fidelity ranking.

**G6 acceptance sharpened (falsifiable).** Replace "produces plausible
output on smoke-test prompts" with the research's novelty-boundary test:

- On held-out tensors, the regime-routed kernel-fidelity composite
  (`Sum_l alpha_l * t_l^2` against L1) must beat the best single proxy
  (best of AWQ-only, rotation-only, low-rank-only) at the same bit
  budget.
- Separately, measure the ranking disagreement between the offline
  ternary-MSE proxy and the kernel-direct fitness. If the proxy already
  ranks identically, the kernel-fidelity novelty collapses to "routing
  only," and we say so honestly.

**CHAMP-Q permutation note.** CHAMP-Q's permutation
(`ts_champq_permutation`) is closed-form. If permutation is to enter the
GA search space, it must be continuously relaxed (Gumbel-Sinkhorn /
differentiable sorting / ShuffleSoftSort) to be evolvable; otherwise it
stays a closed-form expert. This is unoccupied white space per the
research (learnable permutation for quantization channel reordering is
explicitly listed as future work in PermLLM).

**FLRQ / LRQ gating note.** Gate the low-rank experts by a cheap
spectral-compactness descriptor (effective rank / singular-value
entropy; LieQ reports Spearman rho > 0.8 with sensitivity). Only invoke
them when compactness is high; otherwise the router skips them.

**Sequencing cross-ref.** Add an explicit dependency: G4 production
fitness requires `runtime-aware-pipeline.md` L1. G6 requires L1 + L6.

### 4.3 `multimodal-calibration-design.md`

Modality is confirmed as a first-class regime axis (the research's
regime structure includes modality, and the modality-weighted fitness
0.5/0.3/0.2 is the alpha-weighting idea extended across modality). Two
refinements:

- The MAP-Elites descriptor space (4.2 above) includes the modality
  bucket, so the archive keeps per-modality champions where they differ.
- The G2-MM GA fitness weighting is the composite objective for the
  multimodal case; state it as an instance of `Sum alpha * t^2` with
  modality-specific alpha.

The locked decisions M1-M8 hold. M2 (missing modality -> ERROR by
default) is consistent with the research's "regime must be observed, not
silently imputed" stance.

### 4.4 `w4a4-calibration-design.md`

Validated as-is. The W4A16/W4A4 weighted fitness (0.5/0.5, configurable)
and the per-semantic-family breakdown already instantiate the composite
objective and regime routing. One flag to add: W4A4 is the lower-bit
regime where QEP cross-layer effects are most likely to matter, so it is
the first place the QEP off-switch (4.1) should be re-examined if W4A4
fitness plateaus or diverges from end-to-end PPL.

### 4.5 `tessera-coreml-conversion-design.md`

Light validation. The ~23% cross-chipset INT8 variance finding is direct
external evidence for the IOReport runtime-telemetry + receipts thesis
and for the Metal auto-fallback (C7): the runtime boundary is where
fidelity actually lives, and it varies per hardware. The
`modality_scales` translation (Section 2.5) carries the regime axis into
the CoreML artifact, so the converted model preserves the routing
information. No structural change.

### 4.6 `tessera-studio-design.md`

Light validation. The `evaluate` tool, the fitness-over-generations
chart, and the A/B compare view are the user-facing surface for L4 (e2e
probe) and L6 (kernel fitness). Two refinements:

- The fitness chart should plot the alpha-weighted composite and, where
  it exists, the regime-indexed archive occupancy, not a single fitness
  line.
- The A/B compare view is the natural place to surface the G6
  composite-beats-single-proxy result as a receipt.

## 5. The novelty-boundary acceptance gate

This is the single falsifiable test that decides whether the project's
core claim holds. It belongs in G6 and should be wired into the Studio
A/B compare view as a receipt.

**Claim.** A regime-conditioned, continuously-parameterized evolutionary
search, scored against actual TESSERA_T640 kernel-dequant fidelity,
produces a better Pareto frontier (PPL vs bit budget) than any single
published proxy objective, across architectures.

**Test.**

1. Hold out a set of tensors not seen during calibration.
2. Run the regime-routed composite (alpha-weighted `t_l^2` against L1
   kernel output).
3. Run each single proxy at the same bit budget: AWQ-only,
   rotation-only (DartQuant), low-rank-only (FLRQ), Hessian-mask-only
   (SEPTQ), and the offline ternary-MSE proxy.
4. The composite must beat the best single proxy on held-out
   kernel-fidelity `t_l^2` AND on end-to-end PPL (L4 probe).
5. Record the ranking disagreement between the offline proxy and the
   kernel-direct fitness. Report it honestly; if it is near zero, the
   kernel-fidelity contribution is null and the novelty reduces to
   routing.

**Prior art to cite, not claim.** SEPTQ (arXiv:2604.10091), FLRQ
(arXiv:2601.05684), PE-QAT (ACL 2026.acl-srw.63), DartQuant, AWQ,
SmoothQuant, GPTQ, EvoPress, HAQ, OHQ are composed or distinguished, not
novel. CHAMP-Q is internal (no external LLM-quantization method by that
name exists; the only external hit is a sonic-logging device).

## 6. Unified sequencing

Collapsing the six independent phasings into one order, keyed on the
kernel-fidelity spine. Weeks are single-engineer estimates.

| Stage | Work | Source | Depends on | Est. |
|---|---|---|---|---|
| S0 | GGUF type registration | G0 | - | 1d |
| S1 | L1 kernel dequant hook (ground truth) | runtime-aware L1 | S0 | 1w |
| S2 | quantize_2d + AWQ + TESSERA writer | G1 + G3 | S0 | 3-4d |
| S3 | regime experts + router | G2 (reframed) | S2 | 4-5d |
| S4 | GA in C++: MAP-Elites(regime axes) + alpha-weighted fitness | G4 (refined) | S2, S1 | 3-4d |
| S5 | kernel-direct fitness wired into GA (== L6) | runtime-aware L6 | S4, S1 | 1.5w |
| S6 | L2-L5 differential / per-token / e2e / adaptive requantize | runtime-aware L2-L5 | S1, S2 | 2w (parallel) |
| S7 | G6 acceptance: composite-beats-single-proxy + ranking disagreement | G6 (sharpened) | S5, S6 | 3-4d |
| S8 | modality as regime axis | G0-MM - G4-MM | S4, S5 | per MM plan |
| S9 | W4A4 activation path (QEP re-exam candidate) | w4a4 | S5 | per w4a4 plan |
| S10 | CoreML conversion + IOReport telemetry | coreml | S2 | per coreml plan |
| S11 | Studio surface for L4/L6 + archive + A/B receipt | Studio Phase 1-8 | S5, S7 | per Studio plan |

The key reordering versus today: **S1 (L1 kernel hook) moves ahead of
S4/S5 (the GA), and S5 (kernel-direct fitness) is a hard prerequisite
for S7 (G6 acceptance).** The offline ternary-MSE fitness is a
stand-in used only until S5 lands.

## 7. Design calls (RATIFIED 2026-07-30)

All six calls below were ratified by the architect on 2026-07-30 (yes
to all). They do not reverse any locked decision (M1-M8, C1-C10,
c++-port 1-14 all hold); they refine and connect. Call 4 carries a
staging decision: ship uniform weights first (all alpha_l equal), then
add HIGGS per-layer alpha_l estimation as a follow-on refinement
(research in flight -> docs/research-higgs-alpha-2026-07-30.md), with a
permanent uniform fallback.

1. **Promote L1 ahead of the GA.** L1 (kernel hook) becomes a
   prerequisite for G4-done and G6, not a parallel workstream. This
   reorders the C++ port around the runtime-aware pipeline. Recommend:
   yes; it is the research's central point.

2. **G2 = experts + router, not modes.** Add the regime router to G2
   (~150-250 LoC). `tensor_families` + imatrix regime stats become
   operative. Recommend: yes; low cost, high leverage, and it is the
   operative use of metadata we already collect.

3. **MAP-Elites descriptor = regime axes.** Repoint the existing
   3-axis archive cell to (kurtosis, effective rank, tensor-family /
   modality). Recommend: yes; nearly free, turns the archive into the
   regime map.

4. **Adopt `Sum alpha_l * t_l^2` as the stated fitness form, with
   kernel-direct `t_l^2` as production.** RATIFIED with staging: ship
   uniform weights first (all alpha_l equal); add HIGGS per-layer
   alpha_l estimation as a follow-on refinement (research agent
   dispatched 2026-07-30; output -> docs/research-higgs-alpha-2026-07-30.md).
   The uniform fallback is permanent: if alpha_l estimation proves
   noisy, uniform stays (the theorem still holds structurally).

5. **QEP off-switch: no cross-layer fitness in v1.** Recommend: yes;
   revisit only for sub-3-bit T640_3D or the W4A4 boundary.

6. **CHAMP-Q permutation: keep closed-form in v1; relax
   (Sinkhorn/SoftSort) only if it enters the GA search space.**
   Recommend: defer the relaxation; it is white space but not on the
   critical path.

## 8. What the research validates unchanged

- The per-tensor calibration + receipts / auditability thesis. The
  ~23% cross-chipset variance finding is external evidence for it.
- The L1.5 FP16-reference-at-quantize-time decision (not in the dequant
  kernel). Consistent with "kernel output is ground truth; references
  are quantize-time inputs."
- The modality-weighted fitness (M1, 0.5/0.3/0.2) and per-modality AWQ
  (M8). These are the composite objective extended to modality.
- The W4A4 weighted fitness + per-semantic-family breakdown. Already an
  instance of composite + regime routing.
- The CoreML stock-ops-v1 / custom-op-v2 gating (C1) and Metal
  auto-fallback (C7). Runtime-boundary concerns the research reinforces.
- The island GA + MAP-Elites + progressive evaluation already in G4.
  The research refines the descriptor and the fitness, not the
  architecture.

## 9. Open risk the research adds

**Proxy-ranking divergence is the real bet.** If the offline ternary-MSE
proxy ranks candidate policies identically to the kernel-direct fitness,
then the expensive kernel-fidelity loop buys nothing beyond what the
cheap proxy gives, and the novelty reduces to regime routing. The G6
ranking-disagreement measurement (Section 5, step 5) is the early
detector. If divergence is near zero on the first model, re-scope toward
routing and away from the kernel-fidelity loop. This is the single most
important thing to measure early.
