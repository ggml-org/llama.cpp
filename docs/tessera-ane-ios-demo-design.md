# Tessera: gemma 4 12B unified on iPhone via ANE+CPU

Status: design. No code changes. Outlines the path from the current
Tessera + ANE integration to a public-facing demo: a Tessera-quantized
gemma 4 12B + drafters running on an iPhone 13 Pro Max, on the Apple
Neural Engine and CPU, with thermal and battery envelopes that don't
kill the device.

This document captures a multi-turn design conversation (2026-08-04
session). Each phase has a definition of done, the open questions for
the architect, and the existing landed work it builds on.

## North star

A SwiftUI iOS app, on an iPhone 13 Pro Max, that:

- Loads a single Tessera-quantized unified GGUF containing the gemma 4
  12B trunk + DFlash + DSpark + MTP drafters (the singular-GGUF
  commitment from M0a/M0b, commit c64e9a85a).
- Quantized at T640_3D (2-bit weights + per-row meta) with HIGGS
  per-layer alphas, total ~3.5-4 GB on disk.
- Runs the full transformer forward pass with **all heavy ops on
  Apple Neural Engine** and the control-flow / sampling ops on CPU.
- Streams tokens at 5-10 tok/s with battery draw ~25-30% per hour.
- Survives 30+ minutes of continuous use before thermal throttling
  forces a graceful CPU-only fallback.

The "hell of a demo" the architect described: ANE-first, low-power,
battery-respectful, on the hardest target (iPhone), with the singular
unified GGUF the architect committed to.

## Why iPhone 13 Pro Max specifically

- **6 GB unified memory** (vs. 8 GB on 15/16 Pro). This is the binding
  constraint that forces T640_3D (sub-2-bit) and shapes the bundle
  split.
- **A15 ANE**: 16-core, 15.8 TOPS INT8, ~80-100 MB per-`.mlmodelc`
  weight cap. Multifunction ANE is supported (A15+).
- Same ANE architecture as the macOS M1 the prior work targets. iOS
  is the smaller-memory / harder-thermal cousin.

iPad M-series 16-32 GB is a kinder target (no memory ceiling) and the
right public-anchor demo before the iPhone port. See "iPad first"
under Decision Points below.

## Memory math (the binding constraint)

At T640_3D with HIGGS per-layer alpha-encoded per-row meta (~3 effective
bits/weight):

| Component | Approx size | Notes |
|---|---|---|
| Trunk 12B | ~3.0-3.3 GB | weights only |
| DFlash encoder | ~150-300 MB | EAGLE-style feature-conditioned |
| DSpark Markov head | ~200-400 MB | semi-AR |
| MTP next-n (n=4-8) | ~150-250 MB | future-token prediction |
| **Total unified GGUF** | **~3.5-4.0 GB** | |
| iOS system overhead on 6 GB | ~1.0-1.5 GB | system, springboard, drivers |
| **App headroom** | **~4.5-5.0 GB** | |
| Available for KV + IOSurface state + app heap | **~0.5-1.0 GB** | |

KV cache for a 12B model at full attention is ~30-50 MB per token. At
4-8k context the KV alone is 240-400 MB, leaving very little for the
app heap. **iPhone mode is short-context chat**, not long-document
summarization. The 13 Pro Max demo target is 1-2k context with the
full unified GGUF, or 4-8k context with the drafter heads offloaded.

## ANE constraints on A15

| Constraint | Value | Source / implication |
|---|---|---|
| Per-`.mlmodelc` weight cap | ~80-100 MB | reverse-engineering: maderix/ANE, A14-A17 |
| Multifunction support | yes (A15+) | one bundle, multiple functionName entries |
| State must be in IOSurface | yes | architect's W0-W7 already implements this |
| Compute units preference | `MLComputeUnitsCPUAndNeuralEngine` | runtime falls back to CPU per-op |
| Cross-function input sharing | no | per-function slot allocation; no share |

For a 3.5-4 GB trunk at the A15 cap, the bundle splits into
~36-40 multifunction `.mlmodelc` files, each holding the dispatch
table for its weight range. The architect's W0-W7 architecture
(IOSurface-pinned state, MTLSharedEvent handoff, E-core SPSC pump)
is **designed for exactly this**. The pattern is proven on the
gemma4 prefill bundle (`test-ane-pinned-slot-dispatch`); applying it
to the full 12B is engineering, not research.

## The HIGGS per-layer alpha story

The fitness form, per the architect's research-alignment doc
(`docs/research-alignment-2026-07-30.md`, Section 6.4):

```
L = Sum_l alpha_l * t_l^2
```

where `t_l^2` is the relative per-tensor Frobenius reconstruction
error for layer `l`, and `alpha_l` is the per-layer weight. The
architect's research-design ratified decision is: **START UNIFORM
(all alpha_l equal)**, then refine with HIGGS per-layer estimation
once L1 kernel-dequant fidelity is available.

**The L1 catch**: in production, `t_l^2` must be measured against the
L1 kernel-dequant output, not the offline ternary MSE proxy. The L1
path is the architect's "kernel-direct fidelity" differentiator. Until
L1 lands, the per-layer alpha is the offline-ternary-MSE estimate
(documented as the "until L1 lands" caveat in the alignment doc).

For the iPhone demo, this means:

1. The L1 kernel-dequant must run on ANE for HIGGS to be a real
   measurement and not a proxy. The T640_3D matmul on ANE consumes
   the packed weight format directly (no separate dequant step); this
   is the W0 spike's `TILE640_MATMUL` TODO at `ggml-ane.mm:1240`.
2. Until L1-on-ANE lands, the HIGGS alpha is the uniform fallback.
   The demo still works (T640_3D uniform alpha is a known-good
   2-bit quantization), it just doesn't yet exercise the research
   contribution.

## Path A vs Path B: the drafter question

The memory math closes only if the drafter heads fit. Two options:

| | **Path A: trunk only on iPhone** | **Path B: full unified on iPhone** |
|---|---|---|
| What ships on-device | Trunk T640_3D (~3.0-3.3 GB) | Trunk + 3 drafters (~3.5-4.0 GB) |
| KV / state headroom | 1.5-2.0 GB (4-8k context) | 0.5-1.0 GB (1-2k context) |
| Drafters | Disabled (single-model decode) | Loaded on demand from sandbox flash |
| Singular-GGUF story | Lost on iPhone (Mac is the unified target) | Honored on iPhone |
| Battery | ~25-30% per hour | ~30-40% per hour (drafter CPU adds ~5-10%) |
| Public story | "12B LLM on iPhone" | "12B + spec-decoding on iPhone" |

The architect's singular-GGUF commitment is the Mac calibration
target. Path A on iPhone demonstrates **deliberate role-aware model
partition** using the M0 infrastructure (`model_role` in
`tessera-quantize-db.cpp`, the 8-role schema in
`tessera_db.py`). Path B honors the singular story on iPhone but
squeezes the context budget.

**My recommendation: Path A for the public demo, Path B as the
follow-on**. The iPhone audience reads "12B + 3 drafters" as "this
won't fit" and you want the headline to be "12B on iPhone, ANE,
sub-2-bit, doesn't kill the battery." Path B becomes the second
chapter once the architect's singular-GGUF is shipped on the Mac
calibration side.

## Phase plan

Six phases. Each has a definition of done, a list of files touched,
and the open questions for the architect.

### Phase 0: L1 kernel-dequant on ANE (the load-bearing work)

**Why first**: until L1 is on ANE, "ANE-first inference" is a lie.
Every other phase assumes L1 works. If L1 hits a hard ANE constraint
(A15 bandwidth, weight-shape cap, integer-only precision), the
architecture pivots here and we know early.

**Definition of done**:
- T640_3D matmul on ANE consumes the packed weight format
  (`weight_packed`, `weight_page_scales`, `weight_lane_scales`,
  `weight_outlier_*`, `weight_act_scale`) directly. No host-side
  dequantization step.
- The matmul output matches the L0.5 (CPU reference) reconstruction
  within T640_3D's documented error bound.
- A 256x256 matmul at T640_3D passes a CPU-vs-ANE parity test
  (max abs error < 1e-2 FP16 equivalent).
- The `TILE640_MATMUL` op routing at `ggml-ane.mm:1240` is no longer
  a TODO.

**Files**:
- `ggml/src/ggml-ane/ggml-ane.mm` (TILE640_MATMUL dispatch case, ~200
  lines)
- `tools/ane-mtp/` (new bundle export for the T640_3D matmul
  function)
- `tests/test-ane-tile640-matmul.cpp` (new parity test)
- `common/tessera-debug/` if the L1 kernel needs any host-side
  helpers

**Estimated scope**: 2-3 weeks.

**Open questions**:
- Does the A15 ANE compiler accept the T640_3D weight format as a
  static input, or does the dequant have to be a separate ANE
  function with the matmul as the second function in the bundle?
  The maderix/ANE reverse-engineering work suggests the latter.
- Per-row meta + per-layer alpha together is ~1 effective bit per
  weight of state. Can this state live in IOSurface and be
  re-supplied per dispatch, or does it have to be baked into the
  bundle at compile time?

### Phase 1: transformer body ops on ANE

**Why**: the architect's W0 spike (`ggml-ane.mm`) covers matmul +
elementwise. A 12B forward pass on the current backend is
"matmul-on-ANE, everything-else-on-CPU" which is a CPU sandwich.
This phase lights up the rest of the transformer body so the L1
path is fully on ANE.

**Definition of done**:
- `GGML_OP_NORM` (RMSNorm) dispatched to ANE, with eps from
  `op_params[0]`.
- `GGML_OP_SOFT_MAX` dispatched to ANE.
- `GGML_OP_ROPE` dispatched to ANE, including the gemma 4 variant
  (rope scaling, mrope sections, freq factors).
- `GGML_OP_GLU` dispatched to ANE (split-then-mul for gated FFN).
- `GGML_OP_GET_ROWS` dispatched to ANE (embedding lookup).
- Each op has a CPU-vs-ANE parity test at representative shapes
  (e.g. RMSNorm at [1, 4096], SoftMax at [1, 1024], RoPE at
  [1, 4096], GLU at [1, 11008]).
- All five ops land in one multifunction `.mlmodelc` (the architect's
  W0 + W1 spike pattern, with all the new functionName entries in
  one bundle).

**Files**:
- `ggml/src/ggml-ane/ggml-ane.mm` (5 new dispatch cases, ~50-200 lines
  each)
- `tools/ane-mtp/make-transformer-body-bundle.py` (new, single
  multifunction bundle export)
- `tests/test-ane-transformer-body.cpp` (new parity suite)

**Estimated scope**: 2-3 weeks (5 ops × 0.5 weeks each, plus the
multifunction bundle export).

**Open questions**:
- Does the gemma 4 ROPE variant decompose into the standard MIL
  `gather` + `mul` + `cos`/`sin` + `add`, or does it need a custom
  coremltools op?
- GLU: gemma 4 uses `geglu` (GELU * gate) or `swiglu` (silu * gate)?
  The MIL op name differs.

### Phase 2: A15 bundle split

**Why**: ANE has a per-`.mlmodelc` weight cap (~80-100 MB on A15).
A 3.5-4 GB trunk at this cap splits into ~36-40 multifunction bundles.
The architect's W0-W7 architecture is designed for this; this phase
applies it to the full 12B.

**Definition of done**:
- A `tools/ane-mtp/split-bundle.py` that takes a Tessera-quantized
  unified GGUF, walks the weight tensor list, partitions into
  per-ANE-budget chunks, and emits a set of multifunction `.mlmodelc`
  files with the IOSurface state contract from W0.
- Each `.mlmodelc` carries the `dispatch_pinned_function` for the
  weight range + the relevant subset of body ops (from Phase 1) that
  are inlined into the same bundle.
- The `ane_state_layout.v1.json` manifest is the per-bundle
  contract; the iOS loader uses it to map pinned IOSurface slots.
- The `test-ane-state-layout` test passes on a 36-bundle split
  (or a test fixture simulating it).

**Files**:
- `tools/ane-mtp/split-bundle.py` (new)
- `common/ane-mtp.mm` (extend the load path to handle a list of
  bundles, not just one)
- `tests/test-ane-bundle-split.cpp` (new)

**Estimated scope**: 1-2 weeks.

**Open questions**:
- Are the body ops (RMSNorm, etc.) duplicated across bundles, or
  factored into a shared "common" bundle? Duplication is simpler;
  factoring saves ~200-500 KB of mlmodelc footprint.
- Bundle ordering: does the load path need explicit prefetch
  hints, or does iOS's `URL` cache handle the streaming?

### Phase 3: HIGGS per-layer alpha estimation

**Why**: the architect's research design ratifies this as the
refinement path. With L1-on-ANE (Phase 0) the alpha is a real
measurement; without L1, it's the offline-ternary-MSE proxy. The
iPhone demo is the killer validation: per-layer alpha is what makes
2-bit T640_3D actually usable for chat.

**Definition of done**:
- A `tools/ane-mtp/estimate_higgs_alpha.py` that:
  1. Runs the unified GGUF through the L1 path on the ANE
     (Phase 0's T640_3D matmul on the dequant kernel).
  2. Per layer, computes `t_l^2` as the L2 reconstruction error
     vs the L0.5 FP16 reference.
  3. Estimates `alpha_l` via the HIGGS Linearity Theorem (per the
     architect's `docs/research-higgs-alpha-2026-07-30.md`).
  4. Stamps `alpha_l` into the `ane_state_layout.v1.json` as
     per-layer metadata consumed by the L1 dispatch.
- The per-layer alpha is computed once per model (not per inference)
  and cached.
- A "uniform alpha" baseline (alpha_l = 1 for all l) is also
  supported; the iPhone demo defaults to the HIGGS estimate when
  available, falls back to uniform.

**Files**:
- `tools/ane-mtp/estimate_higgs_alpha.py` (new, ~300 lines)
- `tools/ane-mtp/test_estimate_higgs_alpha.py` (new)
- `docs/research-higgs-alpha-2026-07-30.md` (already exists; this
  phase implements the estimator it documents)

**Estimated scope**: 1-2 weeks (the math is research; the
implementation is engineering).

**Open questions**:
- Is the L1 measurement (Phase 0) per-chunk, per-layer, or per-model?
  The HIGGS estimator wants a stable per-layer measurement, so
  probably per-layer averaged over a representative chunk count.
- Does the alpha estimate need to be re-done if the model is
  re-quantized at a different granularity, or is the estimate
  stable across quant settings?

### Phase 4: iOS app

**Why**: macOS development is the algorithm work; iOS is the
deployment. The demo is the iOS app.

**Definition of done**:
- A fork of `examples/llama.swiftui/` that uses the tessera backend
  (ggml-cpu + ggml-ane, with the ANE backend on by default).
- Role-aware loader: trunk on the ANE path, drafters mmap'd from
  sandbox flash and loaded on demand (Path B) or disabled (Path A).
- `QOS_CLASS_BACKGROUND` on the E-core pump thread.
- `ProcessInfo.thermalState` observer: when state >= `.serious`, the
  app degrades to CPU-only decode at lower tok/s rather than letting
  iOS throttle the ANE under us.
- A SwiftUI chat surface that streams tokens as they generate, with
  a battery/thermal telemetry strip ("ANE: 2.1 W, CPU: 0.4 W, SoC:
  38°C, projected runtime: 47 min").
- App bundle ships the .mlmodelc files (multifunction split) and
  the unified GGUF, OR the GGUF is downloaded on first run (depends
  on App Store size policy; 4 GB exceeds the 200 MB cellular cap).

**Files**:
- `examples/llama.swiftui/` (fork)
- `examples/llama.swiftui/llama.swiftui/Sources/Backend/TesseraANE.swift`
  (new)
- `examples/llama.swiftui/llama.swiftui/Sources/Backend/RoleAwareLoader.swift`
  (new)
- `examples/llama.swiftui/llama.swiftui/Sources/Telemetry/PowerStrip.swift`
  (new)
- `Info.plist` updates for ANE permission + thermal usage
  description

**Estimated scope**: 3-4 weeks.

**Open questions**:
- Bundle download vs in-app shipping. 4 GB on cellular requires
  a download on first run; 4 GB in the .app is fine for TestFlight
  but problematic for App Store.
- ANE access in background mode: iOS restricts ANE in
  `UIApplication.backgroundTimeRemaining`. Is the demo
  background-resilient or foreground-only?

### Phase 5: battery / thermal characterization

**Why**: the demo is the data, not the talk. Battery %, thermal
throttle curve, projected runtime — these are the things that
make the audience say "oh, that's real."

**Definition of done**:
- A `tools/ios/power_audit.py` that reads the per-tok telemetry
  strip from the running app via a debug bridge (XCUITest harness
  or the LM's stderr log capture).
- A characterization report on a real iPhone 13 Pro Max with
  battery at 100% / 50% / 5%:
  - tok/s at each battery level
  - W drawn by ANE (from `powermetrics` or equivalent)
  - W drawn by CPU (similarly)
  - thermal state over 30 minutes of continuous use
  - projected runtime at 100% battery, room temp
- The characterization is reproducible: same prompt, same model
  hash, same battery preconditioning.

**Files**:
- `tools/ios/power_audit.py` (new)
- `docs/tessera-ane-ios-demo-results.md` (characterization report
  published as part of the demo)

**Estimated scope**: 1 week.

**Open questions**:
- Is the demo's battery telemetry accurate enough to publish, or
  do we need an external instrumented setup (USB power meter, FLIR)?
  Apple's `powermetrics` is reasonably accurate but not
  publication-grade.

## Decision points for the architect

These are the choices that change the whole shape. Each is a
binary or small-set decision; my recommendation is noted but
override is fine.

### 1. Path A (trunk only on iPhone) vs Path B (full unified)

**Recommend: Path A**. The iPhone audience reads "12B + 3 drafters"
as "won't fit" and you want the headline. Singular-GGUF is the Mac
calibration story; iPhone demonstrates the role-aware partition
the M0 work enabled. Path B becomes the second chapter.

### 2. iPad M-series first, then iPhone port?

**Recommend: iPad first**. The architect's iPad M-series 16-32 GB
removes the memory ceiling and lets the same iOS app showcase the
full unified GGUF + drafters without compromise. iPhone port is
the follow-on using Path A. iPad-first gets the public demo out in
~6-8 weeks; iPhone-first is ~10-12 weeks.

### 3. Phase 0 (L1 on ANE) gate

**Recommend: gate hard**. If L1 on ANE hits a hard A15 constraint
(weight-shape, bandwidth, precision), the architecture pivots and
we know at 2-3 weeks in. No silent fallback to "uniform alpha +
T640_3D" without telling the architect.

### 4. HIGGS alpha: implement now or after L1?

**Recommend: implement after L1**. The estimator is the same code
either way (per-layer t_l^2 measurement), but without L1 the
measurement is the offline ternary MSE proxy. Implementing the
estimator with the proxy and then "upgrading" to L1 later is the
right sequencing: the math + the cache + the manifest stamping are
all L1-agnostic; only the measurement source changes.

### 5. App distribution

**Recommend: TestFlight + download on first run**. 4 GB exceeds
the 200 MB cellular App Store cap; TestFlight has a 4 GB limit
that just barely fits. The on-device download is the right
production story anyway: the model changes faster than the app
release cycle.

## What this is NOT

- **Not a T640_3D matmul on a server**. The L1 work is ANE-specific
  (the IOSurface state contract, the multifunction bundle, the
  per-model weight cap). Server-side matmul is a different
  problem.
- **Not a long-context solution**. 1-2k context is the iPhone
  budget. 8k+ context needs the iPad or Mac.
- **Not a multi-model chat surface**. The singular unified GGUF
  is one model. Multi-model (router + N specialists) is a
  different design.
- **Not a research claim on HIGGS until L1 lands**. The architect's
  research design ratifies the per-layer alpha as a refinement of
  the uniform fallback; the measurement is the proof, and the
  measurement needs L1.

## Total scope and ordering

| Phase | Work | Estimated scope | Dependency |
|---|---|---|---|
| 0 | L1 kernel-dequant on ANE | 2-3 weeks | - |
| 1 | Transformer body ops | 2-3 weeks | Phase 0 (uses L1 path) |
| 2 | A15 bundle split | 1-2 weeks | Phase 1 (body ops land in bundle) |
| 3 | HIGGS per-layer alpha | 1-2 weeks | Phase 0 (measurement source) |
| 4 | iOS app | 3-4 weeks | Phases 0-2 (uses the bundles) |
| 5 | Battery / thermal characterization | 1 week | Phase 4 (needs the running app) |

**Total: 10-12 weeks for Path A on iPhone, 12-14 weeks for Path B
on iPhone.** iPad M-series first: 6-8 weeks for the iPad demo,
then iPhone port as a follow-on.

**Critical path**: Phase 0 (L1) is the long pole and the
architect's research differentiator. Every other phase assumes
L1-on-ANE works. If it doesn't, the architecture pivots early.

## What lands first, in priority

If only one phase can be done: **Phase 0 (L1 on ANE)**. It
validates the whole architecture; the rest is plumbing.

If two: **Phase 0 + Phase 1** (L1 + body ops). Together they
prove the L1 path is fully on ANE, which is the load-bearing claim.

If three: **+ Phase 4** (iOS app). The demo needs the app.

The character of each phase: Phase 0 is the architect's research
contribution, Phase 1 is engineering, Phase 2-4 are plumbing,
Phase 5 is data.

## Open questions for the architect (one round)

1. **Path A or Path B for the iPhone demo?** My recommendation is
   Path A (trunk only on iPhone, full unified on iPad). Override
   if the singular-GGUF story on iPhone is the higher priority.
2. **iPad M-series first?** My recommendation is yes (6-8 weeks
   instead of 10-12). Override if iPhone is the only acceptable
   public target.
3. **Implement HIGGS estimator now (with proxy) or after L1?** My
   recommendation is after L1. Override if you want the estimator
   code shape locked in before L1 is proven.
4. **App distribution?** My recommendation is TestFlight + on-
   device download. Override if you want the .app to ship the
   model in-bundle (only feasible for an internal demo).
5. **Phase 0 hard-gate criteria?** What does the architect need
   to see to commit to L1-on-ANE as the architecture? CPU-vs-ANE
   parity within 1e-2? A full block forward pass at the model's
   actual shapes? Something else?

The doc is the scope. Pick the decision points and we go.
