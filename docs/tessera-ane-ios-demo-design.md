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
- Runs the full transformer forward pass through **one singular
  stateless multifunction `.mlmodelc`** on the Apple Neural Engine,
  with the GGUF ternary weights streamed into IOSurface at runtime.
  **Accelerate** (vDSP / BLAS) handles the CPU-side ops and the
  fallbacks the ANE refuses.
- Streams tokens at 5-10 tok/s with battery draw ~25-30% per hour.
- Survives 30+ minutes of continuous use before thermal throttling
  forces a graceful CPU-only fallback.

**The "hell of a demo" the architect described: ANE-first, low-power,
battery-respectful, on the hardest target (iPhone), with the singular
unified GGUF the architect committed to.**

The architecture is **stateless from the ANE's perspective**: the
`.mlmodelc` is a kernel library (one file, all the functionName
entries for matmul / RMSNorm / SoftMax / RoPE / GLU / GetRows). The
GGUF holds the model weights in the T640_3D packed format; the host
streams each layer's weight tensor into IOSurface as the forward
pass advances, and the ANE program consumes it. The per-`.mlmodelc`
weight cap that bites the weight-baked pattern does not apply here
because nothing is baked.

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

**The ANE bundle itself is not part of this budget.** The `.mlmodelc`
is stateless — function definitions only, no baked weights — so it
sits in the few-MB range and lives in the app bundle. The 3.5-4 GB
weight budget is the GGUF file in the sandbox; the .mlmodelc adds
negligible overhead.

## ANE constraints on A15

| Constraint | Value | Source / implication |
|---|---|---|
| Per-`.mlmodelc` weight cap | ~80-100 MB | applies to **baked** weights, NOT to runtime IOSurface-fed weights |
| Multifunction support | yes (A15+) | one bundle, many functionName entries (matmul, RMSNorm, SoftMax, RoPE, GLU, GetRows, etc.) |
| State must be in IOSurface | yes | architect's W0-W7 already implements this |
| Compute units preference | `MLComputeUnitsCPUAndNeuralEngine` | runtime falls back per-op |
| Cross-function input sharing | no | per-function slot allocation; no share |
| Per-function weight binding | YES — each functionName has baked input shapes | the function is shape-specific; the weight format inside is fixed |

**The weight cap is not a binding constraint for the iPhone demo**
because the `.mlmodelc` is **stateless from the ANE's perspective**.
The bundle contains function definitions (matmul-of-shape-X, RMSNorm-
of-shape-Y, etc.) but no model weights. The weights come from the
GGUF at runtime, streamed into IOSurface. The `.mlmodelc` is therefore
~few MB (function tables + a small amount of internal state) regardless
of model size. The 3.5-4 GB weight budget lives entirely in the GGUF
file in the app sandbox, not in the ANE bundle.

The architect's W0-W7 architecture (IOSurface-pinned state, MTLSharedEvent
handoff, E-core SPSC pump) is exactly this pattern: the bundle is
stateless, the host writes activations + the current layer's weights
into pinned IOSurface slots, the pump dispatches. The pattern is
proven on the gemma4 prefill bundle (`test-ane-pinned-slot-dispatch`);
applying it to the full 12B is engineering, not research.

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
- The weight is supplied at runtime via IOSurface (no baked weight);
  Phase 2 wires the GGUF→IOSurface stream.

**Files**:
- `ggml/src/ggml-ane/ggml-ane.mm` (TILE640_MATMUL dispatch case, ~200
  lines)
- `tools/ane-mtp/` (the `.mlmodelc` for the T640_3D matmul function
  is one entry in the same multifunction bundle as Phase 1's body
  ops — the L0.5 reference is exported as a separate Python script
  for parity testing)
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
  bundle at compile time? (Same question applies for Phase 2's
  streaming layer.)

### Phase 0.5: EXL2 cross-check (research credibility layer)

**Why**: Tessera's HIGGS per-layer alpha is the architect's research
contribution. To claim it's well-founded, the per-layer sensitivity
ranking from HIGGS should agree with at least one independent
estimator. **EXL2** is the natural choice: it's the current
quality-per-bit leader on NVIDIA (open source, MIT, turboderp's
implementation of GPTQ-style calibration error with per-layer bit
allocation). The cross-check is "do the two per-layer sensitivity
estimators — Tessera's HIGGS and the EXL2-style algorithm — agree on
which layers are sensitive?"

**Important distinction: the algorithm vs the inference engine**.
- **ExLlamaV2** (turboderp's CUDA runtime that loads `.exl2` files)
  is NVIDIA-only and irrelevant to the cross-check. We are NOT
  loading `.exl2` files.
- **The EXL2 calibration algorithm** (quantize each layer at
  multiple bpw, measure error, choose the combination under a
  target average bpw) is pure math and hardware-agnostic. The
  GPTQ paper (Frantar et al. 2022, open access) documents the
  math; turboderp's README documents the per-layer allocation.
  We **reimplement** the algorithm in pure NumPy on Apple Silicon.
  No CUDA, no external hardware dependency.

This is strictly better than running ExLlamaV2 on a separate box:
both estimators run on the **same Mac, same corpus, same model**,
removing the hardware confound from the cross-check. The Spearman
comparison is a clean head-to-head between two algorithms with
different math and different proxies.

The two estimators measure different things:
- **HIGGS (Tessera)**: per-layer alpha via the Linearity Theorem,
  weighting the L1 kernel-dequant reconstruction error by the
  per-tensor Hessian. This is the kernel-direct fidelity measurement
  (proxy: offline ternary MSE until L1 lands).
- **EXL2-style (reimplemented)**: per-layer sensitivity via GPTQ-style
  calibration error, where the algorithm quantizes each layer at
  multiple bpw (2, 3, 4, 5, 6, 8) and measures the L2 reconstruction
  error. The bpw EXL2's allocator would choose for that layer is a
  side-effect of the search.

Different math, different proxy, same hardware — both estimate
the same underlying signal: **which transformer layers are the
most sensitive to quantization error?** If the two rankings agree,
that's evidence the design is shaped by SOTA. If they disagree on
specific layers, that's a research finding (the disagreement is a
paper, not a bug).

**This phase is orthogonal to the iPhone demo work** and runs
locally on the architect's Mac. No external hardware, no CUDA,
no cloud cost. Starts immediately in parallel with Phase 0/1/2/3.

**Definition of done**:
- A new `exl2_layer_stats` table in the unified DuckDB with columns:
  `model_hash`, `layer_index`, `exl2_per_layer_error` (the per-layer
  quantization error at the EXL2-chosen bpw), `exl2_per_layer_bpw`
  (the bpw the allocator chose for that layer), `exl2_calibration_corpus`
  (the corpus the EXL2 calibration was run against, for audit
  trail). One row per (model_hash, layer_index).
- A `tools/tessera/exl2_calibrate.py` that:
  1. **Reimplements** the GPTQ-style calibration in pure NumPy
     (column-wise quantization with error correction; calibration
     data + Hessian-weighted reconstruction error). The math is
     from Frantar et al. 2022.
  2. Reimplements the EXL2 per-layer bit allocation (search for
     the best bpw combination under a target average bpw; minimize
     max per-layer error). The algorithm is in turboderp's
     README.
  3. Runs both on the same calibration corpus Tessera uses
     (Wikitext-103 + COCO + LibriSpeech for the multimodal case).
  4. Captures the per-layer error + per-layer bpw, writes to
     `exl2_layer_stats`, and emits a sidecar JSON in the L5 retune
     shape.
- The L5 orchestrator reads `exl2_layer_stats` and folds the
  EXL2 per-layer error into the per-tensor sensitivity score as
  a third evidence signal (alongside HIGGS alpha and the
  imatrix stats). The fold is **independent** — the orchestrator
  does not bias toward either estimator; both are evidence, the
  disagreement is logged.
- A `tools/tessera/test_exl2_cross_check.py` that:
  1. Runs both estimators on a known model (gemma 4 12B at BF16
     is the gold standard).
  2. Computes the per-layer Spearman rank correlation between
     HIGGS alpha ranking and EXL2 error ranking.
  3. Asserts Spearman > 0.6 (high agreement, expected outcome on
     well-behaved models).
  4. Reports the top-5 disagreeing layers as a research finding
     (would be a paper, not a test failure).
- Documentation: a `docs/tessera-higgs-vs-exl2-sensitivity.md`
  report on the gemma 4 12B measurement, with the Spearman plot
  and the top-5 disagreements.

**Files**:
- `tools/tessera/tessera_db.py` (new `exl2_layer_stats` table +
  additive column on the per-tensor sensitivity path; no
  destructive schema change)
- `tools/tessera/exl2_calibrate.py` (new, ~400-500 lines: the
  reimplemented GPTQ + EXL2 allocation algorithm in NumPy)
- `tools/tessera/l5_orchestrator.py` (extend the per-tensor
  sensitivity score to consume EXL2 as a third evidence signal;
  log disagreement)
- `tools/tessera/test_exl2_cross_check.py` (new, ~200 lines)
- `docs/tessera-higgs-vs-exl2-sensitivity.md` (the report)

**Estimated scope**: 1.5-2 weeks (1 week coding + integration, 3-5
days running the calibration + writing the report, 1-2 days for
the cross-validation test). The GPTQ reimplementation is the bulk;
the EXL2 allocation is a small search algorithm on top.

**Open questions**:
- Is the per-tensor EXL2 error useful, or only the per-layer
  ranking? Per-tensor gives finer granularity for the L5 loop's
  per-tensor verdicts; per-layer is what the cross-check needs.
  Both fit in the same `exl2_layer_stats` table (one row per
  tensor, with the layer index as a column).
- The Spearman threshold (0.6) is a guess. The actual floor is
  determined empirically on a known model. The test should
  report the Spearman value and let the architect set the
  threshold after the first run.
- **Why the reimplementation, not ExLlamaV2's exact algorithm?**
  ExLlamaV2's per-layer allocator has implementation details
  not in the README (the exact search strategy, the per-tensor
  vs per-layer granularity, the outlier handling). The
  reimplementation captures the documented intent (search for
  best bpw combination under target average bpw, minimize max
  per-layer error) using the math the GPTQ paper makes explicit.
  If turboderp's specific search heuristic matters, the
  reimplementation can be tuned to match. The Spearman threshold
  will tell us.

**Research claim**:

> *"The HIGGS Linearity Theorem per-layer sensitivity ranking
> agrees with the EXL2-style GPTQ-based per-layer sensitivity
> ranking (reimplemented in NumPy, run on the same Apple Silicon
> hardware as the HIGGS estimator) at Spearman ρ > 0.6 on gemma
> 4 12B. The top-5 disagreements are in layers that the EXL2
> allocator over-allocates bits to (early attention QKV) and
> HIGGS under-weights (late FFN down projections); the
> disagreement is consistent with the kernel-direct vs.
> offline-proxy measurement difference."*

If true, this is a real paper. The two estimators are independent
(Tessera's HIGGS is kernel-direct, the EXL2-style is GPTQ Hessian
proxy), same hardware, same corpus, same model — the cross-check
is a clean validation of the design direction.

### Phase 1: transformer body ops on ANE

**Why**: the architect's W0 spike (`ggml-ane.mm`) covers matmul +
elementwise. A 12B forward pass on the current backend is
"matmul-on-ANE, everything-else-on-CPU" which is a CPU sandwich.
This phase lights up the rest of the transformer body so the L1
path is fully on ANE.

**The host-side split: ANE vs Accelerate**. Not every op goes to
ANE. The dispatch policy is:

| Op class | Primary backend | Fallback | Why |
|---|---|---|---|
| `MUL_MAT` (T640_3D) | ANE (L1 path, Phase 0) | n/a | the matmul is the whole point |
| `MUL_MAT` (BF16/fp16) | ANE (the W0 spike) | Accelerate BLAS | bake-shape constraint, fallback if shape mismatches |
| `NORM` (RMSNorm) | ANE (new, Phase 1) | Accelerate vDSP (if shape doesn't fit) | per-row reduction |
| `SOFT_MAX` | ANE (new, Phase 1) | Accelerate vDSP | row softmax |
| `ROPE` (gemma 4 variant) | ANE (new, Phase 1) | Accelerate vDSP | elementwise + gather |
| `GLU` (gated FFN) | ANE (new, Phase 1) | Accelerate vDSP | split + elementwise mul |
| `GET_ROWS` (embedding) | ANE (new, Phase 1) | memcpy (vocab is small enough) | simple gather |
| `ADD` / `MUL` / `SCALE` | Accelerate vDSP (always) | n/a | ANE dispatch overhead > vDSP cost for elementwise |
| `RESHAPE` / `VIEW` / `PERMUTE` | layout-only, free | n/a | no compute |
| `CPY` | memcpy | n/a | no compute |
| Sampling (argmax, top-k, etc.) | CPU | n/a | control flow, not compute |

The dispatcher in `ggml_ane_program_dispatch_op` checks ANE
eligibility (shape, dtype, dispatch cost vs. fall-through cost);
if ANE is the better fit, it runs the functionName; otherwise it
returns `false` and the ggml scheduler routes the op to the CPU
backend (which uses Accelerate via `ggml-cpu`). The hard rule:
**ANE is used when ANE is faster, not when ANE is available**.

**Definition of done**:
- `GGML_OP_NORM` (RMSNorm) dispatched to ANE, with eps from
  `op_params[0]`.
- `GGML_OP_SOFT_MAX` dispatched to ANE.
- `GGML_OP_ROPE` dispatched to ANE, including the gemma 4 variant
  (rope scaling, mrope sections, freq factors).
- `GGML_OP_GLU` dispatched to ANE (split-then-mul for gated FFN).
- `GGML_OP_GET_ROWS` dispatched to ANE (embedding lookup).
- The fallback policy above is implemented in
  `ggml_ane_program_dispatch_op`: per op, ANE first if eligible,
  else `return false` for the scheduler to route to CPU.
- Each ANE-eligible op has a CPU-vs-ANE parity test at
  representative shapes (e.g. RMSNorm at [1, 4096], SoftMax at
  [1, 1024], RoPE at [1, 4096], GLU at [1, 11008]).
- All five ops land in the one multifunction `.mlmodelc` (the
  architect's W0 + W1 spike pattern, with all the new functionName
  entries in the same bundle).

**Files**:
- `ggml/src/ggml-ane/ggml-ane.mm` (5 new dispatch cases, ~50-200
  lines each, plus the dispatch-policy table)
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
- Per-op ANE eligibility: how do we decide the cutoff "ANE faster
  than vDSP"? A 64-element add on ANE has ~1 ms dispatch
  overhead, vs. vDSP's <1 us. Below ~256 elements, ANE is never
  the right answer. We need a benchmark table.

### Phase 2: GGUF-to-IOSurface weight streaming

**Why**: the `.mlmodelc` is stateless but each functionName still
has a fixed input shape. The current layer's weight tensor has to
land in the right IOSurface slot before the functionName is
dispatched. This phase is the wire from "GGUF on disk" to "ANE sees
the weight as input."

**Definition of done**:
- A `common/ane-mtp/gguf_weight_stream.{h,mm}` that, given a
  `ggml_context *` over the unified GGUF and a layer index `L`,
  locates the `blk.L.*` weight tensors (and any per-layer alpha /
  page / lane / outlier / act_scale meta tensors) and copies
  them into the corresponding IOSurface-pinned slots.
- The E-core pump's dispatch path consumes the streamed weight
  and the activation as inputs to the bundle's `dispatch_pinned_function`
  for that layer.
- A "weight cache" that keeps the current layer's weights warm
  in IOSurface across consecutive dispatches (decode is M=1 per
  layer, so the same weights are reused N times before the layer
  index advances). Reduces ANE-side state thrash.
- The `test-ane-state-layout` test passes on a streamed 12B load
  (or a test fixture simulating it): the manifest drives the slot
  allocation, the streaming write lands the weight, the dispatch
  runs, the output lands in the destination slot, the next layer
  is streamed in.

**Files**:
- `common/ane-mtp/gguf_weight_stream.h` + `.mm` (new)
- `common/ane-mtp/ane-pump.mm` (extend the pump to use the
  streamer)
- `tests/test-ane-gguf-stream.cpp` (new)

**Estimated scope**: 1-2 weeks.

**Open questions**:
- Per-layer alpha + per-row meta: do these stream as part of the
  weight tensor's meta-region, or as separate IOSurface slots?
  Architecturally cleaner as a single meta-slot, byte-packed.
- Prefetch: the pump can stream the next layer's weight while
  the ANE is computing the current layer's output. The IOSurface
  write is memcpy-fast; the question is whether iOS's `URL`
  cache can keep the GGUF pages warm or if we need an explicit
  `posix_fadvise(POSIX_FADV_WILLNEED)` on the GGUF fd.

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

> See `docs/tessera-higgs-estimator.md` for the math, the
> L1-agnostic design, the sidecar JSON shape, and the
> measurement-function contract.

**Estimated scope**: 1-2 weeks (the math is research; the
implementation is engineering).

**Open questions**:
- Is the L1 measurement (Phase 0) per-chunk, per-layer, or per-model?
  The HIGGS estimator wants a stable per-layer measurement, so
  probably per-layer averaged over a representative chunk count.
- Does the alpha estimate need to be re-done if the model is
  re-quantized at a different granularity, or is the estimate
  stable across quant settings?

### Phase 3.5: HIGGS structural proxy -> C++ first-class

**Why**: Phase 3 just landed the structural proxy as a sidecar-time
NumPy script (`tools/ane-mtp/estimate_higgs_alpha.py`, 1002 lines,
52 tests). The proxy math is small, but the wiring is exactly the
shape the calibration + quantization pipeline has in C++ today:
read GGUF, dequant, compute, write sidecar JSON. Shipping it as a
sidecar NumPy script makes the "L1-agnostic by design" claim
partial: the L1 measurement source for the calibration / imatrix
path is C++; the proxy that consumes it is not. Phase 3.5 closes
that gap so the iOS dispatch's input is produced by the same
first-class C++ module that produces every other L5 input.

The proxy stays a **proxy**: `tessera-higgs.cpp` (HIGGS Algorithm 3,
the paper's gold standard) is the L1-aware fallback when the
model is large enough. The structural proxy is the fast path for
"iPhone demo on a 12B model" where the paper's J-level Gaussian
sweep would be infeasible. The C++ port shares the sidecar JSON
shape (`ane.alpha-coefficients.v1`) and the family prior with
the NumPy version byte-for-byte, so a sidecar produced by the
NumPy path is interchangeable with one produced by the C++ path.

**Definition of done**:
- A `tools/quantize/tessera/tessera-higgs-proxy.{h,cpp}` (new)
  that exposes the same API shape as the existing
  `tessera-higgs.cpp`:
  - `ts_higgs_proxy_params { J, t_min, t_max, alpha_floor, ... }`
  - `ts_higgs_proxy_layer_result { name, alpha_l, t_squared, n_elem, family }`
  - `ts_higgs_proxy_result { layers, n_valid, n_fallback_uniform, mean_alpha }`
  - `ts_higgs_proxy_estimate(gguf_path, params, result)` that reads
    the GGUF, dequantizes each tensor to F32, computes `t_l^2`
    against a ternary reference, and estimates `alpha_l` from the
    Frobenius norm + family prior + Hessian-trace surrogate.
  - `ts_higgs_proxy_to_json` / `ts_higgs_proxy_from_json` for the
    sidecar shape.
- A `tools/quantize/tessera/test_higgs_proxy.cpp` (new) that
  asserts:
  - The C++ output is **byte-equivalent** to the NumPy output
    on a fixed-seed fixture (round-trip via
    `ts_higgs_proxy_to_json` -> `ts_higgs_proxy_from_json`).
  - The family-prior rank order (K/V > attn_output > attn_q >
    norm > token_embd > ffn_down > ffn_gate/up) holds on a
    tinyllamas fixture.
  - Uniform fallback engages below
    `min_params_for_pert_estimate` (the same threshold the
    NumPy version uses).
  - The sidecar JSON validates against
    `docs/tessera-higgs-estimator.md` (schema version
    `ane.alpha-coefficients.v1`).
- A `tools/tessera/estimate_higgs_alpha.py` (rewritten, thin
  wrapper) that:
  - Subprocesses the C++ binary (`tessera-higgs-proxy --gguf ...
    --output ...`) and exposes the same CLI surface as today's
    NumPy script.
  - Falls back to the in-process NumPy implementation if the
    C++ binary is not on PATH (e.g. dev environment without a
    C++ build). The fallback is logged and tagged in the
    sidecar so the consumer knows which path produced the
    estimate.
  - The 52 existing tests in
    `tools/ane-mtp/test_estimate_higgs_alpha.py` are migrated
    to the new layout: the parity tests assert the C++ and
    NumPy paths produce the same sidecar; the C++-specific
    tests move to `tests/test-higgs-proxy.cpp`.

**Files**:
- `tools/quantize/tessera/tessera-higgs-proxy.h` (new, ~80 lines)
- `tools/quantize/tessera/tessera-higgs-proxy.cpp` (new,
  ~600-700 lines, mirrors the structure of `tessera-higgs.cpp`)
- `tools/quantize/tessera/CMakeLists.txt` (register the new
  source + a new `tessera-higgs-proxy` binary target)
- `tools/quantize/tessera/test_higgs_proxy.cpp` (new, ~30 tests)
- `tools/tessera/estimate_higgs_alpha.py` (rewritten as thin
  wrapper; the in-process NumPy path stays as the dev fallback)
- `tools/ane-mtp/test_estimate_higgs_alpha.py` (migrated to
  parity tests; the C++-only tests move out)
- `docs/tessera-higgs-estimator.md` (extend the "L1-agnostic
  design" section to document the C++ path and the parity
  invariant)

**L1-agnostic invariant (landed)**: `ts_higgs_proxy_estimate`
accepts a `ts_higgs_proxy_measurement_fn` callback (same shape as
`ts_higgs_metric_fn` in the existing `tessera-higgs.h`). The
default measurement is the L1-on-ANE kernel-dequant path
(`ts_higgs_proxy_measure_l1`, `t_squared_source =
"l1_kernel_dequant"`): each tensor is packed into the flat TILE640
row layout and dequantized with the same dispatch the
`GGML_OP_TILE640_MATMUL` inference path uses (v2 dequant at
`in_dim >= GGML_TESSERA_T640_V2_MIN_K` when v2 is enabled, the C
reference below the cutoff), round-tripped through fp16 (the ANE
bundle's pinned slot dtype), and compared element-wise:
`t_l^2 = mean |W - W_deq| / max |W|`. This captures the ternary
quantization error AND the ANE fp16 precision loss.
`TS_HIGGS_PROXY_LEGACY_OFFLINE=1` restores the legacy offline
ternary MSE proxy (`t_squared_source = "offline_ternary_mse"`); a
caller-supplied measurement function also keeps the legacy
behavior bit-identical.

**Estimated scope**: 1-2 weeks. The math is small; the work is
the GGUF reader plumbing, the CMake wiring, the parity tests,
and the Python wrapper migration. No external dependencies.

**Open questions**:
- Does the iOS dispatch need a C ABI for the proxy (callable
  from Swift), or does it only need the sidecar JSON? Today the
  Swift side reads the JSON; the C ABI is a future
  optimization, not a Phase 3.5 deliverable.
- Should the family prior live in C++ as a static table (same
  shape as the NumPy `FAMILY_PRIOR_*` constants), or read from a
  TOML sidecar alongside `ane.alpha-coefficients.v1`? C++ table
  is the path of least resistance and matches the calibration
  pipeline's prior encoding.

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

### 6. EXL2 cross-check (Phase 0.5): ship with the iPhone demo or follow-on?

**Recommend: ship with the demo, runs locally.** Phase 0.5 is a
reimplementation of the EXL2 algorithm (GPTQ calibration +
per-layer bit allocation) in pure NumPy. It runs on the same
Apple Silicon Mac the architect already has. No CUDA, no cloud
cost, no external hardware. The cross-validation result (Spearman
ρ between HIGGS and EXL2 per-layer sensitivity rankings) is the
research credibility layer that the demo's public story rests
on. Without it, the demo is "Tessera 2-bit on iPhone." With it,
the demo is "Tessera 2-bit on iPhone, and our per-layer
sensitivity estimator agrees with the SOTA per-layer allocation
algorithm at ρ > 0.6 on the same hardware." The second version
is a paper.

Override: skip if the iPhone demo needs to ship faster than the
research-credibility work can be done. The demo still works
without the cross-check; it just doesn't have the validation
behind the HIGGS claim.

## What this is NOT

- **Not a weight-baked bundle**. The `.mlmodelc` is stateless;
  weights come from the GGUF via IOSurface. The per-`.mlmodelc`
  weight cap does not apply because nothing is baked.
- **Not a T640_3D matmul on a server**. The L1 work is ANE-specific
  (the IOSurface state contract, the multifunction bundle, the
  E-core pump). Server-side matmul is a different problem.
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
| 0.5 | EXL2 cross-check (reimplemented, local) | 1.5-2 weeks | runs in parallel; no external hardware |
| 1 | Transformer body ops | 2-3 weeks | Phase 0 (uses L1 path) |
| 2 | GGUF-to-IOSurface weight streaming | 1-2 weeks | Phase 0+1 (multifunction bundle) |
| 3 | HIGGS per-layer alpha | 1-2 weeks | Phase 0 (measurement source) |
| 4 | iOS app | 3-4 weeks | Phases 0-2 (uses the running bundle) |
| 5 | Battery / thermal characterization | 1 week | Phase 4 (needs the running app) |

**Total: 12-14 weeks for Path A on iPhone, 14-16 weeks for Path B
on iPhone.** iPad M-series first: 6-8 weeks for the iPad demo,
then iPhone port as a follow-on. Phase 0.5 runs in parallel on
the same Apple Silicon Mac the architect already has; it does not
block the Apple-side work and there is no external hardware
dependency.

**Critical path**: Phase 0 (L1) is the long pole and the
architect's research differentiator. Every other phase assumes
L1-on-ANE works. If it doesn't, the architecture pivots early.
Phase 0.5 is independent and runs in parallel on the same Mac;
no external hardware.

## What lands first, in priority

If only one phase can be done: **Phase 0 (L1 on ANE)**. It
validates the whole architecture; the rest is plumbing.

If two: **Phase 0 + Phase 0.5** (L1 + EXL2 cross-check). Together
they validate the iPhone demo on the architect's research claim
and the independent cross-check. The result is a paper-grade
research claim: "HIGGS sensitivity agrees with EXL2 sensitivity
at Spearman ρ > X on gemma 4 12B."

If three: **+ Phase 1** (body ops). Together with the above, the
L1 path is fully on ANE and the research credibility is published.

If four: **+ Phase 4** (iOS app). The demo needs the app.

The character of each phase: Phase 0 is the architect's research
contribution, Phase 0.5 is the research credibility layer (paper-grade
cross-validation), Phase 1 is engineering, Phase 2-4 are plumbing,
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
6. **EXL2 cross-check (Phase 0.5) included in the demo or
   follow-on?** My recommendation is ship with the demo, in
   parallel on a separate CUDA box. The cost is 1.5-2 weeks of
   CUDA-box time; the value is the research-credibility layer
   that turns the demo into a paper. Override: skip if the
   demo needs to ship faster than the cross-check can be done.
7. ~~**NVIDIA box for EXL2 calibration**: do you have one in
   your environment, or does this need to run on a rented
   instance (vast.ai, RunPod, etc.)?~~ **Resolved: not needed.**
   Phase 0.5 reimplements the EXL2 algorithm locally on Apple
   Silicon (no CUDA, no ExLlamaV2, no external hardware). The
   earlier NVIDIA assumption was wrong; the calibration algorithm
   is hardware-agnostic and we don't need ExLlamaV2 to produce
   the per-layer error vector. The Spearman comparison is now
   head-to-head on the same Mac, removing the hardware confound.

The doc is the scope. Pick the decision points and we go.
