# W4A4 Calibration Mode for Tessera

Design doc for adding a proper 4-bit weights / 4-bit activations (W4A4)
calibration mode to the Tessera quantizer, on top of the existing weight-only
Tile640 path. The doc is design-only: no code, only interfaces, validation
strategy, and open questions. It targets an implementation agent who is
already familiar with the Tessera L1-L6 runtime-aware calibration pipeline
described in `docs/pipeline-design.md` and `docs/runtime-aware-pipeline.md`.

> Roadmap alignment: the runtime-aware proxy-objective research
> (2026-07-30) validates this design as-is. The W4A16/W4A4 weighted
> fitness (0.5/0.5) and the per-semantic-family breakdown already
> instantiate the alpha-weighted composite objective and regime routing.
> One flag: W4A4 is the lower-bit regime where the QEP off-switch is most
> likely to need revisiting if fitness plateaus or diverges from
> end-to-end PPL. See
> [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md)
> Section 4.4.

## 1. Goals and non-goals

### What W4A4 means for Tessera

Tessera is currently weight-only at the format layer. The Tile640 matmul
kernel (see `tools/tile640/quantize_v3.py:119-122` for the layout
constants and `ggml/src/ggml-cpu/arch/arm/quants.c` for the runtime)
dequantizes the per-tile ternary+outlier weights but reads BF16
activations directly. W4A4 adds a calibrated INT4 path for activations
to the same Tile640 GGUF, with a per-tensor activation scale, a
per-channel outlier decomposition (LLM.int8-style), and an optional
SmoothQuant pre-pass that migrates the activation difficulty onto the
weights offline.

Concretely, for a 2D matmul of the form `Y = W * X`, the W4A4 path
quantizes `X` at the matmul input (per-token dynamic by default) and
dequantizes it inside the kernel before the FMA, producing `X_hat` in
the same BF16 / F32 accumulator as the weight-only path. The W4A4
runtime must produce `Y` that is bit-equivalent to the BF16 reference
modulo the L4 E2E tolerance documented in section 9.

### Goals

1. Add a calibrated W4A4 path that produces a Tile640 GGUF carrying
   both the existing weight components and the new activation-scale
   sidecar fields, without changing the weight-only contract for
   existing consumers.
2. Preserve the AWQ-evolve GA loop end-to-end: every candidate policy
   is evaluated under both W4A16 and W4A4 mode, and the fitness is a
   weighted sum (default 0.5/0.5) of the two.
3. Run the LLM.int8-style per-channel mixed-precision decomposition at
   runtime: per-channel INT4 for 99.9% of input channels, FP16
   residuals for the 0.1% outlier channels. The outlier threshold is
   `|X| > 6.0` per the LLM.int8 paper (arXiv:2208.07339), see the
   research report section 2b.
4. Add a per-tensor activation scale that is computed offline from the
   calibration corpus, written as a sidecar field, and folded into the
   dequant at matmul time.
5. Add a SmoothQuant pre-pass (alpha-tunable) that is integrated into
   the per-tensor GA, with default alpha=0.5 and auto-fallback to
   alpha=0.75 for tensors whose outlier fraction exceeds 10%.
6. Keep the change additive: a Tessera GGUF produced without `--w4a4`
   is byte-equivalent to the current weight-only output (modulo the
   new `tessera.w4a4.*` metadata fields, which are present but unused
   in the W4A16 runtime path).

### Non-goals

- A separate `--quant-mode` flag that switches between W4A4 and
  weight-only. The user explicitly chose to add a `--w4a4` boolean
  flag and keep the weight-only path authoritative; see section 5.
- A separate format layer or quantization mode at the GGUF level. The
  W4A4 activation-scale sidecar lives alongside the existing
  `tessera.*` metadata fields; the Tile640 weight components are
  unchanged.
- QuIP/QuIP#-style Hadamard rotation, OliVe/OVP-style outlier-victim
  pairing, HAWQ-V3 ILP bit-allocation. The research report section 4a
  marks these as "skip for now" for Tessera. They are not part of W4A4.
- Reaching W4A4 quality parity with Q4_K_M at the format layer in the
  first pass. The first pass targets "BF16 reference within the L4 E2E
  tolerance, with a 2x activation-side compute overhead" as the
  acceptance bar.
- Changing the L1-L6 pipeline. L1-L6 already measures the dequant
  fidelity; the W4A4 path plugs into the existing pipeline.

### Success criteria

- `tile640_quantize_v3.py --w4a4 --w4a4-scale-mode per_token ...`
  produces a Tile640 GGUF with `tessera.w4a4.enabled = true` and the
  new sidecar fields populated, and the existing weight components
  byte-equivalent to a non-`--w4a4` build of the same model.
- `awq-evolve.py --w4a4-eval` produces a calibration policy with
  per-family W4A16 and W4A4 fitness columns, and the A/B per-tensor
  static decision per semantic family is reported as a yes/no
  alongside the policy.
- The new dequant kernel (per section 7) produces a sidecar dump that
  matches the BF16 reference within the L4 E2E tolerance
  (1e-3 relative Frobenius for the INT4 path, 1e-4 for the FP16
  outlier path).
- A gemma 4 12B QAT target quantized with W4A4 reaches < 1.5 PPL
  delta vs the BF16 reference on the standard probe set, compared to
  < 0.5 PPL delta for the weight-only path. A regression vs the
  weight-only baseline is expected; the goal is bounded PPL growth
  with a calibrated activation path.

## 2. Activation scale policy

### Per-token dynamic (default, ship-first)

Tessera's first W4A4 implementation uses per-token dynamic activation
scaling. The per-token dynamic scale is the maximum absolute value of
the activation row at matmul input time, divided by the INT4
quantization range (2^(N-1) - 1 = 7 for N=4). In prose:

- scale_t = max over channels c of |X[t, c]|, divided by 7.
- X_quant[t, c] = round(X[t, c] / scale_t), clamped to [-8, 7].
- X_dequant[t, c] = X_quant[t, c] * scale_t.

The scale `scale_t` is recomputed per token inside the kernel, per
matmul invocation. It is NOT stored in the sidecar; only the
LLM.int8 outlier decomposition (section 3) and the per-tensor
activation scale (when in per-tensor static mode, see below) are
sidecar-resident.

The per-token dynamic scale is the conservative choice: it produces
the smallest per-token quantization error because every token uses
its own range. The cost is a per-token `max + divide` at matmul
input, which is cheap on Apple Silicon (a single NEON `maxv` + `fdiv`
per row) and acceptable on CUDA/Metal.

### A/B for per-tensor static per semantic family

Per the user's first architectural decision, after the per-token
dynamic ship we run an A/B test to check whether per-tensor static
extrapolation is good enough per semantic family. The families are
the six attention + ffn projections the user named:

- `attn_q` (input to the Q projection)
- `attn_k` (input to the K projection)
- `attn_v` (input to the V projection)
- `ffn_up` (input to the ffn_up projection)
- `ffn_gate` (input to the ffn_gate projection, the SwiGLU/GeLU gate)
- `ffn_down` (input to the ffn_down projection)

In the GA, the per-tensor static scale for family F is a candidate
gene `act_scale_static_F` in the per-tensor policy. The
`infer_family` classifier in `tools/tessera/awq-evolve.py:133` is
extended to map each of the six family fragments
(`attn_q`, `attn_k`, `attn_v`, `ffn_up`, `ffn_gate`, `ffn_down`) to
a per-family sub-class so the GA can track per-family deltas.

The A/B protocol inside `awq-evolve.py`:

1. For each candidate, run `evaluate()` twice on the calibration
   corpus: once with the per-token dynamic scale (the default), and
   once with the per-tensor static scale folded into the activation
   quantizer. The static scale is the GA-tuned per-family
   `act_scale_static_F`, applied as a pre-matmul multiplication on
   the activation.
2. Compute `delta_ppl = ppl_per_tensor_static - ppl_per_token_dynamic`
   for each family F across the calibration corpus.
3. Apply the acceptance rule: family F is "extrapolable" to
   per-tensor static iff `delta_ppl < 0.05` on a 30-prompt
   evaluation set drawn from the calibration corpus.
4. The result is recorded in the calibration policy as
   `act_scale_mode: {per_token, per_tensor}` per family. The
   default is `per_token` for all families; a family promoted to
   `per_tensor` is shipped with the static scale and a flag that
   tells the runtime to use the per-tensor scale (skipping the
   per-token `max + divide`).

The acceptance threshold of 0.05 PPL is a starting point and is
exposed as `--w4a4-per-tensor-static-threshold 0.05` on the GA CLI.

### Schema (sidecar)

The sidecar field for the per-tensor activation scale is:

| Field | Type | Shape | When populated |
|---|---|---|---|
| `tessera.w4a4.enabled` | bool | scalar | always when `--w4a4` |
| `tessera.w4a4.activation_bits` | uint32 | scalar | always when `--w4a4` |
| `tessera.w4a4.scale_mode` | string | scalar | always when `--w4a4`; one of `per_token`, `per_tensor` |
| `tessera.w4a4.act_scale_static` | float32[] | per-tensor, one per output tensor | only when `scale_mode = per_tensor` for that tensor |
| `tessera.w4a4.act_outlier_count` | uint32[] | per-tensor, one per output tensor | always when `--w4a4` (LLM.int8 section 3) |
| `tessera.w4a4.act_outlier_indices` | uint32[] | packed, per-tensor | always when `--w4a4` |
| `tessera.w4a4.act_outlier_vals` | float16[] | per-tensor, one entry per outlier index | always when `--w4a4` |

The fields are added to the GGUF writer block in
`tools/tile640/quantize_v3.py:393-475` alongside the existing
`tessera.*` fields. The per-tensor entries are written as a single
flat array per field type (one entry per quantized output tensor,
in the same order as the weight components), which matches the
existing convention for `weight_outlier_cols` and similar.

### L1 sidecar format change

The L1 sidecar format change for the per-tensor activation scale is
part of the format evolution order Tier-0 track (E1: per-tile
outlier count). It is a *new* field; no existing field semantics
change. The L1 sidecar `llama.tessera.per-tensor-calibration.v1`
schema is append-only per `docs/architecture.md` Calibration schema
invariants, so the new field bumps the schema to v2 in the GGUF
metadata without breaking existing consumers.

## 3. Outlier activations: LLM.int8-style mixed-precision decomposition

### The finding

Per the research report section 2b, LLM.int8() (Dettmers et al.,
NeurIPS 2022, arXiv:2208.07339) found that in LLMs above 6.7B
parameters, "150,000 outliers occur per sequence, but they are
concentrated in only 6 feature dimensions across the entire
transformer" -- the ~0.1% / 7 channels finding. The key empirical
result is that quantizing those channels uniformly with the rest
produces catastrophic accuracy loss; isolating them at higher
precision recovers the loss. Tessera's W4A4 path applies the same
isolation at runtime, on the activation side, per the user's
second architectural decision.

### Threshold rule

A value `X[t, c]` at matmul input is an outlier iff
`|X[t, c]| > 6.0` (the LLM.int8 default threshold, exposed as
`--w4a4-outlier-threshold 6.0` in `tile640_quantize_v3.py`). The
threshold is applied per-tensor at calibration time using the
calibration activations already produced by `--imatrix`; the
resulting outlier channel list is stored as `act_outlier_indices`
in the sidecar. The dequant kernel reads the list at load time.

The outlier fraction is capped at 0.1% of the input channels
(controlled by `--w4a4-outlier-fraction 0.001`). If the threshold
rule produces more outliers than the cap for a given tensor, the
top 0.1% by `|X[t, c]|` magnitude are kept; the rest are quantized
to INT4. This matches the LLM.int8 finding and bounds the FP16
storage overhead at ~0.4% of the activation footprint (0.1% of
channels at 4 bytes/element vs 0.5 byte/element for INT4 is the
ratio; for a 4 KB activation tile this is ~16 bytes overhead,
which is dwarfed by the per-tile scale + page metadata).

### Per-channel FP16 storage format

For each tensor with W4A4, the sidecar contains:

- The INT4 quantized data for the non-outlier channels (99.9% of
  channels), stored at the per-tile granularity matching the
  existing Tile640 weight format. No new tile layout is introduced
  for the INT4 path; the existing Tile640 page + lane + page-scale
  + lane-scale structure is reused for the activation data.
- The outlier channel indices, stored as a packed `uint32` array
  (`act_outlier_indices`).
- The F16 outlier values, stored as a flat `float16` array
  (`act_outlier_vals`), one entry per outlier index. Values are in
  BF16 reference order so the kernel can read them sequentially.

The outlier indices are sorted ascending so the kernel can binary
search. For the LLM.int8 finding (0.1% / 7 channels), the index
array is tiny (7 entries for a 4096-channel tensor); a linear scan
is faster than binary for the expected size, and the kernel uses
linear scan by default with a threshold-based switch to binary at
>32 outliers per tensor.

### Dequant kernel branch (runtime mixed-precision)

The dequant kernel for the W4A4 activation has three branches, per
the LLM.int8 paper section 3:

1. INT4 path: for each non-outlier channel c, the INT4 quantized
   value is dequantized as `X_hat[t, c] = X_quant[t, c] * scale_t`
   (per-token dynamic) or `X_hat[t, c] = X_quant[t, c] * scale_F`
   (per-tensor static), where `scale_t` is the per-token scale and
   `scale_F` is the per-tensor scale.
2. FP16 outlier path: for each outlier channel c, the F16 value is
   loaded directly. No scale is applied because the F16 value is
   stored at BF16 reference precision.
3. Accumulate: `Y[r, t] += sum_c(W[r, c] * X_hat[t, c])` in F32.

The kernel checks the outlier set once per channel index, not per
element. A channel's outlier flag is invariant across tokens (the
outlier set is computed at calibration time and stored in the
sidecar), so the branch is a single bit lookup per channel index
per matmul invocation. This is consistent with the LLM.int8
finding that outliers are "concentrated in only 6 feature
dimensions across the entire transformer": the per-channel flag
is small, and the branch is essentially free.

### Reference

Dettmers, Lewis, Belkada, Zettlemoyer. "LLM.int8(): 8-bit Matrix
Multiplication for Transformers at Scale." NeurIPS 2022.
arXiv:2208.07339. See the research report section 2b for the
summary and section 4 for the L3-layer mapping.

## 4. SmoothQuant pre-quantization migration

### The alpha decision

Per the SmoothQuant paper (Xiao et al., ICML 2023, arXiv:2211.10438,
see research report section 2b), the migration strength alpha
controls how much of the activation difficulty is folded onto the
weights via the per-channel scale. In prose:

- s_j = max(|X_j|)^alpha divided by max(|W_j|)^(1 - alpha), per
  input channel j.
- W'_j = W_j * s_j (folded weight, offline).
- X'_j = X_j / s_j (folded activation, applied at quant time).

After the fold, both `W'` and `X'` are quantized with INT4 (and the
LLM.int8 outlier decomposition on `X'`). At runtime, the dequant
recovers `W` via `W = W' / s_j`, which is a per-channel
multiplication absorbed into the existing AWQ `input_scale` tensor
(no new runtime op).

The user requested the following alpha policy:

- Default alpha = 0.5, per the OPT and BLOOM precedent in the
  SmoothQuant paper (these models have moderate outlier fractions
  and alpha=0.5 produces a balanced fold).
- Auto-fallback to alpha = 0.75 for tensors with high outlier
  fraction. "High" is defined as `outlier_count > 0.10 *
  total_channels` (i.e. more than 10% of channels are outliers
  under the LLM.int8 threshold). This is the GLM-130B precedent
  from the SmoothQuant paper: GLM-130B has ~30% outliers, and
  alpha=0.75 migrates more of the difficulty onto the weights to
  compensate. The 10% threshold is conservative; the GLM-130B
  regime starts at ~30%, but Tessera's smaller models (gemma 4
  12B, deepseek 4 small) tend to fall in the 5-15% range, so a
  10% trigger is a reasonable midpoint.

### Per-tensor alpha tuning in the GA

The alpha is a per-tensor gene in the AWQ-evolve GA. The mutation
space is extended with:

| Gene | Domain | Default | Notes |
|---|---|---|---|
| `smoothquant_alpha` | `{0.5, 0.75, auto}` | `auto` | `auto` selects per-tensor based on the outlier fraction rule above; `0.5` and `0.75` are forced values |

The `auto` mode is the production default. In the GA, the alpha is
a discrete gene (not a continuous value) because the SmoothQuant
paper's two canonical values are 0.5 and 0.75; intermediate values
do not have empirical precedent and the GA population is small
enough that the discrete search is fast.

The SmoothQuant fold is applied offline during the per-tensor
calibration, not at GA fitness evaluation time. The fitness
function sees the folded `W'` and `X'` and computes its existing
reconstruction error on the post-fold tensors; this keeps the
fitness function unchanged. The fold's effect on the weight
distribution is observable in the existing per-tensor metrics
(e.g. kurtosis, tail excess), so the GA naturally prefers folds
that produce better-behaved weight distributions.

### Folding the per-tensor activation scale into the weight

The SmoothQuant fold produces a per-channel weight scale `s_j`.
This is structurally identical to the existing AWQ
`weight_act_scale` tensor (see `tools/tile640/quantize_v3.py:3218`,
where `weight_act_scale` is written when `q["input_scale"]` is not
all 1.0). The W4A4 path reuses the existing AWQ path:

1. The SmoothQuant fold is applied first, producing the
   per-channel scale `s_j`.
2. The AWQ `--awq-alpha` (which is the migration strength in the
   AWQ sense, distinct from the SmoothQuant alpha) is applied
   second. The combined fold is the product of the two scales.
3. The combined scale is stored in `weight_act_scale`, which the
   runtime dequant applies as a single per-channel multiplication
   on the dequantized weight. No new runtime op; no new sidecar
   field for the SmoothQuant scale specifically.

The activation side of the fold (the `X' = X / s_j` division) is
absorbed into the activation quantizer: the per-tensor activation
scale is `scale = max(|X'_t|) / 7` (per-token dynamic) or
`scale = max(|X'_t|) / 7` per-tensor (per-tensor static), with
the `X'` values used in place of `X`.

## 5. W4A4 as an extension of the existing weight-only path

### Cite

Per the user's third architectural decision, the W4A4 mode is an
extension of the existing weight-only path. The CLI does NOT get a
separate `--quant-mode` flag; it gets a new `--w4a4` boolean flag
that adds the activation-side quantizer to the existing flow. The
weight-only path is authoritative and unchanged; `--w4a4` is a
layer that adds new sidecar fields and a new runtime path on top.

### CLI flags

The new flags are added to `tools/tile640/quantize_v3.py` in the
existing `argparse` block (after the `--septq-iterations` flag at
line 2614, alongside the SEPTQ flags, to keep related
quantization-mode flags grouped):

| Flag | Type | Default | Description |
|---|---|---|---|
| `--w4a4` | bool | False | Enable the W4A4 path. When False, the tool produces a weight-only Tile640 GGUF (existing behavior). When True, the tool adds the activation-scale sidecar fields and the LLM.int8 outlier decomposition. |
| `--w4a4-activation-bits` | int | 4 | Activation bit width. The first pass is fixed at 4; the flag is here to make the bit width explicit and forward-compatible with W4A8 / W4A16 comparisons. |
| `--w4a4-outlier-threshold` | float | 6.0 | LLM.int8 outlier threshold (see section 3). |
| `--w4a4-outlier-fraction` | float | 0.001 | Maximum outlier fraction (0.001 = 0.1%). |
| `--w4a4-smoothquant-alpha` | choice | `auto` | `0.5`, `0.75`, or `auto` (see section 4). |
| `--w4a4-scale-mode` | choice | `per_token` | `per_token` (per-token dynamic, default) or `per_tensor` (per-tensor static, only for families promoted by the GA A/B). |

### Position in the quantize_v3 if/elif chain

The existing if/elif chain in
`tools/tile640/quantize_v3.py:3114-3171` is:

1. `use_lrq` (LRQ mode from a calibration policy with rank-r)
2. `use_septq` (SEPTQ two-step PTQ, mutually exclusive with
   imatrix_mse)
3. `use_imatrix_mse` (per-row MSE grid search)
4. `else` (the default Tessera flow)

The CHAMP-Q permute hoist sits above the chain (lines 3083-3101).

The W4A4 path does not introduce a new branch in this chain. It is
a wrapper that runs alongside the existing flow:

- The existing weight quantizer (one of the four branches above) is
  run as before. The output is the same `q` dict with the same
  Tile640 components.
- After the weight quantizer, if `--w4a4` is set, the activation
  quantizer is run on the calibration activations. This produces
  the per-tensor activation scale, the LLM.int8 outlier
  decomposition, and the optional SmoothQuant-folded activation
  tensor.
- The activation quantizer's output is written to the sidecar
  fields in section 2 (the `tessera.w4a4.*` fields in the GGUF
  writer block at lines 393-475).

The activation quantizer sits in the same `if is_quant_2d:` branch
as the weight quantizer (around line 3050) and is gated on
`args.w4a4` and on a non-empty `--imatrix` (the W4A4 path needs
calibration activations to compute the outlier set and the
per-tensor scale; without an imatrix, the W4A4 path raises a
ValueError at startup, not at quantization time).

The order within the branch:

1. CHAMP-Q permute hoist (existing, line 3083-3101).
2. SEPTQ mutual-exclusion check (existing, line 3103-3112).
3. W4A4 activation quantizer setup (new, gated on `args.w4a4`):
   - Apply SmoothQuant fold if `args.w4a4_smoothquant_alpha` is
     set.
   - Compute the per-tensor activation scale and the LLM.int8
     outlier decomposition.
   - This produces the `act_outlier_indices`, `act_outlier_vals`,
     `act_scale_static`, and `act_outlier_count` sidecar entries.
4. The existing if/elif chain (LRQ / SEPTQ / imatrix_mse / else)
   runs the weight quantizer. Unchanged.
5. After the weight quantizer, the activation sidecar fields are
   written to the GGUF alongside the existing weight components.

### Why no separate --quant-mode flag

The user explicitly chose to keep the weight-only path authoritative
and add `--w4a4` as an extension. The reasons:

- The weight components (`weight_packed`, `weight_page_scales`,
  `weight_lane_scales`, `weight_outlier_*`) are byte-equivalent in
  the W4A16 (no `--w4a4`) and W4A4 (with `--w4a4`) cases. A
  consumer that doesn't read the `tessera.w4a4.*` fields gets the
  same weight data either way.
- The activation scale sidecar fields are optional. A consumer
  that doesn't implement the W4A4 dequant kernel ignores them and
  reads the activation as BF16. The runtime falls back to the
  weight-only path on a per-tensor basis: if the tensor has no
  `tessera.w4a4.*` fields, the existing Tile640 weight-only
  dequant is used.
- A separate `--quant-mode` flag would require the runtime to
  branch on the mode, which is invasive across the L1-L6 pipeline.
  A boolean `--w4a4` flag is a simple additive layer.

The trade-off is that the W4A4 path runs the same weight
quantizer as the weight-only path; it cannot opt into a different
weight scheme (e.g. Q4_K weights) without breaking the contract.
This is acceptable for the first pass; a future change could add
`--w4a4-weight-quant` if needed.

## 6. A/B validations as part of the evolutionary search

### Cite

Per the user's third architectural decision, the A/B validation
between W4A16 (weight-only) and W4A4 is integrated into the
`awq-evolve.py` GA. Each candidate policy is evaluated under both
modes, and the GA fitness is a weighted sum of the W4A16 and W4A4
perplexity deltas. This is the runtime-aware counterpart of the
L1-L6 fitness function: it tests the candidate against the actual
runtime dequant, not just the offline `_ternary_reconstruct`
reference in `tools/tessera/awq-evolve.py:219`.

### Fitness composition

The existing `_aggregate_layer_scores` in
`tools/tessera/awq-evolve.py:380` produces a per-candidate `Score`
with `train_error`, `heldout_error`, `tail_error`, `size_cost`,
`fitness`, and `worst_layer_error`. The existing `fitness` formula
is the sum of five terms: `train_error` plus `2.0 * heldout_error`
plus `0.25 * worst_layer_error` plus `0.05 * tail_error` plus
`0.15 * size_cost`.

The W4A4 path extends this with a parallel evaluation:

1. The existing `evaluate()` is called with the W4A16 setup
   (current behavior). This produces `score_w4a16` with the same
   fields as above.
2. A new `evaluate_w4a4()` is called with the W4A4 setup: the
   activation quantizer is applied to the calibration activations,
   the LLM.int8 outlier decomposition runs, and the candidate's
   per-tensor activation scale is folded in. This produces
   `score_w4a4` with the same fields.
3. The combined `fitness` is
   `0.5 * score_w4a16.fitness + 0.5 * score_w4a4.fitness`. The
   0.5/0.5 weight is the default and is exposed as
   `--w4a4-fitness-weight 0.5` on the GA CLI. Setting it to 0.0
   makes the GA optimize weight-only; setting it to 1.0 makes the
   GA optimize W4A4 only.

The 0.5/0.5 default is a starting point. The A/B log records both
scores per candidate per family, so the user can rerun the GA
with a different weight if the 0.5/0.5 default is found to over- or
under-weight the W4A4 signal in practice.

### Per-semantic-family breakdown

The existing family classifier in
`tools/tessera/awq-evolve.py:133` (`infer_family`) maps each
tensor to one of `attention`, `ffn`, `router`, `routed_expert`,
`shared_expert`, `fusion`, `output_embedding`. The W4A4 A/B
extends this to the six user-named sub-families (`attn_q`,
`attn_k`, `attn_v`, `ffn_up`, `ffn_gate`, `ffn_down`) via a new
`infer_subfamily()` helper:

| Family | Sub-family match |
|---|---|
| `attention` | `attn_q`, `attn_k`, `attn_v`, `attn_output` |
| `ffn` | `ffn_up`, `ffn_gate`, `ffn_down` |
| `router` | `ffn_gate_inp` (no sub-family split) |
| `routed_expert` | `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` |
| `shared_expert` | `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` |
| `fusion`, `output_embedding` | no sub-family split |

The fitness report (the per-tensor JSON policy written by
`build_policy` at `tools/tessera/awq-evolve.py:924`) gains two new
fields per tensor:

| Field | Type | Description |
|---|---|---|
| `w4a16_fitness` | float | The existing `evaluate()` fitness under W4A16 mode. |
| `w4a4_fitness` | float | The new `evaluate_w4a4()` fitness under W4A4 mode. |
| `w4a4_per_tensor_static_extrapolable` | bool | A/B result from section 2: True if `ppl_per_tensor_static - ppl_per_token_dynamic < 0.05` for this tensor's sub-family. |
| `w4a4_scale_mode` | string | `per_token` or `per_tensor`; populated only if `--w4a4-scale-mode` was set to `per_tensor` for the A/B. |

The report also gains a per-family rollup block. Each row carries
the four metrics (w4a16, w4a4, extrapolable_per_tensor,
scale_mode) for one sub-family. The schema is fully specified;
the values are populated by the GA at run time. The shape,
illustrated with representative values for a gemma 4 12B QAT
target:

| Family | Sub-family | w4a16 fitness | w4a4 fitness | extrapolable_per_tensor | scale_mode |
|---|---|---|---|---|---|
| attention | attn_q | 0.18 | 0.22 | true | per_token |
| attention | attn_k | 0.17 | 0.21 | true | per_token |
| attention | attn_v | 0.16 | 0.20 | true | per_token |
| attention | attn_output | 0.20 | 0.25 | false | per_token |
| ffn | ffn_up | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| ffn | ffn_gate | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| ffn | ffn_down | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| router | ffn_gate_inp | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| routed_expert | ffn_gate_exps | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| routed_expert | ffn_up_exps | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| routed_expert | ffn_down_exps | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| shared_expert | ffn_gate_shexp | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| shared_expert | ffn_up_shexp | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| shared_expert | ffn_down_shexp | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| fusion | (no split) | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |
| output_embedding | (no split) | (populated by GA) | (populated by GA) | (populated by GA) | (populated by GA) |

The "(populated by GA)" cells are deliberately left blank in
this design doc: the schema is fully specified, but the values
are not known until the GA runs. This is not a TODO; the
implementation agent does not need to fill these in by hand.

### GA plumbing

The GA itself does not change. The 6D mutation space, the
population, generation, and island count are all unchanged
(see `tools/tessera/awq-evolve.py:799-895` for the main `evolve`
loop). The change is in the fitness function: `evaluate_w4a4()` is
added alongside the existing `evaluate()`, and `_aggregate_layer_scores`
gains a new aggregation mode that takes both scores and produces
the weighted sum.

The new `evaluate_w4a4()` function:

- Reuses the existing `_evaluate_layer()` for the weight
  reconstruction. The weight side is unchanged.
- Adds a new `_evaluate_layer_w4a4()` that, after the weight
  reconstruction, applies the activation quantizer to the
  calibration activations and computes the activation-side error.
  The activation-side error is added to the existing per-layer
  `train_error` and `heldout_error` with a configurable weight
  (default 1.0; exposed as `--w4a4-activation-error-weight`).
- Returns a `Score` with the same field shape as the existing
  `Score`, so `_aggregate_layer_scores` can be reused.

The new `--w4a4` flag on `awq-evolve.py` enables the W4A4 fitness
path. Without it, the GA runs in W4A16-only mode (current
behavior); with it, the GA runs the A/B and writes the
`w4a4_*` fields to the policy.

### Cost

Each candidate requires two `evaluate()` calls instead of one.
The W4A4 evaluation is roughly 1.3x the cost of the W4A16
evaluation (the activation quantizer is small relative to the
weight reconstruction, and the LLM.int8 outlier decomposition
is a single pass over the calibration data). The total GA cost
doubles in the worst case (per-tensor activation-side scoring on
every candidate) and grows by ~30% in the typical case (the
activation quantizer is fast). This is acceptable for the
calibration pass; the per-tensor GA at
`tools/tessera/per_tensor_calibrate.py` runs at a similar
multiplier.

## 7. Runtime changes

### New dequant kernel path

The dequant kernel for the W4A4 activation is implemented in a new
file rather than edited into the existing kernel files. The
reason: `ggml/src/ggml-cpu/arch/arm/quants.c` is 4319 lines and
already carries the Q2-Q8, Tile640, and ARM-specific NEON paths;
editing in place is a large merge surface. A new helper file is
cleaner.

The new files:

- `ggml/src/ggml-cpu/arch/arm/quant-tessera-w4a4.c` (CPU/ARM
  NEON implementation; ~300 lines for the activation dequant
  branch, the LLM.int8 outlier branch, and the per-tensor scale
  application).
- `ggml/src/ggml-cuda/quant-tessera-w4a4.cuh` (CUDA
  implementation; a single header with the activation dequant as
  a `__device__` function, called from `vecdotq.cuh`).
- `ggml/src/ggml-metal/quant-tessera-w4a4.metal` (Metal shader;
  the activation dequant as a Metal kernel function, called
  from `ggml-metal.metal` via the standard dispatch).

The new files are NOT modifications to the existing
`ggml-cpu/arch/arm/quants.c`, `ggml-cuda/vecdotq.cuh`, or
`ggml-metal/ggml-metal.metal`; they are additive. The existing
Tile640 weight dequant in those files is unchanged. The new
files are wired in by:

- Adding `quant-tessera-w4a4.c` to the `ggml-cpu/arch/arm/`
  build (CMake change, see `ggml/src/ggml-cpu/CMakeLists.txt`).
- Adding `quant-tessera-w4a4.cuh` to the `ggml-cuda` build
  (`ggml/src/ggml-cuda/CMakeLists.txt`).
- Adding `quant-tessera-w4a4.metal` to the `ggml-metal` build
  (`ggml/src/ggml-metal/CMakeLists.txt`).

The dispatch logic for the W4A4 activation lives in the existing
matmul op kernels (e.g. `ggml-cpu/ops.cpp` for the CPU dispatch,
`ggml-cuda` for CUDA, `ggml-metal-ops.cpp` for Metal). The
dispatch checks `tessera_debug::w4a4_enabled()` and, if true,
routes the activation dequant to the new helper file.

### Per-tensor activation scale sidecar (L1 sidecar)

The per-tensor activation scale is a new field in the L1 sidecar,
written by the GGUF writer in
`tools/tile640/quantize_v3.py:393-475`. The fields are listed in
the section 2 table. The runtime reads these fields at model load
time and caches them per-tensor for the duration of the model
session.

The per-token dynamic scale is NOT stored in the sidecar; it is
computed at matmul time inside the kernel. The per-tensor static
scale, when used, IS stored; it is the `act_scale_static` field.

### L1 sidecar format change (E1 / format evolution)

The L1 sidecar format change for the W4A4 path is part of the
format evolution order Tier-0 track. The user's fourth
architectural decision specifies:

- Tier-0 unlocks (in order): E1 (per-tile outlier count) and E5
  (per-tile Hessian trace). These are added to the L1 sidecar
  without changing the Tile640 weight layout.
- Tier-1 unlocks (first): E2 (Tile640+CSR). This is the first
  format-layer change.

The W4A4 activation outlier count and outlier indices are an E1
addition: the sidecar gets a new `act_outlier_count` and
`act_outlier_indices` per tensor, but the Tile640 weight layout is
unchanged. The E1 change is implemented as part of the W4A4 doc
because the LLM.int8 outlier decomposition is a Tier-0 prerequisite
for the runtime mixed-precision path. The format evolution doc
(separate from this one) will track the E1 schema bump and the
`llama.tessera.per-tensor-calibration.v1 -> v2` migration.

The E2 (Tile640+CSR) change is NOT part of this doc. The W4A4
path uses the existing Tile640 weight format for the INT4 portion
and stores the F16 outlier values as a separate sidecar tensor,
not as a CSR-augmented Tile640 weight. A future format evolution
pass could merge the outlier storage into the Tile640 weight
format, but that is out of scope here.

### Debug sidecar (L1 capture)

The kernel dequant fidelity path (Layer 1, see
`common/tessera-debug.h` and `common/tessera-debug.cpp`)
is extended for the W4A4 path:

- A new env var `LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4` enables
  the W4A4 sidecar capture. The existing `LLAMA_TILE640_DEBUG_DEQUANT_DIR`
  env var still controls the output directory.
- The W4A4 sidecar writes the dequantized activation to
  `<dequant_dir>/<tensor_name>.act.dequant.f32` (parallel to the
  existing `<tensor_name>.dequant.f32` for the weight). The
  dequantized activation is the post-LLM.int8-decomposition,
  post-scale-application value, which is what the matmul
  accumulator sees.
- The existing sidecar format (28-byte header + F32 rows, see
  `common/tessera-debug.h:38-52`) is reused. The dtype field
  uses `DEQUANT_DTYPE_F32` for the activation dump; a future
  F16 dump is forward-compatible (the dtype field is already
  defined).

The activation dump is a Layer 1 artifact: it is the ground truth
for the activation dequant fidelity, and the L2 differential
forward consumes it. The activation dump is the per-tile,
per-matmul counterpart of the weight dump that the L1 hook
already produces.

## 8. L1-L6 mapping

The L1-L6 pipeline is described in `docs/pipeline-design.md` and
`docs/runtime-aware-pipeline.md`. The W4A4 path is a layer on top
of L1-L6; it does not change the pipeline structure. The
per-layer changes are:

### L1 (kernel dequant fidelity)

- The W4A4 activation dequant is added to the kernel instrumentation
  in the new helper files
  (`ggml/src/ggml-cpu/arch/arm/quant-tessera-w4a4.c` etc.). The
  L1 capture path (via `tessera_debug::open_dequant_writer`)
  is extended to dump the activation dequant alongside the weight
  dequant. The L1 sidecar format gets a new `.act.dequant.f32`
  file per tensor; the existing `.dequant.f32` is unchanged.
- New env var: `LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4`. The
  `tessera_debug` API gains a `w4a4_enabled()` function (parallel
  to the existing `dequant_debug_enabled()`) that returns True
  when the W4A4 path is active and the dequant sidecar is
  configured.
- Modified: `common/tessera-debug.h:60-90` (add `w4a4_enabled()`,
  `set_dequant_mode()`), `common/tessera-debug.cpp` (parallel
  state for the W4A4 dir, mirroring `g_dequant_dir`).

### L2 (BF16 vs quant differential forward)

- `tools/tessera/runtime_probe.py` is extended to report a
  per-tensor W4A16 PPL delta and a per-tensor W4A4 PPL delta as
  two columns in the L2 report (schema
  `llama.tessera.runtime-probe.v1`). The differential is computed
  identically for both modes; the only difference is which
  quantized model is loaded (weight-only or W4A4). The L2
  report's `divergence` block gains `w4a4_max_abs`,
  `w4a4_relative_frobenius`, etc. alongside the existing
  `max_abs` and `relative_frobenius` (which become the W4A16
  columns by default).
- Modified: `tools/tessera/runtime_probe.py` (add the W4A4
  column). The schema is appended, not replaced
  (per `docs/architecture.md` calibration schema invariants).

### L3 (per-token coherence)

- The per-token coherence script (currently a sub-mode of
  `runtime_probe.py` per `docs/runtime-aware-pipeline.md:289-296`)
  produces a per-token row with `kl_divergence` and
  `top1_mismatch`. The W4A4 path extends this with a per-token
  breakdown of the W4A4-specific error: the
  `kl_w4a4_int4_path` (the KL contribution from the INT4
  non-outlier path) and `kl_w4a4_outlier_path` (the KL
  contribution from the FP16 outlier path). The total
  `kl_divergence` is unchanged in meaning (BF16 vs quantized
  divergence at the token level).
- Modified: `tools/tessera/runtime_probe.py` (add the
  per-path breakdown).

### L4 (E2E probe)

- The E2E probe in `tools/tessera/e2e_probe.py` produces a single
  PPL delta today. The W4A4 path produces both a `w4a16_ppl_delta`
  and a `w4a4_ppl_delta` in the L4 report
  (schema `llama.tessera.e2e-probe.v1`). The probe's PASS/FAIL
  decision uses the W4A16 criteria for the W4A16 column and
  separate (relaxed) W4A4 criteria for the W4A4 column
  (see section 9). The probe can be run with
  `--w4a16-only` or `--w4a4-only` to test one mode in isolation.
- Modified: `tools/tessera/e2e_probe.py` (add the dual-column
  report), and the L4 schema in
  `docs/runtime-aware-pipeline.md:308-336` is updated to
  reflect the dual-column shape.

### L5 (adaptive requantization)

- Unchanged. The adaptive requantization loop in
  `docs/runtime-aware-pipeline.md:347-411` consumes the L2
  report; the L2 report now has both W4A16 and W4A4 columns,
  and L5's divergence threshold is applied to both. Tensors that
  exceed the W4A4 threshold are requantized; the L5 decision is
  independent of the W4A16 decision.
- No code change in L5 itself.

### L6 (kernel-direct fitness)

- Unchanged in shape. The kernel-direct fitness in
  `tools/tessera/per_tensor_calibrate.py` (per
  `docs/runtime-aware-pipeline.md:415-461`) consumes the L1
  sidecar. With the W4A4 activation dump added, the L1 sidecar
  has both weight and activation dequant; the L6 fitness
  function is extended to consume both, but the API surface
  (the `kernel-direct` fitness mode) is unchanged.
- Modified: `tools/tessera/per_tensor_calibrate.py` (extend
  the `kernel-direct` mode to consume the activation sidecar).

### New vs modified artifacts

| Layer | New | Modified |
|---|---|---|
| L1 | `ggml/src/ggml-cpu/arch/arm/quant-tessera-w4a4.c`, `ggml/src/ggml-cuda/quant-tessera-w4a4.cuh`, `ggml/src/ggml-metal/quant-tessera-w4a4.metal` | `common/tessera-debug.h`, `common/tessera-debug.cpp`, `ggml-cpu/CMakeLists.txt`, `ggml-cuda/CMakeLists.txt`, `ggml-metal/CMakeLists.txt` |
| L2 | -- | `tools/tessera/runtime_probe.py` |
| L3 | -- | `tools/tessera/runtime_probe.py` |
| L4 | -- | `tools/tessera/e2e_probe.py` |
| L5 | -- | (none) |
| L6 | -- | `tools/tessera/per_tensor_calibrate.py` |
| Quantizer | `tools/tessera/w4a4_quantize.py` (the activation quantizer) | `tools/tile640/quantize_v3.py` (new CLI flags + sidecar writes) |
| GA | `tools/tessera/awq-evolve.py:evaluate_w4a4` (new function) | `tools/tessera/awq-evolve.py:infer_subfamily` (new), `build_policy` (extended output) |

The W4A4 activation quantizer at `tools/tessera/w4a4_quantize.py`
is a new module. It is the Python-side counterpart of the C++
dequant kernel: it takes a calibration activation tensor, a
weight tensor, and the W4A4 CLI flags, and produces the per-tensor
activation scale, the LLM.int8 outlier decomposition, and the
optional SmoothQuant-folded weight. It is called from
`quantize_v3.py` at the position described in section 5.

## 9. Validation strategy

### Concrete tests

The validation gates below are mandatory. Each is a CI check that
fails on regression.

#### V1. Weight contract (W4A4 must be additive)

- Build a gemma 4 12B QAT model with `tile640_quantize_v3.py` and
  the same model with `--w4a4`. The weight components
  (`weight_packed`, `weight_page_scales`, `weight_lane_scales`,
  `weight_outlier_*`) must be byte-equivalent between the two
  builds. The activation sidecar fields
  (`tessera.w4a4.*`) must be absent in the first build and
  present in the second.
- Fail condition: any weight component differs.

#### V2. L4 E2E probe dual-column

- `e2e_probe.py --bf16 model-f16.gguf --quantized model-tessera-w4a4.gguf`
  produces a report with both `w4a16_ppl_delta` and
  `w4a4_ppl_delta` columns, and a per-tensor breakdown of
  the W4A4 outlier path contribution. The expected ranges for
  the gemma 4 12B QAT target:
  - `w4a16_ppl_delta < 0.5` (existing, per
    `docs/PROJECT-STATUS.md` Phase 4).
  - `w4a4_ppl_delta < 1.5` (new; the relaxed threshold is
    expected because the activation quantizer is a
    significant new source of error).
  - The first 5 tokens of the Paris probe
    (`docs/runtime-aware-pipeline.md:317-321`) must match
    the BF16 reference in both modes.
- Fail conditions:
  - `w4a16_ppl_delta > 0.5` (regression on the weight-only
    contract).
  - `w4a4_ppl_delta > 2.0` (regression beyond the relaxed
    W4A4 threshold).
  - Any first-5-token mismatch on the Paris probe (the
    `docs/runtime-aware-pipeline.md:280` "first 5 tokens
    include any mismatch" criterion).

#### V3. A/B per-tensor static extrapolation

- The `awq-evolve.py` A/B run produces a per-family yes/no
  decision for the per-tensor static extrapolation. For each of
  the six sub-families (`attn_q`, `attn_k`, `attn_v`,
  `ffn_up`, `ffn_gate`, `ffn_down`):
  - Yes (extrapolable): `ppl_per_tensor_static - ppl_per_token_dynamic < 0.05`
    on the 30-prompt evaluation set.
  - No (not extrapolable): the difference exceeds 0.05.
- The default threshold is 0.05
  (`--w4a4-per-tensor-static-threshold 0.05`). The expected
  outcome on gemma 4 12B is that `attn_q`, `attn_k`, and
  `attn_v` are extrapolable (small range of activation
  magnitudes per head) and `ffn_gate` is not (the SiLU/GeLU
  gate has high dynamic range across tokens).
- Fail conditions:
  - More than 4 of the 6 sub-families are "not extrapolable"
    (suggests the per-tensor scale mode is broken in general).
  - The per-family fitness report is missing or has missing
    fields (schema validation failure).

#### V4. LLM.int8 outlier count

- For each tensor quantized with W4A4, the sidecar's
  `act_outlier_count` is computed and reported in the L2
  report. The expected fraction is <= 0.1% of channels (the
  LLM.int8 finding) for the standard OPT/BLOOM class of
  models. For gemma 4 12B (which has moderate outliers), the
  expected fraction is <= 0.5%.
- Fail condition: any tensor has more than 5% outlier channels
  (a clear signal that the LLM.int8 threshold is wrong or
  the model has a non-standard outlier structure).

#### V5. Tile-by-tile dequant dump

- The C++ dequant kernel produces a tile-by-tile F32 dump via
  the L1 sidecar (`tessera_debug::open_dequant_writer`). The
  Python validation loads the dump and compares each tile to
  the BF16 reference (loaded from the safetensors source).
- Expected tolerances:
  - INT4 non-outlier path: relative Frobenius < 1e-3 per tile.
  - FP16 outlier path: relative Frobenius < 1e-4 per tile.
- Fail conditions:
  - Any tile exceeds the tolerance.
  - The dump is missing a tile (the kernel skipped a tile,
    which is a bug).

#### V6. SmoothQuant fold sanity

- For each tensor with W4A4, the SmoothQuant-folded weight
  `W'` is compared to the original weight `W`. The expected
  per-channel scale `s_j` is in the range
  `[max(|X_j|)^0.5 / max(|W_j|)^0.5, max(|X_j|)^0.75 / max(|W_j|)^0.25]`
  (for alpha=0.5 and alpha=0.75 respectively). For tensors
  with `outlier_count > 10%`, the alpha is auto-set to 0.75;
  for tensors with `outlier_count < 10%`, the alpha is 0.5
  by default.
- Fail conditions:
  - Any per-channel scale is zero or NaN (the fold is broken).
  - Any per-channel scale is outside the expected range (the
    alpha selection logic is wrong).

### Validation run sequence

The validation runs in this order; each is gated on the previous
passing:

1. V1 (weight contract) -- fails fast on the simplest
   additivity check.
2. V4 (LLM.int8 outlier count) -- fails fast if the threshold
   or fraction is misconfigured.
3. V6 (SmoothQuant fold sanity) -- fails fast on a wrong
   alpha selection.
4. V5 (tile-by-tile dequant dump) -- the slowest check,
   requires the C++ side to be built.
5. V3 (A/B per-tensor static) -- requires the GA to have
   produced a policy, so it runs after the GA.
6. V2 (L4 E2E probe) -- the end-to-end gate, requires a
   complete model and probe set.

This sequence is run in CI on a 1B model (gemma 4 1B or
equivalent) for fast PR feedback, and on a 12B model nightly.
The runtime for the 1B pass is < 5 minutes; the 12B pass is
< 1 hour.

## 10. Open questions for the architect

The following questions are blocking for the implementation agent.
Each is a specific design decision that the doc does not fully
resolve.

1. **Activation imatrix format**. The standard `tools/imatrix/`
   imatrix is a per-tensor file with per-input-channel importance
   (`<key>.in_sum2` in the v3 quantizer convention, see
   `tools/tile640/quantize_v3.py:21-25`). The W4A4 outlier
   detection (LLM.int8 threshold) needs per-channel outlier
   identification, not just per-channel importance. Should the
   activation imatrix be extended with an outlier mask (one bit
   per channel per tensor), or should the W4A4 path use a
   separate activation-side imatrix with the outlier
   decomposition baked in? The first option reuses the existing
   imatrix infrastructure; the second adds a new file format
   but keeps the W4A4 path isolated.

2. **LLM.int8 outlier threshold auto-tuning**. The doc specifies
   `--w4a4-outlier-threshold 6.0` as a global default. The
   LLM.int8 paper uses 6.0 as a global threshold, but Tessera's
   per-tensor GA could in principle search the threshold per
   tensor (with the 0.1% outlier fraction as a hard cap). The
   trade-off: a global threshold is simpler and matches the
   LLM.int8 paper, but a per-tensor threshold could
   accommodate model-specific outlier distributions (e.g. the
   gemma 4 12B attention output has tighter outlier distribution
   than the ffn_up). Should the GA extend the mutation space to
   include `w4a4_outlier_threshold` per tensor, or is the global
   6.0 the right default?

3. **L4 E2E probe W4A4/W4A16 divergence attribution**. The probe
   currently reports a single PPL delta. With W4A4 added, the
   report has two columns. If both columns show divergence, the
   report should ideally attribute the divergence to one mode
   or the other. The current design (section 9 V2) reports
   them as independent columns, but the LLM.int8 paper's
   finding is that the activation outlier path contributes
   disproportionately to the W4A4 error. Should the L4 probe
   add a third column `w4a4_outlier_path_ppl_delta` that
   isolates the outlier path's contribution, or is the dual
   column sufficient for the first pass?

4. **Per-token dynamic scale: kernel or pre-matmul**. The
   per-token dynamic scale (section 2) is computed at matmul
   input time. The implementation has two options: (a) compute
   `scale_t = max_c(|X[t, c]|) / 7` inside the kernel as a
   pre-step to the FMA, or (b) emit the per-token scale as a
   pre-matmul auxiliary tensor (one F32 per token) that the
   kernel reads. Option (a) is faster (no auxiliary tensor,
   no extra memory traffic) but more invasive in the kernel
   code; option (b) is simpler but adds a memory round-trip
   per matmul. Apple Silicon's per-tile FMA is bandwidth-bound,
   so option (a) is probably the right choice on Metal; CUDA
   may favor (b) for the SM-level scheduling. The doc does not
   commit to one or the other; this needs an architect
   decision per backend.

5. **W4A4 path interaction with the Tile640 weight
   format**. The E2 (Tile640+CSR) format evolution is
   the first Tier-1 format change; the W4A4 outlier storage
   is a CSR-like addition to the activation side, but it is
   implemented as a separate sidecar tensor (not as a
   Tile640 weight modification). The user explicitly said E1
   (per-tile outlier count) and E5 (per-tile Hessian trace)
   are Tier-0 unlocks; the W4A4 activation outlier count is
   an E1 addition, but the F16 outlier value storage is a
   new sidecar tensor (not a per-tile count). Is the
   F16 outlier value storage part of E1 (count-only sidecar)
   or does it wait for E2 (Tile640+CSR)? The doc's
   interpretation is that the F16 outlier values are an
   E1-side addition because they are sidecar metadata (not
   part of the Tile640 weight format), but a strict reading
   of the format evolution order would put them in E2.

6. **SmoothQuant fold for MoE experts**. The 3D expert
   weight path (the `routed_expert` and `shared_expert`
   families in `awq-evolve.py:42-46`) has the in_dim axis
   shared across experts but the out_dim and (n_experts)
   axes per-expert. The SmoothQuant fold is per-channel on
   the in_dim axis; for experts, the fold should be shared
   across experts (the same `s_j` for every expert's
   in_dim channel). The CHAMP-Q permute in
   `tools/tile640/quantize_v3.py:3243-3308` already does
   this kind of cross-expert sharing. Should the W4A4
   path add a per-family SmoothQuant fold for the
   `routed_expert` family, or skip the SmoothQuant pre-pass
   for MoE tensors and use the raw INT4 path?

These are the design questions the implementation agent cannot
resolve without the architect. The doc's recommendations are
noted in each case but a final decision is needed before the
implementation starts.

## Appendix: references

- `tools/tile640/quantize_v3.py:1-50` (overview, CHAMP-Q import)
- `tools/tile640/quantize_v3.py:119-122` (Tile640 layout constants)
- `tools/tile640/quantize_v3.py:393-475` (GGUF writer block,
  existing `tessera.*` metadata fields)
- `tools/tile640/quantize_v3.py:2394-2614` (CLI flag block,
  position for new `--w4a4*` flags)
- `tools/tile640/quantize_v3.py:3050-3220` (the per-tensor
  quantize loop, position for the W4A4 activation quantizer
  and the sidecar writes)
- `tools/tessera/awq-evolve.py:35-49` (FAMILIES, MATCHES, the
  per-family classifier)
- `tools/tessera/awq-evolve.py:133-138` (`infer_family`,
  extended by `infer_subfamily` for W4A4)
- `tools/tessera/awq-evolve.py:219-252` (`_ternary_reconstruct`,
  the offline reference; the W4A4 fitness is parallel to this)
- `tools/tessera/awq-evolve.py:269-447` (`evaluate` and
  `_aggregate_layer_scores`, extended by `evaluate_w4a4`)
- `tools/tessera/awq-evolve.py:799-895` (the main `evolve` loop,
  unchanged)
- `tools/tessera/awq-evolve.py:967-1197` (CLI and main, new
  `--w4a4` flag goes here)
- `tools/tessera/per_tensor_calibrate.py` (per-tensor GA; the
  W4A4 fitness mode is added here)
- `common/tessera-debug.h:1-90` (L1 sidecar API; extended by
  `w4a4_enabled()`, `set_dequant_mode()`)
- `common/tessera-debug.cpp:1-132` (L1 sidecar implementation;
  extended by W4A4 state)
- `ggml/src/ggml-cpu/arch/arm/quants.c` (existing ARM NEON
  dequant, NOT modified by W4A4)
- `ggml/src/ggml-cuda/vecdotq.cuh` (existing CUDA dequant, NOT
  modified by W4A4)
- `ggml/src/ggml-metal/ggml-metal.metal` (existing Metal
  dequant, NOT modified by W4A4)
- `docs/pipeline-design.md` (L1-L6 design)
- `docs/runtime-aware-pipeline.md` (L1-L6 implementation plan)
- `docs/per-tensor-calibration.md` (per-tensor GA calibration)
- `docs/architecture.md` (schema invariants, calibration tooling
  overview)
- Research report section 2a (HAWQ, SmoothQuant, LLM.int8
  diagnostics)
- Research report section 2b (LLM.int8 outlier-driven hardness,
  SmoothQuant migration, OliVe isolation, QuIP rotation)
- Research report section 2d (GPTQ, AWQ, SpQR, QuIP reconstruction
  methods)
- Research report section 2e (QEP error accumulation, Lens of
  Perturbation)
- Research report section 4 (L1-L6 mapping and concrete
  per-layer opportunities)
- Research report section 4a (methods to adopt, defer, skip)
