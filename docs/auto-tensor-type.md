# `llama-auto-tensor-type`

A tool that picks per-tensor-role quantization types to meet a target bits-per-weight
(BPW) budget while minimizing the quality impact of quantization. It produces a
`tensor-type-file` that `llama-quantize --tensor-type-file` consumes directly.

## Why this tool exists

For a given target BPW, the question of "how do I distribute bits across tensor roles?"
does not have a universal answer — the right allocation depends on the model's
architecture, the role's sensitivity to quantization, and the set of available quant
types. Hand-crafted recipes (e.g. "Q6_K for attention output, IQ3_TQ for FFN gate/up")
embed implicit judgments about which tensors are load-bearing. This tool tries to
derive those judgments from measurements and produce a numerically-justified recipe.

## Quick start

```
build/bin/llama-auto-tensor-type \
  -m /path/to/model.gguf \
  -i /path/to/model.imatrix.gguf \
  -q IQ3_TQ,Q4_DPT,IQ4_XS,Q4_K,Q5_K,Q6_K \
  -b 4.25 \
  -o /tmp/recipe.types \
  --test-data /path/to/calibration.txt \
  --test-sizes 128,512 \
  --threads 32 \
  --cost-matrix-cache /tmp/model.cm.cache
```

Then quantize and validate:

```
build/bin/llama-quantize \
  --imatrix /path/to/model.imatrix.gguf \
  --tensor-type-file /tmp/recipe.types \
  /path/to/model.gguf \
  /tmp/quantized.gguf \
  IQ4_XS 32

build/bin/llama-perplexity \
  -m /tmp/quantized.gguf \
  --kl-divergence-base /path/to/reference.kld \
  --kl-divergence \
  -ngl 99 -t 32 -fa 1 -c 512 -b 512 -ub 512 \
  -f /path/to/calibration.txt
```

The `IQ4_XS 32` positional args at the end of `llama-quantize` are the **fallback
ftype** and thread count. Tensors not matched by the recipe (small 2D weights below
`--min-elements`) get the fallback ftype. Pick a fallback whose BPW is close to
your target so the unmeasured tensors don't push BPW off-budget.

## Pipeline

The tool runs in five phases:

1. **Metadata load.** Opens the source GGUF with `no_alloc=true` (we only need tensor
   names/shapes/offsets; raw weight data is read later on demand). Enumerates tensor
   roles via regex (`blk.N.ROLE.weight` → `ROLE`; global tensors like `token_embd`
   keep their full name). Computes layer equivalence classes and per-(class, bucket)
   layer assignments.

2. **Activation capture.** Loads the model via the llama API, runs one forward pass
   per `--test-size` × `--test-samples` over either real text (`--test-data`,
   recommended) or synthetic tokens, and captures the input (`src[1]`) and reference
   output of every `MUL_MAT` op whose `src[0]` weight is a quantization target.
   Captures are organized per (role, bucket).

3. **Cost matrix.** For each (role, bucket, candidate type) cell, re-quantizes each
   captured weight to the candidate type, replays the captured input through
   `MUL_MAT`, and measures the relative L2 error of the output against the reference
   (see [Cost metric](#cost-metric)). Results are averaged over captures within the
   cell. **This is the slow phase** — typically 30-50 minutes for a 9B model with
   7 candidate types — and is the target of the [cost-matrix cache](#cost-matrix-cache).

4. **Optimization.** Exact multi-choice knapsack DP over per-(role, bucket) items.
   Each item picks one candidate type; the objective is the element-weighted sum of
   per-cell relative-L2 errors, subject to the BPW budget (`target - tol_low ≤ BPW ≤
   target + tol_high`). See [Optimizer](#optimizer).

5. **Emit recipe.** Writes a `tensor-type-file` with anchored regexes:
   `blk\.(<layers>)\.ROLE\.=TYPE` for layer tensors, `^ROLE=TYPE` for globals.

## Cost metric

For each captured MUL_MAT output (a `[ne_out, n_tokens]` F32 matrix), per-row error
is computed as

```
err(ref_row, quant_row) = ||ref - quant||² / ||ref||²
```

(scale-free relative squared L2 per row), then averaged over rows and captures. This
is the per-cell value the optimizer minimizes.

### Why relative L2 and not softmax-KLD

Softmax over an intermediate MUL_MAT output is not a meaningful probability
distribution by itself, and its peakedness depends on the row's absolute magnitude.
Rows with large magnitudes (typical of attention Q/K/V/QKV projections, which read
from the post-norm residual stream) produce near one-hot softmaxes and dramatic
KLDs; rows with small magnitudes (post-gate FFN inputs, post-attention values)
produce flat softmaxes and tiny KLDs. The signal is thus **inversely correlated**
with the downstream cost of quantizing the tensor — softmax-KLD penalises exactly
the wrong tensors.

Relative L2 is scale-free per-row: it measures fractional output perturbation
directly, which is a better proxy for how the error propagates through the residual
stream. The tool used softmax-KLD historically (with an `--importance-alpha` hack
to compensate for the inverse-correlation); the relative-L2 metric obviates the
hack for typical mid-BPW targets, but [importance-alpha](#importance-alpha-bias-correction)
is still available as an opt-in tuning knob.

### Caveats

- **Local metric ≠ global impact.** Per-cell relative L2 measures the fractional
  perturbation the tensor introduces *into its own output*. It does not model how
  that perturbation amplifies through downstream nonlinearities. Pre-softmax
  tensors (`attn_q`, `attn_k`, `attn_qkv`) are particularly under-weighted: a small
  pre-softmax error becomes a large attention-weight shift after exp + normalize.
  This shows up at low BPW targets (≤ ~3.75), where the DP cheerfully pushes
  IQ2_TQ onto these roles based on local metric alone.
- **Errors compound across layers, not summed.** The DP minimises a sum-of-cells
  objective. Real model output KLD compounds through the residual stream and
  norms; the sum is a useful proxy but not the true loss.
- **Trained-quant captures use the model's own (unquantized) activations.** For
  trained types (IQ2_TQ, IQ3_TQ, Q4_DPT, Q3_PT, Q3_KPT, Q2_KPT), the measured
  error uses levels trained on the same data the imatrix was built from, which
  is a mildly optimistic estimate of end-use behavior.

## Layer equivalence classes and bucketing

Hybrid models (e.g. Qwen3.5's SSM + attention mix) have different tensor signatures
per layer. To avoid measuring every layer:

1. Compute per-layer signature `sorted([(role, ne0, ne1)])` over all quantizable
   weights in the layer.
2. Group layers by signature into equivalence classes.
3. Within each class, divide layers into `--layer-buckets` (default 3) — first
   third, middle third, last third by position. Different buckets of the same
   role can get different quant types (matches hand-tuned recipes that protect
   first/last layers).
4. Sample `--reps-per-bucket` (default 3) layers per (class, bucket) for activation
   capture, evenly stratified. The first and last layer of each class are *always*
   pinned regardless of `K` — boundary layers are most sensitive (cf.
   `llama-quant.cpp:use_more_bits`).

Per-(role, bucket) cost-matrix entries are averaged across all sampled captures
for that cell. Larger `--reps-per-bucket` reduces noise at the cost of more
Phase-3 quantize+eval work.

## Optimizer

Exact dynamic-programming multi-choice knapsack over per-(role, bucket) items:

- One item per (role, bucket) cell with a positive element count. Globals
  (`token_embd`, `output`) are pinned before the DP runs (see [Globals](#globals))
  and enter as items with one valid choice and forbidden alternatives.
- One choice per candidate `qtype` in `--quants`. Each choice has:
  - bits = `n_elements * compute_bpw(qtype)`
  - cost = `n_elements * relative_L2_avg(role, bucket, qtype)` (element-weighted)
- The DP maximises -cost (minimises element-weighted error) subject to total
  bits ∈ [`(target - tol_low) * total_elements`, `(target + tol_high) * total_elements`],
  with overhead from non-quantizable tensors (norms, biases, small 2D matrices)
  added to the budget floor.
- Budget axis discretised at 16384 units (~5×10⁻⁵ BPW step). Table size is
  `O(n_items × budget_units)` — fast even for 36-cell hybrid models.

If the lower budget bound is unreachable (rare; happens with tied embeddings
forcing `token_embd` to a high BPW), the DP falls back to the lowest-cost
assignment within the upper bound. If the target itself is infeasible (e.g.
embedding alone exceeds it), the tool reports the smallest achievable BPW.

### Globals

`token_embd` and `output` are not measured for KLD — `token_embd` uses
`ggml_get_rows` (not MUL_MAT, so no signal), and `output` is force-pinned regardless
of measurement, so capturing it would waste Phase-3 work on a typically-huge tensor.

- `output` → `--output-tensor-type` if set, else `default_output_type_for_target()`:
  the lowest candidate with `bpw ≥ max(target_bpw + 1.0, 5.0)`, falling back to the
  highest candidate. The 5.0-bpw floor reflects the empirical sweet spot for output
  on Qwen3.5-9B (Q5_K beats Q6_K by 20% Mean KLD at 4.25 BPW target with the same
  total budget) and matches hand-tuned recipes (bartowski/unsloth use Q5_K for
  output across IQ4_XS, Q4_K_S, Q3_K_M).
- `token_embd` → if tied embeddings detected (no distinct `output.weight` in the
  model), force-matches `output_type` because `llama-quantize` will do the same
  silently regardless of what we ask. Otherwise → closest-BPW `get_rows`-compatible
  candidate to target.

## Cost-matrix cache

Phase 3 is the wall-time bottleneck (30-50 min on a 9B model with 7 candidate types,
of which 2-3 are CPU-bound trained types). For BPW sweeps on the same model the
cost matrix doesn't change between runs — only the DP target shifts.

Pass `--cost-matrix-cache PATH` to read+write a text-format cache:

```
build/bin/llama-auto-tensor-type ... -b 4.5  --cost-matrix-cache /tmp/m.cm.cache
build/bin/llama-auto-tensor-type ... -b 4.25 --cost-matrix-cache /tmp/m.cm.cache  # cache hit, ~0.2 s
build/bin/llama-auto-tensor-type ... -b 4.0  --cost-matrix-cache /tmp/m.cm.cache  # cache hit, ~0.2 s
```

Cache invalidation is exact: the header records every parameter that affects what
the cost matrix would contain, and **any** mismatch invalidates the cache (the tool
runs Phase 2+3 normally and overwrites the file). Validated fields:

- `model_path`, `model_size`, `model_mtime`
- `quant_types` (sorted)
- `min_elements`
- `n_layer_buckets`, `n_reps_per_bucket`
- `test_data`, `test_data_size`
- `test_sizes`, `n_test_samples`

`--target-bpw`, `--bpw-tolerance`, `--output-tensor-type`, `--importance-alpha`,
and `--max-iterations` are *not* part of the cache key — these only affect
Phase 4, which always re-runs (it's near-instant). That's the whole point of the
cache: tune them freely.

The cache is a plain text file (one entry per `(role, bucket, qtype)` line); safe
to inspect, diff, or delete.

## Importance-alpha bias correction (optional)

Each role's per-cell error can be multiplied by an imatrix-energy weight:

```
role_energy[r]   = mean over tensors in r of mean(imatrix[tensor])
geomean          = geometric mean of role_energy[r] across roles with data
role_weight[r]   = (geomean / role_energy[r]) ^ alpha
weighted_err[r]  = local_err[r] * role_weight[r]
```

`--importance-alpha 0` (default) disables the correction; `1` is full inverse-energy.
With the relative-L2 metric the correction is much less critical than under
softmax-KLD (where it was on by default), but it still has empirical use when the
target BPW is mid-range and the imatrix is high-quality. Setting α = 1 promotes
low-energy residual-write tensors (`ffn_down`, `attn_output`, `ssm_out`) at the
expense of high-energy pre-softmax tensors. The weights are displayed in the run
log.

## Fusion map

When a model fuses projections (e.g. `attn_qkv` = Q‖K‖V as one MUL_MAT), the capture
belongs to the fused role. A `fusion_map` in the source splits the measured error
proportionally (by element count) to the component roles that lack their own
captures. Current entries:

- `attn_qkv` → `{attn_q, attn_k, attn_v}`

Extend in source as needed per arch.

## Implementation notes

- **Per-(weight, qtype) quant caching.** Same weight captured at multiple test
  sizes is quantized once per type; the cached `quant_result` is reused by every
  capture's `eval_mul_mat`. This avoids a latent CUDA race on per-tensor grid/level
  device symbols when multiple captures of the same (weight, qtype) would
  otherwise run concurrently.
- **Parallelism.** `--threads N` launches N `std::async` workers per (weight,
  capture) batch. The CUDA backend pool pre-creates one backend per thread because
  the VMM pool requires strict LIFO alloc/free per backend. Effective quantize
  parallelism is bounded by `n_quant_types` (the outer weight loop is sequential);
  eval parallelism scales with `n_threads`.
- **Output skip.** `output.weight` is excluded from Phase-2 capture because it's
  force-pinned in Phase 4 and the per-tensor KLD measurement would be discarded
  — significant savings on models with large vocab × hidden output projections.
- **llama-quantize override fix.** `llama-quant.cpp` historically treated
  `--tensor-type-file` overrides equal to the ftype default as "no override" and
  then ran the auto-upgrade rules (e.g. n_gqa ≥ 4 promotes `attn_v` IQ4_XS → Q5_K).
  The fix: regex match counts as `manual = true` regardless of whether the type
  literally changes, so user intent is honoured. Without this, recipes that
  happen to coincide with the ftype default get silently re-mapped.
- **Memory.** After Phase 2 finishes, the llama model is reset and the capture
  scaffolding (`target_weight_names`, `weight_to_role`, tokenized test inputs) is
  freed. During Phase 3 each role's capture buffers are released as soon as the
  corresponding cost-matrix row is populated. The GGUF context is opened with
  `no_alloc=true` to avoid duplicating the model in host RAM.

## CLI reference

```
-m, --model PATH         Input GGUF model (required)
-i, --imatrix PATH       Importance matrix (required)
-q, --quants LIST        Comma-separated candidate quant types (required)
-b, --target-bpw N       Target bits per weight (required)
-o, --output PATH        Output tensor-type-file (required)
--bpw-tolerance H,L      BPW tolerance: +H, -L from target (default: +0,-0.2)
--test-data PATH         Text file for test inputs (recommended; synthetic otherwise)
--test-samples N         Samples per test size (default: 1)
--test-sizes S1,S2,S3    Token counts for test inputs (default: 32,128,512)
--min-elements N         Skip KLD measurement for tensors below this size (default: 40000)
--output-tensor-type T   Quant type for output.weight (default: lowest candidate with
                         bpw >= max(target_bpw + 1.0, 5.0), else highest)
--max-iterations N       Reserved (the DP is exact, not iterative; default: 100)
--threads N              Parallel workers (default: 1)
--layer-buckets N        Per-class layer buckets for independent quant assignment
                         (default: 3 — first/middle/last third). 1 reproduces
                         the older per-role-only behavior.
--reps-per-bucket K      Layers sampled per (class, bucket) for activation capture
                         (default: 3). Boundary layers always sampled additionally.
--importance-alpha A     Imatrix-energy bias exponent (default: 0; opt-in).
--cost-matrix-cache PATH Read/write Phase-3 cost matrix (skip Phase 2+3 on hit).
```

## Empirical results

Measured 2026-05-01 on `Qwen3.5-9B-BF16.gguf` with imatrix calibrated on
`calibration_data_v5_rc.txt`, KLD basis from the unquantized BF16 model on
`eval_dataset_260412-0326.txt` (103 chunks). Quants list:
`IQ2_TQ,IQ3_TQ,Q4_DPT,IQ4_XS,Q4_K,Q5_K,Q6_K`. Settings: `--test-sizes 128,512
--threads 32` (defaults otherwise).

| Target BPW | Mean KLD | Median | 95% | 99% | Max | Notes |
|---|---|---|---|---|---|---|
| 4.5  | **0.0275** | 0.0172 | 0.0798 | 0.1862 | 11.96 | output Q5_K |
| 4.25 | **0.0366** | 0.0223 | 0.1088 | 0.2768 |  2.57 | output Q5_K |
| 4.0  | 0.0608 | 0.0363 | 0.1857 | 0.4318 |  9.48 | IQ2_TQ on 6 cells |
| 3.75 | 0.0830 | 0.0473 | 0.2550 | 0.6368 | 13.35 | IQ2_TQ on 8 cells |
| 3.5  | 0.1243 | 0.0720 | 0.3932 | 0.9375 |  9.62 | IQ2_TQ on 16 cells |

The 4.5 and 4.25 results clear the user-stated targets (< 0.03 and < 0.04
respectively). Quality degrades super-linearly below 4.0 BPW because the DP
exhausts low-error options and pushes IQ2_TQ onto pre-softmax tensors
(attn_q, attn_qkv) — exactly the case where the local relative-L2 metric most
under-predicts global impact (see [Caveats](#caveats)).

## Known limitations

- **BPW estimate ignores GGUF metadata.** Tokenizer vocab, model config, and
  alignment padding contribute ~2% to the file size. The tool's predicted BPW is
  tensor-only. Expect final file size ≈ predicted size × 1.02.
- **Tied embeddings force `token_embd` up.** As noted in [Globals](#globals), the
  tool detects the tied case and reports the promoted BPW correctly. On such
  models, raising `target_bpw` or overriding `--output-tensor-type` with a cheaper
  type is the only way to free budget for other roles.
- **Small matrix weights are best-effort.** 2D `.weight` tensors below
  `--min-elements` aren't measured for KLD; they're assumed to be quantized to
  the candidate type closest to the target BPW for accounting purposes. On most
  archs this is within ~1 MB of reality but can drift for unusual tensor shapes.
- **Synthetic tokens are worse than real text.** Default is `rand() % n_vocab`,
  which has different activation statistics from real inputs. Always pass
  `--test-data` when quality matters.
- **Local error is a proxy.** See the caveats above — pre-softmax tensors at low
  BPW are the most exposed. For critical quantizations, verify with
  `llama-perplexity --kl-divergence` on held-out text and compare against a
  known-good reference.
