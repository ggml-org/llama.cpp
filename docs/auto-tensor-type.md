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

## Pipeline

The tool runs in five phases:

1. **Metadata load.** Opens the source GGUF with `no_alloc=true` (we only need tensor
   names/shapes/offsets; raw weight data is read later on demand). Enumerates tensor
   roles via regex (`blk.N.ROLE.weight` → `ROLE`; global tensors like `token_embd`
   keep their full name).

2. **Activation capture.** Loads the model via the llama API, runs one forward pass per
   `--test-size` over either synthetic tokens or real text (`--test-data`), and
   captures the input (`src[1]`) and reference output of every `MUL_MAT` op whose
   `src[0]` weight is a quantization target. Captures are organized per role.

3. **Cost matrix.** For each (role, candidate type) pair, re-quantizes each captured
   weight to the candidate type, replays the captured input through `MUL_MAT`, and
   measures the KL divergence of the output against the reference. See
   [KLD measurement](#kld-measurement) below. Results are averaged over captures
   within the role.

4. **Optimization.** Greedily upgrades role→type assignments to minimize (possibly
   energy-weighted) total KLD subject to the BPW budget. See
   [Optimizer](#optimizer).

5. **Emit recipe.** Writes a `tensor-type-file` with anchored regexes:
   `\.ROLE\.=TYPE` for layer tensors, `^ROLE=TYPE` for globals.

## KLD measurement

For each captured MUL_MAT output (a `[ne_out, n_tokens]` F32 matrix), per-row KLD is
computed as

```
p = softmax(ref_row), q = softmax(quant_row)
KLD(p || q) = sum_i p_i * log(p_i / q_i)
```

and averaged over rows and captures. This is the signal the optimizer minimizes.

### Caveats

- Softmax over an intermediate MUL_MAT output is not a meaningful probability
  distribution by itself. Rows with large absolute magnitudes (typical of
  attention Q/K/QKV projections, which read from the post-norm residual stream)
  produce visibly peaked softmaxes and dramatic-looking KLDs. Rows with small
  magnitudes (post-gate FFN inputs, post-attention values) produce flatter
  softmaxes and small KLDs. **This systematically under-weights residual-path
  tensors like `ffn_down` and `attn_output`**, which often carry more downstream
  importance than their local KLD suggests. The `--importance-alpha` correction
  (below) compensates for this.
- Local per-tensor KLD is summed as if tensors were independent, but errors
  compound non-linearly through residuals and norms. No attempt is made to model
  that propagation directly.
- Captures use the model's own (unquantized) activations. For trained-type
  quantizations (IQ2_TQ, IQ3_TQ, Q4_DPT, ...), the measured KLD uses the levels
  trained on the same data the imatrix was built from, which is a mildly
  optimistic estimate of end-use behavior.

## Importance-alpha bias correction

To counter the local-KLD bias above, each role's KLD is optionally multiplied by a
weight derived from its imatrix energy:

```
role_energy[r]     = mean over tensors in r of mean(imatrix[tensor])
geomean            = geometric mean of role_energy[r] across roles with data
role_weight[r]     = (geomean / role_energy[r]) ^ alpha
weighted_kld[r]    = local_kld[r] * role_weight[r]
```

`alpha = 0` disables the correction; `alpha = 1` is a full inverse-energy flip.
The weights are displayed in the run log.

### Why imatrix energy?

The imatrix records per-column input activation energy. Residual-path inputs
(post-gate FFN, post-attention) have low activation energy because the non-linearity
or attention weighting compresses the signal into a subspace. Those same tensors
write back into the residual stream, where errors accumulate. Inverse-energy
weighting promotes exactly those tensors. Empirically (see
[Calibration](#calibration)) this aligns with hand-tuned recipes.

### Calibration

Measured on `Qwen3.5-0.8B-BF16` and `Nanbeige4.1-3B-BF16` at target BPW 4.5,
quant-list `IQ3_TQ,Q4_DPT,IQ4_XS,Q4_K,Q5_K,Q6_K`, real-text test data:

| α | Qwen3.5-0.8B PPL | Nanbeige-3B PPL |
|---|---|---|
| 0.0 | 54.2149 | 45.4793 |
| 0.5 | 54.2657 | 46.7783 (worse than baseline) |
| **1.0** | **53.2286** | **44.8957** |
| 1.5 | 52.1232 | (same recipe as 1.0) |
| 2.0 | 52.4972 (starts regressing) | (same recipe as 1.0) |

`alpha=1.0` is the default. It's the first setting where the correction reliably
flips allocations on residual-path tensors (promotes `ffn_down`, `ssm_out`,
`attn_output`) on both tested architectures. Lower values (`0.5`) are marginal and
can regress on some architectures. Higher values (`1.5+`) sometimes win on dense
architectures but over-correct on sparser ones.

## Optimizer

Greedy, two-stage, on a (role × type) cost matrix:

1. **Seed.** Every role starts at the lowest-BPW type in the candidate list that has
   a finite KLD.
2. **Greedy upgrade.** Repeatedly pick the single role→type upgrade with the highest
   `Δkld / Δbpw` ratio that still fits the BPW budget (`target + bpw_tol_high`),
   until no improving move exists.
3. **Swap improvement.** Up to `--max-iterations` rounds, try upgrading one role
   while downgrading another to stay in budget and reduce total weighted KLD.
   Terminates on convergence.

BPW accounting uses `ggml_get_bpw` (block size + per-tensor-trained overhead).
Global tensors (`token_embd`, `output`) are pinned before optimization begins:
- `output` → `--output-tensor-type`, or highest-BPW candidate if unset.
- `token_embd` → if tied embeddings detected (no distinct `output.weight` in the
  model), force-matches `output_type` because `llama-quantize` will do the same
  upgrade silently regardless of what we ask for. Otherwise → closest-BPW
  `get_rows`-compatible candidate to target.

## Layer equivalence classes

Hybrid models (e.g. Qwen3.5's SSM + attention mix) have different tensor signatures
per layer. To avoid measuring every layer:

1. Compute per-layer signature `sorted([(role, ne0, ne1)])`.
2. Group layers by signature into equivalence classes.
3. Sample representative layers per class (first, middle, last; deduped).

Only representative layers are captured in Phase 2. Per-role KLDs are averaged across
all representatives within their class.

## Fusion map

When a model fuses projections (e.g. `attn_qkv` = Q‖K‖V as one MUL_MAT), the capture
belongs to the fused role. A `fusion_map` in the source splits the measured KLD
proportionally (by element count) to the component roles that lack their own
captures. Current entries:

- `attn_qkv` → `{attn_q, attn_k, attn_v}`

Extend as needed per arch.

## Implementation notes

- **Per-(weight, qtype) quant caching.** Same weight captured at multiple test
  sizes is quantized once per type; the cached `quant_result` is reused by every
  capture's `eval_mul_mat`. This is ~3× faster than the naive per-capture approach
  and also avoids a latent CUDA race on per-tensor grid/level device symbols when
  multiple captures of the same (weight, qtype) would otherwise run concurrently.
- **Parallelism.** `--threads N` launches N `std::async` workers per (weight,
  capture) batch. The CUDA backend pool pre-creates one backend per thread
  because the VMM pool requires strict LIFO alloc/free per backend.
- **Memory.** After Phase 2 finishes, the llama model is reset and the capture
  scaffolding (`target_weight_names`, `weight_to_role`, tokenized test inputs) is
  freed. During Phase 3 each role's capture buffers are released as soon as the
  corresponding cost-matrix row is populated. The GGUF context is opened with
  `no_alloc=true` to avoid duplicating the model in host RAM.

## CLI reference

```
-m, --model PATH         Input GGUF model (required)
-i, --imatrix PATH       Importance matrix (required, also used for bias correction)
-q, --quants LIST        Comma-separated candidate quant types (required)
-b, --target-bpw N       Target bits per weight (required)
-o, --output PATH        Output tensor-type-file (required)
--bpw-tolerance H,L      BPW tolerance: +H, -L from target (default: +0,-0.2)
--test-data PATH         Text file for test inputs (recommended; synthetic otherwise)
--test-sizes S1,S2,S3    Token counts for test inputs (default: 32,128,512)
--min-elements N         Skip KLD measurement for tensors below this size (default: 40000)
--output-tensor-type T   Quant type for output.weight (default: highest from list)
--max-iterations N       Max swap-improvement iterations (default: 100)
--threads N              Parallel workers (default: 1)
--importance-alpha A     Imatrix-energy bias exponent (default: 1.0, range 0..2)
```

## Known limitations

- **BPW estimate ignores GGUF metadata.** Tokenizer vocab, model config, and
  alignment padding contribute ~2% to the file size. The tool's predicted BPW is
  tensor-only. Expect final file size ≈ predicted size × 1.02.
- **Tied embeddings force `token_embd` up.** As noted above, the tool now detects
  the tied case and reports the promoted BPW correctly. On such models, raising
  `target_bpw` or overriding `--output-tensor-type` with a cheaper type is the only
  way to free budget for other roles.
- **Small matrix weights are best-effort.** 2D `.weight` tensors below
  `--min-elements` aren't measured for KLD; they're assumed to be quantized to the
  candidate type closest to the target BPW for accounting purposes. On most archs
  this is within ~1 MB of reality but can drift for unusual tensor shapes.
- **Synthetic tokens are worse than real text.** Default is `rand() % n_vocab`,
  which has different activation statistics from real inputs. Always pass
  `--test-data` when quality matters.
- **Local KLD is a proxy.** See the caveats above. For critical quantizations,
  verify with `llama-perplexity` on held-out text and compare against a known-good
  reference.
