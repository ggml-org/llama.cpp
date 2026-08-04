# imatrix calibration - optimization and quality study

Run: `imatrix-study`. Mode: research + design (no code, no builds).
Baseline SHA: `10222c950`. Date: 2026-08-02.

This document answers three questions:

1. **Quality** - what does the current imatrix calibration actually do, and
   where are the algorithmic opportunities to get BETTER calibration data?
2. **Optimization** - can the temporal/spatial/memory optimizations from the
   tessera quantization pipeline be applied to the imatrix producer?
3. **Drafter feed** - how does this feed the multi-drafter work (drafter
   training consumes the imatrix tool's feature capture)?

The headline finding (verified by the orchestrator and re-confirmed here by
reading the source): the imatrix PRODUCER (`tools/imatrix/imatrix.cpp`, 2975
lines) and the tessera quantize pipeline (`tools/quantize/tessera/`) are two
independent inference paths over the same model. The pipeline has ~11 commits
of optimization (mmap, streaming weights, Metal, vDSP, sharded eval, streaming
MSE, DuckDB). The producer uses none of them. Section 3 verdicts each one.

---

## 1. Current-state analysis

### 1.1 Three pieces, do not confuse them

| Piece | Path | Role |
|-------|------|------|
| PRODUCER | `tools/imatrix/imatrix.cpp` (2975 ln) | Runs inference over a calibration corpus, accumulates per-tensor activation stats |
| DATA SHAPE | `common/imatrix-loader.{h,cpp}` | On-disk GGUF/dat contract: per-tensor `sums, abs_sums, fourth_sums, max_abs, counts` |
| CONSUMERS | `tools/quantize/tessera/tessera-imatrix.cpp` (352 ln), `tessera-mm-imatrix.cpp` (416 ln) | npz/GGUF readers; derive regime stats; expose act_scales |

### 1.2 What the producer collects (the "4 moments + max" story)

`struct Stats` (`imatrix.cpp:54-60`) holds five per-channel vectors:

- `values`        = sum of x^2 (the squared-activation second moment)
- `abs_values`    = sum of |x| (first absolute moment)
- `fourth_values` = sum of x^4 (fourth moment, for kurtosis)
- `max_values`    = running max of |x| (the only non-additive stat)
- `counts`        = number of activations accumulated

The "rich observer" path (`collect_graph_observers`, line 565) reads a single
fused compact observer tensor per ubatch that packs all four moments + a count
into `[4*channels + 1]` floats per expert (line 766). The reduction
(`reduce_graph_observers`, line 706) is already async across two staging slots
(`k_observer_slots = 2`, line 153) so CPU accumulation overlaps the next Metal
graph. This is the producer's ONLY performance optimization today.

Per-tensor stats are bucketed by `(scope, name)` (`stats_key`, line 101):
`LLAMA_OBSERVER_SCOPE_VERIFIER` (bare name, unchanged on-disk contract) vs
`LLAMA_OBSERVER_SCOPE_DRAFTER` (prefixed `dft.`). This is the multi-drafter
hook - a verifier and a drafter sharing one collector land in separate ledgers.

### 1.3 Incremental collection (progressive transfer)

`struct observer_transfer_state` (line 77) plus `update_progressive_transfer`
(line 868) implement a streaming convergence detector:

- Every `imatrix_convergence_interval` chunks, a downsampled "signature" of the
  3 moments (stride = `channels/32`, ~96 floats/expert) is compared to the
  previous window's signature via RMS delta.
- If `rms_delta <= imatrix_convergence_tolerance` for
  `imatrix_convergence_patience` consecutive windows AND expert coverage is
  adequate (`imatrix_min_expert_coverage`), the tensor is **frozen**: the
  observer filter (`observer_enabled`, line 1061) returns false for it, so the
  graph builder stops inserting observer nodes for that tensor. This is real
  compute savings on long corpora.
- A frozen tensor is **probed** every `4 * convergence_interval` chunks
  (`next_probe`, line 991): briefly re-enabled to check for distribution
  drift; if `rms_delta > 2 * tolerance`, it reopens (`frozen=false`, line 983).

The transfer ledger is serialized beside the imatrix (`<imatrix>.transfer.json`,
schema `llama.tessera.progressive-observer-ledger.v1`, line 1336) so collection
can resume across runs. This is the producer's most sophisticated feature and
is already a partial answer to "better calibration at lower compute".

### 1.4 The three operating modes (selected at main:, line 2778)

| Mode | Flag | Function | Output |
|------|------|----------|--------|
| Plain chunked | (default) | `compute_imatrix` (line 1608) | `imatrix.gguf` |
| Spec-decoding cal | `--model-draft --spec-steps N` | `compute_imatrix_spec` (line 2053) | `imatrix.gguf` (+ optional `--telemetry-out`) |
| Offline feature capture | `--features-out P --feature-layers 0,15,31` | `compute_features` (line 1873) | `P.bin` + `P.json` (NO imatrix) |

The three are mutually exclusive (asserted line 2905-2908). The spec mode is
the most intricate: it runs a real spec-decoding loop (DRAFT_SIMPLE), with
per-prefix verifier forwards so each draft token gets a real verifier softmax
(line 2264). Only the verifier accumulates imatrix; the drafter is
observer-free (line 2276). Telemetry can be v1 (`acceptance.v1`, just
confidence) or v2 (`spec_calib.v2`, full top-k from both models, for
rejection-sampling drafter realignment).

### 1.5 The chunk strategy (where activations come from)

`compute_imatrix` (line 1676): the corpus is tokenized once, then chunked into
`n_ctx`-sized windows (default `n_ctx=512`). Each window clears KV
(`llama_memory_clear`, line 1694) and decodes `num_batches` ubatches of
`n_batch` tokens. `chunk_size = n_ctx / n_parallel` (line 386) is the
activation count per "call" used to gate periodic saves.

The batch filter is `src1->ne[1] < 16` (line 394): batches smaller than 16
tokens are NOT collected (the comment asks "why?"). This means the final
short batch of every chunk is silently dropped. Small-batch filtering is an
upstream llama.cpp heuristic to avoid noise; in tessera's fused-observer path
this filter is bypassed because the observer tensor is always emitted.

### 1.6 What the consumers actually use (the critical part)

This is the load-bearing finding for section 2's quality analysis. The
consumers were read in full:

**`tessera-imatrix.cpp::ts_imatrix_load_gguf` (line 239)**: only reads
`sums` and `counts`, computes `sums[i] / counts[i]` (mean squared activation)
-> stores as `out->data[tensor]`. **`abs_sums`, `fourth_sums`, `max_abs` are
NOT read by the GGUF path at all.** They are only consumed via
`ts_imatrix_regime` (line 303), which takes the per-channel mean-squared-act
vector and re-derives `mean_magnitude, kurtosis, eff_rank, p99` FROM THAT
VECTOR ALONE - by treating `act_data[i]` as if it were a sample, computing
`sum(|v|), sum(v^2), sum(v^4)`. So the consumer reconstructs the 4th moment
of the per-channel mean-squared-activation distribution, not the 4th moment
of the original activations. The on-disk `fourth_sums`/`max_abs` are dead
weight for the standard (non-multi-modal) path.

**`tessera-mm-imatrix.cpp::ts_mm_imatrix_load` (line 232)**: same - only the
per-modality `act` vector (mean squared activation) survives; `mm_regime`
(line 176) re-derives stats from it.

**Where the regime stats go**: `ts_regime_compute_descriptor`
(`tessera-regime.cpp:302`) fills `kurtosis, eff_rank, mean_magnitude, p99`,
then `ts_regime_classify` (line 71) thresholds them to route each tensor to
one of 6 quantization experts:

| Threshold | Routes to | Reason |
|-----------|-----------|--------|
| `kurt > 10` in `down` family | DartQuant | rotation handles massive outliers |
| `kurt > 10` | DartQuant | distribution-aware rotation |
| `kurt > 5` | CHAMP-Q | channel permutation smooths heavy tails |
| `er < 0.15` | FLRQ | factored low-rank residual |
| `er < 0.3` | LRQ | low-rank residual |
| `er > 0.7 && kurt < 3` | AWQ | well-conditioned, plain diagonal |
| attn_k / attn_v | AWQ | "well-behaved" |
| default | AWQ | confidence 0.5 |

So the ENTIRE routing decision rests on two scalars per tensor (kurtosis and
eff_rank), each derived from the per-channel mean-squared-activation vector.
The richer per-tensor statistics computed by the producer's `compute_statistics`
(line 257: total_sqract, stddev, active, entropy, zd, cossim) are display-only
(`--show-statistics`) and NEVER feed a quantization decision.

### 1.7 The downstream quantization pipeline (consumers in context)

The dispatch (`tessera-dispatch.cpp`) calls imatrix data in two places, both
narrowly scoped:

- `ts_imatrix_lookup` / `ts_mm_imatrix_act_scales` (line 154, 208) -> the AWQ
  expert consumes per-channel act_scales as the salience signal for the alpha
  search. This is the classic AWQ mechanism: scale salient channels up before
  quantization, undo after.
- `ts_regime_compute_descriptor` (line 1566, 2047) -> the regime classifier
  uses (kurtosis, eff_rank) to pick the expert AND to tune the expert profile
  (`ts_expert_default_profile`: alpha_scale, clip_scale, awq_grid,
  max_outliers).

This means imatrix quality directly controls (a) AWQ's per-channel salience
ranking and (b) which of 6 quantizers runs per tensor. Both are first-order
drivers of final model quality.

---

## 2. Quality opportunities (part 1)

Each recommendation is grounded in a cited paper, demonstrated peer practice,
or a specific code fact from section 1. Ranked by expected quality gain per
unit implementation risk, highest first.

### Q1. Collect per-channel activation MAGNITUDE histograms, not just moments

**Today**: 4 moments + max per channel. The consumers reconstruct distribution
shape (kurtosis) from these moments of the mean-squared-act vector - a lossy
summary of a summary.

**Opportunity**: keep a small per-channel histogram (e.g. 16-32 log-spaced
bins on |x|, ~64-128 KB per typical tensor). This gives the consumer the
ACTUAL activation distribution, not a moment-derived proxy. The downstream
gain is twofold: (a) the regime classifier can use real distributional
features (true kurtosis, true tail mass above threshold) instead of a moment
reconstruction; (b) AWQ can use a true outlier-fraction per channel, which is
exactly what AWQ's salience metric approximates with `act_scales`.

**Quality gain expected**: medium. AWQ and the routing thresholds are the two
biggest quality levers; both get more accurate distributional inputs.

**Compute / memory cost**: per-token, incrementing 16-32 bins per channel is
~2-4x the cost of the current 4-moment accumulation; total imatrix RAM grows
~4-8x (still small - 100s of MB for an 8B model, not GB). The fused observer
tensor grows from `[4*C+1]` to `[4*C + H*C + 1]` (H = bins), which may need
observer-op reshaping.

**Risk**: medium - changes the on-disk contract unless added as an optional
sidecar (the producer already supports optional fields, see
`load_optional` in `imatrix-loader.cpp:171`).

**Grounding**: AWQ (Lin et al. 2023, arXiv:2306.00978) explicitly identifies
"salient channels" by activation magnitude and fraction; the histogram
captures both exactly. EXL2 uses a per-row importance histogram to allocate
bits across bits-and-weight (Maxime Labonne, ExLlamaV2 overview).

### Q2. Multi-scale (variable-length) calibration corpus

**Today**: every chunk is `n_ctx` tokens (default 512). Activation statistics
are collected only at one sequence length.

**Opportunity**: the MaCa paper (Matryoshka Calibration, arXiv:2602.07465)
shows that "Hessian estimates derived from fixed-length calibration may fail
to represent the true importance of weights across diverse input scenarios"
because input length alters both activation distributions AND the weight
importance captured by any moment-based proxy. MaCa mixes multiple sequence
lengths in the calibration set and regularizes each sequence as an
independent sample, reporting consistent accuracy improvements under low-bit
quantization on Qwen3, Gemma3, LLaMA3 at "lightweight enhancement"
cost. The producer could mix window sizes (e.g. 128, 512, 2048) in one run,
weighting each chunk's contribution to the running moments.

**Quality gain expected**: medium-high at low bit widths (sub-4-bit). This is
exactly the regime tessera targets (the regime classifier exists to make
2-4 bit viable).

**Compute cost**: ~1.2-1.5x for a mixed-length run (more windows total, but
smaller ones are cheaper; some recomputation when context length changes if
RoPE scaling differs).

**Risk**: low-medium. The collection loop already supports variable `n_ctx`;
the work is corpus curation + ensuring the moments are correctly weighted
across lengths.

**Concrete change**: a `--calib-mix "128:0.25,512:0.5,2048:0.25"` flag that
selects window size per chunk by sampling the distribution. Counts already
weight the accumulation, so the only invariant to preserve is "counts reflect
tokens seen at each scale".

### Q3. Per-channel max-abs is collected but UNUSED by the consumers - wire it in

**Today**: the producer carefully tracks `max_values` (running max of |x| per
channel) through the fused observer. The GGUF consumer path
(`ts_imatrix_load_gguf`) loads only `sums/counts` and ignores `max_abs`. The
regime classifier never sees per-channel max.

**Opportunity**: per-channel max-abs is the cleanest signal for outlier-aware
quantization. DartQuant (rotation) and CHAMP-Q (permutation) are triggered by
`kurt > 5/10` thresholds - but kurtosis computed from per-channel
mean-squared-act is a global scalar, missing WHICH channels carry the
outliers. Routing a tensor to "rotation expert" without knowing the outlier
channels forces the rotation to be global. Per-channel max gives the expert
the channel-localized outlier information it actually needs.

**Quality gain expected**: high for the rotation/permutation experts
specifically (the ones triggered by heavy tails). This is the cheapest win on
the list - the data is already collected and stored, it just needs a
consumer.

**Compute cost**: zero at collection (already done); tiny at consumer
(`max_abs` vector is `nval` floats, one pass to find the top-k outlier
channels).

**Risk**: low. Add a `max_abs` lookup to `ts_imatrix_lookup` or a new
`ts_imatrix_outlier_channels` accessor; thread it into
`ts_regime_compute_descriptor` and `ts_expert_default_profile`.

**Grounding**: this is a strict improvement to information the producer
already pays to collect. The fact that the consumer discards it is a
discovered bug, not a design choice.

### Q4. Hessian / Fisher diagonal for AWQ salience (the GPTQ-style signal)

**Today**: AWQ uses `sums/counts` (mean squared activation) as the per-channel
salience. This is a first-moment-of-squared-activation proxy for "how much
this channel matters to the loss".

**Opportunity**: GPTQ derives an actual Hessian-diagonal proxy
`H = 2 * X^T X` (where X is the calibration activation matrix), which is the
quantity the OBSERVER already accumulates element-wise into `values`. The
diagonal of `X^T X` over the calibration corpus IS the per-channel sum of
x^2, which IS `e.values`. So tessera is *already* computing the Hessian
diagonal - it is just averaging it (dividing by counts) before handing it to
AWQ. The Fisher information matrix (for a softmax cross-entropy loss) is a
different quantity: `F = E[grad grad^T]`, which needs the gradient of the loss
w.r.t. the input to the matmul - i.e. the backpropagated error, not just the
forward activation.

**Quality gain expected**: low for the Hessian (already implicit);
potentially high for the FISHER, because Fisher weights channels by how much
they affect the output distribution, not just by how active they are. A
high-activation channel that the model rarely uses to discriminate among
outputs should NOT get a high quantization budget; Fisher captures that,
squared-activation does not.

**Compute / memory cost**: Hessian = free (already there). Fisher = expensive:
one backward pass per chunk to get input gradients, plus accumulating their
outer products. Roughly doubles the per-chunk cost (forward + backward vs
forward-only). Requires ggml-opt backprop through the trunk - non-trivial
plumbing.

**Risk**: high for Fisher (new backward path, calibration corpus must support
a real target for the loss; the producer currently does pure inference with
no labels). Low for Hessian (just stop dividing by counts in the consumer).

**Grounding**: GPTQ (Frantar et al. 2022) uses the Hessian for column-wise
importance; AWQ (Lin et al. 2023) uses the simpler act_scales and shows it is
within noise of GPTQ at 4-bit, which is WHY tessera chose the simpler path.
Fisher-style calibration is the SOTA frontier for sub-3-bit; "First-Order
Error Matters" (AAAI 2026, ojs.aaai.org/index.php/AAAI/article/view/40123)
refines the Hessian compensation.

**Recommendation**: ship Q3 (wire in max_abs) first; treat Q4 Fisher as a
phase-4 research item gated on a sub-3-bit quality target.

### Q5. Corpus selection / diversity

**Today**: the corpus is one text file (`-f`), tokenized, chunked linearly.
No diversity selection, no domain balancing, no perplexity filtering. The
`--in-file` mechanism lets you accumulate across multiple imatrix runs, so
multi-corpus combination is possible but manual.

**Opportunity**: a large body of 2024 work (LLM-QAT, EasyQuant, the Red Hat
half-million-evaluation study) shows calibration corpus diversity matters
more than corpus size for downstream quantized accuracy. Concretely:
(a) sample chunks across documents rather than linearly (avoid running one
long document's distribution); (b) optionally weight chunks by inverse
perplexity to favor "in-distribution" examples; (c) for multi-modal models,
ensure each modality (text/image/audio) gets adequate coverage.

**Quality gain expected**: medium, but cheap.

**Compute cost**: ~0 at collection (just chunk ordering).

**Risk**: low. The producer already supports `--in-file` accumulation; this
is a corpus-curation tooling improvement, not a producer change.

**Grounding**: Microsoft's INT4 guide (medium.com/data-science-at-microsoft,
"A Practical Guide to INT4 Quantization for SLMs") found GPTQ's accuracy
varies meaningfully with calibration set choice; Red Hat's study
(developers.redhat.com, half-million evals) corroborates.

### Q6. Other smaller items

- **Spec-mode telemetry depth**: `--telemetry-topk > 0` already emits v2
  records with full verifier + drafter distributions (line 2499). The
  acceptance-rate histogram is the right signal for drafter training. No
  quality work needed here; the producer is ahead of the consumers.
- **Neutral prior** (`--prior-weight`, commit `cb616cc56`): lets the imatrix
  be blended toward a uniform prior to avoid overfitting to one corpus. The
  right answer to "how much" is empirical and per-model; expose it as a
  per-tensor knob rather than global, OR replace it with corpus-weighting (Q5)
  which is more principled.
- **`compute_statistics` display stats** (entropy, zd, cossim): currently
  display-only. The cossim between adjacent layers' activation patterns is a
  genuinely useful signal for routing (layers with similar patterns to their
  predecessor may tolerate more aggressive quant) and could be fed into the
  regime descriptor cheaply.

### Quality summary table

| ID | Opportunity | Gain | Cost | Risk | Phase |
|----|-------------|------|------|------|-------|
| Q3 | Wire per-channel max_abs into routing/expert | high | ~0 | low | 1 |
| Q5 | Corpus diversity sampling | medium | ~0 | low | 1 |
| Q2 | Multi-scale calibration | med-high | 1.2-1.5x | low-med | 2 |
| Q1 | Per-channel histograms | medium | 2-4x | medium | 2 |
| Q6c| Cossim as a routing feature | low-medium | ~0 | low | 2 |
| Q4 | Fisher information (backward pass) | high (sub-3-bit) | 2x | high | 4 |

---

## 3. Porting optimizations (part 2)

For each of the 11 tessera quantize pipeline optimization commits, a verdict
(PORT / ADAPT / SKIP) with rationale, the specific change, and expected
benefit. The producer runs INFERENCE over a corpus (one model, many chunks);
the pipeline runs GRID-SEARCH QUANTIZATION (one model, many candidates). Some
patterns transfer directly, some do not.

The producer's current memory profile: full model resident (line 1256,
`no_alloc=false`), full tokenized corpus in RAM, two staging slots of order
`sum(observer sizes)` (~MB), and the `m_stats` accumulator (~MB per tensor).
For an 8B model: ~16 GB resident model + small overhead. No mmap, no
streaming, no Metal dispatch in the producer path.

### 3.1 Streaming weight loading (commit `02ac74294`)

**Pattern**: load one layer's weights at a time via load/release callbacks,
holding only N_THREADS layers live (~1.4 GB instead of 32 GB for 8B).

**Verdict: ADAPT (medium priority)**. The producer holds the full model
resident because inference needs any layer at any decode step (unlike the GA,
which evolves one layer at a time). But for a CALIBRATION run on a memory-
contended machine, a few tricks apply:

- The producer only decodes the FORWARD pass; if the backend supports tensor
  eviction (ggml's mmap path), the OS will page in active layers and evict
  cold ones under pressure. This is exactly what `5b566f919` (next item)
  enables.
- A more aggressive adaptation: shard the corpus by layer-affinity is not
  possible (every chunk touches every layer). So the GA's "one layer at a
  time" pattern does NOT transfer.

**Specific change**: rely on mmap (3.2) rather than porting the GA's
streaming callbacks. The producer does not have a per-layer loop to hook.

**Expected benefit**: indirect (via 3.2).

### 3.2 mmap the input GGUF (commit `5b566f919`) - PORT

**Pattern**: `no_alloc=true` + mmap, lazy paging, RSS drops from ~8 GB to
~180 MB for an 8 GB q8_0 model.

**Verdict: PORT (high priority, lowest-risk win)**. The producer's
`save_imatrix` line 1256 explicitly uses `no_alloc = false` (eager
allocation), which the PREFLIGHT flagged. This is the same flip the pipeline
made. The model GGUF is read-only; nothing in the producer mutates weights.

**Specific change**: in the producer's model-load path (which uses
`common_init_from_params`), set the mmap path. This is a one-flag change
exposed by `common`'s loader; the producer already inherits it. The line
1256 `no_alloc=false` is for the OUTPUT imatrix context (a tiny ~MB context
to hold the outgoing stats), not the model - re-verify which context this
is. (Reading the code: line 1256 is in `save_imatrix`, allocating the
outgoing stats context; it is NOT the model load. So the "no_alloc=false"
evidence in PREFLIGHT is real but is about save, not load. The model load
goes through `common_init_from_params` and already respects the standard
mmap flag - so this may already be free if the user passes `--mlock off` or
the platform default. Verify before claiming the win.)

**Expected benefit**: 8 GB -> ~180 MB resident for an 8B q8_0 model on macOS,
assuming mmap is not already enabled by default. This is the single biggest
memory win on the list and the cheapest to ship.

**Correction to PREFLIGHT**: the `no_alloc=false` at line 1256 is in
`save_imatrix` for the OUTGOING stats context (size `data_size`, line 1254),
not the model. The outgoing stats context is small (MB-scale). The real
question is whether `common_init_from_params` mmap's the model; this needs
runtime verification, not a source-read claim.

### 3.3 Parallel candidate eval: serial layers, shared weights (commit `0449cfdbe`)

**Pattern**: many threads evaluate different candidates against ONE shared
read-only weight buffer.

**Verdict: SKIP (does not transfer)**. The producer has only one "candidate"
(the model). It already uses `n_parallel` sequences in parallel
(`n_seq = n_batch / n_ctx`, line 1660), which is the inference-native
analogue: many sequences sharing one set of weights. There is no per-thread
fan-out to port.

**Caveat**: if Q1 (histograms) or Q4 (Fisher) make per-channel reduction
expensive, the async two-slot reduction already in the producer (line 188)
is the right place to extend, not a port of the GA's parallel-candidate
pattern.

### 3.4 Metal GPU acceleration (commit `18f871ef1`)

**Pattern**: three Metal kernels (scale_clip_ternarize, dequant_mse_recon,
awq_grid_batch) plus GPU-resident weight buffers uploaded once per layer.

**Verdict: PORT (high priority) but via a different mechanism**. The producer
does not have scale/clip/ternarize ops (those are quantization primitives).
But it DOES run inference, and the inference backend already has Metal
support (`llama_decode` dispatches to Metal when configured). The producer
gets Metal acceleration for free if the user runs with the Metal backend
enabled - which is the default on Apple Silicon.

**What is NOT free**: the producer's CPU-side reduction (`reduce_graph_observers`,
line 706) runs on CPU. If the observer tensor is large (Q1 histograms), this
becomes a bottleneck. The port is: add a Metal reduce kernel that fuses the
4-moment + histogram accumulation into the observer-op itself, so the GPU
hands back already-reduced stats. The pipeline's "GPU-resident buffers, small
scalars back to CPU" principle (commit message) applies directly.

**Specific change**: extend the existing imatrix observer ggml op (the one
that produces the `[4*C+1]` compact tensor) to reduce on-GPU; CPU only reads
the final per-tensor accumulators.

**Expected benefit**: 2-5x on the decode path for large models on Apple
Silicon (matching the pipeline's reported Metal speedups); near-elimination
of the CPU reduce bottleneck if Q1 lands.

### 3.5 vDSP fusion + sharded eval_ctx + parallel screening (commit `97f757843`)

**Pattern**: vDSP (`ts_mat_scale_cols`, `ts_vec_maxabs`, `ts_vec_meanabs`)
for hot elementwise loops; sharded concurrent map (`ts_sharded_map`) instead
of a global mutex; parallel screening / acceptance loops via atomic work
queues.

**Verdict: ADAPT (medium priority)**. Three sub-parts:

- **vDSP for stat accumulation**: the producer's reduction loop (line 764-779)
  is exactly the elementwise pattern vDSP accelerates. On macOS/arm64,
  `vDSP_svesq` (sum of squares), `vDSP_maxmgv` (max abs), and friends replace
  the inner `for channel` loop. This is a clean PORT and pays for itself the
  moment Q1 (histograms) makes reduction more work. The producer already uses
  `std::async` for the reduction (line 672); adding vDSP inside the async
  task is local.
- **Sharded map**: the producer's `m_stats` is guarded by ONE mutex
  (`m_mutex`, line 179) and the reduction task takes it for the merge
  (line 789). For models with many tensors (MoE), this becomes contended.
  A sharded map keyed by `(scope, name)` hash would let reductions from
  different backends proceed in parallel. PORT - the pattern transfers
  directly.
- **Parallel screening**: SKIP - no analogue in the producer.

**Specific change**: add `vDSP_svesq`/`vDSP_maxmgv` calls inside
`reduce_graph_observers`; replace `m_stats` with a small sharded map (8-16
shards by name hash).

**Expected benefit**: 1.3-2x on the reduction phase for large MoE models;
near-zero benefit for small dense models. The sharded map matters more once
histograms (Q1) inflate per-tensor state.

### 3.6 FUSE C + cache-blocked dequant + AWQ grid batch (commit `45eeab7b2`)

**Pattern**: fused scale/clip/ternarize + cache-blocked dequant + 20-alpha
AWQ grid search in one pass.

**Verdict: SKIP**. All these ops are quantization-specific (scale/clip/ternarize
have no meaning during inference). The producer's "fusion" opportunity is the
observer-op fusion (already done - the `[4*C+1]` compact tensor IS the fused
observer). Nothing to port.

### 3.7 BLAS-accelerate B7 optimizer matmuls (commit `ccf9fa803`)

**Verdict: SKIP**. The B7 optimizer is a GA-internal primitive. The producer
has no optimizer.

### 3.8 BLAS-accelerate GA fitness matmul + per-tensor thread pool (commit `7c6d85681`)

**Pattern**: BLAS for the fitness matmul; per-tensor thread pool to avoid
cross-tensor contention.

**Verdict: ADAPT (low priority)**. The producer has no fitness matmul. But
the "per-tensor thread pool" idea is relevant if the producer parallelizes
the per-tensor reduction across many tensors - which is exactly what the
sharded map (3.5) gives. Treat this as already covered by 3.5.

### 3.9 Streaming MSE fitness, 132 KB vs 700 MB (commit `770bddee4`) - PORT

**Pattern**: do not materialize the full per-candidate workspace; stream
row-by-row through the reduction, keeping O(in_dim) scratch. 132 KB instead
of 700 MB per worker.

**Verdict: PORT (high priority, the single most applicable pattern)**. This
is the "spatial memory optimization" pattern, and it applies directly to the
producer's reduction:

- The producer currently copies the observer tensor into a staging arena
  (`m_observer_staging[slot]`, line 593, resized to `m_observer_offsets.back()`
  per batch). For a large MoE model with many experts, this arena can be
  large.
- The streaming-MSE insight: process the observer tensor row-by-row (per
  expert, per channel block), accumulating into the global `m_stats` directly,
  without the intermediate staging arena. The arena is replaced by a
  per-thread `~channels` scratch buffer.

**Specific change**: refactor `collect_graph_observers` (line 565) and
`reduce_graph_observers` (line 706) to do the per-row accumulation inline
during the tensor-get, rather than staging then reducing. Keep the two-slot
async structure (it overlaps GPU and CPU); just shrink each slot from
"sum of all observer sizes" to "max single observer size".

**Expected benefit**: the staging arena shrinks from O(total observer
payload per batch) to O(largest single observer). For MoE models with
hundreds of experts, this is the difference between MB and GB. Same pattern,
same win as the pipeline's 700 MB -> 132 KB.

**Note**: this also makes Q1 (histograms) cheap to ship, because the
histogram state lives in `m_stats` (persistent), not in the staging arena.

### 3.10 Streaming MSE for acceptance gate (commit `18d3aaaf1`, `cf0c49fbf`)

**Verdict: SKIP (covered by 3.9)**. Same pattern, applied to more call sites
in the pipeline. The producer has only one reduction site, so 3.9 covers it.

### 3.11 DuckDB persistent pipeline store (commit `d0ba47b49`)

**Pattern**: persistent, queryable store with cross-run warm-start, crash
resumability, analytical queries.

**Verdict: ADAPT (medium priority, deferred)**. The producer ALREADY has a
form of this: the `.transfer.json` sidecar (line 1333) and the GGUF imatrix
itself support resume across runs. What it lacks:

- **Cross-run warm-start keyed by model hash**: the pipeline's
  `SELECT best_alpha FROM ga_results WHERE family=?` is the pattern; for
  the producer, it would be "load the prior imatrix for this model and
  continue accumulating". This is what `--in-file` does manually; DuckDB
  would make it automatic and queryable.
- **Crash resumability**: the producer writes periodic checkpoints
  (`save_imatrix(n_chunk)`, line 1814), so it is already crash-resilient for
  the data. The transfer-ledger resume (line 1387) is the mechanism.
- **Analytical queries**: this is where DuckDB adds real value - querying
  per-tensor convergence history, comparing two calibration runs, etc. But
  it is a tooling improvement, not a producer improvement.

**Specific change**: optional, low priority. Add DuckDB-backed imatrix
metadata if/when the Studio UI needs to query calibration history. The
producer's existing resume is adequate for the core loop.

**Expected benefit**: tooling / observability, not speed or memory.

### 3.12 Porting summary

| # | Commit | Verdict | Priority | Benefit |
|---|--------|---------|----------|---------|
| 3.2 | mmap GGUF | PORT (verify first) | high | 8 GB -> 180 MB RSS |
| 3.9 | streaming MSE -> streaming reduction | PORT | high | MB -> KB staging; enables Q1 |
| 3.4 | Metal acceleration | PORT (via observer-op fusion) | high | 2-5x decode on Apple Silicon |
| 3.5 | vDSP fusion + sharded map | ADAPT | medium | 1.3-2x reduction on MoE |
| 3.1 | streaming weight loading | ADAPT (via 3.2) | medium | indirect |
| 3.11| DuckDB persistent store | ADAPT (deferred) | low | tooling |
| 3.3 | parallel candidate eval | SKIP | - | n/a (inference already parallel) |
| 3.6 | FUSE C + AWQ grid | SKIP | - | quantization-specific |
| 3.7 | BLAS B7 optimizer | SKIP | - | GA-specific |
| 3.8 | BLAS GA fitness | SKIP | - | GA-specific |
| 3.10| streaming acceptance | SKIP | - | covered by 3.9 |

**Tally: 3 PORT, 3 ADAPT, 5 SKIP.** The 3 PORTs (mmap, streaming reduction,
Metal-via-fusion) are the concrete wins; the 3 ADAPTs (vDSP/sharded-map,
streaming-via-mmap, DuckDB) are secondary; the 5 SKIPs are quantization-
specific or already provided by inference-native mechanisms.

---

## 4. Drafter-training feed (part 3)

### 4.1 How the `--features-out` flow works today

`compute_features` (`imatrix.cpp:1873`) is the offline trunk-feature capture
path, documented as "Path 1" in `docs/tessera-dflash-training-design.md`
(section 4). The flow:

1. User invokes `llama-imatrix -m trunk.gguf -f calib.txt --features-out P
   --feature-layers 0,15,31 [--features-warmup 256]`. Mutually exclusive with
   `--model-draft` (asserted line 2905).
2. The drafter's `target_layer_ids` are passed as `--feature-layers` in
   concatenation order (e.g. `0,15,31` for a 32-layer trunk with the drafter
   consuming layers 0, 15, 31).
3. For each target layer, `llama_set_embeddings_layer_inp(ctx, lid, true)`
   (line 1920) enables the existing runtime tap for per-layer input hidden
   states.
4. The plain windowed forward runs (same structure as `compute_imatrix`,
   KV-cleared per window). For each ubatch, `llama_get_embeddings_layer_inp`
   (line 1979) returns the layer's input hidden states, token-major.
5. Rows are streamed token-major to `<prefix>.bin` (raw f32,
   `[n_tokens, n_layers * n_embd]`, layers concatenated in
   `--feature-layers` order). Header in `<prefix>.json`, schema
   `llama.tessera.features.v1`.

### 4.2 The overlap-window trick (already shipped)

A subtlety worth recording because it is the producer's best trick: windows
advance by `stride = n_ctx - warmup` (NOT `n_ctx`), so consecutive windows
overlap by exactly `warmup` tokens (line 1905). Each window re-decodes the
previous window's tail to prime its KV; only positions `[warmup, n_ctx)` are
emitted. The emitted rows form ONE contiguous corpus sequence
(`row r == corpus token warmup + r`), and every emitted token sees
`>= warmup` genuine left-context tokens spanning the window boundary.

Cost: `n_ctx / (n_ctx - warmup)` decode overhead (~1.07x at `warmup=256,
n_ctx=4096`). This recovers the warmup-prefix tokens the naive layout would
discard, and matches the steady-state inference regime (the trunk also
conditions on a finite KV window at inference). This is exactly how
EAGLE-style feature capture is done (cited in the source comment, line 1867).

### 4.3 What features are captured, and how training consumes them

Per token, the drafter encoder consumes `n_target_layers * n_embd` floats
(`dflash.cpp:15`, confirmed in `tessera-dflash-training-design.md` section 1).
Example: 3 target layers x 4096 = 12288 floats ~= 48 KB/token at F32. The
drafter is EAGLE-style: the encoder fuses trunk hidden states through an FC
layer (`dflash.cpp:138-153`), the decoder borrows `tok_embd` and `output`
from the trunk via `ctx_other`, and runs block-diffusion over
`[anchor, MASK x B]` to draft B tokens (`dflash.cpp:243-435`).

The training driver (section 5 of the design doc, NOT YET WRITTEN - Stages
1-4 are pending) will:

1. Load the feature sidecar into a `ggml_opt_dataset` alongside the block
   dataset (`llama.tessera.dflash-block.v1`).
2. Run the combined encoder + decoder forward on cached features (no trunk
   forward at train time - this is what makes idle-time training viable on
   a 16 GB machine).
3. Apply weighted CE (D-PACE) on the B drafted positions.

The producer's role is DONE for this flow: the feature sidecar is the
contract. Improvements split into "make the sidecar better" (producer-side)
and "consume it better" (training-driver-side, out of scope for this study).

### 4.4 What the multi-drafter work specifically needs

Three concrete improvements, ranked by leverage:

#### D1. F16 / Q8_0 feature quantization (storage is the dominant cost)

**Today**: F32 capture is exact but is ~48 KB/token. For a 100M-token
calibration corpus, that is ~4.8 TB. The header reserves a `dtype` field
(line comment at `tessera-features.h`) but F16/Q8_0 are "not yet
implemented".

**Why it matters for multi-drafter**: the design doc (section 0c) flags
"F16 halves it; Q8_0-style quantization is the likely production choice. This
is the dominant storage cost." Risk #1 in the design doc is literally "Feature
storage size / quantization fidelity."

**Opportunity**: ship F16 first (bit-exact-enough for hidden states, halves
storage, zero conversion risk). Q8_0 second (8x storage reduction, small
accuracy hit that the drafter FC layer can absorb - EAGLE-3 uses quantized
features). The writer already reserves the dtype; the work is in the write
and read paths.

**Risk**: low. F16 is a strict format add. Q8_0 needs a calibration pass on
the drafter to confirm acceptance rate is preserved (the design doc's Stage 2
A/B harness exists for exactly this).

**Grounding**: EAGLE-3 (per the llama.cpp discussion #15902 and the AAAI
"Steering Pretrained Drafters" paper) uses multi-level hidden state features
with quantization; the drafter head is robust to feature compression because
it learns to read whatever it gets.

#### D2. Multi-level feature layers (EAGLE-3 pattern)

**Today**: `--feature-layers` is a free-form CSV, but the typical invocation
is `0,15,31` (first, middle, last) - already the EAGLE-3 pattern. The
producer does not validate or recommend a layer set.

**Opportunity**: EAGLE-3 explicitly uses "low-, middle-, and high-level
features" (after the first, middle, and last decoding layers). The
`--feature-layers 0,N/2,N-1` pattern is the empirically-validated set for
EAGLE-3-style drafters. The producer could:

- Validate the requested set is non-empty and in range (already done, line
  1885).
- Offer a `--feature-layers eagle3` shorthand that auto-selects first /
  middle / last for the loaded model.
- Document the layer-ablation recipe (try `0,N-1`, then `0,N/2,N-1`, measure
  acceptance rate).

**Why it matters for multi-drafter**: the multi-drafter work will need to
tune `target_layer_ids` per drafter architecture. The producer is the supply
side; making it easy to experiment with layer sets accelerates the
drafter-design loop.

**Risk**: very low (mostly documentation / a CLI shorthand).

#### D3. Spec-mode feature capture (combine the two paths)

**Today**: `--features-out` and `--model-draft` are mutually exclusive (line
2905). You can capture features over plain text OR run the spec-decoding
calibration loop, but not both.

**Opportunity**: the multi-drafter work will eventually want drafter training
data that reflects the DRAFTING distribution, not the plain-text
distribution. EAGLE trains on the trunk's hidden states over real text; the
DFlash block-diffusion drafter may benefit from features captured during
spec-decoding (where the trunk sees the drafter's outputs). This requires
fusing `compute_imatrix_spec` and `compute_features` into one loop.

**Why it matters for multi-drafter**: the design doc section 7.2 ("Teacher =
offline verifier traces") frames DFlash training as distillation from the
trunk-as-teacher. The teacher signal is currently captured as either features
(trunk forward over text) OR spec telemetry (verifier argmax / topk). Fusing
the two gives the trainer BOTH signals aligned to the same tokens.

**Risk**: high. The spec loop is already intricate (per-prefix forwards, KV
rollback, telemetry JSONL). Interleaving feature capture into it is the
"fragile" combination the design doc warns against (line 1846: "the
speculative telemetry loop interleaves drafter forwards, KV trimming, and
output_reorder in ways that make hidden-state readback fragile"). This is a
phase-4 item, not a phase-1 item.

**Grounding**: P-EAGLE (arXiv:2602.01469) trains the drafter on the trunk's
hidden states during parallel drafting; the feature capture during spec
decoding is the on-device analogue.

### 4.5 Drafter-feed summary

| ID | Improvement | Producer work | Driver work | Risk | Phase |
|----|-------------|---------------|-------------|------|-------|
| D1 | F16 / Q8_0 feature dtype | writer + reader | loader | low | 1 |
| D2 | `--feature-layers eagle3` shorthand + docs | CLI | none | very low | 1 |
| D3 | Spec-mode feature capture | fuse two loops | consume both | high | 4 |

The producer is ahead of the consumers here. The feature-capture flow is
already correct (overlap windows, contiguous rows, bit-exact verified). The
multi-drafter work's bottleneck is the TRAINING DRIVER (Stages 1-4 of the
design doc, not yet written), not the feature supply. D1 is the one
producer-side improvement that unblocks the driver at corpus scale.

---

## 5. Recommended wave plan

Five phases, each a shippable milestone, ordered from "quickest porting wins"
to "ambitious quality improvements". This is the bridge to a future
implementation wave.

### Phase 1 - Verify the cheap wins (1-2 days)

- **3.2 mmap verify**: confirm whether `common_init_from_params` mmap's the
  model by default on macOS. If not, enable it. (Source-read could not
  confirm; needs runtime RSS measurement.) Expected: 8 GB -> ~180 MB for an
  8B model, or "already on, no change".
- **Q3 wire max_abs**: add `max_abs` accessor to `tessera-imatrix.cpp`; thread
  per-channel max into `ts_regime_compute_descriptor` and the
  DartQuant/CHAMP-Q expert profiles. The data is already on disk; this is a
  consumer-only change. Expected: better routing accuracy for heavy-tailed
  tensors.
- **Q5 corpus sampling**: add a `--calib-sample stride=N` flag that samples
  every Nth chunk rather than running linearly, plus a multi-file corpus
  mode that interleaves documents. Expected: more diverse calibration, small
  accuracy lift.
- **D2 eagle3 shorthand**: trivial CLI add, unblocks drafter-design
  experiments.

Shippable as one PR; each sub-item independent.

### Phase 2 - The big producer ports (3-5 days)

- **3.9 streaming reduction**: refactor `collect_graph_observers` /
  `reduce_graph_observers` to do per-row accumulation during tensor-get,
  eliminating the staging arena. Preserves the two-slot async structure.
  Expected: MB -> KB staging; same bit-exact output.
- **3.4 Metal observer-op fusion**: extend the imatrix observer ggml op to
  reduce on-GPU; CPU reads only final per-tensor stats. Fall back to the
  current CPU path on any dispatch failure (matching the pipeline's pattern).
  Expected: 2-5x decode on Apple Silicon for large models.
- **3.5 vDSP + sharded map**: add `vDSP_svesq`/`vDSP_maxmgv` to the
  reduction; shard `m_stats` by name hash. Expected: 1.3-2x reduction on MoE.

Each is independently shippable; together they are the "producer parity with
the pipeline" milestone.

### Phase 3 - Quality data (1-2 weeks)

- **Q2 multi-scale calibration**: `--calib-mix "128:0.25,512:0.5,2048:0.25"`
  flag; weight moments correctly across scales. Grounded in MaCa. Expected:
  accuracy lift at sub-4-bit.
- **Q1 per-channel histograms**: extend the observer op to emit a 16-32 bin
  log-spaced histogram per channel; add a histogram sidecar to the imatrix
  format. Depends on 3.9 (streaming reduction) and 3.4 (Metal fusion) to be
  cheap. Expected: better distributional inputs to AWQ and the regime
  classifier.
- **Q6c cossim routing feature**: feed the existing adjacent-layer cossim
  into the regime descriptor. Expected: small routing accuracy lift.

### Phase 4 - Drafter-feed scale-out (parallel to Phase 3)

- **D1 F16 / Q8_0 feature dtype**: implement the reserved dtype field; add a
  conversion pass and a drafter-side loader. Unblocks multi-drafter training
  at corpus scale. Expected: 2x (F16) or 8x (Q8_0) storage reduction.

### Phase 5 - Research (gated, speculative)

- **Q4 Fisher information**: backward pass through the trunk to accumulate
  per-channel input gradients. Gated on a sub-3-bit quality target. High
  risk, high reward.
- **D3 spec-mode feature capture**: fuse `compute_imatrix_spec` and
  `compute_features`. Gated on the multi-drafter training driver landing
  first. High risk.
- **3.11 DuckDB persistent store**: optional, only if Studio UI needs
  calibration-history queries.

---

## 6. Risks and open questions

Honest list, not filtered.

1. **The mmap claim needs runtime verification.** Source-read could not
   confirm whether `common_init_from_params` mmap's the model by default.
   The `no_alloc=false` at line 1256 is in `save_imatrix` for the OUTGOING
   stats context (small), NOT the model load. This corrects the PREFLIGHT
   headline. The mmap win may be "already free" or "one flag", but it must
   be measured, not assumed. (Correction logged; the rest of the study does
   not depend on this.)

2. **Q1 (histograms) changes the on-disk contract.** Adding a histogram
   sidecar is backward-compatible if done as an optional field (the loader
   already supports optional fields, `imatrix-loader.cpp:171`), but the
   consumer must opt in. Risk of bloating the imatrix file for users who do
   not need histograms.

3. **Q4 (Fisher) is the highest-risk item.** Backward pass through the trunk
   requires ggml-opt backprop plumbing, a labeled calibration target (the
   producer currently does pure inference), and roughly doubles compute. The
   quality payoff is speculative at 4-bit (where AWQ's act_scales are within
   noise of GPTQ's Hessian, per the AWQ paper) and only clearly wins at
   sub-3-bit. Gate on a concrete sub-3-bit target.

4. **The streaming reduction (3.9) must remain bit-exact.** The producer's
   existing async two-slot reduction has a non-trivial invariant
   (`static_assert(k_observer_slots == ...)`, line 572, with a comment about
   a past bug with three staging vectors). Any refactor must preserve the
   "staging and completion ownership one-to-one" property. Test with the
   existing `compute_statistics` display path which would surface any
   numerical drift.

5. **Metal observer-op fusion is the riskiest of the PORTs.** The pipeline's
   Metal kernels took several iterations to get bit-exact (commit messages
   cite "kernel 1 bit-exact, kernel 2 within 3e-6 relative, kernel 3 argmin
   within 0.5%"). The producer's reduction is simpler (no grid search), but
   the same "fall back to CPU on any failure" discipline is required.

6. **Multi-scale calibration (Q2) needs corpus curation, not just producer
   changes.** The MaCa paper's benefit comes from DIVERSE sequence lengths
   in the calibration set. A flag that mixes window sizes on a homogeneous
   corpus may not get the full benefit. This couples to Q5 (corpus
   diversity).

7. **The downstream consumers may not be ready for richer data.** Wiring
   `max_abs` (Q3) into the regime descriptor is only useful if the experts
   actually use per-channel outlier information. DartQuant's rotation is
   currently global; making it channel-aware is a separate piece of work in
   the expert itself. The producer can supply the data, but the consumer
   must be ready to use it.

8. **Drafter-feed work depends on the unwritten training driver.** D1 (F16
   features) is producer-side and shippable standalone. D3 (spec-mode
   features) is gated on the training driver (Stages 1-4 of the design doc)
   landing first, because there is no point capturing spec-mode features
   until something consumes them.

9. **Compute budget is contended.** The PREFLIGHT notes 3 other agents
   running (MoE quantize, UX study, wave-6 multi-drafter). Any producer
   change that requires running a full calibration to validate (e.g. Q1, Q2)
   should be validated on a small model first (the design doc used
   stories260K tiny llama, n_embd=64).

---

## 7. Sources

Markdown links, inline citations above.

### Calibration / quantization methods
- AWQ: Protecting Salient Weights for Efficient LLM Inference (Lin et al. 2023) - https://arxiv.org/pdf/2306.00978
- A Practical Guide to INT4 Quantization for SLMs (Microsoft) - https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d
- AWQ activation-aware quantization explainer - https://mbrenndoerfer.com/writing/awq-activation-aware-weight-quantization-llm
- Accelerating LLM Inference with AWQ and GPTQ (AWS) - https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/
- ExLlamaV2 overview (Maxime Labonne) - https://maximelabonne.substack.com/p/exllamav2-the-fastest-library-to-run-llms-32aeda294d26
- GPTQ mechanics (APXML course) - https://apxml.com/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/gptq-mechanics
- GPTQ-based quantization (EmergentMind) - https://www.emergentmind.com/topics/gptq-based-quantization
- A Visual Guide to Quantization (Maarten Grootendorst) - https://maartengrootendorst.com/blog/quantization/
- First-Order Error Matters, AAAI 2026 - https://ojs.aaai.org/index.php/AAAI/article/view/40123

### Calibration data / multi-scale
- MaCa: On the Importance of a Multi-Scale Calibration - https://arxiv.org/abs/2602.07465
- A Comprehensive Study on Quantization Techniques for LLMs - https://arxiv.org/html/2411.02530v1
- EasyQuant: Efficient Data-free Quantization for LLMs - https://openreview.net/forum?id=RWJYEeaW1d
- LLM-QAT: Data-Free Quantization Aware Training - https://www.researchgate.net/publication/384207258_LLM-QAT_Data-Free_Quantization_Aware_Training_for_Large_Language_Models
- We Ran Over Half a Million Evaluations on Quantized LLMs (Red Hat) - https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms
- Awesome-LLM-Quantization - https://github.com/pprp/awesome-llm-quantization

### Drafter feature capture (speculative decoding)
- P-EAGLE: Parallel-Drafting EAGLE with Scalable Training - https://arxiv.org/html/2602.01469v1
- P-EAGLE blog (vLLM) - https://vllm.ai/blog/2026-03-13-p-eagle
- Introduction to Speculative Decoding (NVIDIA) - https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/
- EAGLE-3 in llama.cpp (discussion #15902) - https://github.com/ggml-org/llama.cpp/discussions/15902
- Steering Pretrained Drafters During Speculative Decoding (AAAI) - https://ojs.aaai.org/index.php/AAAI/article/view/40255/44216
- tessera DFlash training design doc (in-tree) - `docs/tessera-dflash-training-design.md`

### In-tree source (read for this study)
- Producer: `tools/imatrix/imatrix.cpp`
- Data shape: `common/imatrix-loader.{h,cpp}`
- Consumers: `tools/quantize/tessera/tessera-imatrix.cpp`, `tessera-mm-imatrix.cpp`
- Regime router: `tools/quantize/tessera/tessera-regime.{h,cpp}`
- Dispatch: `tools/quantize/tessera/tessera-dispatch.cpp`
- Drafter design: `docs/tessera-dflash-training-design.md`
- Pipeline optimization commits: `02ac74294`, `5b566f919`, `0449cfdbe`,
  `18f871ef1`, `97f757843`, `45eeab7b2`, `ccf9fa803`, `7c6d85681`,
  `770bddee4`, `18d3aaaf1`, `cf0c49fbf`, `d0ba47b49`
