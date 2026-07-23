# RDNA2 tensor-parallel sharded output plan

## Goal

Reduce LM-head latency in tensor mode without changing sampler semantics.
The feature is opt-in through `GGML_TP_SHARDED_OUTPUT=1` and initially supports
validated Qwen3.5/Qwen3.6 35B/122B heads only.

## Important non-goal

This design does **not** split logits by vocabulary and does not implement a
distributed sampler. Every rank produces a full-vocabulary partial contribution;
RCCL or the existing generic all-reduce sums those contributions into the same
full mirrored logits tensor that the current sampler already consumes.

## Tensor algebra

Current mirrored path:

```text
W: [n_embd, n_vocab] mirrored
h: [n_embd, n_rows] mirrored
logits = W^T h independently on every rank
```

Proposed path for N ranks:

```text
W_r = W[embedding slice r, :]       shape [n_embd_r, n_vocab]
h_r = h[embedding slice r, :]       strided view [n_embd_r, n_rows]
p_r = W_r^T h_r                     full-vocab PARTIAL [n_vocab, n_rows]
logits = FP32_SUM_r(p_r)             full-vocab MIRRORED [n_vocab, n_rows]
```

Bias, LoRA contributions, and other non-distributive terminal operations execute
only after the partial base projection is reduced.

## Pre-load feature selection

Weight placement is fixed at model load and cannot silently return to mirrored
placement later. Enabling the feature therefore has two classes of checks:

### Pre-load predicates

A head is sharded only when all are true:

- `GGML_TP_SHARDED_OUTPUT=1` exactly;
- architecture is initially `LLM_ARCH_QWEN35` or `LLM_ARCH_QWEN35MOE`;
- tensor pointer is `model.output` or a validated
  `layer.nextn.shared_head_head` pointer;
- main output is not the same tensor as `model.tok_embd`;
- tensor is 2-D, non-empty, and its embedding axis can be divided into non-empty
  quant-block-aligned slices for the selected tensor split.

All other heads remain mirrored.

### Post-load invariants

Once a head is sharded, invalid graph layout is a clear context-creation failure,
not a silent mirrored fallback. RCCL is preferred, but the existing generic FP32
all-reduce is a correct slower fallback. Runtime logs must state which heads are
sharded and which collective provider is selected.

## Planned implementation

1. Add an internal tensor flag for **FP32-only all-reduce**. The flag is copied
   to simple tensors and causes RCCL/NCCL to bypass its generic BF16 crossover.
2. Split eligible main and Qwen shared MTP LM-head weights on axis 0 using a
   granularity of at least `lcm(quant_block_size, 128)`.
3. Mark the base LM-head `MUL_MAT` with the FP32-reduction flag.
4. Extend meta `MUL_MAT` inference only for flagged plain `GGML_OP_MUL_MAT` with
   one-segment axis-0 weight plus mirrored RHS:
   - `assume_sync=false` -> `PARTIAL`;
   - `assume_sync=true`  -> `MIRRORED`.
5. During per-rank simple-graph construction, replace the mirrored RHS with a
   synthetic rank-local descriptor:
   - `ne[0] = simple_weight->ne[0]`;
   - offset equals the cumulative preceding weight-slice width times `rhs->nb[0]`;
   - preserve original `nb[0..3]`, especially full-row `nb[1]`;
   - bounds-check the final row and all dimensions;
   - allocate descriptor metadata in the active alternating compute container.
6. Add a zero-allocation same-shape synchronization alias after every eligible
   main/MTP head. This guarantees the PARTIAL matmul is not the final subgraph,
   so CPU sampling, raw logits, and speculative target verification also trigger
   the all-reduce.
7. Keep output scale, LoRA, bias, and the existing sampler after that alias.
   Initial implementation never delays the all-reduce through these operations.
8. Preserve existing compact sampled-logit/probability/candidate API behavior.
   The sampler input is full-vocabulary mirrored logits; APIs that already return
   compact sampled data remain compact.

## Communication and allocation

- Output reductions are always FP32, including two-GPU and multi-row cases above
  the generic BF16 threshold.
- RCCL remains the preferred provider; generic butterfly is correctness fallback.
- The initial prototype supports normal small output-row counts. Before enabling
  all-logits/many-row production workloads, make generic-fallback scratch lazy so
  successful RCCL does not eagerly reserve multiple full-vocabulary buffers per
  GPU.

## MTP behavior

- Main and Qwen shared MTP heads are recognized by tensor identity.
- Draft backend sampling receives full mirrored logits after reduction.
- Speculative target verification keeps its existing CPU/raw-logit behavior; the
  terminal synchronization alias guarantees those logits are reduced.
- Unsupported/tied MTP heads remain mirrored independently of the main head.

## Numeric expectations

Splitting accumulation across ranks changes floating-point addition order. The
gate is not byte-identical stochastic output. Required checks are:

- finite full logits of identical shape and token order;
- tight max/mean logit error versus mirrored mode;
- identical argmax except classified near-ties;
- no material perplexity regression;
- no material MTP acceptance regression;
- deterministic fixed-seed behavior within each mode.

## Validation stages

1. Unit/meta tests for feature parsing, pointer-role selection, axis inference,
   terminal alias behavior, FP32 flag propagation, and rank offsets/strides.
2. Quant-block, unequal split, rotation, zero-slice, and tied-head fallback tests.
3. Matmul comparison for `n_rows = 1,2,4,32,256` on two and four ranks.
4. Main and separate/shared MTP heads; no sampler, CPU sampler, backend sampler,
   target verification, output scale, and active output-head LoRA.
5. Full raw logits plus existing compact sampled logits/probs/candidates and
   pre/post-sampling logprobs.
6. Parallel slots 1/2/4, graph reserve/replay, prompt reuse, recurrent state,
   multimodal text continuation, and repeated decode.
7. Explicit assertions that every eligible output collective remains FP32.
8. 8k/64k/128k prompt processing and long decode stability.
9. 35B and 122B A/B profiles: LM-head kernels, added collective, TG, PP, VRAM,
   acceptance, logits/perplexity, and output quality.

Existing unrelated large-K ROCm TOP_K fallback failures are tracked separately;
changed-path Qwen `k=20`/`k=256` tests must remain green.

## Performance hypothesis

The mirrored output projection accounts for roughly 14.7% of profiled MTP
kernel time. Four-way projection sharding can remove at most about three quarters
of that critical work, but adds one roughly 1 MiB FP32 reduction per projected
row. Expected net gain is approximately 3-8%; benchmark evidence decides whether
the feature is retained.