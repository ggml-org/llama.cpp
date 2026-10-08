# P12 - Non-materialised causal mask

**Kind:** port
**Depends on:** P08, which shrinks the same `n_ubatch` dimension

## Purpose

`build_attn_inp_kq_mask` allocates an f16 tensor of `n_kv x n_tokens` and
`fill_mask` writes all of it. Kmic-68 reports that for a single sequence in
position order, query *t* sees exactly the first `L0 + t` cells, so a list of
prefix lengths carries the same information, freeing room to run `-ub 2048` with
vision enabled, bit-exact.

**That premise does not hold for our cache.** `set_input_kq_mask_impl`
(`src/llama-kv-cache.cpp:2258-2281`) builds an *index list*, not a prefix: it
tests `cells.is_empty(j)`, `cells.seq_has(j, seq_id)` and `cells.pos_get(j)`
per cell, and pushes surviving indices. Deletion, reuse and shifting leave
holes and let a cell's stored position differ from its index, so visible cells
need not occupy `[0, L)`. A prefix length would attend to invalid cells and miss
valid ones unless compactness is proven or the full mask is kept. This is the
hardest precondition in the set and R12.0a makes it explicit.

At our depths the tensor is tens of MiB, not their 1 GiB, but it is the same
tensor whose differing width produced the 8 KiB allocation-plan variance that
PR #90 T36 declined to bound: a decode's smaller mask moves an 8 KiB tensor into
a different hole, and best fit is not monotonic in tensor sizes.

Replacing the tensor removes the variance by construction rather than documenting
it.

## Source

- Kmic-68 `p100-docs/FINDINGS.md`, *Do not materialise the causal mask*.

## In this fork

- `src/llama-graph.cpp:30-42` `build_attn_inp_kq_mask`, the allocation.
- `src/llama-graph.cpp:3007-3008` the call site, and `:458-463` `fill_mask`.
- `common/speculative.cpp:2210-2216`, the chained path that reads packed logits
  and depends on mask widths staying stable.

## Requirements

- **R12.1** (optional feature) WHERE a decode batch carries a single sequence in position order, `cparams.causal_attn` is true and `hparams.use_alibi` is false, the <mask builder> shall represent causality as a per-query prefix length instead of an `n_kv x n_tokens` tensor.
- **R12.0a** (optional feature) WHERE the prefix representation is selected, the <mask builder> shall first establish that every visible KV cell occupies a contiguous index range `[0, L)` in position order for this sequence, as `set_input_kq_mask_impl` at `:2258-2281` assumes of nothing; otherwise the full tensor shall be allocated.
- **R12.0b** (optional feature) WHERE `cparams.flash_attn` is false, the <mask builder> shall allocate the full tensor, because the non-FA path at `src/llama-graph.cpp:2809-2844` routes through `ggml_soft_max_ext` with a complete mask and consumes no prefix lengths.
- **R12.0c** (optional feature) WHERE the scheduled backend has no prefix support, the <mask builder> shall allocate the full tensor. The mask is built in backend-independent graph code while `GGML_OP_FLASH_ATTN_EXT` is implemented by CPU, Vulkan, OpenVINO and SYCL here, so a representation chosen from model and batch properties alone would reach backends that cannot consume it.
- **R12.0d** (event-driven) WHEN the graph is reused, the <reuse predicate> shall re-establish R12.0a and the eligibility conditions, because `llm_graph_input_attn_kv::can_reuse` at `src/llama-graph.cpp:496-508` today checks only tensor dimensions through `can_reuse_kq_mask`, and a same-shaped graph would be reused after deletion, reuse or shifting had broken the compact layout.
- **R12.2** (event-driven) WHEN the prefix-length representation is in use, the <FA kernel> shall derive each query's visible cell count from it rather than from a mask tensor.
- **R12.3** (ubiquitous) The <representation> shall yield bit-identical attention output to the materialised mask for every query position.
- **R12.4** (unwanted) IF a batch carries multiple sequences, or non-contiguous positions, or sliding-window attention, THEN the <mask builder> shall allocate the full tensor as it does today.
- **R12.4a** (unwanted) IF `cparams.causal_attn` is false, THEN the <mask builder> shall allocate the full tensor, because `fill_mask` only masks future tokens under `cparams.causal_attn` and a prefix representation would hide cells a noncausal attention is meant to see.
- **R12.4b** (unwanted) IF `hparams.use_alibi` is true, THEN the <mask builder> shall allocate the full tensor, because `fill_mask` writes `-abs(p0 - p1)` rather than 0, so the tensor carries a position-dependent bias that a visible-cell count cannot express.
- **R12.5** (event-driven) WHEN the reservation sizes a compute buffer, the <planner> shall size against the prefix-length representation's memory.
- **R12.6** (unwanted) IF the prefix representation would exceed the materialised mask's own size at the current depth, THEN the <builder> shall materialise the tensor instead.
- **R12.7** (event-driven) WHEN this plan lands, the <PR #90 T36 residual> shall be updated to record that the 8 KiB variance is bounded by construction.

## Acceptance

Bit-exact logit comparison at depths 4096 and 16384 against the materialised mask,
plus a compute-buffer size before and after. The turbo oracle suite passes with no
regression in nmse or cosine.
