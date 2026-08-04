# DSV4-Flash Indexed CSA — design note (gather-only-selected)

Status: **on hold** behind the target-only raw-decode baseline and M5.6 scaling gate in `deepseek-v4-flash-rocm-performance.md`; not implemented. Owner commit for the original note: `925d93700`. The initial review's "feasible" verdict applies to a single-query decode proof, not a general multi-query PP gather.

## 1. Problem this solves

The current CSA path is dense-masked (source-confirmed in `deepseek4.cpp`
`build_csa_lid_attention`):

1. `build_lid_top_k` scores every visible compressed entry with the lightning
   indexer and selects `n_top_k = min(n, indexer_top_k)` (<=512) per token.
2. `build_top_k_mask` fills a full `-INF` mask and scatters zeros at the
   selected indices.
3. `build_csa_lid_attention` concatenates the local raw-window K with **all**
   `T = ctx/4` compressed CSA keys and runs generic flash attention with the
   dense arbitrary mask (+ sink).

The softmax *math* reduces to `selected + local + sink` (non-selected run
`exp(-inf)=0`). The graph presents every compressed key to flash, so its
operand length grows with `T = ctx/4`; source alone does not prove how many
masked elements the HIP kernel physically reads. A single-run 16K -> 32K PP
sweep raised whole-graph time ~6.3x, which is consistent with CSA/LID scaling
but does not attribute the increase to flash.

## 2. Design

Phase A, if selected by M5.6, is a **raw-decode-only** gather proof enabled
only for one query token per stream:

- Keep `build_lid_top_k` unchanged: it emits per-token indices
  `top_k [n_top_k, n_batch, 1, n_stream]` into the compressed axis.
- Gather selected compressed K/V from the same shared-K=V cache plus the valid
  local raw-window K/V, producing at most 512 + 128 = 640 physical KV rows.
- Continue passing per-head sinks through the existing separate `sinks`
  argument; a sink is a virtual softmax input, not a concatenated KV row.
- Gather the selected original mask values or compact invalid selected tails.
  When fewer than 512 positions are valid, `top_k` may return masked `-INF`
  entries; the gather must not make them visible.
- Run the existing flash path over the compact KV only if its tensor layout
  correctly represents the single query and introduces no backend fallback.

For multi-query PP, each query has a different selected set while existing
flash consumes common K/V per stream. A plain `GGML_OP_GET_ROWS` + existing
flash is therefore not an established PP implementation; prefill remains
on the dense-mask path until a per-query indexed layout or direct indexed
attention kernel exists.

## 3. Correctness contract (indexed CSA)

Must preserve (from dense-masked, which is the accepted reference):

- Shared K=V MQA (single KV head); q has ~64 query heads; MQA broadcast over
  the per-(batch,stream) gathered KV set.
- Per-token selected index sets; `top_k <= 512`.
- Local raw/SWA branch (<=128 tokens) + per-head attention sink.
- One stable softmax over `selected compressed + local + sink`;
  mathematically equivalent to the dense-masked union (non-selected
  contribute exp(-inf)=0), but not assumed bitwise identical.
- Causal visibility and compression/overlap completion boundaries: the score
  mask sends invalid entries to `-INF`, but top-k tails can still name them
  when fewer than K entries are valid. Preserve/gather the original mask or
  compact invalid entries; never treat every returned index as visible.
- Inverse partial RoPE / k_rot: `csa_sel_k` must be gathered from the exact
  same `csa_k` tensor in the same rotation state as the dense path (gather
  from the cached compressed K before/after the identical rotation).
- Allocated cache stream stride and logical-to-physical mapping across cache
  wrap, reuse, reset, and any moving cache base; include streams with unequal
  lengths and phases.
- Deterministic duplicate/tie policy: reuse the accepted TOP_K operation and
  define tie semantics explicitly. Repeated output must be deterministic;
  exact CPU index equality for tied values is required only if an index
  tie-break is part of the contract.
- Multi-GPU tensor-split: gather is valid if CSA K is present on the device
  executing each token's attention (mirrored/replicated KV); must not rely on
  cross-device scatter of arbitrary per-token indices.
- Dense generic fallback + a force-reference switch for A/B and correctness
  comparison.

## 4. Reviewer findings to resolve in implementation

- **R1 (medium-high):** the per-token gather index must map to the actual CSA
  cache strides (flattened `ne0/ne1/ne2/ne3` + stream offset), not assumed
  contiguous ordering. Verify with the real layout before coding.
- **R5 (medium-high):** cross-device determinism/consistency of `top_k`
  indices under tensor split; confirm each device's gather uses indices that
  reference that device's reachable KV and that all devices agree on the same
  selection.
- **Bitwise equivalence is NOT guaranteed.** Dense-masked (large `N_v` with
  `-INF`/`0.0` rows) vs gather-only (small `N_v`, those rows absent) may differ
  in flash online-softmax running max, `l`/`seql`, and accumulation order.
  Before implementation acceptance, declare absolute and relative tolerances
  and their comparison formula; require identical NaN/Inf classification;
  report max and percentile error; and pass fixed downstream logit/token
  gates. Report any bitwise subset separately.
- MQA broadcast and rotation state are compatible with gathering from the
  same cache tensor. Multi-GPU locality still requires scheduler proof.
- **R6 (high for PP):** existing `ggml_get_rows` batch rules and
  `ggml_flash_attn_ext` common-KV semantics do not directly represent a
  different compact selected-KV set per query. "No new kernel needed" is not
  accepted for `n_batch > 1`.

## 5. Where the next bottleneck is

If flash is bounded for decode, the remaining long-context cost includes the
**indexer/top-k selection**. llama.cpp's fused indexer reduces the 64 indexer
heads internally and materializes `[T,ubatch,1,n_stream]`, not a resident
full-sequence `[S,H_I,T]` tensor. It still scans O(T) and writes/re-reads F32
scores for every raw decode token, and performs O(S*T) total work over full
prefill. The external 256 GB figure describes a full-sequence reference
materialization, not llama.cpp's ubatched peak allocation. A later measured
phase may add chunked/streaming top-k (tile candidates -> hierarchical merge)
to bound intermediate traffic; exact selection remains linear in candidates.

## 6. Proposal / follow-ups

1. Hold implementation until the target-only raw-TG sweep and M5.6 component
   profile select CSA.
2. If selected, add a decode-only sparse-attention microbenchmark and prove
   the stream/cache indexing and selected-mask semantics before integration.
3. Build a one-query gather proof with dense fallback and force-reference
   switch; A/B 8K/16K/32K/64K and establish the measured crossover.
4. Then consider direct indexed ROCm attention; keep multi-query PP dense until
   a correct per-query indexed representation/kernel exists.
5. If selection becomes material, consider fused/streaming top-k.
6. Do not change top-k semantics, ratios (CSA 4 / HCA 128), sinks, or model
   math; keep generic/non-HIP fallbacks intact.