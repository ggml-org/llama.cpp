# DSV4-Flash Indexed CSA — design note (gather-only-selected)

Status: design/proposal (not yet implemented). Owner commit for this note:
`3cf35253f`. Read-only review verdict: **feasible, no hard blocker**.

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

The softmax *math* already reduces to `selected + local + sink` (non-selected
run `exp(-inf)=0`), but the *compute* touches every compressed key, so flash
cost grows with `T = ctx/4`. Local measured scaling confirms this is severe:
16K -> 32K raised PP time ~6.3x (372 -> 117 t/s).

## 2. Design

Replace the dense all-keys flash with a **gather-only-selected** flash:

- Keep `build_lid_top_k` unchanged: it emits per-token indices
  `top_k [n_top_k, n_batch, 1, n_stream]` into the compressed axis.
- Instead of concatenating all `csa_k`, gather only the selected compressed
  K (`csa_sel_k`) from the **same** shared-K=V CSA cache tensor, concatenate
  with the local raw-window K and sink, and call the existing `build_attn_mha`
  over `O(n_top_k + n_local)` keys (~641) rather than `O(T)`.
- The non-selected compressed entries are simply absent; no `-INF` rows.

Result: flash attention cost becomes **constant in context** (~512 + 128 +
sink) instead of linear in `ctx/4`.

## 3. Correctness contract (indexed CSA)

Must preserve (from dense-masked, which is the accepted reference):

- Shared K=V MQA (single KV head); q has ~64 query heads; MQA broadcast over
  the per-(batch,stream) gathered KV set.
- Per-token selected index sets; `top_k <= 512`.
- Local raw/SWA branch (<=128 tokens) + per-head attention sink.
- One stable softmax over `selected compressed + local + sink`; identical to
  the dense-masked union (non-selected contribute exp(-inf)=0).
- Causal visibility and compression/overlap completion boundaries: top_k
  indices are already restricted to the visible set by `kq_mask`; the gather
  must not expose entries beyond them.
- Inverse partial RoPE / k_rot: `csa_sel_k` must be gathered from the exact
  same `csa_k` tensor in the same rotation state as the dense path (gather
  from the cached compressed K before/after the identical rotation).
- Stream/batch sequences with different lengths/phases.
- Deterministic duplicate/tie policy: reuse `ggml_top_k` so index sets
  (including ties) match the reference.
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
  Gate: a deterministic dense-masked vs indexed A/B with a strict relative
  tolerance (and report whether a small subset is bitwise), plus routing/layer
  checks on the fixed natural proxy.
- MQA broadcast, rotation state, and multi-GPU gather locality were all
  reviewed as fine for this shared-K layout.

## 5. Where the next bottleneck is

After flash is reduced to constant `N_v`, the remaining long-context cost is
the **indexer/top-k selection**: `indexer_score [ubatch, n_indexer_head, T]`
is materialized and `ggml_top_k` runs over all `T` per token (O(S*T)). That is
still linear in `ctx/4` and, per StreamIndex (arXiv 2605.02568), the full score
tensor at 64K+ (256 GB at S=65,536 with V4-Flash dims) and the raw top-k
reduction are the binding wall. A later phase must add **chunked/streaming
top-k** (tile-partition-merge) for 32K-1M; the indexed-flash change and the
streaming-indexer change are separable and should be staged in that order.

## 6. Proposal / follow-ups

1. Build a minimal gather-only-selected CSA with a dense fallback and
   force-reference switch (backend: reuse existing GGML_OP_GET_ROWS gather +
   existing flash; no new flash kernel needed).
2. A/B dense-masked vs indexed at 8K/16K/32K on the fixed natural proxy and a
   fixed synthetic context to (a) confirm equivalence within tolerance and
   (b) confirm the flash t/s win grows with context.
3. If green, consider chunked/streaming top-k for the indexer selection.
4. Do not change the top-k semantics, tie policy, ratios (CSA 4 / HCA 128),
   or the model math; keep generic/non-HIP fallbacks intact.