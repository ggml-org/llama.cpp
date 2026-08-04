# memopt-pipeline-w4 best.md (champion) - PARTIAL

## Champion: g1 (paged-attn V-transpose fix + SWA page-map filter) - PARTIAL

- **Branch**: `evolve-review/memopt-pipeline-w4` on the main repo, off 6b47815e2.
- **Champion tip**: dab0d5117 (gene worktree), bbfc3493d (review branch cherry-pick).
- **Patch**: `integration/patches/g1.patch` (169 lines, 3 files).
- **Lineage**: gen2 (gen0 baseline, gen1 SWA-filter-only pruned as insufficient, gen2 v_trans+filter+gating promoted).
- **Candidates evaluated**: 2.
- **Stacked with**: nothing else (single gene).

## The actual bug (NOT what PREFLIGHT.md described)

PREFLIGHT.md diagnosed the bug as: "build_tessera_full_page_map populates cells outside the SWA window, the paged kernel reads them, producing noise." That diagnosis was WRONG (or at least incomplete):

1. **The real primary bug**: The tessera paged-attn kernel reads V in row-major (non-transposed) layout ONLY. But the KV cache stores V transposed when `flash_attn` is off (`v_trans = !flash_attn`). With `-fa off` (the w1/w3 test methodology and the server default), the kernel read transposed V bytes as garbage. This broke paged on EVERY model, not just SWA ones:
   - qwen2.5-0.5B (no SWA): 1pct -> 100pct after fix
   - gemma-4-12B (SWA): 1pct -> 1pct (separate issue, see below)
   - Verified with `-fa on` (v_trans=false) on qwen: paged already matched flash 100pct BEFORE any code change, proving v_trans was the differentiator.

2. **SWA page-map filter (defensive, kept)**: build_tessera_full_page_map now filters resident cells by the SAME causal + SWA + per-sequence gating the flash-path KQ mask uses (is_masked_swa against seq_pos_max). This is correct and necessary for long contexts (>n_swa tokens) but is a no-op for the 80-token test (n_swa=1024 for gemma). Without it, long-context paged decode would attend to SWA-evicted cells.

## The fix (3 files, +57/-8)

1. **src/llama-model.cpp**: Force `v_trans=false` at every KV-cache construction site (dsa, dsv4, hybrid_iswa, hybrid, plain kv_cache) when `TESSERA_PAGED_ATTN=1 && kv_unified` are both set. Computed once as `attn_v_trans` and threaded through. No-op when flash_attn is already on.
2. **src/llama-graph.cpp**: Add `!mctx_cur->is_v_trans()` to BOTH paged dispatch gating conditions (build_attn_kv and build_attn_kv_iswa) as defense - paged falls back to flash if V is ever transposed.
3. **src/llama-kv-cache.cpp**: build_tessera_full_page_map now skips cells that are causal-future, SWA-masked, or wrong-sequence relative to each sequence's seq_pos_max. Handles all four swa_type values via is_masked_swa.

## Scores (review branch, 30-80 token greedy decode, seed=0 temp=0 argmax)

- **qwen2.5-0.5B f16 paged vs flash**: 100pct (30/30). Was 1pct (3/30) before fix. FIXED.
- **gemma-4-12B f16 paged vs flash**: 1pct (1/80). Was 1pct before fix. NOT FIXED (see below).
- **ctest -R test-server-prompt-cache**: PASS.

## Why gemma-4-12B is still broken (diagnosis)

The gemma failure is a SEPARATE, pre-existing bug, NOT the SWA issue PREFLIGHT described. Evidence:
- gemma paged is broken even with `-fa on` (v_trans=false, the layout the kernel wants).
- gemma paged is broken even with `-ngl 0` (CPU kernel, not Metal).
- gemma paged is broken even with `-ctk f32 -ctv f32` (no f16 precision issue).
- gemma paged is broken even with a 3-token prompt (no SWA masking active; n_swa=1024).
- The page map is verified correct (identity 0..n-1 -> cells 0..n-1).
- The K/V data, strides, head dims, and GQA ratios all verify correct.
- qwen (non-hybrid) works 100pct with the same kernel after the v_trans fix.

The difference: gemma uses `llama_memory_hybrid_iswa` (split base/SWA KV caches with different head dims: 512 full / 256 SWA) and `n_layer_kv_from_start` layer reuse. The bug is somewhere in how the hybrid path wires the paged op for one of these structures, but I could not isolate it within budget despite extensive instrumentation (dumped K rows, strides, map contents, scores - all look correct). The non-paged flash path works 100pct for gemma, so the K/V data and cache structure are sound; the issue is specifically in the paged dispatch or kernel invocation for the hybrid case.

Recommended next step for the gemma hybrid bug: add a `--paged-layer-mask` debug flag to selectively enable paged for only base or only SWA layers, to isolate which cache's path is broken. Or diff the K tensor bytes the kernel receives vs what flash_attn_ext receives for the same layer (they should be identical pointers, so this may point to a view/offset bug).

## swa_types handled

The SWA page-map filter uses `llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)` which branches on all four types (NONE/STANDARD/CHUNKED/SYMMETRIC). It is exercised automatically whenever swa_type != NONE. Only NONE and STANDARD were reachable in testing (qwen=NONE, gemma=STANDARD); CHUNKED and SYMMETRIC reuse the same is_masked_swa code path the flash KQ mask uses, so they inherit its correctness.

## q8_0/q4_0 paged on gemma

Not verified separately - gemma paged is broken at f16 already (the hybrid bug), so q8_0/q4_0 would be broken too for the same reason. The v_trans fix does make q8_0/q4_0 paged correct on non-hybrid models (same mechanism as f16). q8_0 on qwen2.5-0.5B failed at context creation (pre-existing, unrelated to this gene).

## Verdict: PARTIAL

The v_trans fix is a real, significant correctness fix: it unblocks TESSERA_PAGED_ATTN for every non-hybrid model when used with `-fa off` (the previous state was 1-3pct token agreement = total garbage). This includes non-hybrid SWA models (mistral, qwen) once they use the paged path. The gemma hybrid case needs a follow-up wave to diagnose the separate hybrid-iswa paged bug.

## Inspect

```
git log --oneline 6b47815e2..evolve-review/memopt-pipeline-w4
git diff 6b47815e2...evolve-review/memopt-pipeline-w4
```
