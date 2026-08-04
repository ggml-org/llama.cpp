# best.md - run: memopt-pipeline-w1

## Champion: gene g1 (single-gene wave 1)

- **Owner**: wave1-single
- **Approach**: S1 - q8_0 K/V through GGML_OP_TESSERA_PAGED_ATTN with on-the-fly
  dequantization, plus the -kvu evaluator plumbing and a fix to the pre-existing
  broken page_map.
- **Champion tip SHA**: `28062fa0ba5a7b009008fcc314237a66e6832c3f`
- **Champion branch**: `champions/g1` (in the integration worktree)
- **Patch**: `integration/patches/g1.patch`
- **Lineage**: baseline 53752d7 -> gen1 (pruned, abort) -> gen2 (champion)
- **Generation evaluated to**: 2 (out of 8 budget). stagnation_limit=4 not hit;
  terminated early because gen2 is correct and beats the f16 baseline on the
  primary metric, and decode-throughput optimization was deemed a follow-up.

## Scores (qwen2.5-0.5B Q4_K_M, pp512 + n64, Metal+BLAS, 3-run medians)

| variant                         | peak RSS (MiB) | tg64 (t/s) | correctness |
|---------------------------------|----------------|------------|-------------|
| f16 KV baseline (flash path)    | 574.3          | 94.0       | reference   |
| q8_0 KV (flash_attn_ext, prior) | 571.9          | 87.3       | reference   |
| **q8_0 KV paged (champion)**    | **571.9**      | 41.0       | pass        |

- **Primary metric -peak_RSS (MAXIMIZE)**: champion -571.9 MiB vs f16 baseline
  -574.3 MiB -> **-2.4 MiB win** (0.4%). Modest because the 0.5B model at 576
  tokens has only ~14 MiB of f16 KV; q8_0 halves the KV portion. Scales with
  model/context size.
- The champion matches the existing q8_0 flash_attn_ext RSS exactly (no f16
  staging buffer materialized), which is the S1 thesis: q8_0 flowing through
  the paged op directly.
- **Correctness**: f16 paged output == f16 reference (0/40 token diff); q8_0
  paged == q8_0 flash_attn_ext (0/40 token diff on two distinct prompts);
  `ctest -R test-server-prompt-cache` passes.

## What worked

1. **On-the-fly q8_0 dequant** in the attention inner loop (CPU `load_k`/`load_v`
   and Metal `kernel_tessera_paged_attn_q8_0`). The flat element index maps
   cleanly to `block[i/QK8_0].d * block[i/QK8_0].qs[i%QK8_0]` because the cache
   stores q8_0 rows contiguously per head, matching the f16/f32 layout the
   kernel already assumed.
2. **`-kvu` plumbing in llama-bench** so the paged path is reachable from the
   evaluator (it requires `kv_unified` + single-token decode).
3. **Fixing `set_input_tessera_page_map`** to invert `v_cells` into the full
   logical->physical map. This was the load-bearing fix: the prior code only
   populated current-ubatch cells, so historical positions read garbage.

## Known limitation

Decode throughput (tg64) is 41 t/s vs 87 t/s for q8_0 flash_attn_ext. The
Metal q8_0 dequant macro does two independent block-pointer dereferences per
element and is not vectorized. A follow-up gen should fuse the dequant into a
blocked/simdgroup form (the existing `dequantize_q8_0_t4` helper used by
flash_attn_ext is the reference). The win is demonstrated; the speed is not.

## Bugs found in baseline tessera (valuable for the orchestrator)

1. **TESSERA_PAGED_ATTN was correctness-broken even for f16.**
   `set_input_tessera_page_map` built the page map from
   `tessera_kv_block_table::make_page_map`, which only walks the *current
   ubatch's* spans (from `slot_info.idxs`). All historical logical positions
   were left at their zero/UINT32_MAX initialization, so during decode the
   kernel attended to garbage cells and the model emitted repetitive junk
   within ~3 tokens. The path only looked "fine" if you never checked the
   output tokens. Reproduce: `TESSERA_PAGED_ATTN=1 llama-cli -kvu ...` on the
   baseline and watch it produce repetitive garbage. Fix is in this champion
   (`build_tessera_full_page_map`).
2. **Metal `supports_op` for TESSERA_PAGED_ATTN** only accepted F32/F16 K/V,
   so any quantized-KV paged op silently fell back to the CPU kernel.
3. **llama-bench has no `-kvu` flag**, so the paged path was unreachable from
   the standard evaluator regardless of `TESSERA_PAGED_ATTN=1`.
4. **ops.cpp transitive-include quirk**: `block_q8_0`/`QK8_0` are only visible
   if `GGML_COMMON_DECL_CPP` is defined *before* the first include of
   `ggml-common.h`; defining `GGML_COMMON_IMPL_C` is NOT sufficient (it only
   activates the tables block, not the struct decls).
5. **Baseline sha `53752d7` does not build the server**: it is a WIP stash
   merge that omits 6 untracked server sources (`server-admission.*`,
   `server-metrics.*`, `server-prefill-policy.*`) referenced by the committed
   `tools/server/CMakeLists.txt`. They must be copied from the main tree to
   build `server-context`, `llama-server`, and `test-server-prompt-cache`.

## Stacking

Single gene, so the champion IS the stack. Applied to integration/main off
baseline 53752d7. The review branch `evolve-review/memopt-pipeline-w1` carries
the same diff for the user's main repo.
