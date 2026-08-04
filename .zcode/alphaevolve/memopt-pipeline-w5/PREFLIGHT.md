# Wave 5 pre-flight: gemma hybrid-iswa paged fix

## Why
Wave 4 fixed the v_trans mismatch and unblocked paged on non-hybrid models
(qwen0.5B: 3% -> 100%). gemma-4-12B (which uses llama_memory_hybrid_iswa:
split base/SWA caches with different head dims 512/256 and layer reuse) is
STILL 1% on the paged path. The K/V data and cache structure are sound (the
non-paged flash path works 100% on the same model), so the bug is in how the
paged op is dispatched for the hybrid case.

## Baseline
- sha: bbfc3493d (wave-4 review tip - has the v_trans fix + SWA page-map filter)
- Builds clean. Build: cmake --build build --target llama-server llama-bench -- -j8

## The symptom (precise)
- Model: /Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf
- 80-token greedy decode, paged path: 1% token agreement (1/80) vs flash_attn_ext
- Same model, flash path: 100%. Same model with -ngl 0 (CPU): also broken on paged.
- qwen2.5-0.5B (non-hybrid): 100% on paged after wave 4. Don't regress it.
- 16 GB RAM: pp512 -n 32 max. SERIALIZE benches.

## What wave 4's agent recommended (two approaches)
1. Add a --paged-layer-mask debug flag to isolate WHICH cache (base vs SWA) is
   producing garbage - test paged one layer at a time vs flash output.
2. Byte-diff the K view the paged kernel receives vs what flash_attn_ext
   receives for the same layer - the divergence point IS the bug.

## The hybrid structure to understand
- src/llama-memory-hybrid-iswa.{h,cpp} - the split base/SWA cache
- src/llama-kv-cache.cpp - how slot_info is built for hybrid (the page map
  builder takes slot_info; for hybrid there may be TWO slot_infos, one per
  cache type, and the paged dispatch may be using the wrong one or mapping
  positions across the split incorrectly)
- src/llama-graph.cpp - paged op dispatch; the gating added by wave 4
- The head-dim difference (base 512, SWA 256) is a prime suspect: the page map
  or the kernel may assume one head dim and misindex the other.

## Correctness gate (defines "fixed")
- 80-token greedy decode on gemma-4-12B paged path >= 95% token agreement with
  flash_attn_ext. Report exact %.
- ctest -R test-server-prompt-cache passes.
- qwen2.5-0.5B stays 100%.

## Mechanics
- Single gene, one worktree off bbfc3493d.
- Budget: 5 gens OR 45 min OR 10 candidates. stagnation_limit=3.
- ASCII only. Commits on evolve/memopt-pipeline-w5/* only. Never master/main.
- Never weaken a test.

## Output
- review branch evolve-review/memopt-pipeline-w5 off bbfc3493d.
- Final message: token agreement % before/after on gemma, qwen regression check,
  which layer/cache was the culprit, candidate count, bugs.
