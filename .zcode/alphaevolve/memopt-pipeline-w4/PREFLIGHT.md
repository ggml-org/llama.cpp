# Wave 4 pre-flight: SWA-in-paged fix (the gatekeeper bug)

## Why this wave
Waves 1 and 3 shipped the q8_0/q4_0 paged-attn path (S1, S4). Wave 3 discovered
that on the gemma-4-12B model (which uses sliding-window attention), the paged
path produces GARBAGE for ALL KV types including f16 - only 3.8% token agreement
vs f16 flash_attn_ext. The non-paged flash path is correct. So the bug is in the
paged path's SWA handling, and it blocks S1/S4 from being real wins on production
SWA models. Fixing it is the highest-value next step.

## The bug, precisely
src/llama-kv-cache.cpp:1327 `build_tessera_full_page_map` inverts cells into a
logical-position -> physical-cell map. It populates EVERY resident cell's
position, including cells OUTSIDE the SWA window for SWA layers. The paged-attn
kernel then reads K/V rows it should be masked from, and produces noise.

## The fix region
The surrounding code ALREADY respects SWA:
- slot_info carries n_swa + swa_type (src/llama-kv-cache.cpp:1698-1699, 1714-1715)
- slot_info.idxs is the per-stream cell list, filtered by is_masked_swa at
  line 1836 - this is the SWA-respecting cell set
- llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1) is the masking check
  (src/llama-hparams.h:380)
- SWA types: LLAMA_SWA_TYPE_NONE/STANDARD/CHUNKED/SYMMETRIC (hparams.h:20-24)
- is_swa_impl[il] marks which layers are SWA (hparams.h:149)

The fix: make build_tessera_full_page_map consult n_swa/swa_type (passed via
slot_info or as args) and skip cells whose position is_masked_swa against the
current token's position window - OR build the map from the SWA-filtered idxs
that the slot already produces. The cleanest path is to reuse slot_info.idxs
since it's already correct, rather than re-inverting v_cells.

CRITICAL: the fix must work for ALL FOUR swa_type values, not just STANDARD.
Test each if you can construct a case. Chunked and Symmetric have different
masking geometry than Standard.

## Build
- Baseline sha: 6b47815e2 (the w3 review tip - has S1 q8_0 + S4 q4_0/q5_0 + the
  WIP server files + ggml-ane). Builds clean.
- Build: cmake --build build --target llama-server llama-bench -- -j8

## Model + the exact failure to fix
- Model: /Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf
- 12B Q5_K_M, uses SWA. Loads with --no-embedded-mtp.
- 16 GB RAM: pp512 -n 32 only. SERIALIZE benches.
- The failure: run the model through the paged path (TESSERA_PAGED_ATTN=1 or -kvu)
  and decode 80 tokens. Current output: 3.8% token agreement with f16 flash_attn_ext
  (multilingual noise). TARGET: 100% agreement (byte-identical greedy decode) for
  f16, matching the qwen0.5B result from wave 1. Anything less than ~95% is not a
  fix - flash_attn_ext achieves 100% on this same model, so the paged path should
  too once SWA is handled.

## Correctness gate (STRONG - this is the whole point)
- Primary: 80-token greedy decode through the paged path must be >=95% token-
  identical to the same decode through flash_attn_ext, for f16 KV, on the gemma
  model. This is the regression that defines "fixed."
- ctest -R test-server-prompt-cache must pass.
- Re-verify qwen0.5B (no SWA) still works at 100% - don't break the non-SWA case.
- If you extend to q8_0/q4_0 paged: verify those match their flash_attn_ext
  equivalents too (the S1/S4 quality claims depend on this).

## Mechanics
- Single gene, one worktree off 6b47815e2.
- Budget: 5 generations OR 45 min OR 10 candidates. This is a focused bug fix;
  don't over-explore. stagnation_limit=3.
- ASCII only. Commits on evolve/memopt-pipeline-w4/* only. Never master/main.
  Never push, never gh.
- Never weaken a test or assertion.

## Output
- review branch evolve-review/memopt-pipeline-w4 off 6b47815e2.
- Run artifacts: gene-ledger.json, changes.md, best.md, integration/patches/g1.patch.
- Final message: the % token agreement before and after the fix (before should be
  ~3.8%, after should be >=95%), which swa_types you handled, whether q8_0/q4_0
  paged now also work on gemma, candidate count, bugs.

This is a correctness fix, not a memory win. The RSS should be roughly unchanged.
The value is that it UNLOCKS S1/S4 on every SWA model (gemma, mistral, qwen, etc.).
