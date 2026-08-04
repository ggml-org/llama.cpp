# memopt-pipeline-w3 best.md (champion)

## Champion: g1 (S4 low-bit KV paged-attn) - PARTIAL

- **Branch**: `evolve-review/memopt-pipeline-w3` on the main repo, off 45fec509.
- **Champion tip**: 6b47815e2 (review branch), 602d795311ca... (gene worktree tip).
- **Patch**: `integration/patches/g1.patch`.
- **Lineage**: gen1 (single generation; gen2 was a correctness-validation pass, pruned).
- **Stacked with**: nothing else (single gene). S1 (q8_0 paged-attn) is already in the baseline.

## Scores (in-session, 3-run median, pp512/n32, Metal, gemma-4-12B Q5_K_M)
- q8_0 baseline (floor): peak RSS 8,989,589,504 B (~8.99 GB), 4.52 tg t/s.
- q4_0 (champion config): peak RSS 8,946,450,432 B (~8.95 GB, **-43 MB / -0.48pct**), **5.72 tg t/s (+27pct)**.
- q5_0: peak RSS 8,957,149,184 B (~8.96 GB), no decode-speed advantage over q4_0.
- PREFLIGHT floor (different env): 8,300,724,224 B (~8.30 GB). My in-session floor was ~8.99 GB; the gap is a measurement-environment difference (Metal residency sets, prior-buffer keep_alive). All candidates compared against the in-session floor.

## Correctness (STRONG gate, 80-token greedy decode, server logprobs)
- q8_0 via flash_attn_ext: **100pct token match to f16** (80/80).
- q4_0 via flash_attn_ext: **100pct token match to f16** (80/80).
- q4_0 / q8_0 / f16 via **paged** path: ALL garbage (3.8-10pct agreement). Pre-existing baseline bug (SWA), not from this gene.
- ctest -R test-server-prompt-cache: PASSED.

## Verdict: PARTIAL
The quantization code is correct and the q4_0 memory/throughput win is real.
End-to-end correctness through the paged path is blocked by a pre-existing
SWA-handling bug in S1's paged-attn (affects f16 too). The q4_0 path should
be correct once S1's SWA bug is fixed; the dequant math is bit-exact vs
ggml-quants.c reference.

## Why not "stacked/ship": the win is small and the end-to-end path is broken.
- At 12B/512 the KV cache is ~tiny vs 8 GB weights, so the RSS delta is small.
- The paged path (the only path that exercises this code) is broken on this model.
- q2_K (the headline 2-bit target) is blocked by a Metal backend gap (no quantize_q2_K / set_rows).
