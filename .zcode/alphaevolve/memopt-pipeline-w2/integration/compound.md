# compound.md - run: memopt-pipeline-w2

Canonical integration log. Per the profile, this is rewritten at finalize to
reflect the authoritative greedy re-stack.

## Baseline

- baseline_sha: 2200c659d4e67e638dab67b43f672a1921bfeb8d
- baseline peak RSS (5-run median, pp512/n32, f16 KV): 9,068,068,864 B (~8.45 GiB)
  (3-run median at bootstrap was 9,040,199,680 B; both within the 0.5% tie band.
   We use the 5-run value as the floor because the compounding test is 5-run.)

## Champion branches (all rooted at baseline)

| gene | champion tip                        | standalone peak RSS (5-run median) | standalone delta vs base | status     |
|------|-------------------------------------|------------------------------------|--------------------------|------------|
| s2   | be087a037ff3f8049c6751eb0e65d58bc71aa506 | 8,891,891,712 B (lazy ON)        | -176,177,152 B (-1.94%)  | promoted   |
| s3   | 051b6b69a3e1565193a78aed8349daf06ffdd178 | 9,068,199,936 B (tie, opt-in)    | ~0 (tie)                 | non-compounding |
| s7   | 69a4eba095289e9a47ae35908ee5f3d726ea1fc7 | 9,067,888,640 B (tie, opt-in)    | ~0 (tie)                 | non-compounding |

## Compounding test (s2)

- score_base (5-run median, no flag): 9,067,905,024 B
- champion_standalone (s2 with LLAMA_KV_LAZY_CLEAR=1): 8,891,891,712 B
- standalone gain: -176,013,312 B (-1.94%)
- stacked (s2 merged onto integration/main, flag ON): 8,891,891,712 B
- stacked gain: -176,013,312 B
- compounding threshold: 0.5 * standalone_gain = -88,006,656 B
- verdict: stacked gain (-176M) <= threshold (-88M) -> **PASSES, STACKED**
- correctness on stacked tree: logit_probe 0/32 diff vs baseline; ctest
  test-server-prompt-cache passes (with LLAMA_KV_LAZY_CLEAR=1 active).

## Final canonical stack

1. **s2** (lazy KV clear) - STACKED at integration/main HEAD 15952f3ad.
   peak RSS: 8,891,891,712 B (~8.28 GiB), a -176 MiB (-1.94%) win over
   baseline. The flag is opt-in (`LLAMA_KV_LAZY_CLEAR=1`); default behavior
   unchanged.

## Skipped champions

- **s3** (sliding-window KV eviction helper): non-compounding. The opt-in
  `llama_memory_seq_evict_oldest` helper is correctness-proven (custom probe
  passes: 77->16 dense, idempotent, post-eviction decode ok) but bench-invisible
  (the helper is unused by llama-bench at pp512/n32). Standalone gain ~= 0,
  so the compounding threshold is not met. The helper is a useful scaffold for
  long-stream workloads and could be wired into the server under memory
  pressure in a later wave. Patch retained at integration/patches/s3.patch.
- **s7** (configurable radix-cache cap): non-compounding. Key architectural
  finding: `server_kv_block_radix` is METADATA-ONLY (string-key trie + block
  position records, no KV tensor data). The CacheGen KV-compression premise
  (S7 research) does NOT apply to this structure - it would need to target the
  unified-cache cells, which requires S1's quantized reader (absent in w2
  baseline). The made-the-cap-tunable change is opt-in and bench-invisible.
  Patch retained at integration/patches/s7.patch.

## Per-gene integration notes

- s2: stacked cleanly (merge --no-ff, no conflicts). integrated-at 15952f3ad.
- s3: skipped (non-compounding, zero standalone gain on bench).
- s7: skipped (non-compounding, zero standalone gain on bench).
