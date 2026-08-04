# best.md - run: memopt-pipeline-w2

## Final champion stack (compounding winner: s2)

- baseline_sha: 2200c659d4e67e638dab67b43f672a1921bfeb8d
- integration/main HEAD: 15952f3ad4f26bd42886f8b7b8b81c8485557b67
- review branch: evolve-review/memopt-pipeline-w2 (off baseline)

| variant                          | peak RSS (5-run median) | pp512 t/s | tg32 t/s | correctness |
|----------------------------------|-------------------------|-----------|----------|-------------|
| baseline (lazy OFF)              | 9,067,905,024 B (8.45 GiB) | 65.12  | 6.06     | reference   |
| **s2 stacked (LLAMA_KV_LAZY_CLEAR=1)** | **8,891,891,712 B (8.28 GiB)** | 68.87 | 6.45 | pass        |
| delta                            | **-176,013,312 B (-1.94%)** | +3.75   | +0.39    | 0/32 diff   |

## Stacked champion

- **s2** - mmap-backed KV residency (lazy physical commit). Owner: w2-single.
  - Approach: skip the eager `ggml_backend_buffer_clear` on KV buffer init
    when `LLAMA_KV_LAZY_CLEAR=1`. On Metal shared memory the memset pins every
    KV page up front; skipping it lets peak RSS track only written pages
    (vAttention-style lazy commit). Cells guard all reads so the clear was
    purely defensive.
  - Champion tip: be087a037ff3f8049c6751eb0e65d58bc71aa506
  - Champion branch: champions/s2
  - Patch: integration/patches/s2.patch
  - Generation evaluated to: 1 (stagnation not hit; gen1 correct + winning,
    terminated early to respect budget).

## Skipped (non-compounding, patches retained)

- **s3** - sliding-window KV eviction helper (`llama_memory_seq_evict_oldest`).
  Correctness-proven via custom probe; bench-invisible (opt-in, unused by
  bench at pp512/n32). Useful scaffold for long-stream workloads.
  Patch: integration/patches/s3.patch
- **s7** - configurable radix-cache cap (`LLAMA_RADIX_MAX_BLOCKS`).
  Architectural finding: `server_kv_block_radix` is metadata-only (no KV data);
  the CacheGen KV-compression premise does not apply to it. Opt-in cap made
  tunable; bench-invisible. Patch: integration/patches/s7.patch

## Honest assessment

- The wave shipped ONE compounding winner (s2, -176 MiB / -1.94% peak RSS),
  which is the documented success criterion.
- s2's win is robust: 5-run OFF/ON medians have non-overlapping ranges
  (OFF: 9.067-9.068G after dropping the cold-start 8.23G outlier;
   ON:  8.891-8.892G).
- s2's win grows with context length (more KV pages deferred) and with KV
  precision (q8/q4 KV has the same lazy-commit benefit). At pp512 on a 12B
  Q5_K_M the win is ~1.94%; at pp2048+ it would be substantially larger (but
  this 16GB host OOMs above pp512, so we cannot measure that here).
- s3/s7 are correct, non-regressing scaffolds that pay off in regimes the
  single-stream bench does not exercise (long streams; multi-prefix serving).
  Neither is force-fit through the compounding test.
