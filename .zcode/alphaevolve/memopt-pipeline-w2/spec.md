# spec.md - run: memopt-pipeline-w2

Frozen by orchestrator in PREFLIGHT.md. Reproduced here for the loop.

- **Goal**: reduce peak RSS of the inference pipeline on a 12B Q5_K_M model
  at 512 context (pp512 / n32), 16 GB Apple M1.
- **Baseline sha**: `2200c659d4e67e638dab67b43f672a1921bfeb8d`
  (evolve-baseline/w2: includes server-admission/metrics/prefill-policy +
   ggml-ane backend so worktrees build clean).
- **Note on wave-1**: PREFLIGHT claims this baseline "includes wave-1's S1
  champion"; verified FALSE at evolve time (ops.cpp still asserts F32/F16 for
  paged-attn, no -kvu in llama-bench). The baseline builds and produces the
  documented RSS regardless; we evolve against the actual tree state. Any
  wave-1 stacking benefit is out of scope for this wave's compounding test.
- **Metric (primary, MAXIMIZE)**: `-peak_RSS_bytes` from `/usr/bin/time -l`
  on `llama-bench -m <model> -p 512 -n 32`. 3 runs, median. Wide tie band:
  deltas under ~0.5% of baseline (~45 MiB) are ties.
- **Metric (secondary, monitor)**: pp512 t/s and tg32 t/s. A memory win that
  crushes throughput >50% is a regression unless the memory delta is large.
- **Correctness gate**:
  1. `cd build && ctest -R test-server-prompt-cache` passes.
  2. logit-diff vs baseline: 32 greedy tokens on a fixed prompt, 0 diff.
- **Evaluator command** (serialized under eval.lock; one bench at a time):
  `/usr/bin/time -l ./build/bin/llama-bench -m "<MODEL>" -p 512 -n 32`
  (run 3x, take median of the reported tg32 + parse peak RSS from time -l)
- **Model**: `/Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf`
  (~8.04 GiB Q5_K_M; 12B). DO NOT use pp2048 or larger (OOMs).
- **Baseline numbers**: pp512 ~64.6 t/s, tg32 ~5.77 t/s,
  peak RSS 8,959,934,464 B (~8.96 GiB).
- **Budget**: 6 generations OR 50 min wall-clock OR 20 candidates per gene,
  whichever first. POC-scale.
- **stagnation_limit**: 4 generations without improvement -> freeze.
- **Genes**: three disjoint islands evolved in parallel worktrees, evaluated
  serially (16 GB RAM cannot run 3 llama-bench on 12B concurrently).
  - **S2** (mmap-backed KV residency): `src/llama-kv-cache.cpp` buffer alloc,
    `src/llama-memory*.cpp`.
  - **S3** (KV eviction policy): `src/llama-kv-cache.{h,cpp}` near seq_rm
    (~line 379), `src/llama-kv-cells.h`.
  - **S7** (CacheGen KV compression for radix cache):
    `tools/server/server-context.cpp` (~line 831), `tools/server/server-queue.cpp`.
- **Honest expectations**: S2's payoff grows with context (marginal at 512);
  S3 needs a recall fixture to prove correctness (be conservative); S7 only
  pays if the workload hits the radix cache (single-stream bench may not).
  A wave that ships ONE compounding winner and honestly reports the other two
  as non-compounding is a success.
- **Output**: review branch `evolve-review/memopt-pipeline-w2` on the main
  repo, off 2200c659d, with the compounding stack applied.
