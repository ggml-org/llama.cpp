# Wave 2 pre-flight facts (resolved by the orchestrator)

Read this and the parent research.md before evolving. Do NOT re-derive.

## Build
- Baseline sha: 2200c659d4e67e638dab67b43f672a1921bfeb8d
  (this is the wave-1 review tip + the previously-untracked tessera WIP files
   committed: server-admission/metrics/prefill-policy + ggml-ane backend.
   Builds clean; server binary works.)
- Includes wave 1's S1 champion (q8_0 paged-attn + the f16 page-map correctness
  fix + the -kvu llama-bench flag + the build_tessera_full_page_map fix).
- Build: `cmake --build build --target llama-server llama-bench -- -j8`

## Model (recalibrated for this 16GB machine)
- Path: /Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf
- ~12B params, Q5_K_M, 8.04 GiB. Loads on Metal (Apple M1, 16 GB RAM).
- IMPORTANT: 16 GB RAM is the binding constraint. The 12B Q5_K_M is ~8 GB of
  weights. DO NOT run pp2048 - it OOMs. Keep bench at pp512 -n 32 (verified
  working, peak RSS ~8.96 GB).
- Each gene worktree has its OWN build dir (multi-GB). With ~67 GB free disk
  and 3 genes, that is fine; with 16 GB RAM you CANNOT run 3 builds/benches
  concurrently - serialize the eval step across genes (flock if needed).

## Baseline numbers (f16 KV, wave-1 paged path NOT used for default decode)
- pp512: 64.63 t/s, tg32: 5.77 t/s, peak RSS: 8,959,934,464 B (~8.96 GB)
- The KV cache at 512 context on 12B is the target. S2/S3 should reduce the
  KV contribution; S7 reduces the radix-cache (prefix-cache) contribution.

## Correctness gate
- `cd build && ctest -R test-server-prompt-cache` must pass.
- logit-diff vs baseline on a fixed short prompt (greedy, 32 tokens, 0 diff).

## The three candidates (disjoint file regions - parallel-safe islands)

### S2: mmap-backed KV residency (lazy physical commit)
Wire the KV cache buffer to mmap with lazy commit so peak RSS tracks touched
KV pages, not n_ctx * head_dim. vAttention philosophy on Apple UMA.
Region: src/llama-kv-cache.cpp (buffer allocation), src/llama-memory*.cpp,
        ggml/src/ggml-backend*.
Risk: page-fault stalls during decode; interaction with the existing block
      table. Measure both RSS AND throughput - a memory win that halts
      throughput is a regression.

### S3: KV eviction policy (H2O / StreamingLLM-style)
Add an eviction hook off the seq_rm path that drops low-attention or non-sink
tokens under a configurable memory budget. Caps KV growth vs sequence length.
Region: src/llama-kv-cache.{h,cpp} (eviction near seq_rm at line 379),
        src/llama-kv-cells.h.
Risk: quality degradation on long-context recall. The correctness gate MUST
      include a long-context recall fixture, not just a short prompt.
      Opt-in policy knob; default off.

### S7: CacheGen-style KV compression for the radix cache
Delta + entropy code the radix-cached KV blocks so the prefix cache shrinks.
Region: tools/server/server-context.cpp (radix cache store ~line 831),
        tools/server/server-queue.cpp.
Risk: CPU cost of compress/decompress; benefit depends on prefix-cache hit
      patterns. Make it opt-in and measure cache hit rate too.

## Mechanics
- 3 genes, one worktree each, branched off 2200c659d.
- Per-gene build dir INSIDE the worktree. Never share build/.
- ASCII only. No em-dash, no unicode arrows.
- Commits on evolve/memopt-pipeline-w2/* only. Never master/main. Never push.
- Budget: 6 generations OR 50 min OR 20 candidates per gene. POC-scale.
  stagnation_limit = 4.
- Memory metric is noisy: 3 runs, median, wide tie band.
- Because 16 GB RAM cannot run 3 llama-bench on a 12B concurrently, SERIALIZE
  the eval step. Use the integration flock or a dedicated eval.lock if you
  parallelize the build but not the bench.

## Output
- review branch evolve-review/memopt-pipeline-w2 on the main repo, off 2200c659d,
  with the compounding stack applied (those of S2/S3/S7 that pass the test).
- In your final message: per-gene verdict (stacked/skipped + reason), final
  peak RSS vs 8.96 GB baseline, correctness status, candidate counts, bugs.
