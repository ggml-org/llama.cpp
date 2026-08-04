# spec.md - run: memopt-pipeline

Goal (verbatim): "optimize the memory usage of the pipeline"

This is a DRY Phase 0+1 run. Source is NOT edited in this phase; this file and
research.md are the only outputs.

## 1. Target

The "pipeline" is the llama-server inference path. Memory-relevant surfaces, in
priority order for this goal:

- KV cache management
  - src/llama-kv-cache.{h,cpp}             - the unified/split KV cache, cells, defrag
  - src/llama-kv-cells.h                   - per-cell metadata, seq ownership
  - src/llama-memory*.{h,cpp}              - the llama_memory_t abstraction (KV + recurrent + hybrid)
  - src/llama-kv-cache.h:101-130           - tessera_kv_block_span / tessera_kv_block_table / make_page_map
                                              (scaffolding for paged/quantized attention; wired in at
                                              src/llama-kv-cache.cpp:2776; "compressed block reader"
                                              backend appears incomplete - candidate region)
  - include/llama.h:384-385                - type_k / type_v (KV cache dtype; default F16)
  - common/common.h:604-605                - server cache_type_k/v defaults

- Weight loading
  - common/common.h:495                    - fit_params_min_ctx (auto-shrink context to fit memory)
  - common/common.h:506-509                - load_mode (MMAP default), fit_params, fit_params_target
  - src/llama-mmap*.{h,cpp}                - mmap/mlock weight loading
  - Recent tessera streaming-weight work   - commits 02ac74294, 770bddee4, 18d3aaaf1, cf0c49fbf
                                              (streaming load for screening already landed)

- Prefill / batch memory
  - tools/server/server-prefill-policy.{h,cpp}  - per-iteration prefill cap (already landed)
  - server-context.cpp                     - integration point (pre_decode), batch construction

- MoE offload
  - docs/moe-disk-offload-study.md         - DESIGN ONLY, no code. WASTE-style expert offload to NVMe
                                              is studied but not implemented. Candidate region if MoE
                                              models are in scope.

## 2. Metric

Primary (MAXIMIZE): -peak_RSS.  negated so "higher is better" per the agent convention.
  Measured as maximum resident set size in bytes of the server process during a
  fixed inference workload. Lower RSS = better.

Secondary (informational, MAXIMIZE): tok/s during the same workload, so a memory
  win that destroys throughput is visible and can be ruled a regression if it
  falls outside an accepted band.

Correctness gate (must pass): the test-binary suite used by ctest for the
  server, plus a numerical check that sampled token logits match baseline within
  rtol. A faster-but-wrong kernel is a regression. Concretely: `ctest -R server`
  must pass, and a logit-diff harness against the baseline binary must report
  max_abs_delta below a fixed threshold on a held-out prompt set.

Workload (must be fixed across all candidates): one model, one prompt set, one
  concurrency level, one context length. Suggested default for the dry run:
  model = a small dense model present on disk (to be confirmed at Phase 4
  start), prompt set = 4 prompts of 2k tokens each at concurrency 4, context
  4096. The exact workload is frozen into this spec before the loop starts.

Peak-RSS measurement is noisy: average N>=5 runs per candidate; widen the tie
band accordingly.

## 3. Evaluator

Concrete commands (runnable from this repo with Bash):

  # correctness gate (fast, runs first - cascade)
  cmake --build build --target test-binaries -j
  ( cd build && ctest -R server --output-on-failure )

  # memory + throughput measurement (the metric)
  /usr/bin/time -l ./build/bin/llama-server \
      -m <model> -c 4096 -np 4 \
      --port 8799 < /dev/null &
  # drive it with the fixed prompt set via curl, then kill and parse
  # "maximum resident set size" from /usr/bin/time -l stderr.

Notes:
- gtime is NOT installed (brew install gnu-time would add it); /usr/bin/time -l
  on macOS does report peak RSS and is sufficient.
- A custom harness under tools/server/tests/ or pocs/ should wrap this so the
  prompt set, concurrency, and RSS parsing are deterministic. Building that
  harness is part of Phase 4 setup, not Phase 0+1.
- The logit-diff correctness harness against baseline also needs to be built at
  Phase 4 start.

## 4. Baseline commit

Tree is DIRTY (uncommitted tessera WIP across server/* and the new ggml-ane
backend). Baseline = git stash create snapshot, NOT a real commit:

  baseline_sha: f02cda5d37d9ff7c940ab77e082876555ec06591   (stash-create object)
  HEAD:         cf0c49fbf37520321c445c4e2cf1d2e58bf593dd

All worktrees branch off baseline_sha, never HEAD. This captures the current
tessera WIP as the starting point so evolved candidates build on top of the
in-flight work rather than the last public commit.

## 5. Budget

Default for a real run (NOT enforced in this dry Phase 0+1):
  - 12 generations per gene, OR
  - 90 minutes wall-clock per joiner, OR
  - 200 candidates evaluated across all agents,
  whichever comes first.

stagnation_limit = 8 generations without improvement -> gene freezes.
migration_interval = 4 generations.
max_genes per machine = min(cores/2, free_disk_GB / 8).

For the dry run, no evolution is performed; only this spec and research.md are
produced.
