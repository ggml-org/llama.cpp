# MoE track: drive the tessera quantization pipeline on Qwen3.6-35B-A3B

## Objective (read this first - it is NOT inference optimization)
The tessera quantization pipeline (tools/quantize/tessera/, entry point compiled
into build/bin/llama-quantize) was used to produce the existing
Qwen3.6-35B-A3B-Tile640-AWQ-unified.gguf artifact on /Volumes/Julian T7/models/.
Your job is to drive this pipeline end-to-end on the f16 source to produce a
fresh AWQ-unified artifact for this MoE model, the same way it was done for the
8B model. The goal is the QUANTIZED ARTIFACT, not inference speed/memory.

## The pipeline (reverse-engineer the actual flow before running)
tools/quantize/tessera/tessera-quant.cpp is compiled into llama-quantize
(tools/quantize/CMakeLists.txt:53). The recent tessera commits tell the story:
- 09fde9d40 tessera : GA early termination + family warm-start
- 4eb19f695 tessera : reuse screening data across layers via family warm-start
- 02ac74294 tessera : streaming weight loading to fix OOM on 16GB systems  <- this is why it fits on 16GB
- 7af655a4d tessera : fix segfault + accept any source GGUF type
- 0449cfdbe tessera : parallel candidate eval (serial layers, shared weights)
- 18f871ef1 tessera : Metal GPU acceleration for the quantize pipeline
- d0ba47b49 tessera : DuckDB persistent pipeline store
- cf0c49fbf tessera : streaming acceptance gate (load weights on-demand)

The DuckDB store (tessera-quantize-db.{h,cpp}) means pipeline state persists
across runs - a previous run on this model may have left state you can resume
from. CHECK the DB before starting from scratch.

## Existing artifacts on /Volumes/Julian T7/models/ (the target lineage)
- Qwen3.6-35B-A3B-f16.gguf  <- the 66 GB source (f16)
- Qwen3.6-35B-A3B-Tile640-AWQ.imatrix.gguf             (raw imatrix)
- Qwen3.6-35B-A3B-Tile640-AWQ-refined.imatrix.gguf     (calibrated)
- Qwen3.6-35B-A3B-Tile640-AWQ-refined.imatrix.gguf.at_16/.at_24/.at_32/.at_40  (per-stage screening)
- Qwen3.6-35B-A3B-Tile640-AWQ-refined.interrupted.imatrix.gguf  (a prior run was interrupted)
- Qwen3.6-35B-A3B-Tile640-AWQ-refined.preoptimized.imatrix.gguf
- Qwen3.6-35B-A3B-Tile640-AWQ-unified.gguf  <- the existing final artifact (12 GB)
These artifacts ARE the pipeline's intermediate outputs. Reuse them where
possible - re-running screening from scratch on a 66 GB source takes hours.

## First steps (do these before any heavy compute)
1. Read tools/quantize/tessera/tessera-quant.cpp (1247 lines) end to end - find
   the actual CLI / subcommand / flag that triggers the AWQ pipeline (it is NOT
   the standard --help flags, which are the upstream llama-quantize interface).
   Look for env vars (TESSERA_*), subcommands, or behavior keyed on the
   qwen35moe arch.
2. Check tools/quantize/tessera/CMakeLists.txt and the built binaries
   (build/bin/llama-quantize, tessera-train-lk, test-tessera-*) for the entry.
3. Look for the DuckDB store file (find . -name "*.duckdb" or similar) and read
   its state - the prior run may be resumable.
4. Read the existing artifact metadata (use build/bin/gguf-dump or the
   llama-quantize --imatrix path) to understand what calibration data was used.

## Machine + disk constraints (real)
- 16 GB RAM Apple Silicon. The pipeline's streaming-weight-load (02ac74294) is
  what makes a 66 GB source fit; rely on it.
- Disk: /Volumes/Julian T7 has ~60 GB free (the source alone is 66 GB and
  already lives there). Local disk has ~67 GB free. Intermediate artifacts are
  multi-GB each. You may need to point intermediates at local disk and stream
  the source from the T7, or clean up stages as you go.
- The source is on an external drive - I/O bound. The pipeline already has
  mmap-the-input-GGUF (5b566f919) for this.

## Baseline
- sha: bbfc3493d (latest review tip; the tessera pipeline code is in tools/quantize/tessera/
  which is mostly untracked WIP - verify it builds)
- Build: cmake --build build --target llama-quantize -- -j8

## Mechanics
- This is a pipeline-DRIVING task, not a deep multi-gen evolution. Single gene
  is fine; the "candidates" are mostly: figure out the invocation, resume from
  DuckDB state if possible, drive the stages, produce the artifact.
- Budget: generous - this is a long-running pipeline. 60 min OR producing the
  artifact OR a clear diagnosis that it can't fit, whichever comes first. If
  you need more time, ship where you got and what's left.
- One worktree off bbfc3493d. ASCII only. Commits on evolve/moe-qwen35b/* only.
  NEVER master/main. Never push, never gh.
- SERIALIZE all heavy I/O against the other running agent (wave 5).

## Success criteria
- IDEAL: a fresh Qwen3.6-35B-A3B-Tile640-AWQ-unified.gguf artifact is produced,
  loadable by llama-bench (verify with pp16 -n 1).
- PARTIAL: pipeline advances a stage or two and you document the exact remaining
  steps + the resume command.
- FAILURE: clear diagnosis of why it can't run (disk, RAM, missing wiring).

## Output contract
- review branch evolve-review/moe-qwen35b off bbfc3493d (only if you made code
  changes to drive the pipeline; if you only ran it, no branch needed).
- Final message: did the artifact get produced? (yes/partial/no), the EXACT
  invocation(s) used, the stages completed, where the artifact landed, any
  tessera pipeline bugs/quirks you hit, and the DuckDB state for resume.

Be honest. A claimed artifact that doesn't load is worse than an honest "got
3 of 5 stages, here's the resume command." Begin.
