# Wave 6 preflight: tessera-quantize the multi-drafter inputs

## Objective
Quantize the 4 inputs for the upcoming multi-drafter bundle (wave 7 = GGUF format,
wave 8 = runtime + disagreement-aware ensemble arbiter). Wave 6 is PURE PIPELINE
DRIVING - no format changes, no runtime changes. Just produce 4 tessera-quantized
artifacts via the existing tessera pipeline.

## The pipeline (reverse-engineered by the MoE-track agent - reuse this knowledge)
The tessera AWQ pipeline is triggered by passing the pseudo-ftype TESSERA_T640 to
the standard llama-quantize binary:
  ./build/bin/llama-quantize \
    --tessera-imatrix <imatrix.gguf> \
    --tessera-awq-alpha 0.5 \
    --tessera-evolve-iters 4 --tessera-evolve-population 8 --tessera-evolve-islands 2 \
    --progress-file <path.jsonl> \
    <source-f16.gguf> <out-tessera.gguf> TESSERA_T640

Flow: tools/quantize/main.cpp -> quantize.cpp:llama_quantize (detects TESSERA_T640
around line 969/1012, sets use_tessera=true) -> ts_dispatch_run at
tools/quantize/tessera/tessera-dispatch.cpp:1040. The --tessera-* flags are parsed
by common/arg.cpp:4148+ into common_tessera_params.
- Streaming weight loading is built in (commit 02ac74294) - fits 16GB.
- mmap's the source GGUF (no_alloc=true).
- The pipeline writes the entire output via a single gguf_write_to_file at the
  END - no incremental checkpointing. A killed process loses quantize progress.
- DuckDB store only persists GA results + family warm-start seeds, NOT the
  long quantize phase. Optional (--quantize-db).
- Silent by default; use --progress-file for NDJSON observability.
- Read /Users/user/Developer/GitHub/tessera/.zcode/alphaevolve/moe-qwen35b/run-state.md
  for the MoE agent's full notes on the pipeline.

## Baseline
- sha: 10222c950 (current main). The tessera pipeline is in main. Build if needed:
  cmake --build build --target llama-quantize -- -j8 (already built and verified).

## The 4 inputs (decide per-input whether tessera quant is needed)
1. **gemma-4-12B unified target + MTP drafter** (BUNDLE - one file holds both):
   /Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-F16.gguf (23 GB, F16)
   This is the only f16 source for the main target AND the MTP drafter together.
   Needs tessera quantization. Output: tessera-quantized unified bundle preserving
   the target+MTP structure. Use existing imatrix if present on T7 (look for
   *unified-mtp*.imatrix*); otherwise the pipeline will compute one.

2. **DFlash drafter**: /Volumes/Julian T7/models/gemma4-12B-dflash-BF16.gguf (1.4 GB)
   - BUT: /Volumes/Julian T7/models/drafters/ has PRE-QUANTIZED variants:
     gemma-4-12B-it-DFlash-{IQ4_XS,Q4_K_M,Q5_K_M,Q6_K,Q8_0}.gguf. VERIFY whether
     these are already tessera-quantized (check metadata via llama-bench or
     gguf-dump). If yes, pick one (Q5_K_M or Q6_K) and skip re-quantizing.
     If no (standard k-quants), tessera-quantize from the BF16 source.

3. **DSpark drafter**: /Volumes/Julian T7/models/drafters/dspark_gemma4_12b_q4pure_v2.gguf
   Already "q4pure" (likely tessera's pure quant, not AWQ). VERIFY the metadata.
   If q4pure is the desired tessera format, this is DONE - just copy/validate.
   Otherwise re-quantize via TESSERA_T640.

4. (There is no separate "MTP drafter" file - it's INSIDE input #1, the unified bundle.)

So the actual work is likely: 1 big quantize (the 23GB unified bundle) + verification
of the existing dflash/dspark artifacts. May be smaller than 4 full runs.

## CRITICAL: avoid RAM contention with the running MoE pipeline
- The MoE quantize (pid 89237) is running, ~435/753 tensors, ETA ~20 min from
  dispatch. Peak RSS ~2.3 GB but it hammers CPU and disk I/O against the T7.
- DO NOT start the big 23GB unified quantize until pid 89237 finishes. Check
  ps -p 89237 first; if alive, do the small drafters first OR wait.
- NEVER run two tessera pipeline instances concurrently.

## Output placement
- Tessera outputs land on the T7 (or /tmp if T7 is full, then move). T7 has
  ~170 GB free after cleanup.
- Naming convention: <base>-tessera-T640.gguf to distinguish from existing k-quants.

## Mechanics
- Single gene. This is pipeline-driving + verification, not deep evolution.
- Budget: 60 min OR all 4 (or as many as needed) artifacts produced/verified.
- One worktree off 10222c950 (only if you need code changes - pure pipeline
  driving needs no worktree). ASCII only. Commits on evolve/multi-drafter-w6/*
  only. Never master/main. Never push, never gh.
- SERIALIZE all heavy I/O against the UX-study agent and the MoE pipeline.

## Success criteria
- IDEAL: 4 tessera-quantized artifacts ready for the wave-7 bundler:
  (1) gemma-4-12B unified (target+MTP) tessera-quantized
  (2) DFlash tessera-quantized (or verified-already)
  (3) DSpark tessera-quantized (or verified-already)
  (4) each loadable by llama-bench -m <artifact> -p 16 -n 1
- PARTIAL: some artifacts produced/verified, others documented as remaining.
- For each, capture: the exact pipeline invocation, the imatrix used (existing
  or computed), the output path + size, and a llama-bench load verification.

## Output contract
- review branch evolve-review/multi-drafter-w6 off 10222c950 (only if code changes).
- Run artifacts in .zcode/alphaevolve/multi-drafter-w6/ (gene-ledger.json, changes.md,
  best.md, integration/patches/ if any code).
- Final message: per-input verdict (quantized / already-tessera / partial / failed),
  exact invocations, output paths + sizes, load verification for each, total time,
  bugs/quirks in tessera's handling of the drafter model archs (dflash/dspark).

Be honest. A claimed artifact that doesn't load is worse than an honest report.
Respect the budget. Begin.
