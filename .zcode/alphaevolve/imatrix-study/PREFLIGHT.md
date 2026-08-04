# imatrix calibration study - orientation facts

## The three parts of the user's question
1. What does the current imatrix calibration actually do, and where are the
   quality/algorithmic opportunities to get BETTER calibration data out of it?
2. Can the temporal/spatial/memory optimizations from the tessera quantization
   pipeline be applied to the imatrix producer?
3. (Implicit) How does this feed into the multi-drafter work, since drafter
   training (DFlash/DSpark) consumes offline feature capture from the imatrix
   tool (--features-out)?

## The architecture (verified by the orchestrator)
There are THREE distinct imatrix pieces - do not confuse them:

1. tools/imatrix/imatrix.cpp (2975 lines) - the PRODUCER. The upstream
   llama.cpp IMatrixCollector wrapped with tessera extensions. Runs inference
   over a calibration corpus, collects per-tensor activation stats via the
   imatrix_observers hook (set_imatrix_observer_filter/scope in llama-context).
   This is what generates calibration data.

2. common/imatrix-loader.{h,cpp} - the on-disk DATA SHAPE. Per-tensor entries
   with: sums, abs_sums, fourth_sums, max_abs, counts (4 moments + max).
   Plus datasets/chunk metadata.

3. tools/quantize/tessera/tessera-imatrix.cpp (352 lines) + tessera-mm-imatrix.cpp
   (416 lines) - the CONSUMERS. npz readers + regime stats (mean/variance/
   skewness/kurtosis derived from the moments). mm-imatrix is the per-modality
   (text/image/audio) variant.

## What the producer already does (tessera extensions on top of upstream)
Read tools/imatrix/imatrix.cpp end-to-end, especially:
- IMatrixCollector class (line ~88): the core accumulator
- struct Stats: values, abs_values, fourth_values, max_values, counts (4 moments + max)
- struct tensor_statistics: richer per-tensor stats (total_sqract, stddev, active,
  entropy, zd, cossim) - understand what these capture and how they're used
- observer_transfer_state (line ~77): previous_moments, previous_counts, signature
  - this is for INCREMENTAL/streaming collection across chunks
- The spec-decoding calibration mode (--model-draft --spec-steps 64):
  calibrate WITH a drafter in the loop
- Offline trunk-feature capture (--features-out --feature-layers 0,15,31):
  per-layer hidden states for DFlash drafter training (Path 1 in some doc)
- Recent commits to read:
  - b4c0ac4ac imatrix : overlap feature-capture windows for contiguous output
  - 25f6c1cab imatrix : skip per-chunk warmup in feature capture
  - 1124be83f imatrix : add offline trunk-feature capture for DFlash drafter training
  - cb616cc56 quantize : add opt-in --prior-weight for neutral imatrix prior
  - 8aaf980f2 (merge) tessera/imatrix-neutral-prior

## THE KEY FINDING (verified): the producer uses NONE of the tessera pipeline's optimizations
The tessera quantize pipeline (tools/quantize/tessera/) has ~20 commits of
optimization. The imatrix producer uses NONE of them. Concrete evidence:
- imatrix.cpp line 1256: no_alloc = false (EAGER allocation, opposite of tessera's
  streaming no_alloc=true approach)
- No mmap of the input model
- No streaming weight loading
- No Metal acceleration for the calibration forward passes
- No sharded eval_ctx / parallel candidate evaluation
- No BLAS/vDSP fusion for the stat accumulation

The tessera pipeline optimizations to evaluate for porting (each is a commit):
- 02ac74294 streaming weight loading to fix OOM on 16GB systems
- 5b566f919 mmap the input GGUF (no_alloc=true)
- 0449cfdbe parallel candidate eval (serial layers, shared weights)
- 18f871ef1 Metal GPU acceleration for the quantize pipeline
- 97f757843 vDSP fusion, sharded eval_ctx, parallel screening/acceptance
- 45eeab7b2 FUSE C + cache-blocked dequant + AWQ grid batch
- ccf9fa803 BLAS-accelerate B7 optimizer matmuls and linalg primitives
- 7c6d85681 BLAS-accelerate GA fitness matmul + per-tensor thread pool
- 770bddee4 streaming MSE fitness (132 KB scratch vs 700 MB per candidate) - the
  "spatial memory optimization" pattern, directly relevant to imatrix stat RAM
- 18d3aaaf1 streaming MSE for acceptance gate + L5 A/B comparison
- d0ba47b49 DuckDB persistent pipeline store - could imatrix runs persist/resume?

## Calibration quality opportunities (the "better calibration data" part)
Beyond performance, survey what state-of-the-art imatrix/importance-matrix work
looks like. Research questions:
- Is collecting 4 moments + max enough? Some calibration work uses full
  per-channel histograms, Hessian diagonals, or Fisher information. What would
  the quality/compute tradeoff look like?
- The spec-decoding calibration mode (--model-draft) - is this producing
  calibration matched to the drafter's distribution? How could it be better?
- Chunk-based collection (chunk_size = n_ctx / n_parallel) - does this miss
  long-range activation patterns? Would whole-sequence collection help?
- The neutral prior (--prior-weight) - what problem does it solve, and is there
  a better solution (e.g. importance-weighting the corpus)?
- Offline trunk-feature capture for drafter training - is the feature set
  (per-layer hidden states) optimal? EAGLE-style capture exists; what else?

## Research peer work (web research encouraged)
- Original llama.cpp imatrix PR/discussions (ggerganov/llama.cpp#4861 era)
- AWQ paper (Lin et al. 2023) - activation-aware weight quantization
- GPTQ, EXL2, OmniQuant, SpinQuant - other activation-aware methods and their
  calibration data requirements
- Higgs (recent) - how it uses activation stats
- EAGLE / EAGLE-2 / EAGLE-3 - drafter feature capture patterns
- Latest 2024-2026 work on calibration data quality for LLM quantization
  (Datafree, AnyQuant, etc.)

## Constraints
- DO NOT do heavy compute. Multiple agents are running: the MoE quantize pipeline
  (pid 89237), the UX study, and wave-6 multi-drafter quantization. 16 GB RAM is
  contended. This is a READ + RESEARCH + DESIGN task, no builds, no model loads.
- DO NOT edit source. Output is a study document, not code changes.
- ASCII only in output.

## Baseline
- sha: 10222c950 (current main). Read the source from this tree.

## Honest scope
This is a research + design study. The output is a single substantial document
that the user (who is deciding whether to fund a follow-up optimization wave)
can use to decide: which optimizations port over, which quality improvements are
worth pursuing, and what the priority order is. Do not implement - that is a
separate future wave using your document as its spec.
