# Multi-drafter plan: gemma-4-12B unified + 3 drafters, runtime extension

Triggered when: wave 5 (gemma hybrid-iswa paged fix) lands successfully AND the
MoE track (Qwen 35B pipeline) has shipped or been honestly diagnosed.

## The user's ask, decomposed

"get the tessera quantization pipeline to digest the gemma 4 12b unified model,
the gemma 4 mtp, dspark and dflash drafters and produce a single gguf and extend
the runtime to actually load all three drafters so we can study which drafters
work better in which scenarios and ways for us to maybe leverage all three
drafters during runtime"

This is FIVE sub-problems, not one. They must be done in order because each
gates the next.

## Sub-problem A: quantize the four inputs via the tessera pipeline

Inputs (current state on /Volumes/Julian T7/models/):
- gemma-4-12B unified target  -> many F16/Q5_K_M/Q6_K variants exist; the
  "unified" form is the MTP-bundled target. Needs a single canonical quantized
  artifact (probably Q5_K_M or Q6_K via the tessera pipeline).
- gemma-4 MTP drafter         -> gemma-4-12B-it-qat-unified-mtp-F16.gguf (24 GB).
  Needs tessera quantization.
- DSpark drafter              -> dspark_gemma4_12b_q4pure_v2.gguf (1.95 GB) EXISTS.
  May already be tessera-quantized (q4pure is a tessera quant type). Verify.
- DFlash drafter              -> gemma4-12B-dflash-BF16.gguf EXISTS in BF16.
  Needs tessera quantization.

This is FOUR pipeline runs (or fewer if some artifacts are reusable as-is).
The MoE track (currently running) is establishing the pipeline-driving flow on
Qwen 35B; that knowledge carries over directly. Each drafter is small (1-2 GB)
so the pipeline is fast on these; the target is 12B so slower but not 35B-slow.

## Sub-problem B: pack into a single GGUF

Tessera's MTP bundling already does this for one drafter (gemma-4-12B-it-qat-
unified-mtp-F16.gguf embeds the MTP). Extending to three drafters in one GGUF
needs:
- A new GGUF convention for the multi-drafter bundle (likely a per-drafter
  tensor prefix, e.g. `dflash.*`, `dspark.*`, `mtp.*`).
- Changes to the GGUF writer (tessera-archive.{h,cpp}) and reader
  (llama-model-loader.cpp).
- Metadata keys declaring which drafters are present.

Risk: this is a non-trivial format change. Alternative: keep drafters as
separate files and load them by path list (much simpler, no format change).
DECISION POINT for the user: bundle into one GGUF (more work, cleaner
distribution) or accept a path-list manifest (less work, ships faster).
Recommend asking before implementing.

## Sub-problem C: extend the runtime to load all three drafters

Current state (verified in the code):
- common_params_speculative_draft has ctx_tgt, ctx_dft, ctx_mtp - TWO drafters
  max (the hybrid path: DFlash + MTP).
- COMMON_SPECULATIVE_TYPE_DRAFT_HYBRID exists ("DFlash and MTP arbitration").
- common_speculative_impl_draft_dflash (common/speculative.cpp:942) handles
  both DFlash and DSpark via an `is_dspark` flag - they share an impl.
- common_speculative_impl_draft_mtp (common/speculative.cpp:1274) handles MTP.

The extension: add a third drafter slot (ctx_dspark or generalize the three
slots to a vector<ctx>) and extend the hybrid arbiter to consider all three.
Files:
- common/common.h           - common_params_speculative_draft struct
- common/speculative.{h,cpp} - the arbiter impl (~2900 lines, the meat)
- common/arg.cpp             - CLI flags
- tools/server/server-task.cpp, server-context.cpp - server wiring

Risk: this is real C++ runtime surgery on a hot path. The arbiter logic decides
which drafter to consult per token; adding a third is conceptually clean but
touches the speculative-correctness invariants. Strong correctness gate needed.

## Sub-problem D: benchmark which drafter wins where

Once C lands, run each drafter alone AND in hybrid combos across scenarios:
- Code generation (DFlash/DSpark typically strong)
- Math/reasoning (MTP may differ)
- Long-context (different KV economics)
- Different batch sizes / concurrency
Metrics: acceptance rate, tokens-per-second, wall time, peak RSS.
This is a measurement task, not an evolution task - run AFTER C is stable.

## Sub-problem E: explore runtime leveraging of all three

Given D's data, explore:
- Per-prompt routing (pick drafter by prompt class)
- Per-token arbitration (the hybrid arbiter, extended)
- Ensemble (consult multiple, vote)
This is research - likely a follow-up evolution wave after D produces data.

## Proposed wave structure

WAVE 6 (after wave 5): quantize the drafters (sub-problem A). Single gene.
  Reuses the MoE track's pipeline knowledge. Produces 3-4 quantized artifacts.
  This is the "digest" part of the ask.

WAVE 7: runtime extension (sub-problem C). SINGLE gene, focused, real C++ work.
  Adds the third drafter slot + extends the hybrid arbiter. Hard correctness
  gate: speculative decoding must not break (existing tests + acceptance-rate
  parity with single-drafter on a fixed workload).

DECISION GATE before wave 7: ask the user bundle-vs-manifest (sub-problem B).
  If manifest, wave 7 is simpler. If bundle, add a wave 7a for the GGUF format
  change first.

WAVE 8: benchmarking (sub-problem D). Not evolution - a measurement script +
  matrix run. Produces the data table.

WAVE 9 (optional): ensemble/routing exploration (sub-problem E). Evolution wave
  using wave 8's data as the evaluator.

## Preflight data captured (for the agent that runs these)

- Tessera already supports DFlash + DSpark in one impl (is_dspark flag at
  common/speculative.cpp:970). They are not truly "two drafters" - they are one
  drafter type with a flag. So "three drafters" really means MTP + (DFlash XOR
  DSpark) + the third, OR extending to MTP + DFlash + DSpark as truly distinct.
  Clarify which interpretation the user wants.
- The dflash.cpp TODO at line 42 says only Qwen3 backbones are supported for
  DSpark, but a Gemma4 DSpark EXISTS on disk (dspark_gemma4_12b_q4pure_v2.gguf).
  Reconcile this - either the TODO is stale or the Gemma4 DSpark is experimental.
- The MTP-unified target variants are large (24-32 GB each F16); the canonical
  quantized target for this work should be picked (Q5_K_M telemetry at 8 GB is
  the existing one used in earlier waves).

## Open questions for the user (ask before dispatching wave 6+)

1. Bundle-into-one-GGUF vs manifest-of-paths? (sub-problem B decision)
2. "Three drafters" = MTP + DFlash + DSpark as three distinct contexts, or
   DFlash/DSpark as one switchable drafter plus MTP? (the is_dspark flag
   question above)
3. Which canonical target quantization? (Q5_K_M telemetry, Q6_K calibration,
   or fresh from this pipeline run?)
4. Priority order if compute is tight: quantize-first (wave 6) is the safe
   starting point regardless.
