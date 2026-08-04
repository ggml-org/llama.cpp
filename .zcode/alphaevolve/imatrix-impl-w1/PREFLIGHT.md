# imatrix implementation wave 1 - phases 1+2 of the study

## The study (READ FIRST)
/Users/user/Developer/GitHub/tessera/.zcode/alphaevolve/imatrix-study/study.md
This wave implements Phase 1 and (if budget allows) Phase 2 of section 5. Read
the full study; this preflight only captures the must-know facts and constraints.

## Phase 1 scope (verify the cheap wins - 1-2 days of work)
1. 3.2 mmap verify: confirm whether common_init_from_params mmap's the model by
   default on macOS. The study could NOT confirm this from source alone - it
   needs a runtime RSS measurement. DO THIS FIRST so the rest of phase 1 knows
   whether mmap is already on.
2. Q3 wire max_abs: add max_abs accessor to tools/quantize/tessera/tessera-imatrix.cpp;
   thread per-channel max into ts_regime_compute_descriptor and the DartQuant/CHAMP-Q
   expert profiles. CONSUMER-ONLY change; data already on disk. The study found this
   is the single cheapest quality win.
3. Q5 corpus sampling: add --calib-sample stride=N flag, plus a multi-file corpus
   mode that interleaves documents.
4. D2 eagle3 shorthand: trivial CLI add.

## Phase 2 scope (only if Phase 1 ships with budget remaining - 3-5 days of work)
1. 3.9 streaming reduction: refactor collect_graph_observers/reduce_graph_observers
   in tools/imatrix/imatrix.cpp to do per-row accumulation during tensor-get,
   eliminating the staging arena. MB -> KB. Bit-exact output preserved.
2. 3.4 Metal observer-op fusion: extend the imatrix observer ggml op to reduce
   on-GPU. Fall back to CPU on dispatch failure.
3. 3.5 vDSP + sharded map: add vDSP_svesq/vDSP_maxmgv to the reduction; shard
   m_stats by name hash.

## Critical correction from the study (READ)
The study CORRECTED an earlier misread: tools/imatrix/imatrix.cpp line 1256
(no_alloc=false) is the SAVE path context (writing outgoing stats), NOT the model
load. Don't repeat the mistake. The mmap question for the model load is open and
must be settled by RSS measurement (item 1.2 above), not source grep.

## Files (read each before editing)
- tools/imatrix/imatrix.cpp (2975 lines) - the producer
- common/imatrix-loader.{h,cpp} - the data shape (sums, abs_sums, fourth_sums, max_abs, counts)
- tools/quantize/tessera/tessera-imatrix.cpp (352 lines) - the consumer / regime stats
- tools/quantize/tessera/tessera-regime.{h,cpp} - the regime descriptor
- common/arg.cpp (~line 4148) - --tessera-* flag parsing
- common/tessera-args.h - common_tessera_params struct

## Baseline
- sha: 10222c950 (main). Branch your worktree off THIS.
- Build: cmake --build build --target llama-quantize -- -j8

## CRITICAL resource constraint
- The MoE quantize pipeline (wave 6) is running and using most of the 16 GB RAM.
- A UX-implementation agent is also running concurrently (Swift edits only).
- DO NOT run the imatrix tool on a real model for the mmap RSS measurement if it
  would contend with wave 6. Instead, either:
  (a) defer the RSS measurement to the end after wave 6 finishes, or
  (b) use a TINY model (the qwen 0.5B) with a tiny corpus for the measurement.
- For build verification: `cmake --build build --target llama-quantize` is CPU-only,
  safe to run concurrently. Just don't run llama-quantize itself on a big model.
- NEVER run two llama-quantize processes concurrently.

## Mechanics
- Single gene. Budget: 60 min OR Phase 1 + as much Phase 2 as fits, whichever first.
- One worktree off 10222c950. ASCII only (no em-dash, no unicode arrows).
- Commits on evolve/imatrix-impl-w1/* only. NEVER master/main. Never push, never gh.
- Never weaken a test or assertion to pass.

## Build + correctness verification
- cmake --build build --target llama-quantize must succeed.
- For the Q3 max_abs change: there's a test_quantize_db and other tests under
  tools/quantize/tessera/test_*.cpp; run any that exercise the regime path.
- The mmap RSS measurement: if you can run it safely (tiny model or after wave 6),
  capture before/after RSS numbers. If you can't, document it as "needs measurement".

## Output contract
- review branch evolve-review/imatrix-impl-w1 off 10222c950.
- Run artifacts: .zcode/alphaevolve/imatrix-impl-w1/{gene-ledger.json, changes.md, best.md, integration/patches/g1.patch}
- Final message: phase 1 sub-items landed (with mmap verdict), phase 2 sub-items
  landed if any, build status, RSS measurement result if obtained, what you skipped,
  bugs/quirks in the imatrix producer.

Be honest. A claimed build pass that doesn't reproduce is worse than an honest
"source done, build couldn't verify because X". Begin.
