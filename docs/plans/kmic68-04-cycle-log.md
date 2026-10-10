# P04 - Speculative cycle log and phase profile

**Kind:** instrumentation
**Depends on:** none
**Supplies:** cycle data for P06 and P10; P07 consumes it through P06

## Context

P06, P07, and P10 need the same unit of evidence: one selected MTP draft cycle
for one sequence, from the start of drafting through target verification and
sampling. P04 keeps logging and profiling together because they share that
boundary, but separates their costs. `LLAMA_SPEC_LOG` records cycle data
without forcing a device synchronization. `LLAMA_SPEC_PROFILE` adds explicit
phase-boundary synchronization so wall times are comparable.

The log is append-only JSONL. A row's `top1_probabilities` covers every attempted
candidate, including any probability-stop candidate and attempts later removed
from the selected proposal; `drafted_count` counts only final selected tokens.
Timing rows identify whether they are synchronized.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`, changes
to `common/speculative.cpp` for `LLAMA_SPEC_LOG`, `LLAMA_SPEC_PROFILE`,
`spec_log_rec`, and phase counters. The source implementation is evidence, not
a direct transplant: its per-token `if (spec_log)` violates this plan's
disabled-path contract.

## EARS requirements

- **R04.1 (State-driven):** WHILE `LLAMA_SPEC_LOG` names a writable path, the speculative-cycle log writer shall append exactly one JSON object for each completed MTP sequence cycle.
- **R04.2 (Ubiquitous):** The speculative-cycle JSON row shall contain schema version, launch/request/proposal/sequence/cycle identity, separate draft and verify phase identities with their participant identities/counts and timing scopes, starting position, draft wall time, verify-and-sample wall time, per-sequence total cycle and inter-phase wait times, drafted count, attempted count, replay-adjusted accepted count, selected state, timing mode, per-phase synchronization overhead, and every evaluated token's top-1 probability.
- **R04.3 (State-driven):** WHILE `LLAMA_SPEC_PROFILE` is enabled, the speculative-cycle profiler shall synchronize the draft and target contexts only at the defined phase boundaries.
- **R04.4 (Event-driven):** WHEN at least 64 unique profiled draft phases, 64 unique verify phases, and 64 sequence cycles complete, the speculative-cycle profiler shall report mean timings with a separate denominator for each scope.
- **R04.5 (State-driven):** WHILE both `LLAMA_SPEC_LOG` and `LLAMA_SPEC_PROFILE` are absent, `common_speculative_impl_draft_mtp` shall execute a compile-time no-record token loop with no per-token instrumentation branch.
- **R04.6 (State-driven):** WHILE `LLAMA_SPEC_PROFILE` is absent, speculative-cycle instrumentation shall issue no context synchronization.
- **R04.7 (Event-driven):** WHEN a top-1 candidate triggers the MTP probability stop, the MTP cycle recorder shall retain that candidate's probability as the row's final attempted-probability entry.
- **R04.8 (Unwanted behaviour):** IF the configured cycle log cannot be opened, appended, or flushed, THEN the speculative-cycle log writer shall fail the instrumented run instead of silently dropping a row.
- **R04.9 (Unwanted behaviour):** IF a pending proposal identity does not match its sequence at completion, THEN the speculative-cycle log writer shall reject the row instead of joining data from different proposals.
- **R04.10 (State-driven):** WHILE multiple sequences share one draft or verify GPU phase, every participating sequence row shall reference the same shared phase identity and wall-time measurement.
- **R04.11 (Event-driven):** WHEN all MTP `n_min` clearing and common `n_max` clamping are complete, the cycle recorder shall snapshot the immutable final selected proposal and its drafted count.
- **R04.12 (State-driven):** WHILE a completed proposal is unselected, its cycle row shall store zero drafted and accepted counts.

## Approach

1. Add `common_speculative_cycle_record` and narrow query/finish APIs to
   `common/speculative.h`. Keep per-sequence candidate attempts inside
   `common_speculative_impl_draft_mtp`, but assign a fresh monotonic
   `proposal_id` before each draft attempt.
2. Parse `LLAMA_SPEC_LOG` and `LLAMA_SPEC_PROFILE` once during speculative
   initialization. Select the instrumented or non-recording implementation once
   outside the token loop. The disabled specialization performs no `getenv`,
   logger-state test, mutex, probability push, or synchronization per token.
3. The recording specialization captures attempted token ID, top-1 probability,
   and emitted/stopped state. After every drafter returns, finalize the proposal
   in `common_speculative_draft` only after all `n_min` clears, implementation
   selection, and common `n_max` resizing. Keep the complete attempted-record
   stream for `top1_probabilities` and `attempted_count`; truncate only the
   separate emitted/proposal-aligned records with the final result and snapshot
   immutable `drafted_count=result.size()`. A cleared or unselected attempt
   retains all diagnostic attempted records but has
   `selected=false`, `drafted_count=0`, and `accepted_count=0`.
4. Preserve the immutable selected snapshot and proposal ID across checkpoint
   replay even when `slot.spec_draft` is replaced by accepted tokens plus a
   correction. Final server statistics and log reconciliation always refer to
   the original finalized proposal, not the replay vector.
5. Assign separate `draft_phase_id` and `verify_phase_id` to their actual
   invocations. Each phase owns its participant proposal/sequence identities,
   participant count, `timing_scope`, wall time, and synchronization overhead;
   draft and verify participant sets may differ. Rows referencing the same phase
   must agree on all its metadata. Shared times use `timing_scope=shared_batch`
   and are never exclusive per-sequence costs or summed across duplicate rows.
   A one-slot phase uses `timing_scope=single_sequence`. An unselected proposal
   with no verification has a null verify reference/time, not a fabricated phase.
6. Start cycle/draft timing immediately before the shared MTP draft phase. With
   profiling enabled, synchronize `ctx_dft` at draft boundaries and measure
   synchronization time. Logging alone records unsynchronized host wall time.
7. In `tools/server/server-context.cpp`, carry the selected proposal into its
   `server_slot`. Measure the shared verify-and-sample invocation from target
   decode through sampling, rollback, replay adjustment, and server-stat update.
   Its start does not inherit any participant's individual draft-completion
   timestamp. Measure each sequence cycle separately from its draft start to
   its completion, including scheduling waits; record `inter_phase_wait_us` from
   draft end to verify start. Synchronize `ctx_tgt` only when profiling.
8. Keep the pending snapshot across the earlier `common_speculative_accept`
   call. Finalize after `spec_is_replay` adjusts `n_accepted` and
   `slot.stats.n_draft_accepted` is updated. Store that adjusted value only on
   the selected row; close stale/unselected proposals with zero reconciled counts.
9. Deduplicate draft and verify means independently by their respective phase
   IDs; compute total-cycle means by sequence-cycle identity. Report counts and
   synchronization means for each scope, never sum differently scoped means.
   Only compute the additive phase residual for one-to-one, single-sequence,
   non-overlapping draft/verify pairs; otherwise report it as unavailable. Emit
   after each phase kind and sequence-cycle sample has at least 64 identities.
10. Add `scripts/perf/verify-spec-cycle-log.py`. It must start an uninstrumented
   cache-warm process, rotate the log, then start an instrumented Qwen4Exp MTP
   process. Assert the initialization log names
   `common_speculative_impl_draft_mtp` before sending requests. Reconcile the
   first six request IDs, then continue profile-only requests until at least 64
   unique phases of each kind and 64 sequence cycles have completed.

## Critical files & anchors

- `common/speculative.h` - new narrow cycle-record query and completion contract.
- `common/speculative.cpp:1576-1680` - `common_speculative_impl_draft_mtp` state and initialization.
- `common/speculative.cpp:2313-2415` - non-chained MTP host sampling loop and probability stop.
- `common/speculative.cpp:2438-2451` - MTP `accept` receives the accepted count.
- `tools/server/server-context.cpp:3424-3435` - common draft call and per-slot draft handoff.
- `tools/server/server-context.cpp:663-701` - target verification batch assembly.
- `tools/server/server-context.cpp:4280-4377` - target sampling, rollback, replay-adjusted acceptance, statistics update, and final cycle-record completion.
- `tools/server/server-common.cpp:99-104` - response fields `draft_n` and `draft_n_accepted`.
- `scripts/perf/bench_spec.py:335-355` - existing response-stat extraction.
- `scripts/perf/verify-spec-cycle-log.py` - planned six-request reconciliation tool.

## Verification

Build normally; the feature is runtime opt-in. On the A770, before timing, stop
`llama-sycl.cpp.service`, verify sole tenancy on `/dev/dri/renderD128`, name
the kernel driver and build, use the persistent AOT cache, and apply the
post-run two-driver fault gate. Then run:

```bash
timeout 2400 python3 scripts/perf/verify-spec-cycle-log.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --repeats 2 --ctx 16384 --spec-args='--spec-type draft-mtp --spec-draft-n-min 1 --spec-draft-n-max 7 --no-spec-draft-backend-sampling' --assert-impl common_speculative_impl_draft_mtp --min-profile-cycles 64 --log /tmp/p04-spec-cycles.jsonl --profile
```

The verifier enables `LLAMA_SPEC_LOG=/tmp/p04-spec-cycles.jsonl` and
`LLAMA_SPEC_PROFILE=1` only after the warmup process exits, rotates any old
file, and assigns the measured launch request IDs. Expected evidence:

- exactly six reconciliation requests complete after the separate
  uninstrumented warmup, and the initialization assertion proves MTP ran;
- every started proposal/sequence identity appears exactly once;
- summed `accepted_count` and `drafted_count` over `selected=true` rows
  equal server `draft_n_accepted` and `draft_n`; unselected rows contribute
  zero to both sums;
- `n_min` clear, common `n_max` clamp, unselected implementation, and partial
  replay cases retain the finalized immutable proposal semantics;
- a fixture with draft participants `{A,B}` and verify participants `{A,C}`
  preserves distinct phase IDs, membership, counts, times, and synchronization
  overhead; each phase contributes once to its own aggregate despite repeated
  row references, and unselected proposals have no verify phase;
- each probability-stop row ends with a non-emitted attempted probability;
- probability stops followed by `n_min` clearing, `n_max` clamping, or loss of
  implementation selection retain the entire attempted stream and its final
  stopped candidate while only the proposal-aligned records change;
- for the one-slot campaign's matched, non-overlapping phase pairs,
  `abs(cycle_wall_us-draft_wall_us-verify_sample_wall_us)` is no greater than
  their combined measured synchronization overhead plus separately recorded
  inter-phase scheduling time; mixed-participant/shared phases have no additive
  cycle residual;
- the verifier continues until each phase-kind and cycle count reaches 64 and
  checks its own denominator, without treating shared times as sequence costs;
- an unwritable log path fails before requests begin, while a disabled run
  creates no log, synchronization, or recording-loop work.

Run `timeout 240 python3 -m unittest scripts.test_bench_spec`,
`timeout 240 ./build-sycl/bin/test-qwen4exp-mtp`, and its `--q8-kv` form
before and after the change.

## Assumptions & contingencies

- `LLAMA_SPEC_LOG` alone records unsynchronized host wall time. Only rows with
  `timing_mode=synchronized` are valid for phase attribution.
- Proposal, sequence, cycle, phase, request, and launch IDs are process-local and
  monotonic; every row carries enough identity to reject cross-cycle joins.
- Shared multi-sequence wall time is duplicated by reference, not attributed or
  summed. The primary reconciliation campaign forces one active slot; separate
  tests cover shared-phase identity.
- P06 and P10 consume the selected proposal dataset. P07 consumes it through
  P06 and is not an additional hard dependency.
