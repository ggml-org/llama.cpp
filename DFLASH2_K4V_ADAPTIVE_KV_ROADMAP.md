# DFlash2 + K4V adaptive KV streaming roadmap

Status: implementation in progress on `feature/adaptive-kv-dflash2-k4v`. NK0 is complete; NK1-NK7 are pending. Mixed streaming admission remains closed, and no mixed-mode performance gain is claimed. Updated: 2026-10-26.

Source baseline: `bcf1a81d641c478db5690d2f2c048c691801fcb2`. The mechanism investigation originally used `1b7c0f46e`; the relevant common, n-gram, context, rollback and server source files were verified unchanged between that revision and this baseline. Recheck those files before implementation if HEAD changes.

The user authorized implementation on a new branch and one commit per completed NK stage on 2026-10-26. Keep this work separate from completed Gemma qualification, vision support and deferred kernel experiments. Preserve unrelated user changes. No pushes or production model/configuration changes are authorized.

## Objective

Support `ngram-map-k4v` together with the currently admitted Qwen DFlash2 pairing under adaptive KV streaming and the shared phase arena. Reuse the existing speculative dispatcher, target verification, recurrent rollback, bounded draft KV and serial workspace owner. Preserve stock target arithmetic, DFlash2 convolution/selector behavior and target sampling semantics.

The expected benefit is avoiding some DFlash2 prediction forwards when history supplies a useful draft. K4V does not remove target verification or required DFlash2 feature injection. A lookup hit, a skipped neural forward or a faster isolated lookup is not by itself evidence of an end-to-end speedup.

Related plans: [DFlash2 integration](DFLASH2_ADAPTIVE_KV_ROADMAP.md), [memory consumers](DEVICE_MEMORY_CONSUMERS_ROADMAP.md), [MTP integration](MTP_ADAPTIVE_KV_ROADMAP.md) and [Gemma integration](GEMMA4_ADAPTIVE_KV_ROADMAP.md). This plan does not renumber their stages or reopen completed work.

## Initial scope and explicit limits

- Current Qwen3.8-27B target, represented by `qwen35` in the implementation, and its matching DFlash2 sidecar. Pin actual GGUFs and metadata rather than admitting arbitrary drafts by name.
- One CUDA GPU, one serial request, Flash Attention enabled, physical-budget qualification with UVM disabled, and an explicitly declared shared arena.
- Text-only, matching existing adaptive DFlash2 admission. No image/projector, parallel-slot, multi-GPU, Gemma+DFlash2, DSpark, EAGLE or simultaneous MTP support is added by this work.
- Independently configured DFlash2 and K4V proposal maxima of 1-7 tokens. Qualify all actual target verification widths from TG1 through TG8, including intermediate widths.
- K4V's lookup length N remains distinct from its continuation/proposal length M. The seven-token initial limit applies to M, not N or the number of tracked values.
- K4V currently defaults to M=48. Mixed streaming configurations above the admitted maximum must fail with an actionable error, not silently clamp or inherit a different meaning for an existing option. Longer K4V verification is a separate extension requiring its own resource, rollback and numerical qualification.
- Preserve native/non-streaming behavior and the existing target-only, embedded-MTP, Gemma-assistant and DFlash-only streaming paths.
- Use existing common ownership and backend operations. No new attention kernel, quantization format, arithmetic order, GPU map, independent arena or generic speculative execution framework is expected.

The 1-7 K4V limit and reset policy below are proposed first-release design choices, not already implemented support. Any material expansion needs explicit review rather than a silent change to this scope.

## Current mechanisms and gaps

| Area | Mechanism at the source baseline | Required work |
| --- | --- | --- |
| K4V storage | CPU vectors hold a hash index and per-key continuation statistics referring to token-history positions | Keep the map in RAM; validate history identity, indices and updates |
| Draft ordering | `common_speculative_init` puts K4V ahead of DFlash; `common_speculative_draft` stops after one implementation supplies a nonempty draft | Reuse this cascade, independent of CLI ordering |
| Admission | `common_streamed_dflash_config_error` rejects any other speculative mode | Narrowly admit K4V only after the complete mixed path is tested |
| Target depth | `common_params_speculative::need_n_rs_seq` returns the neural drafter's `draft.n_max`; n-gram length does not participate | Resolve target verification and rollback from both proposal sources |
| Backend capability | Target streaming and recurrent spill already support up to seven proposals/eight verification rows | Reuse this qualified range; do not increase a constant to admit longer drafts |
| Child workload | The serial DFlash child derives `block_rows` from target decode rows | Distinguish target verification, neural prediction and feature injection |
| Feature maintenance | Every successful target batch passes through `common_speculative_process`; DFlash gathers target features and injects them into its windowed cache | Continue this when K4V supplies the draft |
| Acceptance | The dispatcher remembers the supplying implementation and sends source-specific acceptance feedback | Keep attribution correct during switching, rejection and replay |
| Recovery | Server sequence removal coordinates target and draft contexts; rejected injected rows must be removed from both | Exercise every accepted prefix and drafter transition |
| Restore state | The common speculative state interface does not provide a general serialization container for multiple stateful implementations | Treat K4V as rebuildable host state initially; do not invent a new state format unnecessarily |

Source anchors:

- [K4V data structures](common/ngram-map.h): `common_ngram_map`, `common_ngram_map_value` and `key_map`.
- [K4V lookup and statistics](common/ngram-map.cpp): `common_ngram_map_draft`, `common_ngram_map_begin` and `common_ngram_map_accept`.
- [Speculative driver](common/speculative.cpp): `common_speculative_init`, `common_speculative_draft`, `common_speculative_process` and `common_speculative_accept`.
- [Common parameters](common/common.h): `common_params_speculative::need_n_rs_seq`.
- [Admission and target parameters](common/common.cpp): `common_streamed_dflash_config_error`, `common_context_params_to_llama` and `common_base_params_to_speculative`.
- [Context construction and reservation](src/llama-context.cpp): serial DFlash admission, `sched_reserve` and `decode`.
- [Decode width](src/llama-cparams.h): `kv_stream_decode_rows` and `serial_graph_reserve_rows`.
- [Rollback plan](src/llama-memory-recurrent-spill.cpp): `llama_recurrent_rollback_plan_make`.
- [DFlash graph](src/models/dflash.cpp): separate embedding-based injection and token-based prediction paths.
- [Server round](tools/server/server-context.cpp): `pre_decode`, `decode` and speculative acceptance in `post_decode`.

### K4V correctness issue to reproduce first

At the baseline, K4V finds `slot_max` and uses that value's acceptance statistics, but emits tokens from `inp[match_pos + n + i]` rather than explicitly from the selected value's `value_idx`. Construct a CPU fixture where these continuations differ before changing the code. Record the failing assertion and then fix the demonstrated inconsistency.

Also audit the inclusive statistics cursor: the loop begins at `curr_key.stat_idx`, which is set to the last processed match position. Test whether repeated lookups recount the same occurrence. Characterize minimum-hit and zero-acceptance behavior rather than assuming the documentation fully describes their implementation. Do not bundle speculative new scoring heuristics into correctness fixes.

## Verification and rollback contract

For the first supported pairing, resolve these quantities before constructing contexts:

```text
F = configured maximum DFlash2 proposals
K = configured maximum K4V proposals
D = max(F, K)

Target maximum verification rows = D + 1
Target recurrent rollback depth  = D
DFlash prediction rows           = F + 1, within trained block capacity
Actual target verification rows = L + 1 for a selected draft of length L
```

The extra row is the already sampled anchor at the next unprocessed target position. K4V's four values are alternative stored continuations, not four simultaneous verification branches.

| DFlash maximum F | K4V maximum K | Target verification capacity | DFlash prediction maximum |
| ---: | ---: | ---: | ---: |
| 3 | 3 | 4 rows | 4 rows |
| 3 | 7 | 8 rows | 4 rows |
| 7 | 3 | 8 rows | 8 rows |
| 7 | 7 | 8 rows | 8 rows |

The `F=3, K=7` case is a mandatory test: it exposes accidental coupling between neural prediction size and target verification capacity. Startup requirements must also fit the effective target and draft microbatches. Keep each non-causal neural prediction block intact.

For Qwen, configure the existing recurrent rollback machinery from D; do not repurpose Gemma's separately admitted `kv_stream_verify_max` mechanism merely to bypass a guard. No dynamic growth beyond the admitted width may occur after a K4V hit.

### Required startup measurements

1. Validate the allowed mode set, proposal bounds, model pairing and effective microbatch capacities before mutating target feature configuration or registering a child.
2. Derive one consistent target verification/rollback/output limit from F and K.
3. Measure every admitted target width, including widths with all logits and required feature rows retained. Maximum width alone is not proof that all intermediate graph shapes fit.
4. Measure the DFlash token-prediction workload independently of target verification width.
5. Preserve and verify the existing embedding-based feature-injection measurement, including verification batches larger than the configured neural prediction block.
6. Admit persistent bounded draft KV, target rollback staging and all phase workspaces through the existing physical owner. Retire warmup graphs before loading sidecar weights as the existing startup path requires.
7. Report effective widths and exact quota failure causes. Reject insufficient budgets; do not create an unbudgeted private workspace or retry a different model/quantization silently.

## Per-round execution and ownership

```text
Committed token history + sampled anchor
                  |
              K4V lookup
             /          \
    usable proposal    miss/ambiguous/empty
             |                 |
             |          DFlash2 prediction
             \                 /
                One selected draft
                        |
             Target verifies L + 1 rows
                        |
             DFlash feature injection
                        |
           Target acceptance and correction
                        |
          Trim rejected target AND draft state
                        |
                    Next round
```

Preserve these invariants:

- A K4V hit skips DFlash prediction, not required feature injection. Skipping injection needs a separate catch-up design and is not part of this plan.
- A rejected K4V draft completes normal target acceptance/correction. Do not retry DFlash against the old frontier in the same round.
- Both-empty rounds use ordinary target decoding and still maintain the DFlash feature history needed by a later prediction.
- Temporary DFlash noise rows and feature-injected rows are different contents, even when their logical positions overlap. Retire temporary prediction suffixes before target verification/injection using existing mechanisms.
- Injection currently precedes acceptance. Its provisional suffix must be removed when the target rejects tokens; injected does not mean committed.
- For L proposed tokens and a accepted proposals, retain the anchor plus a accepted proposal rows and discard L-a provisional rows. The correction/bonus token becomes the next unprocessed anchor.
- Keep target feature identity, position, source generation and completion valid until injection has consumed the data. Serial graph handoff must not release or overwrite those outputs early.
- Reuse the current supplying-implementation record for acceptance accounting. A DFlash-selected round must not train K4V as though it supplied that draft.
- CPU map access adds no GPU synchronization requirement. Preserve mandatory target/draft completion, copy fences and cleanup rather than moving or deleting them to improve a lookup timing.

## Memory contract

K4V stays in ordinary host memory. Its 262144-entry `uint32_t` hash index is 1 MiB per map, plus dynamic key/value statistics and token history. Measure vector capacity and growth; it is not a fixed 1 MiB total-memory guarantee.

Keep the existing device ownership model:

```text
Outside the arena:
  target and DFlash2 model weights, native driver/executable resources

Inside the participating shared arena:
  target resident KV and transfer ring
  persistent bounded DFlash2 KV
  admitted phase graph/attention workspace
  live rollback publication staging and required handoff resources

Host memory:
  authoritative streamed KV history
  recurrent rollback snapshots
  retained feature transport and CPU token history
  K4V hash index and continuation statistics
```

This is not automatically VRAM-neutral. With F=7 and K<=7, target width is unchanged and a larger verification grant should not be needed; verify the actual allocations. With F=3 and K=7, rollback/output/workspace requirements can grow and reduce target resident KV capacity. Report the resulting pool, resident and ring layout rather than calling CPU-only lookup free.

Do not multiply separate phase costs when they share mutually exclusive lifetimes, and do not overlap resources whose data are still live. Reuse the current memory coordinator and bounded-cache requirements, not a parallel allocator.

## Staged implementation and acceptance

### NK0 - Pin controls and observability

Status: complete. See the [pinned NK0 report](benchmarks/data/dflash2-k4v-nk0.json).

Added read-only common-layer snapshots of existing source counters and last supplying implementation, plus a successful DFlash/DSpark prediction-forward counter. These are host observations, not new C inference APIs or backend synchronization. The forward count distinguishes a dispatcher call from a completed neural prediction, including context-end abstention.

The missing-observation API compiled red before implementation. Five CPU cases/42 assertions pass in Release and ASAN/UBSAN, and the existing argument suite passes in both builds. The preserved pre-NK DFlash control passes 1075 assertions; the observed native and shared-arena controls pass 1083 and 1097 assertions. The owned-command descendant-cleanup self-test passes, production port 1234 is restored healthy, and the control owner is inactive.

Raw evidence: `benchmarks/results/dflash2-k4v-20261026-nk0-A53EiR/`, including artifact inspection, the 24-file pre-NK archive, source/binary hashes, CMake-derived build commands, API red, seven accepted gates, loaded-library paths and restoration health. Native qualification rebuilt only changed host objects with exact CMake compile/link commands against hash-pinned existing core/CUDA libraries; no dependency timestamps were forged and no shader was rebuilt. This is focused observability qualification, not a complete fresh mixed-runtime build or mixed-mode acceptance proof. NK4/NK6 still require actual source switching and whole-path qualification.

- Pin source, compiler/build configuration, target/sidecar/configuration/corpus hashes, libraries, KV types, batch sizes and declared arena profiles.
- Preserve a DFlash-only control and use the current native-versus-streamed exact DFlash fixtures.
- Extend existing tests and harnesses. A focused CPU n-gram test target is justified if no existing test directly covers this unit; do not introduce a separate test framework or a public testing API.
- Make tests observe the selected drafter, returned proposal length, actual target verification width and actual DFlash prediction forwards.
- Keep total draft counts separate from source-specific counts. Success must not mean K4V was accepted by the CLI but never used, or speculation was silently disabled.

Exit: reproducible controls, an inventory of reused test hooks, and fixtures that can prove both proposal sources execute.

### NK1 - K4V CPU correctness

Status: pending. Start here before full-model testing.

Write failing tests before fixes for demonstrated defects. Cover:

- Most-frequent continuation different from the hash-selected historical continuation.
- Selection of each of the four value slots, ambiguous frequencies and the dominance threshold boundary.
- Minimum-hit behavior and acceptance feedback applied to the continuation actually emitted.
- Zero acceptance and subsequent calls; characterize recovery behavior without inventing an unreviewed heuristic.
- Repeated lookups with no new historical occurrences, incremental history growth and statistics accounting.
- Short histories, exact length boundaries, hash collisions, capacity/count limits and index bounds.
- Truncation, cleared history and equal-length replacement with different tokens.
- Existing key-only n-gram behavior remaining intact.

Run focused normal and ASAN/UBSAN CPU tests. Preserve red evidence. If behavior differs from the intended algorithm, separate a correctness fix from any optional quality/performance policy change.

Exit: correct emitted continuation and statistics attribution, no invalid history reads, and explicit tested reset semantics.

### NK2 - Common mixed-workload limits and admission

Status: pending. Depends on NK1.

- Add or extend a small common resolver for target proposal capacity, verification rows, rollback depth and output-row requirements. Avoid a new registry or speculative scheduler.
- Apply it consistently in target parameter construction, common validation and server output sizing. Keep existing non-streaming parameter semantics unchanged.
- Admit only DFlash2 plus optional K4V in the new path. Preserve errors for MTP, other n-gram types, unrelated model-based drafters, auxiliary MTP layers, parallel slots and media configurations.
- Reject K4V M>7 with an actionable message; do not quietly clamp the current M=48 default.
- Check effective batch/microbatch requirements and integer bounds before allocation or target mutation.
- Keep public mixed-mode admission closed until the internal contract and execution tests exist; open it narrowly as part of the completed integration.

Host tests must include F<K, F=K, F>K, both extrema, duplicate/type-order handling, invalid lengths, absent sidecar, explicit draft-cache overrides, insufficient microbatches and unchanged legacy modes.

Exit: one tested source of verification limits; invalid combinations fail before expensive execution.

### NK3 - Target and child workspace sizing

Status: pending. Depends on NK2.

- Audit `serial_block_rows` and all consumers before changing its meaning. Keep target verification capacity, neural prediction limit and embedding-injection extent distinct using the smallest necessary private changes.
- Reserve all admitted target graph widths and feature/logit rows using the existing scheduler measurement path.
- Preserve trained DFlash prediction bounds and intact non-causal blocks even when K4V increases target capacity.
- Verify injection graphs cover K4V verification rows, ordinary prefill injection and all relevant serial handoffs.
- Retain the same physical parent through graph remeasurement, target/draft switching and child teardown.
- Keep persistent draft KV and rollback staging inside their declared regions; no independent-workspace fallback for this admitted serial consumer.

Tests must include exact-quota construction, one-byte-short rejection, all TG1-TG8 widths, F=3/K=7, short drafts, output canaries, child destruction/target resume and stale graph-address rejection. Derive quota expectations from the actual resolved workload, not a copied old minimum.

Exit: every admitted workload is measured and executes inside the declared ownership contract without new backend arithmetic.

### NK4 - Switching, feature continuity and rollback

Status: pending. Depends on NK3.

Exercise real driver sequences:

- K4V hit followed by another hit.
- K4V miss, ambiguity or empty result followed by DFlash.
- K4V -> DFlash -> K4V, including multiple consecutive rounds of each source.
- Both-empty fallback to a single target row.
- Zero, every partial prefix and full acceptance from each source.
- Context/output headroom shortening a selected proposal.

After each round compare target and draft frontiers, visible KV, recurrent state, feature-row identity and positions, temporary-row retirement and source-specific statistics. Verify a failed/stale feature match rejects cleanly rather than using another batch's output.

A mandatory continuity test runs several K4V-selected rounds and then compares the next DFlash selector candidates and scores against a reference that processed the identical committed target history. This catches the error of skipping DFlash maintenance merely because its prediction forward was skipped.

Keep exact forced-decision tests separate from ordinary model-output/performance tests. Synthetic acceptance is not valid output evidence.

Exit: no stale DFlash history, rejected suffix, incorrect accepted-row offset or cross-source feedback contamination.

### NK5 - Lifecycle, failure recovery and sampling

Status: pending. Depends on NK4.

Cover fresh/cached/changed prompts, low-high-low requests, sliding-window wrap, repeated rejection, context end, cancellation, save/restore, restart and construction/destruction failures.

Initial proposed K4V restore policy: treat the map as disposable derived CPU acceleration state. Invalidate/rebuild it after state restore or history replacement instead of adding a serialized map format. Do not keep indices into replaced history. Compare restored execution against a reference using the same reset policy. Do not promise uninterrupted-run draft-count identity when the acceptance-history map was intentionally reset.

Inject failures in target verification, feature extraction/injection and serial handoff. Require completed cleanup or closed admission before retry. Do not leave a partially updated draft cache advertised as current. Test successful reconstruction and exact retry after failures.

Preserve current sampler and grammar behavior. Ensure an n-gram-selected round has no stale draft probability distributions. Greedy exact qualification is mandatory; exercise existing nonzero-temperature and replay paths separately without assuming fixed-seed cross-layout identity or claiming newly qualified probabilistic rejection sampling. Cover EOS, stop strings, grammar constraints, cancellation and output limits.

Exit: no stale map/cache/feature state after reset or failure, no leaked borrowed region, and correct target sampling/termination behavior.

### NK6 - Whole-path numerical and safety qualification

Status: pending. Depends on NK1-NK5.

Use two different comparisons and do not confuse their invariants:

1. Correctness: native versus streamed hybrid execution with identical proposal inputs, verification shapes and acceptance decisions. Compare every target logit vector, extracted feature row, subsequent DFlash selector candidate/score, recurrent state and visible KV. Public matched-protocol controls compare output token IDs and source-specific draft/acceptance counts.
2. Performance: streamed DFlash-only versus streamed DFlash+K4V. Draft counts and selected sources are expected to differ; they must be recorded, not required to match. Verify model-output correctness separately from these changed schedules.

Qualification includes:

- Focused CPU and ASAN/UBSAN tests plus relevant existing portable owner/cache/rollback tests.
- Strict CUDA memcheck, all API errors reported and a nonzero error exit. Keep ordinary graph-enabled coverage; use eager execution for tool-only profiles where established. Never call a sanitizer OOM or timeout a zero-error pass.
- Actual Qwen/DFlash native-matched controls through all verification widths and required window/rejection/lifecycle cases.
- Full-server Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1 profiles. Distinguish their measured scope from unchanged operator-level coverage of other encodings.
- Existing Qwen target-only and embedded-MTP, Gemma target/assistant, DFlash-only, phase-arena and applicable attention-format/width regressions.
- Real long streamed history with observed active streaming/ring use, plus fitting native controls. Stream-only long-context self-consistency is not native equivalence.
- Exact source, model, profile and loaded-library hashes before/after, with explicit provenance for any legitimately reused evidence.

Trace numerical mismatches to the first differing inputs, row, state or execution family. Do not relax comparisons, silently enlarge fixtures, alter quants, suppress API errors or remove a failing geometry to force qualification.

Exit: exact matched-path correctness, clean strict memory qualification, bounded resources and no unrelated supported-mode regression.

### NK7 - Performance, arena finder and documentation

Status: pending. Depends on NK6 for promotion; small diagnostic measurements may follow focused correctness earlier.

Benchmark repetitive code/copying, ordinary prose and ambiguous repetition. Preserve the corpus, prompt token IDs, tokenizer/model identities, greedy sampling settings, output counts, batch sizes, UVM setting and declared budgets. Start with a small representative screen before expanding expensive runs.

Report:

- Whole K4V draft-call p50/p95/p99, separating initial indexing and first-use statistics from steady state. Timing one hash access is insufficient.
- Lookup calls, supplied drafts, ambiguity/empty outcomes, selected-source counts, acceptance length and actual DFlash prediction forwards avoided. Distinguish driver calls from forwards: a called implementation can have no eligible sequence.
- Target verification widths and cost, feature-injection cost, end-to-end decode throughput and prefill/startup changes.
- Host vector/history memory, rollback resources, physical arena size and target pool/resident/ring layouts.
- Light/heavy streaming behavior under explicitly declared profiles. Use matched histories to separate history-size effects from CPU scheduling/host-memory contention; do not claim bandwidth saturation without measurement.

First compare equal arena budgets. Separately compare independently maximum-fit deployment configurations through the existing sizing tool. A larger K4V proposal capacity can change the target working set and must not be hidden in the comparison.

Extend [the arena finder](benchmarks/find-max-arena.py), [DFlash controls](benchmarks/run_dflash_controls.py) and their existing host tests as needed to describe the mixed mode and explicit K4V parameters. Keep inherited target cache types, environment sanitization and owned-server cleanup. Preflight must execute real verification and prove both K4V and DFlash paths are exercised; one successful route is not validation of the other. Preserve the selected full-workload validation and tested upper-neighbor evidence where the finder reports a maximum.

Use uninstrumented matched measurements for speed claims. Profiling can explain changes but cannot replace end-to-end timing. If confirmation samples are needed, record their protocol and variability; do not promote an unsupported percentage or merge unlike cohorts. Do not require noisy wall-clock speed thresholds in correctness tests.

Document supported configurations, proposal limits, reset behavior, memory costs, actual measured gains/regressions and reproduction commands. Keep the mixed path opt-in; do not make it the default merely because selected repetitive prompts improve.

Exit: a reproducible mixed-mode invocation, faithful sizing, workload-specific performance evidence and complete limitations. A correctness pass without useful performance must be reported as such, not sold as an optimization.

## Expected file boundaries

| Responsibility | Expected files to inspect or extend |
| --- | --- |
| K4V correctness | `common/ngram-map.cpp`, `common/ngram-map.h`, focused CPU tests and `tests/CMakeLists.txt` if needed |
| Limits and admission | `common/common.h`, `common/common.cpp`, `common/speculative.cpp`, `common/arg.cpp` only where existing validation needs wiring, `tests/test-arg-parser.cpp` |
| Serial resource integration | `src/llama-context.cpp`, `src/llama-cparams.h`, existing context-memory/rollback helpers only where demonstrated necessary |
| Round and lifecycle | `common/speculative.cpp`, `tools/server/server-context.cpp`, existing checkpoint mechanisms only where a test exposes a gap |
| Exact model tests | `tests/kv-stream-dflash-test.h`, `tests/test-kv-stream-context.cpp`, existing owner/cache/phase tests and server tests |
| Reproduction and docs | `benchmarks/find-max-arena.py`, `benchmarks/run_dflash_controls.py`, their host tests, `docs/speculative.md` and concise supported-scope README updates after qualification |

These are investigation boundaries, not authorization to modify every listed file. Prefer extending existing helpers and tests. No public C API change or new device code is expected; revisit the design if either becomes necessary. Rebuild affected normal and sanitizer consumers consistently if private layouts change, without starting a clean compile-time benchmark or an unnecessary heavy shader rebuild.

## Proposed user-facing invocation after qualification

The following fragment documents intended options, not a currently working streaming configuration. Combine it with the ordinary admitted target, arena, cache, context and device settings only after qualification:

```sh
--spec-type ngram-map-k4v,draft-dflash \
--model-draft /path/to/matching-DFlash2.gguf \
--spec-draft-n-max 7 \
--spec-ngram-map-k4v-size-n 12 \
--spec-ngram-map-k4v-size-m 7 \
--spec-ngram-map-k4v-min-hits 2
```

CLI order must not change the existing K4V-before-DFlash priority. This fragment does not add MTP, a Gemma sidecar, a projector or draft-cache overrides.

## Execution discipline and review boundaries

- The first executable step is the CPU continuation-selection red, followed by asymmetric workload-limit tests such as F=3/K=7.
- Preserve controls, expected reds, unexpected failures and accepted artifacts. Resume only missing or invalidated gates after validating the source/binary/profile cohort.
- Do not launch duplicate GPU jobs, agents, production configuration/model changes, pushes or clean code-generation benchmarks as part of this plan.
- Benchmark programs own and stop only their own server descendants. They must not assume that a production server exists. If a local validation session separately coordinates an existing service, drain owned work before restoring and verifying that service.
- Keep numerical and API checks unchanged. New budgets, quantization, model scope, proposal limits or ownership mechanisms require an explicit recorded decision.
- Commit once per completed NK stage under the user's implementation authorization. Review the exact stage files and evidence before each commit; do not mark a subgate as a complete stage or include unrelated files.

## Completion checklist

- [x] NK0: pinned controls and proof that both draft sources can be observed independently; actual mixed switching remains NK4.
- [ ] NK1: CPU reds reproduced and demonstrated K4V defects fixed with regression coverage.
- [ ] NK2: one consistent bounded target-workload calculation and narrow admission.
- [ ] NK3: independent verification/prediction/injection measurement and exact quota coverage.
- [ ] NK4: source switching, feature continuity and every accepted prefix are correct.
- [ ] NK5: cache reuse, restore/restart, cancellation, sampling and failure recovery pass.
- [ ] NK6: matched native/streamed exactness, strict safety and supported-mode regressions pass.
- [ ] NK7: meaningful performance scope, faithful finder, examples and limitations are documented.

## Primary references

- [Stock speculative decoding and K4V documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).
- [Stock n-gram map implementation](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-map.cpp).
- [Stock speculative dispatcher](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp).
- [Publisher DFlash2 configuration](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json).

External links can change. The source baseline above governs the implementation observations; pin artifact/configuration revisions and checksums in NK0 before treating any external metadata as qualification input.
