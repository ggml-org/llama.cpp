# Generic adaptive KV quantization

## Goal and agreed policy

Support upstream KV cache combinations through actual backend capabilities, not a Q8_0/Q4_0-only policy. Cover target-only, embedded MTP and DFlash on the existing serial Qwen/CUDA implementation. Do not change model architecture support or introduce independent MTP page formats.

The target's K/V settings are the single source of truth for adaptive speculative execution. Both MTP and DFlash inherit them. Explicit draft K/V overrides are rejected, even if they name the same format, instead of being silently ignored. Preserve ordinary non-streaming argument behavior. DFlash has its own bounded sliding-window storage within the arena; this consistency rule is policy rather than a claim that it uses target ring slots.

Candidate upstream cache types are F32, F16, BF16, Q8_0, Q5_0, Q5_1, Q4_0, Q4_1 and IQ4_NL. A CLI-valid type is not sufficient evidence that the backend can write it or execute its native and streamed attention. Missing upstream functionality, excluded build pairs, unsupported device geometry and insufficient budgets must be diagnosed distinctly. Do not advertise unsupported combinations as qualified.

## Execution rules

- Work on the merged adaptive V2 branch, starting at 1c4469df3.
- Use existing tests and private backend interfaces. Keep common ownership and policy backend-neutral.
- Write a failing regression before each behavior change. Preserve existing test bodies and tolerances.
- Preserve the Q8_0/Q4_0 arithmetic and optimized paths. Generalization must not replace them with a universally slower fallback.
- Commit each completed stage with a title-only message, as explicitly authorized by the fork owner. Do not push.
- Update this document with commands, red/green evidence, qualification limits and commit checkpoints.
- Preserve unrelated untracked files and accepted benchmark records.
- Benchmark without UVM or an arbitrary reserve. Stop production only when GPU tests require it and restore its existing configuration afterward.
- Keep long-running jobs supervised and disclose log/progress locations. The old exploratory quant matrix is stopped; start a new qualified matrix only at KQ6.

## KQ1 - Configuration versus capabilities

Introduce shared structural validation and target-cache inheritance for adaptive speculation. Track explicit draft cache overrides from CLI and environment without mistaking default draft values for overrides. Reject overrides and invalid structural settings before model allocation. Test all candidate K/V pairs, both speculative modes, argument order, aliases, environment input and unchanged native behavior. Keep the existing MTP quant guard until KQ4 connects the qualified replacement.

Acceptance: configuration tests pass, native behavior is unchanged, no kernel or per-token work is added.

Status: completed. The ASan/UBSan argument suite first fails on DFlash retaining F16 instead of the target's asymmetric Q5_1/Q8_0 pair (`/tmp/kq1-red.log`), then passes with inheritance and explicit-override tracking. Added tests cover all 81 pairs in both modes, both CLI argument orders, six aliases, environment overrides, invalid formats and unchanged native overrides. The harness first fails on emitting prohibited draft flags, then all 54 artifact/control tests pass after command normalization (`/tmp/kq1-harness-{red,green}.log`). The full argument suite passes with leak detection and UBSan halt-on-error (`/tmp/kq1-final.log`). No decode kernel or hot-path code changes.

## KQ2 - Generic streamed attention dispatch

Audit compiled-pair discovery and stock-family selection across TG1-TG8, shorter terminal batches and prefill. Remove unjustified quant-specific dispatch restrictions. Reuse existing encoded loaders and native conversion rules; add missing streamed counterparts where stock execution supports the requested type. F32 and IQ4_NL need explicit investigation, not an assumption that all type enums have direct kernels. Retain capability-based rejection when no correct counterpart exists and document why.

Tests cover vector/tile/MMA selection, dense and compact masks, stock conversion order, one/two/three spans, wrap boundaries, partial tiles, resource rejection and selective builds. Compare matching native execution modes, including converted inputs, rather than TG1 versus wider verification.

Acceptance: native/spanned/resumed arithmetic qualifies for implemented pairs; missing pairs fail before execution; Q8/Q4 optimized dispatch is preserved.

Status: completed. The all-pairs release server and libraries rebuild coherently. All 64 compiled pairs execute TG1-TG8 against native attention (3075 assertions); the complete backend span suite passes 23 tests/6214 assertions. Focused F32 complete/wrapped/resumed cases pass 2/924 with zero reported maximum error. CUDA memcheck and synccheck report zero errors; racecheck reports zero hazards, errors or warnings. Unchanged seven-type CUDA reference objects are relinked against the coherent libraries: conversion 112 cases, staging 112, encoded MMA 1372, FP16 tile 48 and FP16 MMA 24 all match; native resumed tile passes 104 stock-exact cases and vector passes 34 wave cases with stock maximum error zero. No ABI or reference arithmetic changes are involved in reusing those unchanged reference objects. Logs are `/tmp/kq2-{f32-green,quant-matrix,spans-final,f32-memcheck,f32-racecheck,f32-synccheck,unchanged-cuda-spans,unchanged-cuda-resume}.log`.

The first broad A/B completes all 96 numerical/timing cases; one measurement line in each combined stdout/stderr log is interleaved and excluded, leaving 95 paired measurements. Focused alternating A/B runs use one identical O2-compiled driver against frozen old/new libraries, three repetitions, 64/128 Ki, TG1/TG4/TG8 and three wrapped spans. Median latency changes range from -0.122% to +0.156%, treated as noise rather than a speedup. Whole-model performance qualification remains KQ6. The exploratory full-matrix second repetition is deliberately interrupted after the first complete pair, not accepted as another full run. Logs are `/tmp/kq2-ab-{old,new}-1.log` and `/tmp/kq2-focused-{old,new}-{1,2,3}.log`. Production is restored healthy.

## KQ3 - Quant-dependent memory accounting

Verify host layout, resident/ring pages, row publication, mutable tails, conversion and attention scratch, and phase transitions for selected encodings. Cover auxiliary MTP cache geometry and gathered-prefill requirements. Use actual block sizes, row sizes, padding and alignment in every quote and transfer.

Tests cover all candidate layouts, asymmetric pairs, exact fits, one-byte-short grants, overflow, alignment, partial publication and resource retention. An allocation quote must cover the actual selected path; a fallback with different requirements must be quoted before execution.

Acceptance: no Q8/Q4 size assumptions, no hidden reserve, no out-of-bounds access under focused sanitizers.

Status: completed as a qualification-only change: the existing sizing implementation passes the expanded checks without alteration. ASan/UBSan passes partition/minimum/exact-fit accounting (36 tests/153360 assertions), host layouts (8/386), row publication (12/4258) and extended lease population (19/339). CUDA passes model ownership/admission and 64-pair auxiliary phase quotes (34/4341), all 81 writer format combinations with partial/cross-page rows (12/4260), lease population (20/339), strict host registration (9/394) and unchanged recurrent spill (22/3769). Unsupported IQ4_NL native attention is rejected at width one before model allocations. Logs are `/tmp/kq3-{policy,host,writer,lease,model-cuda,writer-cuda,lease-cuda,host-cuda,recurrent-cuda}.log`; no numerical tolerance or original failure assertion is weakened. Production is restored healthy.

## KQ4 - Capability-based MTP integration

Replace the server's Q8_0/Q4_0-only check after the lower layers qualify. Prepare required native and streamed prediction/catch-up/verification/replay variants before arena allocation. Keep target and MTP cache formats identical and the auxiliary cache in the existing shared physical policy. DFlash inherits target formats under KQ1 policy while retaining bounded draft storage.

Add regressions for Q8/Q8, Q5_1/Q5_1, Q8/Q4 and asymmetric formats: context construction, attachment, publication, lease reuse, rejection, truncation and cleanup. Verify bad pairs and inadequate budgets return actionable initialization errors rather than assertions or recovered execution failures.

Audit the existing vision-arena quant guard as well: apply the same structural/capability distinction without changing its serial, projector, model, LoRA or speculative-mode restrictions. Qualify inherited cache formats across target/MTP suspension and restoration before claiming image-path coverage.

Acceptance: supported pairs start and execute through all admitted widths; unsupported paths are rejected before requests.

Status: completed. The old Q8/Q8 MTP startup failure is preserved in the stopped exploratory matrix. Live 8 Ki-capacity/4096-prompt/32-output smoke runs now pass all nine combinations of Q8/Q8, Q5_1/Q5_1 and Q8/Q4 with target-only, MTP=3 and DFlash=7; speculative cases generate actual drafts and the harness rejects any recovered execution failure. These are fixed-quota compatibility checks, not performance data or primary sweep points. Results and logs are under `benchmarks/results/kq-qualified-20261016/smoke`, with runner output `/tmp/kq4-live-smoke.log`. Vision validation first fails on 160 newly admitted format/mode cases, then passes 7 tests/200 assertions under ASan/UBSan and the CUDA-linked build. All other vision restrictions remain. Initialization and all-width backend qualification reuse KQ2/KQ3 checks; full numerical and vision-handoff qualification follows in KQ5. Production is restored healthy.

## KQ5 - Numerical and lifecycle qualification

Extend the existing real-model native/streamed oracle with the actual selected quant pair. Exercise target-only, MTP and DFlash with matching execution settings. Cover resident, light-ring, wrapped-ring and histories larger than the ring, mutable tails, rejection, context exhaustion, cancellation and serial request reuse.

Use exact output/state comparisons where the stock execution order permits them. Different formats are not expected to generate identical text. Neither is TG1 automatically interchangeable with stock wider-query arithmetic. Record and investigate any mismatch without weakening old tests or tolerances.

Acceptance: unchanged suites and expanded numerical/lifecycle cases pass; focused ASan/UBSan/CUDA checks are clean; any external qualification limitation is explicit.

Status: completed. The audited 140-case real-model matrix passes with no skipped cases: all 64 CUDA-supported pairs in matched native/streamed MTP and DFlash, plus the 12 primary-pair extended controls. Short MTP forces actual target/MTP streaming with an eight-row diagnostic microbatch and forced acceptance/rejection through TG1-TG4; DFlash exercises depths 1/3/7, and its longer controls force actual target streaming. Primary long MTP retains b/ub=256 and all lengths 1/2/3. Rejection, truncation, cancellation, serial restoration and shared-parent vision/MTP suspension are covered. Native controls use identical staged host recurrent rollback to fit this GPU; these are native-attention comparisons, not pristine upstream memory-allocation tests. Vision handoff checks do not constitute image-encoder inference qualification for other cache pairs. Results and runtime provenance are recorded in `benchmarks/results/kq-qualified-20261016/numerical/qualification-review.json`.

The freshly rebuilt, unchanged CUDA oracles also pass: 112 encoded conversions, 112 staging cases, 1372 MMA comparisons, 48 FP16 tile probes, 24 FP16 MMA comparisons, 104 resumable tile cases and 34 vector wave cases, with zero measured output/metadata/byte mismatches (`/tmp/kq5-fresh-cuda-{spans,resume}.log`). CPU ASan/UBSan passes 11 tests/926 assertions. Earlier KQ2/KQ3 sanitizer and boundary checks remain valid because KQ5 changes test fixtures only. Test setup failures and fixes are documented in the progress log; no production code, comparison tolerance or actual-streaming prerequisite was relaxed. All accepted records are preserved with original runtime fingerprints, and only source-verified unaffected controls are reused after fixture changes. Production is restored healthy on port 1234.

## KQ6 - Performance matrix and documentation

Run the requested fresh 45-point matrix on IQ4_XS with the checked-in article corpus:

- Context capacities: 8192, 65536, 131072, 196608 and 262144.
- Modes: target-only, MTP=3 and DFlash=7.
- Cache pairs: Q8_0/Q8_0, Q5_1/Q5_1 and Q8_0/Q4_0.
- 256 generated tokens, matched prompt headroom, b/ub=256, one slot, no UVM.
- Independently determine each context/mode's largest full-workload fitting arena.
- Record prefill/decode throughput, effective KV grant, acceptance, errors and runtime provenance.

Use shorter correctness controls for remaining supported pairs. Compare Q8/Q4 against the frozen pre-change runtime; repeat equal-budget representative comparisons to separate performance changes from maximum-fit/residency effects. Review differences above 3%, using repeated alternating runs rather than declaring a regression from one noisy sample. Do not retain a reproducible implementation regression or claim improved performance from unsupported/failed points.

Update README support and invocation guidance, inherited draft-cache behavior, capability diagnostics and plotting instructions. Preserve full commands, logs, manifests and the final graph. Verify production's host health endpoint after restoration.

After implementation and correctness qualification, restart the full requested matrix under independent termination/stall supervision. If a worker aborts before completion, inspect its exact failure, fix the in-scope issue, rerun affected correctness checks and resume missing points without discarding accepted records. A clean process exit or launched background job is not completion. Review all points and final artifacts before reporting KQ6 complete.

Acceptance: all 45 requested points pass, matched-mode correctness qualifies, Q8/Q4 performance remains comparable, documentation is accurate, production is healthy, and no unqualified support claim remains.

Status: pending.

## Progress log

- Roadmap saved before implementation. No runtime changes or new qualification matrix started yet.
- KQ1 committed as c6ce1800e. Only title text was used in the commit, as requested.
- KQ2 in progress: the new host dispatch regression fails 144 assertions before removing the pair-specific TG2 gate, then passes under ASan/UBSan. An additional Volta unquantized-TG1 regression first fails three assertions; final host admission passes 24 tests/7211 assertions. A separately compiled new CUDA fixture against the frozen baseline reproduces F32 capability rejection (`/tmp/kq2-f32-red.log`). F32 readers round into FP16 like stock and use mapped F16 native vector occupancy. Raw F32 cannot fit an FP16-sized staging tile; the initial compile rejection and temporary direct-loader build are archived in `/tmp/kq2-cuda-build{,-2}.log`. The refined provider instead rounds F32 on ingress into the existing FP16-sized staging region, then preserves the common staged MMA pipeline without additional buffers. Existing encoded quant paths are unchanged. The coherent all-pairs CUDA release build is running in build-upstream-attention-port (`/tmp/kq2-cuda-build-3.log`); actual GPU validation and this stage commit remain pending. IQ4_NL is absent from the current stock CUDA FA selector and remains a documented upstream limitation, not newly advertised support.
- KQ3 test preparation, not yet committed: expanded 81-pair partition/minimum-budget checks pass 36 tests/153360 assertions; host layout passes 8/386; stock-versus-writer partial/cross-page publication passes 12/4258 under ASan/UBSan. No sizing implementation change was necessary for these checks. Additional CUDA phase-quote and lease cases await the coherent CUDA build. Tests remain in the working tree for their own stage commit.
- While the all-pairs CUDA build runs, later-stage test preparation proceeds without committing those stages early. KQ4 vision configuration first fails 160 of 199 assertions on the old fixed-pair check, then passes 7 tests/200 assertions with the shared type policy (`/tmp/kq4-vision-{red,green}.log`). The server's MTP type guard is removed in the working tree; real-model admission remains pending. KQ5's new oracle type option first fails as unknown, then its host suite passes 8 tests/207 assertions (`/tmp/kq5-{red,green}.log`); the native/streamed MTP oracle, DFlash oracle and vision handoff can select matching cache formats without changing their defaults or tolerances. KQ6's fixed-span command test first fails on hardcoded Q8_0, then 36 fixed-span and 54 DFlash harness checks pass with configurable target cache types. GPU qualification and stage commits remain pending in order after KQ2.
- Production remains on its existing container image; no new sweep is running. The user explicitly requested resuming and supervising the complete matrix after implementation and correctness qualification.
- KQ1-KQ4 are now committed as c6ce1800e, a5feda843, c3312484d and 78a98458d. The earlier preparation entries above describe the red/green sequence, not the current stage status.
- KQ5 qualification is running under independent termination/stall supervision in `benchmarks/results/kq-qualified-20261016/numerical`. Its first small-history fixture correctly rejected an all-resident run: phase-arena reclamation grew the requested 6 MiB into a 267 MiB decode pool. The explicit short oracle now uses an eight-row diagnostic batch and one draft length of three to exercise TG1-TG4 and rejection without that large reclaimed prefill workspace. Four rows are invalid for the native recurrent splitter's four-row protected rollback tail; the fixture parser now rejects that boundary before execution. Default long controls retain b/ub=256 and all three lengths. CPU ASan/UBSan passes 11 tests/926 assertions; strict byte comparisons and actual-streaming prerequisites are unchanged. Frozen attempts and their failures remain available, with production restored between attempts.
- KQ5 run 1158f116 passes all 128 short numerical cases and three longer Q8/Q8 checks, then stops on the vision/MTP fixture. The fixture acquired a raw lease while still using the prefill layout, so the subsequent shared decode transition retired it. The existing phase-aware `llama_kv_stream_mtp_prepare()` fixes the fixture, with explicit prefill/decode assertions; no runtime code or tolerance changes. Q8/Q8 then passes 259 handoff assertions with zero measured logits error (`/tmp/kq5-handoff-green.log`). Resume run 688e1ec1 preserves the original 131 accepted records and their original runtime fingerprints: the new manifest checks identical production libraries, unchanged numerical headers, and an exact source diff limited to this vision-only branch. Only the nine missing cases run on the corrected fixture. Production was verified healthy between runs.
- The next resume passes the Q5_1/Q5_1 long MTP control, then correctly rejects its long DFlash fixture because a 512 MiB parent holds the compact 16K history without streaming. Explicit-format long DFlash fixtures now use 384 MiB; legacy defaults and all short-profile quotas are unchanged. Resume 02ee63d5 preserves 131 unaffected controls with checked source/library provenance and reruns the affected long/lifecycle DFlash cases as well as missing controls. The actual-streaming prerequisite and byte comparisons remain unchanged; the tighter Q8/Q8 long DFlash control already passes. This changes test setup only, not production allocation policy.
