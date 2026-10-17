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

Status: pending.

## KQ3 - Quant-dependent memory accounting

Verify host layout, resident/ring pages, row publication, mutable tails, conversion and attention scratch, and phase transitions for selected encodings. Cover auxiliary MTP cache geometry and gathered-prefill requirements. Use actual block sizes, row sizes, padding and alignment in every quote and transfer.

Tests cover all candidate layouts, asymmetric pairs, exact fits, one-byte-short grants, overflow, alignment, partial publication and resource retention. An allocation quote must cover the actual selected path; a fallback with different requirements must be quoted before execution.

Acceptance: no Q8/Q4 size assumptions, no hidden reserve, no out-of-bounds access under focused sanitizers.

Status: pending.

## KQ4 - Capability-based MTP integration

Replace the server's Q8_0/Q4_0-only check after the lower layers qualify. Prepare required native and streamed prediction/catch-up/verification/replay variants before arena allocation. Keep target and MTP cache formats identical and the auxiliary cache in the existing shared physical policy. DFlash inherits target formats under KQ1 policy while retaining bounded draft storage.

Add regressions for Q8/Q8, Q5_1/Q5_1, Q8/Q4 and asymmetric formats: context construction, attachment, publication, lease reuse, rejection, truncation and cleanup. Verify bad pairs and inadequate budgets return actionable initialization errors rather than assertions or recovered execution failures.

Acceptance: supported pairs start and execute through all admitted widths; unsupported paths are rejected before requests.

Status: pending.

## KQ5 - Numerical and lifecycle qualification

Extend the existing real-model native/streamed oracle with the actual selected quant pair. Exercise target-only, MTP and DFlash with matching execution settings. Cover resident, light-ring, wrapped-ring and histories larger than the ring, mutable tails, rejection, context exhaustion, cancellation and serial request reuse.

Use exact output/state comparisons where the stock execution order permits them. Different formats are not expected to generate identical text. Neither is TG1 automatically interchangeable with stock wider-query arithmetic. Record and investigate any mismatch without weakening old tests or tolerances.

Acceptance: unchanged suites and expanded numerical/lifecycle cases pass; focused ASan/UBSan/CUDA checks are clean; any external qualification limitation is explicit.

Status: pending.

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

Acceptance: all 45 requested points pass, matched-mode correctness qualifies, Q8/Q4 performance remains comparable, documentation is accurate, production is healthy, and no unqualified support claim remains.

Status: pending.

## Progress log

- Roadmap saved before implementation. No runtime changes or new qualification matrix started yet.
