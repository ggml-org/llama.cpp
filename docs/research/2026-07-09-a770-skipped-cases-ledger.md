# Arc A770 skipped cases ledger

Date: 2026-07-09
Branch: `benchmark/fork-vs-upstream-a770`
Purpose: row-by-row audit trail of every skipped, omitted, reduced, or discarded case during fork-vs-upstream comparison and benchmarking.

Legend:
- Status `skipped`: code path available, not run in that invocation
- Status `omitted`: not meaningful or impossible in comparison set
- Status `reduced`: run happened, but reduced scope means broader variant not yet run
- Status `discarded`: run started, artifact/result not trustworthy, not cited

## Audit table

| source | status | exact case / path | gate / env / command needed to run | why skipped / omitted / reduced / discarded | omission risk | follow-up |
|---|---|---|---|---|---|---|
| test harness | skipped | `[5] flash attention turbo KV` in default `./build-port/bin/test-sycl-turbo-correctness` run | `LLAMA_TEST_TURBO_FA=1` | Harness keeps turbo FA opt-in because broken turbo FA can hang JIT/device on A770. Default run intentionally avoids hang-risk path. | High for reachability claim if no separate opt-in run. | Done separately with `LLAMA_TEST_TURBO_FA=1`. Cite that run, not default harness. |
| test harness | skipped | `[5] flash attention turbo KV` in XMX-only harness run | `LLAMA_TEST_TURBO_FA=1 GGML_SYCL_FA_XMX=1` | XMX-only run proved f16 XMX path, but left turbo gate off. Did not cover turbo-XMX router path. | High if claiming all FA kernel work reachable. | Done separately with both env vars set. |
| test harness | skipped | `[7] flash attention d=256` generic opt-in block in default and turbo runs | `LLAMA_TEST_FA256=1` | Harness comments mark d=256 generic FA as known A770 hang-risk. Default sweep leaves it off so validation terminates instead of wedging GPU. | Medium. Leaves generic d=256 FA unverified in this session. | Keep in skip ledger until explicit hang-risk retest on changed driver/IGC stack. |
| test harness | skipped | `[8] InnerQ FA` block | `LLAMA_TEST_INNERQ=1` | InnerQ remains default-off per P3.2 policy. Reachability hook exists, but this session focused fork-vs-upstream plus kernel reachability already demanded by user. | Medium-high. InnerQ code reachable in source, not runtime-validated here. | Run dedicated InnerQ opt-in pass after main matrix stable. |
| test harness | omitted | any support-check skip inside executed turbo/XMX probes | none; would show in harness output | No such skips observed in executed `LLAMA_TEST_TURBO_FA=1`, `GGML_SYCL_FA_XMX=1`, or combined runs. This row exists to record absence. | Low. | None. |
| benchmark matrix | reduced | all matrix rows run with `--quick` profile | rerun `python scripts/bench-a770-fork-unique.py --timeout 600 --out-dir ...` without `--quick`, or use longer `p/n/r` | Current matrix intentionally uses `p64/n16/r1`. One sample only. Zero stddev by construction. Good for smoke/direction, bad for final throughput claims. | High for stable perf claims; low for dispatch/reachability. | Rerun full profile after partial smoke matrix finishes and problem rows known. |
| benchmark matrix | reduced | coherence/correctness pairing for each `llama-bench` row | run `llama-cli` or `llama-completion` per selected case with `--temp 0` / deterministic prompt | `llama-bench` correctness-blind. This session started with harness + bench matrix first. Per-case coherence probes still pending. | High if bench-only numbers used to claim working output. | Add deterministic coherence probes after matrix completion. |
| benchmark matrix | reduced | full-context / longer decode characterization | longer `-p`, `-n`, `-r`; likely no `--quick` | Current bench smoke too short to expose deep-context crossover or stable variance. | Medium-high. | Full rerun after smoke matrix. |
| benchmark matrix | reduced | non-quick mistral and qwen3 rows | same runner without `--quick` | Current job still in quick mode for tractable first-pass coverage across 60 cases. | Medium. | Second pass. |
| benchmark matrix | discarded | first background matrix run `bg_2` | none; rerun corrected script | First matrix invocation canceled after script correction. Old runner had misleading XMX labels and produced unusable artifact state. | High if cited. | Ignore `bg_2`. Use `bg_3` artifacts only. |
| benchmark matrix | omitted | upstream turbo2/turbo3/turbo4 rows | impossible in upstream tree | Upstream checkout lacks `GGML_TYPE_TURBO2_0`, `GGML_TYPE_TURBO3_0`, `GGML_TYPE_TURBO4_0`, `TQ3_1S`, `TQ4_1S` in `ggml/include/ggml.h`. No like-for-like upstream turbo command exists. | Low for comparison honesty; impossible case. | Use upstream f16/q8_0 as controls, fork turbo as fork-only surface. |
| benchmark matrix | omitted | upstream mixed `q8_0/turbo3` row | impossible in upstream tree | Mixed post-auto-asymmetric control is fork-only because turbo V type absent upstream. | Low. | None; document as fork-only. |
| benchmark matrix | omitted | upstream turbo-XMX rows | impossible in upstream tree | Upstream lacks turbo KV types, so combined turbo-XMX path cannot exist there even if XMX FA exists for f16/q8_0. | Low. | None. |
| benchmark matrix | reduced | default turbo rows as exact-type statements | use pure rows with `TURBO_LAYER_ADAPTIVE=0 TURBO_AUTO_ASYMMETRIC=0` | Default fork runtime can mutate requested KV layout via static `adaptive_mode` and auto-asymmetric K policy. Default rows still useful, but not exact-type proof. | High if mislabeled. | Treat default rows as policy-on behavior. Use pure rows for exact-type comparisons. |
| benchmark matrix | reduced | XMX default turbo rows as exact-type statements | add `GGML_SYCL_FA_XMX=1 TURBO_LAYER_ADAPTIVE=0 TURBO_AUTO_ASYMMETRIC=0` | High-GQA model can auto-convert K to q8_0, which would turn claimed turbo-XMX row into mixed q8_0/turbo VEC path. | High. | Fixed in current runner via `fork-xmx-pure-*` rows. |
| benchmark matrix | reduced | one-process multi-KV bench sweeps | one process per case only | `src/llama-kv-cache.cpp` uses `static const int adaptive_mode`; one process with many KV cases can become order-dependent. | High if violated. | Current runner already isolates one process per case. |
| upstream comparison | omitted | no-index whole-directory diff | use tracked Git diff only | Raw directory diff polluted by generated/build/untracked files. Not trustworthy basis for technical comparison. | High if used. | Already replaced with tracked diff against upstream commit. |
| upstream comparison | omitted | cross-repo commit ancestry assumptions | fetch remote or compare trees by explicit path/object alternate | Fork and upstream are separate repos. Shared object graph not guaranteed. | Medium. | Comparison already done with explicit upstream checkout and alternate object lookup for read-only diff. |
| gate script | skipped | `scripts/turbo-quality-gate.sh` default correctness binary path | set `CORRECTNESS_BIN=build-port/bin/test-sycl-turbo-correctness` and `LLAMA=build-port/bin` | Script defaults point at `../build-sycl-fp32/bin/test-sycl-turbo-correctness` and `~/local_llms/...`, wrong for this tree. Running raw would test wrong binary or fail for wrong reason. | High if used as proof unmodified. | Only run with explicit overrides, or patch script in future. |
| results doc | reduced | final 60-case matrix analysis | wait for `bg_3` completion | This document built incrementally while job still running. Partial snapshot cannot support end-state claims. | Medium. | Append/update when job completes. |

## Current executed follow-ups that cleared earlier skips

These rows started as skips, then were explicitly covered later in session:

| previously skipped thing | follow-up command | outcome |
|---|---|---|
| turbo FA reachability | `LLAMA_TEST_TURBO_FA=1 ./build-port/bin/test-sycl-turbo-correctness` | covered; turbo3/turbo4 pass, turbo2 xfail |
| f16 XMX reachability | `GGML_SYCL_FA_XMX=1 ./build-port/bin/test-sycl-turbo-correctness` | covered; XMX pass |
| turbo-XMX combined reachability | `LLAMA_TEST_TURBO_FA=1 GGML_SYCL_FA_XMX=1 ./build-port/bin/test-sycl-turbo-correctness` | covered; no support skip, no GATE fail |
| full 60-case matrix | `python scripts/bench-a770-fork-unique.py --quick` (bg_3) | covered; 60/60 ok, `docs/research/a770-fork-unique-2026-07-09/` |
| coherence probes | `llama-completion` deterministic smoke, 5 fork cases x 3 models | covered; 15/15 coherent, `docs/research/coherence-2026-07-09/` |

## Newly-recorded skips found while mining RALPH/ASSUMPTIONS docs

| source | status | exact case / path | gate / env / command needed | why skipped / caveat | risk | follow-up |
|---|---|---|---|---|---|---|
| benchmark matrix | discarded-for-correctness | `fork-nonfa-turbo3-turbo3` bench row | run turbo under `-fa on` only; use CPU-FA for turbo PPL | Prior HARD RULE: turbo KV is FA-only. Non-FA `-fa off` transposes block-quant V; correctness not valid there even though the fork added a dequant-before-transpose graph fix. Row measures throughput only, correctness-blind. | High if cited as a valid turbo output path. | Keep row as throughput datapoint only; never cite for quality. |
| upstream comparison | omitted | fresh full-corpus PPL this session | `llama-perplexity` CPU-FA 564-chunk per model | Prior RALPH PPL matrices already exist and are the authoritative quality axis; not re-run this session. | Low; prior numbers stand. | Rerun only if kernels change. |
| upstream comparison | omitted | long-context throughput crossover | longer `-d`/depth sweep with `llama-bench -d` | `--quick` smoke is depth 0. Prior perf-findings doc measured the turbo-loses-at-depth crossover already. | Medium for "turbo fast?" framing; reframe says capacity not speed. | Depth sweep if speed ever re-chased. |
| test harness / runtime | not-proven | InnerQ consumer path (`scale_inv tensor updated finalized=1`) | `LLAMA_ENABLE_INNERQ=1` live Qwen3 probe on a build where consumer fires | Producer fires, consumer never observed. Inapplicable to qwen3 by design (auto-asymmetric K downgrade). Prior runtime proof blocked by AOT/IGC offload-link failure on `build-turbo-aot`. | High for any "InnerQ works end-to-end" claim. | Needs consumer-path fix + non-qwen3 turbo-K model, or AOT unblock. |
| build | omitted | AOT (`ocloc` acm-g10) build path | `/home/svnbjrn/build-turbo-aot` AOT rebuild | This session used the JIT `build-port` which builds/runs clean. The AOT path is the one the RALPH InnerQ loop was blocked on. | Low for reachability; the JIT build is a valid runtime. | AOT unblock is a separate track. |

## Still-open skip debt

Open items after this write:
- non-quick (long `p/n/r`) throughput matrix
- long-context depth crossover sweep
- generic d=256 FA opt-in section (`LLAMA_TEST_FA256=1`)
- InnerQ opt-in consumer-path live proof (`LLAMA_TEST_INNERQ=1` / `LLAMA_ENABLE_INNERQ=1`)
- fresh full-corpus PPL rerun (prior RALPH numbers currently authoritative)
- AOT build path
