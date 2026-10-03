# Benchmark specification and operator interface

Status: the initial baseline runner is implemented and the first IQ3_S 4096-token MTP baseline completed successfully. It launches an owned server, verifies health, device inventory, server context, model path, and MTP draft-context initialization, builds an exact-depth token prompt, performs one warmup plus repeated fresh requests, records raw responses/logs/manifest data, and reads existing speculative counters from `/metrics`. Prefix reuse, suites, MTP sweeps, paired comparison, packaging, and stress runs remain planned. See `progress.md` for the measured baseline.
Specification date: 2026-10-01. Goals are defined in `goal.md`.

## 1. Purpose and score

Measure real **single-user MTP serving performance** on the RX 7900 XTX. Use `llama-server` as the primary test surface. A kernel timer or `llama-bench` result can explain a bottleneck but cannot replace an MTP serving result.

Primary score: final output tokens per second for a completed request, with real MTP verification enabled. Also report interactive latency, prompt ingestion, request duration, acceptance, and environment information. Do not count rejected draft tokens as generated output. Do not score synthetic acceptance modes.

The benchmark is mandatory evidence for retained optimizations and is designed for manual execution by the user. The coding agent prepares binaries, presets, and documented commands; the user can run them independently. A terminal timing line is useful for quick checks, but is not a controlled comparison.

## 2. Fixed workload matrix

| Preset | Model filename | Server context capacity | Measured prompt depths |
|---|---|---:|---|
| `qwen35-iq3s` | `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | 200000 | 4096, 16384, 32768, 65536, 98304 |

Here 4k means 4096 tokens and 96k means 98304 tokens. Context capacity is fixed even for the smallest test. The harness requests 200000 tokens; upstream rounds this to the verified effective 200192-token context on its 256-token alignment boundary, and records both values. Benchmark prompt depth never exceeds 98304. Generated tokens naturally increase live state beyond initial prompt depth; reserve capacity for output and speculative work. The cap applies to the initial populated benchmark depth, not an artificial cutoff of every generated token.

One slot, one outstanding request, no user traffic during measurement. The IQ3_S model is the only benchmark workload in the current project.

Verify actual context capacity and MTP activation from the server, not only the requested command line. Do not silently shorten a prompt, reduce capacity, disable MTP, change cache types, or offload extra work to the CPU to make a failed run pass.

## 3. Suites and execution cost

| Suite | Depths | Prompt selection | Output | Measured repetitions |
|---|---|---|---:|---:|
| `quick` | 4096 | One fixed chat fixture | 512 tokens | 5 |
| `common` | 4096, 16384, 32768 | Two chat fixtures and one code fixture | 512 tokens | 10 |
| `full` | All five depths | Same three fixture families with depth-specific context | 512 tokens | 10 |
| `stress` | Selected agreed depth, default 16384 | Sequential chat requests | Up to 512 each | 100 |

Defaults are starting policies, not claims that these durations fit every machine. `--runs` and `--output-tokens` override them. Warm up once per server/configuration/depth before measured generation; retain warmup records but exclude them from scores. A quick test is a filter, not final evidence.

Long prompts are expensive to prefill. Separate two workload modes:

- `fresh`: each measured request starts from empty request state with no prefix reuse. Measures full prompt ingestion and end-to-end latency.
- `reused-prefix`: prepare the context, then restore/reuse the same prefix for each generation measurement. Measures warm chat continuation at the specified depth. Record exactly how many prompt tokens were reused and newly evaluated.

Use `reused-prefix` for frequent generation iteration if the pinned server safely supports reuse for this model/state. Validate that each repetition begins from the same prefix, without previous generated output. Do not assume persistent-slot save/restore supports every hybrid or recurrent architecture. If reliable restore/reuse cannot be established, use fresh preparation and label its cost explicitly.

For `full`, run the generation matrix and a fresh-prompt pass with 3 repetitions per fixture/depth. `--fresh-runs` controls that second pass. Never mix fresh and reused-prefix TTFT/prompt scores in one statistic. Display an estimated workload count before execution; do not require confirmation to proceed.

A stress run checks sustained throughput, acceptance degradation, temperature behavior, and state reuse. It is useful after scheduling/state changes, not mandatory after every small shader edit.

## 4. Prompt fixtures and exact token depth

Maintain a versioned prompt manifest. Each fixture records ID, category, source/license, text SHA-256, construction rules, tokenizer identity, and intended task. Use reproducible public or project-authored text, not private conversations.

Use chat-like requests as the main fixtures and a smaller code fixture to reflect actual use. Long-depth fixtures should contain varied coherent text/code, not one token repeated thousands of times. Acceptance depends on content, so a single easy repetitive prompt is insufficient for final validation.

Use the model's actual tokenizer and the pinned server's tokenization interface. Construct depth after all templates, system messages, separators, and special tokens are included. Prefer a controlled, pre-rendered prompt and validated token-array request when supported. Record the exact token sequence hash and observed prompt token count. If the native endpoint cannot preserve chat semantics, record that limitation and add a chat endpoint validation case.

Target exact depths. If endpoint behavior inserts tokens, adapt construction based on observed counts. A request at the wrong depth must not silently appear under the target label. Never estimate depth from character count.

Use fixed sampling, initially greedy for controlled comparisons, plus a small validation set using the user's eventual real sampling preset. Save all sampler parameters, seed, chat template, and reasoning-mode settings where applicable. Greedy acceptance tuning might not be optimal for sampled conversation; validate the selected settings in the real mode before deployment.

Request a fixed output budget. For a supported controlled speed fixture, suppress early end-of-generation consistently in both variants and record this choice. Include normal stopping behavior in real-chat validation. If output is shorter than requested, retain actual counts, mark the reason, and avoid treating an early-stopped request as equivalent to a 512-token run.

## 5. Metric definitions

Use structured upstream response data when available. Capability-detect fields for the pinned revision; missing metrics are `null` with an explanation, never fabricated zeroes. Keep raw responses and server logs.

| Metric | Definition and handling |
|---|---|
| `final_output_tokens` | Actual sampled/emitted tokens, including internal output tokens if the server counts them; distinguish visible text from reasoning tokens |
| `mtp_output_tps` | `final_output_tokens / server_generation_seconds` when upstream generation timing covers the complete MTP draft/verify/commit loop; validate that definition in source |
| `client_delivery_tps` | Final token count divided by client time from first content-bearing event to last content-bearing event; record denominator and streaming burst limitations |
| `ttft_ms` | Monotonic client time from request submission to first content-bearing streaming event, excluding role-only and keepalive events |
| `request_ms` | Submission to completed response received |
| `prompt_eval_tokens` | Tokens actually evaluated this request, separately from total logical prompt depth |
| `prompt_tps` | Newly evaluated prompt tokens divided by server prompt-evaluation time; meaningful for nonzero evaluated counts |
| `draft_proposed` | Number of speculative candidate tokens proposed, using the pinned implementation's documented counter semantics |
| `draft_accepted` | Number of proposed draft tokens committed through verification; exclude ordinary target/bonus tokens |
| `draft_acceptance` | `draft_accepted / draft_proposed`; null if denominator is zero |
| `accepted_per_draft_step` | Accepted draft tokens divided by drafting iterations; report separately from total output per cycle |
| `draft_ms`, `verify_ms` | Phase timings if reliably supplied/instrumented; do not invent a decomposition from total time |
| `vram_peak_bytes` | Sampled peak with sampling interval, device, and source; distinguish whole-device use from process allocation |
| Environment telemetry | GPU clocks/temperature/power when available; missing telemetry does not itself invalidate a run |

Server generation timing is the main score only after its MTP coverage is verified. Until then, report it as an unvalidated server timing and use clearly labeled client throughput/end-to-end timings. Streaming events may contain several tokens, especially with MTP; one SSE event is not one token. Do not estimate final token count by counting chunks or retokenizing visible text when authoritative server token counts exist.

Define measurement boundaries in the result schema. A first-to-last delivery interval can be very short for bursty or tiny outputs; mark unusable denominators rather than producing a misleading huge rate. Optional inter-event gaps describe delivery behavior, not true per-token latency.

## 6. MTP tuning protocol

MTP must be verified active, not merely requested. Establish GGUF head support, compatible upstream implementation, actual draft/verify execution, and valid counters. If MTP fails to activate, fail the scored run and diagnose it; do not report fallback generation as an MTP score.

Initial candidate `n-max` values: `1,2,4,8`. These are sweep candidates, not promised valid settings. Check the pinned implementation/model's limits; skip unsupported values with an explicit reason. Sweep other MTP controls only if real help/source and profiling justify them.

1. Screen candidates at 4k and 16k across representative fixtures.
2. Validate the best candidates at 32k, then 64k/96k.
3. Select for final throughput, TTFT, and stability. Acceptance alone does not choose the winner.
4. Freeze settings during a kernel A/B comparison.
5. If a patch changes the best setting, report a separate retuned comparison alongside the fixed-setting result.

Do not tune on one prompt then claim all-content improvement. Do not introduce automatic depth-dependent policy until its benefit and guard conditions are measured. Preserve the actual sampler and verification rules; changing acceptance semantics is not legitimate tuning.

## 7. Proposed command-line interface

The implemented baseline runner supports `--preset qwen35-iq3s`, `--model`, `--server`, `--mmproj`, `--config`, `--depth`, `--runs`, `--output-tokens`, `--host`, `--port`, `--timeout-seconds`, `--startup-timeout-seconds`, `--results-dir`, `--label`, and `--dry-run`. All other options in this section remain planned. JSON presets are versioned, and CLI values override presets.

| Option | Meaning / default |
|---|---|
| `--preset NAME` | `qwen35-iq3s`; model path must be supplied locally |
| `--model PATH` | Exact GGUF path; required unless stored in local config |
| `--server PATH` | llama-server executable to launch; required for launch mode |
| `--baseline-server PATH` | Control executable for paired A/B mode |
| `--server-url URL` | Attach to an already running server instead of launching; mutually exclusive with launch/A-B settings |
| `--suite NAME` | `quick`, `common`, `full`, `stress`; default `quick` |
| `--ctx N` | Allocated capacity; preset default 200000 or 131072 |
| `--depth N` | One exact prompt depth from the agreed five; overrides suite depth list |
| `--depths LIST` | Comma-separated subset of agreed depths; mutually exclusive with `--depth` |
| `--runs N` | Measured repetitions per fixture, depth, configuration, and variant |
| `--warmup N` | Excluded warmups per configuration/depth; default 1 |
| `--fresh-runs N` | Fresh-prompt pass repetitions for full suite; default 3, 0 disables and labels omission |
| `--output-tokens N` | Generation budget; default 512 |
| `--prompt-mode MODE` | `fresh` or `reused-prefix`; default reused-prefix with explicit capability handling |
| `--mtp-n-max N` | One supported candidate draft length; use selected local value, initially upstream default |
| `--mtp-sweep LIST` | Candidate draft lengths to test; mutually exclusive with `--mtp-n-max` |
| `--batch N`, `--ubatch N` | Upstream logical and physical batch settings; initially inherit recorded preset/server defaults |
| `--cache-k TYPE`, `--cache-v TYPE` | Cache types supported by model/backend; keep constant for A/B |
| `--flash-attn MODE` | Supported upstream flash-attention setting; pin resolved mode |
| `--gpu-layers VALUE` | Upstream offload request, initially all supported layers; record actual placement |
| `--threads N` | CPU inference thread setting; record CPU and resolved count |
| `--sampling-config PATH` | Versioned complete sampler settings; initially greedy fixture configuration |
| `--seed N` | Reproducibility seed where supported; default 1234 |
| `--config PATH` | Local paths/runtime overrides as JSON; not committed with machine-specific paths |
| `--label TEXT` | Human-readable experiment name |
| `--results-dir PATH` | Default `bench-results/` relative to checkout |
| `--compare PATH` | Compare completed run with prior result directory/manifest |
| `--timeout-seconds N` | Per-request timeout; initial default 1800, configurable for slow prefill |
| `--startup-timeout-seconds N` | Startup/health timeout; initial default 600 |
| `--port N` | Loopback server port; default 8080, fail on collision |
| `--dry-run` | Print fully resolved configuration and command without starting inference |
| `--strict-metrics` | Require acceptance counters and validated server throughput; default off while adapters are developed |
| `--help`, `--version` | Usage and harness/build/schema identity |

The harness always sets MTP and single-slot serving in launch mode. Do not add a normal target-only preset. Do not provide arbitrary shell command interpolation: pass executable and arguments as an argv array. If extra server arguments are eventually needed, store a validated JSON array and reject conflicting preset-owned settings.

Attach mode must verify or require an exported server manifest for model identity, allocated capacity, MTP settings, sampler/configuration, and offload. If critical facts cannot be established, label the run exploratory and exclude it from acceptance evidence. The harness must never stop or restart a server it did not launch.

## 8. Planned usage examples

After the harness is implemented and packaged:

```bash
./rx7900xtx-bench --preset qwen35-iq3s --model /models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --server ./build-rx7900xtx/bin/llama-server --runs 5
```

Run the common-use suite:

```bash
./rx7900xtx-bench --preset qwen35-iq3s --config ./bench.local.json --suite common --runs 10
```

Tune MTP:

```bash
./rx7900xtx-bench --preset qwen35-iq3s --config ./bench.local.json --depths 4096,16384 --mtp-sweep 1,2,4,8 --runs 5
```

Compare fixed settings, using an illustrative candidate length of 4 rather than an established optimum:

```bash
./rx7900xtx-bench --preset qwen35-iq3s --config ./bench.local.json --server ./build-rx7900xtx/bin/llama-server --baseline-server ./build-control/bin/llama-server --suite common --mtp-n-max 4 --runs 10 --label iq3s-kernel-v1
```

Full validation, without any depth above 96k:

```bash
./rx7900xtx-bench --preset qwen35-iq3s --config ./bench.local.json --suite full --runs 10 --fresh-runs 3
```

One 64k measurement and an offline comparison:

```bash
./rx7900xtx-bench --preset qwen35-iq3s --config ./bench.local.json --depth 65536 --runs 20
./rx7900xtx-bench compare --baseline bench-results/control-run --candidate bench-results/optimized-run
```

`--config` examples assume the local file supplies required model/server paths. On Windows the packaged executable is `rx7900xtx-bench.exe`; adapt executable and model paths. A single executable is the intended deliverable, packaged per OS from a Python implementation, for example with PyInstaller. Source execution must remain available for development. All Python development and packaging commands must run in the project-local `.venv-bench` virtual environment; do not install benchmark dependencies globally. No executable is created by this specification.

## 9. Fair comparison and statistical reporting

Prefer control and optimized builds from the **same source commit**, differing only in the project optimization switch. Both use upstream MTP. Keep driver, compiler, build type, model hash, prompt tokens, sampler, MTP settings, capacity, cache types, offload, and batch settings identical.

Execute only one GPU server at a time. Alternate or randomize A/B order in blocks rather than measuring all control runs before all optimized runs. Record the schedule. Blocks may contain several requests to avoid model-load overhead; prepare equivalent state for each measured request. Treat blocks as independent units when computing uncertainty if repetitions share server/state conditions.

Record every sample, mean, median, standard deviation, sample count, and a 95% confidence interval for paired relative improvement using a documented paired/block bootstrap method. Use deterministic bootstrap seeds. With very few blocks, explicitly mark uncertainty as weak. Do not equate “larger than standard deviation” with a significance test.

Calculate per-fixture/depth improvement as `(candidate / baseline - 1) * 100` for throughput. For latency, label lower as better and use a consistent improvement convention. Acceptance percentages have percentage-point changes, not throughput gains.

Do not delete slow runs merely because they hurt the result. Record failures, timeouts, background interference, and thermal events. Exclusions require explicit objective reasons and retain the raw records. Avoid repeatedly testing a marginal result until it becomes favorable; confirm promising experiments on a fresh run or held-out fixture.

Report every depth, with 4k/16k as the primary decision rows and 32k as the next priority. Do not invent usage weights or hide a regression inside one aggregate score. A full comparison with missing rows is incomplete. No universal percentage threshold is imposed; weigh evidence, practical benefit, and complexity.

## 10. Results and provenance

```text
bench-results/<timestamp>-<label>/
  manifest.json
  samples.jsonl
  summary.json
  summary.csv
  report.md
  logs/
    <variant>-<block>.log
  responses/
    <sample-id>.json
  traces/                              optional profiling artifacts
```

Include a schema version and units. The manifest stores source/upstream base SHAs, dirty-tree state and diff hash, binary hashes, harness version, build flags, compiler, OS/kernel, CPU/RAM, Vulkan device/driver, model SHA-256/size/metadata, prompt/token hashes, effective command, environment controls, and all resolved runtime settings. Hash the model once per trusted identity change; do not reread a 12 GB file for every sample.

Samples contain unique IDs, variant, block/order, fixture, requested/observed depth, cache mode, actual output count, timings, speculation counters, telemetry, validity status, and errors. Record warmups separately. Store raw counters so definitions can be corrected later.

Terminal output should show a compact table of model/depth/fixture, control and candidate MTP throughput, relative change and interval, TTFT, acceptance, and measured VRAM. `report.md` provides readable details. JSON/CSV permit later analysis. There is no required hosted dashboard.

Exit codes: `0` completed successfully; `1` operational/test failure; `2` invalid configuration; `3` incomplete strict evidence. A completed run can still show a performance regression; the result file states that outcome. Comparisons must reject incompatible configurations by default rather than manufacture a percent delta.

## 11. Process lifecycle and profiling

Launch on loopback, wait for real readiness, track the child process, capture logs, and stop only owned processes on completion/error/Ctrl+C. Never kill a process just because it occupies the requested port. Use monotonic clocks for durations and UTC timestamps for artifacts.

Begin profiling with end-to-end phase attribution and Vulkan dispatch timing if available. Then inspect the actual dominant kernels, transfers, synchronization, occupancy/register/LDS behavior, and memory utilization using tools supported by the real driver. Radeon profiling availability is environment-dependent; do not promise counters unavailable on the machine.

Keep profiling runs separate from scored runs because instrumentation changes timings. No profiling hypothesis is an accepted speed gain until ordinary MTP serving measurements confirm it.

## 12. Delivery milestones

1. Launch/health/client/token-depth handling and raw request timings.
2. Model inventory, manifests, summaries, comparison, and manually runnable package.
3. Capability-checked MTP counters, sweep support, and paired control/optimized runs.
4. Reliable prefix reuse/fresh separation, complete suite, and stress validation.
5. Optional detailed profiling integrations.

Before starting kernel experiments, require a reproducible MTP-active baseline, accurate final token counts, trustworthy time boundaries, and correct prompt depth. Acceptance counters can initially be missing with explicit labels, but are required to conclude the MTP tuning task.

## Reference

[Upstream server API and arguments](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Current documented MTP arguments include `--spec-type draft-mtp` and `--spec-draft-n-max`; verify the checked-out binary's help and implementation before mapping harness options. Upstream flags, metrics, and state-reuse capabilities can change.
