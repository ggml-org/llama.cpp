# DeepSeek-V4 ROCm benchmark harness

This directory contains the controlled PP baseline/profiling harness for
`docs/deepseek-v4-flash-rocm-performance.md`.

## Safety

`run-pp.sh` acquires `$HOME/llama-jobs/gpu.lock`, then checks
`rocm-smi --showpids` twice: once before manifest capture and again immediately
before launch. Discovery errors fail closed. Any active ROCm process causes a
refusal. The benchmark runs in its own process group; cleanup and both watchdogs
signal the complete group with bounded TERM-to-KILL handling. The script never
signals a process that existed before its own launch.

Do not use `DSV4_ALLOW_BUSY_GPUS=1` for controlled measurements.

Inspect the exact command without querying ROCm or starting a GPU process:

```bash
scripts/dsv4-rocm/run-pp.sh --dry-run
scripts/dsv4-rocm/profile-pp.sh --dry-run
```

## Five-minute quick baseline

After the GPU owner has stopped inference and confirmed a benchmark window:

```bash
cd /home/edwin/llama.cpp-rdna2
DSV4_LABEL=base-quick \
DSV4_PROMPTS=512,2048 DSV4_UBATCHES=256 \
DSV4_REPS=3 DSV4_TIMEOUT=300 \
scripts/dsv4-rocm/run-pp.sh
```

Defaults match the current target:

- IQ2_M first shard under `/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/`;
- all layers on four GPUs;
- tensor split `1/1/1/1`;
- F16 K/V cache;
- flash attention on;
- batch 512 and ubatch 256;
- NCCL all-reduce, P2P, HIP graphs, gfx1030 override, and no scratch reclaim.

The five-minute absolute cap starts when llama-bench emits its first measured
prompt-run marker, after the initial model load, context creation, and first
warmup. TERM is sent two seconds before the deadline and KILL is sent at the
absolute deadline if any process-group member remains. A separate 20-minute
startup cap prevents an infinite load/warmup hang.

JSONL preserves fully completed cases if a run is truncated. A truncated or
shape-incomplete run is explicitly marked `complete=false` and exits 3 after
writing its summary. It is evidence, not a matched A/B result. Compare only
identical complete shape/repetition sets. Run 8K separately so a slower 8K case
cannot hide completion of the short grid:

```bash
DSV4_LABEL=base-8k DSV4_PROMPTS=8192 DSV4_REPS=2 \
DSV4_TIMEOUT=300 scripts/dsv4-rocm/run-pp.sh
```

The script runs PP only (`--n-gen 0`). Speculative draft settings are excluded
because they do not belong in first-pass PP attribution.

## Frozen baseline binaries

Selecting `/home/edwin/baseline-bin/bin/llama-bench` also selects sibling DSOs
through the harness-managed `LD_LIBRARY_PATH`, preventing a candidate rebuild
from silently changing the baseline libraries:

```bash
DSV4_BENCH=/home/edwin/baseline-bin/bin/llama-bench \
DSV4_LABEL=frozen-base scripts/dsv4-rocm/run-pp.sh
```

The manifest stores the executable SHA-256, `ldd` resolution, SHA-256 for all
resolved local llama/ggml DSOs, selected CMake cache when available, git diff,
and untracked-file hashes. Verify resolved DSO paths before accepting an A/B.

## RDNA2 routed-MMQ screening

`GGML_HIP_RDNA2_MMQ_J` is an experimental, opt-in tile-width override for
routed MMQ on RDNA2 only. The default dispatch is unchanged when it is unset.
Valid values are supported multiples of eight from 8 through 128; an
unsupported type/configuration fails rather than silently changing the test.
For example:

```bash
GGML_HIP_RDNA2_MMQ_J=16 \
DSV4_LABEL=mmq-j16-quick DSV4_PROMPTS=512,2048 DSV4_REPS=3 \
scripts/dsv4-rocm/run-pp.sh
```

On the target IQ2_M model, J=16 is the current exploratory winner. It is not
yet a general RDNA2 default: the focused fixture shows that strongly skewed
expert routing changes the optimum. Keep the variable and its value in the
manifest, use complete paired runs, and do not carry this setting to unrelated
models without screening their routing shape.

`test-mmid-rdna2` defaults to a fast prototype-weight fixture for performance
screens. Its `--fixture unique` mode independently quantizes every
expert/output row for correctness. The target-shape check uses N=512,
batch=256, 256 experts, top-6, and K=256 (one quant block) so coupled
expert/row addressing is distinguishable without minutes of setup. Dump a J64
reference and compare J16 for both `iq2_xxs`/`iq3_xxs` and `uniform`/`hot`:

```bash
GGML_HIP_RDNA2_MMQ_J=64 build/bin/test-mmid-rdna2 \
  --type iq2_xxs --fixture unique --k 256 --n 512 --batch 256 \
  --experts 256 --top-k 6 --routing uniform --dump-output /tmp/mmid.bin
GGML_HIP_RDNA2_MMQ_J=16 build/bin/test-mmid-rdna2 \
  --type iq2_xxs --fixture unique --k 256 --n 512 --batch 256 \
  --experts 256 --top-k 6 --routing uniform --compare-output /tmp/mmid.bin
```

## RDNA2 DSV4 hidden-channel mixer screening

`GGML_HIP_RDNA2_HC_MIXES=1` opts the exact contiguous F32
M=24,N=256,K=16384 DSV4 hidden-channel mixer shape into a 12x16x256 LDS-tiled
kernel on wave32 RDNA2. Unset/`0`, other devices, other types/layouts/shapes,
and all non-HIP backends keep the existing dispatcher and rocBLAS/generic
fallbacks. Values other than `0` or `1` fail closed.

The setting is still a screening control, not a general F32 GEMM default. Run
the focused benchmark through the normal GGML backend graph before whole-model
A/B:

```bash
unset GGML_HIP_RDNA2_HC_MIXES
build/bin/test-hc-mixes-rdna2 --iterations 100 --dump-output /tmp/hc-mixes.bin
GGML_HIP_RDNA2_HC_MIXES=1 \
build/bin/test-hc-mixes-rdna2 --iterations 100 \
  --compare-output /tmp/hc-mixes.bin
```

The benchmark also checks a double-accumulation CPU reference. Hold
`GGML_HIP_RDNA2_MMQ_J=16` constant in every arm of whole-model tests so only
the hidden-channel path changes.

## RDNA2 DSV4 lightning-indexer subwave path

`GGML_HIP_RDNA2_LID_SUBWAVE=4` enables the bitwise-preserving four-lane
subwave reduction for the guarded DSV4 F16 shape: 128 embedding values, 64
heads, batch 256, one stream, and KV 1–4096 on wave32 RDNA2. Unset/`0` and all
other devices, shapes, types, and backends keep the generic vector kernel.
Other values fail closed. The manual fixture requires explicit path identity;
run reference and candidate in separate processes:

```bash
GGML_HIP_RDNA2_LID_SUBWAVE=0 build/bin/test-lightning-indexer-rdna2 \
  --kv 4096 --expect-path reference --dump-output /tmp/lid-kv4096.bin
GGML_HIP_RDNA2_LID_SUBWAVE=4 build/bin/test-lightning-indexer-rdna2 \
  --kv 4096 --expect-path subwave4 --compare-output /tmp/lid-kv4096.bin
```

The fixture's candidate expectation is intentionally bound to the validated
Radeon Pro V620 target. Acceptance artifacts must additionally preserve
reference/candidate kernel-name traces; byte equality alone cannot prove the
candidate dispatched.

## Natural-text proxy validation

`scripts/dsv4-rocm/corpus/technical-proxy.txt` is a fixed 2,527-token
engineering proxy, not a user-supplied production corpus. Its SHA-256 is
`396c178b3f77e7a920473fedaa54d79d3c98df5a27baebfa9b7de62a793a71df`.
Set `DSV4_OUTPUT_DIR` when running `tests/test-dsv4-validation.sh` to preserve
layer/tensor server logs and response JSON. Run the same command with the MMQ
override unset and set to 16, using different output directories, then compare:

```bash
scripts/dsv4-rocm/compare-validation.py \
  "$artifact_root/base" "$artifact_root/candidate" \
  --json "$artifact_root/comparison.json"
```

The comparison fails if required response fields are missing/malformed, if
timings are internally inconsistent, or if content, generated token IDs,
prompts, or token counts differ. Its timing row is only one natural-text
observation; use bracketed `run-pp.sh` repetitions for performance claims.

For an acceptance-quality base/J16 run, use the safety-guarded attested wrapper
from a clean checkout:

```bash
cd /home/edwin/llama.cpp-rdna2
DSV4_HASH_MODE=full scripts/dsv4-rocm/run-corpus-validation.sh
```

The default arms remain no overrides versus J16. To isolate the second
optimization, hold J16 in both arms and enable only the candidate HC mixer:

```bash
DSV4_BASE_MMQ_J=16 DSV4_CANDIDATE_MMQ_J=16 \
DSV4_BASE_HC_MIXES=0 DSV4_CANDIDATE_HC_MIXES=1 \
scripts/dsv4-rocm/run-corpus-validation.sh
```

To isolate the lightning-indexer optimization, hold J16 and HC fixed:

```bash
DSV4_BASE_MMQ_J=16 DSV4_CANDIDATE_MMQ_J=16 \
DSV4_BASE_HC_MIXES=1 DSV4_CANDIDATE_HC_MIXES=1 \
DSV4_BASE_LID_SUBWAVE=0 DSV4_CANDIDATE_LID_SUBWAVE=4 \
scripts/dsv4-rocm/run-corpus-validation.sh
```

Empty per-arm controls mean unset; HC controls accept `0`/`1` and LID controls
accept `0`/`4`. MMQ controls must be supported multiples of eight. The wrapper
removes inherited MMQ/HC/LID variables before applying each arm, and records
the resolved controls in both self-contained command files and
`effective-settings.sh`.

It holds the shared GPU lock, refuses active ROCm processes, rechecks before
each variant, pins server batch/ubatch to 512/256 by default, and reads the
exact UTF-8 prompt bytes through `DSV4_PROMPT_FILE`,
and requires full hashes of all model shards. The artifact includes clean
source identity, executable and resolved llama/ggml DSO hashes, exact exported
settings and base/candidate command files, ROCm/hardware details, all response
JSON/logs, comparison output, and response hashes. Full model hashing occurs
before inference and can take time, but it does not consume the measured PP
budget.

## Artifacts

Runs are written to collision-resistant directories under
`$HOME/llama-jobs/dsv4-rocm-pp/`:

- `manifest.txt`: host, source, binary/DSO hashes, model shard metadata, ROCm, topology, and environment;
- `source.patch`, `source-status.txt`, `untracked-files.sha256`: dirty-source identity;
- `effective-settings.sh`, `command.sh`, `executed-command.sh`: exact effective controls and argv;
- `result.jsonl`: raw llama-bench records, including every `samples_ts`/`samples_ns` value;
- `summary.tsv` and `summary.json`: completion state, expected/missing shapes, median/range, latency, and raw samples;
- `bench.log`: loader/progress/errors;
- `measurement-start.ns`, `result-completed-at.ns`: trace-alignment timestamps;
- `clock-domain.txt`: run-time realtime-to-monotonic clock mapping and boot ID;
- `measured-region-summary.{txt,json}`: optional filtered rocprof attribution;
- `rocm-smi.log`: one-second utilization/memory/power/clock samples;
- `status.txt`: nanosecond timestamps, truncation, and process exit code.

Quick three-sample summaries report median and range. p05/p95 are deliberately
`NA` until at least 20 samples are available. Use `DSV4_HASH_MODE=full` only when
a one-time full read of every GGUF shard is acceptable.

## Kernel trace

Start with one shape because rocprof tracing is expensive:

```bash
DSV4_LABEL=trace-base-8k \
DSV4_PROMPTS=8192 DSV4_UBATCHES=256 DSV4_REPS=1 \
scripts/dsv4-rocm/profile-pp.sh
```

For long-context attribution where full HIP API and JSON output would consume
several GiB, use compact CSV-only tracing. It retains the kernel dispatch,
memory-copy, and RCCL inputs required by `summarize-trace.py`:

```bash
DSV4_PROFILE=kernel DSV4_PROMPTS=16384 \
DSV4_LABEL=kernel-trace-base-16k scripts/dsv4-rocm/profile-pp.sh
```

The profile wrapper disables llama-bench's warmup. Full `trace` mode records
whole-process HIP runtime, kernel, memory, and RCCL CSV/JSON under `rocprof/`;
compact `kernel` mode omits HIP API events and JSON. Model-load events are still
present in either mode. Do **not** use whole-process summary percentages for
optimization decisions. Filter the trace to the interval beginning at
`measurement-start.ns` and ending at the corresponding line in
`result-completed-at.ns`; traced throughput is not comparable to ordinary A/B
throughput.

Each new run stores start/end calibrations in `clock-domain.txt`, which map the
harness's realtime markers to rocprof's monotonic timestamps and bound clock
drift across the run. Generate measured-region text and JSON summaries with:

```bash
run_dir=/home/edwin/llama-jobs/dsv4-rocm-pp/<trace-run>
scripts/dsv4-rocm/summarize-trace.py "$run_dir" --top 30 \
  --json "$run_dir/measured-region-summary.json" \
  | tee "$run_dir/measured-region-summary.txt"
```

Kernel durations are clipped to the measured interval and summed across all
devices/queues, so their wall-equivalent percentage can exceed 100%. The
summarizer requires a kernel trace/nonempty measured interval and rejects more
than 1 ms of start-to-end clock-offset drift by default. A legacy trace without
`clock-domain.txt` requires its run-time realtime-minus-monotonic offset via
`--clock-offset-ns`; do not reconstruct that offset after a reboot. A single
legacy calibration has unknown boundary uncertainty and is labeled as such.

For multi-GPU imbalance, preserve a per-agent view after generating the main
summary:

```bash
scripts/dsv4-rocm/analyze-trace-agents.py "$run_dir" \
  --json "$run_dir/measured-region-agents.json" \
  | tee "$run_dir/measured-region-agents.txt"
```

This maps rocprof KFD agents to PCI BDFs through the run's `agent_info.csv` and
reports measured-region all-kernel sums plus conservative kernel-name-match
families and copy endpoints. Name matches overlap and are not operation-level
proof. Durations are clipped and then summed over queues; they expose imbalance
but do not by themselves prove PCIe causality or
link bandwidth.

## Production MTP validation

After the main-model PP controls pass, exercise the accepted stack with the
production context/KV/batch settings and the Q4 MTP draft model:

```bash
timeout --signal=TERM --kill-after=30s 2400s env \
GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 \
DSV4_OUTER_TIMEOUT=2400 \
scripts/dsv4-rocm/run-production-mtp-validation.sh
```

The runner holds the shared GPU lock, rejects active ROCm processes again
immediately before each server launch, uses a 262144-token unified F16 KV
context with batch/ubatch 512/256, and compares a fresh main-only server with a
fresh MTP server (`draft-mtp`, Q4_0, n-max 3, ROCm0+ROCm1). It fully hashes the
main shards, draft model, server/DSOs, corpus, and its embedded request client.
The fixed request processes the engineering proxy before generating 128 tokens.
Content and token IDs must match exactly; the MTP arm must report nonzero draft
attempts and drafted tokens. PP and TG timings plus draft acceptance are one
matched main-then-MTP observation, not a stable speed claim.

Default output is a new `$HOME/llama-jobs/dsv4-production-mtp-*` directory. The
outer timeout includes model hashing/loading; the request itself is capped at
five minutes, so model preparation remains outside the measured PP/TG fields.
Do not use production port 8080.

Bounded diagnostics may set `DSV4_DRAFT_N_MAX=1`, `DSV4_N_PREDICT=64`, and
`DSV4_HASH_MODE=quick` after linking back to one full-hash run. Set
`GGML_CUDA_DISABLE_GRAPHS=1` to disable runtime HIP/CUDA graph capture; absence
means enabled. `GGML_HIP_GRAPHS` is a build-time CMake option in this checkout,
not a runtime off switch. Every diagnostic arm records the resolved n-max,
hash mode, and graph-disable presence.

## Summarize existing output

```bash
scripts/dsv4-rocm/summarize.py result.jsonl \
  --expected-prompts 512,2048 --expected-ubatches 256 --expected-reps 3 \
  --json summary.json --tsv summary.tsv
```

`--allow-trailing-partial` is accepted only together with `--truncated`. It may
drop one malformed final JSONL fragment after forced termination; malformed
middle records remain errors and the raw file is never rewritten.

## Validation sequence

1. Run a complete 512/2K quick baseline at ubatch 256.
2. Run a separate capped 8K baseline.
3. Run one 8K trace and filter it to the recorded measured interval.
4. Select a patch only from matched complete shapes and measured-region trace data.
5. Validate exploratory wins on one fixed, recorded production-representative token corpus because llama-bench's synthetic tokens can change MoE routing.
6. Use paired/interleaved base-candidate-base ordering and review clocks/temperature.
7. Add ubatch 128/512 and 32K+ only for final validation.
8. Preserve all raw directories; never cite only the fastest sample.