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

The profile wrapper disables llama-bench's warmup and records whole-process HIP
runtime, kernel, memory, and RCCL trace/statistics under `rocprof/`. Model-load
events are still present. Do **not** use whole-process summary percentages for
optimization decisions. Filter the trace to the interval beginning at
`measurement-start.ns` and ending at the corresponding line in
`result-completed-at.ns`; traced throughput is not comparable to ordinary A/B
throughput.

Each new run stores `clock-domain.txt`, which maps the harness's realtime
markers to rocprof's monotonic timestamps. Generate measured-region text and
JSON summaries with:

```bash
run_dir=/home/edwin/llama-jobs/dsv4-rocm-pp/<trace-run>
scripts/dsv4-rocm/summarize-trace.py "$run_dir" --top 30 \
  --json "$run_dir/measured-region-summary.json" \
  | tee "$run_dir/measured-region-summary.txt"
```

Kernel durations are clipped to the measured interval and summed across all
devices/queues, so their wall-equivalent percentage can exceed 100%. A legacy
trace without `clock-domain.txt` requires its run-time realtime-minus-monotonic
offset via `--clock-offset-ns`; do not reconstruct that offset after a reboot.

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