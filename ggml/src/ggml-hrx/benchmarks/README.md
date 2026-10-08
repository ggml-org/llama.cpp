# HRX Loom Benchmarks

This directory contains model-scoped Loom benchmarks for HRX kernels. Benchmark sources stay separate from the production kernel corpus: they declare the kernels they need with `kernel.decl`, and the runner links those declarations against the production `.loom` files before benchmarking.

Use these benchmarks to measure production kernels with `iree-benchmark-loom` while keeping model-shaped benchmark cases out of the embedded kernel catalog. Each model has one `.loom` file under `loom/`, and each workload shape has a sidecar manifest next to it, for example `llama32_3b_f16.pp512.json`.

Generated model benchmarks are deduplicated by kernel shape. The sidecar manifest keeps a compact `dispatches` list with the generated benchmark name, kernel, compile/runtime parameters, source files, and `count`, so benchmark results can be weighted back to the full model without materializing one check case for every duplicate dispatch.

Generated model benchmarks are performance fixtures, not numerical validators. They launch the model-shaped kernels with synthetic buffers and intentionally omit `check.expect.*` assertions; backend model smoke tests and the kernel corpus checks remain responsible for numerical validation.

## Generating Llama Benchmarks

First dump the HRX command programs for each model run:

```sh
GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-llama32-pp256-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/llama-3.2/Llama-3.2-3B-Instruct-F16.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 64 \
  --ubatch-size 64 \
  --repetitions 1 \
  --no-warmup \
  --output jsonl \
  --n-prompt 256 \
  --n-gen 0 \
  --n-depth 0

GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-llama32-pp512-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/llama-3.2/Llama-3.2-3B-Instruct-F16.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 64 \
  --ubatch-size 64 \
  --repetitions 1 \
  --no-warmup \
  --output jsonl \
  --n-prompt 512 \
  --n-gen 0 \
  --n-depth 0

GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-llama32-tg8-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/llama-3.2/Llama-3.2-3B-Instruct-F16.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 64 \
  --ubatch-size 64 \
  --repetitions 1 \
  --no-warmup \
  --output jsonl \
  --n-prompt 0 \
  --n-gen 8 \
  --n-depth 0
```

For the Llama 3.2 1B Q4_K_XL decode benchmark, dump the tg32 command program with:

```sh
GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-llama32-1b-tg32-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/lemonade/llamacpp-gguf-models/unsloth_Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-UD-Q4_K_XL.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 512 \
  --ubatch-size 512 \
  --repetitions 1 \
  --output jsonl \
  --n-prompt 0 \
  --n-gen 32 \
  --n-depth 0
```

Then generate the shared model benchmark file and per-scenario sidecars:

```sh
ggml/src/ggml-hrx/tools/benchmarks/generate-model-benchmarks.py \
  --model llama32_3b_f16 \
  --scenario-dump pp256=/tmp/hrx-llama32-pp256-dumps \
  --scenario-dump pp512=/tmp/hrx-llama32-pp512-dumps \
  --scenario-dump tg8=/tmp/hrx-llama32-tg8-dumps

ggml/src/ggml-hrx/tools/benchmarks/generate-model-benchmarks.py \
  --model llama32_1b_q4_k_xl \
  --scenario tg32 \
  --dump-dir /tmp/hrx-llama32-1b-tg32-dumps
```

Review the generated invoked-kernel set with:

```sh
jq '.kernel_counts' ggml/src/ggml-hrx/benchmarks/loom/llama32_3b_f16.<scenario>.json

ggml/src/ggml-hrx/tools/benchmarks/run-model-benchmarks.sh \
  --model llama32_3b_f16 \
  --scenario <scenario> \
  --build-dir <build-dir> \
  --list
```

Run the generated benchmarks with:

```sh
ggml/src/ggml-hrx/tools/benchmarks/run-model-benchmarks.sh \
  --model llama32_3b_f16 \
  --scenario <scenario> \
  --build-dir <build-dir> \
  --output-dir /home/rsuderman/codex/project-workspaces/llama.cpp/gates/hrx-loom-benchmarks/llama32-<scenario>
```

Use `--dry-run-only` to stop after planning, `--list` to inspect the generated benchmark set, or `--benchmark @name` to run a single invocation. Set `LOOM_LINK` or `IREE_BENCHMARK_LOOM` to override tool discovery. Set `DEVICE` to override the default `amdgpu` HAL device.

The runner applies the same workload-argument specialization used by the HRX runtime after `loom-link`, and benchmarks the resulting `linked.runtime-specialized.loom` source by default. Use `--no-runtime-specialization` to benchmark the raw linked source.

`loom-link` also receives each captured compile parameter as `--config=<name>=<value>`. Some model kernels select templates from compile-time config values, so passing those configs during link keeps standalone benchmarks aligned with the HRX runtime path.

Use `--continue-on-failure` to record failed or timed-out standalone cases and keep running the rest of the model benchmark set. Use `--benchmark-timeout-sec` to bound each link, dry-run, and benchmark command. Each benchmark directory keeps `loom-link`, dry-run, and benchmark command lines plus stdout/stderr so failures can be reproduced directly.

Use `--profile-final-batch=false`, `--input-ring-count`, or repeated `--benchmark-extra-arg` flags when isolating `iree-benchmark-loom` behavior from production runtime behavior. After a run, summarize the weighted model time with:

```sh
ggml/src/ggml-hrx/tools/benchmarks/summarize-model-benchmarks.py \
  /home/rsuderman/codex/project-workspaces/llama.cpp/gates/hrx-loom-benchmarks/llama32-pp512/results.jsonl
```

The summary uses `operation_timing_ns.p50 * count` by default and writes `summary.json` plus `summary.md` next to the runner results.

To map likely fusion opportunities from a command-program dump and the weighted benchmark summary:

```sh
ggml/src/ggml-hrx/tools/benchmarks/analyze-model-fusion-adjacency.py \
  --dump-dir /tmp/hrx-llama32-tg8-dumps \
  --scenario-manifest ggml/src/ggml-hrx/benchmarks/loom/llama32_3b_f16.tg8.json \
  --summary-json /home/rsuderman/codex/project-workspaces/llama.cpp/gates/hrx-loom-benchmarks/llama32-tg8/summary.json \
  --output-json /home/rsuderman/codex/project-workspaces/llama.cpp/gates/hrx-loom-benchmarks/llama32-tg8/fusion-adjacency.json \
  --output-md /home/rsuderman/codex/project-workspaces/llama.cpp/gates/hrx-loom-benchmarks/llama32-tg8/fusion-adjacency.md
```

The adjacency report uses transient value producer-consumer edges for true data dependencies and sequential cache-update windows for side-effect patterns such as RoPE, SET_ROWS, and decode flash attention.

## Generating Qwen 30B Benchmarks

Dump the HRX command programs with the Qwen 30B shard and the same pp256, pp512, and tg8 shapes used for model benchmarking:

```sh
GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-qwen30b-pp256-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/qwen-30b/qwen3-30b-a3b-q4_k_m-00001-of-00020.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 64 \
  --ubatch-size 64 \
  --repetitions 1 \
  --no-warmup \
  --output jsonl \
  --n-prompt 256 \
  --n-gen 0 \
  --n-depth 0

GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-qwen30b-pp512-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/qwen-30b/qwen3-30b-a3b-q4_k_m-00001-of-00020.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 64 \
  --ubatch-size 64 \
  --repetitions 1 \
  --no-warmup \
  --output jsonl \
  --n-prompt 512 \
  --n-gen 0 \
  --n-depth 0

GGML_HRX_DUMP_COMMAND_PROGRAM_DIR=/tmp/hrx-qwen30b-tg8-dumps \
<build-dir>/bin/llama-bench \
  --model /home/rsuderman/Downloads/gguf/qwen-30b/qwen3-30b-a3b-q4_k_m-00001-of-00020.gguf \
  --device HRX0 \
  --n-gpu-layers -1 \
  --batch-size 64 \
  --ubatch-size 64 \
  --repetitions 1 \
  --no-warmup \
  --output jsonl \
  --n-prompt 0 \
  --n-gen 8 \
  --n-depth 0
```

Then regenerate the shared Qwen benchmark source and sidecars:

```sh
ggml/src/ggml-hrx/tools/benchmarks/generate-model-benchmarks.py \
  --model qwen3_30b_a3b_q4_k_m \
  --scenario-dump pp256=/tmp/hrx-qwen30b-pp256-dumps \
  --scenario-dump pp512=/tmp/hrx-qwen30b-pp512-dumps \
  --scenario-dump tg8=/tmp/hrx-qwen30b-tg8-dumps
```
