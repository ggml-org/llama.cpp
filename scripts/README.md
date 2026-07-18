# Repository scripts

Run scripts from the repository root unless an entry says otherwise. Most build
and benchmark helpers expect their dependencies on `PATH`; SYCL helpers also
expect the oneAPI environment to be sourced first.

## Arc A770 and SYCL research

| Path | Purpose |
|---|---|
| `audit-sycl-op-candidates.py` | Maps exported graph op IDs to `ggml_op` names and reports whether selected SYCL port candidates are present, absent, or already implemented. |
| `bench-a770-fork-unique.py` | Benchmarks fork-only A770 modes. Its `product` campaign enforces sole tenancy, alternates paired arms, discards warm-up sample 0, records raw JSON, confidence intervals, provenance, dmesg deltas, and effective KV bandwidth. `--candidate-bin-dir` compares separate builds; omit it for environment-only comparisons. |
| `test_bench_a770_fork_unique.py` | Unit tests for the A770 product campaign, including pairing, invalid samples, dmesg gating, environment routing, and separate candidate binaries. |
| `sweep-a770-mmvq-geometry.py` | Reproducible `MMV_Y={1,2,4}` x `MMVQ_NUM_SUBGROUPS={4,8,16,32}` orchestrator. It can build isolated JIT binaries, run correctness plus dmesg gates, and benchmark multiple models sequentially against the `1x16` baseline. |
| `test_sweep_a770_mmvq_geometry.py` | Unit tests for geometry naming, dmesg overlap calculation, and fail-closed render-node tenancy checks. |
| `validate-dense-turbo4-capacity.sh` | Runs the dense low-GQA turbo4 capacity/quality validation workflow and preserves the paired evidence used by the research queue. |
| `turbo-quality-gate.sh` | Pre-push TurboQuant correctness, perplexity, and context-scaling gate. Strict mode rejects skips, XFAILs, and XPASSes. |
| `bench-smem-m5.sh` | Apple M5 Max experiment comparing shared-memory pre-dequantization with a baseline at multiple context depths. It is not an A770 runner. |

### MMVQ geometry sweep

Source oneAPI, then run all phases with content-addressed build and result
locations:

```bash
set +u
source /opt/intel/oneapi/setvars.sh
set -u

python scripts/sweep-a770-mmvq-geometry.py \
  --phase all \
  --source "$PWD" \
  --build-root "$HOME" \
  --tag "$(git rev-parse --short=9 HEAD)" \
  --model mistral=/path/to/mistral.gguf \
  --model qwen=/path/to/qwen.gguf \
  --out-root /tmp/a770-mmvq-geometry-"$(git rev-parse --short=9 HEAD)"
```

The build phase is the only concurrent phase. Correctness and benchmark legs
are sequential and fail closed unless `fuser -v /dev/dri/renderD128` proves the
render node is idle. The script never kills a holder. Benchmark output is one
`product.json`/`product.md` pair per geometry and model, plus a top-level
`manifest.json`. Use `--phase build`, `correctness`, or `benchmark` to run an
individual phase against existing build directories.

## Benchmark and performance analysis

| Path | Purpose |
|---|---|
| `bench-models.sh` | Runs a configured set of models through the standard benchmark tooling. |
| `compare-llama-bench.py` | Reads llama-bench or test-backend-ops result data, selects baseline/compare commits, aggregates repetitions, and emits comparison tables. |
| `tool_bench.py` | Python benchmark analysis and plotting tool with inline dependency metadata. |
| `tool_bench.sh` | Shell entry point/wrapper for the benchmark tool workflow. |
| `server-bench.py` | Measures throughput of a running OpenAI-compatible `llama-server` and writes console summaries and plots. |
| `perf/bench_spec.py` | A770 SYCL speculative-decoding and KV-type HTTP benchmark harness. Launches `llama-server`, executes fixed prompts, and records request/server evidence. |
| `perf/prompts.jsonl` | Normal prompt fixture for speculative-decoding comparisons. |
| `perf/prompts_adversarial.jsonl` | Adversarial prompt fixture for proving the ngram-mod hard-off mechanism. |
| `perf/FINDINGS.md` | Preserved interpretation and reproduction commands for the speculative-decoding experiments. |
| `perf/results/` | Generated/specimen output directory for speculative-decoding campaigns. |

## Server behavior tests

These scripts target an already running server and do not configure the model
for you.

| Path | Purpose |
|---|---|
| `server-test-model.py` | Basic llama-server functionality smoke test. |
| `server-test-function-call.py` | Multi-turn chat-completions tool-calling test suite with mocked tool responses and semantic validators. |
| `server-test-parallel-tc.py` | Parallel tool-call test suite; only use with a model/server configuration that supports parallel calls. |
| `server-test-structured.py` | Structured-output and JSON-schema test suite for chat completions. |
| `fetch_server_test_models.py` | Downloads the small model fixtures required by server tests. |
| `serve-static.js` | Minimal static-file server used by local UI/server test workflows. |

## Data, model, and template utilities

| Path | Purpose |
|---|---|
| `hf.sh` | Downloads a Hugging Face model from a URL or `--repo`/`--file` pair and prints the local path for command substitution. |
| `verify-checksum-models.py` | Checks model files listed in repository `SHA256SUMS` and prints validity/missing status. |
| `get_chat_template.py` | Extracts or prints the chat template stored in a model. |
| `get-hellaswag.sh` | Downloads the HellaSwag validation fixture. |
| `get-wikitext-2.sh` | Downloads and extracts WikiText-2 raw test data. |
| `get-winogrande.sh` | Downloads the Winogrande evaluation CSV used by llama.cpp tests. |
| `get-pg.sh` | Fetches and formats Paul Graham essay text for prompt/corpus experiments. |
| `gen-unicode-data.py` | Downloads Unicode data and generates the C++ Unicode lookup tables. |
| `jinja/jinja-tester.py` | GUI/CLI Jinja chat-template editor and renderer. |
| `jinja/requirements.txt` | Python dependencies for the Jinja tester. |
| `compare-logprobs.py` | Captures and compares token log-probability logs from llama.cpp and another implementation, then writes a report. |

## Build, test, and maintenance

| Path | Purpose |
|---|---|
| `build-info.sh` | Emits build/version metadata consumed by the build system. |
| `get-flags.mk` | Make helper for obtaining the compiler/linker flags used by selected targets. |
| `check-requirements.sh` | Creates fresh virtual environments for top-level conversion scripts, installs each requirements file, and checks imports. This is intentionally I/O-heavy. |
| `debug-test.sh` | Configures a debug build, selects a CTest test by regex/index, and runs it directly or under GDB. |
| `git-bisect.sh` | Convenience driver for the repository's automated git-bisect workflow. |
| `git-bisect-run.sh` | Per-revision command invoked by the bisect driver. |
| `install-oneapi.bat` | Windows helper for installing/configuring Intel oneAPI dependencies. |
| `create_ops_docs.py` | Converts `docs/ops/*.csv` backend support data into the generated operations documentation. |
| `gen-authors.sh` | Regenerates/normalizes the `AUTHORS` file from Git history. Review its platform-specific `sed` invocation before use. |
| `sync_vendor.py` | Synchronizes vendored source content and metadata. |
| `sync-ggml.sh` | Simple ggml-to-llama.cpp synchronization entry point. |
| `sync-ggml-am.sh` | Generates/applies ggml patch series with configurable context and skipped commits, then updates the sync marker. |
| `sync-ggml.last` | Data file containing the last synchronized ggml commit; it is consumed by sync scripts, not executed. |
| `ui-assets.cmake` | CMake helper that selects prebuilt, npm-built, or Hugging Face-hosted UI assets and generates `ui.cpp`/`ui.h`. |

## Worktree helpers

| Path | Purpose |
|---|---|
| `wc2wt.sh` | Creates a sibling worktree and branch from the current checkout, optionally preparing an agent/build command. |
| `pr2wt.sh` | Creates a sibling worktree from a GitHub PR's fork/branch, optionally preparing an agent/build command. |

## Generated files

`__pycache__/` contains local Python bytecode and is not a maintained script
surface. Benchmark result directories should remain outside the source tree
unless a specific experiment deliberately preserves fixtures under
`perf/results/`.
