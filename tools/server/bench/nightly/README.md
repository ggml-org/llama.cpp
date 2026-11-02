# Nightly perf benchmark on DGX Spark

Runs the newest llama.cpp pre-release build (`llama-<tag>-bin-ubuntu-cuda-13.4-arm64`) on a DGX Spark and records one JSON manifest per night. Driven by [suite.yaml](suite.yaml) and the [bench-nightly-spark](../../../../.github/workflows/bench-nightly-spark.yml) workflow.

This first cut covers two models:

| ID | Model | Suites |
| --- | --- | --- |
| M3 | Qwen3.8-27B UD-Q4_K_M (MTP built in, DFlash2 draft) | engine, speedbench, serving, agentic, quality |
| M5 | Laya Q8_0 (decision model, `/v1/systemone`) | decision |

| Suite | Tool | What it records |
| --- | --- | --- |
| engine | `llama-bench` | pp2048 / tg128 tokens/s at depth 0, 8k, 32k |
| speedbench | [speed_bench.py](../speed-bench/speed_bench.py) | per-category decode speed, latency, draft acceptance for spec none / MTP / DFlash2 at concurrency 1 and 4 |
| serving | AIPerf | TTFT and ITL p50/p99, throughput, goodput over input length, depth and concurrency |
| agentic | AIPerf `mooncake_trace` | the same latency metrics plus prefix-cache reuse, replaying [traces/](traces) at their original timestamps |
| quality | `llama-perplexity` | Wikitext-2 perplexity (10 x 512 chunks) |
| decision | `/v1/systemone` | client latency p50/p99 and decisions/s over state length 64-2048 and concurrency 1-16 |

The agentic traces are Mooncake JSONL (`timestamp`, `input_length`, `output_length`, `hash_ids`, 512 tokens per block) with no prompt text. Output lengths are capped at `agentic.max_output_tokens` (some turns ask for 35k tokens).

## Box setup

- NVIDIA driver only. The binaries come from the release tarballs, no CUDA toolkit or compiler is needed.
- `python3` with `venv`, plus `tar`, `unzip` and `curl` or `wget`. The bench venv uses a uv-managed Python 3.12, because AIPerf builds `crick` from source on arm64 and the system Python often has no headers (`python3-dev`).
- A writable `NIGHTLY_HOME` (default `/opt/llama-nightly`) with about 40 GiB free. Layout: `builds/<tag>/`, `models/`, `data/`, `results/<date>_<tag>/`, `venv/`. Tokenizers and SPEED-Bench go to the default Hugging Face cache.
- A self-hosted runner with the labels `self-hosted, Linux, ARM64, spark-nightly-bench`, used by nothing else (a concurrent job corrupts the numbers).

## Run by hand

```bash
python3 -m venv /opt/llama-nightly/uv-venv && /opt/llama-nightly/uv-venv/bin/pip install uv
/opt/llama-nightly/uv-venv/bin/uv venv --python 3.12 --managed-python /opt/llama-nightly/venv
VIRTUAL_ENV=/opt/llama-nightly/venv /opt/llama-nightly/uv-venv/bin/uv pip install -r tools/server/bench/nightly/requirements.txt
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py prefetch --home /opt/llama-nightly
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py run --home /opt/llama-nightly --suites engine --models M3
```

`--dry-run` prints every server and client command without starting anything. `--tag bNNNN` (prefetch) pins a release.

## Workflow

- `workflow_dispatch` with optional `tag`, `suites` and `models` inputs.
- The 01:00 UTC schedule only runs when the repository variable `SPARK_BENCH_NIGHTLY` is `true`. `SPARK_BENCH_HOME` overrides `NIGHTLY_HOME`.
- The run step writes a summary table to the job summary, with a compare link to the previous night's commit, and uploads `results/<night>/` as an artifact for 30 days. It fails if any run failed.

Not in this cut: M1, M2, M4, backend-sampling runs, the greedy spec on/off check, the decision accuracy set, the rolling-baseline compare and the dashboard.
