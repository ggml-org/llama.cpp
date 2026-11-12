# Nightly perf benchmark on DGX Spark

Runs the newest llama.cpp pre-release build (`llama-<tag>-bin-ubuntu-cuda-13.x-arm64`) on a DGX Spark and records one JSON manifest per night. Driven by [suite.yaml](suite.yaml) and the [bench-nightly-spark](../../../../.github/workflows/bench-nightly-spark.yml) workflow. Once a week, [weekly.yaml](weekly.yaml) and the [bench-weekly-vllm-spark](../../../../.github/workflows/bench-weekly-vllm-spark.yml) workflow run the same serving and agentic suites against vLLM (see [Weekly vLLM comparison](#weekly-vllm-comparison)).

| ID | Model | Suites |
| --- | --- | --- |
| M1 | Qwen3.6-35B-A3B UD-Q4_K_M | engine |
| M2 | Qwen3.8-Flash-Next UD-IQ4_XS (3 shards, separate MTP and DFlash2 heads) | engine, speedbench, serving, agentic, quality |
| M3 | Qwen3.8-27B UD-Q4_K_M (MTP built in, DFlash2 draft) | engine, speedbench, serving, agentic, quality |
| M4 | Nemotron 3.5 Lightning 30B-A3B UD-Q4_K_M | engine |
| M5 | Laya Q8_0 (decision model, `/v1/systemone`) | decision |

| Suite | Tool | What it records |
| --- | --- | --- |
| engine | `llama-bench` | pp2048 / tg128 tokens/s at depth 0, 8k, 32k |
| speedbench | [speed_bench.py](../speed-bench/speed_bench.py) | per-category decode speed, latency, draft acceptance for spec none / MTP / DFlash2 at concurrency 1 and 4 |
| serving | AIPerf | TTFT and ITL p50/p99, throughput over input length, depth and concurrency |
| agentic | AIPerf `mooncake_trace` | the same latency metrics plus prefix-cache reuse, replaying [traces/](traces) one turn after the other |
| quality | `llama-perplexity` | Wikitext-2 perplexity (10 x 512 chunks) |
| decision | `/v1/systemone` | client latency p50/p99 and decisions/s over state length 64-2048 and concurrency 1-16 |

The agentic traces are Mooncake JSONL (`timestamp`, `input_length`, `output_length`, `hash_ids`, 512 tokens per block) with no prompt text. Output lengths are capped at `agentic.max_output_tokens` (some turns ask for 35k tokens). Both traces are one agent session, so the server has one slot and each turn is sent when the previous one finishes: the Spark is slower than the traced system, and replaying at the original timestamps only measured queueing.

## Box setup

- NVIDIA driver only. The binaries come from the release tarballs, no CUDA toolkit or compiler is needed.
- `python3` with `venv`, plus `tar`, `unzip` and `curl` or `wget`. The bench venv uses a uv-managed Python 3.12, because AIPerf builds `crick` from source on arm64 and the system Python often has no headers (`python3-dev`).
- A writable `NIGHTLY_HOME` (default `/opt/llama-nightly`) with about 160 GiB free for the nightly, plus about 135 GiB of checkpoints and two images for the weekly. Layout: `builds/<tag>/`, `models/`, `data/`, `results/<date>_<tag>/`, `venv/`, and `vllm-cache/` for the weekly. Tokenizers and SPEED-Bench go to the default Hugging Face cache.
- For the weekly: Docker with the NVIDIA Container Toolkit, and the runner user in the `docker` group.
- A self-hosted runner with the labels `self-hosted, Linux, ARM64, spark-nightly-bench`, used by nothing else (a concurrent job corrupts the numbers).

## Run by hand

```bash
python3 -m venv /opt/llama-nightly/uv-venv && /opt/llama-nightly/uv-venv/bin/pip install uv
/opt/llama-nightly/uv-venv/bin/uv venv --python 3.12 --managed-python /opt/llama-nightly/venv
VIRTUAL_ENV=/opt/llama-nightly/venv /opt/llama-nightly/uv-venv/bin/uv pip install -r tools/server/bench/nightly/requirements.txt
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py prefetch --home /opt/llama-nightly
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py run --home /opt/llama-nightly --suites engine --models M3
```

`--dry-run` prints every server and client command without starting anything. `--suites`, `--models` and `--configs` (shell patterns on config names, e.g. `--configs 'mooncake_T02*'`) narrow a run; a re-run on the same day and build replaces only the runs it repeats in that night's `night.json`. `--resume <night_id>` (e.g. `2026-11-11_vllm`) finishes an interrupted night into its own report: its ok runs are kept in place, the rest run with that night's build or images (by digest), and the summary lists the parts. `--tag bNNNN` (prefetch) pins a release.

## Workflow

- `workflow_dispatch` with optional `tag`, `suites`, `models`, `configs` and `resume` inputs.
- The 01:00 UTC schedule only runs when the repository variable `SPARK_BENCH_NIGHTLY` is `true`. `SPARK_BENCH_HOME` overrides `NIGHTLY_HOME`.
- The run step writes a summary table to the job summary, with a compare link to the previous night's commit, and uploads `results/<night>/` as an artifact for 30 days. It fails if any run failed.

## Weekly vLLM comparison

vLLM only: each vLLM row is shown next to the llama.cpp row of its paired nightly model, from that day's nightly `night.json`, with the same AIPerf tokenizer and seed (so the same synthetic prompts).

| Column | Pair | Checkpoint | Image |
| --- | --- | --- | --- |
| `M3-vllm-nvfp4` | M3 | `nvidia/Qwen3.8-27B-NVFP4` | NVIDIA recipe `qwen/qwen3.8-27b/dgx-spark/linux/vllm/128gb/nvfp4` (NGC vLLM) |
| `M3-vllm-int4` | M3 | `RedHatAI/Qwen3.8-27B-INT4` (W4A16) | same image and flags, minus the NVFP4 `--kernel-config` |
| `M2-vllm-fp4local` | M2 | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | NVIDIA recipe `qwen/qwen3.8-flash-next/dgx-spark/linux/vllm/fp4-local` (community b12x image) |

- **Images:** each week prefetch reads the image of each column from [NVIDIA's DGX Spark recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html), so the weekly follows NVIDIA's updates. It falls back to the `image` in `weekly.yaml` when the catalog can't be read. Each run records the digest, the vLLM version and the kernel/backend lines from the server log.
- **Flags:** copied from the recipes, minus speculative decoding and the reasoning/tool parsers (the llama.cpp runs use neither), with `--max-num-seqs` / `--max-model-len` per config. Prefetch warns when a recipe was re-validated after the flags were copied (`recipe_validated_at`).
- **Known difference:** vLLM runs with an FP8 KV cache (as the recipes do), llama.cpp with f16.
- **Inputs and schedule:** `workflow_dispatch` takes optional `columns`, `suites`, `configs`, `vllm_image` (`COLUMN=IMAGE` overrides) and `resume` inputs. The Saturday schedule only runs when `SPARK_BENCH_WEEKLY` is `true`, and it queues behind that night's nightly.

```bash
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py prefetch --config tools/server/bench/nightly/weekly.yaml --home /opt/llama-nightly
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py run --config tools/server/bench/nightly/weekly.yaml --home /opt/llama-nightly --models M3-vllm-nvfp4 --configs 'depth0_*_c1'
```

Not covered: backend-sampling runs, the greedy spec on/off check, the decision accuracy set, vLLM with speculative decoding, SGLang, llama.cpp NVFP4 GGUFs, the rolling-baseline compare and the dashboard.
