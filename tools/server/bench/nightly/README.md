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

- NVIDIA driver, Docker with the NVIDIA Container Toolkit, and a runner that can use Docker (it needs that for container jobs anyway). The binaries come from the release tarballs, no CUDA toolkit or compiler is needed.
- The bench venv needs Python 3.11-3.13 with its headers: AIPerf 0.13.0 rejects 3.14 and builds `crick` from source on arm64. Dependencies are pinned in `requirements.txt`.
- The nightly job runs in `nvidia/cuda:13.5.0-devel-ubuntu24.04` and installs what it needs there, with the image's Python 3.12 and `python3-dev`. The weekly's prefetch job runs in the same image and builds the bench venv in `NIGHTLY_HOME` with a pinned standalone uv (`uv-bin/`) and uv's own Python 3.12. Then one job per vLLM column runs in that column's image (by digest, with the recipe's docker options), uses that venv and starts `vllm serve` there.
- About 160 GiB free in the runner's work folder for the nightly, plus about 135 GiB of checkpoints and two images for the weekly. Both workflows use `NIGHTLY_HOME` = `<work folder>/llama-nightly`, next to the checkouts: the runner mounts the work folder into every job container and keeps it between jobs. The jobs run as root in their containers and hand what they wrote back to the work folder's owner at the end. Runs by hand default to `/opt/llama-nightly` (`--home` or `NIGHTLY_HOME` overrides it). Layout: `builds/<tag>/`, `models/`, `data/`, `results/<date>_<tag>/`, `container/` (the Hugging Face cache of both workflows, and the weekly's `weekly-venv/`, `uv-python/` and `uv-cache/`), and `uv-bin/` and `vllm-cache/` for the weekly. Runs by hand use the default Hugging Face cache for tokenizers and SPEED-Bench.
- A self-hosted runner with the labels `self-hosted, Linux, gb10, perf`. Nothing else may run on the box during a night: a concurrent job corrupts the numbers.

## Run by hand

```bash
curl -LsSf https://astral.sh/uv/0.13.7/install.sh | env UV_UNMANAGED_INSTALL=/opt/llama-nightly/uv-bin sh
/opt/llama-nightly/uv-bin/uv venv --python 3.12 --managed-python /opt/llama-nightly/venv
VIRTUAL_ENV=/opt/llama-nightly/venv /opt/llama-nightly/uv-bin/uv pip install -r tools/server/bench/nightly/requirements.txt
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py prefetch --home /opt/llama-nightly
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py run --home /opt/llama-nightly --suites engine --models M3
```

`--dry-run` prints every server and client command without starting anything. `--suites`, `--models` and `--configs` (shell patterns on config names, e.g. `--configs 'mooncake_T02*'`) narrow a run; a re-run on the same day and build replaces only the runs it repeats in that night's `night.json`. `--resume <night_id>` (e.g. `2026-11-11_vllm`) finishes an interrupted night into its own report: its ok runs are kept in place, the rest run with that night's build or images (by digest), and the summary lists the parts. `--tag bNNNN` (prefetch) pins a release.

## Workflow

- `workflow_dispatch` with optional `tag`, `suites`, `models`, `configs` and `resume` inputs.
- Scheduled at 01:00 UTC. On a fork without the runner, disable the workflow (`gh workflow disable`), or its scheduled jobs wait for a runner that never comes.
- The run step writes a summary table to the job summary, with a compare link to the previous night's commit, and uploads `results/<night>/` as an artifact for 30 days. It fails if any run failed.

## Weekly vLLM comparison

vLLM only: each vLLM row is shown next to the llama.cpp row of its paired nightly model, from that day's nightly `night.json`, with the same AIPerf tokenizer and seed (so the same synthetic prompts).

| Column | Pair | Checkpoint | Image |
| --- | --- | --- | --- |
| `M3-vllm-nvfp4` | M3 | `nvidia/Qwen3.8-27B-NVFP4` | NVIDIA recipe `qwen/qwen3.8-27b/dgx-spark/linux/vllm/128gb/nvfp4` (NGC vLLM) |
| `M3-vllm-int4` | M3 | `RedHatAI/Qwen3.8-27B-INT4` (W4A16) | same image and flags, minus the NVFP4 `--kernel-config` |
| `M2-vllm-fp4local` | M2 | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | NVIDIA recipe `qwen/qwen3.8-flash-next/dgx-spark/linux/vllm/fp4-local` (community b12x image) |

- **Images:** each week prefetch reads the image of each column from [NVIDIA's DGX Spark recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html), so the weekly follows NVIDIA's updates. It falls back to the `image` in `weekly.yaml` when the catalog can't be read. Each run records the digest, the vLLM version and the kernel/backend lines from the server log.
- **Flags:** copied from the recipes, minus speculative decoding and the reasoning/tool parsers (the llama.cpp runs use neither), with `--max-num-seqs` / `--max-model-len` per config and `--max-num-batched-tokens 2048` (llama.cpp runs `-ub 2048`). Serving runs with prefix caching off on both engines (`serving.vllm_args` / `serving.server_args` in `suite.yaml`). Prefetch warns when a recipe was re-validated after the flags were copied (`recipe_validated_at`).
- **Known difference:** vLLM runs with an FP8 KV cache (as the recipes do), llama.cpp with f16.
- **Inputs and schedule:** `workflow_dispatch` takes optional `columns`, `suites`, `configs`, `vllm_image` (`COLUMN=IMAGE` overrides) and `resume` inputs. Scheduled Saturdays at 02:00 UTC, it queues behind that night's nightly.
- **Jobs:** a prefetch job, then one job per column, one after the other, each in its column's image. They all add to one night (`results/<date>_vllm/`), and each uploads it, so the last upload has every column.

By hand, `run` goes inside the column's image, like the workflow's column jobs. The venv's Python must be visible there, so create the venv with `UV_PYTHON_INSTALL_DIR=/opt/llama-nightly/uv-python`:

```bash
export HF_HOME=/opt/llama-nightly/container/hf-cache
/opt/llama-nightly/venv/bin/python tools/server/bench/nightly/nightly.py prefetch --config tools/server/bench/nightly/weekly.yaml --home /opt/llama-nightly
docker run --rm --gpus all --ipc host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 -v /opt/llama-nightly:/opt/llama-nightly -v "$PWD:/src" -w /src -e HF_HOME --entrypoint /opt/llama-nightly/venv/bin/python nvcr.io/nvidia/vllm:26.10.pre1-py3 tools/server/bench/nightly/nightly.py run --config tools/server/bench/nightly/weekly.yaml --home /opt/llama-nightly --models M3-vllm-nvfp4 --configs 'depth0_*_c1'
```

Not covered: the greedy spec on/off check, the decision accuracy set, vLLM with speculative decoding, SGLang, llama.cpp NVFP4 GGUFs, the rolling-baseline compare and the dashboard.
