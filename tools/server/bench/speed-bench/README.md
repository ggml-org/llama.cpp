# SPEED-Bench server benchmark

A lightweight [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) client for benchmarking an already-running `llama-server` through its OpenAI-compatible API. It is primarily meant to evaluate speculative decoding (draft model, n-gram, MTP, EAGLE3, ...) by reporting per-category throughput, latency, and draft acceptance.

The dataset handling follows the [aiperf SPEED-Bench tutorial](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/speed-bench.md), which also documents the dataset layout in more detail.

## Install

```bash
pip install -r tools/server/bench/speed-bench/requirements.txt
```

## Start a server

The client does not launch the server, so start `llama-server` yourself first. If you care about throughput numbers, set the client `--concurrency` to the server's slot count (`--np`):

```bash
llama-server \
  -m target.gguf \
  -c 8192 \
  --port 8080 \
  -ngl 99 -fa on \
  --np 1 \
  --jinja
```

For speculative decoding, start the server with the appropriate flags for your setup (e.g. a draft model with `-md`, or `--spec-type ngram-mod`). See the [speculative decoding doc](../../../../docs/speculative.md) for details.

## Run

```bash
python tools/server/bench/speed-bench/speed_bench.py \
  --url localhost:8080 \
  --bench qualitative \
  --category coding \
  --osl 1024 \
  --concurrency 1
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--url` | `localhost:8080` | Server URL. The scheme and `/v1` are optional and a trailing slash is fine, so `localhost:8080` and `http://localhost:8080/v1/` both work. |
| `--model` | none | Optional `model` field sent in each request. |
| `--bench` | `qualitative` | SPEED-Bench config, e.g. `qualitative`, `throughput_1k`. See [available dataset variants](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/speed-bench.md#available-dataset-variants). |
| `--category` | `all` | Category filter within the bench; comma-separated list or `all`. For `qualitative` the categories are `coding`, `humanities`, `math`, `multilingual`, `qa`, `rag`, `reasoning`, `roleplay`, `stem`, `summarization`, `writing`. For the `throughput_{ISL}` splits they are `high_entropy`, `low_entropy`, `mixed`. |
| `--osl` | `1024` | Output sequence length, mapped to `max_tokens`. |
| `--extra-inputs` | `{"temperature":0}` | Extra request fields as a JSON object. |
| `--concurrency` | `1` | Concurrent client requests; usually match `--np`. |
| `--limit` | none | Max samples per category (handy for smoke tests). |
| `--timeout` | `600` | Per-request timeout in seconds. |
| `--output` | none | Save raw per-request results and the summary to JSON. |

A few common ones:

- `--category all` runs every category in the bench.
- `--category coding,math` runs just those two.
- `--bench throughput_8k` runs a fixed-input-length throughput split.
- `--limit 8` keeps at most 8 samples per category, which is enough for a quick check.

The `throughput_{ISL}` splits use fixed input lengths (1k - 32k), so they are handy for long-context testing and for comparing different `llama-server` batching settings (e.g. sweeping `-ub` / `--ubatch-size`) on prompts of a known size. Make sure the server `-c` is large enough for the chosen split. When raising `-ub`, also raise `-b` to at least the same value, since the physical ubatch cannot exceed the logical batch.

When `--output` is given, the JSON file holds the run `config`, the `selected_samples` / `completed_samples` / `failed_samples` counts, the per-category `summary` rows, and the per-sample `results`.

## Metrics

The summary prints one row per category plus an `overall` row:

- `samples` - how many samples finished successfully.
- `avg_prompt_t/s` - prefill throughput from llama.cpp (`timings.prompt_per_second`), averaged over the category's samples.
- `avg_pred_t/s` - decode throughput from llama.cpp (`timings.predicted_per_second`), averaged over the category's samples.
- `avg_latency` - average end-to-end request latency seen by the client.
- `accept_rate` - `accepted / draft_n` over the category, or `n/a` if nothing was drafted (`draft_n == 0`).

## Baseline vs speculative decoding

Save a run from each server with `--output`, then diff the two JSON files with `speed_bench_compare.py`.

First, start a plain `llama-server` (no speculative decoding) and save a baseline:

```bash
python tools/server/bench/speed-bench/speed_bench.py \
  --url localhost:8080 \
  --bench qualitative \
  --category all \
  --osl 1024 \
  --concurrency 1 \
  --output baseline.json
```

Then restart `llama-server` with speculative decoding enabled and save another run:

```bash
python tools/server/bench/speed-bench/speed_bench.py \
  --url localhost:8080 \
  --bench qualitative \
  --category all \
  --osl 1024 \
  --concurrency 1 \
  --output spec.json
```

Finally compare the two:

```bash
python tools/server/bench/speed-bench/speed_bench_compare.py \
  --baseline baseline.json \
  --speculative spec.json
```

The comparison table adds:

- `decode_speedup = spec_avg_pred_t/s / base_avg_pred_t/s`
- `latency_speedup = base_avg_latency / spec_avg_latency`

Keep `--bench`, `--category`, `--osl`, and `--limit` the same across both runs, otherwise they won't be using the same prompts.
