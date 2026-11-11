# Nightly llama.cpp benchmark on DGX Spark: what runs

One unattended run per night on a single DGX Spark (GB10, 128 GB unified LPDDR5X, ~273 GB/s, DGX OS on aarch64). Nothing is built on the box.

## Build under test

- The newest [llama.cpp pre-release](https://github.com/ggml-org/llama.cpp/releases) that has both `llama-<tag>-bin-ubuntu-cuda-13.4-arm64.tar.gz` and `cudart-llama-<tag>-bin-ubuntu-cuda-13.4-arm64.tar.gz`. Both tarballs are unpacked into one folder; the last 3 builds are kept.
- A tag can be pinned for a manual run. The summary links the commit range since the previous night.

## Models

GGUFs are pinned by Hugging Face commit and downloaded once.

| ID | Model | Quant | Size | Workloads |
|---|---|---|---|---|
| M1 | [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | UD-Q4_K_M | 21 GiB | engine |
| M2 | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) + [MTP / DFlash2 heads](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF) | UD-IQ4_XS, heads Q8_0 | 92 GiB | all LLM workloads |
| M3 | [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (dense) + [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) | UD-Q4_K_M | 16 GiB | all LLM workloads |
| M4 | [Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | UD-Q4_K_M | 24 GiB | engine |
| M5 | [Laya](https://huggingface.co/ggml-org/Laya-GGUF) (decision model) | Q8_0 | 0.4 GiB | decision |

The nightly runs all five. M2 is split into three shards; its MTP and DFlash2 heads are separate files.

## Common setup

- `llama-server` on `127.0.0.1`, started fresh per configuration: `-ngl 999 -fa on --jinja --metrics`, `-np` = client concurrency, `-c` = per-slot context x slots.
- 60 s idle cooldown before every `llama-server`, `llama-bench` and `llama-perplexity` start.
- Telemetry sampled every second: power, SM clock and GPU temperature from `nvidia-smi`; memory from `/proc/meminfo` (unified memory reports N/A in `nvidia-smi`).
- Clients: in-tree `tools/server/bench/speed-bench/speed_bench.py`, [AIPerf](https://github.com/ai-dynamo/aiperf) 0.13.0, and a small Python client for `/v1/systemone`.

## Workloads

| Workload | Tool | Configurations | Metrics |
|---|---|---|---|
| Engine | `llama-bench -p 2048 -n 128 -d 0,8192,32768 -r 5 -fa 1 -ub 2048` | one per model | pp2048 and tg128 tok/s at each depth |
| Spec decode | `speed_bench.py` on [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) qualitative, 4 prompts x 11 categories, 256 output tokens, temp 1.0 / top_p 0.95 / top_k 20 / seed 42 | none, MTP (`--spec-type draft-mtp --spec-draft-n-max 3`), DFlash2 (`--spec-type draft-dflash --spec-draft-n-max 7`), each at concurrency 1 and 4; drafts sample on the backend | per category and overall: decode tok/s, prompt tok/s, latency, draft acceptance |
| Serving | AIPerf `profile`, chat endpoint, streaming, fixed-length synthetic prompts, 256 output tokens (`ignore_eos`) | input 2k / 8k at depth 0, input 2k after a 32k shared prefix; concurrency 1 / 4 / 8; 16 / 16 / 32 requests; at least one warmup request per slot | TTFT and ITL p50 / p99, output tok/s per user and total, req/s |
| Agentic | AIPerf `mooncake_trace` | 2 recorded coding-agent sessions (51 and 39 turns, 20k-174k input tokens, no prompt text), replayed one turn after the other on one slot; output capped at 256 tokens | TTFT and ITL p50 / p99, output tok/s, prefix-cache reuse from `/metrics` |
| Quality | `llama-perplexity -c 512 --chunks 10` on Wikitext-2 raw test | one per model | perplexity |
| Decision | `/v1/systemone`, 3 questions per request (choice, score, yes/no) | state 64 / 512 / 2048 tokens x concurrency 1 / 4 / 16; 8 warmup + 64 requests; server `-np 16 -ub 8192 -b 8192` | client latency p50 / p99, decisions/s |

Traces use the [Mooncake](https://github.com/kvcache-ai/Mooncake) JSONL format (`timestamp`, `input_length`, `output_length`, `hash_ids`, 512 tokens per block). AIPerf synthesizes prompt text whose shared blocks repeat across turns, so prefix caching behaves as in the original session.

## Weekly vLLM comparison

Once a week (Saturday, after the nightly) the serving and agentic workloads run against vLLM in docker, with the same AIPerf commands, tokenizer and seed. The llama.cpp side is the paired model's rows from that day's nightly; the weekly runs vLLM only.

| Column | Pair | Checkpoint | Quant | Size | Image |
|---|---|---|---|---|---|
| `M3-vllm-nvfp4` | M3 | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | NVFP4 (+ FP8 layers) | 20 GiB | NGC `nvcr.io/nvidia/vllm`, per NVIDIA's recipe |
| `M3-vllm-int4` | M3 | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | INT4 W4A16, groups of 128 | 18 GiB | same as above |
| `M2-vllm-fp4local` | M2 | [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 experts, MXFP8 attention | 95 GiB | community `eugr/spark-vllm-b12x`, per NVIDIA's recipe |

- Images and flags come from [NVIDIA's DGX Spark recipes](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html). The image follows the recipe each week; each run records the digest, vLLM version and kernel/backend lines.
- Deviations from the recipes, for parity with llama.cpp: no speculative decoding, no reasoning/tool parsers, `--max-num-seqs` and `--max-model-len` set per configuration like llama.cpp's slots and per-slot context.
- Kept from the recipes: FP8 KV cache (llama.cpp uses f16); the Flash-Next n-gram embedding table on disk.
- The NGC image runs 27B NVFP4 with the SM12x NVFP4 backend (vllm#54614), which upstream vLLM does not have yet. No Flash-Next INT4 checkpoint fits in 128 GB (the n-gram table stays BF16), and upstream vLLM cannot read FP4-Local's compressed scales yet.
- Prefix reuse comes from vLLM's `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total`.

## Failure handling

A configuration that fails is recorded with its error and the night continues. A server that cannot start fails only its own configurations. A spec-decode run that drafts no tokens counts as failed.

## Outputs

- Per night: `night.json` (build, commit, models, every metric, telemetry, exact commands), `summary.md`, and every client and server log.
- On GitHub Actions: the summary table in the job summary, the night folder as a 30-day artifact. Nothing is committed to the repository.

## Cost (build b12072)

| | Runs | Wall time | Disk |
|---|---|---|---|
| M3 + M5 (measured) | 28 | 3.1 h | 21 GiB |
| Nightly, all five models | 49 | ~5.5 h (estimated) | 157 GiB (measured) |
| Weekly vLLM, 3 columns | 33 | ~5.5-6.5 h (estimated) | ~135 GiB + 2 images |

The nightly time scales M3's measured runs by M2's measured prefill and decode speeds (M2: 1232 / 30.5 tok/s vs M3: 1039 / 13.7 tok/s at depth 0), plus M2's 77 s load per server start; M1, M2 and M4 engine runs took 126 / 274 / 116 s. The nightly workflow allows 480 min, the weekly 600 min.
