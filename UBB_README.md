# UBBoost

UBBoost is a private `llama.cpp` experiment for faster `llama-server` prompt
processing on VRAM-limited systems.

The idea is simple: use a temporary prompt-processing runtime with a larger
physical ubatch and different offload settings, then switch back to the normal
runtime for token generation. Without a UBBoost argument, the server should stay
on the official runtime path.

Note that UBB is configured to only run in the first prompt and supposed to also trigger when context is invalidated and reprocessed.

## What It Adds

UBBoost adds an opt-in prompt-processing runtime for `llama-server`.

| Area | Behavior |
|---|---|
| Normal server path | Used when `--promptprocessing-ubatchboost-size` is not set or is `0`. |
| UBBoost path | Used only when `--promptprocessing-ubatchboost-size N` is set. |
| Prompt processing | Runs with the UBBoost runtime settings only on first prompt.|
| Token generation | Returns to the normal/main runtime settings. |
| Reprocessing context| Full prompt reprocessing can switch back into the UBBoost runtime. |
| Checkpoints | Usually continues in main runtime settings with normal settings |

## Current Flags

| Flag | Purpose |
|---|---|
| `--promptprocessing-ubatchboost-size N` | Enables UBBoost and sets the prompt-processing physical ubatch size. `0` disables UBBoost. |
| `--promptprocessing-ubatchboost-gpu-layers N` | GPU layer count for the temporary prompt-processing runtime. Supports `auto` and `all`. |
| `--promptprocessing-ubatchboost-n-cpu-moe N` | MoE CPU offload count for the temporary prompt-processing runtime. |
| `--tokengeneration-no-warmup` | Skips warmup when loading the main runtime after UBBoost prompt processing. |

## Benchmark Summary

## New Benchmark Q2_K_XL UBBoost Tests

All rows use about 7.1 GB of the RTX 2080 8 GB VRAM, with a Ryzen 9 5900X
and 64 GB RAM.

Shared base settings: `n-gpu-layers 99`, `n-cpu-moe 36`, `batch-size 12288`,
`no-warmup true`, `spec-type draft-mtp`, `spec-draft-n-max 5`.


# I have yet to find the best sweetspot.

| Comparison Pair | Configuration | MTP | UBB | UB Size | UBB Size | GPU Layers | CPU MoE | Prefill Speed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Q2 | UBB | :heavy_check_mark: | :heavy_check_mark: | 1024 | 2560 | 17 | 99 | 538.69 T/s | 8K context |
| Q2 | UBB | :heavy_check_mark: | :heavy_check_mark: | 1024 | 2560 | 16 | 99 | 525 T/s | 8K context |
| Q2 | UBB | :heavy_check_mark: | :heavy_check_mark: | 1024 | 2048 | 32 | 99 | 516 T/s | 8K context |
| Q2 | UBB | :heavy_check_mark: | :heavy_check_mark: | 1024 | 4096 | 0 | 99 | ~510 T/s | 8K context |
| Q2 | UBB | :heavy_check_mark: | :heavy_check_mark: | 1024 | 3072 | 1 | 99 | ~510 T/s | 8K context |
| Q2 | Main reference | :heavy_check_mark: | :x: | 1024 | - | 99 | 36 | 389.12 T/s | 8K context |

# (Old Benchmark) First token real world prefill speed
Older "Qwen3.6-35BA3B-MTP-Q8_0"
Newer Unsloth Qwen3.6 35B A3B Q2_K_XL

| Comparison Pair | Configuration | MTP | UBB | Batch Size | UB Size | UBB Size | GPU Layers | CPU MoE | Prefill Speed | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Q8_0 | :x: | :heavy_check_mark: | 11288 | 512 | 6144 | 1 | 40 | 400 T/s | Reload included; worst case with lower context |
| 1 | Q8_0 | :x: | :x: | 11264 | 2816 | - | 40 | 40 | 375 T/s | - |

| Comparison Pair | Configuration | MTP | UBB | Batch Size | UB Size | UBB Size | GPU Layers | CPU MoE | Prefill Speed | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | Q8_0 | :heavy_check_mark: | :heavy_check_mark: | 11264 | 512 | 2816 | 1 | 40 | 230 T/s |-|
| 2 | Q8_0 | :heavy_check_mark: | :x: | 12288 | 512 | - | 40 | 40 | 120 T/s |-|
| 2 | Q8_0 | :heavy_check_mark: | :x: | - | 768 | - | 40 | 40 | ~80 T/s | VRAM spill |



## Raw Batch Speed (Old Test)

This table keeps the raw prompt-processing batch speed separate from the
practical MTP/server comparison above.

| Case | Model / setting | MTP | UBB | Batch Size | Standard UB Size | UBB Size | GPU Layers | CPU MoE | Prompt Processing | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Q8 raw UBBoost | `Qwen3.6-35BA3B-MTP-Q8_0.gguf` | :x: | :heavy_check_mark: | 12888 | 512 | 6144 | 1 | 40 | 1288.8 T/s, about 10 s/batch | - |
| Q8 raw reference | `Qwen3.6-35BA3B-MTP-Q8_0.gguf` | :x: | :x: | 11264 | 2816 | - | 40 | 40 | 375 T/s, about 30 s/batch | - |

## Q2_K_XL Setup

From `build/bin/Release/models.ini`:

| Key | Value |
|---|---|
| Section | `Hyper` |
| `model` | `.\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` |
| `threads` | `23` |
| `parallel` | `1` |
| `top-k` | `20` |
| `min-p` | `0.0` |
| `temp` | `0.6` |
| `ctx-size` | `131072` |
| `ctk` | `q8_0` |
| `ctv` | `q8_0` |
| `n-gpu-layers` | `99` |
| `n-cpu-moe` | `38` |
| `batch-size` | `12288` |
| `ubatch-size` | `1024` |
| `no-warmup` | `true` |
| `spec-type` | `draft-mtp` |
| `spec-draft-n-max` | `5` |
| `promptprocessing-ubatchboost-size` | `2048` |
| `promptprocessing-ubatchboost-n-cpu-moe` | `99` |
| `promptprocessing-ubatchboost-gpu-layers` | `32` |
| `metrics` | `true` |
| `reasoning` | `off` |

## Older MTP Model - Qwen3.6-35BA3B-MTP-Q8_0:

| Key / flag | Value |
|---|---|
| Model | `Qwen3.6-35BA3B-MTP-Q8_0.gguf` |
| CPU | Ryzen 9 5900X |
| GPU | RTX 2080 8 GB |
| RAM | 64 GB DDR4 |
| `--threads` | `23` |
| `--ctx-size` / `-c` | `131072` |
| `--cache-type-k` / `-ctk` | `q4_0` |
| `--cache-type-v` / `-ctv` | `q4_0` |
| `--gpu-layers` / `-ngl` | `99` |
| `--n-cpu-moe` | `40` |
| `--batch-size` / `-b` | `11264` |
| `--ubatch-size` / `-ub` | `512` |
| `--no-mmap` | enabled |
| `--no-warmup` | enabled |
| `--spec-type` | `draft-mtp` |
| `--spec-draft-n-max` | `2` |
| `--promptprocessing-ubatchboost-size` | `2816` |
| `--promptprocessing-ubatchboost-gpu-layers` | `1` |
| `--promptprocessing-ubatchboost-n-cpu-moe` | `40` |

## Expected Runtime Flow

```text
Initial load
  -> UBBoost runtime if --promptprocessing-ubatchboost-size > 0
  -> prompt processing
  -> save prompt state
  -> load normal/main runtime
  -> restore prompt state
  -> token generation

Normal generation
  -> stays on the normal/main runtime

Full prompt reprocess
  -> switches to UBBoost runtime
  -> reprocesses prompt
  -> switches back to normal/main runtime
```

## Changed Files

| File | Purpose |
|---|---|
| `common/common.h` | Adds UBBoost parameter fields. |
| `common/arg.cpp` | Adds UBBoost CLI arguments. |
| `tools/server/server-context.cpp` | Adds the opt-in UBBoost runtime path for prompt processing and reprocessing. |
