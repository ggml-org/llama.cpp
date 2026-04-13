# llama.cpp-tq3

Runtime fork for TurboQuant GGUFs in `llama.cpp`.

Public focus:
- `TQ3_1S`
- `TQ3_4S`

This repo is the runtime side only. The quantization tooling is intentionally not part of the public fork.

## Current 27B Results

`Qwen3.5-27B`, `wiki.test.raw`, full pass, `c=2048`:

| Format | PPL | Size |
|--------|-----|------|
| `TQ3_4S` | `6.8224 +/- 0.04534` | `12.9 GiB` |
| `Q3_K_S` | `6.8630 +/- 0.04583` | `11.4 GiB` |
| `TQ3_1S` | `6.9807 +/- 0.04690` | `12.9 GiB` |
| `EXL3 3.0bpw` | `7.027580` | `~13.0 GiB` |

Notes:
- `EXL3 3.0bpw` is from a local `145 x 2048` eval, not `llama-perplexity`.
- The public runtime path here is the clean base `TQ3_1S` / `TQ3_4S` support.

## Current Runtime Speed

Measured on:

- GPU: `RTX 5060 Ti 16GB`
- build: public runtime branch
- command family: `llama-bench`

### Qwen3.5-27B-TQ3_4S

| Metric | Result |
|--------|--------|
| `pp2048` | `~708 tok/s` |
| `tg128` | `23.2 tok/s` |
| size | `12.99 GiB` |

Notes:

- these are the current public runtime numbers for the 27B `TQ3_4S` path
- the prompt-path headline is the important public speed claim: `~708 tok/s` at `pp2048`
- `pp` and `tg` should be compared separately
- if you benchmark with custom KV-cache settings, keep that separate from the base weight-format numbers

## Build From Source

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git
cd llama.cpp-tq3

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Download A Model

Example base `TQ3_4S` release:

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download YTan2000/Qwen3.5-27B-TQ3_4S \
  Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  --local-dir ./models/tq3_4s
```

## Run

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 2048 \
  --port 8090
```

## KV Cache Settings

For the cleanest baseline, run with normal KV cache:

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 4096 \
  --port 8090
```

For the TurboQuant KV-cache path used in local experiments, use:

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 8192 \
  -ctk tq3_0 \
  -ctv tq3_0 \
  --port 8090
```

Notes:
- `-ctk tq3_0 -ctv tq3_0` is the currently supported TurboQuant KV-cache setting in this fork.
- Keep weight-format and KV-cache experiments separate when benchmarking.
- If you want the exact weight-format baseline, leave KV cache at the default types.

## Benchmark

```bash
./build/bin/llama-bench \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -p 2048 \
  -n 0 \
  -r 3

./build/bin/llama-bench \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -p 0 \
  -n 128 \
  -r 3

./build/bin/llama-perplexity \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -f /path/to/wiki.test.raw \
  -c 2048 \
  -ngl 99 \
  -fa 1 \
  -t 8 \
  --no-warmup
```

## Scope

- runtime support for `TQ3_1S` and `TQ3_4S`
- CUDA path for local inference
- pre-quantized GGUF workflow

Not in scope for this public fork:
- quantizer research code
- private mixed-policy search tooling
- HF upload helpers

## Credits

- `TheTom/turboquant_plus` for public TurboQuant KV-cache engineering, benchmarking, and implementation notes:
  - https://github.com/TheTom/turboquant_plus
- `TheTom/ik_llama.cpp` for related open-source runtime work in the same area:
  - https://github.com/TheTom/ik_llama.cpp
