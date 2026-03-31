# llama.cpp-tq3

**Run a 27B language model on a single 16GB GPU — with near-Q4_0 quality.**

TQ3_1S is a 3.5-bit quantization format that compresses model weights using Walsh-Hadamard rotation and dual-scale encoding. On Qwen3.5-27B it reaches near-`Q4_0` quality while being materially smaller — small enough to fit models on consumer GPUs that `Q4_0` cannot fit fully on in the same setup.

This fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) adds the CUDA kernels needed to run TQ3_1S models at full speed.

## Why This Matters

Large language models are powerful but expensive to run. A 27B-parameter model in `Q4_0` is large enough that full-GPU deployment on a 16GB card becomes impractical in this setup.

TQ3_1S solves that deployment problem. By squeezing the same model into about 12.9 GB with near-`Q4_0` quality, it fits fully on GPU. The practical result on this card is much better end-to-end throughput than a partially offloaded `Q4_0` run.

## Benchmark (Qwen3.5-27B, RTX 5060 Ti 16GB)

Gold-standard `wiki.test.raw` pass, `c=512`, full `580` chunks:

| | TQ3_1S | Q4_0 |
|---|---:|---:|
| Perplexity | `7.2570 +/- 0.0480` | `7.2431 +/- 0.0482` |
| Model size | **12.9 GB** | 14.4 GB |
| Fits 16GB GPU fully | **✅** | ❌ |

Perplexity gap: `+0.0139`, about **0.19%**.

Safe interpretation:

- `TQ3_1S` is materially smaller than `Q4_0`
- `TQ3_1S` reaches near-`Q4_0` quality on this 27B witness
- the practical speed win is a **deployment / fit advantage on a 16GB GPU**, not a claim that native `TQ3_1S` kernels are universally faster than native `Q4_0`

## Pre-quantized Model

**[turbo-tan/Qwen3.5-27B-TQ3_1S on Hugging Face](https://huggingface.co/turbo-tan/Qwen3.5-27B-TQ3_1S)** — download and run immediately.

## Build From Source

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git
cd llama.cpp-tq3

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Quick Start

Set portable path variables:

```bash
export USERNAME="${USERNAME:-$USER}"
export CODE_ROOT="/home/$USERNAME/code"
export MODEL_ROOT="/home/$USERNAME/models"
```

Download the published GGUF from Hugging Face:

```bash
mkdir -p "$MODEL_ROOT/turboquant27"

python3 - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="turbo-tan/Qwen3.5-27B-TQ3_1S",
    filename="Qwen3.5-27B-TQ3_1S.gguf",
    local_dir=f"/home/{__import__('os').environ.get('USERNAME', __import__('os').environ.get('USER'))}/models/turboquant27",
)
PY
```

```bash
# Serve the published 27B model
./build/bin/llama-server \
    -m "$MODEL_ROOT/turboquant27/Qwen_Qwen3.5-27B-TQ3_1S.gguf" \
    -ngl 99 \
    -fa on \
    -c 4096 \
    --port 8090 \
    --reasoning off \
    --reasoning-budget 0 \
    --reasoning-format none
```

## Benchmark

Gold-standard perplexity run:

```bash
./build/bin/llama-perplexity \
    -m "$MODEL_ROOT/turboquant27/Qwen_Qwen3.5-27B-TQ3_1S.gguf" \
    -f "$CODE_ROOT/llama.cpp/wikitext-2-raw/wiki.test.raw" \
    -c 512 \
    -ngl 99 \
    -fa 1 \
    -t 8 \
    --no-warmup
```

Quick throughput check:

```bash
./build/bin/llama-bench \
    -m "$MODEL_ROOT/turboquant27/Qwen_Qwen3.5-27B-TQ3_1S.gguf" \
    -ngl 99 \
    -fa on
```

If you downloaded the GGUF from Hugging Face into a different location, replace `MODEL_ROOT` accordingly.

## How TQ3_1S Works

Standard quantization maps each weight to the nearest value on a uniform grid. TQ3_1S takes a different approach:

1. **Rotate** — Apply a Walsh-Hadamard Transform to each 32-element weight block. This spreads information across elements, making the distribution more uniform and easier to quantize. Inspired by [RaBitQ](https://arxiv.org/abs/2405.12497).

2. **Quantize** — Map each rotated value to one of 8 learned centroids (3 bits). The centroids are optimized for the post-rotation distribution.

3. **Dual-scale** — Store two fp16 scale factors per block: one for elements 0–15, one for 16–31. This captures local magnitude variation that a single scale would miss.

Block layout: `[d0: fp16][d1: fp16][qs: 12 bytes]` = 16 bytes per 32 elements (4.0 bits per weight)

During inference, activations are pre-rotated into the same WHT domain, allowing the CUDA kernel to compute dot products directly against centroids without inverse-transforming the weights.

## The Journey

This project grew from a practical need: running capable models on consumer hardware without renting cloud GPUs. We explored adaptive block promotion, imatrix-weighted quantization, mixed-precision hybrids, and multiple transform ideas. Many of them improved tensor metrics but failed to hold up at model level. The dual-scale WHT approach survived because it delivered the strongest practical 27B result while keeping the runtime path simple enough to deploy.

The MMVQ kernel that powers token generation was validated against CPU baselines and then stress-tested over the full `580`-chunk `wiki.test.raw` pass to catch numerical drift that short evaluations can miss.

## Evidence

Recorded artifacts used for the headline claim:

- full `TQ3_1S` PPL run: `7.2570 +/- 0.04802`
- full `Q4_0` PPL run: `7.2431 +/- 0.04822`
- model sizes:
  - `TQ3_1S`: about `12.9 GB`
  - `Q4_0`: about `14.4 GB`

If you want to reproduce the result, use the same base model, the same `wiki.test.raw` corpus, and the same `llama-perplexity` settings on both formats.

## Credits

- [RaBitQ](https://arxiv.org/abs/2405.12497) — Walsh-Hadamard transform inspiration
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine
- [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) — base model

## License

MIT — same as llama.cpp
