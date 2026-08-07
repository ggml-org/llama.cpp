<div align="center">

# Summer.cpp

### A llama.cpp fork for running GGUF models larger than VRAM across NVIDIA GPU, DRAM, and SSD

**Tiered-memory execution · split GGUF support · local CUDA inference**

[日本語](README.md) · **English** · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#requirements)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#requirements)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#manual-build)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

</div>

> [!IMPORTANT]
> The most stable configuration currently uses **VRAM + DRAM**. On a GTX 1660 SUPER, `--vram-mib 3800 --dram-mib 6500` has been validated with 0 MiB assigned to SSD. Selective SSD streaming on Turing GPUs remains experimental.

## Overview

Summer.cpp adds a tiered-memory backend and a dedicated `llama-tiered` executable to llama.cpp. Large GGUF tensors are placed according to workload and memory budgets.

| Tier | Location | Purpose |
|---|---|---|
| VRAM | CUDA device memory | Frequently used dense weights, embeddings, and hot tensors |
| DRAM | CUDA-mapped host memory | Weights that do not fit in VRAM, accessed through zero-copy or mapped pinned copies |
| SSD | File-backed GGUF mapping | Selected MoE expert weights staged on demand, with hot experts retained in an adaptive cache |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## Validated configuration

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| Model | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| GGUF size | About 9.35 GiB |
| VRAM budget | 3800 MiB |
| DRAM budget | 6500 MiB |
| SSD placement | 0 MiB |
| Load time | About 5.65 s |
| Prompt processing | About 31.7 tokens/s |
| Token generation | About 27.7 tokens/s |

These figures are one short-prompt measurement. Results vary with the model, context, sampler, CPU, PCIe link, driver, and background workload.

## Requirements

- Linux; Ubuntu 22.04 or 24.04 is recommended
- NVIDIA GPU and driver
- CUDA Toolkit with `nvcc`
- CMake and a C++17 compiler
- Python 3.10 or later
- A GGUF model
- Enough SSD capacity for the model
- Enough system RAM when using the DRAM tier

For a GTX 1660 SUPER and a roughly 9.35 GiB model, at least 16 GiB of system RAM is recommended, including the OS and other processes. The DRAM fallback may allocate a mapped pinned copy separately from the file mapping.

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## Quick installation

### 1. Install dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

Install a CUDA Toolkit compatible with your NVIDIA driver and GPU if it is not already available.

### 2. Clone

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. Build and install

```bash
bash scripts/install-summer.sh
```

The installer applies the Turing/GTX 16 DRAM fallback patches, builds the CUDA `llama-tiered` target in Release mode, installs it to `~/.local/bin`, removes legacy SummerCLI commands, and creates `~/models`.

GPU architecture is detected through `nvidia-smi`. To set it explicitly for a GTX 1660 SUPER:

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

For Tensor Core GPUs where forced MMQ is unnecessary:

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. Configure PATH

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. Place a model

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

For split GGUF files, place every part in the same directory.

### 6. Run

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "Introduce yourself."
```

The CLI prints the llama.cpp block logo and Summer.cpp banner to standard error, so generated text on standard output remains suitable for piping.

## Manual build

```bash
cd "$HOME/Summer.cpp"

python3 scripts/apply-tiered-dram-pinned-fallback.py
python3 scripts/apply-tiered-dram-matmul-staging.py
python3 scripts/apply-tiered-no-prompt-echo.py

rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_BACKEND_DL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build --target llama-tiered -j"$(nproc)"
install -Dm755 build/bin/llama-tiered "$HOME/.local/bin/llama-tiered"
```

## Adaptive expert cache

For large MoE models with weights placed on SSD, enable the adaptive cache with `--cache-mib`. Its capacity is part of `--vram-mib` and is deducted automatically from the resident-weight budget.

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "Hello"
```

The cache learns hot experts from router history during single-row decode. It is bypassed for multi-row prompt batches to avoid cache pollution. Final logs report hit rate, H2D/D2D traffic, admissions, and evictions.

## Memory-budget tuning

A practical starting point for a GTX 1660 SUPER is:

```text
--vram-mib 3800 --dram-mib 6500
```

- Reduce `--vram-mib` after GPU allocation errors.
- Increase `--dram-mib` if tensors are placed on SSD.
- Use a smaller or more aggressively quantized GGUF when system RAM is insufficient.
- Leave additional VRAM reserve when the GPU also drives a desktop.

Check the startup log:

```text
tiered weights: VRAM ... MiB, DRAM ... MiB, SSD 0.00 MiB (0 streamed tensors)
```

For maximum stability, use a configuration that reports `SSD 0.00 MiB`.

## Troubleshooting

- Missing `build/bin/llama-tiered`: rerun `bash scripts/install-summer.sh`.
- `llama_model_load: ... invalid argument` on Turing/GTX 16: apply `scripts/apply-tiered-dram-pinned-fallback.py` and rebuild.
- `tensor_state layout did not match expected source`: restore `ggml/src/ggml-cuda/tiered.cu`, then rerun the installer.
- `operation not supported`: remove `build`, then rebuild with the current patches.
- Illegal CUDA memory access: update and rebuild; reproduce short prompts with `compute-sanitizer --tool memcheck`.
- `summer: command not found`: the supported executable is `llama-tiered`; ensure `~/.local/bin` is in `PATH`.
- Model not found: run `find "$HOME/models" -type f -iname '*.gguf'`.

## SSD streaming status

The SSD tier keeps stacked MoE expert tensors in the GGUF mapping and transfers only the selected expert slab into reusable VRAM scratch during `MUL_MAT_ID`. Current limitations include resident page-locked source memory, scratch sized to the largest stacked expert tensor, correctness-first synchronization without transfer/compute overlap, single-row adaptive caching, serialized graph execution for contexts sharing a model, and the need for physical validation on each GPU architecture.

GTX 1660 SUPER with Laguna-S-2.1 IQ1_S has been tested with 1, 16, and 128 generated tokens and under `compute-sanitizer`. Start with short generations on other GPUs and models.

## Library API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();
llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.vram_budget_bytes = 3800ull * 1024 * 1024;
memory.dram_budget_bytes = 6500ull * 1024 * 1024;

llama_tiered_model * owner = llama_tiered_model_load_from_file(
        "model.gguf", model_params, memory);
if (!owner) {
    fprintf(stderr, "tiered load failed: %s\n", llama_tiered_last_error());
    return 1;
}

llama_model * model = llama_tiered_model_get_model(owner);
// Create and use llama_context while owner remains alive.
llama_tiered_model_free(owner);
```

The pointer returned by `llama_tiered_model_get_model()` is borrowed. Do not pass it directly to `llama_model_free()`.

## Project structure

```text
Summer.cpp/
├── examples/tiered-memory/                 tiered-memory CLI and design
├── ggml/src/ggml-cuda/tiered.cu           CUDA tiered backend
├── scripts/                                patch and installation scripts
├── src/                                    llama library
└── build/bin/llama-tiered                  built executable
```

## Uninstall

```bash
rm -f "$HOME/.local/bin/llama-tiered"
rm -f "$HOME/.local/bin/Summer.CPP"
rm -f "$HOME/.local/bin/summer"
```

Remove source and models only when they are no longer needed.

## Upstream

Summer.cpp is based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Thanks to the upstream contributors working on llama.cpp, ggml, CUDA backends, quantization, and model loading.

## License

This repository uses the same MIT License as upstream llama.cpp. See [LICENSE](LICENSE).
