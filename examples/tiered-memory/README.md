<div align="center">

# ⚡ Llama Tiered Memory

### Run models larger than VRAM — without duplicating the GGUF in host memory.

**VRAM residency · registered DRAM · SSD expert streaming**

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#requirements)
[![Backend](https://img.shields.io/badge/backend-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#requirements)
[![API](https://img.shields.io/badge/API-llama--tiered.h-2563EB)](#library-api)
[![Status](https://img.shields.io/badge/status-production--oriented-F59E0B)](#current-constraints)

</div>

> [!IMPORTANT]
> This is a production-oriented integration of tiered model memory into the llama model loader and CUDA backend. It is not the earlier `LD_PRELOAD` prototype: no external placement manifest and no second host copy of the GGUF weights are required.

## Why this exists

Large Mixture-of-Experts models often fit on SSD but not in VRAM. Moving every weight through a conventional CPU staging buffer wastes memory and bandwidth, while loading every expert eagerly defeats the sparse nature of MoE inference.

Tiered Memory assigns each tensor to the cheapest usable memory tier while prioritizing the tensors that matter most per generated token.

```text
                    ┌──────────────────────────┐
                    │       GGUF model         │
                    │   single or split files  │
                    └────────────┬─────────────┘
                                 │ mmap
                   ┌─────────────▼─────────────┐
                   │   Placement planner       │
                   │ expected active bytes/token│
                   └───────┬────────┬──────────┘
                           │        │
              ┌────────────▼─┐  ┌───▼───────────────┐
              │ VRAM         │  │ registered DRAM   │
              │ hot tensors  │  │ file-backed mmap  │
              └──────────────┘  └───────────────────┘
                           │
                            │ router-selected experts
                     ┌──────▼──────────────┐
                     │ adaptive VRAM cache │
                     │ reusable VRAM scratch│
                     └─────────────────────┘
```

## Memory tiers

| Tier | Placement | Runtime behavior | Best suited for |
|---|---|---|---|
| **VRAM** | Normal CUDA allocation | Uploaded once and kept resident | Dense and high-activity tensors |
| **Registered DRAM** | Writable private GGUF `mmap` pages registered with CUDA | Selected experts are copied into reusable VRAM scratch before execution | Warm tensors that do not fit in VRAM |
| **SSD streaming** | File-backed GGUF mapping, page-locked when RAM permits | Router-selected MoE slabs use an adaptive VRAM cache before reusable scratch | Cold stacked expert matrices |

The planner ranks tensors by expected bytes read per token:

```text
active bytes = tensor bytes × active fraction
```

- Dense tensors use an active fraction of `1`.
- Stacked MoE tensors use `expert_used_count / expert_count`.
- Non-streamable tensors must fit in VRAM or registered DRAM.
- SSD placement is restricted to stacked MoE expert weight tensors consumed by `MUL_MAT_ID`.

## Installation

### 1. Install build tools

Ubuntu 22.04/24.04 or another compatible Linux distribution is recommended.

```sh
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config
```

Install an NVIDIA driver and CUDA Toolkit supported by your GPU. Confirm that both the driver and compiler are visible:

```sh
nvidia-smi
nvcc --version
```

> [!NOTE]
> CUDA Virtual Memory Management is required for the SSD tier. A successful CUDA build does not guarantee that an older GPU or driver supports the required VMM operations.

### 2. Clone the tiered-memory branch

Until the feature is merged into the default branch, clone the PR branch directly:

```sh
git clone \
  --branch feature/llamay-tiered-memory-planner \
  --single-branch \
  https://github.com/vnlpscale/Summer.cpp.git

cd Summer.cpp
```

### 3. Configure the production build

The tiered backend currently requires the statically linked backend registry. Keep `GGML_BACKEND_DL` disabled.

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_BACKEND_DL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF
```

To reduce CUDA compilation time, target the compute capability of the installed GPU. For example, Ada GPUs such as the RTX 4090 use architecture `89`:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON \
  -DGGML_BACKEND_DL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF
```

### 4. Build

```sh
cmake --build build --target llama-tiered -j"$(nproc)"
```

The executable is created at:

```text
build/bin/llama-tiered
```

### 5. Optional user-local install

Copy the statically linked executable into a directory on your user `PATH`:

```sh
install -Dm755 build/bin/llama-tiered "$HOME/.local/bin/llama-tiered"
```

If `~/.local/bin` is not already in `PATH`:

```sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

### 6. Verify the installation

```sh
llama-tiered --help
```

When running directly from the build directory, use:

```sh
build/bin/llama-tiered --help
```

## Quick start

Let the runtime use currently available VRAM while preserving a safety reserve:

```sh
llama-tiered \
  -m /path/to/model.gguf \
  --dram-mib 24000 \
  --reserve-mib 2048 \
  -n 64 \
  "Explain virtual memory"
```

Set an explicit VRAM budget when the GPU is shared with other processes:

```sh
llama-tiered \
  -m /path/to/model.gguf \
  --vram-mib 5000 \
  --dram-mib 24000 \
  -n 64
```

Enable the adaptive expert cache for models with streamed MoE tensors:

```sh
llama-tiered \
  -m /path/to/model.gguf \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 256 \
  -n 128
```

The cache is included in `--vram-mib`; it does not add 256 MiB above that placement budget. Cache slots are partitioned across streamed tensors. Single-row decode requests use aged, rank-weighted hotness and recency for admission and eviction. Multi-row prompt batches bypass the cache so they do not evict decode entries.

### Troubleshooting

| Symptom | Check |
|---|---|
| `nvcc: command not found` | Install the CUDA Toolkit or add its `bin` directory to `PATH`. |
| Tiered CUDA backend is unavailable | Reconfigure with `GGML_CUDA=ON`, `GGML_BACKEND_DL=OFF`, and `BUILD_SHARED_LIBS=OFF`. |
| CUDA VMM operation fails | Check GPU architecture, driver version, and CUDA VMM support. |
| Model load exceeds DRAM | Lower `--dram-mib`, increase the SSD tier, or reduce the model size/quantization. |
| GPU allocation fails | Increase `--reserve-mib` or set a lower explicit `--vram-mib`. |

## What happens during inference

1. The GGUF is opened with `mmap`; tiered loading disables whole-model mmap prefetch.
2. The planner assigns every tensor to VRAM, DRAM, or SSD.
3. VRAM tensors are uploaded normally.
4. DRAM and SSD expert tensors retain their original file-backed pages. Linux uses a writable private mapping so CUDA can page-lock the source without modifying the GGUF.
5. The backend allocates one reusable VRAM scratch area sized for the largest streamed expert stack.
6. If configured, a bounded expert cache is split into per-tensor slab slots.
7. When a `MUL_MAT_ID` node runs, strided router IDs are packed and copied to the host.
8. Cache hits use device-to-device copies. Misses copy page-locked GGUF data to the slab's natural scratch offset.
9. Aged router rank, frequency, and recency control admission and eviction.
10. The expert operation executes, then the original tensor pointer is restored after synchronization.

## Library API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();

llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.dram_budget_bytes  = 24ull * 1024 * 1024 * 1024;
memory.vram_reserve_bytes =  2ull * 1024 * 1024 * 1024;
memory.ssd_cache_bytes    = 256ull * 1024 * 1024;

llama_tiered_model * owner = llama_tiered_model_load_from_file(
        "model.gguf",
        model_params,
        memory);

if (!owner) {
    fprintf(stderr, "tiered load failed: %s\n", llama_tiered_last_error());
    return 1;
}

llama_model * model = llama_tiered_model_get_model(owner);

// Create and destroy llama_context objects normally while owner remains alive.

llama_tiered_model_free(owner);
```

`llama_tiered_model` owns:

- the generated device list;
- the tensor buffer override;
- the placement plan;
- the underlying `llama_model`.

Keep the owner alive until every associated `llama_context` has been destroyed. The pointer returned by `llama_tiered_model_get_model()` is borrowed; do not pass it to `llama_model_free()`.

## Requirements

- Linux
- CMake 3.14 or newer
- a C++ compiler with C++17 support
- NVIDIA CUDA with CUDA Virtual Memory Management support
- a statically linked backend registry via `GGML_BACKEND_DL=OFF`
- a GGUF model, including conventionally named split GGUF files

## Current constraints

- Dynamic backend discovery is not yet wired into the tiered registry.
- Custom split-file naming is not exposed by the public API.
- SSD streaming currently targets stacked MoE expert weight tensors used by `MUL_MAT_ID`.
- SSD transfers are correctness-first and synchronized around each expert operation.
- Page-locking can make most streamed expert weights resident in host RAM; registration falls back to pageable transfers when it fails.
- The reusable scratch allocation is as large as the largest streamed expert tensor and must fit inside the VRAM reserve.
- Graph execution is serialized when multiple contexts share one tiered model.
- The adaptive cache is decode-only, uses fixed per-tensor partitions, and falls back to direct staging if allocation fails.
- Transfer/compute overlap is not implemented.
- Physical-GPU validation is still required for each target architecture before production rollout, including logits parity, memory pressure, PCIe traffic, and throughput measurements.

## Validation checklist

For a target model and GPU, compare tiered execution against a fully resident baseline:

```text
[ ] identical prompt and sampler configuration
[ ] logits or deterministic token parity
[ ] stable VRAM and DRAM usage over long generation
[ ] no invalid memory accesses under CUDA sanitizers
[ ] expected SSD reads for selected experts only
[ ] acceptable prompt and token-generation throughput
```

---

<div align="center">

**Keep hot weights close. Stream cold experts only when the router asks for them.**

</div>
