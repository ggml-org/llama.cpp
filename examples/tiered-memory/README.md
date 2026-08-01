<div align="center">

# ⚡ Llama Tiered Memory

### Run models larger than VRAM — without duplicating the GGUF in host memory.

**VRAM residency · DRAM zero-copy · SSD expert streaming**

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
              │ VRAM         │  │ DRAM zero-copy    │
              │ hot tensors  │  │ CUDA-mapped mmap  │
              └──────────────┘  └───────────────────┘
                           │
                           │ router-selected experts
                    ┌──────▼──────────────┐
                    │ SSD expert streaming│
                    │ temporary CUDA VMM  │
                    └─────────────────────┘
```

## Memory tiers

| Tier | Placement | Runtime behavior | Best suited for |
|---|---|---|---|
| **VRAM** | Normal CUDA allocation | Uploaded once and kept resident | Dense and high-activity tensors |
| **DRAM zero-copy** | Read-only CUDA registration of GGUF `mmap` pages | Kernels access the file-backed pages through a device alias | Warm tensors that do not fit in VRAM |
| **SSD streaming** | Reserved CUDA VMM address range | Only router-selected MoE expert chunks are mapped, copied, executed, and released | Cold stacked expert matrices |

The planner ranks tensors by expected bytes read per token:

```text
active bytes = tensor bytes × active fraction
```

- Dense tensors use an active fraction of `1`.
- Stacked MoE tensors use `expert_used_count / expert_count`.
- Non-streamable tensors must fit in VRAM or registered DRAM.
- SSD placement is restricted to stacked MoE expert weight tensors consumed by `MUL_MAT_ID`.

## Quick start

### Build

```sh
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON

cmake --build build --target llama-tiered -j
```

### Run

Let the runtime use currently available VRAM while preserving a safety reserve:

```sh
build/bin/llama-tiered \
  -m model.gguf \
  --dram-mib 24000 \
  --reserve-mib 2048 \
  -n 64 \
  "Explain virtual memory"
```

Set an explicit VRAM budget when the GPU is shared with other processes:

```sh
build/bin/llama-tiered \
  -m model.gguf \
  --vram-mib 5000 \
  --dram-mib 24000 \
  -n 64
```

## What happens during inference

1. The GGUF is opened with `mmap`; tiered loading disables whole-model mmap prefetch.
2. The planner assigns every tensor to VRAM, DRAM, or SSD.
3. VRAM tensors are uploaded normally.
4. DRAM tensors retain their original file-backed pages and receive CUDA device aliases through read-only host registration.
5. SSD-tier tensors reserve virtual GPU address ranges without allocating the complete tensor in VRAM.
6. When a `MUL_MAT_ID` node runs, router IDs are copied to the host.
7. Selected expert chunks are deduplicated, faulted from the GGUF mapping, and mapped into CUDA VMM.
8. The expert operation executes, then its temporary mappings are synchronized and released.

## Library API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();

llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.dram_budget_bytes  = 24ull * 1024 * 1024 * 1024;
memory.vram_reserve_bytes =  2ull * 1024 * 1024 * 1024;

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
- NVIDIA CUDA with CUDA Virtual Memory Management support
- a statically linked backend registry via `GGML_BACKEND_DL=OFF`
- a GGUF model, including conventionally named split GGUF files

## Current constraints

- Dynamic backend discovery is not yet wired into the tiered registry.
- Custom split-file naming is not exposed by the public API.
- SSD streaming currently targets stacked MoE expert weight tensors used by `MUL_MAT_ID`.
- SSD transfers are correctness-first and synchronized around each expert operation.
- Persistent expert caching and transfer/compute overlap are not implemented.
- Physical-GPU validation is still required for each target architecture before production rollout, including logits parity, memory pressure, PCIe traffic, and throughput measurements.

## Validation checklist

For a target model and GPU, compare tiered execution against a fully resident baseline:

```text
[ ] identical prompt and sampler configuration
[ ] logits or deterministic token parity
[ ] stable VRAM and DRAM usage over long generation
[ ] no invalid VMM accesses under CUDA sanitizers
[ ] expected SSD reads for selected experts only
[ ] acceptable prompt and token-generation throughput
```

---

<div align="center">

**Keep hot weights close. Stream cold experts only when the router asks for them.**

</div>
