# CUDA tiered-memory inference

This directory demonstrates the production `llama-tiered.h` API. The implementation is integrated into the model loader and CUDA backend; it does not require `LD_PRELOAD`, an external manifest, or a second host copy of the GGUF weights.

## Memory tiers

- **VRAM** — dense and high-activity tensors are allocated with the normal CUDA device allocator.
- **DRAM zero-copy** — selected GGUF `mmap` pages are registered read-only with CUDA and exposed to kernels through their device alias. The original file-backed pages are used directly.
- **SSD streaming** — stacked MoE expert matrices reserve a CUDA VMM address range. After the router produces expert IDs, only the selected expert granularity chunks are faulted from the GGUF mapping and copied into temporary VMM mappings. The mappings are released after the `MUL_MAT_ID` operation.

The placement score is expected bytes read per token:

```text
active bytes = tensor bytes × active fraction
```

Dense tensors have an active fraction of `1`. Stacked MoE tensors use `expert_used_count / expert_count`. Non-streamable tensors must fit in VRAM or registered DRAM; the SSD tier is restricted to MoE expert weight matrices.

## Build

The initial production implementation requires Linux, NVIDIA CUDA, CUDA VMM, and a statically linked backend registry:

```sh
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --target llama-tiered -j
```

## Run

Use an explicit DRAM budget. Omit `--vram-mib` to use currently free VRAM minus the reserve:

```sh
build/bin/llama-tiered \
  -m model.gguf \
  --dram-mib 24000 \
  --reserve-mib 2048 \
  -n 64 \
  "Explain virtual memory"
```

Use an explicit weight budget when several processes share the GPU:

```sh
build/bin/llama-tiered \
  -m model.gguf \
  --vram-mib 5000 \
  --dram-mib 24000
```

## Library API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();
llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.dram_budget_bytes = 24ull * 1024 * 1024 * 1024;
memory.vram_reserve_bytes = 2ull * 1024 * 1024 * 1024;

llama_tiered_model * owner = llama_tiered_model_load_from_file(
        "model.gguf", model_params, memory);
if (!owner) {
    fprintf(stderr, "%s\n", llama_tiered_last_error());
    return 1;
}

llama_model * model = llama_tiered_model_get_model(owner);
// Create and destroy llama_context objects normally while owner remains alive.
llama_tiered_model_free(owner);
```

`llama_tiered_model` owns the generated device list, tensor-buffer override, placement plan, and underlying `llama_model`. Keep it alive until all contexts are destroyed, and do not call `llama_model_free` on the borrowed model pointer.

## Current production constraints

- Linux and NVIDIA CUDA only.
- `GGML_BACKEND_DL=OFF` until tiered registry discovery is added to dynamic backend loading.
- Conventionally named split GGUF files are supported; custom split names are not yet exposed by the public tiered API.
- SSD streaming is limited to stacked MoE expert weight tensors consumed by `MUL_MAT_ID`.
- SSD expert transfers are correctness-first and synchronized around the expert operation. Persistent caching and transfer/compute overlap are separate optimizations.
