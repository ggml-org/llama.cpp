# Tiered-memory runtime prototype

This example implements the core memory hierarchy used by LlamaY on top of the existing llama.cpp graph scheduler:

- **VRAM:** tensors are left on the CUDA device.
- **DRAM:** tensors stay in the GGUF `mmap`, and only their file-backed pages are registered with `cudaHostRegister` for direct asynchronous DMA. No second host copy is allocated.
- **SSD:** tensors stay in the same `mmap` but are not registered. The operating system faults pages in from storage when the scheduler requests them.

The existing `op_offload` scheduler path runs supported operations on CUDA even when their weight buffer is on the CPU. It creates the required per-split device copies at graph execution time. Registering the DRAM ranges makes those copies use pinned pages; leaving SSD ranges unregistered preserves demand paging.

## Build

```sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --target llama-tiered-plan llama-tiered-preload llama-cli -j
```

The preload component is currently Linux-only.

## Generate a plan

```sh
build/bin/llama-tiered-plan model.gguf \
  --vram-mib 5000 \
  --dram-mib 24000 \
  --manifest model.tiers \
  --print-command
```

The planner assigns whole tensors using the expected bytes read per token:

```text
active bytes = tensor bytes × active fraction
```

Dense attention, normalization, router, and dense FFN weights use an active fraction of `1`. Stacked MoE expert tensors use `expert_used_count / expert_count`. Higher active fractions receive VRAM first, then page-locked DRAM, with the remainder left on SSD-backed `mmap` pages.

The tool writes:

1. A manifest containing exact GGUF file offsets for each tier.
2. A `--override-tensor` expression that keeps DRAM/SSD tensors in the CPU mmap buffer.
3. With `--print-command`, a complete example invocation.

## Run

Use the command printed by the planner. Its essential form is:

```sh
LLAMA_TIERED_MANIFEST=model.tiers \
LD_PRELOAD=/absolute/path/to/libllama-tiered-preload.so \
build/bin/llama-cli \
  -m model.gguf \
  -ngl 999 \
  --override-tensor '^(?:...non-VRAM tensor names...)$=CPU'
```

The preload library only changes mappings whose canonical path appears in the manifest. For those mappings it uses `MAP_PRIVATE` with writable copy-on-write pages, then registers only the page-aligned DRAM ranges. The process does not write model data, so the private mapping does not create a duplicate unless another component modifies it.

## Current scope

- Single-file GGUF models. Split GGUF manifests are not generated yet.
- Linux and CUDA runtime (`libcudart`).
- Whole-tensor placement; tensors are not split across tiers.
- Runtime copies are scheduled by the existing ggml backend scheduler. This prototype does not yet add expert-slab gathering or transfer/compute overlap beyond the scheduler's existing asynchronous copy facilities.

Use this as an experimental path. Verify generated outputs against a fully resident run before relying on it.
