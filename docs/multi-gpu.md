# Using multiple GPUs with llama.cpp

This guide explains how to run [llama.cpp](https://github.com/ggml-org/llama.cpp) across more than one GPU. It covers the split modes, the command-line flags that control them, the limitations you need to know about, and ready-to-use recipes for `llama-cli` and `llama-server`.

The CLI arguments listed here are the same for both tools - or most llama.cpp binaries for that matter.

---

## When you need multi-GPU

Reach for multi-GPU when one of these is true:

- **The model doesn't fit in a single GPU's VRAM.** By spreading the weights across two or more GPUs the whole model can stay on accelerators. Otherwise part of the model will need to be run off of the comparatively slower system RAM.
- **You want more throughput.** By distributing the computation across multiple GPUs, each individual GPU has to do less work. This can result in better prefill and/or token generation performance, depending on the split mode and interconnect speed vs. the speed of an individual GPU.

---

## The split modes

Set with `--split-mode` / `-sm`.

| Mode | What it does | When to use |
|---|---|---|
| `none` | Use a single GPU only. Pick which one with `--main-gpu`. | You explicitly want to confine the model to one GPU even though more are visible. |
| `layer` (**default**) | Pipeline parallelism. Each GPU holds a contiguous slice of layers. The KV cache for layer *l* lives on the GPU that owns layer *l*. | Default and most compatible multi-GPU choice. You want more memory than a single GPU provides and your priority is a fast prefill. Can tolerate slow interconnect speeds between GPUs. |
| `row` | **Deprecated.** Older row-split tensor-parallel path with comparatively poor performance. Splits only dense weights across GPUs. Superseded by `tensor` which should be universally superior if it can be used. | Avoid in new deployments. |
| `tensor` | **EXPERIMENTAL.** Tensor parallelism that splits both weights *and* KV across the participating GPUs via a "meta device" abstraction. | You want more memory than a single GPU provides and your priority is fast token generation. Prefill speeds approach pipeline parallel speeds for large, dense models and fast GPU interconnect speeds. Treat as experimental as the code is less mature than pipeline parallelism. Performance should be good for multiple NVIDIA GPUs using the CUDA backend, no guarantees otherwise. |

> Pipeline parallel (`layer`) vs. tensor parallel (`tensor`): pipeline-parallel runs different layers on different GPUs and processes tokens sequentially through the pipeline. This minimizes data transfers between GPUs but requires many tokens to scale well. Tensor-parallel splits each layer across GPUs and does multiple cross-GPU reductions per layer. This enables parallelizing any workload but is much more bottlenecked by the GPU interconnect speed. Pipeline-parallel maximizes batch throughput; tensor-parallel minimizes latency.

---

## Command-line arguments reference

| Short | Long | Value | Default | Notes |
|---|---|---|---|---|
| `-sm` | `--split-mode` | `none` \| `layer` \| `tensor` | `layer` | See modes above. |
| `-ts` | `--tensor-split` | comma-separated proportions, e.g. `3,1` | mode-dependent | How much of the model goes to each GPU. If omitted, `layer`/`row` use automatic splitting proportional to memory, while `tensor` splits tensor segments evenly. With `3,1` on two GPUs, GPU 0 gets 75 %, GPU 1 gets 25 %. The values follow the order in `--device`. |
| `-mg` | `--main-gpu` | integer device index | `0` | The single GPU used in `--split-mode none`. |
| `-ngl` | `--n-gpu-layers` / `--gpu-layers` | integer \| `auto` \| `all` | `auto` | Maximum number of layers to keep in VRAM. Use `999` or `all` to push everything possible to the GPUs. |
| `-dev` | `--device` | comma-separated device names, or `none` | auto | Restrict which devices llama.cpp may use. See `--list-devices` for names. |
| | `--list-devices` | - | - | Print the available devices and their memory. Run this first to learn the names you'd pass to `--device`. |
| `-fa` | `--flash-attn` | `on` \| `off` \| `auto` | `auto` | Required when using `--split-mode tensor` and/or quantized V cache. Supported (and therefore enabled by default) for most combinations of models and backends. |
| `-ctk` | `--cache-type-k` | `f32` \| `f16` \| `bf16` \| `q8_0` \| `q4_0` \| ... | `f16` | KV cache type for K. |
| `-ctv` | `--cache-type-v` | same as `-ctk` | `f16` | KV cache type for V. |
| `-fit` | `--fit` | `on` \| `off` | `on` | Auto-fit unset args to device memory. **Not supported with `tensor`. You may need to manually set the `--ctx-size` to make the model fit.**  |

As for any CUDA program, the environment variable `CUDA_VISIBLE_DEVICES` can be used to control which GPUs to use for the CUDA backend: if you set it, llama.cpp only sees the specified GPUs. Use `--device` for selecting GPUs from among those visible to llama.cpp, this works for any backend.

---

## Recipes

### 1. Default - pipeline parallel across all visible GPUs

```bash
llama-cli -m model.gguf
llama-server -m model.gguf
```

Easiest configuration. KV cache spreads across the GPUs along with the layers. `--fit` (on by default) sizes things automatically.

### 2. Pipeline parallel with a custom split ratio

```bash
llama-cli -m model.gguf -ts 3,1
```

Useful when GPUs have different memory: GPU 0 (3 parts) and GPU 1 (1 part). Proportions are normalized so `-ts 3,1` is the same as e.g. `-ts 75,25`.

### 3. Single-GPU mode, picking a specific GPU

```bash
llama-cli --list-devices
llama-cli -m model.gguf -dev CUDA1
```

Use only the device listed as `CUDA1` when calling with `--list-devices`.

### 4. Tensor parallelism (experimental)

```bash
llama-cli -m model.gguf -sm tensor -ctk f16 -ctv f16
```

- `--flash-attn off` or (`--flash-attn auto` resolving to `off` when it isn't supported) is a hard error.
- KV cache types must be non-quantized: `f32`, `f16`, or `bf16`. Support for quantized KV cache is not implemented and trying to use it will result in an error.
- Mark this configuration as experimental in your tooling: validate output quality before deploying.
- `--split-mode tensor`is not implemented for all architectures. The following will fail with *"LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '...'"*:

  - **MoE / hybrid:** Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H, Nemotron-H-MoE, Granite-Hybrid, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1
  - **State-space / RWKV-style:** Mamba, Mamba2 (and the hybrid Mamba-attention models above)
  - **Other:** PLAMO2, MiniCPM3, Gemma-3n, OLMo2, BitNet, T5

### 5. With NCCL

There's no runtime flag for NCCL - it's selected at build time (`-DGGML_CUDA_NCCL=ON`, this is the default). Note that NCCL is **not** automatically distributed with CUDA and you may need to install it manually - when in doubt check the CMake log to see whether or not it can find the package. When llama.cpp is compiled with NCCL support it uses it automatically for cross-GPU reductions in `tensor` mode. When NCCL is missing on a multi-GPU build, you'll see this one-time warning and performance will be lower:

```
NVIDIA Collective Communications Library (NCCL) is unavailable, multi GPU performance will be suboptimal
```

When using the "ROCm" backend (which is the ggml CUDA code translated for AMD via HIP), the AMD equivalent RCCL can be used by compiling with `-DGGML_HIP_RCCL=ON`. Note that RCCL is by default *disabled* because (unlike NCCL) it was not universally beneficial during testing.
### 6. With CUDA peer-to-peer access (`GGML_CUDA_P2P`)

CUDA peer-to-peer (P2P) lets GPUs transfer data directly between each other instead of going through system memory, which generally improves multi-GPU performance. It is **opt-in** at runtime - set the environment variable `GGML_CUDA_P2P` to any value to enable it:

```bash
GGML_CUDA_P2P=1 llama-cli -m model.gguf -sm tensor
```

P2P requires driver support (usually restricted to workstation/datacenter GPUs) and **may cause crashes or corrupted outputs on some motherboards or BIOS configurations** (e.g. when IOMMU is enabled). If you see instability after enabling it, unset the variable.

---

## Troubleshooting

| Symptom | How to fix |
|---|---|
| Startup error *"SPLIT_MODE_TENSOR requires flash_attn to be enabled"* | Add `-fa on` or remove `-fa off`. |
| Startup error *"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented"* | Use `-ctk f16 -ctv f16` (or `bf16`/`f32`) with `--split-mode tensor`. |
| Startup error *"LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'X'"* | Architecture not on the TENSOR allow-list. Use `--split-mode layer`. |
| Warning *"NCCL is unavailable, multi GPU performance will be suboptimal"* | llama.cpp wasn't built with NCCL. Either accept the lower performance or install NCCL and rebuild. |
| CUDA OOM at startup or during prefill in `--split-mode tensor` | Auto-fit is disabled in this mode, so reduce memory pressure yourself. In order from least to most disruptive: lower `--ctx-size` (`-c`) (KV cache is roughly proportional to `n_ctx`); for `llama-server`, lower `--parallel` (`-np`) (a slot KV cache is allocated per concurrent sequence); as a last resort, reduce `--n-gpu-layers` (`-ngl`) (the remaining layers run on CPU and inference will be much slower). |
| Performance is worse with multi-GPU than single-GPU | The performance is bottlenecked by GPU interconnect speed. For `--split-mode tensor`, verify that NCCL is being used. Try `--split-mode layer` (less communication than `tensor`). Increase GPU interconnect speed via more PCIe lanes or e.g. NVLink (if available). |
| GPU not used at all | `--n-gpu-layers` is `0` or too low - try explicitly setting `-ngl all`. Or you are accidentally hiding the GPUs via an environment variable like `CUDA_VISIBLE_DEVICES=-1`. Or your build doesn't include support for the relevant backend. |
| Crashes or corrupted outputs after setting `GGML_CUDA_P2P=1` | Some motherboards and BIOS settings (e.g. with IOMMU enabled) don't support CUDA peer-to-peer reliably. Unset `GGML_CUDA_P2P`. |
