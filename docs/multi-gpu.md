# Using multiple GPUs with llama.cpp

This guide explains how to run [llama.cpp](https://github.com/ggml-org/llama.cpp) across more than one GPU. It covers the split modes, the command-line flags that control them, the limitations you need to know about, and ready-to-use recipes for `llama-cli` and `llama-server`.

The same flags work in both tools - they share `common/arg.cpp`.

---

## When you need multi-GPU

Reach for multi-GPU when one of these is true:

- **The model doesn't fit in a single GPU's VRAM.** Spread the weights across two or more GPUs so the whole model can stay on accelerators.
- **You want more throughput.** Distributing computation across GPUs can reduce per-token latency or lift batch throughput, depending on the split mode.

---

## The split modes

Set with `--split-mode` / `-sm`.

| Mode | What it does | When to use |
|---|---|---|
| `none` | Use a single GPU only. Pick which one with `--main-gpu`. | You explicitly want to confine the model to one GPU even though more are visible. |
| `layer` (**default**) | Pipeline parallelism. Each GPU holds a contiguous slice of layers. The KV cache for layer *l* lives on the GPU that owns layer *l*. | Default and most compatible multi-GPU choice. |
| `row` | **Deprecated.** Older row-split tensor-parallel path. Superseded by `tensor`. Avoid in new deployments. | Not recommended. |
| `tensor` | **EXPERIMENTAL.** Tensor parallelism that splits both weights *and* KV across the participating GPUs via a "meta device" abstraction. | If you want true TP including KV distribution and your model architecture is supported (see limitations below). Treat as experimental - verify correctness on your model before relying on it. |

> Pipeline parallel (`layer`) vs. tensor parallel (`tensor`): pipeline-parallel runs different layers on different GPUs and processes tokens sequentially through the pipeline. Tensor-parallel splits each layer across GPUs and does cross-GPU reductions inside every layer. Pipeline-parallel maximizes batch throughput; tensor-parallel can reduce single-token latency but adds communication on every layer.

---

## Command-line flags reference

| Short | Long | Value | Default | Notes |
|---|---|---|---|---|
| `-sm` | `--split-mode` | `none` \| `layer` \| `tensor` | `layer` | See modes above. |
| `-ts` | `--tensor-split` | comma-separated proportions, e.g. `3,1` | mode-dependent | How much of the model goes to each GPU. If omitted, `layer`/`row` use automatic memory-based splitting, while `tensor` splits tensor segments evenly. With `3,1` on two GPUs, GPU 0 gets 75 %, GPU 1 gets 25 %. The values follow the order in `--device`. |
| `-mg` | `--main-gpu` | integer device index | `0` | The single GPU used in `--split-mode none`. |
| `-ngl` | `--n-gpu-layers` / `--gpu-layers` | integer \| `auto` \| `all` | `auto` | Maximum number of layers to keep in VRAM. Use `999` or `all` to push everything possible to the GPUs. |
| `-dev` | `--device` | comma-separated device names, or `none` | auto | Restrict which devices llama.cpp may use. See `--list-devices` for names. |
| | `--list-devices` | - | - | Print the available devices and their memory. Run this first to learn the names you'd pass to `--device`. |
| `-fa` | `--flash-attn` | `on` \| `off` \| `auto` | `auto` | Required by `--split-mode tensor`. Recommended whenever you also use quantized V cache. |
| `-ctk` | `--cache-type-k` | `f32` \| `f16` \| `bf16` \| `q8_0` \| `q4_0` \| ... | `f16` | KV cache type for K. |
| `-ctv` | `--cache-type-v` | same as `-ctk` | `f16` | KV cache type for V. Quantized V requires `-fa on`. |
| `-fit` | `--fit` | `on` \| `off` | `on` | Auto-fit unset args to device memory. **Not supported with `tensor`** - set `-fit off` for that mode. |

`CUDA_VISIBLE_DEVICES` is honored implicitly through CUDA itself: if you set it, llama.cpp only sees the GPUs CUDA exposes. Use `--device` for finer control across multiple backends.

---

## Recipes

### 1. Default - pipeline parallel across all visible GPUs

```bash
llama-cli   -m model.gguf -ngl all
llama-server -m model.gguf -ngl all --host 0.0.0.0 --port 8080
```

Easiest configuration. KV cache spreads across the GPUs along with the layers. `--fit` (on by default) sizes things automatically.

### 2. Pipeline parallel with a custom split ratio

```bash
llama-cli -m model.gguf -ngl all -sm layer -ts 3,1
```

Useful when GPUs have different memory: GPU 0 (3 parts) and GPU 1 (1 part). Proportions are normalized.

### 3. Single-GPU mode, picking a specific GPU

```bash
llama-cli -m model.gguf -ngl all -sm none -mg 1
```

Use only device index 1, even if more GPUs are visible.

### 4. Tensor parallelism - TENSOR mode (experimental)

```bash
llama-cli -m model.gguf -ngl all -sm tensor -fa on -ctk f16 -ctv f16 -fit off
```

- `--flash-attn on` is **mandatory**. Setting it to `auto` is fine (it gets upgraded automatically); setting it to `off` is a hard error.
- KV cache types must be non-quantized: `f32`, `f16`, or `bf16`. Quantized KV cache will refuse to start with this mode.
- `-fit off` because auto-fit isn't implemented for `tensor`.
- The model architecture must be on the allow-list - see limitations below.
- Mark this configuration as experimental in your tooling: validate output quality before deploying.

### 5. Restricting which GPUs participate

```bash
llama-cli --list-devices
# CUDA0: NVIDIA RTX A6000 (48 GiB)
# CUDA1: NVIDIA RTX 4090 (24 GiB)
# CUDA2: NVIDIA RTX 4090 (24 GiB)

llama-cli -m model.gguf -ngl all -sm tensor -dev CUDA0,CUDA2 -fa on -fit off
```

You can also use `CUDA_VISIBLE_DEVICES=0,2 llama-cli ...` to do this at the CUDA layer.

### 6. With NCCL

There's no runtime flag for NCCL - it's selected at build time (`-DGGML_CUDA_NCCL=ON`). When NCCL is compiled in and the system has it installed, llama.cpp uses it automatically for cross-GPU reductions in `tensor` mode. When NCCL is missing on a multi-GPU build, you'll see this one-time warning at startup and performance will be lower:

```
NVIDIA Collective Communications Library (NCCL) is unavailable, multi GPU performance will be suboptimal
```
---

## Limitations and gotchas

### `--split-mode tensor` (TENSOR, experimental)

This mode is gated by several hard checks at startup. Hitting any of them is a fatal error.

1. **Flash attention is required.** `--flash-attn off` will fail with *"SPLIT_MODE_TENSOR requires flash_attn to be enabled"*. Use `-fa on` (or leave it `auto`, which auto-enables for this mode).
2. **KV cache must not be quantized.** `--cache-type-k` and `--cache-type-v` must be `f32`, `f16`, or `bf16`. Anything else fails with *"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented"*.
3. **`--fit` is not supported.** Set `-fit off` and size things yourself. The auxiliary `--fit-target` / `--fit-ctx` flags do not make fitting work in this mode either.
4. **At least one non-CPU device required.** `--device none` is rejected with `--split-mode tensor`.
5. **Architecture must be supported.** TENSOR mode is implemented for most architectures, but a number are explicitly excluded today and will fail with *"LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '...'"*. The current exclusion list includes:

   - **MoE / hybrid:** Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H, Nemotron-H-MoE, Granite-Hybrid, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1
   - **State-space / RWKV-style:** Mamba, Mamba2 (and the hybrid Mamba-attention models above)
   - **Other:** PLAMO2, MiniCPM3, Gemma-3n, OLMo2, BitNet, T5

   The list grows over time - if you hit the error, fall back to `--split-mode layer`.
6. **CPU devices are skipped.** TENSOR mode wraps the selected GPUs into a single "meta device"; CPU-typed devices are excluded automatically. The selected device list and `--tensor-split` list must also stay within the `GGML_CUDA_MAX_DEVICES` cap of 16.

### Backends

- **CUDA** - fullest support: `none`, `layer`, and `tensor` all work.
- **HIP / ROCm** - same modes as CUDA, but cross-GPU reduction performance benefits from RCCL (the AMD analog of NCCL); RCCL is opt-in at build time.
- **SYCL** - supports `none` and `layer`.
- **Metal, Vulkan, CANN, etc.** - each backend exposes its own set of devices; multi-device support varies. `layer` is the most portable choice across backends.

---

## Handling OOM with `--fit off`

`--fit off` (required for `--split-mode tensor`) disables the auto-fitter, so llama.cpp will not shrink anything to make the model fit. If you OOM at startup or during inference, you have to reduce memory pressure yourself. The knobs below are listed roughly from least to most disruptive - try them in order.

1. **Lower `--ctx-size`.** The KV cache is roughly proportional to `n_ctx`. Halving `-c` halves the KV cache. Pick a context size you actually need rather than the model maximum.

2. **For `llama-server`: lower `--parallel`.** `-np N` allocates a slot KV cache for every concurrent sequence; total KV memory ≈ `n_ctx × n_parallel`. The default is often higher than you need.

3. **Reduce `--n-gpu-layers`.** Last performance-preserving knob: offload fewer layers (e.g. `-ngl 30` instead of `all`). The remaining layers run on CPU and inference will be **much slower** - only do this when no other knob suffices.

---

## Diagnosing problems

| Symptom | Likely cause |
|---|---|
| Startup error *"SPLIT_MODE_TENSOR requires flash_attn to be enabled"* | Add `-fa on` or remove `-fa off`. |
| Startup error *"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented"* | Use `-ctk f16 -ctv f16` (or `bf16`/`f32`) with `--split-mode tensor`. |
| Startup error *"LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'X'"* | Architecture not on the TENSOR allow-list. Use `--split-mode layer`. |
| Warning *"NCCL is unavailable, multi GPU performance will be suboptimal"* | llama.cpp wasn't built with NCCL. Either accept the lower performance or rebuild with `-DGGML_CUDA_NCCL=ON`. |
| CUDA OOM at startup or during prefill in `--split-mode tensor` | Auto-fit is disabled in this mode. See "Handling OOM with `--fit off`" above for the order in which to lower `-c`, `-np`, `-ngl`, etc. |
| Performance is worse with multi-GPU than single-GPU | Inter-GPU bandwidth is the bottleneck. Try `--split-mode layer` (less communication than `tensor`), and verify that NCCL is being used. |
| GPU not used at all | `--n-gpu-layers` is `0` or too low - set `-ngl all`. Or your build doesn't include the relevant backend. |
