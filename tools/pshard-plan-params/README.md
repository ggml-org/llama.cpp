# pshard-plan-params

`llama-pshard-plan-params` writes a pshard registry for a model and VRAM budget.

It probes batch-size tiers such as `bs=1`, `bs=16`, and `bs=512`, compares the available placement strategies, and writes the selected tensor overrides to:

```text
<model>.gguf.tensor_overrides.pshard_registry
```

When CPU/GPU profiler results are available, the planner uses them to estimate throughput for each candidate plan. At runtime, pshard selects the smallest tier that covers the current batch.

The registry is plain text so it can be inspected when debugging planner output.

## Example usage

```bash
# 1. Generate roofline profiles once per machine
#    (writes profile JSONs next to the binary)
./build/bin/llama-profiler-cpu
./build/bin/llama-profiler-gpu

# 2. Plan placement for a model + VRAM budget
./build/bin/llama-pshard-plan-params \
    --model /opt/models/Qwen3.5-27B-Q4_0.gguf \
    --max-vram-alloc 12000 \
    -c 8192 \
    -fa on

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):
  Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB
main: planning pshard tensor overrides...
llama_params_fit_pshard: probing pshard plans: 64 layers, 12000.0 MiB VRAM free
pshard_registry_save: saved budget=12000 MiB cache_ubatch=8192 variant with 7 tier plans to /opt/models/Qwen3.5-27B-Q4_0.gguf.tensor_overrides.pshard_registry
main: planning complete, registry written next to model file

# 3. Inspect the registry
cat /opt/models/Qwen3.5-27B-Q4_0.gguf.tensor_overrides.pshard_registry
```

Example registry output:

```text
# Generated file. Edit at your own risk.

[fingerprint=0x9041bb5b253cf89f]
# n_ctx=8192 n_seq_max=1 n_threads=8 fa=on type_k=1 type_v=1 strategy=auto

[variant budget=12000 cache_ubatch=8192]
[tier 0 bs=1]
strategy=DYNAMIC_FFNCPU_ATTNSTREAM n_pinned=54 n_attn_pinned=0 overflow=NONE tps=12.85   vram=11975.7 output_on_gpu=0 pin_from_back=0
ot=^output=CUDA_Host:3,^token_embd=CUDA_Host:3,blk\.0\..*=CUDA_Host:0, ... ,blk\.53\..*=CUDA_Host:0,blk\.54\.ffn_(up|gate|down).*=CUDA_Host:3,blk\.54\..*=CUDA_Host:1,blk\.55\.ffn_(up|gate|down).*=CUDA_Host:3,blk\.55\..*=CUDA_Host:2, ... ,blk\.63\.ffn_(up|gate|down).*=CUDA_Host:3,blk\.63\..*=CUDA_Host:2
[tier 1 bs=16]
strategy=GPUONLY_ATTNPIN_FFNSTREAM n_pinned=48 n_attn_pinned=0 overflow=NONE tps=208.27  vram=11964.1 output_on_gpu=0 pin_from_back=0
ot=...
[tier 2 bs=512]
strategy=GPUONLY_LAYERPIN_LAYERSTREAM n_pinned=52 n_attn_pinned=0 overflow=NONE tps=1994.90 vram=11973.3 output_on_gpu=0 pin_from_back=0
ot=...
[tier 3 bs=1024]
strategy=GPUONLY_LAYERPIN_LAYERSTREAM n_pinned=51 n_attn_pinned=0 overflow=NONE tps=2286.87 vram=11909.8 output_on_gpu=0 pin_from_back=0
ot=...
[tier 4 bs=2048]
strategy=GPUONLY_ATTNPIN_FFNSTREAM n_pinned=44 n_attn_pinned=0 overflow=NONE tps=2249.44 vram=11914.3 output_on_gpu=0 pin_from_back=0
ot=...
[tier 5 bs=4096]
strategy=GPUONLY_ATTNPIN_FFNSTREAM n_pinned=40 n_attn_pinned=0 overflow=NONE tps=2095.88 vram=11868.6 output_on_gpu=0 pin_from_back=0
ot=...
[tier 6 bs=8192]
strategy=GPUONLY_ATTNPIN_FFNSTREAM n_pinned=33 n_attn_pinned=0 overflow=NONE tps=2148.21 vram=11937.2 output_on_gpu=0 pin_from_back=0
ot=^output=CUDA_Host:3,^token_embd=CUDA_Host:3,blk\.0\..*=CUDA_Host:0, ... ,blk\.32\..*=CUDA_Host:0,blk\.33\.ffn_(up|gate|down).*=CUDA_Host:2,blk\.33\..*=CUDA_Host:0, ... ,blk\.63\.ffn_(up|gate|down).*=CUDA_Host:2,blk\.63\..*=CUDA_Host:0
```

## Planning for llama-bench

Use `--bench-plan` when the registry is intended for `llama-bench -pshard`. The planner accepts the bench shape flags `-p` / `--n-prompt`, `-n` / `--n-gen`, `-pg`, and `-d` / `--n-depth`, then plans each unique context that llama-bench can run:

```bash
./build/bin/llama-pshard-plan-params \
    --model /opt/models/Qwen3.5-35B-A3B-Q8_0.gguf \
    --bench-plan \
    -pg 512,200 \
    -pg 2048,200 \
    -d 0,1024
```

For each generated context, `n_ctx` is the number of tokens that can be resident during the test:

- prompt-only: `n_ctx = p + d`
- generation-only: `n_ctx = n + d`
- prompt+generation: `n_ctx = p + n + d`

The largest planned tier for a bench context is capped by prompt batch demand, not by depth. For example, `-pg 2048,200 -d 1024` plans `n_ctx=3272` with `tier_cap=2048`, so the registry includes decode tiers plus prompt tiers up to `bs=2048`.

If `-fa` / `--flash-attn` is not provided and `LLAMA_ARG_FLASH_ATTN` is not set, `--bench-plan` uses Flash Attention off to match llama-bench defaults. Normal non-bench planning keeps the regular common parameter defaults.

## Strategies

In the strategy names, `ATTN` refers to the attention/dense side of the layer, as opposed to FFN/MoE weights.

Static schedules run GPU-resident tensors on GPU and CPU-resident tensors on CPU, with no streamed GPU execution for host-resident weights.

- `STATIC_ATTNPRIO_ALLMODELS`: static attention-priority placement. It pins the attention/dense side across as many layers as fit, then uses the remaining budget to pin full layers. FFN/MoE that does not fit remains on CPU. Unlike `llama_params_fit`, this attention-priority placement applies to dense models too.

Dynamic schedules split the layer between CPU and GPU execution. Some host-resident tensors are streamed to GPU scratch for execution, while other parts of the layer remain on CPU.

- `DYNAMIC_FFNCPU_ATTNSTREAM`: pin as many full layers as fit, keep FFN/MoE on CPU in the remaining layers, and stream the attention/dense side for GPU execution.

GPU-only schedules execute repeating-layer compute on GPU. Weights that do not fit in VRAM stay resident in host memory and are streamed to GPU scratch before use.

- `GPUONLY_LAYERPIN_LAYERSTREAM`: pin as many full layers as fit, then stream the remaining layers for GPU execution.
- `GPUONLY_ATTNPIN_FFNSTREAM`: pin the attention/dense side for all layers, pin as many complete layers as the remaining budget allows, and stream FFN/MoE weights for GPU execution.

## Notes

- `--max-vram-alloc` / `-mva <MiB>` sets the absolute planning budget. If it is `0` or omitted, the planner uses the free VRAM available at planning time minus `--fit-target` / `-fitt` (default 1024 MiB). When `-mva` is non-zero, `-fitt` is ignored for pshard.
- `--pshard-tier-max <N>` caps the largest batch size the planner probes. The default is `min(max(n_batch, 16384), n_ctx)`.
- The `[fingerprint=...]` line invalidates the registry when plan-compatible inputs change (`n_ctx`, `n_seq_max`, threads, FA mode, KV cache types, GGUF file size, or forced `PSHARD_STRATEGY`). The comment below the fingerprint lists those same inputs.
- A fingerprint can contain multiple `[variant budget=... cache_ubatch=...]` blocks. Re-running the planner with a new budget or cache ubatch replaces only that variant and keeps the others.
- `cache_ubatch` records the runtime ubatch used for context/KV/SWA cache sizing. Tier compute scratch is still measured with the tier batch size.
- `pshard_disabled=1 baseline_vram=<MiB>` is variant-scoped. Runtime skips pshard only when the measured baseline VRAM fits the current budget.
- `backend_id` values:
  - `0` = GPU pinned compute
  - `1` / `2` = shard compute lanes used for pipeline overlap
  - `3` = CPU
