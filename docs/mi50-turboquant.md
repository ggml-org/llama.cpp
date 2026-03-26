# MI50 TurboQuant Validation

This tree carries the milestone-1 `TQ3_0` bring-up for ROCm on `gfx906`.

## Runtime environment

Use this environment for MI50 validation:

```bash
export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCBLAS_LAYER=0
export ROCBLAS_LOG_LEVEL=0
export HIPBLASLT_LOG_LEVEL=0
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0
export GPU_SINGLE_ALLOC_PERCENT=100
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
```

## Build

```bash
cmake -S . -B build-mi50 \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-mi50 --config Release -j"$(nproc)"
```

## Smoke tests

Baseline:

```bash
./build-mi50/bin/llama-cli \
  -m /home/stefan/.lmstudio/models/mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF/Huihui-Qwen3.5-27B-abliterated.Q4_K_S.gguf \
  -ngl 999 \
  --flash-attn off \
  --cache-type-k f16 \
  --cache-type-v f16 \
  -c 4096 -n 64 \
  -p "Write one paragraph about TurboQuant."
```

TurboQuant:

```bash
./build-mi50/bin/llama-cli \
  -m /home/stefan/.lmstudio/models/mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF/Huihui-Qwen3.5-27B-abliterated.Q4_K_S.gguf \
  -ngl 999 \
  --flash-attn off \
  --cache-type-k tq3_0 \
  --cache-type-v q8_0 \
  -c 4096 -n 64 \
  -p "Write one paragraph about TurboQuant."
```

Fallback V cache:

```bash
./build-mi50/bin/llama-cli \
  -m /home/stefan/.lmstudio/models/mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF/Huihui-Qwen3.5-27B-abliterated.Q4_K_S.gguf \
  -ngl 999 \
  --flash-attn off \
  --cache-type-k tq3_0 \
  --cache-type-v f16 \
  -c 4096 -n 64 \
  -p "Write one paragraph about TurboQuant."
```

## Notes

- `GGML_OP_SOLVE_TRI` is intentionally not offloaded on `gfx906`.
- FlashAttention is intentionally disabled when `K=tq3_0`.
- Start at `4k`, then move to `16k`, then raise context only after a clean run.
