# Why llama.cpp Runs Out of Memory on Ryzen AI Max 395 Despite 128GB RAM

## The Core Problem

The Ryzen AI Max 395 is an APU (Accelerated Processing Unit) with a **unified memory architecture (UMA)**. The GPU (Radeon 8060S) does not have its own dedicated VRAM -- it shares your 128GB of system RAM with the CPU. However, `rocm-smi` only reports a small **carve-out** of memory (typically 4GB) that the BIOS has pre-allocated as "VRAM" for the GPU. This is the root of your problem: llama.cpp sees only ~4GB of GPU memory and cannot fit a 65B model into that.

## Why This Happens in llama.cpp

Looking at the llama.cpp source code (`ggml/src/ggml-cuda/ggml-cuda.cu`), there are two interrelated issues:

### 1. The `integrated` GPU flag is hardcoded to `false`

At line 233 of `ggml-cuda.cu`:

```cpp
info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
```

Even though the ROCm/HIP runtime correctly reports `prop.integrated > 0` for your APU, llama.cpp forcibly overrides this to `false` due to a corruption bug reported in issue #15034. This means llama.cpp does not recognize your device as an integrated GPU and will not enable any of the UMA-aware memory handling logic.

### 2. Memory reporting falls back to `hipMemGetInfo`

The function `ggml_backend_cuda_device_get_memory()` (line 4420) has UMA-aware code that reads `/proc/meminfo` to determine the true available system memory. However, this code path is only activated when `prop.integrated > 0` OR when the environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is set. Since `integrated` is hardcoded to `false`, the function falls through to `hipMemGetInfo`, which returns only the small BIOS-allocated VRAM carve-out (~4GB).

### 3. Memory allocation uses `hipMalloc` instead of `hipMallocManaged`

The allocation function `ggml_cuda_device_malloc()` (line 119) only uses `cudaMallocManaged` (mapped to `hipMallocManaged` on ROCm) when the environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is set. Without it, allocations go through `hipMalloc`, which can only allocate from the small VRAM carve-out.

## The Fix: Set the Unified Memory Environment Variable

Set the `GGML_CUDA_ENABLE_UNIFIED_MEMORY` environment variable before running llama.cpp:

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
```

This does two things:
1. **Switches memory allocation to `hipMallocManaged`**, which allows the GPU to allocate from the full shared system memory pool (your 128GB) rather than the tiny VRAM carve-out. On HIP/ROCm, it also sets `hipMemAdviseSetCoarseGrain` for performance.
2. **Enables the UMA memory reporting path**, so llama.cpp reads `/proc/meminfo` to determine how much memory is actually available rather than relying on the misleading `hipMemGetInfo` result.

## Additional Recommendations

### Increase the BIOS VRAM Allocation

Even with unified memory enabled, increasing the BIOS VRAM allocation can help. Many ASUS and Lenovo systems with Ryzen AI Max chips allow setting the GPU memory allocation up to 96GB in BIOS settings (look for "UMA Frame Buffer Size" or "VRAM Size" in BIOS). This is not strictly required when using `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, but it may improve behavior with certain driver configurations.

### Use Enough GPU Layers

With unified memory enabled, you can offload all layers to the GPU:

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-cli \
  -m your-65B-model.gguf \
  -ngl 999
```

The `-ngl 999` (or `-ngl all`) tells llama.cpp to offload all layers to the GPU. Since CPU and GPU share the same physical memory on your APU, this does not actually double memory usage -- it just determines which processor computes each layer.

### Use a Quantized Model

A 65B model in FP16 requires ~130GB, which exceeds your 128GB even before accounting for OS and KV cache overhead. Use a quantized version:

- **Q4_K_M**: ~35GB -- fits comfortably, good quality
- **Q5_K_M**: ~44GB -- fits well, better quality
- **Q6_K**: ~53GB -- fits with room for KV cache
- **Q8_0**: ~65GB -- fits but leaves less room for context

### Consider Vulkan as an Alternative Backend

If you encounter driver issues with ROCm on the Ryzen AI Max (ROCm support for Strix Halo APUs has historically been spotty), the Vulkan backend is an alternative. Vulkan can use the `--vk-mem-budget` flag to specify how much memory the GPU backend should use, sidestepping the VRAM reporting issue entirely:

```bash
llama-cli -m your-model.gguf -ngl 999 --vk-mem-budget 96000
```

## Summary

| Problem | Cause | Solution |
|---------|-------|----------|
| `rocm-smi` shows 4GB VRAM | BIOS default carve-out for iGPU | Increase in BIOS or use unified memory |
| llama.cpp OOM on load | `hipMalloc` limited to VRAM carve-out | `export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` |
| llama.cpp does not detect APU | `integrated` flag hardcoded to `false` | Same env var works around this |
| 65B FP16 too large for 128GB | Model exceeds RAM after overhead | Use Q4_K_M or Q5_K_M quantization |
