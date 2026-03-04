# Diagnosing Slow Prompt Processing on Strix Halo (AMD ROCm/HIP)

## Summary of the Problem

You are seeing ~350 tokens/sec for prompt processing (PP) but ~48 t/s for token generation (TG) on a 7B model. This is a classic symptom of a **memory bandwidth / batch size mismatch**, but on a Strix Halo system the most likely culprits are specific to the AMD integrated GPU architecture and ROCm/HIP configuration.

A healthy PP/TG ratio for a 7B model running on an iGPU should be roughly 10-30x TG speed (so 500-1500+ t/s PP would be more typical). 350 t/s PP with 48 t/s TG is suspiciously low, suggesting the GPU is not batching the prompt efficiently.

---

## Most Likely Causes

### 1. Unified Memory Bandwidth Bottleneck (Most Likely)

Strix Halo uses AMD's integrated GPU sharing system memory. The key issue is that **the memory bandwidth available to the iGPU is shared with the CPU and depends on the memory subsystem configuration**.

- Strix Halo has a large 256-bit LPDDR5X memory interface, which provides significant bandwidth — but it must be shared between CPU and GPU.
- If the system is not fully utilizing the memory channels, or if the OS/BIOS has constrained VRAM allocation, PP speed will suffer dramatically.
- Check how much VRAM has been allocated to the iGPU in your BIOS/UEFI (GTT/UMA buffer size). For a Strix Halo system running large models, you want this set to **at least 16GB or more** if the option exists, or rely on ROCm's unified memory.

### 2. Small `-ub` (Ubatch Size) Parameter

This is the **single most common cause** of low PP speed in llama.cpp.

- `--ubatch-size` (or `-ub`) controls how many tokens are processed in a single GPU kernel launch during prompt processing.
- The default is often 512, but for iGPU systems, the optimal value may differ.
- If the ubatch size is too small, the GPU is underutilized during PP. If it is too large, you may hit memory allocation issues.

Try increasing the ubatch size:
```bash
llama-cli ... -ub 1024
# or even
llama-cli ... -ub 2048
```

Also check `-b` (batch size), which should be at least as large as `-ub`.

### 3. ROCm Not Using the Correct GPU / Falling Back to CPU for Attention

On Strix Halo, the integrated GPU is an RDNA 3.5 architecture. Verify that llama.cpp is actually offloading all layers to the GPU:

```bash
llama-cli ... -ngl 999
```

If layers are split between CPU and GPU, prompt processing will be dramatically slower because the attention computation (which is compute-intensive during PP) may run on CPU.

Check the startup logs for lines like:
```
llm_load_tensors: offloading XX repeating layers to GPU
```

All layers should be offloaded. If you see fewer layers offloaded than the model has, you are hitting a VRAM limit.

### 4. ROCm Target Architecture Mismatch

If llama.cpp was compiled without specifying `gfx1151` (Strix Halo's GPU architecture), ROCm may be JIT-compiling kernels at runtime or falling back to a generic path.

Check what target was used:
```bash
# If you compiled from source, ensure you set:
AMDGPU_TARGETS=gfx1151 cmake ...
# or for older build systems:
HSA_OVERRIDE_GFX_VERSION=11.5.1
```

If you are using a pre-built binary, it may not have been compiled for your exact architecture. The Strix Halo iGPU is `gfx1151` (or `gfx1150` depending on the variant). Setting the environment variable at runtime can help:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

### 5. Flash Attention Disabled

Flash attention significantly improves prompt processing speed by reducing memory transfers during the attention calculation. Ensure it is enabled:

```bash
llama-cli ... --flash-attn
# or the short form:
llama-cli ... -fa
```

Without flash attention, PP speed can be 2-5x slower on GPU, especially for longer prompts or larger context windows.

### 6. Context Size Too Large

If you have set a very large context size (`-c`), the KV cache occupies more VRAM, and attention computation during PP requires more memory bandwidth.

- For a 7B model at 350 PP t/s, if you are using a context of 8192 or larger, try reducing to 4096 and see if PP improves.
- Also consider using a quantized KV cache: `--cache-type-k q8_0 --cache-type-v q8_0` or even `q4_0`.

### 7. GPU Power / Thermal Throttling

Laptop iGPUs are constrained by TDP. If the system is thermal throttling or operating in a low-power mode:

- PP is more compute-intensive than TG (which is memory-bandwidth-bound), so throttling affects PP more.
- Check GPU frequency under load: `rocm-smi` or `amdgpu_top`.
- Ensure the laptop is plugged in and set to a high-performance power profile.

```bash
rocm-smi --showclocks
amdgpu_top  # if installed
```

### 8. ROCm Sparse Attention / Kernel Selection

For some ROCm versions, the attention kernels used during PP are not optimally selected for integrated GPU configurations. This can be version-dependent.

- Try updating to the latest ROCm version (6.3+ recommended for Strix Halo).
- Check if `ROCBLAS_GEMM_TYPE` or similar environment variables can help select better kernels.

---

## Recommended Debugging Steps (in order)

1. **Confirm all layers are on GPU**: Check startup logs for layer offload count.
2. **Enable flash attention**: Add `-fa` and retest.
3. **Increase ubatch size**: Try `-ub 1024` and `-ub 2048`.
4. **Set architecture override**: `export HSA_OVERRIDE_GFX_VERSION=11.5.1` before running.
5. **Check VRAM allocation**: In BIOS, ensure adequate UMA/GTT buffer is allocated (16GB+).
6. **Check thermals and power**: Use `rocm-smi` to verify clocks are not throttled.
7. **Reduce context size**: Try `-c 2048` or `-c 4096` to isolate context impact.
8. **Try quantized KV cache**: Add `--cache-type-k q8_0 --cache-type-v q8_0`.

---

## Example Command with All Optimizations Applied

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1

llama-cli \
  -m /path/to/model-7b-q4_k_m.gguf \
  -ngl 999 \
  -fa \
  -b 2048 \
  -ub 1024 \
  -c 4096 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -p "Your prompt here"
```

---

## Expected Results After Fixes

On a Strix Halo system with properly configured ROCm and the above settings, you should expect:

- **PP speed**: 1500-4000+ t/s for a 7B model (depending on quantization and context length)
- **TG speed**: 40-60 t/s (this is memory-bandwidth-bound and harder to improve)

Your current 350 PP t/s with 48 TG t/s suggests the GPU is being severely underutilized during prompt processing. The flash attention flag and ubatch size are the first things to try.
