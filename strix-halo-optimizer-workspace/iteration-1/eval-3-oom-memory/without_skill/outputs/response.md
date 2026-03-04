# llama.cpp Out of Memory on Ryzen AI Max 395 with 128GB RAM

## What Is Happening

The Ryzen AI Max 395 (Strix Halo) uses a **unified memory architecture (UMA)**. There is no discrete VRAM — the CPU and integrated GPU share the same physical 128GB LPDDR5X memory pool. However, by default, the system firmware (BIOS/UEFI) reserves only a small fixed slice of that unified memory and exposes it to the OS as dedicated GPU memory. This is what `rocm-smi` reports: the BIOS-reserved segment, commonly defaulting to 4GB or 8GB depending on your OEM/firmware settings.

llama.cpp's HIP/ROCm backend allocates GPU buffers from this BIOS-reserved segment, not from the full system RAM, unless told otherwise. When you try to load a 65B model (which requires roughly 35–130GB depending on quantization), you immediately exhaust the 4GB reservation and get an OOM error even though 128GB of physical RAM is physically present and largely idle.

---

## Root Causes

### 1. BIOS/UEFI UMA Frame Buffer Size Setting

The firmware exposes only a fraction of the unified memory pool as "VRAM" to the GPU driver. The rest is managed by the CPU memory controller and is not automatically available to GPU allocators.

**Fix:** Enter your BIOS/UEFI settings and look for an option like:
- "UMA Frame Buffer Size"
- "iGPU Memory"
- "Integrated Graphics Shared Memory"
- "VRAM Size" or "GPU Memory Allocation"

On Strix Halo systems this option often supports values up to 64GB. Set it to the largest value your BIOS allows (e.g., 16GB, 32GB, or 64GB). After saving and rebooting, `rocm-smi` should reflect the new size.

> Note: Increasing this value does not actually reserve that RAM exclusively for the GPU at boot time on all implementations — on true UMA systems it mainly tells the driver how large the GPU-addressable window is. The behavior is firmware-dependent.

### 2. ROCm/HIP Driver and the `HSA_OVERRIDE_GFX_VERSION` or Memory Visibility Issue

ROCm on integrated AMD GPUs may not automatically expose the full unified memory pool as GPU-accessible. The driver stack may treat it as a split system where GPU memory is only what the firmware declared.

### 3. llama.cpp GPU Offload Configuration

Even if the GPU memory window is correctly sized, llama.cpp needs to be told to offload layers to the GPU. If the model loads entirely on CPU, it uses system RAM through the CPU allocator (not ROCm), which should work — but performance will be poor. If it is trying to use the GPU (`-ngl` flag) and only 4GB is visible, it will OOM.

---

## Solutions

### Solution A: Increase UMA Frame Buffer Size in BIOS (Primary Fix)

1. Reboot into BIOS/UEFI (typically Delete, F2, or F10 at POST).
2. Navigate to Advanced > AMD CBS > GFX Configuration (exact path varies by OEM/motherboard).
3. Find "UMA Frame Buffer Size" and increase it (try 16GB or 32GB to start).
4. Save and reboot.
5. Verify with `rocm-smi` — it should now report the larger size.

### Solution B: Run llama.cpp on CPU Only (No GPU Offload)

If you do not need GPU acceleration and just want to load the model:

```bash
./llama-cli -m your-65b-model.gguf -ngl 0 -c 2048 -n 128 -p "Your prompt here"
```

Setting `-ngl 0` disables GPU layer offloading entirely. llama.cpp will load the model into system RAM via the CPU path, which can access all 128GB. A Q4_K_M quantized 65B model is approximately 35–40GB, which fits comfortably. Inference will be slower than GPU-accelerated, but it will work.

### Solution C: Use Memory Mapping to Reduce RAM Pressure

llama.cpp uses `mmap` by default for GGUF files, which means it does not load the entire model into RAM immediately — pages are loaded on demand. Ensure you are not disabling this:

```bash
# Good: mmap is on by default, do not pass --no-mmap unless you have a reason
./llama-cli -m your-65b-model.gguf -ngl 0 ...
```

### Solution D: Partial GPU Offload

After increasing your UMA Frame Buffer Size in BIOS (Solution A), you can offload only as many layers as fit in the GPU-visible memory and run the rest on CPU:

```bash
# For a 65B model, experiment with the number of layers (e.g., 20 out of ~80)
./llama-cli -m your-65b-model.gguf -ngl 20 -c 2048 -p "Your prompt here"
```

Increase `-ngl` incrementally until you find the maximum that does not OOM.

### Solution E: Set ROCm Environment Variables for Unified Memory

Some ROCm versions support flags that allow the GPU to access all system memory. Try:

```bash
export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0
# On some systems this helps expose more memory:
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
```

Then launch llama.cpp. These variables instruct the ROCm runtime to be more aggressive about allocating from the full memory pool. Results vary by ROCm version and firmware.

---

## Recommended Approach for Your Setup

1. **First**, go into BIOS and increase UMA Frame Buffer Size to 32GB or 64GB.
2. **Verify** with `rocm-smi` that the new size is visible.
3. **Run** llama.cpp with a Q4_K_M or Q5_K_M quantized 65B model and use `-ngl` to offload as many layers as fit.
4. **If BIOS does not have the option**, fall back to `-ngl 0` and run CPU-only — your 128GB RAM is more than sufficient for any quantized 65B model.

---

## Summary Table

| Scenario | What to Do |
|---|---|
| BIOS shows UMA Frame Buffer option | Increase to 32GB or 64GB, then use `-ngl` |
| No BIOS option available | Use `-ngl 0` for CPU-only inference |
| Want partial GPU acceleration | Set BIOS UMA size, then tune `-ngl` gradually |
| ROCm still sees wrong size after BIOS change | Set `GPU_MAX_HEAP_SIZE=100` env vars |

The core issue is a firmware/driver boundary: your physical hardware has 128GB of unified memory, but the ROCm stack only sees what the BIOS declared as GPU memory (4GB by default). Fixing the BIOS setting is the cleanest and most effective solution.
