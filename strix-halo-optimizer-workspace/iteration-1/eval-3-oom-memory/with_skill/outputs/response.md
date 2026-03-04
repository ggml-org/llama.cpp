# Why llama.cpp OOMs on Your Ryzen AI Max 395 Despite 128GB RAM

You're running into two separate but related issues that are both very common on Strix Halo hardware. The good news is both have clear fixes.

---

## Issue 1: `rocm-smi` Showing Only 4GB — This Is Normal and Not the Real Problem

The Ryzen AI Max 395 uses **unified memory** — there is no dedicated VRAM. The CPU and GPU share the same 128GB LPDDR5X pool. Tools like `rocm-smi` report the small "dedicated" chunk configured in BIOS firmware, which is typically 4GB or less. That number is essentially meaningless for actual inference capacity.

You can verify this:
```bash
rocm-smi --showmeminfo vram
cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null
```

The low number from `rocm-smi` is not why you're OOMing. The real cause is below.

---

## Issue 2: Linux TTM Memory Limits — This Is Why You're OOMing

The Linux kernel's **TTM (Translation Table Maps)** subsystem manages GPU memory mappings. Its default page limits are very conservative and were designed for systems with small discrete GPUs — they don't account for APUs with 128GB unified memory pools.

By default, TTM may only allow the GPU to access 4–8GB of system RAM, regardless of how much physical RAM you have. Attempting to load a 65B model — which requires 35–40GB at Q4 quantization — hits this wall and causes an out-of-memory error even though `free -h` shows plenty of available RAM.

### The Fix

Create the following file (requires root):

```
/etc/modprobe.d/increase_amd_memory.conf
```

With this content:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Then apply and reboot:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

With 4KB pages, `25600000 * 4096 = ~100GB` of GPU-accessible memory — enough for 70B+ models at Q4 quantization.

**Verify the fix after reboot:**
```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

**Temporary fix (no reboot required, but resets on next boot):**
```bash
echo 25600000 | sudo tee /sys/module/ttm/parameters/pages_limit
```

---

## After the TTM Fix: Correct Runtime Flags for Large Models

Once TTM is configured, you'll want these flags to reliably load a 65B model:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-cli -m your-65b-model.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 \
  -t 4
```

Key flags explained:

- **`--no-mmap`**: Critical for models larger than ~64GB. Without this, HIP's `hipMemcpy()` must lock and unlock individual memory-mapped pages, causing extreme slowdowns or failures during model loading. Always use `--no-mmap` for large models.
- **`-ngl 99`**: Offload all layers to the iGPU. On Strix Halo, unified memory means there's no PCIe transfer penalty — this is essentially free and gives you faster inference.
- **`-ctk q8_0 -ctv q8_0`**: Quantize the KV cache to reduce memory footprint. For a 65B model you'll want every bit of headroom.
- **`-t 4`**: Keep CPU thread count low. CPU and GPU share the same ~256 GB/s memory bus, so high CPU thread counts compete with GPU for bandwidth during inference.
- **`ROCBLAS_USE_HIPBLASLT=1`**: Fixes a separate issue where default rocBLAS kernels are 2–3x slower than needed for prompt processing on gfx1150/gfx1151.

---

## Summary of What's Happening

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM despite 128GB free RAM | Linux TTM pages_limit too low | Increase TTM limits in modprobe.d, reboot |
| `rocm-smi` shows 4GB VRAM | Strix Halo has no dedicated VRAM; reports BIOS firmware allocation | Expected behavior — ignore this number |
| Slow/hanging model load (>64GB) | HIP + mmap page locking overhead | Use `--no-mmap` flag |

The OOM is entirely a kernel configuration issue, not a hardware limitation. Your 128GB is more than sufficient to run a 65B model once TTM is configured correctly.
