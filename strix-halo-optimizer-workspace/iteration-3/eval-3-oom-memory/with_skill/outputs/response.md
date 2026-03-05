# OOM Loading 65B Model on Ryzen AI Max 395 with 128GB RAM

## What's Going On

You're hitting two separate but related issues that are both well-known on Strix Halo systems.

### 1. rocm-smi Showing Only 4GB VRAM Is Normal

The Ryzen AI Max 395 has **no dedicated VRAM**. It uses unified memory -- the CPU and GPU share your 128GB of LPDDR5X. The 4GB figure that `rocm-smi` reports is just the BIOS-configured "dedicated" portion, not the actual GPU-accessible memory. This is a reporting artifact, not a real limitation.

You can verify what the GPU actually sees:

```bash
rocm-smi --showmeminfo vram
cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null
```

### 2. The OOM Is Caused by Linux TTM Limits (This Is the Real Problem)

The Linux kernel's TTM (Translation Table Maps) subsystem controls how much system memory the GPU is allowed to access. The **default limits are extremely conservative** and don't account for APUs with large unified memory pools like Strix Halo. With default TTM settings, the GPU can only use a small fraction of your 128GB, so loading a 65B model triggers an OOM even though `free -h` shows plenty of available RAM.

## The Fix

Create the file `/etc/modprobe.d/increase_amd_memory.conf` with:

```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Then apply and reboot:

```bash
sudo update-initramfs -u -k all
sudo reboot
```

With 4KB pages, this sets the GPU-accessible memory pool to approximately 100GB, which is more than enough for a 65B model in Q4 quantization.

**If you want a quick test without rebooting**, you can temporarily increase the limit:

```bash
echo 25600000 | sudo tee /sys/module/ttm/parameters/pages_limit
```

### Verify the Fix

After reboot (or the temporary fix), confirm the new limit is active:

```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

## Additional Recommendations for Loading Your 65B Model

Once TTM is fixed, use these flags for best results:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-cli -m your-65b-model.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -t 4
```

Key points:

- **`--no-mmap`**: Critical for models this large. Without it, HIP's `hipMemcpy()` has to lock/unlock individual mmap pages, causing extremely slow model loading or hangs for models over ~64GB.
- **`-ngl 99`**: Offload all layers to GPU. On unified memory there is no PCIe transfer penalty, so full offload is optimal.
- **`ROCBLAS_USE_HIPBLASLT=1`**: Fixes a ~2.5x prompt processing speed penalty caused by poorly optimized default rocBLAS kernels for gfx1150.
- **`-t 4`**: Keep CPU threads low during GPU inference since CPU and GPU share the ~256 GB/s memory bus.
- **`-ctk q8_0 -ctv q8_0`**: Quantized KV cache reduces memory footprint, giving you more headroom for larger contexts.

## Summary

| Observation | Explanation |
|---|---|
| rocm-smi shows 4GB VRAM | Normal -- this is the BIOS-reported "dedicated" portion; actual GPU-accessible memory is the full unified pool |
| OOM loading 65B model | Linux TTM limits restrict GPU memory access; increase `pages_limit` to 25600000 |
| Model loads very slowly after fixing OOM | Use `--no-mmap` flag to avoid HIP page-locking overhead on large models |
