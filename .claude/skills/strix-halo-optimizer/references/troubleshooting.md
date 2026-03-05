# Troubleshooting: llama.cpp on AMD Strix Halo

Common issues and solutions sourced from real GitHub issues and community experience.

## Table of Contents
- [OOM Despite Free RAM](#oom-despite-free-ram)
- [Slow Prompt Processing (2-3x Slower Than Expected)](#slow-prompt-processing)
- [Model Loading Hangs or Extremely Slow (>64GB Models)](#model-loading-hangs)
- [No HIP Devices Found](#no-hip-devices-found)
- [ROCm Performance Regression](#rocm-performance-regression)
- [Vulkan Backend Failures](#vulkan-backend-failures)
- [Slow Compilation](#slow-compilation)
- [HSA_OVERRIDE_GFX_VERSION Confusion](#hsa-override-confusion)
- [Unified Memory and VRAM Reporting](#unified-memory-reporting)

---

## OOM Despite Free RAM

**Symptoms**: Out-of-memory errors when loading models, even though `free -h` shows plenty of available RAM. Typically affects models >8GB.

**Root Cause**: Linux kernel TTM (Translation Table Maps) limits restrict how much system memory the GPU can access. Default limits are very conservative and don't account for APUs with large unified memory pools.

**Solution**: Increase TTM limits.

Create `/etc/modprobe.d/increase_amd_memory.conf`:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Apply:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

Alternatively, for a temporary fix without reboot:
```bash
echo 25600000 | sudo tee /sys/module/ttm/parameters/pages_limit
```

**Verification**:
```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

**Source**: [Lychee-Technology/llama-cpp-for-strix-halo](https://github.com/Lychee-Technology/llama-cpp-for-strix-halo), [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)

---

## Slow Prompt Processing

**Symptoms**: Prompt processing (pp) speed is 2-3x slower than expected. For example, a 7B Q4_0 model gets ~350 t/s instead of ~880 t/s on pp512.

**Root Cause**: Default rocBLAS kernels for gfx1150/gfx1151 are poorly optimized. The standard Tensile-generated kernels underperform significantly compared to the hipBLASLt (TensileLt) alternatives.

**Solution**: Set the hipBLASLt environment variable:
```bash
export ROCBLAS_USE_HIPBLASLT=1
```

Add this to your `.bashrc` or shell profile for persistence.

**Impact**:
- pp512 (7B Q4_0): 348 t/s → 882 t/s (~2.5x improvement)
- Token generation: Unchanged (~48 t/s)

**Source**: [Issue #13565](https://github.com/ggml-org/llama.cpp/issues/13565)

---

## Model Loading Hangs

**Symptoms**: Loading models larger than ~64GB takes extremely long (minutes to hours) or appears to hang. The process uses high CPU but barely any GPU activity. Vulkan loads the same model fine.

**Root Cause**: When memory mapping (mmap) is enabled, HIP's `hipMemcpy()` must lock and unlock individual memory pages. For large models that span many pages, this creates severe overhead. The issue is specific to HIP's interaction with mmap'd memory regions.

**Solution**: Disable memory mapping:
```bash
./build/bin/llama-cli -m model.gguf --no-mmap -ngl 99
```

For `llama-bench`:
```bash
./build/bin/llama-bench -m model.gguf --no-mmap -ngl 99
```

**Note**: This increases initial memory usage since the model is loaded directly into RAM rather than memory-mapped. For 128GB Strix Halo systems, this is usually not an issue.

**Source**: [Issue #15018](https://github.com/ggml-org/llama.cpp/issues/15018)

---

## No HIP Devices Found

**Symptoms**: llama.cpp doesn't detect the GPU, falls back to CPU-only inference.

**Diagnosis**:
```bash
# Check ROCm installation
rocminfo | grep -i "Name:"
# Should show: gfx1150 or gfx1151

# Check HIP runtime
hipinfo 2>/dev/null || /opt/rocm/bin/hipconfig --check
```

**Common Causes and Fixes**:

1. **ROCm not installed or too old**: Requires ROCm 6.1+
   ```bash
   apt list --installed 2>/dev/null | grep rocm
   # Or
   /opt/rocm/bin/rocm-smi --version
   ```

2. **User not in video/render groups**:
   ```bash
   sudo usermod -a -G video,render $USER
   # Log out and back in
   ```

3. **ROCM_PATH not set** (if ROCm is in non-standard location):
   ```bash
   export ROCM_PATH=/opt/rocm
   ```

4. **Built without HIP**: Verify the build included `GGML_HIP=ON`:
   ```bash
   ./build/bin/llama-bench --list-devices
   ```

---

## ROCm Performance Regression

**Symptoms**: Performance suddenly drops between llama.cpp versions.

**Diagnosis**:
```bash
# Compare with a known-good version
git log --oneline -20
# Identify the regression commit range

# Benchmark current version
export ROCBLAS_USE_HIPBLASLT=1
./build/bin/llama-bench -m model.gguf -ngl 99 --no-mmap -r 5

# Compare with older version (build from that commit)
```

**Known Regressions**:
- Build 4df6e85 (Dec 2025): Performance regression on `llama-server` and `llama-bench` with ROCm on Strix Halo. See [Issue #17917](https://github.com/ggml-org/llama.cpp/issues/17917).

**Workarounds**:
- Pin to a known-good commit
- Try both `GGML_CUDA_FORCE_MMQ=ON` and `OFF` — the optimal setting can change between versions
- Report the regression with benchmark data on GitHub

---

## Vulkan Backend Failures

**Symptoms**: "failed to load model" error when using Vulkan backend on Strix Halo.

**Status**: Known issue. See [Issue #18741](https://github.com/ggml-org/llama.cpp/issues/18741).

**Workaround**: Use the ROCm/HIP backend instead:
```bash
# Build with HIP (not Vulkan)
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS="gfx1150" -DCMAKE_BUILD_TYPE=Release
```

The ROCm/HIP backend is better optimized for Strix Halo and generally provides equal or better performance than Vulkan on this hardware.

---

## Slow Compilation

**Symptoms**: Build takes 30+ minutes.

**Cause**: Building for multiple GPU architectures when only gfx1150 is needed.

**Solution**: Target only your architecture:
```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j $(nproc)
```

If unsure of your exact architecture:
```bash
rocminfo | grep -i "Name:" | grep gfx
```

---

## HSA Override Confusion

**Symptoms**: Older guides suggest using `HSA_OVERRIDE_GFX_VERSION` to make Strix Halo pretend to be a different GPU. This was necessary before proper gfx1150/gfx1151 support was added.

**Current Status**: llama.cpp now has native RDNA 3.5 support. **Do not use `HSA_OVERRIDE_GFX_VERSION`** — it forces generic kernels that are slower than the native gfx1150 code paths.

If you previously set this, remove it:
```bash
# Check if it's set
echo $HSA_OVERRIDE_GFX_VERSION

# Remove from .bashrc if present
grep -n HSA_OVERRIDE_GFX_VERSION ~/.bashrc
```

---

## Unified Memory Reporting

**Symptoms**: `rocm-smi` or other tools show very little VRAM (e.g., 512MB or 4GB) despite having 128GB unified memory.

**Explanation**: Strix Halo uses unified memory — there's no dedicated VRAM. Tools report only the BIOS-configured "dedicated" portion. The actual GPU-accessible memory is the full unified pool (after TTM configuration).

**Verification**:
```bash
# Check what ROCm sees
rocm-smi --showmeminfo vram

# Check total system memory available to GPU
cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null

# Actual usable memory depends on TTM pages_limit
cat /sys/module/ttm/parameters/pages_limit
# Multiply by 4096 (page size) to get bytes
```

**Practical test**: Try loading a model larger than the reported VRAM. If TTM is configured correctly, it will work:
```bash
./build/bin/llama-cli -m large_model.gguf -ngl 99 --no-mmap
```
