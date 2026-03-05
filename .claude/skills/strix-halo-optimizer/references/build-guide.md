# Build Guide: llama.cpp for AMD Strix Halo

## Table of Contents
- [Prerequisites](#prerequisites)
- [ROCm/HIP Build (GPU-Primary)](#rocmhip-build)
- [ZenDNN Build (CPU-Primary)](#zendnn-build)
- [Dual-Backend Build](#dual-backend-build)
- [Build Flags Reference](#build-flags-reference)
- [Verifying the Build](#verifying-the-build)

## Prerequisites

### ROCm Installation
ROCm 6.1+ is required. Install from AMD's official repository:
```bash
# Check if ROCm is installed
rocminfo 2>/dev/null && echo "ROCm OK" || echo "Install ROCm first"

# Verify GPU is detected
rocm-smi --showid
# Should show: gfx1150 or gfx1151
```

### TTM Memory Configuration (Critical)
Without this, GPU can only access a small portion of system RAM.

Create `/etc/modprobe.d/increase_amd_memory.conf`:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Apply and reboot:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

Verify after reboot:
```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

## ROCm/HIP Build

### Minimal Build
```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

### Optimized Build (Recommended)
```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

### With rocWMMA Flash Attention
Requires rocWMMA v2.0+ installed. This enables optimized flash attention kernels using WMMA instructions on RDNA 3.5:
```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

### GPU Target Selection
- `gfx1150` — Ryzen AI Max+ 395 (Strix Halo) — the primary target
- `gfx1151` — Some Strix Halo SKUs report this variant
- `gfx1150;gfx1151` — Build for both (slightly longer compile time)

If unsure which your device reports:
```bash
rocminfo | grep -i "Name:" | grep gfx
```

## ZenDNN Build

ZenDNN accelerates matrix multiplication on AMD Zen CPUs using optimized BLAS primitives.

### Automatic (CMake downloads ZenDNN)
```bash
cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

### With Custom ZenDNN Installation
```bash
export ZENDNN_ROOT=/path/to/ZenDNN/build/install

cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

### ZenDNN Runtime Configuration
```bash
export ZENDNNL_MATMUL_ALGO=1    # Blocked AOCL DLP (best performance)
```

ZenDNN supports FP32 and BF16. BF16 provides best performance on Zen 5. Quantized models are not accelerated by ZenDNN — they use the standard CPU backend with AVX-512.

## Dual-Backend Build

Build with both HIP (GPU) and ZenDNN (CPU) for maximum flexibility:
```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_ZENDNN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

## Build Flags Reference

### HIP/ROCm Flags
| Flag | Default | Purpose |
|------|---------|---------|
| `GGML_HIP` | OFF | Enable ROCm/HIP backend |
| `GPU_TARGETS` | auto | GPU architecture targets (e.g., `gfx1150`) |
| `GGML_HIP_GRAPHS` | OFF | HIP graph capture (reduces dispatch overhead) |
| `GGML_HIP_NO_VMM` | ON | Disable virtual memory management |
| `GGML_HIP_ROCWMMA_FATTN` | OFF | rocWMMA flash attention (needs rocWMMA v2.0+) |
| `GGML_HIP_MMQ_MFMA` | ON | MFMA for MMQ (CDNA only, not relevant for RDNA 3.5) |
| `GGML_HIP_EXPORT_METRICS` | OFF | Export kernel resource usage metrics |
| `GGML_CUDA_FORCE_MMQ` | OFF | Force quantized matmul kernels over hipBLAS |
| `GGML_CUDA_FORCE_CUBLAS` | OFF | Force hipBLAS for all matmul |
| `GGML_CUDA_FA` | ON | Enable flash attention CUDA/HIP kernels |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF | Build FA kernels for all quant types |

### CPU Flags (Relevant for Zen 5)
| Flag | Default | Purpose |
|------|---------|---------|
| `GGML_NATIVE` | OFF | Optimize for build machine's CPU |
| `GGML_AVX512` | OFF | Enable AVX-512 (supported on Zen 5) |
| `GGML_AVX512_VNNI` | OFF | AVX-512 VNNI (check `lscpu` for support) |
| `GGML_AVX512_BF16` | OFF | AVX-512 BF16 (Zen 4+) |
| `GGML_ZENDNN` | OFF | AMD ZenDNN backend |

### Recommended: Native Build for CPU
If building on the target Strix Halo machine:
```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

`GGML_NATIVE=ON` automatically detects and enables all CPU features available on the build machine (AVX-512, BF16, etc.), so you don't need to set individual flags.

## Verifying the Build

### Check GPU Detection
```bash
./build/bin/llama-bench --list-devices
# Should show: AMD Radeon Graphics (gfx1150 or gfx1151)
```

### Quick Benchmark
```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench -m model.gguf \
  -ngl 99 --no-mmap \
  -p 512 -n 128 -r 3
```

Expected ballpark for 7B Q4_0:
- pp512: ~800-900 t/s (with hipBLASLt)
- tg128: ~45-50 t/s

### Check CPU Features
```bash
./build/bin/llama-cli --version
# Look for: AVX512 = 1 in the feature flags
```
