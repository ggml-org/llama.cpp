---
name: strix-halo-optimizer
description: >
  Optimize llama.cpp builds and inference for AMD Ryzen AI Max+ 395 (Strix Halo) processors
  with Radeon 890M integrated GPU. Use this skill whenever the user mentions AMD Strix Halo,
  Ryzen AI Max, Radeon 890M, gfx1150, gfx1151, RDNA 3.5, or wants to optimize llama.cpp
  performance on AMD integrated GPUs or hybrid CPU+iGPU setups. Also use when the user asks
  about ROCm/HIP build optimization for laptop or integrated AMD GPUs, unified memory
  inference strategies, or getting the best token generation speed on AMD APUs. Even if the
  user just says "make it faster on AMD" or "optimize for my laptop GPU", this skill likely
  applies if they're on Strix Halo hardware.
---

# Strix Halo Optimizer for llama.cpp

This skill helps you get maximum inference performance from llama.cpp on AMD Ryzen AI Max+ 395 (Strix Halo) systems. The Strix Halo pairs Zen 5 CPU cores with an RDNA 3.5 integrated GPU (Radeon 890M) sharing a unified memory pool of up to 128GB. This architecture creates optimization opportunities — and pitfalls — that differ significantly from discrete GPU or CPU-only setups.

## Architecture Quick Reference

| Component | Spec |
|-----------|------|
| CPU | Zen 5, up to 16 cores / 32 threads, AVX-512 |
| GPU | Radeon 890M (RDNA 3.5), 40 CUs, gfx1150/gfx1151 |
| Memory | Up to 128GB unified LPDDR5X (shared CPU/GPU) |
| GPU arch class | RDNA 3.5 (`GGML_CUDA_CC_RDNA3_5` in llama.cpp) |
| WMMA | Supported (inherited from RDNA3) |
| Memory bandwidth | ~256 GB/s (shared between CPU and GPU) |

## Critical First Steps (Linux)

Before anything else, the Linux kernel's TTM (Translation Table Maps) limits must be increased. Without this, the GPU can only access a fraction of system RAM, and models >8B will OOM despite having plenty of memory.

Create `/etc/modprobe.d/increase_amd_memory.conf`:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Then apply:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

With 4KB pages, this allows ~100GB of GPU-accessible memory. This is essential for running 70B+ models. Without it, you'll hit mysterious OOM errors even with free RAM.

## When to Use Each Strategy

- **GPU-primary** (recommended for most cases): Offload all layers to iGPU with `-ngl 99`. The Radeon 890M handles token generation well, and unified memory means zero PCIe transfer penalty. Works for models up to ~70B Q4 on 128GB systems.
- **CPU-primary with ZenDNN**: For FP32/BF16 models leveraging Zen 5's AVX-512 and ZenDNN's optimized matmul. Good for smaller models where CPU throughput matches or exceeds iGPU.
- **Hybrid CPU+GPU**: Partially offload layers. Useful when experimenting or when certain layers perform better on CPU.

## Build Configuration

The essential build command:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

`GPU_TARGETS="gfx1150"` compiles kernels specifically for RDNA 3.5 rather than a generic multi-arch build — this reduces compile time from 30+ minutes to under 10 and produces better-optimized code.

Read `references/build-guide.md` for all build configurations including ZenDNN and dual-backend builds.

### Key Build Flags

| Flag | Purpose | Recommendation |
|------|---------|----------------|
| `GGML_HIP=ON` | Enable ROCm/HIP GPU backend | Always for GPU accel |
| `GPU_TARGETS=gfx1150` | Target Strix Halo RDNA 3.5 | Always — avoids bloated multi-arch |
| `GGML_HIP_GRAPHS=ON` | HIP graph capture for kernel dispatch | Reduces dispatch overhead |
| `GGML_CUDA_FORCE_MMQ=ON` | Force quantized matmul kernels | Often faster than hipBLAS on RDNA 3.5 for quantized models |
| `GGML_HIP_ROCWMMA_FATTN=ON` | rocWMMA flash attention | Enable if rocWMMA v2.0+ is installed |
| `GGML_ZENDNN=ON` | ZenDNN CPU backend | For CPU-primary or hybrid |

## Runtime Optimization

### The hipBLASLt Fix (Critical for Prompt Processing)

The default rocBLAS kernels for gfx1150/gfx1151 are poorly optimized — prompt processing can be 2-3x slower than expected. Setting this environment variable forces the hipBLASLt kernel library, which dramatically improves performance:

```bash
export ROCBLAS_USE_HIPBLASLT=1
```

**Real benchmark impact** (Llama 2 7B Q4_0):
- Without: pp512 = 348 t/s
- With hipBLASLt: pp512 = 882 t/s (~2.5x improvement)
- Token generation is unaffected (~48 t/s either way)

This is a rocBLAS limitation, not a llama.cpp issue. Always set this variable.

### GPU-Primary Inference

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-cli -m model.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 \
  -t 4
```

Key points:
- **`--no-mmap`**: Critical for models >64GB. Without this, HIP's `hipMemcpy()` must lock/unlock mmap pages, causing extreme slowdown during model loading.
- **`-t 4`**: Keep CPU threads low during GPU inference. Memory bandwidth is shared, so high CPU thread count competes with GPU for bandwidth.
- **`-fa on`**: Flash attention is WMMA-accelerated on RDNA 3.5.
- **`-ctk q8_0 -ctv q8_0`**: Quantized KV cache saves memory for larger contexts.

### CPU-Primary with ZenDNN

```bash
export ZENDNNL_MATMUL_ALGO=1    # Blocked AOCL DLP (best algorithm)

./build/bin/llama-cli -m model.gguf \
  -ngl 0 \
  -t $(nproc) \
  -b 2048 -ub 512
```

ZenDNN accelerates MUL_MAT with FP32 and BF16 data types. BF16 performs best on Zen 5. For quantized models, the standard CPU backend with AVX-512 handles the work.

### Hybrid CPU+GPU

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-cli -m model.gguf \
  -ngl 20 \
  --no-mmap \
  -t 8 \
  -fa on \
  -b 2048 -ub 512
```

Finding the optimal `-ngl` split requires benchmarking. Read `references/tuning-guide.md` for a systematic approach.

## Benchmarking

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench -m model.gguf \
  -ngl 0,20,40,99 \
  -t 4,8,16 \
  -p 512 -n 128 \
  --no-mmap \
  -r 3 \
  -o csv > bench_results.csv
```

Watch for:
- **pp t/s** (prompt processing): Should be 800+ for 7B Q4 models with hipBLASLt
- **tg t/s** (token generation): ~48 t/s for 7B Q4, scales with model size
- Compare `-ngl` values to find the sweet spot between CPU and GPU

## Memory Considerations

The unified memory architecture means CPU and GPU share the same physical memory with no dedicated VRAM. This has implications:

1. **No transfer overhead**: Offloading layers to GPU has no PCIe penalty — partial offloading is cheap.
2. **Bandwidth sharing**: CPU and GPU compete for the ~256 GB/s memory bus. Heavy GPU load degrades CPU performance and vice versa.
3. **TTM limits**: Linux kernel must be configured to allow GPU access to large memory pools (see Critical First Steps above).
4. **`--no-mmap` for large models**: HIP's memory copy from mmap'd pages has severe overhead past ~64GB. Always use `--no-mmap` for large models.
5. **KV cache quantization**: `-ctk q8_0 -ctv q8_0` (or q4_0) reduces memory footprint and frees bandwidth.

## Troubleshooting

Read `references/troubleshooting.md` for the full list. Quick fixes:

| Problem | Solution |
|---------|----------|
| OOM with plenty of free RAM | Increase TTM pages_limit (see Critical First Steps) |
| Slow prompt processing | Set `ROCBLAS_USE_HIPBLASLT=1` |
| Slow/hanging model loading (>64GB) | Use `--no-mmap` flag |
| "No HIP devices found" | Install ROCm 6.1+, verify with `rocminfo` |
| Slow compilation | Use `GPU_TARGETS=gfx1150` instead of multi-arch |
| Poor TG speed with hipBLAS | Rebuild with `GGML_CUDA_FORCE_MMQ=ON` |
| Vulkan model loading failure | Use ROCm/HIP backend instead (better optimized for Strix Halo) |

## Reference Files

- `references/build-guide.md` — Detailed build instructions for all configurations
- `references/tuning-guide.md` — Systematic performance tuning methodology
- `references/troubleshooting.md` — Common issues, root causes, and solutions from real GitHub issues
