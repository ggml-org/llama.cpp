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
| CPU | Zen 5, up to 16C/32T (2 CCDs x 8 cores), boost 5.1 GHz |
| CPU cache | 80 KB L1/core, 1 MB L2/core, 32 MB L3/CCD (64 MB total on 16C) |
| CPU AVX-512 | Supported (BF16, VNNI, VBMI) — **half-width 256-bit datapaths** on mobile Zen 5 (double-pumped, still beneficial but not full 512-bit throughput) |
| GPU | Radeon 8060S (RDNA 3.5), 40 CUs / 2560 SPs, up to 2.9 GHz, gfx1151 |
| GPU compute | ~59.4 FP16/BF16 TFLOPS peak |
| Infinity Cache | 32 MB MALL (shared CPU/GPU memory-side cache) |
| Memory | Up to 128GB unified LPDDR5X-8000, ~256 GB/s theoretical (~212 GB/s measured) |
| GPU arch class | RDNA 3.5 (`GGML_CUDA_CC_RDNA3_5` in llama.cpp) |
| WMMA | Supported (inherited from RDNA3) |
| NPU | XDNA 2, 50 TOPS INT8 (not yet a llama.cpp target) |

**SKU Variants**: Max+ 395 (16C/40CU), Max+ 392 (12C/40CU), Max 390 (12C/40CU), Max+ 388 (8C/40CU), Max 385 (8C/32CU), Max Pro 380 (6C/16CU). Adjust `-t` thread count and GPU expectations based on your specific SKU.

**MoE Model Sweet Spot**: Strix Halo excels at Mixture-of-Experts models due to high memory bandwidth. Qwen3-30B-A3B achieves ~52 tok/s, Llama 4 Scout (109B params, 17B active) runs ~4x faster than Llama 3.3 70B dense.

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

- **GPU-primary via HIP/ROCm** (recommended for most cases): Offload all layers to iGPU with `-ngl 99`. The Radeon 890M handles token generation well, and unified memory means zero PCIe transfer penalty. Works for models up to ~70B Q4 on 128GB systems.
- **GPU via Vulkan**: An alternative GPU backend. Vulkan auto-detects RDNA 3.5 and uses wave64 subgroups (64 threads per wavefront). Useful when ROCm is unavailable or for portability, but HIP generally provides better performance on Strix Halo due to hipBLASLt and HIP graph support. See "Vulkan Backend Details" below.
- **CPU-primary with ZenDNN**: For FP32/BF16 models leveraging Zen 5's AVX-512 and ZenDNN's optimized matmul. Good for smaller models where CPU throughput matches or exceeds iGPU. See "Zen 5 CPU Backend Details" below.
- **CPU-primary without ZenDNN**: For quantized models (Q4/Q5/Q8), the standard CPU backend uses AVX-512 SIMD paths with 16-wide float operations, AVX-512 VNNI for int8 dot products, and optional AVX-512 BF16 instructions. ZenDNN only accelerates FP32/BF16 matmul — quantized inference runs on the native GGML CPU backend regardless.
- **Hybrid CPU+GPU**: Partially offload layers. Useful when experimenting or when certain layers perform better on CPU.

## Build Configuration

The essential build command:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1151" \
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
| `GPU_TARGETS=gfx1151` | Target Strix Halo RDNA 3.5 | Always — avoids bloated multi-arch (use `gfx1150` if `rocminfo` reports that instead) |
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

## Vulkan Backend Details

The Vulkan backend (`ggml/src/ggml-vulkan/`) provides a portable GPU compute path. On Strix Halo:

### Build with Vulkan
```bash
cmake -S . -B build \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

### How Vulkan Works on RDNA 3.5
- **Device detection**: Vulkan auto-detects the Radeon 890M via vendor ID (AMD = 0x1002). No need for `GPU_TARGETS` — Vulkan uses runtime SPIR-V shaders, not precompiled ISA.
- **Subgroup size**: RDNA 3.5 supports both wave32 and wave64. The Vulkan backend queries `VkPhysicalDeviceSubgroupProperties` and adapts shader dispatch. Matmul shaders use subgroup operations (`subgroupAdd`, `subgroupShuffle`) for efficient reductions.
- **Memory**: Vulkan sees device-local memory (the BIOS-allocated VRAM portion) plus host-visible memory. For unified memory APUs like Strix Halo, both map to the same physical LPDDR5X, but device-local allocations may get priority bandwidth. The backend prefers device-local for model weights.
- **Matmul shaders**: Located in `ggml/src/ggml-vulkan/vulkan-shaders/`. Key files: `mul_mat_vec*.comp` (token generation), `mul_mm*.comp` (batched matmul for prompt processing). Shaders are precompiled to SPIR-V at build time.

### Vulkan vs HIP: When to Choose Which

| Factor | HIP/ROCm | Vulkan (RADV) |
|--------|----------|---------------|
| Prompt processing | ~880 t/s 7B Q4 (with hipBLASLt) / ~350 without | ~850 t/s — competitive with HIP+hipBLASLt |
| Token generation | ~48 t/s (7B Q4) | ~44 t/s — RADV is ~4% faster on tg but this varies |
| Graph capture | HIP graphs reduce dispatch overhead | Not available |
| Flash attention | WMMA-accelerated via rocWMMA | Shader-based, scales well at long context |
| Setup complexity | Requires ROCm 6.1+ installation | Just needs Vulkan drivers (Mesa RADV, usually pre-installed) |
| Known issues | None critical (with TTM + hipBLASLt) | Model loading failures on some configs ([#18741](https://github.com/ggml-org/llama.cpp/issues/18741)) |
| Driver choice | N/A | RADV ~4% faster tg, AMDVLK ~16% faster pp |

**Recommendation**: Both backends are viable. Vulkan (RADV) is the easiest path — no ROCm install, competitive performance. HIP/ROCm provides the most tuning options (HIP graphs, rocWMMA flash attention, FORCE_MMQ). If you have ROCm installed, use HIP. If you want simplicity or are on a distro without easy ROCm support, Vulkan is excellent. Note: Linux kernel version matters — 6.15+ showed ~15% improvement over 6.14.

## Zen 5 CPU Backend Details

The CPU backend (`ggml/src/ggml-cpu/`) leverages Zen 5's AVX-512 instruction set for high-throughput inference.

### AVX-512 on Zen 5
Zen 5 supports AVX-512F, AVX-512CD, AVX-512VL, AVX-512DQ, AVX-512BW, AVX-512 VNNI, and AVX-512 BF16. The GGML CPU backend uses these as follows:

- **16-wide SIMD**: All vector operations use 512-bit registers (`__m512`), processing 16 floats per instruction — 2x the throughput of AVX2's 256-bit paths. Defined in `ggml/src/ggml-cpu/simd-mappings.h`.
- **Fused multiply-add**: `_mm512_fmadd_ps` for efficient dot products and reductions.
- **BF16 dot product**: `_mm512_dpbf16_ps` (AVX-512 BF16) processes 32 BF16 values per operation. This is the fastest CPU path for BF16 models on Zen 5. Defined in `ggml/src/ggml-cpu/vec.cpp`.
- **VNNI int8 dot product**: `_mm256_dpbusd_epi32` (AVX-512 VNNI with VL) accelerates quantized inference for Q8 and mixed-precision operations. Defined in `ggml/src/ggml-cpu/arch/x86/quants.c`.
- **Runtime feature scoring**: `ggml_backend_cpu_x86_score()` in `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp` detects available CPU features via CPUID and selects the best kernel variant at runtime.

### Build for Maximum CPU Performance
```bash
cmake -S . -B build \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

`GGML_NATIVE=ON` auto-detects and enables all CPU features on the build machine. Alternatively, enable specific features:
```bash
-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON
```

### ZenDNN Integration
ZenDNN (`ggml/src/ggml-zendnn/`) is AMD's optimized primitives library. It accelerates:
- **F32 x F32 → F32** matmul
- **BF16 x BF16 → BF16** matmul (best on Zen 5)
- **BF16 x BF16 → F32** mixed-precision matmul

ZenDNN does **not** accelerate quantized (Q4/Q5/Q8) operations — those use the standard GGML CPU kernels with AVX-512. The key environment variable:
```bash
export ZENDNNL_MATMUL_ALGO=1    # Blocked AOCL DLP algorithm (fastest)
```

ZenDNN depends on AOCL-DLP (AMD Optimized CPU Libraries) and libxsmm internally.

### CPU-Only Quantized Inference
For quantized models without ZenDNN, build with:
```bash
cmake -S . -B build \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
```
No additional flags needed — the runtime scorer selects AVX-512 VNNI kernels automatically. Use all physical cores: `-t 16` (Strix Halo has 16 Zen 5 cores). Performance scales well up to the physical core count but degrades with hyperthreading.

## llama-server Production Deployment

For running `llama-server` as a persistent service on Strix Halo:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-server \
  -m model.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  -np 4 \
  -t 4 \
  --host 0.0.0.0 \
  --port 8080
```

Key considerations for production:
- **`-np 4`**: Number of parallel sequences. Each needs its own KV cache, so memory usage scales with `np * context_length`. With q8_0 KV cache, 4 parallel slots at 8K context is manageable on 128GB.
- **`-c 8192`**: Context length. Larger contexts need more KV cache memory. Use `-ctk q8_0 -ctv q8_0` (or q4_0 under memory pressure) to keep memory in check.
- **`-t 4`**: Keep threads low — the GPU does the heavy lifting, and CPU threads compete for shared bandwidth.
- **`-fa on`**: Flash attention reduces memory usage for long contexts and is WMMA-accelerated on RDNA 3.5.
- **`--no-mmap`**: Essential for stable operation with large models under sustained load.
- **Systemd service**: For persistence, create a systemd unit file. Set `Environment=ROCBLAS_USE_HIPBLASLT=1` in the `[Service]` section.
- **Monitoring**: The server exposes `/health` and `/metrics` endpoints. Monitor VRAM usage via `rocm-smi` and watch for TTM-related OOM under sustained multi-user load.

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
