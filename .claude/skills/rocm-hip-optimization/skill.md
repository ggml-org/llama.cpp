---
name: rocm-hip-optimization
description: >
  Optimize llama.cpp HIP/ROCm backend performance on AMD GPUs, especially Strix Halo
  (gfx1151, RDNA 3.5) iGPUs. Covers build configuration, runtime tuning, HIP graphs,
  UMA optimizations, flash attention, WMMA, and closing the gap with the Vulkan backend.
  Use when the user mentions "ROCm", "HIP", "rocBLAS", "hipBLAS", "gfx1151", "gfx1150",
  "HIP graphs", "RDNA", or wants to improve HIP inference speed on AMD hardware.
---

# ROCm/HIP Optimization for llama.cpp on Strix Halo

## Current Performance Gap

| Backend | tok/s (tg128) | Notes |
|---------|--------------|-------|
| **Vulkan** | **67.09** | Our optimized branch (fused SSM + mega-kernel + tiling) |
| **HIP/ROCm** | **~21** | Stock llama.cpp, gfx1151, ROCm 7.1 |
| **Gap** | **3.2x** | HIP is missing several optimizations Vulkan has |

Model: Qwen3.5-35B-A3B Q4_K_M on AMD Ryzen AI Max+ 395 (Strix Halo), Windows 11.

## Why HIP Is Slower (Root Causes)

1. **No fused SSM recurrence kernel** — HIP uses separate `ssm-scan.cu` and `ssm-conv.cu` kernels, not our tiled+fused shader. The Vulkan fused kernel with shared memory tiling gave +14.8% alone.
2. **No batched elementwise mega-kernel** — HIP dispatches ~260 tiny elementwise ops individually. Vulkan batches them into one dispatch (+7%).
3. **No wave64 optimization** — HIP/RDNA uses wave32 natively. Our Vulkan backend explicitly uses wave64 for key shaders (+8.7%).
4. **HIP graphs disabled** — `GGML_HIP_GRAPHS=OFF` by default. CUDA graphs eliminate CPU dispatch overhead entirely (vLLM gets 2.2x from this alone).
5. **Immature Windows ROCm drivers** — ROCm is primarily developed for Linux datacenter GPUs (MI300X). Windows iGPU support is secondary.
6. **No shared memory tiling for SSM** — The SSM state access pattern is non-coalesced in HIP (same problem Vulkan had before our fix).

## Hardware: Strix Halo (gfx1151)

- **Architecture**: RDNA 3.5, 40 CUs, wave32 native
- **Memory**: 68 GB LPDDR5X, ~212 GB/s bandwidth (shared with CPU)
- **UMA**: Integrated GPU, unified memory (no PCIe bottleneck)
- **Compute capability**: `GGML_CUDA_CC_RDNA3_5` (0x1150 offset + AMD)
- **WMMA**: Available (16x16 wave matrix multiply accumulate)
- **MFMA**: NOT available (CDNA-only, MI-series datacenter GPUs)
- **Infinity Cache**: 32 MB L3 cache for memory compression
- **Shared memory (LDS)**: 64 KB per CU (vs 32 KB maxComputeSharedMemorySize in Vulkan)

**Important**: HIP on RDNA 3.5 has 64 KB LDS per CU — double what Vulkan exposes. This means HIP SSM kernels could use TILE_K=128 (no tiling needed!) if properly optimized.

## Build Configuration

### Optimal CMake flags for Strix Halo

```bash
cmake -B build-hip \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DAMDGPU_TARGETS=gfx1151 \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-hip --config Release -j 16
```

### Build flags explained

| Flag | Effect | Recommended |
|------|--------|-------------|
| `GGML_HIP=ON` | Enable HIP backend | Required |
| `GPU_TARGETS=gfx1151` | Target Strix Halo specifically | Required |
| `GGML_HIP_GRAPHS=ON` | Enable HIP graph capture (experimental) | Try it |
| `GGML_HIP_ROCWMMA_FATTN=ON` | Use rocWMMA for flash attention | Try it |
| `GGML_HIP_NO_VMM=ON` | Disable virtual memory management | Default ON |
| `GGML_HIP_MMQ_MFMA=ON` | MFMA matrix multiply (CDNA only) | N/A on RDNA |
| `GGML_HIP_EXPORT_METRICS=ON` | Kernel profiling/metrics | For profiling |
| `GGML_CUDA_FORCE_MMQ=ON` | Force MMQ over hipBLAS | Test both |

### Windows build notes

- ROCm 7.1 on Windows uses `clang++` from `/opt/rocm/7.1/bin/`
- Windows HIP builds use `hipcc` compiler mode (not CMake HIP language)
- The HIP backend compiles ALL CUDA `.cu` files via HIP compatibility layer
- No separate HIP implementation — it's CUDA code with `vendors/hip.h` mapping

## Runtime Environment Variables

### Critical for performance

```bash
# 2-3x faster prompt processing on RDNA 3.5 — MUST SET
export ROCBLAS_USE_HIPBLASLT=1

# Force UMA path (auto-detected on Strix Halo, but can force)
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
```

The code warns if `ROCBLAS_USE_HIPBLASLT` is not set on RDNA 3.5:
```
RDNA 3.5 detected: set ROCBLAS_USE_HIPBLASLT=1 for 2-3x faster prompt processing
```

### Other tuning variables

```bash
# UMA expert prefetch for MoE models
export GGML_CUDA_UMA_PREFETCH=1
export GGML_CUDA_UMA_PREFETCH_LOOKAHEAD=2

# Force specific tensor split (for hybrid CPU+GPU)
export GGML_CUDA_SPLIT_MODE=layer  # or "row"
```

## Architecture Details

### Code structure (HIP reuses CUDA)

```
ggml/src/ggml-cuda/          # All kernel source code (.cu, .cuh)
  ├── ggml-cuda.cu           # Main backend file, UMA detection, dispatch
  ├── common.cuh             # Architecture detection (RDNA3_5 macros)
  ├── vendors/hip.h          # CUDA→HIP API mapping header
  ├── fattn*.cu/cuh          # Flash attention (MMA, WMMA, vec variants)
  ├── mmq*.cu/cuh            # Matrix multiply quantized
  ├── ssm-scan.cu            # SSM state recurrence (NOT fused)
  ├── ssm-conv.cu            # SSM convolution
  └── ...                    # 43 .cu files total

ggml/src/ggml-hip/
  └── CMakeLists.txt         # Build config, globs ggml-cuda sources
```

### RDNA 3.5 compile-time macros

In `common.cuh`:
```cpp
#define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 0x1150)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)

// Shader-level macros (set by compiler for gfx1150/gfx1151):
#if defined(__GFX11__)
  #define RDNA3
  #if defined(__gfx1150__) || defined(__gfx1151__)
    #define RDNA3_5
  #endif
#endif
```

### Current RDNA 3.5-specific optimizations in codebase

1. **Flash attention vec**: Uses `nthreads_KQ_q = 4` (vs 2 for general RDNA)
2. **hipBLASLt auto-enablement**: Detected and recommended at startup
3. **UMA detection**: `prop.integrated` flag for iGPU identification
4. **Soft scheduling**: Sets "spinning" strategy for iGPU sync to avoid delays
5. **Expert-aware prefetch**: `hipMemPrefetchAsync()` during MoE gate computation

### What's missing vs Vulkan (optimization opportunities)

| Optimization | Vulkan Status | HIP Status | Potential Gain |
|-------------|---------------|------------|----------------|
| Fused SSM recurrence | Done (+14.8%) | Missing | +10-15% |
| Shared memory tiling for SSM | Done (TILE_K=64) | Missing | Included above |
| Batched elementwise mega-kernel | Done (+7%) | Missing | +5-7% |
| Wave64 for key shaders | Done (+8.7%) | N/A (wave32 native) | Different approach needed |
| HIP graphs | N/A | OFF by default | +10-30% (est.) |
| rocWMMA flash attention | N/A | OFF by default | Unknown |
| Infinity Cache optimization | N/A | Not tuned | Unknown |

## Optimization Roadmap (Prioritized)

### Tier 1: Quick wins (build/runtime config only)

#### 1A. Enable HIP graphs
```bash
cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON
```
HIP graphs capture the dispatch sequence and replay it, eliminating per-token CPU dispatch overhead. CUDA graphs give 2.2x speedup in vLLM. On our small-batch (batch=1) workload, expect 10-30% improvement.

**Risk**: Marked experimental. May crash or produce wrong results. Test with `llama-bench` first.

#### 1B. Enable rocWMMA flash attention
```bash
cmake -B build-hip -DGGML_HIP_ROCWMMA_FATTN=ON
```
Uses hardware WMMA instructions for flash attention on RDNA 3.5. Currently disabled by default.

**Risk**: Low for correctness, unknown for performance on SSM-dominated models.

#### 1C. Set ROCBLAS_USE_HIPBLASLT=1
Already auto-detected but worth verifying. 2-3x faster prompt processing.

#### 1D. Enable UMA expert prefetch
```bash
export GGML_CUDA_UMA_PREFETCH=1
export GGML_CUDA_UMA_PREFETCH_LOOKAHEAD=2
```
Prefetches next expert weights while current expert is computing.

### Tier 2: Port Vulkan optimizations to HIP (moderate effort)

#### 2A. Fused SSM recurrence kernel for HIP
Port `ssm_recurrence.comp` (Vulkan) → `ssm-scan.cu` (HIP).

Key differences:
- HIP has 64 KB LDS (vs Vulkan 32 KB) — can fit entire 128x128 state without tiling!
- Wave32 on RDNA means 32 threads per wavefront (vs 64 subgroup in Vulkan)
- Use `__shared__` instead of `shared float s_tile[]`
- Use `__syncthreads()` instead of `barrier()`
- Thread indexing: `threadIdx.x` instead of `gl_LocalInvocationIndex`

```cpp
// HIP kernel sketch (untiled — 64 KB LDS fits full state)
__global__ void ssm_recurrence_fused(
    const float* state_in, const float* q, const float* k,
    const float* v, const float* gate, const float* beta,
    float* dst, uint32_t S, uint32_t H, uint32_t n_seqs, uint32_t s_off)
{
    __shared__ float s_shared[128 * 128];  // 64 KB — fits in RDNA LDS!

    const uint32_t j = threadIdx.x;  // row index
    const uint32_t head = blockIdx.x % H;
    const uint32_t seq = blockIdx.x / H;

    // Load entire state into shared memory (coalesced)
    // ... (cooperative loading, same pattern as Vulkan but no tiling needed)

    // Compute sk_j, d_j, update state, compute o_j — single pass!
    // ... (no 2-pass needed since entire state fits)
}
```

**Advantage over Vulkan**: No tiling needed! Single-pass algorithm possible.
**Expected gain**: +10-15% (largest single opportunity)

#### 2B. Batched elementwise for HIP
Port `batched_elementwise.comp` → HIP kernel. Same concept: batch SILU/EXP/SIGMOID/etc into one launch.

HIP advantage: Could use HIP graphs to batch even more aggressively.

### Tier 3: HIP-specific optimizations (high effort)

#### 3A. Infinity Cache tuning
Strix Halo has 32 MB Infinity Cache (L3). For MoE models, the active expert weights (~1.9 GB) don't fit, but the SSM state matrices (128x128x4 = 64 KB per head) and small vectors (q, k, v, gate, beta) should be cache-resident.

Investigate: Are repeated SSM state reads hitting Infinity Cache? Profiling with `rocprof` would reveal cache hit rates.

#### 3B. Cooperative groups for SSM
HIP supports cooperative groups (similar to CUDA cooperative groups). Could enable more flexible thread cooperation patterns for the SSM recurrence.

#### 3C. Custom GEMV for small matmuls
The SSM state update involves 128x128 matrix-vector products. hipBLAS overhead for these tiny GEMVs may exceed the compute time. Custom kernels could be faster.

## Benchmarking

### Quick comparison benchmark

```bash
# Vulkan baseline
build-win/bin/llama-bench -m model.gguf -ngl 99 -t 8 -r 5 -p 0 -n 128

# HIP with all flags
ROCBLAS_USE_HIPBLASLT=1 GGML_CUDA_UMA_PREFETCH=1 \
  build-hip/bin/llama-bench -m model.gguf -ngl 99 -t 8 -r 5 -p 0 -n 128
```

### Expected baselines (Strix Halo, Qwen3.5-35B-A3B Q4_K_M)

| Config | pp512 tok/s | tg128 tok/s |
|--------|------------|-------------|
| HIP stock | ~200 | ~21 |
| HIP + HIPBLASLT | ~600 | ~21 |
| HIP + graphs (est.) | ~200 | ~25-28 |
| HIP + fused SSM (est.) | — | ~30-35 |
| Vulkan optimized | ~300 | **67** |

### Profiling

```bash
# ROCm profiler (Linux)
rocprof --stats build-hip/bin/llama-bench -m model.gguf -ngl 99 -p 0 -n 128

# HIP export metrics (build-time flag)
cmake -DGGML_HIP_EXPORT_METRICS=ON ...

# Windows: use AMD Radeon GPU Profiler (RGP) or rocprof via WSL
```

## Key Files Reference

| Component | Path |
|-----------|------|
| HIP build config | `ggml/src/ggml-hip/CMakeLists.txt` |
| CUDA→HIP API mapping | `ggml/src/ggml-cuda/vendors/hip.h` |
| Main backend + UMA | `ggml/src/ggml-cuda/ggml-cuda.cu` |
| Architecture detection | `ggml/src/ggml-cuda/common.cuh` |
| SSM scan kernel | `ggml/src/ggml-cuda/ssm-scan.cu` |
| SSM conv kernel | `ggml/src/ggml-cuda/ssm-conv.cu` |
| Flash attention (MMA) | `ggml/src/ggml-cuda/fattn-mma-f16.cuh` |
| Flash attention (WMMA) | `ggml/src/ggml-cuda/fattn-wmma-f16.cu` |
| Flash attention (vec) | `ggml/src/ggml-cuda/fattn-vec.cuh` |
| Matrix multiply (MMQ) | `ggml/src/ggml-cuda/mmq.cu` |
| Benchmark script | `scripts/bench-strix-halo.sh` |
| MoE benchmark | `scripts/bench-strix-halo-moe.sh` |
| Build cache | `build-hip/CMakeCache.txt` |

## Lessons from Vulkan Optimization (Apply to HIP)

1. **Fusing SSM recurrence is the biggest win** — 11 dispatches → 1, with coalesced shared memory access. Port this first.
2. **Shared memory tiling transforms performance** — non-coalesced reads waste 94% of cache lines. HIP has 64 KB LDS, so no tiling needed (entire 128x128 state fits).
3. **Don't use HostCoherent memory on UMA** — same lesson applies to HIP. Use device-local allocation, let the driver manage coherence.
4. **Measure before optimizing** — profile with `rocprof` or `GGML_HIP_EXPORT_METRICS` to find the actual bottleneck before writing code.
5. **The SSM model is dispatch-bound, not compute-bound** — reducing dispatch count matters more than optimizing individual kernels.
