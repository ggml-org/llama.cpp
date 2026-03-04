# Diagnosing Slow Prompt Processing on Strix Halo

Your symptom is a classic and well-documented issue on Strix Halo (AMD Ryzen AI Max+ 395 / Radeon 890M). The numbers you describe — ~350 t/s prompt processing (pp) and ~48 t/s token generation (tg) — are a textbook match for a specific root cause.

## Root Cause: rocBLAS Using Unoptimized Kernels for gfx1150

The default rocBLAS kernels for the gfx1150/gfx1151 architecture (RDNA 3.5) are poorly optimized for prompt processing workloads. Prompt processing is dominated by large matrix multiplications (the prefill pass), and the standard Tensile-generated rocBLAS kernels severely underperform on this GPU for those operations.

Token generation, on the other hand, involves smaller memory-bound operations that are not as sensitive to this kernel quality issue — which is exactly why your tg speed (~48 t/s) looks correct while pp speed is degraded.

This is confirmed by real benchmark data on 7B Q4_0 models at pp512:

- **Without fix**: ~348 t/s (matches what you're seeing)
- **With fix**: ~882 t/s (~2.5x improvement)
- **Token generation**: ~48 t/s (unaffected either way)

## The Fix: One Environment Variable

Set this before running llama.cpp:

```bash
export ROCBLAS_USE_HIPBLASLT=1
```

This forces rocBLAS to use the hipBLASLt (TensileLt) kernel library instead of the default Tensile kernels. hipBLASLt has dramatically better performance for the matrix multiplication shapes used during prompt processing on RDNA 3.5.

To make it permanent, add it to your shell profile:

```bash
echo 'export ROCBLAS_USE_HIPBLASLT=1' >> ~/.bashrc
source ~/.bashrc
```

## Full Recommended Command

With the env var set, here is the complete recommended invocation for a 7B model on Strix Halo:

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

Key flags and why they matter on Strix Halo:

- **`-ngl 99`**: Offload all layers to the iGPU. Since Strix Halo uses unified memory (no separate VRAM), there is zero PCIe transfer penalty for full GPU offload.
- **`--no-mmap`**: HIP's `hipMemcpy()` has severe overhead when working with mmap'd memory pages. This matters less for a 7B model but is critical for models >64GB and is good practice regardless.
- **`-fa on`**: Flash attention is WMMA-accelerated on RDNA 3.5. Enable it.
- **`-ctk q8_0 -ctv q8_0`**: Quantizes the KV cache to reduce memory usage and bandwidth pressure.
- **`-t 4`**: Strix Halo's ~256 GB/s memory bandwidth is shared between CPU and GPU. Using too many CPU threads during GPU inference creates bandwidth contention. Keep it low (2-4) when using GPU-primary mode.

## Verifying the Fix

You can benchmark before and after with:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench -m model.gguf \
  -ngl 99 \
  --no-mmap \
  -p 512 -n 128 \
  -r 3
```

You should see pp512 jump from ~350 t/s to ~800-900 t/s. Token generation will remain at ~48 t/s, which is expected and correct for a 7B Q4 model on this hardware.

## One Thing to Check: HSA_OVERRIDE_GFX_VERSION

If you previously followed older guides that suggested setting `HSA_OVERRIDE_GFX_VERSION`, remove it. That workaround was needed before llama.cpp had native gfx1150 support. It now forces generic, slower kernel paths. Check whether it's set:

```bash
echo $HSA_OVERRIDE_GFX_VERSION
grep HSA_OVERRIDE_GFX_VERSION ~/.bashrc
```

If it's present, remove it — it will counteract the native RDNA 3.5 optimizations.

## Summary

| Symptom | Cause | Fix |
|---------|-------|-----|
| pp ~350 t/s on 7B Q4 | rocBLAS using slow Tensile kernels for gfx1150 | `export ROCBLAS_USE_HIPBLASLT=1` |
| tg ~48 t/s (normal) | Memory-bandwidth bound, not kernel-quality sensitive | No action needed |

The single environment variable `ROCBLAS_USE_HIPBLASLT=1` is the fix. It should bring your prompt processing speed up to the ~880 t/s range, which is the expected performance for a 7B Q4_0 model on the Radeon 890M.
