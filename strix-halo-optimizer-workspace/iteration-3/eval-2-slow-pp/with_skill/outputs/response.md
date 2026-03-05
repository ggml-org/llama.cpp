# Slow Prompt Processing on Strix Halo: Missing `ROCBLAS_USE_HIPBLASLT=1`

## Diagnosis

Your numbers -- ~350 t/s prompt processing and ~48 t/s token generation on a 7B model -- are a textbook match for a known issue on Strix Halo (Ryzen AI Max+ 395 / Radeon 890M) systems. The default rocBLAS kernels for gfx1150/gfx1151 are poorly optimized. Specifically, the standard Tensile-generated GEMM kernels used during prompt processing significantly underperform on RDNA 3.5 hardware.

Token generation is unaffected because it is memory-bandwidth bound (reading weights), not compute bound. Prompt processing, on the other hand, involves large matrix multiplications that hit the slow rocBLAS code path.

## The Fix

Set this environment variable before running llama.cpp:

```bash
export ROCBLAS_USE_HIPBLASLT=1
```

This forces rocBLAS to use the hipBLASLt (TensileLt) kernel library instead of the default Tensile kernels. The hipBLASLt kernels are substantially better optimized for gfx1150/gfx1151.

Add this to your `~/.bashrc` or shell profile so it persists across sessions:

```bash
echo 'export ROCBLAS_USE_HIPBLASLT=1' >> ~/.bashrc
source ~/.bashrc
```

## Expected Impact

Based on real benchmarks with a Llama 2 7B Q4_0 model on Strix Halo:

| Metric | Without hipBLASLt | With hipBLASLt |
|--------|-------------------|----------------|
| pp512 (prompt processing) | ~348 t/s | ~882 t/s |
| tg128 (token generation) | ~48 t/s | ~48 t/s |

That is roughly a **2.5x improvement** in prompt processing speed. Your pp speed should jump from ~350 t/s to ~880+ t/s. Token generation stays the same since it is not affected by this issue.

## Additional Optimizations

While you are at it, make sure your full runtime setup is optimized for Strix Halo:

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

Key flags explained:
- **`-ngl 99`**: Offload all layers to the Radeon 890M iGPU. Unified memory means no PCIe transfer penalty.
- **`--no-mmap`**: Avoids HIP page-locking overhead when loading models. Critical for models over ~64GB, good practice for all sizes.
- **`-fa on`**: Enables flash attention, which is WMMA-accelerated on RDNA 3.5.
- **`-ctk q8_0 -ctv q8_0`**: Quantized KV cache saves memory with negligible quality impact.
- **`-t 4`**: Keep CPU threads low during GPU inference. CPU and GPU share the ~256 GB/s memory bus, so high thread counts create bandwidth contention.

## Also Check: HSA_OVERRIDE_GFX_VERSION

If you followed older setup guides, you may have `HSA_OVERRIDE_GFX_VERSION` set in your environment. This was needed before llama.cpp had native RDNA 3.5 support but is now counterproductive -- it forces generic kernels that are slower than the native gfx1150 code paths.

```bash
# Check if it's set
echo $HSA_OVERRIDE_GFX_VERSION

# If it returns anything, remove it from your .bashrc
```

## Build Verification

Also confirm your build is targeting the correct architecture:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

Using `GPU_TARGETS="gfx1150"` ensures kernels are compiled specifically for your RDNA 3.5 GPU rather than a generic multi-architecture build.

## Root Cause Reference

This is a rocBLAS limitation, not a llama.cpp bug. It was documented in [llama.cpp Issue #13565](https://github.com/ggml-org/llama.cpp/issues/13565). The fix is purely an environment variable -- no code changes or rebuilds required.
