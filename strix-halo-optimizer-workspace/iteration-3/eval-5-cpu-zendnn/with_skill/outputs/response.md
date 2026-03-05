# CPU-Only Inference with ZenDNN on Ryzen AI Max 395 (Strix Halo)

## Overview

Your Ryzen AI Max 395 has Zen 5 cores with full AVX-512 support (including AVX-512 BF16 instructions), making it a strong CPU-only inference platform. ZenDNN is AMD's optimized primitives library that accelerates matrix multiplication on Zen CPUs. In llama.cpp, it acts as an accelerator backend that intercepts `MUL_MAT` operations for FP32 and BF16 data types and runs them through AMD's hand-tuned BLAS routines instead of the generic CPU path.

For your specific scenario -- a 7B BF16 model, CPU-only -- ZenDNN is the ideal backend because BF16 is one of its two supported data types and Zen 5 has native hardware BF16 support.

## Step 1: Build llama.cpp with ZenDNN and Native CPU Optimizations

You need two key build flags:

- `GGML_ZENDNN=ON` -- enables the ZenDNN backend for accelerated matmul
- `GGML_NATIVE=ON` -- auto-detects and enables all CPU features on your build machine (AVX-512, AVX-512 BF16, AVX-512 VNNI, etc.) so you do not need to set each flag individually

Since you want CPU-only (no GPU), you do **not** need `GGML_HIP=ON` or any GPU flags.

```bash
cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

CMake will automatically download and build ZenDNN if it is not already installed on your system. If you have a custom ZenDNN installation, point to it before running cmake:

```bash
export ZENDNN_ROOT=/path/to/ZenDNN/build/install
```

### Alternative: Explicit AVX-512 Flags (if not building on the target machine)

If you are cross-compiling or want explicit control instead of `GGML_NATIVE=ON`:

```bash
cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_BF16=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

## Step 2: Verify the Build

After building, confirm AVX-512 support is compiled in:

```bash
./build/bin/llama-cli --version
# Look for: AVX512 = 1 in the feature flags
```

## Step 3: Runtime Configuration

### Set the ZenDNN Algorithm

ZenDNN exposes an environment variable to select its matmul algorithm. Algorithm 1 (Blocked AOCL DLP) provides the best performance:

```bash
export ZENDNNL_MATMUL_ALGO=1
```

Add this to your `~/.bashrc` for persistence.

### Run Inference

For CPU-only inference, set `-ngl 0` (zero GPU layers) and use all available CPU threads:

```bash
export ZENDNNL_MATMUL_ALGO=1

./build/bin/llama-cli -m your-7b-bf16-model.gguf \
  -ngl 0 \
  -t $(nproc) \
  -b 2048 -ub 512
```

Key parameters explained:

| Flag | Value | Purpose |
|------|-------|---------|
| `-ngl 0` | 0 | No GPU layer offloading -- pure CPU inference |
| `-t $(nproc)` | All threads | Use all CPU threads. On the 395, this is 32 (16 cores x 2 threads). See tuning note below. |
| `-b 2048` | 2048 | Logical batch size for prompt processing |
| `-ub 512` | 512 | Physical (compute) batch size |

## Step 4: Thread Count Tuning

Performance typically scales well up to the physical core count (16 on the Ryzen AI Max 395), then plateaus or even degrades when hyperthreading threads are added. Benchmark to find your sweet spot:

```bash
export ZENDNNL_MATMUL_ALGO=1

./build/bin/llama-bench -m your-7b-bf16-model.gguf \
  -ngl 0 \
  -t 4,8,12,16,24,32 \
  -p 512 -n 128 \
  -r 3 -o csv > cpu_thread_sweep.csv
```

Start with `-t 16` (physical core count) and compare against `-t 32` (all threads). For BF16 matmul workloads, the physical core count is usually optimal since ZenDNN's BLAS routines are compute-dense and benefit less from SMT.

## How ZenDNN Works Under the Hood

Looking at the llama.cpp source, the ZenDNN backend (`ggml/src/ggml-zendnn/ggml-zendnn.cpp`) works as follows:

1. It registers itself as an accelerator backend (`GGML_BACKEND_DEVICE_TYPE_ACCEL`).
2. It only handles `MUL_MAT` operations -- all other operations (attention, normalization, etc.) are handled by the standard CPU backend with AVX-512.
3. For `MUL_MAT`, it supports exactly two weight types: **FP32** and **BF16**. Quantized types (Q4_0, Q4_K_M, Q8_0, etc.) are **not** supported by ZenDNN and fall back to the standard AVX-512 CPU kernels.
4. For BF16 weights, ZenDNN supports both BF16 and FP32 output accumulation.
5. The matmul is dispatched through `zendnnl::lowoha::matmul::matmul_direct()`, which uses AMD's optimized AOCL DLP (Deep Learning Primitives) routines tuned for Zen architecture cache hierarchy.

This is why BF16 is the ideal format for your use case: ZenDNN accelerates the matmul (the dominant operation during inference), and Zen 5's native BF16 instructions handle the arithmetic efficiently.

## Why BF16 Instead of Quantized for CPU-Only ZenDNN

For CPU-only inference with ZenDNN, you have a meaningful choice:

| Approach | Backend Used for MUL_MAT | Model Size (7B) | Best For |
|----------|--------------------------|------------------|----------|
| BF16 model + ZenDNN | ZenDNN (optimized BLAS) | ~14 GB | Maximum quality, good throughput |
| Q4_K_M model (no ZenDNN needed) | Standard CPU + AVX-512 | ~4 GB | Maximum speed, smaller memory footprint |

With a 7B BF16 model (~14 GB), you are well within the memory capacity of the Ryzen AI Max 395 (up to 128 GB unified LPDDR5X). ZenDNN's optimized BF16 matmul on Zen 5 should provide competitive throughput.

However, if you find that CPU throughput with BF16 is not fast enough for your needs, consider that a quantized model (e.g., Q4_K_M) at ~4 GB will be significantly faster for token generation because it is memory-bandwidth bound -- smaller model means fewer bytes to read per token. The standard AVX-512 CPU backend handles quantized formats natively and does not need ZenDNN.

## Memory Bandwidth Consideration

Token generation speed is fundamentally limited by memory bandwidth:

```
tg_speed (approx) = memory_bandwidth / model_size
```

For a 7B BF16 model (~14 GB) on ~256 GB/s shared LPDDR5X:
```
theoretical max = 256 / 14 = ~18 tokens/s
```

For a 7B Q4_K_M model (~4 GB):
```
theoretical max = 256 / 4 = ~64 tokens/s
```

Real-world throughput is typically 60-70% of theoretical. So CPU-only BF16 at 7B will give you roughly 11-13 t/s for token generation, while Q4_K_M would give roughly 38-45 t/s. The BF16 advantage is in quality (no quantization loss) and prompt processing throughput (where ZenDNN's optimized BLAS shines with large matrix multiplications).

## Quick-Start Summary

```bash
# 1. Build
cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

# 2. Set runtime env
export ZENDNNL_MATMUL_ALGO=1

# 3. Run (CPU-only, BF16 model)
./build/bin/llama-cli -m your-7b-bf16-model.gguf \
  -ngl 0 \
  -t 16 \
  -b 2048 -ub 512

# 4. Benchmark to tune thread count
./build/bin/llama-bench -m your-7b-bf16-model.gguf \
  -ngl 0 \
  -t 8,12,16,24,32 \
  -p 512 -n 128 \
  -r 3
```

## No TTM or ROCm Needed

Since you are running CPU-only with no GPU offloading at all, you do **not** need to:
- Configure TTM memory limits (that is only needed for GPU memory access)
- Install ROCm or HIP
- Set `ROCBLAS_USE_HIPBLASLT=1` (that is a GPU-path optimization)
- Use `--no-mmap` (that workaround is specific to HIP's interaction with memory-mapped files)

Your build and runtime setup is simpler than the GPU-accelerated path. Just ZenDNN + native CPU optimizations.
