# Running a 7B BF16 Model on Ryzen AI Max 395 -- CPU-Only with ZenDNN

Great choice of hardware. The Ryzen AI Max 395 (Strix Halo) has Zen 5 cores with full AVX-512 support including AVX-512 BF16, which makes it one of the best consumer CPUs for BF16 inference. Here is how to set it up for maximum CPU-only performance.

## What ZenDNN Actually Does

ZenDNN is AMD's optimized primitives library for Zen CPUs. In llama.cpp, it accelerates the matrix multiplication operations (MUL_MAT) specifically for FP32 and BF16 data types. Since you are running a BF16 model, ZenDNN is the ideal accelerator -- it uses AMD's AOCL-DLP (AMD Optimized CPU Libraries) internally, which contains hand-tuned matmul kernels for Zen 5 that outperform the generic AVX-512 paths.

Importantly, ZenDNN does **not** help with quantized models (Q4, Q5, Q8). Those use the standard GGML CPU backend with AVX-512 VNNI intrinsics. ZenDNN is specifically valuable for your use case: unquantized BF16 inference.

## Step 1: Build llama.cpp with ZenDNN

The build system can automatically download ZenDNN for you:

```bash
cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

Key flags explained:

- **`GGML_ZENDNN=ON`** -- Enables the ZenDNN backend, which provides optimized BF16 matmul kernels for Zen CPUs. CMake will automatically download and build ZenDNN if it is not already installed on your system.
- **`GGML_NATIVE=ON`** -- Tells the compiler to auto-detect and enable all CPU features available on your machine. On the Ryzen AI Max 395, this enables AVX-512F, AVX-512CD, AVX-512VL, AVX-512DQ, AVX-512BW, AVX-512 VNNI, and AVX-512 BF16. This is simpler and more reliable than setting individual flags like `-DGGML_AVX512=ON -DGGML_AVX512_BF16=ON`, etc.
- **`Release`** -- Enables compiler optimizations (-O2/-O3).

You do **not** need `GGML_HIP=ON`, `GGML_VULKAN=ON`, or any GPU-related flags since you want CPU-only.

### Alternative: Custom ZenDNN Installation

If you have already installed ZenDNN separately (e.g., from AMD's release packages), point CMake to it:

```bash
export ZENDNN_ROOT=/path/to/ZenDNN/build/install

cmake -S . -B build \
  -DGGML_ZENDNN=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

### Verify the Build

After building, confirm AVX-512 support is active:

```bash
./build/bin/llama-cli --version
# Look for: AVX512 = 1 in the feature flags
```

## Step 2: Set the ZenDNN Runtime Environment Variable

This is easy to miss but matters for performance:

```bash
export ZENDNNL_MATMUL_ALGO=1
```

This selects the "Blocked AOCL DLP" algorithm, which is the fastest matmul strategy available in ZenDNN for Zen 5. Without this, ZenDNN may default to a slower algorithm.

Add it to your `~/.bashrc` for persistence:

```bash
echo 'export ZENDNNL_MATMUL_ALGO=1' >> ~/.bashrc
```

## Step 3: Run Inference

```bash
export ZENDNNL_MATMUL_ALGO=1

./build/bin/llama-cli -m your-7b-bf16-model.gguf \
  -ngl 0 \
  -t 16 \
  -b 2048 -ub 512
```

Parameters explained:

- **`-ngl 0`** -- Zero GPU layers. All computation stays on the CPU.
- **`-t 16`** -- Use 16 threads, matching the physical core count of the Ryzen AI Max 395. This is the sweet spot. Performance scales well up to the physical core count but **degrades** with hyperthreading (using 32 threads is slower than 16 because of resource contention and cache thrashing).
- **`-b 2048 -ub 512`** -- Batch size settings for prompt processing. `-b` is the logical batch size (how many tokens per batch), `-ub` is the physical/computation batch size. These defaults work well for the 395's large L3 cache and memory bandwidth.

## Step 4: Benchmark to Confirm Performance

Run a benchmark to validate everything is working optimally:

```bash
export ZENDNNL_MATMUL_ALGO=1

./build/bin/llama-bench -m your-7b-bf16-model.gguf \
  -ngl 0 \
  -t 4,8,12,16 \
  -p 512 -n 128 \
  -r 3
```

This sweeps thread counts so you can confirm that 16 threads is optimal on your specific system. Watch the `pp` (prompt processing) and `tg` (token generation) numbers. If you see performance plateau at 12 threads or drop at 16, adjust your `-t` accordingly.

## Performance Expectations and Considerations

### Memory Bandwidth

A 7B BF16 model is approximately 14GB in memory (7 billion parameters at 2 bytes each). Token generation speed is memory-bandwidth-bound. The Ryzen AI Max 395 has approximately 256 GB/s of LPDDR5X bandwidth (shared between CPU and GPU). Since you are running CPU-only, the CPU gets full use of this bandwidth, which is favorable.

Rough theoretical estimate for token generation:
```
tg_speed ~ 256 GB/s / 14 GB ~ 18 tokens/s theoretical max
```
Real-world will be lower (60-70% of theoretical) due to overhead, so expect roughly 11-13 t/s for token generation. Prompt processing will be faster since it is compute-bound and benefits heavily from ZenDNN's optimized BF16 matmul and Zen 5's `_mm512_dpbf16_ps` instruction (AVX-512 BF16), which processes 32 BF16 values per operation.

### Why BF16 + ZenDNN Is a Good Match for Zen 5

Zen 5 has native AVX-512 BF16 support via the `DPBF16PS` instruction. This instruction computes a dot product of BF16 pairs and accumulates into FP32, giving you 2x the throughput of FP32 operations. ZenDNN's matmul kernels are specifically tuned to exploit this instruction along with Zen 5's improved front-end and wider execution resources.

### When You Might Consider Quantization Instead

If token generation speed is more important to you than maintaining full BF16 precision, a quantized model (e.g., Q4_K_M at ~4GB) would give much faster inference since the model is 3-4x smaller and therefore uses proportionally less memory bandwidth. Quantized models do not use ZenDNN -- they run on the standard GGML CPU backend with AVX-512 VNNI kernels, which are also well-optimized on Zen 5. You would not need the `GGML_ZENDNN=ON` flag for quantized models; just `GGML_NATIVE=ON` is sufficient.

## Quick Reference

| Setting | Value |
|---------|-------|
| Build flags | `GGML_ZENDNN=ON`, `GGML_NATIVE=ON`, `Release` |
| Environment variable | `ZENDNNL_MATMUL_ALGO=1` |
| GPU layers (`-ngl`) | `0` |
| Thread count (`-t`) | `16` (physical core count) |
| Batch sizes | `-b 2048 -ub 512` |
| GPU flags needed | None |
| ROCm required | No |

## Summary

1. Build with `GGML_ZENDNN=ON` and `GGML_NATIVE=ON` to get ZenDNN BF16 matmul and full AVX-512 support.
2. Set `ZENDNNL_MATMUL_ALGO=1` at runtime for the fastest matmul algorithm.
3. Run with `-ngl 0 -t 16` to use all physical cores, no GPU.
4. Benchmark with `llama-bench` to verify performance on your specific system.

This gives you the best CPU-only BF16 inference path on Zen 5 without touching the GPU at all.
