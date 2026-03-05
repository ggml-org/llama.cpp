# Running a 7B BF16 Model on Ryzen AI Max 395 (CPU-Only) with ZenDNN

## Overview

The Ryzen AI Max 395 (codename "Strix Halo") uses Zen 5 cores with full AVX-512 support, including AVX-512 BF16 instructions. This makes it well-suited for running BF16 models on the CPU. This fork of llama.cpp includes a **ZenDNN backend** that accelerates matrix multiplication operations using AMD's optimized libraries, which is exactly what you want for best CPU-only performance.

## What is ZenDNN?

ZenDNN (Zen Deep Neural Network Library) is AMD's high-performance deep learning inference library optimized for AMD Zen-based CPUs. In this llama.cpp fork, it provides an accelerated backend for **MUL_MAT** (matrix multiplication) operations -- the dominant compute kernel in transformer-based LLMs. Under the hood, ZenDNN uses its **LowOHA (Low Overhead Hardware Accelerated)** MatMul operator, which provides:

- Built-in weight caching
- Minimal execution overhead
- Access to backend libraries: AOCL DLP, LibXSMM, and OneDNN

The ZenDNN backend explicitly supports the Ryzen AI MAX (Strix Halo) and BF16 data type, which is the ideal combination for your setup.

## Step-by-Step Setup

### 1. Build llama.cpp with ZenDNN

The build system can automatically download and compile ZenDNN for you. No manual installation is required.

```bash
cd /home/user/llama.cpp

# Clean any previous build
rm -rf build

# Configure with ZenDNN enabled, no GPU backends
cmake -B build \
    -DGGML_ZENDNN=ON \
    -DGGML_VULKAN=OFF \
    -DGGML_HIP=OFF \
    -DGGML_OPENCL=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build (first build takes several minutes due to ZenDNN compilation)
cmake --build build --config Release -j $(nproc)
```

**Note:** The first build will take a while because CMake downloads and compiles ZenDNN and its dependencies (AOCL DLP, LibXSMM, OneDNN) from source. Subsequent rebuilds are fast.

### 2. Enable AVX-512 and Native CPU Optimization

By default, llama.cpp builds with `-march=native` (the `GGML_NATIVE` option defaults to ON). This automatically enables all AVX-512 extensions supported by your Zen 5 cores, including:

- AVX-512F, AVX-512CD, AVX-512BW, AVX-512DQ, AVX-512VL
- AVX-512 BF16 (hardware BFloat16 conversion and dot-product instructions)
- AVX-512 VNNI (vector neural network instructions)

You can verify this by checking the compiler flags in the CMake output. If for some reason `GGML_NATIVE` is not picking up everything, you can explicitly force the flags:

```bash
cmake -B build \
    -DGGML_ZENDNN=ON \
    -DGGML_NATIVE=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON \
    -DGGML_VULKAN=OFF \
    -DCMAKE_BUILD_TYPE=Release
```

### 3. Download a BF16 Model

You need a model in BF16 GGUF format. For example, a 7B-class model:

```bash
# Example: Llama 3.1 8B Instruct BF16
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct-GGUF \
    --local-dir models/

# Or any other 7B-class BF16 GGUF model you prefer
```

A 7B BF16 model requires approximately 14 GB of RAM. The Ryzen AI Max 395 typically comes with 64-128 GB of unified LPDDR5X memory, so this is not a concern.

### 4. Set Environment Variables for Best Performance

```bash
# Use the Blocked AOCL DLP algorithm (recommended for best performance)
export ZENDNNL_MATMUL_ALGO=1

# Set thread count to match your physical core count
# The Ryzen AI Max 395 has 16 Zen 5 cores
export OMP_NUM_THREADS=16
```

### 5. Run Inference

#### Using llama-server:

```bash
./build/bin/llama-server \
    -m models/your-7b-bf16-model.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -t 16 \
    -ngl 0
```

The `-ngl 0` flag explicitly ensures zero layers are offloaded to GPU, keeping everything on the CPU.

#### Using llama-cli for quick testing:

```bash
./build/bin/llama-cli \
    -m models/your-7b-bf16-model.gguf \
    -t 16 \
    -ngl 0 \
    -p "Hello, how are you?"
```

#### Benchmarking with llama-bench:

```bash
./build/bin/llama-bench \
    -m models/your-7b-bf16-model.gguf \
    -t 16 \
    -ngl 0
```

This will give you tokens/second measurements for both prompt processing and generation.

## Performance Tuning Tips

### Thread Count

- The Ryzen AI Max 395 has **16 Zen 5 cores / 32 threads** (with SMT).
- For LLM inference, using the **physical core count** (16 threads with `-t 16`) typically gives the best results. SMT (hyperthreading) threads rarely help and can hurt due to cache contention.
- Experiment with `-t 16` vs `-t 32` and benchmark to confirm.

### Memory Bandwidth

- The Strix Halo has **256-bit LPDDR5X** memory with substantial bandwidth (up to ~120 GB/s depending on configuration).
- BF16 models use half the memory bandwidth of FP32, which is a significant advantage for memory-bandwidth-bound token generation.
- Autoregressive token generation in LLMs is almost always memory-bandwidth-bound, so BF16 gives you close to 2x speedup over FP32 in this phase.

### ZenDNN Algorithm Selection

The `ZENDNNL_MATMUL_ALGO` environment variable controls which backend algorithm ZenDNN uses:

| Value | Algorithm | Notes |
|-------|-----------|-------|
| 1     | Blocked AOCL DLP | **Recommended** - best overall performance |

For additional algorithm options and environment variables, refer to the [ZenDNN runtime environment documentation](https://github.com/amd/ZenDNN).

### Batch Size for Prompt Processing

If you are processing long prompts, larger batch sizes improve throughput because matrix multiplications become more compute-bound (favoring ZenDNN). The default batch size of 512 is generally good. You can adjust with `-ub` (micro-batch size) and `-b` (batch size).

## How It Works Architecturally

When you build with `-DGGML_ZENDNN=ON`, the ZenDNN backend registers itself alongside the standard CPU backend:

1. **ZenDNN backend** handles `MUL_MAT` operations using optimized AMD-specific GEMM routines (via `zendnnl::lowoha::matmul::matmul_direct`).
2. **CPU backend** handles all other operations (attention, activations, normalization, etc.) using AVX-512 SIMD paths compiled via `-march=native`.
3. The two backends cooperate automatically -- no manual configuration needed to split work between them.

This means you get the best of both worlds: AMD-optimized matrix multiplication from ZenDNN, plus the standard llama.cpp CPU backend with full AVX-512 BF16 support for everything else.

## Verifying ZenDNN is Active

When you start llama-server or llama-cli, check the startup logs. You should see the ZenDNN backend being initialized. You can also look for backend information in the log output confirming that MUL_MAT operations are being routed to ZenDNN.

## Summary

| Setting | Value |
|---------|-------|
| Build flag | `-DGGML_ZENDNN=ON` |
| GPU offload | `-ngl 0` (none) |
| Thread count | `-t 16` (physical cores) |
| Environment | `ZENDNNL_MATMUL_ALGO=1` |
| Data type | BF16 GGUF model |
| CPU features used | AVX-512, AVX-512 BF16, AVX-512 VNNI (via `-march=native`) |

This configuration gives you optimized CPU-only inference leveraging both ZenDNN's tuned matrix multiplication and the Zen 5 AVX-512 instruction set for a 7B BF16 model on your Ryzen AI Max 395.
