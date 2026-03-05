# Running a 7B BF16 Model on Ryzen AI Max 395 (CPU-Only) with ZenDNN

## Hardware Overview

The Ryzen AI Max 395 (codename "Strix Halo") features Zen 5 CPU cores with full AVX-512 support, including the AVX-512 BF16 extension. This is important because your BF16 model can take advantage of native BF16 instructions for matrix multiplication. The chip has 16 Zen 5 cores (32 threads) and a large unified LPDDR5X memory pool, which is more than sufficient for a 7B BF16 model (~14 GB).

## What Is ZenDNN and Why Use It?

ZenDNN (Zen Deep Neural Network Library) is AMD's optimized deep learning inference library for AMD CPUs. In llama.cpp, it exists as a **separate backend** (not part of the CPU backend) that accelerates **matrix multiplication (MUL_MAT) operations** specifically. It uses AMD's LowOHA (Low Overhead Hardware Accelerated) MatMul operator, which internally leverages AOCL DLP, LibXSMM, and OneDNN for highly optimized GEMM kernels tuned for Zen architectures.

Key facts about the ZenDNN backend in this codebase:

- It accelerates **only MUL_MAT** operations. All other operations (attention, normalization, activations, etc.) fall back to the standard CPU backend.
- It supports **FP32 and BF16** data types. BF16 is explicitly noted as providing best performance on Zen 4/Zen 5.
- It does **not** support quantized formats (Q4_0, Q8_0, etc.) -- only full-precision FP32 and BF16.
- The Ryzen AI Max (Strix Halo) is explicitly listed as a supported processor in the ZenDNN documentation.

Since you are running a BF16 model, ZenDNN is a strong match. The combination of native AVX-512 BF16 instructions on Zen 5 plus ZenDNN's optimized GEMM kernels should deliver the best CPU-only MUL_MAT throughput.

## Build Instructions

### Option 1: ZenDNN with Automatic Download (Recommended)

The simplest approach -- CMake will automatically download, build, and link ZenDNN:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_ZENDNN=ON \
    -DGGML_NATIVE=ON

cmake --build build --config Release -j $(nproc)
```

The key flags:

| Flag | Purpose |
|------|---------|
| `-DGGML_ZENDNN=ON` | Enables the ZenDNN backend, which intercepts MUL_MAT ops for BF16/FP32 |
| `-DGGML_NATIVE=ON` | Compiles the CPU backend with `-march=native`, which on Zen 5 automatically enables AVX-512, AVX-512 BF16, AVX-512 VNNI, and all other supported instruction sets |
| `-DCMAKE_BUILD_TYPE=Release` | Optimization level -O3 for best performance |

`GGML_NATIVE=ON` is the default when building natively on the target machine, so technically you may omit it. But it is worth being explicit. When set, the CPU backend CMakeLists.txt (at `ggml/src/ggml-cpu/CMakeLists.txt`) passes `-march=native` to the compiler, which on your Zen 5 chip will automatically enable:
- AVX, AVX2, FMA
- AVX-512F, AVX-512CD, AVX-512VL, AVX-512DQ, AVX-512BW
- AVX-512 VBMI, AVX-512 VNNI, AVX-512 BF16

The first build with ZenDNN auto-download will take several minutes because it clones and builds ZenDNN and its dependencies (AOCL DLP, LibXSMM, OneDNN, etc.).

### Option 2: Manual ZenDNN Installation

If you want more control over the ZenDNN version:

```bash
# Build ZenDNN separately
git clone https://github.com/amd/ZenDNN.git
cd ZenDNN
mkdir build && cd build
cmake ..
cmake --build . --target all
cd ../..

# Build llama.cpp pointing to your ZenDNN
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_ZENDNN=ON \
    -DZENDNN_ROOT=/path/to/ZenDNN/build/install \
    -DGGML_NATIVE=ON

cmake --build build --config Release -j $(nproc)
```

### Ensuring No GPU Backend Is Used

By default, llama.cpp does not enable GPU backends unless you explicitly pass flags like `-DGGML_VULKAN=ON` or `-DGGML_HIP=ON`. So a plain build with only `-DGGML_ZENDNN=ON` will be CPU-only. But to be absolutely certain:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_ZENDNN=ON \
    -DGGML_NATIVE=ON \
    -DGGML_VULKAN=OFF \
    -DGGML_HIP=OFF \
    -DGGML_OPENCL=OFF
```

## Runtime Configuration

### Environment Variables

Set the ZenDNN matrix multiplication algorithm before running:

```bash
export ZENDNNL_MATMUL_ALGO=1    # Blocked AOCL DLP algorithm (recommended for best performance)
```

This selects the blocked AOCL DLP algorithm inside ZenDNN's LowOHA MatMul, which the documentation identifies as the optimal choice.

For additional ZenDNN environment variables (logging, profiling, etc.), consult the [ZenDNN runtime environment documentation](https://github.com/amd/ZenDNN/blob/a18adf8c605fb5f5e52cefd7eda08a7b18febbaf/docs/runtime_env.md).

### Running Inference

```bash
# Set the algorithm
export ZENDNNL_MATMUL_ALGO=1

# Run with llama-cli (interactive)
./build/bin/llama-cli \
    -m /path/to/your-7b-model.BF16.gguf \
    -t 16 \
    -p "Your prompt here"

# Or run the server
./build/bin/llama-server \
    -m /path/to/your-7b-model.BF16.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -t 16
```

### Thread Count (`-t`)

The Ryzen AI Max 395 has 16 Zen 5 cores with SMT (32 threads). For inference workloads:

- **Start with `-t 16`** (one thread per physical core). This is typically optimal because LLM inference is memory-bandwidth-bound and SMT provides diminishing returns.
- If you see improved throughput, try `-t 32` (all hardware threads), but expect marginal gains at most.
- Benchmark with `llama-bench` to find the sweet spot:

```bash
export ZENDNNL_MATMUL_ALGO=1
./build/bin/llama-bench \
    -m /path/to/your-7b-model.BF16.gguf \
    -t 16
```

## How It Works Under the Hood

When you build with `-DGGML_ZENDNN=ON`, a separate backend library (`ggml-zendnn`) is compiled. At runtime, llama.cpp's backend scheduling system works as follows:

1. **ZenDNN backend** claims all `MUL_MAT` operations where both operands are FP32 or BF16 and tensors are contiguous.
2. **CPU backend** handles everything else (RMS norm, RoPE, softmax, element-wise ops, KV cache management, etc.) using AVX-512-optimized code paths compiled via `-march=native`.

This is a cooperative arrangement. ZenDNN does not replace the CPU backend; it supplements it for the most computationally expensive operation (matrix multiplication), which dominates transformer inference time.

## Memory Considerations

A 7B parameter model in BF16 format requires approximately 14 GB of memory. The Ryzen AI Max 395 supports up to 128 GB of unified LPDDR5X, so memory capacity is not a concern. The LPDDR5X bandwidth (up to ~256 GB/s in 8-channel configurations) is also favorable for memory-bound LLM inference.

## Performance Expectations

With ZenDNN + AVX-512 BF16 on the Ryzen AI Max 395, you should see:

- **Prompt processing (prefill)**: This is compute-bound, and ZenDNN's optimized BF16 GEMM kernels will provide the most benefit here. The ZenDNN documentation notes 1.1x-2x speedup compared to standard CPU inference for matrix multiplication operations.
- **Token generation (decode)**: This is largely memory-bandwidth-bound (single-token batch size). ZenDNN will still handle the MUL_MAT, but gains may be more modest since bandwidth is the bottleneck, not compute.

## Summary Checklist

1. Build with `cmake -B build -DGGML_ZENDNN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`
2. Set `export ZENDNNL_MATMUL_ALGO=1` before running
3. Use a BF16 GGUF model file (not quantized, since ZenDNN only supports FP32/BF16)
4. Run with `-t 16` (physical core count)
5. Benchmark with `llama-bench` to validate performance
6. Do NOT enable `-DGGML_VULKAN=ON`, `-DGGML_HIP=ON`, or any GPU flags if you want pure CPU execution

## Relevant Source Files

- ZenDNN backend implementation: `ggml/src/ggml-zendnn/ggml-zendnn.cpp`
- ZenDNN CMake build: `ggml/src/ggml-zendnn/CMakeLists.txt`
- ZenDNN header: `ggml/include/ggml-zendnn.h`
- ZenDNN documentation: `docs/backend/ZenDNN.md`
- CPU backend CMake (AVX-512 flags): `ggml/src/ggml-cpu/CMakeLists.txt`
- Top-level GGML CMake (GGML_ZENDNN option): `ggml/CMakeLists.txt`
