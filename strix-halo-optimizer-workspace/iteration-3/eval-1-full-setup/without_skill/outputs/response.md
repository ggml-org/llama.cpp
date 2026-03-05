# Running Llama 3.1 70B on AMD Ryzen AI Max+ 395 with llama.cpp

## Hardware Overview

The AMD Ryzen AI Max+ 395 (codenamed "Strix Halo") is a unified memory architecture (UMA) processor that combines:

- **CPU**: Zen 5 cores (16C/32T) with AVX-512 support
- **iGPU**: Radeon 890M (RDNA 3.5), 40 Compute Units, identified as `gfx1150` in ROCm
- **Memory**: 128GB unified LPDDR5X shared between CPU and iGPU

The unified memory is the key advantage here. Unlike discrete GPUs with limited VRAM, the iGPU can access the full 128GB, which is more than enough for Llama 3.1 70B in quantized formats. Since CPU and GPU share the same physical memory, there is no PCIe transfer penalty for GPU offload.

## Model Selection

Llama 3.1 70B has 70 billion parameters. At different quantization levels, the model sizes are approximately:

| Quantization | Model Size | Quality    |
|-------------|-----------|------------|
| Q4_K_M      | ~40 GB    | Good       |
| Q5_K_M      | ~48 GB    | Very Good  |
| Q6_K        | ~56 GB    | Excellent  |
| Q8_0        | ~70 GB    | Near-FP16  |

With 128GB of RAM, you can comfortably run Q4_K_M through Q8_0. For the best balance of speed and quality, **Q4_K_M is recommended** since token generation speed on an iGPU is primarily memory-bandwidth-bound, and the smaller the model, the faster it generates tokens.

Download a GGUF quantized model from Hugging Face. For example:

```bash
# Install huggingface-hub CLI if needed
pip install huggingface-hub

# Download Q4_K_M quantization (recommended for speed)
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
    Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --local-dir models/
```

## Build Option 1: ROCm/HIP Backend (Recommended)

The HIP backend uses your Radeon 890M iGPU via ROCm. Since you already have ROCm 6.3 installed, this is the most straightforward path to GPU-accelerated inference.

### Build Steps

```bash
cd /path/to/llama.cpp

# Clean any previous build
rm -rf build

# Configure with HIP backend targeting RDNA 3.5
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="gfx1150" \
    -DCMAKE_BUILD_TYPE=Release

# Build with parallel jobs (adjust -j to your core count)
cmake --build build --config Release -- -j 16
```

### Key Build Flags Explained

| Flag | Purpose |
|------|---------|
| `GGML_HIP=ON` | Enables the ROCm/HIP GPU backend |
| `GPU_TARGETS="gfx1150"` | Compiles kernels specifically for RDNA 3.5 (Radeon 890M). Without this, it builds for all detected GPU architectures, which dramatically increases compile time and produces less optimized code. |
| `CMAKE_BUILD_TYPE=Release` | Enables compiler optimizations |

### Running Inference

```bash
# Enable unified memory (critical for iGPU - lets the GPU access all system RAM)
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

# Run with all layers offloaded to GPU
./build/bin/llama-cli \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    -t 4 \
    --flash-attn \
    -cnv
```

### Runtime Flags Explained

| Flag | Purpose |
|------|---------|
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` | **Critical.** Enables unified memory so the iGPU can allocate from system RAM. Without this, the iGPU is limited to its small dedicated VRAM allocation and the model will fail to load. |
| `-ngl 99` | Offload all model layers to the GPU. On a UMA system there is no memory transfer cost, so full offload is optimal. |
| `-c 4096` | Context size of 4096 tokens. Increase if you need longer conversations, but larger contexts use more memory and slow down generation. |
| `-t 4` | Number of CPU threads for any operations that remain on CPU. With full GPU offload, this mainly affects prompt processing of small batches. A moderate thread count (4-8) is usually sufficient. |
| `--flash-attn` | Enables Flash Attention, which reduces memory usage for the KV cache and improves performance, especially at longer context lengths. |
| `-cnv` | Enables conversation mode for interactive chat. |

## Build Option 2: Vulkan Backend (Alternative)

Vulkan is an alternative GPU backend that does not require ROCm. It can work well on RDNA 3.5 and may be simpler to set up if you encounter ROCm issues.

### Build Steps

```bash
# Install Vulkan development packages if not already present
sudo apt-get install libvulkan-dev glslc

# Build with Vulkan
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j 16
```

### Running with Vulkan

```bash
./build/bin/llama-cli \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    -t 4 \
    --flash-attn \
    -cnv
```

The Vulkan backend automatically detects unified memory on integrated GPUs. No environment variable is needed.

## Build Option 3: CPU-Only (Fallback)

If you want to avoid GPU complexities, the Zen 5 CPU with AVX-512 support can run inference directly. This will be slower for token generation but can be competitive for prompt processing with enough threads.

```bash
cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j 16

./build/bin/llama-cli \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -c 4096 \
    -t 16 \
    -cnv
```

With CPU-only, use more threads (e.g., `-t 16` or higher, up to your physical core count) since all computation happens on the CPU.

## Performance Tuning Tips

### 1. Benchmark Before and After Changes

Use `llama-bench` to measure performance systematically:

```bash
# Benchmark token generation and prompt processing with HIP
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-bench \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -fa 1 \
    -t 4
```

This reports tokens per second for both prompt processing (pp) and token generation (tg).

### 2. KV Cache Quantization

To reduce memory usage and potentially improve speed at longer context lengths, quantize the KV cache:

```bash
./build/bin/llama-cli \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 8192 \
    -ctk q8_0 -ctv q8_0 \
    --flash-attn \
    -cnv
```

The `-ctk q8_0 -ctv q8_0` flags quantize the key and value caches to 8-bit, cutting KV cache memory in half compared to the default f16 with minimal quality loss. This is especially beneficial when using larger context sizes.

### 3. Context Size vs Speed Tradeoff

Token generation speed decreases as the context fills up. If you do not need the full 128K context that Llama 3.1 supports, use a smaller value:

- `-c 4096`: Fast, suitable for most conversations
- `-c 8192`: Good balance for longer documents
- `-c 16384`: For tasks requiring more context
- `-c 32768`+: Only when you genuinely need it; speed will drop noticeably

### 4. Using the Server for Continuous Use

For ongoing use, the server mode provides an OpenAI-compatible HTTP API:

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-server \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    -t 4 \
    --flash-attn \
    --host 0.0.0.0 \
    --port 8080
```

You can then use any OpenAI-compatible client to connect to `http://localhost:8080`.

### 5. Thread Count Optimization

With full GPU offload (`-ngl 99`), most computation is on the GPU and fewer CPU threads are needed. However, some operations (prompt processing, token sampling) still use the CPU. Experiment with `-t` values:

- Start with `-t 4` for full GPU offload
- Try `-t 8` and `-t 16` to see if prompt processing improves
- More threads are not always better due to NUMA effects and memory bandwidth contention

### 6. HSA_OVERRIDE_GFX_VERSION

If ROCm does not properly detect your Radeon 890M (gfx1150), you can force the GFX version:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.0
```

This tells ROCm to treat the GPU as gfx1150. Try this if you get errors about unsupported GPU architectures at runtime.

## Expected Performance

On the Ryzen AI Max+ 395 with 128GB LPDDR5X and Llama 3.1 70B Q4_K_M with full GPU offload:

- **Token generation**: Expect roughly 8-14 tokens/second, depending on context length and quantization. The iGPU's memory bandwidth (up to ~256 GB/s for LPDDR5X-8533) is the primary bottleneck for token generation, since each token requires reading the entire model from memory.
- **Prompt processing**: Several hundred tokens/second, leveraging the GPU's parallel compute capability.

These numbers are approximate and depend on your specific LPDDR5X speed, ROCm driver version, and llama.cpp version. Always benchmark with `llama-bench` on your actual hardware.

## Quick Start Summary

For the fastest path to running Llama 3.1 70B:

```bash
# 1. Build with ROCm/HIP
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="gfx1150" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -- -j 16

# 2. Run inference
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    ./build/bin/llama-cli \
    -m models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    -t 4 \
    --flash-attn \
    -cnv
```
