# Running Llama 3.1 70B on Framework Desktop with AMD Ryzen AI Max+ 395

## Hardware Overview

The AMD Ryzen AI Max+ 395 (Strix Halo) is an exceptional platform for running large language models locally. Here's what makes it special:

- **GPU**: Integrated RDNA 3.5 GPU with 40 Compute Units
- **Memory**: Up to 128 GB unified LPDDR5X RAM shared between CPU and GPU
- **Memory Bandwidth**: ~256 GB/s (shared unified memory pool)
- **ROCm Support**: Fully supported via ROCm 6.x (gfx1151 architecture)

The key advantage here is the **unified memory architecture** — the GPU can access the full 128 GB pool, meaning Llama 3.1 70B (which requires ~40 GB in Q4 quantization) fits entirely in GPU-accessible memory without any CPU offloading.

---

## Step 1: Verify ROCm Installation

Before building, confirm your ROCm 6.3 installation is working correctly:

```bash
# Check ROCm installation
rocminfo | grep -E "Name|gfx"

# Verify HIP compiler
hipcc --version

# Check GPU is visible
/opt/rocm/bin/rocm-smi
```

For Strix Halo, you should see the GPU listed as `gfx1151` in rocminfo output.

---

## Step 2: Build llama.cpp with HIP/ROCm

Clone and build llama.cpp with ROCm support enabled:

```bash
cd ~/llama.cpp  # or wherever you have it

# Configure with CMake, targeting Strix Halo's GPU architecture
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="gfx1151" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP_UMA=ON

# Build with all available cores
cmake --build build --config Release -j $(nproc)
```

### Key Build Flags Explained

| Flag | Purpose |
|------|---------|
| `DGGML_HIP=ON` | Enables the ROCm/HIP GPU backend |
| `DAMDGPU_TARGETS="gfx1151"` | Targets Strix Halo's specific GPU architecture — avoids compiling for all AMD GPUs which takes much longer |
| `DCMAKE_BUILD_TYPE=Release` | Optimized release build |
| `DGGML_HIP_UMA=ON` | Enables Unified Memory Architecture mode — critical for Strix Halo since CPU and GPU share the same physical memory pool |

> **Important**: `GGML_HIP_UMA=ON` is especially important for Strix Halo. Without it, llama.cpp would try to allocate separate GPU memory and copy data between CPU/GPU memory regions, which doesn't make sense (and won't work well) on a unified memory platform. With UMA enabled, it directly uses the shared memory pool.

---

## Step 3: Download Llama 3.1 70B

Use a quantized version that balances quality and speed. For 128 GB of RAM, you have plenty of headroom.

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Download Q4_K_M (good balance of quality and speed)
huggingface-cli download \
    bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
    --include "Meta-Llama-3.1-70B-Instruct-Q4_K_M*.gguf" \
    --local-dir ~/models/llama-3.1-70b/
```

### Recommended Quantizations for Strix Halo

| Quantization | File Size | Quality | Notes |
|---|---|---|---|
| Q8_0 | ~74 GB | Excellent | Fits in 128 GB, near-lossless |
| Q6_K | ~57 GB | Very Good | Good quality/size tradeoff |
| Q4_K_M | ~40 GB | Good | Fast, widely used |
| Q4_K_S | ~38 GB | Good | Slightly smaller than K_M |
| IQ4_XS | ~36 GB | Good | Good perplexity for size |

With 128 GB unified memory, you can comfortably run **Q8_0** if you want near-original quality. Q4_K_M is the sweet spot for speed.

---

## Step 4: Run Inference — Optimal Settings

```bash
./build/bin/llama-cli \
    -m ~/models/llama-3.1-70b/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 8192 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    -p "You are a helpful assistant." \
    -i
```

### For Benchmarking / Speed Testing

```bash
./build/bin/llama-bench \
    -m ~/models/llama-3.1-70b/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    -n 128 \
    -p 512
```

---

## Step 5: Parameter Tuning for Best Performance

### Critical Parameters

**`-ngl 99` (Number of GPU Layers)**
Set this to 99 (or any large number). This offloads all model layers to the GPU. On Strix Halo with unified memory, the GPU can access all 128 GB, so you want every layer on the GPU for maximum throughput. Llama 3.1 70B has 80 transformer layers, so `-ngl 99` offloads everything.

**`-c` (Context Size)**
```bash
-c 8192    # Good starting point
-c 16384   # Larger context, uses more memory, slightly slower
-c 32768   # Large context — only use if needed
```
Larger context windows use more memory and can slow generation. Start with 8192 unless you need longer conversations.

**`-b` and `-ub` (Batch Size / Micro-Batch Size)**
These control prompt processing speed:
```bash
-b 512 -ub 512    # Default, good for most cases
-b 1024 -ub 512   # Larger batch, can speed up prompt ingestion
```

**`--threads` / `-t` (CPU Threads)**
With `-ngl 99`, most work is on the GPU, but setting this for the remaining CPU work:
```bash
-t 16    # Use half the CPU threads for CPU-side work
```

### Full Optimized Command

```bash
./build/bin/llama-cli \
    -m ~/models/llama-3.1-70b/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 8192 \
    -b 512 \
    -ub 512 \
    -t 16 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    --color \
    -i \
    -p "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
```

---

## Step 6: Environment Variables for Performance

Set these before running llama.cpp for better ROCm performance:

```bash
# Use the integrated GPU (device 0 on Strix Halo)
export HIP_VISIBLE_DEVICES=0

# Tune for Strix Halo's memory subsystem
export GPU_MAX_HW_QUEUES=4

# Optional: increase async queue depth
export AMD_SERIALIZE_KERNEL=0

# Then run llama.cpp
./build/bin/llama-cli ...
```

---

## Expected Performance

On the Ryzen AI Max+ 395 with Q4_K_M Llama 3.1 70B, you can expect approximately:

- **Prompt processing (PP)**: 300-600 tokens/sec
- **Token generation (TG)**: 8-15 tokens/sec

The memory bandwidth of ~256 GB/s is the primary bottleneck for token generation with quantized models (it's a memory-bandwidth-bound workload). This is still significantly faster than CPU-only inference and comparable to discrete mid-range GPUs.

### Improving Generation Speed

1. **Use smaller quantizations** (Q4_K_S, IQ4_XS) — less data to transfer per token
2. **Reduce context size** — larger KV cache uses more memory bandwidth
3. **Try Q2_K or IQ2_XXS** — if quality is less important, much faster generation
4. **Flash attention** (if supported in your build): add `--flash-attn` flag

```bash
# Flash attention (if available) — reduces KV cache memory footprint
./build/bin/llama-cli -m model.gguf -ngl 99 --flash-attn -c 8192 ...
```

---

## Troubleshooting

### GPU Not Detected
```bash
# Check that ROCm sees the GPU
rocm-smi
rocminfo | grep "Device Type"

# Ensure your user is in the render/video groups
sudo usermod -aG render,video $USER
# Then log out and back in
```

### Build Fails with Missing HIP
```bash
# Add ROCm to your PATH if not already done
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Out of Memory Errors
If you see OOM errors (unlikely with 128 GB but possible with large context):
```bash
# Reduce context size
-c 4096

# Or use a smaller quantization
# Switch from Q4_K_M to Q4_K_S or IQ4_XS
```

### Slow Performance / GPU Not Being Used
Check that the build correctly detected HIP:
```bash
# Look for HIP in the cmake output
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1151" -DGGML_HIP_UMA=ON . 2>&1 | grep -i hip

# Verify GPU layers are being loaded at runtime
./build/bin/llama-cli -m model.gguf -ngl 99 ...
# Should show "llm_load_tensors: offloaded XX/XX layers to GPU"
```

---

## Summary

The Ryzen AI Max+ 395 with 128 GB unified memory is an excellent platform for Llama 3.1 70B. The key points:

1. Build with `-DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1151" -DGGML_HIP_UMA=ON`
2. Use `-ngl 99` to offload all layers to the GPU
3. Use Q4_K_M for speed, or Q8_0 if you want near-original quality (fits in 128 GB)
4. The unified memory architecture means no separate VRAM limit — the full 128 GB is available to the GPU
5. Expect ~8-15 tokens/sec generation speed for 70B Q4_K_M

The `GGML_HIP_UMA=ON` flag is what unlocks the full potential of this platform by properly handling the unified memory architecture.
