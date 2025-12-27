# Dynamic VRAM Allocation for Vulkan Backend

This document describes the dynamic VRAM allocation heuristic for `llama.cpp`'s Vulkan backend, which automatically optimizes GPU layer offloading based on available VRAM.

## Overview

The Vulkan backend now includes a **dynamic heuristic** that automatically calculates the optimal number of GPU layers to offload based on:
- Available VRAM on your GPU
- Model size and layer count (from GGUF metadata)
- Reserved overhead for KV cache and compute buffers

This enables **optimal performance** on low-VRAM devices (like AMD RX 6500 XT with 4GB) without manual configuration or OOM errors.

## How It Works

When you run `llama-cli` or `llama-server` **without** specifying `-ngl` (or with `-ngl -1`), the heuristic:

1. **Queries available VRAM** from your Vulkan device
2. **Parses model metadata** to determine model size and layer count
3. **Reserves overhead** (800MB) for KV cache, compute buffers, and system
4. **Calculates optimal layers**: `(available_vram - overhead) / bytes_per_layer`
5. **Offloads automatically** without risking OOM

### Example Results

**AMD RX 6500 XT (4GB VRAM)**:
- Gemma 2B (1.6GB): **26/27 layers** offloaded → **2.5-3.1x faster**
- Llama 3.2 3B (1.9GB): **28/29 layers** offloaded → **~2x faster**
- Llama 2 7B (3.9GB): **21/33 layers** offloaded → **1.6x faster**
- Llama 2 13B (7.5GB): **14/41 layers** offloaded → **No OOM** ✅

## Usage

### Automatic (Recommended)

Simply run without `-ngl` to enable the dynamic heuristic:

```bash
# Heuristic calculates optimal layers automatically
llama-cli -m models/gemma-2b-q4.gguf -p "Hello"
```

The heuristic will print debug info showing the calculation:
```
Vulkan dynamic heuristic: available_vram=3434 MB, model_size=1623 MB, 
n_layers=27, overhead=800 MB, calculated_layers=26
```

### Manual Override

You can still manually specify layers to override the heuristic:

```bash
# Force specific number of layers
llama-cli -m models/gemma-2b-q4.gguf -p "Hello" -ngl 20

# Force CPU-only
llama-cli -m models/gemma-2b-q4.gguf -p "Hello" -ngl 0
```

## Performance

Compared to CPU-only (`-ngl 0`), the dynamic heuristic provides:

**Gemma 2B Q4_K_M on AMD RX 6500 XT**:
- Prompt processing: **2.5x faster** (497 → 1231 t/s)
- Token generation: **3.1x faster** (19.4 → 60.4 t/s)

## Troubleshooting

### Still Getting OOM Errors?

If you encounter "Out of Device Memory" errors despite the heuristic:

1. **Reduce context size**: Use `-c 2048` or lower
2. **Force fewer layers**: Use `-ngl 10` or lower
3. **Check available VRAM**: Close other GPU applications
4. **Use smaller model**: Try a smaller quantization (Q4_K_M → Q3_K_S)

### Heuristic Not Triggering?

The heuristic only activates when:
- ✅ Vulkan backend is enabled (`GGML_USE_VULKAN=1` during build)
- ✅ `-ngl` is not specified (or set to `-1`)
- ✅ GGUF file can be parsed for metadata

If you explicitly set `-ngl`, the heuristic is bypassed.

## Technical Details

### Overhead Calculation

The heuristic reserves **800MB** for:
- KV cache (dynamically allocated by llama.cpp)
- Compute buffers (temporary tensors during inference)
- System overhead (driver, fragmentation)

This value is conservative and works well across different model sizes.

### Model Compatibility

The heuristic generalizes across model architectures by searching for:
- `*.block_count` (layer count)
- `*.embedding_length` (model dimensions)

Tested architectures:
- ✅ Gemma / Gemma 2
- ✅ Llama / Llama 2 / Llama 3
- ✅ Qwen / Qwen 2.5

## Benchmark Script

The `tests/6500xt_benchmark.ps1` script automates testing across different configurations:

```powershell
cd tests
.\6500xt_benchmark.ps1
```

This will test CPU-only vs GPU heuristic and report performance improvements.
