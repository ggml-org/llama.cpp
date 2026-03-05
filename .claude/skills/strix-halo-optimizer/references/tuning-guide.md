# Performance Tuning Guide: llama.cpp on Strix Halo

## Table of Contents
- [Environment Setup](#environment-setup)
- [Finding the Optimal GPU Layer Count](#finding-the-optimal-gpu-layer-count)
- [Batch Size Tuning](#batch-size-tuning)
- [KV Cache Optimization](#kv-cache-optimization)
- [Thread Count Tuning](#thread-count-tuning)
- [Memory Bandwidth Optimization](#memory-bandwidth-optimization)
- [Model Quantization Selection](#model-quantization-selection)
- [Server Configuration](#server-configuration)
- [Advanced: Tensor Overrides](#advanced-tensor-overrides)

## Environment Setup

Before any benchmarking, set these environment variables:
```bash
# Critical: Use hipBLASLt for 2-3x better prompt processing
export ROCBLAS_USE_HIPBLASLT=1

# Optional: Pin to specific GPU if multiple are present
export HIP_VISIBLE_DEVICES=0
```

## Finding the Optimal GPU Layer Count

The `-ngl` (number of GPU layers) parameter has the biggest impact on performance. For Strix Halo with unified memory, the cost of partial offloading is lower than discrete GPUs (no PCIe transfers), but GPU computation is still faster than CPU for most operations.

### Systematic Benchmarking
```bash
export ROCBLAS_USE_HIPBLASLT=1

# Sweep across GPU layer counts
./build/bin/llama-bench -m model.gguf \
  -ngl 0,10,20,30,40,50,60,99 \
  --no-mmap \
  -t 4 \
  -p 512 -n 128 \
  -r 3 \
  -o csv > ngl_sweep.csv
```

### What to Expect
- `-ngl 0` (CPU only): Baseline. Good prompt processing with AVX-512, moderate token generation.
- `-ngl 99` (full GPU): Best token generation speed. Prompt processing excellent with hipBLASLt.
- Intermediate values: Diminishing returns. Usually either full CPU or full GPU wins.

For most models on Strix Halo, **`-ngl 99` is optimal**. The unified memory means the GPU can access all layers without transfer overhead, and the 40 CUs provide good compute throughput. The main exception is when you're running other GPU workloads simultaneously.

## Batch Size Tuning

Two batch size parameters control prompt processing throughput:

- `-b` (logical batch size): How many tokens are processed per batch. Default: 2048.
- `-ub` (physical batch size): Actual computation batch size. Default: 512.

```bash
# Sweep batch sizes
./build/bin/llama-bench -m model.gguf \
  -ngl 99 --no-mmap \
  -b 512,1024,2048,4096 \
  -ub 128,256,512 \
  -t 4 \
  -p 2048 -n 128 \
  -r 3 -o csv > batch_sweep.csv
```

Larger batch sizes generally improve prompt processing speed but increase memory usage. For Strix Halo with ample unified memory, `-b 2048 -ub 512` is a good default. If you have 128GB and are processing long prompts, try `-b 4096`.

## KV Cache Optimization

The KV cache stores attention key/value pairs and grows with context length. Quantizing it saves significant memory:

| Cache Type | Memory per token (per layer) | Quality Impact |
|-----------|------------------------------|----------------|
| f16 (default) | 2 bytes/element | None |
| q8_0 | 1 byte/element | Negligible |
| q4_0 | 0.5 bytes/element | Slight degradation on long contexts |

```bash
# Compare KV cache quantization impact
./build/bin/llama-bench -m model.gguf \
  -ngl 99 --no-mmap \
  -ctk f16 -ctv f16 \
  -p 512 -n 128 -r 3

./build/bin/llama-bench -m model.gguf \
  -ngl 99 --no-mmap \
  -ctk q8_0 -ctv q8_0 \
  -p 512 -n 128 -r 3
```

**Recommendation**: Use `-ctk q8_0 -ctv q8_0` by default. The quality impact is negligible for most use cases, and the memory savings let you run larger models or longer contexts. Use q4_0 only when memory is truly constrained.

## Thread Count Tuning

On Strix Halo, CPU and GPU share memory bandwidth (~256 GB/s). Too many CPU threads during GPU inference creates bandwidth contention.

### For GPU-Primary Inference
```bash
# Sweep thread counts with full GPU offload
./build/bin/llama-bench -m model.gguf \
  -ngl 99 --no-mmap \
  -t 1,2,4,8,16 \
  -p 512 -n 128 \
  -r 3 -o csv > thread_sweep.csv
```

**Expected result**: 2-4 threads is optimal for GPU-primary. More threads can actually slow things down due to bandwidth contention.

### For CPU-Primary Inference
```bash
# Sweep thread counts with CPU-only
./build/bin/llama-bench -m model.gguf \
  -ngl 0 \
  -t 4,8,12,16,24,32 \
  -p 512 -n 128 \
  -r 3 -o csv > cpu_thread_sweep.csv
```

**Expected result**: Performance scales well up to physical core count (16), then plateaus or degrades with hyperthreading.

## Memory Bandwidth Optimization

Strix Halo's ~256 GB/s bandwidth is shared between CPU and GPU. Token generation is memory-bandwidth bound (reading model weights), so maximizing effective bandwidth is critical.

### Strategies
1. **Minimize CPU activity during GPU inference**: Use `-t 4` or lower.
2. **Use quantized models**: Q4_0/Q4_K_M use 4 bits per weight vs 16 for FP16, meaning 4x more effective bandwidth.
3. **Quantize KV cache**: Reduces cache bandwidth requirements.
4. **Disable mmap for large models**: `--no-mmap` avoids HIP page-locking overhead.
5. **Close other applications**: Browsers, IDEs, and other GPU-using apps compete for bandwidth.

### Estimating Token Generation Speed
Token generation speed is approximately:
```
tg_speed ≈ memory_bandwidth / (model_size_bytes / num_layers * layers_on_device)
```

For a 7B Q4_0 model (~3.5GB) on 256 GB/s bandwidth:
```
≈ 256 GB/s / 3.5 GB ≈ 73 tokens/s theoretical max
```

Real-world is typically 60-70% of theoretical due to overhead.

## Model Quantization Selection

For Strix Halo, model quantization choice affects both quality and speed:

| Quantization | Bits/weight | Relative Quality | Speed Impact |
|-------------|-------------|-------------------|-------------|
| Q4_0 | 4.0 | Baseline | Fastest |
| Q4_K_M | 4.8 | Better | Slightly slower |
| Q5_K_M | 5.5 | Good | Moderate |
| Q6_K | 6.5 | Very good | Slower |
| Q8_0 | 8.0 | Near-lossless | Slowest quantized |
| BF16 | 16.0 | Lossless | CPU-only (ZenDNN) |

**Recommendations**:
- For GPU inference: Q4_K_M offers the best quality/speed tradeoff
- For maximum speed: Q4_0
- For CPU with ZenDNN: BF16 (leverages Zen 5 BF16 hardware)
- IQ4_XS: Good option for fitting larger models in memory with acceptable quality

## Server Configuration

For running `llama-server` on Strix Halo:

```bash
export ROCBLAS_USE_HIPBLASLT=1
export ZENDNNL_MATMUL_ALGO=1

./build/bin/llama-server \
  -m model.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  -np 4 \
  -t 4 \
  --host 0.0.0.0 \
  --port 8080
```

For concurrent users (`-np 4`), KV cache quantization becomes even more important since each parallel sequence needs its own cache.

## Advanced: Tensor Overrides

For fine-grained control, you can override where specific tensor types are computed:

```bash
# Keep MoE expert layers on CPU (sometimes faster for sparse models)
./build/bin/llama-cli -m model.gguf \
  -ngl 99 --no-mmap \
  -ot "*.ffn_up=CPU" \
  -ot "*.ffn_down=CPU"
```

This is mostly useful for Mixture-of-Experts models where the sparse expert layers may not benefit from GPU acceleration.

## Quick Reference: Best Defaults

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

Then benchmark and adjust from there.
