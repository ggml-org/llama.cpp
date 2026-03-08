---
name: parallel-model-serving
description: >
  Run and benchmark parallel model serving for coding agents and multi-user workloads.
  Use when the user wants to run multiple models simultaneously, serve multiple agents
  from one model, benchmark parallel throughput, or compare single vs multi-slot serving.
  Also use when the user mentions "parallel", "multi-slot", "coding agents", "concurrent",
  "multiple conversations", or "serving".
---

# Parallel Model Serving for Coding Agents

Guide for running and benchmarking parallel inference on llama.cpp, optimized for
AMD Strix Halo (Vulkan) but applicable to any backend.

## Architecture Options

### Option 1: Multi-slot llama-server (Recommended)

One model loaded once, multiple independent conversation slots sharing the weights.

```bash
# 4 parallel slots, single model instance
llama-server \
  -m <model.gguf> \
  -ngl 99 \
  -np 4 \           # number of parallel slots
  -c 8192 \         # total context across all slots (divided equally)
  --port 8080
```

**Advantages:**
- Model weights loaded once in VRAM (~8 GiB for 35B-A3B)
- Single Vulkan device context — no inter-process GPU contention
- Shared dispatch pipeline — one command queue
- For SSM models: state is constant-size per slot, no growing KV cache
- For MoE models: expert weights shared, only active experts read per token

**Disadvantages:**
- Slots share bandwidth — per-slot throughput decreases with more slots
- All slots use the same model — can't mix different models
- Single point of failure

**Memory per slot:**
- Transformer models: KV cache grows with context length. Budget ~2 MB per 1K context per B params
- SSM models (Delta-Net, Mamba): Fixed state size regardless of context. ~128x128x4 bytes x heads x layers per slot

### Option 2: Multiple llama-bench/llama-cli processes

Separate processes, each loading its own model copy.

```bash
# Run two models in parallel (background + wait)
llama-bench -m model_a.gguf -ngl 99 -t 4 -r 3 -p 0 -n 128 &
PID1=$!
llama-bench -m model_b.gguf -ngl 99 -t 4 -r 3 -p 0 -n 128 &
PID2=$!
wait $PID1 $PID2
```

**Advantages:**
- Can run different models simultaneously
- Process isolation — one crash doesn't affect the other
- Simple to set up

**Disadvantages:**
- Each process loads its own copy of weights (2x memory)
- Competing Vulkan device contexts cause scheduling overhead
- Memory bandwidth split between processes — expect ~50% per-process throughput
- Higher variance (±2-3 tok/s) due to contention

### Option 3: vLLM (Python/PyTorch serving framework)

**Website**: https://docs.vllm.ai/
**Install**: `pip install vllm`
**Backends**: CUDA (NVIDIA), ROCm (AMD discrete GPUs)

vLLM is the industry-standard high-throughput LLM serving framework. It supports
continuous batching, PagedAttention, tensor parallelism, and speculative decoding.

**Qwen 3.5 support** (as of early 2026):
- Full GatedDeltaNet support via Triton `fused_recurrent_gated_delta_rule` kernel
- Fuses the entire SSM recurrence into one kernel (same idea as our GGML fused op)
- FP8 quantization, MTP speculative decoding, `--performance-mode` flag
- Hybrid KV cache manager for mixed linear-attention + full-attention layers
- CUDA graphs enabled by default to eliminate CPU dispatch overhead
- On NVIDIA B200: 92.5 tok/s single-user, 723.9 tok/s at batch 16 (Qwen3.5-27B)

```bash
# vLLM example: serve Qwen3.5-35B-A3B with 4 concurrent users
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 1 \
  --max-num-seqs 4 \
  --performance-mode throughput
```

**When to use vLLM:**
- Linux + NVIDIA GPU (best support)
- Linux + AMD Instinct MI300X/MI325X (ROCm, Day 0 Qwen 3.5 support from AMD)
- Many concurrent users (>4) where continuous batching pays off
- Need FP8/NVFP4 quantization or speculative decoding

**Not recommended for this Strix Halo setup because:**
- No Vulkan backend — requires CUDA or ROCm
- ROCm on Strix Halo gets 21 tok/s vs Vulkan 67 tok/s (3x slower)
- Windows ROCm support is limited and unreliable
- vLLM's PagedAttention doesn't help SSM-dominated models
- For <=4 concurrent users, llama-server's simple slot system is sufficient

### Option 4: ROCm / HIP (AMD GPU compute)

**Website**: https://rocm.docs.amd.com/
**Backends**: AMD discrete GPUs (MI-series, Radeon RX), AMD iGPUs (limited)

ROCm is AMD's open-source GPU compute platform, similar to CUDA. llama.cpp has a
HIP backend that uses ROCm, and vLLM/PyTorch can also run on ROCm.

**Strix Halo ROCm status:**
- llama.cpp HIP backend: **21 tok/s** tg128 (vs 67 tok/s Vulkan — 3x slower)
- The gap is due to immature Windows ROCm drivers and iGPU-specific issues
- ROCm is primarily optimized for discrete MI-series datacenter GPUs
- gfx1151 (RDNA 3.5 iGPU) is not a primary ROCm target

**When ROCm makes sense:**
- AMD Instinct MI300X/MI325X on Linux (primary target, well optimized)
- AMD Radeon RX 7900 XTX on Linux (community supported)
- When you need PyTorch/vLLM ecosystem compatibility on AMD hardware

**When ROCm does NOT make sense:**
- Strix Halo iGPU on Windows (Vulkan is 3x faster)
- Any setup where Vulkan backend is available and faster

```bash
# llama.cpp HIP build (for reference, not recommended on Strix Halo)
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
cmake --build build-hip --config Release

# Benchmark comparison
build-hip/bin/llama-bench -m model.gguf -ngl 99     # ~21 tok/s
build-win/bin/llama-bench -m model.gguf -ngl 99     # ~67 tok/s (Vulkan)
```

### Tool Comparison Summary

| Feature | llama-server | vLLM | ROCm/HIP |
|---------|-------------|------|----------|
| **Best for** | Local/edge, few users | Cloud, many users | AMD datacenter GPUs |
| **Backend** | Vulkan, CUDA, Metal, HIP | CUDA, ROCm | HIP (AMD only) |
| **Qwen 3.5 SSM** | Fused kernel (our work) | Triton fused kernel | Via vLLM/PyTorch |
| **Strix Halo tok/s** | **67** (Vulkan) | ~21 (via ROCm) | 21 |
| **Continuous batching** | Basic (slot-based) | Advanced (PagedAttention) | Via vLLM |
| **CUDA graphs** | No | Yes | HIP graphs (Linux) |
| **Windows support** | Full | Limited | Limited |
| **Multi-model** | Separate processes | Single process | Via vLLM |
| **Quantization** | GGUF (Q4_K, Q8, etc.) | FP8, NVFP4, AWQ, GPTQ | Via vLLM |
| **Setup complexity** | Low (single binary) | Medium (Python env) | High (driver stack) |

## Benchmarking Parallel Throughput

### Single model, measuring multi-slot server throughput

```bash
# Start server with N slots
llama-server -m model.gguf -ngl 99 -np 4 -c 8192 --port 8080 &
SERVER_PID=$!

# Wait for server to be ready
sleep 5

# Send N concurrent requests (using curl in parallel)
for i in $(seq 1 4); do
  curl -s http://localhost:8080/completion \
    -d '{"prompt":"Write a function that","n_predict":128}' &
done
wait

# Check server metrics
curl -s http://localhost:8080/health | python -m json.tool

kill $SERVER_PID
```

### Two separate models in parallel

```bash
# Benchmark two models simultaneously
llama-bench -m model_a.gguf -ngl 99 -t 4 -r 3 -p 0 -n 128 2>&1 &
PID1=$!
llama-bench -m model_b.gguf -ngl 99 -t 4 -r 3 -p 0 -n 128 2>&1 &
PID2=$!
wait $PID1
echo "=== Model A done ==="
wait $PID2
echo "=== Model B done ==="
```

### Key benchmark parameters
- `-t N`: CPU threads per process (halve when running 2 processes: use `-t 4` instead of `-t 8`)
- `-r 3`: At least 3 repeats for stable numbers
- `-p 0 -n 128`: Token generation only (tg128), skip prompt processing
- Always warm up GPU first (discard first run)

## Measured Results on Strix Halo

### Solo vs parallel (Vulkan, warm GPU)

| Model | Quant | Size | Solo tok/s | Parallel tok/s | Notes |
|-------|-------|------|-----------|----------------|-------|
| Qwen3.5-35B-A3B | Q4_K_M | ~8 GiB | **67.09** | ~35 (est.) | MoE+SSM, only 3B active |
| Qwen3.5-4B | Q4_K_XL | 2.70 GiB | **56.09** | 27.01 | Dense model |
| Qwen3.5-9B | Q4_K_XL | 5.55 GiB | **34.52** | 24.71 | Dense model |
| 4B + 9B combined | — | 8.25 GiB | — | **51.72** | Total throughput |

### Why MoE wins for parallel serving

The 35B-A3B reads ~1.9 GB per token (only active experts). A dense 4B reads ~2.7 GB per token (all params). The MoE model is more bandwidth-efficient despite being "larger":

- 2x 35B-A3B parallel: ~3.8 GB/tok bandwidth needed → ~35 tok/s each, ~70 combined
- 2x 4B parallel: ~5.4 GB/tok bandwidth needed → ~27 tok/s each, ~54 combined
- The "bigger" MoE model gives **better parallel throughput** than the smaller dense model

### Bandwidth is the bottleneck

On Strix Halo with ~212 GB/s memory bandwidth:
- Solo: one model uses ~60% of bandwidth → 67 tok/s
- Parallel: two models split bandwidth → ~50% throughput each
- The variance increases in parallel (±2-3 tok/s) due to cache line contention

## Recommendations for Coding Agent Setups

### Single agent (maximize speed)
```bash
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -np 1 -c 8192
```
Expected: ~67 tok/s

### 2 concurrent agents (same model)
```bash
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -np 2 -c 16384
```
Expected: ~35 tok/s per slot, ~70 tok/s combined

### 4 concurrent agents (same model)
```bash
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -np 4 -c 32768
```
Expected: ~18 tok/s per slot, ~72 tok/s combined

### Mixed models (different agents, different needs)
Run separate processes — each loads its own model:
```bash
llama-server -m large-model.gguf -ngl 99 --port 8080 &   # "thinking" agent
llama-server -m small-model.gguf -ngl 99 --port 8081 &   # "fast" agent
```
Note: bandwidth is split, both will be slower than solo.

## Scaling Rules of Thumb

1. **Total throughput scales sub-linearly**: 2 slots ≈ 1.0-1.1x total throughput of 1 slot (bandwidth-bound)
2. **Per-slot throughput halves** with each doubling of slots
3. **MoE models scale better** than dense models (less bandwidth per token)
4. **SSM models have constant state** — adding slots doesn't increase per-slot memory like KV cache does
5. **CPU threads should be split** between processes: if you have 16 threads, use `-t 8` per process with 2 processes
6. **Diminishing returns past 4 slots** on Strix Halo — per-slot throughput drops below ~15 tok/s

## Monitoring

```bash
# Server health and slot status
curl -s http://localhost:8080/health

# Detailed metrics
curl -s http://localhost:8080/metrics

# Check which slots are active
curl -s http://localhost:8080/slots
```
