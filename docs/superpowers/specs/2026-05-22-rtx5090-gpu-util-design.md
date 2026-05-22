# RTX 5090 GPU Utilization Optimization Design

**Date:** 2026-05-22
**Branch:** refactor/CUDA-backend

## Hardware & Software Context

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GB VRAM |
| CPU | AMD EPYC 9V74, 80 cores / 160 threads, 1 NUMA node |
| CUDA | 12.0 (sm_120a-real), CUDA runtime 13.2, driver 595.71.05 |
| Compiler | GCC 15.2.0 (AOCC 5.2.0 available but not in use) |
| Model | Qwen3.6-27B-UD-Q6_K_XL, ~29.6 GB VRAM, ~2.4 GB headroom |
| Server | --ctx-size 262144 --parallel 4 --threads 32 --threads-batch 32 --batch-size 4096 --ubatch-size 1024 -ngl 999 --flash-attn on --cont-batching --cache-type-k q8_0 --cache-type-v q8_0 |

## Problem

GPU utilization during token generation caps at ~50% even with a single request at short context. The RTX 5090 finishes each decode forward pass quickly and then sits idle. The GPU is starved — the CPU and GPU operations share the same CUDA stream, so the SM waits for the logits readback to complete before it can accept new work.

### Root Cause

**Current flow (single-stream, sequential):**

```
llama-context.cpp:1444 — ggml_backend_tensor_get_async() starts logits copy to host
sampling.cpp:538       — llama_synchronize(ctx) blocks CPU until GPU + copy done
common_sampler_sample  — CPU reads logits.data for sampling (synchronous dependency)
```

`ggml_backend_tensor_get_async` launches `cudaMemcpyAsync` on the main compute stream. Then `llama_synchronize` calls `cudaStreamSynchronize` on that same stream — waiting for both the compute kernels AND the memcpy. The GPU SM is idle during the ~100 MB logits readback because compute and copy compete for the same stream.

## Solution: Two-Stream Logits Readback Overlap

Split the readback onto a dedicated stream so the SM finishes compute and becomes available for the next forward pass while the GPU copy engine handles the logits independently.

```
Before (single stream):
  [compute kernels] → [logits memcpy] → CPU sync → sampling → next step
  SM idle during memcpy

After (two stream):
  Stream 0: [compute kernels] → done (SM immediately ready)
  Stream 1:                    ← wait(kernels_done) → [logits memcpy] → CPU sync → sampling
  Copy engine handles memcpy; SM is free for next forward pass
```

**Implementation steps:**

1. Create a dedicated readback stream (`cudaStreamCreateWithFlags(cudaStreamNonBlocking)`) per CUDA backend device.
2. In `ggml_backend_cuda_get_tensor_async`: record a dependency event on the compute stream, wait on it in the readback stream, then `cudaMemcpyAsync` on the readback stream.
3. In `ggml_backend_cuda_synchronize`: wait on the readback stream (not compute stream) for the sync that follows an async get.
4. Allocate logits host memory as pinned (`cudaMallocHost`) — required for fast async memcpy. This is a `llama-context.cpp` change.
5. Fall back to single-stream sync copy if pinned allocation fails.

**Files affected:**
- `ggml/src/ggml-cuda/common.cuh` — add `cudaStream_t readback_stream` and `cudaEvent_t readback_event` to `ggml_backend_cuda_context`
- `ggml/src/ggml-cuda/ggml-cuda.cu` — split `get_tensor_async` onto readback stream; update `synchronize` logic; init/cleanup
- `src/llama-context.cpp` — use pinned memory for logits buffer allocation

**Safety:** The readback stream waits on the compute stream's event before reading. No data race. If any CUDA call fails, fall back to the current single-stream sync path.

### Out of Scope

- Speculative decoding tuning — user confirmed no effect on GPU utilization
- CPU thread count tuning — user confirmed no effect on GPU utilization
- GGML cgraph caching between decode steps — too complex for this round
- AOCC rebuild — orthogonal build experiment

## Testing

1. **Correctness:** Run `test-backend-ops` to verify two-stream change doesn't break tensor operations
2. **Output match:** Run server with a known prompt and compare output token-for-token against single-stream
3. **GPU utilization:** Monitor `nvidia-smi` during generation — target >80% at 1 request
4. **Fallback path:** Verify single-stream fallback works by testing error conditions

## Files to Change

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/common.cuh` | Add `readback_stream` + `readback_event` to `ggml_backend_cuda_context` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Split `get_tensor_async` onto readback stream; update `synchronize` to wait on correct stream; init/cleanup |
| `src/llama-context.cpp` | Allocate logits buffer with pinned (page-locked) memory for async memcpy |
