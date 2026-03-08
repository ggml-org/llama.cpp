---
name: vulkan-dispatch-optimization
description: >
  Techniques for reducing Vulkan compute dispatch overhead in llama.cpp inference.
  Use when optimizing token generation speed by reducing the number or cost of GPU dispatches,
  implementing operation fusion, batched elementwise kernels, command buffer caching, or
  speculative computation strategies. Applies to any model architecture on Vulkan backend.
---

# Vulkan Dispatch Optimization

## Proven Optimizations (Implemented & Benchmarked)

### 1. Zero-Copy mmap Import for UMA (ggml-vulkan.cpp:14919)
- **Change**: Set `buffer_from_host_ptr = device->uma && device->external_memory_host`
- **Mechanism**: Uses `VK_EXT_external_memory_host` to import mmap'd GGUF data directly as Vulkan buffers
- **Impact**: +6.2% tg speed (55.2 -> 58.6 tok/s), saves ~20 GB RAM
- **Why it works**: Eliminates 20 GB memcpy during load, reduces memory pressure, better TLB coverage
- **Constraints**: Requires page-aligned mmap (Windows MapViewOfFile is always page-aligned)

### 2. Batched Elementwise Mega-Kernel (BDA-based)
- **Files**: `vulkan-shaders/batched_elementwise.comp`, `ggml-vulkan.cpp` (multiple locations)
- **Mechanism**: Accumulates unary element-wise ops (SILU, EXP, SOFTPLUS, SCALE, RELU, NEG, TANH) into a single SSBO-driven dispatch using Buffer Device Address
- **Impact**: +5.1% tg speed (58.6 -> 61.6 tok/s)
- **Design decisions**:
  - Only batches unary ops + SCALE; binary ops (MUL, ADD, SUB) excluded to preserve existing fusion patterns
  - SIGMOID excluded (participates in topk_moe fusion)
  - Host-visible SSBO required (works on UMA, ReBAR)
  - Runs BEFORE existing fusion detection to avoid conflicts
  - Conservative barriers before each batch flush for correctness
- **Key structs**: `vk_batched_op` (48 bytes, matches GLSL layout), stored in pre-allocated 24 KB SSBO

### 3. Combined Result
- Baseline: 55.2 tok/s
- Zero-copy mmap: 58.6 tok/s (+6.2%)
- + Mega-kernel: 61.6 tok/s (+11.6% total)

## Profiling Reference (Qwen3.5-35B-A3B Q4_K_M, per token)

### Dispatch Breakdown
- ~1,500 total dispatches per token
- Matmul ops: ~12,000 us (bandwidth-bound, near-optimal)
- Element-wise ops: ~5,200 us (dispatch-bound, target for optimization)
- Key element-wise consumers: MUL(260x, 1410us), REPEAT(90x, 519us), ADD(91x, 487us)

### Profiling Tools
- `GGML_VK_PERF_LOGGER=1` - Per-op timing summary (adds ~30% overhead)
- `GGML_VK_SYNC_LOGGER=1` - Barrier/sync logging

## Remaining Optimization Opportunities

### Speculative Vocabulary Pruning (~1,400 us savings)
- Track "hot set" of ~4K tokens from recent sampling
- Compute lm_head matmul for hot set first (4.6 MB vs 286 MB)
- Use Cauchy-Schwarz bound to verify correctness
- Fall back to full vocab when confidence check fails (~10% of tokens)
- Expected: 90% hit rate, 56->62 tok/s

### Command Buffer Pre-Recording (~1,200 us savings)
- Record entire forward pass command buffer once, replay per token
- Tensor shapes constant between tokens (only data changes)
- Requires dynamic descriptor offset updates for MoE expert selection

### Binary Op Batching (~1,000 us potential)
- The 260 MUL dispatches are not consecutive (per-layer, not fuseable as multi_mul)
- Could batch if dependency tracking proves they're independent between barriers
- Must not conflict with existing MUL_MAT_ID_MUL, RMS_NORM_MUL fusions

## Build Environment Notes (Windows)
- vcvarsall.bat fails from git-bash; must set INCLUDE/LIB manually
- Build batch file at `/c/Users/fabia/build_llama.bat`
- Touch shader-gen stamps to skip ExternalProject sub-build (include path issue)
- Shader SPIR-V files persist in build-win/ggml/src/ggml-vulkan/*.comp.cpp
