---
name: llama-cpp-vulkan-internals
description: >
  Deep knowledge of llama.cpp's Vulkan backend internals for debugging, profiling, and
  optimizing GPU inference. Use when investigating Vulkan shader behavior, dispatch patterns,
  memory allocation, pipeline creation, subgroup size selection, or backend-specific bugs.
  Also use when comparing Vulkan vs HIP/CUDA performance characteristics or when the user
  wants to modify Vulkan shaders or backend code.
---

# llama.cpp Vulkan Backend Internals

## Architecture Overview

The Vulkan backend (`ggml/src/ggml-vulkan/`) implements GPU-accelerated tensor operations using Vulkan compute shaders. Key components:

- **`ggml-vulkan.cpp`** (~12K lines): Backend implementation, device detection, pipeline creation, memory management, dispatch logic
- **`vulkan-shaders/`**: GLSL compute shaders compiled to SPIR-V at build time
- **`vulkan-shaders-gen.cpp`**: Shader compilation tool that generates C header files with embedded SPIR-V

## Device Detection and Configuration

### AMD RDNA 3.5 (Strix Halo) Detection
```cpp
// Vendor ID 0x1002 = AMD
// Device name contains "gfx1150" or "gfx1151"
// RDNA 3.5 is classified under GGML_VK_AMD_RDNA3 in the backend
```

Key auto-detected features:
- **UMA (Unified Memory Architecture)**: Detected via `vk::MemoryHeap::eDeviceLocal` check. Enables zero-copy buffer sharing.
- **Wave64**: RDNA 3.5 defaults to subgroupSize=64. The backend queries `VkPhysicalDeviceSubgroupProperties`.
- **Cooperative matrix**: `VK_KHR_cooperative_matrix` enabled only for RDNA3+ (older AMD GPUs have buggy support).
- **Integer dot product**: `VK_KHR_shader_integer_dot_product` for quantized matmul acceleration.

### Subgroup Size Selection (Critical for Performance)

RDNA 3/3.5 supports both wave32 and wave64. The backend allows per-pipeline subgroup size override via `gpu_pipeline_configs`:

```cpp
// DO NOT add an RDNA3 entry forcing wave32 as default!
// Benchmarks show wave64 is faster: 56.4 tok/s vs 51.5 tok/s (8.7% regression)
// Only specific shaders like soft_max and im2col might benefit from wave64 override
```

The `RDNA_DEFAULT_SUBGROUP_SIZE` is 32 (wave32), but the backend's auto-selection picks wave64 when no override is configured. Adding an AMD_RDNA3 entry to `gpu_pipeline_configs` forces wave32 as the default, which hurts performance.

## Memory Management

### UMA Buffer Allocation
On UMA devices (iGPUs), the backend prefers:
1. **Device-local + host-visible**: Best of both worlds on UMA
2. **Zero-copy host buffers**: When device-local is exhausted
3. Buffer import via `VK_EXT_external_memory_host` when available

### Buffer Binding for MUL_MAT_ID
The entire expert weight tensor (all 256 experts) is bound as a single storage buffer. Individual expert data is accessed via offset calculation in the shader. This means:
- No per-expert buffer allocation
- No per-expert descriptor set updates
- GPU reads only the bytes it needs via offset addressing

## Pipeline Dispatch Patterns

### Token Generation (MUL_MAT_VEC)
Single-token inference uses vector-matrix multiply shaders:
- `mul_mat_vec_*.comp` for regular layers
- Same shaders with `#define MUL_MAT_ID` for MoE expert layers
- Dispatch: `(output_rows, batch_or_expert_count, groups_z)`

### Prompt Processing (MUL_MAT)
Batched inference uses matrix-matrix multiply:
- `mul_mm.comp` (standard path)
- `mul_mm_cm2.comp` (cooperative matrix path for RDNA3+)
- `mul_mmq.comp` (quantized matmul path)
- For MoE: `count_experts.comp` first counts tokens per expert, then dispatches per-expert matmuls

### MoE Expert Dispatch (Token Gen)
```
Per token, per MoE layer:
1. Router matmul: hidden[1,2048] x router[2048,256] -> logits[1,256]
2. Top-K selection on CPU or GPU
3. MUL_MAT_VEC_ID for gate_exps: 8 workgroups (one per active expert)
4. MUL_MAT_VEC_ID for up_exps: 8 workgroups
5. Activation (SiLU)
6. Element-wise multiply (gate * up)
7. MUL_MAT_VEC_ID for down_exps: 8 workgroups
8. Shared expert: 3 separate matmul dispatches
Total: ~10-12 dispatches for MoE portion alone
```

### MoE Expert Dispatch (Prompt Processing)
```
1. count_experts.comp: Count how many tokens go to each expert
2. For each expert with tokens: MUL_MAT with token subset
   - Uses row_ids shared memory to track which tokens belong to which expert
   - MUL_MAT_ID_USE_SUBGROUPS path uses ballot operations for efficient row filtering
```

## Performance Characteristics on RDNA 3.5

### Dispatch Overhead
- ~685 total dispatches per token for Qwen3.5-35B-A3B (40 layers)
- ~3.5 microseconds per dispatch (Vulkan command buffer submission)
- Total overhead: ~2.4 ms per token (~13% at 56 tok/s)
- **No Vulkan equivalent of HIP graphs** - each dispatch is a separate command

### Memory Bandwidth Utilization
- Peak system bandwidth: 212 GB/s (LPDDR5X-8000)
- Measured utilization at 56 tok/s: ~106 GB/s (50%)
- Gap due to: dispatch overhead, small matmul sizes, synchronization barriers
- MoE expert matmuls (0.59-6.8 MB) are too small to saturate bandwidth per dispatch

### Cooperative Matrix
- Enabled for RDNA3+ via `VK_KHR_cooperative_matrix`
- Used in `mul_mm_cm2.comp` for batched matmul (prompt processing)
- NOT used in MUL_MAT_VEC (token generation) - too small for coopmat benefit

## Key Source Locations

| What | File | Lines |
|------|------|-------|
| Device detection | `ggml-vulkan.cpp` | ~2500-2700 |
| UMA detection | `ggml-vulkan.cpp` | ~2698 |
| Pipeline config (subgroup size) | `ggml-vulkan.cpp` | ~2800-2900 |
| MUL_MAT_VEC_ID dispatch | `ggml-vulkan.cpp` | 8393-8623 |
| MUL_MAT_ID (batched) dispatch | `ggml-vulkan.cpp` | 8101-8390 |
| Expert counting | `vulkan-shaders/count_experts.comp` | entire file |
| Vec-mat shader base | `vulkan-shaders/mul_mat_vec_base.glsl` | entire file |
| Expert ID row loading | `vulkan-shaders/mul_mm_id_funcs.glsl` | entire file |
| Mat-mat shader | `vulkan-shaders/mul_mm.comp` | entire file |
| Cooperative matrix shader | `vulkan-shaders/mul_mm_cm2.comp` | entire file |

## Common Pitfalls

1. **Wave32 regression**: Adding RDNA3 to `gpu_pipeline_configs` forces wave32 default, causing 8.7% tg regression
2. **Buffer size limits**: `maxStorageBufferRange` may be too small for full expert tensor. Backend has 64-bit indexing fallback.
3. **Shared memory limits**: MUL_MAT_ID needs extra shared memory for row_ids and ballot operations. `ggml_vk_matmul_shmem_support()` checks this.
4. **Flash attention occupancy**: On RDNA, FA shader pads shared memory to reduce occupancy to 4 subgroups, preventing cache thrashing.
