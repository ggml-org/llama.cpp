---
name: vulkan-moe-bandwidth-analysis
description: >
  Analyze and optimize memory bandwidth utilization for Mixture-of-Experts (MoE) models
  on AMD iGPUs with Vulkan backend. Use when investigating token generation speed,
  calculating theoretical throughput limits, understanding per-token bandwidth costs,
  or trying to close the gap between measured and theoretical performance on UMA systems.
  Applies to any MoE model (Qwen3.5, Mixtral, DeepSeek, Llama 4 Scout) on bandwidth-limited hardware.
---

# Vulkan MoE Bandwidth Analysis

Detailed analysis of how llama.cpp's Vulkan backend handles Mixture-of-Experts models,
with focus on memory bandwidth utilization and optimization opportunities.

## Key Finding: Vulkan MUL_MAT_ID Only Reads Active Experts

**Verified by shader analysis** (not speculation):

The Vulkan `MUL_MAT_ID` operation dispatches workgroups only for active experts, NOT all experts.

### How It Works

1. **Dispatch** (`ggml-vulkan.cpp:8614`):
   ```cpp
   ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
       { d_X, d_Y, d_D, d_F0, d_F1, d_ids },
       pc, { groups_x, (uint32_t)nei0, groups_z });
   ```
   - `groups_y = nei0` = number of active expert slots (e.g., 8 for Qwen3.5-35B-A3B)
   - NOT 256 (total expert count)

2. **Shader indexing** (`mul_mat_vec_base.glsl:66`):
   ```glsl
   expert_id = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
   a_offset = expert_id * (p.batch_stride_a / QUANT_K);
   ```
   - Each workgroup reads ONE expert's weight slice via computed offset
   - Only 8 experts' data is read from memory, not 256

3. **Entry point** (`ggml-vulkan.cpp:8625-8630`):
   ```cpp
   static bool ggml_vk_use_mul_mat_vec_id(...) {
       return (src2->ne[1] <= 8) && (src0->type == F32 || F16 || quantized);
   }
   ```
   - Token generation (batch=1) always uses the vec path
   - The mat-mat path (`ggml_vk_mul_mat_id_q_f16`) is used for prompt processing with larger batches

### Implications for Bandwidth Calculation

For Qwen3.5-35B-A3B (256 experts, 8 active, FFN=512, embed=2048):
- Per expert per layer: gate(2048x512 Q4_K) + up(2048x512 Q4_K) + down(512x2048 Q6_K)
- 8 active experts per layer: ~16.25 MB
- NOT 256 experts: would be ~520 MB (32x more)

## Qwen3.5-35B-A3B Bandwidth Budget

### Per-Token Memory Read Breakdown

| Component | Count | Per Unit | Total |
|-----------|-------|----------|-------|
| SSM layers (attn_gate+qkv+ssm) | 30 | 25.3 MB | 759.4 MB |
| SSM layers (MoE 8/256 experts) | 30 | 16.25 MB | 487.5 MB |
| Full-attn layers (Q+K+V+O) | 10 | 19.75 MB | 197.5 MB |
| Full-attn layers (MoE 8/256) | 10 | 16.25 MB | 162.5 MB |
| Shared experts (all 40 layers) | 40 | ~2.7 MB | 108.0 MB |
| Router + norms + small tensors | 40 | ~2.2 MB | 88.0 MB |
| Output head (lm_head 248Kx2048) | 1 | 286.1 MB | 286.1 MB |
| **Total** | | | **~1,893 MB** |

### Theoretical Speed Limits

| Bandwidth | Efficiency | tok/s |
|-----------|-----------|-------|
| 212 GB/s (measured max) | 100% | 112.0 |
| 212 GB/s | 80% | 89.6 |
| 212 GB/s | 70% | 78.4 |
| **Measured: 56 tok/s** | **~50%** | **56** |

### Where is the Other 50%?

At 56 tok/s, only ~106 GB/s of 212 GB/s is used. The gap comes from:

1. **Dispatch overhead (~13%)**: ~685 Vulkan compute dispatches per token at ~3.5us each = 2.4ms of 17.9ms token time. Unlike HIP, Vulkan has no graph capture to amortize dispatch costs.

2. **Small matmul inefficiency (~15-20%)**: MoE expert matmuls are only 0.59-0.85 MB each. Even batched as 8 experts, 4.7-6.8 MB transfers don't fully saturate the memory bus on 40 CUs. The GPU needs ~10+ MB transfers to reach peak bandwidth.

3. **Synchronization barriers (~5-10%)**: Each layer has barriers between attention, MoE routing, expert dispatch, and shared expert computation. These create pipeline bubbles.

4. **Compute overhead (~5%)**: Softmax, RMS norm, router top-k, activation functions are compute-bound rather than bandwidth-bound, adding fixed latency.

## Optimization Opportunities

### High Impact (potentially +10-20 tok/s)
- **Vulkan compute graph/command buffer batching**: Pre-record command buffers for the full forward pass. Eliminates per-dispatch CPU overhead. Would be equivalent to HIP graphs.
- **Expert weight fusion**: Combine gate+up expert matmuls into a single dispatch (they share the same input). Halves MoE dispatch count.
- **Output head pruning/speculation**: For greedy decoding, compute logits for only top-K candidates from previous token. Reduces 286 MB output head read.

### Medium Impact (potentially +5-10 tok/s)
- **Larger workgroup dispatch**: Pack multiple experts into one workgroup to improve memory bandwidth utilization per dispatch.
- **Shared memory prefetch**: Use cooperative loading within workgroups to hide memory latency.
- **KV cache on device-local**: Ensure KV cache uses device-local memory for lower latency access.

### Low Impact (< 5 tok/s)
- Flash attention tuning (already near-optimal for tg)
- Thread count changes (doesn't affect bandwidth-bound tg)
- Batch size tuning (only matters for pp, not tg)

## How to Reproduce This Analysis

```bash
# 1. Get model architecture
python gguf-py/gguf/scripts/gguf_dump.py model.gguf --no-tensors 2>&1 | grep -E "expert|embed|block_count|ffn"

# 2. Get tensor sizes per layer
PYTHONIOENCODING=utf-8 python gguf-py/gguf/scripts/gguf_dump.py model.gguf 2>&1 | grep "blk\.0\."

# 3. Benchmark
llama-bench -m model.gguf -ngl 99 -t 8 -n 128 -p 16

# 4. Calculate: tok/s * bytes_per_token = bandwidth_used
# Compare to system bandwidth (212 GB/s for Strix Halo)
```

## Key Source Files

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:8393-8623` - MUL_MAT_VEC_ID dispatch
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl` - Shader with expert indexing
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl` - Expert row ID loading
- `ggml/src/ggml-vulkan/vulkan-shaders/count_experts.comp` - Expert counting shader
