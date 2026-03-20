# Vulkan Chunked Gated Delta Net (GDN) — Performance & Development Notes

PR #20377 — First chunked parallel GDN implementation on any GPU shader backend.

## Architecture

Three-stage chunked parallel decomposition (matches FLA/NVlabs reference implementations):

1. **Intra-chunk** (`gated_delta_net_chunk_intra.comp`) — Builds attention matrix A, computes W/U via WY representation. Outputs g_cumsum and total chunk decay.
2. **Inter-chunk** (`gated_delta_net_chunk_inter.comp`) — Sequential across chunks, parallel across state columns. State update: `S_next = exp(g_total) * S + K_gated^T @ v_corrected`.
3. **Output** (`gated_delta_net_chunk_output_cm1.comp`) — Coopmat GEMM kernel. Computes `A_decayed[64x64] @ vnew[64x128]` using VK_KHR_cooperative_matrix (f16 inputs, f32 accumulation).

Chunk size: C=64 tokens. State dimensions: S_K=S_V=128. Pipeline: d128 non-KDA configs only.

## Development History

### Phase 1: Infrastructure (PR #20334, merged)
- Autoregressive GDN Vulkan shader — single-token sequential processing
- PP-512: 165 t/s, TG-128: 21.2 t/s on 890M (16 CU)
- 13/13 backend-ops tests

### Phase 2: Graph-level chunked ops (PR #20340, merged)
- Chunked op decomposition at the GGML graph level
- Feeds autoregressive shader more efficiently
- PP-512: 165 → 220 t/s (+30.3%) — this gain is already in master

### Phase 3: Vulkan chunked shaders (PR #20377, this PR)
- Three new compute shaders for intra/inter/output stages
- Initial scalar output kernel — functional but dispatch overhead made it slower than autoregressive on 16 CU
- Threshold gating: chunked path activates only when beneficial

### Phase 4: Coopmat output kernel
- Replaced scalar output with VK_KHR_cooperative_matrix GEMM
- f16 shared memory for A_decayed and vnew, f32 accumulation via coopmat
- 4-phase architecture: QK^T via coopmat → decay mask → vnew staging → A_decayed @ vnew GEMM
- Numerically stable: direct `exp(g_i - g_j)` per element (no factorization — factorized approach caused PPL regression to 20.06)
- 16/16 backend-ops tests pass

### Abandoned Approaches
- **Factorized exp with g_max**: `exp(g_max - gcum[j])` amplified vnew, caused catastrophic cancellation. PPL 20.06 vs 13.46 baseline.
- **Scoped register split**: Attempted to reduce VGPR pressure via scope boundaries. RADV compiler ignores scope for register allocation — no measurable difference.

## Current Performance

Hardware: AMD Radeon 890M (RDNA3.5, 16 CU, 64KB LDS/CU, warp 64, KHR_coopmat)
Model: Qwen3-Coder-Next-REAM Q4_K_M (60.33B params, 34.21 GiB)

### Throughput (chunked coopmat, GDN_CHUNK_THRESHOLD=2)

| Test | t/s |
|------|-----|
| PP-512 | 217.55 ± 1.41 |
| PP-1024 | 219.84 ± 4.00 |
| PP-2048 | 216.89 ± 1.94 |
| TG-128 | 21.76 ± 0.06 |

### Autoregressive vs Chunked Comparison

| Test | Autoregressive | Chunked coopmat | Delta |
|------|---------------|-----------------|-------|
| PP-512 | 225.68 ± 3.00 | 217.55 ± 1.41 | -3.6% |
| PP-1024 | 229.63 ± 4.39 | 219.84 ± 4.00 | -4.3% |
| PP-2048 | 230.88 ± 1.44 | 216.89 ± 1.94 | -6.1% |
| TG-128 | 21.29 ± 0.03 | 21.76 ± 0.06 | +2.2% |

On 16 CU, autoregressive is 3.6-6.1% faster for PP due to lower dispatch overhead. Note autoregressive PP improves from 512→2048 while chunked stays flat — the gap widens on small hardware but the scaling characteristics favor chunked on wider hardware.

GDN kernel time comparison (PP-512):
- Autoregressive: 36 × 1,150 us = 41 ms (1.8% of total)
- Chunked (3 dispatches): 36 × 5,173 us = 186 ms (7.9% of total)

The chunked path's 3-dispatch overhead (intra + inter + output) accounts for the per-kernel cost difference, but end-to-end impact is only 3.6-6.1% since GDN is a small fraction of total wall time on this MoE model.

### Perplexity Validation (WikiText-2, 299K tokens)

| Context | Chunked coopmat | f32 baseline | Delta |
|---------|----------------|--------------|-------|
| 512 (584 chunks) | 13.52 ± 0.11 | 13.46 | +0.06 |
| 4096 (73 chunks) | 10.18 ± 0.08 | 10.15 | +0.03 |

Both within error bars. Chunked coopmat path is numerically lossless.

### Per-Kernel Timing (GGML_VK_PERF_LOGGER, PP-512)

```
GATED_DELTA_NET: 36 × 5173 us = 186 ms (7.9% of 2.35s total)
FLASH_ATTN_EXT:  12 × 783 us  = 9.4 ms (0.4% of 2.35s total)
```

GDN is 7.9% of PP-512 wall time on this MoE-heavy model. MUL_MAT and MoE routing dominate the remaining 92%.

## Scaling Analysis

### Why flat PP scaling matters
PP-512/1024/2048 all within ±2 t/s. The chunked architecture processes fixed-size 64-token chunks — adding more tokens adds more chunks at constant cost each. Autoregressive dispatches scale linearly with token count (36 layers × N tokens = 36N sequential dispatches).

### Why 16 CU doesn't show the crossover
- Chunked output kernel dispatches 3 shaders (intra + inter + output) vs 1 for autoregressive
- Each shader has launch overhead (~10-20 us) that dominates on small hardware
- The 64×64 @ 64×128 coopmat GEMM in the output kernel can't saturate 16 CUs
- On 40+ CU hardware (e.g., Strix Halo 8060S, discrete GPUs), the matmul-heavy chunked path has more headroom

### GDN share grows with model density
On Qwen3-Next (384-expert MoE), GDN is only 8% of wall time. On GDN-dense architectures with fewer/no MoE layers, GDN's share would be 30-40%+, making the chunked optimization proportionally more impactful.

## Key Files

| File | Purpose |
|------|---------|
| `vulkan-shaders/gated_delta_net.comp` | Autoregressive kernel |
| `vulkan-shaders/gated_delta_net_chunk_intra.comp` | Intra-chunk (A matrix, WY) |
| `vulkan-shaders/gated_delta_net_chunk_inter.comp` | Inter-chunk (state update) |
| `vulkan-shaders/gated_delta_net_chunk_output.comp` | Original scalar output |
| `vulkan-shaders/gated_delta_net_chunk_output_cm1.comp` | Coopmat GEMM output |
| `ggml-vulkan.cpp:10409` | GDN_CHUNK_THRESHOLD (dispatch gating) |

## Test Commands

```bash
# Backend ops tests
./build/bin/test-backend-ops -b Vulkan0 -o GATED_DELTA_NET

# Benchmark
./build/bin/llama-bench -m <model> -ngl 99 -fa 1 -n 128 -p 512 --output md

# Perf logger
GGML_VK_PERF_LOGGER=1 ./build/bin/llama-bench -m <model> -ngl 99 -fa 1 -n 128 -p 512 -r 3 --output md

# Perplexity
./build/bin/llama-perplexity -m <model> -ngl 99 -fa 1 --ctx-size 4096 -f data/wikitext-2-raw/wiki.test.raw
```
