# Strix Halo Vulkan Optimization Log

## Target: Qwen3.5-35B-A3B Q4_K_M — Token Generation (tg128)

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S (gfx1151, RDNA 3.5, 40 CUs), 68 GB LPDDR5X (~212 GB/s)

**Model**: Hybrid SSM (Delta-Net) + MoE architecture. 24 SSM layers + attention layers. Only ~3B params active per token (8 of 256 experts selected).

---

## Performance Timeline

| Stage | tok/s | Change | Description |
|-------|------:|-------:|-------------|
| Baseline (master d969e933e) | 58.07 | — | Upstream llama.cpp, wave64 default |
| + Batched elementwise mega-kernel | ~58.0 | +7% vs pre-wave64 | Fuse ~260 tiny dispatches into batched kernel |
| + Fused SSM recurrence op | 58.71 | +1.1% | Replace 264 dispatches/tok with 24 fused shaders |
| + SSM shared memory tiling | **67.09** | **+15.5%** | Coalesced LDS access for state matrix |

**Final result: 58.07 → 67.09 tok/s (+15.5% over upstream master)**

---

## What Worked

### 1. Batched Elementwise Mega-Kernel (+7%)
**54.25 → 58.0 tok/s** (measured against pre-wave64 baseline)

Created `batched_elementwise.comp` — a single Vulkan compute shader that absorbs ~260 tiny elementwise dispatches per token (SILU, EXP, SOFTPLUS, SCALE, MUL, SUB, SIGMOID, RELU, NEG, TANH). Instead of submitting one dispatch per op, the backend detects consecutive same-shape elementwise ops and batches them into a single dispatch with an SSBO describing the operation sequence.

- Same-shape operands only (broadcast not supported — flat modulo indexing can't handle multi-dim broadcast)
- ADD excluded (see Regressions below)
- SSBO capacity: 2048 ops with graceful fallback

### 2. GGML_OP_DELTA_NET_RECURRENCE — Fused SSM Kernel
**Reduces dispatch count by ~264 per token**

Each SSM layer previously decomposed into 11 individual Vulkan dispatches (multiply, scale, exp, dot products, outer product update). Created a new GGML op that fuses all 11 into a single `ssm_recurrence.comp` shader dispatch.

- 8-file change: ggml.h, ggml.c, ops.cpp, ops.h, ggml-cpu.c, ggml-vulkan.cpp, vulkan-shaders-gen.cpp, delta-net-base.cpp
- 128 threads per workgroup, one per state row
- CPU fallback implemented for correctness testing

### 3. SSM Shared Memory Tiling (+14.8%)
**58.44 → 67.09 tok/s** — the biggest single win.

The original fused SSM shader stored state rows in registers (`float s_row[128]`), causing non-coalesced global memory reads — each thread read a different row, 512 bytes apart, wasting 31/32 cache line bytes (6.25% utilization).

Redesigned to use shared memory (LDS) with tiled, coalesced access:
- **TILE_K=64**: Process 64 columns at a time (64 x 128 x 4 = 32 KB, exactly fits RDNA 3.5's 32 KB shared memory limit)
- **Transposed LDS layout**: `s_tile[col * 128 + row]` gives 2-way bank conflicts instead of 64-way
- **Coalesced global reads**: Thread j loads column j, consecutive threads read consecutive addresses, 100% cache line utilization
- **2-pass algorithm**: Pass 1 loads tiles and computes `sk_j = dot(state_row, k)`. Pass 2 reloads tiles, applies the rank-1 update, and computes output
- **Cooperative loading**: 128 threads split into 2 halves, each loading 64 rows of each column
- Also required adding `ggml_cont()` guards in delta-net-base.cpp for non-contiguous tensor views from server batching

### 4. Wave64 Subgroup Size (+8.7% vs wave32)
Correctly defaulting to wave64 on RDNA 3.5 for soft_max, im2col, and flash_attn shaders. Wave32 caused an 8.7% regression.

### 5. Flash Attention row_split for UMA iGPUs
Enabled row_split in flash attention for integrated GPUs sharing system memory, improving occupancy on RDNA 3.5.

---

## What Didn't Work

### Regressions (Made Things Worse)

| Attempt | Result | Why It Failed |
|---------|--------|---------------|
| **ADD in mega-kernel batching** | +1500 us regression | ADD ops have complex dependency patterns. The barrier/flush overhead exceeds the dispatch savings. |
| **UMA HostVisible+HostCoherent allocation** | 58 → 56 tok/s (-3.4%) | On UMA, requesting HostVisible+HostCoherent causes the driver to use uncached/write-combined memory instead of GPU-optimized cached memory. eDeviceLocal is correct even on UMA. |
| **SSM double-read (remove register array)** | 58.71 → 54.13 tok/s (-7.8%) | Removing `float s_row[128]` and reading state_in twice doubled non-coalesced global reads (threads 512B apart, 32x cache line waste). This observation directly led to the shared memory tiling solution. |
| **MUL_MAT substitution for SSM dot products** | 4.2 tok/s (garbage output) | `ggml_mul_mat` computes A^T @ B, not A @ B. The substitution was mathematically wrong. Reverted immediately. |
| **Fusion plan caching (goto-based)** | 43 tok/s (worse than baseline) | The `goto` skipped essential state setup (fused_ops_write_mask, dependency tracking). CPU time dropped but GPU throughput regressed. |

### Dead Ends (Fundamentally Won't Work)

| Attempt | Result | Why It's a Dead End |
|---------|--------|---------------------|
| **Speculative decoding** | 4.65 tok/s (92% slower) | Qwen3.5 uses M-RoPE, incompatible with speculative decoding. Only 4% acceptance rate. |
| **KV cache quantization** (`-ctk`/`-ctv`) | Context creation fails | Not supported on Vulkan backend. Would need backend-level changes. |
| **Flash attention tuning** (`-fa 1`) | No measurable effect | Model is SSM-dominated (Delta-Net). FA only affects the few attention layers. |
| **Zero-copy mmap** (`VK_EXT_external_memory_host`) | Crash with `-mmp 1` | GGUF tensor data not 4096-byte aligned within file. Would need format changes. |
| **REPEAT op batching** | Impossible (analysis only) | SSM REPEAT ops have sequential read-after-write hazards. Each op depends on the previous op's output. Cannot be parallelized. |

---

## Comparative Benchmarks (All Vulkan, Strix Halo)

| Model | Quant | Size | Solo tok/s | 2-in-parallel tok/s |
|-------|-------|------|-----------|---------------------|
| Qwen3.5-35B-A3B (MoE+SSM) | Q4_K_M | ~8 GiB | **67.09** | — |
| Qwen3.5-4B (dense) | Q4_K_XL | 2.70 GiB | 56.09 | 27.01 |
| Qwen3.5-9B (dense) | Q4_K_XL | 5.55 GiB | 34.52 | 24.71 |
| 4B + 9B parallel | — | 8.25 GiB | — | 51.72 combined |

The 35B MoE model is faster than even the 4B dense model because MoE only reads ~3B active params per token, while the dense 4B reads all 4.2B every token.

---

## Theoretical Limits

- Memory bandwidth: ~212 GB/s measured
- Current efficiency: ~60% (127/212 GB/s)
- Matmul ops already near peak bandwidth
- Theoretical max at 100% BW: ~112 tok/s
- Realistic ceiling at 70% BW: ~78 tok/s
- **Gap to close: ~11 tok/s** (67 → 78)

---

## Remaining Optimization Ideas (Not Yet Tried)

| Option | Estimated Gain | Complexity | Notes |
|--------|---------------|------------|-------|
| Broadcast MUL batching | ~500-800 us/tok (+2-3 tok/s) | High | Need multi-dim broadcast support in mega-kernel |
| Command buffer caching | Unknown | Very high | llama.cpp rebuilds command buffers every token |
| Barrier reduction | ~5 us x hundreds/tok | Medium | Need dependency analysis to prove safety |
| Matmul shader tuning | Unknown | Very high | Already near bandwidth limit |

---

## Profiling Snapshot (GGML_VK_PERF_LOGGER, per 5-token iteration)

Total: ~75,000 us (5 tokens) = ~15,000 us/tok

| Op | Time (us) | Count | Per-op (us) |
|----|-----------|-------|-------------|
| DELTA_NET_RECURRENCE | 13,758 | 120 | 115 |
| CPY | 1,571 | 240 | 6.5 |
| MULTI_ADD | 1,447 | 320 | 4.5 |
| GET_ROWS | 1,165 | 248 | 4.7 |
| MUL (broadcast) | 1,108 | 440 | 2.5 |
| TOPK_MOE | 1,060 | 160 | 6.6 |
| RMS_NORM_MUL (fused) | 988 | 524 | 1.9 |
| SIGMOID | 625 | 320 | 2.0 |
| GLU | 543 | 320 | 1.7 |
| CONT | 346 | 200 | 1.7 |
| CONCAT | 328 | 120 | 2.7 |
| SCALE | 273 | 120 | 2.3 |
| SILU | 262 | 240 | 1.1 |
| ROPE | 213 | 80 | 2.7 |
| REPEAT | 202 | 240 | 0.8 |

---

## Key Lessons Learned

1. **GPU warmup matters**: Cold-start benchmarks are ~24% slower on RDNA 3.5 iGPU. Always discard the first run.
2. **ggml_mul_mat computes A^T @ B**, not A @ B. Always verify matrix operation semantics before substituting.
3. **Sequential dependencies kill batching**. The SSM architecture creates long dependency chains. You need fused kernels, not batched dispatches.
4. **Non-coalesced reads are expensive**. Threads reading rows 512B apart waste 31/32 cache line bytes. Shared memory tiling with transposed layout solved this for +14.8%.
5. **Wave64 > wave32 on RDNA 3.5** by 8.7%. Let the driver choose (subgroup size = 0).
6. **HostCoherent memory is slow on UMA**. Use DeviceLocal only — the driver manages coherence more efficiently.
7. **32 KB shared memory limit on RDNA 3.5**, not 64 KB. Tile size must respect `maxComputeSharedMemorySize`.
8. **MoE models are bandwidth-efficient**: The 35B MoE (3B active) is faster than a 4B dense model because it reads fewer weights per token.
9. **Clear lingering processes before building**: `taskkill /f /im llama-cli.exe` prevents LNK1104 DLL lock errors.
10. **Measure before optimizing**: Assumptions about bottlenecks are wrong more often than right. Use GGML_VK_PERF_LOGGER.
