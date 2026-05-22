---
name: blackwell-wgmma-cuda-backend
description: Refactor ggml-cuda backend from synchronous mma.sync (Ampere-style) to Asynchronous Warpgroup MMA (WGMMA) for full Blackwell (SM100/120/130) with TMA pipeline.
metadata:
  type: project
---

# Architecting Blackwell (SM 100/120/130) Support in llama.cpp — WGMMA Refactor

**Status**: Approved design, pending implementation plan
**Date**: 2026-05-22
**Target hardware**: RTX 50-series (SM100/101), datacenter Blackwell (SM120/121), Rubin (SM130)
**CUDA toolkit**: 12.8+ with `__CUDA_ARCH__` compile-time guards (fatbin per CC)
**Approach**: A — "Direct Path" WGMMA rewrite, full design with phased implementation

## Problem Statement

Current `mma.cuh` uses `m16n8k16.sync` instructions (warp-level, 32 threads) which force constant stalls and limit RTX 5090 utilization to ~40%. The `mmq.cuh` kernel uses `__syncthreads()` between every shared-memory tile load and MMA operation, with no async pipeline. No TMA (Tensor Memory Accelerator) or `cp.async.bulk` exists. Blackwell support is limited to SM120a FP4 block-scale `mma.sync`.

## Goals

1. Replace `mma.sync` with `wgmma.mma_sync` for all SM >= 1000, using 128-thread warpgroups on tiles up to 128x128
2. Integrate TMA + `.mbarrier` on SM120+ for zero-register data movement from gmem to smem
3. Achieve >2x tokens/second on RTX 5090 with Q4_K quantization
4. Support INT8 (Q4_K/Q8_0), FP8 (E4M3/E5M2), and FP4 (MXFP4/NVFP4) WGMMA variants
5. Maintain full backward compatibility — all pre-Blackwell paths unchanged

## Architecture

```
Kernel layer (mmq.cuh, mmf.cu)
  ├── mul_mat_q_wgmma<>  — new Blackwell kernel, 128-thread warpgroups
  └── mul_mat_q (existing) — unchanged for pre-Blackwell, default when CC < 1000

WGMMA primitives (mma.cuh — new #if __CUDA_ARCH__ >= 1000 block)
  ├── wgmma_mma_sync<>   — async WGMMA issue, accumulates in register fragments
  ├── wgmma_commit_sync  — commit all outstanding WGMMA ops
  ├── wgmma_wait<N>      — wait for N most recent commits
  └── wgmma_store_gmem<> — write-back accumulator to global memory (SM120+ only)

TMA pipeline (cp-async.cuh — new #if __CUDA_ARCH__ >= 1200 block)
  ├── tma_desc_t         — 256-bit TMA descriptor
  ├── tma_load_bulk<>    — cp.async.bulk.tensor.shared.global via descriptor
  └── mbarrier primitives — .arrived(), .wait(), .complete() for phase buffers

Legacy path (unchanged)
  └── mma.sync (current) — Volta/Turing/Ampere/Ada, zero modification
```

### Feature Availability Matrix

| Feature | SM100/101 | SM120/121 | SM130 |
|---------|-----------|-----------|-------|
| WGMMA `wgmma.mma_sync` | yes | yes | yes |
| WGMMA tile sizes m64N | yes | yes | yes |
| `cp.async.bulk` | no | yes | yes |
| TMA descriptor | no | yes | yes |
| `.mbarrier` | no | yes | yes |
| `wgmma.ldmatrix` | no | yes | yes |
| `wgmma.store_gmem` | no | yes | yes |

SM100 uses 2-stage `cp.async.cg.shared.global` pipelining as software fallback for data movement.

## Design Details

### WGMMA Primitives (mma.cuh)

#### Register Fragment Structs

Explicit `uint32_t` arrays — no `half2` or `float` reinterpreting — to guarantee register mapping for WGMMA PTX:

```cpp
struct frag_c_m16n8  { uint32_t x[1]; };   // 16x8  / 32 threads = 1 elem
struct frag_c_m16n16 { uint32_t x[2]; };   // 16x16 / 32 = 2
struct frag_c_m64n16 { uint32_t x[8]; };   // 64x16 / 32 = 8
struct frag_c_m64n32 { uint32_t x[16]; };  // 64x32 / 32 = 16
struct frag_c_m64n64 { uint32_t x[32]; };  // 64x64 / 32 = 32
struct frag_c_m64n128{ uint32_t x[64]; };  // 64x128 / 32 = 64
```

#### WGMMA PTX Intrinsics

Three data-type families:
- **INT8**: `wgmma.mma_sync.sync.aligned.m64nNh16.s32.s8.s8.s32` — for Q4_K, Q8_0, IQ-series (dequant→s8)
- **FP8 E4M3**: `wgmma.mma_sync.sync.aligned.m64nNh16.s32.e4m3.e4m3.s32`
- **FP8 E5M2**: `wgmma.mma_sync.sync.aligned.m64nNh16.s32.e5m2.e5m2.s32`

Pipeline control:
- `wgmma.commit_sync.sync` — commit all outstanding WGMMA ops
- `wgmma.wait_sync.sync N` — wait for N most recent commits. For triple-buffering, `wgmma_wait<2>` waits for the 2 most recent completed commits while the 3rd is still in flight.

#### Triple-Buffering Pipeline

```
Stage 0: load_tile_A[0], load_tile_B[0]  →  wgmma(0)  →  commit
Stage 1: load_tile_A[1], load_tile_B[1]  →  wgmma(1)  →  commit
Stage 2: load_tile_A[2], load_tile_B[2]  →  wgmma(2)  →  commit
         wgmma_wait<2>  // wait for stage 0+1 while 2 is flying
         process_result[0]
         wrap: load_tile_A[0]... (reuse buffer)
```

#### Feature Guards

```cpp
#if __CUDA_ARCH__ >= 1000
#define BLACKWELL_WGMMMA_AVAILABLE    // WGMMA compute on all Blackwell
#endif
#if __CUDA_ARCH__ >= 1200
#define BLACKWELL_TMA_AVAILABLE       // cp.async.bulk + .mbarrier
#define WGMMMA_STORE_GMEM_AVAILABLE   // wgmma.store_sync to gmem
#endif
#if __CUDA_ARCH__ >= 1300
#define RUBIN_WGMMA_AVAILABLE         // Future Rubin-specific extensions
#endif
```

### TMA Pipeline (cp-async.cuh + mmq.cuh)

#### TMA Descriptor (SM120+ only)

A 256-bit (32-byte) TMA store descriptor, built on host at kernel launch time and passed via constant memory:

```cpp
struct alignas(32) tma_desc_t {
    uint32_t gmem_address_lo, gmem_address_hi;
    uint32_t gmem_stride_lo, gmem_stride_hi;
    uint32_t gmem_element_stride;
    uint32_t smem_size_lo, smem_stride_lo, smem_element_stride;
    uint32_t data_type;        // S8, E4M3, E5M2, etc.
    uint32_t smem_swizzle;     // 64B/128B/256B/512B
    uint32_t gmem_round;
    uint32_t padding[3];
};
```

Descriptors built per-matrix (src0, src1, dst) at dispatch time in `mmq.cu`, placed in `__constant__` memory.

#### mbarrier Synchronization

Two phase buffers per stage in shared memory. The TMA load increments one barrier; WGMMA waits on the barrier of the stage it's consuming:

```cpp
__shared__ uint64_t barrier_phase[StageCount];
// Init: mbarrier.init.shared::cta.b64 %0, 128 (warpgroup size)
```

#### SM100 Fallback

Software-pipelined `cp.async.cg.shared.global` with 2-stage double-buffering. Reuses existing `cp-async.cuh` primitives extended with per-stage wait (`cp_async_wait_stage<N>`).

#### Alignment

Buffer allocator already guarantees 128-byte alignment (`ggml_backend_cuda_buffer_type_get_alignment` returns 128). TMA tile offsets within quantized buffers are aligned to 128-byte boundaries; non-aligned inner tiles fall back to `cp.async.cg` (negligible branching). `MATRIX_ROW_PADDING` (512 elements) is a multiple of 128 bytes for all quant types.

### Blackwell-Optimized mul_mat_q Kernel

#### Warpgroup Orchestration

- **Block size**: 128 threads (4 warps), dim `(128, 1, 1)`
- **Tile sizes**: 64x64, 64x128, 128x128 — selected by dispatch based on batch size and SM count
- **One warpgroup per SM per tile** — maximizes occupancy

#### Quant Type Support

| Quant Type | WGMMA Input Format | Notes |
|------------|-------------------|-------|
| Q4_K, Q5_K, Q6_K | Dequant→s8, then `wgmma.s32.s8.s8.s32` | Dequant in shared mem load stage |
| Q8_0 | Direct `wgmma.s32.s8.s8.s32` | No dequant needed |
| IQ2_XXS…IQ4_NL | Dequant→s8, then `wgmma.s32.s8.s8.s32` | Grid lookup in load stage |
| MXFP4/NVFP4 | `wgmma.s32.e2m1.e2m1.s32` (native FP4) | Reuse existing Blackwell block-scale PTX |
| Future FP8 | `wgmma.s32.e4m3.e4m3.s32` / `e5m2.e5m2.s32` | Template parameter |

#### Result Writeback

- **SM120+**: `wgmma.store_sync.sync.aligned.m64nNh16.f32.s32` — direct warpgroup→gmem
- **SM100 / fallback**: Each thread writes its `frag_c.x[i]` elements via strided global stores with L2 cache hint

#### Dispatch

New `blackwell_wgmma_available(cc)` runtime check (CC >= 1000). Kernel selection in `ggml_cuda_mul_mat_q()`:

```cpp
if (blackwell_wgmma_available(cc)) {
    // Select tile: prefer 64x128 for large batches, 64x64 for small
    // Build TMA descriptors for SM120+
    // Launch mul_mat_q_wgmma<<128, shared_mem>>
} else {
    // Existing mma.sync path — unchanged
}
```

### NUMA-Awareness (Phase 1: log + hints)

- Detect PCI-to-NUMA mapping in `ggml_cuda_init()`
- Log GPU NUMA node and suggest `numactl --cpunodebind=N` or `taskset`
- Full NUMA-aware allocation (`numa_alloc_onnode`) deferred to Phase 3

## Phased Implementation

### Phase 1 — WGMMA Direct Path (mma.cuh + CMake)

- `common.cuh`: add `GGML_CUDA_CC_BLACKWELL_WG = 1000`, new guards
- `mma.cuh`: register fragments, `wgmma_mma_sync<>`, `wgmma_commit_sync`, `wgmma_wait<>`
- `CMakeLists.txt`: add `100a-real`, `101a-real` architectures (CUDA 12.8+)
- `mmq.cu`: `blackwell_wgmma_available(cc)` runtime check
- **Test**: Build each CC, verify PTX with `cuobjdump --dump-ptx`

### Phase 2 — TMA Pipeline + mmq Kernel (throughput)

- `cp-async.cuh`: TMA descriptor, `tma_load_bulk<>`, `.mbarrier` (SM120+); `cp_async_wait_stage` (SM100)
- `mmq.cuh`: new `mul_mat_q_wgmma<>` kernel with triple-buffered pipeline
- `mmq.cu`: dispatch logic, tile selection, TMA descriptor setup
- **Test**: `test-backend-ops.cpp` — WGMMA vs mma.sync result comparison, delta < 1e-3

### Phase 3 — NUMA Polish + FP8

- `ggml-cuda.cu`: NUMA detection + logging in `ggml_cuda_init()`
- `mma.cuh`: FP8 WGMMA templates (`e4m3`, `e5m2`)
- `mma.cuh`: SM120+ `wgmma.store_gmem` writeback
- **Test**: NUMA log verification, FP8 numerical test

## Testing Strategy

- **Correctness**: Per-quant-type WGMMA vs mma.sync comparison in `test-backend-ops`
- **Performance**: `llama-bench` t/s on Llama-3.1-70B-Q4_K, target >2x baseline
- **Build**: All CC variants compile; `cuobjdump` confirms WGMMA PTX presence
- **Edge**: Non-aligned tile fallback, boundary conditions for stream-K fixup

## Risks

- **nvcc WGMMA support**: CUDA 12.8+ may have incomplete WGMMA PTX parsing. Mitigation: test early with `cuobjdump`, fall back to `mma.sync` if PTX generation fails
- **Register pressure**: 128-thread warpgroups with `frag_c_m64n128` (64 uint32 per thread) could exceed register budget. Mitigation: `__launch_bounds__` with register spill budget analysis
- **TMA descriptor alignment**: Per-tile offsets must be 128-byte aligned. Mitigation: fallback to `cp.async.cg` for non-aligned tiles
