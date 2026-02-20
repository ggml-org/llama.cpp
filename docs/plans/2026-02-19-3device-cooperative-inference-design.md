# 3-Device Cooperative Inference Design

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Maximize token generation throughput by splitting every MUL_MAT across
Arc B580 + Arc Pro B50 + CPU simultaneously, combining ~459 GB/s total bandwidth.

**Architecture:** Per-op row-split (HeteGen-style) where each weight matrix is
partitioned across devices proportional to measured memory bandwidth. All devices
compute in parallel on every MUL_MAT, writing to non-overlapping output regions.

**Tech Stack:** SYCL (oneAPI), Level Zero, TBB, existing MMVQ kernels + CPU vec_dot

---

## Measured Hardware Data

All values from micro-benchmarks run on this machine (Feb 19, 2026):

| Device | Mem BW (GB/s) | TG tok/s | VRAM | PCIe D2H | PCIe H2D |
|--------|:------------:|:--------:|:----:|:--------:|:--------:|
| Arc B580 | 276 | 72.07 | 12 GB | 14.2 GB/s | 13.5 GB/s |
| Arc Pro B50 | 145 | 37.76 | 14 GB | 26.1 GB/s | 27.4 GB/s |
| CPU (20t DDR5) | 38 | ~10 | 128 GB | N/A | N/A |
| **Combined** | **~459** | **~120 theoretical** | **154 GB** | | |

**Validated assumptions:**
- Multi-device doesn't hang (SYCL alloc + memset + kernel on both GPUs: OK)
- Bandwidths are additive: proportional 3-device split = **1.67x** measured (1.66x theoretical)
- Concurrent dual-GPU execution: 1.97x speedup over sequential
- B50 has 2x PCIe bandwidth vs B580 (PCIe 5.0 x8 vs 4.0 x8)
- No Intel discrete GPU P2P: all cross-device goes through host memory
- CPU DDR5 peaks at 20 threads (37.8 GB/s), drops at 22+ threads

**Test artifacts:** `Testing/test-device-bandwidth.cpp`, `Testing/test-cpu-mt-bandwidth.cpp`,
`Testing/test-3device-parallel.cpp`

---

## Design Overview

### Per-Op Row-Split Execution (TG batch=1)

```
Weight Matrix W [K, N] for each MUL_MAT:

  B580 VRAM (SOA):  rows [0, N_b580)          60% of N
  B50 VRAM (SOA):   rows [N_b580, N_b50_end)  32% of N
  Host DDR5 (AOS):  rows [N_b50_end, N)        8% of N

  All 3 execute concurrently, write to non-overlapping output dst[0..N-1]
```

### Per-Token Execution Timeline

```
Time -->
B580 queue: ──[MMVQ partial rows 0..N_b580]──────────┤
B50 queue:  ──[MMVQ partial rows 0..N_b50]────────────┤
CPU thread: ──[vec_dot rows N_b50_end..N]──┤           │
                                            │           │
                                    merge ──┘───────────┘
                                    (~2KB total D2H + H2D)
```

### Split Ratio Configuration

```bash
# Bandwidth-proportional (default)
GGML_SYCL_SPLIT_RATIO="60,32,8"     # B580%, B50%, CPU%

# GPU-only split (no CPU contribution)
GGML_SYCL_SPLIT_RATIO="65,35,0"

# Auto-detect from measured bandwidth at init
GGML_SYCL_SPLIT_RATIO="auto"

# Device selector must expose both GPUs
ONEAPI_DEVICE_SELECTOR="level_zero:0,1"
```

---

## Weight Distribution

### At Model Load Time

The unified cache distributes each weight tensor across devices:

| Device | Layout | Storage | Access Pattern |
|--------|--------|---------|---------------|
| B580 | SOA (reordered) | 60% of rows per tensor | MMVQ kernel |
| B50 | SOA (reordered) | 32% of rows per tensor | MMVQ kernel |
| CPU | AOS (original) | 8% of rows per tensor | vec_dot |

**Total memory overhead:** 1.0x (each row stored once, on exactly one device).

For Mistral 7B Q4_0 (3.83 GB weights):
- B580: 2.30 GB VRAM used (of 12 GB)
- B50: 1.22 GB VRAM used (of 14 GB)
- CPU: 0.31 GB DDR5 (mmap-backed, negligible)

### SOA Layout Per Device

Both GPUs get SOA-reordered weights for their row ranges. The unified cache
already handles per-tensor SOA reordering. For multi-device, each device's cache
stores and reorders only its assigned rows.

CPU portion stays AOS (original quantized format) — `vec_dot` works on AOS.

---

## KV Cache Placement

KV cache stays on B580 (primary GPU):
- FLASH_ATTN_EXT runs entirely on B580
- K/V writes target B580 device memory
- No cross-device KV transfer during inference
- For large context: existing KV tiering (`GGML_SYCL_KV_HOST=1`) spills cold tokens
  to host-pinned memory accessible by all devices via PCIe zero-copy

---

## Multi-Device Queue Architecture

```
              ┌──────────────────────────────┐
              │  Shared sycl::context         │
              │  (B580 + B50 devices)         │
              └──────┬───────────┬───────────┘
                     │           │
              ┌──────┴──┐ ┌─────┴───┐
              │ Queue 0  │ │ Queue 1  │
              │ (B580)   │ │ (B50)    │
              │ in-order │ │ in-order │
              └──────────┘ └─────────┘

  CPU: TBB task_arena (existing g_task_arena from cpu-dispatch.cpp)

  Host memory: sycl::malloc_host() visible to ALL devices
```

Reuses existing TP shared context infrastructure (`common.cpp:117-150`):
one `sycl::context` containing both GPU devices, with per-device in-order queues.

### Src1 (Activation) Distribution

Per token, src1 [K] (16 KB at K=4096, float32) staged to all devices:

| Device | How src1 arrives | Latency |
|--------|-----------------|---------|
| B580 | Already there (primary) | 0 |
| B50 | H2D via `stream_b50->memcpy()` | ~0.6 us |
| CPU | D2H from B580 or read from host | ~1.1 us |

Negligible cost. Stage once per token, reuse for all MUL_MATs.

### Output Merge

Each MUL_MAT produces partial output on each device. Merge to B580's dst:

| Transfer | Size per MUL_MAT | Bandwidth | Latency |
|----------|:----------------:|:---------:|:-------:|
| B50 → host → B580 | N_b50 * 4 bytes (~4KB) | 14 GB/s | ~0.3 us |
| CPU → B580 | N_cpu * 4 bytes (~1KB) | 14 GB/s | ~0.1 us |

Total merge cost per MUL_MAT: ~0.4 us. For 32 layers * ~4 MUL_MATs = 128 merges
per token: ~50 us total. Negligible vs ~14 ms per token at 72 tok/s.

---

## Batched Dispatch Optimization

**Problem:** Current Phase 1 tensor split synchronizes per-MUL_MAT (stream->wait()
after each op), serializing the dispatch pipeline.

**Solution:** Batch all partial MUL_MATs per device:

1. **Pre-scan compute graph**: Identify all MUL_MAT ops in the TG graph
2. **Submit B580 batch**: All partial MMVQs to `stream_b580` (in-order queue pipelines them)
3. **Submit B50 batch**: All partial MMVQs to `stream_b50` (concurrent with B580)
4. **Run CPU batch**: All vec_dots sequentially on calling thread (concurrent with both GPUs)
5. **Wait once**: `stream_b580->wait(); stream_b50->wait();`
6. **Batch merge**: All B50/CPU output portions to B580 in one pass

This eliminates 128 per-op synchronizations, replacing them with 1 global sync.

---

## Per-Device Staging Buffers

Current single-device code uses global singletons. Multi-device requires per-device:

```cpp
struct split_device_state {
    sycl::queue *  queue       = nullptr;     // Device's queue
    float *        src1_staged = nullptr;     // Pinned src1 copy (16KB)
    float *        output_buf  = nullptr;     // Pinned output buffer
    size_t         output_size = 0;
    // Per-device weight cache for SOA partial tensors
    // managed by unified_cache in PER_DEVICE mode
};

static split_device_state g_split_devices[3];  // B580, B50, CPU
```

Allocated lazily via `sycl::malloc_host()` on first use, freed at backend cleanup.

---

## 3-Tier Model Size Strategy

### Tier 1: Model fits B580 alone (< 10 GB)
- Single-device, existing path
- 72 tok/s baseline

### Tier 2: Model fits combined GPU VRAM (10-20 GB)
- 3-device row-split per-op
- B580=60%, B50=32%, CPU=8%
- Target: ~100-112 tok/s (1.4-1.56x)

### Tier 3: Model exceeds GPU VRAM (> 20 GB)
- GPU layers: B580+B50 row-split per-op
- CPU layers: full layers via existing cpu-dispatch
- Activation pipeline between GPU and CPU sections
- Target: 15-40 tok/s depending on overflow ratio

---

## Existing Code Reused (No Modification)

| Component | Location | Why reusable |
|-----------|----------|-------------|
| MMVQ row splitting | `mmvq.cpp:4679-4840` | Has `row_low`/`row_high`/`total_nrows` |
| CPU vec_dot + TBB | `cpu-dispatch.cpp` | Thread pool, `g_task_arena`, per-type vec_dot |
| Host pointer map | `cpu-dispatch.cpp:g_host_ptr_map` | All weight mmap ptrs registered |
| TP shared context | `common.cpp:117-150` | Multi-device `sycl::context` + queues |
| SOA reorder | `unified-cache.cpp` | Per-tensor SOA layout transformation |
| Q8 quantization | `ggml-quants.c` | `from_float` for CPU vec_dot input |

## New Code Required

| Component | Est. Lines | Files |
|-----------|:----------:|-------|
| Split ratio config + init | ~40 | `ggml-sycl.cpp` |
| Per-device weight distribution | ~100 | `unified-cache.cpp` |
| 3-device MUL_MAT dispatch | ~120 | `ggml-sycl.cpp` |
| Batched dispatch + graph pre-scan | ~80 | `ggml-sycl.cpp` |
| Per-device staging buffers | ~40 | `ggml-sycl.cpp` |
| B50 MMVQ dispatch + src1 staging | ~50 | `ggml-sycl.cpp` |
| Output merge + sync | ~30 | `ggml-sycl.cpp` |
| **Total** | **~460** | |

---

## Performance Projections

### Mistral 7B Q4_0 (Tier 2)

| Configuration | TG128 (tok/s) | Speedup |
|--------------|:-------------:|:-------:|
| B580 alone (baseline) | 72 | 1.0x |
| B580 + B50 row-split | ~92 | 1.28x |
| B580 + B50 + CPU row-split | ~100-112 | 1.4-1.56x |

Conservative estimate accounts for:
- Merge overhead (~50 us/token → ~0.5% at 72 tok/s)
- Src1 staging (~2 us/token → negligible)
- Batched dispatch overhead (~100 us/token → ~1%)
- CPU being slower than optimal (38 vs theoretical 50 GB/s)

### Upper bound validation:
- Micro-benchmark showed 1.67x with proportional bandwidth split
- Real inference has non-MUL_MAT ops (attention, norms, SILU) that don't split
- MUL_MAT is ~85% of TG compute → max theoretical speedup ≈ 1/(0.15 + 0.85/1.67) = 1.51x
- Target range 1.4-1.56x is realistic

---

## Verification Plan

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# 1. Correctness (deterministic output)
GGML_SYCL_SPLIT_RATIO="60,32,8" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10" (matches single-GPU output)

# 2. No regression (GPU-only baseline)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expect: PP512 >= 1200, TG128 >= 68

# 3. Performance sweep
for ratio in "65,35,0" "60,32,8" "55,30,15" "50,25,25"; do
  echo "=== SPLIT_RATIO=$ratio ==="
  GGML_SYCL_SPLIT_RATIO="$ratio" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
    ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
  sleep 30  # Thermal cooldown
done
# Expect: best TG128 > 90 tok/s

# 4. Large model (Tier 3 overflow)
GGML_SYCL_SPLIT_RATIO="auto" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 'Hello,' -n 10 --seed 42 --temp 0
# Expect: correct output, no crash
```

---

## Research References

- **HeteGen** (MLSys 2024): Heterogeneous parallel inference with GPU+CPU cooperative
  MUL_MAT, bandwidth-proportional row splitting
- **NEO** (MLSys 2025): CPU offloading for online LLM inference with asymmetric pipelining
- **Nonuniform Tensor Parallelism** (arXiv 2504.06095): Handling heterogeneous device
  capabilities in parallel inference
- **Intel Level Zero**: No discrete GPU P2P — cross-device transfers go through host memory
- **SYCL Multi-Queue**: Out-of-order or separate in-order queues enable concurrent
  kernel execution across devices
