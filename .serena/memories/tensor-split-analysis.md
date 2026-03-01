# Tensor Split Implementation Analysis (Feb 19, 2026)

## 1. Current Single-Device Tensor Split Implementation

### Location: `/Apps/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:19932-20080`

**Function**: `ggml_sycl_mul_mat_tensor_split(ctx, src0, src1, dst, cpu_pct, src0_layout)`

**Purpose**: Cooperative GPU+CPU MUL_MAT where GPU computes rows [0, N_gpu) and CPU computes rows [N_gpu, ne01).

**Step-by-step flow**:

1. **Graph Recording Guard** (line 19943): Disabled during graph recording (returns false)
   - Reason: Async H2D copies and host_task incompatible with graph capture

2. **Row Split Calculation** (lines 19950-19957):
   - `N_cpu = ne01 * cpu_pct / 100` (user percentage)
   - `N_gpu = ne01 - N_cpu`
   - Round N_gpu UP to multiple of 16 (SOA MMVQ work-group granularity)
   - Clamp: if N_gpu > ne01, set N_gpu = ne01, N_cpu = 0
   - Minimum CPU work: if N_cpu < 32 rows, abort split (overhead not worth it)

3. **AOS Device Pointer Lookup** (lines 19964-19969):
   - Get AOS layout pointer from device via `ggml_sycl_get_layout_ptr_for(src0, device, GGML_LAYOUT_AOS)`
   - Returns false if AOS not available (fallback to full GPU)

4. **Pre-Stage Phase** (lines 19987-20009):
   - Weight lookup: `split_get_cached_weights(name, src0_aos, N_gpu, src0_row_bytes, cpu_weight_bytes, stream)`
     - Persistent host-pinned cache keyed by tensor name
     - D2H copies CPU rows (rows [N_gpu, ne01)) once, reused across tokens
   - Fallback: `split_weight_staging_ensure()` allocates temporary host-pinned buffer if cache miss
   - src1 pre-stage: `stream->memcpy(g_split_staging.src1_host, src1->data, src1_bytes).wait()`

5. **GPU Phase** (lines 20011-20034):
   - Q8_1 quantize of src1 on GPU
   - Call `ggml_sycl_op_mul_mat_vec_q()` with row range [0, N_gpu)
   - Outputs rows [0, N_gpu) to `dst_dd[0..N_gpu-1]`

6. **CPU Phase** (lines 20035-20045 or 20046-20077):
   - **Overlap path**: If src1 was pre-staged:
     - CPU runs `ggml_sycl_cpu_vec_dot_rows()` concurrently with GPU
     - Outputs to `g_split_staging.output`
     - Async H2D copy: `stream->memcpy(dst_dd + N_gpu, g_split_staging.output, out_bytes)` (no .wait())
   - **Fallback path**: If pre-staging failed:
     - D2H weights from device (blocking), then CPU compute (blocking), then H2D output (async)

### Bottlenecks Identified:

1. **Single-Device Only**: Currently dispatches all work to `ctx.device`
2. **Graph Incompatible**: Auto-disables graph replay during TG (line 22600+)
3. **Async H2D**: Output not awaited → CPU work must complete before graph replay next token
4. **Sequential D2H**: Weight D2H is blocking (`.wait()` line 19997, 20059)

---

## 2. MMVQ Kernel Row-Splitting Support

### Location: `/Apps/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:4679-4840`

**Function**: `ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high, src1_ncols, src1_padded_col_size, stream)`

**Row Splitting Parameters** (lines 4687-4690):
- `row_low`: First output row to compute (inclusive)
- `row_high`: Last output row to compute (exclusive)
- `src1_ncols`: Number of input columns
- `src1_padded_col_size`: K dimension padded for alignment

**Usage in Tensor Split** (line 20031-20033):
```cpp
ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                            src0_dd, nullptr, src1_ddq, dst_dd,
                            0, N_gpu, 1, K_padded, stream);  // GPU: rows [0, N_gpu)
```

**Capability**: YES, can be called with different row ranges
- Line 4763: `stream_ctx.row_base = row_low;` (base offset for row addressing)
- Line 4712: `const int64_t row_diff = row_high - row_low;` (computes diff for kernel dispatch)
- **Critical**: Kernel internally uses `row_low` to offset into src0 (quantized weights)
  - Line 4750: `const int64_t dst_row_stride = dst->ne[0];` (dst column count, fixed)
  - Outputs written to `dst_dd_i[row_low * dst_row_stride]` range

**Constraint**: SOA layout is per-tensor, not per-row-range
- Line 4730: `layout_base = ggml_sycl_get_layout_ptr_for(src0, device_id, GGML_LAYOUT_SOA);`
- All rows must use same layout (cannot mix SOA/AOS per device in single tensor)

---

## 3. Multi-Device Handling (Existing Non-Tensor-Split Case)

### Location: `/Apps/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:11900-12000`

**Multi-Device Row Split** (lines 11905-11920):
- Only used in traditional MUL_MAT (TP mode or weight sharding), NOT in tensor split
- Splits rows across devices using `tensor_split[device]` array
- Each device gets row range [row_low, row_high) per device

**Device Loop** (lines 11926-12161):
- Iterates over all devices in TP group or detected devices
- Each device creates its own queue: `queue_ptr stream = ctx.stream(i, 0)` (line 11935)
- Each device calls kernel independently with its own row range
- No cross-device coordination in single MUL_MAT

**Queue Creation** (line 11935, line 2195-2214 in common.hpp):
- `ctx.stream(device, stream_id)` returns queue for (device, stream_id)
- Cached in `qptrs[device][stream]` array (line 2177)
- Returns TP shared queue if TP enabled, else device's default queue

**Key Insight**: Multi-device support already exists for independent row ranges per device, just not integrated into tensor split.

---

## 4. Unified Cache & Weight Management

### Single-Device Unified Cache: `/Apps/llama.cpp/ggml/src/ggml-sycl/unified-cache.cpp:31`

```cpp
static std::unordered_map<int, std::unique_ptr<unified_cache>> g_device_caches;
```

**Per-Device Caches**: ONE cache instance per device ID
- Weights are per-device when in unified_cache_mode::PER_DEVICE (default)
- Each device has independent device VRAM, host pinned, mmap tiers

**Weight Replication Story**:
- **Single GPU fit**: Weights loaded to device 0 VRAM only
- **Multi-GPU (no TP)**: Each device creates its own cache, weights replicated to each VRAM
- **Multi-GPU (TP sharded)**: Each device gets sharded copy (K/world_size columns)

### No Multi-Device Tensor Split Currently:
- Tensor split `g_split_weight_cache` (line 19904) is global, not per-device
- Uses single host-pinned pool: `g_split_weight_staging` (line 19879)
- Assumes single queue/device for all operations

---

## 5. PCIe Topology & Shared Host Memory

### PCIe Topology (Arc B580 + Arc Pro B50):
- **Same root complex**: YES, both discrete GPUs share PCIe Root Complex in host chipset
- **Direct GPU-to-GPU access**: Supported via PCIe (P2P) but NOT used in tensor split
- **Shared Host Memory**: All GPUs can access host-pinned (malloc_host) via PCIe

### Host-Pinned Memory Types:

From `/Apps/llama.cpp/ggml/src/ggml-sycl/common.hpp` and memory notes:
- `sycl::malloc_host`: Host-pinned, PCIe-accessible by all GPUs, does NOT consume VRAM
- `sycl::malloc_shared`: USM shared, migrates between host/device, DOES consume VRAM
- **Current implementation**: Uses `malloc_host` for all staging (correct for multi-device)

### Current Staging Buffers:
```cpp
static struct {
    float * src1_host;          // malloc_host, shared by all devices (if single-threaded)
    float * output;             // malloc_host, shared by all devices
} g_split_staging;              // Lines 19841-19846

static struct {
    void * data;                // malloc_host for CPU weight rows
} g_split_weight_staging;       // Lines 19876-19879
```

**Limitation**: Single global buffers, not thread-safe for multi-device concurrent access.

---

## 6. Graph Recording Incompatibility

### Current Auto-Disable Logic:

**Location**: `/Apps/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:22595-22610`

```cpp
if (use_sycl_graph && cached_is_decode && ggml_sycl_tensor_split_pct() > 0) {
    // Tensor split provides ~14% TG improvement via GPU/CPU overlap, while
    // graph replay only adds ~2.8% for TG. Auto-disable graph for decode phase.
    use_sycl_graph = false;
}
```

**Reason**: 
1. Async H2D copy (line 20045) is not awaited → output may not be ready before next graph
2. Host_task in graph incompatible with replay (host_task requires queue sync)
3. Graph would serialize GPU→H2D→graph_start instead of allowing overlap

**Performance Impact**: 
- Tensor split TG: 9-12 tok/s (CPU offload, overlap benefit)
- Graph replay: +2.8% vs +14% benefit → not worth the complexity

---

## 7. Recent Implementation Status (Feb 19, 2026)

### Phase 1: Basic Tensor Split (e0668da88 - bfbcf6a51)
- GPU partial MMVQ + CPU vec_dot rows (sequential)

### Phase 2: Graph Replay Integration (Phase 1-2 complete)
- Commits: e0668da88, d70d47097
- Weight caching, async H2D, pre-staging
- Graph auto-disabled for TG

### Phase 3: Multi-Device Readiness (INCOMPLETE)
- **Missing**: Multi-device queue coordination
- **Missing**: Per-device weight staging
- **Missing**: Cross-device host memory synchronization
- **Blocker**: Each device needs independent D2H staging to avoid resource contention

### Recent CPU Optimization (c77f7a678 - a35bdb724)
- 4-row, 8-row, 16-row AVX2+VNNI kernels for CPU vec_dot
- Performance: ~2x improvement over generic vec_dot
- **No multi-device changes yet**

---

## 8. Key Files & Line References

| Component | File | Line | Purpose |
|-----------|------|------|---------|
| Tensor split main | ggml-sycl.cpp | 19932-20080 | Single-device GPU+CPU split |
| MMVQ kernel | mmvq.cpp | 4679-4840 | Row-ranged MUL_MAT vec_q |
| Multi-device (non-split) | ggml-sycl.cpp | 11900-12161 | Traditional per-device row split |
| Context structure | common.hpp | 2144-2214 | Queue management per device |
| TP config | common.hpp | 986-1003 | Multi-GPU coordination |
| Unified cache | unified-cache.cpp | 31-100 | Per-device cache instances |
| Graph auto-disable | ggml-sycl.cpp | 22595-22610 | Disable graph for TG + tensor split |
| Weight cache | ggml-sycl.cpp | 19897-19930 | Persistent host-pinned weights |
| Staging buffers | ggml-sycl.cpp | 19841-19894 | Global src1/output/weight buffers |
| Tensor split pct | ggml-sycl.cpp | 19820-19836 | Get split % from env var |

---

## 9. 3-Device Cooperative Inference (Feb 19, 2026)

### Status: Implementation Plan Created

**Epic:** llama.cpp-2rsa (P0)
**Design:** docs/plans/2026-02-19-3device-cooperative-inference-design.md
**Plan:** docs/plans/2026-02-19-3device-cooperative-inference.md

### Task Breakdown

| ID | Task | Track | Depends On | Status |
|----|------|-------|------------|--------|
| llama.cpp-a97p | T1: Split ratio config + device discovery | A | None | READY |
| llama.cpp-l74s | T2: Per-device weight distribution | B | None | READY |
| llama.cpp-8lwt | T3: Per-device staging buffers | A | T1 | Blocked |
| llama.cpp-xbpc | T4: 3-device MUL_MAT dispatch | A | T1,T2,T3 | Blocked |
| llama.cpp-aqzi | T5: Batched dispatch optimization | A | T4 | Blocked |
| llama.cpp-2lpg | T6: Integration verification | — | T4 | Blocked |

### Architecture Summary

Per-op MUL_MAT row-split: B580 (60%) + B50 (32%) + CPU (8%)
- B580 + B50: MMVQ on SOA, separate queues, concurrent
- CPU: vec_dot on AOS via TBB, concurrent with both GPUs
- Output merge: B50+CPU → B580 dst via host-pinned staging
- Env var: GGML_SYCL_SPLIT_RATIO="60,32,8"
- Graph replay disabled when multi-device split active

### Measured Hardware Data

| Device | Mem BW | TG tok/s | PCIe D2H |
|--------|--------|----------|----------|
| Arc B580 | 276 GB/s | 72 | 14.2 GB/s |
| Arc Pro B50 | 145 GB/s | 37.76 | 26.1 GB/s |
| CPU 20t DDR5 | 38 GB/s | ~10 | N/A |
| Combined | ~459 GB/s | ~100-112 target | |

---

## 10. Blocker for Multi-Device Tensor Split

### Issue: Single Global Staging Buffers (ADDRESSED in T3)
- `g_split_staging.src1_host` (line 19843): 1 buffer shared by all devices
- Fix: Per-device array g_split_staging[3]
- All allocated via sycl::malloc_host on shared context