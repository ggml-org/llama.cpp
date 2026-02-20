# 3-Device Cooperative Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Extend tensor split to use B580 + B50 + CPU simultaneously for per-op MUL_MAT row splitting, targeting ~100 tok/s TG (1.4x over single-GPU 72 tok/s).

**Architecture:** Each MUL_MAT weight matrix is row-split across 3 devices proportional to measured bandwidth (B580=60%, B50=32%, CPU=8%). All devices compute concurrently on every MUL_MAT, writing to non-overlapping output regions. No all-reduce — just concatenation.

**Tech Stack:** SYCL (oneAPI), Level Zero, TBB, existing MMVQ kernels + CPU vec_dot

**Design Doc:** `docs/plans/2026-02-19-3device-cooperative-inference-design.md`

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 4, 5 | Multi-device init + 3-device dispatch + batched optimization (ggml-sycl.cpp) |
| B | 2 | Per-device weight distribution (unified-cache.cpp/hpp) |
| — | 3 | Per-device staging buffers (depends on A.1, touches ggml-sycl.cpp) |
| — | 6 | Integration + verification (depends on all) |

### Dependency Graph

```
Task 1 (A: device init) ──────────┐
                                   ├── Task 3 (staging) ──┐
Task 2 (B: weight distribution) ──┘                       ├── Task 4 (3-device dispatch) ── Task 5 (batched) ── Task 6 (verification)
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 1, 3, 4, 5 | Sequential (same track after merge) |
| `ggml/src/ggml-sycl/unified-cache.cpp` | 2 | None (single task) |
| `ggml/src/ggml-sycl/unified-cache.hpp` | 2 | None (single task) |
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 4 | None (minor addition) |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | 4 | None (minor addition) |

---

## Existing Infrastructure (No Modification Needed)

| What | Where | Why reusable |
|------|-------|-------------|
| MMVQ row splitting | `mmvq.cpp:4679-4840` | Has `row_low`/`row_high`/`total_nrows` params |
| CPU vec_dot + TBB | `cpu-dispatch.cpp` | Thread pool, `g_task_arena`, per-type vec_dot |
| Host pointer map | `cpu-dispatch.cpp:75` `g_host_ptr_map` | All weight mmap ptrs registered |
| SOA reorder | `unified-cache.cpp` | Per-tensor SOA layout transformation |
| Q8 quantization | `ggml-quants.c` | `from_float` for CPU vec_dot input |
| Single-device tensor split | `ggml-sycl.cpp:19932-20080` | GPU+CPU split, staging, weight cache |

---

### Task 1: Multi-Device Discovery and Split Ratio Configuration

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (add ~60 lines near line 19820)

**Description:**

Replace the single `GGML_SYCL_TENSOR_SPLIT` env var (CPU percentage) with `GGML_SYCL_SPLIT_RATIO` (3-way percentage: `"60,32,8"` for B580/B50/CPU). Discover all Level Zero GPU devices at init. Create a shared `sycl::context` containing both GPUs with per-device in-order queues. This builds on the existing TP shared context infrastructure at `common.cpp:117-150`.

**Acceptance Criteria:**

- [ ] `GGML_SYCL_SPLIT_RATIO="60,32,8"` parsed into 3 percentages that sum to 100
- [ ] `GGML_SYCL_SPLIT_RATIO="auto"` computes ratios from measured bandwidth
- [ ] Old `GGML_SYCL_TENSOR_SPLIT=N` still works (maps to `100-N,0,N`)
- [ ] B580 and B50 devices discovered via Level Zero enumeration
- [ ] Shared `sycl::context` created with both GPUs
- [ ] Per-device in-order queues stored in static struct
- [ ] Graceful fallback: if <2 GPUs found, disable multi-device (single-GPU+CPU as before)
- [ ] No regression when `SPLIT_RATIO` not set (default disabled)

**Implementation Guide:**

1. **Replace `ggml_sycl_tensor_split_pct()`** at line 19820 with new config struct:

```cpp
struct split_device_config {
    float    ratio[3]     = {1.0f, 0.0f, 0.0f};  // B580, B50, CPU fractions
    sycl::queue * queue[2] = {nullptr, nullptr};   // GPU queues (0=B580, 1=B50)
    int      gpu_count    = 0;                     // 1 or 2
    bool     enabled      = false;                 // any non-trivial split
};

static split_device_config g_split_config;

static void split_config_init(sycl::queue * primary_queue) {
    if (g_split_config.queue[0]) return;  // already init'd
    g_split_config.queue[0] = primary_queue;

    // Parse GGML_SYCL_SPLIT_RATIO
    const char * ratio_env = std::getenv("GGML_SYCL_SPLIT_RATIO");
    const char * legacy_env = std::getenv("GGML_SYCL_TENSOR_SPLIT");

    if (ratio_env) {
        if (strcmp(ratio_env, "auto") == 0) {
            // Auto-detect: TODO in Task 6, use fixed 60/32/8 for now
            g_split_config.ratio[0] = 0.60f;
            g_split_config.ratio[1] = 0.32f;
            g_split_config.ratio[2] = 0.08f;
        } else {
            int r0, r1, r2;
            if (sscanf(ratio_env, "%d,%d,%d", &r0, &r1, &r2) == 3) {
                float sum = (float)(r0 + r1 + r2);
                g_split_config.ratio[0] = r0 / sum;
                g_split_config.ratio[1] = r1 / sum;
                g_split_config.ratio[2] = r2 / sum;
            }
        }
    } else if (legacy_env) {
        int cpu_pct = std::atoi(legacy_env);
        g_split_config.ratio[0] = (100.0f - cpu_pct) / 100.0f;
        g_split_config.ratio[1] = 0.0f;
        g_split_config.ratio[2] = cpu_pct / 100.0f;
    }

    g_split_config.enabled = (g_split_config.ratio[1] > 0.0f
                              || g_split_config.ratio[2] > 0.0f);

    // Discover B50 (secondary GPU)
    if (g_split_config.ratio[1] > 0.0f) {
        // Enumerate Level Zero GPUs beyond device 0
        for (auto & p : sycl::platform::get_platforms()) {
            if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            auto devs = p.get_devices(sycl::info::device_type::gpu);
            if (devs.size() >= 2) {
                // Create shared context with both GPUs
                sycl::context shared_ctx({devs[0], devs[1]});
                static sycl::queue q1(shared_ctx, devs[1],
                    sycl::property::queue::in_order());
                g_split_config.queue[1] = &q1;
                g_split_config.gpu_count = 2;
                GGML_LOG_INFO("SYCL 3-device split: %s + %s + CPU "
                    "(%.0f%%/%.0f%%/%.0f%%)\n",
                    devs[0].get_info<sycl::info::device::name>().c_str(),
                    devs[1].get_info<sycl::info::device::name>().c_str(),
                    g_split_config.ratio[0] * 100,
                    g_split_config.ratio[1] * 100,
                    g_split_config.ratio[2] * 100);
                break;
            }
        }
        if (!g_split_config.queue[1]) {
            // B50 not found — redistribute to B580+CPU only
            GGML_LOG_WARN("SYCL: B50 not found, falling back to 2-device split\n");
            g_split_config.ratio[0] += g_split_config.ratio[1];
            g_split_config.ratio[1] = 0.0f;
        }
    }
}
```

2. **Update the existing tensor split gate** at line ~19852 to use new config:

```cpp
// Old:
// if (tensor_split_cpu_pct > 0 && has_soa_reorder && !g_ggml_sycl_graph_recording) {
// New:
split_config_init(stream);
if (g_split_config.enabled && has_soa_reorder && !g_ggml_sycl_graph_recording) {
```

3. **Test: Verify env var parsing**

Build and run with various env vars:
```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# Legacy mode (should work as before)
GGML_SYCL_TENSOR_SPLIT=13 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10" and log line showing single-device split

# New 3-device mode (should log discovery)
GGML_SYCL_SPLIT_RATIO="60,32,8" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: log "SYCL 3-device split: Arc B580 + Arc Pro B50 + CPU (60%/32%/8%)"
# Output may be wrong (weight distribution not done yet) — that's OK for Task 1
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add 3-device split ratio config and multi-device discovery"
```

**Notes for implementer:**
- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` exposes both discrete GPUs
- The shared context is critical — both queues must share a context for host-pinned memory visibility
- `sycl::malloc_host` allocated on the shared context is accessible by BOTH GPUs via PCIe zero-copy
- Don't worry about actually dispatching to B50 yet — Task 4 handles that

---

### Task 2: Per-Device Weight Distribution in Unified Cache

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (~80 lines)
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (~15 lines)

**Description:**

Add a `load_partial_rows()` method to the unified cache that loads only a contiguous subset of rows from a weight tensor to a specific device's VRAM, with SOA reordering for that row range. This enables B580 and B50 each storing only their assigned rows. The CPU portion stays AOS (mmap-backed, already in `g_host_ptr_map`).

Current weight loading does full-tensor SOA reorder to a single device. This task adds row-range-aware loading for the multi-device case.

**Acceptance Criteria:**

- [ ] New method `load_partial_rows(tensor, device, row_start, row_count)` on unified cache
- [ ] SOA reorder applied only to the specified row range (not full tensor)
- [ ] VRAM consumption proportional to row count (B580 uses 60% VRAM, B50 uses 32%)
- [ ] AOS pointer for CPU portion accessible via existing `g_host_ptr_map` (no new code)
- [ ] Budget tracking updated: each device tracks only its portion
- [ ] Existing single-device path unchanged when multi-device disabled
- [ ] New public function to query device-specific weight pointer: `get_split_weight_ptr(tensor, device)`

**Implementation Guide:**

1. **Add row-range load to unified cache header** (`unified-cache.hpp`):

```cpp
// After existing load() method declaration:

// Load a contiguous row range of a weight tensor to a specific device.
// Performs SOA reorder for the specified rows only.
// Returns device pointer to the reordered partial tensor.
void * load_partial_rows(const ggml_tensor * tensor,
                         int device,
                         int64_t row_start,
                         int64_t row_count,
                         ggml_type type);
```

2. **Implement `load_partial_rows()`** in `unified-cache.cpp`:

The SOA reorder logic in the existing `reorder_to_soa()` works on full tensors. For partial rows, we need to:
- Compute byte offset for `row_start` in the AOS source: `row_start * ggml_row_size(type, ne00)`
- Allocate `row_count * ggml_row_size(type, ne00)` bytes on the target device
- Copy the AOS row range from host to a temporary host buffer
- Apply SOA reorder on just those rows
- Copy the SOA result to device VRAM

```cpp
void * unified_cache::load_partial_rows(const ggml_tensor * tensor,
                                        int device,
                                        int64_t row_start,
                                        int64_t row_count,
                                        ggml_type type) {
    const int64_t ne00 = tensor->ne[0];
    const size_t row_bytes = ggml_row_size(type, ne00);
    const size_t partial_bytes = row_count * row_bytes;

    // Source: AOS data from host (mmap or pinned)
    const char * src = (const char *)tensor->data + row_start * row_bytes;

    // Allocate on target device
    auto & queue = get_queue_for_device(device);
    void * dev_ptr = sycl::malloc_device(partial_bytes, queue);
    if (!dev_ptr) return nullptr;

    // SOA reorder the partial rows
    // Create a temporary tensor descriptor for just the row range
    // to reuse existing reorder_to_soa() logic
    void * host_soa = sycl::malloc_host(partial_bytes, queue);
    if (!host_soa) {
        sycl::free(dev_ptr, queue);
        return nullptr;
    }

    // Reorder AOS→SOA for row_count rows
    // Use existing SOA reorder with adjusted ne[1] = row_count
    ggml_tensor partial_desc = *tensor;
    partial_desc.ne[1] = row_count;
    partial_desc.data = (void *)src;
    reorder_tensor_to_soa(&partial_desc, host_soa, type);

    // H2D copy SOA data to device
    queue.memcpy(dev_ptr, host_soa, partial_bytes).wait();
    sycl::free(host_soa, queue);

    // Track in cache for later lookup
    // Key: tensor name + device ID
    std::string key = std::string(tensor->name) + ":" + std::to_string(device);
    partial_cache_[key] = {dev_ptr, device, partial_bytes};

    // Update budget
    update_budget(device, partial_bytes);

    return dev_ptr;
}
```

3. **Add lookup function**:

```cpp
void * unified_cache::get_split_weight_ptr(const ggml_tensor * tensor, int device) {
    std::string key = std::string(tensor->name) + ":" + std::to_string(device);
    auto it = partial_cache_.find(key);
    return (it != partial_cache_.end()) ? it->second.ptr : nullptr;
}
```

4. **Add member variables** to the cache class:

```cpp
struct partial_entry {
    void * ptr;
    int    device;
    size_t bytes;
};
std::unordered_map<std::string, partial_entry> partial_cache_;
```

5. **Test: Verify partial row loading**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# Test that existing single-device path still works
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expect: PP512 >= 1200, TG128 >= 68 (no regression)
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-cache.cpp ggml/src/ggml-sycl/unified-cache.hpp
git commit -m "sycl: add partial row loading to unified cache for multi-device weight distribution"
```

**Notes for implementer:**
- The existing `reorder_tensor_to_soa()` function handles per-tensor SOA layout. You need to understand how it works with ne[1] (number of rows) to correctly reorder a subset.
- Each device's VRAM only stores its portion — B580 gets 60% of rows, B50 gets 32%. CPU reads AOS directly from mmap.
- Budget tracking: each device's cache should only account for its partial allocation, not the full tensor.
- Use Serena's `find_symbol` with `reorder_tensor_to_soa` to find the existing implementation.
- The `partial_cache_` map must be thread-safe if accessed from multiple device queues (use a mutex).

---

### Task 3: Per-Device Staging Buffers

**Track:** — (convergence point)
**Depends on:** Task 1
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~40 lines, refactor near line 19841)

**Description:**

Replace the global `g_split_staging` singleton with a per-device array `g_split_staging[3]` (B580, B50, CPU). Each device needs independent staging buffers for src1 (activation input) and output (partial MUL_MAT result). This prevents contention when B580 and B50 submit work concurrently.

**Acceptance Criteria:**

- [ ] `g_split_staging` becomes `g_split_staging[3]` with per-device src1 and output buffers
- [ ] B580 staging: `sycl::malloc_host` on B580's queue context
- [ ] B50 staging: `sycl::malloc_host` on shared context (accessible by B50)
- [ ] CPU staging: plain `aligned_alloc` (no SYCL allocation needed)
- [ ] Lazy allocation on first use, freed at backend cleanup
- [ ] Existing single-device tensor split uses `g_split_staging[0]` (no regression)

**Implementation Guide:**

1. **Replace global staging struct** at line 19841:

```cpp
// Old:
// static struct { ... } g_split_staging;

// New:
struct split_staging {
    float * src1_host = nullptr;   // host-pinned activation staging
    float * output    = nullptr;   // host-pinned partial output
    size_t  src1_size = 0;
    size_t  out_size  = 0;
};
static split_staging g_split_staging[3];  // [0]=B580, [1]=B50, [2]=CPU

static void split_staging_ensure(int dev_idx, size_t src1_bytes, size_t out_bytes,
                                  sycl::queue * queue) {
    auto & s = g_split_staging[dev_idx];
    if (s.src1_size < src1_bytes) {
        if (s.src1_host) sycl::free(s.src1_host, *queue);
        s.src1_host = sycl::malloc_host<float>(src1_bytes / sizeof(float), *queue);
        s.src1_size = src1_bytes;
    }
    if (s.out_size < out_bytes) {
        if (s.output) sycl::free(s.output, *queue);
        s.output = sycl::malloc_host<float>(out_bytes / sizeof(float), *queue);
        s.out_size = out_bytes;
    }
}
```

2. **Update existing tensor split function** to use `g_split_staging[0]` instead of `g_split_staging`:

Search and replace within `ggml_sycl_mul_mat_tensor_split()`:
- `g_split_staging.src1_host` → `g_split_staging[0].src1_host`
- `g_split_staging.output` → `g_split_staging[0].output`

3. **Test: Verify no regression in single-device mode**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

GGML_SYCL_TENSOR_SPLIT=13 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10" (identical to before)
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: refactor tensor split staging to per-device arrays"
```

**Notes for implementer:**
- `sycl::malloc_host` must be allocated with a queue from the shared context so BOTH GPUs can access it
- The CPU (dev_idx=2) staging output buffer also uses `sycl::malloc_host` because the B580 GPU needs to read it (D2H is to B580's dst)
- Check the existing `split_staging_ensure()` pattern (if it exists) near line 19860

---

### Task 4: 3-Device MUL_MAT Dispatch

**Track:** A (after Task 3)
**Depends on:** Task 1, Task 2, Task 3
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~120 lines, extend tensor split function)
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (~10 lines, add batched vec_dot)
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.hpp` (~5 lines, declaration)

**Description:**

Extend `ggml_sycl_mul_mat_tensor_split()` to dispatch MUL_MAT across 3 devices concurrently. B580 and B50 each run MMVQ on their SOA row ranges via separate queues. CPU runs vec_dot on its AOS row range via TBB. All three compute in parallel. Output merge copies B50 and CPU partial results to B580's dst.

This is the core performance-critical change. The key is:
- B580 MMVQ submitted to `g_split_config.queue[0]` (async)
- B50 MMVQ submitted to `g_split_config.queue[1]` (async)
- CPU vec_dot runs on calling thread (concurrent with both GPUs)
- Single barrier: wait for both GPU queues
- Merge: H2D copy B50+CPU outputs to B580 dst

**Acceptance Criteria:**

- [ ] `GGML_SYCL_SPLIT_RATIO="60,32,8"` dispatches to B580+B50+CPU
- [ ] B580 MMVQ processes rows [0, N_b580)
- [ ] B50 MMVQ processes rows [N_b580, N_b580+N_b50)
- [ ] CPU vec_dot processes rows [N_b580+N_b50, ne01)
- [ ] All three run concurrently (not serialized)
- [ ] Output merge: B50 and CPU outputs copied to B580 dst
- [ ] Correct output: `llama-completion` with seed 42 matches single-GPU output
- [ ] GPU-only split works: `GGML_SYCL_SPLIT_RATIO="65,35,0"` (no CPU)
- [ ] Graceful fallback if B50 weight not loaded (return false, single-device path)

**Implementation Guide:**

1. **Modify `ggml_sycl_mul_mat_tensor_split()`** to handle 3-device:

```cpp
static bool ggml_sycl_mul_mat_tensor_split(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    int cpu_pct,  // legacy param, ignored if g_split_config has 3-way
    layout_mode src0_layout) {

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // N (rows)

    // Compute 3-way row split
    int64_t N_b580, N_b50, N_cpu;
    if (g_split_config.gpu_count >= 2) {
        N_b580 = (int64_t)(ne01 * g_split_config.ratio[0]);
        N_b50  = (int64_t)(ne01 * g_split_config.ratio[1]);
        // Round to multiples of 16 (MMVQ work-group granularity)
        N_b580 = (N_b580 + 15) & ~15;
        N_b50  = (N_b50 + 15) & ~15;
        N_cpu  = ne01 - N_b580 - N_b50;
        if (N_cpu < 0) { N_b50 += N_cpu; N_cpu = 0; }
    } else {
        // Legacy 2-device (GPU+CPU)
        N_cpu  = ne01 * cpu_pct / 100;
        N_b580 = ne01 - N_cpu;
        N_b50  = 0;
    }

    if (N_b580 < 16) return false;  // Not enough work for primary GPU

    // === Get weight pointers ===

    // B580: use existing SOA weight on device 0
    const void * src0_b580 = /* existing SOA device pointer */;

    // B50: lookup partial SOA weight loaded by Task 2
    const void * src0_b50 = nullptr;
    if (N_b50 > 0) {
        src0_b50 = unified_cache_get_split_weight_ptr(src0, 1 /*device*/);
        if (!src0_b50) return false;  // Not loaded yet, fallback
    }

    // CPU: AOS from host mmap
    const void * src0_cpu = nullptr;
    if (N_cpu > 0) {
        auto * base = ggml_sycl_cpu_dispatch_get_host_ptr(src0->name);
        if (!base) return false;
        src0_cpu = (const char *)base + (N_b580 + N_b50) * ggml_row_size(src0->type, ne00);
    }

    // === Stage src1 to all devices ===

    const size_t src1_bytes = ne00 * sizeof(float);
    auto * stream_b580 = g_split_config.queue[0];

    // Ensure staging buffers for all devices
    split_staging_ensure(0, src1_bytes, N_b580 * sizeof(float), stream_b580);
    if (N_b50 > 0) {
        auto * stream_b50 = g_split_config.queue[1];
        split_staging_ensure(1, src1_bytes, N_b50 * sizeof(float), stream_b50);
        // Stage src1 to B50's staging buffer (H2D)
        stream_b50->memcpy(g_split_staging[1].src1_host, src1->data, src1_bytes);
        // Don't wait — B50 queue is in-order, MMVQ will wait implicitly
    }
    if (N_cpu > 0) {
        split_staging_ensure(2, src1_bytes, N_cpu * sizeof(float), stream_b580);
        // Stage src1 to CPU staging (D2H from B580 or read from host)
        stream_b580->memcpy(g_split_staging[2].src1_host, src1->data, src1_bytes).wait();
    }

    // === Submit GPU MMVQ (both async) ===

    // B580: rows [0, N_b580)
    // Q8_1 quantize src1 on B580 (existing code)
    quantize_src1_to_q8(ctx, src0, src1, stream_b580);
    ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
        src0_b580, nullptr, src1_ddq, dst_dd,
        0, N_b580, 1, K_padded, stream_b580);

    // B50: rows [0, N_b50) on B50's SOA data (row indices are 0-based within B50's partial tensor)
    if (N_b50 > 0) {
        auto * stream_b50 = g_split_config.queue[1];
        // Q8_1 quantize src1 on B50
        quantize_src1_to_q8_on_device(stream_b50, src1, ...);
        ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
            src0_b50, nullptr, src1_ddq_b50, g_split_staging[1].output,
            0, N_b50, 1, K_padded, stream_b50);
    }

    // === CPU vec_dot (concurrent with GPU) ===

    if (N_cpu > 0) {
        ggml_sycl_cpu_vec_dot_rows(
            src0->type, ne00, src0_cpu,
            g_split_staging[2].src1_host,
            g_split_staging[2].output, N_cpu);
    }

    // === Wait for both GPUs ===
    stream_b580->wait();
    if (N_b50 > 0) g_split_config.queue[1]->wait();

    // === Merge outputs to B580 dst ===
    float * dst_dd = (float *)dst->data;
    if (N_b50 > 0) {
        // B50 output → B580 device memory
        stream_b580->memcpy(dst_dd + N_b580,
                            g_split_staging[1].output,
                            N_b50 * sizeof(float));
    }
    if (N_cpu > 0) {
        stream_b580->memcpy(dst_dd + N_b580 + N_b50,
                            g_split_staging[2].output,
                            N_cpu * sizeof(float));
    }
    if (N_b50 > 0 || N_cpu > 0) stream_b580->wait();

    return true;
}
```

2. **Handle B50 MMVQ dispatch details:**

The MMVQ kernel needs the B50 queue, B50's SOA weight pointer, and B50's Q8 quantized src1. Key challenges:
- B50 needs its own Q8_1 quantized src1 buffer on B50 VRAM
- MMVQ dispatch uses `ggml_sycl_op_mul_mat_vec_q()` which takes a stream parameter — pass B50's queue
- B50's partial SOA data has rows 0..N_b50 (zero-indexed), not N_b580..N_b580+N_b50

3. **Test: Correctness**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# 3-device correctness
GGML_SYCL_SPLIT_RATIO="60,32,8" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# GPU-only split (no CPU)
GGML_SYCL_SPLIT_RATIO="65,35,0" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/cpu-dispatch.cpp \
  ggml/src/ggml-sycl/cpu-dispatch.hpp
git commit -m "sycl: implement 3-device MUL_MAT dispatch with B580+B50+CPU row split"
```

**Notes for implementer:**
- The B50 MMVQ dispatch must use a SEPARATE Q8_1 buffer allocated on B50's device. Cannot share B580's Q8 buffer.
- B50's SOA weight pointer has row indices starting from 0 (not N_b580). The `row_low=0, row_high=N_b50` is correct because B50's partial tensor only contains its rows.
- `g_split_staging[1].output` is `sycl::malloc_host` — B50 writes to it, then B580 reads it. This works because both share the same host-pinned memory via the shared `sycl::context`.
- The existing `ggml_sycl_op_mul_mat_vec_q()` takes a `stream` parameter — verify you can pass B50's queue.
- Use Serena to explore the MMVQ function signature and understand all required parameters.
- Graph replay MUST remain disabled when multi-device split is active (existing auto-disable at line 22595).

---

### Task 5: Batched Dispatch Optimization

**Track:** A (after Task 4)
**Depends on:** Task 4
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~80 lines)

**Description:**

Current per-op dispatch calls `stream->wait()` after each MUL_MAT, serializing the pipeline. Batch all partial MUL_MATs: pre-scan the compute graph to identify all MUL_MAT ops, submit all B580 partial MMVQs to B580's queue (in-order queue pipelines them), submit all B50 partial MMVQs to B50's queue concurrently, run all CPU vec_dots sequentially, then wait once and merge all outputs in one pass.

**Acceptance Criteria:**

- [ ] Pre-scan identifies all MUL_MAT ops in the TG compute graph
- [ ] All B580 partial MMVQs submitted to B580 queue without intermediate waits
- [ ] All B50 partial MMVQs submitted to B50 queue without intermediate waits
- [ ] CPU vec_dot runs all work items sequentially (concurrent with both GPUs)
- [ ] Single wait point after all submissions
- [ ] Batch merge: all B50/CPU outputs copied in one pass
- [ ] No correctness regression
- [ ] Performance improvement: eliminate ~128 per-op synchronizations

**Implementation Guide:**

1. **Add batch descriptor structures:**

```cpp
struct split_batch_item {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor *       dst;
    int64_t N_b580, N_b50, N_cpu;
    // Per-device weight pointers (resolved at batch time)
    const void * src0_b580;
    const void * src0_b50;
    const void * src0_cpu;
};

static std::vector<split_batch_item> g_split_batch;
```

2. **Pre-scan in `ggml_backend_sycl_graph_compute_impl()`:**

Before the main node loop, scan for MUL_MAT nodes and build the batch:
```cpp
if (g_split_config.enabled) {
    g_split_batch.clear();
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[1]->ne[1] == 1) {
            // TG MUL_MAT — add to batch
            split_batch_item item = {};
            item.src0 = node->src[0];
            item.src1 = node->src[1];
            item.dst   = node;
            // Resolve row splits and weight pointers
            resolve_split_batch_item(item);
            g_split_batch.push_back(item);
        }
    }
}
```

3. **Submit all partial work per device, then wait once and merge.**

4. **Test: Verify correctness and measure performance improvement**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# Correctness
GGML_SYCL_SPLIT_RATIO="60,32,8" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# Performance
GGML_SYCL_SPLIT_RATIO="60,32,8" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
# Expect: TG128 > 90 tok/s (batched should eliminate per-op sync overhead)
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: batched 3-device dispatch eliminates per-op synchronization"
```

**Notes for implementer:**
- The in-order queue property means B580's queue will pipeline all submitted MMVQs in order — no explicit barriers needed between them.
- B50's queue is also in-order and runs concurrently with B580's queue.
- CPU vec_dot work items run sequentially on the calling thread (they're already fast enough with TBB internally).
- Non-MUL_MAT ops (SILU, NORM, ROPE, etc.) still run on B580 only. The batch approach only affects MUL_MAT.
- The merge pass needs to be careful about dst pointers — each MUL_MAT has a different dst tensor.
- This is the performance-critical optimization. Without it, per-op sync limits throughput to ~40 tok/s.

---

### Task 6: Integration Verification and Performance Sweep

**Track:** — (convergence point)
**Depends on:** Task 4 (minimum), Task 5 (preferred)
**File scope:**
- No code changes (testing only)

**Description:**

End-to-end verification that 3-device inference produces correct output and achieves target performance. Sweep split ratios to find optimum. Verify no single-GPU regression.

**Acceptance Criteria:**

- [ ] Deterministic output matches single-GPU: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`
- [ ] GPU-only baseline unaffected: PP512 >= 1200, TG128 >= 68
- [ ] 3-device TG128 > 90 tok/s with optimal split ratio
- [ ] GPU-only split (no CPU) works: `GGML_SYCL_SPLIT_RATIO="65,35,0"`
- [ ] Legacy `GGML_SYCL_TENSOR_SPLIT=13` still works
- [ ] No crash or hang with any reasonable split ratio
- [ ] Auto-detect (`GGML_SYCL_SPLIT_RATIO=auto`) selects reasonable ratios

**Implementation Guide:**

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# 1. GPU-only baseline (no regression)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expect: PP512 >= 1200, TG128 >= 68

sleep 45  # Thermal cooldown (Arc B580 throttles aggressively)

# 2. Correctness test
GGML_SYCL_SPLIT_RATIO="60,32,8" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

sleep 45

# 3. Performance sweep
for ratio in "65,35,0" "60,32,8" "55,30,15" "50,25,25"; do
  echo "=== SPLIT_RATIO=$ratio ==="
  GGML_SYCL_SPLIT_RATIO="$ratio" ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
    ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
  sleep 45  # Thermal cooldown
done
# Expect: best TG128 > 90 tok/s

# 4. Legacy mode still works
GGML_SYCL_TENSOR_SPLIT=13 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# 5. Auto-detect
GGML_SYCL_SPLIT_RATIO=auto ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
# Expect: reasonable performance, log shows detected ratios
```

**Commit:**
No commit for testing. If fixes are needed, commit them as bugfixes.

**Notes for implementer:**
- Arc B580 throttles EXTREMELY after back-to-back benchmarks (70→2.4 tok/s). 45-second cooldown between runs is critical.
- `ONEAPI_DEVICE_SELECTOR="level_zero:0,1"` exposes both discrete GPUs. Using just `level_zero:0` gives single-GPU only.
- The split ratio "65,35,0" tests dual-GPU without CPU contribution — useful for isolating GPU vs CPU overhead.
- If output is wrong, debug by testing each device individually: "100,0,0", "0,100,0" (won't work for CPU-only).
