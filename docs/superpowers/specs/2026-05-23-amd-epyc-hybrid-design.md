# Hybrid TMA-to-RAM Bridge for ggml-cpu

> **Part 2 of 2**: Pinned system RAM + TMA descriptor setup + PCIe benchmark. Part 1 (CPU optimization) is at `2026-05-23-amd-epyc-cpu-design.md`.

**Goal:** Enable the RTX 5090 to pull model weights directly from system RAM via PCIe Gen5 x16 TMA, bypassing CPU register loading. Validate with raw bandwidth and layer-transfer overlap benchmarks.

**Architecture:** New `ggml_cpu_pinned` buffer type (page-locked RAM via `mmap(MAP_LOCKED)`), CUDA TMA descriptor creation for Blackwell SM100+, and two benchmarks measuring PCIe throughput.

**Tech Stack:** Linux `mmap(MAP_ANONYMOUS|MAP_SHARED|MAP_LOCKED)`, CUDA `cudaMallocAsync`/`cudaMemcpyAsync`, PTX `cp.async.bulk` TMA descriptors, CMake

---

## 1. Pinned CPU Buffer Type

### Problem

Standard `malloc()` returns pageable memory. When the GPU initiates a DMA or TMA transfer from pageable system RAM, the OS must ensure the pages are resident, adding latency and potential page faults. Pinned (page-locked) memory guarantees physical address stability, allowing the GPU's DMA engine to read directly without CPU involvement.

### Design

Add a new buffer type `GGML_BACKEND_CPU_PINNED`. Implemented in `ggml/src/ggml-cpu/pinned.c`.

**Allocation:**
```c
void* ggml_cpu_pinned_alloc(size_t size) {
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (ptr == MAP_FAILED) {
        // Fallback: allocate then mlock (requires RLIMIT_MEMLOCK)
        ptr = malloc(size);
        mlock(ptr, size);
    }
    return ptr;
}
```

**Free:**
```c
void ggml_cpu_pinned_free(void* ptr, size_t size) {
    munmap(ptr, size);
}
```

**Cross-platform:**
- Linux: `mmap(MAP_LOCKED)` — locks pages at allocation time
- Windows: `VirtualAlloc(NULL, size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE)` + `VirtualLock(ptr, size)`
- Fallback: if locking fails (insufficient `RLIMIT_MEMLOCK`, permissions), log warning and use `malloc()`. The CUDA side will still work via pageables DMA (slower, but correct).

**Registration:** Exposed via `ggml_backend_cpu_pinned_init()` returning a `ggml_backend_t` with the pinned buffer operations (alloc, free, zero, set).

### Files
- `ggml/src/ggml-cpu/pinned.c` — CREATE: pinned allocation backend (~200 lines)
- `ggml/src/ggml-cpu/pinned.h` — CREATE: declarations
- `ggml/CMakeLists.txt` — add pinned.c to ggml sources

---

## 2. TMA Descriptor Setup

### Problem

Blackwell's TMA (Tensor Memory Accelerator) pipeline requires a 16-byte TMA descriptor to be loaded into the TMA descriptor cache before any `cp.async.bulk` transfer. The descriptor encodes the source address, shape, stride, data type, and surface type. Currently, ggml-cuda has no TMA descriptor creation code.

### Design

Add `ggml_cuda_tma_descriptor` struct and creation functions to `ggml/src/ggml-cuda/tma.cuh` (new file, alongside the existing WGMMA infrastructure).

**Descriptor struct:** The TMA descriptor is a 16-byte (128-bit) structure. On Blackwell, it's typically stored in constant memory or a dedicated TMA descriptor buffer.

```c
struct ggml_cuda_tma_desc {
    uint64_t d[2];  // 16 bytes total
};

// Create a TMA load descriptor for a 2D tensor (rows x cols) of type.
// base_ptr: pinned system RAM address (must be page-locked, 4K-aligned)
// Returns a 16-byte descriptor to be loaded via the TMA descriptor cache.
ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc(
    const void* base_ptr,
    int64_t rows, int64_t cols,
    enum ggml_type type,
    int64_t row_stride_bytes);  // leading dimension in bytes
```

**Descriptor cache load:** Before any kernel using TMA, the host code loads the descriptor into the GPU's TMA descriptor cache via `cudaLaunchKernel` or inline PTX. For the benchmark (Section 3), a simple `cudaMemcpyAsync` suffices. For future WGMMA kernel integration, the descriptor is loaded via `cp.async.bulk.tensor.2d` PTX.

**Pinned RAM requirement:** TMA requires the source memory to be:
1. Page-locked (pinned) — guaranteed by `ggml_cpu_pinned_alloc()`
2. 4096-byte aligned — `mmap` returns page-aligned memory by default

### Files
- `ggml/src/ggml-cuda/tma.cuh` — CREATE: TMA descriptor types and creation functions (~150 lines)
- `ggml/src/ggml-cuda/CMakeLists.txt` — include tma.cuh in headers

---

## 3. Benchmarks

### 3.1 Raw PCIe Bandwidth

Transfer a large buffer (1-4 GB) from pinned RAM to GPU global memory via `cudaMemcpyAsync`, timed with CUDA events. Warm up with 10 iterations, then average over 50.

```
Benchmark output:
  Pinned RAM -> GPU (cudaMemcpyAsync): XX.X GB/s
  Expected ceiling: ~63 GB/s (PCIe Gen5 x16)
  Expected realistic: ~50 GB/s
```

Measures the raw DMA throughput. Validates that the pinned buffer is correctly accessible from the GPU.

### 3.2 Layer Transfer Overlap

Simulate a partial offload scenario:
1. Transfer one Kimi layer's weights (~2-4 GB for a 600GB model) from pinned RAM to GPU via `cudaMemcpyAsync`
2. Launch a dummy compute kernel (e.g., a GEMM on the previously transferred layer) on the GPU
3. Overlap the transfer of layer N+1 with the compute of layer N
4. Measure effective throughput and compute utilization

```
Benchmark output:
  Layer size: X.X GB
  Transfer time: XX ms (XX GB/s)
  Compute time: XX ms
  Overlap efficiency: XX% (perfect = 100% hidden transfer)
```

Shows how well the PCIe transfer can be hidden behind GPU compute. If compute takes longer than transfer, the system is compute-bound. If transfer dominates, the pinned RAM + TMA is critical.

### Implementation

New executable `llama-pcie-bench` (or integrated into `llama-bench` as a new mode). Uses:
- `ggml_backend_cpu_pinned_init()` for allocation
- CUDA runtime API for transfers and timing
- Command-line args: buffer size, layer count, GPU device

### Files
- `tools/pcie-bench/pcie-bench.cpp` — CREATE: benchmark binary (~300 lines)
- `tools/pcie-bench/CMakeLists.txt` — CREATE: build rules (links ggml + CUDA)

---

## 4. Integration and Testing

### File Change Summary

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/pinned.c` | CREATE: pinned allocation backend (~200 lines) |
| `ggml/src/ggml-cpu/pinned.h` | CREATE: declarations |
| `ggml/src/ggml-cuda/tma.cuh` | CREATE: TMA descriptor types and creation (~150 lines) |
| `tools/pcie-bench/pcie-bench.cpp` | CREATE: PCIe benchmark (~300 lines) |
| `tools/pcie-bench/CMakeLists.txt` | CREATE: build rules |
| `ggml/CMakeLists.txt` | MODIFY: add pinned.c |
| `ggml/src/ggml-cuda/CMakeLists.txt` | MODIFY: include tma.cuh |

### Testing

1. **Pinned allocation correctness:** Allocate 1 GB pinned, write a pattern, read back via CPU. Verify `mincore()` shows pages locked.
2. **GPU accessibility:** `cudaMemcpyAsync` from pinned RAM to GPU, verify data integrity.
3. **Raw bandwidth:** Should achieve >40 GB/s on PCIe Gen5 x16 RTX 5090.
4. **RLIMIT_MEMLOCK handling:** Test with low `ulimit -l` — should fallback gracefully.
5. **Non-CUDA build:** Pinned buffer compiles and works without CUDA (TMA descriptor is CUDA-only).

### Zero Regression

- Pinned buffer is opt-in (`ggml_backend_cpu_pinned_init()`)
- Default CPU backend unchanged (still uses `malloc`)
- TMA code gated by `#ifdef __CUDA_ARCH__` and SM100+
- Benchmark is a separate binary; doesn't affect main build

---

## 5. Follow-up Work (Out of Scope)

- `cp.async.bulk` kernel integration with WGMMA pipeline (Phase 3 of Blackwell refactor)
- Automatic pinned RAM allocation for offloaded layers in the ggml scheduler
- Multi-GPU TMA with NVLink-aware routing
- TMA store (`wgmma.store_gmem`) for GPU→RAM writeback
