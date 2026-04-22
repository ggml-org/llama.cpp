# AllReduce Provider Abstraction — Working Notes

## Context

Tensor-parallel mode (`LLAMA_SPLIT_MODE_TENSOR = 3`) splits attention and FFN weight
matrices across N GPUs. Each GPU computes a partial result; an AllReduce sums them
before the next layer begins.

## Where the Reduction Happens

```
src/llama-context.cpp          — validates SPLIT_MODE_TENSOR requirements (FlashAttn required, no KV quant)
src/llama-model.cpp            — llama_meta_device_get_split_state(): assigns split axis per tensor
                                  attn_q/k/v, ffn_up/gate → PARTIAL (needs AllReduce)
                                  output → MIRRORED (no AllReduce)
ggml/src/ggml-backend-meta.cpp — ggml_backend_meta_graph_compute(): drives the subgraph loop
                                  after each PARTIAL subgraph: calls comm_allreduce(), or
                                  falls back to allreduce_fallback() (CPU-based)
ggml/src/ggml-cuda/ggml-cuda.cu — the CUDA-side implementations (NCCL + future internal)
```

### Subgraph Execution Loop (ggml-backend-meta.cpp ~line 2023)

```
for each subgraph i:
    compute subgraph on each GPU in parallel
    if i < last_subgraph:
        if comm_ctx set:
            try comm_allreduce(comm_ctx, last_nodes_per_gpu[])
        if allreduce failed (or no comm_ctx):
            allreduce_fallback(i)   ← copies to CPU, reduces, copies back
```

## Data Structures

| Struct | File | Purpose |
|--------|------|---------|
| `ggml_backend_cuda_comm_context` | `ggml-cuda.cu` | holds provider enum + NCCL comms (or future internal state) |
| `ggml_backend_meta_context` | `ggml-backend-meta.cpp` | holds `comm_ctx` (opaque) + `comm_allreduce` fn ptr |
| `ggml_cuda_device_info` | `common.cuh` | per-device CC, VRAM, default split ratios |

## Provider Abstraction Added

### New file: `ggml/src/ggml-cuda/comm.cuh`

Defines `enum ggml_cuda_allreduce_provider`:
- `GGML_CUDA_ALLREDUCE_NCCL` — NCCL/RCCL (default when compiled in)
- `GGML_CUDA_ALLREDUCE_INTERNAL` — internal host/CUDA staged reduction (stub for now)

### Changes to `ggml/src/ggml-cuda/ggml-cuda.cu`

- `ggml_backend_cuda_comm_context` now always exists; holds `provider` + conditionally `comms`.
- `ggml_cuda_select_allreduce_provider()` — new selection function (see below).
- `ggml_backend_cuda_comm_init()` — constructs context, selects provider, inits NCCL comms or internal state.
- `ggml_backend_cuda_comm_allreduce_tensor()` — dispatches to `_nccl` or `_internal` helper.
- `ggml_backend_cuda_comm_allreduce_nccl()` — extracted from old monolithic function; logic unchanged.
- `ggml_backend_cuda_comm_allreduce_internal()` — stub returning `false` (triggers meta fallback).

The public interface (`ggml_backend_comm_init` / `_free` / `_allreduce_tensor` proc addresses) is unchanged.

## Provider Selection Logic (`ggml_cuda_select_allreduce_provider`)

Priority order:
1. `GGML_CUDA_ALLREDUCE=nccl` env var — force NCCL (warn if not compiled in).
2. `GGML_CUDA_ALLREDUCE=internal` env var — force internal.
3. NCCL when `GGML_USE_NCCL` is defined at compile time.
4. INTERNAL otherwise (with a warning on NVIDIA non-HIP/MUSA builds).

Future: inspect hardware topology before choosing the default:

```cpp
// Check if all device pairs have direct NVLink:
int native_atomic;
cudaDeviceGetP2PAttribute(&native_atomic,
    cudaDevP2PAttrNativeAtomicSupported, dev_i, dev_j);
// If any pair lacks NVLink, internal may win for small tensors on PCIe.
```

## Files Changed

```
ggml/src/ggml-cuda/comm.cuh         NEW  — provider enum
ggml/src/ggml-cuda/ggml-cuda.cu     MOD  — provider selection, dispatch, NCCL helper extracted, internal stub
```

## Files NOT Changed (intentionally)

- `ggml/src/ggml-backend-meta.cpp` — no changes needed; uses opaque `comm_ctx` + fn ptr already.
- `ggml/include/ggml-backend.h` — public `comm_*` typedef signatures unchanged.
- `include/llama.h` — `llama_split_mode` enum unchanged.

---

## Prototype Analysis: `nccl_injector_prototype/`

### What the prototype is

A Windows DLL injected via Microsoft Detours that intercepts NCCL calls and reroutes
AllReduce to a faster internal kernel for the 2-GPU float32 case. We are NOT using the
injection/Detours machinery — we're implementing directly inside llama.cpp.

### Single-Phase Kernel (what we're using)

The prototype has two strategies. We only want the **single-phase merged kernel**
(`allreduce_f32_kernel` in `src/kernels.cu`). It merges D2H copy + cross-GPU
synchronization + reduction into one kernel launch per GPU.

**Execution: 1 block × 256 threads per GPU.**

```
Phase A (all 256 threads): vectorized D2H copy, sendbuf → host_mine
    - float4 loads (16 bytes/thread/iteration) for the bulk
    - scalar tail for remainder if count % 4 != 0
    __threadfence_system() + __syncthreads()   ← make D2H visible system-wide

Phase B (thread 0 only): signal + spin
    signal_publish(arrival_mine, 1)            ← volatile write + __threadfence_system()
    while signal_observe(arrival_other) == 0:  ← volatile read, __nanosleep(100) between polls
        (optional: log spin count to debug buf every 4096 iters)
    __syncthreads()                            ← broadcast "both D2H done" to all threads
    __threadfence_system()                     ← acquire peer's host_other writes

Phase C (all 256 threads): reduce
    recvbuf[i] = sendbuf[i] + host_other[i]   ← float4 vectorized
```

**Why it's fast:** the D2H copy and the cross-GPU spin overlap naturally — GPU-0 starts
spinning while GPU-1's 256 threads are still copying their data. No extra kernel launches
or host round-trips.

### Signal Mechanism

Three options exist via `SIGNAL_MECHANISM` macro; default (and recommended) is 1:

```cuda
// Publish: volatile write + system fence
*(volatile int*)p = value;
__threadfence_system();

// Observe: volatile read (no fence needed — __threadfence_system() after syncthreads covers it)
return *(const volatile int*)p;
```

One int per GPU. Values: 0 = not arrived, 1 = arrived. Reset to 0 before each call.
Single writer per slot (owning GPU), single reader (peer GPU) — no atomics needed.

### Host-Side Setup (what to port, minus the NCCL hooks)

**Per-pipeline state to allocate at `comm_init` time:**

```
host_buf[N]      float* cudaMallocHost, one per GPU, >= max_tensor_bytes
arrival[POOL×N]  int*   cudaMallocHost, ring buffer, one int slot per GPU per in-flight call
stream[N]        cudaStream_t cudaStreamCreateWithFlags(cudaStreamNonBlocking)
ev_pool[N][POOL] cudaEvent_t  cudaEventCreateWithFlags(cudaEventDisableTiming)
                   × 2 events per slot (app = "wait for upstream work", ker = "kernel done")
debug[N×4]       int*   cudaMallocHost, optional, 4 ints per GPU for spin diagnostics
```

**Pool size:** 128 slots in the prototype. Events + arrival slots wrap together; must
sync on `ev_pool[r][slot].ker` before reusing arrival slot (slot ownership check).

**Kernel dispatch sequence:**

```cpp
// For each GPU r in parallel:
cudaEventRecord(ev[r].app, upstream_stream[r]);     // capture upstream work
cudaStreamWaitEvent(internal_stream[r], ev[r].app); // internal stream waits for it
launch_allreduce_kernel(..., internal_stream[r]);    // launch merged kernel
cudaEventRecord(ev[r].ker, internal_stream[r]);     // record kernel completion
cudaStreamWaitEvent(upstream_stream[r], ev[r].ker); // upstream waits for kernel
```

This inserts the allreduce into the existing CUDA streams without blocking the host.

**Warmup:** 64 iterations with 32 KB payloads at `comm_init` time. Amortizes
driver overhead and encourages GPU clock boost before real inference begins.

**Watchdog (optional):** poll arrival + debug values from host every ~20 ms to detect
deadlocks without killing the process.

### What to Discard (Detours/Injection Overhead)

| File | Reason to skip |
|------|---------------|
| `src/dllmain.cpp` | DLL entry point, Detours attach/detach |
| `src/launcher.cpp` | Standalone DLL injector executable |
| `src/nccl_types.h` | NCCL function pointer typedefs (not needed when calling directly) |
| `src/hooks.cpp` (partially) | NCCL function wrapping, PendingOp queue, GroupStart/End logic |
| `src/hooks.h` | Hook declarations |

**Keep from `hooks.cpp`:**
- `AllReducePipeline` struct (minus NCCL-specific fields)
- `init_ar_pipeline()` logic
- `execute_all_reduce_kernel()` dispatch logic (adapted for our stream model)

**Keep from `kernels.cu`:**
- `allreduce_f32_kernel` exactly as-is (can rename)
- `signal_publish` / `signal_observe` device functions
- `launch_allreduce_f32` wrapper (adapt to our context)

**Discard from `kernels.cu`:**
- `allreduce_d2h_f32_kernel` — phase 1 of two-phase approach
- `allreduce_reduce_f32_kernel` — phase 2 of two-phase approach

### Current Limitations & Extension Plan

**Current prototype only handles:**
- Exactly 2 GPUs
- `float32` data type
- Tensors ≤ 256 KB (64K floats, `AR_KERNEL_THRESHOLD`)
- Sum reduction only

---

## Extension Plan for the Internal Implementation

### Data Types Beyond float32

The prototype's kernel is float32 only. In llama.cpp the allreduce tensors are always
FP32 (the NCCL path already converts larger tensors to BF16 before sending and back
after). We should follow the same pattern:

**Strategy A — FP32 kernel only (simplest, sufficient for most cases):**
- Tensors ≤ threshold: run internal kernel as FP32 directly (matches prototype)
- Tensors > threshold: convert F32→BF16 on GPU, run BF16 kernel, convert back
  - Halves PCIe/pinned-host bandwidth for large tensors
  - BF16 kernel is identical structure but with `__nv_bfloat16` / `__nv_bfloat162`

**Strategy B — templated kernel:**

```cuda
template<typename T, typename AccT = float>
__global__ void allreduce_kernel(
    const T* sendbuf, T* recvbuf,
    AccT* host_mine, const AccT* host_other,
    int count, int* arrival_mine, int* arrival_other, ...)
{
    // D2H: convert T → AccT on the fly (if T != AccT), store AccT to host_mine
    // Reduce: read AccT from sendbuf (via on-the-fly upcast) + host_other, write T to recvbuf
}
```

Instantiate for:
- `<float, float>` — FP32 direct (fast, bulk of tensors)
- `<nv_bfloat16, float>` — BF16 tensors, accumulate as FP32 in host_mine
- `<half, float>` — FP16 tensors, accumulate as FP32

The host_mine staging buffer always stores the accumulation type (float), so size is
always `count * sizeof(float)` regardless of tensor type. Simpler than varying buffer types.

### Tensor Size Beyond 256 KB

The prototype bails to CPU sync for large tensors. Options:

**Option 1 — Multi-block kernel (recommended):**
Launch `ceil(count / BLOCK_ELEMENTS)` blocks instead of 1. Each block handles its own
arrival signaling independently (need one arrival int pair per block, or use a shared
atomic). This allows pipelining — later blocks can start D2H while earlier blocks
have already signaled.

**Option 2 — Chunked sequential:**
Call the single-block kernel in a loop, each call covering `CHUNK_SIZE` elements.
Simple but adds kernel launch overhead.

**Option 3 — Keep threshold, fall back to NCCL/CPU for large:**
The NCCL path already handles large tensors well (BF16 compressed). Use internal
only for tensors under a tuned threshold where it beats NCCL. This is probably the
right first step — just match or beat NCCL in the size range where NCCL has latency
overhead.

### More Than 2 GPUs

The prototype is hardcoded 2-GPU. The single-phase approach generalizes to N GPUs:

**For N=3 or N=4 (small N), tree or ring approach:**

**Ring AllReduce (reduce-scatter + all-gather):**
1. Reduce-scatter: each GPU sends to next, keeps accumulated result for its chunk
2. All-gather: each GPU sends its final chunk to all others

For N=2 the ring degenerates to the simple pairwise protocol already in the prototype.
The arrival mechanism needs one slot per `(gpu, neighbor)` pair.

**Alternative for small N: star topology** (one GPU is root):
1. All non-root GPUs send to root's host_buf in parallel
2. Root reduces all contributions
3. Root broadcasts to all non-root

Simpler to implement than ring but root becomes bottleneck for N > 2.

For the initial implementation: focus on N=2 (covers the most common dual-GPU case),
then extend to N=4 for 4×GPU servers.

### Size Threshold Tuning

The prototype uses 256 KB for PCIe 4.0 x16. Our threshold should be determined by
benchmarking; likely different on PCIe 5.0 and definitely different on NVLink.
Expose as `GGML_CUDA_ALLREDUCE_INTERNAL_THRESHOLD` env var (elements, default 65536)
so users can tune without recompiling.

---

## Open Questions

1. What are the actual tensor shapes/sizes in the allreduce calls during inference?
   Need a trace to know what the P50/P95 sizes are.
2. Target GPU topology? NVLink or PCIe? Determines whether internal can beat NCCL.
3. Is BF16 staging acceptable precision-wise, or is FP32 end-to-end required?
4. How many GPUs max? Design differs significantly between N=2 and N≥8.
5. Should we support the watchdog/spin limit for hang detection in production?
