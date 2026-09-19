# Hexagon backend developer details

## Backend libraries

The Hexagon backend consists of two parts:

  - `libggml-hexagon`
    This is the regular CPU-side GGML backend library, either shared or statically linked.

  - `libggml-htp-vNN`
    This is the NPU-side (HTP stands for Hexagon Tensor Processor) shared library that contains the Op dispatcher and kernels.
    The correct library is selected automatically at runtime based on the HW version.

Here is an example of the build artifacts:

```
~/src/llama.cpp$ ls -l pkg-adb/llama.cpp/lib/libggml*
pkg-adb/llama.cpp/lib/libggml-base.so
pkg-adb/llama.cpp/lib/libggml-cpu.so
pkg-adb/llama.cpp/lib/libggml-hexagon.so      <<< CPU library
pkg-adb/llama.cpp/lib/libggml-htp-v73.so      <<< HTP op/kernels for Hexagon v73
pkg-adb/llama.cpp/lib/libggml-htp-v75.so
pkg-adb/llama.cpp/lib/libggml-htp-v79.so
pkg-adb/llama.cpp/lib/libggml-htp-v81.so
```

## Memory buffers

The Hexagon NPU backend takes advantage of Snapdragon unified memory where all DDR buffers are accessible by CPU, GPU, and NPU.
The NPU has dedicated tightly-coupled memory called VTCM (Vector Tightly-Coupled Memory). VTCM is used for intermediate data (such as
dynamically quantized activations) and streaming buffers (chunks of weight and activation tensors fetched via DMA).

## Large model handling

Hexagon NPU sessions have a 32-bit virtual address space window of around 3.5GB.
In llama.cpp/GGML, each Hexagon session is mapped to a single GGML backend device (e.g., `HTP0:0`, `HTP0:1`, etc. when using
`GGML_HEXAGON_DEVICES`, or `HTP0`, `HTP1` in legacy mode).

To support running models larger than 3.5GB on a single device, the Hexagon backend dynamically maps and unmaps buffers:
- Buffers are allocated in shared DDR (RPCMEM) via file descriptors (`fastrpc_mmap` using `FASTRPC_MAP_FD_DELAYED`).
- Pinned buffers (such as KV cache and active compute buffers) remain mapped throughout execution.
- Inactive weight buffers are dynamically mapped into the NPU session via `HAP_mmap()` during batch buffer preparation
  (`prep_op_bufs()` in `htp/main.c`) and unmapped via `htp_iface_munmap()` when no longer needed by the active batch.
- This dynamic sliding window allows a single NPU session to execute models that exceed the 3.5GB window.

Alternatively, users can partition and split the model across multiple virtual sessions or physical NPUs using layer-splitting,
tensor-splitting, or row-splitting modes. For user-facing execution modes and examples, see the
[Snapdragon user guide](README.md#multi-device-execution-modes).

## Op and Kernel Development Guidelines

Writing high-performance operators for Hexagon requires following specific guidelines.

### DDR -> DMA -> VTCM Execution Pipeline

- Strongly prefer the `DDR -> DMA -> VTCM -> compute (HVX/HMX) -> VTCM -> DMA -> DDR` data flow.
- Direct HVX reads/writes from/to DDR are less efficient and should only be used as a fallback.
- The DMA queue is a strict FIFO where operations must be pushed and popped in strict order.
- Follow the pipelined multi-buffering sequence properly (typically 2x to 16x buffering) so every push has a corresponding pop:

  1. In the prologue, push initial DDR -> VTCM transfers to prime the pipeline.
  2. In the loop body, wait for buffer N via DMA pop, launch HVX/HMX compute on buffer N, push VTCM -> DDR writeback of result N,
     and push DDR -> VTCM prefetch of buffer N+2.
  3. In the epilogue, pop all remaining in-flight transfers to drain the pipeline.

- Because every push must be matched by a pop, `dma_queue_flush()` is not required when the pipeline sequence is followed
  properly. Flushing is only used in rare exceptions where a batch of operations is pushed without individual pops.
- Use the DMA queue interface from [`dma-queue.h`](../../../ggml/src/ggml-hexagon/htp/dma-queue.h)
  (`dma_queue_push_ddr_to_vtcm`, `dma_queue_pop`, `dma_queue_push_vtcm_to_ddr`).
  See [`cumsum-ops.c`](../../../ggml/src/ggml-hexagon/htp/cumsum-ops.c) and
  [`act-ops.c`](../../../ggml/src/ggml-hexagon/htp/act-ops.c) for reference implementations.

### Avoid Scalar Reads and Writes to VTCM

- Access VTCM data using DMA transfers or HVX/HMX vector instructions rather than scalar reads and writes.

### Avoid Scalar Division in Inner Loops

- Hexagon cores do not have hardware division instructions.
- For recurring divisions across iterations or threads, use `fastdiv` from
  [`hex-fastdiv.h`](../../../ggml/src/ggml-hexagon/htp/hex-fastdiv.h) with precomputed divisors (such as
  `octx->ctx->mdev.count_div` or `octx->n_threads_div`).
- Do not call `init_fastdiv_values()` for single-use divisions; use standard compiler division (`/`) instead.

### Host-Side Precomputation via `kernel_params`

- Precompute tensor shapes, strides, scale conversions, tiling layouts, and validation checks on the host CPU during graph
  preparation in [`ggml-hexagon.cpp`](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp).
- Pack precomputed parameters into the operator's fixed `kernel_params` structure in `htp_op_node` (such as
  `htp_mm_kernel_params`, `htp_unary_kernel_params`, `htp_fa_kernel_params`, `htp_get_rows_kernel_params`).
- The NPU executes directly using `octx->kernel_params` without redundant runtime metadata extraction or validation.
- **Strict Host-Kernel Alignment**:
  - Verify that parameters calculated by the host CPU are strictly honored by the NPU kernel.
  - Ensure the kernel does not ignore host-computed fields (for example, falling back to `octx->n_threads` instead of
    using `kparams->n_threads`, or ignoring precomputed `tasks_per_thread` and chunk counts).
  - Both human developers and coding agents must audit both sides of the interface: ensure fields populated in `kernel_params`
    in [`ggml-hexagon.cpp`](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp) are actively and consistently utilized by the
    corresponding operator entry point and worker threads in `htp/*-ops.c`.

### Tracing Instrumentation

- All kernels must include trace events for performance profiling and timeline visualization in Perfetto
  ([`hex-profile.h`](../../../ggml/src/ggml-hexagon/htp/hex-profile.h)).
- Surround compute sections with `htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) info)` and
  `htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) info)`.
- Use specific event types for major phases:
  - `HTP_TRACE_EVT_HVX_COMP`: Vector compute execution.
  - `HTP_TRACE_EVT_DMA`: DMA transfer wait or poll cycles.
  - `HTP_TRACE_EVT_FENCE`: Multi-device fence barrier synchronization.
  - `HTP_TRACE_EVT_L2FLUSH`: L2 cache cleaning operations.
- Pass meaningful progress metrics (such as row index, chunk index, or token index) in the 16-bit `info` parameter.

### Work Queue and Threading

- Distribute parallel work across NPU worker threads using the thread pool work queue:

  ```c
  work_queue_run(ctx->work_queue, worker_func, &op_ctx, n_threads);
  ```

- Keep worker functions independent and re-entrant. Worker threads should only operate on their designated chunk of rows or elements.

### Avoid Redundant Defensive NULL Checks

- Do not add defensive NULL checks or assertions for internal framework pointers or required graph operands and outputs.
  Internal pointers include `ctx`, `octx`, local context structs like `*ctx`, `kparams`, and worker callback `data`.
- These pointers are architectural invariants during kernel execution and host-side graph preparation.
  Graph compute receives allocated nodes with valid required `node->src[N]` and `node->data` pointers.
- Do not turn an invariant violation into an unsupported operation or missed fusion.
  Checks such as `if (!octx || !octx->ctx)` clutter the code, obscure intent, and hide upstream errors.
- **Distinction**: `octx->src[N]` pointers *can* be NULL by design and must be checked when optional.
  Examples include attention masks, optional bias or weights in fused kernels, and frequency factors.

### Multiline Macro Formatting

- Keep trailing backslashes in multiline `#define` macros cleanly aligned to a consistent column.
- Avoid trailing whitespace after macro backslashes.
- Use [`scripts/snapdragon/ggml-hexagon-align-macros.py`](../../../scripts/snapdragon/ggml-hexagon-align-macros.py) to inspect, diff,
  or automatically align macro definitions across Hexagon kernel sources:

  ```bash
  # Check for misaligned macros
  python3 scripts/snapdragon/ggml-hexagon-align-macros.py ggml/src/ggml-hexagon/htp/

  # Fix misaligned macros in-place
  python3 scripts/snapdragon/ggml-hexagon-align-macros.py --fix ggml/src/ggml-hexagon/htp/
  ```

## Multi-Device Partitioning (mdev)

Multi-device (mdev) mode enables row-level tensor parallel execution across multiple physical NPU cores or virtual NPU
sessions.

### 128-Byte Cache Line Alignment

- Shared tensor buffers reside in DDR (RPCMEM) with a 128-byte cache line granularity
  (`HEX_L2_LINE_SIZE` = 128 bytes, `HTP_TENSOR_MDEV_LINE_SIZE`).
- **Rule**: Multi-device work partitions must align destination write regions to 128-byte cache line boundaries so distinct
  devices never share or overwrite the same cache line.

### Partitioning Helpers in `htp-tensor.h`

Common partitioning logic is factored into reusable inline helpers in
[`htp-tensor.h`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h):

1. [`htp_tensor_mdev_rows_per_chunk`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h#L67):
   Determines the minimum number of rows per chunk so that the chunk byte size is a multiple of 128 bytes:

   ```
   rows_per_chunk = 128 / hex_gcd_u32(row_size, 128)
   ```

   If row stride `nb[1]` is already a multiple of 128 bytes, `rows_per_chunk = 1`.
   Returns `false` if the tensor cannot be safely row-partitioned (such as unaligned base pointer, permuted layout,
   or non-128-byte aligned outer strides).

2. [`htp_tensor_mdev_partition`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h#L94):
   Calculates the per-device work range `struct htp_tensor_mdev_range { uint32_t start; uint32_t count; }` given
   `total_units`, `units_per_chunk`, `mdev_idx`, `mdev_count`, and the precomputed `mdev_count_div`.
   Handles chunk distribution across devices, assigns remainder units to the last device, and automatically triggers
   single-device fallback when partitioning is unsafe.

### Row-Partitioned Operators

For row-wise operators
(such as activations in [`act-ops.c`](../../../ggml/src/ggml-hexagon/htp/act-ops.c),
binary ops in [`binary-ops.c`](../../../ggml/src/ggml-hexagon/htp/binary-ops.c),
unary ops in [`unary-ops.c`](../../../ggml/src/ggml-hexagon/htp/unary-ops.c), and
sameshape copies in [`cpy-ops.c`](../../../ggml/src/ggml-hexagon/htp/cpy-ops.c)):

```c
const uint32_t total_rows   = ne01 * ne02 * ne03;
const size_t   dst_row_size = dst->ne[0] * elem_size;

uint32_t row_start = 0;
uint32_t nrows     = total_rows;

if (octx->ctx->mdev.count > 1) {
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
    const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
        total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
    row_start = range.start;
    nrows     = range.count;
}

if (nrows == 0) {
    return HTP_STATUS_OK;
}
```

### Element-Partitioned Operators

For flat element-wise operations (such as reshape copies in
[`cpy-ops.c`](../../../ggml/src/ggml-hexagon/htp/cpy-ops.c)):
- Partition total linear elements N = ne0 * ne1 * ne2 * ne3 in 128-byte cache line chunks (`elems_per_line = (elem_size == 4) ? 32 : 64`).
- Requires strict 1D contiguity:
  [`htp_tensor_is_contiguous(dst, elem_size)`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h#L28)
  and 128-byte aligned destination pointer
  [`htp_tensor_mdev_data_aligned(dst)`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h#L47).
- If contiguous and aligned, pass `elems_per_line` to
  [`htp_tensor_mdev_partition`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h#L94);
  otherwise pass 0 to trigger Device 0 fallback.

### Single-Device Fallback (Device 0)

- Fallback to Device 0 (`mdev.idx == 0`) when partitioning would cause cache line tearing or when work cannot be evenly distributed.
- Triggers:
  1. Destination tensor cannot be safely partitioned (`rows_per_chunk == 0` or non-contiguous/unaligned buffer).
  2. Total aligned chunks < `mdev_count`.
- Device 0 processes the entire tensor `[0, total_units)`.
- Devices 1 ... N-1 receive `count = 0` and return `HTP_STATUS_OK` immediately.

### Flatten Outer Dimensions Globally

- **Never partition solely on `ne01` (dimension 1).**
- Partitioning only on `ne01` repeats the device boundary across every 2D slice (`ne02`, `ne03`). If each 2D slice is small,
  false sharing occurs repeatedly throughout the tensor.
- Always flatten outer dimensions globally: `total_rows = ne01 * ne02 * ne03` and partition once across the combined row space.

### Stateless Starting Coordinates

- Do not use incremental state variables across slices that assume the thread or device starts at index 0.
- Precompute starting multidimensional coordinates at `r = row_start` (or `e = elem_start`) once using `fastdiv`.
- In inner loops, step base pointers directly (`ptr += stride`) or reset/wrap coordinates explicitly (`if (++i01 == ne01) { ... }`).

### Clean Range Encapsulation

- Initialize single-device default ranges at declaration:

  ```c
  uint32_t row_start = 0;
  uint32_t nrows     = total_rows;
  ```

- Encapsulate all multi-device logic inside `if (octx->ctx->mdev.count > 1)`. If the block is omitted or compiled out,
  the operator runs standard single-device execution untouched.
- Do not propagate `mdev_` prefixes to worker functions or context structs. Worker threads are device-agnostic and
  should only receive standard range parameters (`ctx.row_start`, `ctx.nrows`).
- In worker threads, calculate row intervals using standard arithmetic:

  ```c
  const uint32_t ir0 = ctx->row_start + dr * ith;
  const uint32_t ir1 = MIN(ir0 + dr, ctx->row_start + ctx->nrows);
  ```

  In single-device mode (`row_start == 0`), this naturally simplifies to `dr * ith` and `MIN(ir0 + dr, ctx->nrows)` with zero overhead.

## Multi-Device Synchronization

Multi-device execution synchronizes worker sessions across devices using explicit barriers and tensor cache flushing.

### Synchronization Fence Protocol

Multi-device execution synchronizes worker sessions through atomic fence slots and barriers defined in
[`htp-fence.h`](../../../ggml/src/ggml-hexagon/htp/htp-fence.h):

```
[NPU Session 0]                              [NPU Session 1]
       |                                            |
  (Input Prep)                                 (Input Prep)
       |                                            |
  Pre-Op Barrier ----------------------------- Pre-Op Barrier
  (mdev_sync_fence)                            (mdev_sync_fence)
       |                                            |
  Kernel Execution                             Kernel Execution
  (Output Slice 0)                             (Output Slice 1)
       |                                            |
  Tensor Cache Flush                           Tensor Cache Flush
  (htp_tensor_flush_all)                       (htp_tensor_flush_all)
       |                                            |
  Post-Op/Batch Barrier ---------------------- Post-Op/Batch Barrier
  (htp_mdev_group_barrier)                     (htp_mdev_group_barrier)
       |                                            |
  Return Response to Host                      Return Response to Host
```

### Atomic Fence Slots and Cache Invalidation

- Fence synchronization operates on dedicated RPCMEM shared memory mapped across all participating sessions (`ctx->mdev.fence_base`).
- Each device owns a dedicated 128-byte cache-line aligned fence slot:

  ```c
  atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
  ```

- **Writing to fence ([`htp_fence_write`](../../../ggml/src/ggml-hexagon/htp/htp-fence.h#L18))**:
  Stores `seq` and `status`, issues a `syncht` thread synchronization barrier, and flushes/invalidates the line
  using `Q6_dccleaninva_A(fence)`.
- **Reading from peer fence ([`htp_fence_read`](../../../ggml/src/ggml-hexagon/htp/htp-fence.h#L26))**:
  Executes `Q6_dccleaninva_A(fence)` and `syncht` before reading atomic values to ensure fresh data from DDR.

### Deterministic Monotonic Sequence Numbers

- Barrier fences use monotonically increasing sequence numbers:

  ```c
  const uint32_t seq = ++ctx->mdev.fence_seq;
  ```

- Comparing sequence numbers with signed arithmetic `(int32_t)(peer_seq - seq) >= 0` prevents race conditions or
  misaligned barrier arrivals across iterations.
- If any peer reports an error status (`peer_status > HTP_STATUS_OK`), the barrier propagates the error and unblocks immediately.

### Tensor Cache Flush and Pipeline Completion

- In the kernel, ensure all pushed DMA operations have been popped in strict FIFO order to drain the queue.
- Use [`htp_tensor_flush_all()`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h) to flush specific dirty tensors back to DDR:
  - [`htp_tensor_flush_all()`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h) flushes only modified tensor address ranges,
    ensuring peer devices and the host CPU observe consistent data in DDR.
- Never signal completion before all DMA transfers are drained and dirty tensor flushes have completed.

