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

Both DDR and VTCM reserve trailing guard space so HVX kernels can issue full-width vector loads (`vmem` / `vmemu`) on tail elements without boundary branches:
- Shared DDR buffers pad allocations to 4 KB and append a 4 KB guard page.
- Usable VTCM (`htp_vtcm_usable_size`) reserves `HTP_VTCM_GUARD_SIZE` (128 bytes) at the end of the physical VTCM page.

## Large model handling

Hexagon NPU sessions have a 32-bit virtual address space window of around 3.5GB.
In llama.cpp/GGML, each Hexagon session is mapped to a single GGML backend device (e.g., `HTP0:0`, `HTP0:1`, etc. when using
`GGML_HEXAGON_DEVICES`, or `HTP0`, `HTP1` in legacy mode).

To support running models larger than 3.5GB on a single device, the Hexagon backend dynamically maps and unmaps buffers:
- Buffers are allocated in shared DDR (RPCMEM) and mapped through FastRPC file descriptors. Non-pinned buffers use delayed
  mappings (`FASTRPC_MAP_FD_DELAYED` or `FASTRPC_MAP_FD_DELAYED_EXTENDED`).
- Pinned buffers (such as KV cache and active compute buffers) remain mapped throughout execution.
- Inactive weight buffers are dynamically mapped into the NPU session during batch buffer preparation
  (`prep_op_bufs()` in `htp/main.c`) and unmapped via `htp_iface_munmap()` when no longer needed by the active batch.
- This dynamic sliding window allows a single NPU session to execute models that exceed the 3.5GB window.

Alternatively, users can partition and split the model across multiple virtual sessions or physical NPUs using layer-splitting,
tensor-splitting, or row-splitting modes. For user-facing execution modes and examples, see the
[Snapdragon user guide](README.md#multi-device-execution-modes).

## Tiled Weight Formats

Weight tensors in the Hexagon backend are repacked from flat row-major GGML layouts into 2D tiles in memory.
Tiling groups 2D matrix blocks contiguously:
- Eliminates strided row reads when fetching 2D tiles into VTCM via DMA.
- Matches native HVX vector width (128 bytes) and HMX systolic array dimensions (32x32).
- Repacking happens on the host during model load (`set_tensor()`) or via runtime repacking (`rt-repack`).
Implementation: [`htp-repack.cpp`](../../../ggml/src/ggml-hexagon/htp-repack.cpp).

### Tile Dimensions and Grid Layout

Each tile represents a submatrix of `N_rows = 32` rows and `K_cols` columns:
- Standard types (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, `MXFP4`, and K-quants): `K_cols = 32` (`HTP_TENSOR_TILE_N_COLS = 32`).
- `Q2_0`: `K_cols = 64` (`HTP_TENSOR_TILE_K_Q2_0 = 64`).
- `Q1_0`: `K_cols = 128` (`HTP_TENSOR_TILE_K_Q1_0 = 128`).

For a weight matrix of logical shape `[ne0, ne1]` (where `ne0 = K`, `ne1 = N`):
- `n_k_tiles   = ne0 / K_cols`
- `n_col_tiles = (ne1 + 31) / 32`
- Tile index for column-tile `kt` and row-tile `ct`: `tile_idx = ct * n_k_tiles + kt`
- Byte offset in matrix buffer: `tile_offset = tile_idx * tile_size`

### Block-32 Quantized Formats

Block-32 formats use a tile size of 32 rows x 32 columns (`K_cols = 32`).
Quants are stored in 4-K-grouped (`vrmpy`) byte order across 128-byte HVX vectors:
columns `k = 0..31` are split into 8 groups `g = k / 4` of 4 consecutive columns (`j = k % 4`), stored at `4 * row + j` within each vector.

- **Q4_0** (576 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q4_0`):
  - **Quants** (bytes 0..511, 512 bytes, 4 HVX vectors `p = 0..3`):
    Contains 32 rows x 32 columns of 4-bit quants.
    Vector `p` packs column group `2*p` in the low nibble and group `2*p + 1` in the high nibble:
    `tile[p * 128 + row * 4 + j]` holds `(quant[row][8*p + 4 + j] << 4) | quant[row][8*p + j]` for `row = 0..31`, `j = 0..3`.
  - **Scales** (bytes 512..575, 64 bytes):
    32 FP16 values (`ggml_half`), storing scale `d` for each row `row = 0..31`.

- **Q8_0** (1088 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q8_0`):
  - **Quants** (bytes 0..1023, 1024 bytes, 8 HVX vectors `p = 0..7`):
    Contains 32 rows x 32 columns of signed 8-bit quants (`int8_t`).
    Vector `p` holds column group `p` (`k = 4*p .. 4*p + 3`):
    `tile[p * 128 + row * 4 + j] = quant[row][4*p + j]` for `row = 0..31`, `j = 0..3`.
  - **Scales** (bytes 1024..1087, 64 bytes):
    32 FP16 values (`ggml_half`), storing scale `d` for each row `row = 0..31`.

- **IQ4_NL** (576 bytes per tile, `HTP_TENSOR_TILE_SIZE_IQ4_NL`):
  - **Quants** (bytes 0..511, 512 bytes):
    Same 4-K-grouped nibble layout as `Q4_0`, storing 4-bit non-linear indices (values 0..15).
  - **Scales** (bytes 512..575, 64 bytes):
    32 FP16 values (`ggml_half`), storing scale `d` for each row `row = 0..31`.
  - **Dequantization**:
    In HVX/HMX kernels, each 4-bit index is looked up in the 16-entry non-linear table (`iq4_nl_to_fp16_lut`) to produce an FP16 value, then multiplied by the row scale `d`.

- **MXFP4** (544 bytes per tile, `HTP_TENSOR_TILE_SIZE_MXFP4`):
  - **Quants** (bytes 0..511, 512 bytes):
    Same 4-K-grouped nibble layout as `Q4_0`, storing 4-bit OCP Microscaling FP4 (E2M1) values.
  - **Scales** (bytes 512..543, 32 bytes):
    32 unsigned 8-bit values (`uint8_t`), storing one 8-bit biased exponent (E8M0, bias 127) scale `e` per row `row = 0..31`.
  - **Dequantization**:
    Kernel converts E8M0 bias to IEEE FP16 bias via `vh = clamp(e - 112, 0, 30) << 10` to form a power-of-two FP16 scale.
    Each 4-bit FP4 nibble is unpacked via `mxfp4_to_fp16_lut` and multiplied by this scale.

- **NVFP4** (576 bytes per tile, `HTP_TENSOR_TILE_SIZE_NVFP4`, lossless tiling with same size as `block_nvfp4`):
  - A 64-element block spans 2 consecutive k-tiles; k-tile `h` holds sub-blocks `2h` and `2h+1`.
  - **Quants** (bytes 0..511, 4 HVX vectors): 4-K-grouped nibble layout per 16-element sub-block `u = 0..1` (`tile[(2*u + 0)*128 + row*4 + j]` holds low nibbles of `qs_sub[j]` and `qs_sub[4+j]`; `tile[(2*u + 1)*128 + row*4 + j]` holds high nibbles).
  - **Scales** (bytes 512..575): UE4M3 bytes, `tile[512 + 32*u + row]` for sub-block `u = 0..1` of the k-tile.
  - The per-tensor F32 scale is not part of the tile.

- **Q4_1** (640 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q4_1`):
  - **Quants** (bytes 0..511, 512 bytes):
    Same 4-K-grouped nibble layout as `Q4_0`.
  - **Scales** (bytes 512..639, 128 bytes):
    32 pairs of FP16 values (`ggml_half`): `scale[2 * row + 0] = d` (scale) and `scale[2 * row + 1] = m` (min) for each row `row = 0..31`.

- **Q5_0** (704 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q5_0`):
  - **Quants** (bytes 0..511): same 4-K-grouped nibble layout as `Q4_0`, holding the low 4 bits.
  - **5th-bit plane** (bytes 512..639, 1 HVX vector): 4-K-grouped; `plane[row * 4 + j]` (`j = 0..3`) holds a flag at bit `g` (`0..7`) for `k = 4*g + j`. The flag is set when the 5th bit of the quant is clear.
  - **Scales** (bytes 640..703, 64 bytes): 32 FP16 values, storing scale `d` for each row `row = 0..31`.
  - **Dequantization**: `0xF0` is OR-ed into the nibble where the flag is set, giving the signed value `q - 16` (-16..15), scaled by `d`.

- **Q5_1** (768 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q5_1`, same layout as the Q5_K tile):
  - **Quants** (bytes 0..511) and **scales** (bytes 512..639): same as `Q4_1`, holding the low 4 bits and `(d, m)` pairs.
  - **5th-bit plane** (bytes 640..767, 1 HVX vector): 4-K-grouped; `plane[row * 4 + j]` (`j = 0..3`) holds bit `g` (`0..7`) for `k = 4*g + j`, set when the 5th bit of the quant is set.
  - Uses the `Q5_K` HVX/HMX kernels (Q8_1 activations with block sums).

### Q1_0 Format (1-Bit Quantization)

`Q1_0` packs 128 columns per tile (`K_cols = 128`, `HTP_TENSOR_TILE_K_Q1_0 = 128`):

- **Tile Layout** (576 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q1_0`):
  - **Quants** (bytes 0..511, 512 bytes):
    Contains 32 rows x 128 columns of 1-bit quants (-1 or +1).
    Divided into 4 quarters (`quarter = 0..3`), where each quarter covers 32 columns in K.
    Each quarter occupies 128 bytes (1 HVX vector) in `vrmpy` byte order.
    Within a quarter, columns `kq = 0..31` map into 8 groups `g = kq / 4` (0..7) of 4 consecutive columns:
    `byte_offset = quarter * 128 + 4 * row + (kq % 4)`
    `bit = g` (0..7)
  - **Scales** (bytes 512..575, 64 bytes):
    32 FP16 values (`ggml_half`), storing scale `d` for each row `row = 0..31`.
    One scale per row applies to all 128 columns (all 4 quarters) of the tile.

- **HMX Mapping (1-to-4 Tile Expansion)**:
  - HMX hardware operates strictly on 32x32 FP16 tiles (`HTP_TENSOR_TILE_N_ROWS = 32`, `HTP_TENSOR_TILE_N_COLS = 32`).
  - During kernel execution, HVX dequantization (`dequantize_tiled_weight_to_fp16_task_q1_0`) expands each 32x128 Q1_0 tile (576 bytes) into 4 contiguous 32x32 FP16 HMX tiles (4 x 2048 bytes = 8192 bytes total) in VTCM.
  - HMX systolic array processes the 4 resulting 32x32 FP16 tiles over 4 compute steps along K.

### Q2_0 Format (2-Bit Quantization)

`Q2_0` packs 64 columns per tile (`K_cols = 64`, `HTP_TENSOR_TILE_K_Q2_0 = 64`):

- **Tile Layout** (576 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q2_0`):
  - **Quants** (bytes 0..511, 512 bytes):
    Contains 32 rows x 64 columns of 2-bit quants (codes 0..3 representing signed values -1..2).
    Divided into 16 column groups `group = 0..15` of 4 columns each.
    Stored in `vrmpy` byte order across 4 HVX vectors (vec 0..3):
    Vector `group >> 2` (0..3) holds `group & 3` at bit offset `2 * (group & 3)`.
    Byte `4 * row + byte` holds row `row = 0..31` and column byte `0..3`.
    Vec 0..1 holds the first 32 columns (half 0), vec 2..3 holds the next 32 columns (half 1).
  - **Scales** (bytes 512..575, 64 bytes):
    32 FP16 values (`ggml_half`), storing scale `d` for each row `row = 0..31` (at offset `512 + 2 * row`).
    One scale per row applies to all 64 columns of the tile.

- **HMX Mapping (1-to-2 Tile Expansion)**:
  - HMX hardware operates strictly on 32x32 FP16 tiles (`HTP_TENSOR_TILE_N_ROWS = 32`, `HTP_TENSOR_TILE_N_COLS = 32`).
  - During kernel execution, HVX dequantization (`dequantize_tiled_weight_to_fp16_task_q2_0`) expands each 32x64 Q2_0 tile (576 bytes) into 2 contiguous 32x32 FP16 HMX tiles (2 x 2048 bytes = 4096 bytes total) in VTCM.
  - HMX systolic array processes the 2 resulting 32x32 FP16 tiles over 2 compute steps along K.

- **HVX Vector Dot-Product**:
  - `tiled_vec_dot_q2_0_32x1` and `tiled_vec_dot_q2_0_32x2` unpack 2-bit quants using `unpack_q3_k_low2`, subtract 1, and accumulate against Q8_0 activation tiles using `Q6_Vw_vrmpyacc_VwVbVb`.

### K-Quants Formats

In GGML, K-quants pack 256 elements per super-block.
On Hexagon, each 256-element super-block splits along K into 8 tiles of 32 columns (`K_cols = 32`).

Byte ordering: `Q2_K`, `Q3_K`, and `Q6_K` use `vrmpy` byte order within each 128B HVX vector:
`k = 0..31` is split into 8 groups `g = k/4` of 4 `k` each; byte `4*r + (k%4)` holds row `r`.

- **Q2_K** (336 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q2_K`, lossless tiling with same size as `block_q2_K`):
  - `bytes   0..255` (vec 0..1): unsigned 2-bit quants; vec `m` holds groups `4m..4m+3` at bit offsets 0, 2, 4, 6.
  - `bytes 256..319`: raw `scales[]` bytes (`min << 4 | sc`): `[0..31]` for k 0..15, `[32..63]` for k 16..31.
  - `bytes 320..335`: 8 FP16 in tile `j` (0..7) of super-block: `d` (`j < 4`) or `-dmin` (`j >= 4`) for rows `8*(j%4)..8*(j%4)+7`.

- **IQ2_XXS** (264 bytes per tile, `HTP_TENSOR_TILE_SIZE_IQ2_XXS`, lossless tiling with same size as `block_iq2_xxs`):
  - `bytes   0..255`: `qs[4*kt .. 4*kt+3]` (grid indices, signs and scales of one 32-element group) of row `r` at `8*r`.
  - `bytes 256..263`: 4 FP16 `d` in tile `j` (0..7) of super-block: rows `4*j..4*j+3`.

- **Q3_K** (512 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q3_K`):
  - `bytes   0..255` (vec 0..1): low 2 bits; vec `m` holds groups `4m..4m+3` at bit offsets 0, 2, 4, 6.
  - `bytes 256..383` (vec 2): sign mask, bit `g` set when `hmask` is 0 (quant = `low2 - 4`).
  - `bytes 384..511` (vec 3): FP16 scales `d * (scales[] - 32)`: `fp16[0..31]` for k 0..15, `fp16[32..63]` for k 16..31.

- **Q4_K** (640 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q4_K`, same layout as Q4_1 tile; each 256-k super-block splits into 8 tiles of 32 k):
  - `bytes   0..511` (vec 0..3): 4-bit quants in 4-K-grouped order; `tile[p * 128 + row * 4 + j]` holds `k = 8*p + j` (low nibble) and `k = 8*p + 4 + j` (high nibble).
  - `bytes 512..639` (vec 4): 32x2 FP16 per row: `(d * sc, -dmin * m)`.

- **Q5_K** (768 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q5_K`):
  - `bytes   0..639` (vec 0..4): Q4_K tile (low 4 bits + FP16 scale/min pairs).
  - `bytes 640..767` (vec 5): 5th-bit plane in 4-K-grouped order; `plane[row * 4 + j]` holds bit `g` (`0..7`) for `k = 4*g + j`.

- **Q6_K** (896 bytes per tile, `HTP_TENSOR_TILE_SIZE_Q6_K`):
  - `bytes   0..511` (vec 0..3): low 4 bits; vec `i` holds group `2i` (low nibble) and `2i+1` (high nibble).
  - `bytes 512..767` (vec 4..5): high 2 bits; vec `4+m` holds groups `4m..4m+3` at bit offsets 0, 2, 4, 6.
  - `bytes 768..895` (vec 6): FP16 scales `d * scales[]`: `fp16[0..31]` for k 0..15, `fp16[32..63]` for k 16..31.

## Op and Kernel Development Guidelines

Writing high-performance operators for Hexagon requires following specific guidelines.

### DDR -> DMA -> VTCM Execution Pipeline

- All operators must use the `DDR -> DMA -> VTCM -> compute (HVX/HMX) -> VTCM -> DMA -> DDR` data flow.
- **No direct reads or writes to `tensor->data`**: Do not dereference `tensor->data` pointers directly (scalar or HVX). All tensor transfers must go through DMA, because bypass DMA writes are not L1D/L2 coherent and tensors may reside in 40-bit extended address space.
- **Rare exception (small index/position tensors)**: For small metadata tensors that require scalar indexing (such as `ids`, `pos`, or row indices), use [`htp_tensor_read()`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h) to copy into a 128-byte aligned local DDR buffer (`ddr_spad_base` or stack). It handles extended addresses, arbitrary strides, and `dccleaninva` + `syncht` cache maintenance in one place.
- Use `dma_addr_t` only for DMA base and final addresses. Form a final address by adding a 32-bit byte offset to a
  `dma_addr_t` tensor base address. This permits a 64-bit mapped base address on newer platforms while retaining 32-bit
  relative addressing.
- The DMA queue is a strict FIFO where operations must be pushed and popped in strict order.
- Follow the pipelined multi-buffering sequence properly (typically 2x to 16x buffering) so every push has a corresponding pop:

  1. In the prologue, push initial DDR -> VTCM transfers to prime the pipeline.
  2. In the loop body, wait for buffer N via DMA pop, launch HVX/HMX compute on buffer N, push VTCM -> DDR writeback of result N,
     and push DDR -> VTCM prefetch of buffer N+2.
  3. In the epilogue, pop all remaining in-flight transfers to drain the pipeline.

- Because every push must be matched by a pop, `dma_queue_flush()` is not required when the pipeline sequence is followed
  properly. Flushing is only used in rare exceptions where a batch of operations is pushed without individual pops.
- Use the DMA queue interface from [`dma-queue.h`](../../../ggml/src/ggml-hexagon/htp/dma-queue.h)
  (`dma_queue_push()`, `dma_queue_pop()`, and `dma_queue_flush()`).
- For new simple ops, prefer the generic double-buffered pipeline drivers in
  [`dma-pipeline.h`](../../../ggml/src/ggml-hexagon/htp/dma-pipeline.h) (`dma_pipe_io_run`, `dma_pipe_in_run`,
  `dma_pipe_out_run`), which use dummy pushes, `dma_queue_pop().src`/`.dst` slot recycling, and `dma_queue_flush()`
  at the end of the op. See [`cumsum-ops.c`](../../../ggml/src/ggml-hexagon/htp/cumsum-ops.c),
  [`diag-ops.c`](../../../ggml/src/ggml-hexagon/htp/diag-ops.c), and
  [`pad-ops.c`](../../../ggml/src/ggml-hexagon/htp/pad-ops.c) for reference implementations.

### Avoid Scalar Reads and Writes to VTCM

- Access VTCM data using DMA transfers or HVX/HMX vector instructions rather than scalar reads and writes.

### Avoid Scalar Division in Inner Loops

- Hexagon cores do not have hardware division instructions.
- For recurring divisions across iterations or threads, use `fastdiv` from
  [`hex-fastdiv.h`](../../../ggml/src/ggml-hexagon/htp/hex-fastdiv.h) with precomputed divisors (such as
  `octx->ctx->mdev.count_div` or `octx->n_hvx_div`).
- Do not call `init_fastdiv_values()` for single-use divisions; use standard compiler division (`/`) instead.

### Host-Side Precomputation via `kernel_params`

- Precompute tensor shapes, strides, scale conversions, tiling layouts, and validation checks on the host CPU during graph
  preparation in [`htp-kparams.cpp`](../../../ggml/src/ggml-hexagon/htp-kparams.cpp).
- Pack precomputed parameters into the operator's fixed `kernel_params` structure in `htp_op_node` (such as
  `htp_mm_kernel_params`, `htp_unary_kernel_params`, `htp_fa_kernel_params`, `htp_get_rows_kernel_params`).
- The NPU executes directly using `octx->kernel_params` without redundant runtime metadata extraction or validation.
- **Strict Host-Kernel Alignment**:
  - Verify that parameters calculated by the host CPU are strictly honored by the NPU kernel.
  - Ensure the kernel does not ignore host-computed fields (for example, falling back to `octx->n_hvx` instead of
    using `kparams->n_hvx`, or ignoring precomputed `tasks_per_hvx` and chunk counts).
  - Both human developers and coding agents must audit both sides of the interface: ensure fields populated in `kernel_params`
    in [`htp-kparams.cpp`](../../../ggml/src/ggml-hexagon/htp-kparams.cpp) are actively and consistently utilized by the
    corresponding operator entry point and worker threads in `htp/*-ops.c`.

### Tracing Instrumentation

- All kernels must include trace events for performance profiling and timeline visualization in Perfetto
  ([`htp-trace.h`](../../../ggml/src/ggml-hexagon/htp/htp-trace.h)).
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
  work_queue_run(ctx->hvx_queue, worker_func, &op_ctx, n_hvx);
  ```

- Keep worker functions independent and re-entrant. Worker threads should only operate on their designated chunk of rows or elements.

### Avoid Redundant Defensive NULL Checks

- Do not add defensive NULL checks or assertions for internal framework pointers or required graph operands and outputs.
  Internal pointers include `ctx`, `octx`, local context structs like `*ctx`, `kparams`, and worker callback `data`.
- These pointers are architectural invariants during kernel execution and host-side graph preparation.
- Do not turn an invariant violation into an unsupported operation or missed fusion.
  Checks such as `if (!octx || !octx->ctx)` clutter the code, obscure intent, and hide upstream errors.
- **Distinction**: `octx->src[N]` pointers *can* be NULL by design and must be checked when optional.
  Examples include attention masks, optional bias or weights in fused kernels, and frequency factors.

### Prefer Early Dispatch and Specialized Micro-Kernels

- Avoid peppering micro-kernels, row or tile compute loops, or DMA worker loops with runtime checks for operand types, optional inputs, broadcast modes, or layout variants.
- Resolve execution variants early during operator setup (`op_*` / `execute_op_*`) via function pointers or dispatch tables so each micro-kernel and inner loop handles a single clean path that the compiler can unroll and software-pipeline without extra branch or register pressure.
- Prefer clean standalone specialized functions; use macros only to avoid genuine code duplication across type or mode instantiations.

### `supports_op` Independence from Buffer and Repack State

- `supports_op` (and op-specific checkers such as `supported_mul_mat()`, `supported_get_rows()`) runs during GGML graph scheduling **before** buffers are allocated and before tensor data is populated.
- **Never check buffer usage or repacking state in `supports_op`**:
  - `t->buffer` may be NULL or uninitialized.
  - `t->data` may be NULL.
  - `buffer->usage` is unknown at scheduling time.
  - `is_tiled` or `extra->flags & GGML_HEXAGON_TENSOR_TILED` is not set yet.
- Operator support decisions must depend strictly on static tensor properties: operand types (`t->type`), shapes (`t->ne`), strides (`t->nb`), and operation parameters.
- Repacking (converting weights to tiled layout) is an internal optimization applied during `set_tensor()` or rt-repack. Non-weight tensors or test harness tensors with the same type remain in flat layout.
- Therefore, any supported type (such as BF16, F16, quantized types) must be supported both when repacked into tiles and when stored flat in memory.

### FP16 and Mixed-Precision Operator Support

- New and updated Hexagon compute operators must support `F16` (and mixed `F16`/`F32` where applicable for graph lowering) across activation inputs and outputs, even when `ggml-cpu` only implements `F32`.

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

### Binary Inspection and Spill Analysis

Use [`scripts/snapdragon/ggml-hexagon-inspect.py`](../../../scripts/snapdragon/ggml-hexagon-inspect.py) to audit Hexagon binaries for register
spills, unexpected float promotions, or disassembly:

- Always verify that compute kernels have zero in-loop vector spills (`--spills --strict`) and no float promotions (`--promotions`).
- Avoid excessive loop unrolling (`#pragma unroll`), which increases register pressure and causes spills.

```bash
# Check for vector and scalar register spills
python3 scripts/snapdragon/ggml-hexagon-inspect.py --spills --strict --func "^compute_"

# Check for float promotions
python3 scripts/snapdragon/ggml-hexagon-inspect.py --promotions --func "^compute_"

# Disassemble with annotated loops and spill markers
python3 scripts/snapdragon/ggml-hexagon-inspect.py --disasm compute_same_shape_div_f32

# Resolve crash addresses to function symbols and lines
python3 scripts/snapdragon/ggml-hexagon-inspect.py --addr2line 0x51a30 0x5ba54
```

## Multi-Device Partitioning (mdev)

Multi-device (mdev) mode enables row-level tensor parallel execution across multiple physical NPU cores or virtual NPU
sessions.

### Non-Overlapping Output Ranges

- Because all DDR writes go through bypass DMA, multi-device partitions are **not** subject to 128-byte cache-line
  alignment rules.
- Devices only need non-overlapping destination output ranges and any kernel-specific tile or vector granularity
  (for example, 32 rows for tiled matmul or 128-byte vector chunks for flat 1D element-wise ops).

### Partitioning Helper in `htp-tensor.h`

Common partitioning logic is factored into [`htp_tensor_mdev_partition`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h)
in [`htp-tensor.h`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h):

- Calculates the per-device work range `struct htp_tensor_mdev_range { uint32_t start; uint32_t count; }` given
  `total_units`, `units_per_chunk`, `mdev_idx`, `mdev_count`, and the precomputed `mdev_count_div`.
- Handles chunk distribution across devices, assigns remainder units to the last device, and automatically falls back
  to Device 0 when `units_per_chunk == 0` or `total_units / units_per_chunk < mdev_count`.

### Row-Partitioned Operators

For row-wise operators (such as binary ops in [`binary-ops.c`](../../../ggml/src/ggml-hexagon/htp/binary-ops.c)
and unary ops in [`unary-ops.c`](../../../ggml/src/ggml-hexagon/htp/unary-ops.c)):

```c
const uint32_t total_rows = ne01 * ne02 * ne03;

uint32_t row_start = 0;
uint32_t nrows     = total_rows;

if (octx->ctx->mdev.count > 1) {
    const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
        total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
    row_start = range.start;
    nrows     = range.count;
}

if (nrows == 0) {
    return HTP_STATUS_OK;
}
```

### Element-Partitioned Operators

For flat 1D element-wise operations:
- Partition total linear elements `N = ne0 * ne1 * ne2 * ne3` in HVX vector-sized chunks (`elems_per_line = 128 / elem_size`).
- Requires 1D contiguity via [`htp_tensor_is_contiguous`](../../../ggml/src/ggml-hexagon/htp/htp-tensor.h); otherwise pass
  `0` to trigger Device 0 fallback.

### Single-Device Fallback (Device 0)

- Fallback to Device 0 (`mdev.idx == 0`) happens when `units_per_chunk == 0` or `total_chunks < mdev_count`.
- Device 0 processes the entire tensor `[0, total_units)`.
- Devices `1 ... N-1` receive `count = 0` and return `HTP_STATUS_OK` immediately.

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
  (htp_mdev_group_barrier)                     (htp_mdev_group_barrier)
       |                                            |
  Kernel Execution                             Kernel Execution
  (Output Slice 0, via bypass DMA)             (Output Slice 1, via bypass DMA)
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

- **Writing to fence ([`htp_fence_write`](../../../ggml/src/ggml-hexagon/htp/htp-fence.h#L17))**:
  Stores `seq` and `status`, issues a `syncht` thread synchronization barrier, and flushes/invalidates the line
  using `Q6_dccleaninva_A(fence)`.
- **Reading from peer fence ([`htp_fence_read`](../../../ggml/src/ggml-hexagon/htp/htp-fence.h#L25))**:
  Executes `Q6_dccleaninva_A(fence)` and `syncht` before reading atomic values to ensure fresh data from DDR.

### Deterministic Monotonic Sequence Numbers

- Barrier fences use monotonically increasing sequence numbers:

  ```c
  const uint32_t seq = ++ctx->mdev.fence_seq;
  ```

- Comparing sequence numbers with signed arithmetic `(int32_t)(peer_seq - seq) >= 0` prevents race conditions or
  misaligned barrier arrivals across iterations.
- If any peer reports an error status (`peer_status > HTP_STATUS_OK`), the barrier propagates the error and unblocks immediately.

### Pipeline Completion

- In the kernel, ensure all pushed DMA operations have been popped in strict FIFO order (or flushed via `dma_queue_flush()`) to drain the queue.
- Because all DDR writes use bypass DMA, no per-tensor output cache flush is needed.
- Never signal completion before all DMA transfers are drained.
