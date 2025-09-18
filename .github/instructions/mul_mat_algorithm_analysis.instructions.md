# GGML `mul_mat` Kernel Algorithm & Logic Analysis

This document analyzes the `ggml_compute_forward_mul_mat` (and helper `ggml_compute_forward_mul_mat_one_chunk`) implementation in `ggml-cpu.c`, focusing on:

- End-to-end execution flow
- How quantized types are handled
- Workspace (`wdata`) usage and layout
- Scheduling & chunking strategy
- The role and vectorization of `vec_dot`
- Differences in the `mul_mat_id` variant
- Performance observations and potential optimizations

---

## 1. Key Symbols and Roles

| Symbol | Meaning |
|--------|---------|
| `src0` | Left matrix (typically model weights; often quantized) |
| `src1` | Right matrix (activations / input; often `F32` initially) |
| `dst`  | Output (always float rows; layout constraints asserted) |
| `vec_dot_type` | Internal element/storage type used by the dot kernel for `src0`'s quantization family |
| `from_float` | Function pointer to quantize/pack a float row into `vec_dot_type` |
| `params->wdata` | Shared per-op workspace buffer (temporary packed `src1` if needed) |
| `nbXY` | Strides (bytes) per dimension (X = tensor index, Y = dimension idx) |
| `neXY` | Extents (elements) per dimension |
| `vec_dot()` | Low-level SIMD kernel performing (de)quantized dot products |
| `vec_dot_num_rows` | 1 or 2: some kernels process two rows simultaneously (e.g. ARM int8 MMLA) |

---

## 2. High-Level Flow

```
Input:
    src0  (quantized or plain)
    src1  (usually F32)
Output:
    dst   (F32)

Phase A: Preparation
    1. Determine vec_dot_type from src0->type (type traits)
    2. If src1->type != vec_dot_type:
           Quantize/pack each src1 row into params->wdata (parallel by threads)
       Else:
           Directly use src1 memory (no conversion)

Phase B: Scheduling & Chunking
    3. Compute logical work dimensions:
           nr0 = ne0
           nr1 = ne1 * ne2 * ne3
    4. Choose chunk_size (default 16; 64 if degenerate)
    5. Derive nchunk0, nchunk1 (ceil divisions)
    6. Possibly re-map chunking if:
           - Total chunks < 4 * threads
           - OR NUMA active (prefer per-thread slabs)
    7. Initialize atomic chunk counter

Phase C: Per-Tile Kernel
    8. Threads atomically claim chunks → (ir0, ir1) ranges
    9. For each chunk, call ggml_compute_forward_mul_mat_one_chunk:
           - 16x16 micro-tiling loops
           - Broadcast + index mapping for higher dims
           - Call vec_dot (SIMD) on 1 or 2 rows at a time
           - Accumulate into `tmp[32]` then store to dst

Result:
    Fully computed dst (F32)
```

---

## 3. Flow Diagram

```mermaid
flowchart LR
  A[Start mul_mat] --> B[Derive vec_dot_type & from_float]
  B --> C{src1->type == vec_dot_type?}
  C -- Yes --> D[Use src1 directly]
  C -- No --> E[Quant/pack src1 rows into wdata]
  E --> F[Barrier: pack complete]
  D --> F
  F --> G[Compute nr0, nr1]
  G --> H[Select chunk_size]
  H --> I[Compute nchunk0,nchunk1]
  I --> J{Few chunks or NUMA?}
  J -- Yes --> K[Rechunk by dominant dim]
  J -- No --> L[Keep 2D grid]
  K --> M[Init atomic current_chunk]
  L --> M
  M --> N[Thread loop: claim chunk]
  N --> O[Map chunk → (ir0,ir1)]
  O --> P[Process micro-tiles (vec_dot)]
  P --> Q{More chunks?}
  Q -- Yes --> N
  Q -- No --> R[Done]
```

---

## 4. Workspace (`wdata`) Layout

### Standard `mul_mat`
If `src1->type != vec_dot_type` (conversion required):

```
params->wdata
  ┌──────────────────────────────────────────┐
  │ Packed Row 0 (row_size bytes)            │
  │ Packed Row 1                             │
  │ ...                                      │
  │ Packed Row (ne11*ne12*ne13 - 1)          │
  └──────────────────────────────────────────┘
```

- `row_size = ggml_row_size(vec_dot_type, ne10)`
- Row index `(i11,i12,i13)` → linear index: `i11 + i12*ne11 + i13*ne11*ne12`
- Address: `base + linear_index * row_size`

If `src1->type == vec_dot_type`: kernel reads directly from `tensor_data(src1)`; no packing.

### `mul_mat_id` Variant
Adds structures after optional packed rows:

```
[wdata packed rows?]
[matrix_row_counts (int64_t[n_as])]
[matrix_rows mapping array]
[per-expert atomic current_chunk cache lines]
```

Supports sparse / expert routing by grouping active rows before executing similar tiling logic.

---

## 5. Quantization & Type Handling

| Aspect | Logic |
|--------|-------|
| `vec_dot_type` selection | `type_traits_cpu[src0->type].vec_dot_type` |
| Packing function | `from_float = type_traits_cpu[vec_dot_type].from_float` |
| When packing happens | Only if `src1->type != vec_dot_type` |
| Purpose | Match activation row format to weight block layout for efficient dot kernels |
| Parallelization | Threads quantize disjoint segments of each row (block-aligned) |
| Block size | `ggml_blck_size(vec_dot_type)` drives partitioning |

Activations are pre-packed once, avoiding per-dot scalar conversions.

---

## 6. Inner Kernel: `ggml_compute_forward_mul_mat_one_chunk`

Micro-tiling parameters:
- Tile sizes: `blck_0 = 16` (rows), `blck_1 = 16` (columns in flattened ir1 space)
- Temporary accumulator: `float tmp[32];` (16 columns × up to 2 row lanes)
- Multi-row dot: `num_rows_per_vec_dot` is 1 or 2 (architectural kernel optimization)

### Loop Skeleton
```
for iir1 in ir1_start..ir1_end step 16:
  for iir0 in ir0_start..ir0_end step 16:
    for ir1 in iir1..min(iir1+16, ir1_end) step num_rows_per_vec_dot:
        // map flattened ir1 → (i11,i12,i13)
        compute broadcast indices → (i02,i03)
        src0_row base = tensor_data(src0) + offset(i02,i03)
        src1_col ptr  = (packed or direct) row
        dst_col base  = output location (float*)

        for ir0 in iir0..tile_end step num_rows_per_vec_dot:
            vec_dot(ne00, &tmp[ir0 - iir0], strides_if_2row, ...)

        copy tmp lanes → dst_col (memcpy)
```

Broadcast factors:
```
r2 = ne12 / ne02 ; r3 = ne13 / ne03
i03 = i13 / r3   ; i02 = i12 / r2
```
These reuse smaller weight broadcast planes without duplication.

### `vec_dot` Interface (conceptual)
```
vec_dot(
   K,                // ne00 (#cols in src0 row)
   out_ptr,          // destination (fp32 partials)
   out_stride_opt,   // used if multi-row kernel
   a_ptr, a_stride,  // weight row(s)
   b_ptr, b_stride,  // activation row(s)
   n_rows            // 1 or 2
)
```
Internally performs (de)quantization + SIMD multiply-accumulate.

---

## 7. Chunking & Scheduling

| Variable | Meaning |
|----------|---------|
| `nr0` | Output dim 0 = `ne0` |
| `nr1` | Flattened higher dims = `ne1 * ne2 * ne3` |
| `chunk_size` | Base partition (16; or 64 for degenerate 1D) |
| `nchunk0` | `ceil(nr0 / chunk_size)` |
| `nchunk1` | `ceil(nr1 / chunk_size)` |
| Re-chunk condition | `(nchunk0 * nchunk1 < 4 * nth) || ggml_is_numa()` |
| Re-chunk effect | Switch to slab-style: parallelize across dominant dimension only |
| Dispatch | Atomic integer `current_chunk` (initialized to `nth` by thread 0) |

Execution loop:
```
current_chunk = ith; // each thread starts at its own ID
while current_chunk < nchunk0 * nchunk1:
    derive (ith0, ith1)
    process tile
    if nth >= total_chunks: break    // each thread got its tile
    current_chunk = atomic_fetch_add(...)
```

Benefits:
- Lightweight dynamic scheduling for load balance
- Early exit avoids atomic traffic when chunk count ≤ thread count
- NUMA path discourages fragmentation by giving larger contiguous regions per thread

---

## 8. Vectorization Assessment

| Stage | Vectorized? | Notes |
|-------|-------------|-------|
| Packing (`from_float`) | Yes | SIMD compress/scale per block |
| Dot core (`vec_dot`) | Yes | Quant deblock + FMA or integer MAC; architecture-specific |
| Multi-row variant | Yes | Uses 2-row fused MMLA / wide registers |
| tmp → dst copy | Typically | `memcpy` small-size unrolled or vector intrinsic |
| Control flow / indexing | No | Scalar pointer arithmetic; negligible cost |

All heavy FLOP operations (packing + multiply-accumulate) are vectorized. Non-vector code is orchestration only.

---

## 9. Higher-Dimensional Broadcasting

`src0` can have fewer planes in dims 2,3 than `src1`. The code computes ratios `r2`, `r3` and maps output plane indices back to source plane indices via integer division, avoiding materializing expanded weight tensors.

---

## 10. `mul_mat_id` Variant Overview

Adds sparse / expert selection logic:
1. `ids` tensor enumerates which row groups (experts) participate.
2. Builds per-expert row mapping arrays (`matrix_rows`) and counts in workspace.
3. For each expert (`cur_a`): performs a similar chunked tile traversal.
4. Optional repacking of `src1` into `vec_dot_type` layout if needed (same logic gate as standard path).
5. Uses per-expert atomic chunk counters stored in padded cache-line segments (to avoid false sharing) for dynamic tile assignment.

Still funnels into the same vectorized `vec_dot` kernel.

---

## 11. Memory Traffic Characteristics

| Access | Pattern |
|--------|---------|
| `src0` | Strided by `nb01` within row-tiles; good locality for contiguous weight blocks |
| `src1` (packed) | Sequential `row_size` segments; reused across all `ir0` iterations for that `(i11,i12,i13)` |
| `dst` | Written in contiguous sub-blocks per micro-tile |
| `tmp` | Stack-resident (L1) scratch; minimizes partial writes to `dst` |

This arrangement reduces cache thrash and supports prefetch-friendly linear scans of activation data.

---

## 12. Edge & Special Cases

| Situation | Handling |
|-----------|----------|
| Odd lengths / alignment issues | Force `num_rows_per_vec_dot = 1` to avoid boundary crossing |
| Very small matrices | Increase `chunk_size` to 64 to cut scheduling overhead |
| Few chunks relative to threads | Switch to slab-style partitioning |
| NUMA active | Force simplified chunking to enhance locality |
| `src1` already packed | Skip conversion; direct pointer path |
| Multi-row kernel impossible (dimension parity) | Fallback to single-row path |

---

## 13. Type Flow Summary

| Stage | `src0` | `src1` | `dst` | Notes |
|-------|--------|--------|-------|-------|
| Input | Original (possibly quant) | Usually `F32` | - | Layout asserts ensure contiguity where needed |
| Packing | Unchanged | Packed to `vec_dot_type` blocks (if needed) | - | One-time cost amortized over matmul |
| Dot kernel | Quant blocks read & dequant (if quant) | Packed (or F32 if same) | Accumulate in F32 registers | Core SIMD work |
| Final write | - | - | F32 | Copy from `tmp` lanes into `dst` |

---

## 14. Potential Optimizations (Non-exhaustive)

1. **Adaptive tile sizes**: Dynamically detect best `(blck_0, blck_1)` based on cache and quant type.
2. **Prefetch hints**: Explicit prefetch of upcoming `src0` blocks may help on large K.
3. **Interleave output store**: Fuse `vec_dot` results directly into `dst` when safe to skip `tmp` for single-row path.
4. **NUMA-aware packing**: Replicate packed `src1` per NUMA node (if reused across many matmuls) to reduce remote accesses in mirror mode.
5. **Work stealing refinement**: Two-level queues for heterogeneous thread speeds (less critical on uniform cores).

---

## 15. Final Takeaways

- The design cleanly separates: (a) type adaptation, (b) scheduling, (c) SIMD compute.
- Quantization of activations (`src1`) is a pre-processing optimization driven by `vec_dot_type` to align with weight block structure.
- The atomic chunk scheduler balances simplicity and efficiency; special handling improves NUMA locality and avoids undersubscription.
- All heavy arithmetic paths are vectorized through `vec_dot` (and optional external fast GEMM like `llamafile_sgemm`).
- `mul_mat_id` extends the same core pattern to sparse/expert scenarios with minimal extra overhead.

---

## 16. Glossary

| Term | Definition |
|------|------------|
| `vec_dot` | Architecture-specific SIMD routine performing one (or two) row dot products with (de)quantization |
| `vec_dot_type` | Storage element type expected by `vec_dot` for best performance (may be quant block format) |
| Block size | Group size per quant type (e.g., 32/64 elements) governing packing granularity |
| Broadcast factors | Ratios mapping higher-dimension indices back to smaller weight tensor shapes |
| NUMA slab | Strategy of giving threads contiguous macro-regions instead of fine tiles to preserve locality |

---

For further exploration, profiling specific quant formats (e.g., Q4_0 vs Q6_K) within `vec_dot` would illuminate instruction mix and memory bandwidth characteristics.

---

## 17. Quantization Block Format Examples

This section gives concrete, self‑contained examples for the principal GGML quant block families referenced (directly or indirectly) by the `mul_mat` path. Each example shows:

1. Raw float values (one block worth)
2. How scale / (optional min or auxiliary data) are derived conceptually
3. The packed block layout in bytes / fields
4. Reconstruction formula applied inside `vec_dot`

Important: Exact per-block scaling algorithms can have implementation nuances (e.g., rounding, per-subgroup extrema). Below we keep formulas representative and aligned with the struct semantics in `ggml-common.h` and `quants.c`. Field names match struct definitions; sizes reflect the `static_assert`s. Numeric examples pick simple round numbers for clarity.

### Legend

| Symbol | Meaning |
|--------|---------|
| `QK*`  | Block (group) length constant for a quant type (e.g. `QK4_0 = 32`) |
| `d`    | Scale / delta (half or float) |
| `m`    | Minimum (for min-based affine variants) |
| `dmin` | Super-block scale for mins (K-series) |
| `qs`   | Quantized values / packed nibbles or bytes |
| `qh`   | High-bit plane for 5-bit variants |
| `scales` | Packed per-sub-block scale+min codes (K-series) |
| `K_SCALE_SIZE` | Size in bytes of packed scale+min arrays for K-series 4/5-bit |

---

### 17.1 Classic Block Families (non-K, per 32 elements)

#### Q4_0 (`block_q4_0`)

Struct:
```
ggml_half d;        // 2 bytes
uint8_t   qs[16];   // 32 x 4-bit (two per byte)
// Total: 2 + 16 = 18 bytes  (bpw ≈ 18*8 / 32 = 4.5)
```
Example raw floats (32 elems):
```
[-2.0, -1.5, ..., 3.0]  (assume min = -2.0, max = 3.0)
```
Compute scale (conceptual):
```
range = max - min = 5.0
d = range / (2^4 - 1) = 5 / 15 ≈ 0.3333 (stored as fp16)
quant(q) = round( (x - min) / d )  in [0..15]
But Q4_0 historically centers around 0 without explicit min: it encodes signed or zero‑offset pattern by shifting midpoint.
Simplified conceptual reconstruction often noted as:  x ≈ d * (q - 8)
```
Packed representation for first 4 values (illustrative):
```
Values: [-2.0, -1.5, -1.0, -0.5]
q (shifted +8): [-2.0/d + 8, ...]  -> nibble codes (0..15)
Bytes: (q0 | q1<<4), (q2 | q3<<4), ...
```
Reconstruction inside dot kernel:
```
for each nibble q: val = d * (q - 8)
accumulate val * activation_val
```

#### Q4_1 (`block_q4_1`)
Struct adds min:
```
ggml_half d; ggml_half m; // scale & min
uint8_t qs[16];
// Size: 4 + 16 = 20 bytes (bpw = 5.0)
```
Encoding:
```
m = min(x)
d = (max(x) - m) / 15
q = round((x - m)/d)
Reconstruct: x ≈ m + d * q
```
Difference vs Q4_0: explicit affine (per-block min) rather than implicit symmetric offset.

#### Q5_0 (`block_q5_0`)
```
ggml_half d;   // 2 bytes
uint8_t qh[4]; // high 1 bit for each of 32 elements (32 bits)
uint8_t qs[16]; // low 4 bits (nibbles)
// Size: 2 + 4 + 16 = 22 bytes (bpw ≈ 5.5)
```
Encoding (conceptual):
```
Split 5-bit integer q in [0..31] into:
    low 4 bits → packed in qs nibbles
    high bit   → bit positions in qh array
Scale: d similar to symmetric approach; often x ≈ d * (q - 16)
```

#### Q5_1 (`block_q5_1`)
Adds a min like Q4_1:
```
ggml_half d; ggml_half m; qh[4]; qs[16];
// Size: 4 + 4 + 16 = 24 bytes (bpw = 6.0)
```
Reconstruct:
```
q = (low_bits | high_bit<<4)
x ≈ m + d * q
```

#### Q8_0 (`block_q8_0`)
```
ggml_half d;    // 2 bytes
int8_t  qs[32]; // 32 signed bytes
// Size: 2 + 32 = 34 bytes (bpw ≈ 8.5)
```
Encoding:
```
Find scale d so that x/d fits in int8 range [-128..127]; store q = round(x/d)
Reconstruct: x ≈ d * q
```

#### Q8_1 (`block_q8_1`)
Adds precomputed sum factor `s`:
```
ggml_half d; ggml_half s; int8_t qs[32];
// Size: 4 + 32 = 36 bytes
```
Used to accelerate certain fused ops where sum(qs) * d term is reused.

### 17.2 K-Series Super-Block Formats (QK_K elements per block)

K-series use larger super-block (e.g., `QK_K = 256`) subdivided into sub-blocks (16 or 32 elements) with shared packed scale/min metadata to improve effective bpw.

Common pattern for reconstruction (schematic):
```
For sub-block j:
    scale_j = d * dequant_scale(scales[j])    // may also involve dmin
    min_j   = dmin * dequant_min(scales[j])   // only for a+b variants
    For element e in sub-block j:
            q = extract_bits(qs, e)
            x ≈ min_j + scale_j * q   (a*q + b form) OR scale_j * q (pure scale form)
```

#### Q2_K (`block_q2_K`)
```
uint8_t scales[QK_K/16];  // packed 4-bit pairs (scale & min indices)
uint8_t qs[QK_K/4];       // 2-bit quants (4 per byte)
ggml_half d; ggml_half dmin; // super-block scale for scales & mins
// Size: 2*2 + QK_K/16 + QK_K/4 bytes
```
Example (simplified for one 16-element sub-block):
```
Raw sub-block floats → local min & scale proxies -> quant indexes (0..3)
Store 2-bit values in qs; scale/min codes (4 bits each) combine into scales[] bytes.
During dequant: decode nibble pair -> scale_code, min_code
scale = d * LUT_scale[scale_code];  min = dmin * LUT_min[min_code]
```

#### Q3_K (`block_q3_K`)
```
hmask[QK_K/8]  // high bit mask
qs[QK_K/4]     // low 2 bits
scales[12]     // packed 6-bit scales (groups)
ggml_half d;   // super scale
```
Reconstruct 3-bit q via combining `hmask` and `qs` low bits; apply grouped scale.

#### Q4_K (`block_q4_K`)
```
ggml_half d; ggml_half dmin;
uint8_t scales[K_SCALE_SIZE]; // contains interleaved quantized scale & min codes
uint8_t qs[QK_K/2]; // 4-bit quants
```
Sub-block (32 elements). Per sub-block encoded scale & min; reconstruct q then x ≈ min_j + scale_j * q.

#### Q5_K (`block_q5_K`)
Same as Q4_K plus high-bit array:
```
ggml_half d; ggml_half dmin; scales[K_SCALE_SIZE]; qh[QK_K/8]; qs[QK_K/2];
// q (5 bits) = (high_bit<<4) | low_4_bits
```

#### Q6_K (`block_q6_K`)
```
ql[QK_K/2]; // low 4 bits
qh[QK_K/4]; // high 2 bits (packed)
scales[QK_K/16]; // signed 8-bit per small group
ggml_half d;     // global multiplier for scales
```
Reconstruct 6-bit q combining ql & qh; per-group scale = d * scales[g].

#### Q8_K (`block_q8_K`)
```
float  d;              // (note: float, not half)
int8_t qs[QK_K];       // signed 8-bit
int16_t bsums[QK_K/16]; // group sums to accelerate dot (bias-like reuse)
```
Reconstruction: x ≈ d * q. Group sums provide quick partial sum reuse (e.g., for a*q + b fusion or norm/scale adjustments).

### 17.3 IQ / TQ Experimental Families (Brief)

Included for completeness (not exhaustive in example math here):
| Type | Key Fields | Notes |
|------|------------|-------|
| `block_tq1_0` | `qs`, `qh`, `d` | Ternary with mixed packing (1.6875 bpw) |
| `block_tq2_0` | `qs`, `d` | 2-bit ternary-like variant |
| `block_iq2_xxs` | `d`, `qs[QK_K/8]` (16-bit) | Near true 2-bit with per-256 block scale |
| `block_iq2_xs` | `d`, `qs`, `scales` | Adds fine-grained scale modulation |

### 17.4 Worked Mini Example (Q4_1)

Given 8 of the 32 values (sub-sample for brevity):
```
x = [0.10, 0.12, 0.05, 0.00, -0.03, 0.07, 0.15, 0.11, ...]
min m = -0.03
max = 0.15
range = 0.18
d = range / 15 = 0.012
Quant:
q = round((x - m)/d)
First 8 q values:
    (0.10 +0.03)/0.012 = 10.83 -> 11
    (0.12 +0.03)/0.012 = 12.50 -> 13
    (0.05 +0.03)/0.012 = 6.66 -> 7
    (0.00 +0.03)/0.012 = 2.50 -> 3
    (-0.03+0.03)/0.012 = 0    -> 0
    (0.07 +0.03)/0.012 = 8.33 -> 8
    (0.15 +0.03)/0.012 = 15.0 -> 15
    (0.11 +0.03)/0.012 = 11.66 -> 12
Pack pairs: (11 | 13<<4), (7 | 3<<4), (0 | 8<<4), (15 | 12<<4), ...
Reconstruct one: q=11 → x' = m + d*q = -0.03 + 0.012*11 = 0.102 (close to 0.10)
```

### 17.5 Interaction with `vec_dot`

During `mul_mat`:
1. Packed activation block (if needed) matches the layout expected by weight’s `vec_dot_type`.
2. `vec_dot` loads `d` (and `m`/`dmin`/scales if present) into SIMD registers.
3. Unpacks `qs` (and `qh` when present) into integer lanes.
4. Applies: `val = m + scale * q` or `val = scale * q` depending on format.
5. Fused multiply-add with activation lanes accumulates into fp32 accumulators.
6. Where group sums (`bsums`) or precomputed `s` exist, kernel may shortcut parts of the expansion.

### 17.6 Summary Table (Approx Bits/Weight)

| Format | Elements/Block (`QK`) | Bytes/Block | Bits/Weight (approx) | Affine (min) | Extra Metadata |
|--------|-----------------------|-------------|----------------------|--------------|----------------|
| Q4_0   | 32 | 18  | 4.5  | implicit (sym) | - |
| Q4_1   | 32 | 20  | 5.0  | yes | per-block min |
| Q5_0   | 32 | 22  | 5.5  | implicit | high-bit plane |
| Q5_1   | 32 | 24  | 6.0  | yes | high-bit + min |
| Q8_0   | 32 | 34  | 8.5  | symmetric | full int8 |
| Q8_1   | 32 | 36  | 9.0  | symmetric | sum factor |
| Q2_K   | 256| 2*2 + 16 + 64 =  (depends on constants) | ~2.625 | yes (a*q+b) | packed scales+mins |
| Q3_K   | 256| (QK/8)+(QK/4)+12+2 =  (≈) | ~3.4375 | scale only | hmask + scales |
| Q4_K   | 256| 2*2 + K_SCALE_SIZE + QK/2 | 4.5 | yes | scales array |
| Q5_K   | 256| 2*2 + K_SCALE_SIZE + QK/2 + QK/8 | 5.5 | yes | qh + scales |
| Q6_K   | 256| 2 + 3*QK/4 + QK/16 | 6.5625 | scale only | split ql/qh + scales |
| Q8_K   | 256| 4 + QK + QK/16*2 | 8+ | scale only | bsums |

Notes:
- Table leaves some expressions symbolic (QK, K_SCALE_SIZE) to stay consistent with code constants.
- “Affine (min)” indicates whether a per-(sub)block additive offset is stored/derivable.

---

End of quantization examples.

### 17.7 Quant Function Cross-Reference

The following table maps the described formats to the primary front-door quantization entry points found in `ggml/src/ggml-cpu/quants.c` (line numbers may drift; symbolic names are stable). Each "front" function typically delegates to a `_ref` or arch-specific implementation after possible runtime dispatch.

| Format | Quant Function | Internal Helper (example) | Notes |
|--------|----------------|---------------------------|-------|
| Q4_0   | `quantize_row_q4_0` | `quantize_row_q4_0_ref` | Packs 32 floats → 18B block (d + 16 nibbles) |
| Q4_1   | `quantize_row_q4_1` | `quantize_row_q4_1_ref` | Affine (d,m) + 16 nibbles |
| Q5_0   | `quantize_row_q5_0` | `quantize_row_q5_0_ref` | Uses `qh` (high bits) + nibbles |
| Q5_1   | `quantize_row_q5_1` | `quantize_row_q5_1_ref` | (d,m) + `qh` + nibbles |
| Q8_0   | `quantize_row_q8_0_generic` | `quantize_row_q8_0_ref` | Int8 symmetric |
| Q2_K   | `quantize_row_q2_K` | `quantize_row_q2_K_ref` | Super-block with packed scales+mins |
| Q3_K   | `quantize_row_q3_K` | `quantize_row_q3_K_ref` | hmask + low bits + 6-bit scales |
| Q4_K   | `quantize_row_q4_K` | `quantize_row_q4_K_ref` | (d,dmin) + packed `scales` + 4-bit qs |
| Q5_K   | `quantize_row_q5_K` | `quantize_row_q5_K_ref` | Adds `qh` for 5th bit |
| Q6_K   | `quantize_row_q6_K` | `quantize_row_q6_K_ref` | Split ql/qh + signed scales |
| Q8_K   | `quantize_row_q8_K_generic` | `quantize_row_q8_K_ref` | Includes `bsums` for group reuse |

Dequantization & vector dot use corresponding architecture-tuned paths (SIMD or intrinsic) that interpret these structures directly during `vec_dot`.

### 17.8 Q6_K Bit Packing Diagram

`block_q6_K` fields recap:
```
uint8_t ql[QK_K/2];   // low 4 bits for each element (2 elems per byte)
uint8_t qh[QK_K/4];   // high 2 bits for each element (4 elems per byte)
int8_t  scales[QK_K/16]; // per 16-element group signed scale code
ggml_half d;          // super scale
```

Let `QK_K = 256`. Then:
```
ql size = 256/2 = 128 bytes
qh size = 256/4 = 64 bytes
scales  = 256/16 = 16 bytes (each int8)
```

Each element's 6-bit quant code q[ i ] is formed by:
```
low4  = (ql[ i/2 ] >> ( (i % 2) * 4 )) & 0xF
// In qh: every byte packs 4 high-2-bit fields for elements (g*4 .. g*4+3):
high2 = (qh[ i/4 ] >> ( (i % 4) * 2 )) & 0x3
q = (high2 << 4) | low4   // range 0..63
```

Group / scale selection (16 elements per scale index):
```
group = i / 16
scale_code = scales[group]       // signed int8
scale = d * decode_scale(scale_code)
// decode_scale may apply linear or LUT-based mapping (implementation-dependent)
value ≈ scale * q                // (pure multiplicative form, no per-element offset)
```

#### Visualization (Packed Layout Slice)

Below shows 8 consecutive elements (indices 0..7) and how their bits are sourced. Two bytes from `ql`, two bytes from `qh` cover these 8 elements:

```text
Indices:   0     1     2     3     4     5     6     7
ql bytes: [ b0 ---------------- ] [ b1 ---------------- ]
                    low4(0)  low4(1)      low4(2)  low4(3)      (each nibble)
                    (i=0)    (i=1)        (i=2)    (i=3)

ql mapping nibble order per byte: bits 3..0 -> element even, bits 7..4 -> element odd

qh byte 0 (covers elements 0..3):
    bits 1..0 -> high2(0)
    bits 3..2 -> high2(1)
    bits 5..4 -> high2(2)
    bits 7..6 -> high2(3)

qh byte 1 (covers elements 4..7):
    bits 1..0 -> high2(4)
    bits 3..2 -> high2(5)
    bits 5..4 -> high2(6)
    bits 7..6 -> high2(7)
```

#### Mermaid Bit Packing Diagram

```mermaid
flowchart TB
    subgraph QL[ql bytes]
        ql0[byte b0\n bits 7..4 -> elem1 low4\n bits 3..0 -> elem0 low4]
        ql1[byte b1\n bits 7..4 -> elem3 low4\n bits 3..0 -> elem2 low4]
    end
    subgraph QH[qh bytes]
        qh0[byte h0\n (1..0) e0 hi2\n (3..2) e1 hi2\n (5..4) e2 hi2\n (7..6) e3 hi2]
        qh1[byte h1\n (1..0) e4 hi2\n (3..2) e5 hi2\n (5..4) e6 hi2\n (7..6) e7 hi2]
    end
    subgraph RECON[Reconstruction]
        r0[q = (hi2<<4)|low4]
        r1[value = scale[group]*q]
    end
    ql0 --> r0
    ql1 --> r0
    qh0 --> r0
    qh1 --> r0
    r0 --> r1
```

#### Example Numeric Mini-Slice

Assume for elements 0..3:
```
low4:  [ 9, 4, 15, 2 ]
high2: [ 1, 0, 2, 3 ]  // from qh bits
q:     [ (1<<4)|9=25, (0<<4)|4=4, (2<<4)|15=47, (3<<4)|2=50 ]
scale (group 0): scale = 0.0123
values ≈ [0.3075, 0.0492, 0.5781, 0.6150]
```

This diagram clarifies how the 6-bit quant code is physically spread across `ql` and `qh` arrays before SIMD expansion in `vec_dot`.

---

## 18. `vec_dot` SIMD Translation & Optimization Analysis

This section drills into how the generic `ggml_vec_dot_*` functions map to SIMD across architectures (x86 AVX2/AVX512, ARM NEON/SVE, fallback scalar) and the implications for `mul_mat` tiling and future optimization.

### 18.1 Function Pointer Dispatch Recap

The `type_traits_cpu[type].vec_dot` field points to a specialized routine (e.g. `ggml_vec_dot_q4_0_q8_0`) selected at runtime compile/arch configuration. Each variant implements:
```
void ggml_vec_dot_X_Y(int n, float *s, size_t bs, const void *x, size_t bx,
                              const void *y, size_t by, int nrc)
```
Parameters (`bs`, `bx`, `by`) provide per-row stride support for multi-row fused calls (when `num_rows_per_vec_dot == 2`). `nrc` reflects number of result columns (lanes) processed simultaneously.

### 18.2 Common Structural Phases Inside Quant Dot Kernels
1. **Prefetch / pointer setup:** Cast raw `vx`, `vy` to block structs.
2. **Block loop:** Iterate over quant blocks (e.g., 32 or 256 elements) accumulating partial sums.
3. **Dequant decode:**
    - Load scale(s) & min where needed.
    - Expand packed bits (nibbles, high-bit planes) to int8 vectors.
    - Apply (a*q + b) or (scale * q) into fp32 or wider integer accumulators.
4. **FMAs / integer dot:**
    - Use vector multiply-add (AVX2 `_mm256_fmadd_ps`, NEON `vfmaq.f32`, SVE `svmad_f16_x`, etc.).
    - For purely integer paths (e.g., int8*int8) rely on widening multiply + horizontal add sequences.
5. **Partial reduction:** Keep several SIMD accumulators live to hide pipeline latency.
6. **Horizontal reduction:** Sum lanes into scalar floats; store into `s[0..nrc-1]`.

### 18.3 FP16 & BF16 Paths (From `vec.h` Snippet)

`ggml_vec_dot_f16_unroll` demonstrates macro-based abstraction:
```
GGML_VEC_DOT_UNROLL = 2  (processes 2 dot products in parallel)
Loop unroll loads multiple vector registers (e.g., ay + ax per unrolled lane) then FMA chains.
Partial sums kept separated (sum_00,sum_01,...) before final reduction.
```
Architectural branches:
| Arch | Key Intrinsics / Ops | Notes |
|------|----------------------|-------|
| SVE  | `svld1_f16`, `svmad_f16_x`, predicated tail | Dynamic vector length; loops use masked remainder |
| NEON | (implied by macros) `vld1q_f16`, `vfmaq.f16` (if FP16) | May fallback to scalar convert if no native FP16 FMA |
| x86 (AVX2+) | Packs half->float expand then `_mm256_fmadd_ps` | Potential cost in conversion bandwidth |
| Fallback | Scalar loop | No vector speedup |

Reduction macros (`GGML_F16x_VEC_REDUCE`) fold multiple accumulator registers minimizing data movement.

### 18.4 Representative Quant Pair (Q4_0 × Q8_0)

Typical inner pattern (conceptual pseudo-SIMD):
```
for each block (32 elements):
  load half d (scale)
  load 16 packed nibbles (qs)
  expand to 32 int8 (sign adjust: q - 8)
  load corresponding 32 int8 from activation (already q8_0)
  widen to int16 or int32
  multiply pairwise → accumulate into 32-bit lanes
  later: convert to float and scale by d * dy (if activation also scaled) or separated scaling factors
```
Optimizations often include:
 - Using `_mm256_maddubs_epi16` / `_mm256_madd_epi16` sequences on x86 for int8 accumulation.
 - Pairing two rows (when `num_rows_per_vec_dot == 2`) to reuse broadcasting of activation values.

### 18.5 K-Series Extended Example (Q6_K × Q8_K)

Process (per 256-element super-block):
1. Load `d` (fp16) → broadcast to f32 register.
2. For each 16-element subgroup:
    - Load signed `scale_code`; convert to float scale = d * decode(scale_code).
    - Gather low nibble vector from `ql`; extract high2 from `qh`; combine into 6-bit q.
    - Convert q → f32 (or int16 then f32) via widening.
    - Load 16 activation int8 values; widen → f32 or int16.
    - FMA accumulate: sum += (scale * q) * act.
3. After all groups: horizontal add partial sums.
4. If activation side includes its own per-group scaling (e.g., q8_K uses global d plus optional bias sums), multiply once at the end.

### 18.6 Unrolling & Latency Hiding

Rationale for `GGML_VEC_DOT_UNROLL = 2`:
 - Keeps at least two independent accumulation chains to cover FMA latency (esp. on ARM / AVX2 ~4-5 cycle dependent latency).
 - Higher unroll can increase register pressure (risking spills). With mixed quant decode logic, two-way unroll is a sweet spot.

Potential improvement avenues:
| Idea | Benefit | Risk |
|------|---------|------|
| Adaptive unroll (arch-specific) | Better ILP on wide SVE/AVX512 | Code complexity, binary size |
| Software pipelining over block decode | Overlap nibble unpack & previous FMAs | Hard to maintain, limited without more accumulators |
| Interleave prefetch for upcoming `ql/qh` | Reduce stalls on memory-bound large K | Might pollute cache if working set small |

### 18.7 Memory Alignment & Access Patterns

Observations:
 - Packed quant data is naturally byte-aligned; vector loads might benefit from aligning block starts to 32 or 64B boundaries for AVX2/AVX512 prefetching.
 - Activation rows (`q8_0`/`q8_K`) are sequential, enabling efficient hardware prefetch.
 - Scale arrays for K-series are small and repeatedly accessed; consider forcing them into L1 via software prefetch on large loops.

### 18.8 Horizontal Reduction Strategies

Current approach: maintain multiple accumulators, then use reduce macros/hadd sequences. For AVX2 typical pattern is:
```
acc0 += acc1; acc2 += acc3; // pairwise
acc0 += acc2;                 // collapse
horizontal_add(acc0)          // final scalar
```
On AVX512: could leverage `_mm512_reduce_add_ps` style intrinsics (or manual shuffles) to reduce instruction count.

### 18.9 Multi-Row Fused Dot (num_rows_per_vec_dot = 2)

When the caller requests 2 rows per vec_dot invocation:
 - Stride parameters (`bs`, `bx`, `by`) pass non-zero distances enabling the kernel to fetch row0/row1 weight data while reusing a single activation column vector.
 - Saves half the activation load bandwidth for that pair.
 - Encourages mmla usage on ARM (matrix multiply-accumulate) for int8 pairs.

### 18.10 Architectural Specific Considerations

| Architecture | Strengths | Potential Gaps |
|--------------|-----------|----------------|
| AVX2 | Rich int8 madd patterns; 256-bit | Lacks native bf16 FMA (needs convert) |
| AVX512 (if enabled) | Wider vectors; masked ops simplify tails | Higher power, potential downclock; code path size |
| NEON | Efficient int8 + widening; low latency FMAs | Limited register file vs wide unroll |
| SVE | Scalable vector length; predication for tails | Complexity in writing hand-tuned decode macros |
| RVV (planned) | Flexible LMUL & tail handling | Implementation pending (TODO comments) |

### 18.11 Optimization Opportunities (Actionable Candidates)

1. **AVX512 Specialized Paths:** Provide alternative `GGML_VEC_DOT_UNROLL = 4` variant guarded by compile-time detection; evaluate register pressure.
2. **Quant Decode Prefetch:** Software prefetch next block's `qs/qh` 2 iterations ahead when bandwidth stalls observed.
3. **Scale/Min Broadcast Hoisting:** For K-series, precompute float scale vector array once per super-block to avoid repeated decode inside inner element loop.
4. **Mixed-Precision Accumulation:** Accumulate int8 products into int32 then convert once (already partly done); explore bf16 accumulation on AVX512-BF16 / SVE2 for specific formats.
5. **Two-Level Blocking with Cache Tuning:** Align `mul_mat` outer chunking so that a thread repeatedly reuses the same activation tile while streaming distinct weight tiles (temporal locality for activations).
6. **NUMA-Aware vec_dot Batching:** In mirror mode, cluster dot calls so each thread finishes a contiguous range of activation columns before moving on—reducing cross-node traffic.
7. **Decode Vectorization Enhancements:** For Q6_K, pack `ql` & `qh` extraction via lookup table shuffle (e.g., `_mm256_shuffle_epi8`) to form full 6-bit values in fewer uops.
8. **Tail Handling Unification:** Replace scalar leftover loops with masked vector ops on AVX512/SVE for reduced branch overhead.
9. **Function Pointer Devirtualization:** For hot loops with known types at compile time (templated builds), inline specific `vec_dot` to enable further compiler auto-vectorization around call site.

### 18.12 Risk / Benefit Summary

| Optimization | Est. Gain | Complexity | Notes |
|--------------|-----------|------------|-------|
| AVX512 unroll 4 | 5–12% (large K) | Medium | Needs thorough perf counters review |
| Prefetch next quant block | 0–5% | Low | Only if memory BW bound |
| K-series scale hoist | 2–6% | Low | Straightforward refactor |
| Q6_K shuffle decode | 3–8% | Medium | Architecture-specific code paths |
| NUMA vec_dot batching | 5–15% (multi-node) | Medium | Integrate with existing mirror scheduler |
| Template devirtualization | 1–4% | High (build variants) | Increases binary size |

---

End of Section 18.

