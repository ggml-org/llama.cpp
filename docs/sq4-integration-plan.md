# SQ4 Integration Plan for llama.cpp

## What Is SQ4

Synaptic Quantization 4-bit. Percentile band calibration + outlier sideband. Same 4 bits per weight as Q4_0, output indistinguishable from FP16. No calibration data needed — one pass over the weights.

- [Format spec](https://github.com/tensorwire/ai/blob/main/docs/sq4-spec.md)
- [White paper](https://github.com/tensorwire/ai/blob/main/docs/sq4-whitepaper.md)
- [Reference implementation](https://github.com/tensorwire/mongoose) (Go + CUDA + Metal)

## Why Add It

Q4_0 encodes 90-99% of weights with a single reconstruction value (bucket 0 absorbs everything near zero). SQ4 uses equal-count percentile bands — every one of the 8 magnitude levels is used equally. The result: FP16-equivalent quality at Q4 memory cost.

IQ4_NL already proved non-linear 4-bit works in llama.cpp. SQ4 is the same pattern (16-entry LUT dequant) with a different encoding: empirical percentile bands instead of k-means clustering, plus an FP32 outlier sideband for the top 0.1% of weights.

## Architecture: How It Maps to GGML

SQ4 maps cleanly to the existing IQ4_NL pattern:

| SQ4 Concept | GGML Equivalent |
|-------------|-----------------|
| 16-entry LUT (8 positive + 8 negative band means) | Same as IQ4_NL's 16-value lookup table |
| Nibble packing [sign:1\|band:3] | Same as Q4_0 nibble packing (2 per byte) |
| Per-tensor band calibration (8 means + 7 boundaries) | Per-super-block metadata (like K-quant scales) |
| Outlier sideband (0.1% at FP32) | New — needs sparse correction pass |

### Block Structure

Use super-block size 256 (matching K-quants) to amortize the band metadata:

```c
#define QK_SQ4 256

typedef struct {
    float bands[8];            // 8 band means (reconstruction LUT) — 32 bytes
    uint8_t qs[QK_SQ4 / 2];   // nibbles [sign:1|band:3], 2 per byte — 128 bytes
    uint8_t n_outliers;        // count of outliers in this block — 1 byte
    // outliers follow as (uint8_t offset, ggml_half value) pairs
    // offset is position within the 256-element block (0-255)
    // ~0.1% = ~0.25 outliers per block on average, max ~3-4
} block_sq4;
// Base size: 161 bytes for 256 weights = 5.03 bits/weight
// With outliers: +3 bytes per outlier (~0.25 avg) = ~5.06 bits/weight
```

Alternative: store outliers out-of-band in a separate tensor (simpler block struct, outlier correction as a post-pass). This avoids variable-size blocks.

**Recommendation**: Fixed block struct with NO inline outliers for v1. Store outliers in a separate sideband tensor per weight matrix. This keeps the block size fixed (required by GGML) and the dequant kernel simple. Outlier correction happens as a post-pass after the main matvec.

```c
// v1: Fixed-size block, no inline outliers
#define QK_SQ4 256

typedef struct {
    float bands[8];            // 8 band means — 32 bytes
    uint8_t qs[QK_SQ4 / 2];   // nibbles — 128 bytes
} block_sq4;
// 160 bytes for 256 weights = 5.0 bits/weight
```

Outlier sideband stored as a separate GGUF tensor `*.outlier_idx` + `*.outlier_val` per weight tensor.

---

## Implementation Steps

### Step 1: GGML Type Definition

**Files:**
- `ggml/include/ggml.h` — add `GGML_TYPE_SQ4` enum value, increment `GGML_TYPE_COUNT`
- `ggml/src/ggml-common.h` — add `QK_SQ4`, `block_sq4` struct

### Step 2: Reference Quantize + Dequant (CPU)

**File:** `ggml/src/ggml-quants.c`

```c
// Quantize: sort absolute values, compute percentile bands, pack nibbles
void quantize_row_sq4_ref(const float * x, block_sq4 * y, int64_t k);

// Dequant: 16-entry LUT lookup per nibble
void dequantize_row_sq4(const block_sq4 * x, float * y, int64_t k);

// Vec dot: dequant + dot product (or fused LUT dot)
void ggml_vec_dot_sq4_q8_0(int n, float * s, size_t bs,
    const void * vx, size_t bx, const void * vy, size_t by, int nrc);
```

**Register in type_traits** (`ggml/src/ggml.c`):
```c
[GGML_TYPE_SQ4] = {
    .type_name = "sq4",
    .blck_size = QK_SQ4,
    .type_size = sizeof(block_sq4),
    .is_quantized = true,
    .to_float = (ggml_to_float_t) dequantize_row_sq4,
    .from_float = (ggml_from_float_t) quantize_row_sq4_ref,
    .vec_dot = (ggml_vec_dot_t) ggml_vec_dot_sq4_q8_0,
    .vec_dot_type = GGML_TYPE_Q8_0,
    .nrows = 1,
},
```

### Step 3: CUDA Kernel

**File:** `ggml/src/ggml-cuda/dequantize.cuh` (or new file `sq4.cu`)

```cuda
// SQ4 dequant: 16-entry register LUT, nibble extract, done.
// Same pattern as IQ4_NL but with per-block band means instead of global table.
static __device__ void dequantize_sq4(const void * vx, const int64_t ib,
    const int iqs, dfloat2 & v) {
    const block_sq4 * x = (const block_sq4 *)vx + ib;
    const uint8_t q = x->qs[iqs];
    v.x = x->bands[q & 0x07] * ((q & 0x08) ? -1.0f : 1.0f);
    // second nibble
    const uint8_t q2 = q >> 4;
    v.y = x->bands[q2 & 0x07] * ((q2 & 0x08) ? -1.0f : 1.0f);
}
```

Register in CUDA backend dispatch.

### Step 4: Metal Shader

**File:** `ggml/src/ggml-metal/ggml-metal.metal`

```metal
kernel void kernel_dequantize_sq4(
    const device block_sq4 * x [[buffer(0)]],
    device float * y [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    const uint block_idx = idx / QK_SQ4;
    const uint elem_idx = idx % QK_SQ4;
    const uint byte_idx = elem_idx / 2;
    const uint nibble_shift = (elem_idx & 1) * 4;

    const uint8_t q = (x[block_idx].qs[byte_idx] >> nibble_shift) & 0x0F;
    const uint band = q & 0x07;
    const float sign = (q & 0x08) ? -1.0f : 1.0f;
    y[idx] = x[block_idx].bands[band] * sign;
}
```

Register in Metal backend (`ggml-metal.cpp`).

### Step 5: Quantize Tool

**Files:**
- `include/llama.h` — add `LLAMA_FTYPE_MOSTLY_SQ4`
- `src/llama-quant.cpp` — add type mapping
- `tools/quantize/quantize.cpp` — add `{"SQ4", LLAMA_FTYPE_MOSTLY_SQ4}` to `QUANT_OPTIONS`

The quantize function:
1. Read FP32/FP16 weights
2. For each block of 256 weights: sort absolute values, compute 8 percentile bands, assign nibbles, identify outliers
3. Write block_sq4 data
4. Write outlier sideband tensors (if implementing outlier correction)

### Step 6: GGUF Converter (Python)

**File:** `convert_hf_to_gguf.py` or new `tools/convert_sq4.py`

Read HuggingFace safetensors → quantize to SQ4 → write GGUF. This can also be done via the C++ quantize tool (Step 5) by first converting to FP16 GGUF then quantizing.

### Step 7: Outlier Sideband (v2, optional for initial PR)

Store outliers as additional GGUF tensors:
- `blk.{L}.attn_q.outlier_idx` — uint16 offsets within each super-block
- `blk.{L}.attn_q.outlier_val` — FP16 exact values

Post-matvec correction kernel:
```cuda
// For each outlier: output[row] += (exact_val - band_approx) * input[col]
```

This can be deferred to a follow-up PR. The base SQ4 without outliers will already beat Q4_0 on quality — the outlier sideband is the last few percent.

---

## PR Strategy

**PR 1 (initial):** SQ4 base — GGML type, block struct, CPU dequant, CUDA kernel, Metal shader, quantize tool. No outlier sideband. This is a clean, reviewable PR that follows the IQ4_NL pattern exactly.

**PR 2 (follow-up):** Outlier sideband — additional GGUF tensors, post-matvec correction kernel, quality comparison showing the outlier benefit.

**PR 3 (follow-up):** Optimized kernels — vectorized reads, shared memory LUT, batch-aware paths.

## File Checklist (PR 1)

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Add `GGML_TYPE_SQ4`, increment `GGML_TYPE_COUNT` |
| `ggml/src/ggml-common.h` | Add `QK_SQ4`, `block_sq4` struct |
| `ggml/src/ggml-quants.h` | Declare `quantize_row_sq4_ref`, `dequantize_row_sq4` |
| `ggml/src/ggml-quants.c` | Implement quantize + dequant + vec_dot |
| `ggml/src/ggml.c` | Register in `type_traits` array |
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU kernel dispatch |
| `ggml/src/ggml-cuda/dequantize.cuh` | CUDA dequant device function |
| `ggml/src/ggml-cuda/mmvq.cu` | CUDA matvec dispatch |
| `ggml/src/ggml-metal/ggml-metal.metal` | Metal dequant + matvec shader |
| `ggml/src/ggml-metal/ggml-metal.cpp` | Metal kernel registration |
| `include/llama.h` | Add `LLAMA_FTYPE_MOSTLY_SQ4` |
| `src/llama-quant.cpp` | Type mapping |
| `tools/quantize/quantize.cpp` | Add to QUANT_OPTIONS |

## Precedent

IQ4_NL (PR #5590) added non-linear 4-bit quantization using a 16-value lookup table. It was merged and is in production. SQ4 follows the same pattern — the difference is the encoding (percentile bands vs k-means) and the addition of an outlier sideband. The community has demonstrated appetite for quality-improving quant types at the same bit budget.
