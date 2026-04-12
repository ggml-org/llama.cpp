# TurboQuant KV Cache Compression — Design Document

**Status:** Draft (Sprint 1 scaffolding, no functional implementation yet)
**Author:** Andrew H. Bond <agi.hpc@gmail.com>
**Branch:** `feature/turboquant-kv-cache`
**Reference:** [turboquant-pro v0.7.0](https://github.com/ahb-sjsu/turboquant-pro)

## Motivation

Long-context inference (32K+ tokens) is VRAM-bound. A 7B model at 32K context with fp16 KV cache uses ~16 GB just for KV — half the VRAM of a 32 GB GPU. Existing KV quantization in `llama.cpp` (Q4_0, Q5_0, Q8_0) reduces this but with quality penalties at lower bit widths.

TurboQuant provides higher compression at better quality by adding a random orthogonal rotation before scalar quantization:

| Method | Compression | Cosine Sim | PPL Loss vs F16 |
|--------|-------------|------------|-----------------|
| F16 (baseline) | 1.0× | 1.000 | 0% |
| Q8_0 (current) | 2.0× | 0.998 | ~0.1% |
| Q4_0 (current) | 3.5× | 0.971 | ~1-2% |
| **TQ KV3 (new)** | **5.1×** | **>0.978** | **<0.5%** |
| **TQ KV4 (new)** | **3.8×** | **>0.995** | **<0.2%** |
| **TQ KV2 (new)** | **7.9×** | **>0.926** | **<2%** |

Quality numbers from Zandieh et al. (ICLR 2026) "PolarQuant + QJL" and the existing Python implementation in `turboquant-pro/turboquant_pro/_kv_cache.py`.

## Algorithm

PolarQuant compresses each `head_dim`-sized vector (K or V row) in three steps:

1. **Extract L2 norm** — store `||v||_2` as fp32 (per-vector, ~4 bytes overhead per head_dim values)
2. **Rotate to unit hypersphere** — multiply unit vector by random orthogonal matrix `Π`. After rotation, coordinates are approximately i.i.d. Gaussian. For `head_dim ≤ 4096`: precomputed QR-decomposed rotation matrix. For larger: structured sign-flip + permutation (O(d) memory and apply cost).
3. **Scalar quantize each coordinate** — Lloyd-Max optimal codebook for N(0, 1/√d), with bit-packing (8 × 3-bit = 3 bytes for TQ KV3).

Decompression inverts: unpack indices → look up centroid → unrotate → scale by stored norm.

## Architecture

### New ggml Types

Add to `ggml/include/ggml.h`:

```c
enum ggml_type {
    /* ... existing types ... */
    GGML_TYPE_TQ_KV3  = 42,  // TurboQuant KV cache, 3-bit
    GGML_TYPE_TQ_KV4  = 43,  // TurboQuant KV cache, 4-bit
    GGML_TYPE_TQ_KV2  = 44,  // TurboQuant KV cache, 2-bit (extreme compression)
    GGML_TYPE_COUNT   = 45,  // bump
};
```

### Block Structure

Each block stores one quantized vector of length `head_dim`:

```c
// TQ_KV3: 3 bits per element + per-vector L2 norm
typedef struct {
    float    norm;                       // L2 norm of original vector (4 bytes)
    uint8_t  indices[(HEAD_DIM*3 + 7)/8]; // bit-packed 3-bit indices (3*HEAD_DIM/8 bytes)
} block_tq_kv3;

// TQ_KV4: 4 bits per element + per-vector norm
typedef struct {
    float    norm;
    uint8_t  indices[HEAD_DIM/2];
} block_tq_kv4;
```

For typical `head_dim=128`: TQ KV3 = 4 + 48 = 52 bytes vs fp16's 256 bytes (4.9× compression) vs Q4_0's ~72 bytes.

### Rotation Matrix Storage

A single rotation matrix is shared across **all** KV blocks for a given `head_dim` (deterministic from a seed). Stored once per model context, not per block. Memory: `4 * head_dim²` bytes (e.g. 64 KB for `head_dim=128`).

Initialized lazily on first KV insertion: `init_rotation(head_dim, seed=42)`.

### CLI Integration

Modify `common/arg.cpp` `kv_cache_types` vector:

```cpp
const std::vector<ggml_type> kv_cache_types = {
    /* ... existing ... */
    GGML_TYPE_TQ_KV2,
    GGML_TYPE_TQ_KV3,
    GGML_TYPE_TQ_KV4,
};
```

User invocation:
```bash
./llama-cli --cache-type-k tq_kv3 --cache-type-v tq_kv4 ...
```

## Integration Points

| File | Lines | Change |
|------|-------|--------|
| `ggml/include/ggml.h` | ~431 | Add `GGML_TYPE_TQ_KV{2,3,4}` enum values |
| `ggml/src/ggml-quants.h` | various | Declare `quantize_tq_kv{2,3,4}_row()`, `dequantize_tq_kv{2,3,4}_row()` |
| `ggml/src/ggml-quants.c` | various | Implement quantize/dequantize (CPU reference, Sprint 2) |
| `ggml/src/ggml-cuda/turboquant.cu` | new | CUDA kernels (Sprint 3) |
| `ggml/src/ggml.c` | type_traits | Register new types in `type_traits` table (block size, name, dot product) |
| `common/arg.cpp` | 383-393 | Add new types to `kv_cache_types` |
| `src/llama-kv-turboquant.{h,cpp}` | new | TurboQuant-specific helpers (rotation init, hot/cold tiering) |
| `src/llama-kv-cache.cpp` | various | Use TQ helpers when `type_k`/`type_v` is TQ |
| `tests/test-tq-kv.cpp` | new | Unit tests (Sprint 2) |

## Hot/Cold Tiering (Sprint 4)

Mirroring `TurboQuantKVCache` from `turboquant-pro`:

- **Hot tier** — last N tokens stored uncompressed (fp16). Default N=512, configurable via `--kv-hot-window`.
- **Cold tier** — older tokens stored TQ-compressed. Auto-flush from hot to cold when hot window exceeds threshold.
- During attention: hot tier read directly; cold tier blocks decompressed on-demand into a scratch buffer.

This amortizes compression cost (only happens at flush time, not per-token) and gives full fp16 precision for the recent tokens that matter most.

## Performance Targets (Sprint 3 acceptance)

| Metric | Target | Baseline (F16) |
|--------|--------|----------------|
| Inference throughput | within 2× of F16 | 100% |
| KV memory at 32K context | <30% of F16 | 100% |
| Perplexity on WikiText-2 | <0.5% loss vs F16 | — |
| First-token latency | within 1.5× of F16 | — |

## Out of Scope (Future Issues)

- **Differential KV** ([turboquant-pro#22](https://github.com/ahb-sjsu/turboquant-pro/issues/22)) — delta-encoding adjacent tokens before PolarQuant
- **Attention-aware eviction** ([#23](https://github.com/ahb-sjsu/turboquant-pro/issues/23)) — drop low-attention tokens
- **FP8/NVFP4 native paths** ([#24](https://github.com/ahb-sjsu/turboquant-pro/issues/24)) — Hopper/Blackwell
- **Quantization-aware micro-LoRA** ([#25](https://github.com/ahb-sjsu/turboquant-pro/issues/25)) — for weight quantization (different problem)

## Roadmap

| Sprint | Status | Deliverable |
|--------|--------|-------------|
| 1 | **In progress** | Fork + design doc + scaffolding + issues |
| 2 | TODO | CPU reference implementation |
| 3 | TODO | CUDA kernels (Volta/Ampere) |
| 4 | TODO | Hot/cold tiering |
| 5 | TODO | Documentation + upstream PR |
