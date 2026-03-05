# UMA Bandwidth-Aware Layer Splitting

## Overview

On unified memory architecture (UMA) systems like AMD Strix Halo, CPU and GPU share the
same physical RAM but access it through different memory controllers with different effective
bandwidths. The GPU-routed memory path achieves ~2x the bandwidth of the CPU-routed path
(~180 GB/s vs ~90 GB/s on Strix Halo with LPDDR5X-8533).

This creates a unique optimization opportunity: when a model is too large to fully offload
to the GPU, we can be *smart* about which parts of each layer stay on GPU vs overflow to CPU.

## APEX-Inspired Approach

This implementation is inspired by the APEX paper (arXiv:2506.03296), which introduces
compute/bandwidth-aware scheduling for hybrid CPU-GPU LLM inference. While APEX targets
discrete GPU systems, we adapt its key insights for UMA:

### Key Insight

During autoregressive decode (token generation):
- **FFN (feed-forward network)** operations are **bandwidth-bound**: they involve large weight
  matrix multiplications with low arithmetic intensity (batch size 1)
- **Attention** operations are more **compute-bound**: flash attention over the KV cache has
  higher arithmetic intensity

### UMA Optimization

On UMA, the GPU has ~2x higher effective memory bandwidth to the same physical RAM. Therefore:

1. **Keep FFN on GPU**: bandwidth-bound operations benefit most from GPU's higher bandwidth
2. **Overflow attention to CPU**: compute-bound operations tolerate CPU execution better

This is the **reverse** of what APEX recommends for discrete GPUs (where the constraint is
VRAM capacity, not bandwidth to shared memory).

## Implementation

### LAYER_FRACTION_FFN (src/llama.cpp)

A new `layer_fraction_t` enum value `LAYER_FRACTION_FFN` enables partial layer offloading
where FFN weights stay on GPU and attention weights (Q, K, V, output projections) overflow
to CPU. This is automatically attempted on UMA systems when full layer offload doesn't fit.

The overflow pattern matches: `blk.N.attn_(q|k|v|output|norm|qkv|gate|q_norm|k_norm).*`

### UMA Auto-Detection in llama_params_fit_impl

When all GPU devices are `GGML_BACKEND_DEVICE_TYPE_IGPU`, the auto-fit logic:
1. First tries to fit full layers as usual
2. If some layers don't fit, attempts `LAYER_FRACTION_FFN` overflow (attention to CPU)
3. Falls back to `LAYER_FRACTION_ATTN` overflow (FFN to CPU) if FFN overflow doesn't fit

### UMA Bandwidth Profiler (common/uma-profiler.h)

A lightweight profiler that hooks into the backend scheduler's eval callback to measure
per-op execution times. It classifies operations using a roofline-model approach:

- **Arithmetic Intensity** (FLOPS/byte) determines if an op is bandwidth-bound or compute-bound
- Threshold: AI < 5.0 = bandwidth-bound, AI >= 5.0 = compute-bound
- Reports per-layer breakdown with time, bandwidth, and recommendations

The profiler is automatically enabled on UMA systems at verbosity >= 1 and runs for the
first 5 inference iterations, then prints a report.

## Usage

The bandwidth-aware splitting is automatic — no user flags needed. On UMA systems:

```bash
# auto-fit will use bandwidth-aware splitting
./llama-cli -m model.gguf

# see profiler output with verbose mode
./llama-cli -m model.gguf -v

# disable auto-fit if needed
./llama-cli -m model.gguf -fit off
```

## Expected Impact

For models that don't fully fit in GPU memory on UMA systems:
- **Token generation (decode)**: improved by keeping bandwidth-heavy FFN on GPU
- **Prompt processing**: less impact since batch size is larger, both CPU and GPU are busy
- **Overall**: more layers effectively on GPU, since attention weights are typically smaller
  than FFN weights (~25% vs ~75% of layer parameters in standard Llama models)

## References

- APEX: arXiv:2506.03296 — Performance-model-driven CPU-GPU scheduling for hybrid LLM inference
- Fiddler: arXiv:2402.07033 — Efficient MoE inference with limited GPU memory
- llama.cpp PR #16653 — Auto parameter fitting (memory-based)
