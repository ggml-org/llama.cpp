# Tessera Calibration Pipeline Design

This document describes the runtime-aware calibration pipeline. The
motivation: the offline `_ternary_reconstruct` reference in
`tools/tessera/awq-evolve.py` does not necessarily match the C++ Tile640
matmul kernel's actual dequant, so calibrating against the reference can
optimize the wrong thing. The pipeline below ties the calibration back to
the runtime.

## Data flow

```
BF16 source
   │
   │  tools/tessera/awq-evolve.py
   │    + tools/tessera/per_tensor_calibrate.py
   │  offline: small GA per tensor over
   │    (ternary_threshold, outlier_fraction, awq_alpha, awq_clip)
   ▼
Tessera-T640 GGUF  (ternary + outliers + scales)
   │
   │  llama.cpp runtime:
   │    Tile640 matmul kernel (C++)
   │    INTERLEAVED with the FMA:
   │      output = (dequant(ternary, outliers) · scale_awq) @ x
   ▼
logits
```

The offline round-trip is cheap but it's not the runtime. The runtime
is the kernel, and the kernel's dequant may differ from the Python
reference in F16 precision, order of operations, and outlier folding.
Until we measure the kernel's actual dequant, calibration is flying
blind.

## Layer 1 — Kernel dequant fidelity (the ground truth)

Add a `LLAMA_TILE640_DEBUG_DEQUANT=1` mode to the Tile640 matmul kernel.
When set, the kernel emits the **effective dequantized weight per row**
to a sidecar buffer (F32, contiguous). For a calibration pass with a
known input, the kernel can also emit `(dequant(W)·x) - (W_bf16·x)` per
row — the **runtime task error** the drafter sees.

This is the only measurement that tells us what the runtime is actually
doing, not what the offline reference thinks.

## Layer 2 — Forward-pass differential

Run two forward passes on the calibration corpus: one with the BF16
source, one with the quantized model. Capture per-tensor matmul outputs
(the kernel's actual output) at every tensor, every layer, every
position. Compute per-tensor divergence: max|Δ|, relative Frobenius,
top-1/top-5 mismatch of the post-matmul distribution. Catches
position-specific errors that per-layer sum differences miss.

## Layer 3 — Per-token coherence

Capture per-token logits at every position for both BF16 and quantized.
Compute per-token KL divergence, top-1 mismatch rate. Aggregate: which
positions in the prompt are most degraded?

## Layer 4 — End-to-end coherence on a probe set

Standard probe prompts (the ones we already use: "The capital of France
is", "What is 2+2?", etc.). Generate 30-50 tokens at temp=0 from both
BF16 and quantized. Measure: exact-match of first-50 tokens, perplexity
delta, logit rank correlation per position. This is what the user
actually sees.

## Layer 5 — Adaptive requantization loop

For tensors with divergence > threshold (say 5% relative Frobenius),
try a small set of parameter variations:

- lower threshold (denser, more -1/+1)
- higher outlier_frac (more exact residuals)
- higher awq_alpha (more per-channel rescaling)

Re-run the kernel dequant fidelity check. Pick the variant with lowest
divergence. This is *runtime-aware* requantization, not just
offline-fitness-aware.

## Layer 6 — Telemetry → re-calibration feedback

The kernel-dequant fidelity becomes the actual fitness function for the
per-tensor GA (not `_ternary_reconstruct`). The GA's mutation space is:
per-tensor `(ternary_threshold, outlier_fraction, awq_alpha, awq_clip)`.
Each candidate is evaluated by running the kernel in debug mode and
measuring the actual dequant error against the BF16 source.

This is slow (each candidate needs a kernel call) but it's the only
fitness that matches the runtime. Once calibrated, the per-tensor
policy is shipped with the GGUF and the runtime uses it directly.

## Code layout

```
tessera/
├── tools/
│   ├── tessera/
│   │   ├── awq_evolve.py            # offline GA fitness (Layer 1-5 start)
│   │   └── per_tensor_calibrate.py  # per-tensor GA
│   ├── imatrix/                     # llama-imatrix (Layer 2 capture)
│   └── ...
├── python/
│   ├── run_kernel_fidelity.py       # Layer 1
│   ├── run_differential_forward.py  # Layer 2
│   ├── run_per_token_coherence.py   # Layer 3
│   ├── run_e2e_probe.py             # Layer 4
│   ├── adaptive_requantize.py       # Layer 5
│   └── fitness_kernel.py            # Layer 6
└── cpp/
    └── patches/                     # Tile640 kernel debug mode patches
```

## Order of work

1. **Layer 1 first** — without the kernel debug mode, the rest of the
   pipeline is optimizing the wrong fitness. The C++ work is the
   bottleneck.
2. **Layers 2-4 next** — Python orchestration around the existing
   `llama-imatrix` capture. Each is small.
3. **Layer 5 next** — adaptive requantization uses Layer 1's output to
   decide which tensors to re-tune.
4. **Layer 6 last** — refit the per-tensor GA to use the kernel
   fitness. Slow, but ships the runtime-aware policy.

## What this is NOT

This pipeline does not fix the drafter alignment problem. The drafter
is still trained against a different distribution. The pipeline's job
is to make sure the verifier's behavior matches BF16 closely enough that
the drafter can predict it. Drafter alignment (LoRA, distillation,
rejection sampling) is a separate concern — see
`docs/dflash-dspark.md` for the drafter-side work.
