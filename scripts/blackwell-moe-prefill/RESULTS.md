# SM120 CUTLASS 4-bit results

This branch adds an optional CUTLASS block-scaled prefill path for GPT-OSS
MXFP4 and Qwen3.6-35B-A3B NVFP4. Measurements were taken on one RTX PRO 6000
Blackwell Server Edition. They are direct prefill measurements, not serving
throughput.

## Test setup

- GPU: RTX PRO 6000 Blackwell Server Edition, 96 GB, SM120
- llama.cpp baseline: `c745be2`
- Batch: 8192
- CPU threads: 25
- Flash Attention enabled
- Three timed repetitions for GPT-OSS
- Five timed repetitions for Qwen
- Native and CUTLASS modes ran in separate processes
- vLLM: `0.1.dev1+g045293d82`, PyTorch 2.12.1+cu130, FlashInfer 0.6.1

## GPT-OSS-120B MXFP4

The comparison uses the best measured ubatch for each implementation.

| Path | Best ubatch | pp8192 | Relative to llama.cpp | Share of vLLM |
| --- | ---: | ---: | ---: | ---: |
| Existing llama.cpp | 2048 | 11,738.58 tok/s | 1.00x | 31.9% |
| CUTLASS | 8192 | 25,713.80 tok/s | 2.19x | 69.8% |
| vLLM FlashInfer CUTLASS | 8192 | 36,819 tok/s | 3.14x | 100% |

The CUTLASS path reached 25,713.80 tok/s after replacing the initial support
kernels with prefix scheduling, one-token-per-CTA input quantization, and a
one-routed-row-per-CTA W13 activation kernel. The complete logits comparison
covered 201,088 values and was bitwise identical to the earlier CUTLASS path,
with NMSE 0 and max absolute error 0.

### Nsys breakdown

The corrected capture used one pp8192 warmup followed by one measured pass.
Times below are normalized to one 36-layer pass. The one-time weight transform
is excluded.

| MoE component | CUTLASS | vLLM CUTLASS |
| --- | ---: | ---: |
| Expert scheduling | 6.578 ms | about 0.924 ms |
| Input expansion and activation quantization | 9.526 ms | about 4.012 ms |
| W13 plus W2 | 127.863 ms | 105.041 ms |
| W13 activation and A2 quantization | 12.973 ms | 13.384 ms |
| W2 finalization | 8.785 ms | 5.856 ms |
| MoE steady total | 165.725 ms | about 129.217 ms |

The CUTLASS W13 and W2 kernels account for most of the remaining MoE gap. The
support kernels are close to the corresponding vLLM kernels, apart from launch
gaps around scheduling and input quantization. The first-use GPT expert-weight
transform took 378.205 ms in the trace and is not included in steady-state
throughput.

## Qwen3.6-35B-A3B NVFP4

This run used batch and ubatch 8192.

| Prompt | Existing llama.cpp | CUTLASS | Speedup | CUTLASS graphs |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 9,819.79 tok/s | 12,998.69 tok/s | 1.324x | 12,979.09 tok/s |
| 2048 | 11,362.81 tok/s | 15,309.67 tok/s | 1.347x | 15,160.20 tok/s |
| 8192 | 10,065.18 tok/s | 13,548.77 tok/s | 1.346x | 13,551.11 tok/s |

CUDA graphs do not materially change the result. The gain comes from the
expert path.

The pp8192 Nsys totals include warmup and measurement. First-use repack is
listed separately.

| Component | Existing llama.cpp | CUTLASS |
| --- | ---: | ---: |
| Profiled prefill | 9,977.8 tok/s | 13,377.0 tok/s |
| All steady CUDA kernels | 1,291.861 ms | 863.054 ms |
| MoE GEMM | 282.364 ms | 63.213 ms |
| Expert scheduling | 61.069 ms | 1.108 ms |
| Activation quantization | 45.941 ms | 27.656 ms |
| CUTLASS epilogues | - | 35.316 ms |
| Attention | 76.195 ms | 75.589 ms |
| First-use weight repack | 0 | 74.715 ms |

Backend correctness passed 6/6 eager cases and 6/6 CUDA Graph cases. A
full-model Qwen logits comparison is still pending.

## Decode result

A separate CUTLASS decode experiment was slower than the existing MMVQ path
and is not included in the retained implementation.

| Path | Latency for 128 tokens | Decode |
| --- | ---: | ---: |
| Existing MMVQ | 634.085 ms | 201.868 tok/s |
| CUTLASS experiment | 752.039 ms | 170.207 tok/s |

Nsys measured 5.882 ms/token for MMVQ and 6.634 ms/token for CUTLASS. At M=1,
CUTLASS cannot fill its tensor-core tiles or amortize activation quantization
and epilogue work. Single-token decode therefore remains on MMVQ.

## Retained implementation

The branch contains the optional CUTLASS prefill path and its direct
dependencies:

- SM120 MXFP4 and NVFP4 grouped W13/W2 kernels
- MXFP4 and NVFP4 weight repacking and cache lifetime management
- Shared expert scheduling, activation quantization, and CUDA epilogues
- GPT-OSS fused W13 conversion and loading
- GPT-OSS MXFP4 and Qwen NVFP4 graph matching
- Dense MXFP4 and NVFP4 GEMMs and a fused parallel SwiGLU FFN path
- Focused conversion and backend tests

The path is gated by `GGML_CUDA_CUTLASS`. Unsupported devices, shapes,
token counts, and graph layouts fall back to the existing CUDA implementation.
Single-token decode is not intercepted.

The dense path was added after the measurements above. Qwen3.6-27B NVFP4
correctness and performance still need to be measured on SM120; no dense result
is inferred from the MoE numbers.

## Remaining gap

For GPT-OSS, the retained path reaches 69.8% of the same-machine vLLM result.
The remaining MoE difference is mainly W13/W2 and launch gaps around the
support kernels. Attention, dense projections, normalization, casts, residual
operations, and graph launch boundaries account for the rest of the
end-to-end gap.
