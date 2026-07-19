# RDNA2 multi-GPU benchmarks

This document records performance for the RDNA2 tensor-split changes on the
[`exp-gpu-sampling`](https://github.com/edwinbrowwn/llama.cpp-rdna2/tree/exp-gpu-sampling)
branch. See [`FIXES.md`](FIXES.md) for implementation details, build flags,
runtime flags, and known limitations.

## Test system

| Component | Configuration |
|---|---|
| GPUs | 4 × AMD Radeon Pro V620 32 GB (`gfx1030`) |
| GPU memory available | 30,704 MiB per device |
| CPU | AMD Ryzen Threadripper PRO 3945WX, 12 cores |
| Backend | ROCm/HIP with RCCL |
| ROCm path | `/opt/rocm/core-7.14` |
| Split mode | Tensor, equal `1/1/1/1` split |
| Flash attention | Enabled |
| KV cache | F16 K and V |
| Batch / microbatch | 2048 / 256 |
| Context | 262,144 tokens |

The tested models were:

- **Qwen3.6-35B-A3B MTP**, Unsloth Dynamic `Q4_K_M`, 35.5B total / 3B
  active parameters, 22.65 GB GGUF.
- **Qwen3.5-122B-A10B MTP**, Unsloth Dynamic `Q4_K_M`, 124.6B total / 10B
  active parameters, 78.25 GB GGUF split across three files.

## Build and runtime configuration

Build this branch with:

```bash
./scripts/build-rdna2-rocm.sh
```

Important build options used by the helper:

```text
-DGGML_HIP=ON
-DGGML_HIP_RCCL=ON
-DGGML_HIP_GRAPHS=ON
-DGGML_HIP_NO_VMM=ON
-DGGML_NATIVE=ON
-DAMDGPU_TARGETS=gfx1030
-DCMAKE_HIP_ARCHITECTURES=gfx1030
```

The benchmark environment was:

```bash
export GGML_CUDA_ALLREDUCE=nccl
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_NO_SCRATCH_RECLAIM=1
export GGML_HIP_GRAPHS=1
```

RCCL was confirmed active by the absence of the internal all-reduce and
meta-backend butterfly fallback warnings.

## Prompt-processing scaling

These tests used `llama-server`, `--parallel 1`, and one 262k context slot.
Context checkpoints and prompt caching were disabled:

```text
--ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

Each request generated one token. Prompt text consisted of a unique prefix and
a repeated string that tokenizes to exactly one token per repetition. The
reported prompt-token count includes the chat template. This isolates prompt
evaluation throughput and avoids measuring tokenization or output generation.

Each row is one fresh request. `Time` is calculated from llama.cpp's reported
prompt-token count and prompt tokens/second rather than HTTP wall time.

### Qwen3.6 35B-A3B

| Target | Actual prompt tokens | Time | Prompt processing |
|---:|---:|---:|---:|
| 8k | 8,181 | 5.159s | **1,585.65 t/s** |
| 64k | 65,526 | 47.236s | **1,387.21 t/s** |
| 128k | 131,063 | 130.914s | **1,001.14 t/s** |

### Qwen3.5 122B-A10B

| Target | Actual prompt tokens | Time | Prompt processing |
|---:|---:|---:|---:|
| 8k | 8,181 | 7.897s | **1,036.01 t/s** |
| 64k | 65,526 | 91.422s | **716.74 t/s** |
| 128k | 131,063 | 543.223s | **241.27 t/s** |

Both models completed all three context lengths without a GPU fault.

The sharp 122B decline at 128k is reproducible for this configuration and is
likely dominated by long-context attention/KV traffic rather than model-weight
bandwidth alone. The repeated synthetic prompt is suitable for measuring
compute throughput but is not representative of prompt quality or cache reuse.

## Standard `llama-bench`

The standard benchmark used pp512 and tg128, three measured repetitions,
four-GPU RCCL tensor split, flash attention, full GPU offload, F16 KV, batch
2048, and microbatch 256.

```bash
GGML_CUDA_ALLREDUCE=nccl \
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_GRAPHS=1 \
./build/bin/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 \
  -sm tensor \
  -ts 1/1/1/1 \
  -fa on \
  -b 2048 \
  -ub 256 \
  -p 512 \
  -n 128 \
  -r 3
```

`llama-bench` requires `/` between tensor-split values. A comma-separated value
is interpreted as a parameter sweep rather than one four-GPU split.

| Model | pp512 | tg128 raw decode |
|---|---:|---:|
| Qwen3.6 35B-A3B Q4_K_M | **1,607.43 ± 18.64 t/s** | **63.17 ± 4.11 t/s** |
| Qwen3.5 122B-A10B Q4_K_M | **955.39 ± 8.65 t/s** | **40.69 ± 1.92 t/s** |

The standard tg128 result does **not** use MTP speculative decoding. End-to-end
`llama-server` generation with MTP is faster when draft acceptance is healthy.

## Content-sensitive MTP generation

The following short server tests generated exactly 1,024 tokens at temperature
0.6 with MTP enabled. They use real code and prose prompts rather than the
synthetic `llama-bench` workload.

| Model | Prompt | Decode | Draft acceptance | Wall time |
|---|---|---:|---:|---:|
| Qwen3.6 35B-A3B | Python implementation and tests | **76.42 t/s** | 65.86% | 14.39s |
| Qwen3.6 35B-A3B | Historical prose essay | **69.67 t/s** | 55.70% | 14.82s |
| Qwen3.5 122B-A10B | Python implementation and tests | **54.61 t/s** | 60.77% | 19.77s |
| Qwen3.5 122B-A10B | Historical prose essay | **57.75 t/s** | 68.36% | 17.92s |

MTP throughput follows draft acceptance more closely than content category. In
these samples the 35B model accepted more coding drafts, while the 122B model
accepted more prose drafts.

## RCCL impact

A separate deterministic 122B comparison measured:

| Four-GPU all-reduce | Decode |
|---|---:|
| Meta-backend butterfly | 50.72 t/s |
| RCCL | **55.61 t/s** |

RCCL improved 122B decode by approximately **9.6%**. An 8.8k prompt improved
from approximately 615 t/s to 750 t/s in the same comparison.

## Notes and caveats

- Results are specific to four V620 cards, this PCIe topology, ROCm version,
  quantization, model build, and branch.
- Prompt processing used one slot to make 128k requests possible. With
  `--parallel 4`, non-unified contexts may divide the configured context among
  four slots.
- Standard `llama-bench` reports raw decode and does not exercise MTP.
- Server MTP results depend on prompt content, temperature, and draft
  acceptance.
- Prompt caching was disabled for scaling tests. Production multi-turn behavior
  can differ substantially when prefixes are reused.
- After any ROCm illegal-memory fault, reset the GPUs or reboot before trusting
  later measurements.