# KV cache quantization with TurboQuant

This fork provides block-128 TurboQuant KV cache formats. The sizes below
include the stored scales; compression compares the same number of values
against f16, before head padding and cache metadata.

| Runtime name | Type | Enum | Bytes / 128 values | Bits / value | Compression |
| --- | --- | ---: | ---: | ---: | ---: |
| `turbo2` | `GGML_TYPE_TURBO2_0` | 43 | 34 | 2.125 | 7.53x |
| `turbo3` | `GGML_TYPE_TURBO3_0` | 44 | 50 | 3.125 | 5.12x |
| `turbo4` | `GGML_TYPE_TURBO4_0` | 45 | 68 | 4.25 | 3.76x |

The layouts in `ggml/src/ggml-common.h` are authoritative. Turbo4 retains the
fork's `rnorm` field. These formats are intended for runtime KV caches;
model-weight formats `TQ3_1S` and `TQ4_1S` use enums 46 and 47, respectively,
and 32-value blocks of 16 and 20 bytes. TheTom's enum assignments and Turbo4
layout differ; GGUF files are not interchangeable merely because names match.

## Usage

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0
llama-cli -m model.gguf -c 8192 -ngl 99 -fa on \
    --cache-type-k q8_0 --cache-type-v turbo3
```

The flags also apply to `llama-server`, `llama-bench`, and `llama-perplexity`.
Use a backend and head shape that support the requested operations. Flash
attention is recommended. With flash attention disabled, the fork permits
TurboQuant through MUL_MAT attention and dequantizes turbo V to F32 at
attention time. Other quantized V formats still require flash attention.

## Model-specific quality

Models with attention sinks may be more sensitive to K-cache quantization
than the models this fork measures. The TurboQuant source tree reports
GPT-OSS as such a case: `q8_0` K is said to shift the output distribution,
and lower-bit K types more so, while codec and kernel accuracy stay normal.
That report came without numbers and this repository has none either: no
GPT-OSS or other sink-heavy model appears in the `docs/research/` results,
for any K or V cache type. Read it as a caution, not as a measured result.

Until such a model is measured, start it from `f16` K and compare every
quantized K or V cache against an `f16` K/V baseline before deploying it.
Short output samples are not a sufficient check, because text can stay fluent
while token probabilities move. Use `llama-perplexity --kl-divergence` or an
equivalent logit comparison when selecting cache types.

## Rotation and policy

The cache-write path applies a fixed Walsh-Hadamard transform before centroid
quantization. The graph rotates Q for attention and inverse-rotates the V
result. Heads are padded to a multiple of 128 where needed. MLA stores V as a
view of latent K and skips separate V rotation and padding.

| Variable | Default | Effect |
| --- | --- | --- |
| `TURBO_LAYER_ADAPTIVE` | unset | Layer precision policy, modes 1, 2, 5, 6 or 7 (other values are ignored with a warning); mode 7 keeps q8_0 V on the first and last two layers. Unset means mode 7 for turbo2 V on models with at least 8 layers, otherwise off; `0` turns it off |
| `TURBO_AUTO_ASYMMETRIC` | on | When K and V request the same turbo type, store K as q8_0 if the GQA ratio is at least 6 or the architecture is Qwen-family; MLA and DeepSeek4 models are excluded; `0` turns it off |
| `LLAMA_ATTN_ROT_K_OVERRIDE` | off | Opt into the separate upstream K rotation path |
| `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Opt into the separate upstream V rotation path |
| `LLAMA_ATTN_ROT_DISABLE` | `0` | Disable both upstream rotation overrides |

The upstream rotation overrides are separate from TurboQuant's required WHT.
The fork retains CPU, BLAS, SYCL, Vulkan, and OpenVINO backends. Operation
coverage varies; the presence of a backend does not imply native TurboQuant
kernels for every operation. CUDA, HIP, and Metal are not built in this fork.

## Weight quantization and validation

```sh
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

Compression ratios are layout facts, not speed or model-quality results.
Use the synthetic correctness tests and model-level PPL/KLD probes described
in [quality-benchmarks.md](quality-benchmarks.md) before drawing conclusions
about a model, cache configuration, or backend.
