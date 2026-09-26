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

## Rotation and policy

The cache-write path applies a fixed Walsh-Hadamard transform before centroid
quantization. The graph rotates Q for attention and inverse-rotates the V
result. Heads are padded to a multiple of 128 where needed. MLA stores V as a
view of latent K and skips separate V rotation and padding.

| Variable | Default | Effect |
| --- | --- | --- |
| `TURBO_LAYER_ADAPTIVE` | `0` | Layer precision policy; mode 7 uses q8_0 V at boundary layers |
| `TURBO_AUTO_ASYMMETRIC` | `1` | Downgrade symmetric turbo K to q8_0 for GQA >= 6, excluding MLA |
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
