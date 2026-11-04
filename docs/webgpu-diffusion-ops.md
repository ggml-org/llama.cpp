# WebGPU Backend: Op Requirements for `llama-diffusion-cli`

Analysis of which GGML operations are needed for the WebGPU backend to support
`llama-diffusion-cli`, cross-referenced against `docs/ops/WebGPU.csv`.

## Supported architectures

`llama-diffusion-cli` supports four model architectures, all of which use
bidirectional (non-causal) attention and no KV cache:

| Architecture | Base | RoPE type | FFN |
|---|---|---|---|
| `dream` | Qwen2 | NeoX | SiLU gated (dense) |
| `llada` | LLaMA | Normal | SiLU gated (dense) |
| `llada-moe` | LLaMA | Normal | SiLU gated (MoE) |
| `rnd1` | Qwen3-MoE | NeoX | SiLU gated (MoE) + QK norm |

All four set `hparams.causal_attn = false` and call `build_attn_inp_no_cache()`,
meaning the full sequence is processed as one batch with no KV cache.

## Core ops (all architectures)

| Op | Purpose | WebGPU support |
|---|---|---|
| `GET_ROWS` | Token embedding lookup | ✅ f16, q4\_0, q8\_0 · ❌ bf16, q1\_0 |
| `MUL_MAT` | Q/K/V/O projections, FFN layers, lm\_head | ✅ {f16, q4\_0, q4\_K, q8\_0, …} × f32 · ❌ bf16 weights · ❌ any × f16 activations |
| `RMS_NORM` | Layer normalization | ✅ fully supported |
| `ROPE` | Rotary position embeddings | ✅ fully supported |
| `ADD` | Residual connections | ✅ fully supported |
| `GLU` / `MUL` | SiLU-gated FFN activation | ✅ fully supported |
| `FLASH_ATTN_EXT` | Full-sequence attention (no KV cache) | ✅ f16 KV · ❌ bf16 / q4\_1 / q5\_0 / q5\_1 / iq4\_nl KV |
| `SOFT_MAX` | Decomposed attention fallback | ✅ fully supported |

### Note on `FLASH_ATTN_EXT` and the no-cache path

Because diffusion models have no KV cache, the K and V tensors are runtime f32
activations. Before calling `ggml_flash_attn_ext`, the backend unconditionally
casts them to f16 (`src/llama-graph.cpp:1969–1975`):

```cpp
if (k->type == GGML_TYPE_F32) {
    k = ggml_cast(ctx0, k, GGML_TYPE_F16);
}
if (v->type == GGML_TYPE_F32) {
    v = ggml_cast(ctx0, v, GGML_TYPE_F16);
}
```

So `type_KV=f16` is always the actual runtime case for diffusion, regardless of
model quantization. `FLASH_ATTN_EXT` with `hsk=128` (Dream-7B, LLaDA-8B) and
`type_KV=f16` is confirmed supported in the CSV.

The WebGPU shader-lib VEC path (`ggml-webgpu-shader-lib.hpp:732`) accepts
`F16 | Q4_0 | Q8_0` KV types; the TILE path requires `F16`. Both paths cover
the diffusion use case.

## Additional ops for MoE architectures (`llada-moe`, `rnd1`)

| Op | Purpose | WebGPU support |
|---|---|---|
| `MUL_MAT_ID` | Sparse expert matmul | ✅ f16, q4\_0, q8\_0 experts · ❌ q1\_0, mxfp4, nvfp4, iq-family, bf16 |
| `SOFT_MAX` | MoE gate normalization | ✅ fully supported |
| `ARGSORT` / `TOP_K` | Expert selection | ✅ fully supported |

## `DIAG_MASK_INF` — not a blocker

`DIAG_MASK_INF` is unsupported in WebGPU but is only inserted for causal
attention. All diffusion architectures use non-causal attention, so it never
appears in their compute graphs.

## What is already working

For models loaded as **f16, q4\_0, q4\_K, q8\_0** (standard GGUF files), every
op on the critical inference path is already present in the WebGPU backend.
`llama-diffusion-cli` should run end-to-end today on those quantizations.

## Missing kernels

The gaps affect **bf16 models** and a small set of unusual quantizations:

| Missing kernel | Impact |
|---|---|
| `MUL_MAT(bf16, f32)` | Blocks all weight projections in bf16-loaded models |
| `GET_ROWS(bf16)` | Blocks token embedding lookup in bf16 models |
| `MUL_MAT_ID` for bf16 / iq-family experts | Blocks `rnd1` and `llada-moe` when experts are in those types |
| `FLASH_ATTN_EXT` for bf16 / q4\_1 / q5\_x KV | Low priority for diffusion (K/V are always cast to f16 at runtime) |

In practice, adding **bf16 support to `MUL_MAT`, `GET_ROWS`, and `MUL_MAT_ID`**
is the only work needed to achieve full coverage for all diffusion model variants.
