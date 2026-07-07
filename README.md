# llama.cpp-hy3

This is a fork of `ggml-org/llama.cpp` with Hy3 support applied.

Hy3 support was originally authored by Yaniss:

```text
Yaniss <yaniss@getasolutions.fr>
```

## What Changed

Compared to upstream `llama.cpp`, this fork adds support for Tencent Hy3 / `HYV3ForCausalLM` models.

The patch adds:

- A new `hy-v3` model architecture.
- Runtime graph support for Hy3's 295B-A21B MoE layout.
- Sigmoid-gated expert routing with expert bias and scale handling.
- Per-head Q/K normalization before NEOX RoPE.
- Support for the leading dense layer and shared expert layout used by Hy3.
- Handling for the extra NextN/MTP layer by loading it but skipping it in the forward graph.
- HF to GGUF conversion support for Hy3 checkpoints.
- FP8 `weight_scale` dequantization support during conversion.
- Conversion fixes for the stock Hy3 chat template so it can be embedded in GGUF and parsed by llama.cpp's Jinja engine.

## Why This Is A Fork

The original Hy3 pull request was removed from upstream review. This repository is kept as a fork so Hy3 users can build and test the patch without consuming upstream maintainer review time or presenting it as an accepted llama.cpp feature.

Use this as an experimental Hy3 branch. For general llama.cpp usage, use upstream:

```text
https://github.com/ggml-org/llama.cpp
```

## Build

Use the normal llama.cpp build flow. For example, with CUDA:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j
```

## Notes

Hy3 chat behavior may still need model-specific stop strings or metadata overrides, depending on the converted GGUF. In testing, the Hy3 EOS token should be treated as a stop string:

```text
<｜hy_eos:opensource｜>
```
