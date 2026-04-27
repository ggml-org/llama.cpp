# TurboQuant-ROCm-RDNA4 Binary Release

Pre-compiled binaries for llama.cpp with TurboQuant enabled.

## Contents
- llama-cli.exe, llama-bench.exe, llama-perplexity.exe
- ggml-hip.dll (Optimized for gfx1201)
- llama.dll, ggml.dll

## Usage
Enable 4-bit KV cache with:
`--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on`
