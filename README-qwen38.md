# Qwen3.8-27B (IQ3_S) specialized llama.cpp build

Private fork specialized to serve **one** model on **one** machine:

- Model: Unsloth `Qwen3.8-27B-UD-IQ3_S.gguf` (architecture `qwen35`, hybrid DeltaNet + full attention).
- Hardware: Windows 10 x64, Ryzen 9 5900X, RTX 3090 (24 GiB), CUDA 12.5, MSVC 19.44.
- Serving: text-only, OpenAI-compatible chat/completions, streaming, tool calling, structured output.

This is NOT upstream-compatible. It deletes unused model and backend code and refuses non-qwen35
models and multimodal input.

## Runtime profile (tuned and validated)

```
--ctx-size 262144 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
--n-gpu-layers 99 --fit off --batch-size 2048 --ubatch-size 512 --threads 8
--no-context-shift --jinja --no-webui
```

Full GPU offload fits with ~2.2 GiB VRAM free. Do not lower context, cache precision, or
Flash Attention; they are required and were validated together.

## Build

Requires VS 2022 Build Tools (MSVC 14.44), CUDA 12.5, and the bundled CMake/Ninja.

Configure + build the production server:

```
cmake --preset qwen38-prod
cmake --build build-qwen38-prod --target llama-server
```

Build flags of note (see `CMakeUserPresets.json`):
- `CMAKE_CUDA_ARCHITECTURES=86-real` (RTX 3090 machine code only).
- `GGML_AVX2`/`GGML_BMI2` on, AVX-512 off (Zen 3), `GGML_NATIVE=OFF` for reproducibility.
- `GGML_CUDA_FA_QUANTS=q8_0-q8_0` (only the Q8_0 Flash Attention kernels, plus f16-f16).
- `GGML_CUDA_GRAPHS=ON`, `BUILD_SHARED_LIBS=OFF` (static libs), UI/OpenSSL/subprocess/LLGuidance off.
- `LLAMA_SERVER_TEXT_ONLY=ON` (refuses `--mmproj`).

The `qwen38-baseline` preset additionally builds tests and `llama-bench` for validation.

## Run

```
pwsh scripts/serve-qwen38.ps1
```

The launcher checks the model path, puts the CUDA 12.5 bin on PATH, binds to 127.0.0.1:8080,
and applies the tuned profile. Override with `-ModelPath`, `-Port`, etc.

## Runtime dependencies

`llama-server.exe` links the project libraries statically but needs these DLLs at run time
(all already present on this machine):

- CUDA 12.5: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` (CUDA bin on PATH).
- Driver: `nvcuda.dll` (system).
- MSVC runtime: `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, `VCOMP140.dll` (OpenMP).

## What was pruned

- `src/models/`: kept only `qwen35.cpp` and `delta-net-base.cpp`; deleted 155 other model sources.
  The model factory serves only `qwen35` and throws for any other architecture.
- `ggml/src/`: kept only `ggml-cpu` and `ggml-cuda`; deleted 16 other backend directories.
- `tools/mtmd/models/`: kept only `models.h`; deleted 45 vision/audio graph sources and narrowed the
  clip graph factory. mtmd still links (its types back the server), but no projector can load.

## Behavior

- Non-qwen35 GGUF: rejected ("unsupported model architecture").
- `--mmproj`: rejected ("text-only server build: --mmproj is not supported").
- Image/audio/video in a request: rejected ("image input is not supported ...").
- Prompt over 262144 tokens: rejected with HTTP 400, not truncated.

## Measured results (RTX 3090)

- Load (full ctx, full offload): ~22.1 GiB VRAM used, ~2.2 GiB free.
- Prompt processing (depth 0): ~1144 tok/s (ubatch 512).
- Generation: ~38 tok/s at low depth, ~13 tok/s near 262k depth.
- Near-limit: 261900-token prompt prefilled and generated with no OOM, no truncation.

See `.copilot` session artifacts (`baseline-results.md`, `validation-results.md`, `tuning-results.md`)
for the full run details.
