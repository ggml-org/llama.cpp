# llama.cpp + TurboQuant+

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) carrying TurboQuant KV-cache and weight quantization support across CUDA / HIP-ROCm / Metal / Vulkan backends.

The codec itself (algorithm, calibration, papers) lives at **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)**. This repo is the llama.cpp integration of that codec.

> Status: work-in-progress. Not merged upstream. Default branch is `feature/turboquant-kv-cache`; ~300 commits ahead of `master`.

---

## What this fork adds (vs upstream ggml-org/llama.cpp)

### New quantization types

- **TQ3_1S** / **TQ4_1S** weight quantization formats (added to `GGMLQuantizationType` + `LlamaFileType` enums) — smaller VRAM than `q8_0` at the 3.5-bit / 4.5-bit tier.
- **turbo3** / **turbo4** KV-cache formats — Walsh-Hadamard rotation + polar codebook, ~4.6× compression at <1.5% PPL loss.

### Backend kernels

- **Metal**: TurboFlash attention kernel for `turbo3` KV-cache decode, `dk=512` FA kernel instances for Gemma 4, TQ weight ops, concurrency handling.
- **CUDA**: `dp4a` kernel for `TQ4_1S` (3.5× faster, 240 t/s vs 68 baseline), warp-cooperative TQ dequant (16× less compute per block), multi-token + multi-GPU kernels, load-time `TQ4_1S → q8_0` conversion path.
- **HIP / ROCm**: AMD RDNA3 (gfx1100), RDNA4, CDNA3 (MI300X/gfx942), CDNA4 (MI355X/gfx950). `ggml_cuda_dp4a` replaces `__dp4a` for portability; scalar half path for `TQ4_1S` on AMD; pool bypass for FA f16 temp buffers; VEC FA path forced for quantized KV.
- **Vulkan**: compute-shader `turbo3` KV cache + coopmat flash attention.

### Flash Attention integration

- `TURBO2_0` added to the FA auto-enable check.
- Mixed `f16/bf16 + q8_0` KV pairs work without requiring `GGML_CUDA_FA_ALL_QUANTS`.
- CPU `vec_dot` heap-allocation fix for turbo / TQ types at `n > 4096`.

### Model-family compatibility

- 256-expert MoE kernel instantiations (large MoE shape support).
- Gemma 4 specifics: `dk512` Metal FA kernels, op concurrency, MoE token routing.
- RPC `GGML_OP_COUNT` assertion fix.

### Memory + stability

- Apple Silicon memory-explosion fix.
- Bench-suite for CUDA dp4a + Metal TurboFlash + Vulkan coopmat.

---

## Build

Standard llama.cpp build flags. TurboQuant types are available automatically once the relevant backend is compiled in.

```bash
# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON && cmake --build build -j

# CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# HIP / ROCm
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx942;gfx950" && cmake --build build -j
```

## Use

`turbo3` / `turbo4` KV-cache quantization is selected at runtime via the existing `--cache-type-k` / `--cache-type-v` flags. `TQ3_1S` / `TQ4_1S` weight quantization is selected at conversion time by passing the type to `llama-quantize`.

See the upstream llama.cpp documentation for everything else. Where this fork diverges, it diverges additively — existing `q4_K`, `q8_0`, `f16` paths are untouched.

## References

- Codec design + calibration + papers: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- Cross-backend bench results: [TheTom/tqkit](https://github.com/TheTom/tqkit)
- Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

## License

MIT, same as upstream llama.cpp.
