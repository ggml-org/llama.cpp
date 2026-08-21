# llama.cpp — ROCmFPx + DFlash2 fork

> A personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp),
> tuned for AMD Strix Halo. It adds AMD-native FP4/FPx quantisation, DFlash2
> speculative decoding, and Vulkan kernel fixes that make the two work well
> *together*. Everything below the divider is the upstream README, unchanged.

## What is in here

| | Where it came from |
|---|---|
| **ROCmFPx quant types** — `Q4_0_ROCMFP4`, `_FAST`, `Q2/Q3/Q6/Q8_0_ROCMFPX`, with CPU codecs and Vulkan dequant / mat-vec / matmul / integer-dot kernels | Hand-ported from [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX). Two decode bugs in that source are fixed here — see [ROCMFPX-NOTES.md](ROCMFPX-NOTES.md) |
| **DFlash2 speculative decoding** | [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) by Jian Chen, not yet merged upstream. Carried unmodified, original authorship intact |
| **Vulkan batched mat-vec fixes** | New here. IQ3_S register spill at `NUM_COLS > 4` (5x at n=8), and the ROCmFPx batch 3-8 rework described below. Both are upstreamable |

## Why it exists

Speculative decoding verifies several tokens in one batch, so a decoder that is
fast at batch 1 and slow at batch 8 throws the speedup away. That is exactly what
the ROCmFPx kernels did: FP4 beat a K-quant of the same model at batch 1 and lost
badly at batch 8. Two shader changes fixed it — amortising the UE4M3 scale decode
over a whole 32-weight block, and replacing the per-weight bit-window gather in the
fp6/fp3 paths with a branch-free one.

Vulkan mat-vec, m=4096 k=14336, n=8, µs/run — lower is better:

| | before | after |
|---|---|---|
| `q4_0_rocmfp4` | 312 | **173** |
| `q6_0_rocmfpx` | 2236 | **402** |
| `q3_0_rocmfpx` | 1942 | **463** |

## Measured results

Qwen3.8-27B, Radeon 8060S (Strix Halo APU), RADV / Mesa 26.0.3, idle GPU.
ROCmFPX-MQ-Q4 is 14.63 GiB against 16.69 for the unsloth UD-Q4_K_XL it is compared
with. Generation, tokens/s:

| | ROCmFPx FP4 | unsloth UD-Q4_K_XL |
|---|---|---|
| batch 1 | **13.3** | 11.8 |
| batch 8 (`llama-batched-bench`) | **60.0** | 44.8 |
| DFlash2, prose | **24.6** | 22.3 |
| DFlash2, HTML | **40.7** | 32.9 |
| DFlash2, counting 0-100 | **53.0** | 35.8 |

The gap widens with how predictable the output is, because that is what raises
draft acceptance and fills the verification batch. Counting runs at 88% of the
model's measured batch-8 ceiling.

## What this does not claim

- **One machine.** Every number above is from a single Radeon 8060S. Nothing here
  has been tested on discrete AMD, NVIDIA, Intel, or Apple hardware.
- **FP4 is not free.** ROCmFPX-MQ-Q4 measures 5.9842 wikitext-2 perplexity against
  5.8965 for UD-Q4_K_XL — about 1.5% worse, with overlapping error bars.
- **The HTML and counting rows are single interactive runs**, not controlled
  benchmarks. The batch and prose rows are repeatable measurements.
- **Incomplete port.** No HIP/CUDA or OpenCL FPx kernels, no Vulkan kernels for
  `Q2_0_ROCMFPX`, no flash-attention with FPx KV cache.
- **Not upstream, and not a stable base.** Branches here are rebuilt against
  `ggml-org/master` by hand.

Correctness is checked with `test-backend-ops` against the CPU backend —
17920/17920 on Vulkan, and again with `GGML_VK_FORCE_MMVQ=1`.

## Try it

```sh
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
build/bin/llama cli \
  -hf  lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF:Q4 \
  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash --spec-draft-n-max 7 -ngl 99 -fa on
```

`--spec-draft-n-max 7` is the ROCmFPx-tuned value: it is free on prose here and
wins on structured output. K-quant models on this box still prefer 3 for prose.
[ROCMFPX-NOTES.md](ROCMFPX-NOTES.md) has the full measurements, the open work, and
the profiling traps.

---

*Everything below is the upstream llama.cpp README. Its badges and links point at
`ggml-org/llama.cpp`, not at this fork.*

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
