# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Patt92 ROCm / Strix Halo fork — V6

This branch is based on upstream llama.cpp commit
[`4197155addb6989875b26e81dda73e9290fe4cad`](https://github.com/ggml-org/llama.cpp/commit/4197155addb6989875b26e81dda73e9290fe4cad).
It is a self-contained cumulative ROCm/HIP optimization branch for AMD Strix Halo
(`gfx1151` / RDNA3.5): it does not depend on the continued existence of any earlier
optimization branch. It retains normal upstream functionality, but is not intended to
replace the portable default upstream build.

### Included changes relative to upstream

#### HIP, RPC and long-context routing

- Enables hipCUB for HIP top-k and argsort, including the distributed RPC execution
  path, so these operations remain GPU-capable on ROCm.
- Adds a gfx1151-local Lightning Indexer top-k specialization for 512, 1024 and 2048
  candidates up to 8192 score entries, avoiding the expensive generic device-wide sort
  during DeepSeek-V4 long-context prefill.
- Keeps the Lightning Indexer key cache in F16 when a quantized KV cache is selected,
  where required by the DeepSeek path.
- Resolves fused operations per layer and device instead of disabling an entire graph
  on a single ROCm/RPC backend mismatch; the default fallback remains unfused GPU work,
  not CPU work.
- Fixes the HIP MMVF dispatch for MoE LoRA `MUL_MAT_ID` tensors. Rank-1 LoRA-B
  expert matrices now use the existing general GPU fallback instead of entering an
  MMVF kernel which requires an even K dimension; this prevents RPC worker crashes
  with valid DeepSeek-V4 expert LoRA adapters.

#### Strix Halo compute tuning

- Adds RDNA3.5/gfx1151 kernel and launch tuning for MMQ, MMVQ, Q8 MoE and expert MMQ.
- Caches MMVQ Q8_1 activations and partitions MMQ waves across rows and columns.
- Adds AMD WMMA Lightning Indexer support and tuned tiled Flash Attention, including a
  fix for NaNs in the compacted-tile mask path.
- Tunes Gated Delta Net and adds quantized-KV Flash Attention support for Strix Halo.
- Extends the gfx1151 MMVQ selection to the relevant Q4/Q5/Q6/Q8 and MXFP4 decode
  types, including the Q6_K VDR=2 decode kernel.
- Keeps HIP integrated-GPU host-buffer handling safe.

#### DeepSeek-V4 and speculative decoding

- Makes the grouped output-projection input contiguous for small multi-token
  DeepSeek-V4 speculative batches.
- Uses matching MMVQ and Flash-Attention kernel configurations for decode and small
  speculative/MTP verification batches, preventing numerical divergence between the
  logits used for acceptance checks.
- Keeps fused DeepSeek-V4 HC and Gated Delta Net operations on devices that support
  them while safely falling back only for mismatching layers.

#### Qwen3.5 / Qwen3.8 and hybrid SSM models

- Fuses SSM gate/beta projections, the SSM convolution-output L2 norm, and the SSM
  pre-scan chain (convolution, L2 norm, gate/beta).
- Folds SSM convolution-input concatenation into QKV MMVQ and fuses paired MMVQ
  matmuls that share an activation.
- Caches graph-local Q8_1 matmul inputs, folds Q8_1 quantization into RMS norm and
  gating-MUL, and fuses a matmul-plus-add-through-view sequence.
- Folds MoE top-k weights into the down projection and fuses the shared-expert output
  chain.
- Fuses IMRoPE and set-rows for BF16 KV cache use, and aligns the attention-gate
  tensor-parallel split with attention-Q.

### Deliberate exclusions

- Native-BF16 Flash-Attention tile changes from an external ROCm research series are
  not included. They overlap the tested gfx1151 tiled-FA path in this branch and need
  their own ROCm correctness and performance benchmark before replacement.
- The external adaptive-MTP depth experiment is not included. It is backend-neutral,
  but needs a separate correctness and throughput test with the RPC memory layout
  before it can be considered for this branch.
- Vulkan shaders, Vulkan resource management and Vulkan-only environment flags are not
  copied into this HIP branch. Algorithmic ideas from them may be ported separately,
  with a dedicated HIP implementation and benchmark.

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
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
