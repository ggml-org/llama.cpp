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

## AMD AI PRO Optimisation and eGPU Compatibility

This fork adds AMD RDNA4 (gfx1201) support and eGPU (Thunderbolt) compatibility to the Vulkan backend. Tested on a Radeon AI PRO R9700 (Navi 48) eGPU with the AMD proprietary Windows driver.

### Changes vs upstream

- **RDNA4 / gfx1201 detection**: the AMD proprietary Windows driver does not expose `VK_NV_cooperative_matrix2`, so upstream mis-detects RDNA4 as RDNA3. This fork detects gfx1201/gfx1200 by PCI device ID (0x755x / 0x759x) and applies the correct pipeline config (subgroup size 32).
- **WMMA (matrix cores) on gfx12**: enables the `VK_KHR_cooperative_matrix` path on RDNA4. The coopmat shaders (`OpCooperativeMatrixLoad/MulAdd/StoreKHR`, 16x16x16 f16/f32/int8/fp8/bf16) lower to `v_wmma_*` instructions on gfx12. Verify at startup: the log line `matrix cores: KHR_coopmat`.
- **eGPU / Thunderbolt memory fix**: the AMD proprietary driver mis-reports buffer `memoryTypeBits` as host-memory-only on eGPUs, which makes every device-local buffer allocation fail with `ErrorOutOfDeviceMemory` despite free VRAM. This fork probes for the broken condition at device init and then trusts the memory type property flags for device-local requests. Force with `GGML_VK_IGNORE_BUFFER_MEMORY_TYPE_BITS=1`.

### Build (Windows)

```sh
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j 8
```

Requires the Vulkan SDK and MSVC. Select the eGPU with `GGML_VK_VISIBLE_DEVICES=<idx>` (see the device list printed at startup).

### Server performance tuning (Qwen3.6-35B-A3B Q4_K_S, R9700 eGPU)

Recommended command:

```sh
llama-server -m model.gguf -ngl 99 -c 65536 --parallel 1
```

Measured decode speed (256-512 generated tokens, warm):

| config | decode t/s |
|---|---|
| `--parallel 4` (default) | ~15 |
| `--parallel 1` | ~80-85 |
| `--parallel 1` + `-fa on` | ~44 |
| `--parallel 1` + `--spec-type draft-mtp` | ~14 (43% acceptance) |

Key findings:

- Use `--parallel 1` for maximum single-stream speed. With 2+ slots on this hybrid (deltanet + attention) model, per-token decode drops ~5x even when only one slot is active.
- MTP speculative decoding (`--spec-type draft-mtp`) is **model-dependent** on this eGPU:
  - **Dense models (e.g. Qwen3.6-27B Q4_K_S): use it, it is a 11-12x win.** 3.8 t/s baseline -> 42-48 t/s (62-70% acceptance). Verification of ~3 drafted tokens in one batch reads the weights once, so a dense bandwidth/latency-bound decode becomes a batched one.
  - **MoE models (e.g. Qwen3.6-35B-A3B): avoid it.** Only ~3B active params/token, so the base decode is already cheap (85 t/s); the draft-head overhead is a net 5x loss (~15 t/s).
- `-fa on` helps prompt processing slightly but hurts single-token decode; leave it on `auto`.

### Roadmap (further speed work)

- Investigate why multi-slot decode is slow on this hybrid model (recurrent state cache copies?)
- KV cache quantization (`-ctk q8_0 -ctv q8_0`), and Q4_0/IQ4_XS quants
- Coopmat is already active in the MoE `mul_mat_id` path for batches (prompt processing, verified via SPIR-V + `matrix cores: KHR_coopmat`); n=1 decode correctly uses the `mul_mat_vec` path (bandwidth-bound). Remaining win: faster K/V or smaller quants for dense decode.
- wave64 pipeline variants for gfx12
- Linux + RADV exposes `VK_NV_cooperative_matrix2` and the decode-vector extension (faster prompt processing); HIP/ROCm is also worth benchmarking against Vulkan on the same GPU

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
