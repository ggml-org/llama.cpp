# llama.cpp (Adaptive KV Streaming & RDNA 4 ROCm Optimizations)

This fork adds **Adaptive KV Streaming** and **RDNA 4 (gfx1201) / ROCm optimizations** to `llama.cpp`. It allows consumer GPUs with 16 GB VRAM (such as the AMD Radeon RX 9070 XT) to execute models like **Qwen3.8-27B** (`IQ4_XS` model weights, `Q8_0` K / `Q4_0` V cache) across their full **262,144-token context window** without running out of memory, delivering **100+ t/s** decode and **1,600+ t/s** prefill.

---

## Improvements

- **Adaptive KV Streaming**:
  - Automatically pages and streams the KV cache between host RAM and GPU VRAM.
  - Active VRAM residency is fixed and bounded within dedicated prefill and decode working sets.
  - Dedicated asynchronous copy engine streams KV pages via PCIe SDMA alongside compute streams.
  - Enables full 262,144-token context depth on a single 16 GB GPU with 100% recall and zero allocation crashes.
- **RDNA 4 (gfx1201) and ROCm Kernel Optimizations**:
  - Prefill attention kernel for the chunked paged prefill that reads the `Q8_0` K / `Q4_0` V cache directly.
  - Decode attention kernel for `Q8_0` K / `Q4_0` V with aligned LDS reads for K fragments and compact V scales.
  - Chunked gated delta net (GDN) prefill on WMMA: the recurrent state stays in registers across 16-token chunks.
  - `mmq`: narrow tiles for weights with few row tiles, int32 sums converted to float without `v_cvt`, LDS-only
    barriers in its loop (and in the FA prefill loop).
  - Multi-column `mmvq` with 4 rows per block tuned for RDNA 4 wave dispatch; 16-byte weight reads for Q4_K and Q5_K;
    IQ2/IQ3 grid signs without emulated byte-wise SIMD ops.
  - Kernel fusions: residual add + RMS norm, SwiGLU and the residual add + RMS norm writing the quantized `q8_1` copy
    for `mmvq`, the GDN gate/beta and alpha gate, and the GDN state read from and written to the recurrent cache.
  - Sibling matmul activation reuse across parallel projections.
  - Launch sizes stepped past the full-card wave cliff (normalizations, FWHT).
- **Speculative decoding**:
  - Multi-Token Prediction (MTP) drafts sampled at request temperature and verified by rejection sampling, preserving
    the model's output distribution.
  - Opportunistic n-gram drafts (`ngram-mod`) stacked with MTP: up to 32-token bursts when the output repeats context.
  - ReplaySSM: delta-net layers roll drafts back from recorded updates instead of state snapshots (smaller recurrent
    buffer, n-gram drafts roll back in place); the memory it frees goes to the KV decode budget.
- **Prefill memory**: the KQ mask covers only the cells that are not fully visible (compact mask).
- **HIP Runtime and Graph Tuning**:
  - Tuned for ROCm 7.2.3 and HIP graphs with single-queue async mode to prevent launch bubbles.
  - Constrains hardware queue allocations to prevent preemption save area VRAM blowups during long prefills.

This work builds on concepts from [RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) and [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts), re-implemented specifically for RDNA 4 hardware targets in ROCm/HIP.

---

## Setup

The configurations and benchmarks in this repository use **Qwen3.8-27B** (ASCII-condensed `IQ4_XS`, 12.63 GiB): [bsaleh03/Qwen3.8-27B-ASCII-Condensed](https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed). Another option would be [troed/Qwen3.8-27B-ASCII-Condensed](https://huggingface.co/troed/Qwen3.8-27B-ASCII-Condensed).

### System Requirements

- **GPU**: AMD Radeon RX 9070 XT (`gfx1201`, RDNA 4) or compatible ROCm GPU with >= 16 GB VRAM.
- **Host Bus**: PCIe 5.0 x16 recommended for optimal streaming throughput at > 80k context (PCIe 4.0 supported).
- **Host RAM**: >= 32 GB RAM (~7.4 GB host memory allocated for 262,144 context; dual-channel DDR5 recommended).
- **ROCm Toolchain**: ROCm 7.2.3 or later on Linux.

### NixOS

```bash
nix run .#serve-openai -- -m /path/to/Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf
```

### Other Linux Distributions

Follow official documentation to install AMD ROCm for your distribution.

```bash
# Build
cmake -B build-rocm -G Ninja \
  -DCMAKE_C_COMPILER=hipcc \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16,q8_0-q4_0" \
  -DGGML_BACKEND_DL=OFF \
  -DGGML_CCACHE=OFF

cmake --build build-rocm --config Release -j$(nproc)
```

```bash
# Launch server
export ROCR_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=2
export DEBUG_HIP_FORCE_ASYNC_QUEUE=1
export DEBUG_HIP_FORCE_GRAPH_QUEUES=1

./build-rocm/bin/llama-server \
  -m /path/to/Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 262144 -np 1 -dev ROCm0 -fa on -fit off \
  -ctk q8_0 -ctv q4_0 -ctkd q8_0 -ctvd q4_0 \
  --kv-stream \
  --kv-resident-mib 450 \
  --kv-resident-mib-decode 2420 \
  --spec-type draft-mtp,ngram-mod \
  --spec-rs-replay \
  --spec-draft-n-max 4 \
  --spec-ngram-mod-n-max 32 \
  --spec-ngram-mod-n-min 12 \
  --spec-ngram-mod-n-match 24 \
  --spec-draft-sampling probabilistic \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  -b 2048 -ub 512
```

---

## Benchmark

- **Hardware**: AMD Radeon RX 9070 XT (`gfx1201`), 300 W power limit, PCIe 5.0 x16, ROCm 7.2.3.
- **Fork Build**: Commit `3ef1275ca` (`feature/adaptive-kv-stream-rocm`).
- **Upstream Build**: Commit `a37cc4610` (`upstream/master`, the fork's base), configured with the fork's flags.
- **Workload**: An agent-style edit turn as a chat request. The context is the SQLite amalgamation (`sqlite3.c`) cut
  to each ladder depth, followed by `sqlite3GetInt32()`; the request asks for a NULL guard as a minimal old/new edit
  and a short explanation. Reasoning effort low; temperature 1.0, top-k 20, top-p 0.95, min-p 0.05; each turn ends at
  the model's end of text (cap 4,096 tokens).
- **Measurement**: One cold prefill per configuration and depth; decode is the median of three seeds, each on a fresh
  server from the restored prompt state. Prefill and decode t/s from the server's timings.
- **Configurations**: The same server flags for both builds (`-fit off`, `Q8_0` K / `Q4_0` V, `-c` = depth + 4,096).
  Decode is shown autoregressive and with MTP + n-gram drafts (the launcher's speculative settings); prefill is the
  autoregressive configuration's (MTP's head adds its own pass over the prompt). OOM: the server could not allocate
  its buffers at startup.

![Benchmark Matrix Summary](docs/benchmark_matrix_summary.png)

### Decode Throughput (tokens/second)

| Context Depth | Stock llama.cpp | Stock + MTP + n-gram | This Fork | This Fork + MTP + n-gram |
| :--- | :--- | :--- | :--- | :--- |
| **1k** (928 tokens) | 34.07 t/s | 84.89 t/s | **36.83 t/s** (+8.1%) | **117.83 t/s** (+38.8%) |
| **4k** (4,005 tokens) | 33.16 t/s | 83.15 t/s | **36.56 t/s** (+10.3%) | **115.59 t/s** (+39.0%) |
| **16k** (16,305 tokens) | 30.62 t/s | 66.28 t/s | **35.84 t/s** (+17.0%) | **109.61 t/s** (+65.4%) |
| **64k** (65,466 tokens) | 23.52 t/s | **OOM** | **33.32 t/s** (+41.7%) | **99.07 t/s** |
| **128k** (130,960 tokens) | **OOM** | **OOM** | **28.37 t/s** | **80.93 t/s** |
| **256k** (258,019 tokens) | **OOM** | **OOM** | **10.47 t/s** | **46.57 t/s** |

### Prefill Throughput (tokens/second)

| Context Depth | Stock llama.cpp | This Fork |
| :--- | :--- | :--- |
| **1k** (928 tokens) | 1,173.8 t/s | **1,383.3 t/s** (+17.8%) |
| **4k** (4,005 tokens) | 1,394.0 t/s | **1,607.8 t/s** (+15.3%) |
| **16k** (16,305 tokens) | 1,385.9 t/s | **1,628.5 t/s** (+17.5%) |
| **64k** (65,466 tokens) | 1,104.9 t/s | **1,417.6 t/s** (+28.3%) |
| **128k** (130,960 tokens) | **OOM** | **1,201.1 t/s** |
| **256k** (258,019 tokens) | **OOM** | **924.8 t/s** |

---

## Multi-Turn Agent Performance

I have been using this setup for multi-turn agentic coding sessions. Here are the aggregated results from these sessions (recorded on the earlier build `bd063a49a`):

![Real-World Agent Workload Performance Profile](docs/agentic_workload_performance.png)

While the measurements above were taken at 300W TDP, operating with a 231W power limit (-24% power reduction relative to 304W default TDP) incurs only a 3.6% prefill latency and exhibits virtually identical decode throughput, providing significant power and thermal savings for real-world workloads.

---

*(Original upstream llama.cpp documentation follows below)*

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

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
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
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
