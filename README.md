# halo-box/llama.cpp

<img src="halo-box.png" alt="Halo Box" width="260">

<b>llama.cpp, close to mainline, with more</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[upstream llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) / [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp)

## Halo Box

The goal is simple: more functionality, and the fastest llama.cpp around. And help the community with a single fast
llama.cpp fork instead of many competing ones.

Halo Box keeps two forks, and which one you want depends on your hardware:

| Fork | What it is |
| --- | --- |
| [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) (this repo) | Stays close to mainline. Tracks upstream `master` and adds features and speedups on top, without diverging from how upstream works. |
| [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) | Purely optimised for AMD Strix Halo machines (Ryzen AI Max+, RDNA 3.5 / gfx1151). Free to diverge from upstream wherever that buys speed. |

Use this repo if you want upstream behaviour plus extras. Use `strix-llama.cpp` if you run a Strix Halo box and
want every last token/s out of it. Everything here is merged into `strix-llama.cpp` regularly, so that repo is a
superset of this one.

## What this is

A community fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) that stays close to mainline. It tracks
upstream `master`, merges it in regularly, and adds features and speedups on top without changing how upstream
behaves. Upstream behaviour is unchanged - this is a superset, not a rewrite.

It is also the staging fork for the pair: anything general enough for upstream is developed here, on `halo/*`
branches, and submitted to `ggml-org/llama.cpp` from here under the upstream project's contribution and AI-usage
rules. What stays here is either not yet ready to go up, or too niche for mainline. Work that only makes sense on
AMD Strix Halo lives in [strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) instead.

## Quick start

Build from source. For example:

**Vulkan** (works on any recent GPU; on AMD, RADV on Mesa is the easiest path)

```sh
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

**ROCm / HIP** (needs ROCm installed; set `GPU_TARGETS` to your GPU, `gfx1151` is Strix Halo)

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Then:

```sh
# chat, pulling the model straight from Hugging Face
./build/bin/llama-cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# OpenAI-compatible API server + web UI on http://localhost:8080
./build/bin/llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF
```

Full build documentation, including Windows and Docker, is in [docs/build.md](docs/build.md).

## Running on Strix Halo

**Give the iGPU enough memory.** The APU's memory is shared, and the GPU can only use what the firmware and kernel let
it map. Two things control this: the UMA / dedicated-VRAM split in your BIOS, and the `amdgpu` GTT limit on Linux
(`amdgpu.gttsize`, in MB, and `ttm.pages_limit`, in 4 KB pages, as kernel command-line parameters). Which of those you
need depends on your kernel version - newer kernels size GTT more generously on their own. If a model that clearly
fits in RAM fails to allocate, this is almost always why.

**Measure things.** `GGML_VK_PERF_LOGGER=1` (any value) gives per-op timings on the Vulkan backend. `llama-bench`
and `llama-perplexity` are the tools for before/after numbers, and performance PRs here are expected to carry them.

Strix Halo specific notes and tuning (ROCm workarounds, Vulkan mat-vec chunking) are in the
[strix-llama.cpp README](https://github.com/halo-box/strix-llama.cpp#running-on-strix-halo).

## What differs from upstream

Everything else is upstream `llama.cpp`. The additions currently carried here:

| Change | Flag / switch | What it does |
| --- | --- | --- |
| Speculative prefill | `--spec-prefill` | A small draft model scores prompt tokens by attention importance so the target model only prefills the ones that matter, cutting time-to-first-token on long prompts |
| N-gram table on disk | `--ngram-on-disk`, `--ngram-cache`, `--ngram-io-threads` | Keeps a model's n-gram hash-embedding table (28.8 GB on Qwen3.8-Flash-Next) off the memory budget entirely, reading only the rows each batch actually gathers |
| Adaptive speculative draft length | `--spec-draft-adaptive` | Sizes each draft from a measured per-sequence acceptance EMA rather than always drafting `--spec-draft-n-max`; speeds up MTP and DFlash |
| Vulkan fixes and tuning for RDNA 3.5 | | Driver-gated coopmat LDS stride padding, UMA bulk readback gated on host-cached mappings, IQ3_S mat-vec at batch sizes > 4, and a radix top-k kernel for large k |
| Hidden server presets | `hidden` in the models `.ini` | Keep a model loadable by name while omitting it from `GET /models` |

Run `--help`, or see [tools/server/README.md](tools/server/README.md), for the full options.

## Supported backends

All of upstream's, unmodified:

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

Most of the work in this fork is measured on AMD Strix Halo (Vulkan and HIP), and CI runs the standard upstream matrix.

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

This is a small community project. A benchmark, a bug report, or a patch is exactly what it is for.

- This repo follows the upstream contribution and AI-usage rules, because what lands here is meant to go upstream.
  See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md); in short, understand every line you submit.
- Performance claims need numbers against a baseline you built and ran yourself, on the same machine in the same
  session.
- If your change is Strix Halo specific, send it to [strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp)
  instead; that repo has its own, more permissive rules.
- Work lands on `halo/*` branches, and upstream is merged in regularly.

## Acknowledgements

This project is a fork and owes everything to the people who built what it forks:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml) - Georgi Gerganov and the llama.cpp contributors - MIT license
- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
