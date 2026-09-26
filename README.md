# ggml-llama.cpp (Raudbjorn fork)

> Single-maintainer fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on the TurboQuant+ codec stack and SYCL/Vulkan deployment on Intel Arc A770. Backends shipped: CPU, BLAS, SYCL, Vulkan, OpenVINO. CUDA, ROCm/HIP, Metal, OpenCL, WebGPU, CANN, MUSA, hexagon, RPC are not built or supported here.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintained by Raudbjorn](https://img.shields.io/badge/maintainer-Raudbjorn-blueviolet.svg)](https://github.com/Raudbjorn)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) integrating the **TurboQuant+** codec stack -- Walsh-Hadamard rotated polar quantization, attention-gated sparse dequantization, and layer-aware V compression policies. The codec design, calibration, and validation papers live at [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus); this repository is the llama.cpp runtime integration.

### Lineage -- why the `+`

This project extends that foundation -- adding the asymmetric K/V policy (V is free, K is everything), layer-aware Boundary V protection, attention-gated sparse V dequantization, the `TQ3_1S` / `TQ4_1S` weight quantization formats, the `turbo2` / `turbo3` / `turbo4` tier variants, the SYCL/Vulkan kernel coverage, and a body of model-family-specific quality and operational fixes. The trailing `+` denotes ongoing extension work; the original TurboQuant codec itself was the Google ICLR 2026 contribution.

This fork is additive within its scope: every shipped backend (CPU, BLAS, SYCL, Vulkan, OpenVINO) continues to work as in upstream, and all TurboQuant+ types (weights, KV cache) are opt-in via the standard `--cache-type-k` / `--cache-type-v` and `llama-quantize` interfaces. Backends not in the shipped set (CUDA, HIP/ROCm, Metal, OpenCL, CANN, MUSA, WebGPU, RPC, Hexagon) are not built here; pull upstream for those.

## Maintenance

This is a single-maintainer fork. No production deployments are tracked here -- if you deploy this fork somewhere public, please open an issue so the README can be updated.

| | |
|---|---|
| Maintainer | Sveinbjorn Geirsson (Raudbjorn) |
| Default branch | `master` |
| Backends shipped | CPU, BLAS, SYCL, Vulkan, OpenVINO |
| Upstream tracking | continuous sync from `ggml-org/llama.cpp` master |
| Upstream PR status | not yet upstreamed; running as a long-lived fork |


---

## What this fork adds

### Quantization types

| Type | Domain | Approx. bits | Notes | Paper |
|---|---|---|---|---|
| `TQ3_1S` | weights | ~3.5 | smaller VRAM than `q8_0` | [weight-compression-tq4](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) |
| `TQ4_1S` | weights | ~4.5 | smaller VRAM than `q8_0`; requires GPU backend (Vulkan or SYCL) to be built in | [weight-compression-tq4](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) |
| `turbo2` | KV cache | ~2.0 | aggressive; pair with Boundary V | [block-size-experiment](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/block-size-experiment.md) |
| `turbo3` | KV cache | ~3.5 | 5.12x analytic compression vs f16; +5.11% PPL vs q8_0 on Llama-3.1-8B-Instruct Q4_K_M at ctx 512 on this fork (see [turbo3 gate note](docs/research/turbo3-quality-gate-llama31-8b-2026-09.md)) | [attn-rotation-and-ppl-artifact](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/attn-rotation-and-ppl-artifact.md) |
| `turbo4` | KV cache | ~4.5 | rehabilitated to beat `q4_0` on fidelity | [turbo4-resurrection](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) |

All turbo formats use Walsh-Hadamard rotation followed by polar codebook quantization on 128-element blocks. Why this works where MSE-driven codecs fail: [why-mse-fails-for-kv-quantization](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md).

### Compression policies

- **Auto-asymmetric K/V compression** -- recognizes that V tolerates aggressive compression while K does not; default policy picks complementary codecs rather than symmetric. [asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)
- **Boundary V (experimental, layer-aware)** -- auto-enabled for `turbo2-V`. Protects layers where aggressive V quantization degrades quality, leaves the rest at full aggression. [layer-aware-v-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md), [moe-v-compression-frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md)
- **Sparse V dequantization** -- skip V dequantization for positions whose softmax attention weight falls below threshold. Enabled across SYCL/Vulkan targets. [sparse-v-dequant](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md)

### Backend coverage

| Backend | Quant kernels | Flash Attention | Notes |
|---|---|---|---|
| **SYCL** (Intel Arc / oneAPI) | turbo `mmvq` + WHT custom op | `q8_0` / `f16` KV at mainline parity; turbo3/turbo4 KV FA pass the correctness gate (gated behind `LLAMA_TEST_TURBO_FA=1` for opt-in validation); turbo2 KV FA remains XFAIL (2-bit precision below the cosine floor) | A770 (DG2) is the canonical target; builds with `GGML_SYCL_F16=ON` or `OFF` |
| **Vulkan** | `TQ4_1S` weights, `SET_ROWS` for `turbo2`/`turbo4` | coopmat flash attention with `turbo3` KV | Compute-shader path; nix-buildable |


OpenVINO is shipped in-tree but is not exercised by the TurboQuant+ probes in this fork -- build with `-DGGML_OPENVINO=ON` only if you need the upstream OpenVINO backend.

### Model-family support

- **Gemma 4** -- large head-dim (`dk=512`) FA kernels, MoE token routing, op-concurrency handling
- **Large MoE** -- kernel instantiations for up to 256-expert routing
- **Hybrid architectures (GDN, Mamba)** -- speculative decoding cherry-picked from upstream feature branches
- All existing llama.cpp model families remain fully supported

### Operational fixes carried by this fork

- CPU `vec_dot` heap-allocation fix for turbo / TQ types at `n > 4096`
- Cross-vendor `-Werror` build fixes
- Defensive `xxd.cmake` handling for missing input files

---

## Quick start

Build with the backends you need. TurboQuant+ types become available once a supported GPU backend is compiled in.

```bash
# CPU only
cmake -B build && cmake --build build -j

# SYCL (Intel Arc / oneAPI) -- canonical target
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && cmake --build build -j

# Vulkan
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j

# SYCL + Vulkan (A770 with Vulkan fallback)
cmake -B build -DGGML_SYCL=ON -DGGML_VULKAN=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && cmake --build build -j
```

KV-cache types are selected per-side via the standard `--cache-type-k` / `--cache-type-v` flags.

> **Start light, then compress.** Some model families -- small models, certain MoE configurations, quant-sensitive instruction-tuned variants -- are more delicate than others. Pick a light asymmetric configuration first, verify output quality (eyeball + PPL on a hold-out set) on your specific model, then ratchet up V aggression if you have memory headroom to gain. **Do not start at maximum compression** and work backwards.

The core finding from the asymmetric-kv-compression paper -- [**Asymmetric K/V Cache Compression: Why V is Free and K is Everything**](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) -- drives all the configs below: **V tolerates aggressive compression, K does not**. Always keep K at higher precision than V; never start symmetric. That paper documents the specific failure modes you'll hit if you ignore this and compress K aggressively (PPL blow-up on certain model families, attention-rotation interaction with low-bit K, etc.) -- read it before considering step 6.

Higher turbo number = more bits per element = less aggressive compression. The V-side compression ladder is `turbo4` (lightest) -> `turbo3` -> `turbo2` (heaviest). On the K side, prefer `f16` or `q8_0`; never lead with a turbo K.

Recommendations, ordered from most conservative to most aggressive:

| Step | `--cache-type-k` | `--cache-type-v` | When | Notes |
|---|---|---|---|---|
| **1. Safest start** | `f16` | `turbo4` | First contact with any new model | K untouched, V at the lightest turbo tier. If output isn't faithful at this step, the model is unusually quant-sensitive -- stop and investigate before escalating. |
| **2. Conservative** | `q8_0` | `turbo4` | Verified safe at step 1, want a memory win without much risk | Light on both sides. Typically near-indistinguishable from `f16`/`f16` outputs. |
| **3. Recommended default** | `q8_0` | `turbo3` | Most dense models, most production workloads | The "asymmetric turbo" sweet spot from the [asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) paper. Near-lossless K, ~4.6x compressed V. Total KV ~3-4x smaller than `f16`/`f16`. |
| **4. Aggressive V** | `q8_0` | `turbo2` | Memory-bound long context, after validating quality at step 3 | Boundary V auto-engages and protects sensitive layers. Expect <2% PPL loss on dense models outside the protected layers. |
| **5. MoE-aware aggressive** | `q8_0` | `turbo2` | Large MoE models (DeepSeek, Qwen3.6, Mixtral-style) | Same flags; Boundary V's per-expert-boundary protection is what makes this work on MoE. See [moe-v-compression-frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md). |
| **6. Discouraged: symmetric K compression** | any `turbo*` | any `turbo*` | Only with model-specific quality validation in hand | Compressing K is where models break. The asymmetric paper documents the failure modes. Not a starting point. |

Example invocations:

```bash
# Step 1 -- safest start (first contact with a new model)
llama-cli -m model.gguf --cache-type-k f16 --cache-type-v turbo4 -p "..."

# Step 3 -- recommended default (asymmetric turbo)
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 -p "..."

# Step 4 -- aggressive V at long context
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo2 -c 131072 -p "..."
```

If output quality drops between steps, walk back to the previous step. The compression frontier is per-model -- there is no global "best" setting.

### Weight quantization (offline)

Weight quantization is selected at conversion time via `llama-quantize`:

```bash
# TQ4_1S -- recommended for most GPU deployments (Vulkan or SYCL)
llama-quantize model.f16.gguf model.tq4_1s.gguf TQ4_1S

# TQ3_1S -- smaller, accept ~1-2 PPL bump
llama-quantize model.f16.gguf model.tq3_1s.gguf TQ3_1S
```

### Automatic behavior

The following activate based on the selected types -- no flags required:

- **Auto-asymmetric K/V** -- when both sides are turbo / TQ types, the policy picks complementary configurations rather than symmetric.
- **Boundary V (layer-aware)** -- auto-enables for any `turbo2-V` selection.
- **Sparse V dequantization** -- on SYCL/Vulkan targets, sparse V activates for all turbo V types.
- **Flash Attention** -- auto-enabled for turbo KV with the relevant backend kernel (XFAIL on SYCL by default; opt in with `LLAMA_TEST_TURBO_FA=1`).

If this fork or any of its quantization types is used in your work, please cite the corresponding paper from the [TurboQuant+ paper corpus](https://github.com/TheTom/turboquant_plus/tree/main/docs/papers).

## License

MIT, same as upstream llama.cpp.

---

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

----


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
- x86_64 (AVX / AVX2 / AVX512 / AMX), ARM NEON, and RISC-V (RVV) SIMD paths on CPU
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization, plus the TurboQuant+ family (`turbo2`/`turbo3`/`turbo4`/`TQ3_1S`/`TQ4_1S`)
- Vulkan and SYCL backend support (this fork's GPU targets)
- BLAS support (OpenBLAS, oneMKL, AOCL) for CPU matrix multiplication
- CPU+GPU hybrid inference to partially accelerate models larger than total VRAM

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends in this fork

Only the backends below are built and tested. CUDA, HIP/ROCm, Metal, OpenCL, CANN, MUSA, WebGPU, RPC, Hexagon, BLIS, ZenDNN, IBM zDNN, and VirtGPU are present in upstream `ggml-org/llama.cpp` but **not shipped here** -- see the upstream README for those.

| Backend | Target devices | Notes |
| --- | --- | --- |
| [CPU](docs/build.md#cpu-build) | x86_64 / ARM / RISC-V | Default backend; SIMD-accelerated |
| [BLAS](docs/build.md#blas-build) | All | OpenBLAS, oneMKL, AOCL, Accelerate |
| [SYCL](docs/backend/SYCL.md) | Intel GPU (Arc / iGPU / Data Center GPU Max) | Canonical target is the Arc A770; oneAPI/DPC++ toolchain required |
| [Vulkan](docs/build.md#vulkan) | GPU (cross-vendor) | Compute-shader path; works on Intel/AMD/NVIDIA |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs | Shipped in-tree; not exercised by TurboQuant+ probes |

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
