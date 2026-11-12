# OpenVINO Backend for llama.cpp

> [!NOTE]
> Performance and memory optimizations, accuracy validation, broader quantization coverage, broader operator and model support are work in progress.

[OpenVINO](https://docs.openvino.ai/) is an open-source toolkit for optimizing and deploying high-performance AI inference, specifically designed for Intel hardware, including CPUs, GPUs, and NPUs, in the cloud, on-premises, and on the edge. [OpenVINO backend for llama.cpp](../../ggml/src/ggml-openvino) enables hardware-accelerated inference on **Intel® CPUs, GPUs, and NPUs** while remaining compatible with the existing **GGUF model ecosystem**. The backend translates GGML compute graphs into OpenVINO graphs and leverages graph compilation, kernel fusion, and device-specific optimizations to improve inference performance on supported Intel hardware.

The OpenVINO backend is implemented in `ggml/src/ggml-openvino` and provides a translation layer for core GGML operations. The OpenVINO backend replaces the standard GGML graph execution path with Intel's OpenVINO inference engine. This approach allows the same GGUF model file to run on Intel CPUs, Intel GPUs (integrated and discrete), and Intel NPUs without changes to the model or the rest of the llama.cpp stack. When a `ggml_cgraph` is dispatched to OpenVINO backend, it:

- Walks the GGML graph and identifies inputs, outputs, weights, and KV cache tensors.
- Translates the GGML operations into an `ov::Model` using OpenVINO's frontend API.
- Compiles and caches the model for the target device.
- Binds GGML tensor memory to OpenVINO inference tensors and runs inference.

For guidance on contributing to the OpenVINO backend, see the [OpenVINO Backend Contributing Guide](https://github.com/ravi9/llamacpp-ov-dev-guide/blob/main/contributing-llamacpp-ov.md).

## Contents

- [Quick Start](#quick-start)
- [Supported Devices](#supported-devices)
- [Supported Model Precisions](#supported-model-precisions)
- [Supported Llama.cpp Tools](#supported-llamacpp-tools)
- [Validated Models](#validated-models)
- [Build Instructions](#build-instructions)
  - [0. Prerequisites](#0-prerequisites)
  - [1. Install OpenVINO Runtime](#1-install-openvino-runtime)
  - [2. Build llama.cpp with OpenVINO Backend](#2-build-llamacpp-with-openvino-backend)
    - [Ubuntu Build Script](#ubuntu-build-script)
    - [Windows Build Script](#windows-build-script)
  - [3. Download Sample Model](#3-download-sample-model)
  - [4. Run Inference with OpenVINO Backend](#4-run-inference-with-openvino-backend)
  - [5. Docker Build](#5-docker-build)
- [GGML OpenVINO Backend Runtime Configurations](#ggml-openvino-backend-runtime-configurations)
  - [Basic](#basic)
  - [Plugin properties](#plugin-properties)
  - [Advanced tuning](#advanced-tuning)
  - [Compiled model cache](#compiled-model-cache)
  - [Deprecated](#deprecated)
  - [Debugging and development](#debugging-and-development)
  - [Example Usage](#example-usage)
    - [GPU Inference with Profiling](#gpu-inference-with-profiling)
- [Known Limitations](#known-limitations)
- [Work in Progress](#work-in-progress)

## Quick Start

On Ubuntu, with the [prerequisites](#0-prerequisites) and the [OpenVINO Runtime](#1-install-openvino-runtime) installed:

```bash
# 1. Build
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel

# 2. Download a model
mkdir -p ~/models/
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     -O ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# 3. Run on CPU, GPU or NPU
export GGML_OPENVINO_DEVICE=GPU
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 1024
```

The [Ubuntu Build Script](#ubuntu-build-script) and the [Windows Build Script](#windows-build-script) do the setup in one step: they install the prerequisites and the OpenVINO Runtime, and build llama.cpp with the OpenVINO backend.

See [Build Instructions](#build-instructions) for Windows and Docker, and [Runtime Configurations](#ggml-openvino-backend-runtime-configurations) for the environment variables.

## Supported Devices

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel GPUs (integrated and discrete)
- Intel NPUs

Although OpenVINO supports a wide range of [Intel hardware](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html), the llama.cpp OpenVINO backend has been validated specifically on AI PCs such as the Intel® Core™ Ultra Series 2 (Lunar Lake) and Series 3 (Panther Lake).

## Supported Model Precisions

- `FP16`
- `BF16` (on Intel Xeon)
- `Q8_0`
- `Q4_0`
- `Q4_1`
- `Q4_K`
- `Q4_K_M`
- `Q5_K` (converted at runtime, see the details below)
- `Q6_K` (converted at runtime, see the details below)

> [!NOTE]
> Accuracy validation and performance optimizations for quantized models are a work in progress.

<details>
<summary>Quantization details for each device</summary>

**All devices:**
- `FP16` and `BF16` tensors are used as they are.
- A quantized token embedding (`token_embd.weight`) and output weight (`output.weight`) are converted to `Q8_0_C`.

**CPU and GPU:**
- `Q4_0`, `Q4_1`, `Q4_K`, `Q5_1` and `Q8_0` tensors keep their layout: weights, scales and zero points are extracted directly.
- `Q5_K` and `Q6_K` tensors are converted to `Q8_0_C`. `GGML_OPENVINO_REQUANT_KQUANT` selects a 4-bit layout instead.
- `Q5_0` tensors are converted to `Q4_0_64`.

**NPU:**
- All other quantized tensors are requantized to `Q4_0_128`.
- For a `Q6_K` token embedding, `GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT` selects another layout.
- The validated models use `Q4_K_M` files.

**Layout names:**
- `Q8_0_C`: 8-bit, one scale per row.
- `Q4_0_128` and `Q4_0_64`: 4-bit, symmetric, one scale per 128 or 64 weights.

**Additional Notes:**
- Both `Q4_0` and `Q4_1` models use `Q6_K` for the token embedding tensor and the final matmul weight tensor (often the same tensor)
- `Q4_0` models may produce some `Q4_1` tensors if an imatrix is provided during quantization using `llama-quantize`
- `Q4_K_M` models may include both `Q6_K` and `Q5_K` tensors (observed in Phi-3)

</details>

## Supported Llama.cpp Tools

The OpenVINO backend integrates with the standard llama.cpp tools listed below.
However, all the tools coverage across all devices is not uniform and exhaustive validation is work in progress.

- llama-bench
- llama-cli
- llama-completion
- llama-embedding
- llama-perplexity
- llama-server
- llama-simple

## Validated Models

Although, the validated models below were tested with `llama-cli` using the `Q4_K_M` quantization format on Intel® Core™ Ultra Series 2 (Lunar Lake), the OpenVINO backend is expected to work across a broader range of [Intel hardware](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html), [supported model precisions](#supported-model-precisions), [supported llama.cpp tools](#supported-llamacpp-tools) and additional model architectures.

> [!NOTE]
> Extensive accuracy validation, performance optimizations, and broader architecture coverage are work in progress.

**Legend & Test Configuration:**
- **Status:** ✓ = Passed | ~ = Accuracy issues | ✗ = Failed or Unsupported
- **Execution mode:** stateless (`GGML_OPENVINO_STATEFUL_EXECUTION=0`) on every device.
- **Validation system:** Intel® Core™ Ultra 7 258V (Lunar Lake) | 32 GB RAM | Ubuntu 26.04.1 | Intel Graphics Compiler 2.42.6 | Intel OpenCL GPU Driver 26.40.40196.8-0 | Intel NPU Driver 1.38.0.
- See [Known Limitations](#known-limitations) for context on observed failures.

<details open>
<summary><b>Validated models</b></summary>

| Model | CPU | GPU | NPU |
| :--- | :---: | :---: | :---: |
| [bartowski/Llama-3.2-1B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Llama-3.2-3B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Meta-Llama-3.1-8B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | ✓ | ✓ | ✓ |
|  |  |  |  |
| [Qwen/qwen2.5-1.5b-instruct-q4_k_m](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [Qwen/qwen2.5-coder-7b-instruct-q4_k_m](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Qwen_Qwen3-0.6B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Qwen_Qwen3-1.7B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF) | ✓ | ✓ | ✓ |
| [Qwen/Qwen3-4B-Q4_K_M](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | ✓ | ✓ | ✓ |
| [lm-kit/Qwen3-8B-Q4_K_M](https://huggingface.co/lm-kit/qwen-3-8b-instruct-gguf) | ✓ | ✓ | ✓ |
| [bartowski/Qwen_Qwen3.5-0.8B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Qwen_Qwen3.5-2B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Qwen_Qwen3.5-4B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF) | ✓ | ✓ | ✓ |
| [lmstudio-community/Qwen3.5-9B-Q4_K_M](https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF) | ✓ | ✓ | ✓ |
|  |  |  |  |
| [unsloth/gemma-3-4b-it-Q4_K_M](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/google_gemma-4-E2B-it-Q4_K_M](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/google_gemma-4-E4B-it-Q4_K_M](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/gemma-4-12B-it-Q4_K_M](https://huggingface.co/bartowski/gemma-4-12B-it-GGUF) | ✓ | ✓ | ✗ |
|  |  |  |  |
| [bartowski/Phi-3-mini-4k-instruct-Q4_K_M](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Phi-3.5-mini-instruct-Q4_K_M](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/microsoft_Phi-4-mini-instruct-Q4_K_M](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF) | ✓ | ✓ | ✓ |
|  |  |  |  |
| [bartowski/Mistral-7B-Instruct-v0.3-Q4_K_M](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF) | ✓ | ✓ | ✓ |
| [QuantFactory/Ministral-3b-instruct.Q4_K_M](https://huggingface.co/QuantFactory/Ministral-3b-instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/Ministral-8B-Instruct-2410-Q4_K_M](https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF) | ✓ | ✓ | ✓ |
|  |  |  |  |
| [bartowski/DeepSeek-R1-Distill-Llama-8B-Q4_K_M](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) | ✓ | ✓ | ✓ |
|  |  |  |  |
| [ibm-granite/granite-4.0-350m-Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF) | ✓ | ✓ | ✓ |
| [ibm-granite/granite-4.0-micro-Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF) | ✓ | ✓ | ✓ |
| [ibm-granite/granite-4.0-1b-Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-1b-GGUF) | ✓ | ✓ | ~ |
| [ibm-research/granite-3.2-8b-instruct-Q4_K_M](https://huggingface.co/ibm-research/granite-3.2-8b-instruct-GGUF) | ✓ | ✓ | ✓ |
|  |  |  |  |
| [HuggingFaceTB/smollm2-1.7b-instruct-q4_k_m](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [openbmb/MiniCPM-V-2_6-Q4_K_M](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) | ✓ | ✓ | ✓ |
| [bartowski/tencent_Hunyuan-7B-Instruct-Q4_K_M](https://huggingface.co/bartowski/tencent_Hunyuan-7B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-Q4_K_M](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF) | ✓ | ✓ | ✓ |
| [bartowski/prism-ml_Bonsai-8B-unpacked-Q4_K_M](https://huggingface.co/bartowski/prism-ml_Bonsai-8B-unpacked-GGUF) | ✓ | ✓ | ✓ |
| [LiquidAI/LFM2.5-2.6B-Q4_K_M](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) | ✓ | ✓ | ✗ |
| [bartowski/inclusionAI_Ling-mini-2.0-Q4_K_M](https://huggingface.co/bartowski/inclusionAI_Ling-mini-2.0-GGUF) | ✓ | ✓ | ✗ |
|  |  |  |  |
| [Qwen/Qwen3-Embedding-0.6B-Q8_0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | ✓ | ✓ | ✗ |

</details>

## Build Instructions

### 0. Prerequisites

- Linux or Windows system with Intel hardware (CPU, GPU, or NPU)
- **For Intel GPU or NPU Usage**: Install the appropriate hardware drivers for your Intel GPU or NPU. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html).

<details open>
<summary><b>Linux</b></summary>

Build tools (Git, CMake, Ninja):

```bash
sudo apt-get update
sudo apt-get install -y build-essential libcurl4-openssl-dev libtbb12 cmake ninja-build python3-pip curl wget tar
```

OpenCL:

```bash
sudo apt install ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd
```

</details>

<details>
<summary><b>Windows</b></summary>

1. Download and install [Microsoft Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe). During installation, select the **"Desktop development with C++"** workload.
2. Install the required tools:

   ```powershell
   # Windows PowerShell
   winget install Git.Git
   winget install GNU.Wget
   winget install Ninja-build.Ninja
   ```

3. Install **OpenCL** using **vcpkg**:

   ```powershell
   # Windows PowerShell
   cd C:\
   git clone https://github.com/microsoft/vcpkg
   cd vcpkg
   .\bootstrap-vcpkg.bat
   .\vcpkg install opencl
   # Optional but recommended: Integrate vcpkg with Visual Studio / CMake:
   .\vcpkg integrate install
   ```

</details>

### 1. Install OpenVINO Runtime

- Follow the guide to install OpenVINO Runtime from an archive file: [Linux](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html) | [Windows](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html)

- Verify OpenVINO is initialized properly:
  ```bash
  echo $OpenVINO_DIR
  ```

### 2. Build llama.cpp with OpenVINO Backend

Clone the llama.cpp repository:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Build on Linux:

```bash
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

<details>
<summary><b>Build on Windows</b></summary>

Open **x64 Native Tools Command Prompt for VS** (so the MSVC toolchain is on `PATH`), then run:

```cmd
C:\Intel\openvino\setupvars.bat
cmake -B build\ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build\ReleaseOV --parallel
```

> [!NOTE]
> The Windows install path is `C:\Intel\openvino` (no spaces) to avoid quoting problems some CMake/Ninja toolchains have with `C:\Program Files (x86)\...`. Adjust to wherever you installed OpenVINO Runtime. From `cmd`, run `C:\Intel\openvino\setupvars.bat`; from PowerShell, run `& "C:\Intel\openvino\setupvars.ps1"` instead. Once the build is finished you can launch the binaries from any `cmd` or `PowerShell` window after sourcing the matching `setupvars` script for that shell.

</details>

#### Ubuntu Build Script

For Ubuntu24 users, the following shell script automates the prerequisite installs (build tools, OpenCL ICD), the OpenVINO Runtime download/extract/setup, and the Ninja-based llama.cpp build.
Save the following as `build-llamacpp-ov.sh` next to where you want the `llama.cpp` folder to land, then run it:

```bash
chmod +x build-llamacpp-ov.sh
./build-llamacpp-ov.sh
```

<details>
<summary>Click to expand <code>build-llamacpp-ov.sh</code></summary>

```bash
#!/usr/bin/env bash
# ============================================
# llama.cpp OpenVINO Build Script (Ninja)
# ============================================
set -euo pipefail

OPENVINO_VERSION_MAJOR="2026.4.1"
OPENVINO_VERSION_FULL="2026.4.1.22982.07f9c262b05"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVINO_INSTALL_DIR="/opt/intel/openvino_${OPENVINO_VERSION_MAJOR}"
OPENVINO_LINK_DIR="/opt/intel/openvino"
OPENVINO_TGZ="${SCRIPT_DIR}/openvino.tgz"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz"

echo "============================================"
echo "Installing prerequisites (apt)..."
echo "============================================"
sudo apt-get update
sudo apt-get install -y \
    build-essential libcurl4-openssl-dev libtbb12 \
    cmake ninja-build python3-pip \
    curl wget tar git

echo "============================================"
echo "Installing OpenCL runtime + headers..."
echo "============================================"
sudo apt-get install -y \
    ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd

cd "${SCRIPT_DIR}"

# ============================================
# Clone llama.cpp if missing
# ============================================
if [[ ! -f "llama.cpp/CMakeLists.txt" ]]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp
fi

# ============================================
# Setup OpenVINO: download & extract to /opt/intel/openvino_${OPENVINO_VERSION_MAJOR},
# then point /opt/intel/openvino at it via symlink so the active version is swappable.
# ============================================
if [[ -f "${OPENVINO_INSTALL_DIR}/setupvars.sh" ]]; then
    echo "OpenVINO ${OPENVINO_VERSION_MAJOR} already installed at ${OPENVINO_INSTALL_DIR}. Skipping download."
else
    echo "OpenVINO not found at ${OPENVINO_INSTALL_DIR}. Starting download..."
    curl -L -o "${OPENVINO_TGZ}" "${OPENVINO_URL}"

    echo "Extracting OpenVINO to ${OPENVINO_INSTALL_DIR}..."
    sudo mkdir -p "${OPENVINO_INSTALL_DIR}"
    sudo tar -xzf "${OPENVINO_TGZ}" -C "${OPENVINO_INSTALL_DIR}" --strip-components=1
    rm -f "${OPENVINO_TGZ}"
fi

# Refresh symlink: /opt/intel/openvino -> /opt/intel/openvino_${OPENVINO_VERSION_MAJOR}
sudo ln -sfn "${OPENVINO_INSTALL_DIR}" "${OPENVINO_LINK_DIR}"

OPENVINO_ROOT="${OPENVINO_LINK_DIR}"
echo "OpenVINO Ready: ${OPENVINO_ROOT} -> ${OPENVINO_INSTALL_DIR}"

# Install OpenVINO's own runtime dependencies (one-time per system).
if [[ -x "${OPENVINO_ROOT}/install_dependencies/install_openvino_dependencies.sh" ]]; then
    echo "============================================"
    echo "Installing OpenVINO runtime dependencies..."
    echo "============================================"
    echo "Y" | sudo -E "${OPENVINO_ROOT}/install_dependencies/install_openvino_dependencies.sh"
fi

# ============================================
# Clean old build cache
# ============================================
cd "${SCRIPT_DIR}/llama.cpp"
if [[ -d "build/ReleaseOV" ]]; then
    echo "Removing old build directory..."
    rm -rf "build/ReleaseOV"
fi

echo "============================================"
echo "Configuring with CMake..."
echo "============================================"
set +u
source "${OPENVINO_ROOT}/setupvars.sh"
set -u

cmake -B build/ReleaseOV -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENVINO=ON

cmake --build build/ReleaseOV --parallel

echo "============================================"
echo "Build completed successfully!"
echo "============================================"
echo "Binaries: $(pwd)/build/ReleaseOV/bin"
echo
echo "NOTE: To run, source setupvars.sh and pick a device:"
echo "  source /opt/intel/openvino/setupvars.sh"
echo "  export GGML_OPENVINO_DEVICE=CPU   # or GPU / NPU"
echo "  ./build/ReleaseOV/bin/llama-cli -m model.gguf"
```

> [!NOTE]
> The script pins OpenVINO `2026.4.1` via the `OPENVINO_VERSION_MAJOR` / `OPENVINO_VERSION_FULL` variables at the top — edit them to track a different release.

</details>

#### Windows Build Script

For Windows users, the following `.bat` script automates the prerequisite installs (Git, Ninja, CMake, Visual Studio 2022 Build Tools, vcpkg + OpenCL), the OpenVINO Runtime download/extract, and the Ninja-based llama.cpp build.
Save the following as `build-llamacpp-ov.bat` next to where you want the `llama.cpp` to land, then run it from either **Command Prompt** or **PowerShell**:

```cmd
:: Command Prompt
build-llamacpp-ov.bat
```

```powershell
# PowerShell
.\build-llamacpp-ov.bat
```

<details>
<summary>Click to expand <code>build-llamacpp-ov.bat</code></summary>

```bat
@echo off
setlocal enabledelayedexpansion

REM ============================================
REM llama.cpp OpenVINO Build Script (Ninja)
REM ============================================

set "OPENVINO_VERSION_MAJOR=2026.4.1"
set "OPENVINO_VERSION_FULL=2026.4.1.22982.07f9c262b05"

set "SCRIPT_DIR=%~dp0"
set "VCPKG_DIR=C:\vcpkg"
set "OPENVINO_INSTALL_DIR=C:\Intel\openvino_%OPENVINO_VERSION_MAJOR%"
set "OPENVINO_LINK_DIR=C:\Intel\openvino"
set "OPENVINO_ZIP=%SCRIPT_DIR%openvino.zip"
set "OPENVINO_EXTRACT_TMP=%SCRIPT_DIR%openvino_extract_tmp"
set "OPENVINO_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/%OPENVINO_VERSION_MAJOR%/windows/openvino_toolkit_windows_%OPENVINO_VERSION_FULL%_x86_64.zip"

echo ============================================
echo Installing prerequisites...
echo ============================================
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id Ninja-build.Ninja -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements 2>nul

REM Ensure Visual Studio Build Tools are installed.
echo Checking for Visual Studio Build Tools...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALLED="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        set "VS_INSTALLED=%%i"
    )
)
if defined VS_INSTALLED (
    echo Visual Studio with VC++ x86/x64 tools already present at "!VS_INSTALLED!". Skipping winget install.
) else (
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo WARNING: winget could not install Visual Studio Build Tools automatically.
        echo Install manually from https://aka.ms/vs/17/release/vs_BuildTools.exe ^(select the "Desktop development with C++" workload^)
        echo and re-run this script from a "Developer Command Prompt for VS 2022".
    )
)

echo ============================================
echo Installing OpenCL via vcpkg...
echo ============================================
if not exist "%VCPKG_DIR%" (
    git clone https://github.com/microsoft/vcpkg "%VCPKG_DIR%"
    cd /d "%VCPKG_DIR%"
    call bootstrap-vcpkg.bat
    call vcpkg integrate install
)
cd /d "%VCPKG_DIR%"
call vcpkg install opencl

cd /d "%SCRIPT_DIR%"

REM ============================================
REM Clone llama.cpp if missing
REM ============================================
if not exist "llama.cpp\CMakeLists.txt" (
    echo Cloning llama.cpp...
    git clone https://github.com/ggml-org/llama.cpp
)

cd /d "llama.cpp"
set "SCRIPT_DIR=%CD%"

REM ============================================
REM Setup OpenVINO: download & extract to C:\Intel\openvino_%OPENVINO_VERSION_MAJOR%,
REM then point C:\Intel\openvino at it via a directory junction (mklink /J).
REM ============================================

if exist "%OPENVINO_INSTALL_DIR%\setupvars.bat" (
    echo OpenVINO %OPENVINO_VERSION_MAJOR% already installed at "%OPENVINO_INSTALL_DIR%". Skipping download.
) else (
    echo OpenVINO not found at "%OPENVINO_INSTALL_DIR%". Starting download...

    curl -L -o "%OPENVINO_ZIP%" "%OPENVINO_URL%"
    if errorlevel 1 (
        echo ERROR: Download failed.
        exit /b 1
    )

    echo Extracting OpenVINO...
    if exist "%OPENVINO_EXTRACT_TMP%" rmdir /s /q "%OPENVINO_EXTRACT_TMP%"
    mkdir "%OPENVINO_EXTRACT_TMP%"
    tar -xf "%OPENVINO_ZIP%" -C "%OPENVINO_EXTRACT_TMP%"
    if errorlevel 1 (
        echo ERROR: Extraction failed.
        exit /b 1
    )

    REM Move the single top-level folder contents into the versioned install dir.
    set "OPENVINO_EXTRACTED="
    for /d %%i in ("%OPENVINO_EXTRACT_TMP%\*") do set "OPENVINO_EXTRACTED=%%i"
    if not defined OPENVINO_EXTRACTED (
        echo ERROR: Could not locate extracted OpenVINO folder under "%OPENVINO_EXTRACT_TMP%".
        exit /b 1
    )
    if not exist "%OPENVINO_INSTALL_DIR%" mkdir "%OPENVINO_INSTALL_DIR%"
    xcopy /e /i /y /q "!OPENVINO_EXTRACTED!\*" "%OPENVINO_INSTALL_DIR%\" >nul
    if errorlevel 1 (
        echo ERROR: Failed to copy OpenVINO from "!OPENVINO_EXTRACTED!" to "%OPENVINO_INSTALL_DIR%".
        echo Re-run this script from an elevated Command Prompt ^(Run as administrator^) if access is denied.
        exit /b 1
    )

    rmdir /s /q "%OPENVINO_EXTRACT_TMP%"
    del "%OPENVINO_ZIP%"
)

REM Refresh junction: C:\Intel\openvino -> C:\Intel\openvino_<version>.
REM `mklink /J` creates a directory junction (no admin / Developer Mode required).
if exist "%OPENVINO_LINK_DIR%" rmdir "%OPENVINO_LINK_DIR%"
mklink /J "%OPENVINO_LINK_DIR%" "%OPENVINO_INSTALL_DIR%" >nul
if errorlevel 1 (
    echo ERROR: Failed to create junction "%OPENVINO_LINK_DIR%" -^> "%OPENVINO_INSTALL_DIR%".
    echo If "%OPENVINO_LINK_DIR%" already exists as a regular non-empty folder, remove it manually and re-run.
    exit /b 1
)

set "OPENVINO_ROOT=%OPENVINO_LINK_DIR%"
echo OpenVINO Ready: %OPENVINO_ROOT% -^> %OPENVINO_INSTALL_DIR%


echo ============================================
echo Setting up compiler environment...
echo ============================================
REM Locate Visual Studio Build Tools vcvars64.bat
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products Microsoft.VisualStudio.Product.BuildTools -property installationPath`) do (
        set "VS_PATH=%%i"
    )
)
if defined VS_PATH (
    call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
) else (
    echo WARNING: Visual Studio Build Tools not found. Compiler may be missing.
)

REM ============================================
REM Clean old build cache
REM ============================================
if exist "build\ReleaseOV" (
    echo Removing old build directory ...
    rmdir /s /q "build\ReleaseOV"
)

echo ============================================
echo Configuring with CMake...
echo ============================================
call "%OPENVINO_ROOT%\setupvars.bat" >nul 2>nul

cmake -B build\ReleaseOV -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_OPENVINO=ON ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake"

if errorlevel 1 (
    echo If you continue to face CMAKE errors, make sure to install:
    echo   winget install Microsoft.VisualStudio.2022.BuildTools
    echo   Then run the "Developer Command Prompt for VS 2022" and launch this script from there.
    exit /b 1
)

cmake --build build\ReleaseOV --config Release
if errorlevel 1 exit /b 1

echo ============================================
echo Build completed successfully!
echo ============================================
echo Binaries: %CD%\build\ReleaseOV\bin
echo.
echo NOTE: To run, source setupvars.bat and pick a device:
echo   call "C:\Intel\openvino\setupvars.bat"
echo   set GGML_OPENVINO_DEVICE=CPU   ^&^& REM or GPU / NPU
echo   build\ReleaseOV\bin\llama-cli.exe -m model.gguf
echo.

endlocal
```

> [!NOTE]
> The script pins OpenVINO `2026.4.1` via the `OPENVINO_VERSION_MAJOR` / `OPENVINO_VERSION_FULL` variables at the top — edit them to track a different release. From any new shell, source the matching `setupvars` script via the junction — `call "C:\Intel\openvino\setupvars.bat"` from `cmd`, or `& "C:\Intel\openvino\setupvars.ps1"` from PowerShell. If `winget` cannot register Visual Studio Build Tools on first run, install them once manually and re-run the script from an elevated **Developer Command Prompt for VS 2022**.

</details>

### 3. Download Sample Model

Download a sample model for testing.

```bash
mkdir -p ~/models/
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     -O ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

<details>
<summary><b>Windows</b></summary>

```powershell
# Windows PowerShell
mkdir C:\models
Invoke-WebRequest -Uri https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -OutFile C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

```cmd
# Windows Command Line
mkdir C:\models
curl -L https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -o C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

</details>

### 4. Run Inference with OpenVINO Backend

When using the OpenVINO backend, the first inference token may have slightly higher latency due to on-the-fly conversion to the OpenVINO graph. Subsequent tokens and runs will be faster.

> [!NOTE]
> Default context size is set to the model training context, which may be very large. For example, 131072 for Llama 3.2 1B, which may result in lower performance, especially on edge/laptop devices. Use `-c` to limit context size in supported llama.cpp tools for better performance:
> - CPU and GPU: for example `-c 1024`.
> - NPU: keep the context small, for example `-c 512`, or `-c 256` for larger models. Decode may slow down as the context grows.

```bash
# If the device is unset, the CPU is used.
# If the system has multiple GPUs, use GPU.0 or GPU.1 to explicitly target a specific GPU.

export GGML_OPENVINO_DEVICE=GPU
# To run llama-simple:
./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 50 "The story of AI is "
# To run in chat mode:
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 1024
# To run llama-bench:
GGML_OPENVINO_DEVICE=GPU ./build/ReleaseOV/bin/llama-bench -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# NPU: keep context small to avoid failures from very large model context windows.
export GGML_OPENVINO_DEVICE=NPU
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 512
```

<details>
<summary><b>Windows</b></summary>

```powershell
# Windows Command Line
set GGML_OPENVINO_DEVICE=GPU
# Windows PowerShell
$env:GGML_OPENVINO_DEVICE = "GPU"

# To run llama-simple
build\ReleaseOV\bin\llama-simple.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -n 50 "The story of AI is "
# To run in chat mode:
build\ReleaseOV\bin\llama-cli.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -c 1024
# To run llama-bench:
build\ReleaseOV\bin\llama-bench.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# NPU: keep context small to avoid failures from very large model context windows.
# Windows Command Line
set GGML_OPENVINO_DEVICE=NPU
# Windows PowerShell
$env:GGML_OPENVINO_DEVICE = "NPU"
build\ReleaseOV\bin\llama-cli.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -c 512
```

</details>

> [!NOTE]
> - Select the OpenVINO device with `GGML_OPENVINO_DEVICE`, not with `-dev`.
> - On systems with multiple GPUs, use `GPU.0` or `GPU.1` to explicitly target specific GPU.
> - To split a model across several GPUs, set a comma-separated list, e.g. `GPU.0,GPU.1`, and use `-sm layer`.
> - A device that is not available is an error (no fallback to CPU). The error message lists the available OpenVINO devices with their names.
> - Run `llama-cli --list-devices` to see the valid values: each OpenVINO device shows the `GGML_OPENVINO_DEVICE=<value>` to set, and `(selected)` marks the active one.
> - See [OpenVINO GPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) for more details.

### 5. Docker Build

You can build and run llama.cpp with OpenVINO backend using Docker.

<details>
<summary><b>Build the images</b></summary>

```bash
# Build the base runtime image with compiled shared libraries and minimal dependencies.
docker build -t llama-openvino:base -f .devops/openvino.Dockerfile .

# Build the complete image with all binaries, Python tools, gguf-py library, and model conversion utilities.
docker build --target=full -t llama-openvino:full -f .devops/openvino.Dockerfile .

# Build a minimal CLI-only image containing just the llama-cli executable.
docker build --target=light -t llama-openvino:light -f .devops/openvino.Dockerfile .

# Builds a server-only image with llama-server executable, health check endpoint, and REST API support.
docker build --target=server -t llama-openvino:server -f .devops/openvino.Dockerfile .

# If you are behind a proxy:
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --target=server -t llama-openvino:server -f .devops/openvino.Dockerfile .
```

</details>

<details>
<summary><b>Run a container</b> (CPU, GPU, NPU)</summary>

Run llama.cpp with OpenVINO backend Docker container. Save sample models in `~/models` as [shown above](#3-download-sample-model). It will be mounted to the container in the examples below.

```bash
#  Run Docker container
docker run --rm -it -v ~/models:/models llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# With Intel GPU access (iGPU or dGPU)
docker run --rm -it -v ~/models:/models \
--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
--env=GGML_OPENVINO_DEVICE=GPU \
llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# With Intel NPU access
docker run --rm -it -v ~/models:/models \
--device=/dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
--env=GGML_OPENVINO_DEVICE=NPU \
llama-openvino:light --no-warmup -c 512 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

</details>

<details>
<summary><b>Run llama-server</b> (container or local build) and test it</summary>

> [!NOTE]
> `llama-server` with OpenVINO backend supports only one chat session/thread, when `GGML_OPENVINO_STATEFUL_EXECUTION=1` is enabled.

```bash
# Run the llama-openvino:server Docker container (CPU)
docker run --rm -it -p 8080:8080 -v ~/models:/models llama-openvino:server --no-warmup -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 1024 --host 0.0.0.0

# Run the llama-openvino:server Docker container with Intel GPU access (iGPU or dGPU)
docker run --rm -it -v ~/models:/models \
--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8080:8080 --env=GGML_OPENVINO_DEVICE=GPU  \
llama-openvino:server --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --host 0.0.0.0

# Run the llama-openvino:server Docker container with Intel NPU access
docker run --rm -it -v ~/models:/models \
--device=/dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8080:8080 --env=GGML_OPENVINO_DEVICE=NPU \
llama-openvino:server --no-warmup -c 512 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --host 0.0.0.0

# Or Using llama-server executable
./build/ReleaseOV/bin/llama-server -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8080 -c 1024

# Option 1: Open your browser to http://localhost:8080 to access the web UI for the llama.cpp server.
# Option 2: In a NEW terminal, test the server with curl

# If you are behind a proxy, make sure to set NO_PROXY to avoid proxy for localhost
export NO_PROXY=localhost,127.0.0.1

# Test health endpoint
curl -f http://localhost:8080/health

# Test with a simple prompt
curl -X POST "http://localhost:8080/v1/chat/completions" -H "Content-Type: application/json" \
 -d '{"messages":[{"role":"user","content":"Write a poem about OpenVINO"}],"max_tokens":100}' | jq .
```

</details>

## GGML OpenVINO Backend Runtime Configurations

The backend needs no environment variable: it runs on the CPU with its defaults. Set `GGML_OPENVINO_DEVICE` to use the GPU or the NPU. All other variables are optional.

```bash
export GGML_OPENVINO_DEVICE=GPU            # String
export GGML_OPENVINO_PROFILING=1           # Integer
export GGML_OPENVINO_EXACT_ZERO_POINT=1    # Boolean: 1 (on) or 0 (off)

# Show the variables in effect (log verbosity 4)
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 1024 -lv 4 2>&1 | grep "settings from the environment"
```

- The backend reads the variables once, at start. Unset or empty means the default.
- A value that is not valid stops the backend. A `GGML_OPENVINO_*` name that the backend does not know gives a warning:

```text
ggml-openvino: GGML_OPENVINO_EXACT_ZERO_POINT=yes is not valid, use 1 (on) or 0 (off)
ggml-openvino: unknown environment variable GGML_OPENVINO_DEVCE is ignored
```

### Basic

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_DEVICE` | all | String | `CPU` | Target device: `CPU`, `GPU` or `NPU`. With several GPUs, `GPU.0` or `GPU.1` selects one, and a comma-separated GPU list splits the model across them with `-sm layer`. A device that is not available is an error (no fallback to the CPU); the message lists the available devices. See [OpenVINO GPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html). Examples: `GGML_OPENVINO_DEVICE=NPU`, `GGML_OPENVINO_DEVICE=GPU.0,GPU.1`. |
| `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` | all | String | `not set` | Directory for the backend's cache of compiled models. A later start with the same model and settings loads faster. See [Compiled model cache](#compiled-model-cache). Example: `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/tmp/ov_cache`. |
| `GGML_OPENVINO_NPU_CONFIG` | NPU | String | `not set` | OpenVINO properties for the NPU plugin, as comma-separated `KEY=VALUE` pairs. See [Plugin properties](#plugin-properties). Example: `GGML_OPENVINO_NPU_CONFIG="NPU_COMPILER_TYPE=PLUGIN"`. |
| `GGML_OPENVINO_GPU_CONFIG` | GPU | String | `not set` | OpenVINO properties for the GPU plugin, as comma-separated `KEY=VALUE` pairs. See [Plugin properties](#plugin-properties). Example: `GGML_OPENVINO_GPU_CONFIG="INFERENCE_PRECISION_HINT=f16"`. |
| `GGML_OPENVINO_CPU_CONFIG` | CPU | String | `not set` | OpenVINO properties for the CPU plugin, as comma-separated `KEY=VALUE` pairs. See [Plugin properties](#plugin-properties). Example: `GGML_OPENVINO_CPU_CONFIG="INFERENCE_PRECISION_HINT=f16"`. |
| `GGML_OPENVINO_PROFILING` | all | Integer | `0` | `1` logs execution timing. `2` or higher also turns on OpenVINO and OpenCL profiling. Example: `GGML_OPENVINO_PROFILING=1`. |

### Plugin properties

`GGML_OPENVINO_NPU_CONFIG`, `GGML_OPENVINO_GPU_CONFIG` and `GGML_OPENVINO_CPU_CONFIG` pass OpenVINO plugin properties to the device in their name.

```bash
# Format: comma-separated KEY=VALUE pairs. The backend uses the variable of the selected device only.
export GGML_OPENVINO_DEVICE=NPU
export GGML_OPENVINO_NPU_CONFIG="NPU_COMPILER_TYPE=PLUGIN,COMPILATION_NUM_THREADS=4"

# CPU: compute in FP16, as the GPU and the NPU do. This helps to debug the accuracy of a quantized model.
export GGML_OPENVINO_DEVICE=CPU
export GGML_OPENVINO_CPU_CONFIG="INFERENCE_PRECISION_HINT=f16"
```

- Property names and values are those of OpenVINO, for example `YES` and `NO`. They are not the `1` and `0` of the `GGML_OPENVINO_*` variables.
- The backend applies the variable last, so it overrides the properties that the backend sets.
- OpenVINO checks the properties when the model loads. If it rejects one, the load fails and the error names the variable.
- All properties and their values are in the OpenVINO documentation: [NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html), [GPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html), [CPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html).
- NPU: the backend sets several properties by default. They are listed in the [developer documentation](../../ggml/src/ggml-openvino/dev_doc/OpenVINO-ENV-VARS.md#npu-plugin-properties).
- GPU and CPU: the backend sets no property, except `CACHE_DIR` and `CACHE_MODE=OPTIMIZE_SIZE` when `GGML_OPENVINO_CACHE_DIR` is set.

`NPU_COMPILER_TYPE` selects the NPU compiler. Both forms below work; use the first one. If both are set, `GGML_OPENVINO_NPU_CONFIG` takes precedence.

```bash
# Values: DRIVER (the compiler in the NPU driver; backend default), PLUGIN (the compiler in OpenVINO),
# PREFER_PLUGIN (OpenVINO default: PLUGIN when it is available, DRIVER if not).
export GGML_OPENVINO_NPU_CONFIG="NPU_COMPILER_TYPE=PLUGIN"   # recommended form
export GGML_OPENVINO_NPU_COMPILER_TYPE=PLUGIN                # older name, still works
```

### Advanced tuning

The defaults fit most models. The variables below trade one property for another, for example speed for accuracy.

<details>
<summary><b>Advanced tuning variables</b></summary>

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_PREFILL_CHUNK_SIZE` | NPU | Integer | `256` | Number of tokens that the NPU processes in one prefill step. A larger value can make long prompts faster, and makes the first load longer. Zero or a negative value means the default. Example: `GGML_OPENVINO_PREFILL_CHUNK_SIZE=512`. |
| `GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT` | NPU | String | `q8_0_c` | How a `Q6_K` `token_embd.weight` is stored for the NPU: `q8_0_c` (8-bit, one scale per row), `f16`, `q4_0_128` (4-bit, one scale per 128 weights) or `q4_0_c` (4-bit, one scale per row). The 4-bit layouts use less memory at some cost in accuracy; `f16` uses more memory. Keep the default unless memory is tight. |
| `GGML_OPENVINO_GDN_DECOMPOSE` | all | Integer | auto | Translate `GATED_DELTA_NET` (Qwen3.5) into MatMul operations instead of the OpenVINO `GatedDeltaNet` op. Unset: on for the Lunar Lake NPU, off elsewhere. `0` turns it off, `1` turns it on for any device. Try `1` for Qwen3.5 on the Panther Lake NPU. |
| `GGML_OPENVINO_MANUAL_GQA_ATTN` | CPU, GPU | Boolean | on for GPU | Build grouped-query attention from separate MatMul operations. Unset: on for GPU, off for CPU and NPU. `1` turns it on, `0` turns it off. Try `1` on the CPU for faster decode. |
| `GGML_OPENVINO_REQUANT_KQUANT` | CPU, GPU | String | `not set` | Requantize `Q6_K` and `Q5_K` weights to a 4-bit layout instead of `Q8_0_C`: faster decode for lower accuracy. One of `q4_asym64`, `q4_asym64_all` (also `Q4_K`), `q4_sym128`, `q4_sym128_all` (also `Q4_K`) or `native` (no requantization). On GPU with `Q4_K_M` files use `q4_asym64_all`, because `q4_asym64` does not compile there. |
| `GGML_OPENVINO_EXACT_ZERO_POINT` | CPU, GPU | Boolean | `0` | Keep the exact zero point of asymmetric weights (`Q4_1`, `Q4_K`, `Q5_1`, `Q5_K`) instead of rounding it to an integer. This improves accuracy and lowers speed. Use it when accuracy matters more than speed. |

</details>

### Compiled model cache

The cache keeps compiled models on disk, so later starts of the same model load faster. Use the same model and the same settings in every run.

```bash
# CPU or GPU: the first run compiles the model and writes the cache. Later runs read it.
export GGML_OPENVINO_DEVICE=GPU
export GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/tmp/ov_cache
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 1024

# CPU or GPU, later runs (optional): require the cached model and skip the weight upload and the compilation.
export GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1

# NPU: add the weightless bank.
export GGML_OPENVINO_DEVICE=NPU
export GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/tmp/ov_cache
export GGML_OPENVINO_NPU_WEIGHTLESS_BANK=1
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 512

# NPU, later runs (optional): use less memory. The run fails if a cached model is missing.
export GGML_OPENVINO_SELF_CONTAINED_MMAP=1
```

<details>
<summary><b>Cache variables</b></summary>

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY` | CPU, GPU | Boolean | `0` | Require an existing cached model and skip the weight upload and the compilation. Needs mmap model loading (the default on Linux and Windows) and a model that runs fully on OpenVINO. |
| `GGML_OPENVINO_NPU_WEIGHTLESS_BANK` | NPU | Boolean | `0` | Requires `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR`. Write the prefill and decode models without weights, plus one shared weights file. Later runs read the weights from that file and keep one copy in memory. |
| `GGML_OPENVINO_SELF_CONTAINED_MMAP` | NPU | Boolean | `0` | Later runs only. Requires `GGML_OPENVINO_NPU_WEIGHTLESS_BANK=1` and an existing cache. Use the memory-mapped GGUF file as the model buffer, which lowers the host memory peak. Fails if a cached model is missing; run once without it to build the cache. |
| `GGML_OPENVINO_CACHE_DIR` | all | String | `not set` | Directory for the cache of the OpenVINO plugin itself. `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` is the recommended cache. |

- One directory can hold the cached models of different models and settings.
- NPU: each prefill length enters the cache in a run without `GGML_OPENVINO_SELF_CONTAINED_MMAP`.
- See [Compiled model cache](../../ggml/src/ggml-openvino/README.md) for the workflow and the restrictions.

</details>

### Deprecated

| Variable | Device | Type | Default | Description |
|---|---|---|---|---|
| `GGML_OPENVINO_STATEFUL_EXECUTION` | CPU, GPU | Boolean | `0` | **Deprecated**, to be removed. Keep the KV cache inside the OpenVINO model. Use the default, stateless execution (`0`). |

<details>
<summary>Limits of stateful execution</summary>

- CPU and GPU only; it has no effect on the NPU.
- One slot only (`-np 1`).
- State save and restore, sequence rewind, context shift and recurrent rollback are not supported.
- Qwen3.5 needs a graph that runs fully on OpenVINO, with model caching on.
- A prompt that starts at position 0 resets all states.

</details>

### Debugging and development

Debug output, switches that turn a default optimization off, experimental options and older variable names are in the [developer documentation](../../ggml/src/ggml-openvino/dev_doc/OpenVINO-ENV-VARS.md).

### Example Usage

#### GPU Inference with Profiling

```bash
# If the system has multiple GPUs, use GPU.0 or GPU.1 to explicitly target a specific GPU.

export GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=/tmp/ov_cache
export GGML_OPENVINO_PROFILING=1
export GGML_OPENVINO_DEVICE=GPU

./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 50 "The story of AI is "
```

<details>
<summary><b>Windows</b></summary>

```powershell
# Windows Command Line
set GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=C:\tmp\ov_cache
set GGML_OPENVINO_PROFILING=1
set GGML_OPENVINO_DEVICE=GPU

# Windows PowerShell
$env:GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR = "C:\tmp\ov_cache"
$env:GGML_OPENVINO_PROFILING = "1"
$env:GGML_OPENVINO_DEVICE = "GPU"

build\ReleaseOV\bin\llama-simple.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -n 50 "The story of AI is "
```

</details>

## Known Limitations

**General (all devices)**

- Llama.cpp OpenVINO backend currently supports a subset of GGML ops and text-only models. Unsupported ops or unsupported op shapes/cases fail during OpenVINO translation.
- Multimodal features (audio/image/video) are a work in progress.
- Embedding models: BERT-based models (`bge-m3`, `nomic-embed-text-v1.5`) fail on all devices. `Qwen3-Embedding-0.6B` runs on CPU and GPU. Reranking models are not validated.
- Llama.cpp tool coverage across CPU/GPU/NPU is not uniform.

**Tool-specific**

- `llama-cli --context-shift`: stateless only (`GGML_OPENVINO_STATEFUL_EXECUTION=0`). In stateful mode the KV cache is owned by the OpenVINO model and cannot be shifted externally.
- `llama-server`: only one chat session/thread when `GGML_OPENVINO_STATEFUL_EXECUTION=1`.

**GPU-specific**

- `llama-server -np > 1`: concurrent requests are batched together, which may slightly reduce per-request throughput.
- Multiple GPUs (`GGML_OPENVINO_DEVICE=GPU.0,GPU.1` with `-sm layer`):
  - `-sm tensor` is not supported.
  - Copies between GPUs go through host memory. There is no asynchronous copy and no peer-to-peer copy.
  - Only GPUs can be combined. A list with several NPUs, or with a GPU and the CPU or the NPU, is rejected.

**NPU-specific**

- Default context resolves to the model's training context (e.g. 131072 for Llama 3.2 1B), which can OOM or fail or degrade performance on NPU. Inspect the resolved value with `-lv 3`.
  - **Workaround:** Pass an explicit `-c <N>`, e.g. `-c 512`.
- NPU device uses a static graph with a fixed prefill chunk size (defaults to 256), configurable with `GGML_OPENVINO_PREFILL_CHUNK_SIZE`. Large prefill/batch settings may need tuning.
- `llama-server -np > 1` (multiple parallel sequences) is not supported.
- `llama-perplexity`: requires `-b 512` or smaller.
- Flash attention must stay on, which is the default (`-fa auto`). With `-fa off`, the run fails on the NPU.
- `llama-embedding` fails on the NPU.
- Lunar Lake NPU: decode slows down sharply when the context is large. The context size at which this starts depends on the model.
  - **Workaround:** Pass the smallest context that fits the use, e.g. `-c 512`, or `-c 256` for larger models.
- Panther Lake NPU: the load of a Qwen3.5 model can end with a segmentation fault in the NPU compiler of OpenVINO 2026.4.1.
  - **Workaround:** Set `GGML_OPENVINO_NPU_CONFIG=COMPILATION_NUM_THREADS=1`.
- Panther Lake NPU: `GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT=f16` and `q4_0_128` can give wrong output. Keep the default.
- Lunar Lake NPU: `granite-4.0-1b` gives wrong output. The activations of the model exceed the FP16 range, and the NPU computes in FP16. The model runs correctly on CPU and GPU.
- Lunar Lake NPU: `gemma-4-12B` does not run. The first inference fails with `ZE_RESULT_ERROR_UNINITIALIZED` from the NPU driver. The model runs on CPU and GPU.
- The first load of a model on the NPU compiles it for the device, which can take minutes for large models.
  - **Workaround:** Use the [compiled model cache](#compiled-model-cache).

> [!NOTE]
> The OpenVINO backend is actively under development. Fixes and improvements are underway, and this document will continue to be updated.

## Work in Progress

- Performance and memory optimizations
- Accuracy validation
- Broader quantization coverage
- Support for additional model architectures
