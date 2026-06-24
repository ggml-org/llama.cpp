# OpenVINO Backend for llama.cpp

> [!NOTE]
> Performance and memory optimizations, accuracy validation, broader quantization coverage, broader operator and model support are work in progress.

[OpenVINO](https://docs.openvino.ai/) is an open-source toolkit for optimizing and deploying high-performance AI inference, specifically designed for Intel hardware, including CPUs, GPUs, and NPUs, in the cloud, on-premises, and on the edge. [OpenVINO backend for llama.cpp](../../ggml/src/ggml-openvino) enables hardware-accelerated inference on **Intel® CPUs, GPUs, and NPUs** while remaining compatible with the existing **GGUF model ecosystem**. The backend translates GGML compute graphs into OpenVINO graphs and leverages graph compilation, kernel fusion, and device-specific optimizations to improve inference performance on supported Intel hardware.

The OpenVINO backend is implemented in `ggml/src/ggml-openvino` and provides a translation layer for core GGML operations. The OpenVINO backend replaces the standard GGML graph execution path with Intel's OpenVINO inference engine. This approach allows the same GGUF model file to run on Intel CPUs, Intel GPUs (integrated and discrete), and Intel NPUs without changes to the model or the rest of the llama.cpp stack. When a `ggml_cgraph` is dispatched to OpenVINO backend, it:

- Walks the GGML graph and identifies inputs, outputs, weights, and KV cache tensors.
- Translates the GGML operations into an `ov::Model` using OpenVINO's frontend API.
- Compiles and caches the model for the target device.
- Binds GGML tensor memory to OpenVINO inference tensors and runs inference.

## Contents

- [Supported Devices](#supported-devices)
- [Supported Model Precisions](#supported-model-precisions)
- [Supported Llama.cpp Tools](#supported-llamacpp-tools)
- [Validated Models](#validated-models)
- [Build Instructions](#build-instructions)
  - [0. Prerequisites](#0-prerequisites)
  - [1. Install OpenVINO Runtime](#1-install-openvino-runtime)
  - [2. Build llama.cpp with OpenVINO Backend](#2-build-llamacpp-with-openvino-backend)
    - [Automated Ubuntu Build Script](#automated-ubuntu-build-script)
    - [Automated Windows Build Script](#automated-windows-build-script)
  - [3. Download Sample Model](#3-download-sample-model)
  - [4. Run Inference with OpenVINO Backend](#4-run-inference-with-openvino-backend)
  - [5. Docker Build](#5-docker-build)
- [GGML OpenVINO Backend Runtime Configurations](#ggml-openvino-backend-runtime-configurations)
- [Known Limitations](#known-limitations)
- [Work in Progress](#work-in-progress)

## Supported Devices

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel GPUs (integrated and discrete)
- Intel NPUs

Although OpenVINO supports a wide range of [Intel hardware](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html), the llama.cpp OpenVINO backend has been validated specifically on AI PCs such as the Intel® Core™ Ultra Series 1 and Series 2.

## Supported Model Precisions

- `FP16`
- `BF16` (on Intel Xeon)
- `Q8_0`
- `Q4_0`
- `Q4_1`
- `Q4_K`
- `Q4_K_M`
- `Q5_K` (converted to `Q8_0_C` at runtime)
- `Q6_K` (converted to `Q8_0_C` at runtime)

> [!NOTE]
> Accuracy validation and performance optimizations for quantized models are a work in progress.

**CPU and GPU Quantization Details:**
- `Q5_K` and `Q6_K` tensors are converted to `Q8_0_C`

**NPU Quantization Details:**
- Primary supported quantization scheme is `Q4_0`
- `Q6_K` tensors are requantized to `Q4_0_128` in general. For embedding weights, `Q6_K` tensors are requantized to `Q8_0_C` except for the token embedding matrix which is dequantized to fp16

**Additional Notes:**
- Both `Q4_0` and `Q4_1` models use `Q6_K` for the token embedding tensor and the final matmul weight tensor (often the same tensor)
- `Q4_0` models may produce some `Q4_1` tensors if an imatrix is provided during quantization using `llama-quantize`
- `Q4_K_M` models may include both `Q6_K` and `Q5_K` tensors (observed in Phi-3)
- `Q5_1` tensors are dequantized natively (weights, scales, and zero-points extracted directly)

## Supported Llama.cpp Tools

The OpenVINO backend integrates with the standard llama.cpp tools listed below.
However, all the tools coverage across all devices is not uniform and exhaustive validation is work in progress.

- llama-bench
- llama-cli
- llama-completion
- llama-embedding
- llama-perplexity
- llama-run
- llama-server
- llama-simple

## Validated Models

Although, the validated models below were tested with `llama-cli` using the `Q4_K_M` quantization format on Intel® Core™ Ultra Series 2 (Lunar Lake), the OpenVINO backend is expected to work across a broader range of [Intel hardware](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html), [supported model precisions](#supported-model-precisions), [supported llama.cpp tools](#supported-llamacpp-tools) and additional model architectures.

> [!NOTE]
> Extensive accuracy validation, performance optimizations, and broader architecture coverage are work in progress.

**Legend & Test Configuration:**
- **Status:** ✓ = Passed | ✗ = Failed or Unsupported
- **Execution Modes:**
  - **SL** = Stateless (`GGML_OPENVINO_STATEFUL_EXECUTION=0`)
  - **SF** = Stateful (`GGML_OPENVINO_STATEFUL_EXECUTION=1`)
  - Note: The NPU operates in stateless mode only.
- **Validation system:** Intel® Core™ Ultra 5 238V (Lunar Lake) | 32 GB RAM | Ubuntu 24.04 | Intel OpenCL GPU Driver 26.18.38308.1 | Intel NPU Driver 1.33.0.
- See [Known Limitations](#known-limitations) for context on observed failures.

| Model | CPU (SL / SF) | GPU (SL / SF) | NPU (SL) |
| :--- | :---: | :---: | :---: |
| [bartowski/Llama-3.2-1B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [bartowski/Llama-3.2-3B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [bartowski/Meta-Llama-3.1-8B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
|  |  |  |  |
| [Qwen/qwen2.5-1.5b-instruct-q4_k_m](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [Qwen/qwen2.5-coder-7b-instruct-q4_k_m](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [bartowski/Qwen_Qwen3-0.6B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [bartowski/Qwen_Qwen3-1.7B-Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [Qwen/Qwen3-4B-Q4_K_M](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [lm-kit/Qwen3-8B-Q4_K_M](https://huggingface.co/lm-kit/qwen-3-8b-instruct-gguf) | ✓ / ✓ | ✓ / ✗ | ✓ |
|  |  |  |  |
| [unsloth/gemma-3-4b-it-Q4_K_M](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [bartowski/google_gemma-4-E2B-it-Q4_K_M](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF) | ✓ / ✗ | ✓ / ✗ | ✓ |
| [bartowski/google_gemma-4-E4B-it-Q4_K_M](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF) | ✓ / ✗ | ✓ / ✗ | ✓ |
| [bartowski/gemma-4-12B-it-Q4_K_M](https://huggingface.co/bartowski/gemma-4-12B-it-GGUF) | ✓ / ✗ | ✓ / ✗ | ✗ |
|  |  |  |  |
| [bartowski/Phi-3-mini-4k-instruct-Q4_K_M](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [bartowski/Phi-3.5-mini-instruct-Q4_K_M](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
|  |  |  |  |
| [bartowski/Mistral-7B-Instruct-v0.3-Q4_K_M](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [QuantFactory/Ministral-3b-instruct.Q4_K_M](https://huggingface.co/QuantFactory/Ministral-3b-instruct-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [bartowski/Ministral-8B-Instruct-2410-Q4_K_M](https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
|  |  |  |  |
| [bartowski/DeepSeek-R1-Distill-Llama-8B-Q4_K_M](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [bartowski/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
|  |  |  |  |
| [ibm-granite/granite-4.0-350m-Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF) | ✓ / ✓ | ✗ / ✗ | ✓ |
| [ibm-granite/granite-4.0-micro-Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [ibm-granite/granite-4.0-1b-Q4_K_M](https://huggingface.co/ibm-granite/granite-4.0-1b-GGUF) | ✓ / ✓ | ✗ / ✗ | ✗ |
| [ibm-research/granite-3.2-8b-instruct-Q4_K_M](https://huggingface.co/ibm-research/granite-3.2-8b-instruct-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
|  |  |  |  |
| [HuggingFaceTB/smollm2-1.7b-instruct-q4_k_m](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✓ | ✓ |
| [openbmb/MiniCPM-V-2_6-Q4_K_M](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [bartowski/tencent_Hunyuan-7B-Instruct-Q4_K_M](https://huggingface.co/bartowski/tencent_Hunyuan-7B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-Q4_K_M](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
| [bartowski/prism-ml_Bonsai-8B-unpacked-Q4_K_M](https://huggingface.co/bartowski/prism-ml_Bonsai-8B-unpacked-GGUF) | ✓ / ✓ | ✓ / ✗ | ✓ |
|  |  |  |  |
| [gpustack/bge-m3-Q4_K_M.gguf](https://huggingface.co/gpustack/bge-m3-GGUF) | ✓ | ✗ | ✗ |

## Build Instructions

### 0. Prerequisites

- Linux or Windows system with Intel hardware (CPU, GPU, or NPU)
- **For Intel GPU or NPU Usage**: Install the appropriate hardware drivers for your Intel GPU or NPU. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html).

- **Linux:**
    - Git, CMake, and Ninja software tools are needed for building.
    ```bash
      sudo apt-get update
      sudo apt-get install -y build-essential libcurl4-openssl-dev libtbb12 cmake ninja-build python3-pip curl wget tar
    ```
    - OpenCL
    ```bash
      sudo apt install ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd
    ```

- **Windows:**
  - Download and install [Microsoft Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe). During installation, select the **"Desktop development with C++"** workload.

  - Install required tools:
    ```powershell
    # Windows PowerShell
    winget install Git.Git
    winget install GNU.Wget
    winget install Ninja-build.Ninja
    ```

  - Install **OpenCL** using **vcpkg**:
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

### 1. Install OpenVINO Runtime

- Follow the guide to install OpenVINO Runtime from an archive file: [Linux](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html) | [Windows](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html)

- Verify OpenVINO is initialized properly:
  ```bash
  echo $OpenVINO_DIR
  ```

### 2. Build llama.cpp with OpenVINO Backend

Clone llama.cpp repo and build :

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

- **Linux:**
```bash
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

- **Windows:** Open a **Developer Command Prompt for VS 2022** (so the MSVC toolchain is on `PATH`), then run:

```cmd
C:\Intel\openvino\setupvars.bat
cmake -B build\ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build\ReleaseOV --parallel
```

> [!NOTE]
> The Windows install path is `C:\Intel\openvino` (no spaces) to avoid quoting problems some CMake/Ninja toolchains have with `C:\Program Files (x86)\...`. Adjust to wherever you installed OpenVINO Runtime. From `cmd`, run `C:\Intel\openvino\setupvars.bat`; from PowerShell, run `& "C:\Intel\openvino\setupvars.ps1"` instead. Once the build is finished you can launch the binaries from any `cmd` or `PowerShell` window after sourcing the matching `setupvars` script for that shell.

#### Automated Ubuntu Build Script

For Ubuntu24 users, the following shell script automates the prerequisite installs (build tools, OpenCL ICD), the OpenVINO Runtime download/extract/setup, and the Ninja-based llama.cpp build.
Save the following as `ubuntu-llamacpp-ov-install.sh` next to where you want the `llama.cpp` folder to land, then run it:

```bash
chmod +x ubuntu-llamacpp-ov-install.sh
./ubuntu-llamacpp-ov-install.sh
```

<details>
<summary>Click to expand <code>ubuntu-llamacpp-ov-install.sh</code></summary>

```bash
#!/usr/bin/env bash
# ============================================
# llama.cpp OpenVINO Build Script (Ninja)
# ============================================
set -euo pipefail

OPENVINO_VERSION_MAJOR="2026.2.1"
OPENVINO_VERSION_FULL="2026.2.1.21919.ede283a88e3"

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
# shellcheck disable=SC1091
source "${OPENVINO_ROOT}/setupvars.sh"

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
> The script pins OpenVINO `2026.2.1` via the `OPENVINO_VERSION_MAJOR` / `OPENVINO_VERSION_FULL` variables at the top — edit them to track a different release.

</details>

#### Automated Windows Build Script

For Windows users, the following `.bat` script automates the prerequisite installs (Git, Ninja, CMake, Visual Studio 2022 Build Tools, vcpkg + OpenCL), the OpenVINO Runtime download/extract, and the Ninja-based llama.cpp build.
Save the following as `windows-llamacpp-ov-install.bat` next to where you want the `llama.cpp` to land, then run it from either **Command Prompt** or **PowerShell**:

```cmd
:: Command Prompt
windows-llamacpp-ov-install.bat
```

```powershell
# PowerShell
.\windows-llamacpp-ov-install.bat
```

<details>
<summary>Click to expand <code>windows-llamacpp-ov-install.bat</code></summary>

```bat
@echo off
setlocal enabledelayedexpansion

REM ============================================
REM llama.cpp OpenVINO Build Script (Ninja)
REM ============================================

set "OPENVINO_VERSION_MAJOR=2026.2.1"
set "OPENVINO_VERSION_FULL=2026.2.1.21919.ede283a88e3"

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
    REM NOTE: delayed expansion (!VAR!) is required because the surrounding else( ... )
    REM block is parsed once up-front, so %OPENVINO_EXTRACTED% would expand to "" here
    REM and xcopy would then treat "\*" as C:\* and fail with "Cannot perform a cyclic copy".
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
> The script pins OpenVINO `2026.2.1` via the `OPENVINO_VERSION_MAJOR` / `OPENVINO_VERSION_FULL` variables at the top — edit them to track a different release. From any new shell, source the matching `setupvars` script via the junction — `call "C:\Intel\openvino\setupvars.bat"` from `cmd`, or `& "C:\Intel\openvino\setupvars.ps1"` from PowerShell. If `winget` cannot register Visual Studio Build Tools on first run, install them once manually and re-run the script from an elevated **Developer Command Prompt for VS 2022**.

</details>


### 3. Download Sample Model

Download sample model for testing.

```bash
# Linux
mkdir -p ~/models/
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     -O ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Windows PowerShell
mkdir C:\models
Invoke-WebRequest -Uri https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -OutFile C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Windows Command Line
mkdir C:\models
curl -L https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -o C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

### 4. Run Inference with OpenVINO Backend

When using the OpenVINO backend, the first inference token may have slightly higher latency due to on-the-fly conversion to the OpenVINO graph. Subsequent tokens and runs will be faster.

> [!NOTE]
> Default context size is set to the model training context, which may be very large. For example, 131072 for Llama 3.2 1B, which may result in lower performance, especially on edge/laptop devices. Use `-c` to limit context size in supported llama.cpp tools for better performance. For example, `-c 512`.

```bash
# If device is unset or unavailable, defaults to CPU.
# If the system has multiple GPUs, use GPU.0 or GPU.1 to explicitly target a specific GPU.

# Linux
export GGML_OPENVINO_DEVICE=GPU
# Optional: enable stateful execution for improved GPU performance (recommended).
export GGML_OPENVINO_STATEFUL_EXECUTION=1
# To run llama-simple:
./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 50 "The story of AI is "
# To run in chat mode:
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 1024
# To run llama-bench, -fa 1 is needed
GGML_OPENVINO_STATEFUL_EXECUTION=1 GGML_OPENVINO_DEVICE=GPU ./build/ReleaseOV/bin/llama-bench -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -fa 1

# NPU: keep context small to avoid failures from very large model context windows.
export GGML_OPENVINO_DEVICE=NPU
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 512

# Windows Command Line
set GGML_OPENVINO_DEVICE=GPU
# Optional: enable stateful execution for improved GPU performance (recommended).
set GGML_OPENVINO_STATEFUL_EXECUTION=1
# Windows PowerShell
$env:GGML_OPENVINO_DEVICE = "GPU"
$env:GGML_OPENVINO_STATEFUL_EXECUTION = "1"

# To run llama-simple
build\ReleaseOV\bin\llama-simple.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -n 50 "The story of AI is "
# To run in chat mode:
build\ReleaseOV\bin\llama-cli.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -c 1024
# To run llama-bench, -fa 1 is needed
build\ReleaseOV\bin\llama-bench.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -fa 1

# NPU: keep context small to avoid failures from very large model context windows.
# Windows Command Line
set GGML_OPENVINO_DEVICE=NPU
# Windows PowerShell
$env:GGML_OPENVINO_DEVICE = "NPU"
build\ReleaseOV\bin\llama-cli.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -c 512
```
> [!NOTE]
> On systems with multiple GPUs, use `GPU.0` or `GPU.1` to explicitly target specific GPU. See [OpenVINO GPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) for more details.

### 5. Docker Build

You can build and run llama.cpp with OpenVINO backend using Docker.

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

Run llama.cpp with OpenVINO backend Docker container.
Save sample models in `~/models` as [shown above](#3-download-sample-model). It will be mounted to the container in the examples below.


```bash
#  Run Docker container
docker run --rm -it -v ~/models:/models llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# With Intel GPU access (iGPU or dGPU)
docker run --rm -it -v ~/models:/models \
--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
--env=GGML_OPENVINO_DEVICE=GPU --env=GGML_OPENVINO_STATEFUL_EXECUTION=1 \
llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# With Intel NPU access
docker run --rm -it -v ~/models:/models \
--device=/dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
--env=GGML_OPENVINO_DEVICE=NPU \
llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Run Llama.cpp Server with OpenVINO Backend.
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
llama-openvino:server --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --host 0.0.0.0

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

## GGML OpenVINO Backend Runtime Configurations

The OpenVINO backend can be configured using the following environment variables at runtime to control device selection, caching, debugging, and profiling behavior.
Boolean flags follow a uniform convention: set to a **positive integer** (e.g. `1`) to enable; unset, empty, `0`, negative, or non-numeric values are treated as disabled.

| Variable                          | Type      | Default    | Description                                                                                                 |
|-----------------------------------|-----------|------------|-------------------------------------------------------------------------------------------------------------|
| `GGML_OPENVINO_DEVICE`            | String    | `CPU`      | Specify the target device (CPU, GPU, NPU). On systems with multiple GPUs, use `GPU.0` or `GPU.1` to explicitly target specific GPU. See [OpenVINO GPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html). When set to **NPU**, static compilation mode is enabled for optimal performance. |
| `GGML_OPENVINO_CACHE_DIR`         | String    | `not set`  | Directory for OpenVINO model caching (recommended: `/tmp/ov_cache`). Enables model caching when set. **Not supported on NPU devices.** |
| `GGML_OPENVINO_PREFILL_CHUNK_SIZE`| Integer   | `256`      | Token chunk size for **NPU** prefill (NPU-only; ignored on CPU/GPU). Must be a positive integer; otherwise the default is used. |
| `GGML_OPENVINO_STATEFUL_EXECUTION`| Boolean   | `0`        | Enable stateful KV cache for better performance. Recommended on CPU, GPU.                                   |
| `GGML_OPENVINO_DISABLE_CACHE`     | Boolean   | `0`        | Disable the in-process compiled-model / decoder cache (cache is on by default). Set to `1` to disable.      |
| `GGML_OPENVINO_DISABLE_KV_SLICE`  | Boolean   | `0`        | Disable the KV-cache input-tensor slicing optimization (slicing is on by default on CPU/GPU). Set to `1` to disable. |
| `GGML_OPENVINO_MANUAL_GQA_ATTN`   | Boolean   | device-based | Tri-state. When **unset**, manual GQA attention is enabled by default on `GPU` and disabled on other devices. Set to a positive integer to force-enable, or `0` to force-disable. |
| `GGML_OPENVINO_PROFILING`         | Boolean   | `0`        | Enable execution-time profiling.                                                                            |
| `GGML_OPENVINO_DUMP_CGRAPH`       | Boolean   | `0`        | Dump the GGML compute graph to `cgraph_ov.txt`.                                                             |
| `GGML_OPENVINO_DUMP_IR`           | Boolean   | `0`        | Serialize OpenVINO IR files with timestamps.                                                                |
| `GGML_OPENVINO_DEBUG_INPUT`       | Boolean   | `0`        | Enable input debugging and print input tensor info.                                                         |
| `GGML_OPENVINO_DEBUG_OUTPUT`      | Boolean   | `0`        | Enable output debugging and print output tensor info.                                                       |
| `GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS` | Boolean | `0` | Print tensor address map once.                                                                           |

> [!NOTE]
>`GGML_OPENVINO_STATEFUL_EXECUTION` is an **Experimental** feature to allow stateful execution for managing the KV cache internally inside the OpenVINO model, improving performance on CPUs and GPUs. Stateful execution is not effective on NPUs, and not all models currently support this feature. This feature is experimental and has been validated only with the llama-simple, llama-cli, llama-bench, and llama-run applications and is recommended to enable for the best performance. Other applications, such as llama-server and llama-perplexity, are not yet supported.

### Example Usage

#### GPU Inference with Profiling

```bash
# If the system has multiple GPUs, use GPU.0 or GPU.1 to explicitly target a specific GPU.

# Linux
export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache
export GGML_OPENVINO_PROFILING=1
export GGML_OPENVINO_DEVICE=GPU
export GGML_OPENVINO_STATEFUL_EXECUTION=1

./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 50 "The story of AI is "

# Windows Command Line
set GGML_OPENVINO_CACHE_DIR=C:\tmp\ov_cache
set GGML_OPENVINO_PROFILING=1
set GGML_OPENVINO_DEVICE=GPU
set GGML_OPENVINO_STATEFUL_EXECUTION=1

# Windows PowerShell
$env:GGML_OPENVINO_CACHE_DIR = "C:\tmp\ov_cache"
$env:GGML_OPENVINO_PROFILING = "1"
$env:GGML_OPENVINO_DEVICE = "GPU"
$env:GGML_OPENVINO_STATEFUL_EXECUTION = "1"

build\ReleaseOV\bin\llama-simple.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" -n 50 "The story of AI is "

```

## Known Limitations

**General (all devices)**

- Llama.cpp OpenVINO backend currently supports a subset of GGML ops and text-only models. Unsupported ops or unsupported op shapes/cases fail during OpenVINO translation.
- Multimodal features (audio/image/video) are a work in progress.
- Limited Embedding and Reranking model support.
- Llama.cpp tool coverage across CPU/GPU/NPU is not uniform.

**Tool-specific**

- `llama-bench`: requires `-fa 1` (flash-attention).
- `llama-cli --context-shift`: stateless only (`GGML_OPENVINO_STATEFUL_EXECUTION=0`). In stateful mode the KV cache is owned by the OpenVINO model and cannot be shifted externally.
- `llama-server`: only one chat session/thread when `GGML_OPENVINO_STATEFUL_EXECUTION=1`.

**GPU-specific**

- `llama-server -np > 1`: concurrent requests are batched together, which may slightly reduce per-request throughput.

**NPU-specific**

- Default context resolves to the model's training context (e.g. 131072 for Llama 3.2 1B), which can OOM or fail or degrade performance on NPU. Inspect the resolved value with `-lv 3`.
  - **Workaround:** Pass an explicit `-c <N>`, e.g. `-c 1024`.
- NPU device uses a static graph with a fixed prefill chunk size (defaults to 256), configurable with `GGML_OPENVINO_PREFILL_CHUNK_SIZE`. Large prefill/batch settings may need tuning.
- `llama-server -np > 1` (multiple parallel sequences) is not supported.
- `llama-perplexity`: requires `-b 512` or smaller.

> [!NOTE]
> The OpenVINO backend is actively under development. Fixes and improvements are underway, and this document will continue to be updated.

## Work in Progress

- Performance and memory optimizations
- Accuracy validation
- Broader quantization coverage
- Support for additional model architectures
