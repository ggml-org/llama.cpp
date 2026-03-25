# OpenVINO Backend for llama.cpp

> [!NOTE]
> Performance and memory optimizations, accuracy validation, broader quantization coverage, broader operator and model support are work in progress.

[OpenVINO](https://docs.openvino.ai/) is an open-source toolkit for optimizing and deploying high-performance AI inference, specifically designed for Intel hardware, including CPUs, GPUs, and NPUs, in the cloud, on-premises, and on the edge. [OpenVINO backend for llama.cpp](../../src/ggml-openvino) enables hardware-accelerated inference on **Intel® CPUs, GPUs, and NPUs** while remaining compatible with the existing **GGUF model ecosystem**. The backend translates GGML compute graphs into OpenVINO graphs and leverages graph compilation, kernel fusion, and device-specific optimizations to improve inference performance on supported Intel hardware.

The OpenVINO backend is implemented in `ggml/src/ggml-openvino` and provides a translation layer for core GGML operations. The OpenVINO backend replaces the standard GGML graph execution path with Intel's OpenVINO inference engine. This approach allows the same GGUF model file to run on Intel CPUs, Intel GPUs (integrated and discrete), and Intel NPUs without changes to the model or the rest of the llama.cpp stack. When a `ggml_cgraph` is dispatched to OpenVINO backend, it:

- Walks the GGML graph and identifies inputs, outputs, weights, and KV cache tensors.
- Translates the GGML operations into an `ov::Model` using OpenVINO's frontend API.
- Compiles and caches the model for the target device.
- Binds GGML tensor memory to OpenVINO inference tensors and runs inference.

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
- `Q5_K` (converted to Q8_0_C at runtime)
- `Q6_K` (converted to Q8_0_C at runtime)

> [!NOTE]
> Accuracy validation and performance optimizations for quantized models are a work in progress.

## Quantization Support Details

### CPU and GPU

- **`Q4_0`, `Q4_1`, `Q4_K_M`, `Q6_K` models are supported**
- `Q5_K` and `Q6_K` tensors are converted to `Q8_0_C`

### NPU

- **Primary supported quantization scheme is `Q4_0`**
- `Q6_K` tensors are requantized to `Q4_0_128` in general. For embedding weights, `Q6_K` tensors are requantized to `Q8_0_C` except for the token embedding matrix which is dequantized to fp16

### Additional Notes

- Both `Q4_0` and `Q4_1` models use `Q6_K` for the token embedding tensor and the final matmul weight tensor (often the same tensor)
- `Q4_0` models may produce some `Q4_1` tensors if an imatrix is provided during quantization using `llama-quantize`
- `Q4_K_M` models may include both `Q6_K` and `Q5_K` tensors (observed in Phi-3)

## Validated Models

The following models have been validated for functionality on Intel® Core™ Ultra Series 1 and Series 2:

- [Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/)
- [Llama-3.1-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
- [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-S-1B-sft-gguf)
- [tencent/Hunyuan-7B-Instruct](https://huggingface.co/bartowski/tencent_Hunyuan-7B-Instruct-GGUF)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF)
- [bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF)

## Build Instructions

### Prerequisites

- Linux or Windows system with Intel hardware (CPU, GPU, or NPU)
- **For Intel GPU or NPU Usage**: Install the appropriate hardware drivers for your Intel GPU or NPU. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2025/get-started/install-openvino/configurations.html).

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

- **Linux:**

    <details>
    <summary>📦 Click to expand OpenVINO installation from an archive file on Ubuntu</summary>
    <br>

    ```bash
    wget https://raw.githubusercontent.com/ravi9/misc-scripts/main/openvino/ov-archive-install/install-openvino-from-archive.sh
    chmod +x install-openvino-from-archive.sh
    ./install-openvino-from-archive.sh
    ```

    Verify OpenVINO is initialized properly:
    ```bash
    echo $OpenVINO_DIR
    ```
    </details>


### 2. Build llama.cpp with OpenVINO Backend

Clone the OpenVINO-enabled llama.cpp fork and build it:

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

- **Windows:**
    ```cmd
    # x64 Native Tools Command Prompt for VS 2022
    "C:\Program Files (x86)\Intel\openvino_2026.0\setupvars.bat"
    cmake -B build\ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON -DLLAMA_CURL=OFF -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
    cmake --build build\ReleaseOV --parallel
    ```
> [!NOTE]
> Use `x64 Native Tools Command Prompt` for Windows build. After building, you could use either `cmd` or `PowerShell` to run the OpenVINO backend.

### 3. Download Sample Model

Download models for testing:

```bash
# Linux
mkdir -p ~/models/
wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf \
     -O ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf

# Windows PowerShell
mkdir C:\models
Invoke-WebRequest -Uri https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf -OutFile C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf

# Windows Command Line
mkdir C:\models
curl -L https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf -o C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf
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
# Enable stateful execution with GPU device to avoid known stateless execution failures.
export GGML_OPENVINO_STATEFUL_EXECUTION=1
# To run llama-simple:
./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -n 50 "The story of AI is "
# To run in chat mode:
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -c 1024
# To run llama-bench, -fa 1 is needed
GGML_OPENVINO_STATEFUL_EXECUTION=1 GGML_OPENVINO_DEVICE=GPU ./build/ReleaseOV/bin/llama-bench -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -fa 1

# NPU: keep context small to avoid failures from very large model context windows.
export GGML_OPENVINO_DEVICE=NPU
./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -c 512

# Windows Command Line
set GGML_OPENVINO_DEVICE=GPU
# Enable stateful execution with GPU device to avoid known stateless execution failures.
set GGML_OPENVINO_STATEFUL_EXECUTION=1
# Windows PowerShell
$env:GGML_OPENVINO_DEVICE = "GPU"
$env:GGML_OPENVINO_STATEFUL_EXECUTION = "1"

# To run llama-simple
build\ReleaseOV\bin\llama-simple.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf" -n 50 "The story of AI is "
# To run in chat mode:
build\ReleaseOV\bin\llama-cli.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf" -c 1024
# To run llama-bench, -fa 1 is needed
build\ReleaseOV\bin\llama-bench.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf" -fa 1

# NPU: keep context small to avoid failures from very large model context windows.
# Windows Command Line
set GGML_OPENVINO_DEVICE=NPU
# Windows PowerShell
$env:GGML_OPENVINO_DEVICE = "NPU"
build\ReleaseOV\bin\llama-cli.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf" -c 512
```
> [!NOTE]
> On systems with multiple GPUs, use `GPU.0` or `GPU.1` to explicitly target specific GPU. See [OpenVINO GPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) for more details.

### Known Issues and Current Workarounds

- GPU stateless execution is currently affected by a known issue.
  - Workaround: set `GGML_OPENVINO_STATEFUL_EXECUTION=1` when using GPU device.
- NPU failures can happen when context size is too large. Recent llama.cpp behavior may resolve context size to the model training context (for example, 131072 for Llama 3.2 1B), which is too large for current NPU usage and can also stress laptop CPU/GPU on larger models. To inspect the selected context size, run `llama-cli` or `llama-server` with `-lv 3`.
  - Workaround: explicitly set context size, for ex. `-c 1024` for NPU runs. Performance will be better with lower context size.
- Additional NPU limitations:
  - Model caching is not yet supported.
  - `llama-server -np > 1` (multiple parallel sequences) is not supported.
  - `llama-perplexity` is only supported with `-b 512` or smaller.
- `--context-shift` with `llama-cli` is currently not supported with OpenVINO backend across CPU, GPU, and NPU devices.
- Encoder models (embedding, reranking) are not supported with the current OpenVINO backend implementation.
- `-fa 1` is required when running llama-bench with the OpenVINO backend.
  - `GGML_OPENVINO_STATEFUL_EXECUTION=1 GGML_OPENVINO_DEVICE=GPU ./llama-bench -fa 1`
- `llama-server` with OpenVINO backend supports only one chat session/thread, when `GGML_OPENVINO_STATEFUL_EXECUTION=1` is enabled.
- For Intel GPU, NPU detection in containers, GPU, NPU user-space drivers/libraries must be present inside the image. We will include in a future PR. Until then, you can use this reference Dockerfile: [openvino.Dockerfile](https://github.com/ravi9/llama.cpp/blob/ov-docker-update/.devops/openvino.Dockerfile)

> [!NOTE]
> The OpenVINO backend is actively under development. Fixes are underway, and this document will continue to be updated as issues are resolved.


### Docker Build

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
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --target=light -t llama-openvino:light -f .devops/openvino.Dockerfile .
```

Run llama.cpp with OpenVINO backend Docker container.
Save sample models in `~/models` as [shown above](#3-download-sample-model). It will be mounted to the container in the examples below.

> [!NOTE]
> Intel GPU, NPU detection in containers will be included in a future PR. Until then, you can use this reference Dockerfile: [openvino.Dockerfile](https://github.com/ravi9/llama.cpp/blob/ov-docker-update/.devops/openvino.Dockerfile).

```bash
#  Run Docker container
docker run --rm -it -v ~/models:/models llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_0.gguf

# With Intel GPU access (iGPU or dGPU)
docker run --rm -it -v ~/models:/models \
--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
--env=GGML_OPENVINO_DEVICE=GPU --env=GGML_OPENVINO_STATEFUL_EXECUTION=1 \
llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_0.gguf

# With Intel NPU access
docker run --rm -it -v ~/models:/models \
--device=/dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
--env=GGML_OPENVINO_DEVICE=NPU \
llama-openvino:light --no-warmup -c 1024 -m /models/Llama-3.2-1B-Instruct-Q4_0.gguf
```

Run Llama.cpp Server with OpenVINO Backend.
> [!NOTE]
> `llama-server` with OpenVINO backend supports only one chat session/thread, when `GGML_OPENVINO_STATEFUL_EXECUTION=1` is enabled.

```bash
# Run the Server Docker container
docker run --rm -it -p 8080:8080 -v ~/models:/models llama-openvino:server --no-warmup -m /models/Llama-3.2-1B-Instruct-Q4_0.gguf -c 1024
# Or Using llama-server executable
./build/ReleaseOV/bin/llama-server -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf --port 8080 -c 1024

# If you are behind a proxy, make sure to set NO_PROXY to avoid proxy for localhost
export NO_PROXY=localhost,127.0.0.1

# Option 1: Open your browser to http://localhost:8080 to access the web UI for the llama.cpp server.
# Option 2: In a NEW terminal, test the server with curl

# Test health endpoint
curl -f http://localhost:8080/health

# Test with a simple prompt
curl -X POST "http://localhost:8080/v1/chat/completions" -H "Content-Type: application/json" \
 -d '{"messages":[{"role":"user","content":"Write a poem about OpenVINO"}],"max_tokens":100}' | jq .
```

## Runtime Configuration

The OpenVINO backend can be configured using the following environment variables at runtime to control device selection, caching, debugging, and profiling behavior.

### Configuration Options

| Variable                          | Default    | Description                                                                                                 |
|-----------------------------------|------------|-------------------------------------------------------------------------------------------------------------|
| `GGML_OPENVINO_DEVICE`            | `CPU`      | Specify the target device (CPU, GPU, NPU). On systems with multiple GPUs, use `GPU.0` or `GPU.1` to explicitly target specific GPU. See [OpenVINO GPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html). When set to **NPU**, static compilation mode is enabled for optimal performance. |
| `GGML_OPENVINO_CACHE_DIR`         | `not set`  | Directory for OpenVINO model caching (recommended: `/tmp/ov_cache`). Enables model caching when set. **Not supported on NPU devices.** |
| `GGML_OPENVINO_PREFILL_CHUNK_SIZE`| `256`      | Token chunk size for **NPU** prefill.                                                                       |
| `GGML_OPENVINO_STATEFUL_EXECUTION`| `0`        | Enable stateful KV cache on for better performance. Recommended on CPU, GPU.                                |
| `GGML_OPENVINO_PROFILING`         | `0`        | Enable execution-time profiling.                                                                            |
| `GGML_OPENVINO_DUMP_CGRAPH`       | `0`        | Dump the GGML compute graph to `cgraph_ov.txt`.                                                             |
| `GGML_OPENVINO_DUMP_IR`           | `0`        | Serialize OpenVINO IR files with timestamps.                                                                |
| `GGML_OPENVINO_DEBUG_INPUT`       | `0`        | Enable input debugging and print input tensor info.                                                         |
| `GGML_OPENVINO_DEBUG_OUTPUT`      | `0`        | Enable output debugging and print output tensor info.                                                       |
| `GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS` | `0` | Print tensor address map once.                                                                           |

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

./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -n 50 "The story of AI is "

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

build\ReleaseOV\bin\llama-simple.exe -m "C:\models\Llama-3.2-1B-Instruct-Q4_0.gguf" -n 50 "The story of AI is "

```

## Llama.cpp Tools

The following tools work with the OpenVINO backend on CPU, GPU, NPU:
- llama-bench
- llama-cli
- llama-completion
- llama-perplexity
- llama-server
- llama-simple

## Work in Progress

- Performance and memory optimizations
- Accuracy validation
- Broader quantization coverage
- Support for additional model architectures
