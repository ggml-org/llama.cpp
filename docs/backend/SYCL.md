# llama.cpp for SYCL

- [Background](#background)
- [Recommended Release](#recommended-release)
- [News](#news)
- [OS](#os)
- [Hardware](#hardware)
- [Performance Reference](#performance-reference)
- [Docker](#docker)
- [Linux](#linux)
- [Windows](#windows-1)
- [Environment Variable](#environment-variable)
- [Design Rule](#design-rule)
- [Known Issue](#known-issues)
- [Q&A](#qa)
- [TODO](#todo)

## Background

**SYCL** is a high-level parallel programming model designed to improve developers productivity writing code across various hardware accelerators such as CPUs, GPUs, and FPGAs. It is a single-source language designed for heterogeneous computing and based on standard C++17.

**oneAPI** is an open ecosystem and a standard-based specification, supporting multiple architectures including but not limited to Intel CPUs, GPUs and FPGAs. The key components of the oneAPI ecosystem include:

- **DPCPP** *(Data Parallel C++)*: The primary oneAPI SYCL implementation, which includes the icpx/icx Compilers.
- **oneAPI Libraries**: A set of highly optimized libraries targeting multiple domains *(e.g. Intel oneMKL, oneMath and oneDNN)*.
- **oneAPI LevelZero**: A high performance low level interface for fine-grained control over Intel iGPUs and dGPUs.

### Llama.cpp + SYCL

The llama.cpp SYCL backend is primarily designed for **Intel GPUs**.
SYCL cross-platform capabilities enable support for other vendor GPUs as well.

## Recommended Release

This fork publishes no release binaries; build from source ([Linux](#linux)). The upstream
ggml-org packages below do not contain the TurboQuant+ codec stack or this fork's SYCL changes.

### Windows

The following upstream releases are verified and recommended upstream:

|Commit ID|Tag|Release|Verified  Platform| Update date|
|-|-|-|-|-|
|24e86cae7219b0f3ede1d5abdf5bf3ad515cccb8|b5377 |[llama-b5377-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b5377/llama-b5377-bin-win-sycl-x64.zip) |Arc B580/Linux/oneAPI 2025.1<br>LNL Arc GPU/Windows 11/oneAPI 2025.1.1|2025-05-15|
|3bcd40b3c593d14261fb2abfabad3c0fb5b9e318|b4040 |[llama-b4040-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b4040/llama-b4040-bin-win-sycl-x64.zip) |Arc A770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1| 2024-11-19|
|fb76ec31a9914b7761c1727303ab30380fd4f05c|b3038 |[llama-b3038-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b3038/llama-b3038-bin-win-sycl-x64.zip) |Arc A770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1||

### Ubuntu 24.04

The upstream release packages for Ubuntu 24.04 x64 (FP32/FP16) only include the binary files of the llama.cpp SYCL backend. They require the target machine to have pre-installed Intel GPU drivers and oneAPI packages that are the same version as the build package. The version is set by upstream's `.github/workflows/release.yml` (job ubuntu-24-sycl); this fork carries no `.github/` directory.

It is recommended to use them with [Intel Docker](https://hub.docker.com/r/intel/deep-learning-essentials).

The packages for FP32 and FP16 would have different accuracy and performance on LLMs. Please choose it according to the test result.

## News

- 2026.09
  - Update the CI build environment for oneAPI 2026.1 (unified oneAPI Toolkit). oneDNN is removed from the Deep Learning Essentials package in 2026.0, so the CI now uses the oneAPI Toolkit installer which still includes oneDNN.
  - oneAPI 2026.1 improves the SYCL build performance: measured with the same code on Arc B570, prompt processing 1331 vs 434 t/s (3.1x) vs the 2025.3-based release build.

- 2026.04-05
  - Optimize mul_mat by reorder feature for data type: Q4_K, Q5_K, Q6_K, Q8_0.
  - Fused MoE.
  - Upgrate CI and built package for oneAPI 2025.3.3, support Ubuntu 24.04 built package.

- 2026.03
  - Support Flash-Attention: less memory usage, performance impact depends on LLM.

- 2026.02
  - Remove support for Nvidia & AMD GPU, because the oneAPI plugin for Nvidia & AMD GPU is unavailable: download/installation channels are out of work. User can't build up the software for Nvidia & AMD GPU.

- 2025.11
  - Support malloc memory on device more than 4GB.

- 2025.2
  - Optimize MUL_MAT Q4_0 on Intel GPU for all dGPUs and built-in GPUs since MTL. Increase the performance of LLM (llama-2-7b.Q4_0.gguf) 21%-87% on Intel GPUs (MTL, ARL-H, Arc, Flex, PVC).
    |GPU|Base tokens/s|Increased tokens/s|Percent|
    |-|-|-|-|
    |PVC 1550|39|73|+87%|
    |Flex 170|39|50|+28%|
    |Arc A770|42|55|+30%|
    |MTL|13|16|+23%|
    |ARL-H|14|17|+21%|

- 2024.11
  - Use syclcompat to improve the performance on some platforms. This requires to use oneAPI 2025.0 or newer.

- 2024.8
  - Use oneDNN as the default GEMM library, improve the compatibility for new Intel GPUs.

- 2024.5
  - Performance is increased: 34 -> 37 tokens/s of llama-2-7b.Q4_0 on Arc A770.
  - Arch Linux is verified successfully.

- 2024.4
  - Support data types: GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M.

- 2024.3
  - Release binary files of Windows.
  - A blog is published: **Run LLM on all Intel GPUs Using llama.cpp**: [intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llm-on-all-gpus-using-llama-cpp-artical.html) or [medium.com](https://medium.com/@jianyu_neo/run-llm-on-all-intel-gpus-using-llama-cpp-fd2e2dcbd9bd).
  - New base line is ready: [tag b2437](https://github.com/ggml-org/llama.cpp/tree/b2437).
  - Support multiple cards: **--split-mode**: [none|layer]; not support [row], it's on developing.
  - Support to assign main GPU by **--main-gpu**, replace $GGML_SYCL_DEVICE.
  - Support detecting all GPUs with level-zero and same top **Max compute units**.
  - Support OPs
    - hardsigmoid
    - hardswish
    - pool2d

- 2024.1
  - Create SYCL backend for Intel GPU.
  - Support Windows build

## OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Linux   | Support | Ubuntu 22.04, Fedora Silverblue 39, Arch Linux |
| Windows | Support | Windows 11                                     |

This fork is built and validated on Linux (Arch) only; the Windows instructions are carried
from upstream and are unverified here.

## Hardware

### Intel GPU

SYCL backend supports Intel GPU Family:

- Intel Data Center Max Series
- Intel Flex Series, Arc Series
- Intel Built-in Arc GPU
- Intel iGPU in Core CPU (11th Generation Core CPU and newer, refer to [oneAPI supported GPU](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html#inpage-nav-1-1)).

On older Intel GPUs, you may try the [Vulkan](../build/build.md#vulkan) backend (the OpenCL backend is not part of this fork), although the performance is not optimal, and some GPUs may not have any GPGPU capabilities.

#### Verified devices

| Intel GPU                     | Status  | Verified Model                        |
|-------------------------------|---------|---------------------------------------|
| Intel Data Center Max Series  | Support | Max 1550, 1100                        |
| Intel Data Center Flex Series | Support | Flex 170                              |
| Intel Arc A-Series            | Support | Arc A770, Arc A730M, Arc A750         |
| Intel Arc B-Series            | Support | Arc B580                              |
| Intel built-in Arc GPU        | Support | built-in Arc GPU in Meteor Lake, Arrow Lake, Lunar Lake |
| Intel iGPU                    | Support | iGPU in 13700k, 13400, i5-1250P, i7-1260P, i7-1165G7  |

This table is upstream's. This fork's canonical and routinely tested device is the Arc A770
(acm-g10, Xe-HPG).

*Notes:*

- **Memory**
  - The device memory is a limitation when running a large model. The loaded model size, *`llm_load_tensors: buffer_size`*, is displayed in the log when running `./bin/llama-completion`.
  - Please make sure the GPU shared memory from the host is large enough to account for the model's size. For e.g. the *llama-2-7b.Q4_0* requires at least 8.0GB for integrated GPU and 4.0GB for discrete GPU.

- **Execution Unit (EU)**
  - If the iGPU has less than 80 EUs, the inference speed will likely be too slow for practical use.

### Other Vendor GPU

NA

## Performance Reference


To get the supported LLMs, GPUs, and performance reference, please check [Performance of llama.cpp on Intel GPU with SYCL backend](https://github.com/ggml-org/llama.cpp/discussions/23313).

You could update your test result in it directly.

## Docker

This fork ships no Dockerfiles: the upstream `.devops/` directory, including the Intel/SYCL
image, is not in this tree (see [Docker](../build/docker.md)). Build from source.

## Quick Development WOW

This chapter is for quick development & try with SYCL backend on Intel GPU.

You need to install following sofeware before development:
   - Intel GPU driver
   - oneAPI package
   - other development tools.

Please refer to [Linux](#linux) or [Windows](#windows-1) for above installation and resolve the trouble in usage. There are the detailed guide.

- Linux

```
## build from source code
./examples/sycl/build.sh

## run CONV_2D_DW unit test cases
./build/bin/test-backend-ops -b SYCL0 -o CONV_2D_DW

## run all unit test cases
./build/bin/test-backend-ops -b SYCL0

## run with LLM on the first GPU
./examples/sycl/test.sh -mg 0 -m xxxx.gguf

## run service with LLM on the first GPU
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
./examples/sycl/start-svr.sh -m xxxx.gguf

## update the docs/ops.md for new/update OPs
./examples/sycl/update-ops-doc.sh
```

- Windows

```
## build from source code
examples\sycl\win-build-sycl.bat

## run CONV_2D_DW unit test cases
build\bin\test-backend-ops.exe -b SYCL0 -o CONV_2D_DW

## run all unit test cases
build\bin\test-backend-ops.exe -b SYCL0

## run LLM on the first GPU
examples\sycl\win-test.bat -mg 0 -m xxxx.gguf

## run service with LLM on the first GPU
set ONEAPI_DEVICE_SELECTOR="level_zero:0"
examples\sycl\win-start-svr.bat -m xxxx.gguf

## update the docs/ops.md for new/update OPs
examples\sycl\win-update-ops-doc.bat
```


## Linux

### I. Setup Environment

1. **Install GPU drivers**

  - **Intel GPU**

Intel data center GPUs drivers installation guide and download page can be found here: [Get Intel dGPU Drivers](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps).

*Note*: for client GPUs *(iGPU & Arc A-Series)*, please refer to the [client iGPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html).

Once installed, add the user(s) to the `video` and `render` groups.

```sh
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```

*Note*: logout/re-login for the changes to take effect.

Verify installation through `clinfo`:

```sh
sudo apt install clinfo
sudo clinfo -l
```

Sample output:

```sh
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics

Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Iris(R) Xe Graphics [0x9a49]
```

2. **Install Intel® oneAPI Toolkit**

SYCL backend depends on:
  - Intel® oneAPI DPC++/C++ compiler/running-time.
  - Intel® oneAPI DPC++/C++ library (oneDPL).
  - Intel® oneAPI Deep Neural Network Library (oneDNN).
  - Intel® oneAPI Math Kernel Library (oneMKL).

- **For Intel GPU**

With the 2026.0 release, the Intel® oneAPI Base toolkit and the HPC toolkit are combined into the **Intel® oneAPI Toolkit**, and **oneDNN is removed from the Intel® Deep Learning Essentials** package (oneDNN is distributed separately since then). The **Intel® oneAPI Toolkit** includes oneDNN until 2027.0.

It's recommended to install the **Intel® oneAPI Toolkit**.

oneAPI 2026.0 dropped oneDNN from the Deep Learning Essentials package; install the unified
**Intel® oneAPI Base toolkit** instead if oneDNN support (`GGML_SYCL_ENABLE_DNN`, the oneDNN FA/GEMM
paths) matters on 2026.0+. Confirmed on this fork: a Deep Learning Essentials-only 2026.0 install
builds with `GGML_SYCL_DNNL=0` and a "Disabling oneDNN support" CMake warning.

The **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Linux, and preferably keep the default installation values unchanged, notably the installation path *(`/opt/intel/oneapi` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

Upon a successful installation, SYCL is enabled for the available Intel devices, along with relevant libraries such as oneAPI oneDNN for Intel GPUs.

|Verified release|
|-|
|2026.1.4 (this fork, Arc A770, 2026-09)|
|2026.0 (this fork, Arc A770, 2026-07)|
|2025.3.3 |
|2025.2.1|
|2025.1|
|2024.1|

3. **Verify installation and environment**

In order to check the available SYCL devices on the machine, please use the `sycl-ls` command.
```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

- **Intel GPU**

When targeting an intel GPU, the user should expect one or more devices among the available SYCL devices. Please make sure that at least one GPU is present via `sycl-ls`, for instance `[level_zero:gpu]` in the sample output below:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.3.29735+27]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) UHD Graphics 730 12.2.0 [1.3.29735+27]
[opencl:cpu][opencl:0] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i5-13400 OpenCL 3.0 (Build 0) [2025.20.8.0.06_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.39.31294]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) UHD Graphics 730 OpenCL 3.0 NEO  [24.39.31294]
```

### II. Build llama.cpp

#### Intel GPU

```sh
# Uses FP32, consider using FP16 for better performance in most cases
./examples/sycl/build.sh
```

or

```sh
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP16 (recommended for better performance in most cases)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

# Option 2: Use FP32
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# build all binary
cmake --build build --config Release -j -v
```

Fork notes: `GGML_SYCL_TARGET=INTEL` is the only accepted target. Leaving
`GGML_SYCL_DEVICE_ARCH` empty gives the default JIT build; `-DGGML_SYCL_DEVICE_ARCH=acm-g10`
builds AOT for the A770 and took about 45 minutes clean on the fork's host
([build and runtime pins](../research/software-stack/sycl-build-runtime-pins.md)). Keep host `CFLAGS`/`CXXFLAGS`
out of the build (see [Known Issues](#known-issues)).

It is possible to come across some precision issues when running tests that stem from using faster
instructions, which can be circumvented by setting the environment variable `SYCL_PROGRAM_COMPILE_OPTIONS`
as `-cl-fp32-correctly-rounded-divide-sqrt`

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Obtaining and quantizing models*](../user/models.md) guide for model preparation, or download an already quantized model like [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf?download=true) or [Meta-Llama-3-8B-Instruct-Q4_0.gguf](https://huggingface.co/aptha/Meta-Llama-3-8B-Instruct-Q4_0-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf).

##### Check device

1. Enable oneAPI running environment

```sh
source /opt/intel/oneapi/setvars.sh
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```sh
./build/bin/llama-ls-sycl-device
```

This command will only display the selected backend that is supported by SYCL. The default backend is level_zero. For example, in a system with 2 *Intel GPU* it would look like the following:
```
found 2 SYCL devices:

|  |                  |                                             |Compute   |Max compute|Max work|Max sub|               |
|ID|       Device Type|                                         Name|capability|units      |group   |group  |Global mem size|
|--|------------------|---------------------------------------------|----------|-----------|--------|-------|---------------|
| 0|[level_zero:gpu:0]|               Intel(R) Arc(TM) A770 Graphics|       1.3|        512|    1024|     32|    16225243136|
| 1|[level_zero:gpu:1]|                    Intel(R) UHD Graphics 770|       1.3|         32|     512|     32|    53651849216|
```

#### Choose level-zero devices

|Chosen Device ID|Setting|
|-|-|
|0|`export ONEAPI_DEVICE_SELECTOR="level_zero:0"` or no action|
|1|`export ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|0 & 1|`export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"`|

#### Execute

Choose one of following methods to run.

1. Script

- Use device 0:

```sh
./examples/sycl/test.sh -mg 0
```
- Use multiple devices:

```sh
./examples/sycl/test.sh
```

- Run llama-server:

```sh
./examples/sycl/start-svr.sh -m PATH/MODEL_FILE
```

2. Command line
Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user. Default device id is 0.
- Multiple devices: Automatically choose the devices with the same backend.

In two device selection modes, the default SYCL backend is level_zero, you can choose other backend supported by SYCL by setting environment variable ONEAPI_DEVICE_SELECTOR.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |
| Multiple devices | --split-mode tensor (tensor parallelism) |

`--split-mode tensor` (tensor parallelism) shards each layer across the selected
GPUs. It requires flash attention, which is auto-enabled when `--flash-attn` is
left at its default `auto`, so `--split-mode tensor` works out of the box.
Passing `--flash-attn off` together with `--split-mode tensor` is rejected at
context creation. The default `f16` KV cache is recommended. Tensor parallelism
is currently optimized for 2 GPUs; other device counts fall back to a generic
all-reduce.

Examples:

- Use device 0:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-completion -no-cnv -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 99 -sm none -mg 0 --load-mode auto
```

- Use multiple devices:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-completion -no-cnv -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 99 -sm layer --load-mode auto
```

*Notes:*

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```
Or
```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```

User can use the device management in [docs/user/multi-gpu.md](../user/multi-gpu.md), like parameter `--device SYCL0,SYCL1` to assign one or more devices.

## Windows

### Install GPU driver

Intel GPU drivers instructions guide and download page can be found here: [Get Intel GPU Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html).

### Option 1: download the binary package directly

Download the binary package for Windows from: https://github.com/ggml-org/llama.cpp/releases.

Extract the package to local folder, run the llama tools directly. Refer to [Run the inference](#iii-run-the-inference-1).

Note, the package includes the SYCL running time and all depended dll files, no need to install oneAPI package and activte them.

### Option 2: build locally from the source code.

#### I. Setup environment

1. Install Visual Studio

If you already have a recent version of Microsoft Visual Studio, you can skip this step. Otherwise, please refer to the official download page for [Microsoft Visual Studio](https://visualstudio.microsoft.com/).

2. Install Intel® oneAPI Base toolkit

SYCL backend depends on:
  - Intel® oneAPI DPC++/C++ compiler/running-time.
  - Intel® oneAPI DPC++/C++ library (oneDPL).
  - Intel® oneAPI Deep Neural Network Library (oneDNN).
  - Intel® oneAPI Math Kernel Library (oneMKL).

All above are included in both **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** packages.

It's recommended to install **Intel® Deep Learning Essentials** which only provides the necessary libraries with less size.

The **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Windows, and preferably keep the default installation values unchanged, notably the installation path *(`C:\Program Files (x86)\Intel\oneAPI` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

b. Enable oneAPI running environment:

- Type "oneAPI" in the search bar, then open the `Intel oneAPI command prompt for Intel 64 for Visual Studio 2022` App.

- On the command prompt, enable the runtime environment with the following:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

- if you are using Powershell, enable the runtime environment with the following:

```
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```

c. Verify installation

In the oneAPI command line, run the following to print the available SYCL devices:

```
sycl-ls.exe
```

There should be one or more *level-zero* GPU devices displayed as **[ext_oneapi_level_zero:gpu]**. Below is example of such output detecting an *Intel Iris Xe* GPU as a Level-zero SYCL device:

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [31.0.101.5186]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.28044]
```

3. Install build tools

a. Download & install cmake for Windows: https://cmake.org/download/ (CMake can also be installed from Visual Studio Installer)
b. The new Visual Studio will install Ninja as default. (If not, please install it manually: https://ninja-build.org/)


#### II. Build llama.cpp

You could download the release package for Windows directly, which including binary files and depended oneAPI dll files.

Choose one of following methods to build from source code.

##### Option 1: Script

```sh
# Uses FP32, consider using FP16 for better performance in most cases
.\examples\sycl\win-build-sycl.bat
```

##### Option 2: CMake

On the oneAPI command line window, step into the llama.cpp main directory and run the following:

```
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

# Option 1: Use FP16 (recommended for better performance in most cases)
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON

# Option 2: Or FP32
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j
```

Or, use CMake presets to build:

```sh
cmake -DGGML_SYCL_F16=ON --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-completion

cmake --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-completion

cmake --preset x64-windows-sycl-debug
cmake --build build-x64-windows-sycl-debug -j --target llama-completion
```

##### Option 3: Visual Studio

You have two options to use Visual Studio to build llama.cpp:
- As CMake Project using CMake presets.
- Creating a Visual Studio solution to handle the project.

**Note**:

All following commands are executed in PowerShell.

###### - Open as a CMake Project

You can use Visual Studio to open the `llama.cpp` folder directly as a CMake project. Before compiling, select one of the SYCL CMake presets:

- `x64-windows-sycl-release`

- `x64-windows-sycl-debug`

*Notes:*
- For a minimal experimental setup, you can build only the inference executable using:

    ```Powershell
    cmake --build build --config Release -j --target llama-completion
    ```

###### - Generating a Visual Studio Solution

You can use Visual Studio solution to build and work on llama.cpp on Windows. You need to convert the CMake Project into a `.sln` file.

If you want to use the Intel C++ Compiler for the entire `llama.cpp` project, run the following command:

```Powershell
cmake -B build -G "Visual Studio 17 2022" -T "Intel C++ Compiler 2025" -A x64 -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release
```

If you prefer to use the Intel C++ Compiler only for `ggml-sycl`, ensure that `ggml` and its backend libraries are built as shared libraries ( i.e. `-DBUILD_SHARED_LIBRARIES=ON`, this is default behaviour):

```Powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
      -DSYCL_INCLUDE_DIR="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" \
      -DSYCL_LIBRARY_DIR="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
```

If successful the build files have been written to: *path/to/llama.cpp/build*
Open the project file **build/llama.cpp.sln** with Visual Studio.

Once the Visual Studio solution is created, follow these steps:

1. Open the solution in Visual Studio.

2. Right-click on `ggml-sycl` and select **Properties**.

3. In the left column, expand **C/C++** and select **DPC++**.

4. In the right panel, find **Enable SYCL Offload** and set it to `Yes`.

5. Apply the changes and save.


*Navigation Path:*

```
Properties -> C/C++ -> DPC++ -> Enable SYCL Offload (Yes)
```

Now, you can build `llama.cpp` with the SYCL backend as a Visual Studio project.
To do it from menu: `Build -> Build Solution`.
Once it is completed, final results will be in **build/Release/bin**

*Additional Note*

- You can avoid specifying `SYCL_INCLUDE_DIR` and `SYCL_LIBRARY_DIR` in the CMake command by setting the environment variables:

    - `SYCL_INCLUDE_DIR_HINT`

    - `SYCL_LIBRARY_DIR_HINT`

- Above instruction has been tested with Visual Studio 17 Community edition and oneAPI 2025.0. We expect them to work also with future version if the instructions are adapted accordingly.

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Obtaining and quantizing models*](../user/models.md) guide for model preparation, or download an already quantized model like [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) or [Meta-Llama-3-8B-Instruct-Q4_0.gguf](https://huggingface.co/aptha/Meta-Llama-3-8B-Instruct-Q4_0-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf).

##### Check device

1. Enable oneAPI running environment

On the oneAPI command line window, run the following and step into the llama.cpp directory:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```
build\bin\llama-ls-sycl-device.exe
```

This command will only display the selected backend that is supported by SYCL. The default backend is level_zero. For example, in a system with 2 *Intel GPU* it would look like the following:
```
found 2 SYCL devices:
|  |                  |                                             |Compute   |Max compute|Max work|Max sub|               |
|ID|       Device Type|                                         Name|capability|units      |group   |group  |Global mem size|
|--|------------------|---------------------------------------------|----------|-----------|--------|-------|---------------|
| 0|[level_zero:gpu:0]|               Intel(R) Arc(TM) A770 Graphics|       1.3|        512|    1024|     32|    16225243136|
| 1|[level_zero:gpu:1]|                    Intel(R) UHD Graphics 770|       1.3|         32|     512|     32|    53651849216|

```

##### Choose level-zero devices

|Chosen Device ID|Setting|
|-|-|
|0|Default option. You may also want to `set ONEAPI_DEVICE_SELECTOR="level_zero:0"`|
|1|`set ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|0 & 1|`set ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"` or `set ONEAPI_DEVICE_SELECTOR="level_zero:*"`|

##### Execute

Choose one of following methods to run.

1. Script

- Run test:

```
examples\sycl\win-test.bat
```

- Run llama-server:

```
examples\sycl\win-start-svr.bat -m PATH\MODEL_FILE
```

2. Command line

Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user. Default device id is 0.
- Multiple devices: Automatically choose the devices with the same backend.

In two device selection modes, the default SYCL backend is level_zero, you can choose other backend supported by SYCL by setting environment variable ONEAPI_DEVICE_SELECTOR.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |
| Multiple devices | --split-mode tensor (tensor parallelism) |

`--split-mode tensor` (tensor parallelism) shards each layer across the selected
GPUs. It requires flash attention, which is auto-enabled when `--flash-attn` is
left at its default `auto`, so `--split-mode tensor` works out of the box.
Passing `--flash-attn off` together with `--split-mode tensor` is rejected at
context creation. The default `f16` KV cache is recommended. Tensor parallelism
is currently optimized for 2 GPUs; other device counts fall back to a generic
all-reduce.

Examples:

- Use device 0:

```
build\bin\llama-completion.exe -no-cnv -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 99 -sm none -mg 0 --load-mode auto
```

- Use multiple devices:

```
build\bin\llama-completion.exe -no-cnv -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 99 -sm layer --load-mode auto
```


Note:

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```

Or

```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```

User can use the device management in [docs/user/multi-gpu.md](../user/multi-gpu.md), like parameter `--device SYCL0,SYCL1` to assign one or more devices.

## Environment Variable

### Build

| Name               | Value                                 | Function                                    |
|--------------------|---------------------------------------|---------------------------------------------|
| GGML_SYCL          | ON (mandatory)                        | Enable build with SYCL code path.           |
| GGML_SYCL_TARGET   | INTEL *(default)*                     | Set the SYCL target device type. INTEL is the only accepted value; anything else stops CMake. |
| GGML_SYCL_FA_LARGE_GRF | OFF *(default)* \|ON *(Optional)* | Compile the 256-GRF variants of the flash-attention tile kernels so the runtime knob `GGML_SYCL_FA_LARGE_GRF` can select them. Doubles the tile-kernel device code, and that share of an AOT build (CMake warns when both are set). |
| GGML_SYCL_DEVICE_ARCH | Optional                           | Set the SYCL device architecture. Setting the device architecture can improve the performance. See the table [--offload-arch](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OffloadDesign.md#--offload-arch) for a list of valid architectures. Empty (default) builds JIT; `acm-g10` selects AOT (`spir64_gen`) for the A770. |
| GGML_SYCL_MAX_PARALLEL_LINK_JOBS | CPU count *(default)* | Parallel `ocloc` jobs for the AOT device link. Only used when `GGML_SYCL_DEVICE_ARCH` is set. |
| GGML_SYCL_DEVICE_CODE_SPLIT | ON *(default)* \|OFF *(Optional)* | Compile and link with `-fsycl-device-code-split=per_kernel`, so JIT compiles each kernel on first use instead of the whole device image. |
| GGML_SYCL_TURBO_QUANT | ON *(default)* \|OFF *(Optional)* | Compile the TurboQuant (turbo2/3/4) K/V flash-attention dispatch. OFF makes flash attention reject turbo K/V. |
| GGML_SYCL_SCALAR_MMQ | OFF *(default)* \|ON *(Optional)* | Leave `SYCL_USE_XMX` undefined (scalar MMQ tile sizes, intended for B70). MMQ is disabled in this tree (`ggml_sycl_supports_mmq()` returns false), so this currently has no runtime effect. |
| GGML_SYCL_SEPARATE_BUILD | OFF *(default)* \|ON *(Optional)* | Build ggml-sycl as a nested project with its own C++ compiler (`GGML_SYCL_SEPARATE_BUILD_CXX`, default `icpx`, `icx` on Windows; extra `-D` arguments in `GGML_SYCL_SEPARATE_BUILD_ARGS`), so the rest of the tree can use another compiler. Requires `GGML_BACKEND_DL=ON` and a non-Visual Studio generator such as Ninja. |
| GGML_SYCL_F16      | OFF *(default)* \|ON *(optional)*     | Enable FP16 build with SYCL code path. (1.) |
| GGML_SYCL_GRAPH    | ON *(default)* \|OFF *(Optional)*     | Enable build with [SYCL Graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc). |
| GGML_SYCL_DNN      | ON *(default)* \|OFF *(Optional)*     | Request oneDNN. Only a request: if CMake finds no oneDNN built for the same GPU target, the build compiles with `GGML_SYCL_DNNL=0` and the oneDNN GEMM and flash-attention paths are compiled out. The `GGML_SYCL_DNNL: yes/no` startup log line is authoritative. |
| GGML_SYCL_HOST_MEM_FALLBACK | ON *(default)* \|OFF *(Optional)* | Allow host memory fallback when device memory is full during quantized weight reorder. Enables inference to continue at reduced speed (reading over PCIe) instead of failing. Requires Linux kernel 6.8+. |
| GGML_SYCL_SUPPORT_LEVEL_ZERO_API | ON *(default)* \|OFF *(Optional)* | Support to use Level Zero API for device memory allocation. Requires Level Zero headers/library at build time and Intel GPU driver (Level Zero runtime) at run time. Reduces system RAM usage during multi-GPU inference. SYCL backend always runs on Level Zero running time even if it's set as OFF (The SYCL api will be usage for memory allocation).|
| GGML_SYCL_XMX_GATHER | AUTO *(default)* \|ON\|OFF | XMX gather build policy. AUTO selects verified AOT targets; ON explicitly includes all requested targets; OFF omits the kernels. See (2.) for mixed targets, JIT, and cache migration. |
| CMAKE_C_COMPILER   | `icx` *(Linux)*, `icx/cl` *(Windows)* | Set `icx` compiler for SYCL code path.      |
| CMAKE_CXX_COMPILER | `icpx` *(Linux)*, `icx` *(Windows)*   | Set `icpx/icx` compiler for SYCL code path. |

1. FP32 or FP16 have different performance impact to LLM. Recommended to test them for better prompt processing performance on your models. You need to rebuild the code after change `GGML_SYCL_F16=OFF/ON`.

2. See [XMX gather GEMMs and DG2 AOT builds](#xmx-gather-gemms-and-dg2-aot-builds).

#### XMX gather GEMMs and DG2 AOT builds

`ggml/src/ggml-sycl/fused-gemm.cpp` implements dequant-in-GEMM for nine IQ weight
formats, through grouped `MUL_MAT_ID` and plain `MUL_MAT` entry points. The kernels
need the SG16 8x16x16 fp16/fp16/fp32 `joint_matrix` combination. The A770 does not
report it, so its runtime gate selects the regular GEMM paths. AOT compilation
still tries to compile every emitted kernel: the original `acm-g10` link crashed
IGC 2.41.5. Other unsupported targets also fail, so matching only DG2 names is
insufficient. Target-link evidence and its limits are in
[the investigation](../research/sycl-xmx-gather-dg2-aot-2026-09-28.md).

| `GGML_SYCL_XMX_GATHER` | `GGML_SYCL_DEVICE_ARCH` | Gather kernel images |
|---|---|---|
| AUTO (default) | empty | portable JIT; runtime checks device support |
| AUTO | AOT list | only exact allow-listed entries: `bmg-g21`, `xe2-hpg`, `pvc`, `lnl-m` |
| AUTO | `acm-g10,bmg-g21` | `bmg-g21` only; other backend kernels retain both targets |
| AUTO | no allow-listed entries | omitted; regular GEMM fallback |
| ON | empty | portable JIT |
| ON | AOT list | all requested targets; unsupported combinations may fail device link |
| OFF | any | omitted |

Selection lower-cases names, normalizes underscores to hyphens, and accepts
comma, semicolon, or space separators. Unknown names, PCI/IP identifiers, and
ranges are omitted by AUTO unless an entry matches the exact allow-list after
normalization. Omission is conservative, not proof that a device is unsupported.
ON is an explicit opt-in for targets outside the list and is never silently
changed to OFF. Existing build directories with the old BOOL option set to ON
keep that explicit ON: use a clean build directory or pass
`-DGGML_SYCL_XMX_GATHER=AUTO` to select the new default.

When AUTO selects a strict subset of an AOT list, the build compiles
`fused-gemm.cpp` separately and device-links only its selected targets, then
includes the resulting objects in the backend. This keeps
gather images for supported targets without sending those kernels through an
unsupported target's device link. No additional runtime library is introduced. All-admitted lists and explicit ON
use the regular backend link. ELF static mixed builds retain the image registration
with the host entry points in one archive member. Windows static mixed builds are
rejected at configure time: use a shared build, OFF, or separate packages.
The runtime checks both matrix support and kernel-image availability before
launching. With no emitted gather images, the existing entry points return false
and callers use regular GEMM. `GGML_SYCL_XMX_GATHER_TYPES` cannot enable a missing
image or bypass the capability gate.

JIT AUTO deliberately keeps portable kernels, including when configured on an
A770: a build can be deployed to a different GPU. Use OFF for a JIT package that
must omit them. The device capability cache is thread-local; this removes the
previous global mutex, but no runtime speedup is claimed.

**Inspect the effective configuration.**

- CMake reports the requested policy and selected gather targets. The top-level
  `build-metadata.json` records `sycl_xmx_gather_requested` and
  `sycl_xmx_gather_effective`; effective values are `OFF`, `JIT`, or a comma-separated
  AOT target list. Preserve these fields alongside source SHA and binary hashes.
- With `GGML_SYCL_SEPARATE_BUILD=ON` and `GGML_BACKEND_DL=ON`, the backend's actual
  compiler checks and command generation run in the nested configure during the
  build. Inspect `<build>/ggml/src/ggml-sycl/nested/CMakeCache.txt` and that
  directory's `compile_commands.json` (enabled by the ggml configure).
  Set `GGML_SYCL_XMX_GATHER` and `GGML_SYCL_DEVICE_ARCH` on the outer configure;
  overriding either through `GGML_SYCL_SEPARATE_BUILD_ARGS` is rejected to keep
  outer metadata consistent. An outer cache or outer configure log alone does
  not prove what was compiled.
- A `fused-gemm.cpp` compile command containing `GGML_SYCL_NO_XMX_GATHER` builds
  fallback entry points. For mixed lists, inspect the separate gather device-link
  command too; the backend's full target list is not its gather target list.
- A fully disabled backend prints
  `GGML_SYCL_XMX_GATHER_TYPES: XMX gather GEMMs disabled by compile flag` at startup.
  A runtime bitmask permits formats; it does not prove a kernel was launched.

Use the CMake option instead of manually injecting `GGML_SYCL_NO_XMX_GATHER` into
compiler flags, so metadata can describe the selected build. No SG8 gather GEMM
variant is provided; its correctness and performance remain unmeasured.

### Runtime

| Name              | Value            | Function                                                                                                                  |
|-------------------|------------------|---------------------------------------------------------------------------------------------------------------------------|
| GGML_SYCL_DEBUG   | 0 (default) or 1 | Enable log function: GGML_SYCL_DEBUG() for common debug. |
| GGML_SYCL_DEV_DEBUG   | 0 (default) or 1 | Enable log function: GGML_SYCL_DEV_DEBUG() for developmental purposes by replacing GGML_SYCL_DEBUG() in special codes. Restore to GGML_SYCL_DEBUG() before committing code.|
| GGML_SYCL_DEV2DEV_MEMCPY | 0 (default), 1, 2 | Choose the method of dev2dev memory copy.<br>Value: <br>*  0: SYCL API (default), only support dGPUs.<br>* 1: L0 API -- Better performance, only support dGPUs, found to lead to abnormal crash in some case. <br>* 2: Host Forward -- Most stable method for all cases (including iGPU + dGPU*N), but with lower performance (-2% to -5%).<br>SYCL & L0 API are easy to be impacted by Intel GPU driver issue. When you meet the garbled output or crash issues in multiple GPUs case, try with this debug flag to work around or check the issue.<br>Forced to 0 when the Level Zero API is not in use (`GGML_SYCL_USE_LEVEL_ZERO_API=0`).|
| GGML_SYCL_ENABLE_FLASH_ATTN | 1 (default) or 0| Enable Flash-Attention. It can reduce memory usage. The performance impact depends on the LLM.|
| GGML_SYCL_ENABLE_OPT | 0 or 1 (default)| Enable optimize features for Intel GPUs. (Recommended to 0 for Intel devices older than Gen 10) |
| GGML_SYCL_ENABLE_GRAPH | 0 (default) or 1 | Enable running computations through SYCL Graphs feature. Disabled by default because SYCL Graph is still on development, performance depends on device. |
| GGML_SYCL_GRAPH_PROFILE | 0 (default) or 1 | Print SYCL graph counters (graph, replay, re-record, finalize and update calls, timings) to stderr at exit. Only the exact value `1` enables it. |
| GGML_SYCL_GRAPH_EVICTION_TIMEOUT | 10 (default) or positive integer | Seconds a cached SYCL graph can go unused before it is evicted. Checked on a sweep that runs every half of this timeout. |
| GGML_SYCL_ENABLE_HOST_PINNED_MEM | 0 or 1 (default) | Enable host pinned memory to speed up copy data from host to device. When disable it, host memory will common malloc() on CPU. Disable it when use `--load-mode mlock`.|
| GGML_SYCL_NO_PINNED | Unset (default) or set | Any value disables pinned host buffers: the device reports no host buffer type and pinned host allocations return null. |
| GGML_SYCL_HOST_PINNED_MEM_2G | 0 (default) or 1 | Limit the max memory allocation to be no more than 2GB when enable host pinned memory. USM allocations above 2 GiB take the relaxed/large-allocation path, which serializes H2D copies with compute and prevents copy/compute overlap. It will impact the startup time. Need more test. Depend on `GGML_SYCL_ENABLE_HOST_PINNED_MEM=1`.|
| GGML_SYCL_GET_MEM_API | 0 (default) or 1  | Set to get memory info (free, total) by Level Zero or SYCL API:<br>0 - Level Zero API: support more GPUs, only run on Level Zero running time. When there is an error, fallback to call SYCL API. Depend on GGML_SYCL_SUPPORT_LEVEL_ZERO_API.<br>1 - SYCL API: legacy, support more running time, it can't get the free size of some GPUs (like Arc770). In such case, return the free size as value of total size.<br>Forced to 1 when `GGML_SYCL_USE_LEVEL_ZERO_API=0`.|
| GGML_SYCL_USE_LEVEL_ZERO_API | 1 (default) or 0 | Use Level Zero API for device memory allocation instead of SYCL. Reduces system RAM usage on Intel dGPUs by avoiding DMA-buf/TTM host memory staging. Requires GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON at build time. SYCL backend always runs on Level Zero running time even if it's set as OFF (The SYCL api will be usage for memory allocation).|
| GGML_SYCL_ENABLE_DNN | 0 or 1 (default)| Use oneDNN for the GEMM paths; 0 uses oneMKL. Inert in builds with `GGML_SYCL_DNNL=0`, which always use oneMKL. |
| GGML_SYCL_FA_ONEDNN | 1 (default) or 0 | Enable the oneDNN fused SDPA (flash-attention) path on supported GPUs. Set to 0 to always use the native SYCL flash-attention kernel. Only effective in builds with `GGML_SYCL_DNNL=1`; checked after the MKL and XMX routes, and head dim 64 is excluded on DG2. |
| GGML_SYCL_FA_LARGE_GRF | 0 (default) or 1 | Request the 256-entry register file for the flash-attention tile kernels on launches with more than one query row (prefill). At the default 128 GRF the FA tile kernels spill 8-15 KB per thread on DG2 (IGC 2.41 shader dumps). Measured on the A770 (paired campaigns, `docs/research/sycl/sycl-fa-large-grf-2026-09-30.md`): pp512 +2.4 % +- 1.1 (d=256) to +10.0 % +- 0.9 (d=128) at short context, decode flat (a tile-and-vec mode was measured and dropped: decode loss at depth). A 256-GRF launch plans with half the work-groups per Xe-core unless `GGML_SYCL_MAX_WG_PER_CU` is set explicitly. The 256-GRF instantiations exist only for the tile kernels and only in builds configured with `-DGGML_SYCL_FA_LARGE_GRF=ON` (default OFF); other builds warn and ignore the variable. The request applies on Xe-HPG (DG2), Xe-HPC and Xe2 devices and is ignored with one warning per device elsewhere; the value must be exactly 0 or 1, anything else warns and is ignored. |
| GGML_SYCL_FA_ONEDNN_MAX_KV | 0 (default, disabled) or positive integer | By default (0), all sequences are handled by the oneDNN fused SDPA path, regardless of KV length; a positive value caps that length, past which sequences fall back to the native kernel. If GPU driver watchdog resets (DEVICE_LOST) occur during long-context inference, set this near the context depth where they start, e.g. 24576. |
| GGML_SYCL_ENABLE_VMM | 0 or 1 (default) | Enable the virtual-memory device pool. |
| GGML_SYCL_ENABLE_MKL_FA | 1 (default) or 0 | Enable oneMKL GEMM flash attention for XMX-accelerated prompt processing with non-turbo K/V. Activates when all conditions are met: mask present, no sinks/ALiBi/softcap, GQA ratio >= 2, head dim a multiple of 64 in [64, 512] with matching K/V head size, `Q->ne[1] >= 32` and `K->ne[1] >= 1024`. Set to 0 to force the TILE/VEC kernel for A/B testing. Example minimum command: `llama-cli -m model.gguf -fa on -ngl 99 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 1024 -p "your prompt"` |
| GGML_SYCL_MKL_FA_Q_TILE | 8192 (default) or positive integer | Query rows per MKL flash-attention tile; bounds the score scratch to `Q_TILE x min(n_kv, 8192)` elements. |
| GGML_SYCL_MKL_FA_DEBUG | 0 (default) or 1 | Enable per-call diagnostic logging for MKL flash attention: GEMM/softmax timings, interleaved-head detection, and buffer memory usage. |
| GGML_SYCL_FA_XMX | Unset (default) or set | Opt-in XMX (DPAS) flash-attention kernel. Presence enables it: any value, including `0`. Taken only for same-type K/V with head dim 128 or 256: turbo, or f16/q8_0 in canonical layout (q8_0 quants-first rows are excluded, so q8_0 needs `GGML_SYCL_Q8_KV_QUANTS_FIRST=0`). ALiBi, logit softcap, attention sinks and multi-sequence batches fall through to VEC/TILE. Experimental (see Known Issues). |
| GGML_SYCL_FA_XMX_DEBUG | Unset (default) or set | Print one line per XMX flash-attention launch (shape, mask). |
| GGML_SYCL_FA_Q8_GQA_TILE | 0 (default) or 1 | Decode only (one query row): route head dim 128 q8_0/q8_0 KV with GQA >= 2 to TILE instead of VEC. Prefill is unaffected. |
| GGML_SYCL_FA_FORCE_VEC_STANDARD | 0 (default) or 1 | Decode only: force VEC for head dim 128 f16/f16 or q8_0/q8_0 KV. Prefill is unaffected. `GGML_SYCL_FA_Q8_GQA_TILE` is checked first. |
| GGML_SYCL_FA_PROFILE | Unset or 0 (default), any other value enables | Print the first flash-attention route per route and phase (`GGML_SYCL_FA_ROUTE:` lines on stderr) and, at exit, VEC/TILE launch counts, timings and launch geometry per KV layout (`GGML_SYCL_FA_PROFILE:` lines). |
| GGML_SYCL_MEMTRACE | 0 (default), 1, 2 | Enable record and output memory allocation diagnostics. Requires `-lv 4`. <br>0 -  Disable<br>1 - Basic memory info, including current and peak allocations, as well allocations from other sources, around 50 lines per model load.<br>2 - More verbose, logging around 900 specific allocations and deallocations. |
| GGML_SYCL_MEMTRACE_STEP | 64 (default) or positive integer | With GGML_SYCL_MEMTRACE=1, the minimum growth in memory usage to trigger another log record. |
| GGML_SYCL_MKL_FA_DIAG | 0 (default) or 1 | Enable output fingerprinting for MKL flash attention. Dumps the first 64 float output values for the first 6 FA calls with n_kv >= 1024, labeled with kernel type (MKL/TILE/VEC) for cross-kernel comparison. |
| GGML_SYCL_ENABLE_FUSION | 0 or 1 (default) | Enable the graph-compute fusions: top-k MoE gating (`ggml_sycl_fuse()`, topk-moe.cpp), every fusion matched through `ggml_sycl_can_fuse()` (fusion.cpp: RMS_NORM+MUL(+ADD), RMS_NORM+SCALE, ADD+ADD, UNARY+MUL, SSM_CONV(+ADD)+SILU, MUL_MAT+MUL_MAT+GLU mat-vec) and the gated-delta-net cache-copy fusion. The Q4_K FFN fusion (`GGML_SYCL_FFN_FUSION`) and L2_NORM batching are not gated by this flag. |
| GGML_SYCL_ROPE_FUSION_PROFILE | 0 (default) or 1 | Count ROPE fusion candidates and rejection reasons per graph and print them to stderr at exit. Only the exact value `1` enables it. |
| GGML_SYCL_MAX_WG_PER_CU | Integer 1-1024 (16 default) | Set the flash-attention resident work-group cap per Xe-core/SM. This is not a cap per SYCL compute unit/EU. Invalid values are ignored with a warning. |
| GGML_SYCL_FFN_FUSION | 0 or 1 (default) | Enable fused dense Q4_K single-token SwiGLU. Promoted to default 2026-08-12 after a paired campaign against the fixed kernel; set to 0 to opt out. |
| GGML_SYCL_FFN_FUSION_DEBUG | Unset (default) or set | Log tensor shape and layout details for the first fused FFN launch attempt. |
| GGML_SYCL_FFN_FUSION_PROFILE | 0 (default) or 1 | Report fused FFN graph eligibility and rejection counters at process exit. |
| GGML_SYCL_ENABLE_ESIMD | 0 or 1 (default)| Enable ESIMD kernels when available. |
| GGML_SYCL_PRIORITIZE_DMMV | 0 (default) or 1 | Prefer the DMMV (dequantize mat-vec) kernels over the reordered MMVQ/ESIMD paths for quantized mat-vec. Also turns off the K-quant reorder for the generic mul_mat path and the reordered same-type Q4_K/Q5_K gate+up+GLU mat-vec fusion. |
| GGML_SYCL_USE_ASYNC_MEM_OP | 1 (default) or 0 | Use asynchronous USM allocation/free (`ext_oneapi_async_memory_alloc`) for temporary buffers when every device supports it; `GGML_SYCL_ENABLE_GRAPH=1` turns it on regardless. Requires a build with `GGML_SYCL_GRAPH=ON`. |
| GGML_OP_OFFLOAD_MIN_BATCH | 32 (default) or integer | Minimum batch size at which an op whose weights are in host memory is offloaded to the SYCL device. |
| GGML_SYCL_MMVQ_WIDE | 0 or 1 (1 default) | Use the wide-load variant of the reordered Q8_0 mat-vec kernel, which reads four contiguous dwords per operand instead of one value at a time. Set to 0 to fall back to the per-value loads. Only affects Q8_0 weights in the reordered layout. |
| GGML_SYCL_XMX_GATHER_TYPES | decimal bitmask, all bits set (default) | Select which quantized weight formats may take the XMX dequant-GEMM paths, where the weights are dequantized inside the GEMM (gathered straight into the XMX tiles) instead of being written out to f16 and read back. This covers the grouped `MUL_MAT_ID` path used by MoE models, and the plain `MUL_MAT` path when built with `GGML_SYCL_F16=ON` (the plain path sits inside that build's f16 branch). Both compute in f16 on the XMX units regardless of `GGML_SYCL_F16`, so enabling them for `MUL_MAT_ID` trades some precision for speed relative to the per-expert library GEMM they replace. Mainly affects prompt processing; token generation is unaffected. One bit per format, so a format can be enabled or benchmarked on its own:<br>* 1: IQ4_NL<br>* 2: IQ3_S<br>* 4: IQ4_XS<br>* 8: IQ3_XXS<br>* 16: IQ2_XXS<br>* 32: IQ2_XS<br>* 64: IQ2_S<br>* 128: IQ1_S<br>* 256: IQ1_M<br>Set to 0 to disable the paths entirely and fall back to the library GEMM, which is the baseline to compare against. A format is only taken when the shape also fits (the weights must cover whole blocks, and the tile is only used while N is narrow), so setting a bit does not force the path. Formats outside this list are never affected by this variable. No effect when the selected build policy omits these kernels or the device has no matching kernel image, see [XMX gather GEMMs and DG2 AOT builds](#xmx-gather-gemms-and-dg2-aot-builds). |
| GGML_SCHED_COPY_SYNC | 1 (default) or exactly 0 | Keep synchronous scheduler input copies by default. Set exactly `0` to opt into experimental stream-ordered copies where eligible. Read once per process. See [scheduler input-copy synchronization](#scheduler-input-copy-synchronization) for eligibility, lifetime, diagnostics, and validation limits. |
| GGML_SYCL_SPARSE_FA | 0 (default) or 1 | Enable Sparse Flash-attention.|
| GGML_SYCL_SPARSE_FA_DEBUG | 0 (default) or 1 | Enable to debug for Sparse Flash-attention.|
| GGML_SYCL_SPARSE_FA_MARGIN | [0,..] default:256 | Set the margin value for Sparse Flash-attention.|
| ZES_ENABLE_SYSMAN | 0 (default) or 1 | Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory.<br>Recommended to use when --split-mode = layer |
| UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS | 0 (default) or 1 | Allow SYCL/Unified Runtime Level Zero device allocations larger than 4 GiB. llama.cpp's direct Level Zero allocation path requests the relaxed maximum-size limit itself when GGML_SYCL_USE_LEVEL_ZERO_API=1. |
| UR_L0_USE_COPY_ENGINE | adapter default, or 0 | Unified Runtime Level Zero (v1 adapter) knob: 0 routes USM copies to the compute queue instead of the blitter (bcs), 1 enables every copy engine, `lower:upper` selects a range. On Linux, when a DG2 GPU is bound to the `xe` kernel driver, ggml-sycl sets it to 0 at startup and logs one line (see Known Issues) unless the variable or its `SYCL_PI_` alias is already set, or any copy-engine variable asks for copy engines (this one, the `UR_L0_USE_COPY_ENGINE_FOR_*` family or the aliases with a value other than 0, or `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=0`). An explicit 0 agrees with the default and does not block the other adapter's variable; empty counts as unset and, on DG2/xe, is removed from the environment (the adapter aborts at load on an empty `UR_L0_USE_COPY_ENGINE_FOR_*` value and at the first queue on an empty value here). Set it to 1 to enable the copy engines anyway; `--prefetch-experts-slots` needs that (and is refused on the v2 adapter regardless). `GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0` gives the adapter's own defaults back. |
| GGML_SYCL_XE_COPY_ENGINE_DEFAULT | 1 (default) or 0 | 0 disables the automatic xe copy-engine defaults above entirely (integer, read like the other GGML_SYCL knobs). The hook's log lines are printed at backend initialization, after the application's logger is installed; when the defaults are skipped because a variable asks for copy engines, one line says which. |
| UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD | 0 (default) or 1 | The same control for the Level Zero v2 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1`, default on Xe2+) (its binary also reads `UR_L0_USE_COPY_ENGINE` and `UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY`, none of the other v1 names). ggml-sycl sets it to 1 on DG2/xe unless it is already set or a copy-engine variable asks for copy engines (rule above). To keep copy offload on the v2 adapter set it to 0 explicitly; on v1 set `UR_L0_USE_COPY_ENGINE=1`. Either override can bring the blitter failure back on DG2. |
| UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE | 1 (default) or 0 | Same as above but only for in-order queues, which is every queue ggml-sycl creates. |
| SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE | alias | Older alias for UR_L0_USE_COPY_ENGINE, read by the adapter only when the UR name is unset. An explicit value here counts like one on the UR name for the xe default; if the UR name is present but empty next to a set alias, ggml-sycl copies the alias into it (the adapter would otherwise fail parsing the empty value). |
| GGML_SYCL_USM_SYSTEM | 0 (default) or 1 | Enable experimental support for [USM system allocations](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html#system-allocations) for large GPU buffers. This requires enough host memory for model weights and caches, an Intel Xe2+ GPU such as BMG or newer and supported on Linux only, with CONFIG_DRM_XE_GPUSVM enabled. |
| GGML_SYCL_Q8_KV_QUANTS_FIRST | 1 (default) or 0 | Store `q8_0` KV cache rows as 128 contiguous quant values followed by four fp16 scales, instead of four interleaved 34-byte `block_q8_0` records. Applies only to SYCL devices with `q8_0` K and V, 128-element heads and non-transposed V (flash attention on); every other cache keeps canonical blocks either way. Set to 0 to fall back. Read by `src/llama-kv-cache.cpp`. |

### Scheduler input-copy synchronization

`GGML_SCHED_COPY_SYNC` controls the generic scheduler's handling of eligible
host-to-device split inputs, such as outputs from CPU-resident MoE experts with
`--n-cpu-moe`. It is a runtime environment variable, not a CMake option or a
server request parameter. It does not enable SYCL graph replay or select the
Level Zero copy engine.

| Environment value | Behavior |
| --- | --- |
| Unset or `1` | Default: retain the existing synchronization and copy path. |
| Exactly `0` | Opt into experimental stream-ordered input copies, subject to the gates below. |
| Empty or any other string, including `false`, `00`, or whitespace around `0` | Retain synchronization; these are not opt-ins. |

The value is cached process-wide when the scheduler first checks it, normally
during scheduler construction. Set it before launching the executable; changing
the environment later does not change existing or subsequent schedulers in that
process. Library callers follow the same rule. No upload-completion event for
this experimental path is allocated while synchronization is selected. Normal
pipeline-copy events are unaffected.

With `0`, the destination must advertise `ggml_backend_async_is_stream_ordered`,
use its default buffer type, and provide an asynchronous tensor setter. Only
single-device SYCL currently advertises the capability. The scheduler must have
one copy (`n_copies == 1`) and no pipeline-copy event. The source must have a
host buffer; CPU-from-pointer buffers are excluded to preserve mapped-memory
staging, including the PVC workaround. Multi-device SYCL, parallel schedulers,
unsupported backends, and ineligible buffers keep their existing copy path even
when `0` is set.

An eligible ordinary split input synchronizes its source backend, then enqueues
`set_tensor_async` on the destination's in-order stream without first draining
that stream. Mutable host sources additionally require a completion event,
recorded immediately after the upload. The scheduler waits on pending upload
events before host splits can overwrite source memory and before graph return,
including compute failure. These waits leave later device compute queued.
Immutable WEIGHTS do not need this mutable-source guard. If event support or
allocation is unavailable, mutable inputs retain blocking copies. Routed expert
copies and prefetch staging keep their existing specialized transfer handling;
this flag is not a guarantee that every copy becomes asynchronous.

Use separate processes for comparison, with identical arguments:

```bash
# Default and explicit synchronous baseline:
env -u GGML_SCHED_COPY_SYNC llama-completion -m MODEL --n-cpu-moe 20 -n 256 -lv 5
GGML_SCHED_COPY_SYNC=1 llama-completion -m MODEL --n-cpu-moe 20 -n 256 -lv 5
# Experimental opt-in:
GGML_SCHED_COPY_SYNC=0 llama-completion -m MODEL --n-cpu-moe 20 -n 256 -lv 5
```

At scheduler destruction, debug logging (`-lv 5` for completion, `-v` for bench)
prints `stream-ordered input copies: N, host-read flushes: H at a host split,
G at graph end` if `N > 0`. Require a nonzero counter to establish that a test
exercised the new ordinary input-copy branch. A missing line alone cannot
distinguish disabled logging, ineligible graphs, early termination, or the
synchronous path. Flush counters count upload-event waits, not full backend
synchronizations or time saved.

**Validation limit:** the 2026-10-06 A770 correctness gate and CPU lifetime tests
passed, but real-model greedy output was not repeatable even in the forced-sync
baseline. Output equivalence remains unproven; five timing pairs showed no
throughput improvement. The optimization therefore remains opt-in. The measured
revision enabled it when unset; use explicit `0` to reproduce that arm with
current code. See the [campaign report](../research/sched-stream-ordered-copies-2026-09-28.md)
for commands, counters, failed comparisons, and untested configurations.

### Intel Arc (A770 / DG2) flash-attention KV cache

For flash attention (`-fa on`) with a quantized KV cache on Arc, use `q8_0`
(`--cache-type-k q8_0 --cache-type-v q8_0`): it is near-lossless and runs at
mainline parity (`docs/research/sycl/standard-sycl-upstream-ab-2026-07-11.md`). Both `GGML_SYCL_F16=ON` and `OFF` builds work; `F16=ON` gives
faster prompt processing. TurboQuant KV cache types (`turbo2`/`turbo3`/`turbo4`)
run on the SYCL flash-attention VEC path; Arc validation and optimization
currently cover `-fa on` only. The non-FA graph handles block-quantized V via a
dequant-before-transpose and inverse-WHT sequence, but that path remains
unqualified for production use with turbo KV.

`ggml_sycl_get_best_fattn_kernel()` (`ggml/src/ggml-sycl/fattn.cpp`) picks the kernel in
this order:

1. Turbo K or V: VEC, for head dims that are multiples of 128. Accepted pairs are turbo K
   and V in any combination, or turbo on one side with f16 or q8_0 on the other.
   `GGML_SYCL_FA_XMX` sends same-type turbo K/V at head dim 128 or 256 to XMX instead.
2. Non-turbo prefill inside the MKL envelope (`GGML_SYCL_ENABLE_MKL_FA`): oneMKL GEMM.
3. Decode overrides: `GGML_SYCL_FA_Q8_GQA_TILE`, then `GGML_SYCL_FA_FORCE_VEC_STANDARD`.
4. XMX (opt-in), then oneDNN SDPA (`GGML_SYCL_DNNL=1` builds only).
5. Otherwise VEC for single-row decode (f16/f32 KV only when the GQA optimization does not
   apply; quantized KV up to 2 rows, which go to TILE on Xe2/BMG instead) and TILE for
   everything else.

Two llama-level variables (`src/llama-kv-cache.cpp`) decide which KV types the kernels see:

- `TURBO_AUTO_ASYMMETRIC`: when K and V request the same turbo type and a non-MLA model has
  GQA >= 6 or is Qwen-family, K is rewritten to `q8_0` (logged as `auto-asymmetric`).
  `TURBO_AUTO_ASYMMETRIC=0` keeps turbo K.
- `TURBO_LAYER_ADAPTIVE`: a single digit selecting per-layer KV types (modes 1, 2, 5, 6, 7).
  When unset, turbo2 V on models with at least 8 layers gets mode 7 (first and last two
  layers V=q8_0, the rest turbo2); `TURBO_LAYER_ADAPTIVE=0` opts out. A mode that would not
  change any turbo type logs a warning and is ignored.

## Compile-time Flags

Pass these via `CXXFLAGS` or add a one-off `#define` to enable a flag on the spot.

| Name            | Function                                                                         |
|-----------------|----------------------------------------------------------------------------------|
| DEBUG_SYCL_POOL | Enable device memory pool logging on teardown. Useful for profiling allocations. |
| DEBUG_SYCL_MALLOC | Enable verbose per-call logging of device pool alloc/free operations. |
| GGML_SYCL_SUPPORT_VMM | Support to building with VMM code. Defined automatically when the compiler provides `SYCL_EXT_ONEAPI_VIRTUAL_MEM`. |
| GGML_SYCL_FA_ALL_QUANTS | Off by default (commented out in `ggml/src/ggml-sycl/common.hpp`). Widens flash-attention type coverage: mixed K/V types and q4_1/q5_0/q5_1 K/V. Without it, K/V types must match (except the turbo pairs in the [Arc flash-attention section](#intel-arc-a770--dg2-flash-attention-kv-cache)) and only f32/f16/bf16, q4_0, q8_0 and turbo K are accepted. |

## Design Rule

- Open to all contributors.

- All code change should be useful to user:
    - Fix bug.
    - Add new function.
    - Improve the performance/usage.
    - Make code be easy to maintain.
    - ...

- Don't accept the codes of following cases:
    - Break legacy function.
    - Reduce the performance of legacy case in default.
    - Not completed work/the functionality cannot be demonstrated.

- Encourage to use environment variable to control features to be opened/closed.
    - User can evaluate the feature without rebuild the code.
    - Recommend the best features to user by setting them be opened as default.

- Design the code based on the published official releases of oneAPI packages: compiler, library, driver, OS kernel.

- Developers need to maintain the code they submit.

## Known Issues

- `--split-mode row` has a SYCL implementation (row-split buffer type) but is not validated
  on this fork, which tests a single A770.

- AOT (Ahead-of-Time) is opt-in (`GGML_SYCL_DEVICE_ARCH`); the default build is JIT.
  - Good: Builds quickly, smaller size of binary file.
  - Bad: The startup is slow (JIT) in first time, but subsequent performance is unaffected.

- Intel Arc A770 (DG2 / Xe-HPG) stacks used on this fork (JIT builds); `sycl-ls` should list
  the Arc under `level_zero:gpu`:
  - 2026-07, i915: oneAPI 2026.0 (icx/icpx), Intel Graphics Compiler (IGC) 2.36.3,
    intel-compute-runtime 26.22.38646, level-zero-loader 1.28.6
    (`docs/research/sycl/sycl-a770-p5-performance-campaign-2026-07-19.md`).
  - 2026-09, xe on kernel 7.3-rc: oneAPI 2026.1.4, IGC 2.41.5, intel-compute-runtime
    26.35.39758, level-zero-loader 1.32.0 (`docs/research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md`);
    see the xe blitter issue below.

- IGC internal compiler error on `joint_matrix` (XMX). Bleeding-edge IGC (e.g. the 2.38.x
  git builds) can crash with `IGC: Internal Compiler Error: Floating point exception` when
  JIT-compiling any `joint_matrix` kernel on DG2. Use a stable IGC release (2.36.3 verified).

- XMX `joint_matrix` on DG2 requires sub-group size 8, not 16. Forcing
  `[[sycl::reqd_sub_group_size(16)]]` on a `joint_matrix` kernel triggers the IGC ICE above;
  the DG2 XMX systolic depth is 8 (the device reports N=8 in `matrix_combinations`). A
  correct SG=8 kernel lowers to real `dpas` and runs several x scalar FMA. Note the rest of
  the ggml-sycl backend uses sub-group 16 (`GGML_SYCL_WARP_SIZE=16`), so any XMX matmul
  region must run in an SG=8 scope.

- Host build-flag injection corrupts the SYCL device pipeline. When packaging (e.g. Arch
  `makepkg`), disable injected compiler flags (`options=(!buildflags)`); host `-march`
  microarch flags leaking into the device compile can produce garbage GPU output. Do not add
  host CFLAGS to the SYCL build.

### Arc A770 (DG2) on the xe KMD: blitter copies hang, then `Engine reset: engine_class=bcs`

With the Unified Runtime Level Zero adapter routing USM copies to the blitter (its default),
workloads that stream host memory to the device every token (MoE experts left on the CPU by
`--fit`) stall silently and later log

```
xe 0000:03:00.0: [drm] Tile0: GT0: Engine reset: engine_class=bcs, logical_mask: 0x1, guc_id=6, state=0x29
xe 0000:03:00.0: [drm] Tile0: GT0: Timedout job: seqno=..., guc_id=6, flags=0x20 in llama-bench [...]
```

followed by `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` from `ggml_backend_sycl_set_tensor_async`
(the VM is banned after the reset, so every later `VM_BIND` fails). Measured on kernel
7.3-rc1 and 7.3-rc5 with intel-compute-runtime 26.35.39758: 12 failures in 15 long-context
runs with the blitter, 0 in 10 without it. Mechanism, from device coredumps, xe tracepoints
and NEO allocation logs (`docs/research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md`): on xe every
userptr bind of the mmap'd model pages fails with `EPERM` (read-only file mapping, NEO asks
for write access); NEO answers each failure with an unused-allocation eviction sweep, and
that sweep unbinds the blitter's KMD-submitted command buffer while its job is still
pending. With scratch pages enabled the blitter parses zeros up to the next mapped
allocation and halts on an invalid instruction; the next LR-mode suspend then cannot
preempt it and the GuC resets the engine after the 640 ms preempt timeout. The compute
queue uses direct submission (allocations stay bound) and is not affected. Same sweep as
intel/compute-runtime issues #973 and #1010, different victim.

Defaults and workarounds:

- ggml-sycl now sets `UR_L0_USE_COPY_ENGINE=0` (v1 adapter) and
  `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` (v2 adapter) itself when it finds a DG2 GPU (PCI
  ids 0x5690-0x56ff) bound to `xe`. Each variable is left alone when it (or, for the v1
  one, its `SYCL_PI_` alias) is already set, and neither is touched when any copy-engine
  variable asks for copy engines with a value other than 0 (logged at backend
  initialization, also when the defaults are skipped; the variable table above has the
  exact rule). Copies run on the compute queue; decode on
  real text was 1.2 % below the blitter path on this workload (32.38 vs 32.78 t/s).
  Override with `UR_L0_USE_COPY_ENGINE=1` (every copy engine) on the v1 adapter or
  `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=0` on the v2 adapter (both are set because the v2 adapter
  binary also carries the v1 name while the v1 binary does not know the v2 variable); overriding reintroduces the failure described above.
  `GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0` disables the hook and gives the adapter's own
  defaults back; revisit it when a compute-runtime release carries the read-only userptr
  retry below. The hook runs from a load-time constructor in `ggml-sycl/xe-kmd.cpp`, ahead
  of every SYCL entry point in the library; on any other GPU or driver it leaves the
  environment untouched. An application that changes these variables after the library
  is loaded must set `UR_L0_USE_COPY_ENGINE` itself.
- `--prefetch-experts-slots` needs the private copy queue, so it is unavailable under this
  default; the "no private stream" warning says so and names the hook. Enabling it
  requires the v1 override and therefore the blitter path this entry is about; the v2
  adapter never gets a private copy queue. The two are mutually exclusive on DG2 until the
  runtime is fixed.
- `NEOReadDebugKeys=1 DirectSubmissionOverrideBlitterSupport=1` keeps the blitter and
  passed one full run, but decode dropped from 14.5 to 8.7 t/s on the random-token bench.
- The runtime fix: `docs/research/software-stack/patches/0001-neo-retry-userptr-bind-readonly-on-eperm.patch`
  against intel/compute-runtime master makes NEO retry the userptr bind read-only on `EPERM`
  instead of running the eviction sweep. With it the blitter path ran clean with zero failed
  binds (and prefill +33 %, since the staging fallback copies disappear too). NEO master as
  of `8ae033266e` still fails without it. Build DG2-only and load it with
  `ZE_ENABLE_ALT_DRIVERS=/path/to/libze_intel_gpu.so.1` to test without replacing the package.
- `--load-mode none` puts the CPU-placed weights in pinned `SYCL_Host` memory: no userptr
  binds, no failing bind, no sweep. Measured +12 % prefill on the Ornith fit, decode flat,
  and clean with the blitter on. Costs an owned RAM copy instead of shared page cache.
- The i915 driver does not hit this: its userptr binds of file-backed pages succeed, so
  the sweep never runs, and execbuf keeps batch buffers alive for the job's lifetime.

## Q&A

- Error:  `error while loading shared libraries: libsycl.so: cannot open shared object file: No such file or directory`.

  - Potential cause: Unavailable oneAPI installation or not set ENV variables.
  - Solution: Install *oneAPI base toolkit* and enable its ENV through: `source /opt/intel/oneapi/setvars.sh`.

- General compiler error:

  - Remove **build** folder or try a clean-build.

- I can **not** see `[ext_oneapi_level_zero:gpu]` after installing the GPU driver on Linux.

  Please double-check with `sudo sycl-ls`.

  If it's present in the list, please add video/render group to your user then **logout/login** or restart your system:

  ```
  sudo usermod -aG render $USER
  sudo usermod -aG video $USER
  ```
  Otherwise, please double-check the GPU driver installation steps.

- Can I report Ollama issue on Intel GPU to llama.cpp SYCL backend?

  No. We can't support Ollama issue directly, because we aren't familiar with Ollama.

  Suggest reproducing on llama.cpp and report similar issue to llama.cpp. We will support it.

  It's same for other projects including llama.cpp SYCL backend.

- `Native API failed. Native API returns: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`, `ggml_backend_sycl_buffer_type_alloc_buffer: can't allocate 3503030272 Bytes of memory on device`, or `failed to allocate SYCL0 buffer`

  You are running out of Device Memory.

  |Reason|Solution|
  |-|-|
  | The default context is too big. It leads to excessive memory usage.|Set `-c 8192` or a smaller value.|
  | The model is too big and requires more memory than what is available.|Choose a smaller model or change to a smaller quantization, like Q5 -> Q4;<br>Alternatively, use more than one device to load model.|

- `ggml_backend_sycl_buffer_type_alloc_buffer: can't allocate 5000000000 Bytes of memory on device`

  With the default `GGML_SYCL_USE_LEVEL_ZERO_API=1`, llama.cpp requests Level Zero's relaxed maximum-size allocation limit directly. If Level Zero support is disabled at build time or runtime and the allocation goes through SYCL/Unified Runtime instead, enable support for allocations larger than 4 GiB by:
  ```
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    set UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
  ```

- When I set `SYCL_CACHE_PERSISTENT=1` in running time, I meet crash.

  `SYCL_CACHE_PERSISTENT=1` is not recommended by llama.cpp SYCL backend.
  When cache is enabled, SYCL runtime will try to cache and reuse JIT-compiled binaries.

  We find some AI will tell user this cmd to speed up SYCL backend. It only speeds up the startup to skip the JIT process, instead of running speed.

  It will bring negative impact when the SYCL binary file is changed frequently in your running environment. The new & old codes mix will lead to crash.

  Compare to the benefit, it has brought more failed cases.
  If you are not familiar with the SYCL compiler principle of JIT and AOT, please don't use it.

  To restore, you need to remove the local cache: `~/.cache/libsycl_cache/` and execute `unset SYCL_CACHE_PERSISTENT` in running time.

- How to use iGPU and dGPU in same time?

  1. Detect the devices in your running time.
  ```
  source /opt/intel/oneapi/setvars.sh
  ./build/bin/llama-server --list-devices

  or
  ./build/bin/llama-cli --list-devices
  ./build/bin/llama-bench --list-devices
  ./build/bin/llama-completion --list-devices

  Available devices:
    SYCL0: Intel(R) Arc(TM) A770 Graphics (15473 MiB, 15473 MiB free)
    SYCL1: Intel(R) UHD Graphics 770 (59675 MiB, 44986 MiB free)
  ```

  The dGPU will be in the head of this list and iGPU will be the end.
  If not all GPUs are listed, please check the env var: ONEAPI_DEVICE_SELECTOR and unset it.

  2. Set the iGPU and dGPU

  Set the iGPU and dGPU by `./build/bin/llama-server --device SYCL0,SYCL1,SYCLxxx`.


### **GitHub contribution**:
Please add the `[SYCL]` prefix/tag in issues/PRs titles to help the SYCL contributors to check/address them without delay.

## TODO

- Review ZES_ENABLE_SYSMAN: https://github.com/intel/compute-runtime/blob/master/programmers-guide/SYSMAN.md#support-and-limitations
