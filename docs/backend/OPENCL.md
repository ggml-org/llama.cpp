# llama.cpp for OpenCL

- [llama.cpp for OpenCL](#llamacpp-for-opencl)
  - [Background](#background)
    - [Llama.cpp + OpenCL](#llamacpp--opencl)
  - [OS](#os)
  - [Hardware](#hardware)
    - [Adreno GPU](#adreno-gpu)
  - [DataType Supports](#datatype-supports)
  - [Model Preparation](#model-preparation)
  - [Binary Kernel Library](#binary-kernel-library)
  - [CMake Options](#cmake-options)
  - [Android](#android)
    - [I. Setup Environment](#i-setup-environment)
    - [II. Build llama.cpp](#ii-build-llamacpp)
  - [Windows 11 Arm64](#windows-11-arm64)
    - [I. Setup Environment](#i-setup-environment-1)
    - [II. Build llama.cpp](#ii-build-llamacpp-1)
  - [Linux](#linux)
    - [I. Setup Environment](#i-setup-environment-2)
    - [II. Build llama.cpp](#ii-build-llamacpp-2)
  - [Known Issues](#known-issues)
  - [TODO](#todo)

## Background

OpenCL (Open Computing Language) is an open, royalty-free standard for cross-platform, parallel programming of diverse accelerators found in supercomputers, cloud servers, personal computers, mobile devices and embedded platforms. OpenCL specifies a programming language (based on C99) for programming these devices and application programming interfaces (APIs) to control the platform and execute programs on the compute devices. Similar to CUDA, OpenCL has been widely used to program GPUs and is supported by most GPU vendors.

### Llama.cpp + OpenCL

The llama.cpp OpenCL backend is designed to enable llama.cpp on **Qualcomm Adreno GPU** firstly via OpenCL. Thanks to the portabilty of OpenCL, the OpenCL backend can also run on certain Intel GPUs such as those that do not have [SYCL](/docs/backend/SYCL.md) support although the performance is not optimal.

## OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Android | Support | Snapdragon 8 Gen 3, Snapdragon 8 Elite         |
| Windows | Support | Windows 11 Arm64 with Snapdragon X Elite       |
| Linux   | Support | Ubuntu 22.04 WSL2 with Intel 12700H            |

## Hardware

### Adreno GPU

**Verified devices**

| Adreno GPU                            | Status  |
|:-------------------------------------:|:-------:|
| Adreno 750 (Snapdragon 8 Gen 3)       | Support |
| Adreno 830 (Snapdragon 8 Elite)       | Support |
| Adreno 840 (Snapdragon 8 Elite Gen 5) | Support |
| Adreno X1-85 (Snapdragon X Elite)     | Support |
| Adreno X2-90 (Snapdragon X2 Elite)    | Support |

> A6x GPUs with a recent driver and compiler are supported; they are usually found in IoT platforms.
However, A6x GPUs in phones are likely not supported due to the outdated driver and compiler.

## DataType Supports

| DataType               | Status                     |
|:----------------------:|:--------------------------:|
| Q1_0                   | Support                    |
| Q4_0                   | Support                    |
| Q4_1                   | Support                    |
| Q5_0                   | Support                    |
| Q5_1                   | Support                    |
| Q8_0                   | Support                    |
| Q4_K                   | Support                    |
| Q5_K                   | Support                    |
| Q6_K                   | Support                    |
| MXFP4                  | Support                    |
| IQ4_NL                 | Support                    |

## Model Preparation

Since common quantizations are supported now, it is recommanded to download GGUF models directly from Huggingface.

## Binary Kernel Library

A prebuilt binary kernel library has been introduced for Adreno GPUs.
It currently targets X2 GPUs (X2-90, X2-85 and X2-45) found in Snapdragon X2 SoC.
The library currently contains kernels for MUL_MAT_ID with Q4_0, Q4_1, Q4_K, MXFP4.
The library must be manually downloaded from https://softwarecenter.qualcomm.com/catalog/item/Adreno_Kernel_Library_GGML.

To allow using the kernel library, add `-DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=ON` when configuring with CMake.
Then, extract `adreno-opencl-kernels.dll` from the zip file downloaded from the above URL and put it alongside the executables.
If kernels compatible with the current GPU are found in the library, they will be loaded and used.


## CMake Options

The OpenCL backend has the following CMake options that control the behavior of the backend.

| CMake options                        | Default value  | Description                               |
|:------------------------------------:|:--------------:|:------------------------------------------|
| `GGML_OPENCL_EMBED_KERNELS`          | `ON`           | Embed OpenCL kernels into the executable. |
| `GGML_OPENCL_USE_ADRENO_KERNELS`     | `ON`           | Use kernels optimized for Adreno.         |
| `GGML_OPENCL_USE_ADRENO_BIN_KERNELS` | `OFF`          | Allow using binary kernel lib for Adreno. |

## Android

Ubuntu 22.04 is used for targeting Android. Make sure the following tools are accessible from command line,

* Git
* CMake 3.29
* Ninja
* Python3

### I. Setup Environment

1. **Install NDK**

```sh
cd ~
wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip && \
unzip commandlinetools-linux-8512546_latest.zip && \
mkdir -p ~/android-sdk/cmdline-tools && \
mv cmdline-tools latest && \
mv latest ~/android-sdk/cmdline-tools/ && \
rm -rf commandlinetools-linux-8512546_latest.zip

yes | ~/android-sdk/cmdline-tools/latest/bin/sdkmanager "ndk;26.3.11579264"
```

2. **Install OpenCL Headers and Library**

```sh
mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/KhronosGroup/OpenCL-Headers && \
cd OpenCL-Headers && \
cp -r CL ~/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include

cd ~/dev/llm

git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && \
cd OpenCL-ICD-Loader && \
mkdir build_ndk26 && cd build_ndk26 && \
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/android-sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$HOME/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24 \
  -DANDROID_STL=c++_shared && \
ninja && \
cp libOpenCL.so ~/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
```

### II. Build llama.cpp

```sh
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && \
cd llama.cpp && \
mkdir build-android && cd build-android

cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/android-sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON

ninja
```

## Windows 11 Arm64

A Snapdragon X Elite device with Windows 11 Arm64 is used. Make sure the following tools are accessible from command line,

* Git
* CMake 3.29
* Clang 19
* Ninja
* Visual Studio 2022
* Powershell 7
* Python

Visual Studio provides necessary headers and libraries although it is not directly used for building.
Alternatively, Visual Studio Build Tools can be installed instead of the full Visual Studio.

> Note that building using Visual Studio's cl compiler is not supported. Clang must be used. Clang depends on libraries provided by Visual Studio to work. Therefore, Visual Studio must be installed. Alternatively, Visual Studio Build Tools can be installed instead of the full Visual Studio.

Powershell 7 is used for the following commands.
If an older version of Powershell is used, these commands may not work as they are.

### I. Setup Environment

1. **Install OpenCL Headers and Library**

```powershell
mkdir -p ~/dev/llm

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-Headers && cd OpenCL-Headers
mkdir build && cd build
cmake .. -G Ninja `
  -DBUILD_TESTING=OFF `
  -DOPENCL_HEADERS_BUILD_TESTING=OFF `
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF `
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" `
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install
```

### II. Build llama.cpp

```powershell

mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
mkdir build && cd build

cmake .. -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$HOME/dev/llm/llama.cpp/cmake/arm64-windows-llvm.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" `
  -DBUILD_SHARED_LIBS=OFF `
  -DGGML_OPENCL=ON
ninja
```

## Linux

The two steps just above also apply to Linux. When building for linux, the commands are mostly the same as those for PowerShell on Windows, but in the second step they do not have the `-DCMAKE_TOOLCHAIN_FILE` parameter, and then in both steps the backticks are replaced with back slashes.

If not installed already, install Git, CMake, Clang, Ninja and Python, then run in the terminal the following:

### I. Setup Environment

1. **Install OpenCL Headers and Library**

```bash
mkdir -p ~/dev/llm

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-Headers && cd OpenCL-Headers
mkdir build && cd build
cmake .. -G Ninja \
  -DBUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install

cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" \
  -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"
cmake --build . --target install
```

### II. Build llama.cpp

```bash
mkdir -p ~/dev/llm
cd ~/dev/llm

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
mkdir build && cd build

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON
ninja
```

## Known Issues

- Flash attention does not always improve performance.
- Currently OpenCL backend works on A6xx GPUs with recent drivers and compilers (usually found in IoT platforms).
  However, it does not work on A6xx GPUs found in phones with old drivers and compilers.

## TODO

- Improve flash attention
- Improve OpenCL C kernels performance
