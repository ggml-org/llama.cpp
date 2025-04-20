# llama.cpp for OpenCL

- [Background](#background)
- [OS](#os)
- [Hardware](#hardware)
- [DataType Supports](#datatype-supports)
- [Model Preparation](#model-preparation)
- [CMake Options](#cmake-options)
- [Android](#android)
- [Windows 11 Arm64](#windows-11-arm64)
- [Known Issue](#known-issues)
- [TODO](#todo)

## Background

OpenCL (Open Computing Language) is an open, royalty-free standard for cross-platform, parallel programming of diverse accelerators found in supercomputers, cloud servers, personal computers, mobile devices and embedded platforms. OpenCL specifies a programming language (based on C99) for programming these devices and application programming interfaces (APIs) to control the platform and execute programs on the compute devices. Similar to CUDA, OpenCL has been widely used to program GPUs and is supported by most GPU vendors.

### Llama.cpp + OpenCL

The llama.cpp OpenCL backend is designed to enable llama.cpp on **Qualcomm Adreno GPU** firstly via OpenCL. Thanks to the portabilty of OpenCL, the OpenCL backend can also run on certain Intel GPUs although the performance is not optimal.

## OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Android | Support | Snapdragon 8 Gen 3, Snapdragon 8 Elite         |
| Windows | Support | Windows 11 Arm64 with Snapdragon X Elite       |
| Linux   | Support | Ubuntu 22.04 WSL2 with Intel 12700H            |

## Hardware

### Adreno GPU

**Verified devices**

| Adreno GPU                           | Status  |
|:------------------------------------:|:-------:|
| Adreno 750 (Snapdragon 8 Gen 3)      | Support |
| Adreno 830 (Snapdragon 8 Elite)      | Support |
| Adreno X85 (Snapdragon X Elite)      | Support |

## DataType Supports

| DataType               | Status                     |
|:----------------------:|:--------------------------:|
| Q4_0                   | Support                    |
| Q6_K                   | Support, but not optimized |

## Model Preparation

You can refer to the general [*Prepare and Quantize*](README.md#prepare-and-quantize) guide for model prepration.

Currently we support `Q4_0` quantization and have optimize for it. To achieve best performance on Adreno GPU, add `--pure` to `llama-quantize`. For example,

```sh
./llama-quantize --pure ggml-model-qwen2.5-3b-f16.gguf ggml-model-qwen-3b-Q4_0.gguf Q4_0
```

Since `Q6_K` is also supported, `Q4_0` quantization without `--pure` will also work. However, the performance will be worse compared to pure `Q4_0` quantization.

## CMake Options

The OpenCL backend has the following CMake options that control the behavior of the backend.

| CMake options                     | Default value  | Description                               |
|:---------------------------------:|:--------------:|:------------------------------------------|
| `GGML_OPENCL_EMBED_KERNELS`       | `ON`           | Embed OpenCL kernels into the executable. |
| `GGML_OPENCL_USE_ADRENO_KERNELS`  | `ON`           | Use kernels optimized for Adreno.         |

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

Visual Studio provides necessary headers and libraries although it is not directly used for building.
Alternatively, Visual Studio Build Tools can be installed instead of the full Visual Studio.

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

## Known Issues

- Currently OpenCL backend does not work on Adreno 6xx GPUs.

## TODO

- Optimization for Q6_K
- Support and optimization for Q4_K
