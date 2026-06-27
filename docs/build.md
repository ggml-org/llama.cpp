# Build llama.cpp locally

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the Code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

The following sections describe how to build with different backends and options.

* [CPU Build](#cpu-build)
* [BLAS Build](#blas-build)
* [SYCL](#sycl)
* [Vulkan](#vulkan)
* [Arm® KleidiAI™](#arm-kleidiai)
* [Android](#android-1)
* [OpenVINO](#openvino)
* [Notes about GPU-accelerated backends](#notes-about-gpu-accelerated-backends)

## CPU Build

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

The following flags are supported by the CPU backend:

| Flag                        | Default | Description                                                                                                                                                                                                              |
|-----------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GGML_CPU_ALL_VARIANTS`     | OFF     | Build all variants of the CPU backend (requires `GGML_BACKEND_DL`). Enables multiple dispatch ISAs (AVX2, AVX512, AMX, ...) under one build.                                                                          |
| `GGML_CPU_REPACK`           | ON      | Use runtime weight conversion of Q4_0 to Q4_X_X                                                                                                                                                                          |
| `GGML_CPU_KLEIDIAI`         | OFF     | Use KleidiAI optimized kernels if applicable                                                                                                                                                                            |
| `GGML_SSE42`                | ON      | Enable SSE 4.2                                                                                                                                                                                                            |
| `GGML_F16C`                 | ON      | Enable F16C                                                                                                                                                                                                               |
| `GGML_AVX`                  | ON      | Enable AVX                                                                                                                                                                                                                |
| `GGML_AVX_VNNI`             | OFF     | Enable AVX-VNNI                                                                                                                                                                                                           |
| `GGML_AVX2`                 | ON      | Enable AVX2                                                                                                                                                                                                               |
| `GGML_BMI2`                 | ON      | Enable BMI2                                                                                                                                                                                                               |
| `GGML_FMA`                  | ON      | Enable FMA                                                                                                                                                                                                                |
| `GGML_AVX512`               | OFF     | Enable AVX512F                                                                                                                                                                                                            |
| `GGML_AVX512_VBMI`          | OFF     | Enable AVX512-VBMI                                                                                                                                                                                                        |
| `GGML_AVX512_VNNI`          | OFF     | Enable AVX512-VNNI                                                                                                                                                                                                        |
| `GGML_AVX512_BF16`          | OFF     | Enable AVX512-BF16                                                                                                                                                                                                        |
| `GGML_AMX_TILE`             | OFF     | Enable AMX-TILE                                                                                                                                                                                                           |
| `GGML_AMX_INT8`             | OFF     | Enable AMX-INT8                                                                                                                                                                                                           |
| `GGML_AMX_BF16`             | OFF     | Enable AMX-BF16                                                                                                                                                                                                           |
| `GGML_CPU_HBM`              | OFF     | Use memkind for CPU HBM                                                                                                                                                                                                   |
| `GGML_OPENMP`               | ON      | Use OpenMP (used for parallelizing some compute loops)                                                                                                                                                                   |

For ARM targets, the following flags are available:

| Flag             | Default | Description                                          |
|------------------|---------|------------------------------------------------------|
| `GGML_CPU_ARM_ARCH` | ""      | CPU architecture for ARM                            |
| `GGML_LASX`      | ON      | Enable lasx (Loongson)                               |
| `GGML_LSX`       | ON      | Enable lsx (Loongson)                                |
| `GGML_RVV`       | ON      | Enable rvv (RISC-V Vector)                           |
| `GGML_RV_ZFH`    | ON      | Enable riscv zfh                                     |
| `GGML_RV_ZVFH`   | ON      | Enable riscv zvfh                                    |
| `GGML_RV_ZICBOP` | ON      | Enable riscv zicbop                                  |
| `GGML_XTHEADVECTOR` | OFF   | Enable xtheadvector                                  |

## BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Using BLAS doesn't affect the generation performance. There are currently several different BLAS implementations available for build and use:

### Accelerate Framework

This is only available on MacOS and is enabled by default. You can explicitly disable it by setting the `GGML_ACCELERATE` cmake option to `OFF`.

### OpenBLAS

```bash
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release
```

### Intel oneMKL

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release
```

### Amazon AOCL BLAS

```bash
sudo apt install libaocl-blas-dev
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME
cmake --build build --config Release
```

### Other BLAS libraries

Any other BLAS library can be used by setting the `GGML_BLAS_VENDOR` option. See the [CMake documentation](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors) for a list of supported vendors.

## SYCL

SYCL is a higher-level programming model to improve programming productivity on various hardware accelerators.

llama.cpp based on SYCL is used to **support Intel GPU** (Data Center Max series, Flex series, Arc series, Built-in GPU and iGPU).

For detailed info, please refer to [llama.cpp for SYCL](./backend/SYCL.md).

## Vulkan

### For Windows Users:
**w64devkit**

Download and extract [`w64devkit`](https://github.com/skeeto/w64devkit/releases).

Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

Launch `w64devkit.exe` and run the following commands to copy Vulkan dependencies:
```sh
SDK_VERSION=1.3.283.0
cp /VulkanSDK/$SDK_VERSION/Bin/glslc.exe $W64DEVKIT_HOME/bin/
cp /VulkanSDK/$SDK_VERSION/Lib/vulkan-1.lib $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/
cp -r /VulkanSDK/$SDK_VERSION/Include/* $W64DEVKIT_HOME/x86_64-w64-mingw32/include/
cat > $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/pkgconfig/vulkan.pc <<EOF
Name: Vulkan-Loader
Description: Vulkan Loader
Version: $SDK_VERSION
Libs: -lvulkan-1
EOF

```

Switch into the `llama.cpp` directory and build using CMake.
```sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

**Git Bash MINGW64**

Download and install [`Git-SCM`](https://git-scm.com/downloads/win) with the default settings

Download and install [`Visual Studio Community Edition`](https://visualstudio.microsoft.com/) and make sure you select `C++`

Download and install [`CMake`](https://cmake.org/download/) with the default settings

Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

Go into your `llama.cpp` directory and right click, select `Open Git Bash Here` and then run the following commands

```
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

Now you can load the model in conversation mode using `Vulkan`

## Arm® KleidiAI™

KleidiAI is a library of optimized microkernels for AI workloads, specifically designed for Arm CPUs. These microkernels enhance performance and can be enabled for use by the CPU backend.

To enable KleidiAI, go to the llama.cpp directory and build using CMake

```bash
cmake -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```

## Android

To read documentation for how to build on Android, [click here](./android.md)

## OpenVINO

[OpenVINO](https://docs.openvino.ai/) is an open-source toolkit for optimizing and deploying high-performance AI inference, specifically designed for Intel hardware (CPUs, GPUs, and NPUs).

For build instructions and usage examples, refer to [OPENVINO.md](backend/OPENVINO.md).

---

## Notes about GPU-accelerated backends

The GPU may still be used to accelerate some parts of the computation even when using the `-ngl 0` option. You can fully disable GPU acceleration by using `--device none`.

In most cases, it is possible to build and use multiple backends at the same time. For example, you can build llama.cpp with both SYCL and Vulkan support by using the `-DGGML_SYCL=ON -DGGML_VULKAN=ON` options with CMake. At runtime, you can specify which backend devices to use with the `--device` option. To see a list of available devices, use the `--list-devices` option.

Backends can be built as dynamic libraries that can be loaded dynamically at runtime. This allows you to use the same llama.cpp binary on different machines with different GPUs. To enable this feature, use the `GGML_BACKEND_DL` option when building.
