# Build llama.cpp locally (Raudbjorn fork)

This fork ships five backends: **CPU, BLAS, SYCL, Vulkan, and OpenVINO**. CUDA, HIP/ROCm, Metal, OpenCL, CANN, MUSA, WebGPU, RPC, Hexagon, BLIS, ZenDNN, IBM zDNN, and VirtGPU are upstream-only -- see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) if you need them. The TurboQuant+ codec stack (`turbo2`/`turbo3`/`turbo4`/`TQ3_1S`/`TQ4_1S`) requires either SYCL or Vulkan to be built in for GPU acceleration.

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the Code:**

```bash
git clone https://github.com/Raudbjorn/ggml-llama.cpp
cd ggml-llama.cpp
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
KleidiAI provides optimized Arm CPU microkernels used by the ggml CPU backend. Enabling it at build time makes those kernels available; it does not force every operation to use KleidiAI. At runtime, llama.cpp selects the best compatible CPU kernel from the detected CPU features, tensor type, operation shape, and active backend priority.

Supported targets:

| Platform | Supported ABI / architecture | Notes |
| --- | --- | --- |
| Linux | AArch64 / arm64 | Runtime CPU feature detection is automatic. |
| Android | `arm64-v8a` | Use the Android NDK command below for a portable build. |
| Apple | arm64 | Runtime CPU feature detection is automatic. Non-streaming SVE vector length is treated as unavailable. |
| Windows | arm64 | Runtime CPU feature detection is automatic. SMCU count is treated as unknown until a detection path is verified. |

`GGML_CPU_KLEIDIAI=ON` is valid only for AArch64/arm64 builds. Do not enable it for x86, 32-bit Arm, or Android ABIs other than `arm64-v8a`.

### Native AArch64/arm64 build

From the llama.cpp source directory:

```bash
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```

### Android arm64-v8a NDK build

Set `ANDROID_NDK` to the Android NDK root, then run the following from the llama.cpp source directory. This command configures a portable Android `arm64-v8a` build with KleidiAI enabled and avoids Android dependencies that are not part of the NDK stable native API set.

```bash
cmake -S . -B build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-android --config Release --parallel
cmake --install build-android --prefix {install-dir} --config Release
```

Important Android options:

- `GGML_CPU_KLEIDIAI=ON` enables KleidiAI for Android `arm64-v8a`.
- `GGML_NATIVE=OFF` is required for cross-compilation because the build host CPU is not the Android target CPU.
- `GGML_OPENMP=OFF` avoids adding an OpenMP runtime dependency to this NDK command-line build.
- `GGML_LLAMAFILE=OFF` avoids the llamafile backend, which is not supported on Android.
- `LLAMA_OPENSSL=OFF` avoids depending on OpenSSL, which is not part of the Android NDK stable native API set.

The Android Studio project under `examples/llama.android` enables KleidiAI automatically for `arm64-v8a`. For Android command-line CMake builds on `arm64-v8a`, pass `-DGGML_CPU_KLEIDIAI=ON` explicitly.

Global -march flags such as `-march=armv8.7a` flag are not required for a portable Android `arm64-v8a` build. Global `-march` flags raise the baseline instruction set for generic code. No manual architecture-specific source selection is required; llama.cpp selects compatible KleidiAI kernels at runtime. The KleidiAI libraries internal CMake handles the -march flags for each particular kernel.

### Verifying the build

Run an installed or in-tree binary:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "What is a car?"
```

If KleidiAI is enabled, the output contains a line similar to:

```
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```

This confirms that the model has tensors allocated through the KleidiAI CPU buffer. It does not prove that every operation, or any specific SME-family operation, used a KleidiAI microkernel. Runtime CPU features, tensor type, operation shape, and backend priority still control dispatch.

Depending on the build target, another backend may have higher priority than the CPU backend. To force CPU execution for a run, disable higher priority backends at build time, for example `-DGGML_METAL=OFF`, or use a runtime device option such as `--device none` where supported.

### Runtime dispatch

KleidiAI microkernels use Arm CPU features such as dotprod, i8mm, SVE, and SME/SME2. Build-time configuration makes the kernels available. Runtime dispatch selects a compatible kernel for the detected CPU and operation. Older or lower-feature CPUs fall back automatically to compatible kernels.

KleidiAI accelerates selected `GGML_OP_MUL_MAT` paths for F32 and common quantized formats. Exact coverage depends on the bundled KleidiAI version and the llama.cpp runtime selector, so unsupported tensor types, unsupported operation shapes, or higher priority backends may bypass KleidiAI even when the CPU supports the required Arm feature. This is also why a model may not use SME-family kernels on SME-capable hardware.

The current llama.cpp KleidiAI SVE selector only enables SVE kernels when the runtime SVE vector length is known to be QK8_0 bytes, currently 32 bytes. Linux and Android query this at runtime. Apple reports SVE capability separately from userspace non-streaming SVE availability, so llama.cpp treats the SVE vector length as unknown there. Windows exposes SVE feature presence but not the runtime SVE vector length used by this selector, so that value is also treated as unknown. Windows arm64 also treats SMCU count as unknown until a detection mechanism is verified.

The set of available SME-family kernels depends on the bundled KleidiAI version and the detected CPU capabilities. Production configuration does not require any KleidiAI runtime environment variables.

### Diagnostics and debug overrides

KleidiAI runtime environment variables are diagnostics/debug overrides, not production configuration. Leave them unset for normal use.

`GGML_KLEIDIAI_SME` controls SME-family kernel selection and overrides the maximum number of threads assigned to selected quantized SME-family kernels:

- Not set: use automatic runtime detection.
- `0`: disable SME-family kernels.
- `<n> > 0`: enable compatible SME-family kernels and allow up to `<n>` threads for quantized SME-family kernels.

On Windows arm64, use `GGML_KLEIDIAI_SME=<n>` as the temporary diagnostics/debug override for SME thread-cap calibration until automatic SMCU count detection is verified.

If the CPU does not support the required SME-family capability for a bundled kernel, that kernel is disabled regardless of the environment variable.


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
