> [!IMPORTANT]
> This build documentation is specific only to IBM Z & LinuxONE mainframes (s390x). You can find the build documentation for other architectures: [build.md](build.md).

# Build llama.cpp locally (for s390x)

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## CPU Build with BLAS

Building llama.cpp with BLAS support is highly recommended as it has shown to provide performance improvements. Make sure to have OpenBLAS installed in your environment.

```bash
cmake -S . -B build             \
    -DCMAKE_BUILD_TYPE=Release  \
    -DGGML_BLAS=ON              \
    -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release -j $(nproc)
```

**Notes**:

-   For faster repeated compilation, install [ccache](https://ccache.dev/)
-   By default, VXE/VXE2 is enabled. To disable it (not recommended):

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Release  \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_VXE=OFF

    cmake --build build --config Release -j $(nproc)
    ```

-   By default, NNPA is disabled by default. To enable it:

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Release  \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_NNPA=ON

    cmake --build build --config Release -j $(nproc)
    ```

-   For debug builds:

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Debug    \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS
    cmake --build build --config Debug -j $(nproc)
    ```

-   For static builds, add `-DBUILD_SHARED_LIBS=OFF`:

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Release  \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DBUILD_SHARED_LIBS=OFF

    cmake --build build --config Release -j $(nproc)
    ```

## Getting GGUF Models

All models need to be converted to Big-Endian. You can achieve this in three cases:

1. **Use pre-converted models verified for use on IBM Z & LinuxONE (easiest)**

    ![File Type - gguf](https://img.shields.io/badge/File_Type-gguf-fff)

    You can find popular models pre-converted and verified at [s390x Verified Models](https://huggingface.co/collections/taronaeo/s390x-verified-models-672765393af438d0ccb72a08) or [s390x Runnable Models](https://huggingface.co/collections/taronaeo/s390x-runnable-models-686e951824198df12416017e).

    These models have already been converted from `safetensors` to `GGUF` Big-Endian and their respective tokenizers verified to run correctly on IBM z15 and later system.

2. **Convert safetensors model to GGUF Big-Endian directly (recommended)**

    ![File Type - safetensors](https://img.shields.io/badge/File_Type-safetensors-da1e28)

    The model you are trying to convert must be in `safetensors` file format (for example [IBM Granite 3.3 2B](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)). Make sure you have downloaded the model repository for this case.

    Ensure that you have installed the required packages in advance

    ```bash
    pip3 install -r requirements.txt
    ```

    Convert the `safetensors` model to `GGUF`

    ```bash
    python3 convert_hf_to_gguf.py \
        --outfile model-name-be.f16.gguf \
        --outtype f16 \
        --bigendian \
        model-directory/
    ```

    For example,

    ```bash
    python3 convert_hf_to_gguf.py \
        --outfile granite-3.3-2b-instruct-be.f16.gguf \
        --outtype f16 \
        --bigendian \
        granite-3.3-2b-instruct/
    ```

3. **Convert existing GGUF Little-Endian model to Big-Endian**

    ![File Type - gguf](https://img.shields.io/badge/File_Type-gguf-fff)

    The model you are trying to convert must be in `gguf` file format (for example [IBM Granite 3.3 2B GGUF](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF)). Make sure you have downloaded the model file for this case.

    ```bash
    python3 gguf-py/gguf/scripts/gguf_convert_endian.py model-name.f16.gguf BIG
    ```

    For example,

    ```bash
    python3 gguf-py/gguf/scripts/gguf_convert_endian.py granite-3.3-2b-instruct-le.f16.gguf BIG
    mv granite-3.3-2b-instruct-le.f16.gguf granite-3.3-2b-instruct-be.f16.gguf
    ```

    **Notes:**

    - The GGUF endian conversion script may not support all data types at the moment and may fail for some models/quantizations. When that happens, please try manually converting the safetensors model to GGUF Big-Endian via Step 2.

## IBM Accelerators

### 1. SIMD Acceleration

Only available in IBM z15 or later system with the `-DGGML_VXE=ON` (turned on by default) compile flag. No hardware acceleration is possible with llama.cpp with older systems, such as IBM z14/arch12. In such systems, the APIs can still run but will use a scalar implementation.

### 2. NNPA Vector Intrinsics Acceleration

Only available in IBM z16 or later system with the `-DGGML_NNPA=ON` (turned off by default) compile flag. No hardware acceleration is possible with llama.cpp with older systems, such as IBM z15/arch13. In such systems, the APIs can still run but will use a scalar implementation.

### 3. zDNN Accelerator

_Only available in IBM z16 / LinuxONE 4 or later system. No support currently available._

### 4. Spyre Accelerator

_Only available with IBM z17 / LinuxONE 5 or later system. No support currently available._

## Performance Tuning

### 1. Virtualization Setup

It is strongly recommended to use only LPAR (Type-1) virtualization to get the most performance.

Note: Type-2 virtualization is not supported at the moment, while you can get it running, the performance will not be the best.

### 2. IFL (Core) Count

It is recommended to allocate a minimum of 8 shared IFLs assigned to the LPAR. Increasing the IFL count past 8 shared IFLs will only improve Prompt Processing performance but not Token Generation.

Note: IFL count does not equate to vCPU count.

### 3. SMT vs NOSMT (Simultaneous Multithreading)

It is strongly recommended to disable SMT via the kernel boot parameters as it negatively affects performance. Please refer to your Linux distribution's guide on disabling SMT via kernel boot parameters.

### 4. BLAS vs NOBLAS

IBM VXE/VXE2 SIMD acceleration depends on the BLAS implementation. It is strongly recommended to use BLAS.

## Frequently Asked Questions (FAQ)

1. I'm getting the following error message while trying to load a model: `gguf_init_from_file_impl: failed to load model: this GGUF file version 50331648 is extremely large, is there a mismatch between the host and model endianness?`

    Answer: Please ensure that the model you have downloaded/converted is GGUFv3 Big-Endian. These models are usually denoted with the `-be` suffix, i.e., `granite-3.3-2b-instruct-be.F16.gguf`.

    You may refer to the [Getting GGUF Models](#getting-gguf-models) section to manually convert a `safetensors` model to `GGUF` Big Endian.

2. I'm getting extremely poor performance when running inference on a model

    Answer: Please refer to the [Appendix B: SIMD Support Matrix](#appendix-b-simd-support-matrix) to check if your model quantization is supported by SIMD acceleration.

3. I'm building on IBM z17 and getting the following error messages: `invalid switch -march=z17`

    Answer: Please ensure that your GCC compiler is of minimum GCC 15.1.0 version, and have `binutils` updated to the latest version. If this does not fix the problem, kindly open an issue.

4. Failing to install the `sentencepiece` package using GCC 15+

    Answer: The `sentencepiece` team are aware of this as seen in [this issue](https://github.com/google/sentencepiece/issues/1108).

    As a temporary workaround, please run the installation command with the following environment variables.

    ```bash
    export CXXFLAGS="-include cstdint"
    ```

    For example,

    ```bash
    CXXFLAGS="-include cstdint" pip3 install -r requirements.txt
    ```

5. `-DGGML_NNPA=ON` generates gibberish output

    Answer: We are aware of this as detailed in [this issue](https://github.com/ggml-org/llama.cpp/issues/14877). Please either try reducing the number of threads, or disable the compile option using `-DGGML_NNPA=OFF`.

## Getting Help on IBM Z & LinuxONE

1. **Bugs, Feature Requests**

    Please file an issue in llama.cpp and ensure that the title contains "s390x".

2. **Other Questions**

    Please reach out directly to [aionz@us.ibm.com](mailto:aionz@us.ibm.com).

## Appendix A: Hardware Support Matrix

|         | Support | Minimum Compiler Version |
| ------- | ------- | ------------------------ |
| IBM z15 | ✅      |                          |
| IBM z16 | ✅      |                          |
| IBM z17 | ✅      | GCC 15.1.0               |

-   ✅ - supported and verified to run as intended
-   🚫 - unsupported, we are unlikely able to provide support

## Appendix B: SIMD Support Matrix

|            | VX/VXE/VXE2 | NNPA | zDNN | Spyre |
| ---------- | ----------- | ---- | ---- | ----- |
| FP32       | ✅          | ✅   | ❓   | ❓    |
| FP16       | ✅          | ✅   | ❓   | ❓    |
| BF16       | 🚫          | 🚫   | ❓   | ❓    |
| Q4_0       | ✅          | ✅   | ❓   | ❓    |
| Q4_1       | ✅          | ✅   | ❓   | ❓    |
| Q5_0       | 🚫          | 🚫   | ❓   | ❓    |
| Q5_1       | 🚫          | 🚫   | ❓   | ❓    |
| Q8_0       | ✅          | ✅   | ❓   | ❓    |
| Q2_K       | 🚫          | 🚫   | ❓   | ❓    |
| Q3_K       | ✅          | ✅   | ❓   | ❓    |
| Q4_K       | ✅          | ✅   | ❓   | ❓    |
| Q5_K       | ✅          | ✅   | ❓   | ❓    |
| Q6_K       | ✅          | ✅   | ❓   | ❓    |
| TQ1_0      | 🚫          | 🚫   | ❓   | ❓    |
| TQ2_0      | 🚫          | 🚫   | ❓   | ❓    |
| IQ2_XXS    | 🚫          | 🚫   | ❓   | ❓    |
| IQ2_XS     | 🚫          | 🚫   | ❓   | ❓    |
| IQ2_S      | 🚫          | 🚫   | ❓   | ❓    |
| IQ3_XXS    | 🚫          | 🚫   | ❓   | ❓    |
| IQ3_S      | 🚫          | 🚫   | ❓   | ❓    |
| IQ1_S      | 🚫          | 🚫   | ❓   | ❓    |
| IQ1_M      | 🚫          | 🚫   | ❓   | ❓    |
| IQ4_NL     | ✅          | ✅   | ❓   | ❓    |
| IQ4_XS     | ✅          | ✅   | ❓   | ❓    |
| FP32->FP16 | 🚫          | ✅   | ❓   | ❓    |
| FP16->FP32 | 🚫          | ✅   | ❓   | ❓    |

-   ✅ - acceleration available
-   🚫 - acceleration unavailable, will still run using scalar implementation
-   ❓ - acceleration unknown, please contribute if you can test it yourself

Last Updated by **Aaron Teo (aaron.teo1@ibm.com)** on July 25, 2025.
