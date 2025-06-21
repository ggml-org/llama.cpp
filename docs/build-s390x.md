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

Building llama.cpp with BLAS support is highly recommended as it has shown to provide performance improvements.

```bash
cmake -S . -B build             \
    -DCMAKE_BUILD_TYPE=Release  \
    -DGGML_BLAS=ON              \
    -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release -j $(nproc)
```

**Notes**:
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- By default, VXE/VXE2 is enabled. To disable it (not recommended):

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Release  \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_VXE=OFF

    cmake --build build --config Release -j $(nproc)
    ```

- For debug builds:

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Debug    \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS

    cmake --build build --config Debug -j $(nproc)
    ```

- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:

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

    You can find popular models pre-converted and verified at [s390x Ready Models](hf.co/collections/taronaeo/s390x-ready-models-672765393af438d0ccb72a08).

    These models and their respective tokenizers are verified to run correctly on IBM Z & LinuxONE.

2. **Convert safetensors model to GGUF Big-Endian directly (recommended)**

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

Only available in IBM z15 or later system with the `-DGGML_VXE=ON` (turned on by default) compile flag. No hardware acceleration is possible with llama.cpp with older systems, such as IBM z14 or EC13. In such systems, the APIs can still run but will use a scalar implementation.

### 2. zDNN Accelerator

*Only available in IBM z16 or later system. No direction at the moment.*

### 3. Spyre Accelerator

*No direction at the moment.*

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

## Getting Help on IBM Z & LinuxONE

1. **Bugs, Feature Requests**

    Please file an issue in llama.cpp and ensure that the title contains "s390x".

2. **Other Questions**

    Please reach out directly to [aionz@us.ibm.com](mailto:aionz@us.ibm.com).

