# llama.cpp for AMD ZenDNN

> [!WARNING]
> **Note:** ZenDNN is **not** the same as zDNN.
> - **ZenDNN** (this page): AMD's deep learning library for AMD EPYC CPUs
> - **zDNN**: IBM's Deep Neural Network acceleration library for IBM Z & LinuxONE Mainframes ([see zDNN documentation](zDNN.md))

- [Background](#background)
- [OS](#os)
- [Hardware](#hardware)
- [Supported Operations](#supported-operations)
- [DataType Supports](#datatype-supports)
- [Linux](#linux)
- [Environment Variable](#environment-variable)
- [Performance Optimization](#performance-optimization)
- [Known Issues](#known-issues)
- [TODO](#todo)

## Background

**ZenDNN** (Zen Deep Neural Network Library) is AMD's high-performance deep learning inference library optimized for AMD EPYC™ CPUs. It provides optimized implementations of key deep learning primitives and operations, delivering significant performance improvements for neural network workloads on AMD Zen-based processor architectures.

**Llama.cpp + ZenDNN**

The llama.cpp ZenDNN backend leverages AMD's optimized matrix multiplication primitives to accelerate inference on AMD CPUs. It utilizes ZenDNN's **LowOHA (Low Overhead Hardware Accelerated)** MatMul operator for efficient GEMM operations with minimal execution overhead, built-in weight caching, and direct access to backend libraries (AOCL BLIS, LibXSMM, OneDNN).

For more information about ZenDNN, visit: https://www.amd.com/en/developer/zendnn.html

## OS

| OS      | Status  | Verified                                       |
|:-------:|:-------:|:----------------------------------------------:|
| Linux   | Support | Ubuntu 20.04, 22.04, 24.04                     |

For the latest list of supported operating systems, see the [ZenDNN Supported OS](https://github.com/amd/ZenDNN/blob/zendnnl/README.md#15-supported-os).

## Hardware

### AMD CPUs

**Recommended Processors**

ZenDNN is optimized for AMD EPYC™ processors and AMD Ryzen™ processors based on "Zen" microarchitecture and newer.

| CPU Family                    | Status  | Notes                              |
|:-----------------------------:|:-------:|:----------------------------------:|
| AMD EPYC™ 9005 Series (Turin)| Support | 5th Gen - Zen 5 architecture       |
| AMD EPYC™ 9004 Series (Genoa)| Support | 4th Gen - Zen 4 architecture       |
| AMD EPYC™ 7003 Series (Milan)| Support | 3rd Gen - Zen 3 architecture       |
| AMD Ryzen™ AI MAX (Strix Halo)| Support | High-performance mobile processors |

*Notes:*

- Best performance is achieved on AMD EPYC™ processors with high core counts (e.g., EPYC 9005 series).
- ZenDNN leverages AMD's advanced CPU features including AVX2 and AVX-512 instruction sets.
- For optimal performance, ensure your system has sufficient memory bandwidth.

## Supported Operations

The ZenDNN backend currently accelerates **matrix multiplication (MUL_MAT)** operations only. Other operations are handled by the standard CPU backend.

| Operation    | Status  | Notes                                          |
|:-------------|:-------:|:----------------------------------------------:|
| MUL_MAT      |    ✓    | Accelerated via ZenDNN LowOHA MatMul           |

*Note:* Since only MUL_MAT is accelerated, models will benefit most from ZenDNN when matrix multiplications dominate the computational workload (which is typical for transformer-based LLMs).

## DataType Supports

| DataType               | Status  | Notes                                         |
|:----------------------:|:-------:|:---------------------------------------------:|
| FP32                   | Support | Full precision floating point                 |
| BF16                   | Support | BFloat16 (best performance on Zen 4/Zen 5)    |

*Notes:*

- **BF16** provides best performance on Zen 4 and Zen 5 EPYC™ processors (Genoa, Turin).

## Linux

### I. Setup Environment

You have two options to set up ZenDNN:

#### Option 1: Automatic Download and Build (Recommended)

CMake will automatically download and build ZenDNN for you:

```sh
# Build llama.cpp - ZenDNN will be automatically downloaded and built
cmake -B build -DGGML_ZENDNN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

No manual ZenDNN installation required. CMake will handle everything automatically.

#### Option 2: Use Custom ZenDNN Installation

If you want to build ZenDNN yourself or use a specific version:

**Step 1: Build ZenDNN from source**

```sh
# Clone ZenDNN repository
git clone https://github.com/amd/ZenDNN.git
cd ZenDNN
git checkout zendnnl

# Build and install (requires CMake >= 3.25)
mkdir build && cd build
cmake ..
cmake --build . --target all
```

Default installation path: `ZenDNN/build/install`

**For detailed build instructions**, refer to the [ZenDNN README](https://github.com/amd/ZenDNN/blob/zendnnl/README.md).

**Step 2: Build llama.cpp with custom ZenDNN path**

```sh
# Using environment variable
export ZENDNN_ROOT=/path/to/ZenDNN/build/install
cmake -B build -DGGML_ZENDNN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

# OR specify path directly in CMake
cmake -B build -DGGML_ZENDNN=ON -DZENDNN_ROOT=/path/to/ZenDNN/build/install -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

### II. Run the Server

#### 1. Download Model

Download LLaMA 3.1 8B Instruct BF16 model:

```sh
# Download from Hugging Face
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct-GGUF --local-dir models/
```

#### 2. Start Server

Run llama.cpp server with ZenDNN acceleration:

```sh
# Set optimal configuration
export OMP_NUM_THREADS=64  # Adjust to your CPU core count
export ZENDNNL_MATMUL_ALGO=2  # Blocked AOCL BLIS for best performance

# Start server
./build/bin/llama-server \
    -m models/Llama-3.1-8B-Instruct.BF16.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -t 64
```

Access the server at `http://localhost:8080`.

**Performance tips**:
- Set `OMP_NUM_THREADS` to match your physical core count
- Use `ZENDNNL_MATMUL_ALGO=2` for optimal performance
- For NUMA systems: `numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server ...`

## Environment Variable

### Build Time

| Name               | Value                                 | Function                                    |
|--------------------|---------------------------------------|---------------------------------------------|
| GGML_ZENDNN        | ON/OFF                                | Enable ZenDNN backend support               |
| ZENDNN_ROOT        | Path to ZenDNN installation           | Set ZenDNN installation directory           |
| GGML_OPENMP        | ON/OFF (recommended: ON)              | Enable OpenMP for multi-threading           |

### Runtime

| Name                    | Value                    | Function                                                          |
|-------------------------|--------------------------|-------------------------------------------------------------------|
| OMP_NUM_THREADS         | Number (e.g., 64)        | Set number of OpenMP threads (recommended: physical core count)   |
| ZENDNNL_MATMUL_ALGO     | 0-5                      | Select MatMul backend algorithm (see Performance Optimization)    |
| ZENDNNL_PROFILE_LOG_LEVEL | 0-4                    | Profiling log level (0=disabled, 4=verbose)                       |
| ZENDNNL_ENABLE_PROFILER | 0 or 1                   | Enable detailed profiling (1=enabled)                             |
| ZENDNNL_API_LOG_LEVEL   | 0-4                      | API log level (0=disabled, 4=verbose)                             |

**Example**:

```sh
export OMP_NUM_THREADS=64
export ZENDNNL_MATMUL_ALGO=2  # Use Blocked AOCL BLIS for best performance
./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "Test" -n 100
```

## Performance Optimization

### MatMul Algorithm Selection

ZenDNN's LowOHA MatMul supports multiple backend algorithms. For **best performance**, use the **Blocked AOCL BLIS** algorithm:

```sh
export ZENDNNL_MATMUL_ALGO=2  # Blocked AOCL BLIS (recommended)
```

**Available algorithms**:

| Value | Algorithm              | Description                                    |
|:-----:|:-----------------------|:----------------------------------------------|
| 0     | Dynamic Dispatch       | Automatic backend selection (default)         |
| 1     | AOCL BLIS              | AOCL BLIS backend                             |
| 2     | AOCL BLIS Blocked      | **Blocked AOCL BLIS (recommended)**           |
| 3     | OneDNN                 | OneDNN backend                                |
| 4     | OneDNN Blocked         | Blocked OneDNN                                |
| 5     | LibXSMM                | LibXSMM backend                               |

### Profiling and Debugging

For detailed profiling and logging options, refer to the [ZenDNN Logging Documentation](https://github.com/amd/ZenDNN/blob/zendnnl/docs/logging.md).

## Known Issues

- **Limited operation support**: Currently only matrix multiplication (MUL_MAT) is accelerated via ZenDNN. Other operations fall back to the standard CPU backend.
- **BF16 support**: BF16 operations require AMD Zen 4 or Zen 5 architecture (EPYC 9004/9005 series). On older CPUs, operations will use FP32.
- **NUMA awareness**: For multi-socket systems, manual NUMA binding may be required for optimal performance.

## Q&A

**Q: How do I verify that ZenDNN backend is being used?**

A: Check the log output when running llama.cpp. You should see messages indicating the ZenDNN backend is initialized. You can also check the backend name in the output.

**Q: What performance improvement can I expect?**

A: Performance gains vary depending on the model size, batch size, and CPU architecture. On AMD EPYC processors, you can typically expect 1.1x-2x speedup compared to standard CPU inference for matrix multiplication operations.

**Q: Can I use ZenDNN on non-AMD processors?**

A: ZenDNN is optimized specifically for AMD processors. While it may work on other x86-64 CPUs, performance benefits are only guaranteed on AMD Zen-based architectures.

**Q: Does ZenDNN support quantized models?**

A: Currently, ZenDNN primarily supports FP32 and BF16 data types. Quantized model support is not available at this time.

**Q: Why is my inference not faster with ZenDNN?**

A: Ensure:
1. You're using an AMD EPYC or Ryzen processor (Zen 2 or newer)
2. `OMP_NUM_THREADS` is set appropriately (physical core count)
3. `ZENDNNL_MATMUL_ALGO=2` is set for best performance (Blocked AOCL BLIS)
4. You're using a sufficiently large model (small models may not benefit as much)
5. Enable profiling to verify ZenDNN MatMul is being called

### **GitHub Contribution**:
Please add the **[ZenDNN]** prefix/tag in issues/PRs titles to help the ZenDNN-team check/address them without delay.

## TODO

- Expand operation support beyond MUL_MAT (attention operations, activations, etc.)
