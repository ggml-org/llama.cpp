# llama.cpp (SYCL Coalescing Worktree)

## Project Overview
`llama.cpp` is a high-performance C/C++ inference engine for Large Language Models (LLMs), designed for local execution on a wide variety of hardware.
This specific worktree, `sycl-coalescing`, is focused on optimizing the **SYCL backend** (targeting Intel GPUs) through memory coalescing techniques and other kernel optimizations.

## Key Directories & Files

*   **`ggml/src/ggml-sycl/`**: Core implementation of the SYCL backend. Contains kernel code (`.cpp`, `.hpp`) and backend logic.
*   **`docs/plans/`**: Contains design documents and implementation plans for the current coalescing work.
    *   *Active Plan:* `2026-01-02-q6k-variable-tile-impl.md`
*   **`docs/backend/SYCL.md`**: Comprehensive documentation for the SYCL backend, including hardware support, build flags, and environment variables.
*   **`CMakeLists.txt`**: Main build configuration.
*   **`examples/sycl/`**: SYCL-specific examples and helper scripts.

## Build Instructions

The project uses **CMake**.

### Prerequisites
*   **Intel oneAPI Base Toolkit** (for `icx`/`icpx` compilers and MKL/oneDNN libraries).
*   Ensure environment variables are set: `source /opt/intel/oneapi/setvars.sh`

### SYCL Build
To build with SYCL support (optimized for Intel GPUs):

```bash
# Configure (Release mode recommended)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release

# Optional: Enable FP16 (recommended for performance)
# cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON ...

# Build
cmake --build build --config Release -j $(nproc)
```

### Useful Build Flags (CMake)
*   `-DGGML_SYCL=ON`: Enable SYCL backend.
*   `-DGGML_SYCL_F16=ON`: Enable FP16 support.
*   `-DGGML_SYCL_DNN=OFF`: Disable oneDNN (use MKL only) if needed.
*   `-DGGML_SYCL_DISABLE_GRAPH=1`: Disable SYCL graph capturing (useful for debugging).

## Running & Testing

### Environment Variables
*   `ONEAPI_DEVICE_SELECTOR`: Select specific devices (e.g., `level_zero:0`).
*   `GGML_SYCL_DEBUG`: Set to `1` for verbose SYCL backend logging.
*   `ZES_ENABLE_SYSMAN=1`: Enable system management (memory queries) - recommended.

### Basic Inference
```bash
./build/bin/llama-cli -m <path_to_model.gguf> -p "Hello world" -n 100 -ngl 99
```
*   `-ngl 99`: Offload all layers to GPU.

### Listing Devices
```bash
./build/bin/llama-ls-sycl-device
```

## Development Conventions

*   **Style**: Adhere to the existing C++ style in `ggml/src/ggml-sycl/`.
*   **Testing**:
    *   Use `llama-bench` to verify performance improvements.
    *   Use `llama-perplexity` to ensure accuracy (no degradation).
*   **Profiling**:
    *   VTune profiling is a key part of the workflow for this worktree.
    *   Enable JIT profiling support if necessary (see `docs/backend/SYCL.md` or internal docs).

## Current Context (SYCL Coalescing)
The current focus is on optimizing memory access patterns (coalescing) in SYCL kernels, particularly for quantized formats like Q6_K. Check `docs/plans/` for the latest design details.
