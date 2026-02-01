# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Standard Build (CPU-only)
```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

**Build time**: ~10 minutes with ccache, ~25 minutes without.

### Backend-Specific Builds
```bash
# CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Metal (macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)

# SYCL (Intel) - requires oneAPI
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
ninja -C build -j $(nproc)
```

### Ninja vs Make
Prefer Ninja (`-G Ninja`) for:
- **Correct header dependency tracking**: Changes to `.hpp` files reliably trigger recompilation
- **Faster no-op builds**: 1.5s vs 73s for Make on large projects

**Warning**: Cannot switch generators in existing build directory:
```bash
rm -rf build && cmake -B build -G Ninja [options]
```

### Running Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

### Code Formatting
```bash
# Format C++ files before committing
clang-format-19 -i <file.cpp>

# Check if formatting would change files (dry-run)
clang-format-19 --dry-run -Werror <file.cpp>
```

## Project Architecture

### Core Directories
- **`src/`**: Main llama library (`llama.cpp`, `llama-*.cpp`)
- **`include/llama.h`**: Public C API header (~2000 lines)
- **`ggml/`**: Core tensor library (vendored ggml framework)
- **`common/`**: Shared utility code for examples
- **`examples/`** and **`tools/`**: 40+ CLI tools
- **`tests/`**: CTest integration

### Key Binaries (in `build/bin/`)
- **`llama-cli`**: Interactive chat/inference
- **`llama-completion`**: Non-interactive text completion (use for scripted tests)
- **`llama-server`**: OpenAI-compatible HTTP server
- **`llama-bench`**: Performance benchmarking
- **`llama-quantize`**: Model quantization

### Backend Structure (`ggml/src/`)
- **`ggml-cpu/`**: CPU backend (AVX/NEON/RVV)
- **`ggml-cuda/`**: NVIDIA CUDA kernels
- **`ggml-metal/`**: Apple Metal shaders
- **`ggml-sycl/`**: Intel SYCL backend
- **`ggml-vulkan/`**: Vulkan compute shaders

## ggml Conventions

### Matrix Multiplication
Matrix multiplication is **unconventional**: `C = ggml_mul_mat(ctx, A, B)` computes:
```
C^T = A * B^T  ⟺  C = B * A^T
```

### Tensor Storage
- Tensors store data in **row-major order**
- Dimension 0 = columns, Dimension 1 = rows, Dimension 2 = matrices

### Naming Patterns
- Use `snake_case` for function, variable, and type names
- Optimize for **longest common prefix**: `number_small`, `number_big` (not `small_number`, `big_number`)
- General pattern: `<class>_<method>` with `<method>` being `<action>_<noun>`
  ```cpp
  llama_model_init();           // class: "llama_model", method: "init"
  llama_sampler_get_seed();     // class: "llama_sampler", method: "get_seed"
  ```

## Coding Guidelines

- **Minimal dependencies**: Avoid adding third-party dependencies
- **Cross-platform**: Test on Linux, macOS, Windows when possible
- **Simple STL**: Avoid fancy modern STL, use basic `for` loops, minimize templates
- **Vertical alignment**: Makes code more readable and easier to batch edit
- **Formatting**: 4 spaces, brackets on same line, `void * ptr`, `int & a`
- **Public API types**: Use `int32_t` etc., `size_t` for allocation sizes

## Development Workflow

### Verification Commands
```bash
# Non-interactive completion (deterministic output for testing)
./build/bin/llama-completion -m model.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Benchmark
./build/bin/llama-bench -m model.gguf -p 512 -n 128

# Test backend operations (after modifying ggml operators)
./build/bin/test-backend-ops
```

### Server Unit Tests
```bash
cd tools/server/tests
pip install -r requirements.txt  # if needed
./tests.sh
```

## CI and Validation

### Before Submitting PRs
1. Format code: `clang-format-19 -i <files>`
2. Build: `cmake --build build --config Release`
3. Test: `ctest --test-dir build --output-on-failure`
4. For ggml changes: Run `test-backend-ops` on multiple backends
5. For server changes: Run server unit tests

### Triggering Heavy CI
Add `ggml-ci` to commit message to trigger extended CI workloads.

## Documentation

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **Add New Model**: `docs/development/HOWTO-add-model.md`
- **Contributing**: `CONTRIBUTING.md`
