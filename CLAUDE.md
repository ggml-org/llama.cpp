# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands (Intel SYCL)

**IMPORTANT**: Always source oneAPI before building or running:
```bash
source /opt/intel/oneapi/setvars.sh --force
```

### Build
```bash
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
ninja -C build -j $(nproc)
```

**Build time**: ~10 minutes with ccache, ~25 minutes without.

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
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build --output-on-failure -j $(nproc)

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

### Code Formatting
```bash
clang-format-19 -i <file.cpp>
clang-format-19 --dry-run -Werror <file.cpp>  # dry-run check
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

### Inference Flow
1. **Model loading** (`llama_model_load`): Reads GGUF file, maps weights to tensors
2. **Context creation** (`llama_init_from_model`): Allocates KV cache, scratch buffers
3. **Tokenization** (`llama_tokenize`): Text → token IDs
4. **Graph building** (`llama_build_graph`): Creates ggml computation graph per batch
5. **Graph execution** (`ggml_backend_graph_compute`): Dispatches to CPU/GPU backends
6. **Sampling** (`llama_sampler_sample`): Token selection from logits

### Weight Caching (GPU Backends)
GPU backends cache weights on-device for repeated inference:
- **CUDA**: `ggml_cuda_pool` with per-device allocation tracking
- **SYCL**: `unified_cache` with tiered memory (device → pinned host → mmap)
- Weights are identified by tensor name hash + model ID for cache keys

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

## Development Workflow (Machine-Specific)

### Model Locations
Models are stored in `/Storage/GenAI/models/`:

**Mistral 7B variants** (standard benchmark model):
- `mistral-7b-v0.1.Q2_K.gguf` (2.9G) - Smallest, lowest quality
- `mistral-7b-v0.1.Q3_K_S.gguf` (3.0G)
- `mistral-7b-v0.1.Q3_K_M.gguf` (3.3G)
- `mistral-7b-v0.1.Q3_K_L.gguf` (3.6G)
- `mistral-7b-v0.1.Q4_0.gguf` (3.9G) - **Default for benchmarks**
- `mistral-7b-v0.1.Q4_K_S.gguf` (3.9G)
- `mistral-7b-v0.1.Q4_K_M.gguf` (4.1G) - Good quality/size balance
- `mistral-7b-v0.1.Q5_0.gguf` (4.7G)
- `mistral-7b-v0.1.Q5_K_S.gguf` (4.7G)
- `mistral-7b-v0.1.Q5_K_M.gguf` (4.8G)
- `mistral-7b-v0.1.Q6_K.gguf` (5.6G)
- `mistral-7b-v0.1.Q8_0.gguf` (7.2G) - Highest quality

**GPT-OSS models** (large MoE, native MXFP4):
- `gpt-oss-20b-mxfp4.gguf` (12G) - Smaller variant
- `gpt-oss-120b-mxfp4-*.gguf` (60G total, 3-part split) - Full model

### Verification Commands
```bash
source /opt/intel/oneapi/setvars.sh --force

# Non-interactive completion (deterministic output for testing)
./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Benchmark prompt processing (PP) and token generation (TG)
./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Test backend operations (after modifying ggml operators)
./build/bin/test-backend-ops
```

### SYCL Device Selection (Critical!)

**WARNING**: On multi-GPU systems, you MUST explicitly select a single device. Without this, llama.cpp uses all visible GPUs and the unified kernel will **hang indefinitely**.

This system has 3 GPUs: Arc B580 (device 0), Arc Pro B50 (device 1), iGPU (device 2).

```bash
# List available devices
sycl-ls

# REQUIRED: Select Arc B580 (device 0) for stable operation
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench ...

# NOTE: GGML_SYCL_VISIBLE_DEVICES=0 does NOT work - it filters at llama.cpp level
# but unified cache still sees all Level Zero devices. Use ONEAPI_DEVICE_SELECTOR.
```

**Performance expectations (PP512, Mistral 7B Q4_0, Arc B580)**:
| Configuration | tok/s | Notes |
|---------------|-------|-------|
| Single device (unified kernel) | ~1180 | oneDNN FP16 path for M≥64 |
| Single device (legacy) | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` |
| Multi-device (default) | HANGS | Unified cache sync issues, avoid |

**Kernel path environment variables**:
- `GGML_SYCL_UNIFIED_FORCE_LEGACY=1`: Force legacy kernels (bypass unified kernel)
- `GGML_SYCL_ONEDNN_PP=0`: Disable oneDNN for prompt processing (use XMX/ESIMD)
- `GGML_SYCL_DEBUG=1`: Enable detailed kernel dispatch logging

## CI and Validation

### Before Submitting PRs
1. Format code: `clang-format-19 -i <files>`
2. Build: `ninja -C build`
3. Test: `ctest --test-dir build --output-on-failure`
4. For ggml changes: Run `test-backend-ops` on multiple backends

### Triggering Heavy CI
Add `ggml-ci` to commit message to trigger extended CI workloads.

## Documentation

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **Add New Model**: `docs/development/HOWTO-add-model.md`
- **Contributing**: `CONTRIBUTING.md`
