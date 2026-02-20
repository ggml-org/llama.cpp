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
# Preferred: format only staged changes (uses .clang-format)
git clang-format

# Format specific files
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
- **`llama-perplexity`**: Model evaluation (perplexity measurement)

### Backend Structure (`ggml/src/`)
- **`ggml-cpu/`**: CPU backend (AVX/NEON/RVV)
- **`ggml-cuda/`**: NVIDIA CUDA kernels
- **`ggml-metal/`**: Apple Metal shaders
- **`ggml-sycl/`**: Intel SYCL backend (see SYCL Backend Structure below)
- **`ggml-vulkan/`**: Vulkan compute shaders

### SYCL Backend Structure (`ggml/src/ggml-sycl/`)
Key files in the SYCL backend:
- **`ggml-sycl.cpp`** (~31K lines): Main backend — graph_compute, mul_mat dispatch, buffer ops, graph replay
- **`unified-kernel.cpp/hpp`** (~6.4K/~2.5K lines): Unified MUL_MAT kernel with XMX/ESIMD/MMVQ dispatch
- **`unified-cache.cpp/hpp`** (~5.3K/~960 lines): Tiered weight cache (device VRAM, pinned host, mmap)
- **`common.hpp/cpp`** (~3K/~2.2K lines): Shared types, `extra_gpu` struct, layout_policy, device management
- **`mmvq.cpp`** (~4.2K lines): Matrix-vector quantized kernels (batch=1 TG fast-path)
- **`mmq.cpp`** (~7K lines): Matrix-matrix quantized kernels (persistent TG, streaming PP)
- **`fattn.cpp`** (~1.5K lines): Flash attention implementation
- **`dispatch.hpp`** (~350 lines): Kernel dispatch policy and routing
- **`quants.hpp`** (~190 lines): SOA block offset calculations for quantized types

### Inference Flow
1. **Model loading** (`llama_model_load`): Reads GGUF file, maps weights to tensors
2. **Context creation** (`llama_init_from_model`): Allocates KV cache, scratch buffers
3. **Tokenization** (`llama_tokenize`): Text to token IDs
4. **Graph building** (`llama_build_graph`): Creates ggml computation graph per batch
5. **Graph execution** (`ggml_backend_graph_compute`): Dispatches to CPU/GPU backends
6. **Sampling** (`llama_sampler_sample`): Token selection from logits

### Weight Caching (GPU Backends)
GPU backends cache weights on-device for repeated inference:
- **CUDA**: `ggml_cuda_pool` with per-device allocation tracking
- **SYCL**: `unified_cache` with tiered memory (device VRAM, pinned host, mmap). Supports SOA (Structure-of-Arrays) layout for coalesced GPU memory access, oneDNN packed layouts, and LRU eviction.
- Weights are identified by tensor name hash + model ID for cache keys

## ggml Conventions

### Matrix Multiplication
Matrix multiplication is **unconventional**: `C = ggml_mul_mat(ctx, A, B)` computes:
```
C^T = A * B^T  <=>  C = B * A^T
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
- The `get` action can be omitted; `_context` class suffix is optional
- Use `init`/`free` for constructor/destructor actions

### Enum Values
Enum values are always UPPER_CASE and prefixed with the enum name:
```cpp
enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,
    LLAMA_VOCAB_TYPE_SPM  = 1,
    LLAMA_VOCAB_TYPE_BPE  = 2,
};
```

### Struct Declarations
Use `struct foo {}` not `typedef struct foo {} foo`. Omit optional `struct`/`enum` keywords in C++ code:
```cpp
llama_context * ctx;              // OK
struct llama_context * ctx;       // not OK
const llama_rope_type rope_type;  // OK (no enum keyword)
```

## Coding Guidelines

- **Minimal dependencies**: Avoid adding third-party dependencies
- **Cross-platform**: Test on Linux, macOS, Windows when possible
- **Simple STL**: Avoid fancy modern STL, use basic `for` loops, minimize templates
- **Vertical alignment**: Makes code more readable and easier to batch edit
- **Formatting**: 4 spaces, brackets on same line, `void * ptr`, `int & a`
- **Public API types**: Use `int32_t` etc., `size_t` for allocation sizes
- **File naming**: C/C++ lowercase with dashes (e.g., `unified-kernel.cpp`), Python lowercase with underscores

## Development Workflow (Machine-Specific)

### Model Locations
Models are stored in `/Storage/GenAI/models/`:

**Mistral 7B variants** (standard benchmark model):
- `mistral-7b-v0.1.Q4_0.gguf` (3.9G) - **Default for benchmarks**
- `mistral-7b-v0.1.Q4_K_M.gguf` (4.1G) - Good quality/size balance
- `mistral-7b-v0.1.Q8_0.gguf` (7.2G) - Highest quality
- Other variants: Q2_K (2.9G), Q3_K_S/M/L, Q4_K_S, Q5_0/K_S/K_M, Q6_K

**GPT-OSS models** (large MoE, native MXFP4):
- `gpt-oss-20b-mxfp4.gguf` (12G) - Smaller variant
- `gpt-oss-120b-mxfp4-*.gguf` (60G total, 3-part split) - Full model

### Verification Commands
```bash
source /opt/intel/oneapi/setvars.sh --force

# Non-interactive completion (deterministic output for testing)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Benchmark prompt processing (PP) and token generation (TG)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Test backend operations (after modifying ggml operators)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops
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

### Performance Expectations (Mistral 7B Q4_0, Arc B580)

| Metric | tok/s | Notes |
|--------|-------|-------|
| PP512 (Level 0, all VRAM) | ~1336 | oneDNN FP16 path for M>=64 |
| TG128 (Level 0, all VRAM) | ~72 | MMVQ fast-path with SOA layout + graph replay |
| TG128 (no graph) | ~70 | MMVQ fast-path alone (graph adds only +3%) |
| PP512 (Level 3, 30% budget) | ~269 | 15/33 GPU layers, rest on CPU |
| TG128 (Level 3, 30% budget) | ~14 | CPU offload via fit_params |
| PP512 (legacy) | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` |
| Multi-device | HANGS | Unified cache sync issues, avoid |

### SYCL Environment Variables

**Performance-critical (all default ON, opt-out)**:
| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable MMVQ fast-path bypass (slower TG) |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (minimal TG impact ~3%, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (bypass unified kernel) |

**Kernel dispatch tuning**:
| Variable | Effect |
|----------|--------|
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels for all batch sizes |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Min batch size for ESIMD dispatch |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Min batch for oneDNN PP path |

**Memory budget and pressure hierarchy**:
| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_VRAM_BUDGET_PCT=N` | 90 | VRAM budget as % of total (triggers CPU offload when model exceeds) |
| `GGML_SYCL_KV_HOST=1` | OFF | Force KV cache to host pinned memory (Level 1 offload) |
| `GGML_SYCL_KV_HOT_TOKENS=N` | auto | Hot window size in tokens for KV hot/cold tiering |
| `GGML_SYCL_KV_HOT_PCT=N` | auto | Hot window as % of total KV buffer |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Enable GPU weight streaming (Level 5, last resort) |
| `GGML_SYCL_HOST_COMPUTE=1` | OFF | Use host-pinned compute buffers (eliminates staging for CPU-dispatched layers) |

**Cache and memory**:
| Variable | Effect |
|----------|--------|
| `GGML_SYCL_UNIFIED_CACHE=0` | Disable unified weight cache entirely |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Cache mode (auto, device, host, mmap) |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction under memory pressure |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |

**Debugging**:
| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DEBUG=1` | Enable detailed kernel dispatch logging (MASSIVE output) |
| `GGML_SYCL_UNIFIED_DEBUG=1` | Debug unified kernel dispatch |
| `GGML_SYCL_NAN_CHECK=1` | Enable NaN detection in outputs |
| `GGML_SYCL_VALIDATE=1` | Enable A/B validation between kernel paths |
| `GGML_SYCL_GRAPH_RERECORD=1` | Use graph re-record instead of replay (very slow, diagnostic only) |

**Note**: There are 100+ additional debug/tuning env vars (GGML_SYCL_*). Search with `grep -r 'getenv("GGML_SYCL' ggml/src/ggml-sycl/` to find them all.

## CI and Validation

### Before Submitting PRs
1. Format code: `git clang-format` (preferred) or `clang-format-19 -i <files>`
2. Build: `ninja -C build`
3. Test: `ctest --test-dir build --output-on-failure`
4. For ggml changes: Run `test-backend-ops` on multiple backends
5. Verify performance: `llama-bench` and `llama-perplexity` should not regress

### Triggering Heavy CI
Add `ggml-ci` to commit message to trigger extended CI workloads.

## Documentation

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **Add New Model**: `docs/development/HOWTO-add-model.md`
- **Contributing**: `CONTRIBUTING.md` (coding/naming guidelines, PR process)
- **Copilot Instructions**: `.github/copilot-instructions.md` (cross-platform build/test patterns)
