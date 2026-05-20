# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Policy

**AGENTS.md** and **CONTRIBUTING.md** must be reviewed before any work. This project does NOT accept fully AI-generated PRs. AI tools may only assist with mechanical tasks, corrections, or expanding on established designs. Never write PR descriptions, commit messages, or reviewer responses. Never commit or push without explicit human approval.

## Build System (CMake Only)

The Makefile build was removed. All builds use CMake:

```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release -j

# With tests (default ON for standalone)
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j

# With CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# Static build
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j
```

**CMakePresets.json** has platform-specific presets (e.g., `x64-windows-msvc-release`, `x64-linux-gcc-release`):
```bash
cmake --preset x64-linux-gcc-release
cmake --build --preset x64-linux-gcc-release
```

**Key CMake options:**
| Option | Default | Description |
|--------|---------|-------------|
| `LLAMA_BUILD_TESTS` | ON | Build test executables |
| `LLAMA_BUILD_TOOLS` | ON | Build cli, server, quantize, etc. |
| `LLAMA_BUILD_EXAMPLES` | ON | Build example programs |
| `LLAMA_BUILD_SERVER` | ON | Build llama-server specifically |
| `LLAMA_FATAL_WARNINGS` | OFF | -Werror |
| `GGML_CPU` | ON | CPU backend (always needed) |
| `GGML_NATIVE` | ON | Auto-detect CPU features |
| `GGML_CUDA` | OFF | NVIDIA CUDA backend |
| `GGML_METAL` | ON (macOS) | Apple Metal backend |
| `GGML_HIP` | OFF | AMD HIP backend |
| `GGML_VULKAN` | OFF | Vulkan backend |

## Tests

Custom `testing.h` framework (NOT Catch2) in `tests/testing.h`. Assertions: `assert_true()`, `assert_equal()`. Tests support regex filter via `argv[1]`.

```bash
# Run all tests via CTest
cd build && ctest -C Release

# Run a single test by name (regex)
cd build && ctest -C Release --output-on-failure -R test-tokenizer-0

# Run tests by label
cd build && ctest -C Release -L model --output-on-failure

# Run a test binary directly with filter
./build/bin/test-chat-auto-parser "some_regex"
```

Key test files in `tests/`: `test-backend-ops.cpp` (~400KB, comprehensive backend tests), `test-chat.cpp`, `test-tokenizer-*.cpp`, `test-grammar-*.cpp`, `test-quantize-*.cpp`, `test-sampling.cpp`.

## Formatting and Linting

```bash
# C/C++ formatting (project uses .clang-format with column limit 120)
clang-format -i -style=file path/to/file.cpp

# C/C++ static analysis (.clang-tidy)
clang-tidy --config-file=.clang-tidy path/to/file.cpp -- -std=c++17 -I include -I src -I common

# Pre-commit hooks
pre-commit run --all-files
```

Coding conventions from CONTRIBUTING.md: `snake_case`, 4-space indent, brackets on same line, `void * ptr`, sized int types (`int32_t`) in public API, no templates, basic `for` loops, no third-party deps.

## Architecture Overview

### Layered Structure

```
ggml/          Tensor operations library (ggml v0.12.0)
  ggml.h       Core tensor math (types, ops, graphs)
  ggml-alloc.h Memory allocators (arena, linear, graph)
  ggml-backend.h  Backend abstraction (buffer, device, scheduler)
  ggml/src/ggml-cpu/   CPU backend (AVX, AVX2, AVX512 variants)
  ggml/src/ggml-cuda/  CUDA backend
  ggml/src/ggml-metal/ Metal backend
  ggml/src/ggml-vulkan/ Vulkan backend
  ... (HIP, SYCL, OpenCL, OpenVINO, RPC, WebGPU, CANN, ZenDNN)

common/        Shared utilities ("llama-common" library)
  common.cpp/h   Main utilities, log, arg parsing
  chat.cpp/h     Chat templating
  chat-peg-parser.cpp  PEG-based output parser
  sampling.cpp/h  Sampling strategies
  console.cpp/h  Terminal/console handling
  json-schema-to-grammar.cpp  JSON schema conversion

src/           Core "llama" library
  llama-arch.cpp     Model architecture detection (per-model)
  llama-context.cpp  Inference context (main eval entry point)
  llama-graph.cpp    Compute graph builder (build_input_forward)
  llama-model.cpp    Model loading, GGUF parsing
  llama-sampling.cpp  High-level sampling API
  llama-kv-cache.cpp  KV cache management (slot allocation, state save/restore)
  llama-memory.cpp   Memory strategies (standard, hybrid, recurrent)
  llama-batch.cpp    Batch processing for ubatches

include/       Public API headers
  llama.h      C API (~82KB, comprehensive)
  llama-cpp.h  C++ API wrapper

tools/         End-user binaries
  server/      OpenAI-compatible HTTP server (largest tool, ~15k LOC)
  cli/         Command-line inference
  quantize/    Model quantization
  perplexity/  Perplexity benchmark
  llama-bench/ Performance benchmark
  gguf-split/  Split/merge GGUF files
  completion/  Autocomplete tool
```

### Key Architectural Concepts

**Compute Graph**: `llama_graph.cpp` builds a `ggml_cgraph` representing the full forward pass. Each layer's tensors (norm, QKV projection, RoPE, attention, FFN, residual) are added as ggml operations. The graph is then allocated and evaluated by the ggml backend scheduler.

**KV Cache**: Managed by `llama_kv_cache` (or hybrid/recurrent variants). The cache is a ring buffer of cells, each holding K/V embeddings for a token position. Cells can belong to multiple sequences. SWA (Sliding Window Attention) is supported. State save/restore writes cells as ranges (not full buffers).

**Memory Strategies**:
- Standard (`llama_kv_cache`): Ring buffer with slot allocation
- Hybrid (`llama_memory_hybrid`): Combines standard + ISWA (infinite sliding window attention)
- Recurrent (`llama_memory_recurrent`): For recurrent models

**Backend Abstraction**: `ggml_backend_t` abstracts CPU, GPU, and other accelerators. Each backend implements `ggml_backend_t` (buffer allocation, tensor copy, compute graph evaluation). The scheduler (`ggml_backend_sched_t`) handles multi-device offload and data transfers.

**Model Loading**: GGUF format, parsed by `llama_model_loader`. Architecture-specific init via `llama_arch` dispatch table. Supports quantization types Q2_K through Q8_0, plus IQ series.

### Important Design Patterns

- Tensors are row-major: dimension 0 = columns, 1 = rows, 2 = matrices
- `ggml_mul_mat(A, B)` computes `C = B @ A` (transposed convention)
- Quantized types use chunked storage with shared quantization parameters
- Model layer offload is per-layer, not per-tensor (a layer's all weights go to same device)

## Useful Resources (load as needed)

- [CONTRIBUTING.md](CONTRIBUTING.md) — coding guidelines, PR process, AI policy
- [AGENTS.md](AGENTS.md) — AI agent guidelines, useful resource links
- [docs/build.md](docs/build.md) — detailed build instructions per backend
- [tools/server/README-dev.md](tools/server/README-dev.md) — server development scope
- [docs/development/parsing.md](docs/development/parsing.md) — PEG parser documentation
- [docs/autoparser.md](docs/autoparser.md) — auto parser documentation
- [common/jinja/README.md](common/jinja/README.md) — Jinja engine
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new model architectures
- [ci/README.md](ci/README.md) — running full CI locally
