# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

llama.cpp is a large-scale C/C++ project for efficient LLM (Large Language Model) inference with minimal setup and dependencies. The codebase is ~200k+ lines across 1000+ files, designed for state-of-the-art performance on diverse hardware (CPU, CUDA, Metal, Vulkan, SYCL, etc.).

**Key Facts:**
- Primary language: C/C++ (C++17 required)
- License: MIT
- Build system: CMake 3.14+ (Makefile is deprecated)
- Core dependency: ggml tensor library (vendored in `ggml/` directory)
- Model format: GGUF (GGML Universal Format)

## Build Commands

### Basic Build (CPU-only)
```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

Built binaries are placed in `build/bin/`. Build time is ~10 minutes on 4-core system with ccache, ~25 minutes without.

### Backend-Specific Builds
```bash
# CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Metal (macOS - enabled by default)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)

# Static builds
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release
```

### Debug Builds
```bash
# Single-config generators (Unix Makefiles)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Multi-config generators (Xcode, Visual Studio)
cmake -B build -G "Xcode"
cmake --build build --config Debug
```

## Testing

### Run All Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)
```

Test suite includes 38 tests covering tokenizers, grammar parsing, sampling, backends, and integration. Runtime is ~30 seconds for passing tests. Some tests may fail without network access (they download models).

### Server Unit Tests
```bash
# Build server first
cmake --build build --target llama-server

# Run tests
cd tools/server/tests
source ../../../.venv/bin/activate
./tests.sh
```

### Manual Testing
```bash
# Test version
./build/bin/llama-cli --version

# Test inference (requires model file)
./build/bin/llama-cli -m path/to/model.gguf -p "Hello" -n 10

# Benchmark performance
./build/bin/llama-bench -m model.gguf

# Evaluate perplexity
./build/bin/llama-perplexity -m model.gguf -f dataset.txt
```

## Code Quality

### Before Committing
```bash
# 1. Format C++ code
git clang-format

# 2. Activate Python environment
source .venv/bin/activate

# 3. Run pre-commit hooks
pre-commit run --all-files

# 4. Build
cmake --build build --config Release

# 5. Test
ctest --test-dir build --output-on-failure
```

### Formatting Rules (`.clang-format`)
- 4-space indentation
- 120 column limit
- Pointer alignment: `void * ptr` (middle)
- Reference alignment: `int & ref` (middle)

## Project Architecture

### Core Library Structure

The `src/` directory contains a highly modular implementation split into focused components:

**Model Architecture (`llama-arch.*`, `llama-model.*`)**
- `llama-arch.h`: Defines `llm_arch` enum with 100+ supported model architectures (LLaMA, GPT-2, Qwen, Gemma, Mamba, etc.)
- `llama-model.h/cpp`: Model loading, structure definitions, and weight management
- `llama-model-loader.h/cpp`: GGUF file loading and parsing
- `llama-model-saver.h/cpp`: Model serialization

**Inference Engine (`llama-context.*`, `llama-graph.*`)**
- `llama-context.h/cpp`: Inference context management, computation state
- `llama-graph.h/cpp`: Computation graph construction and execution (builds ggml graphs)
- `llama-batch.h/cpp`: Batch processing for efficient multi-sequence inference

**Memory Management**
- `llama-kv-cache.h/cpp`: Key-value cache for transformer attention
- `llama-kv-cells.h`: KV cache cell management
- `llama-memory.h/cpp`: Base memory abstraction
- `llama-memory-recurrent.h/cpp`: Memory for recurrent models (Mamba, RWKV)
- `llama-memory-hybrid.h/cpp`: Hybrid memory architectures
- `llama-mmap.h/cpp`: Memory-mapped file I/O for efficient model loading

**Text Processing**
- `llama-vocab.h/cpp`: Tokenizer implementations (BPE, SPM, WPM, etc.)
- `llama-chat.h/cpp`: Chat template processing
- `llama-grammar.h/cpp`: GBNF grammar parsing for constrained generation
- `llama-sampling.h/cpp`: Sampling strategies (temperature, top-p, top-k, etc.)
- `unicode.h/cpp`, `unicode-data.h`: Unicode handling

**Quantization & Optimization**
- `llama-quant.h/cpp`: Quantization implementations (2-bit, 3-bit, 4-bit, 5-bit, 6-bit, 8-bit)
- `llama-adapter.h/cpp`: LoRA adapter support

**Public API**
- `include/llama.h`: Main C API (~2000 lines) - this is the stable public interface
- `include/llama-cpp.h`: C++ convenience wrappers

### Key Executables

Built in `build/bin/`:

**Primary Tools:**
- `llama-cli` - Main inference CLI (formerly `main`)
- `llama-server` - OpenAI-compatible HTTP server with streaming support
- `llama-quantize` - Model quantization utility
- `llama-perplexity` - Model quality evaluation
- `llama-bench` - Performance benchmarking
- `llama-run` - Comprehensive inference runner (used by RamaLama)
- `llama-simple` - Minimal example for developers

**Specialized Tools:**
- `llama-imatrix` - Importance matrix generation for quantization
- `llama-export-lora` - LoRA adapter export
- `gguf-split` - Split large GGUF files
- `llama-cvector-generator` - Control vector generation
- `llama-tokenize` - Tokenization testing

### Directory Layout

```
├── src/                    # Core llama library (modular .cpp/.h pairs)
├── include/                # Public API headers (llama.h)
├── ggml/                   # Core tensor library (submodule)
├── common/                 # Shared utilities for examples/tools
├── tools/                  # Primary executables (server, quantize, etc.)
├── examples/               # Example applications
├── tests/                  # Test suite (CTest integration)
├── docs/                   # Documentation (build guides, API docs)
├── scripts/                # CI, data processing, automation
├── grammars/               # GBNF grammar examples
├── gguf-py/               # Python GGUF utilities and model conversion
├── cmake/                  # CMake modules and configuration
└── .github/workflows/      # CI/CD workflows
```

### GGML Integration

llama.cpp depends on the `ggml` tensor library (in `ggml/` directory). Key concepts:

- **Computation graphs**: Built declaratively, executed in one shot
- **Tensor layout**: Row-major order; dim 0 = columns, dim 1 = rows, dim 2 = matrices
- **Matrix multiplication**: `C = ggml_mul_mat(ctx, A, B)` computes C^T = AB^T ⟺ C = BA^T
- **Backends**: CPU (AVX/NEON), CUDA, Metal, Vulkan, SYCL implementations in `ggml/src/`

When modifying ggml operators, run `test-backend-ops` to verify consistency across backends.

### Model Support Pattern

Adding a new model architecture involves:
1. Add entry to `llm_arch` enum in `llama-arch.h`
2. Define tensor names and hyperparameters in `llama-arch.cpp`
3. Implement graph construction in `llama-graph.cpp`
4. Add tokenizer support if needed in `llama-vocab.cpp`
5. Create Python conversion script in `gguf-py/` (e.g., `convert_hf_to_gguf.py`)

See `docs/development/HOWTO-add-model.md` for detailed instructions.

## Common Development Workflows

### Adding a New Model Architecture
1. Study existing similar architecture in `src/llama-arch.cpp`
2. Define architecture constants and tensor mappings
3. Implement graph builder in `src/llama-graph.cpp`
4. Add conversion script in `gguf-py/`
5. Test with `llama-cli` and `llama-perplexity`

### Modifying Inference Logic
- Core inference: `src/llama-context.cpp` and `src/llama-graph.cpp`
- Sampling: `src/llama-sampling.cpp`
- Always benchmark with `llama-bench` to verify no performance regression

### Working with Quantization
- Implementations in `src/llama-quant.cpp` and `ggml/src/ggml-quants.c`
- Test with `llama-quantize` tool
- Validate quality with `llama-perplexity`

### Server Development
- Main server: `tools/server/server.cpp`
- Run tests: `cd tools/server/tests && source ../../../.venv/bin/activate && ./tests.sh`
- API compatibility: Must maintain OpenAI API compatibility

## Python Environment

Always activate `.venv` before running Python scripts:
```bash
source .venv/bin/activate
```

**Python Tools:**
- `convert_hf_to_gguf.py` - Main HuggingFace model converter
- `gguf-py/` - GGUF format manipulation library
- `requirements.txt` - Python dependencies

Configuration files: `.flake8`, `pyrightconfig.json`

## CI/CD

### Local CI Validation
```bash
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

Runtime: 30-60 minutes. Add `ggml-ci` to commit message to trigger heavy CI workloads.

### GitHub Actions
- `.github/workflows/build.yml` - Multi-platform builds
- `.github/workflows/server.yml` - Server tests
- `.github/workflows/python-lint.yml` - Python linting
- `.github/workflows/python-type-check.yml` - Type checking

## Coding Conventions

### C++ Style
- Use basic `for` loops over fancy STL constructs
- Avoid templates when simple code suffices
- Use sized integer types (`int32_t`) in public API
- Vertical alignment improves readability
- `snake_case` for functions, variables, types
- Enum values: UPPER_CASE with enum prefix

### Naming Pattern
`<class>_<action>_<noun>` pattern:
```cpp
llama_model_init();           // class: llama_model, method: init
llama_sampler_get_seed();     // class: llama_sampler, method: get_seed
llama_adapter_lora_free();    // class: llama_adapter_lora, method: free
```

### Pull Requests
- Squash-merge format: `<module> : <description> (#<PR_number>)`
- Modules list: https://github.com/ggml-org/llama.cpp/wiki/Modules
- Test locally before pushing (run full CI if possible)
- Separate PRs for each feature/fix

## Important Guidelines

### Performance First
This is a performance-critical inference library. Always:
- Benchmark changes with `llama-bench`
- Verify quality with `llama-perplexity`
- Test backend operations with `test-backend-ops` if modifying ggml

### Cross-Platform Compatibility
- Test on Linux, macOS, Windows when possible
- Avoid platform-specific code unless necessary
- Note: While all backends can build, only CPU backend runs everywhere

### Minimal Dependencies
- Avoid adding third-party dependencies
- Use header-only libraries when unavoidable
- Current bundled deps: httplib, json, stb-image, minja

### API Stability
- Changes to `include/llama.h` require careful consideration
- Maintain backward compatibility when possible
- Document breaking changes in changelogs:
  - https://github.com/ggml-org/llama.cpp/issues/9289 (libllama API)
  - https://github.com/ggml-org/llama.cpp/issues/9291 (server API)

## Model Format (GGUF)

Models are stored in GGUF format. Key tools:
- Convert from HuggingFace: `python convert_hf_to_gguf.py <model_dir>`
- Quantize: `llama-quantize model-f16.gguf model-q4_0.gguf q4_0`
- Inspect: Use `gguf-py` utilities or HuggingFace GGUF editor
- Split large files: `gguf-split --split model.gguf model-split`

## Resources

- Build documentation: `docs/build.md`
- Adding models: `docs/development/HOWTO-add-model.md`
- Backend guides: `docs/backend/` (CUDA, SYCL, etc.)
- GBNF grammars: `grammars/README.md`
- Contributing: `CONTRIBUTING.md`
- GitHub Projects: https://github.com/ggml-org/llama.cpp/projects
