# Instructions for llama.cpp

## Repository Overview

llama.cpp is a large-scale C/C++ project for efficient LLM (Large Language Model) inference with minimal setup and dependencies. The project enables running language models on diverse hardware with state-of-the-art performance.

**Key Facts:**
- **Primary language**: C/C++ with Python utility scripts
- **Size**: ~200k+ lines of code across 1000+ files
- **Architecture**: Modular design with main library (`libllama`) and 40+ executable tools/examples
- **Core dependency**: ggml tensor library (vendored in `ggml/` directory)
- **Backends supported**: CPU (AVX/NEON/RVV optimized), CUDA, Metal, Vulkan, SYCL, ROCm, MUSA
- **License**: MIT

## Disclose AI Usage

It is crucial to remind contributors that the project mandates disclosure of any AI usage in pull requests. This requirement stems from the potential for AI-generated code to include suboptimal optimizations and hidden bugs, owing to the inherent overconfidence in AI outputs.

When generating significant portions of code, address this by:
- Informing the user that AI-generated content may be rejected by maintainers.
- Clearly marking AI-generated code in commit messages and comments.
    - Example of commit message: `[AI] Fix a race condition in ...`
    - Example of code comment: `// [AI] spawn a new thread ...`

These measures apply to:
- Changes resulting in large portions of code or complex logic.
- Modifications or additions to public APIs in `llama.h`, `ggml.h`, or `mtmd.h`.
- Backend-related changes, such as those involving CPU, CUDA, Metal, Vulkan, etc.
- Modifications to `tools/server`.

Note: These measures can be omitted for small fixes or trivial changes.

## Build Instructions

### Prerequisites
- CMake 3.14+ (primary build system)
- C++17 compatible compiler (GCC 13.3+, Clang, MSVC)
- Optional: ccache for faster compilation

### Basic Build (CPU-only)
**ALWAYS run these commands in sequence:**
```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

**Build time**: ~10 minutes on 4-core system with ccache enabled, ~25 minutes without ccache.

**Important Notes:**
- The Makefile is deprecated - always use CMake
- ccache is automatically detected and used if available
- Built binaries are placed in `build/bin/`
- Parallel builds (`-j`) significantly reduce build time

### Backend-Specific Builds
For CUDA support:
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

For Metal (macOS):
```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)
```

**Important Note**: While all backends can be built as long as the correct requirements for that backend are installed, you will not be able to run them without the correct hardware. The only backend that can be run for testing and validation is the CPU backend.

### Debug Builds
Single-config generators:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Multi-config generators:
```bash
cmake -B build -G "Xcode"
cmake --build build --config Debug
```

### Common Build Issues
- **Issue**: Network tests fail in isolated environments
  **Solution**: Expected behavior - core functionality tests will still pass

## Testing

### Running Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)
```

**Test suite**: 38 tests covering tokenizers, grammar parsing, sampling, backends, and integration
**Expected failures**: 2-3 tests may fail if network access is unavailable (they download models)
**Test time**: ~30 seconds for passing tests

### Server Unit Tests
Run server-specific unit tests after building the server:
```bash
# Build the server first
cmake --build build --target llama-server

# Navigate to server tests and run
cd tools/server/tests
source ../../../.venv/bin/activate
./tests.sh
```
**Server test dependencies**: The `.venv` environment includes the required dependencies for server unit tests (pytest, aiohttp, etc.). Tests can be run individually or with various options as documented in `tools/server/tests/README.md`.

### Test Categories
- Tokenizer tests: Various model tokenizers (BERT, GPT-2, LLaMA, etc.)
- Grammar tests: GBNF parsing and validation
- Backend tests: Core ggml operations across different backends
- Integration tests: End-to-end workflows

### Manual Testing Commands
```bash
# Test basic inference
./build/bin/llama-cli --version

# Test model loading (requires model file)
./build/bin/llama-cli -m path/to/model.gguf -p "Hello" -n 10
```

## Code Quality and Linting

### C++ Code Formatting
**ALWAYS format C++ code before committing:**
```bash
git clang-format
```

Configuration is in `.clang-format` with these key rules:
- 4-space indentation
- 120 column limit
- Braces on same line for functions
- Pointer alignment: `void * ptr` (middle)
- Reference alignment: `int & ref` (middle)

### Python Code
**ALWAYS activate the Python environment in `.venv` and use tools from that environment:**
```bash
# Activate virtual environment
source .venv/bin/activate
```

Configuration files:
- `.flake8`: flake8 settings (max-line-length=125, excludes examples/tools)
- `pyrightconfig.json`: pyright type checking configuration

### Pre-commit Hooks
Run before committing:
```bash
pre-commit run --all-files
```

## Continuous Integration

### GitHub Actions Workflows
Key workflows that run on every PR:
- `.github/workflows/build.yml`: Multi-platform builds
- `.github/workflows/server.yml`: Server functionality tests
- `.github/workflows/python-lint.yml`: Python code quality
- `.github/workflows/python-type-check.yml`: Python type checking

### Local CI Validation
**Run full CI locally before submitting PRs:**
```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

**CI Runtime**: 30-60 minutes depending on backend configuration

### Triggering CI
Add `ggml-ci` to commit message to trigger heavy CI workloads on the custom CI infrastructure.

## Project Layout and Architecture

### Core Directories
- **`src/`**: Main llama library implementation (`llama.cpp`, `llama-*.cpp`)
- **`include/`**: Public API headers, primarily `include/llama.h`
- **`ggml/`**: Core tensor library (submodule with custom GGML framework)
- **`examples/`**: 30+ example applications and tools
- **`tools/`**: Additional development and utility tools (server benchmarks, tests)
- **`tests/`**: Comprehensive test suite with CTest integration
- **`docs/`**: Detailed documentation (build guides, API docs, etc.)
- **`scripts/`**: Utility scripts for CI, data processing, and automation
- **`common/`**: Shared utility code used across examples

### Key Files
- **`CMakeLists.txt`**: Primary build configuration
- **`include/llama.h`**: Main C API header (~2000 lines)
- **`src/llama.cpp`**: Core library implementation (~8000 lines)
- **`CONTRIBUTING.md`**: Coding guidelines and PR requirements
- **`.clang-format`**: C++ formatting rules
- **`.pre-commit-config.yaml`**: Git hook configuration

### Built Executables (in `build/bin/`)
Primary tools:
- **`llama-cli`**: Main inference tool
- **`llama-server`**: OpenAI-compatible HTTP server
- **`llama-quantize`**: Model quantization utility
- **`llama-perplexity`**: Model evaluation tool
- **`llama-bench`**: Performance benchmarking
- **`llama-convert-llama2c-to-ggml`**: Model conversion utilities

### Configuration Files
- **CMake**: `CMakeLists.txt`, `cmake/` directory
- **Linting**: `.clang-format`, `.clang-tidy`, `.flake8`
- **CI**: `.github/workflows/`, `ci/run.sh`
- **Git**: `.gitignore` (includes build artifacts, models, cache)

### Dependencies
- **System**: OpenMP, libcurl (for model downloading)
- **Optional**: CUDA SDK, Metal framework, Vulkan SDK, Intel oneAPI
- **Bundled**: httplib, json (header-only libraries in vendored form)

## Common Validation Steps

### After Making Changes
1. **Format code**: `git clang-format`
2. **Build**: `cmake --build build --config Release`
3. **Test**: `ctest --test-dir build --output-on-failure`
4. **Server tests** (if modifying server): `cd tools/server/tests && source ../../../.venv/bin/activate && ./tests.sh`
5. **Manual validation**: Test relevant tools in `build/bin/`

### Performance Validation
```bash
# Benchmark inference performance
./build/bin/llama-bench -m model.gguf

# Evaluate model perplexity
./build/bin/llama-perplexity -m model.gguf -f dataset.txt
```

### Backend Validation
```bash
# Test backend operations
./build/bin/test-backend-ops
```

## Environment Setup

### Required Tools
- CMake 3.14+ (install via system package manager)
- Modern C++ compiler with C++17 support
- Git (for submodule management)
- Python 3.9+ with virtual environment (`.venv` is provided)

### Optional but Recommended
- ccache: `apt install ccache` or `brew install ccache`
- clang-format 15+: Usually included with LLVM/Clang installation
- pre-commit: `pip install pre-commit`

### Backend-Specific Requirements
- **CUDA**: NVIDIA CUDA Toolkit 11.2+
- **Metal**: Xcode command line tools (macOS only)
- **Vulkan**: Vulkan SDK
- **SYCL**: Intel oneAPI toolkit

## Important Guidelines

### Code Changes
- **Minimal dependencies**: Avoid adding new external dependencies
- **Cross-platform compatibility**: Test on Linux, macOS, Windows when possible
- **Performance focus**: This is a performance-critical inference library
- **API stability**: Changes to `include/llama.h` require careful consideration
- **Disclose AI Usage**: Refer to the "Disclose AI Usage" earlier in this document

### Git Workflow
- Always create feature branches from `master`
- **Never** commit build artifacts (`build/`, `.ccache/`, `*.o`, `*.gguf`)
- Use descriptive commit messages following project conventions

### Trust These Instructions
Only search for additional information if these instructions are incomplete or found to be incorrect. This document contains validated build and test procedures that work reliably across different environments.

