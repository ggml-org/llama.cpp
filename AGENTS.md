# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### SYCL Build (Recommended - Guaranteed Single-Pass)

Use the build script for reliable, single-pass compilation:

```bash
# Full build (handles oneAPI sourcing, Ninja, ccache automatically)
./scripts/sycl-build.sh

# Build specific target
./scripts/sycl-build.sh llama-completion

# Force reconfigure (after CMakeLists.txt changes)
./scripts/sycl-build.sh -r

# Clean build (from scratch)
./scripts/sycl-build.sh -c

# Quick incremental rebuild after editing a file
./scripts/quick-rebuild.sh mmq.cpp llama-completion
```

**Why use the script instead of raw cmake:**
- Sources oneAPI automatically
- Uses Ninja (better dependency tracking than Make)
- Uses ccache for faster rebuilds
- Auto-detects when CMake reconfigure is needed
- Handles generator switching (Make → Ninja) cleanly

### Standard Build (CPU-only)
```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### Backend-Specific Builds
```bash
# CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Metal (macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)

# SYCL (Intel) - Manual method (prefer ./scripts/sycl-build.sh instead)
source /opt/intel/oneapi/setvars.sh --force
cmake -G Ninja -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --config Release -j $(nproc)
```

### Running Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

### Code Formatting
```bash
# Format staged C++ files before committing (Ubuntu uses versioned binary)
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

## Development Workflow

### Subagent Model Selection
When dispatching subagents via the Task tool:
- **Implementation tasks**: Use `model: "opus"` for code writing, debugging, and complex implementation
- **Review tasks**: Use `model: "opus"` for spec compliance and code quality reviews
- **Exploration/search tasks**: Default model (sonnet) is fine for quick lookups

Example:
```
Task tool with model: "opus" for implementing features
```

### Test-Driven Development
1. Write unit tests to reproduce bugs or validate features
2. Tests should exercise actual production code paths
3. Verify with reference models before claiming fixes work

### Verification Commands
```bash
# Non-interactive completion (deterministic output for testing)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /path/to/model.gguf -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"

# Benchmark
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -fa 0,1
```

### Python Environment
```bash
source .venv/bin/activate  # Use project venv for Python tools
```

## Professional Engineering Standards

**Spinach Rule**: When you detect a visible flaw the user may not see (wrong assumption, hidden risk, flawed logic), correction is mandatory. Do not optimize for agreement.

- Challenge assumptions directly: "There's spinach here: this approach has X risk because..."
- Question unclear requirements before implementing
- Identify performance trade-offs and security implications
- Never fake progress or simulate certainty

## Documentation Index

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **Development**: `docs/development/`

## Local Environment (Intel SYCL)

### GPU Selection
- **Single GPU**: `ONEAPI_DEVICE_SELECTOR=level_zero:1`
- **Dual GPU**: `ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"`

### Model Paths
Models in `/Storage/GenAI/models/`:
- **Mistral 7B Q4**: `mistral-7b-v0.1.Q4_0.gguf` (fast testing)
- **Mistral 7B Q6_K**: `mistral-7b-v0.1.Q6_K.gguf` (pure Q6_K testing)
- **GPT-OSS 20B Q8**: `gpt-oss-20b-Q8_0.gguf` (MoE model)

### SYCL Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_SYCL_DISABLE_GRAPH` | 0 | Disable SYCL command graphs |
| `GGML_SYCL_DEBUG` | 0 | Debug output level (0-2) |
| `GGML_SYCL_LAYOUT_OVERRIDE` | (unset) | Force weight layout for debugging: `aos`, `soa`, `coalesced`, `xmx_tiled` |

### SYCL Layout Overrides (Debug)
Use `GGML_SYCL_LAYOUT_OVERRIDE` to force a specific layout. `aos` disables reordering.

```bash
# Default: auto layout selection (no override)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -ngl 99

# Force AoS (no reorder):
GGML_SYCL_LAYOUT_OVERRIDE=aos ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench ...
```
