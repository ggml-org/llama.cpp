# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Work

**Active Epic:** XMX-Optimized Unified Kernel Architecture (`llama.cpp-a3t`)
- **Design Doc:** `docs/plans/xmx-mmvq-optimization.md`
- **Next Task:** `llama.cpp-wzx` (Build benchmark harness) - READY TO START
- **Goal:** Optimize MMVQ kernels for all batch sizes using XMX appropriately

Run `bd show llama.cpp-a3t` for full epic details and `bd ready` for available tasks.

## Build Commands

**CRITICAL**: Always source Intel oneAPI before building or running SYCL code:
```bash
source /opt/intel/oneapi/setvars.sh --force
```

### Recommended Build (Ninja + SYCL)

**Use Ninja for reliable dependency tracking and faster incremental builds.**

Ninja advantages over Make:
- **Correct header dependency tracking**: Changes to `.hpp` files reliably trigger recompilation
- **Faster no-op builds**: 1.5s vs 73s for Make on large projects (stored in binary `.ninja_deps`)
- **Better parallelism**: Optimal resource utilization without configuration
- **Command-line change detection**: Rebuilds when compile flags change

```bash
# One-time setup: Configure with Ninja generator
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -G Ninja \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx

# Build (use this for all subsequent builds)
ninja -C build -j $(nproc)

# Build specific target only
ninja -C build llama-bench

# Check what would be rebuilt (dry-run)
ninja -C build -n

# Debug: explain why files are being rebuilt
ninja -C build -d explain
```

### Alternative: Standard CMake Build

If Ninja is unavailable, fall back to Make (less reliable for header changes):
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j $(nproc)
```

### Backend-Specific Builds (non-SYCL)
```bash
# CUDA
cmake -B build -G Ninja -DGGML_CUDA=ON
ninja -C build -j $(nproc)

# Metal (macOS)
cmake -B build -G Ninja -DGGML_METAL=ON
ninja -C build -j $(nproc)

# CPU-only
cmake -B build -G Ninja
ninja -C build -j $(nproc)
```

### Switching Between Build Systems

**Warning**: Cannot switch generators in existing build directory. Remove and recreate:
```bash
rm -rf build && cmake -B build -G Ninja [options]
```

### Fast Incremental Rebuild (SYCL)
SYCL compilation is slow (~58K lines). Use the quick-rebuild script:
```bash
./scripts/quick-rebuild.sh mmq.cpp      # After editing mmq.cpp
./scripts/quick-rebuild.sh mmvq.cpp     # After editing mmvq.cpp
./scripts/quick-rebuild.sh              # Full rebuild
```
### Using subagents
#Use Opus for subagents responsible for implementing task

### Running Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)
```

### Code Formatting
```bash
# Format C++ files before committing (Ubuntu uses versioned binary)
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
- **`ggml-sycl/`**: Intel SYCL backend (84 .hpp, 46 .cpp files)
- **`ggml-vulkan/`**: Vulkan compute shaders

### SYCL Backend Architecture (`ggml/src/ggml-sycl/`)
Key components for XMX optimization work:
- **`common.hpp`** (~118KB): Core definitions, `reorder_mode` enum, tensor layout system
- **`dmmv.cpp`** (~155KB): Dense matrix-vector multiply (batch=1)
- **`mmvq.cpp`**: Matrix-matrix vector quantized (small batch)
- **`mmq.cpp`**: Matrix-matrix quantized (large batch, XMX path)
- **`gemm.hpp`**: GEMM infrastructure including oneDNN integration
- **`dnnl-ops.hpp`**: oneDNN wrappers for softmax, eltwise, binary ops

#### Memory Layout Modes (`reorder_mode` enum)
| Mode | Value | Description |
|------|-------|-------------|
| `NONE` | 0 | Original AoS layout (Array of Structures) |
| `SOA` | 1 | SoA layout: all qs bytes contiguous, then all d values |
| `COALESCED` | 2 | Tile-based layout for MMVQ (word-major interleaved) |
| `XMX_COALESCED` | 3 | XMX-optimized layout for MoE GEMM (K_TILE=32 aligned) |
| `XMX_GEMM_TILED` | 4 | XMX GEMM tiled layout for quantized weights |

#### Kernel Dispatch Logic
- **Batch=1**: DMMV kernel (memory-bandwidth bound)
- **Batch=1-32**: MMVQ kernel (transitional)
- **Batch>32**: MMQ/XMX kernel (compute-bound)

See `docs/plans/xmx-mmvq-optimization.md` for the unified kernel architecture design.

## Development Workflow

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

- **Build Details**: `docs/BUILD.md`
- **Architecture**: `docs/ARCH.md`
- **Environment/Hardware**: `docs/ENV.md`
- **Debugging/Profiling**: `docs/DEBUG.md`
- **SYCL Kernel Benchmarks**: `docs/SYCL_KERNEL_BENCHMARKS.md`
- **XMX Kernel Optimization Plan**: `docs/plans/xmx-mmvq-optimization.md` (Active Epic)

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
| `GGML_SYCL_DISABLE_OPT` | 0 | Disable SoA weight reordering (29% slower for TG) |
| `GGML_SYCL_DEBUG` | 0 | Debug output level (0-2) |
| `GGML_SYCL_FORCE_MMQ` | unset | Force MMQ kernel regardless of batch size |
| `GGML_SYCL_FORCE_DMMV` | unset | Force DMMV kernel for batch=1 |

See `docs/SYCL_KERNEL_BENCHMARKS.md` for detailed kernel/layout performance data.

## XMX Optimization Concepts

### Intel XMX (Xe Matrix eXtensions)
Matrix acceleration hardware in Intel Arc GPUs. Key characteristics:
- **dpas instruction**: Dot Product Accumulate Systolic
- **Tile size**: 8×16×32 (M×N×K) for FP16/INT8
- **Repeat count**: 1-8 (controls M dimension)
- **Throughput ceiling**: ~300 TOPS for INT8 on Arc B580

### ESIMD (Explicit SIMD)
Low-level SYCL extension for direct hardware control:
```cpp
// ESIMD dpas example (8x16x32 tile)
auto result = dpas<8, 1>(acc, a_tile, b_tile);  // repeat=8
```
Provides explicit register and SLM management, bypassing SYCL abstraction overhead.

### Quantization Types (Optimization Priority)
| Type | Block Size | Bits | Priority | Notes |
|------|------------|------|----------|-------|
| Q4_0 | 32 | 4.5 | P0 | Most common, optimize heavily |
| Q8_0 | 32 | 8.5 | P0 | INT8 XMX native support |
| Q6_K | 256 | 6.6 | P1 | K-quant, different patterns |
| Q4_K | 256 | 4.5 | P1 | K-quant variant |
| MXFP4 | 32 | 4 | P2 | Future hardware support |

### Performance Regimes
| Batch | Regime | Bottleneck | Best Approach |
|-------|--------|------------|---------------|
| 1-4 | Memory-bound | Bandwidth | Wide loads, prefetch, SLM cache |
| 8-64 | Transitional | Mixed | Small XMX tiles (8x8, 16x16) |
| 64+ | Compute-bound | ALU | Large XMX tiles (64x64), persistent threads |
