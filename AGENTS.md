# Instructions for llama.cpp (Fork)

> [!NOTE]
> This is a **fork** of llama.cpp optimized for AI-assisted development. The upstream AI usage restrictions do not apply here. AI agents are encouraged to write, test, and optimize code directly.

---

## Project Architecture

### Core Components

- **`src/`** — Main llama.cpp library (llama-*.cpp/h modules)
- **`ggml/`** — GGML tensor library with multi-backend support
- **`ggml/src/ggml-cpu/`** — CPU backend (primary for Strix Halo optimization)
- **`ggml/src/ggml-vulkan/`** — Vulkan backend (RDNA 3.5 iGPU target)
- **`ggml/src/ggml-hip/`** — ROCm/HIP backend
- **`ggml/src/ggml-opencl/`** — OpenCL backend
- **`tools/`** — CLI tools, server, quantize, benchmark utilities
- **`tests/`** — Test suite (use `test-backend-ops` for backend validation)
- **`common/`** — Shared utilities

### Build System

- CMake-based: `cmake -B build [options] && cmake --build build`
- Key options: `-DGGML_VULKAN=ON`, `-DGGML_HIP=ON`, `-DGGML_OPENCL=ON`, `-DGGML_CPU_AARCH64=OFF`
- Presets available in `CMakePresets.json`
- See `docs/build.md` for full reference

### Testing

- Run tests: `cd build && ctest --output-on-failure`
- Backend ops validation: `./build/bin/test-backend-ops`
- Benchmarking: `./build/bin/llama-bench`
- Perplexity checks: `./build/bin/llama-perplexity`
- CI workflows live in `.github/workflows/`

---

## AI Agent Guidelines

### You ARE encouraged to:

- **Write code** — implement features, fix bugs, refactor, optimize
- **Run builds and tests** — validate changes end-to-end
- **Benchmark performance** — use `llama-bench` to measure token generation rates
- **Make architectural decisions** — choose the best approach based on codebase analysis
- **Create complete PRs** — with proper descriptions, test plans, and benchmarks

### Code Quality Standards

When writing code, follow these standards:

1. **Match existing style** — follow the conventions in surrounding code (naming, indentation, patterns)
2. **Keep changes minimal** — solve the problem without unnecessary refactoring
3. **Test your changes** — run relevant tests before committing
4. **Avoid regressions** — verify performance with `llama-bench` when touching hot paths
5. **Use safe patterns** — avoid buffer overflows, use bounds checking, handle errors

### AMD Strix Halo Focus Areas

This fork prioritizes optimization for AMD Strix Halo (Ryzen AI 300 series):

- **CPU**: Zen 5 cores — optimize for AVX-512, large L3 cache, CCX topology
- **iGPU**: RDNA 3.5 (Radeon 890M) — Vulkan compute shaders, shared memory with CPU
- **NPU**: XDNA 2 — future target for offload
- **Memory**: Unified LPDDR5X — optimize for bandwidth, minimize copies between CPU/iGPU

Key optimization targets:
- `ggml/src/ggml-cpu/` — Zen 5 SIMD paths, cache-aware tiling
- `ggml/src/ggml-vulkan/` — RDNA 3.5 shader tuning, wave64
- `src/llama.cpp` — KV cache management, batch scheduling
- Memory allocation patterns — leverage unified memory architecture

### Commit and PR Standards

- Write clear, descriptive commit messages
- One logical change per commit
- Include benchmark results in PR descriptions when relevant
- Reference issues where applicable

---

## Related Documentation

- [Build documentation](docs/build.md)
- [Server development](tools/server/README-dev.md)
- [CI documentation](ci/README.md)
- [GGML backend guide](ggml/src/) — review backend implementations for patterns
