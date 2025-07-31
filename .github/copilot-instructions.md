# Copilot Instructions for llama.cpp

This document provides instructions for AI assistants (GitHub Copilot, Claude, etc.) working on the llama.cpp project with NUMA improvements and development container setup.

## 🎯 Project Overview

This is a fork of llama.cpp with **NUMA-aware improvements** for better CPU threading and memory allocation. The project includes:

- **Fixed NUMA thread assignment** - Proper CPU topology detection instead of naive modulo arithmetic
- **Configurable hyperthreading** - Default enabled, user can disable with `--no-hyperthreading`
- **Intel hybrid CPU support** - Detects P-cores vs E-cores
- **Development container** - Ubuntu 24.04 with all dependencies for consistent building

## 🏗️ Build Environment Setup

### Primary Development Method: Dev Container

**Always prefer the dev container for consistency**:

1. **Check if in container**: Look for `/.dockerenv` or check environment
2. **Start container**: If in VS Code, use "Dev Containers: Reopen in Container"
3. **Dependencies included**: All NUMA tools, build tools, debugging tools pre-installed

### Quick Build Commands

```bash
# Manual build steps
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel $(nproc)

# Run tests
ctest --list --output-on-failure
```

### Available VS Code Tasks

- **Ctrl+Shift+P** → "Tasks: Run Task":
  - `cmake-configure` - Configure CMake
  - `cmake-build` - Build project (default)
  - `cmake-release` - Release build
  - `cmake-clean` - Clean build directory
  - `test-cpu-topology` - Test CPU topology detection
  - `check-numa` - Display NUMA hardware info

## 🧠 Key Areas of Focus

### 1. NUMA Memory Management
**Files**: `ggml/src/ggml-cpu.c`, `src/llama-mmap.cpp`

- **NUMA mirroring**: Model weights duplicated across NUMA nodes
- **Thread-to-NUMA mapping**: Each thread accesses local memory
- **Memory allocation**: `numa_alloc_onnode()` for local allocation

### 2. CPU Topology Detection
**Files**: `common/common.cpp`, `common/common.h`

- **Linux-specific**: Reads `/sys/devices/system/cpu/` topology
- **Hyperthreading detection**: Groups sibling threads correctly
- **Intel hybrid support**: Distinguishes P-cores from E-cores

Key functions:
```cpp
detect_cpu_topology()           // Main topology detection
cpu_count_math_cpus()          // Count available CPUs with options
cpu_print_topology_info()     // Debug information display
```

### 3. Command-Line Interface
**Files**: `common/arg.cpp`

New arguments added:
- `--no-hyperthreading` - Disable hyperthreading (default: enabled)
- `--use-efficiency-cores` - Include E-cores in thread pool
- `--cpu-topology` - Display CPU topology and exit

### 4. Environment Variables
```bash
LLAMA_NO_HYPERTHREADING=1     # Disable hyperthreading
LLAMA_USE_EFFICIENCY_CORES=1  # Enable efficiency cores
```

## 🧪 Testing Strategy

### 1. Basic Functionality Tests

```bash
# Test CPU topology detection
./build/bin/llama-server --cpu-topology

# Test help output includes new flags
./build/bin/llama-server --help | grep -E "(hyperthreading|efficiency|topology)"

# Test NUMA hardware detection
numactl --hardware
```

### 2. Performance Validation

```bash
# Compare hyperthreading on/off
./build/bin/llama-bench -m model.gguf
./build/bin/llama-bench -m model.gguf --no-hyperthreading

# Test different thread counts
for threads in 4 8 16; do
    ./build/bin/llama-bench -m model.gguf --threads $threads
done

# NUMA binding test
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server --model model.gguf
```

### 3. Memory Access Monitoring

```bash
# Monitor NUMA memory access
perf stat -e node-loads,node-stores,node-load-misses,node-store-misses \
    ./build/bin/llama-bench -m model.gguf

# Check memory allocation patterns
numastat -p $(pgrep llama-server)
```

## 🔧 Development Workflow

### Making Changes

1. **Identify the area**: NUMA allocation, CPU detection, CLI args, etc.
2. **Use dev container**: Ensure consistent environment
3. **Build incrementally**: Use `cmake --build build` for faster iteration
4. **Test immediately**: Run `./build/bin/llama-server --cpu-topology` after changes
5. **Check compilation**: Use `get_errors` tool to validate syntax

### Common Edit Patterns

#### Adding New CPU Parameters
1. Update `cpu_params` struct in `common/common.h`
2. Add argument parsing in `common/arg.cpp`
3. Update `cpu_count_math_cpus()` logic in `common/common.cpp`
4. Test with `--cpu-topology` flag

#### Modifying NUMA Logic
1. Check `ggml-cpu.c` for thread computation changes
2. Update `llama-mmap.cpp` for memory allocation
3. Test on multi-NUMA system or simulate with `numactl`

#### CLI Changes
1. Add/modify arguments in `common/arg.cpp`
2. Update help text and descriptions
3. Test argument parsing with `--help`

### Debugging Approach

```bash
# Debug build for better symbols
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Use GDB with VS Code integration
# Set breakpoints in VS Code, use "Debug llama-server" launch config

# Monitor system calls
strace -e sched_setaffinity,numa_alloc_onnode ./build/bin/llama-server --cpu-topology

# Check CPU affinity assignment
taskset -cp $(pgrep llama-server)
```

## 📝 Code Standards

### Error Handling
- Always check return values for system calls
- Use `LOG_WRN()` for warnings, `LOG_ERR()` for errors
- Graceful fallbacks when NUMA/topology detection fails

### Platform Compatibility
- NUMA features are Linux-specific (`#if defined(__x86_64__) && defined(__linux__)`)
- Provide fallbacks for other platforms
- Test Windows compatibility doesn't break

### Performance Considerations
- Cache topology detection results
- Minimize system calls in hot paths
- Use `pin_cpu()` carefully - restore original affinity

### Testing Guidelines
1. Unit tests live in the `tests/` folder
2. Write tests with the Arrange, Act, Assert pattern
2. Ensure 90%+ coverage for new features
3. Run tests like this:
    ```bash
      set -e
      rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
      CMAKE_EXTRA="-DLLAMA_FATAL_WARNINGS=ON -DLLAMA_CURL=ON"
      time cmake -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} ..  2>&1 
      time make -j$(nproc) 2>&1 
      time ctest --list --output-on-failure 2>&1
    ```

## 🐛 Common Issues and Solutions

### Build Issues
```bash
# Missing dependencies
apt list --installed | grep -E "(numa|hwloc|cmake)"

# Clean build
rm -rf build && cmake -B build

# Verbose build output
cmake --build build --verbose
```

## 📚 Key Documentation Files

- `NUMA_IMPROVEMENTS.md` - Comprehensive technical documentation
- `.devcontainer/README.md` - Dev container usage guide
- `docs/build.md` - Official build instructions
- `build-numa.sh` - Automated build and test script

## 🎯 Success Criteria for Changes

1. **Builds successfully** in dev container
2. **No compilation errors** across all modified files
3. **Unit test coverage** for new features
3. **No failing unit tests** after changes

## 💡 Tips for AI Agents

1. **Always use the dev container** - it has all dependencies and correct environment
2. **Test incrementally** - build and test after each significant change
3. **Check multiple scenarios** - different thread counts, NUMA configurations
4. **Read existing code carefully** - NUMA and threading logic is subtle
5. **Use the build script** - `./build-numa.sh` provides comprehensive testing
6. **Check for platform-specific code** - many features are Linux-only
7. **Validate with real workloads** - not just compilation success

Remember: NUMA and CPU topology changes can have subtle effects. Always validate performance and correctness thoroughly before considering changes complete.
