# llama.cpp Development Container

This dev container provides a complete Ubuntu 24.04 environment for building and testing llama.cpp with NUMA support and optional GPU acceleration.

## Quick Start

1. Open the project in VS Code
2. When prompted, click "Reopen in Container" or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
3. The container will build with the basic development tools (no GPU support by default)

## Optional Components

By default, the container includes only the essential build tools. You can enable additional components by editing `.devcontainer/devcontainer.json`:

### CUDA Support (NVIDIA GPUs)
```json
"INSTALL_CUDA": "true"
```
Installs CUDA 12.9 toolkit for NVIDIA GPU acceleration.

### ROCm Support (AMD GPUs)  
```json
"INSTALL_ROCM": "true"
```
Installs ROCm 6.4 for AMD GPU acceleration.

### Python Dependencies
```json
"INSTALL_PYTHON_DEPS": "true"
```
Installs Python packages for model conversion tools:
- numpy, torch, transformers, sentencepiece, protobuf, gguf

## Example Configurations

### Full GPU Development (NVIDIA + Python)
```json
"build": {
    "args": {
        "INSTALL_CUDA": "true",
        "INSTALL_ROCM": "false", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

### AMD GPU Development
```json
"build": {
    "args": {
        "INSTALL_CUDA": "false",
        "INSTALL_ROCM": "true", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

### CPU-only with Python tools
```json
"build": {
    "args": {
        "INSTALL_CUDA": "false",
        "INSTALL_ROCM": "false", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

## Making Changes

### Method 1: Interactive Configuration Script (Recommended)
```bash
# Run the configuration helper
chmod +x .devcontainer/configure.sh
./.devcontainer/configure.sh
```

### Method 2: Manual Configuration
1. Edit `.devcontainer/devcontainer.json` 
2. Set the desired components to `"true"` or `"false"`
3. Rebuild the container: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

## Features

- **Ubuntu 24.04 LTS** base image
- **Complete build toolchain**: gcc, cmake, ninja, ccache
- **NUMA support**: libnuma-dev, numactl, hwloc for CPU topology detection
- **Optional GPU acceleration**: CUDA 12.9 and/or ROCm 6.4 support
- **Optional Python environment**: with packages for GGUF conversion tools
- **VS Code integration**: with C/C++, CMake, and Python extensions
- **Development tools**: gdb, valgrind for debugging

## Quick Start

1. **Open in VS Code**: Make sure you have the "Dev Containers" extension installed, then:
   - Open the llama.cpp folder in VS Code
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Select it and wait for the container to build and start

2. **Build the project**:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ```

3. **Test NUMA functionality**:
   ```bash
   # Check NUMA topology
   numactl --hardware
   
   # Test CPU topology detection
   ./build/bin/llama-server --cpu-topology
   
   # Run with specific NUMA settings
   numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server --model path/to/model.gguf
   ```

## Available Tools

### System Tools
- `numactl`: NUMA policy control
- `hwloc-info`: Hardware locality information
- `lscpu`: CPU information
- `ccache`: Compiler cache for faster rebuilds

### Build Configurations

#### Debug Build (default post-create)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

#### Release Build (optimized)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

#### With Additional Options
```bash
# Enable OpenBLAS
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

# Static build
cmake -B build -DBUILD_SHARED_LIBS=OFF

# Disable CURL if not needed
cmake -B build -DLLAMA_CURL=OFF
```

## Testing NUMA Improvements

The container includes tools to test the NUMA improvements:

### CPU Topology Detection
```bash
# View detailed CPU information
./build/bin/llama-server --cpu-topology

# Check current NUMA configuration
numactl --show

# Display NUMA hardware topology
numactl --hardware
```

### Performance Testing
```bash
# Test with default settings (hyperthreading enabled)
./build/bin/llama-bench -m model.gguf

# Test without hyperthreading
./build/bin/llama-bench -m model.gguf --no-hyperthreading

# Test with specific thread count
./build/bin/llama-bench -m model.gguf --threads 8

# Test with NUMA binding
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-bench -m model.gguf
```

### Environment Variables
```bash
# Disable hyperthreading via environment
LLAMA_NO_HYPERTHREADING=1 ./build/bin/llama-server --model model.gguf

# Enable efficiency cores
LLAMA_USE_EFFICIENCY_CORES=1 ./build/bin/llama-server --model model.gguf
```

## Development Workflow

1. **Code changes**: Edit files in VS Code with full IntelliSense support
2. **Build**: Use `Ctrl+Shift+P` → "CMake: Build" or terminal commands
3. **Debug**: Set breakpoints and use the integrated debugger
4. **Test**: Run executables directly or through the testing framework

## Troubleshooting

### Container Build Issues
- Ensure Docker Desktop is running
- Try rebuilding: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

### NUMA Issues
- Check if running on a NUMA system: `numactl --hardware`
- Verify CPU topology detection: `lscpu` and `hwloc-info`
- Test CPU affinity: `taskset -c 0-3 ./your-program`

### Build Issues
- Clear build cache: `rm -rf build && cmake -B build`
- Check ccache stats: `ccache -s`
- Use verbose build: `cmake --build build --verbose`
