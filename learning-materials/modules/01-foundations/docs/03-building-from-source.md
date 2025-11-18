# Building llama.cpp from Source

**Learning Module**: Module 1 - Foundations
**Estimated Reading Time**: 15 minutes
**Prerequisites**: Basic command-line knowledge, familiarity with build systems (Make or CMake)
**Related Content**:
- [What is llama.cpp?](./01-what-is-llama-cpp.md)
- [Lab 1: Setup and First Inference](../../labs/lab-01/)
- [Codebase Architecture](./05-codebase-architecture.md)

---

## Why Build from Source?

### Pre-built Binaries vs. Source Build

| Aspect | Pre-built Binary | Build from Source |
|--------|-----------------|-------------------|
| **Setup Time** | Minutes | 10-30 minutes |
| **Hardware Optimization** | Generic | Optimized for your CPU/GPU |
| **Customization** | None | Full control |
| **Latest Features** | Release versions | Git head/branches |
| **Dependencies** | None | Build tools required |
| **Performance** | Good | Better (native optimizations) |

### When to Build from Source

✅ **You should build from source when**:
- You want maximum performance (native CPU optimizations)
- You need GPU acceleration (CUDA, Metal, Vulkan, etc.)
- You want the latest features from Git
- You need to customize the code
- Pre-built binaries don't work on your platform
- You're developing or contributing to llama.cpp

❌ **Pre-built binaries are fine when**:
- You just want to try llama.cpp quickly
- You're on a common platform (macOS/Linux x86_64)
- You don't need GPU acceleration
- Performance is not critical

---

## Prerequisites

### Required Tools

#### All Platforms

1. **Git**: For cloning the repository
2. **CMake**: Build system generator (version 3.14+)
3. **C/C++ Compiler**:
   - Linux: GCC 11+ or Clang 14+
   - macOS: Xcode Command Line Tools
   - Windows: MSVC 2022 or MinGW

#### Optional (for acceleration)

- **CUDA Toolkit**: For NVIDIA GPU support (11.7+)
- **Vulkan SDK**: For cross-platform GPU support
- **ROCm**: For AMD GPU support
- **Metal**: Built-in on macOS (M1/M2/M3)

### Installing Prerequisites

#### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake (via Homebrew)
brew install cmake

# Install Git (usually pre-installed)
git --version
```

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install build essentials
sudo apt install build-essential git cmake

# Optional: Install libcurl (for model downloading)
sudo apt install libcurl4-openssl-dev

# Verify installation
gcc --version
cmake --version
```

#### Linux (Fedora/RHEL)

```bash
# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git

# Optional: Install libcurl
sudo dnf install libcurl-devel
```

#### Windows

**Option 1: Visual Studio**
1. Install [Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/vs/community/)
2. Select "Desktop development with C++" workload
3. Install [CMake](https://cmake.org/download/) (or use VS built-in)
4. Install [Git for Windows](https://git-scm.com/download/win)

**Option 2: MSYS2/MinGW**
1. Install [MSYS2](https://www.msys2.org/)
2. Open MSYS2 UCRT64 terminal:
```bash
pacman -S git mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake
```

---

## Basic CPU Build

### Step 1: Clone the Repository

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Optional: checkout specific version
# git checkout b5046  # Use a specific commit/tag
```

### Step 2: Build with CMake

#### Linux/macOS

```bash
# Configure build (creates build directory)
cmake -B build

# Build the project (use -j for parallel compilation)
cmake --build build --config Release -j $(nproc)

# On macOS, use: -j $(sysctl -n hw.ncpu)
```

#### Windows (Visual Studio)

```powershell
# Open "Developer Command Prompt for VS 2022"

# Configure
cmake -B build

# Build
cmake --build build --config Release
```

#### Windows (MSYS2)

```bash
# In MSYS2 UCRT64 terminal
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### Step 3: Verify Build

```bash
# Test the build
./build/bin/llama-cli --version

# Should output something like:
# llama-cli version: b5046 (build 3e0ba0e6)
```

### Build Output

After successful build, you'll find binaries in `build/bin/`:

```
build/bin/
├── llama-cli              # Main CLI tool
├── llama-server           # HTTP API server
├── llama-bench            # Benchmarking tool
├── llama-perplexity       # Quality measurement
├── llama-quantize         # Model quantization
├── llama-embedding        # Embedding extraction
└── ...                    # Other tools
```

---

## CMake Options Explained

### Common Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: Release, Debug, RelWithDebInfo |
| `BUILD_SHARED_LIBS` | `ON` | Build shared libraries instead of static |
| `GGML_NATIVE` | `OFF` | Enable native CPU optimizations (-march=native) |
| `GGML_OPENMP` | `ON` | Enable OpenMP for parallelization |
| `GGML_CUDA` | `OFF` | Enable CUDA support (NVIDIA GPUs) |
| `GGML_METAL` | `ON` (macOS) | Enable Metal support (Apple Silicon) |
| `GGML_VULKAN` | `OFF` | Enable Vulkan support |
| `GGML_HIP` | `OFF` | Enable HIP support (AMD GPUs) |
| `LLAMA_CURL` | `ON` | Enable curl for model downloading |

### Using CMake Options

```bash
# Example: Native optimizations + static build
cmake -B build \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### Optimization Levels

```bash
# Debug build (with debug symbols, no optimization)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release build (maximum optimization, no debug info)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Release with debug info (optimization + debug symbols)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Native CPU Optimizations

```bash
# Enable CPU-specific optimizations (not portable!)
cmake -B build -DGGML_NATIVE=ON

# This enables -march=native, which optimizes for YOUR specific CPU
# Warning: Binary won't run on different CPUs
```

**Performance Impact**:
```
Generic build:     100% (baseline)
Native build:      120-150% (20-50% faster)
```

---

## Platform-Specific Builds

### macOS: Metal Acceleration (Apple Silicon)

Metal is enabled by default on macOS. To explicitly control it:

```bash
# Build with Metal (default on macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# Disable Metal (CPU only)
cmake -B build -DGGML_METAL=OFF
cmake --build build --config Release
```

**Testing Metal**:
```bash
# Run with GPU acceleration
./build/bin/llama-cli -m model.gguf -ngl 99

# Should see output like:
# ggml_metal_init: device = Apple M1 Pro
# ggml_metal_init: hasUnifiedMemory = true
```

### Linux: CUDA Support (NVIDIA GPUs)

#### Install CUDA Toolkit

```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Or use package manager:

# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# Fedora
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf install cuda-toolkit-12-6
```

#### Build with CUDA

```bash
# Configure with CUDA
cmake -B build -DGGML_CUDA=ON

# Specify compute capability (optional, for your specific GPU)
# RTX 3090: 86, RTX 4090: 89, A100: 80
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"

# Build
cmake --build build --config Release
```

#### Test CUDA Build

```bash
# Check CUDA detection
./build/bin/llama-cli -m model.gguf -ngl 99

# Should see:
# ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
# ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
# ggml_cuda_init: found 1 CUDA devices:
#   Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9
```

### Linux: AMD GPU Support (ROCm/HIP)

#### Install ROCm

```bash
# Ubuntu/Debian
sudo apt install rocm-hip-sdk rocm-libs

# Fedora
sudo dnf install rocm-hip-sdk rocm-libs
```

#### Build with HIP

```bash
# Specify your GPU architecture
# RX 7900 XTX: gfx1100, RX 6800 XT: gfx1030
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### Windows: CUDA Build

```powershell
# Open "x64 Native Tools Command Prompt for VS 2022"

# Configure with CUDA
cmake -B build -DGGML_CUDA=ON

# Build
cmake --build build --config Release
```

### Cross-Platform: Vulkan

#### Install Vulkan SDK

- **Windows/macOS/Linux**: Download from [LunarG](https://vulkan.lunarg.com/sdk/home)

#### Build with Vulkan

```bash
# macOS/Linux
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release

# Windows
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

---

## Advanced Build Configurations

### Static Build (Portable Binary)

```bash
# Build static binary (no shared library dependencies)
cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release

# Test portability
ldd build/bin/llama-cli  # Should show minimal dependencies
```

### Cross-Compilation

#### ARM64 on x86_64 (Linux)

```bash
# Install cross-compilation tools
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Configure for ARM64
cmake -B build \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --config Release
```

#### Android

```bash
# Set Android NDK path
export ANDROID_NDK=/path/to/android-ndk

# Configure for Android
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --config Release
```

### Faster Builds with Ninja

```bash
# Install Ninja
# Linux: sudo apt install ninja-build
# macOS: brew install ninja
# Windows: Download from ninja-build.org

# Use Ninja generator (much faster than Make)
cmake -B build -G Ninja
cmake --build build --config Release

# Typical speedup: 2-3x faster compilation
```

### Parallel Compilation

```bash
# Use all CPU cores
cmake --build build --config Release -j $(nproc)

# Use specific number of cores (e.g., 8)
cmake --build build --config Release -j 8

# Typical compilation time (8 cores):
# - CPU only: 2-5 minutes
# - With CUDA: 10-15 minutes
```

### ccache (Faster Recompilation)

```bash
# Install ccache
# Linux: sudo apt install ccache
# macOS: brew install ccache

# Configure CMake to use ccache
cmake -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build --config Release

# Subsequent builds will be much faster (cached)
```

---

## Troubleshooting

### Common Build Errors

#### Error: "CMake version too old"

```
CMake Error: CMake 3.14 or higher is required.
```

**Solution**:
```bash
# Update CMake
# macOS
brew upgrade cmake

# Linux - download latest from cmake.org
wget https://cmake.org/files/v3.28/cmake-3.28.0-linux-x86_64.sh
sudo sh cmake-3.28.0-linux-x86_64.sh --prefix=/usr/local --skip-license
```

#### Error: "CUDA not found"

```
CMake Error: Could not find CUDA
```

**Solution**:
```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Set CUDA path explicitly
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

#### Error: "cannot find -lcurl"

```
/usr/bin/ld: cannot find -lcurl
```

**Solution**:
```bash
# Install libcurl development package
# Ubuntu/Debian
sudo apt install libcurl4-openssl-dev

# Fedora/RHEL
sudo dnf install libcurl-devel

# Or disable curl
cmake -B build -DLLAMA_CURL=OFF
```

#### Error: "Unsupported GPU architecture"

```
nvcc warning: Cannot find valid GPU for '-arch=native'
```

**Solution**:
```bash
# Explicitly specify compute capability
# Find yours: https://developer.nvidia.com/cuda-gpus
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
```

#### Error: Metal compilation fails (macOS)

```
error: metal: command not found
```

**Solution**:
```bash
# Update Xcode Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install

# Or disable Metal
cmake -B build -DGGML_METAL=OFF
```

### Platform-Specific Issues

#### macOS: "xcrun: error: invalid active developer path"

**Solution**:
```bash
xcode-select --install
```

#### Linux: "GLIBCXX version not found"

**Solution**:
```bash
# Update GCC
sudo apt install gcc-11 g++-11

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

#### Windows: "MSB8066: Custom build exited with code 1"

**Solution**:
- Ensure you're using "Developer Command Prompt for VS"
- Not regular CMD or PowerShell
- Or use MSYS2 instead

---

## Performance Optimization

### Compiler Flags for Maximum Performance

```bash
# Aggressive optimizations (may reduce stability)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native" \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
  -DGGML_NATIVE=ON

cmake --build build --config Release
```

### OpenMP Configuration

```bash
# Default: OpenMP enabled
cmake -B build -DGGML_OPENMP=ON

# Set number of threads at runtime
export OMP_NUM_THREADS=8
./build/bin/llama-cli -m model.gguf -t 8

# Disable OpenMP (use manual threading)
cmake -B build -DGGML_OPENMP=OFF
```

### BLAS Acceleration (CPU)

```bash
# Install OpenBLAS
# Linux
sudo apt install libopenblas-dev

# macOS (uses Accelerate framework by default)
# Already included

# Build with OpenBLAS
cmake -B build \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release
```

---

## Building Specific Components

### Build Only CLI (Faster)

```bash
# Configure normally
cmake -B build

# Build only llama-cli
cmake --build build --config Release --target llama-cli
```

### Build Only Server

```bash
cmake --build build --config Release --target llama-server
```

### Build Everything

```bash
# Build all targets
cmake --build build --config Release --target all
```

---

## Testing Your Build

### Basic Functionality Test

```bash
# 1. Check version
./build/bin/llama-cli --version

# 2. Check help
./build/bin/llama-cli --help

# 3. Download and test with a small model
./build/bin/llama-cli -hf microsoft/Phi-3-mini-4k-instruct-gguf \
  -m Phi-3-mini-4k-instruct-q4.gguf \
  -p "Hello, world!" -n 50

# 4. Benchmark
./build/bin/llama-bench -m model.gguf
```

### Verify GPU Acceleration

```bash
# CUDA
./build/bin/llama-cli -m model.gguf -ngl 99 2>&1 | grep -i cuda

# Metal
./build/bin/llama-cli -m model.gguf -ngl 99 2>&1 | grep -i metal

# Vulkan
./build/bin/llama-cli -m model.gguf -ngl 99 2>&1 | grep -i vulkan
```

---

## Updating Your Build

### Update to Latest Version

```bash
# Pull latest changes
cd llama.cpp
git pull

# Rebuild (CMake detects changes automatically)
cmake --build build --config Release

# Clean rebuild if needed
rm -rf build
cmake -B build -DGGML_CUDA=ON  # Your original options
cmake --build build --config Release
```

### Switching Branches

```bash
# List available branches
git branch -a

# Switch to specific branch
git checkout feature-branch

# Rebuild
cmake --build build --config Release
```

---

## Best Practices

### Development Workflow

```bash
# 1. Use separate build directories for different configs
cmake -B build-cpu
cmake -B build-cuda -DGGML_CUDA=ON
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug

# 2. Use ccache for faster rebuilds
cmake -B build -DCMAKE_C_COMPILER_LAUNCHER=ccache

# 3. Use Ninja for faster compilation
cmake -B build -G Ninja

# 4. Keep source and build separate
# Never build in source directory
```

### Production Builds

```bash
# Optimized, static, portable
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=ON

# Strip symbols for smaller binary
strip build/bin/llama-cli

# Test on target system before deploying
```

---

## Interview Questions

**Q: "Why might you need to build llama.cpp from source instead of using pre-built binaries?"**

**A**: Discuss:
- Hardware-specific optimizations (CUDA, Metal, AVX512)
- Native CPU optimizations (-march=native)
- Latest features not in releases
- Custom modifications
- Static linking for portability
- Platform-specific features

**Q: "What's the difference between GGML_NATIVE=ON and not using it?"**

**A**: Cover:
- GGML_NATIVE enables -march=native
- Optimizes for specific CPU architecture
- Trade-off: Performance vs. portability
- Native build won't run on different CPUs
- Typical speedup: 20-50%

**Q: "How does CMake detect and enable GPU acceleration?"**

**A**: Explain:
- CUDA: Finds nvcc compiler, CUDA toolkit
- Metal: Detects macOS, looks for Metal framework
- Vulkan: Searches for Vulkan SDK
- Can explicitly enable/disable with GGML_* options
- Runtime detection also occurs (device availability)

---

## Further Reading

### Official Documentation
- [llama.cpp Build Guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [Backend Documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend)
- [Docker Guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)

### Related Content
- [What is llama.cpp?](./01-what-is-llama-cpp.md)
- [Lab 1: Setup and First Inference](../../labs/lab-01/)
- [Codebase Architecture](./05-codebase-architecture.md)

### Platform-Specific
- [Android Build Guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [CUDA on Fedora](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CUDA-FEDORA.md)
- [SYCL for Intel GPUs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Feedback**: [Submit feedback](../../../feedback/)
