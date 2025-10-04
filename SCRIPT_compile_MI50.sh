#!/bin/bash
#
# SCRIPT MI50 Compilation Script for llama.cpp
# Optimized build for AMD MI50 (gfx906) with ROCm/HIP support
# 
# This script compiles llama.cpp with maximum optimizations for the MI50 GPU
# including server support, flash attention, and all performance features
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE} SCRIPT MI50 llama.cpp Builder  ${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if we're in the right directory
if [[ ! -f "CMakeLists.txt" ]]; then
    echo -e "${RED}Error: Not in llama.cpp root directory${NC}"
    echo "Please run this script from the llama.cpp root directory"
    exit 1
fi

# Verify ROCm installation
echo -e "${YELLOW}Checking ROCm installation...${NC}"
if ! command -v rocm_agent_enumerator &> /dev/null; then
    echo -e "${RED}Error: ROCm not found. Please install ROCm first.${NC}"
    exit 1
fi

# Check for gfx906 support
GPUS=$(rocm_agent_enumerator)
if [[ ! "$GPUS" =~ "gfx906" ]]; then
    echo -e "${RED}Warning: gfx906 (MI50) not detected in system${NC}"
    echo "Available GPUs: $GPUS"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ ROCm installation verified${NC}"
echo -e "${GREEN}✓ Available GPUs: $GPUS${NC}"

# Set ROCm environment variables for optimal gfx906 compilation
echo -e "${YELLOW}Setting ROCm environment variables for gfx906...${NC}"
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HCC_AMDGPU_TARGET=gfx906
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export AMDGPU_TARGETS=gfx906
export GPU_TARGETS=gfx906

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -rf build
mkdir -p build

# Configure with maximum optimizations
echo -e "${YELLOW}Configuring CMake with MI50 optimizations...${NC}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_HIP_COMPILER_FORCED=1 \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -DGGML_HIP_GFX906_OPTIMIZED -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_HIP_FLAGS=" --offload-arch=gfx906 -DGGML_HIP_GFX906_OPTIMIZED -Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result -mllvm -amdgpu-simplify-libcall -mllvm -amdgpu-internalize-symbols -mllvm -amdgpu-enable-lower-module-lds=false -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -ffast-math -ffp-contract=fast" \
    -DGGML_HIP=ON \
    -DGGML_HIP_MMQ_MFMA=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_EXPORT_METRICS=ON \
    -DGGML_HIP_GFX906_OPTIMIZED=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_NO_PEER_COPY=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_STATIC=OFF

if [[ $? -ne 0 ]]; then
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configuration successful${NC}"

# Compile with all CPU cores and dump detailed logs
NPROC=$(nproc)
LOG_FILE="compilation_log.txt"
echo -e "${YELLOW}Compiling with $NPROC cores...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
echo -e "${YELLOW}Detailed compilation log will be saved to: $LOG_FILE${NC}"

# Clear previous log
> $LOG_FILE

# Run make with detailed output and save to log file
make -j$NPROC 2>&1 | tee $LOG_FILE

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo -e "${RED}✗ Compilation failed${NC}"
    echo -e "${RED}Check $LOG_FILE for detailed error information${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilation successful!${NC}"

# Verify the build
echo -e "${YELLOW}Verifying build...${NC}"

# Check if main executables were built
EXECUTABLES=(
    "bin/llama-cli"
    "bin/llama-server"
    "bin/llama-bench"
    "bin/libggml-hip.so"
)

ALL_GOOD=true
for exec in "${EXECUTABLES[@]}"; do
    if [[ -f "$exec" ]]; then
        echo -e "${GREEN}✓ $exec built successfully${NC}"
        
        # Check HIP linking for executables (not libraries)
        if [[ "$exec" =~ ^bin/llama- && ! "$exec" =~ \.so$ ]]; then
            if ldd "$exec" | grep -q "libggml-hip.so"; then
                echo -e "${GREEN}  ✓ HIP backend linked${NC}"
            else
                echo -e "${RED}  ✗ HIP backend not linked${NC}"
                ALL_GOOD=false
            fi
        fi
    else
        echo -e "${RED}✗ $exec not found${NC}"
        ALL_GOOD=false
    fi
done

if [[ "$ALL_GOOD" = false ]]; then
    echo -e "${RED}✗ Build verification failed${NC}"
    exit 1
fi

# Display ROCm libraries linked
echo -e "${YELLOW}ROCm libraries linked:${NC}"
ldd bin/llama-cli | grep -E "(hip|roc)" | head -5

# Quick functionality test
echo -e "${YELLOW}Testing HIP backend availability...${NC}"
if ./bin/llama-cli --help 2>/dev/null | grep -q "backend"; then
    echo -e "${GREEN}✓ llama-cli responding correctly${NC}"
else
    echo -e "${RED}✗ llama-cli test failed${NC}"
fi

# Success message
echo
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    ✓ BUILD COMPLETED SUCCESSFULLY    ${NC}"
echo -e "${GREEN}======================================${NC}"
echo
echo -e "${BLUE}Built executables:${NC}"
echo "  • CLI:    ./build/bin/llama-cli"
echo "  • Server: ./build/bin/llama-server" 
echo "  • Bench:  ./build/bin/llama-bench"
echo
echo -e "${BLUE}Optimizations enabled:${NC}"
echo "  • Target GPU: AMD MI50 (gfx906)"
echo "  • HIP/ROCm backend with MFMA support"
echo "  • Flash Attention kernels"
echo "  • All quantization formats"
echo "  • Performance metrics export"
echo "  • Native CPU optimizations"
echo "  • Optimization 5: GFX906 compiler flags (-ffast-math, early-inline, function-calls=false)"
echo
echo -e "${BLUE}Ready to run:${NC}"
echo "  ./SCRIPT_launch_server_MI50.sh <model.gguf>"
echo
echo -e "${YELLOW}Note: Make sure to set proper ROCm environment variables before running!${NC}"
echo
echo -e "${BLUE}For debugging with maximum HIP logging:${NC}"
echo "  export AMD_LOG_LEVEL=8"
echo "  export AMD_LOG_MASK=0xFFFFFFFF" 
echo "  export AMD_SERIALIZE_KERNEL=3"
echo "  ./SCRIPT_launch_server_MI50.sh <model.gguf> 2>&1 | tee hip_debug.log"
