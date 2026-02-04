#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝           


EOF

set -e

# 1. Check location
[[ ! -f "CMakeLists.txt" ]] && echo "Error: Not in llama.cpp root directory" && exit 1

# 2. Setup ROCm Environment Variables
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}

if command -v amdgpu-arch &> /dev/null; then
    AMDGPU_ARCH=$(amdgpu-arch | head -n 1)
    echo "Detected AMD GPU Architecture: $AMDGPU_ARCH"
else
    echo "Warning: amdgpu-arch tool not found. Defaulting to 'native'."
    AMDGPU_ARCH="native"
fi

rm -rf build && mkdir -p build && cd build

# ============================================================================
# CMAKE FLAGS DOCUMENTATION 
# ============================================================================
# 
#  CMAKE BUILD CONFIGURATION ===                                            
#
# CMAKE_BUILD_TYPE                 Build type: Release, Debug, RelWithDebInfo, MinSizeRel
# CMAKE_C_COMPILER                 C compiler path (use ROCm clang for HIP)
# CMAKE_CXX_COMPILER               C++ compiler path (use ROCm clang++ for HIP)
# CMAKE_HIP_ARCHITECTURES          Target GPU arch: gfx906 (MI50/60)
# CMAKE_HIP_COMPILER_FORCED=1      Skip HIP compiler detection. Fixes bfloat16 duplicate symbol
#                                  error on ROCm 6.x (hip_bf16.h multi-include bug) - see issue 
#
#  COMPILER FLAGS ===                                                             
#
# -O3                              Maximum optimization level
# -march=native                    Optimize for host CPU architecture
# -mtune=native                    Tune instruction scheduling for host CPU
# -DNDEBUG                         Disable assert() checks (release mode)
# -Wno-ignored-attributes          Suppress CUDA __host__/__device__ attribute warnings
# -Wno-cuda-compat                 Suppress CUDA compatibility warnings in HIP
# -Wno-unused-result               Suppress unused return value warnings
#
#  GGML GENERAL OPTIONS ===                                                       
#
# GGML_STATIC=OFF                  Static link libraries (ON=static, OFF=shared/dynamic)
# GGML_NATIVE=ON                   Enable CPU-native optimizations (AVX, AVX2, etc)
# GGML_LTO=OFF                     Link Time Optimization (slower build, faster binary)
# GGML_CCACHE=ON                   Use ccache for faster rebuilds if available
# GGML_OPENMP=ON                   Enable OpenMP for CPU parallelization
# GGML_CPU=ON                      Enable CPU backend
# GGML_CPU_HBM=OFF                 Use memkind for High Bandwidth Memory (HBM)
# GGML_CPU_REPACK=ON               Runtime weight conversion Q4_0 -> Q4_X_X
# GGML_BACKEND_DL=OFF              Build backends as dynamic libraries
# GGML_SCHED_NO_REALLOC=OFF        Disable reallocations in ggml-alloc (debug)
#
#  CPU SIMD INSTRUCTION SETS ===                                                  
#
# GGML_SSE42=ON                    Enable SSE 4.2 instructions
# GGML_AVX=ON                      Enable AVX instructions
# GGML_AVX2=ON                     Enable AVX2 instructions
# GGML_AVX_VNNI=OFF                Enable AVX-VNNI (Alder Lake+)
# GGML_AVX512=OFF                  Enable AVX-512F instructions
# GGML_AVX512_VBMI=OFF             Enable AVX-512 VBMI
# GGML_AVX512_VNNI=OFF             Enable AVX-512 VNNI
# GGML_AVX512_BF16=OFF             Enable AVX-512 BF16
# GGML_FMA=ON                      Enable FMA (Fused Multiply-Add)
# GGML_F16C=ON                     Enable F16C (half-float conversions)
# GGML_BMI2=ON                     Enable BMI2 bit manipulation
# GGML_AMX_TILE=OFF                Enable Intel AMX tile instructions
# GGML_AMX_INT8=OFF                Enable Intel AMX INT8
# GGML_AMX_BF16=OFF                Enable Intel AMX BF16
#
#  AMD HIP/ROCm BACKEND ===                                                       
#
# GGML_HIP=ON                      Enable AMD ROCm/HIP backend
# GGML_HIP_GRAPHS=OFF              Use HIP graphs for kernel batching (experimental)
# GGML_HIP_NO_VMM=ON               Disable Virtual Memory Management (required for MI50)
# GGML_HIP_ROCWMMA_FATTN=OFF       Use rocWMMA for Flash Attention (CDNA2+ only)
# GGML_HIP_MMQ_MFMA=ON             Use MFMA matrix instructions for MMQ (CDNA GPUs)
# GGML_HIP_EXPORT_METRICS=OFF      Export kernel performance metrics
# GGML_HIP_NO_HIPBLASLT=OFF        Disable hipBLASLt (enable if crashes on your ROCm)
#
#  NVIDIA CUDA BACKEND ===                                                        
#
# GGML_CUDA=OFF                    Enable NVIDIA CUDA backend
# GGML_CUDA_FORCE_MMQ=OFF          Force MMQ kernels instead of cuBLAS
# GGML_CUDA_FORCE_CUBLAS=OFF       Force cuBLAS instead of MMQ kernels
# GGML_CUDA_NO_PEER_COPY=OFF       Disable peer-to-peer GPU copies (multi-GPU)
# GGML_CUDA_NO_VMM=OFF             Disable CUDA Virtual Memory Management
# GGML_CUDA_GRAPHS=ON              Use CUDA graphs for kernel batching
#
#  FLASH ATTENTION  ===                                                           
#
# GGML_CUDA_FA=ON                  Enable Flash Attention CUDA/HIP kernels
# GGML_CUDA_FA_ALL_QUANTS=OFF      Compile FA for all quant types (Q4, Q5, Q8, etc)
#                                  ON = slower build, supports all quants
#                                  OFF = faster build, only F16 FA
#
#  OTHER GPU BACKENDS ===                                                         
#
# GGML_VULKAN=OFF                  Enable Vulkan backend (cross-platform GPU)
# GGML_VULKAN_DEBUG=OFF            Enable Vulkan debug output
# GGML_VULKAN_VALIDATE=OFF         Enable Vulkan validation layers
# GGML_METAL=OFF                   Enable Apple Metal backend (macOS/iOS)
# GGML_METAL_EMBED_LIBRARY=ON      Embed Metal shaders in binary
# GGML_SYCL=OFF                    Enable Intel SYCL backend (oneAPI)
# GGML_OPENCL=OFF                  Enable OpenCL backend (Adreno GPUs)
# GGML_MUSA=OFF                    Enable Moore Threads MUSA backend
# GGML_WEBGPU=OFF                  Enable WebGPU backend (browsers)
# GGML_RPC=OFF                     Enable RPC for distributed inference
#
#  OTHER ACCELERATORS ===                                                         
#
# GGML_BLAS=OFF                    Use BLAS library (OpenBLAS, MKL, etc)
# GGML_ACCELERATE=ON               Use Apple Accelerate framework (macOS)
# GGML_LLAMAFILE=ON                Use llamafile SGEMM kernels
# GGML_HEXAGON=OFF                 Enable Qualcomm Hexagon DSP backend
# GGML_ZENDNN=OFF                  Enable AMD ZenDNN for Zen CPUs
# GGML_ZDNN=OFF                    Enable IBM zDNN for Z mainframes
#
#  LLAMA.CPP BUILD TARGETS ===                                                    
#
# LLAMA_BUILD_SERVER=ON            Build llama-server (OpenAI-compatible HTTP API)
# LLAMA_BUILD_EXAMPLES=ON          Build example programs (simple, batched, etc)
# LLAMA_BUILD_TOOLS=ON             Build tools (quantize, bench, perplexity, etc)
# LLAMA_BUILD_TESTS=OFF            Build test suite (slower, for development)
# LLAMA_BUILD_COMMON=ON            Build common utilities library
# LLAMA_TOOLS_INSTALL=ON           Install tools to system
#
#  LLAMA.CPP FEATURES ===                                                         
#
# LLAMA_CURL=ON                    Enable libcurl for HuggingFace downloads (-hf flag)
# LLAMA_HTTPLIB=ON                 Use cpp-httplib if curl disabled
# LLAMA_OPENSSL=OFF                Use OpenSSL for HTTPS support
# LLAMA_LLGUIDANCE=OFF             Include LLGuidance for structured output
#
#  DEBUG & SANITIZERS ===                                                         
#
# GGML_ALL_WARNINGS=ON             Enable all compiler warnings
# GGML_FATAL_WARNINGS=OFF          Treat warnings as errors (-Werror)
# GGML_SANITIZE_THREAD=OFF         Enable ThreadSanitizer (race detection)
# GGML_SANITIZE_ADDRESS=OFF        Enable AddressSanitizer (memory errors)
# GGML_SANITIZE_UNDEFINED=OFF      Enable UndefinedBehaviorSanitizer
# GGML_GPROF=OFF                   Enable gprof profiling
#
# ============================================================================

{
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
    -DCMAKE_HIP_COMPILER_FORCED=1 \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -fno-finite-math-only -ffp-contract=fast" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_EXPORT_METRICS=ON \
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


make -j$(nproc)

echo ""
echo "Build complete: ./build/bin/llama-cli, llama-server, llama-bench"
} 2>&1 | tee ../compilation_log.txt
