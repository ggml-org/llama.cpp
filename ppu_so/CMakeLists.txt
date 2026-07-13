# Standalone build for the two external kernel .so files. NOT part of the llama.cpp build tree.
#
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 \
#         -DTorch_DIR=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake/Torch"))')
#   cmake --build build -j
#
# On the PPU box (ppu001) instead pass -DCMAKE_CUDA_COMPILER=<ppu-nvcc> -DCMAKE_CUDA_ARCHITECTURES=OFF: a forced
# -arch=sm_XX routes onto ppu0015 and rejects ppu001-only atoms. See the ppu-ptx skill.
cmake_minimum_required(VERSION 3.18)
project(ppu_so LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")

find_package(CUDAToolkit REQUIRED)
find_package(Git REQUIRED)

# Submodules live under <llama.cpp>/thirdparty (see .gitmodules).
set(FA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/flash-attention")
set(DG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/DeepGEMM")

# Torch is a COMPILE-time dependency only (headers). We never link ${TORCH_LIBRARIES}:
#   * FA:  flash.h includes <ATen/cuda/CUDAGeneratorImpl.h> for the header-only `at::PhiloxCudaState philox_args`.
#   * MoE: DeepGEMM's JIT headers include <torch/python.h> for ScalarType / Tensor decls.
# Verify after building:  ldd libppu_fa.so libppu_moe.so | grep -i torch   -> must be EMPTY.
find_package(Torch REQUIRED)

# ---------------- libppu_fa.so ----------------
# FA forward kernels (per-headdim template instances) + our torch-free shim. NO libtorch linked.
# Only the NON-SPLIT forward kernels: the shim pins num_splits=1, so run_mha_fwd_splitkv_dispatch is never called.
file(GLOB FA_KERNEL_SRCS "${FA_ROOT}/csrc/flash_attn/src/flash_fwd_hdim*_sm80.cu")
if (NOT FA_KERNEL_SRCS)
    message(FATAL_ERROR "no FA kernel .cu found under ${FA_ROOT} -- did you `git submodule update --init`?")
endif()
list(LENGTH FA_KERNEL_SRCS FA_N)
message(STATUS "ppu_fa: ${FA_N} flash_fwd kernel translation units")

add_library(ppu_fa SHARED fa/flash_api_c.cpp ${FA_KERNEL_SRCS})
target_include_directories(ppu_fa PRIVATE
    "${FA_ROOT}/csrc/flash_attn"
    "${FA_ROOT}/csrc/flash_attn/src"
    "${FA_ROOT}/csrc/cutlass/include"     # cutlass stays INSIDE the .so
    ${TORCH_INCLUDE_DIRS})                # headers only (PhiloxCudaState); not linked
target_compile_definitions(ppu_fa PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_ALIBI)
target_compile_options(ppu_fa PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections -Xcompiler=-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections -fdata-sections>)
target_link_options(ppu_fa PRIVATE "LINKER:--gc-sections")
target_link_libraries(ppu_fa PRIVATE CUDA::cudart)
set_target_properties(ppu_fa PROPERTIES OUTPUT_NAME ppu_fa)

# test_fa: standalone correctness check (packed layout, vs fp32 CPU ref). Links libppu_fa.
add_executable(test_fa fa/test_fa.cpp)
target_link_libraries(test_fa PRIVATE ppu_fa CUDA::cudart)

# ---------------- libppu_moe.so ----------------  [TORCH-FREE runtime]
# moe/deepgemm_c.cpp re-implements DeepGEMM's sm90_m_grouped_bf16_gemm_contiguous body on RAW POINTERS, reusing its
# torch-free primitives (get_best_config, SM90BF16GemmRuntime::{generate,launch_impl}, compiler->build,
# construct_launch_config) and launching on the CALLER's stream. It calls no torch function; -ffunction-sections +
# --gc-sections strips the unreferenced torch-using helpers so the shipped .so has no libtorch dependency.
#
# DeepGEMM JIT-compiles its kernel with NVRTC on first use and disk-caches the cubin, so the runtime box needs the
# CUDA driver + nvrtc but NOT libtorch. Set -DPPU_BUILD_MOE=OFF to build only libppu_fa.so.
option(PPU_BUILD_MOE "build libppu_moe.so (needs torch HEADERS to compile; does NOT link libtorch)" ON)
if (PPU_BUILD_MOE)
    # DeepGEMM's DeviceRuntime ctor unconditionally allocates a cuBLASLt workspace via torch::empty() and holds it in
    # a torch::Tensor member -- a HARD libtorch link dependency (at::_ops::empty_memory_format, UndefinedTensorImpl,
    # AutogradMeta vtable, ...) that --gc-sections cannot strip because the ctor really references it. Our bf16
    # grouped-GEMM path never touches cuBLASLt, so we compile the submodule with -DDG_NO_TORCH and apply a small
    # patch that guards that member + its ctor/dtor/accessors. Idempotent: skipped if already applied.
    execute_process(COMMAND ${GIT_EXECUTABLE} apply --reverse --check
                            "${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-deepgemm-no-torch-cublaslt.patch"
                    WORKING_DIRECTORY "${DG_ROOT}" RESULT_VARIABLE DG_PATCH_APPLIED ERROR_QUIET)
    if (NOT DG_PATCH_APPLIED EQUAL 0)
        message(STATUS "ppu_moe: applying patches/0001-deepgemm-no-torch-cublaslt.patch to ${DG_ROOT}")
        execute_process(COMMAND ${GIT_EXECUTABLE} apply
                                "${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-deepgemm-no-torch-cublaslt.patch"
                        WORKING_DIRECTORY "${DG_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
    else()
        message(STATUS "ppu_moe: DeepGEMM no-torch patch already applied")
    endif()

    # DeepGEMM's JIT compiles the generated kernel with just two -I flags: $DG_LIBRARY_ROOT/include and $CUDA_HOME/
    # include. The generated code includes <cutlass/...> and <cute/...>, so upstream's develop.sh symlinks cutlass
    # into deep_gemm/include. Do the same here -- it is a RUNTIME requirement of the .so, not just a dev convenience.
    foreach(dir cutlass cute)
        if (NOT EXISTS "${DG_ROOT}/deep_gemm/include/${dir}")
            file(CREATE_LINK "${DG_ROOT}/third-party/cutlass/include/${dir}"
                             "${DG_ROOT}/deep_gemm/include/${dir}" SYMBOLIC)
            message(STATUS "ppu_moe: linked cutlass '${dir}' into deep_gemm/include (needed by the JIT)")
        endif()
    endforeach()

    set_source_files_properties(moe/deepgemm_c.cpp PROPERTIES LANGUAGE CUDA)
    add_library(ppu_moe SHARED moe/deepgemm_c.cpp)
    target_compile_options(ppu_moe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections -Xcompiler=-fdata-sections>)
    target_compile_definitions(ppu_moe PRIVATE FMT_HEADER_ONLY DG_NO_TORCH)  # header-only fmt; no torch/cuBLASLt
    # DeepGEMM's jit headers include <torch/python.h> -> pulls Python.h + pybind11. Headers only; nothing links.
    # (Ask the interpreter directly: cmake 3.22's FindPython3 misses conda's Development.Module component.)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sysconfig;print(sysconfig.get_paths()['include'])"
                    OUTPUT_VARIABLE PY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11;print(pybind11.get_include())"
                    OUTPUT_VARIABLE PYBIND11_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_include_directories(ppu_moe PRIVATE
        ${TORCH_INCLUDE_DIRS}                          # torch headers (compile-time only; not linked)
        ${PY_INCLUDE} ${PYBIND11_INCLUDE}
        "${DG_ROOT}/csrc"                              # apis/ + jit/ + jit_kernels/
        "${DG_ROOT}/deep_gemm/include"                 # deep_gemm/common/types.cuh (also the JIT's -I at runtime)
        "${DG_ROOT}/third-party/cutlass/include"
        "${DG_ROOT}/third-party/fmt/include")
    set_target_properties(ppu_moe PROPERTIES OUTPUT_NAME ppu_moe)
    # NO ${TORCH_LIBRARIES}. --gc-sections drops the unreferenced torch-using helpers.
    target_link_options(ppu_moe PRIVATE "LINKER:--gc-sections")
    target_link_libraries(ppu_moe PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)

    # test_moe: standalone grouped-GEMM check vs an fp32 CPU reference. Links libppu_moe.
    add_executable(test_moe moe/test_moe.cpp)
    target_link_libraries(test_moe PRIVATE ppu_moe CUDA::cudart)
endif()
