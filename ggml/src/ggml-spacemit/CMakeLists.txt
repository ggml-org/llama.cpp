set(TARGET_NAME ggml-spacemit)

# spine-runtime is consumed as a prebuilt release package.
if(NOT DEFINED SPERT_DIR AND DEFINED ENV{SPERT_DIR})
    set(SPERT_DIR "$ENV{SPERT_DIR}")
endif()
if(NOT SPERT_DIR)
    message(FATAL_ERROR "SPERT_DIR is not set. Point it at the extracted spine-runtime release (with include/ and lib/).")
endif()
if(NOT IS_DIRECTORY "${SPERT_DIR}/include" OR NOT IS_DIRECTORY "${SPERT_DIR}/lib")
    message(FATAL_ERROR "Invalid SPERT_DIR at ${SPERT_DIR}: expected include/ and lib/")
endif()
find_library(SPERT_LIB NAMES spert PATHS "${SPERT_DIR}/lib" NO_CMAKE_FIND_ROOT_PATH REQUIRED)
message(STATUS "SpacemiT backend using spine-runtime at ${SPERT_DIR} (${SPERT_LIB})")

# Detect IME support (inline version of FindSMTIME.cmake)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    include(CheckCSourceCompiles)
    set(SMT_MARCH_STR "-march=rv64gcv_zfh_zvfh_zba_zicbop")
    if((CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15) OR
       CMAKE_C_COMPILER_ID MATCHES "Clang")
        string(APPEND SMT_MARCH_STR "_xsmtvdotii")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${SMT_MARCH_STR}")

    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1, i4\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4)
    check_c_source_compiles("int main() {__asm__ volatile(\"vpack.vv v2, v0, v1, 2\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK)
    check_c_source_compiles("int main() {__asm__ volatile(\"vnspack.vv v2, v0, v1, 2\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK)
    unset(CMAKE_REQUIRED_FLAGS)

    set(RISCV64_SPACEMIT_IME_SPEC "")
    if(SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
        set(RISCV64_SPACEMIT_IME_SPEC "RISCV64_SPACEMIT_IME1")
    endif()
    if(SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4 AND SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK AND SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK)
        list(APPEND RISCV64_SPACEMIT_IME_SPEC "RISCV64_SPACEMIT_IME2")
    endif()
    message("RISCV64_SPACEMIT_IME_SPEC: ${RISCV64_SPACEMIT_IME_SPEC}")
endif()

# global_spine_env_info is provided by ggml-spacemit.cpp and initialized
# from spert::backend_info() and GGML_SPACEMIT_WORKERS.
ggml_add_backend_library(${TARGET_NAME}
    ggml-spacemit.cpp
    ime.cpp
    ime1_kernels.cpp
    ime2_kernels.cpp
    rvv_kernels.cpp
    scalar_kernels.cpp
    repack.cpp
    spe_kernel_dispatch.cpp
    spacemit-context.h
    spacemit-fused.cpp
    spacemit-fused.h
    spacemit-env.h
    spine_mem_pool.cpp
    ../../include/ggml-spacemit.h)

set_target_properties(${TARGET_NAME} PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden)

target_link_libraries(${TARGET_NAME} PRIVATE ${SPERT_LIB})
target_link_directories(${TARGET_NAME} PUBLIC "${SPERT_DIR}/lib")
target_link_options(${TARGET_NAME} PUBLIC "-Wl,-rpath-link,${SPERT_DIR}/lib")

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SPERT_DIR}/include)

target_compile_definitions(${TARGET_NAME} PRIVATE
    ${RISCV64_SPACEMIT_IME_SPEC}
)
if (GGML_BUILD_PROFILE)
    target_compile_definitions(${TARGET_NAME} PRIVATE GGML_BUILD_PROFILE)
endif()

# Pass the full march string to the backend target.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(SPACEMIT_MARCH "-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb")
    if(RISCV64_SPACEMIT_IME_SPEC MATCHES "IME" AND
       ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15) OR
        CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        set(SPACEMIT_MARCH "-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb_xsmtvdotii")
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE ${SPACEMIT_MARCH})
endif()
