# ggml-et: Device kernel build (cross-compiled via ExternalProject)
#
# This is invoked automatically by the parent CMakeLists.txt via ExternalProject_Add
# with the RISC-V toolchain. It can also be built standalone:
#
#   cmake -B build-kernels \
#     -DCMAKE_TOOLCHAIN_FILE=/opt/et/lib/cmake/riscv64-ec-toolchain.cmake \
#     -DCMAKE_PREFIX_PATH=/opt/et \
#     -DCMAKE_MODULE_PATH=/opt/et/lib/cmake/cmake-modules \
#     -DCMAKE_BUILD_TYPE=Release \
#     ggml/src/ggml-et/et-kernels/
#   cmake --build build-kernels

cmake_minimum_required(VERSION 3.17)
project(ggml-et-kernels LANGUAGES C ASM)

include(DeviceUtils)

# Ensure we have the base address parameter
if(NOT DEFINED ADDRESS)
    set(ADDRESS "0x8005801000")
    message(STATUS "ADDRESS not specified, using default: ${ADDRESS}")
endif()

set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/linker.ld)
set(CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_unimplemented_instructions.sh)

find_package(et-common-libs REQUIRED)
find_package(esperantoTrace REQUIRED)

# addr2line is not set by the toolchain file
if(DEFINED TOOLCHAIN_DIR)
    set(CMAKE_ADDR2LINE "${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-addr2line")
else()
    set(CMAKE_ADDR2LINE "/opt/et/bin/riscv64-unknown-elf-addr2line")
endif()

# Track address changes to trigger relinking
set(ADDRESS_FILE ${CMAKE_CURRENT_BINARY_DIR}/et_address.txt)
file(CONFIGURE OUTPUT ${ADDRESS_FILE} CONTENT "${ADDRESS}" @ONLY)

# Define kernel list
set(KERNELS
    el_map_f32
    glu_f32
    mul_mat_f32
    mul_mat_id_f32
    rope_f32
    rms_norm_f32
    softmax_f32
    get_rows_f32
    cont_f32
    cont_f16
    set_rows_f32
    memops
)

foreach(KERNEL ${KERNELS})
    add_riscv_executable(${KERNEL})
    target_sources(${KERNEL}.elf PRIVATE
        src/${KERNEL}.c
        src/crt.S
    )
    target_include_directories(${KERNEL}.elf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    target_link_libraries(${KERNEL}.elf
        PRIVATE
            et-common-libs::cm-umode
            esperantoTrace::et_trace
    )
    
    # C-only flags — must not apply to .S files
    target_compile_options(${KERNEL}.elf PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-fno-zero-initialized-in-bss>
        $<$<COMPILE_LANGUAGE:C>:-ffreestanding>
        $<$<COMPILE_LANGUAGE:C>:-std=c99>
    )
    target_link_options(${KERNEL}.elf PRIVATE
        -Wl,--defsym=BASE_ADDRESS=${ADDRESS}
        -Wl,--entry=_start
    )
    # Append to LINK_DEPENDS (macro already sets it for the linker script)
    set_property(TARGET ${KERNEL}.elf APPEND PROPERTY
        LINK_DEPENDS "${ADDRESS_FILE}"
    )

    # Post-build: strip and check (fails build if check script fails)
    add_custom_command(TARGET ${KERNEL}.elf POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-debug $<TARGET_FILE:${KERNEL}.elf>
        COMMAND ${CHECK_SCRIPT}
            ${CMAKE_OBJDUMP} ${CMAKE_ADDR2LINE} $<TARGET_FILE:${KERNEL}.elf>
        DEPENDS ${CHECK_SCRIPT}
        VERBATIM
    )
endforeach()

# Print summary
message(STATUS "GGML ET Kernels configured:")
foreach(KERNEL ${KERNELS})
    message(STATUS "  - ${KERNEL}")
endforeach()
message(STATUS "Base address: ${ADDRESS}")
