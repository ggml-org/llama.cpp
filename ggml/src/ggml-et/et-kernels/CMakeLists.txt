cmake_minimum_required(VERSION 3.16)
project(ggml-kernels C)

# Configure RISC-V toolchain
if(DEFINED ENV{ET_TOOLCHAIN})
    set(TOOLCHAIN_ROOT $ENV{ET_TOOLCHAIN})
elseif(DEFINED ENV{TOOLCHAIN_ROOT})
    set(TOOLCHAIN_ROOT $ENV{TOOLCHAIN_ROOT})
else()
    set(TOOLCHAIN_ROOT /opt/et)
endif()

set(RISCV_PREFIX ${TOOLCHAIN_ROOT}/bin/riscv64-unknown-elf-)
set(CMAKE_C_COMPILER ${RISCV_PREFIX}gcc)
set(CMAKE_STRIP ${RISCV_PREFIX}strip)
set(CMAKE_OBJDUMP ${RISCV_PREFIX}objdump)
set(CMAKE_ADDR2LINE ${RISCV_PREFIX}addr2line)

message(STATUS "Using RISC-V toolchain at: ${TOOLCHAIN_ROOT}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")

# Ensure we have the address parameter
if(NOT DEFINED ADDRESS)
    set(ADDRESS "0x8005801000")
    message(STATUS "ADDRESS not specified, using default: ${ADDRESS}")
endif()

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

set(ADDRESS_FILE ${CMAKE_CURRENT_BINARY_DIR}/et_address.txt)
file(CONFIGURE OUTPUT ${ADDRESS_FILE} CONTENT "${ADDRESS}" @ONLY)

# Function to create bare metal kernel
function(create_bare_metal_kernel KERNEL_NAME)
    set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/${KERNEL_NAME}.c)
    set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/linker.ld)
    set(CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_unimplemented_instructions.sh)

    add_executable(${KERNEL_NAME}.elf ${KERNEL_SRC})

    target_compile_options(${KERNEL_NAME}.elf PRIVATE
        -march=rv64imf
        -mabi=lp64f
        -mcmodel=medany
        -fno-zero-initialized-in-bss
        -ffunction-sections
        -fdata-sections
        -O3
        -g0
        -nostdlib
        -ffreestanding
        -std=c99
    )

    set_target_properties(${KERNEL_NAME}.elf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        LINK_DEPENDS "${LINKER_SCRIPT};${ADDRESS_FILE}"
    )

    target_include_directories(${KERNEL_NAME}.elf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_options(${KERNEL_NAME}.elf PRIVATE
        -nostartfiles
        -T ${LINKER_SCRIPT}
        -Wl,--defsym=BASE_ADDRESS=${ADDRESS}
        -Wl,--entry=_start
        -Wl,--gc-sections
    )

    # Track linker script AND address file changes
    set_target_properties(${KERNEL_NAME}.elf PROPERTIES
        LINK_DEPENDS "${LINKER_SCRIPT};${ADDRESS_FILE}"
    )

    # Post-build: strip and check (fails build if check script fails)
    add_custom_command(TARGET ${KERNEL_NAME}.elf POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-debug $<TARGET_FILE:${KERNEL_NAME}.elf>
        COMMAND ${CHECK_SCRIPT}
            ${CMAKE_OBJDUMP} ${CMAKE_ADDR2LINE} $<TARGET_FILE:${KERNEL_NAME}.elf>
        # Make check script a dependency so changes to it trigger re-check
        DEPENDS ${CHECK_SCRIPT}
        VERBATIM
    )
endfunction()

# Create all kernels
foreach(KERNEL ${KERNELS})
    create_bare_metal_kernel(${KERNEL})
endforeach()

# Create a single target that depends on all kernels
add_custom_target(et-kernels-all ALL)
foreach(KERNEL ${KERNELS})
    add_dependencies(et-kernels-all ${KERNEL}.elf)
endforeach()

# Find Python3 for embedding kernels
find_package(Python3 REQUIRED)

# Generate embedded kernel source files
set(EMBED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_kernels.py)
set(EMBED_HPP ${CMAKE_CURRENT_BINARY_DIR}/ggml-et-kernels-embed.hpp)
set(EMBED_CPP ${CMAKE_CURRENT_BINARY_DIR}/ggml-et-kernels-embed.cpp)

# Build list of kernel ELF file dependencies
set(KERNEL_ELF_FILES)
foreach(KERNEL ${KERNELS})
    list(APPEND KERNEL_ELF_FILES ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.elf)
endforeach()

add_custom_command(
    OUTPUT ${EMBED_HPP} ${EMBED_CPP}
    COMMAND ${Python3_EXECUTABLE} ${EMBED_SCRIPT}
        --input-dir ${CMAKE_CURRENT_BINARY_DIR}
        --output-hpp ${EMBED_HPP}
        --output-cpp ${EMBED_CPP}
        --kernels ${KERNELS}
    DEPENDS ${KERNEL_ELF_FILES} ${EMBED_SCRIPT}
    COMMENT "Embedding ET kernels into source code"
    VERBATIM
)

# Create target for embedded sources
add_custom_target(et-kernels-embed ALL
    DEPENDS ${EMBED_HPP} ${EMBED_CPP}
)
add_dependencies(et-kernels-embed et-kernels-all)

# Note: Embedded sources are build-time artifacts only
# They are included in ggml-et backend compilation but not installed separately

# Print summary
message(STATUS "GGML Bare Metal Kernels configured:")
foreach(KERNEL ${KERNELS})
    message(STATUS "  - ${KERNEL}")
endforeach()
message(STATUS "Base address: ${ADDRESS}")
message(STATUS "Embedded sources: ${EMBED_HPP}, ${EMBED_CPP}")
