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
    mul_mat_f32
    rope_f32
    rms_norm_f32
    memops
)

# Function to create bare metal kernel
function(create_bare_metal_kernel KERNEL_NAME)
    # Release version
    add_custom_target(${KERNEL_NAME}.elf ALL
        COMMAND ${CMAKE_C_COMPILER}
            -march=rv64imf
            -mabi=lp64f
            -mcmodel=medany
            -fno-zero-initialized-in-bss
            -ffunction-sections
            -fdata-sections
            -O3
            -g0
            -nostdlib
            -nostartfiles
            -ffreestanding
            -std=c99
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -T ${CMAKE_CURRENT_SOURCE_DIR}/src/linker.ld
            -Wl,--defsym=BASE_ADDRESS=${ADDRESS}
            -Wl,--entry=_start
            -Wl,--gc-sections
            -o ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.elf
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${KERNEL_NAME}.c
        COMMAND ${CMAKE_STRIP} --strip-debug ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.elf
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_unimplemented_instructions.sh ${CMAKE_OBJDUMP} ${CMAKE_ADDR2LINE} ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.elf
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building bare metal kernel: ${KERNEL_NAME}"
        VERBATIM
    )

    # Install the kernel
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.elf
        DESTINATION kernels
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

# Print summary
message(STATUS "GGML Bare Metal Kernels configured:")
foreach(KERNEL ${KERNELS})
    message(STATUS "  - ${KERNEL}")
endforeach()
message(STATUS "Base address: ${ADDRESS}")
