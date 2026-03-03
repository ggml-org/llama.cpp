# ggml-et: Device kernels (cross-compiled within the main build)
#
# The RISC-V toolchain is set up in-scope so these targets use the
# cross-compiler while the rest of the build uses the host compiler.
# This keeps kernels in compile_commands.json for full IDE support.

# --- RISC-V toolchain setup (scoped to this directory) ---
set(TOOLCHAIN_DIR ${ET_PLATFORM_PATH})
include(${ET_PLATFORM_PATH}/lib/cmake/riscv64-ec-toolchain.cmake)
set(CMAKE_ADDR2LINE "${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-addr2line")
set(CMAKE_LINKER_TYPE LLD)

message(STATUS "ET kernels using RISC-V toolchain at: ${TOOLCHAIN_DIR}")

# DeviceUtils provides the add_riscv_executable macro
list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_PATH}/lib/cmake/cmake-modules")
list(APPEND CMAKE_PREFIX_PATH "${ET_PLATFORM_PATH}/lib/cmake")
include(DeviceUtils)

find_package(et-common-libs REQUIRED)
find_package(esperantoTrace REQUIRED)

# --- Kernel configuration ---
if(NOT DEFINED ADDRESS)
    set(ADDRESS "0x8005801000")
    message(STATUS "ADDRESS not specified, using default: ${ADDRESS}")
endif()

set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/linker.ld)
set(CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_unimplemented_instructions.sh)

# Track address changes to trigger relinking
set(ADDRESS_FILE ${CMAKE_CURRENT_BINARY_DIR}/et_address.txt)
file(CONFIGURE OUTPUT ${ADDRESS_FILE} CONTENT "${ADDRESS}" @ONLY)

# --- Kernel list ---
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
    target_link_libraries(${KERNEL}.elf PRIVATE et-common-libs::cm-umode)
    # C-only flags — must not apply to .S files
    target_compile_options(${KERNEL}.elf PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-fno-zero-initialized-in-bss>
        $<$<COMPILE_LANGUAGE:C>:-ffreestanding>
        $<$<COMPILE_LANGUAGE:C>:-std=c99>
        $<$<COMPILE_LANGUAGE:C>:-ffat-lto-objects>
        $<$<COMPILE_LANGUAGE:C>:-mcmodel=medany>
        $<$<COMPILE_LANGUAGE:C>:-mabi=lp64f>
        $<$<COMPILE_LANGUAGE:C>:-march=rv64imf>
        $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
        $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
        $<$<COMPILE_LANGUAGE:C>:-O3>
        $<$<COMPILE_LANGUAGE:C>:-g0>
        $<$<COMPILE_LANGUAGE:C>:-nostdlib>
        $<$<COMPILE_LANGUAGE:C>:-ffreestanding>
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

# --- Embed kernels into C++ source for the host library ---
find_package(Python3 REQUIRED)

set(EMBED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_kernels.py)
set(EMBED_HPP ${CMAKE_CURRENT_BINARY_DIR}/ggml-et-kernels-embed.hpp)
set(EMBED_CPP ${CMAKE_CURRENT_BINARY_DIR}/ggml-et-kernels-embed.cpp)

set(KERNEL_ELF_TARGETS)
foreach(KERNEL ${KERNELS})
    list(APPEND KERNEL_ELF_TARGETS ${KERNEL}.elf)
endforeach()

add_custom_command(
    OUTPUT ${EMBED_HPP} ${EMBED_CPP}
    COMMAND ${Python3_EXECUTABLE} ${EMBED_SCRIPT}
        --input-dir ${CMAKE_CURRENT_BINARY_DIR}
        --output-hpp ${EMBED_HPP}
        --output-cpp ${EMBED_CPP}
        --kernels ${KERNELS}
    DEPENDS ${KERNEL_ELF_TARGETS} ${EMBED_SCRIPT}
    COMMENT "Embedding ET kernels into source code"
    VERBATIM
)

add_custom_target(et-kernels-embed ALL
    DEPENDS ${EMBED_HPP} ${EMBED_CPP}
)

# Print summary
message(STATUS "GGML ET Kernels configured:")
foreach(KERNEL ${KERNELS})
    message(STATUS "  - ${KERNEL}")
endforeach()
message(STATUS "Base address: ${ADDRESS}")
message(STATUS "Embedded sources: ${EMBED_HPP}, ${EMBED_CPP}")
