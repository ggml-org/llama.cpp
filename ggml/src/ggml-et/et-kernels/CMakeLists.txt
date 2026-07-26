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

# Ensure kernels are built in this directory even if a global output directory is set
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

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

# KERNELS defined in upper CMakeLists.txt
foreach(KERNEL ${KERNELS})
    add_riscv_executable(${KERNEL})
    target_sources(${KERNEL}.elf PRIVATE
        src/${KERNEL}.c
        src/crt.S
    )
    target_include_directories(${KERNEL}.elf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/ggml/src
    )
    target_link_libraries(${KERNEL}.elf PRIVATE et-common-libs::cm-umode)
    # C-only flags — must not apply to .S files
    target_compile_options(${KERNEL}.elf PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-fno-zero-initialized-in-bss>
        $<$<COMPILE_LANGUAGE:C>:-ffreestanding>
        $<$<COMPILE_LANGUAGE:C>:-std=gnu99>
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

add_dependencies(uberkernel.elf et-uberkernel-map)

# Each supported kernel is compiled in its own translation unit with
#   -Dentry_point=<kernel>_entry
# so symbols and macros don't leak between kernels. The dispatcher
# (uberkernel.c) calls the renamed entries via extern declarations.
#
# HACK: we need to supresse _me kernels from setting up SCP themselves
set(_UBER_ME_KERNELS mul_mat_f16_matrix_engine mul_mat_f32_matrix_engine flash_attn_ext_f16_me)

foreach(UK_KERNEL ${UBERKERNEL_SUPPORTED_KERNELS})
    set(_obj uber_${UK_KERNEL})
    add_library(${_obj} OBJECT src/${UK_KERNEL}.c)
    target_compile_definitions(${_obj} PRIVATE "entry_point=${UK_KERNEL}_entry" ET_UBERKERNEL)
    target_include_directories(${_obj} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/ggml/src
    )
    target_link_libraries(${_obj} PRIVATE et-common-libs::cm-umode)
    target_compile_options(${_obj} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-fno-zero-initialized-in-bss>
        $<$<COMPILE_LANGUAGE:C>:-ffreestanding>
        $<$<COMPILE_LANGUAGE:C>:-std=gnu99>
        $<$<COMPILE_LANGUAGE:C>:-ffat-lto-objects>
        $<$<COMPILE_LANGUAGE:C>:-mcmodel=medany>
        $<$<COMPILE_LANGUAGE:C>:-mabi=lp64f>
        $<$<COMPILE_LANGUAGE:C>:-march=rv64imf>
        $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
        $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
        $<$<COMPILE_LANGUAGE:C>:-O3>
        $<$<COMPILE_LANGUAGE:C>:-g0>
        $<$<COMPILE_LANGUAGE:C>:-nostdlib>
    )
    # ME kernels: suppress setup_cache_scp() (called once by the dispatcher)
    if(UK_KERNEL IN_LIST _UBER_ME_KERNELS)
        target_compile_definitions(${_obj} PRIVATE UBERKERNEL_SUPPRESS_SCP_SETUP)
    endif()
    target_sources(uberkernel.elf PRIVATE $<TARGET_OBJECTS:${_obj}>)
endforeach()

# Print summary
message(STATUS "GGML ET Kernels configured:")
foreach(KERNEL ${KERNELS})
    message(STATUS "  - ${KERNEL}")
endforeach()
message(STATUS "Base address: ${ADDRESS}")
