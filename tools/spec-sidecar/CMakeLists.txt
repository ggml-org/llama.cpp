# Optional model-specific speculative sidecar HIP sidecars.
# These are not part of the default llama.cpp build: they are hard-coded to
# the Qwen3.8-27B MTP/DFlash artifact contracts and must be explicitly enabled.

if (NOT GGML_HIP)
    message(FATAL_ERROR "LLAMA_BUILD_SPEC_SIDECARS requires GGML_HIP=ON")
endif()

if (WIN32)
    message(FATAL_ERROR "speculative sidecar CMake builds are currently supported on Linux; use a HIP build script on Windows")
endif()

find_package(hip REQUIRED)

set(LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES "" CACHE STRING
    "HIP architectures for speculative sidecars (empty inherits CMAKE_HIP_ARCHITECTURES)")

function(spec_sidecar_add_sidecar target output source)
    add_library(${target} SHARED ${source})
    # Mirror ggml's HIP compiler split.  ROCm's hipcc wrapper is used as a
    # CXX compiler on this tree; native CMake HIP mode is only used when the
    # caller has explicitly enabled it.
    if (CMAKE_CXX_COMPILER MATCHES "hipcc(\\.bat)?$")
        set_source_files_properties(${source} PROPERTIES LANGUAGE CXX)
        set(_spec_sidecar_hip_link hip::device)
    else()
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
        set(_spec_sidecar_hip_link hip::host)
    endif()
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/spec_sidecar)
    target_link_libraries(${target} PRIVATE ${_spec_sidecar_hip_link})
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${output}"
        POSITION_INDEPENDENT_CODE ON)
    if (LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES)
        set_property(TARGET ${target} PROPERTY HIP_ARCHITECTURES
            "${LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES}")
    endif()
    if (LLAMA_TOOLS_INSTALL)
        install(TARGETS ${target} LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
    endif()
endfunction()

spec_sidecar_add_sidecar(
    spec-sidecar-hip-mtp
    spec_hip_sidecar
    ${PROJECT_SOURCE_DIR}/src/spec_sidecar/mtp/sidecar.hip)

spec_sidecar_add_sidecar(
    spec-sidecar-hip-dflash
    spec_dflash_sidecar
    ${PROJECT_SOURCE_DIR}/src/spec_sidecar/dflash/dflash_sidecar.hip)
