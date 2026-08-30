# Optional model-specific speculative HIP sidecars.
# These are not part of the default llama.cpp build: they are hard-coded to
# validated model/artifact contracts and must be explicitly enabled.

if (NOT GGML_HIP)
    message(FATAL_ERROR "LLAMA_BUILD_SPEC_SIDECARS requires GGML_HIP=ON")
endif()

if (WIN32)
    message(FATAL_ERROR "speculative sidecar CMake builds are currently supported on Linux; use a HIP build script on Windows")
endif()

find_package(hip REQUIRED)
if (NOT CMAKE_CXX_COMPILER MATCHES "hipcc(\\.bat)?$")
    enable_language(HIP)
endif()

set(LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES "" CACHE STRING
    "HIP architectures for speculative sidecars (empty inherits CMAKE_HIP_ARCHITECTURES)")

function(spec_sidecar_add_sidecar target output)
    add_library(${target} SHARED ${ARGN})
    # Mirror ggml's HIP compiler split.  ROCm's hipcc wrapper is used as a
    # CXX compiler on this tree; native CMake HIP mode is only used when the
    # caller has explicitly enabled it.
    if (CMAKE_CXX_COMPILER MATCHES "hipcc(\\.bat)?$")
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CXX)
        set(_spec_sidecar_hip_link hip::device)
    else()
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
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
    set(_spec_arches "${LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES}")
    if (LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES)
        set_property(TARGET ${target} PROPERTY HIP_ARCHITECTURES
            "${LLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES}")
    elseif (CMAKE_HIP_ARCHITECTURES)
        set(_spec_arches "${CMAKE_HIP_ARCHITECTURES}")
    endif()
    # HIP exposes __gfx1030__ only in the device compilation pass. The host
    # pass also needs a matching constant to select the corresponding kernel
    # launch; define it only for an exact single-architecture sidecar build,
    # never for a multi-arch/fallback target.
    if (_spec_arches STREQUAL "gfx1030")
        target_compile_definitions(${target} PRIVATE SPEC_SIDECAR_GFX1030=1)
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

spec_sidecar_add_sidecar(
    spec-sidecar-hip-qwen35moe-mtp
    spec_qwen35moe_mtp_sidecar
    ${PROJECT_SOURCE_DIR}/src/spec_sidecar/qwen35moe_mtp/sidecar.hip)

spec_sidecar_add_sidecar(
    spec-sidecar-hip-qwen4exp-mtp
    spec_qwen4exp_mtp_sidecar
    ${PROJECT_SOURCE_DIR}/src/spec_sidecar/qwen4exp_mtp/sidecar.hip
    ${PROJECT_SOURCE_DIR}/src/spec_sidecar/qwen4exp_mtp/native_attention.hip)
target_compile_definitions(spec-sidecar-hip-qwen4exp-mtp PRIVATE
    QWEN4EXP_SIDECAR_PRODUCTION=1
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    GGML_SCHED_MAX_COPIES=4)
target_include_directories(spec-sidecar-hip-qwen4exp-mtp PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
