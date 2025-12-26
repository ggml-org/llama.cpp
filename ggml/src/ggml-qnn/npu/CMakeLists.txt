enable_language(ASM)
cmake_policy(SET CMP0115 OLD)

if(DEFINED ENV{HEXAGON_SDK_ROOT})
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
    message("HEXAGON_SDK_ROOT (from environment): ${HEXAGON_SDK_ROOT}")
elseif(DEFINED HEXAGON_SDK_ROOT)
    message("HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}")
else()
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not defined")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(PREBUILT_LIB_DIR "android_aarch64")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(PREBUILT_LIB_DIR "UbuntuARM_aarch64")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    # Windows
    set(PREBUILT_LIB_DIR "windows_aarch64")
endif()

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

# Base Include dirs for the Project
set(common_incs
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/incs/HAP/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)

include_directories(${common_incs})

# host build
file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
file(GLOB host_srcs "${CMAKE_CURRENT_LIST_DIR}/host/*.cpp")
set(stub_srcs "${CMAKE_CURRENT_BINARY_DIR}/npu_device_stub.c")
add_library(hexagon-npu-host STATIC
    ${common_srcs}
    ${host_srcs}
    ${stub_srcs}
)

# disable warnings for the stub
set_source_files_properties(
    ${stub_srcs}
    PROPERTIES
    COMPILE_FLAGS "-w"
)

build_idl(idl/hexagon_npu.idl hexagon-npu-host)

# Add compile definitions to the target
target_compile_definitions(hexagon-npu-host PUBLIC
    VERIFY_PRINT_ERROR
    GGML_QNN_ENABLE_HEXAGON_BACKEND
)

if(GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS)
    target_compile_definitions(hexagon-npu-host PUBLIC
        GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
    )
endif()

target_include_directories(hexagon-npu-host PRIVATE
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
    ${QNN_SDK_ROOT}/include/QNN/
    ${CMAKE_CURRENT_LIST_DIR}/host/
    ${CMAKE_CURRENT_LIST_DIR}/
)

target_include_directories(hexagon-npu-host PUBLIC
    ${HEXAGON_SDK_ROOT}/incs/ # TODO: this is for rpc-mem
)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set_target_properties(hexagon-npu-host PROPERTIES OUTPUT_NAME "hexagon_npu")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux")
    target_link_options(hexagon-npu-host PUBLIC -pie)
endif()

if(GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING)
    message("GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING is enabled")
    target_compile_definitions(hexagon-npu-host PUBLIC GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING)
else()
    message("GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING is disabled")
endif()

link_options(hexagon-npu-host)

choose_dsprpc("3" dsprpc) # cdsprpc
link_custom_library(hexagon-npu-host ${dsprpc})

cmake_host_system_information(RESULT BUILD_CPU_COUNT QUERY NUMBER_OF_PHYSICAL_CORES)

# Build HTP bits
set(HTP_CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/../../ggml-hexagon/htp/cmake-toolchain.cmake
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}
    -DHEXAGON_SDK_ROOT=$ENV{HEXAGON_SDK_ROOT}
    -DHEXAGON_TOOLS_ROOT=$ENV{HEXAGON_TOOLS_ROOT}
    -DHEXAGON_HTP_DEBUG=${GGML_HEXAGON_HTP_DEBUG}
    -DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=${GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE})

ExternalProject_Add(hexagon_npu_skel_v73
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v73 -DPREBUILT_LIB_DIR="toolv19_v73")
ExternalProject_Add(hexagon_npu_skel_v75
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v75 -DPREBUILT_LIB_DIR="toolv19_v75")
ExternalProject_Add(hexagon_npu_skel_v79
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v79 -DPREBUILT_LIB_DIR="toolv19_v79")

list(APPEND NPU_RUNTIME_LIBS "${HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonApp")
list(APPEND NPU_RUNTIME_LIBS "${HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonAppLE")

# Install Hexagon skels required at runtime
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libhexagon_npu_skel_v73.so
    ${CMAKE_CURRENT_BINARY_DIR}/libhexagon_npu_skel_v75.so
    ${CMAKE_CURRENT_BINARY_DIR}/libhexagon_npu_skel_v79.so
    ${HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonApp
    ${HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonAppLE
    TYPE LIB)
