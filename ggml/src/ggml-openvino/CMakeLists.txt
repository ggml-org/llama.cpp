find_package(OpenVINO REQUIRED)
list(APPEND GGML_EXTRA_LIBS_PRIVATE openvino::runtime)

# Set header and libs
file(GLOB GGML_HEADERS_OPENVINO "ggml-openvino/*.h")
list(APPEND GGML_HEADERS_OPENVINO "../include/ggml-openvino.h")
file(GLOB GGML_SOURCES_OPENVINO "ggml-openvino/*.cpp")
list(APPEND GGML_SOURCES_OPENVINO "ggml-openvino.cpp")

list(APPEND GGML_CDEF_PUBLIC GGML_USE_OPENVINO)

if (OPENVINO_DEVICE)
    if (OPENVINO_DEVICE STREQUAL "GPU")
        add_compile_definitions(GGML_OPENVINO_GPU)
    elseif (OPENVINO_DEVICE STREQUAL "NPU")
        add_compile_definitions(GGML_OPENVINO_NPU)
    endif()
endif()

if(NOT DEFINED GGML_OV_FRONTEND)
    set(GGML_OV_FRONTEND OpenVINO_DIR)
endif()
add_definitions(-DGGML_OV_FRONTEND="${GGML_OV_FRONTEND}")

if (OpenVINO_DIR)
    if (GGML_OPENVINO)
        if (NOT UNIX)
            set(GGML_OPENVINO OFF)
            message(WARNING "OpenVINO: OpenVINO toolkit supports unix but not ${CMAKE_SYSTEM_NAME}. Turning off GGML_OPENVINO")
        endif()
    endif()

    if (GGML_OPENVINO)
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
        else()
            set(GGML_OPENVINO OFF)
            message(WARNING "OpenVINO: OpenVINO toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}. Turning off GGML_OPENVINO")
        endif()
    endif()

endif()
