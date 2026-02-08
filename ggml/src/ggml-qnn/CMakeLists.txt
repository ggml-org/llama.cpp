message(STATUS "Using QNN backend")

option(GGML_HEXAGON_NPU_ONLY "ggml-qnn: Only use Hexagon NPU" OFF)
option(GGML_QNN_ENABLE_HEXAGON_BACKEND "ggml-qnn: Enable Hexagon custom package" ${GGML_HEXAGON_NPU_ONLY})
option(GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS "ggml-qnn: Enable quantized tensors support" OFF)
option(GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING "ggml-qnn: Enable performance tracking" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(LOG_LIB log)
    set(COMMON_LINK_LIBRARIES ${LOG_LIB})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message("Building for Linux or Windows")
else()
    message(FATAL_ERROR "QNN now only available on Android, Windows and Linux")
endif()

if(NOT DEFINED GGML_QNN_SDK_PATH)
    # try read from environment variable
    # TODO: create a function to search for the SDK path
    if(DEFINED ENV{QNN_SDK_PATH})
        set(GGML_QNN_SDK_PATH $ENV{QNN_SDK_PATH})
    elseif(DEFINED ENV{QNN_SDK_ROOT})
        message("found QNN_SDK_ROOT: ${QNN_SDK_ROOT}")
        set(GGML_QNN_SDK_PATH $ENV{QNN_SDK_ROOT})
    else()
        message(FATAL_ERROR "GGML_QNN_SDK_PATH not defined")
    endif()
endif()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("QNN_SDK_PATH: ${GGML_QNN_SDK_PATH}")

message("GGML_QNN: ${GGML_QNN}")
message("GGML_QNN_ENABLE_HEXAGON_BACKEND: ${GGML_QNN_ENABLE_HEXAGON_BACKEND}")
message("GGML_HEXAGON_NPU_ONLY: ${GGML_HEXAGON_NPU_ONLY}")
message("GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS: ${GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS}")
message("GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING: ${GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING}")

ggml_add_backend_library(ggml-qnn
    ../../include/ggml-qnn.h
)
target_link_libraries(ggml-qnn PRIVATE ${COMMON_LINK_LIBRARIES})

add_subdirectory(shared)

if(GGML_HEXAGON_NPU_ONLY)
    message("GGML_HEXAGON_NPU_ONLY is enabled")
    set(GGML_QNN_ENABLE_HEXAGON_BACKEND ON)
else()
    message("GGML_HEXAGON_NPU_ONLY is disabled")
    add_subdirectory(qnn)
    target_link_libraries(runtime-common PUBLIC qnn-backend)
endif()

if(GGML_QNN_ENABLE_HEXAGON_BACKEND)
    message("GGML_QNN_ENABLE_HEXAGON_BACKEND is enabled")
    add_subdirectory(npu)
    target_link_libraries(hexagon-npu-host runtime-common)
    target_link_libraries(ggml-qnn PRIVATE hexagon-npu-host)
else()
    message("GGML_QNN_ENABLE_HEXAGON_BACKEND is disabled")
    target_link_libraries(ggml-qnn PRIVATE runtime-common)
endif()

# Copy dynamic libraries
set(BACKEND_RUNTIME_LIBS "")

if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        # Android
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/aarch64-android")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        # Linux x86_64
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/x86_64-linux-clang")
    else()
        # Linux aarch64
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/aarch64-oe-linux-gcc11.2")
    endif()

    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/libQnnSystem.so")
    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/libQnnCpu.so")
    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/libQnnGpu.so")
    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/libQnnHtp.so")
    file(GLOB HTP_STUB_LIBS "${QNN_SDK_LIB_PATH}/libQnnHtp*.so")
    list(APPEND BACKEND_RUNTIME_LIBS ${HTP_STUB_LIBS})

    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        file(GLOB HTP_SKEL_LIBS "${GGML_QNN_SDK_PATH}/lib/hexagon-*/unsigned/libQnnHtp*Skel.so")
        list(APPEND BACKEND_RUNTIME_LIBS ${HTP_SKEL_LIBS})

        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            if(EXISTS "${CMAKE_ANDROID_NDK}/prebuilt/android-arm64/gdbserver/gdbserver")
                list(APPEND BACKEND_RUNTIME_LIBS "${CMAKE_ANDROID_NDK}/prebuilt/android-arm64/gdbserver/gdbserver")
                message("old ndk, copy gdbserver")
            else()
                file(GLOB LLDB_SERVER "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/*/lib/linux/aarch64/lldb-server")
                list(APPEND BACKEND_RUNTIME_LIBS ${LLDB_SERVER})
                message("new ndk, copy lldb-server")
            endif()

            file(GLOB OMP_LIBS "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/*/lib/linux/aarch64/libomp.so")
            file(GLOB ASAN_LIBS "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/*/lib/linux/libclang_rt.asan-aarch64-android.so")
            list(APPEND BACKEND_RUNTIME_LIBS ${OMP_LIBS})
            list(APPEND BACKEND_RUNTIME_LIBS ${ASAN_LIBS})
        endif()
    else()
        # Linux
        list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/libHtpPrepare.so")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        # x86_64
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/x86_64-windows-msvc")
    else()
        # aarch64
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/aarch64-windows-msvc")
    endif()

    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/QnnSystem.dll")
    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/QnnCpu.dll")
    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/QnnGpu.dll")
    list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/QnnHtp.dll")
    file(GLOB HTP_STUB_LIBS "${QNN_SDK_LIB_PATH}/QnnHtp*.dll")

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        list(APPEND BACKEND_RUNTIME_LIBS "${QNN_SDK_LIB_PATH}/HtpPrepare.dll")
    endif()

    list(APPEND BACKEND_RUNTIME_LIBS ${HTP_STUB_LIBS})
endif()

foreach(RUNTIME_LIB ${BACKEND_RUNTIME_LIBS})
    message("Copy: ${RUNTIME_LIB} -> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    add_custom_command(
        TARGET ggml-qnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${RUNTIME_LIB}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()
