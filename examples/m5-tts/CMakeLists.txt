cmake_minimum_required(VERSION 3.18)
project(m5tts CXX)

# Native build for the M5-TTS Android app.
#
#   cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
#         -DM5_ORT_ROOT=<extracted onnxruntime aar> -B build-android
#
# M5_ORT_ROOT must contain headers/onnxruntime_c_api.h and
# jni/<abi>/libonnxruntime.so (both ship inside onnxruntime-android.aar).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# f16 dot product for the depth heads (m5_engine.cpp); the fallback f32 kernel
# compiles in when the target lacks it.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+fp16+fp16fml>)

if(NOT M5_ORT_ROOT)
    set(M5_ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ort" CACHE PATH "ONNX Runtime root (headers + jni/<abi>)")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${M5_ORT_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${M5_ORT_ROOT}/headers")

find_library(ANDROID_LIB android)
find_library(LOG_LIB log)

set(M5_SOURCES m5_engine.cpp m5_tokenizer.cpp)

# JNI library loaded by NativeEngine.java.
add_library(m5engine SHARED ${M5_SOURCES} m5_jni.cpp)
target_include_directories(m5engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(m5engine onnxruntime ${LOG_LIB} ${ANDROID_LIB})

# Command line binary, handy for on-device A/B runs without the app.
add_executable(m5-engine m5_engine_main.cpp ${M5_SOURCES})
target_include_directories(m5-engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(m5-engine onnxruntime ${LOG_LIB} ${ANDROID_LIB})

# OpenAI-style HTTP server (see m5_server.cpp header for endpoints and usage).
add_executable(m5-server m5_server.cpp ${M5_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/cpp-httplib/httplib.cpp)
target_include_directories(m5-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/cpp-httplib)
target_link_libraries(m5-server onnxruntime ${LOG_LIB} ${ANDROID_LIB})

# Transport benchmark; ring-stub impersonates a stage process.
add_executable(ring-stub ring_stub.cpp)
target_include_directories(ring-stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(ring-bench ring_bench.cpp)
target_include_directories(ring-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
