set(TARGET rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${TARGET} PRIVATE seccomp)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
