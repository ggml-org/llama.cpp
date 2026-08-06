# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

# Optional measured-region control for disk-safe rocprofv3 target-only profiles.
# Non-HIP builds and ROCm installations without the SDK marker package are unchanged.
if(GGML_HIP AND UNIX)
    find_package(rocprofiler-sdk-roctx QUIET)
    if(TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
        target_link_libraries(${TARGET} PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
        target_compile_definitions(${TARGET} PRIVATE LLAMA_BENCH_ROCTX_PROFILE=1)
    endif()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
