llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            profiler/llama_perfetto.cpp
            llama_vulkan_prof.cpp
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

#
# Perfetto SDK (amalgamated) integration
#
# The SDK is vendored under `src/profiler` as `perfetto.cc/.h`.
# Build it into a static lib and link it so tracepoints are available across platforms
# (macOS, Linux, Android; incl. Vulkan/Mali).
add_library(perfetto_sdk STATIC profiler/perfetto.cc)

target_include_directories(perfetto_sdk PUBLIC ${CMAKE_SOURCE_DIR}/src/profiler)
target_compile_features(perfetto_sdk PRIVATE cxx_std_17)

# Link Perfetto into the core library for all builds
target_link_libraries(llama PRIVATE perfetto_sdk)
