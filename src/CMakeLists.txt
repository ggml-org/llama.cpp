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
            llama-kv-cache-unified.cpp
            llama-kv-cache-unified-iswa.cpp
            llama-kv-cache-recurrent.cpp
            llama-memory.cpp
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
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_NUMA)
    find_library(NUMA_LIB numa)
    check_include_file_cxx("numa.h" HAVE_NUMA_HEADERS)
    check_include_file_cxx("numaif.h" HAVE_NUMAIF_HEADERS)
    if (HAVE_NUMA_HEADERS AND HAVE_NUMAIF_HEADERS AND NUMA_LIB)
        target_compile_definitions(llama PRIVATE USE_LIBNUMA)
        target_link_libraries(llama PRIVATE numa)
        message(STATUS "libnuma found, page cache will be moved to the local node using mbind() syscall. Disable with LLAMA_NUMA=OFF")
    else()
        message(STATUS "Warning: NUMA headers not found - consider disabling this Warning with LLAMA_NUMA=OFF")
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
