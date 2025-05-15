# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(cvector-generator)
    add_subdirectory(baby-llama)
    add_subdirectory(batched-bench)
    add_subdirectory(batched)
    add_subdirectory(convert-llama2c-to-ggml)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)
    add_subdirectory(export-lora)
    add_subdirectory(gbnf-validator)
    add_subdirectory(gguf-hash)
    add_subdirectory(gguf-split)
    add_subdirectory(gguf)
    add_subdirectory(gritlm)
    add_subdirectory(imatrix)
    add_subdirectory(infill)
    add_subdirectory(llama-bench)
    add_subdirectory(llava)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(main)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(perplexity)
    add_subdirectory(quantize-stats)
    add_subdirectory(quantize)
    add_subdirectory(retrieval)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (LLAMA_BUILD_SERVER)
    add_subdirectory(server)
    endif()
    if (GGML_SYCL)
        add_subdirectory(sycl)
    endif()
    add_subdirectory(save-load-state)
    add_subdirectory(simple)
    add_subdirectory(speculative)
    add_subdirectory(tokenize)
endif()
