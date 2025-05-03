# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(main)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
    add_subdirectory(run)
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
        add_subdirectory(llava)
        if (GGML_RPC)
            add_subdirectory(rpc)
        endif()
    endif()
endif()
