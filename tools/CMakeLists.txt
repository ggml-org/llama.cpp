# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # scripts/prune-to-lib.sh deletes some of these directories to produce the
    # library-only SDK tree; guard each prunable one with EXISTS so a pruned
    # configure doesn't hard-fail on a missing add_subdirectory target.
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batched-bench)
        add_subdirectory(batched-bench)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gguf-split)
        add_subdirectory(gguf-split)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/imatrix)
        add_subdirectory(imatrix)
    endif()
    add_subdirectory(llama-bench)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/completion)
        add_subdirectory(completion)
    endif()
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ui)
            add_subdirectory(ui)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli)
            add_subdirectory(cli)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/server)
            add_subdirectory(server)
        endif()
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tokenize)
        add_subdirectory(tokenize)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tts)
        add_subdirectory(tts)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mtmd)
        add_subdirectory(mtmd)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cvector-generator)
            add_subdirectory(cvector-generator)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/export-lora)
            add_subdirectory(export-lora)
        endif()
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fit-params)
        add_subdirectory(fit-params)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/results)
        add_subdirectory(results)
    endif()
endif()
