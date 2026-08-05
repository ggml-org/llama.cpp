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
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()

    # tessera-cli: lightweight CLI shim for the Tessera Studio Swift app.
    # Same engine modules as llama-quantize-impl; thin subcommand wrappers
    # around ts_dispatch_run / ts_awq_evolve / ts_sidecar_v3_read. Built
    # unconditionally (does not require the server). Source lives in
    # tools/cli/ next to llama-cli's own sources; the two targets coexist
    # without conflict because tools/cli/CMakeLists.txt only defines
    # llama-cli and is only included when LLAMA_BUILD_SERVER=ON.
    set(TARGET tessera-cli)
    add_executable(${TARGET} cli/tessera-cli.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
endif()
