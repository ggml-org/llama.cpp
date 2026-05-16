set(TARGET llama-mimov2asr-cli)
add_executable(${TARGET} mimov2asr-cli.cpp)

# 链接 llama-common (它会自动帮你链接 llama 核心库和 ggml)
target_link_libraries(${TARGET} PUBLIC llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()