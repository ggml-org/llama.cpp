set(TARGET llama-app)

add_executable(${TARGET} llama.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama)

target_link_libraries(${TARGET} PRIVATE llama-server-impl llama-cli-impl llama-completion-impl llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
