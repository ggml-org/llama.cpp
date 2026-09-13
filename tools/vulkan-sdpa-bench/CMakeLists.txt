set(TARGET vulkan-sdpa-bench)

add_executable(
    ${TARGET}
    vulkan-sdpa-bench.cpp
)

target_link_libraries(
    ${TARGET}
    PRIVATE
    ggml
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(
    ${TARGET}
    PRIVATE
    cxx_std_17
)

if (LLAMA_TOOLS_INSTALL)
    install(
        TARGETS ${TARGET}
        RUNTIME
    )
endif()