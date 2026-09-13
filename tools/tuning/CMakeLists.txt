set(TARGET ggml-metal-tuning)

add_executable(${TARGET} main.cpp bench.cpp fa-vec.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src/ggml-metal)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
