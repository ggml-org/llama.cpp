set(TARGET rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
