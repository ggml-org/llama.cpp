set(TARGET llama-expected-attention)
add_executable(${TARGET} expected-attention.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(llama-expected-attention PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
