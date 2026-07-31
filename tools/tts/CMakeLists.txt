set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET llama-tts-qwen3)
add_executable(${TARGET} tts-qwen3.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama mtmd llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
