set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
