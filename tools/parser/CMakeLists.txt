if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # this tool is disabled on Windows when building with shared libraries because it uses internal functions not exported with LLAMA_API
    set(TARGET llama-debug-template-parser)
    add_executable(${TARGET} debug-template-parser.cpp)
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
endif()

set(TARGET llama-template-analysis)
add_executable(${TARGET} template-analysis.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
