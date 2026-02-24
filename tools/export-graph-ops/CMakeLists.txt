set(TARGET llama-export-graph-ops)
add_executable(${TARGET} export-graph-ops.cpp)
target_include_directories(${TARGET} PRIVATE ../../src)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()