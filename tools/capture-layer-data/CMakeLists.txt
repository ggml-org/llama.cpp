set(TARGET llama-capture-layer-data)
add_executable(${TARGET} capture-layer-data.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
