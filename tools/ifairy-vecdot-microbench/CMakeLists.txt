set(TARGET ifairy-vecdot-microbench)
add_executable(${TARGET} ifairy-vecdot-microbench.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

