set(TARGET llama-auto-tensor-type)
add_executable(${TARGET} auto-tensor-type.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../common ../../ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
