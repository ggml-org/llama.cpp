set(TARGET llama-hs-extract)
add_executable(${TARGET} hs-extract.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
# Resolve the CUDA backend registration symbol (ggml_backend_cuda_reg) that
# libggml references when GGML_BACKEND_DL=OFF. The backend is a separate library
# that the executable must link explicitly. Guard so CPU-only builds still work.
if (GGML_CUDA)
    target_link_libraries(${TARGET} PRIVATE ggml-cuda)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
