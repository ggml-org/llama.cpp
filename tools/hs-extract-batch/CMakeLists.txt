set(TARGET llama-hs-extract-batch)
add_executable(${TARGET} hs-extract-batch.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
# Resolve the CUDA backend registration symbol (ggml_backend_cuda_reg) that
# libggml references when GGML_BACKEND_DL=OFF. The backend is a separate library
# that the executable must link explicitly. Guard so CPU-only builds still work.
if (GGML_CUDA)
    target_link_libraries(${TARGET} PRIVATE ggml-cuda)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
