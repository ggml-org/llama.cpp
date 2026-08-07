message(STATUS "Using FakeGPU backend")

ggml_add_backend_library(ggml-fakegpu
                         ggml-fakegpu.cpp)

target_compile_definitions(ggml-fakegpu PRIVATE GGML_USE_FAKEGPU)
