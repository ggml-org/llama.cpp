message(STATUS "Using ET backend")

find_package(runtime REQUIRED)

ggml_add_backend_library(ggml-et
                         ggml-et.cpp
                         ggml-et-kernels.cpp
                         ggml-et-ops.cpp
                        )

target_link_libraries(ggml-et PRIVATE runtime::etrt_static deviceLayer::deviceLayer)
