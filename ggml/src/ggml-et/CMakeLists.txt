message(STATUS "Using ET backend")

ggml_add_backend_library(ggml-et
                         ggml-et.cpp
                        )