message(STATUS "Including the CXL Type 2 device backend")

ggml_add_backend_library(ggml-cxl
    cxl-device.cpp
    cxl-graph-serialize.cpp
    ggml-backend-buffer.cpp
    ggml-backend-buffer-type.cpp
    ggml-backend-device.cpp
    ggml-backend.cpp
    ggml-backend-reg.cpp
    ../../include/ggml-cxl.h
)

# No external dependencies beyond standard Linux headers
