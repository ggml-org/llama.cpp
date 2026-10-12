#pragma once

#include "common.cuh"

struct ggml_backend_copy_engine_i;

// the copy engine interface of this backend (ggml-copy-engine.h)
const ggml_backend_copy_engine_i * ggml_cuda_copy_engine_get_iface(void);

// CUDA device index of a device of this backend (ggml-cuda.cu)
int ggml_cuda_dev_index(ggml_backend_dev_t dev);
