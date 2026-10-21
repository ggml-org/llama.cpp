#pragma once

#include "ggml-backend.h"

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor ** tensors, void * data, const size_t * offsets,
        const size_t * sizes, const size_t * data_offsets, size_t n_tensors);
