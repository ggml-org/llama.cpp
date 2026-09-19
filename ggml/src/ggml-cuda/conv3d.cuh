#ifndef GGML_CUDA_CONV3D_CUH
#define GGML_CUDA_CONV3D_CUH

#include "common.cuh"

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif
