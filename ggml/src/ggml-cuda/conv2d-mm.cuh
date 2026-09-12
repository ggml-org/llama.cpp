#include "common.cuh"

#define CONV_SHAPE_128x128 0
#define CONV_SHAPE_64x32 1
#define CONV_SHAPE_32x256 2

#define NUM_VARIANTS 3

void ggml_cuda_op_conv2d_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
