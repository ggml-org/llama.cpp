#include "common.cuh"

#define CONV_SHAPE_128x128 0
#define CONV_SHAPE_64x32   1
#define CONV_SHAPE_32x256  2

#define NUM_VARIANTS 3

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

void ggml_cuda_op_conv2d_tensor_core(const uint32_t &     IW,
                                     const uint32_t &     IH,
                                     const uint32_t &     OW,
                                     const uint32_t &     OH,
                                     const uint32_t &     KW,
                                     const uint32_t &     KH,
                                     const uint32_t &     ST_X,
                                     const uint32_t &     ST_Y,
                                     const uint32_t &     PD_X,
                                     const uint32_t &     PD_Y,
                                     const uint32_t &     DL_X,
                                     const uint32_t &     DL_Y,
                                     const uint32_t &     IC,
                                     const uint32_t &     OC,
                                     const uint32_t &     B,
                                     const float *        IN,
                                     const half *         IK,
                                     float *              output,
                                     const cudaStream_t & st);
