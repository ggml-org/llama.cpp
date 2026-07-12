#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_CUDA_USE_CUB
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows);
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);
