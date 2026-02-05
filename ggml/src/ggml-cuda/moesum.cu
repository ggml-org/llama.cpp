#include "moesum.cuh"

template <typename T>
__device__ __forceinline__ T ldg_cg(const T* p) {
  return __ldg(p);
}

union Pack16B {
  uint4 v;
  __half u16[8];
};

template <int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_vec_kernel(
    const half* __restrict__ x,
    half* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t topk_num,
    const int32_t stride_token,      // in elements
    const int32_t stride_topk,       // in elements
    const int32_t out_stride_token   // in elements
) {
  constexpr int VEC = 16;
  constexpr int PACKS = VEC / 8;

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int32_t t = blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  const int32_t n_chunks = hidden_dim / VEC;

  for (int32_t chunk = blockIdx.x * 32 + lane; chunk < n_chunks; chunk += (int32_t)gridDim.x * 32) {
    const int32_t d = chunk * VEC;
    const int32_t base = t * stride_token + d;

    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] = 0.f;

#pragma unroll
    for (int k = 0; k < topk_num; ++k) {
#pragma unroll
      for (int p = 0; p < PACKS; ++p) {
        const int32_t offset = base + (int32_t)k * stride_topk + p * 8;
        Pack16B pack = {ldg_cg(reinterpret_cast<const uint4*>(x + offset))};

#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[p * 8 + i] += static_cast<float>(pack.u16[i]);
        }
      }
    }

#pragma unroll
    for (int p = 0; p < PACKS; ++p) {
      Pack16B outp;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        outp.u16[i] = static_cast<half>(acc[p * 8 + i]);
      }
      const int32_t dst = t * out_stride_token + d + p * 8;
      *reinterpret_cast<uint4*>(y + dst) = outp.v;
    }
  }
}

template <typename scalar_t, int TOPK, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int32_t t = blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int32_t d = blockIdx.x * 32 + lane; d < hidden_dim; d += gridDim.x * 32) {
    float acc = 0.f;
    const int32_t base = t * stride_token + d;

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += static_cast<float>(x[base + k * stride_topk]);
    }

    y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_kernel_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token,
    const int topk_num) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int32_t t = blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int32_t d = blockIdx.x * 32 + lane; d < hidden_dim; d += gridDim.x * 32) {
    float acc = 0.f;
    const int32_t base = t * stride_token + d;
#pragma unroll 1
    for (int k = 0; k < topk_num; ++k) {
      acc += static_cast<float>(x[base + k * stride_topk]);
    }

    y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_reduce_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int32_t base = t * stride_token + d;
      float acc = 0.f;

#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += static_cast<float>(x[base + k * stride_topk]);
      }

      y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
    }
  }
}

// -------------------- general-topk fallback kernels --------------------
// small-token
template <typename scalar_t>
__global__ void moe_sum_reduce_kernel_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token,
    const int topk_num) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int32_t base = t * stride_token + d;
      float acc = 0.f;

#pragma unroll 1
      for (int k = 0; k < topk_num; ++k) {
        acc += static_cast<float>(x[base + k * stride_topk]);
      }

      y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
    }
  }
}

#define LAUNCH_SMALL_TOKEN_KERNEL(scalar_t, TOPK)                       \
    moe_sum_reduce_kernel<scalar_t, TOPK><<<grid, block, 0, stream>>>(  \
        static_cast<scalar_t*>(src0->data),                             \
        static_cast<scalar_t*>(dst->data),                              \
        token_num,                                                      \
        hidden_dim,                                                     \
        stride_token,                                                   \
        stride_topk,                                                    \
        out_stride_token);

#define LAUNCH_GENERIC_KERNEL(scalar_t)                                 \
    moe_sum_reduce_kernel_general<scalar_t>                             \
            <<<grid, block, 0, stream>>>(                               \
        static_cast<scalar_t*>(src0->data),                             \
        static_cast<scalar_t*>(dst->data),                              \
        token_num,                                                      \
        hidden_dim,                                                     \
        stride_token,                                                   \
        stride_topk,                                                    \
        out_stride_token,                                               \
        topk_num);

#define LAUNCH_WARP_PER_TOKEN_KERNEL(scalar_t, TOPK)                    \
    moe_sum_reduce_warp_token_kernel<scalar_t, TOPK, WARPS_PER_BLOCK>   \
            <<<grid, block, 0, stream>>>(                               \
        static_cast<scalar_t*>(src0->data),                             \
        static_cast<scalar_t*>(dst->data),                              \
        token_num,                                                      \
        hidden_dim,                                                     \
        stride_token,                                                   \
        stride_topk,                                                    \
        out_stride_token);

#define LAUNCH_WARP_PER_TOKEN_GENERIC_KERNEL(scalar_t)                  \
    moe_sum_reduce_warp_token_kernel_general<scalar_t, WARPS_PER_BLOCK> \
            <<<grid, block, 0, stream>>>(                               \
        static_cast<scalar_t*>(src0->data),                             \
        static_cast<scalar_t*>(dst->data),                              \
        token_num,                                                      \
        hidden_dim,                                                     \
        stride_token,                                                   \
        stride_topk,                                                    \
        out_stride_token,                                               \
        topk_num);

void ggml_cuda_op_moe_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // [hidden_dim, n_experts_used, tokens]
    ggml_tensor * src0 = dst->src[0];


    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[0] == dst->ne[0]);
    GGML_ASSERT(src0->ne[2] == dst->ne[1]);

    const int token_num = src0->ne[2];
    const int topk_num = src0->ne[1];
    const int hidden_dim = src0->ne[0];

    const int stride_token = src0->nb[2] / src0->nb[0];
    const int stride_topk = src0->nb[1] / src0->nb[0];
    const int out_stride_token = dst->nb[1] / dst->nb[0];

    auto stream = ctx.stream();

    const bool fast_fp16_vec_ok = (src0->type == GGML_TYPE_F16) &&
                    (token_num > 256) && (hidden_dim % 8 == 0);
    if (fast_fp16_vec_ok) {
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;

        const int n_chunks = hidden_dim / 8;
        int grid_x = (n_chunks + 32 - 1) / 32;
        int grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        dim3 block(THREADS);
        dim3 grid(grid_x, grid_y);

        moe_sum_reduce_warp_token_vec_kernel<WARPS_PER_BLOCK>
                <<<grid, block, 0, stream>>>(
            static_cast<half*>(src0->data),
            static_cast<half*>(dst->data),
            token_num,
            hidden_dim,
            topk_num,
            stride_token,
            stride_topk,
            out_stride_token);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const bool per_token_use_one_warp = (token_num > 128);
    if (!per_token_use_one_warp) {
        // small token num
        const int block_size = 256;
        int grid_x = (hidden_dim + block_size - 1) / block_size;
        int grid_y = token_num;

        dim3 block(block_size);
        dim3 grid(grid_x, grid_y);

        if (src0->type == GGML_TYPE_F32) {
            if (topk_num == 2) {
                LAUNCH_SMALL_TOKEN_KERNEL(float, 2);
            } else if (topk_num == 4) {
                LAUNCH_SMALL_TOKEN_KERNEL(float, 4);
            } else if (topk_num == 8) {
                LAUNCH_SMALL_TOKEN_KERNEL(float, 8);
            } else if (topk_num == 9) {
                LAUNCH_SMALL_TOKEN_KERNEL(float, 9);
            } else {
                LAUNCH_GENERIC_KERNEL(float);
            }
        } else if (src0->type == GGML_TYPE_F16) {
            if (topk_num == 2) {
                LAUNCH_SMALL_TOKEN_KERNEL(half, 2);
            } else if (topk_num == 4) {
                LAUNCH_SMALL_TOKEN_KERNEL(half, 4);
            } else if (topk_num == 8) {
                LAUNCH_SMALL_TOKEN_KERNEL(half, 8);
            } else if (topk_num == 9) {
                LAUNCH_SMALL_TOKEN_KERNEL(half, 9);
            } else {
                LAUNCH_GENERIC_KERNEL(half);
            }
        } else {
            GGML_ASSERT(false);
        }
    } else {
        // warp-per-token
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;

        int grid_x = (hidden_dim + 32 - 1) / 32;
        int grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        dim3 block(THREADS);
        dim3 grid(grid_x, grid_y);

        if (src0->type == GGML_TYPE_F32) {
            if (topk_num == 2) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(float, 2);
            } else if (topk_num == 4) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(float, 4);
            } else if (topk_num == 8) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(float, 8);
            } else if (topk_num == 9) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(float, 9);
            } else {
                LAUNCH_WARP_PER_TOKEN_GENERIC_KERNEL(float);
            }
        } else if (src0->type == GGML_TYPE_F16) {
            if (topk_num == 2) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(half, 2);
            } else if (topk_num == 4) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(half, 4);
            } else if (topk_num == 8) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(half, 8);
            } else if (topk_num == 9) {
                LAUNCH_WARP_PER_TOKEN_KERNEL(half, 9);
            } else {
                LAUNCH_WARP_PER_TOKEN_GENERIC_KERNEL(half);
            }
        } else {
            GGML_ASSERT(false);
        }
    }
}
