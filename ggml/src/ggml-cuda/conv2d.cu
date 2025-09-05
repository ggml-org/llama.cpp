#include "conv2d.cuh"
#include "convert.cuh"

#include <mma.h>
using namespace nvcuda;

struct conv_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
    // helpers
    const int64_t IC_KH_KW, N_OH_OW;
};

auto ceil_div = [](int a, int b) {
    return (a + b - 1) / b;
};

__device__ __forceinline__ static int calculate_input_coord(int64_t out_coord,
                                                            int64_t kern_coord,
                                                            int64_t stride,
                                                            int64_t dilation,
                                                            int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ __forceinline__ static int64_t input_index(int64_t             n,
                                                          int64_t             c,
                                                          int64_t             y,
                                                          int64_t             x,
                                                          const conv_params & P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ __forceinline__ static int64_t kernel_index(int64_t             c_out,
                                                           int64_t             c_in,
                                                           int64_t             ky,
                                                           int64_t             kx,
                                                           const conv_params & P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ __forceinline__ static int64_t output_index(int64_t             n,
                                                           int64_t             c,
                                                           int64_t             y,
                                                           int64_t             x,
                                                           const conv_params & P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ __forceinline__ static void unpack_ickhkw(int64_t             idx,
                                                         int64_t &           ic,
                                                         int64_t &           kh,
                                                         int64_t &           kw,
                                                         const conv_params & P) {
        ic        = idx / (P.KW * P.KH);
        int64_t r = idx - ic * (P.KW * P.KH);
        kh        = r / P.KW;
        kw        = r - kh * P.KW;
    }

    __device__ __forceinline__ static void unpack_nohow(int64_t             idx,
                                                        int64_t &           n,
                                                        int64_t &           oh,
                                                        int64_t &           ow,
                                                        const conv_params & P) {
        n         = idx / (P.OH * P.OW);
        int64_t r = idx - n * (P.OH * P.OW);
        oh        = r / P.OW;
        ow        = r - oh * P.OW;
    }
};

class float_mma {
  public:
    float * buf;

    __device__ __forceinline__ float_mma(float * scratch) {
        buf               = scratch;
        const int lane_id = threadIdx.x % warpSize;
#pragma unroll
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += warpSize) {
            buf[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void mma(const float * A_sh, const float * B_sh, const int strideA, const int strideB) {
        const int lane_id = threadIdx.x % warpSize;
#pragma unroll
        for (int e = lane_id; e < (WMMA_M * WMMA_N); e += warpSize) {
            int   m   = e / WMMA_N;
            int   n   = e % WMMA_N;
            float sum = buf[m * WMMA_N + n];
#pragma unroll
            for (int k = 0; k < WMMA_K; k++) {
                float a = A_sh[m * strideA + k];
                float b = B_sh[k * strideB + n];
                sum     = fmaf(a, b, sum);
            }
            buf[m * WMMA_N + n] = sum;
        }
    }

    __device__ __forceinline__ float * store_result() const { return buf; }
};

class half_mma {
  private:
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              acc;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  public:
    float * buf;

    __device__ __forceinline__ half_mma(float * scratch) {
        buf = scratch;
        wmma::fill_fragment(acc, 0.0f);
    }

    __device__ __forceinline__ void mma(const half * A_sh, const half * B_sh, const int strideA, const int strideB) {
        wmma::load_matrix_sync(a_frag, A_sh, strideA);
        wmma::load_matrix_sync(b_frag, B_sh, strideB);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    __device__ __forceinline__ float * store_result() const {
        wmma::store_matrix_sync(buf, acc, WMMA_N, wmma::mem_row_major);
        return buf;
    }
};

template <typename T, typename layout, typename mma>
static __global__ void conv2d_kernel(const float * IN, const T * IK, float * OUT, const conv_params P) {
    extern __shared__ unsigned char smem_raw[];

    const int64_t OUTPUT_NUMEL = WMMA_M * WMMA_N;
    const int64_t NUM_IC_TILES = (P.IC_KH_KW + BS_ICKHKW - 1) / BS_ICKHKW;

    const int64_t WARPS_PER_NOHOW = max(1, BS_NOHOW / WMMA_N);

    const int64_t NUM_BL_NOHOW     = (P.N_OH_OW + BS_NOHOW - 1) / BS_NOHOW;
    const int64_t tile_id          = blockIdx.x;
    const int64_t tile_oc          = tile_id / NUM_BL_NOHOW;
    const int64_t tile_nohow       = tile_id % NUM_BL_NOHOW;
    const int64_t BLOCK_OC_BASE    = tile_oc * BS_OC;
    const int64_t BLOCK_NOHOW_BASE = tile_nohow * BS_NOHOW;

    const int64_t laneId = threadIdx.x % WARP_SIZE;
    const int64_t warpId = threadIdx.x / WARP_SIZE;

    const int64_t WARP_OC    = warpId / WARPS_PER_NOHOW;
    const int64_t WARP_NOHOW = warpId % WARPS_PER_NOHOW;

    const int64_t OC_BASE    = BLOCK_OC_BASE + WARP_OC * WMMA_M;
    const int64_t NOHOW_BASE = BLOCK_NOHOW_BASE + WARP_NOHOW * WMMA_N;

    unsigned char * ptr  = smem_raw;
    T *             A_sh = reinterpret_cast<T *>(ptr);

    size_t offsetA = BS_OC * BS_ICKHKW * sizeof(T);
    ptr += offsetA;

    T * B_sh = reinterpret_cast<T *>(ptr);
    ptr += BS_ICKHKW * BS_NOHOW * sizeof(T);

    float * shared_scratch = reinterpret_cast<float *>(ptr);
    float * warp_scratch   = shared_scratch + warpId * (WMMA_M * WMMA_N);

    const T * A_warp_base = A_sh + WARP_OC * WMMA_M * BS_ICKHKW;
    const T * B_warp_base = B_sh + WARP_NOHOW * WMMA_N;

    mma acc(warp_scratch);

    const int64_t A_total = BS_OC * BS_ICKHKW;
    const int64_t B_total = BS_ICKHKW * BS_NOHOW;

#pragma unroll
    for (int64_t t = 0; t < NUM_IC_TILES; ++t) {
#pragma unroll
        for (int64_t tid = (threadIdx.x); tid < A_total; tid += blockDim.x) {
            const int row = tid / BS_ICKHKW;
            const int col = tid % BS_ICKHKW;

            int64_t shared_oc     = BLOCK_OC_BASE + row;
            int64_t shared_ickhkw = t * BS_ICKHKW + col;

            T val = ggml_cuda_cast<T>(0);
            if (shared_oc < P.OC && shared_ickhkw < P.IC_KH_KW) {
                int64_t ic, kh, kw;
                layout::unpack_ickhkw(shared_ickhkw, ic, kh, kw, P);

                const int64_t kidx = layout::kernel_index(shared_oc, ic, kh, kw, P);
                val                = IK[kidx];
            }
            A_sh[row * BS_ICKHKW + col] = val;
        }

#pragma unroll
        for (int64_t tid = (threadIdx.x); tid < B_total; tid += blockDim.x) {
            const int brow = tid / BS_NOHOW;
            const int bcol = tid % BS_NOHOW;

            int64_t IC_KH_KW_IDX = t * BS_ICKHKW + brow;
            int64_t N_OH_OW_IDX  = BLOCK_NOHOW_BASE + bcol;

            T val = ggml_cuda_cast<T>(0);
            if (N_OH_OW_IDX < P.N_OH_OW && IC_KH_KW_IDX < P.IC_KH_KW) {
                int64_t n, oh, ow;
                layout::unpack_nohow(N_OH_OW_IDX, n, oh, ow, P);
                int64_t ic, kh, kw;
                layout::unpack_ickhkw(IC_KH_KW_IDX, ic, kh, kw, P);
                int in_y = calculate_input_coord(oh, kh, P.ST_Y, P.DL_Y, P.PD_Y);
                int in_x = calculate_input_coord(ow, kw, P.ST_X, P.DL_X, P.PD_X);
                if (in_y >= 0 && in_y < P.IH && in_x >= 0 && in_x < P.IW) {
                    const int64_t in_idx = layout::input_index(n, ic, in_y, in_x, P);
                    val                  = ggml_cuda_cast<T>(IN[in_idx]);
                }
            }
            B_sh[brow * BS_NOHOW + bcol] = val;
        }

        __syncthreads();

#pragma unroll
        for (int k_tile = 0; k_tile < BS_ICKHKW; k_tile += WMMA_K) {
            const T * A_k_ptr = A_warp_base + k_tile;
            const T * B_k_ptr = B_warp_base + k_tile * BS_NOHOW;

            acc.mma(A_k_ptr, B_k_ptr, BS_ICKHKW, BS_NOHOW);
        }
        __syncthreads();
    }

    const float * out_buf = acc.store_result();
#pragma unroll
    for (int e = laneId; e < OUTPUT_NUMEL; e += WARP_SIZE) {
        const int m = e / WMMA_N;
        const int n = e % WMMA_N;

        const int64_t oc    = OC_BASE + m;
        const int64_t nohow = NOHOW_BASE + n;

        if (oc < P.OC && nohow < (P.N_OH_OW)) {
            int64_t n, oh, ow;
            layout::unpack_nohow(nohow, n, oh, ow, P);
            const int64_t out_idx = layout::output_index(n, oc, oh, ow, P);
            OUT[out_idx]          = out_buf[e];
        }
    }
}

template <typename T, typename mma>
static void conv2d_cuda(const float * X_D, const T * K_D, float * Y_D, conv_params P, cudaStream_t st)

{
    const int64_t NUM_BL_OC    = (P.OC + BS_OC - 1) / BS_OC;
    const int64_t NUM_BL_NOHOW = (P.N_OH_OW + BS_NOHOW - 1) / BS_NOHOW;

    int64_t TOTAL_TILES = NUM_BL_OC * NUM_BL_NOHOW;
    TOTAL_TILES         = std::min(TOTAL_TILES, (int64_t) INT_MAX);

    const int WARPS_PER_OC    = std::max(1, ceil_div(BS_OC, WMMA_M));
    const int WARPS_PER_NOHOW = std::max(1, ceil_div(BS_NOHOW, WMMA_N));
    const int EXPECTED_WARPS  = WARPS_PER_OC * WARPS_PER_NOHOW;
    int       N_THREADS       = EXPECTED_WARPS * WARP_SIZE;

    const int MAX_TPB = 1024;
    if (N_THREADS > MAX_TPB) {
        N_THREADS = (MAX_TPB / WARP_SIZE) * WARP_SIZE;
    }

    if (N_THREADS < WARP_SIZE) {
        N_THREADS = WARP_SIZE;
    }

    const int N_WARPS = N_THREADS / WARP_SIZE;

    // scratch_buff to store output, can't store directly using wmma,
    // output mapping is unknown
    const int64_t scratch_bytes = N_WARPS * (WMMA_M * WMMA_N) * sizeof(float);

    const int64_t A_bytes      = BS_OC * BS_ICKHKW * sizeof(T);
    const int64_t B_bytes      = BS_ICKHKW * BS_NOHOW * sizeof(T);
    const int64_t shared_bytes = A_bytes + B_bytes + scratch_bytes;

    dim3 grid(TOTAL_TILES, 1, 1);
    conv2d_kernel<T, whcn_layout, mma><<<grid, N_THREADS, shared_bytes, st>>>(X_D, K_D, Y_D, P);
}

static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, conv_params & P, cudaStream_t st) {
    conv2d_cuda<half, half_mma>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, conv_params & P, cudaStream_t st) {
    conv2d_cuda<float, float_mma>(X_D, K_D, Y_D, P, st);
}

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int IW = input->ne[0];   // input_w
    const int IH = input->ne[1];   // input_h
    const int OW = dst->ne[0];     // output_w
    const int OH = dst->ne[1];     // output_h
    const int KW = kernel->ne[0];  // kernel_w
    const int KH = kernel->ne[1];  // kernel_h
    const int IC = input->ne[2];   // input_channels
    const int OC = kernel->ne[3];  // ouptut_chanles
    const int B  = input->ne[3];   // n_batches

    const int64_t TOTAL    = B * OC * OH * OW;
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t N_OH_OW  = B * OH * OW;
    conv_params   params   = { IW,   IH,   OW,   OH, KW, KH, ST_X,  ST_Y,     PD_X,
                               PD_Y, DL_X, DL_Y, IC, OC, B,  TOTAL, IC_KH_KW, N_OH_OW };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, (const half *) K_D, Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}
