#include "conv2d.cuh"
#include "convert.cuh"

struct conv_params {
    const int     IW, IH;
    const int     OW, OH;
    const int     KW, KH;
    const int     ST_X, ST_Y;
    const int     PD_X, PD_Y;
    const int     DL_X, DL_Y;
    const int     IC, OC;
    const int     B;
    const int64_t TOTAL;
    // helpers
    const int     IC_KH_KW, N_OH_OW;
};

__device__ __forceinline__ static int calculate_input_coord(int out_coord,
                                                            int kern_coord,
                                                            int stride,
                                                            int dilation,
                                                            int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ __forceinline__ static int64_t input_index(int n, int c, int y, int x, const conv_params & P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ __forceinline__ static int64_t kernel_index(int c_out, int c_in, int ky, int kx, const conv_params & P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ __forceinline__ static int64_t output_index(int n, int c, int y, int x, const conv_params & P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ __forceinline__ static void unpack_ickhkw(int64_t             idx,
                                                         int &               ic,
                                                         int &               kh,
                                                         int &               kw,
                                                         const conv_params & P) {
        ic        = idx / (P.KW * P.KH);
        int64_t r = idx - ic * (P.KW * P.KH);
        kh        = r / P.KW;
        kw        = r - kh * P.KW;
    }

    __device__ __forceinline__ static void unpack_nohow(int64_t             idx,
                                                        int &               n,
                                                        int &               oh,
                                                        int &               ow,
                                                        const conv_params & P) {
        n         = idx / (P.OH * P.OW);
        int64_t r = idx - n * (P.OH * P.OW);
        oh        = r / P.OW;
        ow        = r - oh * P.OW;
    }
};

template <typename layout> class float_mma {
  public:
    static constexpr int num_acc = (WMMA_M * WMMA_N + WARP_SIZE - 1) / WARP_SIZE;

    float acc[num_acc];

    __device__ __forceinline__ float_mma() {
#pragma unroll
        for (int i = 0; i < num_acc; i++) {
            acc[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < num_acc; i++) {
            acc[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void mma(const float * __restrict__ A_sh,
                                        const float * __restrict__ B_sh,
                                        const int strideA,
                                        const int strideB) {
        const int lane_id = threadIdx.x % WARP_SIZE;

#pragma unroll
        for (int e = lane_id, i = 0; e < WMMA_M * WMMA_N; e += WARP_SIZE, i++) {
            const int m = e / WMMA_N;
            const int n = e % WMMA_N;

#pragma unroll
            for (int k = 0; k < WMMA_K; k++) {
                const float a = A_sh[m * strideA + k];
                const float b = B_sh[k * strideB + n];
                acc[i]        = fmaf(a, b, acc[i]);
            }
        }
    }

    __device__ __forceinline__ void store_result(const int OC_BASE,
                                                 const int NOHOW_BASE,
                                                 float * __restrict__ OUT,
                                                 const conv_params & P) const {
        const int lane_id = threadIdx.x % WARP_SIZE;

#pragma unroll
        for (int e = lane_id, i = 0; e < WMMA_M * WMMA_N; e += WARP_SIZE, i++) {
            const int m = e / WMMA_N;
            const int n = e % WMMA_N;

            const int oc    = OC_BASE + m;
            const int nohow = NOHOW_BASE + n;

            if (oc < P.OC && nohow < P.N_OH_OW) {
                int n_, oh, ow;
                layout::unpack_nohow(nohow, n_, oh, ow, P);
                OUT[layout::output_index(n_, oc, oh, ow, P)] = acc[i];
            }
        }
    }
};

#if (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || (defined(FP16_MMA_AVAILABLE)))
#    include "mma.cuh"
using namespace ggml_cuda_mma;

typedef ggml_cuda_mma::tile<WMMA_M, WMMA_K / 2, half2> tile_a;
typedef ggml_cuda_mma::tile<WMMA_N, WMMA_K / 2, half2> tile_b;
typedef ggml_cuda_mma::tile<WMMA_M, WMMA_N, float>     tile_acc;

template <typename layout> class half_mma {
  private:
    tile_a   a_frag;
    tile_b   b_frag;
    tile_acc c_frag;
  public:
    __device__ __forceinline__ half_mma() {}

    __device__ __forceinline__ void clear() {
#    pragma unroll
        for (int l = 0; l < c_frag.ne; ++l) {
            c_frag.x[l] = 0.0f;
        }
    }

    __device__ __forceinline__ void mma(const half * __restrict__ A_sh,
                                        const half * __restrict__ B_sh,
                                        const int strideA,
                                        const int strideB) {
        ggml_cuda_mma::load_ldmatrix(a_frag, (const half2 *) A_sh, strideA / 2);
        ggml_cuda_mma::load_ldmatrix_trans(b_frag, (const half2 *) B_sh, strideB / 2);
        ggml_cuda_mma::mma(c_frag, a_frag, b_frag);
    }

    __device__ __forceinline__ void store_result(const int OC_BASE,
                                                 const int NOHOW_BASE,
                                                 float * __restrict__ OUT,
                                                 const conv_params & P) const {
#    pragma unroll
        for (int l = 0; l < tile_acc::ne; ++l) {
            const int e = tile_acc::get_i(l) * WMMA_N + tile_acc::get_j(l);
            const int m = e / WMMA_N;
            const int n = e % WMMA_N;

            const int oc    = OC_BASE + m;
            const int nohow = NOHOW_BASE + n;

            if (oc < P.OC && nohow < (P.N_OH_OW)) {
                int n, oh, ow;
                layout::unpack_nohow(nohow, n, oh, ow, P);
                OUT[layout::output_index(n, oc, oh, ow, P)] = c_frag.x[l];
            }
        }
    }
};

#else

template <typename layout> class half_mma {
  public:
    static constexpr int num_acc = (WMMA_M * WMMA_N + WARP_SIZE - 1) / WARP_SIZE;

    float acc[num_acc];

    __device__ __forceinline__ half_mma() {
#    pragma unroll
        for (int i = 0; i < num_acc; i++) {
            acc[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void clear() {
#    pragma unroll
        for (int i = 0; i < num_acc; i++) {
            acc[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void mma(const half * __restrict__ A_sh,
                                        const half * __restrict__ B_sh,
                                        const int strideA,
                                        const int strideB) {
        const int lane_id = threadIdx.x % WARP_SIZE;

#    pragma unroll
        for (int e = lane_id, i = 0; e < WMMA_M * WMMA_N; e += WARP_SIZE, i++) {
            const int m = e / WMMA_N;
            const int n = e % WMMA_N;

#    pragma unroll
            for (int k = 0; k < WMMA_K; k++) {
                const half a = A_sh[m * strideA + k];
                const half b = B_sh[k * strideB + n];
                acc[i]       = fmaf(__half2float(a), __half2float(b), acc[i]);
            }
        }
    }

    __device__ __forceinline__ void store_result(const int OC_BASE,
                                                 const int NOHOW_BASE,
                                                 float * __restrict__ OUT,
                                                 const conv_params & P) const {
        const int lane_id = threadIdx.x % WARP_SIZE;

#    pragma unroll
        for (int e = lane_id, i = 0; e < WMMA_M * WMMA_N; e += WARP_SIZE, i++) {
            const int m = e / WMMA_N;
            const int n = e % WMMA_N;

            const int oc    = OC_BASE + m;
            const int nohow = NOHOW_BASE + n;

            if (oc < P.OC && nohow < P.N_OH_OW) {
                int n_, oh, ow;
                layout::unpack_nohow(nohow, n_, oh, ow, P);
                OUT[layout::output_index(n_, oc, oh, ow, P)] = acc[i];
            }
        }
    }
};

#endif  // defined((__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || defined(FP16_MMA_AVAILABLE))

template <typename T, typename layout, typename mma, int num_warps>
__global__ void conv2d_kernel(const float * IN, const T * IK, float * Out, const conv_params P) {
    extern __shared__ unsigned char smem_raw[];

    const int NUM_IC_TILES = (P.IC_KH_KW + BS_ICKHKW - 1) / BS_ICKHKW;
    const int warpId       = threadIdx.y;

    const int WARPS_PER_NOHOW    = max(1, BS_NOHOW / WMMA_N);
    const int total_warps_need   = (((BS_OC * BS_NOHOW) + (WMMA_M * WMMA_N) - 1) / (WMMA_M * WMMA_N));
    const int num_work_per_warps = (total_warps_need + num_warps - 1) / num_warps;

    mma acc[num_work_per_warps];

    const int num_block_nohow = (P.N_OH_OW + BS_NOHOW - 1) / BS_NOHOW;
    const int BL_IDX_OC       = blockIdx.x / num_block_nohow;
    const int BL_IDX_NOHOW    = blockIdx.x % num_block_nohow;

    const int BLOCK_OC_BASE    = BL_IDX_OC * BS_OC;
    const int BLOCK_NOHOW_BASE = BL_IDX_NOHOW * BS_NOHOW;

    unsigned char * ptr = smem_raw;

    const int A_total = BS_OC * BS_ICKHKW;
    const int B_total = BS_ICKHKW * BS_NOHOW;

    size_t offsetA = (size_t) A_total * sizeof(T);
    T *    A_sh    = reinterpret_cast<T *>(ptr);
    ptr += offsetA;

    size_t offsetB = (size_t) B_total * sizeof(T);
    T *    B_sh    = reinterpret_cast<T *>(ptr);
    ptr += offsetB;

    int ic, kh, kw;
    int n, oh, ow;
    for (int t = 0; t < NUM_IC_TILES; ++t) {
#pragma unroll
        for (int tid = (threadIdx.y * blockDim.x + threadIdx.x); tid < A_total; tid += (blockDim.x * blockDim.y)) {
            const int row = tid / BS_ICKHKW;
            const int col = tid % BS_ICKHKW;

            int shared_oc     = BLOCK_OC_BASE + row;
            int shared_ickhkw = t * BS_ICKHKW + col;

            T val = ggml_cuda_cast<T>(0);
            if (shared_oc < P.OC && shared_ickhkw < P.IC_KH_KW) {
                layout::unpack_ickhkw(shared_ickhkw, ic, kh, kw, P);

                const int kidx = layout::kernel_index(shared_oc, ic, kh, kw, P);
                val            = IK[kidx];
            }
            A_sh[row * BS_ICKHKW + col] = val;
        }
#pragma unroll
        for (int tid = (threadIdx.y * blockDim.x + threadIdx.x); tid < B_total; tid += (blockDim.x * blockDim.y)) {
            const int brow = tid / BS_NOHOW;
            const int bcol = tid % BS_NOHOW;

            int IC_KH_KW_IDX = t * BS_ICKHKW + brow;
            int N_OH_OW_IDX  = BLOCK_NOHOW_BASE + bcol;

            T val = ggml_cuda_cast<T>(0);
            if (N_OH_OW_IDX < P.N_OH_OW && IC_KH_KW_IDX < P.IC_KH_KW) {
                layout::unpack_nohow(N_OH_OW_IDX, n, oh, ow, P);
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
        for (int warp = warpId, i = 0; warp < total_warps_need; warp += num_warps, i++) {
            const int WARP_OC     = warp / WARPS_PER_NOHOW;
            const int WARP_NOHOW  = warp % WARPS_PER_NOHOW;
            const T * A_warp_base = A_sh + WARP_OC * WMMA_M * BS_ICKHKW;
            const T * B_warp_base = B_sh + WARP_NOHOW * WMMA_N;
#pragma unroll
            for (int k_tile = 0; k_tile < BS_ICKHKW; k_tile += WMMA_K) {
                const T * A_k_ptr = A_warp_base + k_tile;
                const T * B_k_ptr = B_warp_base + k_tile * BS_NOHOW;
                acc[i].mma(A_k_ptr, B_k_ptr, BS_ICKHKW, BS_NOHOW);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int warp = warpId, i = 0; warp < total_warps_need; warp += num_warps, i++) {
        const int WARP_OC    = warp / WARPS_PER_NOHOW;
        const int WARP_NOHOW = warp % WARPS_PER_NOHOW;
        const int OC_BASE    = BLOCK_OC_BASE + WARP_OC * WMMA_M;
        const int NOHOW_BASE = BLOCK_NOHOW_BASE + WARP_NOHOW * WMMA_N;
        acc[i].store_result(OC_BASE, NOHOW_BASE, Out, P);
    }
}

template <typename T, template <typename> class mma>
static void conv2d_cuda(const float * X_D, const T * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    const int warp_size      = 32;
    const int max_block_size = 256;

    GGML_ASSERT(BS_OC >= WMMA_M && BS_ICKHKW >= WMMA_K && BS_NOHOW >= WMMA_N);

    const int num_block_oc    = (P.OC + BS_OC - 1) / BS_OC;
    const int num_block_nohow = (P.N_OH_OW + BS_NOHOW - 1) / BS_NOHOW;
    const int num_blocks      = num_block_oc * num_block_nohow;

    int nwarps_best = 1;
    int niter_best  = (BS_OC * BS_NOHOW + warp_size - 1) / (warp_size);
    for (int nwarps = 2; nwarps <= max_block_size / warp_size; ++nwarps) {
        const int niter = (BS_OC * BS_NOHOW + nwarps * warp_size - 1) / (nwarps * warp_size);
        if (niter < niter_best) {
            niter_best  = niter;
            nwarps_best = nwarps;
        }
    }

    const size_t A_bytes      = BS_OC * BS_ICKHKW * sizeof(T);
    const size_t B_bytes      = BS_ICKHKW * BS_NOHOW * sizeof(T);
    const size_t shared_bytes = A_bytes + B_bytes;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(warp_size, nwarps_best);

    switch (nwarps_best) {
        case 1:
            conv2d_kernel<T, whcn_layout, mma<whcn_layout>, 1><<<grid, block, shared_bytes, st>>>(X_D, K_D, Y_D, P);
            break;
        case 2:
            conv2d_kernel<T, whcn_layout, mma<whcn_layout>, 2><<<grid, block, shared_bytes, st>>>(X_D, K_D, Y_D, P);
            break;
        case 4:
            conv2d_kernel<T, whcn_layout, mma<whcn_layout>, 4><<<grid, block, shared_bytes, st>>>(X_D, K_D, Y_D, P);
            break;
        case 8:
            conv2d_kernel<T, whcn_layout, mma<whcn_layout>, 8><<<grid, block, shared_bytes, st>>>(X_D, K_D, Y_D, P);
            break;
        case 16:
            conv2d_kernel<T, whcn_layout, mma<whcn_layout>, 16><<<grid, block, shared_bytes, st>>>(X_D, K_D, Y_D, P);
            break;
        case 32:
            conv2d_kernel<T, whcn_layout, mma<whcn_layout>, 32><<<grid, block, shared_bytes, st>>>(X_D, K_D, Y_D, P);
            break;
        default:
            GGML_ABORT("UNSUPPROTED NWARPS_BEST");
    }
}

static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const conv_params & P, cudaStream_t st) {
    conv2d_cuda<half, half_mma>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const conv_params & P, cudaStream_t st) {
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
    const int     IC_KH_KW = IC * KH * KW;
    const int     N_OH_OW  = B * OH * OW;
    conv_params   params   = { IW,   IH,   OW,   OH, KW, KH, ST_X,  ST_Y,     PD_X,
                               PD_Y, DL_X, DL_Y, IC, OC, B,  TOTAL, IC_KH_KW, N_OH_OW };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, (const half *) K_D, Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}
