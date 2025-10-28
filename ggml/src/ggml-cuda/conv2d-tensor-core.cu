#include "common.cuh"
#include "conv2d-tensor-core.cuh"
#include "convert.cuh"
#include "mma.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

static uint32_t ceil_div(uint32_t M, uint32_t N);
static int      get_sm_count();

uint32_t ceil_div(uint32_t M, uint32_t N) {
    return (M + N - 1) / N;
}

__align__(16) struct Params {
    uint32_t IW, IH;
    uint32_t OW, OH;
    uint32_t KW, KH;
    uint32_t ST_X, ST_Y;
    uint32_t PD_X, PD_Y;
    uint32_t DL_X, DL_Y;
    uint32_t Cin, Cout;
    uint32_t B;
    // helpers
    uint32_t IC_KH_KW, N_OH_OW;
    uint32_t IK_TOTAL, IN_TOTAL;

    uint32_t KWmp;
    uint32_t KWL;
    uint32_t KWKHmp;
    uint32_t KWKHL;
    uint32_t OWmp;
    uint32_t OWL;
    uint32_t OWOHmp;
    uint32_t OWOHL;
};

__constant__ __device__ Params P;

// see init_fastdiv_values in ggml-vulkan.cpp
__inline__ __device__ uint fastdiv(uint n, uint mp, uint L) {
    return (__umulhi(n, mp) + n) >> L;
}

__device__ struct T_ICKHKW {
    const uint32_t ic, kh, kw;
};

__device__ struct T_NOHOW {
    const uint32_t B, OH, OW;
};

__device__ __forceinline__ static int32_t calculate_input_coord(const uint32_t & out_coord,
                                                                const uint32_t & kern_coord,
                                                                const uint32_t & stride,
                                                                const uint32_t & dilation,
                                                                const uint32_t & padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ __forceinline__ static uint32_t input_index(const uint32_t & n,
                                                           const uint32_t & c,
                                                           const uint32_t & y,
                                                           const uint32_t & x) {
        return n * (P.Cin * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ __forceinline__ static uint32_t kernel_index(const uint32_t & c_out,
                                                            const uint32_t & c_in,
                                                            const uint32_t & ky,
                                                            const uint32_t & kx) {
        return c_out * (P.Cin * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ __forceinline__ static uint32_t output_index(const uint32_t & n,
                                                            const uint32_t & c,
                                                            const uint32_t & y,
                                                            const uint32_t & x) {
        return n * (P.Cout * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ __forceinline__ static T_ICKHKW unpack_ickhkw(const uint32_t & idx) {
        // const uint32_t ic = idx / (P.KW * P.KH);
        const uint32_t ic = fastdiv(idx, P.KWKHmp, P.KWKHL);
        const uint32_t r  = idx - ic * (P.KW * P.KH);
        // const uint32_t kh = r / P.KW;
        const uint32_t kh = fastdiv(r, P.KWmp, P.KWL);
        const uint32_t kw = r - kh * P.KW;
        return T_ICKHKW{ ic, kh, kw };
    }

    __device__ __forceinline__ static T_NOHOW unpack_nohow(const uint32_t & idx) {
        // const uint32_t n  = idx / (P.OH * P.OW);
        const uint32_t n  = fastdiv(idx, P.OWOHmp, P.OWOHL);
        const uint32_t r  = idx - n * (P.OH * P.OW);
        // const uint32_t oh = r / P.OW;
        const uint32_t oh = fastdiv(r, P.OWmp, P.OWL);
        const uint32_t ow = r - oh * P.OW;
        return T_NOHOW{ n, oh, ow };
    }
};

using namespace ggml_cuda_mma;

typedef tile<WMMA_M, WMMA_K / 2, half2> tile_a;
typedef tile<WMMA_N, WMMA_K / 2, half2> tile_b;
typedef tile<WMMA_M, WMMA_N, float>     tile_acc;

// --> conv_2d kernel modified to function as a matmul
template <typename layout,
          const uint32_t BS_OC,
          const uint32_t BS_NOHOW,
          const uint32_t BS_ICKHKW,
          const uint32_t NUM_TILES_PER_WARP,
          const uint32_t NUM_WARPS_NEED,
          const uint32_t NUM_WARPS_NOHOW,
          const uint32_t NUM_WARPS,
          const uint32_t WG_SIZE>
__global__ void __launch_bounds__(NUM_WARPS * WARP_SIZE) conv2d_tensor_cores_kernel(const float * __restrict__ IN,
                                                                                    const half * __restrict__ IK,
                                                                                    float * __restrict__ Out) {
    const uint32_t warpId    = threadIdx.y;
    const uint32_t block_tid = threadIdx.y * blockDim.x + threadIdx.x;

    const uint32_t OC_BASE    = blockIdx.x * BS_OC;
    const uint32_t NOHOW_BASE = blockIdx.y * BS_NOHOW;

    __shared__ half A_sh[BS_OC * BS_ICKHKW];
    __shared__ half B_sh[BS_NOHOW * BS_ICKHKW];

    const uint32_t Ar = block_tid / BS_ICKHKW;
    const uint32_t Ac = block_tid % BS_ICKHKW;

    constexpr uint32_t ArpWg = WG_SIZE / BS_ICKHKW;

    const uint32_t Br = block_tid / BS_ICKHKW;
    const uint32_t Bc = block_tid % BS_ICKHKW;

    constexpr uint32_t BrpWg = WG_SIZE / BS_ICKHKW;

    tile_a   a_frag;
    tile_b   b_frag;
    tile_acc c_frag[NUM_TILES_PER_WARP];

#pragma unroll
    for (uint32_t id_ickhkw = 0; id_ickhkw < P.IC_KH_KW; id_ickhkw += BS_ICKHKW) {
        const uint32_t cached_ickhkw_idx = id_ickhkw + Ac;

        const T_ICKHKW ickhkw = layout::unpack_ickhkw(cached_ickhkw_idx);

#pragma unroll
        for (uint32_t i = 0; i < BS_OC; i += ArpWg) {
            const uint32_t gOC = OC_BASE + (Ar + i);
            half           val = IK[min(cached_ickhkw_idx + (gOC * P.IC_KH_KW), P.IK_TOTAL - 1)];

            if (((cached_ickhkw_idx) >= P.IC_KH_KW) || (gOC >= P.Cout)) {
                val = 0.0f;
            }
            A_sh[(i + Ar) * BS_ICKHKW + Ac] = val;
        }
#pragma unroll
        for (uint32_t i = 0; i < BS_NOHOW; i += BrpWg) {
            const uint32_t gNOHOW = NOHOW_BASE + (i + Br);
            half           val    = 0.0f;
            const T_NOHOW  nohow  = layout::unpack_nohow(gNOHOW);

            const int32_t in_y = calculate_input_coord(nohow.OH, ickhkw.kh, P.ST_Y, P.DL_Y, P.PD_Y);
            const int32_t in_x = calculate_input_coord(nohow.OW, ickhkw.kw, P.ST_X, P.DL_X, P.PD_X);

            val = ggml_cuda_cast<half>(
                IN[min(max(layout::input_index(nohow.B, ickhkw.ic, in_y, in_x), 0), P.IN_TOTAL - 1)]);
            if (in_y < 0 || in_y >= P.IH || in_x < 0 || in_x >= P.IW) {
                val = 0.0f;
            }
            B_sh[(i + Br) * BS_ICKHKW + Bc] = val;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t i = 0; i < NUM_TILES_PER_WARP; i++) {
            const uint32_t warp       = warpId * NUM_TILES_PER_WARP + i;
            const uint32_t WARP_OC    = warp / NUM_WARPS_NOHOW;
            const uint32_t WARP_NOHOW = warp % NUM_WARPS_NOHOW;

            const half * A_warp_base = A_sh + WARP_OC * WMMA_M * BS_ICKHKW;
            const half * B_warp_base = B_sh + WARP_NOHOW * WMMA_N * BS_ICKHKW;

#pragma unroll
            for (uint32_t k_tile = 0; k_tile < BS_ICKHKW; k_tile += WMMA_K) {
                const half * A_k_ptr = A_warp_base + k_tile;
                const half * B_k_ptr = B_warp_base + k_tile;
                load_ldmatrix(a_frag, (const half2 *) A_k_ptr, BS_ICKHKW / 2);
                load_ldmatrix(b_frag, (const half2 *) B_k_ptr, BS_ICKHKW / 2);
                ggml_cuda_mma::mma(c_frag[i], a_frag, b_frag);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint32_t i = 0; i < NUM_TILES_PER_WARP; i++) {
        const uint32_t warp            = warpId * NUM_TILES_PER_WARP + i;
        const uint32_t WARP_OC         = warp / NUM_WARPS_NOHOW;
        const uint32_t WARP_NOHOW      = warp % NUM_WARPS_NOHOW;
        const uint32_t OC_WARP_BASE    = OC_BASE + WARP_OC * WMMA_M;
        const uint32_t NOHOW_WARP_BASE = NOHOW_BASE + WARP_NOHOW * WMMA_N;
#pragma unroll
        for (uint32_t l = 0; l < tile_acc::ne; ++l) {
            const uint32_t e = tile_acc::get_i(l) * WMMA_N + tile_acc::get_j(l);
            const uint32_t m = e / WMMA_N;
            const uint32_t n = e % WMMA_N;

            const uint32_t oc        = OC_WARP_BASE + m;
            const uint32_t nohow     = NOHOW_WARP_BASE + n;
            const T_NOHOW  out_nohow = layout::unpack_nohow(nohow);
            if (oc < P.Cout && nohow < (P.N_OH_OW)) {
                Out[layout::output_index(out_nohow.B, oc, out_nohow.OH, out_nohow.OW)] = c_frag[i].x[l];
            }
        }
    }
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static void init_fastdiv_values(uint32_t d, uint32_t & mp, uint32_t & L) {
    // compute L = ceil(log2(d));
    L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
}

constexpr int conv_shapes[][NUM_VARIANTS] = {
    { 128, 64, 32  }, // BS_OC
    { 16,  32, 16  }, // BS_ICKHKW
    { 128, 32, 256 }, // BS_NOHOW
};

int get_sm_count() {
    int device;
    cudaGetDevice(&device);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    return sm_count;
}

template <uint CONV_SHAPE>
void conv_2d_tensor_core(const float *        src0,
                         const half *         src1,
                         float *              dst,
                         const Params &       p,
                         const cudaStream_t & st) {
    constexpr uint32_t WG_SIZE = 256;
    static_assert(WG_SIZE % WARP_SIZE == 0);

    constexpr uint32_t NUM_WARPS = WG_SIZE / WARP_SIZE;

    constexpr uint32_t BS_OC     = conv_shapes[0][CONV_SHAPE];
    constexpr uint32_t BS_ICKHKW = conv_shapes[1][CONV_SHAPE];
    constexpr uint32_t BS_NOHOW  = conv_shapes[2][CONV_SHAPE];

    static_assert(BS_OC % WMMA_M == 0 && BS_NOHOW % WMMA_N == 0);

    constexpr uint32_t NUM_WARPS_NEED  = (BS_OC * BS_NOHOW) / (WMMA_M * WMMA_N);
    constexpr uint32_t NUM_WARPS_NOHOW = BS_NOHOW / WMMA_N;

    static_assert(NUM_WARPS_NEED % NUM_WARPS == 0);

    constexpr uint32_t NUM_TILES_PER_WARP = NUM_WARPS_NEED / NUM_WARPS;

    const int64_t  NOHOW    = p.B * p.OW * p.OH;
    const uint32_t NB_OC    = CEIL_DIV(p.Cout, BS_OC);
    const uint32_t NB_NOHOW = CEIL_DIV(NOHOW, BS_NOHOW);

    cudaMemcpyToSymbolAsync(P, &p, sizeof(Params), 0, cudaMemcpyHostToDevice, st);

    dim3           gridDim(NB_OC, NB_NOHOW);
    constexpr dim3 blockDim(WARP_SIZE, NUM_WARPS);

    conv2d_tensor_cores_kernel<whcn_layout, BS_OC, BS_NOHOW, BS_ICKHKW, NUM_TILES_PER_WARP, NUM_WARPS_NEED,
                               NUM_WARPS_NOHOW, NUM_WARPS, WG_SIZE><<<gridDim, blockDim, 0, st>>>(src0, src1, dst);
}

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
                                     const cudaStream_t & st) {
    using Conv2DFuncPtr = void (*)(const float *, const half *, float *, const Params &, const cudaStream_t &);

    Conv2DFuncPtr conv2d_variants[NUM_VARIANTS];

    conv2d_variants[CONV_SHAPE_128x128] = &conv_2d_tensor_core<CONV_SHAPE_128x128>;
    conv2d_variants[CONV_SHAPE_64x32]   = &conv_2d_tensor_core<CONV_SHAPE_64x32>;
    conv2d_variants[CONV_SHAPE_32x256]  = &conv_2d_tensor_core<CONV_SHAPE_32x256>;

    Params p{};
    p.Cout = OC;
    p.Cin  = IC;
    p.B    = B;

    p.KW = KW;
    p.KH = KH;
    p.IW = IW;
    p.IH = IH;
    p.OW = OW;
    p.OH = OH;

    p.ST_X     = ST_X;
    p.ST_Y     = ST_Y;
    p.PD_X     = PD_X;
    p.PD_Y     = PD_Y;
    p.DL_X     = DL_X;
    p.DL_Y     = DL_Y;
    p.IC_KH_KW = IC * KH * KW;
    p.IK_TOTAL = p.IC_KH_KW * p.Cout;

    p.N_OH_OW  = B * OH * OW;
    p.IN_TOTAL = B * IC * IH * IW;

    init_fastdiv_values(p.KW, p.KWmp, p.KWL);
    init_fastdiv_values(p.KW * p.KH, p.KWKHmp, p.KWKHL);
    init_fastdiv_values(p.OW, p.OWmp, p.OWL);
    init_fastdiv_values(p.OW * p.OH, p.OWOHmp, p.OWOHL);

    // Problem size (Cout x NOHOW)
    std::array<uint32_t, 3> elements = { p.Cout, p.B * p.OW * p.OH, 1 };

    const uint32_t sm_count = get_sm_count();

    uint32_t variant_ntiles[NUM_VARIANTS];

    for (int var_id = 0; var_id < NUM_VARIANTS; var_id++) {
        const uint32_t ntilesy = ceil_div(elements[0], conv_shapes[var_id][0]);  // CEIL_DIV(Cout, NB_OC)
        const uint32_t ntilesx = ceil_div(elements[1], conv_shapes[var_id][2]);  // CEIL_DIV(NOHOW, NB_NOHOW)
        variant_ntiles[var_id] = ntilesy * ntilesx;
    }

    uint32_t selected_variant_id = CONV_SHAPE_128x128;

    if (elements[0] > 64 && variant_ntiles[CONV_SHAPE_128x128] >= sm_count * 2) {
        selected_variant_id = CONV_SHAPE_128x128;
    } else if (elements[0] <= 32 && variant_ntiles[CONV_SHAPE_32x256] >= sm_count * 2) {
        selected_variant_id = CONV_SHAPE_32x256;
    } else {
        selected_variant_id = CONV_SHAPE_64x32;
    }

    conv2d_variants[selected_variant_id](IN, IK, output, p, st);
}
