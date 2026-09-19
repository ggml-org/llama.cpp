#include "conv2d-mm.cuh"

#include <cuda_runtime.h>

// If defined, indices are computed once and re-used by each thread
#if __CUDA_ARCH__ < 700
#    define USE_COLLECTIVES
#endif

//#define A_TRANS       // Transposes the A matrix in shmem
//#define A_OPT         // Optimizes A for reducing bank conflicts
#define B_OPT  // Optimizes B for reducing bank conflicts

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

uint32_t ceil_div(uint32_t M, uint32_t N);
int      get_sm_count();

uint32_t ceil_div(uint32_t M, uint32_t N) {
    return (M + N - 1) / N;
}

__align__(16) struct Params {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t N;

    uint32_t KW;
    uint32_t KH;
    uint32_t W;
    uint32_t H;
    uint32_t OW;
    uint32_t OH;

    uint32_t s0;
    uint32_t s1;
    uint32_t p0;
    uint32_t p1;
    uint32_t d0;
    uint32_t d1;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;

    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;

    uint32_t nb1;
    uint32_t nb2;
    uint32_t nb3;

    uint32_t KWmp;
    uint32_t KWL;
    uint32_t KWKHmp;
    uint32_t KWKHL;
    uint32_t OWmp;
    uint32_t OWL;
    uint32_t OWOHmp;
    uint32_t OWOHL;
};

__constant__ __device__ Params dp;

// see init_fastdiv_values in ggml-vulkan.cpp
__inline__ __device__ uint fastdiv(uint n, uint mp, uint L) {
    return (__umulhi(n, mp) + n) >> L;
}

// --> conv_2d kernel modified to function as a matmul
template <uint BS_K, uint BS_NPQ, uint BS_CRS, uint TS_K, uint TS_NPQ, uint WG_SIZE, uint VEC_SIZE>
__global__ void __launch_bounds__(WG_SIZE, 1) mm(uint          K,
                                                 uint          NPQ,
                                                 uint          CRS,
                                                 const float * knl_data,
                                                 const float * src_data,
                                                 float *       dst_data) {
    // Each block computes a tile of the result of size BS_K*BS_NPQ
    const uint B_idx_K   = blockIdx.x;
    const uint B_idx_NPQ = blockIdx.y;
    assert(gridDim.z == 1);

    // T_y, T_x: the tile position this thread is resposible for computing.
    assert(BS_NPQ % TS_NPQ == 0);
    assert(TS_NPQ <= BS_NPQ);
    const uint NT_x = BS_NPQ / TS_NPQ;
    assert(BS_K % TS_K == 0);
    assert(TS_K <= BS_K);
    // const uint NT_y = BS_K / TS_K; // unused

    // Ensure that the kernel is properly called
    // 1. each thread processes a threadtile of size TS_K*TS_NPQ, that is exactly the WG_SIZE
    assert((BS_K / TS_K) * (BS_NPQ / TS_NPQ) == WG_SIZE);
    // 2. the number of threads is exactly the WG_SIZE
    assert(blockDim.x == WG_SIZE && blockDim.y == 1 && blockDim.z == 1);

    const uint T_y = threadIdx.x / NT_x;
    const uint T_x = threadIdx.x % NT_x;

    __shared__ float Ash[BS_K * BS_CRS];
    __shared__ float Bsh[BS_CRS * BS_NPQ];

    const uint Ar = threadIdx.x / BS_CRS;
    const uint Ac = threadIdx.x % BS_CRS;
    assert(WG_SIZE >= BS_CRS);
    const uint ArpWg = WG_SIZE / BS_CRS;

    const uint Br = threadIdx.x / BS_NPQ;
    const uint Bc = threadIdx.x % BS_NPQ;
    assert(WG_SIZE >= BS_NPQ);
    const uint BrpWg = WG_SIZE / BS_NPQ;

    float regA[TS_K]          = { 0.0 };
    float regB[TS_NPQ]        = { 0.0 };
    float regC[TS_K * TS_NPQ] = { 0.0 };

    /* Advance block in CRS dim */
    for (uint idx_CRS = 0; idx_CRS < CRS; idx_CRS += BS_CRS) {
/* Load kernel to A_block: (BS_K x BS_CRS)*/
#ifdef USE_COLLECTIVES
        const int laneId = threadIdx.x & 0x1f;
        // Each thread in CRS dim computes a result that will be broadcast among them
        assert(CRS <= warpSize);
        const uint32_t cached_CRS_idx = idx_CRS + laneId;
        const uint32_t cached_Cin_idx = cached_CRS_idx / (dp.KW * dp.KH);
        uint32_t       rem            = (cached_CRS_idx - cached_Cin_idx * dp.KW * dp.KH);
        const uint32_t cached_KH_idx  = rem / dp.KW;
        const uint32_t cached_KW_idx  = rem - cached_KH_idx * dp.KW;

        const uint32_t CRS_idx_a = __shfl_sync(0xffffffff, cached_CRS_idx, Ac);
        const uint32_t KH_idx_a  = __shfl_sync(0xffffffff, cached_KH_idx, Ac);
        //const uint32_t KW_idx_a = __shfl_sync(0xffffffff, cached_KW_idx, Ac); // unused
        const uint32_t Cin_idx_a = __shfl_sync(0xffffffff, cached_Cin_idx, Ac);
#else
        uint32_t CRS_idx_a     = idx_CRS + Ac;  //Global CRS_idx (column index of A)
        //uint32_t Cin_idx_a = CRS_idx_a / (dp.KW*dp.KH);
        uint32_t Cin_idx_a     = fastdiv(CRS_idx_a, dp.KWKHmp, dp.KWKHL);  // divide by (p.KW * p.KH); / (p.KW * p.KH);
        uint32_t CRS_remainder = CRS_idx_a - Cin_idx_a * dp.KW * dp.KH;
        //uint32_t KH_idx_a = (CRS_idx_a - Cin_idx_a*dp.KW*dp.KH) / dp.KW;
        uint32_t KH_idx_a      = fastdiv(CRS_remainder, dp.KWmp, dp.KWL);  // divide by p.KW;
//uint32_t KW_idx_a = CRS_idx_a - Cin_idx_a*dp.KW*dp.KH - KH_idx_a*dp.KW; // unused
#endif

#pragma unroll
        for (uint r_offset = 0; r_offset < BS_K; r_offset += ArpWg) {
            const uint32_t K_idx_a = B_idx_K * BS_K + r_offset + Ar; /* Global K_idx (row index of A)*/
            // General addressing (does not assume contiguity)
            //const uint32_t knl_idx = KW_idx_a + KH_idx_a*dp.nb01 + Cin_idx_a*dp.nb02 + K_idx_a*dp.nb03;
            // Contiguous addressing
            float          val     = knl_data[min(CRS_idx_a + K_idx_a * dp.nb03, K * CRS - 1)];
            if (CRS_idx_a >= CRS || K_idx_a >= K) {
                val = 0.0;
            }

#ifdef A_TRANS
#    ifdef A_OPT
            uint32_t T_id        = (r_offset + Ar) / TS_K;                      // E.g.: 41/16 = 2
            uint32_t vec_in_TT   = ((r_offset + Ar) - T_id * TS_K) / VEC_SIZE;  // E.g.: 41-2*16 =     9 -> 9/4 = 2
            uint32_t elem_in_vec = ((r_offset + Ar) - T_id * TS_K) % VEC_SIZE;  // E.g.:               9 -> 9%4 = 1
            uint32_t col_offset  = vec_in_TT * (NT_y * VEC_SIZE) + T_id * VEC_SIZE + elem_in_vec;
#    else
            uint32_t col_offset = (r_offset + Ar);
#    endif
            Ash[Ac * BS_K + col_offset] = val;
#else
            Ash[(r_offset + Ar) * BS_CRS + Ac] = val;
#endif
        }

#pragma unroll
        for (uint r_offset = 0; r_offset < BS_CRS; r_offset += BrpWg) {
            // Compute indices for N, OH, OW from NPQ_idx
            const uint32_t NPQ_idx       = B_idx_NPQ * BS_NPQ + Bc; /* Global NPQ index (column index of B) */
            //const uint32_t N_idx = NPQ_idx / (dp.OH*dp.OW);
            uint32_t       N_idx         = fastdiv(NPQ_idx, dp.OWOHmp, dp.OWOHL);  // divide by p.OH * p.OW;
            uint32_t       NPQ_remainder = NPQ_idx - N_idx * dp.OH * dp.OW;
            //const uint32_t OH_idx = (NPQ_idx - N_idx*dp.OH*dp.OW) / dp.OW;
            uint32_t       OH_idx        = fastdiv(NPQ_remainder, dp.OWmp, dp.OWL);  // divide by p.OW;
            const uint32_t OW_idx        = NPQ_idx - N_idx * dp.OH * dp.OW - OH_idx * dp.OW;

#ifdef USE_COLLECTIVES
            const uint32_t CRS_idx_b = __shfl_sync(0xffffffff, cached_CRS_idx, r_offset + Br);
            const uint32_t KH_idx_b  = __shfl_sync(0xffffffff, cached_KH_idx, r_offset + Br);
            const uint32_t KW_idx_b  = __shfl_sync(0xffffffff, cached_KW_idx, r_offset + Br);
            const uint32_t Cin_idx_b = __shfl_sync(0xffffffff, cached_Cin_idx, r_offset + Br);
#else
            // Compute indices KH, KW, Cin from CRS_idx
            uint32_t CRS_idx_b = idx_CRS + r_offset + Br;
            //uint32_t Cin_idx_b = CRS_idx_b / (dp.KW*dp.KH);
            uint32_t Cin_idx_b = fastdiv(CRS_idx_b, dp.KWKHmp, dp.KWKHL);  // divide by (p.KW * p.KH); / (p.KW * p.KH);
            uint32_t CRS_remainder = CRS_idx_b - Cin_idx_b * dp.KW * dp.KH;
            //uint32_t KH_idx_b = (CRS_idx_b - Cin_idx_b*dp.KW*dp.KH) / dp.KW;
            uint32_t KH_idx_b      = fastdiv(CRS_remainder, dp.KWmp, dp.KWL);  // divide by p.KW;
            uint32_t KW_idx_b      = CRS_idx_b - Cin_idx_b * dp.KW * dp.KH - KH_idx_b * dp.KW;
#endif

            // Compute indices for W, H from OH, OW, KH, KW
            const int32_t  H_idx   = OH_idx * dp.s1 + KH_idx_b * dp.d1 - dp.p1;
            const int32_t  W_idx   = OW_idx * dp.s0 + KW_idx_b * dp.d0 - dp.p0;
            const uint32_t src_idx = min(max(W_idx + H_idx * dp.nb11 + Cin_idx_b * dp.nb12 + N_idx * dp.nb13, 0),
                                         dp.Cin * dp.N * dp.W * dp.H - 1);
            float          val;
            if (CRS_idx_b >= CRS || NPQ_idx >= NPQ || H_idx < 0 || H_idx >= dp.H || W_idx < 0 || W_idx >= dp.W) {
                val = 0.0;
            } else {
                val = src_data[src_idx];
            }

#ifdef B_OPT
            assert(VEC_SIZE <= TS_NPQ);
            const uint32_t T_id        = Bc / TS_NPQ;                      // E.g.: 41/16 = 2
            const uint32_t vec_in_TT   = (Bc - T_id * TS_NPQ) / VEC_SIZE;  // E.g.: 41-2*16 =     9 -> 9/4 = 2
            const uint32_t elem_in_vec = (Bc - T_id * TS_NPQ) % VEC_SIZE;  // E.g.:               9 -> 9%4 = 1
            const uint32_t col_offset  = vec_in_TT * (NT_x * VEC_SIZE) + T_id * VEC_SIZE + elem_in_vec;
#else
            uint32_t col_offset = Bc;
#endif
            Bsh[(r_offset + Br) * BS_NPQ + col_offset] = val;
        }

        __syncthreads();

        if (T_y * TS_K < K) {
#pragma unroll
            for (uint32_t CRS_lidx = 0; CRS_lidx < BS_CRS; ++CRS_lidx) {
#pragma unroll
                for (uint32_t T_ly = 0; T_ly < TS_K; ++T_ly) {
#ifdef A_TRANS
#    ifdef A_OPT
                    uint32_t T_id        = T_y;
                    uint32_t vec_in_TT   = T_ly / VEC_SIZE;
                    uint32_t elem_in_vec = T_ly % VEC_SIZE;
                    uint32_t col_offset  = vec_in_TT * (NT_y * VEC_SIZE) + T_id * VEC_SIZE + elem_in_vec;
#    else
                    uint32_t col_offset = (T_y * TS_K + T_ly);
#    endif
                    regA[T_ly] = Ash[CRS_lidx * BS_K + col_offset];
#else
                    regA[T_ly] = Ash[(T_y * TS_K + T_ly) * BS_CRS + CRS_lidx];
#endif
                }
                for (uint32_t T_lx = 0; T_lx < TS_NPQ; ++T_lx) {
#ifdef B_OPT
                    const uint32_t T_id        = T_x;
                    const uint32_t vec_in_TT   = T_lx / VEC_SIZE;
                    const uint32_t elem_in_vec = T_lx % VEC_SIZE;
                    const uint32_t col_offset  = vec_in_TT * (NT_x * VEC_SIZE) + T_id * VEC_SIZE + elem_in_vec;
#else
                    const uint32_t col_offset = T_x * TS_NPQ + T_lx;
#endif
                    regB[T_lx] = Bsh[CRS_lidx * BS_NPQ + col_offset];
                }
                for (uint32_t T_ly = 0; T_ly < TS_K; ++T_ly) {
                    for (uint32_t T_lx = 0; T_lx < TS_NPQ; ++T_lx) {
                        regC[T_ly * TS_NPQ + T_lx] = fmaf(regA[T_ly], regB[T_lx], regC[T_ly * TS_NPQ + T_lx]);
                    }
                }
            }
        }
        __syncthreads();
    }

    /* Save C* */
    for (uint32_t T_ly = 0; T_ly < TS_K; T_ly++) {
        for (uint32_t T_lx = 0; T_lx < TS_NPQ; T_lx++) {
            const uint32_t K_idx     = B_idx_K * BS_K + T_y * TS_K + T_ly;
            const uint32_t NPQ_idx_c = B_idx_NPQ * BS_NPQ + T_x * TS_NPQ + T_lx;
            //const uint32_t N_idx_c = NPQ_idx_c / (dp.OH*dp.OW);
            const uint32_t N_idx_c   = fastdiv(NPQ_idx_c, dp.OWOHmp, dp.OWOHL);  // divide by p.OH * p.OW;
            //const uint32_t OH_idx_c = (NPQ_idx_c - N_idx_c*dp.OH*dp.OW) / dp.OW;
            const uint32_t OH_idx_c = fastdiv(NPQ_idx_c - N_idx_c * dp.OH * dp.OW, dp.OWmp, dp.OWL);  // divide by p.OW;
            const uint32_t OW_idx_c = NPQ_idx_c - N_idx_c * dp.OH * dp.OW - OH_idx_c * dp.OW;
            const uint32_t dst_idx  = OW_idx_c + OH_idx_c * dp.nb1 + K_idx * dp.nb2 + N_idx_c * dp.nb3;
            if (K_idx < K && NPQ_idx_c < NPQ) {
                dst_data[dst_idx] = regC[T_ly * TS_NPQ + T_lx];
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
    { 128, 64, 32  }, // BS_K
    { 16,  32, 16  }, // BS_CRS
    { 128, 32, 256 }, // BS_NPQ
    { 8,   4,  8   }  // TS_K
    //{8,	8,	8}	// TS_NPQ		// Option 2
};

int get_sm_count() {
    int device;
    cudaGetDevice(&device);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    return sm_count;
}

template <uint CONV_SHAPE>
void ggml_cuda_op_conv_2d_variant(ggml_backend_cuda_context & ctx,
                                  ggml_tensor *               src0,
                                  ggml_tensor *               src1,
                                  ggml_tensor *               dst,
                                  const Params &              p) {
    // Tile size calculation options:
    // Option 1: fix block size and all tile sizes except TS_NPQ as it is the free parameter (used in the Vulkan backend).
    // Option 2: fix all tile sizes and block size is the free parameter.
    const uint32_t WG_SIZE = 256;  // Option 1

    const uint32_t BS_K   = conv_shapes[0][CONV_SHAPE];
    const uint32_t BS_CRS = conv_shapes[1][CONV_SHAPE];
    const uint32_t BS_NPQ = conv_shapes[2][CONV_SHAPE];
    const uint32_t TS_K   = conv_shapes[3][CONV_SHAPE];
    //const uint32_t TS_NPQ = sh[4][CONV_SHAPE];			// Option 2
    const uint32_t TS_NPQ = BS_K * BS_NPQ / WG_SIZE / TS_K;

    // Some architectures can use 128-bit loads that might be more efficient.
    const uint32_t VEC_SIZE = TS_NPQ >= 4 ? 4 : 1;

    //const uint32_t WG_SIZE = (BS_K*BS_NPQ) / (TS_K*TS_NPQ);		// Option 2

    // Kernel runtime parameters
    int64_t  NPQ    = p.N * p.OW * p.OH;
    uint32_t NB_K   = CEIL_DIV(p.Cout, BS_K);
    uint32_t NB_NPQ = CEIL_DIV(NPQ, BS_NPQ);

    cudaMemcpyToSymbol(dp, &p, sizeof(Params));

    // Kernel arguments
    float * src0_data = (float *) src0->data;
    float * src1_data = (float *) src1->data;
    float * dst_data  = (float *) dst->data;

    dim3         gridDim(NB_K, NB_NPQ);
    dim3         blockDim(WG_SIZE);
    cudaStream_t stream = ctx.stream();

    mm<BS_K, BS_NPQ, BS_CRS, TS_K, TS_NPQ, WG_SIZE, VEC_SIZE>
        <<<gridDim, blockDim, 0, stream>>>(p.Cout, NPQ, p.Cin * p.KW * p.KH, src0_data, src1_data, dst_data);
}

void ggml_cuda_op_conv2d_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Initialize kernel variants

    using Conv2DFuncPtr =
        void (*)(ggml_backend_cuda_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *, const Params &);

    Conv2DFuncPtr conv2d_variants[NUM_VARIANTS];

    conv2d_variants[CONV_SHAPE_128x128] = &ggml_cuda_op_conv_2d_variant<CONV_SHAPE_128x128>;
    conv2d_variants[CONV_SHAPE_64x32]   = &ggml_cuda_op_conv_2d_variant<CONV_SHAPE_64x32>;
    conv2d_variants[CONV_SHAPE_32x256]  = &ggml_cuda_op_conv_2d_variant<CONV_SHAPE_32x256>;

    // Parse op input, prepare kernel input

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    Params p{};
    p.Cout = static_cast<uint32_t>(ne03);
    p.Cin  = static_cast<uint32_t>(ne02);
    p.N    = static_cast<uint32_t>(ne13);

    p.KW = static_cast<uint32_t>(ne00);
    p.KH = static_cast<uint32_t>(ne01);
    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.s0 = static_cast<uint32_t>(dst->op_params[0]);
    p.s1 = static_cast<uint32_t>(dst->op_params[1]);
    p.p0 = static_cast<uint32_t>(dst->op_params[2]);
    p.p1 = static_cast<uint32_t>(dst->op_params[3]);
    p.d0 = static_cast<uint32_t>(dst->op_params[4]);
    p.d1 = static_cast<uint32_t>(dst->op_params[5]);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    init_fastdiv_values(p.KW, p.KWmp, p.KWL);
    init_fastdiv_values(p.KW * p.KH, p.KWKHmp, p.KWKHL);
    init_fastdiv_values(p.OW, p.OWmp, p.OWL);
    init_fastdiv_values(p.OW * p.OH, p.OWOHmp, p.OWOHL);

    GGML_ASSERT(ne03 == ne2);
    GGML_ASSERT(ne02 == ne12);

    // Select the proper variant based on problem size and device parameters (sm count)

    // Problem size (Cout x NPQ)
    std::array<uint32_t, 3> elements = { p.Cout, p.N * p.OW * p.OH, 1 };

    const uint32_t sm_count = get_sm_count();

    uint32_t variant_ntiles[NUM_VARIANTS];

    for (int var_id = 0; var_id < NUM_VARIANTS; var_id++) {
        const uint32_t ntilesy = ceil_div(elements[0], conv_shapes[var_id][0]);  // CEIL_DIV(Cout, NB_K)
        const uint32_t ntilesx = ceil_div(elements[1], conv_shapes[var_id][2]);  // CEIL_DIV(NPQ, NB_NPQ)
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

    conv2d_variants[selected_variant_id](ctx, src0, src1, dst, p);
}
