#include "mmvq-q8-cdna.cuh"
#include "ggml-cuda/quantize.cuh"

#include <algorithm>
#include <climits>

// Q8_0 matrix-vector product for few src1 columns on CDNA1, the src1 columns are quantized to q8_1.
// A group of lpr lanes works on 64/lpr rows per wave, lane j handles the blocks j + lpr*s, s < nsteps, of its row.
// So 16 lanes read 16 consecutive blocks of 34 bytes, few 64 byte lines per load.
// The waves are persistent and loop over the row groups, the loads of the next group are issued before the current one is computed.
// src1 stays in LDS for the whole kernel.

static constexpr int MMVQ_Q8_CDNA_NWARPS    = 4;
static constexpr int MMVQ_Q8_CDNA_MAX_NCOLS = 8;
static constexpr int MMVQ_Q8_CDNA_MAX_NB     = 2*64*MMVQ_Q8_CDNA_NWARPS; // blocks per row, see mmvq_q8_cdna_get_config

// the columns are computed in pairs, an odd count > 1 was slower
static constexpr __host__ __device__ int mmvq_q8_cdna_ncp(const int ncols) {
    return ncols == 1 ? 1 : ncols + ncols % 2;
}

#if defined(CDNA)
// the raw data of a lane for one block, off is the block address % 4 (0 or 2)
struct mmvq_q8_cdna_x {
    int4 a; // bytes [-off, 16 - off) of the block
    int4 b; // bytes [16 - off, 32 - off)
    int  c; // bytes [30, 34), the last 4 qs
};

// 16 bytes from a 4 byte aligned address
static __device__ __forceinline__ int4 mmvq_q8_cdna_load_b4(const void * p) {
    int4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 4), sizeof(int4));
    return v;
}

// 4 bytes from a 2 byte aligned address
static __device__ __forceinline__ int mmvq_q8_cdna_load_b2(const void * p) {
    int v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), sizeof(int));
    return v;
}

// sum over the n lanes of a group, the result is in all lanes of the group
template <int n>
static __device__ __forceinline__ float mmvq_q8_cdna_group_sum(float x) {
    // DPP: xor 1, xor 2, row_half_mirror, row_mirror
    if constexpr (n >= 2) {
        x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));
    }
    if constexpr (n >= 4) {
        x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));
    }
    if constexpr (n >= 8) {
        x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x141, 0xF, 0xF, false));
    }
    if constexpr (n >= 16) {
        x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x140, 0xF, 0xF, false));
    }
    if constexpr (n >= 32) {
        const float s0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x),  0));
        const float s1 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 16));
        const float s2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 32));
        const float s3 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 48));
        if constexpr (n == 32) {
            x = threadIdx.x < 32 ? s0 + s1 : s2 + s3;
        } else {
            x = (s0 + s1) + (s2 + s3);
        }
    }
    static_assert(n <= 64, "unsupported group size");
    return x;
}
#endif // defined(CDNA)

// ks == 1: a group of lpr lanes of a wave per row, the waves work on different rows.
// ks  > 1: the ks waves of a block split the row, each wave then needs lpr == 64. For matrices with few rows.
template <int ncols, int nsteps, int lpr, int ks>
__launch_bounds__(MMVQ_Q8_CDNA_NWARPS*64, 1)
static __global__ void mul_mat_vec_q8_cdna(
        const char * __restrict__ x, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int nb, const int nrows, const int stride_row, const int stride_col_y, const int stride_col_dst) {
#if defined(CDNA)
    constexpr int nwarps = MMVQ_Q8_CDNA_NWARPS;
    static_assert(ks == 1 || (ks == nwarps && lpr == 64), "unsupported split");
    constexpr int lprt   = lpr*ks;                   // lanes per row
    constexpr int gpw    = ks == 1 ? 64/lpr : 1;     // rows per group of a wave resp. block
    constexpr int ncp    = mmvq_q8_cdna_ncp(ncols);

    // ks == 1: src1 in LDS. ks > 1: a block works on few rows, the src1 values of a lane are kept in registers.
    constexpr bool y_lds = ks == 1;
    static_assert(y_lds || ncols == 1, "too many registers");

    extern __shared__ int4 data_mmvq_q8_cdna[];
    int4  * yqa = data_mmvq_q8_cdna;            // qs [0, 16) of block b of column c at c*nb + b
    int4  * yqb = yqa + ncp*nb;                 // qs [16, 32)
    float * yd  = (float *) (yqb + ncp*nb);     // d
    float * ps  = (float *) data_mmvq_q8_cdna;  // ks > 1: the sums of the waves, 2 buffers used in turn

    const int lane    = threadIdx.x;
    const int iw      = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int j       = ks == 1 ? lane % lpr : iw*64 + lane;
    const int unit    = ks == 1 ? blockIdx.x*nwarps + iw : blockIdx.x;  // first row group
    const int nunits  = ks == 1 ? gridDim.x*nwarps : gridDim.x;
    const int ngroups = (nrows + gpw - 1)/gpw;
    const int last    = nb - 1 - j; // steps beyond the row load its last block again

    // rows beyond nrows load the last row again. 32 bit offsets from the uniform x take fewer registers than 64 bit addresses.
    auto load = [&](mmvq_q8_cdna_x (&xb)[nsteps], int & off, const int g) {
        const int      row = min(g*gpw + (ks == 1 ? lane/lpr : 0), nrows - 1);
        const uint32_t o   = row*stride_row + 34*j;
        off = o % 4;
#pragma unroll
        for (int s = 0; s < nsteps; ++s) {
            // ob % 4 == off except for the repeated last block, its result is not used
            const uint32_t ob = o + 34*min(lprt*s, last);
            xb[s].a = mmvq_q8_cdna_load_b4(x + (ob & ~3u));
            xb[s].b = mmvq_q8_cdna_load_b4(x + (ob & ~3u) + 16);
            xb[s].c = mmvq_q8_cdna_load_b2(x + (ob + 30));
        }
    };

    mmvq_q8_cdna_x xb0[nsteps];
    mmvq_q8_cdna_x xb1[nsteps];
    int off0;
    int off1;
    load(xb0, off0, unit);

    int4  yra[y_lds ? 1 : ncp*nsteps];
    int4  yrb[y_lds ? 1 : ncp*nsteps];
    float yrd[y_lds ? 1 : ncp*nsteps];
    if constexpr (y_lds) {
        for (int i = threadIdx.y*64 + lane; i < ncp*nb; i += nwarps*64) {
            const int c = i / nb;
            const int b = i % nb;
            int4  qa = make_int4(0, 0, 0, 0);
            int4  qb = make_int4(0, 0, 0, 0);
            float d  = 0.0f;
            if (c < ncols) {
                const block_q8_1 * by = y + c*stride_col_y + b;
                qa = mmvq_q8_cdna_load_b4(by->qs +  0);
                qb = mmvq_q8_cdna_load_b4(by->qs + 16);
                d  = __low2float(by->ds);
            }
            yqa[i] = qa;
            yqb[i] = qb;
            yd[i]  = d;
        }
        __syncthreads();
    } else {
#pragma unroll
        for (int c = 0; c < ncp; ++c) {
#pragma unroll
            for (int s = 0; s < nsteps; ++s) {
                const block_q8_1 * by = y + c*stride_col_y + j + min(lprt*s, last);
                yra[c*nsteps + s] = mmvq_q8_cdna_load_b4(by->qs +  0);
                yrb[c*nsteps + s] = mmvq_q8_cdna_load_b4(by->qs + 16);
                yrd[c*nsteps + s] = __low2float(by->ds);
            }
        }
    }

    // lane j keeps the result of column min(j, ncols - 1), all lanes store: a branch at the store makes the waits for the loads conservative
    const int jc = min(j, ncols - 1);
    auto compute = [&](const mmvq_q8_cdna_x (&xb)[nsteps], const int off, const int g, const int buf) {
        int   qs[nsteps][8];
        float dx[nsteps];
        const unsigned int sel = off == 0 ? 0x05040302 : 0x07060504; // bytes [2 + off, 6 + off) of w[i], w[i + 1]
#pragma unroll
        for (int s = 0; s < nsteps; ++s) {
            const int w[8] = {xb[s].a.x, xb[s].a.y, xb[s].a.z, xb[s].a.w, xb[s].b.x, xb[s].b.y, xb[s].b.z, xb[s].b.w};
#pragma unroll
            for (int i = 0; i < 7; ++i) {
                qs[s][i] = __builtin_amdgcn_perm(w[i + 1], w[i], sel);
            }
            qs[s][7] = xb[s].c;
            const int d = __builtin_amdgcn_alignbyte(w[1], w[0], off) & 0xFFFF;
            dx[s] = lprt*s <= last ? __half2float(__ushort_as_half((unsigned short) d)) : 0.0f;
        }

        float res = 0.0f;
#pragma unroll 2
        for (int c = 0; c < ncp; ++c) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < nsteps; ++s) {
                const int   ib = c*nb + j + min(lprt*s, last);
                const int4  ua = y_lds ? yqa[ib] : yra[c*nsteps + s];
                const int4  ub = y_lds ? yqb[ib] : yrb[c*nsteps + s];
                const float dy = y_lds ? yd[ib]  : yrd[c*nsteps + s];
                int sumi = 0;
                sumi = ggml_cuda_dp4a(qs[s][0], ua.x, sumi);
                sumi = ggml_cuda_dp4a(qs[s][1], ua.y, sumi);
                sumi = ggml_cuda_dp4a(qs[s][2], ua.z, sumi);
                sumi = ggml_cuda_dp4a(qs[s][3], ua.w, sumi);
                sumi = ggml_cuda_dp4a(qs[s][4], ub.x, sumi);
                sumi = ggml_cuda_dp4a(qs[s][5], ub.y, sumi);
                sumi = ggml_cuda_dp4a(qs[s][6], ub.z, sumi);
                sumi = ggml_cuda_dp4a(qs[s][7], ub.w, sumi);
                sum += dx[s]*dy*float(sumi);
            }
            sum = mmvq_q8_cdna_group_sum<lpr>(sum);
            if constexpr (ks == 1) {
                res = c == jc ? sum : res;
            } else {
                ps[(buf*ncp + c)*ks + iw] = sum;
            }
        }
        if constexpr (ks > 1) {
            __syncthreads();
#pragma unroll
            for (int k = 0; k < ks; ++k) {
                res += ps[(buf*ncp + jc)*ks + k];
            }
        }
        // rows beyond nrows store the last row again, with the same value
        const int row = min(g*gpw + (ks == 1 ? lane/lpr : 0), nrows - 1);
        dst[jc*stride_col_dst + row] = res;
    };

    int g = unit;
    for (; g + nunits < ngroups; g += 2*nunits) {
        load(xb1, off1, g + nunits);
        compute(xb0, off0, g, 0);
        load(xb0, off0, g + 2*nunits);
        compute(xb1, off1, g + nunits, 1);
    }
    if (g < ngroups) {
        compute(xb0, off0, g, 0);
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(CDNA)
}

// lanes per row and wave, steps per lane, waves per row
struct mmvq_q8_cdna_config {
    int lpr;
    int nsteps;
    int ks;
};

static mmvq_q8_cdna_config mmvq_q8_cdna_get_config(const int64_t nb, const int64_t nrows, const int nsm) {
    if (nb <= 16) {
        return {16, 1, 1};
    }
    if (nb <= 32) {
        return {16, 2, 1};
    }
    // more steps take too many registers. With few rows a wave per row gives less than 1 wave per SIMD.
    if (nb <= 2*64 || (nb <= 3*64 && nrows >= 4*nsm)) {
        return {64, int((nb + 63)/64), 1};
    }
    // the waves of a block split each row
    return {64, int((nb + 64*MMVQ_Q8_CDNA_NWARPS - 1)/(64*MMVQ_Q8_CDNA_NWARPS)), MMVQ_Q8_CDNA_NWARPS};
}

static size_t mmvq_q8_cdna_nbytes_shared(const int ncols, const int nb, const mmvq_q8_cdna_config cfg) {
    const int ncp = mmvq_q8_cdna_ncp(ncols);
    return cfg.ks == 1 ? ncp*nb*(2*sizeof(int4) + sizeof(float)) : 2*ncp*cfg.ks*sizeof(float);
}

template <int ncols, int nsteps, int lpr, int ks>
static void launch_mul_mat_vec_q8_cdna(
        const char * x, const block_q8_1 * y, float * dst, const int nb, const int nrows, const int stride_row,
        const int stride_col_y, const int stride_col_dst, const int nsm, cudaStream_t stream) {
    constexpr int gpw = ks == 1 ? 64/lpr : 1;
    constexpr int upb = ks == 1 ? MMVQ_Q8_CDNA_NWARPS : 1; // row groups per block
    const int ngroups = (nrows + gpw - 1)/gpw;

    const dim3   block_dims(64, MMVQ_Q8_CDNA_NWARPS, 1);
    const size_t nbytes_shared = mmvq_q8_cdna_nbytes_shared(ncols, nb, {lpr, nsteps, ks});

    // all blocks at once, each wave gets the same number of row groups
    int max_blocks_per_cu = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, mul_mat_vec_q8_cdna<ncols, nsteps, lpr, ks>,
        block_dims.x*block_dims.y, nbytes_shared));
    const int max_units = std::max(max_blocks_per_cu, 1)*nsm*upb;
    const int nrounds   = (ngroups + max_units - 1)/max_units;
    const int nunits    = (ngroups + nrounds - 1)/nrounds;

    const dim3   block_nums((nunits + upb - 1)/upb, 1, 1);
    mul_mat_vec_q8_cdna<ncols, nsteps, lpr, ks><<<block_nums, block_dims, nbytes_shared, stream>>>
        (x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst);
}

template <int ncols>
static void mul_mat_vec_q8_cdna_switch_config(
        const char * x, const block_q8_1 * y, float * dst, const int nb, const int nrows, const int stride_row,
        const int stride_col_y, const int stride_col_dst, const int nsm, cudaStream_t stream) {
    const mmvq_q8_cdna_config cfg = mmvq_q8_cdna_get_config(nb, nrows, nsm);
    if constexpr (ncols == 1) {
        if (cfg.ks > 1) {
            GGML_ASSERT(cfg.ks == MMVQ_Q8_CDNA_NWARPS && cfg.lpr == 64);
            switch (cfg.nsteps) {
                case 1: launch_mul_mat_vec_q8_cdna<ncols, 1, 64, MMVQ_Q8_CDNA_NWARPS>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
                case 2: launch_mul_mat_vec_q8_cdna<ncols, 2, 64, MMVQ_Q8_CDNA_NWARPS>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
                default: GGML_ABORT("fatal error");
            }
            return;
        }
    }
    GGML_ASSERT(cfg.ks == 1);
    switch (cfg.lpr*16 + cfg.nsteps) {
        case 16*16 + 1: launch_mul_mat_vec_q8_cdna<ncols, 1, 16, 1>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
        case 16*16 + 2: launch_mul_mat_vec_q8_cdna<ncols, 2, 16, 1>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
        case 64*16 + 1: launch_mul_mat_vec_q8_cdna<ncols, 1, 64, 1>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
        case 64*16 + 2: launch_mul_mat_vec_q8_cdna<ncols, 2, 64, 1>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
        case 64*16 + 3: launch_mul_mat_vec_q8_cdna<ncols, 3, 64, 1>(x, y, dst, nb, nrows, stride_row, stride_col_y, stride_col_dst, nsm, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

bool ggml_hip_should_use_mmvq_q8_cdna(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!GGML_CUDA_CC_IS_CDNA1(cc) || src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const int64_t ncols = src1->ne[1];
    const int64_t nb    = src0->ne[0]/QK8_0;
    if (ncols < 1 || ncols > MMVQ_Q8_CDNA_MAX_NCOLS || nb > MMVQ_Q8_CDNA_MAX_NB) {
        return false;
    }
    // the few-row MMQ (rows of at least 8*QK_K values) is faster from 7 columns, from 5 columns for large matrices: it has the
    // higher throughput once running, this kernel starts faster
    if (src0->ne[0] >= 8*QK_K && (ncols > 6 || (ncols > 4 && ggml_nbytes(src0) > 64*1024*1024))) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // the blocks are read with 4 byte loads from 4 byte aligned rows, or 2 bytes before 2 byte aligned blocks
    if ((uintptr_t) src0->data % 4 != 0 || src0->nb[1] % 2 != 0 || ggml_nbytes(src0) > INT_MAX) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const mmvq_q8_cdna_config cfg = mmvq_q8_cdna_get_config(nb, src0->ne[1], nsm);
    // MMVQ is as fast with 2 columns and long rows, 3 steps with 2 columns take too many registers
    if ((cfg.ks > 1 && ncols > 1) || (ncols == 2 && nb > 2*64)) {
        return false;
    }
    // with more LDS a CU fits only one block of 4 waves, too few for the memory latency
    return mmvq_q8_cdna_nbytes_shared(ncols, nb, cfg) <= 32*1024;
}

void ggml_hip_mul_mat_vec_q8_cdna(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne11 >= 1 && ne11 <= MMVQ_Q8_CDNA_MAX_NCOLS);

    cudaStream_t stream = ctx.stream();
    const int    nsm    = ggml_cuda_info().devices[ctx.device].nsm;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne11*ne10_padded*sizeof(block_q8_1)/QK8_1);
    quantize_row_q8_1_cuda((const float *) src1->data, src1_q8_1.get(), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
        ne10_padded, ne11, 1, 1, stream);

    const char       * x = (const char *) src0->data;
    const block_q8_1 * y = (const block_q8_1 *) src1_q8_1.get();
    float            * d = (float *) dst->data;

    const int nb             = ne00/QK8_0;
    const int stride_col_y   = ne10_padded/QK8_1;
    const int stride_col_dst = nb1/sizeof(float);

    switch (ne11) {
        case 1: mul_mat_vec_q8_cdna_switch_config<1>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 2: mul_mat_vec_q8_cdna_switch_config<2>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 3: mul_mat_vec_q8_cdna_switch_config<3>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 4: mul_mat_vec_q8_cdna_switch_config<4>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 5: mul_mat_vec_q8_cdna_switch_config<5>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 6: mul_mat_vec_q8_cdna_switch_config<6>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 7: mul_mat_vec_q8_cdna_switch_config<7>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        case 8: mul_mat_vec_q8_cdna_switch_config<8>(x, y, d, nb, ne01, nb01, stride_col_y, stride_col_dst, nsm, stream); break;
        default: GGML_ABORT("fatal error");
    }
}
