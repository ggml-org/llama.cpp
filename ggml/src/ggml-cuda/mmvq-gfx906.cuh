#pragma once

// MMVQ kernels for Q1_0 and Q2_0 on gfx906 (MI50 / MI60 / Radeon VII), 1-4 src1 columns.
// Each lane owns fixed 32-weight chunks of a row and keeps their q8_1 values in registers for all columns.
// The weights of 2 rows are loaded before any math. One lane loads the scale of a block and shuffles it to the others.
// Weight loads use 2-byte aligned addresses, gfx9 handles them in unaligned mode.

#include "common.cuh"
#include "unary.cuh"

#if defined(GGML_USE_HIP)

static constexpr int mmvq_gfx906_rows_per_block = 2;

static __device__ __forceinline__ float mmvq_gfx906_h2f(const uint32_t w) {
    const uint16_t b = (uint16_t) (w & 0xFFFFu);
    half h;
    __builtin_memcpy(&h, &b, sizeof(h));
    return __half2float(h);
}

// one q8_1 block, its scale and the sum of its quants
struct mmvq_gfx906_y {
    int   q[8];
    float d;
    int   sum;
};

static __device__ __forceinline__ void mmvq_gfx906_load_y(const block_q8_1 * y, mmvq_gfx906_y & v) {
    const int4 a = ((const int4 *) y->qs)[0];
    const int4 b = ((const int4 *) y->qs)[1];
    v.q[0] = a.x; v.q[1] = a.y; v.q[2] = a.z; v.q[3] = a.w;
    v.q[4] = b.x; v.q[5] = b.y; v.q[6] = b.z; v.q[7] = b.w;
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        sum = ggml_cuda_dp4a(0x01010101, v.q[k], sum);
    }
    v.d   = __low2float(y->ds);
    v.sum = sum;
}

template <ggml_type type> struct mmvq_gfx906_type;

// 2 blocks = 256 weights = 8 chunks. bit 1 = +d, bit 0 = -d, so dot = 2 * sum(y where bit is 1) - sum(y).
template <> struct mmvq_gfx906_type<GGML_TYPE_Q1_0> {
    static constexpr int bytes_per_256 = 2 * sizeof(block_q1_0);
    struct w_t { uint32_t d; uint32_t q; };

    static __device__ __forceinline__ void load(const uint8_t * sb, const int c, w_t & w) {
        const uint8_t * b = sb + sizeof(block_q1_0) * (c >> 2);
        w.d = (c & 3) == 0 ? (uint32_t) *(const uint16_t *) b : 0u;
        w.q = *(const uint32_t *) (b + 2 + 4 * (c & 3));
    }
    static __device__ __forceinline__ int d_lane(const int c) { return c & 4; }

    static __device__ __forceinline__ float dot(const w_t & w, const float d, const mmvq_gfx906_y & y) {
        int r = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t bits = (((w.q >> (4 * m)) & 0xFu) * 0x00204081u) & 0x01010101u; // 4 bits -> 4 bytes of 0/1
            r = ggml_cuda_dp4a((int) bits, y.q[m], r);
        }
        return d * y.d * (float) (2 * r - y.sum);
    }
};

// 4 blocks = 256 weights = 8 chunks. value = crumb - 1, so dot = sum(crumb * y) - sum(y).
template <> struct mmvq_gfx906_type<GGML_TYPE_Q2_0> {
    static constexpr int bytes_per_256 = 4 * sizeof(block_q2_0);
    struct w_t { uint32_t d; uint2 q; };

    static __device__ __forceinline__ void load(const uint8_t * sb, const int c, w_t & w) {
        const uint8_t * b = sb + sizeof(block_q2_0) * (c >> 1);
        w.d = (c & 1) == 0 ? (uint32_t) *(const uint16_t *) b : 0u;
        w.q = *(const uint2 *) (b + 2 + 8 * (c & 1));
    }
    static __device__ __forceinline__ int d_lane(const int c) { return c & 6; }

    static __device__ __forceinline__ float dot(const w_t & w, const float d, const mmvq_gfx906_y & y) {
        int r = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t b = ((m < 4 ? w.q.x : w.q.y) >> (8 * (m & 3))) & 0xFFu;
            const uint32_t q = (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u; // 4 crumbs -> 4 bytes
            r = ggml_cuda_dp4a((int) q, y.q[m], r);
        }
        return d * y.d * (float) (r - y.sum);
    }
};

// thread t owns chunks t, t + blockDim.x, ...; blockDim.x is a multiple of the warp size, so the chunk index in its 256-weight group is lane % 8
template <ggml_type type, int ncols_dst, bool has_gate, bool has_bias, int cpt>
__launch_bounds__(256, 1)
static __global__ void mul_mat_vec_q_gfx906(
        const void * __restrict__ vx, const void * __restrict__ vgate, const float * __restrict__ x_bias,
        const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int nchunks, const int nrows, const int stride_row_bytes, const int stride_col_y, const int stride_col_dst) {
#if defined(__gfx906__)
    using T = mmvq_gfx906_type<type>;
    constexpr int R = mmvq_gfx906_rows_per_block;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;
    const int c    = lane % 8;
    const int base = lane - c;
    const int row0 = blockIdx.x * R;

    mmvq_gfx906_y Y[cpt][ncols_dst];
    int  sb[cpt];
    bool live[cpt];
#pragma unroll
    for (int k = 0; k < cpt; ++k) {
        const int i = tid + blockDim.x * k;
        live[k] = i < nchunks;
        sb[k]   = (live[k] ? i : nchunks - 1) / 8;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            mmvq_gfx906_load_y(y + j * stride_col_y + 8 * sb[k] + c, Y[k][j]);
        }
    }

    typename T::w_t W[R][cpt];
    typename T::w_t WG[has_gate ? R : 1][has_gate ? cpt : 1];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, nrows - 1);
#pragma unroll
        for (int k = 0; k < cpt; ++k) {
            const size_t off = (size_t) row * stride_row_bytes + (size_t) sb[k] * T::bytes_per_256;
            T::load((const uint8_t *) vx + off, c, W[r][k]);
            if constexpr (has_gate) {
                T::load((const uint8_t *) vgate + off, c, WG[r][k]);
            }
        }
    }

    float acc[R][ncols_dst];
    float accg[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            acc[r][j] = 0.0f;
        }
        accg[r] = 0.0f;
#pragma unroll
        for (int k = 0; k < cpt; ++k) {
            const float d = mmvq_gfx906_h2f(__shfl(W[r][k].d, base + T::d_lane(c), warp_size));
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float v = T::dot(W[r][k], d, Y[k][j]);
                acc[r][j] += live[k] ? v : 0.0f;
            }
            if constexpr (has_gate) {
                const float dg = mmvq_gfx906_h2f(__shfl(WG[r][k].d, base + T::d_lane(c), warp_size));
                const float v  = T::dot(WG[r][k], dg, Y[k][0]);
                accg[r] += live[k] ? v : 0.0f;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            acc[r][j] = warp_reduce_sum<warp_size, true>(acc[r][j]);
        }
        if constexpr (has_gate) {
            accg[r] = warp_reduce_sum<warp_size, true>(accg[r]);
        }
    }

    constexpr int nv = ncols_dst + (has_gate ? 1 : 0);
    __shared__ float partial[256 / warp_size][R][nv];
    const int warp = tid / warp_size;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                partial[warp][r][j] = acc[r][j];
            }
            if constexpr (has_gate) {
                partial[warp][r][ncols_dst] = accg[r];
            }
        }
    }
    __syncthreads();

    if (tid < R * ncols_dst) {
        const int r = tid / ncols_dst;
        const int j = tid % ncols_dst;
        if (row0 + r < nrows) {
            const int nwarps = blockDim.x / warp_size;
            float a = 0.0f;
            float g = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                a += partial[w][r][j];
                if constexpr (has_gate) {
                    g += partial[w][r][ncols_dst];
                }
            }
            if constexpr (has_bias) {
                a += x_bias[row0 + r];
            }
            if constexpr (has_gate) {
                a *= ggml_cuda_op_silu_single(g);
            }
            dst[j * stride_col_dst + row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(vx, vgate, x_bias, y, dst, nchunks, nrows, stride_row_bytes, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(__gfx906__)
}

template <ggml_type type, int ncols_dst, bool has_gate, bool has_bias>
static bool mul_mat_vec_q_gfx906_launch(
        const void * vx, const void * vgate, const float * x_bias, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int warp_size, cudaStream_t stream) {
    const int nchunks = ncols_x / 32;
    const int cpt     = (nchunks + 255) / 256; // chunks per thread
    if (cpt * ncols_dst > 8) {
        return false; // too many registers
    }
    const int  nthreads = ((nchunks + cpt - 1) / cpt + warp_size - 1) / warp_size * warp_size; // the reduction reads all lanes
    const dim3 grid((nrows_x + mmvq_gfx906_rows_per_block - 1) / mmvq_gfx906_rows_per_block);
    const int  stride_row_bytes = stride_row_x * (int) ggml_type_size(type);

#define MMVQ_GFX906_LAUNCH(cpt_)                                                                                  \
    mul_mat_vec_q_gfx906<type, ncols_dst, has_gate, has_bias, cpt_><<<grid, nthreads, 0, stream>>>(                \
        vx, vgate, x_bias, (const block_q8_1 *) vy, dst, nchunks, nrows_x, stride_row_bytes, stride_col_y, stride_col_dst)

    bool ran = false;
    switch (cpt) {
        case 1:
            MMVQ_GFX906_LAUNCH(1);
            ran = true;
            break;
        case 2:
            if constexpr (ncols_dst <= 4) {
                MMVQ_GFX906_LAUNCH(2);
                ran = true;
            }
            break;
        case 3:
            if constexpr (ncols_dst <= 2) {
                MMVQ_GFX906_LAUNCH(3);
                ran = true;
            }
            break;
        case 4:
            if constexpr (ncols_dst <= 2) {
                MMVQ_GFX906_LAUNCH(4);
                ran = true;
            }
            break;
        default:
            break;
    }
#undef MMVQ_GFX906_LAUNCH
    return ran;
}

// returns false if the generic MMVQ kernel must run
template <ggml_type type>
static bool mul_mat_vec_q_gfx906_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const int nsamples_dst, const int cc, const int warp_size,
        cudaStream_t stream) {
    static_assert(type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q2_0, "unsupported type");

    if (cc != GGML_CUDA_CC_VEGA20 || ids != nullptr || nchannels_dst != 1 || nsamples_dst != 1 ||
            ncols_dst < 1 || ncols_dst > 4 || ncols_x % 256 != 0 || ncols_x > 32768) {
        return false;
    }
    const bool has_gate = fusion.gate != nullptr;
    const bool has_bias = fusion.x_bias != nullptr;
    if (fusion.gate_bias || fusion.x_scale || fusion.gate_scale || fusion.shared_up || fusion.shared_gate ||
            ((has_gate || has_bias) && ncols_dst != 1) || (has_gate && fusion.glu_op != GGML_GLU_OP_SWIGLU)) {
        return false;
    }

    const float * x_bias = (const float *) fusion.x_bias;
    bool ran = false;
#define MMVQ_GFX906_ARGS vx, fusion.gate, x_bias, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, warp_size, stream
    switch (ncols_dst) {
        case 1:
            if (has_gate) {
                ran = has_bias ? mul_mat_vec_q_gfx906_launch<type, 1, true,  true >(MMVQ_GFX906_ARGS)
                               : mul_mat_vec_q_gfx906_launch<type, 1, true,  false>(MMVQ_GFX906_ARGS);
            } else {
                ran = has_bias ? mul_mat_vec_q_gfx906_launch<type, 1, false, true >(MMVQ_GFX906_ARGS)
                               : mul_mat_vec_q_gfx906_launch<type, 1, false, false>(MMVQ_GFX906_ARGS);
            }
            break;
        case 2:
            ran = mul_mat_vec_q_gfx906_launch<type, 2, false, false>(MMVQ_GFX906_ARGS);
            break;
        case 3:
            ran = mul_mat_vec_q_gfx906_launch<type, 3, false, false>(MMVQ_GFX906_ARGS);
            break;
        case 4:
            ran = mul_mat_vec_q_gfx906_launch<type, 4, false, false>(MMVQ_GFX906_ARGS);
            break;
        default:
            break;
    }
#undef MMVQ_GFX906_ARGS
    if (ran) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ran;
}

#endif // defined(GGML_USE_HIP)
