#include "mmvdq.cuh"
#include "unary.cuh"

#include <cstdlib>

// -1 = unset (use arch default), 0 = force off, 1 = force on.
static int dq_env_override(const char * name) {
    const char * v = getenv(name);
    if (!v) return -1;
    return (v[0] == '0' && v[1] == '\0') ? 0 : 1;
}

bool ggml_cuda_dq_mmv_enabled(bool arch_default) {
    static const int ov = dq_env_override("GGML_CUDA_DQ_MMV");
    return ov < 0 ? arch_default : (bool) ov;
}

bool ggml_cuda_dq_q6k_enabled(bool arch_default) {
    static const int ov = dq_env_override("GGML_CUDA_DQ_Q6K");
    return ov < 0 ? arch_default : (bool) ov;
}

// Rows-per-block tuning knob. Only 1/2/4/8 are instantiated; anything else
// warns once and falls back to 1. Cached so we don't re-read the env per matvec.
static int dq_num_rows_init() {
    const char * v = getenv("GGML_CUDA_DQ_ROWS");
    if (!v) return 1;
    const int r = atoi(v);
    if (r == 1 || r == 2 || r == 4 || r == 8) return r;
    GGML_LOG_WARN("%s: unsupported GGML_CUDA_DQ_ROWS=%s (expected 1/2/4/8), using 1\n", __func__, v);
    return 1;
}

static int dq_num_rows() {
    static const int rows = dq_num_rows_init();
    return rows;
}

// Q4_K scale/min extraction, identical to get_scale_min_k4 in convert.cu.
static __device__ __forceinline__ void dq_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Per-super-block dot for one Q4_K block against pre-loaded (float4) activation.
// Shared by the plain and gate+up-fused kernels so the dequant math lives once.
static __device__ __forceinline__ float dq_dot_q4_K(
        const block_q4_K * b, int q_offset, int v_im,
        const float4 & by10, const float4 & by132, const float4 & by20, const float4 & by232,
        float sum10, float sum32, float sum20, float sum42) {
    const float dall = __low2float(b->dm);
    const float dmin = __high2float(b->dm);

    uint8_t dx, mx, dy, my, dz, mz, dw, mw;
    dq_get_scale_min_k4(2*v_im + 0, b->scales, dx, mx);
    dq_get_scale_min_k4(2*v_im + 1, b->scales, dy, my);
    dq_get_scale_min_k4(2*v_im + 4, b->scales, dz, mz);
    dq_get_scale_min_k4(2*v_im + 5, b->scales, dw, mw);

    const uint32_t * qs32 = (const uint32_t *) b->qs;
    const uint32_t qs0  = qs32[q_offset/4     ];
    const uint32_t qs64 = qs32[q_offset/4 + 16];

    const float sx =
        by10.x * (float) ((qs0 >>  0) & 0xF) + by10.y * (float) ((qs0 >>  8) & 0xF) +
        by10.z * (float) ((qs0 >> 16) & 0xF) + by10.w * (float) ((qs0 >> 24) & 0xF);
    const float sy =
        by132.x * (float) ((qs0 >>  4) & 0xF) + by132.y * (float) ((qs0 >> 12) & 0xF) +
        by132.z * (float) ((qs0 >> 20) & 0xF) + by132.w * (float) ((qs0 >> 28) & 0xF);
    const float sz =
        by20.x * (float) ((qs64 >>  0) & 0xF) + by20.y * (float) ((qs64 >>  8) & 0xF) +
        by20.z * (float) ((qs64 >> 16) & 0xF) + by20.w * (float) ((qs64 >> 24) & 0xF);
    const float sw =
        by232.x * (float) ((qs64 >>  4) & 0xF) + by232.y * (float) ((qs64 >> 12) & 0xF) +
        by232.z * (float) ((qs64 >> 20) & 0xF) + by232.w * (float) ((qs64 >> 28) & 0xF);

    const float smin = sum10*mx + sum32*my + sum20*mz + sum42*mw;
    return dall * (sx*dx + sy*dy + sz*dz + sw*dw) - dmin*smin;
}

// Per-super-block dot for one Q5_K block against pre-loaded (float2) activation.
// Scale unpacking (s04l/s04h/s8) and the qh bit-plane merges mirror the canonical
// vec_dot_q5_K_q8_1 in vecdotq.cuh — keep in sync if that changes.
static __device__ __forceinline__ float dq_dot_q5_K(
        const block_q5_K * b, int q_offset, int l0, int v_im,
        const float2 & by10, const float2 & by116, const float2 & by132, const float2 & by148,
        const float2 & by20, const float2 & by216, const float2 & by232, const float2 & by248,
        float smin_x, float smin_y, float smin_z, float smin_w) {
    const float dall = __low2float(b->dm);
    const float dmin = __high2float(b->dm);

    const uint16_t * sc16 = (const uint16_t *) b->scales;
    const uint32_t scale0 = sc16[v_im    ];
    const uint32_t scale4 = sc16[v_im + 2];
    const uint32_t scale8 = sc16[v_im + 4];
    const uint32_t s04l = (scale4 << 16) | scale0;
    const uint32_t s04h = (s04l & 0xC0C0C0C0u) >> 2;
    const uint32_t s04m = s04l & 0x3F3F3F3Fu;
    const uint32_t s8   = (((scale8 << 12) | scale8) & 0x0F0F0F0Fu) | s04h;

    const float sc0 = (float) ((s04m >>  0) & 0xFF);
    const float sc1 = (float) ((s04m >>  8) & 0xFF);
    const float sc2 = (float) ((s04m >> 16) & 0xFF);
    const float sc3 = (float) ((s04m >> 24) & 0xFF);
    const float sc4 = (float) ((s8   >>  0) & 0xFF);
    const float sc5 = (float) ((s8   >>  8) & 0xFF);
    const float sc6 = (float) ((s8   >> 16) & 0xFF);
    const float sc7 = (float) ((s8   >> 24) & 0xFF);

    const uint16_t * qs16 = (const uint16_t *) b->qs;
    const uint32_t qs0  = (uint32_t) qs16[q_offset/2     ] | ((uint32_t) qs16[q_offset/2 +  8] << 16);
    const uint32_t qs64 = (uint32_t) qs16[q_offset/2 + 32] | ((uint32_t) qs16[q_offset/2 + 40] << 16);

    uint32_t qs0_lo  = qs0  & 0x0F0F0F0Fu;
    uint32_t qs0_hi  = (qs0  >> 4) & 0x0F0F0F0Fu;
    uint32_t qs64_lo = qs64 & 0x0F0F0F0Fu;
    uint32_t qs64_hi = (qs64 >> 4) & 0x0F0F0F0Fu;

    const uint16_t * qh16 = (const uint16_t *) b->qh;
    const uint32_t qh = (uint32_t) qh16[l0/2] | ((uint32_t) qh16[l0/2 + 8] << 16);

    qs0_lo  += ((qh >> (2*v_im)) & 0x01010101u) << 4;
    qs0_hi  += ((qh >> (2*v_im)) & 0x02020202u) << 3;
    qs64_lo += ((qh >> (2*v_im)) & 0x10101010u);
    qs64_hi += ((qh >> (2*v_im)) & 0x20202020u) >> 1;

    const float sx =
        by10.x  * (float) ((qs0_lo  >>  0) & 0xFF) + by10.y  * (float) ((qs0_lo  >>  8) & 0xFF) +
        by116.x * (float) ((qs0_lo  >> 16) & 0xFF) + by116.y * (float) ((qs0_lo  >> 24) & 0xFF);
    const float sy =
        by132.x * (float) ((qs0_hi  >>  0) & 0xFF) + by132.y * (float) ((qs0_hi  >>  8) & 0xFF) +
        by148.x * (float) ((qs0_hi  >> 16) & 0xFF) + by148.y * (float) ((qs0_hi  >> 24) & 0xFF);
    const float sz =
        by20.x  * (float) ((qs64_lo >>  0) & 0xFF) + by20.y  * (float) ((qs64_lo >>  8) & 0xFF) +
        by216.x * (float) ((qs64_lo >> 16) & 0xFF) + by216.y * (float) ((qs64_lo >> 24) & 0xFF);
    const float sw =
        by232.x * (float) ((qs64_hi >>  0) & 0xFF) + by232.y * (float) ((qs64_hi >>  8) & 0xFF) +
        by248.x * (float) ((qs64_hi >> 16) & 0xFF) + by248.y * (float) ((qs64_hi >> 24) & 0xFF);

    const float smin = smin_x*sc2 + smin_y*sc3 + smin_z*sc6 + smin_w*sc7;
    return dall * (sx*sc0 + sy*sc1 + sz*sc4 + sw*sc5) - dmin*smin;
}

// Per-super-block dot for one Q6_K block against pre-loaded (float4) activation.
// The ql/qh bit-plane merges (q0u..q3u) and the -32 bias mirror the canonical
// vec_dot_q6_K_q8_1 in vecdotq.cuh — keep in sync if that changes.
static __device__ __forceinline__ float dq_dot_q6_K(
        const block_q6_K * b, int ql_offset, int qh_offset, int s_offset,
        const float4 & by0, const float4 & by32, const float4 & by64, const float4 & by96) {
    const float d = __half2float(b->d);

    const uint16_t * ql16 = (const uint16_t *) b->ql;
    const uint32_t ql0  = (uint32_t) ql16[ql_offset/2     ] | ((uint32_t) ql16[ql_offset/2 +  1] << 16);
    const uint32_t ql32 = (uint32_t) ql16[ql_offset/2 + 16] | ((uint32_t) ql16[ql_offset/2 + 17] << 16);

    const uint32_t ql0_lo  = ql0  & 0x0F0F0F0Fu;
    const uint32_t ql0_hi  = (ql0  >> 4) & 0x0F0F0F0Fu;
    const uint32_t ql32_lo = ql32 & 0x0F0F0F0Fu;
    const uint32_t ql32_hi = (ql32 >> 4) & 0x0F0F0F0Fu;

    const uint16_t * qh16 = (const uint16_t *) b->qh;
    const uint32_t qh = (uint32_t) qh16[qh_offset/2] | ((uint32_t) qh16[qh_offset/2 + 1] << 16);

    const uint32_t q0u = ql0_lo  | ((qh & 0x03030303u) << 4);
    const uint32_t q1u = ql32_lo | ((qh & 0x0C0C0C0Cu) << 2);
    const uint32_t q2u = ql0_hi  |  (qh & 0x30303030u);
    const uint32_t q3u = ql32_hi | ((qh & 0xC0C0C0C0u) >> 2);

    const int8_t * sc = b->scales + s_offset;
    const float sc0 = (float) sc[0];
    const float sc2 = (float) sc[2];
    const float sc4 = (float) sc[4];
    const float sc6 = (float) sc[6];

    const float sum0 =
        by0.x * (float) ((int) ((q0u >>  0) & 0xFF) - 32) + by0.y * (float) ((int) ((q0u >>  8) & 0xFF) - 32) +
        by0.z * (float) ((int) ((q0u >> 16) & 0xFF) - 32) + by0.w * (float) ((int) ((q0u >> 24) & 0xFF) - 32);
    const float sum1 =
        by32.x * (float) ((int) ((q1u >>  0) & 0xFF) - 32) + by32.y * (float) ((int) ((q1u >>  8) & 0xFF) - 32) +
        by32.z * (float) ((int) ((q1u >> 16) & 0xFF) - 32) + by32.w * (float) ((int) ((q1u >> 24) & 0xFF) - 32);
    const float sum2 =
        by64.x * (float) ((int) ((q2u >>  0) & 0xFF) - 32) + by64.y * (float) ((int) ((q2u >>  8) & 0xFF) - 32) +
        by64.z * (float) ((int) ((q2u >> 16) & 0xFF) - 32) + by64.w * (float) ((int) ((q2u >> 24) & 0xFF) - 32);
    const float sum3 =
        by96.x * (float) ((int) ((q3u >>  0) & 0xFF) - 32) + by96.y * (float) ((int) ((q3u >>  8) & 0xFF) - 32) +
        by96.z * (float) ((int) ((q3u >> 16) & 0xFF) - 32) + by96.w * (float) ((int) ((q3u >> 24) & 0xFF) - 32);

    return d * (sum0*sc0 + sum1*sc2 + sum2*sc4 + sum3*sc6);
}

// ---- Q4_K geometry ----
struct dq_geom_q4_K {
    int itid, ix, v_im, q_offset, y_offset;
};
static __device__ __forceinline__ dq_geom_q4_K dq_setup_q4_K(int tid) {
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int il   = itid / 4;
    const int ir   = itid % 4;
    const int v_im = il / 2;
    const int v_in = il % 2;
    const int l0   = 4 * (2 * ir + v_in);
    return { itid, ix, v_im, 32 * v_im + l0, 64 * v_im + l0 };
}

// mul_mat_vec_q4_k shader: 16 threads process one super-block, each block
// computes NUM_ROWS output rows, activations are loaded once as float4 and
// reused across rows. No q8_1 activation pass (unlike mul_mat_vec_q).
template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_q4_K(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q4_K g = dq_setup_q4_K(threadIdx.x);
    const block_q4_K * x = (const block_q4_K *) vx;

    float sumf[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) sumf[n] = 0.0f;

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        const float4 by10  = *(const float4 *) (yb + g.y_offset      );
        const float4 by132 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by20  = *(const float4 *) (yb + g.y_offset + 128);
        const float4 by232 = *(const float4 *) (yb + g.y_offset + 160);

        const float sum10 = by10.x  + by10.y  + by10.z  + by10.w;
        const float sum32 = by132.x + by132.y + by132.z + by132.w;
        const float sum20 = by20.x  + by20.y  + by20.z  + by20.w;
        const float sum42 = by232.x + by232.y + by232.z + by232.w;

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const block_q4_K * b = &x[(int64_t) row * nblocks + i];
            sumf[n] += dq_dot_q4_K(b, g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float total = warp_reduce_sum<warp_size>(sumf[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = total;
    }
}

// Fused gate+up SwiGLU dequant matvec for Q4_K: computes up and gate matvecs
// from the shared activation in one pass, writes silu(gate)*up.
template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_glu_q4_K(
        const void * __restrict__ vx_up, const void * __restrict__ vx_gate,
        const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q4_K g = dq_setup_q4_K(threadIdx.x);
    const block_q4_K * xu = (const block_q4_K *) vx_up;
    const block_q4_K * xg = (const block_q4_K *) vx_gate;

    if (num_rows >= 8) {
        // Two-pass: one accumulator array at a time keeps register pressure at
        // plain-kernel levels. up is reduced and stashed in dst, then pass 2
        // computes gate and combines. Weights are still read once each; only the
        // small activation vector y is re-read. Avoids the single-pass VGPR spill.
        for (int pass = 0; pass < 2; ++pass) {
            const block_q4_K * xw = pass == 0 ? xu : xg;
            float acc[num_rows];
#pragma unroll
            for (int n = 0; n < num_rows; ++n) acc[n] = 0.0f;

            for (int i = g.ix; i < nblocks; i += it_size) {
                const float * yb = y + (int64_t) i * QK_K;
                const float4 by10  = *(const float4 *) (yb + g.y_offset      );
                const float4 by132 = *(const float4 *) (yb + g.y_offset +  32);
                const float4 by20  = *(const float4 *) (yb + g.y_offset + 128);
                const float4 by232 = *(const float4 *) (yb + g.y_offset + 160);

                const float sum10 = by10.x  + by10.y  + by10.z  + by10.w;
                const float sum32 = by132.x + by132.y + by132.z + by132.w;
                const float sum20 = by20.x  + by20.y  + by20.z  + by20.w;
                const float sum42 = by232.x + by232.y + by232.z + by232.w;

#pragma unroll
                for (int n = 0; n < num_rows; ++n) {
                    const int row = min(first_row + n, nrows_x - 1);
                    acc[n] += dq_dot_q4_K(&xw[(int64_t) row * nblocks + i], g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
                }
            }

#pragma unroll
            for (int n = 0; n < num_rows; ++n) {
                const float r = warp_reduce_sum<warp_size>(acc[n]);
                if (threadIdx.x == 0 && first_row + n < nrows_x) {
                    if (pass == 0) dst[first_row + n] = r;
                    else           dst[first_row + n] = ggml_cuda_op_silu_single(r) * dst[first_row + n];
                }
            }
        }
        return;
    }

    float up[num_rows], gate[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) { up[n] = 0.0f; gate[n] = 0.0f; }

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        const float4 by10  = *(const float4 *) (yb + g.y_offset      );
        const float4 by132 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by20  = *(const float4 *) (yb + g.y_offset + 128);
        const float4 by232 = *(const float4 *) (yb + g.y_offset + 160);

        const float sum10 = by10.x  + by10.y  + by10.z  + by10.w;
        const float sum32 = by132.x + by132.y + by132.z + by132.w;
        const float sum20 = by20.x  + by20.y  + by20.z  + by20.w;
        const float sum42 = by232.x + by232.y + by232.z + by232.w;

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const int64_t off = (int64_t) row * nblocks + i;
            up[n]   += dq_dot_q4_K(&xu[off], g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
            gate[n] += dq_dot_q4_K(&xg[off], g.q_offset, g.v_im, by10, by132, by20, by232, sum10, sum32, sum20, sum42);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float u = warp_reduce_sum<warp_size>(up[n]);
        const float gt = warp_reduce_sum<warp_size>(gate[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = ggml_cuda_op_silu_single(gt) * u;
    }
}

// ---- Q5_K geometry ----
struct dq_geom_q5_K {
    int ix, l0, v_im, q_offset, y_offset;
};
static __device__ __forceinline__ dq_geom_q5_K dq_setup_q5_K(int tid) {
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int il   = itid / 4;
    const int ir   = itid % 4;
    const int v_im = il / 2;
    const int v_in = il % 2;
    const int l0   = 4 * ir + 2 * v_in;
    return { ix, l0, v_im, 32 * v_im + l0, 64 * v_im + l0 };
}

#define DQ_Q5_K_LOAD_ACT()                                                          \
    const float2 by10  = *(const float2 *) (yb + g.y_offset      );                 \
    const float2 by116 = *(const float2 *) (yb + g.y_offset +  16);                 \
    const float2 by132 = *(const float2 *) (yb + g.y_offset +  32);                 \
    const float2 by148 = *(const float2 *) (yb + g.y_offset +  48);                 \
    const float2 by20  = *(const float2 *) (yb + g.y_offset + 128);                 \
    const float2 by216 = *(const float2 *) (yb + g.y_offset + 144);                 \
    const float2 by232 = *(const float2 *) (yb + g.y_offset + 160);                 \
    const float2 by248 = *(const float2 *) (yb + g.y_offset + 176);                 \
    const float smin_x = by10.x  + by10.y  + by116.x + by116.y;                     \
    const float smin_y = by132.x + by132.y + by148.x + by148.y;                     \
    const float smin_z = by20.x  + by20.y  + by216.x + by216.y;                     \
    const float smin_w = by232.x + by232.y + by248.x + by248.y

#define DQ_Q5_K_ARGS by10, by116, by132, by148, by20, by216, by232, by248, smin_x, smin_y, smin_z, smin_w

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_q5_K(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q5_K g = dq_setup_q5_K(threadIdx.x);
    const block_q5_K * x = (const block_q5_K *) vx;

    float sumf[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) sumf[n] = 0.0f;

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        DQ_Q5_K_LOAD_ACT();

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const block_q5_K * b = &x[(int64_t) row * nblocks + i];
            sumf[n] += dq_dot_q5_K(b, g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float total = warp_reduce_sum<warp_size>(sumf[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = total;
    }
}

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_glu_q5_K(
        const void * __restrict__ vx_up, const void * __restrict__ vx_gate,
        const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q5_K g = dq_setup_q5_K(threadIdx.x);
    const block_q5_K * xu = (const block_q5_K *) vx_up;
    const block_q5_K * xg = (const block_q5_K *) vx_gate;

    if (num_rows >= 8) {
        // Two-pass: one accumulator array at a time keeps register pressure at
        // plain-kernel levels. up is reduced and stashed in dst, then pass 2
        // computes gate and combines. Weights are still read once each; only the
        // small activation vector y is re-read. Avoids the single-pass VGPR spill.
        for (int pass = 0; pass < 2; ++pass) {
            const block_q5_K * xw = pass == 0 ? xu : xg;
            float acc[num_rows];
#pragma unroll
            for (int n = 0; n < num_rows; ++n) acc[n] = 0.0f;

            for (int i = g.ix; i < nblocks; i += it_size) {
                const float * yb = y + (int64_t) i * QK_K;
                DQ_Q5_K_LOAD_ACT();

#pragma unroll
                for (int n = 0; n < num_rows; ++n) {
                    const int row = min(first_row + n, nrows_x - 1);
                    acc[n] += dq_dot_q5_K(&xw[(int64_t) row * nblocks + i], g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
                }
            }

#pragma unroll
            for (int n = 0; n < num_rows; ++n) {
                const float r = warp_reduce_sum<warp_size>(acc[n]);
                if (threadIdx.x == 0 && first_row + n < nrows_x) {
                    if (pass == 0) dst[first_row + n] = r;
                    else           dst[first_row + n] = ggml_cuda_op_silu_single(r) * dst[first_row + n];
                }
            }
        }
        return;
    }

    float up[num_rows], gate[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) { up[n] = 0.0f; gate[n] = 0.0f; }

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        DQ_Q5_K_LOAD_ACT();

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const int64_t off = (int64_t) row * nblocks + i;
            up[n]   += dq_dot_q5_K(&xu[off], g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
            gate[n] += dq_dot_q5_K(&xg[off], g.q_offset, g.l0, g.v_im, DQ_Q5_K_ARGS);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float u = warp_reduce_sum<warp_size>(up[n]);
        const float gt = warp_reduce_sum<warp_size>(gate[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = ggml_cuda_op_silu_single(gt) * u;
    }
}

// ---- Q6_K geometry ----
struct dq_geom_q6_K {
    int ix, ql_offset, qh_offset, s_offset, y_offset;
};
static __device__ __forceinline__ dq_geom_q6_K dq_setup_q6_K(int tid) {
    const int itid = tid % 16;
    const int ix   = tid / 16;
    const int v_im = itid / 8;
    const int v_in = itid % 8;
    const int l0   = 4 * v_in;
    const int is   = v_in / 4;
    return { ix, 64 * v_im + l0, 32 * v_im + l0, 8 * v_im + is, 128 * v_im + l0 };
}

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_q6_K(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q6_K g = dq_setup_q6_K(threadIdx.x);
    const block_q6_K * x = (const block_q6_K *) vx;

    float sumf[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) sumf[n] = 0.0f;

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        const float4 by0  = *(const float4 *) (yb + g.y_offset      );
        const float4 by32 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by64 = *(const float4 *) (yb + g.y_offset +  64);
        const float4 by96 = *(const float4 *) (yb + g.y_offset +  96);

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const block_q6_K * b = &x[(int64_t) row * nblocks + i];
            sumf[n] += dq_dot_q6_K(b, g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float total = warp_reduce_sum<warp_size>(sumf[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = total;
    }
}

template <int warp_size, int num_rows>
static __global__ void mul_mat_vec_dq_glu_q6_K(
        const void * __restrict__ vx_up, const void * __restrict__ vx_gate,
        const float * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x) {
    const int first_row = num_rows * blockIdx.x;
    const int nblocks   = ncols_x / QK_K;
    const int it_size   = warp_size / 16;

    const dq_geom_q6_K g = dq_setup_q6_K(threadIdx.x);
    const block_q6_K * xu = (const block_q6_K *) vx_up;
    const block_q6_K * xg = (const block_q6_K *) vx_gate;

    if (num_rows >= 8) {
        // Two-pass: one accumulator array at a time keeps register pressure at
        // plain-kernel levels. up is reduced and stashed in dst, then pass 2
        // computes gate and combines. Weights are still read once each; only the
        // small activation vector y is re-read. Avoids the single-pass VGPR spill.
        for (int pass = 0; pass < 2; ++pass) {
            const block_q6_K * xw = pass == 0 ? xu : xg;
            float acc[num_rows];
#pragma unroll
            for (int n = 0; n < num_rows; ++n) acc[n] = 0.0f;

            for (int i = g.ix; i < nblocks; i += it_size) {
                const float * yb = y + (int64_t) i * QK_K;
                const float4 by0  = *(const float4 *) (yb + g.y_offset      );
                const float4 by32 = *(const float4 *) (yb + g.y_offset +  32);
                const float4 by64 = *(const float4 *) (yb + g.y_offset +  64);
                const float4 by96 = *(const float4 *) (yb + g.y_offset +  96);

#pragma unroll
                for (int n = 0; n < num_rows; ++n) {
                    const int row = min(first_row + n, nrows_x - 1);
                    acc[n] += dq_dot_q6_K(&xw[(int64_t) row * nblocks + i], g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
                }
            }

#pragma unroll
            for (int n = 0; n < num_rows; ++n) {
                const float r = warp_reduce_sum<warp_size>(acc[n]);
                if (threadIdx.x == 0 && first_row + n < nrows_x) {
                    if (pass == 0) dst[first_row + n] = r;
                    else           dst[first_row + n] = ggml_cuda_op_silu_single(r) * dst[first_row + n];
                }
            }
        }
        return;
    }

    float up[num_rows], gate[num_rows];
#pragma unroll
    for (int n = 0; n < num_rows; ++n) { up[n] = 0.0f; gate[n] = 0.0f; }

    for (int i = g.ix; i < nblocks; i += it_size) {
        const float * yb = y + (int64_t) i * QK_K;
        const float4 by0  = *(const float4 *) (yb + g.y_offset      );
        const float4 by32 = *(const float4 *) (yb + g.y_offset +  32);
        const float4 by64 = *(const float4 *) (yb + g.y_offset +  64);
        const float4 by96 = *(const float4 *) (yb + g.y_offset +  96);

#pragma unroll
        for (int n = 0; n < num_rows; ++n) {
            const int row = min(first_row + n, nrows_x - 1);
            const int64_t off = (int64_t) row * nblocks + i;
            up[n]   += dq_dot_q6_K(&xu[off], g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
            gate[n] += dq_dot_q6_K(&xg[off], g.ql_offset, g.qh_offset, g.s_offset, by0, by32, by64, by96);
        }
    }

#pragma unroll
    for (int n = 0; n < num_rows; ++n) {
        const float u = warp_reduce_sum<warp_size>(up[n]);
        const float gt = warp_reduce_sum<warp_size>(gate[n]);
        if (threadIdx.x == 0 && first_row + n < nrows_x) dst[first_row + n] = ggml_cuda_op_silu_single(gt) * u;
    }
}

// ---- launchers ----
#define DQ_LAUNCH_PLAIN(KERN, NR)                                                                 \
    do {                                                                                          \
        const dim3 bn((nrows_x + (NR) - 1) / (NR), 1, 1);                                         \
        const dim3 bd(warp_size, 1, 1);                                                           \
        if (warp_size == 64) KERN<64, NR><<<bn, bd, 0, stream>>>(vx, y, d, ncols_x, nrows_x);     \
        else                 KERN<32, NR><<<bn, bd, 0, stream>>>(vx, y, d, ncols_x, nrows_x);     \
    } while (0)

#define DQ_LAUNCH_GLU(KERN, NR)                                                                         \
    do {                                                                                                \
        const dim3 bn((nrows_x + (NR) - 1) / (NR), 1, 1);                                               \
        const dim3 bd(warp_size, 1, 1);                                                                 \
        if (warp_size == 64) KERN<64, NR><<<bn, bd, 0, stream>>>(vx_up, vx_gate, y, d, ncols_x, nrows_x); \
        else                 KERN<32, NR><<<bn, bd, 0, stream>>>(vx_up, vx_gate, y, d, ncols_x, nrows_x); \
    } while (0)

template <int num_rows>
static void launch_dq_q4_K(const void * vx, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_PLAIN(mul_mat_vec_dq_q4_K, num_rows);
}
template <int num_rows>
static void launch_dq_q5_K(const void * vx, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_PLAIN(mul_mat_vec_dq_q5_K, num_rows);
}
template <int num_rows>
static void launch_dq_q6_K(const void * vx, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_PLAIN(mul_mat_vec_dq_q6_K, num_rows);
}
template <int num_rows>
static void launch_dq_glu_q4_K(const void * vx_up, const void * vx_gate, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_GLU(mul_mat_vec_dq_glu_q4_K, num_rows);
}
template <int num_rows>
static void launch_dq_glu_q5_K(const void * vx_up, const void * vx_gate, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_GLU(mul_mat_vec_dq_glu_q5_K, num_rows);
}
template <int num_rows>
static void launch_dq_glu_q6_K(const void * vx_up, const void * vx_gate, const float * y, float * d, int ncols_x, int nrows_x, int warp_size, cudaStream_t stream) {
    DQ_LAUNCH_GLU(mul_mat_vec_dq_glu_q6_K, num_rows);
}

#define DQ_DISPATCH_ROWS(LAUNCH, ...)                          \
    switch (dq_num_rows()) {                                   \
        case 2:  LAUNCH<2>(__VA_ARGS__); break;               \
        case 4:  LAUNCH<4>(__VA_ARGS__); break;               \
        case 8:  LAUNCH<8>(__VA_ARGS__); break;               \
        default: LAUNCH<1>(__VA_ARGS__); break;               \
    }

void ggml_cuda_mul_mat_vec_dq_q4_K(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx = src0->data;
    const float * y  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    DQ_DISPATCH_ROWS(launch_dq_q4_K, vx, y, d, ncols_x, nrows, warp_size, stream);
}

void ggml_cuda_mul_mat_vec_dq_q5_K(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx = src0->data;
    const float * y  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    DQ_DISPATCH_ROWS(launch_dq_q5_K, vx, y, d, ncols_x, nrows, warp_size, stream);
}

void ggml_cuda_mul_mat_vec_dq_q6_K(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx = src0->data;
    const float * y  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    DQ_DISPATCH_ROWS(launch_dq_q6_K, vx, y, d, ncols_x, nrows, warp_size, stream);
}

void ggml_cuda_mul_mat_vec_dq_glu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols_x = up->ne[0];
    const int nrows   = up->ne[1];
    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const void  * vx_up   = up->data;
    const void  * vx_gate = gate->data;
    const float * y = (const float *) src1->data;
    float       * d = (float *) dst->data;
    switch (up->type) {
        case GGML_TYPE_Q4_K: DQ_DISPATCH_ROWS(launch_dq_glu_q4_K, vx_up, vx_gate, y, d, ncols_x, nrows, warp_size, stream); break;
        case GGML_TYPE_Q5_K: DQ_DISPATCH_ROWS(launch_dq_glu_q5_K, vx_up, vx_gate, y, d, ncols_x, nrows, warp_size, stream); break;
        default:             DQ_DISPATCH_ROWS(launch_dq_glu_q6_K, vx_up, vx_gate, y, d, ncols_x, nrows, warp_size, stream); break;
    }
}
