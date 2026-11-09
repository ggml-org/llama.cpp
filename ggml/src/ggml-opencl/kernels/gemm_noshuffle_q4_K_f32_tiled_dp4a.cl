// DP4A GEMM for the canonical 64-row Q4_K layout.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_K       256
#define TILE_ROWS  64
#define NTILES     4
#define BN_DP      8
#define WG_THREADS (NTILES * TILE_ROWS)

inline void q4k_scale_min(int j, __global const uchar * q, uchar * d, uchar * m) {
    if (j < 4) {
        *d = q[j]   & 0x3F;
        *m = q[j+4] & 0x3F;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] & 0xC0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0F) | ((q[j] & 0xC0) >> 2);
    }
}

// dp4a GEMM over the tiled q4_K layout (ne1 > 1) with a q8_1 activation; per 32-element sub-block:
//   acc += d*sd*qad*dp4a(q, qa) - dm*sm*qas
#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_K_f32_tiled_dp4a(
    __global uint4 * src0_q,
    __global half  * src0_d,
    __global half  * src0_dm,
    __global uchar * src0_s,
    __global uint  * qa,        // q8_1 int8 activation, 4/uint  [ne11][ne00/4]
    __global half  * qad,       // q8_1 per-32-block scale       [ne11][ne00/32]
    __global half  * qas,       // q8_1 per-32-block sum (qad*Sum)[ne11][ne00/32]
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01,
    int ne11
) {
    int rit = get_local_id(0);
    int sg  = get_local_id(1);
    int lid = sg * TILE_ROWS + rit;
    int row = get_group_id(0) * WG_THREADS + lid;
    int rt  = row / TILE_ROWS;
    int col0 = get_global_id(2) * BN_DP;
    int nb  = ne00 / QK_K;
    int qcs = ne00 / 4;
    int dcs = ne00 / 32;

    const bool row_ok = row < ne01;
    __local uint sh_qa[BN_DP * 64];
    __local half sh_qad[BN_DP * 8];
    __local half sh_qas[BN_DP * 8];

    float acc[BN_DP];
    #pragma unroll
    for (int j = 0; j < BN_DP; ++j) acc[j] = 0.0f;

    for (int sb = 0; sb < nb; ++sb) {
        for (int p = lid; p < BN_DP * 64; p += WG_THREADS) {
            int j = p >> 6, u = p & 63, c = col0 + j;
            sh_qa[p] = (c < ne11) ? qa[c * qcs + sb * 64 + u] : 0u;
        }
        for (int p = lid; p < BN_DP * 8; p += WG_THREADS) {
            int j = p >> 3, b = p & 7, c = col0 + j;
            sh_qad[p] = (c < ne11) ? qad[c * dcs + sb * 8 + b] : (half)0;
            sh_qas[p] = (c < ne11) ? qas[c * dcs + sb * 8 + b] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row_ok) {
            int tile_blk = rt * nb + sb;
            float dval  = (float)src0_d [tile_blk * TILE_ROWS + rit];
            float dmval = (float)src0_dm[tile_blk * TILE_ROWS + rit];
            __global uchar * sc = src0_s + (tile_blk * TILE_ROWS + rit) * 12;

            uint q[32];
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                uint4 v = src0_q[(tile_blk * 8 + g) * TILE_ROWS + rit];
                q[g*4+0] = v.x; q[g*4+1] = v.y; q[g*4+2] = v.z; q[g*4+3] = v.w;
            }

            for (int is = 0; is < 8; ++is) {
                uchar sd, sm;
                q4k_scale_min(is, sc, &sd, &sm);
                float scl = dval * (float)sd, mnv = dmval * (float)sm;

                uint cw[8];
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    uint pk = 0;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        int e = is * 32 + t * 4 + u;
                        uint code = (q[e >> 3] >> ((e & 7) * 4)) & 0xF;
                        pk |= (code & 0xFF) << (u * 8);
                    }
                    cw[t] = pk;
                }
                #pragma unroll
                for (int j = 0; j < BN_DP; ++j) {
                    __local uint * a = sh_qa + j * 64 + is * 8;
                    int raw = 0;
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) raw = dot_acc_sat_4x8packed_ss_int(cw[t], a[t], raw);
                    acc[j] += scl * (float)sh_qad[j*8 + is] * (float)raw
                            - mnv * (float)sh_qas[j*8 + is];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_ok) {
        dst = (global float*)((global char*)dst + offsetd);
        #pragma unroll
        for (int j = 0; j < BN_DP; ++j) {
            int c = col0 + j;
            if (c < ne11) dst[(ulong)c * ne01 + row] = acc[j];
        }
    }
}
