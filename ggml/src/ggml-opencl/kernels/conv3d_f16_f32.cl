#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_128
#endif

#define T_ACCUM float4
#define VEC_SIZE 4

#define BS_K 64
#define BS_NPQ 64
#define BS_CRS 16

#define TS_K 4
#define TS_NPQ 8

#define WG_K (BS_K / TS_K)
#define WG_NPQ (BS_NPQ / TS_NPQ)

#define BS_NPQ_VEC (BS_NPQ / VEC_SIZE)
#define TS_NPQ_VEC (TS_NPQ / VEC_SIZE)

static inline ulong splitWork(ulong work_size, uint block_size) {
    return (work_size + block_size - 1) / block_size;
}

// Weight  a: [KW, KH, KD, Cin*Cout]  packed as coc_idx = ioc*Cin + ic (ic fast, ioc slow)
// Input   b: [W,  H,  ID, Cin*N]     packed as cn_idx  = n*Cin  + ic (ic fast, n   slow)
// Output    : [OW, OH, OD, Cout*N]   packed as ocn_idx = n*Cout + ioc (ioc fast, n slow)
// (matches ggml_compute_forward_conv_3d_impl, ggml-cpu/ops.cpp)
REQD_SUBGROUP_SIZE_128
kernel void kernel_conv_3d(
    global void * p_knl,
    ulong off_knl,
    global void * p_src,
    ulong off_src,
    global void * p_dst,
    ulong off_dst,
    local void * shared,
    uint Cout, uint Cin, uint N,
    uint KW, uint KH, uint KD, uint W, uint H, uint ID, uint OW, uint OH, uint OD,
    uint s0, uint s1, uint s2, uint p0, uint p1, uint p2, uint d0, uint d1, uint d2,
    ulong nb01, ulong nb02, ulong nb03,
    ulong nb11, ulong nb12, ulong nb13,
    ulong nb1, ulong nb2, ulong nb3
) {
    global half * knl_data = (global half *) ((global char *) p_knl + off_knl);
    global float * src_data = (global float *) ((global char *) p_src + off_src);
    global float * dst_data = (global float *) ((global char *) p_dst + off_dst);

    const uint K = Cout;
    const ulong KHW = (ulong) KH * KW;
    const ulong KDHW = (ulong) KD * KHW;
    const ulong CRS = (ulong) Cin * KDHW;
    const ulong OHW = (ulong) OH * OW;
    const ulong ODHW = (ulong) OD * OHW;
    const ulong NPQ = (ulong) N * ODHW;

    const uint lid_k = get_local_id(0);
    const uint lid_npq = get_local_id(1);
    const uint tid = lid_npq * WG_K + lid_k;

    const ulong B_idx_K = get_group_id(0);
    const ulong B_idx_NPQ = get_group_id(1);

    const ulong offset_k = B_idx_K * BS_K;
    const ulong offset_npq = B_idx_NPQ * BS_NPQ;

    local half * Ash = (local half *) shared;
    local float4 * Bsh = (local float4 *) &Ash[BS_K * BS_CRS];

    T_ACCUM regC[TS_K][TS_NPQ_VEC];
    for (int i = 0; i < TS_K; ++i) {
        for (int j = 0; j < TS_NPQ_VEC; ++j) {
            regC[i][j] = (T_ACCUM) (0.0f);
        }
    }

    const ulong NB_CRS = splitWork(CRS, BS_CRS);

    for (ulong B_idx_CRS = 0; B_idx_CRS < NB_CRS; ++B_idx_CRS) {
        const ulong offset_crs = B_idx_CRS * BS_CRS;

        for (int i = tid; i < BS_K * BS_CRS; i += (WG_K * WG_NPQ)) {
            const uint k_l = i / BS_CRS;
            const uint crs_l = i % BS_CRS;
            const ulong k_g = offset_k + k_l;
            const ulong crs_g = offset_crs + crs_l;

            if (k_g < K && crs_g < CRS) {
                const ulong Cin_idx = crs_g / KDHW;
                const ulong kd_rem = crs_g - Cin_idx * KDHW;
                const ulong KD_idx = kd_rem / KHW;
                const ulong kh_rem = kd_rem - KD_idx * KHW;
                const ulong KH_idx = kh_rem / KW;
                const ulong KW_idx = kh_rem - KH_idx * KW;
                const ulong knl_idx = KW_idx + KH_idx*nb01 + KD_idx*nb02 + (k_g*Cin + Cin_idx)*nb03;
                Ash[k_l * BS_CRS + crs_l] = knl_data[knl_idx];
            } else {
                Ash[k_l * BS_CRS + crs_l] = (half) 0.0f;
            }
        }

        for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
            const uint crs_l = i / BS_NPQ_VEC;
            const uint npq_l_vec = i % BS_NPQ_VEC;
            const ulong crs_g = offset_crs + crs_l;

            float4 val = (float4) (0.0f);
            if (crs_g < CRS) {
                const ulong Cin_idx = crs_g / KDHW;
                const ulong kd_rem = crs_g - Cin_idx * KDHW;
                const ulong KD_idx = kd_rem / KHW;
                const ulong kh_rem = kd_rem - KD_idx * KHW;
                const ulong KH_idx = kh_rem / KW;
                const ulong KW_idx = kh_rem - KH_idx * KW;
                for (int v = 0; v < VEC_SIZE; ++v) {
                    const ulong npq_g = offset_npq + npq_l_vec * VEC_SIZE + v;
                    if (npq_g < NPQ) {
                        const ulong N_idx = npq_g / ODHW;
                        const ulong od_rem = npq_g - N_idx * ODHW;
                        const ulong OD_idx = od_rem / OHW;
                        const ulong oh_rem = od_rem - OD_idx * OHW;
                        const ulong OH_idx = oh_rem / OW;
                        const ulong OW_idx = oh_rem - OH_idx * OW;
                        const long D_idx = (long) (OD_idx * s2 + KD_idx * d2) - p2;
                        const long H_idx = (long) (OH_idx * s1 + KH_idx * d1) - p1;
                        const long W_idx = (long) (OW_idx * s0 + KW_idx * d0) - p0;

                        if (D_idx >= 0 && D_idx < ID && H_idx >= 0 && H_idx < H && W_idx >= 0 && W_idx < W) {
                            const ulong src_idx = W_idx + H_idx*nb11 + D_idx*nb12 + (N_idx*Cin + Cin_idx)*nb13;
                            ((float *) &val)[v] = src_data[src_idx];
                        }
                    }
                }
            }
            Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint crs_l = 0; crs_l < BS_CRS; ++crs_l) {
            half regA[TS_K];
            for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
                regA[k_l_reg] = Ash[(lid_k * TS_K + k_l_reg) * BS_CRS + crs_l];
            }

            for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
                float4 regB = Bsh[crs_l * BS_NPQ_VEC + lid_npq * TS_NPQ_VEC + npq_l_vec_reg];
                for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
                    regC[k_l_reg][npq_l_vec_reg] = mad(convert_float(regA[k_l_reg]), regB, regC[k_l_reg][npq_l_vec_reg]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
        const ulong k_g = offset_k + lid_k * TS_K + k_l_reg;
        if (k_g >= K) continue;

        for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
            const ulong npq_g_base = offset_npq + (lid_npq * TS_NPQ_VEC + npq_l_vec_reg) * VEC_SIZE;

            const ulong N_idx = npq_g_base / ODHW;
            const ulong od_rem = npq_g_base - N_idx * ODHW;
            const ulong OD_idx = od_rem / OHW;
            const ulong oh_rem = od_rem - OD_idx * OHW;
            const ulong OH_idx = oh_rem / OW;
            const ulong OW_idx = oh_rem - OH_idx * OW;

            if (nb1 == OW && OW_idx + VEC_SIZE <= OW && npq_g_base + VEC_SIZE <= NPQ) {
                const ulong dst_idx = OW_idx + OH_idx*nb1 + OD_idx*nb2 + (N_idx*Cout + k_g)*nb3;
                vstore4(regC[k_l_reg][npq_l_vec_reg], 0, &dst_data[dst_idx]);
            } else {
                T_ACCUM res = regC[k_l_reg][npq_l_vec_reg];
                for (int v = 0; v < VEC_SIZE; ++v) {
                    const ulong npq_g = npq_g_base + v;
                    if (npq_g < NPQ) {
                        const ulong N_idx_s = npq_g / ODHW;
                        const ulong od_rem_s = npq_g - N_idx_s * ODHW;
                        const ulong OD_idx_s = od_rem_s / OHW;
                        const ulong oh_rem_s = od_rem_s - OD_idx_s * OHW;
                        const ulong OH_idx_s = oh_rem_s / OW;
                        const ulong OW_idx_s = oh_rem_s - OH_idx_s * OW;
                        const ulong dst_idx_s = OW_idx_s + OH_idx_s*nb1 + OD_idx_s*nb2 + (N_idx_s*Cout + k_g)*nb3;
                        dst_data[dst_idx_s] = ((float *) &res)[v];
                    }
                }
            }
        }
    }
}
