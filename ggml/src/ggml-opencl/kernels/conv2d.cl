#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define T_FLOAT half
#define T_FLOAT4 half4
#define VSTORE_T_FLOAT4(data, offset, p) vstore_half4_rte(data, offset, p)
#else
#define T_FLOAT float
#define T_FLOAT4 float4
#define VSTORE_T_FLOAT4(data, offset, p) vstore4(data, offset, p)
#endif

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

static inline uint splitWork(uint work_size, uint block_size){
    return (work_size + block_size - 1) / block_size;
}

REQD_SUBGROUP_SIZE_128
kernel void kernel_conv_2d(
    global void* p_knl,
    ulong off_knl,
    global void* p_src,
    ulong off_src,
    global void* p_dst,
    ulong off_dst,
    local void* shared,
    uint Cout, uint Cin, uint N,
    uint KW, uint KH, uint W, uint H, uint OW, uint OH,
    uint s0, uint s1, uint p0, uint p1, uint d0, uint d1,
    uint nb01, uint nb02, uint nb03,
    uint nb11, uint nb12, uint nb13,
    uint nb1, uint nb2, uint nb3
) {
    global T_FLOAT* knl_data = (global T_FLOAT*) ((global char*)p_knl + off_knl);
    global T_FLOAT* src_data = (global T_FLOAT*) ((global char*)p_src + off_src);
    global T_FLOAT* dst_data = (global T_FLOAT*) ((global char*)p_dst + off_dst);

    const uint K = Cout;
    const uint CRS = Cin*KH*KW;
    const uint NPQ = N*OH*OW;

    const uint lid_k = get_local_id(0);
    const uint lid_npq = get_local_id(1);
    const uint tid = lid_npq * WG_K + lid_k;

    const uint B_idx_K = get_group_id(0);
    const uint B_idx_NPQ = get_group_id(1);

    const uint offset_k = B_idx_K * BS_K;
    const uint offset_npq = B_idx_NPQ * BS_NPQ;

    local T_FLOAT* Ash = (local T_FLOAT*)shared;
    local T_FLOAT4* Bsh = (local T_FLOAT4*) &Ash[BS_K * BS_CRS];

    T_ACCUM regC[TS_K][TS_NPQ_VEC];
    for (int i = 0; i < TS_K; ++i) {
        for (int j = 0; j < TS_NPQ_VEC; ++j) {
            regC[i][j] = (T_ACCUM)(0.0f);
        }
    }

    const uint NB_CRS = splitWork(CRS, BS_CRS);

    for (uint B_idx_CRS = 0; B_idx_CRS < NB_CRS; ++B_idx_CRS) {
        const uint offset_crs = B_idx_CRS * BS_CRS;

        for (int i = tid; i < BS_K * BS_CRS; i += (WG_K * WG_NPQ)) {
            const uint k_l = i / BS_CRS;
            const uint crs_l = i % BS_CRS;
            const uint k_g = offset_k + k_l;
            const uint crs_g = offset_crs + crs_l;

            if (k_g < K && crs_g < CRS) {
                const uint Cin_idx = crs_g / (KW*KH);
                const uint KH_idx = (crs_g - Cin_idx*KW*KH) / KW;
                const uint KW_idx = crs_g - Cin_idx*KW*KH - KH_idx*KW;
                const uint knl_idx = KW_idx + KH_idx*nb01 + Cin_idx*nb02 + k_g*nb03;
                Ash[k_l * BS_CRS + crs_l] = knl_data[knl_idx];
            } else {
                Ash[k_l * BS_CRS + crs_l] = (T_FLOAT)0.0f;
            }
        }

        for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
            const uint crs_l = i / BS_NPQ_VEC;
            const uint npq_l_vec = i % BS_NPQ_VEC;
            const uint crs_g = offset_crs + crs_l;

            T_FLOAT4 val = (T_FLOAT4)(0.0f);
            if (crs_g < CRS) {
                const uint Cin_idx = crs_g / (KW * KH);
                const uint KH_idx = (crs_g - Cin_idx * KW * KH) / KW;
                const uint KW_idx = crs_g - Cin_idx * KW * KH - KH_idx * KW;
                for (int v = 0; v < VEC_SIZE; ++v) {
                    const uint npq_g = offset_npq + npq_l_vec * VEC_SIZE + v;
                    if (npq_g < NPQ) {
                        const uint N_idx = npq_g / (OH * OW);
                        const uint pq_idx = npq_g % (OH * OW);
                        const uint OH_idx = pq_idx / OW;
                        const uint OW_idx = pq_idx % OW;
                        const int H_idx = (int)(OH_idx * s1 + KH_idx * d1 - p1);
                        const int W_idx = (int)(OW_idx * s0 + KW_idx * d0 - p0);

                        if (H_idx >= 0 && H_idx < H && W_idx >= 0 && W_idx < W) {
                            const uint src_idx = W_idx + H_idx * nb11 + Cin_idx * nb12 + N_idx * nb13;
                            ((T_FLOAT*)&val)[v] = src_data[src_idx];
                        }
                    }
                }
            }
            Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint crs_l = 0; crs_l < BS_CRS; ++crs_l) {
            T_FLOAT regA[TS_K];
            for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
                regA[k_l_reg] = Ash[(lid_k * TS_K + k_l_reg) * BS_CRS + crs_l];
            }

            for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
                T_FLOAT4 regB = Bsh[crs_l * BS_NPQ_VEC + lid_npq * TS_NPQ_VEC + npq_l_vec_reg];
                for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
                    regC[k_l_reg][npq_l_vec_reg] = mad(convert_float(regA[k_l_reg]), convert_float4(regB), regC[k_l_reg][npq_l_vec_reg]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
        const uint k_g = offset_k + lid_k * TS_K + k_l_reg;
        if (k_g >= K) continue;

        for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
            const uint npq_g_base = offset_npq + (lid_npq * TS_NPQ_VEC + npq_l_vec_reg) * VEC_SIZE;

            const uint N_idx = npq_g_base / (OH * OW);
            const uint pq_idx = npq_g_base % (OH * OW);
            const uint OH_idx = pq_idx / OW;
            const uint OW_idx = pq_idx % OW;

            if (nb1 == OW && OW_idx + VEC_SIZE <= OW && npq_g_base + VEC_SIZE <= NPQ) {
                const uint dst_idx = OW_idx + OH_idx*nb1 + k_g*nb2 + N_idx*nb3;
                VSTORE_T_FLOAT4(regC[k_l_reg][npq_l_vec_reg], 0, &dst_data[dst_idx]);
            } else {
                T_ACCUM res = regC[k_l_reg][npq_l_vec_reg];
                for (int v = 0; v < VEC_SIZE; ++v) {
                    const uint npq_g = npq_g_base + v;
                    if (npq_g < NPQ) {
                        const uint N_idx_s = npq_g / (OH*OW);
                        const uint pq_idx_s = npq_g % (OH*OW);
                        const uint OH_idx_s = pq_idx_s / OW;
                        const uint OW_idx_s = pq_idx_s % OW;
                        const uint dst_idx_s = OW_idx_s + OH_idx_s*nb1 + k_g*nb2 + N_idx_s*nb3;
                        dst_data[dst_idx_s] = (T_FLOAT)(((float*)&res)[v]);
                    }
                }
            }
        }
    }
}
