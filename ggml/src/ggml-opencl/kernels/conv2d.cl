#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#define T_FLOAT half
#define T_FLOAT4 half4
#define T_ACCUM float4
#define VEC_SIZE 4

#define BS_K 128
#define BS_CRS 16
#define BS_NPQ 64
#define TS_K 8
#define TS_NPQ 8
#define WG_SIZE 128

#define BS_NPQ_VEC (BS_NPQ / VEC_SIZE)
#define TS_NPQ_VEC (TS_NPQ / VEC_SIZE)

#define NT_K (BS_K / TS_K)
#define NT_NPQ (BS_NPQ / TS_NPQ)

static inline uint splitWork(uint work_size, uint block_size){
    return (work_size + block_size - 1) / block_size;
}

REQD_SUBGROUP_SIZE_64
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
    global float* knl_data = (global float*) ((global char*)p_knl + off_knl);
    global float* src_data = (global float*) ((global char*)p_src + off_src);
    global float* dst_data = (global float*) ((global char*)p_dst + off_dst);

    const uint tid = get_local_id(0);

    const uint K = Cout;
    const uint CRS = Cin*KH*KW;
    const uint NPQ = N*OH*OW;

    const uint NB_CRS = splitWork(CRS, BS_CRS);

    const uint Ash_stride = BS_CRS + 1;
    const uint Bsh_stride_vec = BS_NPQ_VEC + 1;

    local T_FLOAT* Ash = (local T_FLOAT*)shared;
    local T_FLOAT4* Bsh = (local T_FLOAT4*) &Ash[BS_K * Ash_stride];

    T_ACCUM regC[TS_K][TS_NPQ_VEC];
    for (int i = 0; i < TS_K; ++i) {
        for (int j = 0; j < TS_NPQ_VEC; ++j) {
            regC[i][j] = (T_ACCUM)(0.0f);
        }
    }

    const uint B_idx_K = get_group_id(0);
    const uint B_idx_NPQ = get_group_id(1);

    const uint T_y = tid / NT_NPQ;
    const uint T_x = tid % NT_NPQ;

    for (uint B_idx_CRS = 0; B_idx_CRS < NB_CRS; ++B_idx_CRS) {
        for(uint i = tid; i < BS_K * BS_CRS; i += WG_SIZE){
            uint k_l = i / BS_CRS;
            uint crs_l = i % BS_CRS;
            uint k_g = B_idx_K*BS_K + k_l;
            uint crs_g = B_idx_CRS*BS_CRS + crs_l;
            if(k_g < K && crs_g < CRS){
                uint Cin_idx = crs_g / (KW*KH);
                uint KH_idx = (crs_g - Cin_idx*KW*KH) / KW;
                uint KW_idx = crs_g - Cin_idx*KW*KH - KH_idx*KW;
                uint knl_idx = KW_idx + KH_idx*nb01 + Cin_idx*nb02 + k_g*nb03;
                Ash[k_l * Ash_stride + crs_l] = (T_FLOAT)knl_data[knl_idx];
            } else {
                Ash[k_l * Ash_stride + crs_l] = (T_FLOAT)0.0h;
            }
        }

        for (uint i = tid; i < BS_CRS * BS_NPQ_VEC; i += WG_SIZE) {
            uint crs_l = i / BS_NPQ_VEC;
            uint npq_l_vec = i % BS_NPQ_VEC;

            float4 val_f = (float4)(0.0f);
            uint crs_g = B_idx_CRS * BS_CRS + crs_l;

            if (crs_g < CRS) {
                uint Cin_idx = crs_g / (KW * KH);
                uint KH_idx = (crs_g - Cin_idx * KW * KH) / KW;
                uint KW_idx = crs_g - Cin_idx * KW * KH - KH_idx * KW;

                for (int v = 0; v < VEC_SIZE; ++v) {
                    uint npq_g = B_idx_NPQ * BS_NPQ + npq_l_vec * VEC_SIZE + v;
                    if (npq_g < NPQ) {
                        uint N_idx = npq_g / (OH * OW);
                        uint pq_idx = npq_g % (OH * OW);
                        uint OH_idx = pq_idx / OW;
                        uint OW_idx = pq_idx % OW;
                        int H_idx = (int)(OH_idx * s1 + KH_idx * d1 - p1);
                        int W_idx = (int)(OW_idx * s0 + KW_idx * d0 - p0);

                        if (H_idx >= 0 && H_idx < H && W_idx >= 0 && W_idx < W) {
                            uint src_idx = W_idx + H_idx * nb11 + Cin_idx * nb12 + N_idx * nb13;
                            switch (v) {
                                case 0: val_f.s0 = src_data[src_idx]; break;
                                case 1: val_f.s1 = src_data[src_idx]; break;
                                case 2: val_f.s2 = src_data[src_idx]; break;
                                case 3: val_f.s3 = src_data[src_idx]; break;
                            }
                        }
                    }
                }
            }
            Bsh[crs_l * Bsh_stride_vec + npq_l_vec] = convert_half4(val_f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint crs_l = 0; crs_l < BS_CRS; ++crs_l){
            T_FLOAT regA[TS_K];
            for(uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg){
                regA[k_l_reg] = Ash[(T_y*TS_K + k_l_reg)*Ash_stride + crs_l];
            }
            for(uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg){
                T_FLOAT4 regB = Bsh[crs_l*Bsh_stride_vec + T_x*TS_NPQ_VEC + npq_l_vec_reg];
                for(uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg){
                    regC[k_l_reg][npq_l_vec_reg] = mad((float)regA[k_l_reg], convert_float4(regB), regC[k_l_reg][npq_l_vec_reg]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg){
        uint k_g = B_idx_K * BS_K + T_y * TS_K + k_l_reg;
        if(k_g >= K) continue;

        for(uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg){
            uint npq_g_base = B_idx_NPQ * BS_NPQ + (T_x * TS_NPQ_VEC + npq_l_vec_reg) * VEC_SIZE;
            uint N_idx = npq_g_base / (OH*OW);
            uint pq_idx = npq_g_base % (OH*OW);
            uint OH_idx = pq_idx / OW;
            uint OW_idx = pq_idx % OW;

            if (nb1 == OW && OW_idx + VEC_SIZE <= OW && npq_g_base + VEC_SIZE <= NPQ) {
                uint dst_idx = OW_idx + OH_idx*nb1 + k_g*nb2 + N_idx*nb3;
                vstore4(regC[k_l_reg][npq_l_vec_reg], 0, &dst_data[dst_idx]);
            } else {
                T_ACCUM res = regC[k_l_reg][npq_l_vec_reg];
                for (int v = 0; v < VEC_SIZE; ++v) {
                    uint npq_g = npq_g_base + v;
                    if (npq_g < NPQ) {
                        uint N_idx_s = npq_g / (OH*OW);
                        uint pq_idx_s = npq_g % (OH*OW);
                        uint OH_idx_s = pq_idx_s / OW;
                        uint OW_idx_s = pq_idx_s % OW;
                        uint dst_idx_s = OW_idx_s + OH_idx_s*nb1 + k_g*nb2 + N_idx_s*nb3;
                        float val_f;
                        switch(v) {
                            case 0: val_f = res.s0; break;
                            case 1: val_f = res.s1; break;
                            case 2: val_f = res.s2; break;
                            default:val_f = res.s3; break;
                        }
                        dst_data[dst_idx_s] = val_f;
                    }
                }
            }
        }
    }
}
