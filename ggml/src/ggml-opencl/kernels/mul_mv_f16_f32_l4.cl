#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

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
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#endif

// Assumes row size (ne00) is a multiple of 4
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nrows = ne11;
    int r0 = get_group_id(0);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half4 * x4 = (global half4 *) (src0 + offset_src0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

        global float4 * y4 = (global float4 *) (src1 + offset_src1);

        float sumf = 0;
        for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
            sumf += convert_float(x4[i].s0) * y4[i].s0;
            sumf += convert_float(x4[i].s1) * y4[i].s1;
            sumf += convert_float(x4[i].s2) * y4[i].s2;
            sumf += convert_float(x4[i].s3) * y4[i].s3;
        }

        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

// Each subgroup produces DR_NDST outputs, assumes ne11 == 1
#define MUL_MAT_F16_F32_L4_DR_NDST 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_dr(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * MUL_MAT_F16_F32_L4_DR_NDST;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // assume ne11 == 1
    const ulong offset_src1 = i12*nb12 + i13*nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    global half4 * x4[MUL_MAT_F16_F32_L4_DR_NDST];
    float          sumf[MUL_MAT_F16_F32_L4_DR_NDST];

    const ulong   k_head_off = (i12/r2)*nb02 + (i13/r3)*nb03;

    #pragma unroll
    for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
        int       r0   = r0_base + n;
        int       r0c  = r0 < ne01 ? r0 : 0;
        ulong     off  = (ulong)r0c*nb01 + k_head_off;
        x4[n]   = (global half4 *)(src0 + off);
        sumf[n] = 0.0f;
    }

    const int n_chunks = ne00 / 4;
    const int sg_size  = get_max_sub_group_size();
    const int lid      = get_sub_group_local_id();

    for (int i = lid; i < n_chunks; i += sg_size) {
        float4 q = y4[i];
        #pragma unroll
        for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
            float4 k = convert_float4(x4[n][i]);
            sumf[n] = mad(k.s0, q.s0, sumf[n]);
            sumf[n] = mad(k.s1, q.s1, sumf[n]);
            sumf[n] = mad(k.s2, q.s2, sumf[n]);
            sumf[n] = mad(k.s3, q.s3, sumf[n]);
        }
    }

    #pragma unroll
    for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
        float reduced = sub_group_reduce_add(sumf[n]);
        int   r0      = r0_base + n;
        if (lid == 0 && r0 < ne01) {
            dst[im*ne1*ne0 + r0] = reduced;
        }
    }
}

// Kernels for decoding, Adreno only for now
#define MUL_MAT_F16_F32_L4_DR_LS_R2_MAX 8

#ifdef ADRENO_GPU
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mat_f16_f32_l4_dr_ls(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * 2;
    const int kv_grp  = get_group_id(2);   // KV head group; im = kv_grp*r2 + q

    const int i12_kv = kv_grp % ne02;
    const int i13_kv = kv_grp / ne02;

    const int lid     = get_sub_group_local_id();
    const int subhalf = lid >> 5;          // 0 or 1 (which K row in the WG)
    const int intra   = lid & 31;          // 0..31 (lane within the half)

    const int r0  = r0_base + subhalf;
    const int r0c = r0 < ne01 ? r0 : 0;    // clamp OOB to row 0; skip write below

    // K row pointer for this lane (one K row per half-wave).
    const ulong k_off = (ulong)r0c*nb01 + (ulong)i12_kv*nb02 + (ulong)i13_kv*nb03;
    global half4 * x4 = (global half4 *)(src0 + k_off);

    global float4 * y4[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        const int i12_q = i12_kv*r2 + q;
        const ulong q_off = (ulong)i12_q*nb12 + (ulong)i13_kv*nb13;
        y4[q] = (global float4 *)(src1 + q_off);
    }

    float partial[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        partial[q] = 0.0f;
    }

    const int n_chunks = ne00 / 4;

    for (int i = intra; i < n_chunks; i += 32) {
        float4 k = convert_float4(x4[i]);

        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                float4 v = y4[q][i];
                partial[q] = mad(k.s0, v.s0, partial[q]);
                partial[q] = mad(k.s1, v.s1, partial[q]);
                partial[q] = mad(k.s2, v.s2, partial[q]);
                partial[q] = mad(k.s3, v.s3, partial[q]);
            }
        }
    }

    // half-wave reduction
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        if (q < r2) {
            partial[q] += sub_group_shuffle_xor(partial[q],  1u);
            partial[q] += sub_group_shuffle_xor(partial[q],  2u);
            partial[q] += sub_group_shuffle_xor(partial[q],  4u);
            partial[q] += sub_group_shuffle_xor(partial[q],  8u);
            partial[q] += sub_group_shuffle_xor(partial[q], 16u);
        }
    }

    if (intra == 0 && r0 < ne01) {
        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                const int im = i12_kv*r2 + q + i13_kv*ne12;
                dst[im*ne1*ne0 + r0] = partial[q];
            }
        }
    }
}

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mat_f16_f32_l4_dr_lq(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * 4;
    const int kv_grp  = get_group_id(2);

    const int i12_kv = kv_grp % ne02;
    const int i13_kv = kv_grp / ne02;

    const int lid   = get_sub_group_local_id();
    const int subq  = lid >> 4;            // 0..3 (which K row)
    const int intra = lid & 15;            // 0..15 (lane within quarter)

    const int r0  = r0_base + subq;
    const int r0c = r0 < ne01 ? r0 : 0;

    const ulong k_off = (ulong)r0c*nb01 + (ulong)i12_kv*nb02 + (ulong)i13_kv*nb03;
    global half4 * x4 = (global half4 *)(src0 + k_off);

    global float4 * y4[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        const int i12_q = i12_kv*r2 + q;
        const ulong q_off = (ulong)i12_q*nb12 + (ulong)i13_kv*nb13;
        y4[q] = (global float4 *)(src1 + q_off);
    }

    float partial[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        partial[q] = 0.0f;
    }

    const int n_chunks = ne00 / 4;

    for (int i = intra; i < n_chunks; i += 16) {
        float4 k = convert_float4(x4[i]);

        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                float4 v = y4[q][i];
                partial[q] = mad(k.s0, v.s0, partial[q]);
                partial[q] = mad(k.s1, v.s1, partial[q]);
                partial[q] = mad(k.s2, v.s2, partial[q]);
                partial[q] = mad(k.s3, v.s3, partial[q]);
            }
        }
    }

    // quarter-wave reduction
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        if (q < r2) {
            partial[q] += sub_group_shuffle_xor(partial[q], 1u);
            partial[q] += sub_group_shuffle_xor(partial[q], 2u);
            partial[q] += sub_group_shuffle_xor(partial[q], 4u);
            partial[q] += sub_group_shuffle_xor(partial[q], 8u);
        }
    }

    if (intra == 0 && r0 < ne01) {
        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                const int im = i12_kv*r2 + q + i13_kv*ne12;
                dst[im*ne1*ne0 + r0] = partial[q];
            }
        }
    }
}
#endif // ADRENO_GPU

#define N_ROWS_PER_WG 8
#define N_OUTS_PER_WG 8

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_ROWS_PER_WG;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    __local float4 q_loc[64];   // ne00/4 max for sub_group_size 64
    if (sgs_lid < ne00 / 4) {
        q_loc[sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int dr = 0; dr < N_ROWS_PER_WG; ++dr) {
        const int r0 = r0_base + dr;
        if (r0 >= ne01) return;

        const ulong offset_src0 = r0 * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        float sumf = 0.0f;
        for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
            const half4   k4 = x4[i];
            const float4  q  = q_loc[i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        const float all_sum = sub_group_reduce_add(sumf);
        if (sgs_lid == 0) {
            dst[im * ne1 * ne0 + r0] = all_sum;  // ne11 == 1, so r1==0
        }
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_OUTS_PER_WG;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    global half4 * x4_o[N_OUTS_PER_WG];
    #pragma unroll
    for (int o = 0; o < N_OUTS_PER_WG; ++o) {
        const int r0 = r0_base + o;
        const int r0c = (r0 < ne01) ? r0 : 0;
        const ulong off = r0c * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        x4_o[o] = (global half4 *)(src0 + off);
    }

    float sum[N_OUTS_PER_WG] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        const float4 q4 = y4[i];
        #pragma unroll
        for (int o = 0; o < N_OUTS_PER_WG; ++o) {
            const half4 v4 = x4_o[o][i];
            sum[o] += convert_float(v4.s0) * q4.s0
                    + convert_float(v4.s1) * q4.s1
                    + convert_float(v4.s2) * q4.s2
                    + convert_float(v4.s3) * q4.s3;
        }
    }

    #pragma unroll
    for (int o = 0; o < N_OUTS_PER_WG; ++o) {
        const int r0 = r0_base + o;
        const float s = sub_group_reduce_add(sum[o]);
        if (sgs_lid == 0 && r0 < ne01) {
            dst[im * ne1 * ne0 + r0] = s;
        }
    }
}

#define N_OUTS_PAIR  8
#define N_PAIRS_PAIR (N_OUTS_PAIR / 2)

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_pair(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int half_id = sgs_lid >> 5;     // 0 = lower half, 1 = upper half
    const int lane_h  = sgs_lid & 31;     // lane 0..31 within half

    const int r0_base = get_group_id(0) * N_OUTS_PAIR;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    __local float4 q_loc[64];   // ne00/4 max for sub_group_size 64
    if (sgs_lid < ne00 / 4) {
        q_loc[sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dk_vec = ne00 / 4;

    #pragma unroll
    for (int p = 0; p < N_PAIRS_PAIR; ++p) {
        const int r0 = r0_base + 2 * p + half_id;

        const ulong offset_src0 = r0 * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        float sumf = 0.0f;
        for (int i = lane_h; i < dk_vec; i += 32) {
            const half4  k4 = x4[i];
            const float4 q  = q_loc[i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        sumf += sub_group_shuffle_xor(sumf, 16);
        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_h == 0) {
            dst[im * ne1 * ne0 + r0] = sumf;
        }
    }
}

#define N_K_ROWS_GQA   16
#define GQA_RATIO_GQA  8
#define LANES_PER_QH   8    // 64 / GQA_RATIO_GQA
#define DK_VEC_GQA     32   // DK / 4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;       // 0..7: which Q-head (8 per WG)
    const int lane_q  = sgs_lid & 7;        // 0..7: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;           // K-head index (also K2 batch)
    const int i03 = im_kv / ne02;           // n13 batch index

    const int q_head_lo = i02 * GQA_RATIO_GQA;

    __local float4 q_loc[GQA_RATIO_GQA * DK_VEC_GQA];   // 4 × 32 = 128 float4
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);

        if (sgs_lid < DK_VEC_GQA) {
            q_loc[qh * DK_VEC_GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K base offset for this WG. All 8 K-rows × 4 Q-heads share this K-head.
    const ulong offset_src0_base = (i02) * nb02 + (i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA; ++dr) {
        const int r0 = r0_base + dr;

        const ulong offset_src0 = r0 * nb01 + offset_src0_base;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        float sumf = 0.0f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = lane_q + t * LANES_PER_QH;   // 8, 16, 24-step
            const half4  k4 = x4[i];
            const float4 q  = q_loc[q_id * DK_VEC_GQA + i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

#define N_DV_ROWS_Y8GQA  8
#define GQA_RATIO_Y8GQA  8

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8_gqa(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_DV_ROWS_Y8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;           // K-head index
    const int i03 = im_kv / ne02;           // n13 batch index

    // GQA Q-heads sharing this K-head.
    const int q_head_lo = i02 * GQA_RATIO_Y8GQA;

    global float4 * y4_q[GQA_RATIO_Y8GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        y4_q[qh] = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
    }

    global half4 * x4_o[N_DV_ROWS_Y8GQA];
    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0 = r0_base + o;
        const int r0c = (r0 < ne01) ? r0 : 0;
        const ulong off = r0c * nb01 + (i02) * nb02 + (i03 / r3) * nb03;
        x4_o[o] = (global half4 *)(src0 + off);
    }

    float sum[N_DV_ROWS_Y8GQA][GQA_RATIO_Y8GQA] = { {0.0f} };

    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        // load 8 V values (one per DV row), same K-head, K-pos = i.
        half4 v[N_DV_ROWS_Y8GQA];
        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            v[o] = x4_o[o][i];
        }

        // load 8 softmax values (one per Q-head).
        float4 q[GQA_RATIO_Y8GQA];
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            q[qh] = y4_q[qh][i];
        }

        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            const float4 vf = (float4)(convert_float(v[o].s0),
                                       convert_float(v[o].s1),
                                       convert_float(v[o].s2),
                                       convert_float(v[o].s3));
            #pragma unroll
            for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
                sum[o][qh] += vf.s0 * q[qh].s0
                            + vf.s1 * q[qh].s1
                            + vf.s2 * q[qh].s2
                            + vf.s3 * q[qh].s3;
            }
        }
    }

    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0 = r0_base + o;
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            const float s = sub_group_reduce_add(sum[o][qh]);
            if (sgs_lid == 0 && r0 < ne01) {
                const int im_out = i03 * ne12 + (q_head_lo + qh);
                dst[im_out * ne1 * ne0 + r0] = s;
            }
        }
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa4_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;       // 0..7: which Q-head (8 per WG)
    const int lane_q  = sgs_lid & 7;        // 0..7: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_GQA;

    __local float4 q_loc[GQA_RATIO_GQA * DK_VEC_GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_GQA) {
            q_loc[qh * DK_VEC_GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int pitch_px_row  = (int)(nb01 >> 4);
    const int pitch_px_head = (int)(nb02 >> 4);
    const int pitch_px_n13  = (int)(nb03 >> 4);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px_base = r0 * pitch_px_row + head_px_base;

        float sumf = 0.0f;
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int p = lane_q + t * LANES_PER_QH;          // pixel idx in row, 0..15
            const half8 k8 = as_half8(read_imagef(src0_img, row_px_base + p));
            const int   i0 = 2 * p;                            // first half4 idx
            const float4 qa = q_loc[q_id * DK_VEC_GQA + i0    ];
            const float4 qb = q_loc[q_id * DK_VEC_GQA + i0 + 1];
            sumf += convert_float(k8.s0) * qa.s0
                  + convert_float(k8.s1) * qa.s1
                  + convert_float(k8.s2) * qa.s2
                  + convert_float(k8.s3) * qa.s3
                  + convert_float(k8.s4) * qb.s0
                  + convert_float(k8.s5) * qb.s1
                  + convert_float(k8.s6) * qb.s2
                  + convert_float(k8.s7) * qb.s3;
        }

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8_gqa_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_DV_ROWS_Y8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Y8GQA;

    // Q (= softmax(KQ)) base pointers per Q-head
    global float4 * y4_q[GQA_RATIO_Y8GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        y4_q[qh] = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
    }

    const int pitch_px_row  = (int)(nb01 >> 3);
    const int pitch_px_head = (int)(nb02 >> 3);
    const int pitch_px_n13  = (int)(nb03 >> 3);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    // per-DV-row pixel base
    int row_px_base[N_DV_ROWS_Y8GQA];
    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0  = r0_base + o;
        const int r0c = (r0 < ne01) ? r0 : 0;
        row_px_base[o] = r0c * pitch_px_row + head_px_base;
    }

    float sum[N_DV_ROWS_Y8GQA][GQA_RATIO_Y8GQA] = { {0.0f} };

    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        half4 v[N_DV_ROWS_Y8GQA];

        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            v[o] = read_imageh(src0_img, row_px_base[o] + i);
        }

        float4 q[GQA_RATIO_Y8GQA];
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            q[qh] = y4_q[qh][i];
        }
        // 64 mads.
        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            const float4 vf = (float4)(convert_float(v[o].s0),
                                       convert_float(v[o].s1),
                                       convert_float(v[o].s2),
                                       convert_float(v[o].s3));
            #pragma unroll
            for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
                sum[o][qh] += vf.s0 * q[qh].s0
                            + vf.s1 * q[qh].s1
                            + vf.s2 * q[qh].s2
                            + vf.s3 * q[qh].s3;
            }
        }
    }

    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0 = r0_base + o;
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            const float s = sub_group_reduce_add(sum[o][qh]);
            if (sgs_lid == 0 && r0 < ne01) {
                const int im_out = i03 * ne12 + (q_head_lo + qh);
                dst[im_out * ne1 * ne0 + r0] = s;
            }
        }
    }
}

#define N_K_ROWS_GQA_R4   16
#define GQA_RATIO_R4      4
#define LANES_PER_QH_R4   16    // = 64 / GQA_RATIO_R4
#define DK_VEC_R4         32    // DK / 4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 4;       // 0..3
    const int lane_q  = sgs_lid & 15;       // 0..15

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA_R4;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_R4;

    __local float4 q_loc[GQA_RATIO_R4 * DK_VEC_R4];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_R4; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_R4) {
            q_loc[qh * DK_VEC_R4 + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int pitch_px_row  = (int)(nb01 >> 4);
    const int pitch_px_head = (int)(nb02 >> 4);
    const int pitch_px_n13  = (int)(nb03 >> 4);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA_R4; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px_base = r0 * pitch_px_row + head_px_base;

        const int p = lane_q;
        const half8 k8 = as_half8(read_imagef(src0_img, row_px_base + p));
        const int   i0 = 2 * p;
        const float4 qa = q_loc[q_id * DK_VEC_R4 + i0    ];
        const float4 qb = q_loc[q_id * DK_VEC_R4 + i0 + 1];

        float sumf =
              convert_float(k8.s0) * qa.s0
            + convert_float(k8.s1) * qa.s1
            + convert_float(k8.s2) * qa.s2
            + convert_float(k8.s3) * qa.s3
            + convert_float(k8.s4) * qb.s0
            + convert_float(k8.s5) * qb.s1
            + convert_float(k8.s6) * qb.s2
            + convert_float(k8.s7) * qb.s3;

        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

#define N_K_ROWS_GQA_R2_DK256   16
#define GQA_RATIO_R2            2
#define LANES_PER_QH_R2         32    // = 64 / GQA_RATIO_R2
#define DK_VEC_DK256            64    // DK / 4 for DK=256

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 5;       // 0..1
    const int lane_q  = sgs_lid & 31;       // 0..31

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA_R2_DK256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_R2;

    __local float4 q_loc[GQA_RATIO_R2 * DK_VEC_DK256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_R2; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_DK256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int pitch_px_row  = (int)(nb01 >> 4);
    const int pitch_px_head = (int)(nb02 >> 4);
    const int pitch_px_n13  = (int)(nb03 >> 4);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA_R2_DK256; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px_base = r0 * pitch_px_row + head_px_base;

        const int p = lane_q;
        const half8 k8 = as_half8(read_imagef(src0_img, row_px_base + p));
        const int   i0 = 2 * p;
        const float4 qa = q_loc[q_id * DK_VEC_DK256 + i0    ];
        const float4 qb = q_loc[q_id * DK_VEC_DK256 + i0 + 1];

        float sumf =
              convert_float(k8.s0) * qa.s0
            + convert_float(k8.s1) * qa.s1
            + convert_float(k8.s2) * qa.s2
            + convert_float(k8.s3) * qa.s3
            + convert_float(k8.s4) * qb.s0
            + convert_float(k8.s5) * qb.s1
            + convert_float(k8.s6) * qb.s2
            + convert_float(k8.s7) * qb.s3;

        sumf += sub_group_shuffle_xor(sumf, 16);
        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}
