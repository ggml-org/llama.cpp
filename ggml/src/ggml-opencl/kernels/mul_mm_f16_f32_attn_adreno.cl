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
// Adreno compilers that expose only cl_qcom_subgroup_shuffle do not declare the KHR
// name, so calling it is an implicit declaration and the program fails to build.
// Route it to the qcom builtin.
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
#endif

#define N_K_ROWS_GQA   16
#define GQA_RATIO_GQA  8
#define LANES_PER_QH   8
#define DK_VEC_GQA     32

#define N_DV_ROWS_Y8GQA  8
#define GQA_RATIO_Y8GQA  8
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
