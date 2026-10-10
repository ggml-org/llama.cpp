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

// MUL_MAT_ID with f16 expert weights (bf16 weights are stored as f16 on the device) and f32
// activations: one GEMV per (token, expert slot), N_R0_F16 rows per subgroup. Same arguments and
// grid as kernel_mul_mv_id_q8_0_f32.

#ifdef INTEL_GPU
#define N_R0_F16 4 // number of rows each subgroup works on (the code below is written for 4)
#define N_SG_F16 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_F16 4
#define N_SG_F16 2
#define N_SIMDWIDTH 64
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_id_f16_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * src2,
    ulong         offset2,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    int           ne11,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    int           ne20,
    int           ne21,
    ulong         nb21,
    int           ne0,
    int           ne1
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    src2 = (global char *)((global char *)src2 + offset2);
    dst  = (global char *)((global char *)dst  + offsetd);

    const int iid1 = get_group_id(2)/ne20;   // token
    const int idx  = get_group_id(2)%ne20;   // expert slot

    const int i02 = ((global int *) (src2 + iid1*nb21))[idx];

    const int i11 = idx % ne11;
    const int i12 = iid1;

    global char * src0_cur = src0 + i02*nb02;
    global float * y = (global float *) (src1 + i11*nb11 + i12*nb12);
    global float * dst_f32 = (global float *) (dst + (idx*ne0 + i12*ne1*ne0)*sizeof(float));

    int first_row = (get_group_id(0)*N_SG_F16 + get_sub_group_id()) * N_R0_F16;
    // The x-grid is padded to whole subgroup row-groups; slide the tail window back so every
    // fetch stays inside src0. Repeated rows are stored identically by the subgroup that owns them.
    first_row = min(first_row, max(ne01 - N_R0_F16, 0));

    // Four explicit row pointers and a float4 accumulator: the A7X compiler (Adreno 740,
    // E031.41) computes wrong results with the same loop written over pointer / float arrays.
    global half * ax0 = (global half *) (src0_cur + (ulong)min(first_row + 0, ne01 - 1)*nb01);
    global half * ax1 = (global half *) (src0_cur + (ulong)min(first_row + 1, ne01 - 1)*nb01);
    global half * ax2 = (global half *) (src0_cur + (ulong)min(first_row + 2, ne01 - 1)*nb01);
    global half * ax3 = (global half *) (src0_cur + (ulong)min(first_row + 3, ne01 - 1)*nb01);

    const int lane = get_sub_group_local_id();
    float4 sumf = (float4)(0.f);

    const int ne00_4 = ne00 / 4;
    for (int k4 = lane; k4 < ne00_4; k4 += N_SIMDWIDTH) {
        const float4 y4 = vload4(k4, y);
        sumf.s0 += dot(vload_half4(k4, ax0), y4);
        sumf.s1 += dot(vload_half4(k4, ax1), y4);
        sumf.s2 += dot(vload_half4(k4, ax2), y4);
        sumf.s3 += dot(vload_half4(k4, ax3), y4);
    }
    for (int k = ne00_4*4 + lane; k < ne00; k += N_SIMDWIDTH) {
        const float yk = y[k];
        sumf.s0 += vload_half(k, ax0) * yk;
        sumf.s1 += vload_half(k, ax1) * yk;
        sumf.s2 += vload_half(k, ax2) * yk;
        sumf.s3 += vload_half(k, ax3) * yk;
    }

    const float4 tot = (float4)(sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
                                sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3));
    if (lane == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = tot.s0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = tot.s1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = tot.s2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = tot.s3;
    }
}
