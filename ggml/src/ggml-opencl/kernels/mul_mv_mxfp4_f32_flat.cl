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

#define QK_MXFP4 32

static inline half4 mxfp4_to_fp16_packed(ushort fp4x4) {
    ushort2 fp16_packed_a, fp16_packed_b, bias_a, bias_b, sign_a, sign_b;
    fp16_packed_a.lo = (fp4x4 << 9) & 0x0E00;
    fp16_packed_a.hi = (fp4x4 << 5) & 0x0E00;
    fp16_packed_b.lo = (fp4x4 << 1) & 0x0E00;
    fp16_packed_b.hi = (fp4x4 >> 3) & 0x0E00;

    bias_a.lo = (fp16_packed_a.lo == 0) ? 0x0 : 0x3800;
    bias_a.hi = (fp16_packed_a.hi == 0) ? 0x0 : 0x3800;
    bias_b.lo = (fp16_packed_b.lo == 0) ? 0x0 : 0x3800;
    bias_b.hi = (fp16_packed_b.hi == 0) ? 0x0 : 0x3800;

    fp16_packed_a.lo = (fp16_packed_a.lo == 0x0200) ? 0x0 : fp16_packed_a.lo;
    fp16_packed_a.hi = (fp16_packed_a.hi == 0x0200) ? 0x0 : fp16_packed_a.hi;
    fp16_packed_b.lo = (fp16_packed_b.lo == 0x0200) ? 0x0 : fp16_packed_b.lo;
    fp16_packed_b.hi = (fp16_packed_b.hi == 0x0200) ? 0x0 : fp16_packed_b.hi;

    sign_a.lo = (fp4x4 << 12) & 0x8000;
    sign_a.hi = (fp4x4 << 8) & 0x8000;
    sign_b.lo = (fp4x4 << 4) & 0x8000;
    sign_b.hi = fp4x4 & 0x8000;

    fp16_packed_a = sign_a + bias_a + fp16_packed_a;
    fp16_packed_b = sign_b + bias_b + fp16_packed_b;

    return as_half4((ushort4)(fp16_packed_a, fp16_packed_b));
}

static inline float e8m0_to_fp32(uchar x) {
    int bits;
    bits = (x == 0) ? 0x00400000 : ((uint) x << 23);
    return as_float(bits);
}

#ifdef INTEL_GPU
#define N_R0_MXFP4 2 // number of rows each subgroup works on
#define N_SG_MXFP4 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_MXFP4 2
#define N_SG_MXFP4 2
#define N_SIMDWIDTH 64
#define SRC0Q_IMG
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_mxfp4_f32_flat(
#ifdef SRC0Q_IMG
    __read_only image1d_buffer_t src0_q,
#else
    global uchar * src0_q,
#endif
    global uchar * src0_e,
    global uchar * src1,
    ulong          offset1,
    global uchar * dst,
    ulong          offsetd,
    int ne00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne12,
    ulong nb11,
    ulong nb12,
    ulong nb13,
    int ne0,
    int ne1,
    int r2,
    int r3
) {
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int nb = ne00 / QK_MXFP4;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SG_MXFP4 + get_sub_group_id()) * N_R0_MXFP4;

    uint i12 = im % ne12;
    uint i13 = im / ne12;

    uint offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    // 17 = sizeof(block_mxfp4)
    offset_src0 /= 17;
#ifdef SRC0Q_IMG
    ulong offset_q = offset_src0;
#else
    global uchar16 * x_q = (global uchar16 *)(src0_q) + offset_src0;
#endif
    global uchar * x_e = src0_e + offset_src0;

    ulong offset_src1 = r1 * nb11 + i12 * nb12 + i13 * nb13;
    global float * y = (global float *)(src1 + offset_src1);

    const short ix = get_sub_group_local_id() >> 1;  // 0...15
    const short it = get_sub_group_local_id() & 1;  // 0 or 1

    float sumf[N_R0_MXFP4] = {0.f};

    global float * yb = y + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        global float4 * y4 = (global float4 *)yb;

        #pragma unroll
        for (short row = 0; row < N_R0_MXFP4; row++) {
            uchar xb_e = x_e[row * nb + ib];
#ifdef SRC0Q_IMG
            ushort4 xb_q = as_ushort4(read_imageui(src0_q, (offset_q + row * nb + ib) * 2 + it).xy);
#else
            ushort4 xb_q = vload4(0, (global ushort *)((global uchar *)(x_q + row * nb + ib) + 8 * it));
#endif

            half4 fp16x4_0 = mxfp4_to_fp16_packed(xb_q.s0);
            half4 fp16x4_1 = mxfp4_to_fp16_packed(xb_q.s1);
            float4 acc1 = y4[0] * (float4)(fp16x4_0.s0, fp16x4_0.s2, fp16x4_1.s0, fp16x4_1.s2);
            acc1 += y4[4] * (float4)(fp16x4_0.s1, fp16x4_0.s3, fp16x4_1.s1, fp16x4_1.s3);

            fp16x4_0 = mxfp4_to_fp16_packed(xb_q.s2);
            fp16x4_1 = mxfp4_to_fp16_packed(xb_q.s3);
            acc1 += y4[1] * (float4)(fp16x4_0.s0, fp16x4_0.s2, fp16x4_1.s0, fp16x4_1.s2);
            acc1 += y4[5] * (float4)(fp16x4_0.s1, fp16x4_0.s3, fp16x4_1.s1, fp16x4_1.s3);

            sumf[row] += e8m0_to_fp32(xb_e) * ((acc1.s0 + acc1.s1) + (acc1.s2 + acc1.s3));
        }

        yb += (N_SIMDWIDTH/2) * QK_MXFP4;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    for (int row = 0; row < N_R0_MXFP4 && first_row + row < ne0; ++row) {
        float sum_all = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}
