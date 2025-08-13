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
typedef struct {
    uchar e; // E8M0
    uchar qs[QK_MXFP4/2];
} block_mxfp4;


// static inline half mxfp4_to_fp16(uchar fp4) {
//     ushort sign = (fp4 >> 3) & 0x1;
//     ushort d = (fp4 >> 2) & 0x1;
//     ushort a = fp4 & 0x7;
    
//     return (1 - sign * 2) * ((1-d) * a * 0.5f + d * (ushort)(1.2999f * a - 3.0799f));
// }

// single ushort contains 4 mxfp4 as input
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
#endif

inline void mul_mv_mxfp4_f32(
    global char * src0,
    global char * src1,
    global char * dst,
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
    int r3,
    local  char * shmem
) {
    // local float * shmem_f32 = (local float *) shmem;
    int nb = ne00/QK_MXFP4;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = 0;

    int first_row = (r0 * N_SG_MXFP4 + get_sub_group_id()) * N_R0_MXFP4;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    ulong offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_mxfp4 * x = (global block_mxfp4 *) (src0 + offset_src0);
    global float       * y = (global float       *) (src1 + offset_src1);

    const short ix = get_sub_group_local_id()/2;  // 0...15
    const short it = get_sub_group_local_id()%2;  // 0 or 1

    float4 yl[4];
    float sumf[N_R0_MXFP4] = {0.f};

    global float * yb = y + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        global float4 * y4 = (global float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < N_R0_MXFP4; row++) {
            global block_mxfp4 * xb = x + row*nb + ib;
            global ushort       * q2 = (global ushort *)(xb->qs + 8*it);

            half4 fp16x4_0 = mxfp4_to_fp16_packed(q2[0]);
            half4 fp16x4_1 = mxfp4_to_fp16_packed(q2[1]);
            float4 acc1 = yl[0]*(float4)(fp16x4_0.s0, fp16x4_0.s2, fp16x4_1.s0, fp16x4_1.s2);
            acc1 += yl[1]*(float4)(fp16x4_0.s1, fp16x4_0.s3, fp16x4_1.s1, fp16x4_1.s3);
            fp16x4_0 = mxfp4_to_fp16_packed(q2[2]);
            fp16x4_1 = mxfp4_to_fp16_packed(q2[3]);
            acc1 += yl[2]*(float4)(fp16x4_0.s0, fp16x4_0.s2, fp16x4_1.s0, fp16x4_1.s2);
            acc1 += yl[3]*(float4)(fp16x4_0.s1, fp16x4_0.s3, fp16x4_1.s1, fp16x4_1.s3);

            // float4 acc1 = yl[0]*(float4)(mxfp4_to_fp16(q2[0] &  0x0F), mxfp4_to_fp16(q2[1] &  0x0F), mxfp4_to_fp16(q2[2] &  0x0F), mxfp4_to_fp16(q2[3] &  0x0F));
            // acc1 += yl[1]*(float4)(mxfp4_to_fp16(q2[0] >> 4   ), mxfp4_to_fp16(q2[1] >> 4   ), mxfp4_to_fp16(q2[2] >> 4   ), mxfp4_to_fp16(q2[3] >> 4   ));
            // acc1 += yl[2]*(float4)(mxfp4_to_fp16(q2[4] &  0x0F), mxfp4_to_fp16(q2[5] &  0x0F), mxfp4_to_fp16(q2[6] &  0x0F), mxfp4_to_fp16(q2[7] &  0x0F));
            // acc1 += yl[3]*(float4)(mxfp4_to_fp16(q2[4] >> 4   ), mxfp4_to_fp16(q2[5] >> 4   ), mxfp4_to_fp16(q2[6] >> 4   ), mxfp4_to_fp16(q2[7] >> 4   ));

            sumf[row] += e8m0_to_fp32(xb->e) * ((acc1.s0 + acc1.s1) + (acc1.s2 + acc1.s3));
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

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_id_mxfp4_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * src2,
    ulong         offset2,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne11,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne20,
    int           ne21,
    ulong         nb21,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3,
    local  char * shmem
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    src2 = (global char *)((global char *)src2 + offset2);
    dst  = (global char *)((global char *)dst  + offsetd);

    const int iid1 = get_group_id(2)/ne20;
    const int idx  = get_group_id(2)%ne20;

    int i02 = ((global int *) (src2 + iid1*nb21))[idx];

    int i11 = idx % ne11;
    int i12 = iid1;

    int i1 = idx;
    int i2 = i12;

    global char * src0_cur = src0 + i02*nb02;
    global char * src1_cur = src1 + i11*nb11 + i12*nb12;

    // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
    //     printf("[kernel_mul_mv_id_mxfp4_f32_flat] src1(%lu): %f, src2(%lu): %d\n", offset1, ((global float*)src1)[0], offset2, ((global int*)src2)[0]);
    //     global block_mxfp4 * block = (global block_mxfp4 *)(src0);
    //     printf("[kernel_mul_mv_id_mxfp4_f32] i02: %d, offset0: %d, e: %d, q[0]: %d, q[16]: %d\n", i02, offset0, block->e, block->qs[0], block->qs[15]);
    // }

    global char * dst_cur = dst + (i1*ne0 + i2*ne1*ne0)*sizeof(float);

    mul_mv_mxfp4_f32(src0_cur, src1_cur, dst_cur,
        ne00, nb01, nb02, nb03, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shmem);
}
