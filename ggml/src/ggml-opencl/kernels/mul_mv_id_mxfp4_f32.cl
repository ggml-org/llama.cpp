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

constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline float e8m0_to_fp32(uchar x) {
    int bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint) x << 23;
    }

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
    local float * shmem_f32 = (local float *) shmem;
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

    shmem_f32[get_sub_group_local_id()] = kvalues_mxfp4_f[get_sub_group_local_id()%16];
    barrier(CLK_LOCAL_MEM_FENCE);

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
            global uchar       * q2 = (global uchar *)(xb->qs + 8*it);

            float4 acc1 = yl[0]*(float4)(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*(float4)(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*(float4)(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*(float4)(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

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

    global char * dst_cur = dst + (i1*ne0 + i2*ne1*ne0)*sizeof(float);

    mul_mv_mxfp4_f32(src0_cur, src1_cur, dst_cur,
        ne00, nb01, nb02, nb03, ne12, nb11, nb12, nb13, ne0, ne1, r2, r3, shmem);
}
