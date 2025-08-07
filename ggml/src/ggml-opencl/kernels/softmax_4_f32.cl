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

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global float4 * pmask = src1 != src0 ? (global float4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float  *)(src2) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
}
