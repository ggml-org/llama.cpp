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
        global float * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = (global float*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * psrc4 = (global float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float4 * pmask = src1 != src0 ? (global float4 *)(src1 + i01*ne00) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;
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

    const float sum = sub_group_reduce_add(lsum);

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
}
