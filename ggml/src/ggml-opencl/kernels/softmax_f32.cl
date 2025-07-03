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
kernel void kernel_soft_max(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
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
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float * psrc0 = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global float * pmask = src1 != src0 ? (global float *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float * pdst  = (global float *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        lmax = fmax(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }
    float max = sub_group_reduce_max(lmax);

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max);
        lsum += exp_psrc0;
        // Remember the result of exp here. exp is expensive, so we really do not
        // wish to compute it twice.
        pdst[i00] = exp_psrc0;
    }

    const float sum = sub_group_reduce_add(lsum);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        pdst[i00] /= sum;
    }
}
