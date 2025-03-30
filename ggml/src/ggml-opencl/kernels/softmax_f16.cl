#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#error "Subgroups not supported on your device."
#endif

#ifdef cl_intel_required_subgroup_size
// Always use subgroup size of 32 on Intel.
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
// Always use subgroups size of 64 on Adreno.
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#error "Selecting subgroup size is not supported on your device."
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_f16(
        global float * src0,
        ulong offset0,
        global half * src1,
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
    src0 = (global float *)((global char *)src0 + offset0);
    src1 = (global half *)((global char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    global half  * pmask = (global char *)src1 != (global char *)src0 ? src1 + i01*ne00 : 0;
    global float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

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
