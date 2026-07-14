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

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_l2_norm_f32(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        float eps,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * x = (global float *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * y = (global float *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float sumf = 0;

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        sumf += x[i00] * x[i00];
    }
    sumf = sub_group_reduce_add(sumf);

    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // broadcast
    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (get_local_id(0) < i) {
           sum[get_local_id(0)] += sum[get_local_id(0) + i];
       }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float scale = 1.0f/max(sqrt(sum[0]), eps);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = x[i00] * scale;
    }
}
