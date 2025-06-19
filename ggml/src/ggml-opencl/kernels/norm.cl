#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

//------------------------------------------------------------------------------
// norm
//------------------------------------------------------------------------------
kernel void kernel_norm(
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
    dst = (global void*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * x = (global float *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);

    // MEAN
    // parallel sum
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        sum[get_local_id(0)] += x[i00];
    }
    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    barrier(CLK_LOCAL_MEM_FENCE);
    global float * y = dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = x[i00] - mean;
        sum[get_local_id(0)] += y[i00] * y[i00];
    }

    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float variance = sum[0] / ne00;

    float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = y[i00] * scale;
    }
}
