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

//------------------------------------------------------------------------------
// rms_norm
//------------------------------------------------------------------------------
// This kernel depends on subgroup size.
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm(
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
        local float * sum // Note, the size depends on number of subgroups
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * x_scalar = (global float *) x;
    float4 sumf = 0;
    float all_sum = 0;

#ifdef NVIDIA_GPU
    // Full-workgroup parallel reduction via __local tree reduction.
    // sum[] has get_local_size(0) entries (allocated by host as sizeof(float)*nth).
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);

    float partial = 0.0f;
    for (int i = lid; i < ne00; i += lsz) {
        float v = x_scalar[i];
        partial += v * v;
    }
    sum[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz/2; s > 0; s >>= 1) {
        if (lid < s) {
            sum[lid] += sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale_nv = rsqrt(sum[0] / (float)ne00 + eps);

    global float * y_nv = dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    for (int i = lid; i < ne00; i += lsz) {
        y_nv[i] = x_scalar[i] * scale_nv;
    }
    return;
#endif

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = sub_group_reduce_add(all_sum);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = all_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // broadcast
    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (get_local_id(0) < i) {
           sum[get_local_id(0)] += sum[get_local_id(0) + i];
       }
    }
    if (get_local_id(0) == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            sum[0] += x_scalar[i];
        }
        sum[0] /= ne00;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float * y_scalar = (global float *) y;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = x[i00] * scale;
    }
    if (get_local_id(0) == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = x_scalar[i00] * scale;
        }
    }
}

//------------------------------------------------------------------------------
// rms_norm_mul
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    // The size of sum is sizeof(float)*subgroup_size.
    // Each subgroup writes its partial sum to this array.
    // So the number of subgroups per workgroup for this kernel cannot exceed the subgroup size.
    // This is generally true -
    // for subgroup size 64, workgroup size should be less than 4096 (the max is usually 1024).
    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);

    float sumf = 0;

#ifdef NVIDIA_GPU
    // Full-workgroup parallel reduction via __local tree reduction.
    // sum[] has get_local_size(0) entries (allocated by host as sizeof(float)*nth).
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);

    // Use scalar pointers to avoid float4 alignment requirements when nb01 is
    // not a multiple of 16 bytes (e.g. views with non-contiguous rows).
    global float * x_nv = (global float *) x;
    global float * y_nv = (global float *)(dst + i03*nb3 + i02*nb2 + i01*nb1);
    global float * f_nv = (global float *) f;

    float partial = 0.0f;
    for (int i = lid; i < ne00; i += lsz) {
        float v = x_nv[i];
        partial += v * v;
    }
    sum[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz/2; s > 0; s >>= 1) {
        if (lid < s) {
            sum[lid] += sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale_nv = rsqrt(sum[0] / (float)ne00 + eps);

    for (int i = lid; i < ne00; i += lsz) {
        y_nv[i] = (x_nv[i] * scale_nv) * f_nv[i % ne10];
    }
    return;
#endif

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
    //   if (get_local_id(0) < i) {
    //       sum[get_local_id(0)] += sum[get_local_id(0) + i];
    //   }
    //}
    //if (get_local_id(0) == 0) {
    //    sum[0] /= ne00;
    //}

    //barrier(CLK_LOCAL_MEM_FENCE);

    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    float mean  = sumf / ne00;
    float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)];
    }
}
