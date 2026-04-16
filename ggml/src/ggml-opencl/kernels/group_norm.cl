#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef GGML_OPENCL_NO_REQD_SUBGROUP_SIZE
// Normal path: use subgroup operations for reduction

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

// Workgroup must be a subgroup
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_group_norm(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne,
        int group_size,
        float eps
) {
    src0 = (global float  *)((global char *)src0 + offset0);
    dst  = (global float *)((global char *)dst  + offsetd);

    int start = get_group_id(0) * group_size;
    int end   = start + group_size;

    start += get_local_id(0);

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f;

    for (int j = start; j < end; j += get_local_size(0)) {
        tmp += src0[j];
    }

    tmp = sub_group_reduce_add(tmp);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += get_local_size(0)) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = sub_group_reduce_add(tmp);

    const float variance = tmp / group_size;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start; j < end; j += get_local_size(0)) {
        dst[j] *= scale;
    }
}

//------------------------------------------------------------------------------
// group_norm_mul_add
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_group_norm_mul_add(
        global float * src0, ulong offset0,
        global float * src1, ulong offset1,
        global float * src2, ulong offset2,
        global float * dst, ulong offsetd,
        int ne,
        int group_size,
        float eps
) {
    src0 = (global float *)((global char *)src0 + offset0);
    src1 = (global float *)((global char *)src1 + offset1);
    src2 = (global float *)((global char *)src2 + offset2);
    dst  = (global float *)((global char *)dst  + offsetd);

    int start = get_group_id(0) * group_size;
    int end = start + group_size;
    if (end > ne) {
        end = ne;
    }

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int j = start + get_local_id(0); j < end; j += get_local_size(0)) {
        float val = src0[j];
        sum += val;
        sum_sq += val*val;
    }

    sum = sub_group_reduce_add(sum);
    sum_sq = sub_group_reduce_add(sum_sq);

    const float mean = sum / group_size;
    const float var = sum_sq / group_size - mean * mean;
    const float scale = rsqrt(var + eps);

    for (int j = start + get_local_id(0); j < end; j += get_local_size(0)) {
        dst[j] = ((src0[j] - mean) * scale) * src1[j] + src2[j];
    }
}

#else // GGML_OPENCL_NO_REQD_SUBGROUP_SIZE
// Fallback path: local-memory tree reduction, no subgroup operations.

#define MAX_LOCAL_SIZE 1024

//------------------------------------------------------------------------------
// group_norm (fallback)
//------------------------------------------------------------------------------
kernel void kernel_group_norm(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne,
        int group_size,
        float eps
) {
    src0 = (global float *)((global char *)src0 + offset0);
    dst  = (global float *)((global char *)dst  + offsetd);

    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    int start = get_group_id(0) * group_size;
    int end   = start + group_size;
    if (end >= ne) {
        end = ne;
    }

    __local float lmem[MAX_LOCAL_SIZE];

    // Phase 1: compute sum for mean
    float tmp = 0.0f;
    for (int j = start + lid; j < end; j += lsize) {
        tmp += src0[j];
    }

    // Tree reduction for sum
    lmem[lid] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            lmem[lid] += lmem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float mean = lmem[0] / group_size;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: compute variance
    tmp = 0.0f;
    for (int j = start + lid; j < end; j += lsize) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    // Tree reduction for variance
    lmem[lid] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            lmem[lid] += lmem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float variance = lmem[0] / group_size;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start + lid; j < end; j += lsize) {
        dst[j] *= scale;
    }
}

//------------------------------------------------------------------------------
// group_norm_mul_add (fallback)
//------------------------------------------------------------------------------
kernel void kernel_group_norm_mul_add(
        global float * src0, ulong offset0,
        global float * src1, ulong offset1,
        global float * src2, ulong offset2,
        global float * dst, ulong offsetd,
        int ne,
        int group_size,
        float eps
) {
    src0 = (global float *)((global char *)src0 + offset0);
    src1 = (global float *)((global char *)src1 + offset1);
    src2 = (global float *)((global char *)src2 + offset2);
    dst  = (global float *)((global char *)dst  + offsetd);

    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    int start = get_group_id(0) * group_size;
    int end = start + group_size;
    if (end > ne) {
        end = ne;
    }

    __local float lmem[MAX_LOCAL_SIZE];
    __local float lmem2[MAX_LOCAL_SIZE];

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int j = start + lid; j < end; j += lsize) {
        float val = src0[j];
        sum += val;
        sum_sq += val*val;
    }

    // Tree reduction for sum and sum_sq
    lmem[lid] = sum;
    lmem2[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            lmem[lid] += lmem[lid + s];
            lmem2[lid] += lmem2[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = lmem[0];
    sum_sq = lmem2[0];

    const float mean = sum / group_size;
    const float var = sum_sq / group_size - mean * mean;
    const float scale = rsqrt(var + eps);

    for (int j = start + lid; j < end; j += lsize) {
        dst[j] = ((src0[j] - mean) * scale) * src1[j] + src2[j];
    }
}

#endif // GGML_OPENCL_NO_REQD_SUBGROUP_SIZE
