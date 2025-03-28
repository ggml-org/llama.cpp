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
#error "Subgroup not supported on your device."
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
// TODO: do not know how to choose subgroup size on other GPUs.
#error "Selecting subgroup size is not supported on your device."
#endif

kernel void kernel_im2col_f32(
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        ulong batch_offset,
        ulong delta_offset,
        long IW,
        long IH,
        long IC,
        long OW,
        long OH,
        long KW,
        long KH,
        long pelements,
        long CHW,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        int  d0,
        int  d1
) {
    // threadIdx.x + blockIdx.x * blockDim.x
    long i = get_global_id(0);
    if (i >= pelements) {
        return;
    }

    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    long  ksize = OW * (KH > 1 ? KW : 1);
    long  kx = i / ksize;
    long  kd = kx * ksize;
    long  ky = (i - kd) / OW;
    long  ix = i % OW;

    long  oh = get_group_id(1);
    long  batch = get_group_id(2) / IC;
    long  ic = get_group_id(2) % IC;

    long iiw = ix * s0 + kx * d0 - p0;
    long iih = oh * s1 + ky * d1 - p1;

    long offset_dst =
        ((batch * OH + oh) * OW + ix) * CHW +
        (ic * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = 0.0f;
    } else {
        long offset_src = ic * delta_offset + batch * batch_offset;
        dst[offset_dst] = src1[offset_src + iih * IW + iiw];
    }
}

kernel void kernel_im2col_f16(
        global float * src1,
        ulong offset1,
        global half  * dst,
        ulong offsetd,
        ulong batch_offset,
        ulong delta_offset,
        long IW,
        long IH,
        long IC,
        long OW,
        long OH,
        long KW,
        long KH,
        long pelements,
        long CHW,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        int  d0,
        int  d1
) {
    long i = get_global_id(0);

    if (i >= pelements) {
        return;
    }

    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global half*)((global char*)dst + offsetd);

    long  ksize = OW * (KH > 1 ? KW : 1);
    long  kx = i / ksize;
    long  kd = kx * ksize;
    long  ky = (i - kd) / OW;
    long  ix = i % OW;

    long  oh = get_group_id(1);
    long  batch = get_group_id(2) / IC;
    long  ic = get_group_id(2) % IC;

    long iiw = ix * s0 + kx * d0 - p0;
    long iih = oh * s1 + ky * d1 - p1;

    long offset_dst =
        ((batch * OH + oh) * OW + ix) * CHW +
        (ic * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = 0.0f;
    } else {
        long offset_src = ic * delta_offset + batch * batch_offset;
        dst[offset_dst] = src1[offset_src + iih * IW + iiw];
    }
}
