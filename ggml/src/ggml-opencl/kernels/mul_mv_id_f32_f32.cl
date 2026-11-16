#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("half")))
#endif
kernel void kernel_mul_mv_id_f32_f32(
        global const char * src0,
        ulong offset0,
        global const char * src1,
        ulong offset1,
        global const char * ids,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ncols,
        int ne11,
        int src0_f16,
        ulong nb01,
        ulong nb02,
        ulong nb11,
        ulong nb12,
        ulong nb20,
        ulong nb21,
        ulong nb1,
        ulong nb2,
        int nrows) {
    const int row = get_group_id(0)*4;
    const int slot = get_group_id(1);
    const int token = get_group_id(2);
    const int tid = get_local_id(0);
    const int expert = *(global const int *) (ids + offset2 + slot*nb20 + token*nb21);
    dst += offsetd + slot*nb1 + token*nb2;
    if (expert == -1) {
        if (tid == 0) {
            for (int r = 0; r < 4 && row + r < nrows; ++r) {
                ((global float *) dst)[row + r] = 0.0f;
            }
        }
        return;
    }
    src0 += offset0 + row*nb01 + expert*nb02;
    src1 += offset1 + (slot % ne11)*nb11 + token*nb12;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = tid*4; i + 3 < ncols; i += 64*4) {
        const float4 b = vload4(0, (global const float *) src1 + i);
        for (int r = 0; r < 4 && row + r < nrows; ++r) {
            global const char * arow = src0 + r*nb01;
            const float4 a = src0_f16 ? convert_float4(vload4(0, (global const half *) arow + i)) : vload4(0, (global const float *) arow + i);
            sum[r] += dot(a, b);
        }
    }
    for (int i = ncols/4*4 + tid; i < ncols; i += 64) {
        const float b = ((global const float *) src1)[i];
        for (int r = 0; r < 4 && row + r < nrows; ++r) {
            global const char * arow = src0 + r*nb01;
            const float a = src0_f16 ? (float) ((global const half *) arow)[i] : ((global const float *) arow)[i];
            sum[r] = mad(a, b, sum[r]);
        }
    }
#ifdef cl_qcom_reqd_sub_group_size
    for (int r = 0; r < 4; ++r) {
        sum[r] = sub_group_reduce_add(sum[r]);
    }
#else
    local float sums[4][64];
    for (int r = 0; r < 4; ++r) {
        sums[r][tid] = sum[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride /= 2) {
        for (int r = 0; r < 4; ++r) {
            if (tid < stride) {
                sums[r][tid] += sums[r][tid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < 4; ++r) {
        sum[r] = sums[r][0];
    }
#endif
    if (tid == 0) {
        for (int r = 0; r < 4 && row + r < nrows; ++r) {
            ((global float *) dst)[row + r] = sum[r];
        }
    }
}
