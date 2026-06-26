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

#define SWAP(x, y, T) { T tmp = (x); (x) = (y); (y) = tmp; }

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

kernel void kernel_argsort_f32_i32(
    global float * src0,
    ulong          offset0,
    global int   * dst,
    ulong          offsetd,
    const int      ne00,
    const int      ne00_pad,
    const int      order,
    local int    * dst_row
) {
    // bitonic sort
    int col = get_local_id(0);
    int row = get_group_id(1);

    if (col >= ne00_pad) {
        return;
    }

    src0 = (global char  *)((global char *)src0 + offset0);
    dst  = (global float *)((global char *)dst  + offsetd);

    global float * x_row = src0 + row * ne00;

    // initialize indices
    dst_row[col] = col;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= ne00_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ne00 ||
                        (dst_row[ixj] < ne00 && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                } else {
                    if (dst_row[ixj] >= ne00 ||
                        (dst_row[col] < ne00 && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // copy the result to dst without the padding
    if (col < ne00) {
        dst[row * ne00 + col] = dst_row[col];
    }
}
