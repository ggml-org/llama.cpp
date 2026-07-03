#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

// each work-item computes a 4 (rows of A / m) x 8 (cols of B / n) output tile.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q1_0_f32(
        global const uint * src0_q,
        global const half  * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        int k,
        int m,
        int n,
        int n_no_padding,
        ulong offsetd
) {
    int n_4 = n >> 2;

    int gy   = get_global_id(0);
    int gx   = get_global_id(1);
    int gx_2 = gx << 2;
    dst  = (global float *)((global char*)dst  + offsetd);

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;

    global const uint* wptr = src0_q + gx_2;
    global const half* sptr = src0_d + gx_2;

    // 32 weights per uint32, 128 weights (one block / one scale) per 4 uint32.
    for (int i = 0; i < k; i += 32) {
        uint4 pack4 = vload4(0, wptr + (i / 32)  * m); // 4 rows, 32 K-values each
        half4 scale = vload4(0, sptr + (i / 128) * m); // 4 rows, one scale per 128

        for (int j = 0; j < 32; ++j) {
            B.s0123 = read_imageh(src1, gy * 2 + (i + j) * n_4);
            B.s4567 = read_imageh(src1, gy * 2 + (i + j) * n_4 + 1);

            // sign bit -> +-1 (half arithmetic avoids unsigned underflow)
            half4 wj = (half4)(
                2.0h * (half)((pack4.s0 >> j) & 1u) - 1.0h,
                2.0h * (half)((pack4.s1 >> j) & 1u) - 1.0h,
                2.0h * (half)((pack4.s2 >> j) & 1u) - 1.0h,
                2.0h * (half)((pack4.s3 >> j) & 1u) - 1.0h) * scale;

            c0 += B * wj.s0;
            c1 += B * wj.s1;
            c2 += B * wj.s2;
            c3 += B * wj.s3;
        }
    }

    int idx = (gy << 3) * m + (gx << 2);

    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}
