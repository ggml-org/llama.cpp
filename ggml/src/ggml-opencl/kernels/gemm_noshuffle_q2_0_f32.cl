#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("full")))
#endif
kernel void kernel_gemm_noshuffle_q2_0_f32(
    global const uchar * src0_qs,
    global const half  * src0_d,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding
) {
    dst = (global float *)((global char *)dst + offsetd);
    const int gy = get_global_id(0);
    const int row = get_global_id(1) * 4;
    const int n_4 = n / 4;
    if (row >= m) {
        return;
    }

    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < k; i += 16) {
        const float4 scale = convert_float4(vload4(0, src0_d + row + (i / 64) * m));

        for (int l = 0; l < 16; l += 4) {
            const int ki = i + l;
            const uchar4 q = vload4(0, src0_qs + row + (ki / 4) * m);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float8 b;
                b.s0123 = convert_float4(read_imageh(src1, gy * 2     + (ki + j) * n_4));
                b.s4567 = convert_float4(read_imageh(src1, gy * 2 + 1 + (ki + j) * n_4));
                const int4 v = convert_int4((q >> (uchar4)(2*j)) & (uchar4)(3)) - (int4)(1);
                const float4 w = convert_float4(v) * scale;
                c0 += b * w.s0;
                c1 += b * w.s1;
                c2 += b * w.s2;
                c3 += b * w.s3;
            }
        }
    }

    int idx = gy * 8 * m + row;
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}
