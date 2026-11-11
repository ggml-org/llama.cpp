#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("full")))
#endif
kernel void kernel_gemm_noshuffle_iq4_xs_f32(
    global const ushort * src0_qs,
    global const uint * src0_sl,
    global const ushort * src0_sh,
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

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < k; i += 16) {
        const uint4 sl = vload4(0, src0_sl + row + (i / 256) * m);
        const uint4 sh = convert_uint4(vload4(0, src0_sh + row + (i / 256) * m));
        const half4 d = vload4(0, src0_d + row + (i / 256) * m);
        const int sb = (i / 32) & 7;
        const int4 ls = convert_int4(((sl >> (uint4)(4*sb)) & (uint4)(15)) | (((sh >> (uint4)(2*sb)) & (uint4)(3)) << (uint4)(4))) - (int4)(32);
        const half4 scale = convert_half4(convert_float4(d) * convert_float4(ls));

        for (int l = 0; l < 16; l += 4) {
            const int ki = i + l;
            const ushort4 q = vload4(0, src0_qs + row + (ki / 4) * m);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                half8 b;
                b.s0123 = read_imageh(src1, gy * 2     + (ki + j) * n_4);
                b.s4567 = read_imageh(src1, gy * 2 + 1 + (ki + j) * n_4);
                const ushort4 indices = (q >> (ushort4)(4*j)) & (ushort4)(15);
                const half4 w = (half4)(iq4_xs_values[indices.s0], iq4_xs_values[indices.s1],
                                       iq4_xs_values[indices.s2], iq4_xs_values[indices.s3]) * scale;
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
