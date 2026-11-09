#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("full")))
#endif
kernel void kernel_gemm_noshuffle_q3_k_f32(
    global const uchar * src0_qs,
    global const uchar * src0_hm,
    global const uint  * src0_sc,
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
        const int is = (i / 16) & 15;
        const int word = is >> 2;
        const int shift = 8 * (is & 3);
        const int sc_base = row + 3 * (i / 256) * m;
        const uint4 low = vload4(0, src0_sc + sc_base + (word & 1) * m) >> (uint4)(4 * (word >> 1));
        const uint4 high = vload4(0, src0_sc + sc_base + 2 * m) >> (uint4)(2 * word);
        const int4 sc = convert_int4(((low >> (uint4)(shift)) & (uint4)(15)) | (((high >> (uint4)(shift)) & (uint4)(3)) << (uint4)(4))) - (int4)(32);
        const half4 d = vload4(0, src0_d + row + (i / 256) * m);
        const half4 scale = convert_half4(convert_float4(d) * convert_float4(sc));

        for (int l = 0; l < 16; l += 4) {
            const int ki = i + l;
            const uchar4 bits = vload4(0, src0_qs + row + (ki / 4) * m);
            const uchar4 hbits = vload4(0, src0_hm + row + (ki / 8) * m);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                half8 b;
                b.s0123 = read_imageh(src1, gy * 2     + (ki + j) * n_4);
                b.s4567 = read_imageh(src1, gy * 2 + 1 + (ki + j) * n_4);
                const int4 low = convert_int4((bits >> (uchar4)(2 * j)) & (uchar4)(3));
                const int4 high = convert_int4((hbits >> (uchar4)((ki & 7) + j)) & (uchar4)(1));
                const half4 w = convert_half4(low - (int4)(4) + (int4)(4) * high) * scale;
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
