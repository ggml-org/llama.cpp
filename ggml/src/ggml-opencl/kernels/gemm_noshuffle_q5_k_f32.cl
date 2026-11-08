#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif
#define QK_K         256
#define K_SCALE_SIZE 12

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q5_k_f32(
    global const ushort * src0_q,
    global const uchar  * src0_qh,
    global const uchar  * src0_s,
    global const half   * src0_d,
    global const half   * src0_dm,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    int num_blocks_K = k / QK_K;

    global const ushort * weight_ptr = src0_q  + gx_2;
    global const uchar  * qh_ptr     = src0_qh + gx_2;
    global const half   * d_ptr      = src0_d  + gx_2;
    global const half   * dm_ptr     = src0_dm + gx_2;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half4 d  = vload4(0, d_ptr  + sb_idx * m);
        half4 dm = vload4(0, dm_ptr + sb_idx * m);

        global const uchar * sc0 = src0_s + (gx_2+0) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        global const uchar * sc1 = src0_s + (gx_2+1) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        global const uchar * sc2 = src0_s + (gx_2+2) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        global const uchar * sc3 = src0_s + (gx_2+3) * num_blocks_K * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;

        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4(sub_idx, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc2, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc3, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        half4 scale = convert_half4(convert_float4(d)  * convert_float4((uchar4)(sv0, sv1, sv2, sv3)));
        half4 mval  = convert_half4(convert_float4(dm) * convert_float4((uchar4)(mn0, mn1, mn2, mn3)));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits4   = vload4(0, weight_ptr + (ki/4) * m);
            uchar4  qh_bits = vload4(0, qh_ptr     + (ki/8) * m);
            int     qh_shift = ki % 8;

            // j=0
            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0x000F) | (((qh_bits.s0 >> (qh_shift+0)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0x000F) | (((qh_bits.s1 >> (qh_shift+0)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0x000F) | (((qh_bits.s2 >> (qh_shift+0)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0x000F) | (((qh_bits.s3 >> (qh_shift+0)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dequantized_weights.s0 = (((bits4.s0 & 0x00F0) >> 4) | (((qh_bits.s0 >> (qh_shift+1)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (((bits4.s1 & 0x00F0) >> 4) | (((qh_bits.s1 >> (qh_shift+1)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (((bits4.s2 & 0x00F0) >> 4) | (((qh_bits.s2 >> (qh_shift+1)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (((bits4.s3 & 0x00F0) >> 4) | (((qh_bits.s3 >> (qh_shift+1)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dequantized_weights.s0 = (((bits4.s0 & 0x0F00) >> 8) | (((qh_bits.s0 >> (qh_shift+2)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (((bits4.s1 & 0x0F00) >> 8) | (((qh_bits.s1 >> (qh_shift+2)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (((bits4.s2 & 0x0F00) >> 8) | (((qh_bits.s2 >> (qh_shift+2)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (((bits4.s3 & 0x0F00) >> 8) | (((qh_bits.s3 >> (qh_shift+2)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dequantized_weights.s0 = (((bits4.s0 & 0xF000) >> 12) | (((qh_bits.s0 >> (qh_shift+3)) & 1) << 4)) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (((bits4.s1 & 0xF000) >> 12) | (((qh_bits.s1 >> (qh_shift+3)) & 1) << 4)) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (((bits4.s2 & 0xF000) >> 12) | (((qh_bits.s2 >> (qh_shift+3)) & 1) << 4)) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (((bits4.s3 & 0xF000) >> 12) | (((qh_bits.s3 >> (qh_shift+3)) & 1) << 4)) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;
        }
    }

    int idx = (gy<<3)*m + (gx<<2);

    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if (idx+3 < m*n_no_padding) {
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}
