#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

constant half kvalues_iq4nl[16] = {
    (half)-127.f, (half)-104.f, (half)-83.f, (half)-65.f,
    (half) -49.f, (half) -35.f, (half)-22.f, (half)-10.f,
    (half)   1.f, (half)  13.f, (half) 25.f, (half) 38.f,
    (half)  53.f, (half)  69.f, (half) 89.f, (half)113.f
};

// Packed LUT: 2 FP16 values per uint, 8 unique constant loads instead of 16
constant uint iq4nl_packed[8] = {
    0xD680D7F0u,  // idx 0,1: -127, -104
    0xD410D530u,  // idx 2,3: -83, -65
    0xD060D220u,  // idx 4,5: -49, -35
    0xC900CD80u,  // idx 6,7: -22, -10
    0x4A803C00u,  // idx 8,9: 1, 13
    0x50C04E40u,  // idx 10,11: 25, 38
    0x545052A0u,  // idx 12,13: 53, 69
    0x57105590u   // idx 14,15: 89, 113
};

// Packed dequant: 1 uint constant load (8-way divergence) + shift + as_half
#define IQ4_NL_DEQUANT(nibble) as_half((ushort)(iq4nl_packed[(nibble) >> 1] >> (((nibble) & 1u) << 4)))

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_gemm_noshuffle_iq4_nl_f32(
        global const ushort * src0_q,
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

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort * weight_ptr = src0_q + gx_2;
    global const half * scale_ptr = src0_d + gx_2;

    for (int i = 0; i < k; i += 4) {
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);

        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));

        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        // j=0
        dequantized_weights.s0 = IQ4_NL_DEQUANT(bits4.s0 & 0x000Fu) * scale.s0;
        dequantized_weights.s1 = IQ4_NL_DEQUANT(bits4.s1 & 0x000Fu) * scale.s1;
        dequantized_weights.s2 = IQ4_NL_DEQUANT(bits4.s2 & 0x000Fu) * scale.s2;
        dequantized_weights.s3 = IQ4_NL_DEQUANT(bits4.s3 & 0x000Fu) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = IQ4_NL_DEQUANT((bits4.s0 >> 4) & 0x000Fu) * scale.s0;
        dequantized_weights.s1 = IQ4_NL_DEQUANT((bits4.s1 >> 4) & 0x000Fu) * scale.s1;
        dequantized_weights.s2 = IQ4_NL_DEQUANT((bits4.s2 >> 4) & 0x000Fu) * scale.s2;
        dequantized_weights.s3 = IQ4_NL_DEQUANT((bits4.s3 >> 4) & 0x000Fu) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = IQ4_NL_DEQUANT((bits4.s0 >> 8) & 0x000Fu) * scale.s0;
        dequantized_weights.s1 = IQ4_NL_DEQUANT((bits4.s1 >> 8) & 0x000Fu) * scale.s1;
        dequantized_weights.s2 = IQ4_NL_DEQUANT((bits4.s2 >> 8) & 0x000Fu) * scale.s2;
        dequantized_weights.s3 = IQ4_NL_DEQUANT((bits4.s3 >> 8) & 0x000Fu) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = IQ4_NL_DEQUANT((bits4.s0 >> 12) & 0x000Fu) * scale.s0;
        dequantized_weights.s1 = IQ4_NL_DEQUANT((bits4.s1 >> 12) & 0x000Fu) * scale.s1;
        dequantized_weights.s2 = IQ4_NL_DEQUANT((bits4.s2 >> 12) & 0x000Fu) * scale.s2;
        dequantized_weights.s3 = IQ4_NL_DEQUANT((bits4.s3 >> 12) & 0x000Fu) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
    }

    int idx = (gy<<3)*m + (gx<<2);

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
