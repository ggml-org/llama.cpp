#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_mul_mm_q4_1_f32_8x4(
        global const ushort * src0_q,
        global const half  * src0_d,
        global const half  * src0_m,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int m,
        int n,
        int k,
        int n_no_padding
) {

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;
    __global const ushort* weight_ptr = src0_q + gx_2;
    __global const half*   scale_ptr  = src0_d + gx_2;
    __global const half*   min_ptr    = src0_m + gx_2;

    for(int i=0; i<k; i+=4){

        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);

        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));

        half4 scale = vload4(0, scale_ptr + (i/32)*(m));
        half4 minv  = vload4(0,   min_ptr + (i/32)*(m));

        // j=0
        dequantized_weights.s0 = (bits4.s0 & (0x000F)) * scale.s0 + minv.s0;
        dequantized_weights.s1 = (bits4.s1 & (0x000F)) * scale.s1 + minv.s1;
        dequantized_weights.s2 = (bits4.s2 & (0x000F)) * scale.s2 + minv.s2;
        dequantized_weights.s3 = (bits4.s3 & (0x000F)) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = ((bits4.s0 & (0x00F0)) >> 4) * scale.s0 + minv.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x00F0)) >> 4) * scale.s1 + minv.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x00F0)) >> 4) * scale.s2 + minv.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x00F0)) >> 4) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = ((bits4.s0 & (0x0F00)) >> 8) * scale.s0 + minv.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x0F00)) >> 8) * scale.s1 + minv.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x0F00)) >> 8) * scale.s2 + minv.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x0F00)) >> 8) * scale.s3 + minv.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = ((bits4.s0 & (0xF000)) >> 12) * scale.s0 + minv.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0xF000)) >> 12) * scale.s1 + minv.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0xF000)) >> 12) * scale.s2 + minv.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0xF000)) >> 12) * scale.s3 + minv.s3;
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
