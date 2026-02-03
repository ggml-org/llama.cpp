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

kernel void kernel_mul_mm_q8_0_f32_8x4(
        global const uint * src0_q,
        global const half  * src0_d,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int k,
        int m,
        int n,
        int n_no_padding,
        ulong offsetd
) {

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy   = get_global_id(0);
    int gx   = get_global_id(1);
    int gx_2 = gx << 2;
    dst  = (global float *)((global char*)dst  + offsetd);


    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 deq;

    __global const uint* wptr = src0_q + gx_2;
    __global const half* sptr = src0_d + gx_2;

      for (int i = 0; i < k; i += 4) {
        uint4 pack4 = vload4(0, wptr + (i / 4) * m);
        half4 scale = vload4(0, sptr + (i / 32) * m);

        char4 p0 = as_char4(pack4.s0);
        char4 p1 = as_char4(pack4.s1);
        char4 p2 = as_char4(pack4.s2);
        char4 p3 = as_char4(pack4.s3);

        // ------------------- j = 0 (k = i+0) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 0) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 0) * n_4 + 1);

        half4 wj0 = convert_half4((char4)(p0.s0, p1.s0, p2.s0, p3.s0)) * scale;

        c0 += B * wj0.s0;
        c1 += B * wj0.s1;
        c2 += B * wj0.s2;
        c3 += B * wj0.s3;

        // ------------------- j = 1 (k = i+1) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 1) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 1) * n_4 + 1);

        half4 wj1 = convert_half4((char4)(p0.s1, p1.s1, p2.s1, p3.s1)) * scale;

        c0 += B * wj1.s0;
        c1 += B * wj1.s1;
        c2 += B * wj1.s2;
        c3 += B * wj1.s3;

        // ------------------- j = 2 (k = i+2) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 2) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 2) * n_4 + 1);

        half4 wj2 = convert_half4((char4)(p0.s2, p1.s2, p2.s2, p3.s2)) * scale;

        c0 += B * wj2.s0;
        c1 += B * wj2.s1;
        c2 += B * wj2.s2;
        c3 += B * wj2.s3;

        // ------------------- j = 3 (k = i+3) -------------------
        B.s0123 = read_imageh(src1, gy * 2 + (i + 3) * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + (i + 3) * n_4 + 1);

        half4 wj3 = convert_half4((char4)(p0.s3, p1.s3, p2.s3, p3.s3)) * scale;

        c0 += B * wj3.s0;
        c1 += B * wj3.s1;
        c2 += B * wj3.s2;
        c3 += B * wj3.s3;
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
