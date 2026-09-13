#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q6_K_f32(
        global const ushort * src0_ql,
        global const uchar  * src0_qh,
        global const ushort * src0_s,
        global const half   * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int m,
        int n,
        int k,
        int n_no_padding,
        ushort mask_f000,
        uchar  mask_c0
) {
    dst = (global float *)( (global char *)dst + offsetd );

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0); // n
    int gx = get_global_id(1); // m
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort * ptr_ql = src0_ql + gx_2;
    global const uchar  * ptr_qh = src0_qh + gx_2;
    global const ushort * ptr_s  = src0_s  + gx_2;
    global const half   * ptr_d  = src0_d  + gx_2;

    for (int i = 0; i < k; i += 4) {
        // load 4x elements (ushort) of ql on M, each ushort contains 4 weights
        // 4x ushort correspons to 4 rows on M
        ushort4 bits4 = vload4(0, ptr_ql + (i/4)*m); // ql packed in 4s in ushort
        uchar4  bits2 = vload4(0, ptr_qh + (i/4)*m); // qh packed in 4s in uchar

        // load 4 consecutive scales
        char8 scale_s_8 = as_char8(vload4(0, ptr_s + (i/16/2)*m)); // 1 char scale every 16 elements, packed in 2s
        char4   scale_s = ((i/16) % 2) == 0 ? scale_s_8.s0246 : scale_s_8.s1357; // transposed as ushort, 2 blocks
        half4   scale_d = vload4(0, ptr_d + (i/256)*m);  // 1 half scale every 256 elements

        // j=0
        // load 2x 4 elements of activations on N, corresponding to 8 rows on N
        B.s0123 = read_imageh(src1, gy*2 + (i + 0)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 0)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((bits4.s0 & 0x000F) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((bits4.s1 & 0x000F) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((bits4.s2 & 0x000F) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((bits4.s3 & 0x000F) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i + 1)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 1)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((((bits4.s0 & 0x00F0) >> 4) | ((bits2.s0 & 0x0C) << 2))) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((((bits4.s1 & 0x00F0) >> 4) | ((bits2.s1 & 0x0C) << 2))) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((((bits4.s2 & 0x00F0) >> 4) | ((bits2.s2 & 0x0C) << 2))) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((((bits4.s3 & 0x00F0) >> 4) | ((bits2.s3 & 0x0C) << 2))) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i + 2)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 2)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((((bits4.s0 & 0x0F00) >> 8) | (bits2.s0 & 0x30))) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((((bits4.s1 & 0x0F00) >> 8) | (bits2.s1 & 0x30))) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((((bits4.s2 & 0x0F00) >> 8) | (bits2.s2 & 0x30))) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((((bits4.s3 & 0x0F00) >> 8) | (bits2.s3 & 0x30))) - 32.f) * scale_s.s3 * scale_d.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i + 3)*n_4 + 0);
        B.s4567 = read_imageh(src1, gy*2 + (i + 3)*n_4 + 1);
        dequantized_weights.s0 = (convert_half((((bits4.s0 & mask_f000) >> 12) | ((bits2.s0 & mask_c0) >> 2))) - 32.f) * scale_s.s0 * scale_d.s0;
        dequantized_weights.s1 = (convert_half((((bits4.s1 & mask_f000) >> 12) | ((bits2.s1 & mask_c0) >> 2))) - 32.f) * scale_s.s1 * scale_d.s1;
        dequantized_weights.s2 = (convert_half((((bits4.s2 & mask_f000) >> 12) | ((bits2.s2 & mask_c0) >> 2))) - 32.f) * scale_s.s2 * scale_d.s2;
        dequantized_weights.s3 = (convert_half((((bits4.s3 & mask_f000) >> 12) | ((bits2.s3 & mask_c0) >> 2))) - 32.f) * scale_s.s3 * scale_d.s3;
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

// Cooperative-K q6_K GEMM for the small-batch (n_q in [2..8]) path. Same idea
// as the q4_K _cok kernel: WG = (COK_SG lanes x COK_NSG subgroups), each lane
// owns ONE output row (half8 over the 8 padded cols), and the COK_NSG
// subgroups split the K iterations round-robin and combine via a __local
// reduction. Replaces the default 4-row-per-WI tile that walked all of K alone
// (~M/512 WGs + serial reduction) at small n_q. REQD_SUBGROUP_SIZE_64 +
// barrier (never sub_group_reduce at full width on X2).
#define COK_NSG 8
#define COK_SG  64
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q6_K_f32_cok(
        global const ushort * src0_ql,
        global const uchar  * src0_qh,
        global const ushort * src0_s,
        global const half   * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int m,
        int n,
        int k,
        int n_no_padding,
        ushort mask_f000,
        uchar  mask_c0
) {
    dst = (global float *)( (global char *)dst + offsetd );

    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // output row
    int sg   = get_local_id(1);      // subgroup index (K-split)
    int lane = get_local_id(0);      // lane within subgroup

    global const ushort * ptr_ql = src0_ql + gx;
    global const uchar  * ptr_qh = src0_qh + gx;
    global const ushort * ptr_s  = src0_s  + gx;
    global const half   * ptr_d  = src0_d  + gx;

    half8 acc = 0;
    half8 B;
    half  dq;

    int num_iter = k >> 2;   // k/4 iterations, 4 k-values each

    for (int ib = sg; ib < num_iter; ib += COK_NSG) {
        int i = ib << 2;     // ib * 4

        ushort bits4 = ptr_ql[ib * m];          // ql for row gx at this 4-block
        uchar  bits2 = ptr_qh[ib * m];          // qh

        ushort s_packed = ptr_s[(i >> 5) * m];  // (i/16/2) = i/32
        char2  sc2      = as_char2(s_packed);
        char   scale_s  = (((i >> 4) & 1) == 0) ? sc2.s0 : sc2.s1; // (i/16)%2
        half   scale_d  = ptr_d[(i >> 8) * m];  // i/256

        // j=0
        B.s0123 = read_imageh(src1, (i + 0)*n_4 + 0);
        B.s4567 = read_imageh(src1, (i + 0)*n_4 + 1);
        dq = (convert_half((bits4 & 0x000F) | ((bits2 & 0x03) << 4)) - 32.f) * scale_s * scale_d;
        acc += B * dq;

        // j=1
        B.s0123 = read_imageh(src1, (i + 1)*n_4 + 0);
        B.s4567 = read_imageh(src1, (i + 1)*n_4 + 1);
        dq = (convert_half(((bits4 & 0x00F0) >> 4) | ((bits2 & 0x0C) << 2)) - 32.f) * scale_s * scale_d;
        acc += B * dq;

        // j=2
        B.s0123 = read_imageh(src1, (i + 2)*n_4 + 0);
        B.s4567 = read_imageh(src1, (i + 2)*n_4 + 1);
        dq = (convert_half(((bits4 & 0x0F00) >> 8) | (bits2 & 0x30)) - 32.f) * scale_s * scale_d;
        acc += B * dq;

        // j=3
        B.s0123 = read_imageh(src1, (i + 3)*n_4 + 0);
        B.s4567 = read_imageh(src1, (i + 3)*n_4 + 1);
        dq = (convert_half(((bits4 & mask_f000) >> 12) | ((bits2 & mask_c0) >> 2)) - 32.f) * scale_s * scale_d;
        acc += B * dq;
    }

    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    if (sg > 0) {
        reduceLM[(sg - 1) * COK_SG + lane] = convert_float8(acc);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg == 0) {
        float8 sum = convert_float8(acc);
        for (int s = 0; s < COK_NSG - 1; s++) {
            sum += reduceLM[s * COK_SG + lane];
        }
        int idx = gx;
        if (idx < m*n_no_padding) { dst[idx] = sum.s0; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s1; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s2; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s3; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s4; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s5; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s6; idx += m; }
        if (idx < m*n_no_padding) { dst[idx] = sum.s7; }
    }
}
