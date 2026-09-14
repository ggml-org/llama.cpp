#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif
#define QK_K         256
#define K_SCALE_SIZE 12

// scales are transposed: consecutive codes of a row are `stride` apart
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    int stride,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j*stride]     & mask_d6;
        *m = q[(j+4)*stride] & mask_d6;
    } else {
        *d = (q[(j+4)*stride] & mask_d4) | ((q[(j-4)*stride] & mask_hi2) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & mask_d4) | ((q[j*stride] & mask_hi2) >> 2);
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32(
    global const ushort * src0_q,
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


    global const ushort * weight_ptr = src0_q + gx_2;
    global const half   * d_ptr      = src0_d  + gx_2;
    global const half   * dm_ptr     = src0_dm + gx_2;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half4 d  = vload4(0, d_ptr  + sb_idx * m);
        half4 dm = vload4(0, dm_ptr + sb_idx * m);

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + (gx_2+0);
        global const uchar * sc1 = sc0 + 1;
        global const uchar * sc2 = sc0 + 2;
        global const uchar * sc3 = sc0 + 3;

        uchar sv0, mn0, sv1, mn1, sv2, mn2, sv3, mn3;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);

        half4 scale = convert_half4(convert_float4(d)  * convert_float4((uchar4)(sv0, sv1, sv2, sv3)));
        half4 mval  = convert_half4(convert_float4(dm) * convert_float4((uchar4)(mn0, mn1, mn2, mn3)));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort4 bits4 = vload4(0, weight_ptr + (ki/4) * m);

            // j=0
            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dequantized_weights.s0 = (bits4.s0 & 0x000F) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (bits4.s1 & 0x000F) * scale.s1 - mval.s1;
            dequantized_weights.s2 = (bits4.s2 & 0x000F) * scale.s2 - mval.s2;
            dequantized_weights.s3 = (bits4.s3 & 0x000F) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0x00F0) >> 4) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0x00F0) >> 4) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0x00F0) >> 4) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0x00F0) >> 4) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0x0F00) >> 8) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0x0F00) >> 8) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0x0F00) >> 8) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0x0F00) >> 8) * scale.s3 - mval.s3;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
            c2 += B * dequantized_weights.s2;
            c3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dequantized_weights.s0 = ((bits4.s0 & 0xF000) >> 12) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits4.s1 & 0xF000) >> 12) * scale.s1 - mval.s1;
            dequantized_weights.s2 = ((bits4.s2 & 0xF000) >> 12) * scale.s2 - mval.s2;
            dequantized_weights.s3 = ((bits4.s3 & 0xF000) >> 12) * scale.s3 - mval.s3;
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

// 1x8 per-WI tile (1 output row x 8 output cols). For the small-batch
// (medium n_q, e.g. MTP/spec verify) path where the 2x8 kernel is starved:
// at ne1<=8 the grid is (1, ceil(M/2)) -> only ~M/256 workgroups, leaving
// the SP under-occupied. 1 row per WI doubles the M-axis workgroup count
// (ceil(M/1)/128 vs ceil(M/2)/128) AND collapses the accumulators to a
// single half8 (16 regs, no spill), so more waves co-reside. Same weight
// traffic as 2x8 (rows never share weights); the win is pure occupancy.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_r1(
    global const ushort * src0_q,
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
    int gx = get_global_id(1);       // 1 row per WI

    half8 c0 = 0;
    half8 B;
    half dq;

    int num_blocks_K = k / QK_K;

    global const ushort * weight_ptr = src0_q + gx;
    global const half   * d_ptr      = src0_d  + gx;
    global const half   * dm_ptr     = src0_dm + gx;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half dd  = d_ptr [sb_idx * m];
        half dmm = dm_ptr[sb_idx * m];

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + gx;

        uchar sv0, mn0;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);

        half scale = convert_half(convert_float(dd)  * (float)sv0);
        half mval  = convert_half(convert_float(dmm) * (float)mn0);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = weight_ptr[(ki/4) * m];

            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dq = (bits & 0x000F) * scale - mval;
            c0 += B * dq;

            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dq = ((bits & 0x00F0) >> 4) * scale - mval;
            c0 += B * dq;

            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dq = ((bits & 0x0F00) >> 8) * scale - mval;
            c0 += B * dq;

            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dq = ((bits & 0xF000) >> 12) * scale - mval;
            c0 += B * dq;
        }
    }

    // Output: 8 cols, 1 row per col-step. Scalar store, coalesced across
    // neighbouring WIs (consecutive gx -> consecutive dst addresses).
    int idx = (gy<<3)*m + gx;
    if (idx < m*n_no_padding) { dst[idx] = c0.s0; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s1; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s2; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s3; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s4; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s5; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s6; idx += m; }
    if (idx < m*n_no_padding) { dst[idx] = c0.s7; }
}

// 2x8 tile, but weights read through an image1d_buffer (CL_R/UINT32 over the
// same packed-q buffer) instead of a plain global buffer. The ne1==1 GEMV
// already does this and is much faster per weight byte than this GEMM at
// small n_q; the structural difference is the image path hits the dedicated
// TPL1 weight cache (L1) while the global path only reaches L2. At small n_q
// the forward is weight-read-bound, so L1-cached weights is the lever.
// The 2 adjacent rows the 2x8 tile reads as a ushort2 are exactly one uint32,
// so the vload2 becomes a single read_imageui at index gx + (ki/4)*(m/2).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_kimg(
    read_only image1d_buffer_t src0_q_img,
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
    int m_2 = m >> 1;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 1;

    half8 c0 = 0, c1 = 0;
    half8 B;
    half2 dequantized_weights;

    int num_blocks_K = k / QK_K;

    global const half * d_ptr  = src0_d  + gx_2;
    global const half * dm_ptr = src0_dm + gx_2;

    for (int i = 0; i < k; i += 32) {
        int sb_idx  = i / QK_K;
        int sub_idx = (i / 32) % 8;

        half2 d  = vload2(0, d_ptr  + sb_idx * m);
        half2 dm = vload2(0, dm_ptr + sb_idx * m);

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + (gx_2+0);
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(sub_idx, sc1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        half2 scale = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        half2 mval  = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            uint wpacked = read_imageui(src0_q_img, gx + (ki/4) * m_2).x;
            ushort2 bits2 = (ushort2)((ushort)(wpacked & 0xFFFFu), (ushort)(wpacked >> 16));

            // j=0
            B.s0123 = read_imageh(src1, gy*2   + (ki+0) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+0) * n_4);
            dequantized_weights.s0 = (bits2.s0 & 0x000F) * scale.s0 - mval.s0;
            dequantized_weights.s1 = (bits2.s1 & 0x000F) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;

            // j=1
            B.s0123 = read_imageh(src1, gy*2   + (ki+1) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+1) * n_4);
            dequantized_weights.s0 = ((bits2.s0 & 0x00F0) >> 4) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits2.s1 & 0x00F0) >> 4) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;

            // j=2
            B.s0123 = read_imageh(src1, gy*2   + (ki+2) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+2) * n_4);
            dequantized_weights.s0 = ((bits2.s0 & 0x0F00) >> 8) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits2.s1 & 0x0F00) >> 8) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;

            // j=3
            B.s0123 = read_imageh(src1, gy*2   + (ki+3) * n_4);
            B.s4567 = read_imageh(src1, gy*2+1 + (ki+3) * n_4);
            dequantized_weights.s0 = ((bits2.s0 & 0xF000) >> 12) * scale.s0 - mval.s0;
            dequantized_weights.s1 = ((bits2.s1 & 0xF000) >> 12) * scale.s1 - mval.s1;
            c0 += B * dequantized_weights.s0;
            c1 += B * dequantized_weights.s1;
        }
    }

    int idx = (gy<<3)*m + (gx<<1);
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s0, c1.s0), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s1, c1.s1), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s2, c1.s2), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s3, c1.s3), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s4, c1.s4), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s5, c1.s5), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s6, c1.s6), 0, dst + idx); idx += m; }
    if (idx+1 < m*n_no_padding) { vstore2((float2)(c0.s7, c1.s7), 0, dst + idx); }
}

// Cooperative-K GEMM for the small-batch (n_q in [2..8]) path. Mirrors the
// ne1==1 GEMV's structure: a WG is (COK_SG lanes x COK_NSG subgroups); each
// lane owns ONE output row and computes its 8 (padded) columns, and the
// COK_NSG subgroups SPLIT the K reduction round-robin, combining via a
// __local reduction. This is the thing the per-WI GEMM lacked — at small n_q
// the old kernel had ~M/256 workgroups each walking all of K serially; this
// has M/64 workgroups AND COK_NSG-way K parallelism. Uses REQD_SUBGROUP_SIZE_64
// + barrier (same safe reduction pattern as the GEMV; never sub_group_reduce
// at full width on X2 per the GDN miscompile note).
#define COK_NSG 8
#define COK_SG  64
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q4_k_f32_cok(
    global const ushort * src0_q,
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
    int n_4  = n >> 2;
    int gx   = get_global_id(0);     // output row
    int sg   = get_local_id(1);      // subgroup index (K-split lane)
    int lane = get_local_id(0);      // lane within subgroup (0..COK_SG-1)

    int num_blocks_K = k / QK_K;
    int num_32blk    = k / 32;

    global const ushort * weight_ptr = src0_q + gx;
    global const half   * d_ptr      = src0_d  + gx;
    global const half   * dm_ptr     = src0_dm + gx;

    half8 acc = 0;
    half8 B;
    half  dq;

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        int i       = blk << 5;       // blk * 32
        int sb_idx  = blk >> 3;       // (blk*32) / QK_K  (QK_K = 256 = 32*8)
        int sub_idx = blk & 7;        // (i/32) % 8

        half dd  = d_ptr [sb_idx * m];
        half dmm = dm_ptr[sb_idx * m];

        global const uchar * sc0 = src0_s + sb_idx * K_SCALE_SIZE * m + gx;
        uchar sv0, mn0;
        get_scale_min_k4(sub_idx, sc0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        half scale = convert_half(convert_float(dd)  * (float)sv0);
        half mval  = convert_half(convert_float(dmm) * (float)mn0);

        for (int l = 0; l < 32; l += 4) {
            int ki = i + l;
            ushort bits = weight_ptr[(ki>>2) * m];

            B.s0123 = read_imageh(src1,     (ki+0) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+0) * n_4);
            dq = (bits & 0x000F) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+1) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+1) * n_4);
            dq = ((bits & 0x00F0) >> 4) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+2) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+2) * n_4);
            dq = ((bits & 0x0F00) >> 8) * scale - mval;
            acc += B * dq;

            B.s0123 = read_imageh(src1,     (ki+3) * n_4);
            B.s4567 = read_imageh(src1, 1 + (ki+3) * n_4);
            dq = ((bits & 0xF000) >> 12) * scale - mval;
            acc += B * dq;
        }
    }

    // cross-subgroup reduction over the K-split (float for accuracy)
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
