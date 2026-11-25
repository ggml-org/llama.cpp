#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COOPK_NSG
#define COOPK_NSG 4
#endif
#define COK_SG   64
#define COK_ROWS 4

#define COK_PACK4(a, b, c, e) \
    ( ((uint)(a) & 0xFFu) | (((uint)(b) & 0xFFu) << 8) \
    | (((uint)(c) & 0xFFu) << 16) | (((uint)(e) & 0xFFu) << 24) )

// Pack (0, 2, 1, 3) K bytes and store scales plus both 16-value sums at width eight.
kernel void kernel_quant_a_q8_1_k4h(
    global const float * src,
    global uint        * qa,
    global half        * ds,
    ulong offset_src,
    int total_blocks,
    int k_b
) {
    const int blk = get_global_id(0);
    if (blk >= 8 * k_b) {
        return;
    }
    const int col = blk / k_b;
    const int b = blk % k_b;
    if (blk >= total_blocks) {
        vstore8((uint8)(0), 0, qa + blk * 8);
        ds[(3 * b) * 8 + col] = (half)0;
        ds[(3 * b + 1) * 8 + col] = (half)0;
        ds[(3 * b + 2) * 8 + col] = (half)0;
        return;
    }
    src = (global const float *)((global const char *)src + offset_src);
    const int base = blk * 32;

    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = src[base + i];
        amax = fmax(amax, fabs(v[i]));
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int q[32];
    int sum_lo = 0, sum_hi = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        q[i] = (int)rint(v[i] * id);
        sum_lo += q[i];
    }
    #pragma unroll
    for (int i = 16; i < 32; ++i) {
        q[i] = (int)rint(v[i] * id);
        sum_hi += q[i];
    }

    uint8 out;
    out.s0 = COK_PACK4(q[ 0], q[ 2], q[ 1], q[ 3]);
    out.s1 = COK_PACK4(q[ 4], q[ 6], q[ 5], q[ 7]);
    out.s2 = COK_PACK4(q[ 8], q[10], q[ 9], q[11]);
    out.s3 = COK_PACK4(q[12], q[14], q[13], q[15]);
    out.s4 = COK_PACK4(q[16], q[18], q[17], q[19]);
    out.s5 = COK_PACK4(q[20], q[22], q[21], q[23]);
    out.s6 = COK_PACK4(q[24], q[26], q[25], q[27]);
    out.s7 = COK_PACK4(q[28], q[30], q[29], q[31]);
    vstore8(out, 0, qa + blk * 8);

    ds[(3 * b)     * 8 + col] = (half)d;
    ds[(3 * b + 1) * 8 + col] = (half)(d * (float)sum_lo);
    ds[(3 * b + 2) * 8 + col] = (half)(d * (float)sum_hi);
}

// Cast before shifting to keep all four K values on Adreno compilers.
#define COK_LO(u) (((uint)(u) | ((uint)(ushort)((u) >> 4) << 16)) & 0x0F0F0F0Fu)
#define COK_HI(h) ((((((uint)(h)) & 0x33u) * 0x110u) & 0x3030u) | (((((uint)(h)) & 0xCCu) * 0x440000u) & 0x30300000u))

#define COK_UNPACK(r) \
    const uint w0_##r = COK_LO(bl0.s##r) | COK_HI(bh0.s##r); \
    const uint w1_##r = COK_LO(bl1.s##r) | COK_HI(bh1.s##r); \
    const uint w2_##r = COK_LO(bl2.s##r) | COK_HI(bh2.s##r); \
    const uint w3_##r = COK_LO(bl3.s##r) | COK_HI(bh3.s##r);

#define COK_DOT_RC(r, c) \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w0_##r, A.s0, d##r.s##c); \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w1_##r, A.s1, d##r.s##c); \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w2_##r, A.s2, d##r.s##c); \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w3_##r, A.s3, d##r.s##c);

#define COK_DOT_COL(c, t) \
    { const uint4 A = read_imageui(src1_qa, (c) * k_t + (t)); \
      COK_DOT_RC(0, c) COK_DOT_RC(1, c) COK_DOT_RC(2, c) COK_DOT_RC(3, c) }

kernel void kernel_gemm_coopk_q6_k_q8_1_dp4a_bin_c8(
    global const uint   * src0_ql,
    global const uint   * src0_qh,
    global const ushort * src0_s,
    global const half   * src0_d,
    read_only image1d_buffer_t src1_qa,
    read_only image1d_buffer_t src1_ds,
    global float * dst,
    ulong offsetd,
    int m,
    int k,
    int n_no_padding,
    uchar  mask_c0,
    int ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);
    const int sg   = get_local_id(1);
    const int lane = get_local_id(0);
    const int ks   = get_group_id(1);

    const bool live = gx * COK_ROWS < m;
    const int  row0 = live ? gx * COK_ROWS : m - COK_ROWS;
    const int num_32blk = k / 32;
    const int k_t = k >> 4;
    const int fence = (int)(mask_c0 >> 6) - 3;

    const int nslice = ksplit * COOPK_NSG;
    const int b_beg  = ks * COOPK_NSG + sg;

    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);

    for (int bg = b_beg; bg * 8 < num_32blk; bg += nslice) {
        ushort8 sr0 = vload8(0, src0_s + (row0 + 0) * num_32blk + bg * 8);
        ushort8 sr1 = vload8(0, src0_s + (row0 + 1) * num_32blk + bg * 8);
        ushort8 sr2 = vload8(0, src0_s + (row0 + 2) * num_32blk + bg * 8);
        ushort8 sr3 = vload8(0, src0_s + (row0 + 3) * num_32blk + bg * 8);
        for (int jb = 0; jb < 8; ++jb) {
            const int blk = bg * 8 + jb;
            const ushort4 spk = (ushort4)(sr0.s0, sr1.s0, sr2.s0, sr3.s0);
            sr0 = sr0.s12345670; sr1 = sr1.s12345670; sr2 = sr2.s12345670; sr3 = sr3.s12345670;
            const half4   scd = vload4(0, src0_d + row0 + (blk >> 3) * m);
            const float8  da  = convert_float8(as_half8(read_imageui(src1_ds, 3 * blk)));

            for (int h = 0; h < 2; ++h) {
                const int g0 = (blk << 3) + (h << 2);
                const int at = (blk << 1) + h;

                const uint4 l01 = vload4(0, src0_ql + row0 + ((g0 >> 1)    ) * m);
                const uint4 l23 = vload4(0, src0_ql + row0 + ((g0 >> 1) + 1) * m);
                const uint4 hq  = vload4(0, src0_qh + row0 + (g0 >> 2) * m);

                const ushort4 bl0 = (ushort4)((ushort)l01.s0, (ushort)l01.s1, (ushort)l01.s2, (ushort)l01.s3);
                const ushort4 bl1 = (ushort4)((ushort)(l01.s0 >> 16), (ushort)(l01.s1 >> 16), (ushort)(l01.s2 >> 16), (ushort)(l01.s3 >> 16));
                const ushort4 bl2 = (ushort4)((ushort)l23.s0, (ushort)l23.s1, (ushort)l23.s2, (ushort)l23.s3);
                const ushort4 bl3 = (ushort4)((ushort)(l23.s0 >> 16), (ushort)(l23.s1 >> 16), (ushort)(l23.s2 >> 16), (ushort)(l23.s3 >> 16));
                const uchar4  bh0 = (uchar4)((uchar)hq.s0, (uchar)hq.s1, (uchar)hq.s2, (uchar)hq.s3);
                const uchar4  bh1 = (uchar4)((uchar)(hq.s0 >> 8), (uchar)(hq.s1 >> 8), (uchar)(hq.s2 >> 8), (uchar)(hq.s3 >> 8));
                const uchar4  bh2 = (uchar4)((uchar)(hq.s0 >> 16), (uchar)(hq.s1 >> 16), (uchar)(hq.s2 >> 16), (uchar)(hq.s3 >> 16));
                const uchar4  bh3 = (uchar4)((uchar)(hq.s0 >> 24), (uchar)(hq.s1 >> 24), (uchar)(hq.s2 >> 24), (uchar)(hq.s3 >> 24));

                COK_UNPACK(0) COK_UNPACK(1) COK_UNPACK(2) COK_UNPACK(3)

                int8 d0 = (int8)(0), d1 = (int8)(0), d2 = (int8)(0), d3 = (int8)(0);
                COK_DOT_COL(0, at) COK_DOT_COL(1, at) COK_DOT_COL(2, at) COK_DOT_COL(3, at)
                const int at4 = at + (d0.s0 & fence);
                COK_DOT_COL(4, at4) COK_DOT_COL(5, at4) COK_DOT_COL(6, at4) COK_DOT_COL(7, at4)

                const float8 sa = convert_float8(as_half8(read_imageui(src1_ds, 3 * blk + 1 + h)));
                // Apply the signed-code offset with the sum for this 16-value half.
                const float8 sm = sa * -32.0f;
                const float s0f = (float)(h ? as_char2(spk.s0).s1 : as_char2(spk.s0).s0) * (float)scd.s0;
                const float s1f = (float)(h ? as_char2(spk.s1).s1 : as_char2(spk.s1).s0) * (float)scd.s1;
                const float s2f = (float)(h ? as_char2(spk.s2).s1 : as_char2(spk.s2).s0) * (float)scd.s2;
                const float s3f = (float)(h ? as_char2(spk.s3).s1 : as_char2(spk.s3).s0) * (float)scd.s3;
                acc0 = mad(mad(da, convert_float8(d0), sm), s0f, acc0);
                acc1 = mad(mad(da, convert_float8(d1), sm), s1f, acc1);
                acc2 = mad(mad(da, convert_float8(d2), sm), s2f, acc2);
                acc3 = mad(mad(da, convert_float8(d3), sm), s3f, acc3);
            }
        }
    }

    local float8 reduceLM[COK_SG * (COOPK_NSG - 1)];
    float8 out0 = (float8)(0.0f), out1 = (float8)(0.0f);
    float8 out2 = (float8)(0.0f), out3 = (float8)(0.0f);
#define COK_REDUCE(accv, outv) \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = (accv); } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (sg == 0) { \
        float8 sum = (accv); \
        for (int s = 0; s < COOPK_NSG - 1; s++) { sum += reduceLM[s * COK_SG + lane]; } \
        (outv) = sum; \
    }
    COK_REDUCE(acc0, out0)
    COK_REDUCE(acc1, out1)
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#undef COK_REDUCE

#define COK_STORE_COL(c) \
    if ((c) < n_no_padding) { \
        vstore4((float4)(out0.s##c, out1.s##c, out2.s##c, out3.s##c), 0, dst + (c) * m + row0); \
    }
    if (sg == 0 && live) {
        dst += (size_t)ks * (size_t)m * (size_t)n_no_padding;
        COK_STORE_COL(0) COK_STORE_COL(1) COK_STORE_COL(2) COK_STORE_COL(3)
        COK_STORE_COL(4) COK_STORE_COL(5) COK_STORE_COL(6) COK_STORE_COL(7)
    }
#undef COK_STORE_COL
}
