// Cooperative-K q6_K GEMM with dp4a at eight columns, for verify batches (ne1 = 2..8) on bin weights.
// A lane folds four rows and computes eight columns (extras dropped at the store); K is split across
// COK_NSG subgroups and ksplit workgroups. q stays unsigned 0..63 in the dp4a word and the -32 is
// applied at the flush (dot(q, a) - 32 sum(a)); ss is per 16 K, so the flush runs twice per 32-K block.
// The high two bits of four K reach bits 4..5 of each byte via two multiplies (0x110, 0x440000) whose
// overlapping products land only in masked-off bits. ne01 must be a multiple of 4.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 4
#endif
#define COK_SG   64
#define COK_ROWS 4

#define COK_PACK4(a, b, c, e)                                         \
    ( ((uint)(a) & 0xFFu) | (((uint)(b) & 0xFFu) << 8)               \
    | (((uint)(c) & 0xFFu) << 16) | (((uint)(e) & 0xFFu) << 24) )

// q8_1 activation quantization in this GEMM's byte order. One work-item per 32-K block;
// column c, block b is global block c*k_b + b. Each 4-K group is stored as (K0, K2, K1, K3).
// ds holds three half8 texels per block: [3b] the column scales d, [3b+1] d*Sum(q) over
// K 0..15, [3b+2] d*Sum(q) over K 16..31. The host allocates at width 8.
kernel void kernel_quant_a_q8_1_k4h(
    global const float * src,     // [N * K] f32, K contiguous per column
    global uint        * qa,      // [N * K / 4] int8 x4, (0 2 1 3) order per 4 K
    global half        * ds,      // [K/32 * 24] block scale and half sums, width 8
    int total_blocks,             // N * (K/32)
    int k_b                       // K/32
) {
    const int blk = get_global_id(0);
    if (blk >= total_blocks) {
        return;
    }

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

    const int col = blk / k_b;
    const int b   = blk - col * k_b;
    ds[(3 * b)     * 8 + col] = (half)d;
    ds[(3 * b + 1) * 8 + col] = (half)(d * (float)sum_lo);
    ds[(3 * b + 2) * 8 + col] = (half)(d * (float)sum_hi);
}

// Four K of one row as one dp4a word, bytes (K0, K2, K1, K3), codes 0..63. u: nibble i = K i;
// h: bits 2i..2i+1 = K i. Pack u and u >> 4, not u | u << 12: the Adreno 840 compiler evaluates
// that ushort shift at 16 bits and zeroes the odd K.
#define COK_LO(u) (((uint)(u) | ((uint)(ushort)((u) >> 4) << 16)) & 0x0F0F0F0Fu)
#define COK_HI(h) ((((((uint)(h)) & 0x33u) * 0x110u) & 0x3030u) | (((((uint)(h)) & 0xCCu) * 0x440000u) & 0x30300000u))

// Row r: K-groups 0..3 of the half block (bl0..bl3 nibbles, bh0..bh3 high bits).
#define COK_UNPACK(r)                                                              \
    const uint w0_##r = COK_LO(bl0.s##r) | COK_HI(bh0.s##r);                       \
    const uint w1_##r = COK_LO(bl1.s##r) | COK_HI(bh1.s##r);                       \
    const uint w2_##r = COK_LO(bl2.s##r) | COK_HI(bh2.s##r);                       \
    const uint w3_##r = COK_LO(bl3.s##r) | COK_HI(bh3.s##r);

// Row r against column c over the same sixteen K, the column's texel in A.
#define COK_DOT_RC(r, c)                                                           \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w0_##r, A.s0, d##r.s##c);             \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w1_##r, A.s1, d##r.s##c);             \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w2_##r, A.s2, d##r.s##c);             \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(w3_##r, A.s3, d##r.s##c);

// Column-outer: one activation texel at a time against the unpacked rows. The wave-
// uniform operands (activation texel, block scale and sums) go through the texture path.
#define COK_DOT_COL(c, t)                                                          \
    { const uint4 A = read_imageui(src1_qa, (c) * k_t + (t));                      \
      COK_DOT_RC(0, c) COK_DOT_RC(1, c) COK_DOT_RC(2, c) COK_DOT_RC(3, c) }

// Weight layout (bin, 32b-transposed):
//  - ql: one uint per (8 K, row), low half the nibbles of the first four K, high half the next four.
//  - qh: one uint per (16 K, row), byte i the high bits of the i-th four K.
//  - s: row-major, one ushort (two int8 scales) per (row, 32-K block), rows k/32 ushorts apart.
//  - d: one half per (row, 256 K), transposed: [row + sb*m].
kernel void kernel_gemm_cok8_q6_k_q8_1_dp4a_bin(
    global const uint   * src0_ql,    // [(K/8)*m + row]
    global const uint   * src0_qh,    // [(K/16)*m + row]
    global const ushort * src0_s,     // [row*(K/32) + blk]
    global const half   * src0_d,     // [row + sb*m]
    read_only image1d_buffer_t src1_qa,  // q8_1 activations, (0 2 1 3) order, RGBA32UI [col*K/16 + K/16]
    read_only image1d_buffer_t src1_ds,  // activation scale / half sums, half8 texels [3*blk .. 3*blk+2]
    global float * dst,               // ksplit == 1: [n][m] output; else [ksplit][n][m] partials
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

    // m % 4 == 0; the lanes past the last row group recompute it and skip the store.
    const bool live = gx * COK_ROWS < m;
    const int  row0 = live ? gx * COK_ROWS : m - COK_ROWS;
    const int num_32blk = k / 32;
    const int k_t = k >> 4;                       // activation texels (16 K) per column
    const int fence = (int)(mask_c0 >> 6) - 3;    // 0 at run time, unknown to the compiler

    // This subgroup's K slice: every (ksplit*COK_NSG)-th group of eight 32-K blocks
    // (K % 256 == 0). One 16-byte load per row fetches a group's eight scales.
    const int nslice = ksplit * COK_NSG;
    const int b_beg  = ks * COK_NSG + sg;

    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);

    for (int bg = b_beg; bg * 8 < num_32blk; bg += nslice) {
        ushort8 sr0 = vload8(0, src0_s + (row0 + 0) * num_32blk + bg * 8);
        ushort8 sr1 = vload8(0, src0_s + (row0 + 1) * num_32blk + bg * 8);
        ushort8 sr2 = vload8(0, src0_s + (row0 + 2) * num_32blk + bg * 8);
        ushort8 sr3 = vload8(0, src0_s + (row0 + 3) * num_32blk + bg * 8);
        for (int jb = 0; jb < 8; ++jb) {
            const int blk = bg * 8 + jb;
            // Both int8 scales of this block for the four rows: low half the first 16 K,
            // high half the second.
            const ushort4 spk = (ushort4)(sr0.s0, sr1.s0, sr2.s0, sr3.s0);
            sr0 = sr0.s12345670; sr1 = sr1.s12345670; sr2 = sr2.s12345670; sr3 = sr3.s12345670;

            const half4   scd = vload4(0, src0_d + row0 + (blk >> 3) * m);
            const float8  da  = convert_float8(as_half8(read_imageui(src1_ds, 3 * blk)));

            for (int h = 0; h < 2; ++h) {
                const int g0 = (blk << 3) + (h << 2);     // first K-group of this half block
                const int at = (blk << 1) + h;            // its texel in the activation

                const uint4 l01 = vload4(0, src0_ql + row0 + ((g0 >> 1)    ) * m);
                const uint4 l23 = vload4(0, src0_ql + row0 + ((g0 >> 1) + 1) * m);
                const uint4 hq  = vload4(0, src0_qh + row0 + (g0 >> 2) * m);
                // Converted per component: a vector conversion of the shifted uint4 compiles poorly.
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
                // Two column groups of four, the second gated on the first's dot so the
                // compiler does not hoist all eight texel reads.
                COK_DOT_COL(0, at) COK_DOT_COL(1, at) COK_DOT_COL(2, at) COK_DOT_COL(3, at)
                const int at4 = at + (d0.s0 & fence);
                COK_DOT_COL(4, at4) COK_DOT_COL(5, at4) COK_DOT_COL(6, at4) COK_DOT_COL(7, at4)

                // Flush: acc += ss*d * (da*dot - 32*sa); the -32 goes through the half's activation sum.
                const float8 sa = convert_float8(as_half8(read_imageui(src1_ds, 3 * blk + 1 + h)));
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

    // Cross-subgroup reduction over the in-workgroup K-split, one row at a time.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out0 = (float8)(0.0f), out1 = (float8)(0.0f);
    float8 out2 = (float8)(0.0f), out3 = (float8)(0.0f);

#define COK_REDUCE(accv, outv)                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = (accv); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg == 0) {                                                   \
        float8 sum = (accv);                                         \
        for (int s = 0; s < COK_NSG - 1; s++) {                      \
            sum += reduceLM[s * COK_SG + lane];                      \
        }                                                            \
        (outv) = sum;                                                \
    }

    COK_REDUCE(acc0, out0)
    COK_REDUCE(acc1, out1)
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#undef COK_REDUCE

    // Columns beyond n_no_padding were computed on whatever the activation buffer held
    // past the quantized region and are dropped here. Slice ks writes its own [n][m] plane.
#define COK_STORE_COL(c)                                                                  \
    if ((c) < n_no_padding) {                                                             \
        vstore4((float4)(out0.s##c, out1.s##c, out2.s##c, out3.s##c), 0,                  \
                dst + (c) * m + row0);                                                    \
    }

    if (sg == 0 && live) {
        dst += (size_t)ks * (size_t)m * (size_t)n_no_padding;
        COK_STORE_COL(0) COK_STORE_COL(1) COK_STORE_COL(2) COK_STORE_COL(3)
        COK_STORE_COL(4) COK_STORE_COL(5) COK_STORE_COL(6) COK_STORE_COL(7)
    }
#undef COK_STORE_COL
}
