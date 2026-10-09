// Cooperative-K q4_0 GEMM with a dp4a inner product at eight columns, for verify batches (ne1 = 2..8)
// on bin (32b-transposed) weights. A lane folds COK_ROWS rows and always computes eight columns;
// columns past n_no_padding are dropped at the store. K is split across COK_NSG subgroups (local
// memory reduce) and ksplit workgroups (partials summed by kernel_gemv_splitk_reduce_f32).
// Nibbles unpack eight K at a time into even-K and odd-K bytes, -8 folded as (q + 0x78) ^ 0x80; the
// activation (kernel_quant_a_q8_1_eo) stores each 8-K group as [K0 K2 K4 K6][K1 K3 K5 K7] to match.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 8
#endif
#define COK_SG  64

#ifndef COK_ROWS
#define COK_ROWS 4
#endif

// q8_1 activation quantization in the byte order this file's GEMM reads. One work-item per
// 32-K block; column c, block b is global block c*k_b + b, and its 32 bytes go to
// qa[(c*K + b*32) ..] as four (even, odd) uint pairs. Scales land at da[b*8 + c] so a block's
// eight column scales are one half8; the host allocates da at width 8 whatever N is.
#define COK_PACK4(a, b, c, e)                                         \
    ( ((uint)(a) & 0xFFu) | (((uint)(b) & 0xFFu) << 8)               \
    | (((uint)(c) & 0xFFu) << 16) | (((uint)(e) & 0xFFu) << 24) )

kernel void kernel_quant_a_q8_1_eo(
    global const float * src,     // [N * K] f32, K contiguous per column
    global uint        * qa,      // [N * K / 4] int8 x4, even/odd permuted per 8 K
    global half        * da,      // [K/32 * 8] block scale, column-interleaved at width 8
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
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (int)rint(v[i] * id);
    }

    uint8 out;
    out.s0 = COK_PACK4(q[ 0], q[ 2], q[ 4], q[ 6]);
    out.s1 = COK_PACK4(q[ 1], q[ 3], q[ 5], q[ 7]);
    out.s2 = COK_PACK4(q[ 8], q[10], q[12], q[14]);
    out.s3 = COK_PACK4(q[ 9], q[11], q[13], q[15]);
    out.s4 = COK_PACK4(q[16], q[18], q[20], q[22]);
    out.s5 = COK_PACK4(q[17], q[19], q[21], q[23]);
    out.s6 = COK_PACK4(q[24], q[26], q[28], q[30]);
    out.s7 = COK_PACK4(q[25], q[27], q[29], q[31]);
    vstore8(out, 0, qa + blk * 8);

    const int col = blk / k_b;
    const int b   = blk - col * k_b;
    da[b * 8 + col] = (half)d;
}

// Eight K of one row as two dp4a words. x holds K-groups g (low ushort) and g+1 (high ushort).
// Even K are the low nibble of each byte, odd K the high; (q + 0x78) ^ 0x80 is (q - 8) as int8, no carry.
#define COK_EVEN(x) ((( (x)       & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
#define COK_ODD(x)  (((((x) >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)

// One row's sixteen K (K-groups bl0..bl3 of the half block) as four dp4a words.
#define COK_UNPACK(r)                                                        \
    const uint x0_##r = (uint)bl0.s##r | ((uint)bl1.s##r << 16);             \
    const uint x1_##r = (uint)bl2.s##r | ((uint)bl3.s##r << 16);             \
    const uint we0_##r = COK_EVEN(x0_##r);                                   \
    const uint wo0_##r = COK_ODD (x0_##r);                                   \
    const uint we1_##r = COK_EVEN(x1_##r);                                   \
    const uint wo1_##r = COK_ODD (x1_##r);

// Row r against column c over the same sixteen K, the column's texel in A.
#define COK_DOT_RC(r, c)                                                     \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(we0_##r, A.s0, d##r.s##c);      \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(wo0_##r, A.s1, d##r.s##c);      \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(we1_##r, A.s2, d##r.s##c);      \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(wo1_##r, A.s3, d##r.s##c);

// Column-outer: one activation texel at a time against the unpacked rows. Both wave-
// uniform operands (activation texel, block scales) come through the texture path, which
// serves a wave-uniform address once rather than once per lane.
#define COK_DOT_COL(c, t)                                                    \
    { const uint4 A = read_imageui(src1_qa, (c) * k_t + (t));                \
      COK_ROWS_DO_C(c) }

// Bin plane: one uint per (8 K, row), low half = the four-K group of the first four K.
// g0 is a multiple of 4, so K-group g0+t is word g0/2 + t/2, half t&1.
#if COK_ROWS == 4
#define COK_WLOAD(t) ushort4 bl##t = convert_ushort4(((t) & 1) ? (vload4(0, src0_q + row0 + ((g0 >> 1) + ((t) >> 1)) * m) >> 16) \
                                                            : (vload4(0, src0_q + row0 + ((g0 >> 1) + ((t) >> 1)) * m) & 0xFFFFu));
#define COK_ROWS_DO(F) F(0) F(1) F(2) F(3)
#define COK_ROWS_DO_C(c) COK_DOT_RC(0, c) COK_DOT_RC(1, c) COK_DOT_RC(2, c) COK_DOT_RC(3, c)
#else
#define COK_WLOAD(t) ushort2 bl##t = convert_ushort2(((t) & 1) ? (vload2(0, src0_q + row0 + ((g0 >> 1) + ((t) >> 1)) * m) >> 16) \
                                                            : (vload2(0, src0_q + row0 + ((g0 >> 1) + ((t) >> 1)) * m) & 0xFFFFu));
#define COK_ROWS_DO(F) F(0) F(1)
#define COK_ROWS_DO_C(c) COK_DOT_RC(0, c) COK_DOT_RC(1, c)
#endif

kernel void kernel_gemm_cok8_q4_0_q8_1_dp4a(
    global const uint   * src0_q,     // q4_0 nibble plane, bin [row + (K/8)*m]
    global const half   * src0_d,     // one scale per 32-K block [row + blk*m]
    read_only image1d_buffer_t src1_qa,  // q8_1 activations, eo order, RGBA32UI texels [col*K/16 + K/16]
    read_only image1d_buffer_t src1_da,  // activation scales, one half8 texel per block [blk*8 + col]
    global float * dst,               // ksplit == 1: [n][m] output; else [ksplit][n][m] partials
    ulong offsetd,
    int m,
    int k,
    int n_no_padding,
    int ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);
    const int sg   = get_local_id(1);
    const int lane = get_local_id(0);
    const int ks   = get_group_id(1);

    const int row0 = gx * COK_ROWS;
    const int num_32blk = k / 32;
    const int k_t = k >> 4;                       // activation texels (16 K) per column
    const int fence = n_no_padding >> 4;          // 0 (n <= 8), unknown to the compiler

    // This workgroup's K slice, in 32-K blocks; the COK_NSG subgroups interleave inside it.
    const int chunk   = (num_32blk + ksplit - 1) / ksplit;
    const int blk_beg = ks * chunk;
    const int blk_end = min(blk_beg + chunk, num_32blk);

    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
#if COK_ROWS == 4
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);
#endif

    for (int blk = blk_beg + sg; blk < blk_end; blk += COK_NSG) {
        int8 d0 = (int8)(0), d1 = (int8)(0);
#if COK_ROWS == 4
        int8 d2 = (int8)(0), d3 = (int8)(0);
#endif
        for (int h = 0; h < 2; ++h) {
            const int g0 = (blk << 3) + (h << 2);     // first K-group of this half block
            const int at = (blk << 1) + h;            // its texel in the activation

            COK_WLOAD(0) COK_WLOAD(1) COK_WLOAD(2) COK_WLOAD(3)
            COK_ROWS_DO(COK_UNPACK)

            // Two column groups of four: the second group's texel index carries a
            // dependency on the first group's dot (fence is 0, so the value is at), which
            // keeps the compiler from issuing all eight texel reads up front.
            COK_DOT_COL(0, at) COK_DOT_COL(1, at) COK_DOT_COL(2, at) COK_DOT_COL(3, at)
            const int at4 = at + (d0.s0 & fence);
            COK_DOT_COL(4, at4) COK_DOT_COL(5, at4) COK_DOT_COL(6, at4) COK_DOT_COL(7, at4)
        }

        // One flush per 32-K block: single scale per block, no min term.
        const float8 da = convert_float8(as_half8(read_imageui(src1_da, blk)));
#if COK_ROWS == 4
        const half4 scale = vload4(0, src0_d + row0 + blk * m);
#else
        const half2 scale = vload2(0, src0_d + row0 + blk * m);
#endif
        acc0 = mad(convert_float8(d0), da * (float)scale.s0, acc0);
        acc1 = mad(convert_float8(d1), da * (float)scale.s1, acc1);
#if COK_ROWS == 4
        acc2 = mad(convert_float8(d2), da * (float)scale.s2, acc2);
        acc3 = mad(convert_float8(d3), da * (float)scale.s3, acc3);
#endif
    }

    // Cross-subgroup reduction over the in-workgroup K-split, one row at a time.
    local float8 reduceLM[COK_SG * (COK_NSG - 1)];
    float8 out0 = (float8)(0.0f), out1 = (float8)(0.0f);
#if COK_ROWS == 4
    float8 out2 = (float8)(0.0f), out3 = (float8)(0.0f);
#endif

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
#if COK_ROWS == 4
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#endif
#undef COK_REDUCE

    // Columns beyond n_no_padding were computed on whatever the activation buffer held
    // past the quantized region and are dropped here. Slice ks writes its own [n][m] plane.
#if COK_ROWS == 4
#define COK_STORE_COL(c)                                                                  \
    if ((c) < n_no_padding) {                                                             \
        vstore4((float4)(out0.s##c, out1.s##c, out2.s##c, out3.s##c), 0,                  \
                dst + (c) * m + row0);                                                    \
    }
#else
#define COK_STORE_COL(c)                                                                  \
    if ((c) < n_no_padding) {                                                             \
        vstore2((float2)(out0.s##c, out1.s##c), 0, dst + (c) * m + row0);                 \
    }
#endif

    if (sg == 0) {
        dst += (size_t)ks * (size_t)m * (size_t)n_no_padding;
        COK_STORE_COL(0) COK_STORE_COL(1) COK_STORE_COL(2) COK_STORE_COL(3)
        COK_STORE_COL(4) COK_STORE_COL(5) COK_STORE_COL(6) COK_STORE_COL(7)
    }
#undef COK_STORE_COL
}
