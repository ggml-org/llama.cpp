// Cooperative-K q4_K GEMM with a dp4a inner product at eight columns, for verify batches (ne1 = 5..8)
// on bin (32b-transposed) weights. A lane folds four rows and always computes eight columns; columns
// past n_no_padding are dropped. K is split across COK_NSG subgroups (local memory reduce) and ksplit
// workgroups (kernel_gemv_splitk_reduce_f32), each slice a contiguous run of 32-K blocks.
// Weight codes stay unsigned in the dp4a word; the min is applied at the flush:
// acc += sc*d * da*dot - mn*dmin * sa, with sa = d_act * Sum(q_act).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 6
#endif
#define COK_SG   64
#define COK_ROWS 4

#define COK_PACK4(a, b, c, e)                                         \
    ( ((uint)(a) & 0xFFu) | (((uint)(b) & 0xFFu) << 8)               \
    | (((uint)(c) & 0xFFu) << 16) | (((uint)(e) & 0xFFu) << 24) )

// q8_1 activation quantization in the byte order this file's GEMM reads. One work-item
// per 32-K block; column c, block b is global block c*k_b + b. Every 4-K group is stored
// as the bytes (K0, K2, K1, K3), matching the nibble spread of the weight word below. ds
// holds two half8 texels per block: [2b] the eight column scales d, [2b+1] the eight
// column sums d*Sum(q); the host allocates at width 8.
kernel void kernel_quant_a_q8_1_k4(
    global const float * src,     // [N * K] f32, K contiguous per column
    global uint        * qa,      // [N * K / 4] int8 x4, (0 2 1 3) order per 4 K
    global half        * ds,      // [K/32 * 16] block scale and sum, width 8
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
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (int)rint(v[i] * id);
        sum += q[i];
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
    ds[(2 * b)     * 8 + col] = (half)d;
    ds[(2 * b + 1) * 8 + col] = (half)(d * (float)sum);
}

// Weight layout (bin): one uint per (8 K, row), nibble i = K i. The scale bytes keep the noshuffle q4_K
// layout, byte b of (row, superblock) at [sb][b][row], so the (scale, min) of sub-block j for the four
// rows are uchar4 loads at wave-uniform offsets.

// Nibble group t (four K) spread to bytes (K0, K2, K1, K3).
#define COK_SPREAD(t) (((t) | ((t) << 12)) & 0x0F0F0F0Fu)

#define COK_DOT_RC4(r, c)                                                          \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(W0.s##r, A.s0, d##r.s##c);            \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(W1.s##r, A.s1, d##r.s##c);            \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(W2.s##r, A.s2, d##r.s##c);            \
    d##r.s##c = dot_acc_sat_4x8packed_ss_int(W3.s##r, A.s3, d##r.s##c);

#define COK_DOT_COL4(c, t)                                                         \
    { const uint4 A = read_imageui(src1_qa, (c) * k_t + (t));                      \
      COK_DOT_RC4(0, c) COK_DOT_RC4(1, c) COK_DOT_RC4(2, c) COK_DOT_RC4(3, c) }

kernel void kernel_gemm_cok8_q4_k_q8_1_dp4a_bin(
    global const uint   * src0_q,     // q4_K bin plane [row + (K/8)*m]
    global const uchar  * src0_s,     // packed scales/mins [sb][12][row]
    global const half   * src0_d,     // superblock scale [row + sb*m]
    global const half   * src0_dm,    // superblock min   [row + sb*m]
    read_only image1d_buffer_t src1_qa,  // q8_1 activations, (0 2 1 3) order, RGBA32UI [col*K/16 + K/16]
    read_only image1d_buffer_t src1_ds,  // activation scale / sum, half8 texels [2*blk], [2*blk+1]
    global float * dst,               // ksplit == 1: [n][m] output; else [ksplit][n][m] partials
    ulong offsetd,
    int m,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2,
    int ksplit
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);
    const int sg   = get_local_id(1);
    const int lane = get_local_id(0);
    const int ks   = get_group_id(1);

    const int row0 = gx * COK_ROWS;
    const int num_32blk = k / 32;
    const int k_t = k >> 4;
    const int fence = (int)(mask_d6 >> 6);        // 0 at run time, unknown to the compiler

    const int nslice = ksplit * COK_NSG;
    const int chunk  = (num_32blk + nslice - 1) / nslice;
    const int b_beg  = (ks * COK_NSG + sg) * chunk;
    const int b_end  = min(b_beg + chunk, num_32blk);

    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);

    for (int sb = b_beg >> 3; sb * 8 < b_end; ++sb) {
        const half4 dd  = vload4(0, src0_d  + row0 + sb * m);
        const half4 dmm = vload4(0, src0_dm + row0 + sb * m);
        global const uchar * sc = src0_s + sb * 12 * m + row0;

        const int j_beg = max(b_beg - sb * 8, 0);
        const int j_end = min(b_end - sb * 8, 8);

        for (int j = j_beg; j < j_end; ++j) {
            const int blk = sb * 8 + j;
            int8 d0 = (int8)(0), d1 = (int8)(0), d2 = (int8)(0), d3 = (int8)(0);

            for (int h = 0; h < 2; ++h) {
                const int u0 = (blk << 2) + (h << 1);     // first bin word of this half block
                const int at = (blk << 1) + h;
                const uint4 wa = vload4(0, src0_q + row0 + (u0    ) * m);
                const uint4 wb = vload4(0, src0_q + row0 + (u0 + 1) * m);
                const uint4 W0 = COK_SPREAD(wa & 0xFFFFu);
                const uint4 W1 = COK_SPREAD(wa >> 16);
                const uint4 W2 = COK_SPREAD(wb & 0xFFFFu);
                const uint4 W3 = COK_SPREAD(wb >> 16);
                // Two column groups of four, the second gated on the first's dot so the
                // compiler does not hoist all eight texel reads.
                COK_DOT_COL4(0, at) COK_DOT_COL4(1, at) COK_DOT_COL4(2, at) COK_DOT_COL4(3, at)
                const int at4 = at + (d0.s0 & fence);
                COK_DOT_COL4(4, at4) COK_DOT_COL4(5, at4) COK_DOT_COL4(6, at4) COK_DOT_COL4(7, at4)
            }

            // get_scale_min_k4 for the four rows at once; j is wave-uniform.
            uchar4 scv, mnv;
            if (j < 4) {
                scv = vload4(0, sc + (j    ) * m) & (uchar4)(mask_d6);
                mnv = vload4(0, sc + (j + 4) * m) & (uchar4)(mask_d6);
            } else {
                const uchar4 bh = vload4(0, sc + (j + 4) * m);
                const uchar4 bl = vload4(0, sc + (j - 4) * m);
                const uchar4 bj = vload4(0, sc + (j    ) * m);
                scv = (bh & (uchar4)(mask_d4)) | ((bl >> (uchar4)(6)) << (uchar4)(4));
                mnv = (bh >> (uchar4)(4))      | ((bj >> (uchar4)(6)) << (uchar4)(4));
            }
            const float4 scf = convert_float4(dd)  * convert_float4(scv);
            const float4 mnf = convert_float4(dmm) * convert_float4(mnv);
            {
                const float8 da = convert_float8(as_half8(read_imageui(src1_ds, 2 * blk)));
                acc0 = mad(convert_float8(d0), da * scf.s0, acc0);
                acc1 = mad(convert_float8(d1), da * scf.s1, acc1);
                acc2 = mad(convert_float8(d2), da * scf.s2, acc2);
                acc3 = mad(convert_float8(d3), da * scf.s3, acc3);
            }
            {
                const float8 sa = convert_float8(as_half8(read_imageui(src1_ds, 2 * blk + 1)));
                acc0 = mad(sa, -mnf.s0, acc0);
                acc1 = mad(sa, -mnf.s1, acc1);
                acc2 = mad(sa, -mnf.s2, acc2);
                acc3 = mad(sa, -mnf.s3, acc3);
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

    if (sg == 0) {
        dst += (size_t)ks * (size_t)m * (size_t)n_no_padding;
        COK_STORE_COL(0) COK_STORE_COL(1) COK_STORE_COL(2) COK_STORE_COL(3)
        COK_STORE_COL(4) COK_STORE_COL(5) COK_STORE_COL(6) COK_STORE_COL(7)
    }
#undef COK_STORE_COL
}
