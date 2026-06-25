#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// v = { mp, L, d }
inline uint fastdiv(uint n, uint4 v) {
    uint msbs;
    msbs = mul_hi(n, v.s0);
    return (msbs + n) >> v.s1;
}
inline uint fastmod(uint n, uint4 v) {
    uint q = fastdiv(n, v);
    return n - q * v.s2;
}

kernel void kernel_set_rows_f32_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_set_rows_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

// f32 -> q8_0 quantize set_rows. Block = half d + char qs[32].
#define QK8_0 32

inline void quantize_q8_0_block(global float * x, global char * qs, global half * d_out) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; j++) {
        amax = fmax(amax, fabs(x[j]));
    }

    float d  = amax / 127.0f;
    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

    vstore_half(d, 0, d_out);

    for (int j = 0; j < QK8_0; j++) {
        qs[j] = (char)((int)round(x[j] * id));
    }
}

kernel void kernel_set_rows_q8_0_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global char  * dst_row = (global char  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x = src_row + blk * QK8_0;
        global char  * y = dst_row + blk * (2 + QK8_0);

        quantize_q8_0_block(x, y + 2, (global half *)y);
    }
}

kernel void kernel_set_rows_q8_0_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global char  * dst_row = (global char  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x = src_row + blk * QK8_0;
        global char  * y = dst_row + blk * (2 + QK8_0);

        quantize_q8_0_block(x, y + 2, (global half *)y);
    }
}

// SoA q8_0 variants. dst_q: int8[QK8_0] per block; dst_d: fp16 scale per block.
// Layout matches kernel_convert_block_q8_0; block index follows dst element order.
kernel void kernel_set_rows_q8_0_soa_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst_q,
        ulong         offset_q,
        global char * dst_d,
        ulong         offset_d,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        int           ne1_dst,
        int           ne2_dst,
        int           ne3_dst
) {
    src0  = src0  + offset0;
    src1  = src1  + offset1;
    dst_q = dst_q + offset_q;
    dst_d = dst_d + offset_d;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    long row_blk_base = ((long)i03 * ne2_dst * ne1_dst + (long)i02 * ne1_dst + i1) * nblk0;

    global half  * d_row = (global half  *)(dst_d) + row_blk_base;
    global char  * q_row = (global char  *)(dst_q) + row_blk_base * QK8_0;
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x = src_row + blk * QK8_0;
        global char  * q = q_row + blk * QK8_0;

        quantize_q8_0_block(x, q, d_row + blk);
    }
}

kernel void kernel_set_rows_q8_0_soa_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst_q,
        ulong         offset_q,
        global char * dst_d,
        ulong         offset_d,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        int           ne1_dst,
        int           ne2_dst,
        int           ne3_dst
) {
    src0  = src0  + offset0;
    src1  = src1  + offset1;
    dst_q = dst_q + offset_q;
    dst_d = dst_d + offset_d;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    long row_blk_base = ((long)i03 * ne2_dst * ne1_dst + (long)i02 * ne1_dst + i1) * nblk0;

    global half  * d_row = (global half  *)(dst_d) + row_blk_base;
    global char  * q_row = (global char  *)(dst_q) + row_blk_base * QK8_0;
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x = src_row + blk * QK8_0;
        global char  * q = q_row + blk * QK8_0;

        quantize_q8_0_block(x, q, d_row + blk);
    }
}

kernel void kernel_set_rows_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

// f32 -> q4_0 quantize set_rows. Block = half d + uchar qs[16] (shuffled
// nibbles: qs[j] low/high = elem j / j+16).
// Dequant: val[i] = d * (nibble_i - 8)
// nblk0 = number of q4_0 blocks per row = ne00 / 32.
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

inline void quantize_q4_0_block(global float * x, global uchar * qs, global half * d_out) {
    // Find the signed value with the largest absolute magnitude (matches ggml ref).
    float max  = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < QK4_0; j++) {
        float v = x[j];
        float a = fabs(v);
        if (a > amax) {
            amax = a;
            max  = v;
        }
    }

    float d  = max / -8.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    vstore_half(d, 0, d_out);

    for (int j = 0; j < QK4_0/2; j++) {
        float x0 = x[j]           * id;
        float x1 = x[j + QK4_0/2] * id;

        int i0 = (int)(x0 + 8.5f);
        int i1 = (int)(x1 + 8.5f);
        if (i0 < 0)  i0 = 0;
        if (i0 > 15) i0 = 15;
        if (i1 < 0)  i1 = 0;
        if (i1 > 15) i1 = 15;

        qs[j] = (uchar)i0 | ((uchar)i1 << 4);
    }
}

kernel void kernel_set_rows_q4_0_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global char  * dst_row = (global char  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x    = src_row + blk * QK4_0;
        global char  * y    = dst_row + blk * Q4_0_BLOCK_SIZE;
        global half  * yd   = (global half  *)(y);
        global uchar * yqs  = (global uchar *)(y + 2);

        quantize_q4_0_block(x, yqs, yd);
    }
}

kernel void kernel_set_rows_q4_0_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global char  * dst_row = (global char  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x    = src_row + blk * QK4_0;
        global char  * y    = dst_row + blk * Q4_0_BLOCK_SIZE;
        global half  * yd   = (global half  *)(y);
        global uchar * yqs  = (global uchar *)(y + 2);

        quantize_q4_0_block(x, yqs, yd);
    }
}

// SoA variants for q4_0 dst. Used when the backend has split block_q4_0 records
// into separate quant (dst_q) and scale (dst_d) sub-buffers — same pattern as
// the q8_0 SoA variants above.
//
// Layout (matches kernel_convert_block_q4_0, the "shuffled" variant):
//   dst_q: contiguous 16 packed nibbles per block, block i at offset i * 16 bytes.
//   dst_d: contiguous fp16 scales, block i at offset i * 2 bytes.
// Nibble layout inside each byte is unchanged from AoS: qs[j] low nibble = element j,
// qs[j] high nibble = element j+16. kernel_restore_block_q4_0 copies bytes as-is.
kernel void kernel_set_rows_q4_0_soa_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst_q,
        ulong         offset_q,
        global char * dst_d,
        ulong         offset_d,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        int           ne1_dst,
        int           ne2_dst,
        int           ne3_dst
) {
    src0  = src0  + offset0;
    src1  = src1  + offset1;
    dst_q = dst_q + offset_q;
    dst_d = dst_d + offset_d;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    long row_blk_base = ((long)i03 * ne2_dst * ne1_dst + (long)i02 * ne1_dst + i1) * nblk0;

    global half  * d_row   = (global half  *)(dst_d) + row_blk_base;
    global uchar * q_row   = (global uchar *)(dst_q) + row_blk_base * (QK4_0/2);
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x    = src_row + blk * QK4_0;
        global uchar * qs   = q_row   + blk * (QK4_0/2);
        global half  * d_bk = d_row   + blk;

        quantize_q4_0_block(x, qs, d_bk);
    }
}

kernel void kernel_set_rows_q4_0_soa_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst_q,
        ulong         offset_q,
        global char * dst_d,
        ulong         offset_d,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        int           ne1_dst,
        int           ne2_dst,
        int           ne3_dst
) {
    src0  = src0  + offset0;
    src1  = src1  + offset1;
    dst_q = dst_q + offset_q;
    dst_d = dst_d + offset_d;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    long row_blk_base = ((long)i03 * ne2_dst * ne1_dst + (long)i02 * ne1_dst + i1) * nblk0;

    global half  * d_row   = (global half  *)(dst_d) + row_blk_base;
    global uchar * q_row   = (global uchar *)(dst_q) + row_blk_base * (QK4_0/2);
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int blk = get_local_id(0); blk < nblk0; blk += get_local_size(0)) {
        global float * x    = src_row + blk * QK4_0;
        global uchar * qs   = q_row   + blk * (QK4_0/2);
        global half  * d_bk = d_row   + blk;

        quantize_q4_0_block(x, qs, d_bk);
    }
}
