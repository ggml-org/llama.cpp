#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK5_0 32
#define N_R0_Q5_0 4

kernel void kernel_mul_mv_q5_0_f32_flat(
    global uchar * src0_q,
    global uchar * src0_qh,
    global half  * src0_d,
    global char  * src1,
    ulong         offset1,
    global char  * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src1 = (global char *) ((global char *) src1 + offset1);
    dst  = (global char *) ((global char *) dst  + offsetd);

    if (get_local_id(0) != 0 || get_local_id(1) != 0) {
        return;
    }

    const int nb = ne00 / QK5_0;

    const int r0 = get_group_id(0);
    const int r1 = get_group_id(1);
    const int im = get_group_id(2);

    const int first_row = r0 * N_R0_Q5_0;

    const uint i12 = im % ne12;
    const uint i13 = im / ne12;

    const ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
    global float * y = (global float *) (src1 + offset_src1);
    global float * dst_f32 = (global float *) dst + (ulong) im*ne0*ne1 + (ulong) r1*ne0;

    for (int row = 0; row < N_R0_Q5_0; ++row) {
        if (first_row + row >= ne01) {
            continue;
        }

        const ulong row_offset = (ulong) (first_row + row)*nb01 + (ulong) (i12/r2)*nb02 + (ulong) (i13/r3)*nb03;
        const ulong block_index = row_offset / 22;

        global uchar * ax  = src0_q  + block_index*(QK5_0/2);
        global uchar * axh = src0_qh + block_index*4;
        global half  * ad  = src0_d  + block_index;

        float total = 0.0f;
        for (int ib = 0; ib < nb; ++ib) {
            const uchar4 qh4 = vload4(0, axh + ib*4);
            const uint qh = (uint) qh4.s0 | ((uint) qh4.s1 << 8) | ((uint) qh4.s2 << 16) | ((uint) qh4.s3 << 24);
            global uchar * qs = ax + ib*(QK5_0/2);

            float sum = 0.0f;
            for (int j = 0; j < QK5_0/2; ++j) {
                const int x0 = ((qs[j] & 0x0Fu) | (((qh >> j)      & 0x1u) << 4)) - 16;
                const int x1 = ((qs[j] >> 4)    | (((qh >> (j+16)) & 0x1u) << 4)) - 16;
                sum += (float) x0 * y[ib*QK5_0 + j];
                sum += (float) x1 * y[ib*QK5_0 + QK5_0/2 + j];
            }

            total += sum * vload_half(0, ad + ib);
        }

        dst_f32[first_row + row] = total;
    }
}
