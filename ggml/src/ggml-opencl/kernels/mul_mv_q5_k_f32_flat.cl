#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// block_q5_K
//------------------------------------------------------------------------------
#define QK_K            256
#define BLOCK_Q5K_SIZE  176
#define K_SCALE_SIZE    12

typedef struct {
    half  d;                    // super-block scale for quantized scales
    half  dmin;                 // super-block scale for quantized mins
    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uchar qh[QK_K/8];           // quants, high bit (1 bit per value, packed 8 per byte)
    uchar qs[QK_K/2];           // quants, low 4 bits (2 values per byte)
} block_q5_K;

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST       4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 16
#elif defined(ADRENO_GPU)
#define N_DST       16
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// number of (super) blocks each subgroup processes
// each thread in a subgroup processes a block (32 weights)
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q5_K_f32_flat(
    global uchar * src0_q,
    global uchar * src0_qh,
    global uchar * src0_s,
    global half  * src0_d,
    global half  * src0_dm,
    global char  * src1,
    int offset1,
    global char  * dst,
    int offsetd,
    int ne00,
    int ne01,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne12,
    ulong nb11,
    ulong nb12,
    ulong nb13,
    int ne0,
    int ne1,
    int r2,
    int r3
) {
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    ushort kmask1 = 0x3f3f;
    ushort kmask2 = 0x0f0f;
    ushort kmask3 = 0xc0c0;

    int ix = get_sub_group_local_id()/8;
    int it = get_sub_group_local_id()%8;
    int iq = it/4;
    int ir = it%4;

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = (first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03)/BLOCK_Q5K_SIZE;
    uint blk = nb01 / BLOCK_Q5K_SIZE;
    global uchar * blk_q  = (global uchar *)src0_q  + offset_src0*(QK_K/2);
    global uchar * blk_qh = (global uchar *)src0_qh + offset_src0*(QK_K/8);
    global uchar * blk_s  = (global uchar *)src0_s  + offset_src0*K_SCALE_SIZE;
    global half  * blk_d  = (global half  *)src0_d  + offset_src0;
    global half  * blk_dm = (global half  *)src0_dm + offset_src0;

    int offset_src1 = r1*nb11 + (i12)*nb12 + (i13)*nb13;
    global float * y = (global float *)(src1 + offset_src1);

    float yl[16];
    float yh[16];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uchar u1_lo = (uchar)(1 << (2*iq));
    uchar u2_lo = (uchar)(2 << (2*iq));
    uchar u1_hi = (uchar)(1 << (2*iq + 4));
    uchar u2_hi = (uchar)(2 << (2*iq + 4));

    ushort  sc16[4];
    uchar * sc8 = (uchar *)sc16;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+0];
            sumy.s0 += yl[i+0];

            yl[i+8] = y4[i+32];
            sumy.s1 += yl[i+8];

            yh[i+0] = y4[i+128];
            sumy.s2 += yh[i+0];

            yh[i+8] = y4[i+160];
            sumy.s3 += yh[i+8];
        }

        global ushort * q1 = (global ushort *)(blk_q  + ib * (QK_K/2)) + (16 * iq + 4 * ir);
        global uchar  * qh = (global uchar  *)(blk_qh + ib * (QK_K/8)) + 8 * ir;
        global ushort * sc = (global ushort *)(blk_s  + ib * K_SCALE_SIZE) + iq;
        global half   * d  = blk_d  + ib;
        global half   * dm = blk_dm + ib;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            global ushort * q2 = q1 + 32;

            // Load the 4 q1 / 4 q2 quant ushorts as 2 uints each. 16-bit integer ops are
            // disproportionately slow on the A7X (E031.41) compiler; keeping the dequant
            // operands in 32-bit registers avoids the ushort path (same fix as q4_K flat).
            // q1/q2 are 4-byte aligned; w & 0x0F00 on the low/high half of a uint equals the
            // original ushort mask value, so the math is unchanged. The qh high-bit term is
            // byte-indexed (qh[0..7]) and left as-is.
            global uint * q1u = (global uint *)q1;
            global uint * q2u = (global uint *)q2;
            uint a0 = q1u[0], a1 = q1u[1], b0 = q2u[0], b1 = q2u[1];
            uint w0 = a0 & 0xFFFF, w1 = a0 >> 16, w2 = a1 & 0xFFFF, w3 = a1 >> 16;
            uint v0 = b0 & 0xFFFF, v1 = b0 >> 16, v2 = b1 & 0xFFFF, v3 = b1 >> 16;

            float4 acc1, acc2;
            acc1.s0 =
                yl[0]*((w0&0x000F)+(qh[0]&u1_lo?16.f:0.f)) +
                yl[2]*((w1&0x000F)+(qh[2]&u1_lo?16.f:0.f)) +
                yl[4]*((w2&0x000F)+(qh[4]&u1_lo?16.f:0.f)) +
                yl[6]*((w3&0x000F)+(qh[6]&u1_lo?16.f:0.f));
            acc1.s1 =
                yl[1]*((w0&0x0F00)+(qh[1]&u1_lo?16.f*256.f:0.f)) +
                yl[3]*((w1&0x0F00)+(qh[3]&u1_lo?16.f*256.f:0.f)) +
                yl[5]*((w2&0x0F00)+(qh[5]&u1_lo?16.f*256.f:0.f)) +
                yl[7]*((w3&0x0F00)+(qh[7]&u1_lo?16.f*256.f:0.f));
            acc1.s2 =
                yl[ 8]*((w0&0x00F0)+(qh[0]&u2_lo?16.f*16.f:0.f)) +
                yl[10]*((w1&0x00F0)+(qh[2]&u2_lo?16.f*16.f:0.f)) +
                yl[12]*((w2&0x00F0)+(qh[4]&u2_lo?16.f*16.f:0.f)) +
                yl[14]*((w3&0x00F0)+(qh[6]&u2_lo?16.f*16.f:0.f));
            acc1.s3 =
                yl[ 9]*((w0&0xF000)+(qh[1]&u2_lo?16.f*4096.f:0.f)) +
                yl[11]*((w1&0xF000)+(qh[3]&u2_lo?16.f*4096.f:0.f)) +
                yl[13]*((w2&0xF000)+(qh[5]&u2_lo?16.f*4096.f:0.f)) +
                yl[15]*((w3&0xF000)+(qh[7]&u2_lo?16.f*4096.f:0.f));
            acc2.s0 =
                yh[0]*((v0&0x000F)+(qh[0]&u1_hi?16.f:0.f)) +
                yh[2]*((v1&0x000F)+(qh[2]&u1_hi?16.f:0.f)) +
                yh[4]*((v2&0x000F)+(qh[4]&u1_hi?16.f:0.f)) +
                yh[6]*((v3&0x000F)+(qh[6]&u1_hi?16.f:0.f));
            acc2.s1 =
                yh[1]*((v0&0x0F00)+(qh[1]&u1_hi?16.f*256.f:0.f)) +
                yh[3]*((v1&0x0F00)+(qh[3]&u1_hi?16.f*256.f:0.f)) +
                yh[5]*((v2&0x0F00)+(qh[5]&u1_hi?16.f*256.f:0.f)) +
                yh[7]*((v3&0x0F00)+(qh[7]&u1_hi?16.f*256.f:0.f));
            acc2.s2 =
                yh[ 8]*((v0&0x00F0)+(qh[0]&u2_hi?16.f*16.f:0.f)) +
                yh[10]*((v1&0x00F0)+(qh[2]&u2_hi?16.f*16.f:0.f)) +
                yh[12]*((v2&0x00F0)+(qh[4]&u2_hi?16.f*16.f:0.f)) +
                yh[14]*((v3&0x00F0)+(qh[6]&u2_hi?16.f*16.f:0.f));
            acc2.s3 =
                yh[ 9]*((v0&0xF000)+(qh[1]&u2_hi?16.f*4096.f:0.f)) +
                yh[11]*((v1&0xF000)+(qh[3]&u2_hi?16.f*4096.f:0.f)) +
                yh[13]*((v2&0xF000)+(qh[5]&u2_hi?16.f*4096.f:0.f)) +
                yh[15]*((v3&0xF000)+(qh[7]&u2_hi?16.f*4096.f:0.f));

            float dall = *d;
            float dmin = *dm;
            sumf[row] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                                 (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                                 (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                                 (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);

            q1 += blk*64;
            qh += blk*32;
            sc += blk*6;
            d  += blk;
            dm += blk;
        }

        y4 += BLOCK_STRIDE * QK_K;
    }

    global float * dst_f32 = (global float *) dst + im*ne0*ne1 + r1*ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = sub_group_reduce_add(sumf[row]);
        if (first_row + row < ne01) {
            if (get_sub_group_local_id() == 0) {
                dst_f32[first_row + row] = all_sum;
            }
        }
    }
}
