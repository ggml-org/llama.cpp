#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

#define QK_K 256

// d, then 64 grid index bytes, then 8 uint32 of packed scale+signs
typedef struct {
    half  d;
    uchar qs[3*QK_K/8];
} block_iq3_xxs;

// iq3xxs_grid: 256 x uint32, 1 KB; small enough for Adreno's constant cache.
constant uint iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
};

// 7 bit index -> 8 sign bits
constant uchar ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// 8 threads cover one super block, one 32 element sub block each
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_iq3_xxs_f32(
        global char * src0,
        int offset0,
        global char * src1,
        int offset1,
        global char * dst,
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
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ix = get_sub_group_local_id()/8;  // super block index
    int it = get_sub_group_local_id()%8;  // sub block inside the super block

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    int offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_iq3_xxs * x = (global block_iq3_xxs *) (src0 + offset_src0);
    global float         * y = (global float         *) (src1 + offset_src1);

    float yl[32];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 32 * it;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        global char * xrow = (global char *)(x + ib);

        // only ne01 output rows exist; reading past them can pull in an inf scale
        for (int row = 0; row < N_DST && first_row + row < ne01; row++) {
            global block_iq3_xxs * xb = (global block_iq3_xxs *)(xrow + row*nb01);

            global uchar * qs = xb->qs + 8*it;

            // the block is 98 bytes, so qs+64 is only 2 byte aligned - read the
            // packed scale+signs word as two ushorts, not one uint
            global ushort * sw = (global ushort *)(xb->qs + 64 + 4*it);
            uint aux32 = (uint)sw[0] | ((uint)sw[1] << 16);

            float db = (float)xb->d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            float acc = 0.f;
            for (int l = 0; l < 4; ++l) {
                uchar signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                uint  g1 = iq3xxs_grid[qs[2*l+0]];
                uint  g2 = iq3xxs_grid[qs[2*l+1]];

                for (int j = 0; j < 4; ++j) {
                    float s1 = (signs & (1 << j))       ? -1.f : 1.f;
                    float s2 = (signs & (1 << (j+4)))   ? -1.f : 1.f;
                    acc += yl[8*l+j+0] * (float)((g1 >> (8*j)) & 0xFF) * s1;
                    acc += yl[8*l+j+4] * (float)((g2 >> (8*j)) & 0xFF) * s2;
                }
            }

            sumf[row] += db * acc;
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
