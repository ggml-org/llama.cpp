#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// IQ3_S decode GEMV over five feature-major planes, [k-group][row] (size preserving, see
// kernel_convert_block_iq3_s_ns): qs (k/4), qh (k/32), sg (k/8), sc (k/64), d (k/256).
//   grid index of operand u = qs[u] | (((qh >> u) & 1) << 8)
//   sign of value j         = bit (4*(u&1) + j) of sg[u/2]
//   sub-scale of block sb   = nibble (sb&1) of sc[sb/2], then d * (1 + 2*nib)
// K is split across IQ3S_MV_NSG subgroups; IQ3S_MV_R2 pairs rows and reads every plane as ushort.

#define QK_K 256

// Subgroups per workgroup; each takes every NSG'th super-block along K.
#ifndef IQ3S_MV_NSG
#define IQ3S_MV_NSG 8
#endif

#define IQ3S_GRID_AS constant

IQ3S_GRID_AS uint iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

#if IQ3S_MV_AIMG
#define IQ3S_YV(g, y) read_imagef(y_img, (int)(y_tex + (g)))
#else
#define IQ3S_YV(g, y) vload4((g), (y))
#endif

// Four grid values with their signs applied. base picks the nibble of sgv.
inline float4 iq3s_vals(uint gv, uint sgv, uint base) {
    const uint s = sgv >> base;
    float4 v;
    v.s0 = (float)((gv      ) & 0xFFu); if (s & 1u) { v.s0 = -v.s0; }
    v.s1 = (float)((gv >>  8) & 0xFFu); if (s & 2u) { v.s1 = -v.s1; }
    v.s2 = (float)((gv >> 16) & 0xFFu); if (s & 4u) { v.s2 = -v.s2; }
    v.s3 = (float)((gv >> 24) & 0xFFu); if (s & 8u) { v.s3 = -v.s3; }
    return v;
}

// IQ3S_MV_AIMG=1: read the activation through an image1d_buffer view of src1 (no copy) instead of
// vload4. The read is wave-uniform and redundant across the workgroup's 64 rows, which the texture cache
// serves well; it pays off for these codebook types, whose per-weight decode is expensive, and not for
// linear quants such as IQ4_XS. Default on for X2-class devices; GGML_OPENCL_IQ3S_MV_AIMG overrides.
#ifndef IQ3S_MV_AIMG
#define IQ3S_MV_AIMG 0
#endif

kernel void kernel_iq3s_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 512u) {
        out[i] = iq3s_grid[i];
    }
}

// One weight quad: the fetched grid word `gw` with its signs applied. The grid
// index `g` is unused.
#define IQ3S_W(g, gw, sgv, base) iq3s_vals(gw, sgv, base)
#define IQ3S_GW(g, expr)         (expr)

kernel void kernel_mul_mv_iq3_s_f32_flat(
        __read_only image1d_buffer_t grid_img,
#if IQ3S_MV_AIMG
        // Declared only when the option is on, so host and kernel argument lists cannot disagree
        // (a mismatch would silently shift the weight planes).
        __read_only image1d_buffer_t y_img,
        uint y_off,
#endif
        global const uchar * src0_qs,
        global const uchar * src0_qh,
        global const uchar * src0_sg,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sgi = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ3S_MV_AIMG
    // src1 is f32 and the image is RGBA/f32, so one texel is four floats.
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#define IQ3S_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;                        // rows per plane row, as ushorts
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            const uint scbase = j + ib * 4u * mh;
            ushort sc4[4];
            sc4[0] = scu[scbase + 0u * mh];
            sc4[1] = scu[scbase + 1u * mh];
            sc4[2] = scu[scbase + 2u * mh];
            sc4[3] = scu[scbase + 3u * mh];

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint scv  = (uint)sc4[sb >> 1];
                const uint nib0 = (sb & 1u) ? ((scv >>  4) & 0xFu) : ( scv        & 0xFu);
                const uint nib1 = (sb & 1u) ? ((scv >> 12) & 0xFu) : ((scv >>  8) & 0xFu);

                const uint qhv = (uint)qhu[j + (ib * 8u + sb) * mh];
                const uint qh0 = qhv & 0xFFu;
                const uint qh1 = qhv >> 8;

                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint sgb = j + (ib * 32u + sb * 4u) * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint qsv = (uint)qsu[qsb + u * mh];   // row pair, one load
                    const uint sgv = (uint)sgu[sgb + (u >> 1) * mh];
                    const uint g0  = ( qsv       & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1  = ((qsv >> 8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = IQ3S_YV(grp + u, y);
                    const uint gw0 = IQ3S_GW(g0, IQ3S_GRID(g0));
                    const uint gw1 = IQ3S_GW(g1, IQ3S_GRID(g1));
                    a0 += dot(yv, IQ3S_W(g0, gw0,  sgv       & 0xFFu, base));
                    a1 += dot(yv, IQ3S_W(g1, gw1, (sgv >> 8) & 0xFFu, base));
                }
                acc0 += (float)(1u + 2u * nib0) * a0;
                acc1 += (float)(1u + 2u * nib1) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ3S_MV_NSG > 1
    __local float2 part[IQ3S_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3S_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
}
