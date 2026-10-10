#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ4_XS decode GEMV over four feature-major planes, [k-group][row] (size preserving):
//   src0_q [row + (k/4)*m] ushort 4 codebook indices; src0_d [row + (k/256)*m] half super-block scale;
//   src0_sh / src0_sl [row + (k/256)*m] ushort scales_h / uint the four scales_l bytes.
// One row per lane makes a wave's weight read contiguous; K is also split across IQ4XS_MV_NSG
// subgroups reduced in local memory. Sub-scale: ls = scales_l nibble | ((scales_h >> 2*sb) & 3) << 4,
// d_w = d * (ls - 32).

#define QK_K 256

// Subgroups per workgroup; each takes every NSG'th super-block along K.
#ifndef IQ4XS_MV_NSG
#define IQ4XS_MV_NSG 8
#endif





#define IQ4XS_YV(g, y) vload4((g), (y))


// n >> 2 picks the uint; its two bits are (n & 4) and (n & 8).
inline float iq4nl_cbsel(uint n) {
    const uint t01 = (n & 4u) ? 0xF6EADDCFu : 0xBFAD9881u;
    const uint t23 = (n & 4u) ? 0x71594535u : 0x26190D01u;
    const uint t   = (n & 8u) ? t23 : t01;
    return (float)(char)((t >> ((n & 3u) * 8u)) & 0xFFu);
}

#define IQ4XS_CB_DECL
#define IQ4XS_CB(n) iq4nl_cbsel((n))

// One super-block quarter: four nibbles of `w` against the float4 `yv`.
#define IQ4XS_ACC(a, yv, w)                                    \
    do {                                                       \
        a += (yv).s0 * IQ4XS_CB(((uint)(w)      ) & 0xFu);     \
        a += (yv).s1 * IQ4XS_CB(((uint)(w) >>  4) & 0xFu);     \
        a += (yv).s2 * IQ4XS_CB(((uint)(w) >>  8) & 0xFu);     \
        a += (yv).s3 * IQ4XS_CB(((uint)(w) >> 12) & 0xFu);     \
    } while (0)

kernel void kernel_mul_mv_iq4_xs_f32_flat(
        global const ushort * src0_q,
        global const half   * src0_d,
        global const ushort * src0_sh,
        global const uint   * src0_sl,
        global const float  * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sg  = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;

    IQ4XS_CB_DECL

    const uint mh  = m >> 1;                    // rows per plane row, as uints
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const uint * qu = (global const uint *)src0_q;
        global const uint * su = (global const uint *)src0_sh;

        for (uint ib = sg; ib < nsb; ib += IQ4XS_MV_NSG) {
            const uint  sbase = j + ib * mh;
            const uint2 slv   = vload2(sbase, src0_sl);
            const uint  shp   = su[sbase];
            const half2 dh    = vload2(sbase, src0_d);
            const float d0    = (float)dh.s0;
            const float d1    = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const int ls0 = (int)(((slv.s0 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp      ) >> (2u * sb)) & 3u) << 4);
                const int ls1 = (int)(((slv.s1 >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                              | (int)((((shp >> 16) >> (2u * sb)) & 3u) << 4);

                const uint grp = ib * 64u + sb * 8u;
                const uint qb  = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 8u; ++u) {
                    const uint   w  = qu[qb + u * mh];   // row pair, one load
                    const ushort w0 = (ushort)(w & 0xFFFFu);
                    const ushort w1 = (ushort)(w >> 16);
                    const float4 yv = IQ4XS_YV(grp + u, y);
                    IQ4XS_ACC(a0, yv, w0);
                    IQ4XS_ACC(a1, yv, w1);
                }
                acc0 += (float)(ls0 - 32) * a0;
                acc1 += (float)(ls1 - 32) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ4XS_MV_NSG > 1
    __local float2 part[IQ4XS_MV_NSG][64];
    part[sg][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    for (uint s = 1; s < IQ4XS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
}

