#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

#ifndef IQ2_XS_MV_NSG
#define IQ2_XS_MV_NSG 4
#endif

inline float4 iq2_xs_vals(uint gv, uint sgv, uint base) {
    const uint  s   = sgv >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
}

kernel void kernel_gemv_noshuffle_iq2_xs_f32(
        __read_only image1d_buffer_t grid_img,
        global const ushort * src0_qs,
        global const uchar  * src0_sc,
        global const half   * src0_d,
        global const float  * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;

#define IQ2_XS_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const uint   * qsu = (global const uint   *)src0_qs;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ2_XS_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint scv = (uint)scu[j + sub * mh];

                const float lo0 = 0.5f + (float)( scv        & 0xFu);
                const float hi0 = 0.5f + (float)((scv >>  4) & 0xFu);
                const float lo1 = 0.5f + (float)((scv >>  8) & 0xFu);
                const float hi1 = 0.5f + (float)((scv >> 12) & 0xFu);

                const uint grp = sub * 4u;
                const uint qsb = j + grp * mh;

                float alo0 = 0.f, ahi0 = 0.f, alo1 = 0.f, ahi1 = 0.f;
                for (uint u = 0; u < 4u; ++u) {
                    const uint qsv = qsu[qsb + u * mh];
                    const uint q0  = qsv & 0xFFFFu;
                    const uint q1  = qsv >> 16;

                    const uint g0  = (q0 & 511u) << 1;
                    const uint g1  = (q1 & 511u) << 1;
                    const uint sg0 = iq2_xs_signs(q0 >> 9);
                    const uint sg1 = iq2_xs_signs(q1 >> 9);

                    const float4 yl = vload4(sub * 8u + u * 2u + 0u, y);
                    const float4 yh = vload4(sub * 8u + u * 2u + 1u, y);

                    const float a0 = dot(yl, iq2_xs_vals(IQ2_XS_GRID(g0    ), sg0, 0u))
                                   + dot(yh, iq2_xs_vals(IQ2_XS_GRID(g0 + 1), sg0, 4u));
                    const float a1 = dot(yl, iq2_xs_vals(IQ2_XS_GRID(g1    ), sg1, 0u))
                                   + dot(yh, iq2_xs_vals(IQ2_XS_GRID(g1 + 1), sg1, 4u));

                    if (u < 2u) { alo0 += a0; alo1 += a1; }
                    else        { ahi0 += a0; ahi1 += a1; }
                }
                acc0 += lo0 * alo0 + hi0 * ahi0;
                acc1 += lo1 * alo1 + hi1 * ahi1;
            }
            sumf  += d0 * 0.25f * acc0;
            sumf1 += d1 * 0.25f * acc1;
        }
    }

#if IQ2_XS_MV_NSG > 1
    __local float2 part[IQ2_XS_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2_XS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#undef IQ2_XS_GRID
}

kernel void kernel_iq2_xs_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 1024u) {
        out[i] = iq2_xs_grid[i];
    }
}
