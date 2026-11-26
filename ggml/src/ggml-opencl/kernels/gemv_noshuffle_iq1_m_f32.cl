#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256
#define IQ1_M_DELTA 0.125f

#ifndef IQ1_M_MV_NSG
#define IQ1_M_MV_NSG 4
#endif

inline float4 iq1_m_vals(uint g) {
    return (float4)((float)( g        & 0xFu), (float)((g >>  8) & 0xFu),
                    (float)((g >> 16) & 0xFu), (float)((g >> 24) & 0xFu));
}

inline float iq1_m_super(uint w0, uint w1, uint w2, uint w3) {
    const ushort u16 = (ushort)(((w0 >> 12)) | ((w1 >> 8) & 0x00f0u)
                              | ((w2 >> 4) & 0x0f00u) | (w3 & 0xf000u));
    return (float)as_half(u16);
}

kernel void kernel_gemv_noshuffle_iq1_m_f32(
        __read_only image1d_buffer_t grid_img,
        global const uchar  * src0_qs,
        global const uchar  * src0_qh,
        global const ushort * src0_sc,
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

#define IQ1_M_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const uint   * scu = (global const uint   *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ1_M_MV_NSG) {
            const uint sb = j + (ib * 4u) * mh;
            const uint w0 = scu[sb + 0u * mh];
            const uint w1 = scu[sb + 1u * mh];
            const uint w2 = scu[sb + 2u * mh];
            const uint w3 = scu[sb + 3u * mh];
            const float d0 = iq1_m_super( w0        & 0xFFFFu,  w1        & 0xFFFFu,
                                         w2        & 0xFFFFu,  w3        & 0xFFFFu);
            const float d1 = iq1_m_super((w0 >> 16) & 0xFFFFu, (w1 >> 16) & 0xFFFFu,
                                        (w2 >> 16) & 0xFFFFu, (w3 >> 16) & 0xFFFFu);

            float acc0 = 0.f, acc1 = 0.f;

            for (uint sub_l = 0; sub_l < 8u; ++sub_l) {
                const uint sub = ib * 8u + sub_l;
                const uint scw = scu[j + (ib * 4u + (sub_l >> 1)) * mh];
                const uint grp = sub * 8u;
                const uint shb = 3u * (2u * (sub_l & 1u));

                for (uint l = 0; l < 4u; ++l) {
                    const uint sh  = shb + 3u * (l >> 1);
                    const float s0 = (float)(2u * (((scw       ) >> sh) & 7u) + 1u);
                    const float s1 = (float)(2u * (((scw >> 16) >> sh) & 7u) + 1u);

                    const uint qhv = (uint)qhu[j + (sub * 2u + (l >> 1)) * mh];
                    const uint h0  =  qhv       & 0xFFu;
                    const uint h1  = (qhv >> 8) & 0xFFu;
                    const uint qsv = (uint)qsu[j + (sub * 4u + l) * mh];

                    const uint msk = 0x08u << (4u * (l & 1u));
                    const float t0 = (h0 & msk) ? (-1.f - IQ1_M_DELTA) : (-1.f + IQ1_M_DELTA);
                    const float t1 = (h1 & msk) ? (-1.f - IQ1_M_DELTA) : (-1.f + IQ1_M_DELTA);

                    const uint g0 = IQ1_M_GRID(( qsv       & 0xFFu)
                                              | ((((h0 >> (4u*(l & 1u))) & 7u) << 8)));
                    const uint g1 = IQ1_M_GRID(((qsv >> 8) & 0xFFu)
                                              | ((((h1 >> (4u*(l & 1u))) & 7u) << 8)));

                    const float4 y0 = vload4(grp + 2u*l + 0u, y);
                    const float4 y1 = vload4(grp + 2u*l + 1u, y);

                    const float4 ys = y0 + y1;
                    const float  as = ys.s0 + ys.s1 + ys.s2 + ys.s3;

                    const float a0 = dot(y0, iq1_m_vals(g0)) + dot(y1, iq1_m_vals(g0 >> 4));
                    const float a1 = dot(y0, iq1_m_vals(g1)) + dot(y1, iq1_m_vals(g1 >> 4));
                    acc0 += s0 * (a0 + t0 * as);
                    acc1 += s1 * (a1 + t1 * as);
                }
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ1_M_MV_NSG > 1
    __local float2 part[IQ1_M_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ1_M_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#undef IQ1_M_GRID
}
