#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

#ifndef IQ3_S_MV_NSG
#define IQ3_S_MV_NSG 4
#endif

inline float4 iq3_s_vals(uint gv, uint sg, uint base) {

    const uint  s   = sg >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
}

kernel void kernel_gemv_noshuffle_iq3_s_f32(
        __read_only image1d_buffer_t grid_img,
        global const uchar * src0_qs,
        global const uchar * src0_sg,
        global const uchar * src0_qh,
        global const uchar * src0_sc,
        global const half  * src0_d,
        global const float * src1,
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

#define IQ3_S_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ3_S_MV_NSG) {
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
                    const uint qsv = (uint)qsu[qsb + u * mh];
                    const uint sgv = (uint)sgu[sgb + (u >> 1) * mh];
                    const uint g0  = ( qsv       & 0xFFu) | (((qh0 >> u) & 1u) << 8);
                    const uint g1  = ((qsv >> 8) & 0xFFu) | (((qh1 >> u) & 1u) << 8);
                    const uint base = (u & 1u) * 4u;
                    const float4 yv = vload4(grp + u, y);
                    const uint gw0 = IQ3_S_GRID(g0);
                    const uint gw1 = IQ3_S_GRID(g1);
                    a0 += dot(yv, iq3_s_vals(gw0,  sgv       & 0xFFu, base));
                    a1 += dot(yv, iq3_s_vals(gw1, (sgv >> 8) & 0xFFu, base));
                }
                acc0 += (float)(1u + 2u * nib0) * a0;
                acc1 += (float)(1u + 2u * nib1) * a1;
            }
            sumf  += d0 * acc0;
            sumf1 += d1 * acc1;
        }
    }

#if IQ3_S_MV_NSG > 1
    __local float2 part[IQ3_S_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ3_S_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
}

kernel void kernel_iq3_s_grid_export(global uint * out) {
    out[get_global_id(0)] = iq3_s_grid[get_global_id(0)];
}
