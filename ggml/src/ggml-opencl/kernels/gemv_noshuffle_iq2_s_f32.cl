#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

#ifndef IQ2_S_MV_NSG
#define IQ2_S_MV_NSG 8
#endif

inline float4 iq2_s_vals(uint gv, uint sg, uint base) {

    const uint  s   = sg >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
}

kernel void kernel_iq2_s_grid_export(global uint * out) {
    out[get_global_id(0)] = iq2_s_grid[get_global_id(0)];
}

kernel void kernel_gemv_noshuffle_iq2_s_f32(
        read_only image1d_buffer_t grid_img,
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

#define IQ2_S_GRID(i) read_imageui(grid_img, (int)(i)).x

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * sgu = (global const ushort *)src0_sg;
        global const ushort * qhu = (global const ushort *)src0_qh;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += IQ2_S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint sub = ib * 8u + sb;
                const uint qhv = (uint)qhu[j + sub * mh];
                const uint scv = (uint)scu[j + sub * mh];
                const uint gb  = j + (sub * 4u) * mh;

                for (uint h = 0; h < 2u; ++h) {
                    const uint n0 = (h == 0u) ? ( scv        & 0xFu) : (( scv        >> 4) & 0xFu);
                    const uint n1 = (h == 0u) ? ((scv >> 8)  & 0xFu) : (((scv >> 8)  >> 4) & 0xFu);

                    float a0 = 0.f, a1 = 0.f;
                    for (uint t = 0; t < 2u; ++t) {
                        const uint l   = 2u*h + t;
                        const uint qsv = (uint)qsu[gb + l * mh];
                        const uint sgv = (uint)sgu[gb + l * mh];
                        const uint gi0 = ( qsv       & 0xFFu) | ((((qhv      ) >> (2u*l)) & 3u) << 8);
                        const uint gi1 = ((qsv >> 8) & 0xFFu) | ((((qhv >> 8) >> (2u*l)) & 3u) << 8);
                        const uint s0  =  sgv       & 0xFFu;
                        const uint s1  = (sgv >> 8) & 0xFFu;

                        const uint grp = (ib * 64u + sb * 8u) + l * 2u;
                        const float4 y0 = vload4(grp + 0u, y);
                        const float4 y1 = vload4(grp + 1u, y);
                        a0 += dot(y0, iq2_s_vals(IQ2_S_GRID(2u*gi0 + 0u), s0, 0u));
                        a0 += dot(y1, iq2_s_vals(IQ2_S_GRID(2u*gi0 + 1u), s0, 4u));
                        a1 += dot(y0, iq2_s_vals(IQ2_S_GRID(2u*gi1 + 0u), s1, 0u));
                        a1 += dot(y1, iq2_s_vals(IQ2_S_GRID(2u*gi1 + 1u), s1, 4u));
                    }
                    acc0 += (0.5f + (float)n0) * a0;
                    acc1 += (0.5f + (float)n1) * a1;
                }
            }
            sumf  += (float)dh.s0 * 0.25f * acc0;
            sumf1 += (float)dh.s1 * 0.25f * acc1;
        }
    }

#if IQ2_S_MV_NSG > 1
    __local float2 part[IQ2_S_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2_S_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#undef IQ2_S_GRID
}
