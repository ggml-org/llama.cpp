#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

#ifndef IQ2_XXS_MV_NSG
#define IQ2_XXS_MV_NSG 4
#endif

inline float4 iq2_xxs_vals(uint gv, uint sgv, uint base) {
    const uint  s   = sgv >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
}

kernel void kernel_gemv_noshuffle_iq2_xxs_f32(
        __read_only image1d_buffer_t grid_img,
        global const uchar * src0_qs,
        global const uint  * src0_sas,
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

#define IQ2_XXS_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;

        for (uint ib = sgi; ib < nsb; ib += IQ2_XXS_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint  sub = ib * 8u + sb;
                const uint2 aux = vload2(j + sub * mh, src0_sas);

                const uint grp = sub * 4u;
                const uint qsb = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 4u; ++u) {
                    const uint qsv = (uint)qsu[qsb + u * mh];
                    const uint sg0 = iq2_xxs_signs((aux.s0 >> (7u * u)) & 127u);
                    const uint sg1 = iq2_xxs_signs((aux.s1 >> (7u * u)) & 127u);

                    const uint g0 = ( qsv       & 0xFFu) << 1;
                    const uint g1 = ((qsv >> 8) & 0xFFu) << 1;

                    const float4 yl = vload4(sub * 8u + u * 2u + 0u, y);
                    const float4 yh = vload4(sub * 8u + u * 2u + 1u, y);

                    a0 += dot(yl, iq2_xxs_vals(IQ2_XXS_GRID(g0    ), sg0, 0u))
                        + dot(yh, iq2_xxs_vals(IQ2_XXS_GRID(g0 + 1), sg0, 4u));
                    a1 += dot(yl, iq2_xxs_vals(IQ2_XXS_GRID(g1    ), sg1, 0u))
                        + dot(yh, iq2_xxs_vals(IQ2_XXS_GRID(g1 + 1), sg1, 4u));
                }
                acc0 += (0.5f + (float)(aux.s0 >> 28)) * a0;
                acc1 += (0.5f + (float)(aux.s1 >> 28)) * a1;
            }
            sumf  += d0 * 0.25f * acc0;
            sumf1 += d1 * 0.25f * acc1;
        }
    }

#if IQ2_XXS_MV_NSG > 1
    __local float2 part[IQ2_XXS_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2_XXS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#undef IQ2_XXS_GRID
}

kernel void kernel_iq2_xxs_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 512u) {
        out[i] = iq2_xxs_grid[i];
    }
}
