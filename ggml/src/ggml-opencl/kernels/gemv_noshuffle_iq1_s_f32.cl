#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

#ifndef IQ1_S_MV_NSG
#define IQ1_S_MV_NSG 4
#endif

inline float4 iq1_s_vals(uint g) {
    return (float4)((float)( g        & 0xFu), (float)((g >>  8) & 0xFu),
                    (float)((g >> 16) & 0xFu), (float)((g >> 24) & 0xFu));
}

kernel void kernel_gemv_noshuffle_iq1_s_f32(
        __read_only image1d_buffer_t grid_img,
        global const uchar  * src0_qs,
        global const ushort * src0_qh,
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

#define IQ1_S_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf = 0.f, sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;

        for (uint ib = sgi; ib < nsb; ib += IQ1_S_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint  sub = ib * 8u + sb;
                const ushort2 qhv = vload2(j + sub * mh, src0_qh);
                const uint  h0  = (uint)qhv.s0, h1 = (uint)qhv.s1;

                const float s0  = (float)(2u * ((h0 >> 12) & 7u) + 1u);
                const float s1  = (float)(2u * ((h1 >> 12) & 7u) + 1u);
                const float t0  = ((h0 & 0x8000u) ? -0.125f : 0.125f) - 1.0f;
                const float t1  = ((h1 & 0x8000u) ? -0.125f : 0.125f) - 1.0f;

                const uint gb  = j + (sub * 4u) * mh;
                const uint grp = sub * 8u;

                float a0 = 0.f, a1 = 0.f;
                float4 as = (float4)(0.f);
                for (uint l = 0; l < 4u; ++l) {
                    const uint qsv = (uint)qsu[gb + l * mh];
                    const uint g0  = IQ1_S_GRID(( qsv       & 0xFFu) | ((((h0 >> (3u*l)) & 7u) << 8)));
                    const uint g1  = IQ1_S_GRID(((qsv >> 8) & 0xFFu) | ((((h1 >> (3u*l)) & 7u) << 8)));

                    const float4 y0 = vload4(grp + 2u*l + 0u, y);
                    const float4 y1 = vload4(grp + 2u*l + 1u, y);
                    as += y0 + y1;
                    a0 += dot(y0, iq1_s_vals(g0)) + dot(y1, iq1_s_vals(g0 >> 4));
                    a1 += dot(y0, iq1_s_vals(g1)) + dot(y1, iq1_s_vals(g1 >> 4));
                }

                const float asum = as.s0 + as.s1 + as.s2 + as.s3;
                acc0 += s0 * (a0 + t0 * asum);
                acc1 += s1 * (a1 + t1 * asum);
            }
            sumf  += (float)dh.s0 * acc0;
            sumf1 += (float)dh.s1 * acc1;
        }
    }

#if IQ1_S_MV_NSG > 1
    __local float2 part[IQ1_S_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ1_S_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#undef IQ1_S_GRID
}

kernel void kernel_iq1_s_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 2048u) {
        out[i] = iq1_s_grid[i];
    }
}
