#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

#ifndef IQ4_XS_MV_NSG
#define IQ4_XS_MV_NSG 8
#endif

inline float4 iq4_xs_vals(ushort q) {
    return (float4)((char)iq4_xs_code(q & 15), (char)iq4_xs_code((q >> 4) & 15),
                    (char)iq4_xs_code((q >> 8) & 15), (char)iq4_xs_code(q >> 12));
}

kernel void kernel_gemv_noshuffle_iq4_xs_f32(
        global const ushort * src0_qs,
        global const uint * src0_sl,
        global const ushort * src0_sh,
        global const half * src0_d,
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
    dst = (global float *)((global char *)dst + offsetd);

    const uint m = (uint)ne01;
    const uint mh = m >> 1;
    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);
    const uint j = get_group_id(0)*64u + lid;
    const uint row = j << 1;
    global const float * y = src1 + (ulong)col * (uint)ne10;

    float sum0 = 0.0f, sum1 = 0.0f;
    if (j < mh) {
        global const uint * qs = (global const uint *)src0_qs;
        global const uint * sh = (global const uint *)src0_sh;
        for (uint ib = sgi; ib < (uint)ne00/QK_K; ib += IQ4_XS_MV_NSG) {
            const uint base = j + ib*mh;
            const uint2 sl = vload2(base, src0_sl);
            const uint shp = sh[base];
            const half2 d = vload2(base, src0_d);
            float acc0 = 0.0f, acc1 = 0.0f;
            for (uint sb = 0; sb < 8; ++sb) {
                const int ls0 = (int)(((sl.s0 >> (4*sb)) & 15u) | (((shp >> (2*sb)) & 3u) << 4)) - 32;
                const int ls1 = (int)(((sl.s1 >> (4*sb)) & 15u) | (((shp >> (16 + 2*sb)) & 3u) << 4)) - 32;
                float a0 = 0.0f, a1 = 0.0f;
                for (uint u = 0; u < 8; ++u) {
                    const uint group = ib*64 + sb*8 + u;
                    const uint q = qs[j + group*mh];
                    const float4 yv = vload4(group, y);
                    a0 += dot(yv, iq4_xs_vals((ushort)q));
                    a1 += dot(yv, iq4_xs_vals((ushort)(q >> 16)));
                }
                acc0 += (float)ls0*a0;
                acc1 += (float)ls1*a1;
            }
            sum0 += (float)d.s0*acc0;
            sum1 += (float)d.s1*acc1;
        }
    }

#if IQ4_XS_MV_NSG > 1
    local float2 part[IQ4_XS_MV_NSG][64];
    part[sgi][lid] = (float2)(sum0, sum1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ4_XS_MV_NSG; ++s) {
        const float2 v = part[s][lid];
        sum0 += v.s0;
        sum1 += v.s1;
    }
#endif
    if (j < mh) {
        vstore2((float2)(sum0, sum1), 0, dst + (ulong)col*(uint)ne0 + row);
    }
}
