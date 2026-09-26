#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Q3_K decode GEMV over the feature-major plane split. Same structure as
// mul_mv_q2_k_f32_flat: K split across Q3K_MV_NSG subgroups because one row per
// lane leaves the GPU short of work items, and Q3K_MV_R gives a lane that many
// adjacent rows, so the uchar planes are read a ushort or a uint at a time.
//
// Every plane here is [position][row] with the row as the FASTEST axis, so
// widening the load is all it takes to pick up more rows, and the four rows also
// share the activation vload4 and the y-side work.
//
// R=4 is offered and swept per type. An earlier note said R=4 loses, but that was
// measured on IQ3_S, a codebook type, and the reason given (it quarters the grid)
// no longer holds: this kernel gained a workgroup-level K split, which puts
// ksplit in the z dimension and restores the occupancy. Q2_K, a linear quant like
// this one, ships R=4 and measures R=2 as 18-23% worse.
//
// R=4 needs ne01 % 4 == 0; the host declines the whole plane split otherwise, so
// a tensor that cannot be read this way is never split in the first place.

#define QK_K 256

#ifndef Q3K_MV_NSG
#define Q3K_MV_NSG 8
#endif

// rows per lane: 1, 2 or 4
#ifndef Q3K_MV_R
#define Q3K_MV_R 2
#endif

// The 16 six-bit sub-scales live in 12 bytes, interleaved the way
// dequantize_row_q3_K unpacks them. Rebuilding just the one this lane needs is
// four cases on the word index, so the whole aux[] shuffle is never materialised.
//   is = 2*sb + half   (half picks the 16-weight run inside the 32-block)
inline int q3k_scale(uint sc0, uint sc1, uint sc2, uint is) {
    const uint w  = is >> 2;
    const uint sh = 8u * (is & 3u);
    const uint tb = (sc2 >> sh) & 0xFFu;
    uint v;
    if      (w == 0u) { v = ( (sc0 >> sh)       & 0xFu) | (((tb >> 0) & 3u) << 4); }
    else if (w == 1u) { v = ( (sc1 >> sh)       & 0xFu) | (((tb >> 2) & 3u) << 4); }
    else if (w == 2u) { v = (((sc0 >> sh) >> 4) & 0xFu) | (((tb >> 4) & 3u) << 4); }
    else              { v = (((sc1 >> sh) >> 4) & 0xFu) | (((tb >> 6) & 3u) << 4); }
    return (int)v - 32;
}

// Four weights of one group: two bits each from pk, one high bit each from the
// nibble hb. value = low - 4 + 4*high, i.e. -4..3.
inline uint q3k_pack(uint pk, uint hb) {
    int v0 = (int)((pk      ) & 3u) - 4 + 4*(int)((hb     ) & 1u);
    int v1 = (int)((pk >> 2) & 3u) - 4 + 4*(int)((hb >> 1) & 1u);
    int v2 = (int)((pk >> 4) & 3u) - 4 + 4*(int)((hb >> 2) & 1u);
    int v3 = (int)((pk >> 6) & 3u) - 4 + 4*(int)((hb >> 3) & 1u);
    return ((uint)v0 & 0xFFu) | (((uint)v1 & 0xFFu) <<  8)
         | (((uint)v2 & 0xFFu) << 16) | (((uint)v3 & 0xFFu) << 24);
}

// Four weights of one group as floats.
inline float4 q3k_vals(uint pk, uint hb) {
    float4 v;
    v.s0 = (float)((int)((pk      ) & 3u) - 4 + 4*(int)((hb     ) & 1u));
    v.s1 = (float)((int)((pk >> 2) & 3u) - 4 + 4*(int)((hb >> 1) & 1u));
    v.s2 = (float)((int)((pk >> 4) & 3u) - 4 + 4*(int)((hb >> 2) & 1u));
    v.s3 = (float)((int)((pk >> 6) & 3u) - 4 + 4*(int)((hb >> 3) & 1u));
    return v;
}

kernel void kernel_mul_mv_q3_k_f32_flat(
        global const uchar * src0_qs,
        global const uchar * src0_hm,
        global const uint  * src0_sc,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
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

#if Q3K_MV_R == 4
    const uint mq  = m >> 2;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 2;

    float sumf = 0.f, sumf1 = 0.f, sumf2 = 0.f, sumf3 = 0.f;

    if (j < mq) {
        global const uint * qsw = (global const uint *)src0_qs;
        global const uint * hmw = (global const uint *)src0_hm;

        for (uint ib = sgi; ib < nsb; ib += Q3K_MV_NSG) {
            const half4 dh = vload4(j + ib * mq, src0_d);
            const uint4 s0 = vload4(j + (3u * ib + 0u) * mq, src0_sc);
            const uint4 s1 = vload4(j + (3u * ib + 1u) * mq, src0_sc);
            const uint4 s2 = vload4(j + (3u * ib + 2u) * mq, src0_sc);

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mq;
                const uint hmb = j + (grp >> 1) * mq;

                for (uint h = 0; h < 2u; ++h) {         // two 16-weight halves
                    const int l0 = q3k_scale(s0.s0, s1.s0, s2.s0, 2u*sb + h);
                    const int l1 = q3k_scale(s0.s1, s1.s1, s2.s1, 2u*sb + h);
                    const int l2 = q3k_scale(s0.s2, s1.s2, s2.s2, 2u*sb + h);
                    const int l3 = q3k_scale(s0.s3, s1.s3, s2.s3, 2u*sb + h);

                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = qsw[qsb + gg * mq];        // four rows, one load
                        const uint hmv = hmw[hmb + (gg >> 1) * mq];
                        const uint hsh = 4u * (gg & 1u);
                        const float4 yv = vload4(grp + gg, y);      // shared by all four
                        a0 += dot(yv, q3k_vals((qsv      ) & 0xFFu, ((hmv      ) >> hsh) & 0xFu));
                        a1 += dot(yv, q3k_vals((qsv >>  8) & 0xFFu, ((hmv >>  8) >> hsh) & 0xFu));
                        a2 += dot(yv, q3k_vals((qsv >> 16) & 0xFFu, ((hmv >> 16) >> hsh) & 0xFu));
                        a3 += dot(yv, q3k_vals((qsv >> 24) & 0xFFu, ((hmv >> 24) >> hsh) & 0xFu));
                    }
                    acc0 += (float)l0 * a0;
                    acc1 += (float)l1 * a1;
                    acc2 += (float)l2 * a2;
                    acc3 += (float)l3 * a3;
                }
            }
            sumf  += (float)dh.s0 * acc0;
            sumf1 += (float)dh.s1 * acc1;
            sumf2 += (float)dh.s2 * acc2;
            sumf3 += (float)dh.s3 * acc3;
        }
    }
#elif Q3K_MV_R == 2
    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * hmu = (global const ushort *)src0_hm;

        for (uint ib = sgi; ib < nsb; ib += Q3K_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const uint2 s0 = vload2(j + (3u * ib + 0u) * mh, src0_sc);
            const uint2 s1 = vload2(j + (3u * ib + 1u) * mh, src0_sc);
            const uint2 s2 = vload2(j + (3u * ib + 2u) * mh, src0_sc);

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mh;
                const uint hmb = j + (grp >> 1) * mh;

                for (uint h = 0; h < 2u; ++h) {         // two 16-weight halves
                    const int l0 = q3k_scale(s0.s0, s1.s0, s2.s0, 2u*sb + h);
                    const int l1 = q3k_scale(s0.s1, s1.s1, s2.s1, 2u*sb + h);

                    float a0 = 0.f, a1 = 0.f;
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = (uint)qsu[qsb + gg * mh];
                        const uint hmv = (uint)hmu[hmb + (gg >> 1) * mh];
                        const uint hsh = 4u * (gg & 1u);
                        const float4 yv = vload4(grp + gg, y);
                        a0 += dot(yv, q3k_vals( qsv       & 0xFFu, ( hmv        >> hsh) & 0xFu));
                        a1 += dot(yv, q3k_vals((qsv >> 8) & 0xFFu, ((hmv >> 8)  >> hsh) & 0xFu));
                    }
                    acc0 += (float)l0 * a0;
                    acc1 += (float)l1 * a1;
                }
            }
            sumf  += (float)dh.s0 * acc0;
            sumf1 += (float)dh.s1 * acc1;
        }
    }
#else
    const uint row = get_group_id(0) * 64u + lid;

    float sumf = 0.f;

    if (row < m) {
        for (uint ib = sgi; ib < nsb; ib += Q3K_MV_NSG) {
            const float d  = (float)src0_d[row + ib * m];
            const uint  s0 = src0_sc[row + (3u * ib + 0u) * m];
            const uint  s1 = src0_sc[row + (3u * ib + 1u) * m];
            const uint  s2 = src0_sc[row + (3u * ib + 2u) * m];

            float acc = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;
                const uint hmb = row + (grp >> 1) * m;

                for (uint h = 0; h < 2u; ++h) {
                    const int l = q3k_scale(s0, s1, s2, 2u*sb + h);
                    float a = 0.f;
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = (uint)src0_qs[qsb + gg * m];
                        const uint hmv = (uint)src0_hm[hmb + (gg >> 1) * m];
                        const float4 yv = vload4(grp + gg, y);
                        a += dot(yv, q3k_vals(qsv, (hmv >> (4u * (gg & 1u))) & 0xFu));
                    }
                    acc += (float)l * a;
                }
            }
            sumf += d * acc;
        }
    }
#endif

#if Q3K_MV_NSG > 1
#if Q3K_MV_R == 4
    __local float4 part[Q3K_MV_NSG][64];
    part[sgi][lid] = (float4)(sumf, sumf1, sumf2, sumf3);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q3K_MV_NSG; ++s) {
        const float4 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
        sumf2 += p.s2;
        sumf3 += p.s3;
    }
#elif Q3K_MV_R == 2
    __local float2 part[Q3K_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q3K_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#else
    __local float part[Q3K_MV_NSG][64];
    part[sgi][lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q3K_MV_NSG; ++s) {
        sumf += part[s][lid];
    }
#endif
#endif

#if Q3K_MV_R == 4
    if (j < mq) {
        vstore4((float4)(sumf, sumf1, sumf2, sumf3), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#elif Q3K_MV_R == 2
    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
#else
    if (row < m) {
        dst[(ulong)col * (uint)ne0 + row] = sumf;
    }
#endif
}
