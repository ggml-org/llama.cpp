#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Q2_K decode GEMV over the feature-major plane split. Same structure as
// mul_mv_q3_k_f32_flat: K is split across Q2K_MV_NSG subgroups, and each lane
// takes Q2K_MV_R adjacent rows so the uchar planes are read a word at a time.
//
// Q2_K needs more rows per lane than the other split types. It carries a min, so
// each 16-weight run also needs the sum of that run's activations, and that sum
// is the same for every row. At 2 rows per lane the sum is paid twice as often
// per row and decode measured 12% slower, so the default fold is 4.

#define QK_K 256

#ifndef Q2K_MV_NSG
#define Q2K_MV_NSG 8
#endif

// Rows per work item: 1, 2 or 4. Each is written out, because a generic r-loop
// measured 8% slower.
#ifndef Q2K_MV_R
#define Q2K_MV_R 4
#endif

inline float4 q2k_vals(uint pk) {
    return (float4)((float)( pk       & 3u), (float)((pk >> 2) & 3u),
                    (float)((pk >> 4) & 3u), (float)((pk >> 6) & 3u));
}

kernel void kernel_mul_mv_q2_k_f32_flat(
        global const uchar * src0_qs,
        global const uchar * src0_sc,
        global const half  * src0_dm,
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

    const uint mr  = m / Q2K_MV_R;                  // row groups
    const uint j   = get_group_id(0) * 64u + lid;   // row group this lane owns
    const uint row = j * Q2K_MV_R;

#if Q2K_MV_R == 4
    float sumf0 = 0.f, sumf1 = 0.f, sumf2 = 0.f, sumf3 = 0.f;

    if (j < mr) {
        global const uint * qsu = (global const uint *)src0_qs;
        global const uint * scu = (global const uint *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            // the d/dmin plane element is a half PAIR, so four rows of one
            // super-block are eight halves starting at 2*(j + ib*mr) half4s
            const half4 dma = vload4(2u*(j + ib*mr) + 0u, src0_dm);  // d0, dmin0, d1, dmin1
            const half4 dmb = vload4(2u*(j + ib*mr) + 1u, src0_dm);  // d2, dmin2, d3, dmin3

            float ad0 = 0.f, am0 = 0.f, ad1 = 0.f, am1 = 0.f;
            float ad2 = 0.f, am2 = 0.f, ad3 = 0.f, am3 = 0.f;

            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mr;

                for (uint h = 0; h < 2u; ++h) {
                    const uint scv = scu[j + (2u * (ib * 8u + sb) + h) * mr];

                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = qsu[qsb + gg * mr];   // four rows, one load
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        a0 += dot(yv, q2k_vals( qsv        & 0xFFu));
                        a1 += dot(yv, q2k_vals((qsv >>  8) & 0xFFu));
                        a2 += dot(yv, q2k_vals((qsv >> 16) & 0xFFu));
                        a3 += dot(yv, q2k_vals((qsv >> 24) & 0xFFu));
                    }
                    // one activation sum, four rows -- this is the whole point
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;

                    const uint s0 =  scv        & 0xFFu;
                    const uint s1 = (scv >>  8) & 0xFFu;
                    const uint s2 = (scv >> 16) & 0xFFu;
                    const uint s3 = (scv >> 24) & 0xFFu;
                    ad0 += (float)(s0 & 0xFu) * a0;   am0 += (float)(s0 >> 4) * asum;
                    ad1 += (float)(s1 & 0xFu) * a1;   am1 += (float)(s1 >> 4) * asum;
                    ad2 += (float)(s2 & 0xFu) * a2;   am2 += (float)(s2 >> 4) * asum;
                    ad3 += (float)(s3 & 0xFu) * a3;   am3 += (float)(s3 >> 4) * asum;
                }
            }
            sumf0 += (float)dma.s0 * ad0 - (float)dma.s1 * am0;
            sumf1 += (float)dma.s2 * ad1 - (float)dma.s3 * am1;
            sumf2 += (float)dmb.s0 * ad2 - (float)dmb.s1 * am2;
            sumf3 += (float)dmb.s2 * ad3 - (float)dmb.s3 * am3;
        }
    }

#if Q2K_MV_NSG > 1
    __local float4 part[Q2K_MV_NSG][64];
    part[sgi][lid] = (float4)(sumf0, sumf1, sumf2, sumf3);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        const float4 p = part[s][lid];
        sumf0 += p.s0; sumf1 += p.s1; sumf2 += p.s2; sumf3 += p.s3;
    }
#endif

    if (j < mr) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = sumf0; o[1] = sumf1; o[2] = sumf2; o[3] = sumf3;
    }

#elif Q2K_MV_R == 2
    float sumf0 = 0.f, sumf1 = 0.f;

    if (j < mr) {
        global const ushort * qsu = (global const ushort *)src0_qs;
        global const ushort * scu = (global const ushort *)src0_sc;

        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            const half4 dm = vload4(j + ib * mr, src0_dm);  // d0, dmin0, d1, dmin1

            float ad0 = 0.f, am0 = 0.f, ad1 = 0.f, am1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = j + grp * mr;

                for (uint h = 0; h < 2u; ++h) {
                    const uint scv = (uint)scu[j + (2u * (ib * 8u + sb) + h) * mr];
                    const uint s0  = scv & 0xFFu;
                    const uint s1  = scv >> 8;

                    float a0 = 0.f, a1 = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg  = 4u*h + u;
                        const uint qsv = (uint)qsu[qsb + gg * mr];
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        a0 += dot(yv, q2k_vals( qsv       & 0xFFu));
                        a1 += dot(yv, q2k_vals((qsv >> 8) & 0xFFu));
                    }
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;
                    ad0 += (float)(s0 & 0xFu) * a0;   am0 += (float)(s0 >> 4) * asum;
                    ad1 += (float)(s1 & 0xFu) * a1;   am1 += (float)(s1 >> 4) * asum;
                }
            }
            sumf0 += (float)dm.s0 * ad0 - (float)dm.s1 * am0;
            sumf1 += (float)dm.s2 * ad1 - (float)dm.s3 * am1;
        }
    }

#if Q2K_MV_NSG > 1
    __local float2 part[Q2K_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf0, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf0 += p.s0; sumf1 += p.s1;
    }
#endif

    if (j < mr) {
        vstore2((float2)(sumf0, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }

#else
    float sumf0 = 0.f;

    if (j < mr) {
        for (uint ib = sgi; ib < nsb; ib += Q2K_MV_NSG) {
            const half2 dm = vload2(row + ib * m, src0_dm);

            float ad = 0.f, am = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;

                for (uint h = 0; h < 2u; ++h) {
                    const uint sc = (uint)src0_sc[row + (2u * (ib * 8u + sb) + h) * m];

                    float a = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg = 4u*h + u;
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        const uint pkv = (uint)src0_qs[qsb + gg * m];
                        a += dot(yv, q2k_vals(pkv));
                    }
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;
                    ad += (float)(sc & 0xFu) * a;   am += (float)(sc >> 4) * asum;
                }
            }
            sumf0 += (float)dm.s0 * ad - (float)dm.s1 * am;
        }
    }

#if Q2K_MV_NSG > 1
    __local float part[Q2K_MV_NSG][64];
    part[sgi][lid] = sumf0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2K_MV_NSG; ++s) {
        sumf0 += part[s][lid];
    }
#endif

    if (j < mr) {
        dst[(ulong)col * (uint)ne0 + row] = sumf0;
    }
#endif
}
