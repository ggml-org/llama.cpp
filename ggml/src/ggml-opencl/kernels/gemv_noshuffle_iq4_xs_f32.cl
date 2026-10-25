#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256

inline float4 iq4_xs_vals(ushort q) {
    return (float4)(iq4_xs_values[q & 15], iq4_xs_values[(q >> 4) & 15],
                    iq4_xs_values[(q >> 8) & 15], iq4_xs_values[q >> 12]);
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
    const uint row = get_global_id(0);
    const uint col = get_group_id(1);
    if (row >= m) {
        return;
    }
    global const float * y = src1 + (ulong)col * (uint)ne10;
    float sum = 0.0f;
    for (uint ib = 0; ib < (uint)ne00/QK_K; ++ib) {
        const uint sl = src0_sl[row + ib*m];
        const uint sh = src0_sh[row + ib*m];
        float acc = 0.0f;
        for (uint sb = 0; sb < 8; ++sb) {
            const int ls = (int)(((sl >> (4*sb)) & 15u) | (((sh >> (2*sb)) & 3u) << 4)) - 32;
            float a = 0.0f;
            for (uint u = 0; u < 8; ++u) {
                const uint group = ib*64 + sb*8 + u;
                a += dot(vload4(group, y), iq4_xs_vals(src0_qs[row + group*m]));
            }
            acc += (float)ls*a;
        }
        sum += (float)src0_d[row + ib*m]*acc;
    }
    dst[(ulong)col*(uint)ne0 + row] = sum;
}
