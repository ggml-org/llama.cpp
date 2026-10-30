#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Q2_0_MV_NSG
#define Q2_0_MV_NSG 8
#endif

kernel void kernel_gemv_noshuffle_q2_0_f32(
        global const uchar * src0_qs,
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

    float2 sum = 0.0f;
    if (j < mh) {
        global const ushort * qs = (global const ushort *)src0_qs;
        for (uint ib = sgi; ib < (uint)ne00/64; ib += Q2_0_MV_NSG) {
            const float2 d = convert_float2(vload2(j + ib*mh, src0_d));
            float2 acc = 0.0f;
            for (uint u = 0; u < 16; ++u) {
                const uint group = ib*16 + u;
                const uint q = qs[j + group*mh];
                const float4 yv = vload4(group, y);
                const int4 v0 = convert_int4(((uint4)(q) >> (uint4)(0, 2, 4, 6)) & (uint4)(3)) - (int4)(1);
                const int4 v1 = convert_int4(((uint4)(q) >> (uint4)(8, 10, 12, 14)) & (uint4)(3)) - (int4)(1);
                acc.s0 += dot(yv, convert_float4(v0));
                acc.s1 += dot(yv, convert_float4(v1));
            }
            sum += d*acc;
        }
    }

#if Q2_0_MV_NSG > 1
    local float2 part[Q2_0_MV_NSG][64];
    part[sgi][lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < Q2_0_MV_NSG; ++s) {
        sum += part[s][lid];
    }
#endif
    if (j < mh) {
        vstore2(sum, 0, dst + (ulong)col*(uint)ne0 + row);
    }
}
