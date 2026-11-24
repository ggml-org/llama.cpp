#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if Q2_0_MV_WIDE
#define Q2_0_MV_ROWS 8
#define Q2_0_MV_LOAD_D vload8
#define Q2_0_MV_STORE vstore8
#define Q2_0_MV_FLOAT convert_float8
typedef float8 q2_0_mv_acc_t;
typedef uchar8 q2_0_mv_quant_t;
#else
#define Q2_0_MV_ROWS 4
#define Q2_0_MV_LOAD_D vload4
#define Q2_0_MV_STORE vstore4
#define Q2_0_MV_FLOAT convert_float4
typedef float4 q2_0_mv_acc_t;
typedef uchar4 q2_0_mv_quant_t;
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

    const uint mr = (uint)ne01/Q2_0_MV_ROWS;
    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint lanes = get_local_size(0);
    const uint nsg = get_local_size(1);
    const uint col = get_group_id(1);
    const uint j = get_group_id(0)*lanes + lid;
    global const float * y = src1 + (ulong)col * (uint)ne10;

    q2_0_mv_acc_t sum = 0.0f;
    if (j < mr) {
        for (uint ib = sgi; ib < (uint)ne00/64; ib += nsg) {
            const q2_0_mv_acc_t d = Q2_0_MV_FLOAT(Q2_0_MV_LOAD_D(j + ib*mr, src0_d));
            q2_0_mv_acc_t acc = 0.0f;
            for (uint u = 0; u < 16; ++u) {
                const uint group = ib*16 + u;
#if Q2_0_MV_WIDE
                const uchar8 q = as_uchar8(((global const uint2 *)src0_qs)[j + group*mr]);
#else
                const uchar4 q = as_uchar4(((global const uint *)src0_qs)[j + group*mr]);
#endif
                const float4 yv = vload4(group, y);
                acc += (Q2_0_MV_FLOAT( q                          & (q2_0_mv_quant_t)(3)) - 1.0f)*yv.s0;
                acc += (Q2_0_MV_FLOAT((q >> (q2_0_mv_quant_t)(2)) & (q2_0_mv_quant_t)(3)) - 1.0f)*yv.s1;
                acc += (Q2_0_MV_FLOAT((q >> (q2_0_mv_quant_t)(4)) & (q2_0_mv_quant_t)(3)) - 1.0f)*yv.s2;
                acc += (Q2_0_MV_FLOAT( q >> (q2_0_mv_quant_t)(6))                          - 1.0f)*yv.s3;
            }
            sum += d*acc;
        }
    }

    local q2_0_mv_acc_t part[512];
    part[sgi*lanes + lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < nsg; ++s) {
        sum += part[s*lanes + lid];
    }
    if (j < mr) {
        Q2_0_MV_STORE(sum, 0, dst + (ulong)col*(uint)ne0 + j*Q2_0_MV_ROWS);
    }
}
