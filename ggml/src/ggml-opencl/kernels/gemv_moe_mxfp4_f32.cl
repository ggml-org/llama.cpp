#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_MXFP4 32
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

static inline half8 mxfp4_to_fp16_packed8(ushort2 fp4x8) { //, ushort 0x0E00, ushort 0x8000) {
    ushort2 fp16_packed_a_0, fp16_packed_b_0, bias_a, bias_b, sign_a, sign_b;
    fp16_packed_a_0.lo = (fp4x8.s0 << 9) & 0x0E00;
    fp16_packed_a_0.hi = (fp4x8.s0 << 5) & 0x0E00;
    fp16_packed_b_0.lo = (fp4x8.s0 << 1) & 0x0E00;
    fp16_packed_b_0.hi = (fp4x8.s0 >> 3) & 0x0E00;

    bias_a.lo = (fp16_packed_a_0.lo != 0) ? 0x3800 : 0x0;
    bias_a.hi = (fp16_packed_a_0.hi != 0) ? 0x3800 : 0x0;
    bias_b.lo = (fp16_packed_b_0.lo != 0) ? 0x3800 : 0x0;
    bias_b.hi = (fp16_packed_b_0.hi != 0) ? 0x3800 : 0x0;

    fp16_packed_a_0.lo = (fp16_packed_a_0.lo != 0x0200) ? fp16_packed_a_0.lo : 0x0;
    fp16_packed_a_0.hi = (fp16_packed_a_0.hi != 0x0200) ? fp16_packed_a_0.hi : 0x0;
    fp16_packed_b_0.lo = (fp16_packed_b_0.lo != 0x0200) ? fp16_packed_b_0.lo : 0x0;
    fp16_packed_b_0.hi = (fp16_packed_b_0.hi != 0x0200) ? fp16_packed_b_0.hi : 0x0;

    sign_a.lo = (fp4x8.s0 << 12) & 0x8000;
    sign_a.hi = (fp4x8.s0 << 8) & 0x8000;
    sign_b.lo = (fp4x8.s0 << 4) & 0x8000;
    sign_b.hi = fp4x8.s0 & 0x8000;

    fp16_packed_a_0 = sign_a + bias_a + fp16_packed_a_0;
    fp16_packed_b_0 = sign_b + bias_b + fp16_packed_b_0;

    ushort2 fp16_packed_a_1, fp16_packed_b_1;
    fp16_packed_a_1.lo = (fp4x8.s1 << 9) & 0x0E00;
    fp16_packed_a_1.hi = (fp4x8.s1 << 5) & 0x0E00;
    fp16_packed_b_1.lo = (fp4x8.s1 << 1) & 0x0E00;
    fp16_packed_b_1.hi = (fp4x8.s1 >> 3) & 0x0E00;

    bias_a.lo = (fp16_packed_a_1.lo != 0) ? 0x3800 : 0x0;
    bias_a.hi = (fp16_packed_a_1.hi != 0) ? 0x3800 : 0x0;
    bias_b.lo = (fp16_packed_b_1.lo != 0) ? 0x3800 : 0x0;
    bias_b.hi = (fp16_packed_b_1.hi != 0) ? 0x3800 : 0x0;

    fp16_packed_a_1.lo = (fp16_packed_a_1.lo != 0x0200) ? fp16_packed_a_1.lo : 0x0;
    fp16_packed_a_1.hi = (fp16_packed_a_1.hi != 0x0200) ? fp16_packed_a_1.hi : 0x0;
    fp16_packed_b_1.lo = (fp16_packed_b_1.lo != 0x0200) ? fp16_packed_b_1.lo : 0x0;
    fp16_packed_b_1.hi = (fp16_packed_b_1.hi != 0x0200) ? fp16_packed_b_1.hi : 0x0;

    sign_a.lo = (fp4x8.s1 << 12) & 0x8000;
    sign_a.hi = (fp4x8.s1 << 8) & 0x8000;
    sign_b.lo = (fp4x8.s1 << 4) & 0x8000;
    sign_b.hi = fp4x8.s1 & 0x8000;

    fp16_packed_a_1 = sign_a + bias_a + fp16_packed_a_1;
    fp16_packed_b_1 = sign_b + bias_b + fp16_packed_b_1;

    return as_half8((ushort8)(fp16_packed_a_0, fp16_packed_b_0, fp16_packed_a_1, fp16_packed_b_1));
}

static inline float e8m0_to_fp32(uchar x) {
    int bits;
    bits = (x == 0) ? 0x00400000 : ((uint) x << 23);
    return as_float(bits);
}


__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_mxfp4_f32(
    __global uint4 * src0_q,
    __global uchar * src0_e,
    __read_only image1d_buffer_t src1,
    __global uint * src2,
    __global float * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    int           ne11
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];
    uint expert_offset = expert_id * ne00 * ne01 / 32;

    __private float sum = 0.0f; // each thread calculate partial sum of one output

    // loop along ne00 in block granularity, skip 4 blocks every iter
    for (uint ib00 = sgid; ib00 < (ne00 / QK_MXFP4); ib00 += N_SIMDGROUP) {

        // load one block of q
        uint4 regQ = src0_q[expert_offset + ib00 * ne01 + i01];

        uint offset = i11 * ne00 / 4 + ib00 * 8;

        half8 fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s0));

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (offset + 0));
        float4 acc = shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);

        shared_y4 = read_imagef(src1, (offset + 4));
        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);


        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s1));

        shared_y4 = read_imagef(src1, (offset + 1));
        acc += shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);

        shared_y4 = read_imagef(src1, (offset + 5));
        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);


        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s2));

        shared_y4 = read_imagef(src1, (offset + 2));
        acc += shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);

        shared_y4 = read_imagef(src1, (offset + 6));
        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);


        fp16x8 = mxfp4_to_fp16_packed8(as_ushort2(regQ.s3));

        shared_y4 = read_imagef(src1, (offset + 3));
        acc += shared_y4 * (float4)(fp16x8.s0, fp16x8.s2, fp16x8.s4, fp16x8.s6);

        shared_y4 = read_imagef(src1, (offset + 7));
        acc += shared_y4 * (float4)(fp16x8.s1, fp16x8.s3, fp16x8.s5, fp16x8.s7);

        uchar regE = src0_e[ib00 * ne01 + i01 + expert_offset];
        sum += e8m0_to_fp32(regE) * ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
    }

    // reduction in local memory, assumes #subgroups=4
    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = sum;
    if (sgid == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = sum;
    if (sgid == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (sgid == 0) sum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    // 1 outputs per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = sum;
    }

}
