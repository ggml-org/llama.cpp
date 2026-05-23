#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

static inline float8 q6_k_to_fp32_packed8(ushort2 ql8, ushort qh8, float d_scale) {
    float8 fp32x8;
    fp32x8.s0 = ((float)(( ql8.s0 & 0x000F)        | ((uint)((qh8      ) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s1 = ((float)((( ql8.s0 >> 4) & 0x000F) | ((uint)((qh8 >> 2) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s2 = ((float)((( ql8.s0 >> 8) & 0x000F) | ((uint)((qh8 >> 4) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s3 = ((float)((( ql8.s0 >> 12)& 0x000F) | ((uint)((qh8 >> 6) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s4 = ((float)(( ql8.s1 & 0x000F)        | ((uint)((qh8 >> 8) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s5 = ((float)((( ql8.s1 >> 4) & 0x000F) | ((uint)((qh8 >>10) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s6 = ((float)((( ql8.s1 >> 8) & 0x000F) | ((uint)((qh8 >>12) & 0x3) << 4)) - 32.f) * d_scale;
    fp32x8.s7 = ((float)((( ql8.s1 >> 12)& 0x000F) | ((uint)((qh8 >>14) & 0x3) << 4)) - 32.f) * d_scale;
    return fp32x8;
}

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q6_k_f32_ns(
    __global uint *         src0_ql,
    __global uint *         src0_qh,
    __global char *         src0_s,
    __global half *         src0_d,
    __read_only image1d_buffer_t src1,
    __global uint *         src2,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01,
    int                     ne11
) {
    uint i01  = get_global_id(0);
    uint i20  = get_global_id(2);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    if (i01 >= ne01) {
        return;
    }

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;  // 8 sub-blocks of 32 per super-block
    int scales_per_row = num_superblocks * 16;

    // Expert offsets in the transposed noshuffle layout
    uint expert_ql_offset = expert_id * (ne00 / 8) * ne01;   // 32 uints per super-block
    uint expert_qh_offset = expert_id * (ne00 / 16) * ne01;  // 16 uints per super-block
    uint expert_d_offset  = expert_id * num_superblocks * ne01;

    __private float sum = 0.0f;

    // Loop over sub-blocks of 32 elements, N_SIMDGROUP sub-blocks per iter
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
        uint sb = ib / 8;   // super-block index
        uint j  = ib % 8;   // 32-element group within super-block

        // Load d for this super-block
        half d_val = src0_d[expert_d_offset + sb * ne01 + i01];

        // Load 2 sub-block scales
        global const char * sc = src0_s + (expert_id * ne01 + i01) * scales_per_row + sb * 16;
        float scale0 = (float)d_val * (float)sc[j * 2];
        float scale1 = (float)d_val * (float)sc[j * 2 + 1];

        // Load 4 uints of ql
        uint ql_base = expert_ql_offset + (ib * 4) * ne01 + i01;
        uint4 regQL;
        regQL.s0 = src0_ql[ql_base];
        regQL.s1 = src0_ql[ql_base + ne01];
        regQL.s2 = src0_ql[ql_base + ne01 * 2];
        regQL.s3 = src0_ql[ql_base + ne01 * 3];

        // Load 2 uints of qh
        uint qh_base = expert_qh_offset + (ib * 2) * ne01 + i01;
        uint2 regQH;
        regQH.s0 = src0_qh[qh_base];
        regQH.s1 = src0_qh[qh_base + ne01];

        // Load activations: 32 floats = 8 float4s
        uint y_offset = i11 * ne00 / 4 + ib * 8;

        float8 fp32x8 = q6_k_to_fp32_packed8(as_ushort2(regQL.s0), (ushort)(regQH.s0 & 0xFFFF), scale0);

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (y_offset + 0));
        float4 acc = shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 1));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q6_k_to_fp32_packed8(as_ushort2(regQL.s1), (ushort)(regQH.s0 >> 16), scale0);

        shared_y4 = read_imagef(src1, (y_offset + 2));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 3));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q6_k_to_fp32_packed8(as_ushort2(regQL.s2), (ushort)(regQH.s1 & 0xFFFF), scale1);

        shared_y4 = read_imagef(src1, (y_offset + 4));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 5));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q6_k_to_fp32_packed8(as_ushort2(regQL.s3), (ushort)(regQH.s1 >> 16), scale1);

        shared_y4 = read_imagef(src1, (y_offset + 6));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 7));
        acc += shared_y4 * fp32x8.hi;

        sum += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
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

    // 1 output per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01 + i20 * ne01] = sum;
    }
}
