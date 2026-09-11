#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK1_0 128
#define N_SIMDGROUP 4

#define dequantizeBlockAccum_q1(total, bits, scale, regB, lb)                                       \
    total += (2.0f*(float)((bits >>  0) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+0); \
    total += (2.0f*(float)((bits >>  1) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+0); \
    total += (2.0f*(float)((bits >>  2) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+0); \
    total += (2.0f*(float)((bits >>  3) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+0); \
    total += (2.0f*(float)((bits >>  4) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+0); \
    total += (2.0f*(float)((bits >>  5) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+0); \
    total += (2.0f*(float)((bits >>  6) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+0); \
    total += (2.0f*(float)((bits >>  7) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+0); \
    total += (2.0f*(float)((bits >>  8) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+1); \
    total += (2.0f*(float)((bits >>  9) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+1); \
    total += (2.0f*(float)((bits >> 10) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+1); \
    total += (2.0f*(float)((bits >> 11) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+1); \
    total += (2.0f*(float)((bits >> 12) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+1); \
    total += (2.0f*(float)((bits >> 13) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+1); \
    total += (2.0f*(float)((bits >> 14) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+1); \
    total += (2.0f*(float)((bits >> 15) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+1); \
    total += (2.0f*(float)((bits >> 16) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+2); \
    total += (2.0f*(float)((bits >> 17) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+2); \
    total += (2.0f*(float)((bits >> 18) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+2); \
    total += (2.0f*(float)((bits >> 19) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+2); \
    total += (2.0f*(float)((bits >> 20) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+2); \
    total += (2.0f*(float)((bits >> 21) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+2); \
    total += (2.0f*(float)((bits >> 22) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+2); \
    total += (2.0f*(float)((bits >> 23) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+2); \
    total += (2.0f*(float)((bits >> 24) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+3); \
    total += (2.0f*(float)((bits >> 25) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+3); \
    total += (2.0f*(float)((bits >> 26) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+3); \
    total += (2.0f*(float)((bits >> 27) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+3); \
    total += (2.0f*(float)((bits >> 28) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+3); \
    total += (2.0f*(float)((bits >> 29) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+3); \
    total += (2.0f*(float)((bits >> 30) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+3); \
    total += (2.0f*(float)((bits >> 31) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+3);


#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q1_0_f32(
        read_only  image1d_buffer_t src0_q,
        global half  * src0_d,
        read_only  image1d_buffer_t src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M;
    uint BLOCK_STRIDE_A = 4 * M;

    uint4  regA;
    half   regS;
    float8 regB;

    float totalSum = 0.0f;

    #pragma unroll 1
    for (uint kb = groupId; kb < (K / QK1_0); kb += N_SIMDGROUP) {
        regS = src0_d[gid + kb * LINE_STRIDE_A]; // each fiber loads its row's scale

        // first 16 fibers load 8 B values each -> 128 activations for this block
        if (slid < 16) {
            regB.s0123 = read_imagef(src1, (slid * 2 + kb * 32));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + kb * 32));
        }

        // load this row's 4 uint32 (128 sign bits)
        regA.s0 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;

        float scale = (float)regS;
        dequantizeBlockAccum_q1(totalSum, regA.s0, scale, regB, 0);
        dequantizeBlockAccum_q1(totalSum, regA.s1, scale, regB, 4);
        dequantizeBlockAccum_q1(totalSum, regA.s2, scale, regB, 8);
        dequantizeBlockAccum_q1(totalSum, regA.s3, scale, regB, 12);
    }

    // reduction in local memory, assumes #wave = N_SIMDGROUP = 4
    local float reduceLM[SIMDGROUP_WIDTH * 3];
    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // Guard the output row. The x-grid is padded to CEIL_DIV(M,wavesize)*wavesize,
        // so when ne01 is not a multiple of the wave size the tail work-items run past
        // row ne01 and would overrun dst into the adjacent tensor. No-op / byte-identical
        // when ne01 is wave-aligned (no padding).
        if (gid < M) dst[gid] = totalSum;
    }
}
