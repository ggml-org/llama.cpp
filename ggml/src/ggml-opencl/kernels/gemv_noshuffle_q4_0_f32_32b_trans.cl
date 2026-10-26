#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK4_0 32
#define N_SIMDGROUP 4

#define dequantizeBlockAccum_ila_1row_hi(total_sum, bits4, scale, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sum += ((bits4.s0 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sum += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sum += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sum += (((bits4.s0 & 0xF000) >> 12) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sum += ((bits4.s1 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sum += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sum += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sum += (((bits4.s1 & 0xF000) >> 12) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sum += ((bits4.s2 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sum += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sum += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sum += (((bits4.s2 & 0xF000) >> 12) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sum += ((bits4.s3 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sum += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sum += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sum += (((bits4.s3 & 0xF000) >> 12) - 8) * scale * shared_y;

#define dequantizeBlockAccum_ila_1row_lo(total_sum, bits4, scale, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sum += ((bits4.s4 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sum += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sum += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sum += (((bits4.s4 & 0xF000) >> 12) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sum += ((bits4.s5 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sum += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sum += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sum += (((bits4.s5 & 0xF000) >> 12) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sum += ((bits4.s6 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sum += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sum += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sum += (((bits4.s6 & 0xF000) >> 12) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sum += ((bits4.s7 & 0x000F) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sum += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sum += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sum += (((bits4.s7 & 0xF000) >> 12) - 8) * scale * shared_y;


#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0_f32_32b_trans(
        __read_only  image1d_buffer_t src0_q,
        global half  * src0_d,
        __read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    __private uint4  regA;
    __private half   regS;
    __private float8 regB;
    __private float  totalSum = 0.0f;

    for (uint k = groupId; k < (K / QK4_0); k += N_SIMDGROUP) {
        regS = src0_d[k * M + gid];
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }
        regA.s0 = read_imageui(src0_q, ((k * 4 + 0) * M + gid)).x;
        regA.s1 = read_imageui(src0_q, ((k * 4 + 1) * M + gid)).x;
        regA.s2 = read_imageui(src0_q, ((k * 4 + 2) * M + gid)).x;
        regA.s3 = read_imageui(src0_q, ((k * 4 + 3) * M + gid)).x;

        dequantizeBlockAccum_ila_1row_hi(totalSum, as_ushort8(regA), regS, regB);
        dequantizeBlockAccum_ila_1row_lo(totalSum, as_ushort8(regA), regS, regB);
    }

    __local float reduceLM[SIMDGROUP_WIDTH * 3];
    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        if (gid < M) {
            dst[gid] = totalSum;
        }
    }
}
