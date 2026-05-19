#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define K_SCALE_SIZE 12
#define N_SIMDGROUP 4
#define SIMDGROUP_WIDTH 64

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m
) {
    if (j < 4) {
        *d = q[j]   & 63;
        *m = q[j+4] & 63;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] & 0xC0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0F) | ((q[j]   & 0xC0) >> 2);
    }
}

static inline float8 q4_k_to_fp32_packed8(ushort2 q4x8, float scale, float minv) {
    float8 fp32x8;
    fp32x8.s0 = (q4x8.s0 & 0x000F) * scale - minv;
    fp32x8.s1 = ((q4x8.s0 & 0x00F0) >> 4) * scale - minv;
    fp32x8.s2 = ((q4x8.s0 & 0x0F00) >> 8) * scale - minv;
    fp32x8.s3 = ((q4x8.s0 & 0xF000) >> 12) * scale - minv;
    fp32x8.s4 = (q4x8.s1 & 0x000F) * scale - minv;
    fp32x8.s5 = ((q4x8.s1 & 0x00F0) >> 4) * scale - minv;
    fp32x8.s6 = ((q4x8.s1 & 0x0F00) >> 8) * scale - minv;
    fp32x8.s7 = ((q4x8.s1 & 0xF000) >> 12) * scale - minv;
    return fp32x8;
}

__attribute__((qcom_reqd_sub_group_size("half")))
__kernel void kernel_gemv_moe_q4_k_f32_ns(
    __global uint *         src0_q,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
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

    uint i11 = i20 % ne11;

    uint expert_id = src2[i20];

    int num_superblocks = ne00 / QK_K;
    int num_subblocks = ne00 / 32;
    int scales_per_row = num_superblocks * K_SCALE_SIZE;

    // Expert offsets in the transposed noshuffle layout
    uint expert_q_offset = expert_id * (ne00 / 8) * ne01;
    uint expert_d_offset = expert_id * num_superblocks * ne01;

    __private float sum = 0.0f;

    // Loop over sub-blocks of 32 elements, N_SIMDGROUP sub-blocks per iter
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
        uint sb = ib / 8;
        uint j  = ib % 8;

        // Load d and dmin for this super-block
        half d_val   = src0_d[expert_d_offset + sb * ne01 + i01];
        half dm_val  = src0_dm[expert_d_offset + sb * ne01 + i01];

        // Load sub-block scale and min
        global const uchar * sc = src0_s + (expert_id * ne01 + i01) * scales_per_row + sb * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(j, sc, &sv, &mn);

        float scale = (float)d_val * (float)sv;
        float minv  = (float)dm_val * (float)mn;

        // Load 4 uints of quants (32 nibbles = 32 elements)
        uint q_base = expert_q_offset + ib * ne01 * 4 + i01;

        uint4 regQ;
        regQ.s0 = src0_q[q_base];
        regQ.s1 = src0_q[q_base + ne01];
        regQ.s2 = src0_q[q_base + ne01 * 2];
        regQ.s3 = src0_q[q_base + ne01 * 3];

        // Load activations: 32 floats = 8 float4s
        uint y_offset = i11 * ne00 / 4 + ib * 8;

        float8 fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s0), scale, minv);

        float4 shared_y4;
        shared_y4 = read_imagef(src1, (y_offset + 0));
        float4 acc = shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 1));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s1), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 2));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 3));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s2), scale, minv);

        shared_y4 = read_imagef(src1, (y_offset + 4));
        acc += shared_y4 * fp32x8.lo;

        shared_y4 = read_imagef(src1, (y_offset + 5));
        acc += shared_y4 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s3), scale, minv);

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
