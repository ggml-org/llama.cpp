#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_K 256
#define K_SCALE_SIZE 12
#define N_SIMDGROUP 8
#define SIMDGROUP_WIDTH 64

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uint stride,
    uchar * d,
    uchar * m
) {
    if (j < 4) {
        *d = q[j*stride]     & 63;
        *m = q[(j+4)*stride] & 63;
    } else {
        *d = (q[(j+4)*stride] & 0x0F) | ((q[(j-4)*stride] & 0xC0) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & 0x0F) | ((q[j*stride] & 0xC0) >> 2);
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
__kernel void gemv_noshuffle_q4_k_f32_32b_trans(
    read_only image1d_buffer_t src0_q,
    __global half *         src0_d,
    __global half *         src0_dm,
    __global uchar *        src0_s,
    __read_only image1d_buffer_t src1,
    __global float *        dst,
    ulong                   offsetd,
    int                     ne00,
    int                     ne01
) {
    uint i01  = get_global_id(0);
    uint sgid = get_local_id(1);
    uint slid = get_sub_group_local_id();

    int num_subblocks   = ne00 / 32;

    __private float sum = 0.0f;

    // Loop over sub-blocks of 32 elements, N_SIMDGROUP sub-blocks per iter
    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
        uint sb = ib / 8;
        uint j  = ib % 8;

        // Load d and dmin for this super-block
        half d_val  = src0_d[sb * ne01 + i01];
        half dm_val = src0_dm[sb * ne01 + i01];

        // Load sub-block scale and min. s is transposed [nb][12][M]; stride ne01 per code.
        global const uchar * sc = src0_s + sb * K_SCALE_SIZE * ne01 + i01;
        uchar sv, mn;
        get_scale_min_k4(j, sc, ne01, &sv, &mn);

        float scale = (float)d_val * (float)sv;
        float minv  = (float)dm_val * (float)mn;

        // Load 4 uints of quants (32 nibbles = 32 elements), column-major stride ne01
        uint q_base = ib * ne01 * 4 + i01;

        uint4 regQ;
        regQ.s0 = read_imageui(src0_q, q_base).x;
        regQ.s1 = read_imageui(src0_q, q_base + ne01).x;
        regQ.s2 = read_imageui(src0_q, q_base + ne01 * 2).x;
        regQ.s3 = read_imageui(src0_q, q_base + ne01 * 3).x;

        // Load activations: 32 floats = 8 float4s
        uint y_offset = ib * 8;

        float4 y_local = (slid < 8) ? read_imagef(src1, (y_offset + slid)) : (float4)0.0f;
        float4 y0 = sub_group_broadcast(y_local, 0);
        float4 y1 = sub_group_broadcast(y_local, 1);
        float4 y2 = sub_group_broadcast(y_local, 2);
        float4 y3 = sub_group_broadcast(y_local, 3);
        float4 y4 = sub_group_broadcast(y_local, 4);
        float4 y5 = sub_group_broadcast(y_local, 5);
        float4 y6 = sub_group_broadcast(y_local, 6);
        float4 y7 = sub_group_broadcast(y_local, 7);

        float8 fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s0), scale, minv);
        float4 acc = y0 * fp32x8.lo;
        acc += y1 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s1), scale, minv);
        acc += y2 * fp32x8.lo;
        acc += y3 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s2), scale, minv);
        acc += y4 * fp32x8.lo;
        acc += y5 * fp32x8.hi;

        fp32x8 = q4_k_to_fp32_packed8(as_ushort2(regQ.s3), scale, minv);
        acc += y6 * fp32x8.lo;
        acc += y7 * fp32x8.hi;

        sum += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
    }

    // reduction in local memory over N_SIMDGROUP subgroups
    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (sgid > 0) {
        reduceLM[SIMDGROUP_WIDTH * (sgid - 1) + slid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid == 0) {
        for (uint i = 0; i < N_SIMDGROUP - 1; ++i) {
            sum += reduceLM[SIMDGROUP_WIDTH * i + slid];
        }
    }

    // 1 output per thread in subgroup 0
    if (sgid == 0) {
        dst = dst + (offsetd >> 2);
        dst[i01] = sum;
    }
}
