#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_K  256
#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

// scales are transposed: consecutive codes of a row are `stride` apart
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uint stride,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j*stride]     & mask_d6;
        *m = q[(j+4)*stride] & mask_d6;
    } else {
        *d = (q[(j+4)*stride] & mask_d4) | ((q[(j-4)*stride] & mask_hi2) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & mask_d4) | ((q[j*stride] & mask_hi2) >> 2);
    }
}

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, minv, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += ((bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += ((bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, minv, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += ((bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += ((bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, minv, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, minv, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    // K-split factor = #subgroups in the WG. Read from the launch (NOT a compile
    // constant) so small-M projections (Kcur/Vcur/Qcur) can dispatch a wider
    // K-split (more waves/SP -> latency hiding) while large-M keeps 4. The
    // physical weight layout stride below is INDEPENDENT of this (see BLOCK_STRIDE_A).
    uint nsg = get_local_size(1);

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    // Physical per-K-block stride in the packed image: 8 uints/block-row-pair *
    // (M/2) row-pairs = 4*M uints. This is a layout constant, not tied to nsg.
    uint BLOCK_STRIDE_A = 4 * M;
    uint scales_per_row = (K / QK_K) * 12;

    // The x-grid is padded to CEIL_DIV(ne01/2,64)*64, so when ne01 % 128 != 0 the
    // tail lanes hold gid >= ne01/2. The output stores below are guarded, but the
    // input fetches are not: src0_d and src0_m are raw global half2 pointers,
    // src0_s is a raw global uchar pointer, and read_imageui on an
    // image1d_buffer_t is UNDEFINED out of range -- an image clamps only for
    // SAMPLER reads, which these are not. Those lanes therefore read past the end
    // of all three allocations. For a [2816, 2112] weight (2112 % 128 == 64) the
    // top tail lane is gid = 1087 while only gid < 1056 is backed, and it runs
    // 32 half2 past src0_d/src0_m, 31 uints past the quant image, and 63 bytes
    // past src0_s.
    //
    // Clamp the row used for every fetch. The lanes stay ACTIVE, which the
    // sub_group_broadcast in the dequant macros requires, and their results are
    // still discarded by the existing output guard. No-op and byte-identical
    // whenever ne01 % 128 == 0.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);

    private uint4     regA;
    private half2     regS;
    private half2     regM;
    private float8    regB;

    private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / 32); k += nsg) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid_s + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid_s + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid_s;
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        // load half weights for two blocks in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // Cross-subgroup reduction in local memory. Generalized to nsg subgroups
    // (was a hard-coded 4-wave unroll). Sized for up to 16 subgroups (the widest
    // K-split we dispatch for small M). At nsg==4 the accumulation order is
    // identical to the original unroll -> byte-identical for the large-M path.
    local float2 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = totalSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            totalSum += reduceLM[SUBGROUP_SIZE * i + slid];
        }
    }

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // Guard the two output rows. The x-grid is padded to CEIL_DIV(ne01/2,64)*64,
        // so when ne01 is not a multiple of 128 the tail row-pairs run past row ne01
        // and would overrun dst into the adjacent tensor. No-op / byte-identical when
        // ne01 % 128 == 0 (M/2 already a multiple of 64 -> no padding).
        if (gid * 2 + 0 < M) dst[gid * 2 + 0] = totalSum.s0;
        if (gid * 2 + 1 < M) dst[gid * 2 + 1] = totalSum.s1;
    }

}

// --- Fused gate+up GEMV + GLU epilogue (FFN) ------------------------------------
// Folds the FFN's two decode GEMVs (ffn_gate, ffn_up) and the following GLU into a
// SINGLE dispatch: {MUL_MAT(Wg,x), MUL_MAT(Wu,x), GLU}. Both matmuls share the same
// activation x (ffn_norm), so the activation image read is issued ONCE per K-block
// and reused for the gate and up dot products (the per-op path re-reads it twice and
// also materializes the two full ffn-wide intermediates to global, which the GLU
// then re-reads). The gate/up partial sums are accumulated in the SAME per-fiber
// order and reduced in the SAME cross-subgroup order as the standalone GEMV, and the
// GLU formula is the exact scalar expression from kernels/glu.cl, so the output is
// BYTE-IDENTICAL to the per-op matmul+matmul+glu path -> safe to default on.
//   glu_op: REGLU=0, GEGLU=1, SWIGLU=2, GEGLU_ERF=4, GEGLU_QUICK=5 (ggml_glu_op).
// Weights: src0g_* = gate (= GLU src[0]); src0u_* = up (= GLU src[1]).
#define GLU_GEGLU_COEF_A      0.044715f
#define GLU_SQRT_2_OVER_PI    0.79788456080286535587989211986876f
#define GLU_SQRT_2_INV        0.70710678118654752440084436210484f
#define GLU_QUICK_COEF       -1.702f

inline float glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(GLU_SQRT_2_OVER_PI*g*(1.0f + GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK (glu_op == 5)
        act = g*(1.0f/(1.0f + exp(GLU_QUICK_COEF*g)));
    }
    return act*u;
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_glu(
        read_only  image1d_buffer_t src0g_q,
        global half2  * src0g_d,
        global half2  * src0g_m,
        global uchar  * src0g_s,
        read_only  image1d_buffer_t src0u_q,
        global half2  * src0u_d,
        global half2  * src0u_m,
        global uchar  * src0u_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int glu_op,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;

    private float2 gateSum = (float2)(0.0f);
    private float2 upSum   = (float2)(0.0f);

    // Two SEQUENTIAL K-loops (gate fully, then up). Keeping only one weight's
    // working set live at a time holds the kernel's register footprint at ~the
    // base single-weight GEMV's, so its max WG stays 1024 (16 subgroups) and the
    // per-subgroup K-split matches the standalone wide GEMV exactly -> the gate
    // and up partial sums are BYTE-IDENTICAL to the per-op path. The macro body
    // is the base kernel's inner loop verbatim, parameterized by weight source.
#define Q4K_GLU_LOOP(SUM, Q, DD, MM, SS)                                                       \
    for (uint k = groupId; k < (K / 32); k += nsg) {                                           \
        uint sb = k / 8;                                                                       \
        uint j  = k % 8;                                                                       \
        half2 d   = DD[gid + sb * LINE_STRIDE_A];                                              \
        half2 dm  = MM[gid + sb * LINE_STRIDE_A];                                              \
        global const uchar * sc0 = SS + sb * 12 * M + 2 * gid;                                 \
        global const uchar * sc1 = sc0 + 1;                                                    \
        uchar sv0, mn0, sv1, mn1;                                                              \
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);                      \
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);                      \
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));         \
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));         \
        if (slid < 4) {                                                                        \
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));                                \
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));                            \
        }                                                                                      \
        regA.s0 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;           \
        regA.s1 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;           \
        regA.s2 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;           \
        regA.s3 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;           \
        DEQ_HI(SUM, as_ushort8(regA), regS, regM, regB);                                       \
        regA.s0 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;           \
        regA.s1 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;           \
        regA.s2 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;           \
        regA.s3 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;           \
        DEQ_LO(SUM, as_ushort8(regA), regS, regM, regB);                                       \
    }

#ifdef VECTOR_SUB_GROUP_BROADCAST
#define DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif

    Q4K_GLU_LOOP(gateSum, src0g_q, src0g_d, src0g_m, src0g_s)
    Q4K_GLU_LOOP(upSum,   src0u_q, src0u_d, src0u_m, src0u_s)

#undef DEQ_HI
#undef DEQ_LO
#undef Q4K_GLU_LOOP

    // Cross-subgroup reduction in local memory. Packs gate (xy) + up (zw) into a
    // float4 so both reduce in one pass; summation order matches the base GEMV's
    // per-channel loop -> byte-identical partial sums.
    local float4 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = (float4)(gateSum, upSum);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            float4 p = reduceLM[SUBGROUP_SIZE * i + slid];
            gateSum += p.xy;
            upSum   += p.zw;
        }
        dst = (global float*)((global char*)dst + offsetd);
        dst[gid * 2 + 0] = glu_apply(glu_op, gateSum.s0, upSum.s0);
        dst[gid * 2 + 1] = glu_apply(glu_op, gateSum.s1, upSum.s1);
    }
}

// --- Split-K-across-workgroups decode GEMV (small-M projections) ----------------
// A single-token GEMV makes only ceil(M/2/64) workgroups; a WG runs on one Adreno
// compute unit, so for small M (Kcur/Vcur, M=512 -> 4 WGs) most of the 16 CUs sit
// idle and the matmul is bandwidth-starved even with a wide intra-WG K-split. This
// variant adds a SECOND grid dimension of `ksplit` workgroups that each reduce a
// disjoint slice of K and write a per-slice partial; kernel_gemv_splitk_reduce_f32
// then sums the partials into dst. Identical math/layout to the base kernel
// (physical block stride 4*M, get_scale_min_k4) -> coherent. Gated host-side to
// M<=1024 (M>=2048
// already fills the CUs and the extra reduce dispatch only hurts).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_splitk(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * partial,          // [ksplit * M], slice-major
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);
    uint ksplit  = get_num_groups(1);
    uint kslice  = get_group_id(1);

    uint K = ne00;
    uint M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;      // physical, independent of the K-split

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;
    private float2 totalSum = (float2)(0.0f);

    // each (kslice, subgroup) pair owns a disjoint set of K-blocks
    for (uint k = kslice * nsg + groupId; k < (K / 32); k += ksplit * nsg) {
        uint sb = k / 8;
        uint j  = k % 8;
        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];
        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid;
        global const uchar * sc1 = sc0 + 1;
        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
    }

    local float2 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            totalSum += reduceLM[SUBGROUP_SIZE * i + slid];
        }
        vstore2(totalSum, 0, &(partial[kslice * M + gid * 2]));
    }
}

// Sum the per-slice partials [ksplit * M] into dst[M]; applies the dst byte offset.
kernel void kernel_gemv_splitk_reduce_f32(
        global float * partial,
        global float * dst,
        ulong offsetd,
        int   ne01,         // M
        int   ksplit)
{
    uint r = get_global_id(0);
    if (r >= (uint)ne01) return;
    float acc = 0.0f;
    for (uint s = 0; s < (uint)ksplit; ++s) {
        acc += partial[s * (uint)ne01 + r];
    }
    dst = (global float*)((global char*)dst + offsetd);
    dst[r] = acc;
}


// --- Dequant-once macros for the mc3 verify GEMV (Q4K_MC3_DEQUANT_ONCE) ---
// The inline dequantizeBlockAccum_* macros recompute the dequantized weight
// ((code & mask)>>shift)*scale - minv ONCE PER COLUMN (3x), and the flat
// 32-FMA unroll spills ~430 B of temporaries. These macros split the work:
// DEQUANT_Q4K_BLOCK computes the 16 weights/row of one 32-block ONCE into a
// half2[] (row0 in .s0, row1 in .s1) — stored as half, the exact type the
// inline expression yields (int*half-half), so no extra rounding. MAC_Q4K_BLOCK
// then accumulates them against a column's broadcast activation in the SAME
// per-accumulator order as the inline macro. Each weight value and each
// accumulator's add-chain is bit-for-bit identical => byte-identical output,
// while the dequant ALU drops 3x->1x and the live set shrinks. Requires the
// Qualcomm vector sub_group_broadcast (float8); enabled opt-in on Adreno.
#define DEQ_Q4K_HALF2(b0, b1, msk, sh, scale, minv) \
    (half2)( ((b0 & msk) >> sh) * scale.s0 - minv.s0, \
             ((b1 & msk) >> sh) * scale.s1 - minv.s1 )

#define DEQUANT_Q4K_BLOCK(wq, bits, scale, minv) \
    wq[0]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0x000F, 0,  scale, minv); \
    wq[1]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0x00F0, 4,  scale, minv); \
    wq[2]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0x0F00, 8,  scale, minv); \
    wq[3]  = DEQ_Q4K_HALF2(bits.s0, bits.s1, 0xF000, 12, scale, minv); \
    wq[4]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0x000F, 0,  scale, minv); \
    wq[5]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0x00F0, 4,  scale, minv); \
    wq[6]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0x0F00, 8,  scale, minv); \
    wq[7]  = DEQ_Q4K_HALF2(bits.s2, bits.s3, 0xF000, 12, scale, minv); \
    wq[8]  = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0x000F, 0,  scale, minv); \
    wq[9]  = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0x00F0, 4,  scale, minv); \
    wq[10] = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0x0F00, 8,  scale, minv); \
    wq[11] = DEQ_Q4K_HALF2(bits.s4, bits.s5, 0xF000, 12, scale, minv); \
    wq[12] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0x000F, 0,  scale, minv); \
    wq[13] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0x00F0, 4,  scale, minv); \
    wq[14] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0x0F00, 8,  scale, minv); \
    wq[15] = DEQ_Q4K_HALF2(bits.s6, bits.s7, 0xF000, 12, scale, minv);

// ln0/ln1 = the two source lanes whose activation float8 this block consumes
// (0,1 for the hi block, 2,3 for the lo block — matching the inline _hi/_lo).
#define MAC_Q4K_BLOCK(ts, wq, y, ln0, ln1) { \
    float8 sy = sub_group_broadcast(y, ln0); \
    ts.s0 += wq[0].s0*sy.s0; ts.s0 += wq[1].s0*sy.s1; ts.s0 += wq[2].s0*sy.s2; ts.s0 += wq[3].s0*sy.s3; \
    ts.s0 += wq[4].s0*sy.s4; ts.s0 += wq[5].s0*sy.s5; ts.s0 += wq[6].s0*sy.s6; ts.s0 += wq[7].s0*sy.s7; \
    ts.s1 += wq[0].s1*sy.s0; ts.s1 += wq[1].s1*sy.s1; ts.s1 += wq[2].s1*sy.s2; ts.s1 += wq[3].s1*sy.s3; \
    ts.s1 += wq[4].s1*sy.s4; ts.s1 += wq[5].s1*sy.s5; ts.s1 += wq[6].s1*sy.s6; ts.s1 += wq[7].s1*sy.s7; \
    sy = sub_group_broadcast(y, ln1); \
    ts.s0 += wq[8].s0*sy.s0;  ts.s0 += wq[9].s0*sy.s1;  ts.s0 += wq[10].s0*sy.s2; ts.s0 += wq[11].s0*sy.s3; \
    ts.s0 += wq[12].s0*sy.s4; ts.s0 += wq[13].s0*sy.s5; ts.s0 += wq[14].s0*sy.s6; ts.s0 += wq[15].s0*sy.s7; \
    ts.s1 += wq[8].s1*sy.s0;  ts.s1 += wq[9].s1*sy.s1;  ts.s1 += wq[10].s1*sy.s2; ts.s1 += wq[11].s1*sy.s3; \
    ts.s1 += wq[12].s1*sy.s4; ts.s1 += wq[13].s1*sy.s5; ts.s1 += wq[14].s1*sy.s6; ts.s1 += wq[15].s1*sy.s7; \
}

// Multi-column (N=3) variant of the q4_K decode GEMV, for the speculative /
// MTP verify batch (ne1=3 = 2 drafts + 1 bonus). Stays on the efficient GEMV
// path (subgroup-broadcast activation, NSUBGROUPS K-split) instead of the
// transposed-GEMM dead-zone path. Each K-block's weights (regA_hi/regA_lo) are
// loaded ONCE and reused across all 3 activation columns — same weight traffic
// as one decode, ~3x the (cheap) dequant ALU. Per-column accumulation is
// independent and identical to 3 standalone GEMVs => byte-identical, so it does
// NOT perturb the lm_head logits / spec accept rate.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_mc3(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = NSUBGROUPS * M;
    uint COL_STRIDE     = K / 4;   // float4 pixels per activation column

    private uint4  regA_hi, regA_lo;
    private half2  regS, regM;
    private float8 regB;

    private float2 ts0 = (float2)(0.0f);
    private float2 ts1 = (float2)(0.0f);
    private float2 ts2 = (float2)(0.0f);

#ifdef Q4K_MC3_DEQUANT_LDS
    // One 16-half2 block buffer per WI (reused hi->lo): forces the dequantized
    // weights into LDS instead of private arrays (which spill to slow global on
    // Adreno). 64*NSUBGROUPS WIs * 16 half2 = 16 KB; each WI owns its own slot
    // range (flat*16) -> no cross-lane sharing, no barrier needed.
    local half2 wstage[SUBGROUP_SIZE * NSUBGROUPS * 16];
    local half2 * ws = wstage + (groupId * SUBGROUP_SIZE + slid) * 16;
#endif

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid;
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        // weights loaded ONCE, reused across the 3 columns
        regA_hi.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA_hi.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA_hi.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA_hi.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA_lo.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA_lo.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA_lo.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA_lo.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

#ifdef Q4K_MC3_DEQUANT_ONCE
        // Dequant the 32 weights/row (16 hi + 16 lo) ONCE into half2[] (byte-
        // identical to the inline intermediate), then MAC against each column's
        // activation. Drops the dequant ALU 3x->1x and the macro-temp spill.
        half2 wq_hi[16], wq_lo[16];
        DEQUANT_Q4K_BLOCK(wq_hi, as_ushort8(regA_hi), regS, regM);
        DEQUANT_Q4K_BLOCK(wq_lo, as_ushort8(regA_lo), regS, regM);
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts0, wq_hi, regB, 0, 1); MAC_Q4K_BLOCK(ts0, wq_lo, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts1, wq_hi, regB, 0, 1); MAC_Q4K_BLOCK(ts1, wq_lo, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts2, wq_hi, regB, 0, 1); MAC_Q4K_BLOCK(ts2, wq_lo, regB, 2, 3); }
#elif defined(Q4K_MC3_DEQUANT_LDS)
        // LDS-staged dequant: dequant a 32-block ONCE into the per-WI LDS slot
        // (hi pass then lo pass, overwriting), MAC each column from LDS. ts*
        // receive hi-then-lo in the same order as DEQUANT_ONCE -> byte-identical.
        // Activations reloaded per pass (cheap, imaged); only one regB + 0 weight
        // regs live -> the weight working set lives in LDS, not spilled private.
        DEQUANT_Q4K_BLOCK(ws, as_ushort8(regA_hi), regS, regM);
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts0, ws, regB, 0, 1); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts1, ws, regB, 0, 1); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts2, ws, regB, 0, 1); }
        DEQUANT_Q4K_BLOCK(ws, as_ushort8(regA_lo), regS, regM);
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts0, ws, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts1, ws, regB, 2, 3); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          MAC_Q4K_BLOCK(ts2, ws, regB, 2, 3); }
#else
        // Per-column: load only this column's activation (single regB live at a
        // time -> 1/3 the activation register pressure vs holding all 3) then
        // dequant against the shared weights. Cuts the private-mem spill.
#ifdef VECTOR_SUB_GROUP_BROADCAST
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_8_hi(ts0, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_8_lo(ts0, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_8_hi(ts1, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_8_lo(ts1, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_8_hi(ts2, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_8_lo(ts2, as_ushort8(regA_lo), regS, regM, regB); }
#else
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 0*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 0*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_1_hi(ts0, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_1_lo(ts0, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 1*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 1*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_1_hi(ts1, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_1_lo(ts1, as_ushort8(regA_lo), regS, regM, regB); }
        { if (slid < 4) { regB.s0123 = read_imagef(src1, 2*COL_STRIDE     + slid*2 + k*8);
                          regB.s4567 = read_imagef(src1, 2*COL_STRIDE + 1 + slid*2 + k*8); }
          dequantizeBlockAccum_ns_sgbroadcast_1_hi(ts2, as_ushort8(regA_hi), regS, regM, regB);
          dequantizeBlockAccum_ns_sgbroadcast_1_lo(ts2, as_ushort8(regA_lo), regS, regM, regB); }
#endif
#endif // Q4K_MC3_DEQUANT_ONCE
    }

    // cross-subgroup reduce: pack the 3 columns' float2 into a float8 (6 used).
    local float8 reduceLM[SUBGROUP_SIZE * 3];
    float8 acc = (float8)(ts0.s0, ts0.s1, ts1.s0, ts1.s1, ts2.s0, ts2.s1, 0.0f, 0.0f);
    if (groupId == 1) { reduceLM[SUBGROUP_SIZE * 0 + slid] = acc; }
    if (groupId == 2) { reduceLM[SUBGROUP_SIZE * 1 + slid] = acc; }
    if (groupId == 3) { reduceLM[SUBGROUP_SIZE * 2 + slid] = acc; }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        acc += reduceLM[SUBGROUP_SIZE * 0 + slid];
        acc += reduceLM[SUBGROUP_SIZE * 1 + slid];
        acc += reduceLM[SUBGROUP_SIZE * 2 + slid];
        dst = (global float*)((global char*)dst + offsetd);
        // dst is column-major [M rows x 3 cols]: (row, col) at col*M + row
        vstore2((float2)(acc.s0, acc.s1), 0, &(dst[0 * M + gid * 2]));
        vstore2((float2)(acc.s2, acc.s3), 0, &(dst[1 * M + gid * 2]));
        vstore2((float2)(acc.s4, acc.s5), 0, &(dst[2 * M + gid * 2]));
    }
}
