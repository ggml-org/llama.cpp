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

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, bits1, scale, minv, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += (((bits4.s0 & 0x000F) | ((bits1.s0 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x000F) | ((bits1.s1 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4) | (((bits1.s0 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4) | (((bits1.s1 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8) | (((bits1.s0 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8) | (((bits1.s1 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s0 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s1 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += (((bits4.s2 & 0x000F) | (((bits1.s0 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x000F) | (((bits1.s1 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4) | (((bits1.s0 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4) | (((bits1.s1 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8) | (((bits1.s0 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8) | (((bits1.s1 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s0 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s1 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += (((bits4.s4 & 0x000F) | ((bits1.s2 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x000F) | ((bits1.s3 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4) | (((bits1.s2 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4) | (((bits1.s3 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8) | (((bits1.s2 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8) | (((bits1.s3 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s2 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s3 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += (((bits4.s6 & 0x000F) | (((bits1.s2 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x000F) | (((bits1.s3 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4) | (((bits1.s2 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4) | (((bits1.s3 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8) | (((bits1.s2 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8) | (((bits1.s3 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s2 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s3 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, bits1, scale, minv, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += (((bits4.s0 & 0x000F) | ((bits1.s4 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x000F) | ((bits1.s5 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4) | (((bits1.s4 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4) | (((bits1.s5 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8) | (((bits1.s4 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8) | (((bits1.s5 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s4 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s5 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += (((bits4.s2 & 0x000F) | (((bits1.s4 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x000F) | (((bits1.s5 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4) | (((bits1.s4 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4) | (((bits1.s5 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8) | (((bits1.s4 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8) | (((bits1.s5 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s4 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s5 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += (((bits4.s4 & 0x000F) | ((bits1.s6 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x000F) | ((bits1.s7 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4) | (((bits1.s6 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4) | (((bits1.s7 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8) | (((bits1.s6 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8) | (((bits1.s7 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s6 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s7 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += (((bits4.s6 & 0x000F) | (((bits1.s6 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x000F) | (((bits1.s7 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4) | (((bits1.s6 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4) | (((bits1.s7 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8) | (((bits1.s6 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8) | (((bits1.s7 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s6 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s7 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, bits1, scale, minv, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += (((bits4.s0 & 0x000F)         | ((bits1.s0 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4)  | (((bits1.s0 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8)  | (((bits1.s0 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s0 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s2 & 0x000F)         | (((bits1.s0 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4)  | (((bits1.s0 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8)  | (((bits1.s0 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s0 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s1 & 0x000F)         | ((bits1.s1 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4)  | (((bits1.s1 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8)  | (((bits1.s1 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s1 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s3 & 0x000F)         | (((bits1.s1 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4)  | (((bits1.s1 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8)  | (((bits1.s1 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s1 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += (((bits4.s4 & 0x000F)         | ((bits1.s2 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4)  | (((bits1.s2 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8)  | (((bits1.s2 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s2 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s6 & 0x000F)         | (((bits1.s2 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4)  | (((bits1.s2 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8)  | (((bits1.s2 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s2 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s5 & 0x000F)         | ((bits1.s3 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4)  | (((bits1.s3 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8)  | (((bits1.s3 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s3 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s7 & 0x000F)         | (((bits1.s3 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4)  | (((bits1.s3 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8)  | (((bits1.s3 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s3 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, bits1, scale, minv, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += (((bits4.s0 & 0x000F)         | ((bits1.s4 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4)  | (((bits1.s4 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8)  | (((bits1.s4 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s4 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s2 & 0x000F)         | (((bits1.s4 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4)  | (((bits1.s4 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8)  | (((bits1.s4 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s4 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s1 & 0x000F)         | ((bits1.s5 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4)  | (((bits1.s5 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8)  | (((bits1.s5 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s5 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s3 & 0x000F)         | (((bits1.s5 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4)  | (((bits1.s5 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8)  | (((bits1.s5 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s5 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += (((bits4.s4 & 0x000F)         | ((bits1.s6 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4)  | (((bits1.s6 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8)  | (((bits1.s6 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s6 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s6 & 0x000F)         | (((bits1.s6 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4)  | (((bits1.s6 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8)  | (((bits1.s6 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s6 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s5 & 0x000F)         | ((bits1.s7 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4)  | (((bits1.s7 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8)  | (((bits1.s7 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s7 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s7 & 0x000F)         | (((bits1.s7 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4)  | (((bits1.s7 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8)  | (((bits1.s7 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s7 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q5_k_f32(
        read_only  image1d_buffer_t src0_q,
        read_only  image1d_buffer_t src0_qh,
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

    uint LINE_STRIDE_A     = M / 2;
    uint BLOCK_STRIDE_A    = NSUBGROUPS * M;

    uint LINE_STRIDE_A_QH  = M / 2;
    uint BLOCK_STRIDE_A_QH = NSUBGROUPS * M / 2;
    uint scales_per_row    = (K / QK_K) * 12;

    private uint4     regA;
    private ushort4   regH;
    private half2     regS;
    private half2     regM;
    private float8    regB;

    private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + 2 * gid * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid + 1) * scales_per_row + sb * 12;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        regH.s0 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 0)).x);
        regH.s1 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 1)).x);
        regH.s2 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 2)).x);
        regH.s3 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 3)).x);

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // reduction in local memory, assumes #wave=4
    local float2 reduceLM[SUBGROUP_SIZE * 3];
    if (groupId == 1) {
        reduceLM[SUBGROUP_SIZE * 0 + slid] = totalSum;
    }
    if (groupId == 2) {
        reduceLM[SUBGROUP_SIZE * 1 + slid] = totalSum;
    }
    if (groupId == 3) {
        reduceLM[SUBGROUP_SIZE * 2 + slid] = totalSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 0 + slid];
    }
    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 1 + slid];
    }
    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 2 + slid];
    }

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        vstore2(totalSum, 0, &(dst[gid * 2]));
    }
}
