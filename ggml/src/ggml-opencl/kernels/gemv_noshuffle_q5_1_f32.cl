#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK5_1 32
#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

#define dequantizeBlockAccum_ns_q5_1_sgbroadcast_1_hi(total_sums, bits4, bits1, scale, minv, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += (((bits4.s0 & 0x000F) | (((bits1.s0      ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x000F) | (((bits1.s4      ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4) | (((bits1.s0 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4) | (((bits1.s4 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8) | (((bits1.s0 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8) | (((bits1.s4 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s0 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s4 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += (((bits4.s2 & 0x000F) | (((bits1.s0 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x000F) | (((bits1.s4 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4) | (((bits1.s0 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4) | (((bits1.s4 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8) | (((bits1.s0 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8) | (((bits1.s4 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s0 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s4 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += (((bits4.s4 & 0x000F) | (((bits1.s1      ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x000F) | (((bits1.s5      ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4) | (((bits1.s1 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4) | (((bits1.s5 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8) | (((bits1.s1 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8) | (((bits1.s5 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s1 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s5 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += (((bits4.s6 & 0x000F) | (((bits1.s1 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x000F) | (((bits1.s5 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4) | (((bits1.s1 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4) | (((bits1.s5 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8) | (((bits1.s1 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8) | (((bits1.s5 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s1 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s5 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_q5_1_sgbroadcast_1_lo(total_sums, bits4, bits1, scale, minv, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += (((bits4.s0 & 0x000F) | (((bits1.s2      ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x000F) | (((bits1.s6      ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4) | (((bits1.s2 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4) | (((bits1.s6 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8) | (((bits1.s2 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8) | (((bits1.s6 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s2 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s6 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += (((bits4.s2 & 0x000F) | (((bits1.s2 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x000F) | (((bits1.s6 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4) | (((bits1.s2 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4) | (((bits1.s6 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8) | (((bits1.s2 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8) | (((bits1.s6 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s2 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s6 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += (((bits4.s4 & 0x000F) | (((bits1.s3      ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x000F) | (((bits1.s7      ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4) | (((bits1.s3 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4) | (((bits1.s7 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8) | (((bits1.s3 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8) | (((bits1.s7 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s3 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s7 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += (((bits4.s6 & 0x000F) | (((bits1.s3 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x000F) | (((bits1.s7 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4) | (((bits1.s3 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4) | (((bits1.s7 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8) | (((bits1.s3 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8) | (((bits1.s7 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s3 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s7 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_q5_1_sgbroadcast_8_hi(total_sums, bits4, bits1, scale, minv, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += (((bits4.s0 & 0x000F)         | (((bits1.s0     ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4)  | (((bits1.s0 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8)  | (((bits1.s0 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s0 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s2 & 0x000F)         | (((bits1.s0 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4)  | (((bits1.s0 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8)  | (((bits1.s0 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s0 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s1 & 0x000F)         | (((bits1.s4     ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4)  | (((bits1.s4 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8)  | (((bits1.s4 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s4 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s3 & 0x000F)         | (((bits1.s4 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4)  | (((bits1.s4 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8)  | (((bits1.s4 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s4 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += (((bits4.s4 & 0x000F)         | (((bits1.s1     ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4)  | (((bits1.s1 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8)  | (((bits1.s1 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s1 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s6 & 0x000F)         | (((bits1.s1 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4)  | (((bits1.s1 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8)  | (((bits1.s1 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s1 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s5 & 0x000F)         | (((bits1.s5     ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4)  | (((bits1.s5 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8)  | (((bits1.s5 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s5 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s7 & 0x000F)         | (((bits1.s5 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4)  | (((bits1.s5 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8)  | (((bits1.s5 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s5 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s7; \


#define dequantizeBlockAccum_ns_q5_1_sgbroadcast_8_lo(total_sums, bits4, bits1, scale, minv, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += (((bits4.s0 & 0x000F)         | (((bits1.s2     ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4)  | (((bits1.s2 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8)  | (((bits1.s2 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s2 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s2 & 0x000F)         | (((bits1.s2 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4)  | (((bits1.s2 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8)  | (((bits1.s2 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s2 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s1 & 0x000F)         | (((bits1.s6     ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4)  | (((bits1.s6 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8)  | (((bits1.s6 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s6 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s3 & 0x000F)         | (((bits1.s6 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4)  | (((bits1.s6 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8)  | (((bits1.s6 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s6 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += (((bits4.s4 & 0x000F)         | (((bits1.s3     ) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4)  | (((bits1.s3 >> 1) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8)  | (((bits1.s3 >> 2) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s3 >> 3) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s6 & 0x000F)         | (((bits1.s3 >> 4) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4)  | (((bits1.s3 >> 5) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8)  | (((bits1.s3 >> 6) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s3 >> 7) & 0x01) << 4)) * scale.s0 + minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s5 & 0x000F)         | (((bits1.s7     ) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4)  | (((bits1.s7 >> 1) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8)  | (((bits1.s7 >> 2) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s7 >> 3) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s7 & 0x000F)         | (((bits1.s7 >> 4) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4)  | (((bits1.s7 >> 5) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8)  | (((bits1.s7 >> 6) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s7 >> 7) & 0x01) << 4)) * scale.s1 + minv.s1) * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q5_1_f32(
        __read_only  image1d_buffer_t src0_qs,  // quantized A
        global ushort * src0_qh,                 // 5th bits
        global half2  * src0_d,                  // A scales
        global half2  * src0_m,                  // A mins
        __read_only  image1d_buffer_t src1,      // B activations
        global float * dst,
        ulong offsetd,
        int ne00,               // K
        int ne01)               // M
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A    = M / 2;
   uint BLOCK_STRIDE_A  = NSUBGROUPS * M;

    __private uint4     regA;
    __private half2     regS;
    __private half2     regM;
    __private float8    regB;

    __private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / QK5_1); k += NSUBGROUPS) {
        regS = src0_d[gid + k * LINE_STRIDE_A];
        regM = src0_m[gid + k * LINE_STRIDE_A];

        ushort4 qh_raw;
        qh_raw.s0 = src0_qh[gid + (4*k + 0) * LINE_STRIDE_A];
        qh_raw.s1 = src0_qh[gid + (4*k + 1) * LINE_STRIDE_A];
        qh_raw.s2 = src0_qh[gid + (4*k + 2) * LINE_STRIDE_A];
        qh_raw.s3 = src0_qh[gid + (4*k + 3) * LINE_STRIDE_A];

        uchar8 raw = as_uchar8(qh_raw);
        uchar8 qh_bytes = (uchar8)(raw.s0, raw.s2, raw.s4, raw.s6,
                                    raw.s1, raw.s3, raw.s5, raw.s7);

        // Load activations
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        regA.s0 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;

#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_q5_1_sgbroadcast_8_hi(totalSum, as_ushort8(regA), qh_bytes, regS, regM, regB);
#else
        dequantizeBlockAccum_ns_q5_1_sgbroadcast_1_hi(totalSum, as_ushort8(regA), qh_bytes, regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_qs, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_q5_1_sgbroadcast_8_lo(totalSum, as_ushort8(regA), qh_bytes, regS, regM, regB);
#else
        dequantizeBlockAccum_ns_q5_1_sgbroadcast_1_lo(totalSum, as_ushort8(regA), qh_bytes, regS, regM, regB);
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
