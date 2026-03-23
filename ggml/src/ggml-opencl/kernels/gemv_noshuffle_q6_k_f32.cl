#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

#define dequantize_block_acc_bcast_8_hi(total_sum, bits4, bits2, scale_d, scale_s, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s7; \

#define dequantize_block_acc_bcast_8_lo(total_sum, bits4, bits2, scale_d, scale_s, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s7; \

#define dequantize_block_acc_bcast_1_hi(total_sum, bits4, bits2, scale_d, scale_s, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \

#define dequantize_block_acc_bcast_1_lo(total_sum, bits4, bits2, scale_d, scale_s, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q6_K_f32(
    read_only image1d_buffer_t src0_ql,
    read_only image1d_buffer_t src0_qh,
    global half2 * src0_s,
    global half2 * src0_d,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp = get_local_id(1);
    int gid = get_global_id(0);
    ushort slid = get_sub_group_local_id();

    int nb = ne00 / 32;

    uint4    reg_a_l;
    ushort4  reg_a_h;
    half2    reg_d;
    char4    reg_s;
    float8   reg_b;

    float2  total_sum = 0.0f;

    int line_stride_a = ne01 / 2;
    int block_stride_a = NSUBGROUPS * ne01;

    for (int k = grp; k < nb; k += NSUBGROUPS) {
        reg_d = src0_d[gid + k/8 * line_stride_a];
        reg_s = as_char4(src0_s[gid + k * line_stride_a]);

        if (slid < 4) {
            reg_b.s0123 = read_imagef(src1, 0 + slid*2 + k*8);
            reg_b.s4567 = read_imagef(src1, 1 + slid*2 + k*8);
        }

        reg_a_l.s0 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*0).x;
        reg_a_l.s1 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*1).x;
        reg_a_l.s2 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*2).x;
        reg_a_l.s3 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*3).x;

        reg_a_h.s0 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*0).x);
        reg_a_h.s1 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*1).x);
        reg_a_h.s2 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*2).x);
        reg_a_h.s3 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*3).x);

#ifdef VECTOR_SUB_GROUP_BROADCAT
        dequantize_block_acc_bcast_8_hi(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#else
        dequantize_block_acc_bcast_1_hi(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#endif // VECTOR_SUB_GROUP_BROADCAT

        reg_a_l.s0 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*4).x;
        reg_a_l.s1 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*5).x;
        reg_a_l.s2 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*6).x;
        reg_a_l.s3 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*7).x;

        reg_a_h.s0 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*4).x);
        reg_a_h.s1 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*5).x);
        reg_a_h.s2 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*6).x);
        reg_a_h.s3 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*7).x);

#ifdef VECTOR_SUB_GROUP_BROADCAT
        dequantize_block_acc_bcast_8_lo(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#else
        dequantize_block_acc_bcast_1_lo(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#endif // VECTOR_SUB_GROUP_BROADCAT
    }

    local float2 reduce_lm[SUBGROUP_SIZE * 3];
    if (grp == 1) {
        reduce_lm[SUBGROUP_SIZE*0 + slid] = total_sum;
    }
    if (grp == 2) {
        reduce_lm[SUBGROUP_SIZE*1 + slid] = total_sum;
    }
    if (grp == 3) {
        reduce_lm[SUBGROUP_SIZE*2 + slid] = total_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        total_sum += reduce_lm[SUBGROUP_SIZE*0 + slid];
    }
    if (grp == 0) {
        total_sum += reduce_lm[SUBGROUP_SIZE*1 + slid];
    }
    if (grp == 0) {
        total_sum += reduce_lm[SUBGROUP_SIZE*2 + slid];
    }

    if (grp == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        vstore2(total_sum, 0, &(dst[gid * 2]));
    }
}
