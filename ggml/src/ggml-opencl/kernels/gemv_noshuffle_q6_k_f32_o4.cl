// 4-output-per-WI variant of kernel_gemv_noshuffle_q6_K_f32.
// Each WI now produces 4 consecutive outputs (output quad). The activation
// fetch (reg_b) is shared across all 4 outputs, doubling per-WI ALU per
// activation broadcast and halving the WG count vs the 2-output kernel.
//
// Implementation: each K-block we fetch TWO sets of (scales + ql + qh)
// — one for the low pair (rows 0,1 of the quad) and one for the high pair
// (rows 2,3) — and invoke the existing 2-output dequant macros twice
// against the *same* reg_b. Identical data layout to the 2-output kernel,
// so the host only needs to halve the grid and double the gid-to-output
// mapping.
//
// Opt-in via the host dispatch when GGML_OPENCL_Q6K_GEMV_O4=1.

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

// Macros are identical to the 2-output kernel — they accept `total_sum` as
// a parameter so we can call them twice (once per pair) against different
// accumulators against the same reg_b.
#define dequantize_block_acc_bcast_8_hi(total_sum, bits4, bits2, cs, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * cs.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * cs.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * cs.s2 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * cs.s2 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * cs.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * cs.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * cs.s2 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * cs.s2 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y.s7; \

#define dequantize_block_acc_bcast_8_lo(total_sum, bits4, bits2, cs, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * cs.s1 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * cs.s1 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * cs.s3 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * cs.s3 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * cs.s1 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * cs.s1 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * cs.s3 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * cs.s3 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y.s7; \

#define dequantize_block_acc_bcast_1_hi(total_sum, bits4, bits2, cs, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * cs.s2 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * cs.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * cs.s2 * shared_y; \

#define dequantize_block_acc_bcast_1_lo(total_sum, bits4, bits2, cs, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * cs.s3 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * cs.s1 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * cs.s3 * shared_y; \

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
// Q6K_O4_GLOBAL: read the (read-once-per-token, no-reuse) lm_head/embed weights
// from __global coalesced instead of image1d_buffer. The texture cache caps the
// streaming (no-reuse) lm_head read bandwidth; global coalesced reaches the
// higher rate the rest of the model gets. src1 (activation) stays an image (it IS reused via
// the cross-subgroup broadcast).
#ifdef Q6K_O4_GLOBAL
#define Q6K_O4_NAME kernel_gemv_noshuffle_q6_K_f32_o4_global
#define QL_ARG __global uint * src0_ql
#define QH_ARG __global half * src0_qh
#define RD_QL(b,i) (b[i])
#define RD_QH(b,i) as_ushort(b[i])
#else
#define Q6K_O4_NAME kernel_gemv_noshuffle_q6_K_f32_o4
#define QL_ARG read_only image1d_buffer_t src0_ql
#define QH_ARG read_only image1d_buffer_t src0_qh
#define RD_QL(b,i) (read_imageui(b,i).x)
#define RD_QH(b,i) as_ushort(read_imageh(b,i).x)
#endif
kernel void Q6K_O4_NAME(
    QL_ARG,
    QH_ARG,
    global half2 * src0_s,
    global half2 * src0_d,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp = get_local_id(1);
    int gid = get_global_id(0);              // 4-output-quad index
    ushort slid = get_sub_group_local_id();

    // Map quad index to the two pair-indices the existing 2-output access
    // pattern uses (consecutive output pairs along ne01). NB: the two pairs are
    // kept ADJACENT (gid*2, gid*2+1) on purpose -- a "stride-1" split (pairs
    // ne01/4 apart) is slower because two distant cache-line streams have worse
    // locality than the adjacent pair whose reads interleave into the same lines
    // each iteration.
    int gid_a = gid * 2;
    int gid_b = gid * 2 + 1;

    int nb = ne00 / 32;

    uint4   reg_a_l_a, reg_a_l_b;
    ushort4 reg_a_h_a, reg_a_h_b;
    half2   reg_d_a,   reg_d_b;
    char4   reg_s_a,   reg_s_b;
    float8  reg_b;

    float2 total_sum_a = 0.0f;
    float2 total_sum_b = 0.0f;

    int line_stride_a = ne01 / 2;
    int block_stride_a = NSUBGROUPS * ne01;

    for (int k = grp; k < nb; k += NSUBGROUPS) {
        reg_d_a = src0_d[gid_a + k/8 * line_stride_a];
        reg_d_b = src0_d[gid_b + k/8 * line_stride_a];
        reg_s_a = as_char4(src0_s[gid_a + k * line_stride_a]);
        reg_s_b = as_char4(src0_s[gid_b + k * line_stride_a]);
        // Precompute the loop-invariant combined scale (sub-block scale * super-block d)
        // once per pair instead of re-multiplying it for every one of the 256 elements.
        float4 cs_a = (float4)((float)reg_s_a.s0*(float)reg_d_a.s0, (float)reg_s_a.s1*(float)reg_d_a.s0,
                               (float)reg_s_a.s2*(float)reg_d_a.s1, (float)reg_s_a.s3*(float)reg_d_a.s1);
        float4 cs_b = (float4)((float)reg_s_b.s0*(float)reg_d_b.s0, (float)reg_s_b.s1*(float)reg_d_b.s0,
                               (float)reg_s_b.s2*(float)reg_d_b.s1, (float)reg_s_b.s3*(float)reg_d_b.s1);

        if (slid < 4) {
            reg_b.s0123 = read_imagef(src1, 0 + slid*2 + k*8);
            reg_b.s4567 = read_imagef(src1, 1 + slid*2 + k*8);
        }

        // Pair a (output rows gid_a*2, gid_a*2+1): read hi+lo then dequant
        // both in one block so the `_lo` macro can see the `shared_y` that
        // `_hi` declared. Pair b follows in its own block — fresh shared_y.
        {
            reg_a_l_a.s0 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*0);
            reg_a_l_a.s1 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*1);
            reg_a_l_a.s2 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*2);
            reg_a_l_a.s3 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*3);
            reg_a_h_a.s0 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*0);
            reg_a_h_a.s1 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*1);
            reg_a_h_a.s2 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*2);
            reg_a_h_a.s3 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*3);
#ifdef VECTOR_SUB_GROUP_BROADCAT
            dequantize_block_acc_bcast_8_hi(total_sum_a, as_ushort8(reg_a_l_a), as_uchar8(reg_a_h_a), cs_a, reg_b);
#else
            dequantize_block_acc_bcast_1_hi(total_sum_a, as_ushort8(reg_a_l_a), as_uchar8(reg_a_h_a), cs_a, reg_b);
#endif

            reg_a_l_a.s0 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*4);
            reg_a_l_a.s1 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*5);
            reg_a_l_a.s2 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*6);
            reg_a_l_a.s3 = RD_QL(src0_ql, gid_a + k*block_stride_a + line_stride_a*7);
            reg_a_h_a.s0 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*4);
            reg_a_h_a.s1 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*5);
            reg_a_h_a.s2 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*6);
            reg_a_h_a.s3 = RD_QH(src0_qh, gid_a + k*block_stride_a + line_stride_a*7);
#ifdef VECTOR_SUB_GROUP_BROADCAT
            dequantize_block_acc_bcast_8_lo(total_sum_a, as_ushort8(reg_a_l_a), as_uchar8(reg_a_h_a), cs_a, reg_b);
#else
            dequantize_block_acc_bcast_1_lo(total_sum_a, as_ushort8(reg_a_l_a), as_uchar8(reg_a_h_a), cs_a, reg_b);
#endif
        }

        {
            reg_a_l_b.s0 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*0);
            reg_a_l_b.s1 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*1);
            reg_a_l_b.s2 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*2);
            reg_a_l_b.s3 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*3);
            reg_a_h_b.s0 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*0);
            reg_a_h_b.s1 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*1);
            reg_a_h_b.s2 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*2);
            reg_a_h_b.s3 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*3);
#ifdef VECTOR_SUB_GROUP_BROADCAT
            dequantize_block_acc_bcast_8_hi(total_sum_b, as_ushort8(reg_a_l_b), as_uchar8(reg_a_h_b), cs_b, reg_b);
#else
            dequantize_block_acc_bcast_1_hi(total_sum_b, as_ushort8(reg_a_l_b), as_uchar8(reg_a_h_b), cs_b, reg_b);
#endif

            reg_a_l_b.s0 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*4);
            reg_a_l_b.s1 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*5);
            reg_a_l_b.s2 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*6);
            reg_a_l_b.s3 = RD_QL(src0_ql, gid_b + k*block_stride_a + line_stride_a*7);
            reg_a_h_b.s0 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*4);
            reg_a_h_b.s1 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*5);
            reg_a_h_b.s2 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*6);
            reg_a_h_b.s3 = RD_QH(src0_qh, gid_b + k*block_stride_a + line_stride_a*7);
#ifdef VECTOR_SUB_GROUP_BROADCAT
            dequantize_block_acc_bcast_8_lo(total_sum_b, as_ushort8(reg_a_l_b), as_uchar8(reg_a_h_b), cs_b, reg_b);
#else
            dequantize_block_acc_bcast_1_lo(total_sum_b, as_ushort8(reg_a_l_b), as_uchar8(reg_a_h_b), cs_b, reg_b);
#endif
        }
    }

    // Cross-subgroup reduce. Same shape as the 2-output kernel but with the
    // pair-a and pair-b accumulators concatenated into a single float4.
    local float4 reduce_lm[SUBGROUP_SIZE * 3];
    float4 acc = (float4)(total_sum_a.s0, total_sum_a.s1, total_sum_b.s0, total_sum_b.s1);
    if (grp == 1) { reduce_lm[SUBGROUP_SIZE*0 + slid] = acc; }
    if (grp == 2) { reduce_lm[SUBGROUP_SIZE*1 + slid] = acc; }
    if (grp == 3) { reduce_lm[SUBGROUP_SIZE*2 + slid] = acc; }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        acc += reduce_lm[SUBGROUP_SIZE*0 + slid];
        acc += reduce_lm[SUBGROUP_SIZE*1 + slid];
        acc += reduce_lm[SUBGROUP_SIZE*2 + slid];
        dst = (global float*)((global char*)dst + offsetd);
        // The dispatch rounds ne01/4 up to the subgroup width, so the tail
        // quads past the last row must not store (they wrote 128 rows past
        // dst on every ne01 % 256 == 128 vocab, e.g. 151936).
        if (gid * 4 + 3 < (uint)ne01) {
            vstore4(acc, 0, &(dst[gid * 4]));
        }
    }
}
