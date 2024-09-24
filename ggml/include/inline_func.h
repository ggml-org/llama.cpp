#include <arm_neon.h>

#define BMEMD 256
#define BBKEMD 256
inline void tbl_impl_EMD(int32_t* c, int8_t* lut, uint8_t* a, tmac_float_type* scale, tmac_float_type* lut_scale) {
#ifdef __ARM_NEON
    const int KK = BBKEMD / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const int8x16_t vec_zero = vdupq_n_s16(0x0000);
    int8x16_t vec_lut[2 * KK];
    int16x8_t vec_c[4];
#pragma unroll
    for (int k = 0; k < 2 * KK; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

#pragma unroll
    for (int i = 0; i < BMEMD; i += 32) {
        #pragma unroll
        for (int i=0; i<4; i++) {
            vec_c[i] = vandq_s16(vec_c[i], vec_zero);
        }

#pragma unroll
        for (int k = 0; k < KK / 4; k++) {
            
            uint8x16_t vec_a_0 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 0 * 16);
            uint8x16_t vec_a0_top = vshrq_n_u8(vec_a_0, 4);
            uint8x16_t vec_a0_bot = vandq_u8(vec_a_0, vec_mask);
            int8x16_t  vec_v_0_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 0], vec_a0_top);
            int8x16_t  vec_v_0_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 1], vec_a0_top);
            int8x16_t  vec_v_0_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 2], vec_a0_bot);
            int8x16_t  vec_v_0_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 3], vec_a0_bot);
            int8x16x2_t  vec_v_left_0 = vzipq_s8(vec_v_0_left_tmp1, vec_v_0_left_tmp0);
            int8x16x2_t  vec_v_right_0 = vzipq_s8(vec_v_0_right_tmp1, vec_v_0_right_tmp0);
            vec_c[0] += vec_v_left_0.val[0];
            vec_c[0] += vec_v_right_0.val[0];
            vec_c[1] += vec_v_left_0.val[1];
            vec_c[1] += vec_v_right_0.val[1];
        
            uint8x16_t vec_a_1 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 1 * 16);
            uint8x16_t vec_a1_top = vshrq_n_u8(vec_a_1, 4);
            uint8x16_t vec_a1_bot = vandq_u8(vec_a_1, vec_mask);
            int8x16_t  vec_v_1_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 4], vec_a1_top);
            int8x16_t  vec_v_1_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 5], vec_a1_top);
            int8x16_t  vec_v_1_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 6], vec_a1_bot);
            int8x16_t  vec_v_1_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 7], vec_a1_bot);
            int8x16x2_t  vec_v_left_1 = vzipq_s8(vec_v_1_left_tmp1, vec_v_1_left_tmp0);
            int8x16x2_t  vec_v_right_1 = vzipq_s8(vec_v_1_right_tmp1, vec_v_1_right_tmp0);
            vec_c[0] += vec_v_left_1.val[0];
            vec_c[0] += vec_v_right_1.val[0];
            vec_c[1] += vec_v_left_1.val[1];
            vec_c[1] += vec_v_right_1.val[1];
        
            uint8x16_t vec_a_2 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 2 * 16);
            uint8x16_t vec_a2_top = vshrq_n_u8(vec_a_2, 4);
            uint8x16_t vec_a2_bot = vandq_u8(vec_a_2, vec_mask);
            int8x16_t  vec_v_2_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 0], vec_a2_top);
            int8x16_t  vec_v_2_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 1], vec_a2_top);
            int8x16_t  vec_v_2_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 2], vec_a2_bot);
            int8x16_t  vec_v_2_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 3], vec_a2_bot);
            int8x16x2_t  vec_v_left_2 = vzipq_s8(vec_v_2_left_tmp1, vec_v_2_left_tmp0);
            int8x16x2_t  vec_v_right_2 = vzipq_s8(vec_v_2_right_tmp1, vec_v_2_right_tmp0);
            vec_c[2] += vec_v_left_2.val[0];
            vec_c[2] += vec_v_right_2.val[0];
            vec_c[3] += vec_v_left_2.val[1];
            vec_c[3] += vec_v_right_2.val[1];
        
            uint8x16_t vec_a_3 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 3 * 16);
            uint8x16_t vec_a3_top = vshrq_n_u8(vec_a_3, 4);
            uint8x16_t vec_a3_bot = vandq_u8(vec_a_3, vec_mask);
            int8x16_t  vec_v_3_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 4], vec_a3_top);
            int8x16_t  vec_v_3_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 5], vec_a3_top);
            int8x16_t  vec_v_3_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 6], vec_a3_bot);
            int8x16_t  vec_v_3_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 7], vec_a3_bot);
            int8x16x2_t  vec_v_left_3 = vzipq_s8(vec_v_3_left_tmp1, vec_v_3_left_tmp0);
            int8x16x2_t  vec_v_right_3 = vzipq_s8(vec_v_3_right_tmp1, vec_v_3_right_tmp0);
            vec_c[2] += vec_v_left_3.val[0];
            vec_c[2] += vec_v_right_3.val[0];
            vec_c[3] += vec_v_left_3.val[1];
            vec_c[3] += vec_v_right_3.val[1];
        
       }

        int32x4_t vec_v_bot_low_low_0 = vmovl_s16(vget_low_s16(vec_c[0]));
        int32x4_t vec_v_bot_low_high_0 = vmovl_high_s16(vec_c[0]);
        vst1q_s32(c + i + 0, vld1q_s32(c + i + 0) + vec_v_bot_low_low_0);
        vst1q_s32(c + i + 4, vld1q_s32(c + i + 4) + vec_v_bot_low_high_0);
        int32x4_t vec_v_bot_low_low_1 = vmovl_s16(vget_low_s16(vec_c[1]));
        int32x4_t vec_v_bot_low_high_1 = vmovl_high_s16(vec_c[1]);
        vst1q_s32(c + i + 8, vld1q_s32(c + i + 8) + vec_v_bot_low_low_1);
        vst1q_s32(c + i + 12, vld1q_s32(c + i + 12) + vec_v_bot_low_high_1);
        int32x4_t vec_v_bot_low_low_2 = vmovl_s16(vget_low_s16(vec_c[2]));
        int32x4_t vec_v_bot_low_high_2 = vmovl_high_s16(vec_c[2]);
        vst1q_s32(c + i + 16, vld1q_s32(c + i + 16) + vec_v_bot_low_low_2);
        vst1q_s32(c + i + 20, vld1q_s32(c + i + 20) + vec_v_bot_low_high_2);
        int32x4_t vec_v_bot_low_low_3 = vmovl_s16(vget_low_s16(vec_c[3]));
        int32x4_t vec_v_bot_low_high_3 = vmovl_high_s16(vec_c[3]);
        vst1q_s32(c + i + 24, vld1q_s32(c + i + 24) + vec_v_bot_low_low_3);
        vst1q_s32(c + i + 28, vld1q_s32(c + i + 28) + vec_v_bot_low_high_3);

    }
#endif
}
#include <arm_neon.h>

#define BMGQA 256
#define BBKGQA 256
inline void tbl_impl_GQA(int32_t* c, int8_t* lut, uint8_t* a, tmac_float_type* scale, tmac_float_type* lut_scale) {
#ifdef __ARM_NEON
    const int KK = BBKGQA / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const int8x16_t vec_zero = vdupq_n_s16(0x0000);
    int8x16_t vec_lut[2 * KK];
    int16x8_t vec_c[4];
#pragma unroll
    for (int k = 0; k < 2 * KK; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

#pragma unroll
    for (int i = 0; i < BMGQA; i += 32) {
        #pragma unroll
        for (int i=0; i<4; i++) {
            vec_c[i] = vandq_s16(vec_c[i], vec_zero);
        }

#pragma unroll
        for (int k = 0; k < KK / 4; k++) {
            
            uint8x16_t vec_a_0 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 0 * 16);
            uint8x16_t vec_a0_top = vshrq_n_u8(vec_a_0, 4);
            uint8x16_t vec_a0_bot = vandq_u8(vec_a_0, vec_mask);
            int8x16_t  vec_v_0_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 0], vec_a0_top);
            int8x16_t  vec_v_0_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 1], vec_a0_top);
            int8x16_t  vec_v_0_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 2], vec_a0_bot);
            int8x16_t  vec_v_0_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 3], vec_a0_bot);
            int8x16x2_t  vec_v_left_0 = vzipq_s8(vec_v_0_left_tmp1, vec_v_0_left_tmp0);
            int8x16x2_t  vec_v_right_0 = vzipq_s8(vec_v_0_right_tmp1, vec_v_0_right_tmp0);
            vec_c[0] += vec_v_left_0.val[0];
            vec_c[0] += vec_v_right_0.val[0];
            vec_c[1] += vec_v_left_0.val[1];
            vec_c[1] += vec_v_right_0.val[1];
        
            uint8x16_t vec_a_1 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 1 * 16);
            uint8x16_t vec_a1_top = vshrq_n_u8(vec_a_1, 4);
            uint8x16_t vec_a1_bot = vandq_u8(vec_a_1, vec_mask);
            int8x16_t  vec_v_1_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 4], vec_a1_top);
            int8x16_t  vec_v_1_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 5], vec_a1_top);
            int8x16_t  vec_v_1_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 6], vec_a1_bot);
            int8x16_t  vec_v_1_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 7], vec_a1_bot);
            int8x16x2_t  vec_v_left_1 = vzipq_s8(vec_v_1_left_tmp1, vec_v_1_left_tmp0);
            int8x16x2_t  vec_v_right_1 = vzipq_s8(vec_v_1_right_tmp1, vec_v_1_right_tmp0);
            vec_c[0] += vec_v_left_1.val[0];
            vec_c[0] += vec_v_right_1.val[0];
            vec_c[1] += vec_v_left_1.val[1];
            vec_c[1] += vec_v_right_1.val[1];
        
            uint8x16_t vec_a_2 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 2 * 16);
            uint8x16_t vec_a2_top = vshrq_n_u8(vec_a_2, 4);
            uint8x16_t vec_a2_bot = vandq_u8(vec_a_2, vec_mask);
            int8x16_t  vec_v_2_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 0], vec_a2_top);
            int8x16_t  vec_v_2_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 1], vec_a2_top);
            int8x16_t  vec_v_2_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 2], vec_a2_bot);
            int8x16_t  vec_v_2_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 3], vec_a2_bot);
            int8x16x2_t  vec_v_left_2 = vzipq_s8(vec_v_2_left_tmp1, vec_v_2_left_tmp0);
            int8x16x2_t  vec_v_right_2 = vzipq_s8(vec_v_2_right_tmp1, vec_v_2_right_tmp0);
            vec_c[2] += vec_v_left_2.val[0];
            vec_c[2] += vec_v_right_2.val[0];
            vec_c[3] += vec_v_left_2.val[1];
            vec_c[3] += vec_v_right_2.val[1];
        
            uint8x16_t vec_a_3 = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + 3 * 16);
            uint8x16_t vec_a3_top = vshrq_n_u8(vec_a_3, 4);
            uint8x16_t vec_a3_bot = vandq_u8(vec_a_3, vec_mask);
            int8x16_t  vec_v_3_left_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 4], vec_a3_top);
            int8x16_t  vec_v_3_left_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 5], vec_a3_top);
            int8x16_t  vec_v_3_right_tmp0 = vqtbl1q_s8(vec_lut[8 * k + 6], vec_a3_bot);
            int8x16_t  vec_v_3_right_tmp1 = vqtbl1q_s8(vec_lut[8 * k + 7], vec_a3_bot);
            int8x16x2_t  vec_v_left_3 = vzipq_s8(vec_v_3_left_tmp1, vec_v_3_left_tmp0);
            int8x16x2_t  vec_v_right_3 = vzipq_s8(vec_v_3_right_tmp1, vec_v_3_right_tmp0);
            vec_c[2] += vec_v_left_3.val[0];
            vec_c[2] += vec_v_right_3.val[0];
            vec_c[3] += vec_v_left_3.val[1];
            vec_c[3] += vec_v_right_3.val[1];
        
       }

        int32x4_t vec_v_bot_low_low_0 = vmovl_s16(vget_low_s16(vec_c[0]));
        int32x4_t vec_v_bot_low_high_0 = vmovl_high_s16(vec_c[0]);
        vst1q_s32(c + i + 0, vld1q_s32(c + i + 0) + vec_v_bot_low_low_0);
        vst1q_s32(c + i + 4, vld1q_s32(c + i + 4) + vec_v_bot_low_high_0);
        int32x4_t vec_v_bot_low_low_1 = vmovl_s16(vget_low_s16(vec_c[1]));
        int32x4_t vec_v_bot_low_high_1 = vmovl_high_s16(vec_c[1]);
        vst1q_s32(c + i + 8, vld1q_s32(c + i + 8) + vec_v_bot_low_low_1);
        vst1q_s32(c + i + 12, vld1q_s32(c + i + 12) + vec_v_bot_low_high_1);
        int32x4_t vec_v_bot_low_low_2 = vmovl_s16(vget_low_s16(vec_c[2]));
        int32x4_t vec_v_bot_low_high_2 = vmovl_high_s16(vec_c[2]);
        vst1q_s32(c + i + 16, vld1q_s32(c + i + 16) + vec_v_bot_low_low_2);
        vst1q_s32(c + i + 20, vld1q_s32(c + i + 20) + vec_v_bot_low_high_2);
        int32x4_t vec_v_bot_low_low_3 = vmovl_s16(vget_low_s16(vec_c[3]));
        int32x4_t vec_v_bot_low_high_3 = vmovl_high_s16(vec_c[3]);
        vst1q_s32(c + i + 24, vld1q_s32(c + i + 24) + vec_v_bot_low_low_3);
        vst1q_s32(c + i + 28, vld1q_s32(c + i + 28) + vec_v_bot_low_high_3);

    }
#endif
}
