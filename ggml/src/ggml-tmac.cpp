#include <vector>
#include <type_traits>

#include "ggml-tmac.h"
#include "ggml-quants.h"
#include "inline_func.h"

// #include "t-mac/tmac_gemm_wrapper.h"

#define GGML_TMAC_MAX_NODES 8192

static bool initialized = false;

// static TMAC::TMACGeMMWrapper<tmac_tmac_float_type> * wrapper = nullptr;

static tmac_tensor_extra * tmac_tensor_extras = nullptr;

static size_t tmac_tensor_extras_index = 0;

static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}

static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void per_tensor_quant(int k, void* lut_scales_, void* b_) {
    tmac_float_type* lut_scales = (tmac_float_type*)lut_scales_;
    tmac_float_type* b = (tmac_float_type*)b_;
    // 0.5 per-tensor
#ifdef __ARM_NEON
    float32x4_t temp_max = vdupq_n_f32(0);
    for (int i=0; i < k / 4; i++) {
      float32x4_t vec_bs = vld1q_f32(b + 4 * i);
      float32x4_t abssum = vabsq_f32(vec_bs);
      temp_max = vmaxq_f32(abssum, temp_max);    
    }
    float32_t scales = 127 / vmaxvq_f32(temp_max);
    *lut_scales = scales;
#endif
}

void partial_max_reset(void* lut_scales_) {
    tmac_float_type* lut_scales = (tmac_float_type*)lut_scales_;
    *lut_scales = 0.0;
}

#ifdef __ARM_NEON
inline void Transpose_8_8(
    int16x8_t *v0,
    int16x8_t *v1,
    int16x8_t *v2,
    int16x8_t *v3,
    int16x8_t *v4,
    int16x8_t *v5,
    int16x8_t *v6,
    int16x8_t *v7)
{
    int16x8x2_t q04 = vzipq_s16(*v0, *v4);
    int16x8x2_t q15 = vzipq_s16(*v1, *v5);
    int16x8x2_t q26 = vzipq_s16(*v2, *v6);
    int16x8x2_t q37 = vzipq_s16(*v3, *v7);

    int16x8x2_t q0246_0 = vzipq_s16(q04.val[0], q26.val[0]);
    int16x8x2_t q0246_1 = vzipq_s16(q04.val[1], q26.val[1]);
    int16x8x2_t q1357_0 = vzipq_s16(q15.val[0], q37.val[0]);
    int16x8x2_t q1357_1 = vzipq_s16(q15.val[1], q37.val[1]);

    int16x8x2_t q_fin_0 = vzipq_s16(q0246_0.val[0], q1357_0.val[0]);
    int16x8x2_t q_fin_1 = vzipq_s16(q0246_0.val[1], q1357_0.val[1]);
    int16x8x2_t q_fin_2 = vzipq_s16(q0246_1.val[0], q1357_1.val[0]);
    int16x8x2_t q_fin_3 = vzipq_s16(q0246_1.val[1], q1357_1.val[1]);

    *v0 = q_fin_0.val[0];
    *v1 = q_fin_0.val[1];
    *v2 = q_fin_1.val[0];
    *v3 = q_fin_1.val[1];
    *v4 = q_fin_2.val[0];
    *v5 = q_fin_2.val[1];
    *v6 = q_fin_3.val[0];
    *v7 = q_fin_3.val[1];
}
#endif

template<int act_k>
inline void lut_ctor(int8_t* qlut, tmac_float_type* b, tmac_float_type* lut_scales) {
#ifdef __ARM_NEON
    int16x8_t vec_lut[16];
    float32_t scales = *lut_scales;

        uint8_t tbl_mask[16];
        tbl_mask[0] = 0;
        tbl_mask[1] = 2;
        tbl_mask[2] = 4;
        tbl_mask[3] = 6;
        tbl_mask[4] = 8;
        tbl_mask[5] = 10;
        tbl_mask[6] = 12;
        tbl_mask[7] = 14;
        tbl_mask[8] = 1;
        tbl_mask[9] = 3;
        tbl_mask[10] = 5;
        tbl_mask[11] = 7;
        tbl_mask[12] = 9;
        tbl_mask[13] = 11;
        tbl_mask[14] = 13;
        tbl_mask[15] = 15;

        uint8x16_t tbl_mask_q = vld1q_u8(tbl_mask);

#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {
        float32x4x2_t vec_bs_x0 = vld2q_f32(b + k * 16);
        float32x4x2_t vec_bs_x1 = vld2q_f32(b + k * 16 + 8);
        float32x4_t vec_f_0 = vmulq_n_f32(vec_bs_x0.val[0], scales);
        float32x4_t vec_f_1 = vmulq_n_f32(vec_bs_x0.val[1], scales);
        float32x4_t vec_f_2 = vmulq_n_f32(vec_bs_x1.val[0], scales);
        float32x4_t vec_f_3 = vmulq_n_f32(vec_bs_x1.val[1], scales);
        
        int32x4_t vec_b_0 = vcvtnq_s32_f32(vec_f_0);
        int32x4_t vec_b_1 = vcvtnq_s32_f32(vec_f_1);
        int32x4_t vec_b_2 = vcvtnq_s32_f32(vec_f_2);
        int32x4_t vec_b_3 = vcvtnq_s32_f32(vec_f_3);
        int16x4_t vec_b16_0 = vmovn_s32(vec_b_0);
        int16x4_t vec_b16_1 = vmovn_s32(vec_b_1);
        int16x4_t vec_b16_2 = vmovn_s32(vec_b_2);
        int16x4_t vec_b16_3 = vmovn_s32(vec_b_3);
        int16x8_t vec_bs_0 = vcombine_s16(vec_b16_0, vec_b16_2);
        int16x8_t vec_bs_1 = vcombine_s16(vec_b16_1, vec_b16_3);

        // -1 -1
        vec_lut[0] = vdupq_n_s16(0);
        vec_lut[0] = vec_lut[0] - vec_bs_0;
        vec_lut[0] = vec_lut[0] - vec_bs_1;

        // -1 0
        vec_lut[1] = vdupq_n_s16(0);
        vec_lut[1] = vec_lut[1] - vec_bs_0;

        // -1 1
        vec_lut[2] = vdupq_n_s16(0);
        vec_lut[2] = vec_lut[2] - vec_bs_0;
        vec_lut[2] = vec_lut[2] + vec_bs_1;

        // 0 -1
        vec_lut[3] = vdupq_n_s16(0);
        vec_lut[3] = vec_lut[3] - vec_bs_1;

        // 0 0
        vec_lut[4] = vdupq_n_s16(0);

        // 0 1
        vec_lut[5] = vec_bs_1;

        // 1 -1
        vec_lut[6] = vec_bs_0;
        vec_lut[6] = vec_lut[6] - vec_bs_1;

        // 1 0
        vec_lut[7] = vec_bs_0;

        // 1 1
        vec_lut[8] = vec_bs_0;
        vec_lut[8] = vec_lut[8] + vec_bs_1;

        Transpose_8_8(&(vec_lut[0]), &(vec_lut[1]), &(vec_lut[2]), &(vec_lut[3]),
                      &(vec_lut[4]), &(vec_lut[5]), &(vec_lut[6]), &(vec_lut[7]));

        Transpose_8_8(&(vec_lut[8]), &(vec_lut[9]), &(vec_lut[10]), &(vec_lut[11]),
                      &(vec_lut[12]), &(vec_lut[13]), &(vec_lut[14]), &(vec_lut[15]));

#pragma unroll
        for (int idx = 0; idx < 8; idx++) {
            int8x16_t q0_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx]), tbl_mask_q);
            int8x8_t q0_low = vget_low_s8(q0_s);
            int8x8_t q0_high = vget_high_s8(q0_s);
            int8x16_t q1_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx + 8]), tbl_mask_q);
            int8x8_t q1_low = vget_low_s8(q1_s);
            int8x8_t q1_high = vget_high_s8(q1_s);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2, q0_high);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 8, q1_high);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 16, q0_low);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 24, q1_low);
        }
    }

#endif
}

void preprocessor_k3200(void* B, void* LUT_Scales, void* QLUT) {
  partial_max_reset((&(((tmac_float_type*)LUT_Scales)[0])));
  // 3200 / 16 == 200
  per_tensor_quant(3200, (&(((tmac_float_type*)LUT_Scales)[0])), (&(((tmac_float_type*)B)[0])));
  
  lut_ctor<3200>((&(((int8_t*)QLUT)[0])), (&(((tmac_float_type*)B)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
}

void preprocessor_k8640(void* B, void* LUT_Scales, void* QLUT) {

  partial_max_reset((&(((tmac_float_type*)LUT_Scales)[0])));
  // 8640 / 16 == 200
  per_tensor_quant(8640, (&(((tmac_float_type*)LUT_Scales)[0])), (&(((tmac_float_type*)B)[0])));
  lut_ctor<8640>((&(((int8_t*)QLUT)[0])), (&(((tmac_float_type*)B)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
}

void preprocessor_k8192(void* B, void* LUT_Scales, void* QLUT) {

  partial_max_reset((&(((tmac_float_type*)LUT_Scales)[0])));
  // 8640 / 16 == 200
  per_tensor_quant(8192, (&(((tmac_float_type*)LUT_Scales)[0])), (&(((tmac_float_type*)B)[0])));
  
  lut_ctor<8192>((&(((int8_t*)QLUT)[0])), (&(((tmac_float_type*)B)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
}

void preprocessor_k45568(void* B, void* LUT_Scales, void* QLUT) {

  partial_max_reset((&(((tmac_float_type*)LUT_Scales)[0])));
  // 8640 / 16 == 200
  per_tensor_quant(45568, (&(((tmac_float_type*)LUT_Scales)[0])), (&(((tmac_float_type*)B)[0])));
  
  lut_ctor<45568>((&(((int8_t*)QLUT)[0])), (&(((tmac_float_type*)B)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
}

void preprocessor_k1536(void* B, void* LUT_Scales, void* QLUT) {

  partial_max_reset((&(((tmac_float_type*)LUT_Scales)[0])));
  // 8640 / 16 == 200
  per_tensor_quant(1536, (&(((tmac_float_type*)LUT_Scales)[0])), (&(((tmac_float_type*)B)[0])));
  
  lut_ctor<1536>((&(((int8_t*)QLUT)[0])), (&(((tmac_float_type*)B)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
}

void preprocessor_k4096(void* B, void* LUT_Scales, void* QLUT) {

  partial_max_reset((&(((tmac_float_type*)LUT_Scales)[0])));
  // 8640 / 16 == 200
  per_tensor_quant(4096, (&(((tmac_float_type*)LUT_Scales)[0])), (&(((tmac_float_type*)B)[0])));
  
  lut_ctor<4096>((&(((int8_t*)QLUT)[0])), (&(((tmac_float_type*)B)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
}


void qgemm_lut_k8640(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
//   alignas(32) tmac_float_type CBits[BMGQA];
alignas(32) int32_t CBits[BMGQA];
  memset(&(CBits[0]), 0, BMGQA * sizeof(int32_t));
#pragma unroll
  // compute 32 nums in one loop
  // 270 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 256 = 32 / 2 * 16
  for (int32_t k_outer = 0; k_outer < 8640 / BBKGQA; ++k_outer) {
    tbl_impl_GQA((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKGQA / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKGQA / 2 / 2 * BMGQA)])), (&(((tmac_float_type*)Scales)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
    }
#pragma unroll
  for (int i = 0; i < BMGQA; i++) {
    ((tmac_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((tmac_float_type*)LUT_Scales)[0] * ((tmac_float_type*)Scales)[0];
  }
}

void qgemm_lut_k3200(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) int32_t CBits[BMEMD];
  memset(&(CBits[0]), 0, BMEMD * sizeof(int32_t));
//   printf("check1\n");
#pragma unroll
  // compute 32 nums in one loop
  // 100 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 128 = 32 / 2 * 8
  for (int32_t k_outer = 0; k_outer < 3200 / BBKEMD; ++k_outer) {
    // printf("k_outer:%d\n", k_outer);
    tbl_impl_EMD((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKEMD / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKEMD / 2 / 2 * BMEMD)])), (&(((tmac_float_type*)Scales)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
    }
// printf("check2\n");
#pragma unroll
  for (int i = 0; i < BMEMD; i++) {
    ((tmac_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((tmac_float_type*)LUT_Scales)[0] * ((tmac_float_type*)Scales)[0];
  }
// printf("check3\n");
}

void qgemm_lut_k45568(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) int32_t CBits[BMGQA];
  memset(&(CBits[0]), 0, BMGQA * sizeof(int32_t));
#pragma unroll
  // compute 32 nums in one loop
  // 270 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 256 = 32 / 2 * 16
  for (int32_t k_outer = 0; k_outer < 45568 / BBKGQA; ++k_outer) {
    tbl_impl_GQA((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKGQA / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKGQA / 2 / 2 * BMGQA)])), (&(((tmac_float_type*)Scales)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
    }
#pragma unroll
  for (int i = 0; i < BMGQA; i++) {
    ((tmac_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((tmac_float_type*)LUT_Scales)[0] * ((tmac_float_type*)Scales)[0];
  }
}

void qgemm_lut_k8192(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) int32_t CBits[BMEMD];
  memset(&(CBits[0]), 0, BMEMD * sizeof(int32_t));
//   printf("check1\n");
#pragma unroll
  // compute 32 nums in one loop
  // 100 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 128 = 32 / 2 * 8
  for (int32_t k_outer = 0; k_outer < 8192 / BBKEMD; ++k_outer) {
    tbl_impl_EMD((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKEMD / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKEMD / 2 / 2 * BMEMD)])), (&(((tmac_float_type*)Scales)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
    }
// printf("check2\n");
#pragma unroll
  for (int i = 0; i < BMEMD; i++) {
    ((tmac_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((tmac_float_type*)LUT_Scales)[0] * ((tmac_float_type*)Scales)[0];
  }
// printf("check3\n");
}

void qgemm_lut_k4096(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) int32_t CBits[BMGQA];
  memset(&(CBits[0]), 0, BMGQA * sizeof(int32_t));
#pragma unroll
  // compute 32 nums in one loop
  // 270 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 256 = 32 / 2 * 16
  for (int32_t k_outer = 0; k_outer < 4096 / BBKGQA; ++k_outer) {
    tbl_impl_GQA((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKGQA / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKGQA / 2 / 2 * BMGQA)])), (&(((tmac_float_type*)Scales)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
    }
#pragma unroll
  for (int i = 0; i < BMGQA; i++) {
    ((tmac_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((tmac_float_type*)LUT_Scales)[0] * ((tmac_float_type*)Scales)[0];
  }
}

void qgemm_lut_k1536(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) int32_t CBits[BMEMD];
  memset(&(CBits[0]), 0, BMEMD * sizeof(int32_t));
//   printf("check1\n");
#pragma unroll
  // compute 32 nums in one loop
  // 100 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 128 = 32 / 2 * 8
  for (int32_t k_outer = 0; k_outer < 1536 / BBKEMD; ++k_outer) {
    tbl_impl_EMD((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKEMD / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKEMD / 2 / 2 * BMEMD)])), (&(((tmac_float_type*)Scales)[0])), (&(((tmac_float_type*)LUT_Scales)[0])));
    }
// printf("check2\n");
#pragma unroll
  for (int i = 0; i < BMEMD; i++) {
    ((tmac_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((tmac_float_type*)LUT_Scales)[0] * ((tmac_float_type*)Scales)[0];
  }
// printf("check3\n");
}

void ggml_preprocessor(int k, void* B, void* LUT_Scales, void* QLUT) {
    if (k == 3200) {
        preprocessor_k3200(B, LUT_Scales, QLUT);
    } 
    else if (k == 8640) {
        preprocessor_k8640(B, LUT_Scales, QLUT);
    }
    else if (k == 45568) {
        preprocessor_k45568(B, LUT_Scales, QLUT);
    }
    else if (k == 8192) {
        preprocessor_k8192(B, LUT_Scales, QLUT);
    }
    else if (k == 1536) {
        preprocessor_k1536(B, LUT_Scales, QLUT);
    }
    else if (k == 4096) {
        preprocessor_k4096(B, LUT_Scales, QLUT);
    }
}

void ggml_qgemm_lut(int k, void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    if (k == 3200) {
        qgemm_lut_k3200(A, LUT, Scales, LUT_Scales, C);
    } 
    else if (k == 8640) {
        qgemm_lut_k8640(A, LUT, Scales, LUT_Scales, C);
    }
    else if (k == 45568) {
        qgemm_lut_k45568(A, LUT, Scales, LUT_Scales, C);
    }
    else if (k == 8192) {
        qgemm_lut_k8192(A, LUT, Scales, LUT_Scales, C);
    }
    else if (k == 1536) {
        qgemm_lut_k1536(A, LUT, Scales, LUT_Scales, C);
    }
    else if (k == 4096) {
        qgemm_lut_k4096(A, LUT, Scales, LUT_Scales, C);
    }
}

void ggml_tmac_init(void) {
    // LOG(INFO) << "ggml_tmac_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new TMAC::TMACGeMMWrapper<tmac_tmac_float_type>();
    // }
    if (tmac_tensor_extras == nullptr) {
        tmac_tensor_extras = new tmac_tensor_extra[GGML_TMAC_MAX_NODES];
    }
    tmac_tensor_extras_index = 0;
}

void ggml_tmac_free(void) {
    // LOG(INFO) << "ggml_tmac_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < tmac_tensor_extras_index; i++) {
        // aligned_free(tmac_tensor_extras[i].qweights);
        // aligned_free(tmac_tensor_extras[i].scales);
    }
    delete[] tmac_tensor_extras;
    tmac_tensor_extras = nullptr;
}

static bool is_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_I2) {
        return true;
    } else {
        return false;
    }
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_I2) {
        // Add additional args to decide if permuted I2 or naive I2
        return false;
    } else {
        return true;
    }
}

// struct BlockQ40TypeAccessor {
//     using block_t = block_q4_0;

//     static constexpr int BITS = 4;
//     static constexpr int SIMD_LEN = 16;
//     static constexpr int group_size = (sizeof(block_t) - sizeof(ggml_fp16_t)) * 8 / BITS;
//     static constexpr int simd_n_elem = SIMD_LEN * 8 / BITS;

//     static uint8_t get_q(const void * data, int idx) {
//         const uint8_t * qs = (const uint8_t *) ((((const block_t *) data)[idx / group_size]).qs);
//         int internal_idx = idx % group_size;
//         const uint8_t * simd_qs = qs + internal_idx / simd_n_elem * SIMD_LEN;
//         int simd_idx = internal_idx % simd_n_elem;
//         return simd_qs[simd_idx % SIMD_LEN] >> (simd_idx / SIMD_LEN * BITS);
//     }

//     static tmac_float_type get_scale(const void * data, int idx) {
//         ggml_fp16_t d = ((const block_t *) data)[idx / group_size].d;
//         if (sizeof(tmac_tmac_float_type) == 2) {
//             tmac_tmac_float_type * fp16dp = reinterpret_cast<tmac_tmac_float_type *>(&d);
//             return *fp16dp;
//         } else {
//             return ggml_fp16_to_fp32(((const block_t *) data)[idx / group_size].d);
//         }
//     }
// };

// struct BlockI2TypeAccessor {
//     static constexpr int BITS = 2;
//     static constexpr int n_elem = 8 / BITS;

//     static uint8_t get_q(const void * data, int idx) {
//         const uint8_t * qs = (const uint8_t *) data;
//         int elem_idx = idx % n_elem;
//         return qs[idx / n_elem] >> (elem_idx * BITS);
//     }

//     static tmac_tmac_float_type get_scale(const void * data, int idx, int group_size) {
//         const float * ss = (const float *) data;
//         float s = ss[idx / group_size];
//         return (tmac_tmac_float_type) s;
//     }
// };

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        if (src1->ne[1] <= 1) {
            return true;
        }
    }
    return false;
}

size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_tmac_get_type_bits(src0->type);
    
    size_t wsize = ne10 * ne11 * 15 * sizeof(int8_t) + 1 * ne11 * 2 * sizeof(tmac_float_type);
    if (sizeof(tmac_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(tmac_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

// m = batch_size
// n = output_dim
// void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits) {
//     // t-mac llama.cpp n and m swapped
//     wrapper->llama_cpp_init(src1, qlut, lut_scales, lut_biases, n, k, m, bits);
// }

// void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
//     wrapper->llama_cpp_compute(src0, scales, qlut, lut_scales, lut_biases, dst, n, k, m, bits);
// }

void ggml_tmac_transform_tensor(struct ggml_tensor * tensor) {
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {
        return;
    }

    int k = tensor->ne[0];
    int m = tensor->ne[1];  // `n` in llama.cpp

    const int bits = ggml_tmac_get_type_bits(tensor->type);
    const int lut_scales_size = 1;
    const int scales_size = 1;
    int n_tile_num = 1;
    if (k == 8192) {
      n_tile_num = m / BMEMD;
    } else if(k == 45568) {
      n_tile_num = m / BMGQA;
    } else if (k == 3200) {
      n_tile_num = m / BMEMD;
    } else if(k == 8640) {
      n_tile_num = m / BMGQA;
    } else if(k == 1536) {
      n_tile_num = m / BMEMD;
    } else if(k == 4096) {
      n_tile_num = m / BMGQA;
    }
    uint8_t * qweights;
    tmac_float_type * scales;

    scales = (tmac_float_type *) aligned_malloc(scales_size * sizeof(tmac_float_type));
    qweights = (uint8_t *) tensor->data;
    float * i2_scales = (float * )(qweights + k * m / 4);
    scales[0] = (tmac_float_type) i2_scales[0];

    tensor->extra = tmac_tensor_extras + tmac_tensor_extras_index;
    tmac_tensor_extras[tmac_tensor_extras_index++] = {
        /* .lut_scales_size = */ lut_scales_size,
        /* .scales_size     = */ scales_size,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    };
}

int ggml_tmac_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_I2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

// void ggml_tmac_set_n_threads(int n_threads) {
//     wrapper->set_num_threads(n_threads);
// }
