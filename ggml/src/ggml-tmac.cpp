#include <vector>
#include <type_traits>

#include "ggml-tmac.h"
#include "ggml-quants.h"
#if defined(GGML_BITNET_ARM_TL1)
#include "inline_func.h"
#endif

// #include "t-mac/tmac_gemm_wrapper.h"

#define GGML_TMAC_MAX_NODES 8192

#if defined(GGML_BITNET_X86_TL2)
#define BM3 128
#define BM2 128
#define BK3 96
#define BK2 32

#define TK 1536
#define TK_GQA 4096
#endif

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
<<<<<<< HEAD

#if defined(GGML_BITNET_ARM_TL1)

=======
#if defined(GGML_BITNET_ARM_TL1)
>>>>>>> upstream/release-dev
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
        type == GGML_TYPE_TL1) {
        return true;
    } else {
        return false;
    }
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
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
#endif
#if defined(GGML_BITNET_X86_TL2)
#if defined __AVX2__

inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi32(va, vb);
    *vh = _mm256_unpackhi_epi32(va, vb);
}

inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi64(va, vb);
    *vh = _mm256_unpackhi_epi64(va, vb);
}

inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));
}


inline void Transpose_8_8(
    __m256i *v0,
    __m256i *v1,
    __m256i *v2,
    __m256i *v3,
    __m256i *v4,
    __m256i *v5,
    __m256i *v6,
    __m256i *v7)
{
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;

    _mm256_merge_epi32(*v0, *v1, &w0, &w1);
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);

    _mm256_merge_epi64(w0, w2, &x0, &x1);
    _mm256_merge_epi64(w1, w3, &x2, &x3);
    _mm256_merge_epi64(w4, w6, &x4, &x5);
    _mm256_merge_epi64(w5, w7, &x6, &x7);

    _mm256_merge_si128(x0, x4, v0, v1);
    _mm256_merge_si128(x1, x5, v2, v3);
    _mm256_merge_si128(x2, x6, v4, v5);
    _mm256_merge_si128(x3, x7, v6, v7);
}

#endif

inline int32_t per_tensor_quant(int k, void* lut_scales_, void* b_) {
    tmac_float_type* lut_scales = (tmac_float_type*)lut_scales_;
    tmac_float_type* b = (tmac_float_type*)b_;
#if defined __AVX2__
    __m256 max_vec = _mm256_set1_ps(0.f);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    // #pragma unroll
    for (int i = 0; i < k / 8; i++) {
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
        max_vec = _mm256_max_ps(vec_babs, max_vec);
    }
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    float scales = 127 / _mm_cvtss_f32(max1);
    *lut_scales = scales;
#endif

    return 0;
}

inline int32_t partial_max_reset(int32_t bs, void* lut_scales_) {
    tmac_float_type* lut_scales = (tmac_float_type*)lut_scales_;
    #pragma unroll
    for (int i=0; i< bs; i++) {
        lut_scales[i] = 0.0;
    }
    return 0;
}

template<int act_k>
inline int32_t three_lut_ctor(int8_t* qlut, tmac_float_type* b, tmac_float_type* lut_scales) {
#if defined __AVX2__
    __m256 vec_lut[16];
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 24; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);

        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b2, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();

        vec_lut[14] = _mm256_setzero_si256();

        // 1 1 1
        vec_lut[13] = vec_b0i;
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b1i);
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b2i);

        // 1 1 0
        vec_lut[12] = vec_b0i;
        vec_lut[12] = _mm256_add_epi32(vec_lut[12], vec_b1i);

        // 1 1 -1
        vec_lut[11] = vec_b0i;
        vec_lut[11] = _mm256_add_epi32(vec_lut[11], vec_b1i);
        vec_lut[11] = _mm256_sub_epi32(vec_lut[11], vec_b2i);

        // 1 0 1
        vec_lut[10] = vec_b0i;
        vec_lut[10] = _mm256_add_epi32(vec_lut[10], vec_b2i);

        // 1 0 0
        vec_lut[9] = vec_b0i;

        // 1 0 -1
        vec_lut[8] = vec_b0i;
        vec_lut[8] = _mm256_sub_epi32(vec_lut[8], vec_b2i);

        // 1 -1 1
        vec_lut[7] = vec_b0i;
        vec_lut[7] = _mm256_sub_epi32(vec_lut[7], vec_b1i);
        vec_lut[7] = _mm256_add_epi32(vec_lut[7], vec_b2i);

        // 1 -1 0
        vec_lut[6] = vec_b0i;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1i);

        // 1 -1 -1
        vec_lut[5] = vec_b0i;
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b1i);
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b2i);

        // 0 1 1
        vec_lut[4] = vec_b1i;
        vec_lut[4] = _mm256_add_epi32(vec_lut[4], vec_b2i);

        // 0 1 0
        vec_lut[3] = vec_b1i;

        // 0 1 -1
        vec_lut[2] = vec_b1i;
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b2i);

        // 0 0 1
        vec_lut[1] = vec_b2i;

        // 0 0 0
        vec_lut[0] = _mm256_setzero_si256();

        __m256i ix[16];

#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }

    *lut_scales = scales;
#endif
    return 0;
}

template<int act_k>
inline int32_t two_lut_ctor(int8_t* qlut, tmac_float_type* b, tmac_float_type* lut_scales) {
#if defined __AVX2__
    __m256 vec_lut[16];
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {
        __m256 vec_b0f = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);
        __m256 vec_b1f = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);

        __m256i vec_b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();

        vec_lut[14] = _mm256_setzero_si256();

        vec_lut[13] = _mm256_setzero_si256();

        vec_lut[12] = _mm256_setzero_si256();

        vec_lut[11] = _mm256_setzero_si256();

        vec_lut[10] = _mm256_setzero_si256();

        vec_lut[9] = _mm256_setzero_si256();

        // 1 1
        vec_lut[8] = vec_b0;
        vec_lut[8] = _mm256_add_epi32(vec_lut[8], vec_b1);

        // 1 0
        vec_lut[7] = vec_b0;

        // 1 -1
        vec_lut[6] = vec_b0;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1);

        // 0 1
        vec_lut[5] = vec_b1;

        // 0 0
        vec_lut[4] = _mm256_setzero_si256();

        // 0 -1
        vec_lut[3] = _mm256_setzero_si256();
        vec_lut[3] = _mm256_sub_epi32(vec_lut[3], vec_b1);

        // -1 1
        vec_lut[2] = _mm256_setzero_si256();
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b0);
        vec_lut[2] = _mm256_add_epi32(vec_lut[2], vec_b1);

        // -1 0
        vec_lut[1] = _mm256_setzero_si256();
        vec_lut[1] = _mm256_sub_epi32(vec_lut[1], vec_b0);

        // -1 -1
        vec_lut[0] = _mm256_setzero_si256();
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b0);
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b1);

        __m256i ix[16];
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }
    *lut_scales = scales;
#endif
    return 0;
}

template<int batch_size, int K3>
inline int32_t three_tbl_impl(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    // compute 96 num
    // one K for 3 num / 32 K
    const int KK = BK3 / 3;

#pragma unroll
    // for (int i = 0; i < m / 2; i += 16) {
        for (int i = 0; i < BM3; i += 32) {
        __m256i vec_as[KK / 2];
        __m256i vec_signs[KK / 8];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
        #pragma unroll
        for (int as = 0; as < KK / 8; as++) {
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));
        }

#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        // KK / 4 for 32 row each row 8index
        for (int k = 0; k < KK / 8; k++) {
            // 16 * 16
            __m256i vec_sign = vec_signs[k];

                __m256i vec_a_0 = vec_as[k * 4 + 0];

                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));

                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);

                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);

                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);

                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);

                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0); 

                __m256i vec_a_1 = vec_as[k * 4 + 1];

                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));

                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);

                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);

                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);

                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);

                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1); 

                __m256i vec_a_2 = vec_as[k * 4 + 2];

                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));

                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);

                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);

                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);

                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);

                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2); 

                __m256i vec_a_3 = vec_as[k * 4 + 3];

                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));

                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);

                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);

                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);

                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);

                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3); 

        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM3 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM3 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM3 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM3 * bs));

        // 8 * int32
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM3 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM3 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM3 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM3 * bs), vec_gc3);

    }

    }
#endif
    return 0;
}

template<int batch_size, int K2>
inline int32_t two_tbl_impl(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);

    const int KK = BK2 / 2;

#pragma unroll
    for (int i = 0; i < BM2; i += 32) {
        // each 4 num / 4 * 8 = 32 num
        __m256i vec_as[KK / 2];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        // KK / 4 for 32 row each row 8index
        for (int k = 0; k < KK / 8; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                __m256i vec_a = vec_as[k * 4 + j];

                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM2 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM2 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM2 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM2 * bs));

        // 8 * int32
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM2 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM2 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM2 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM2 * bs), vec_gc3);
    }
    }
#endif
    return 0;
}

template<int BATCH_SIZE>
 int32_t three_qgemm_lut_k1536(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) uint32_t CBits[BATCH_SIZE * BM3];
  memset(&(CBits[0]), 0, BATCH_SIZE * BM3 * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 16
  // 8 * BM = 96 / 3 * BM / 4
  for (int32_t k_outer = 0; k_outer < 1536 / BK3; ++k_outer) {
    three_tbl_impl<BATCH_SIZE, 1536>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK3 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BK3 / 3 / 2 * BM3)])), (&(((uint8_t*)sign)[(k_outer * BK3 / 3 / 8 * BM3)])));
  }
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BM3; i++) {
    ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM3]);
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

template<int BATCH_SIZE>
 int32_t three_qgemm_lut_k4032(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) uint32_t CBits[BATCH_SIZE * BM3];
  memset(&(CBits[0]), 0, BATCH_SIZE * BM3 * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 16
  // 8 * BM = 96 / 3 * BM / 4
  for (int32_t k_outer = 0; k_outer < 4032 / BK3; ++k_outer) {
    three_tbl_impl<BATCH_SIZE, 4032>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK3 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BK3 / 3 / 2 * BM3)])), (&(((uint8_t*)sign)[(k_outer * BK3 / 3 / 8 * BM3)])));
  }
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BM3; i++) {
    ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM3]);
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

template<int BATCH_SIZE>
 int32_t two_qgemm_lut_k64(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) uint32_t CBits[BATCH_SIZE * BM2];
  memset(&(CBits[0]), 0, BATCH_SIZE * BM2 * sizeof(int32_t));
#pragma unroll
  // compute 32 nums in one loop
  // 270 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 128 = 32 / 2 * 8
  for (int32_t k_outer = 0; k_outer < 64 / 32; ++k_outer) {
    two_tbl_impl<BATCH_SIZE, 64>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM2)])));
  }
  // ???
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BM2; i++) {
    ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM2]);
    ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

template<int BATCH_SIZE>
 int32_t two_qgemm_lut_k0(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BM2; i++) {
    ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

void ggml_preprocessor(int bs, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));
  // 8640 / 24 == 200
    for (int32_t b = 0; b < bs; b++) {
        per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
        if (three_k == 1536) {
            three_lut_ctor<1536>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
        }
        else if (three_k == 4032) {
            three_lut_ctor<4032>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<64>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + 4032])), (&(((float*)LUT_Scales)[b])));
        }
    }
}

void ggml_qgemm_lut(int bs, int k, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    if (k == 1536) {
        if (bs == 1) {
            three_qgemm_lut_k1536<1>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 8) {
            three_qgemm_lut_k1536<8>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 32) {
            three_qgemm_lut_k1536<32>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 128) {
            three_qgemm_lut_k1536<128>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 256) {
            three_qgemm_lut_k1536<256>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 512) {
            three_qgemm_lut_k1536<512>(A, sign, LUT, Scales, LUT_Scales, C);
        }
    }
    else if (k == 4032) {
        if (bs == 1) {
            three_qgemm_lut_k4032<1>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 8) {
            three_qgemm_lut_k4032<8>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 32) {
            three_qgemm_lut_k4032<32>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 128) {
            three_qgemm_lut_k4032<128>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 256) {
            three_qgemm_lut_k4032<256>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 512) {
            three_qgemm_lut_k4032<512>(A, sign, LUT, Scales, LUT_Scales, C);
        }
    }
    else if (k == 64) {
        if (bs == 1) {
            two_qgemm_lut_k64<1>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 8) {
            two_qgemm_lut_k64<8>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 32) {
            two_qgemm_lut_k64<32>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 128) {
            two_qgemm_lut_k64<128>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 256) {
            two_qgemm_lut_k64<256>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 512) {
            two_qgemm_lut_k64<512>(A, LUT, Scales, LUT_Scales, C);
        }
    }
    else if (k == 0) {
        if (bs == 1) {
            two_qgemm_lut_k0<1>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 8) {
            two_qgemm_lut_k0<8>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 32) {
            two_qgemm_lut_k0<32>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 128) {
            two_qgemm_lut_k0<128>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 256) {
            two_qgemm_lut_k0<256>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 512) {
            two_qgemm_lut_k0<512>(A, LUT, Scales, LUT_Scales, C);
        }
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
        type == GGML_TYPE_TL2) {
        return true;
    } else {
        return false;
    }
}
#endif

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
#if defined(GGML_BITNET_ARM_TL1)
        if (src1->ne[1] <= 1) {
            return true;
        }
#endif
#if defined(GGML_BITNET_X86_TL2)
        return true;
#endif
    }
    return false;
}

#if defined(GGML_BITNET_ARM_TL1)
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
#elif defined(GGML_BITNET_X86_TL2)
size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    
    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(tmac_float_type);
    if (sizeof(tmac_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(tmac_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}
#endif

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
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

#endif
#if defined(GGML_BITNET_X86_TL2)
// BMGQA BM3 should be same due to hack
#define BK2 32
#if defined __AVX2__

inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi32(va, vb);
    *vh = _mm256_unpackhi_epi32(va, vb);
}

inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi64(va, vb);
    *vh = _mm256_unpackhi_epi64(va, vb);
}

inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));
}


inline void Transpose_8_8(
    __m256i *v0,
    __m256i *v1,
    __m256i *v2,
    __m256i *v3,
    __m256i *v4,
    __m256i *v5,
    __m256i *v6,
    __m256i *v7)
{
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;

    _mm256_merge_epi32(*v0, *v1, &w0, &w1);
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);

    _mm256_merge_epi64(w0, w2, &x0, &x1);
    _mm256_merge_epi64(w1, w3, &x2, &x3);
    _mm256_merge_epi64(w4, w6, &x4, &x5);
    _mm256_merge_epi64(w5, w7, &x6, &x7);

    _mm256_merge_si128(x0, x4, v0, v1);
    _mm256_merge_si128(x1, x5, v2, v3);
    _mm256_merge_si128(x2, x6, v4, v5);
    _mm256_merge_si128(x3, x7, v6, v7);
}

#endif

inline int32_t per_tensor_quant(int k, void* lut_scales_, void* b_) {
    tmac_float_type* lut_scales = (tmac_float_type*)lut_scales_;
    tmac_float_type* b = (tmac_float_type*)b_;
#if defined __AVX2__
    __m256 max_vec = _mm256_set1_ps(0.f);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    // #pragma unroll
    for (int i = 0; i < k / 8; i++) {
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
        max_vec = _mm256_max_ps(vec_babs, max_vec);
    }
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    float scales = 127 / _mm_cvtss_f32(max1);
    *lut_scales = scales;
#endif

    return 0;
}

inline int32_t partial_max_reset(int32_t bs, void* lut_scales_) {
    tmac_float_type* lut_scales = (tmac_float_type*)lut_scales_;
    #pragma unroll
    for (int i=0; i< bs; i++) {
        lut_scales[i] = 0.0;
    }
    return 0;
}

template<int act_k>
inline int32_t three_lut_ctor(int8_t* qlut, tmac_float_type* b, tmac_float_type* lut_scales) {
#if defined __AVX2__
    __m256 vec_lut[16];
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 24; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);

        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b2, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();

        vec_lut[14] = _mm256_setzero_si256();

        // 1 1 1
        vec_lut[13] = vec_b0i;
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b1i);
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b2i);

        // 1 1 0
        vec_lut[12] = vec_b0i;
        vec_lut[12] = _mm256_add_epi32(vec_lut[12], vec_b1i);

        // 1 1 -1
        vec_lut[11] = vec_b0i;
        vec_lut[11] = _mm256_add_epi32(vec_lut[11], vec_b1i);
        vec_lut[11] = _mm256_sub_epi32(vec_lut[11], vec_b2i);

        // 1 0 1
        vec_lut[10] = vec_b0i;
        vec_lut[10] = _mm256_add_epi32(vec_lut[10], vec_b2i);

        // 1 0 0
        vec_lut[9] = vec_b0i;

        // 1 0 -1
        vec_lut[8] = vec_b0i;
        vec_lut[8] = _mm256_sub_epi32(vec_lut[8], vec_b2i);

        // 1 -1 1
        vec_lut[7] = vec_b0i;
        vec_lut[7] = _mm256_sub_epi32(vec_lut[7], vec_b1i);
        vec_lut[7] = _mm256_add_epi32(vec_lut[7], vec_b2i);

        // 1 -1 0
        vec_lut[6] = vec_b0i;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1i);

        // 1 -1 -1
        vec_lut[5] = vec_b0i;
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b1i);
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b2i);

        // 0 1 1
        vec_lut[4] = vec_b1i;
        vec_lut[4] = _mm256_add_epi32(vec_lut[4], vec_b2i);

        // 0 1 0
        vec_lut[3] = vec_b1i;

        // 0 1 -1
        vec_lut[2] = vec_b1i;
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b2i);

        // 0 0 1
        vec_lut[1] = vec_b2i;

        // 0 0 0
        vec_lut[0] = _mm256_setzero_si256();

        __m256i ix[16];

#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }

    *lut_scales = scales;
#endif
    return 0;
}

template<int act_k>
inline int32_t two_lut_ctor(int8_t* qlut, tmac_float_type* b, tmac_float_type* lut_scales) {
#if defined __AVX2__
    __m256 vec_lut[16];
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {
        __m256 vec_b0f = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);
        __m256 vec_b1f = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);

        __m256i vec_b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();

        vec_lut[14] = _mm256_setzero_si256();

        vec_lut[13] = _mm256_setzero_si256();

        vec_lut[12] = _mm256_setzero_si256();

        vec_lut[11] = _mm256_setzero_si256();

        vec_lut[10] = _mm256_setzero_si256();

        vec_lut[9] = _mm256_setzero_si256();

        // 1 1
        vec_lut[8] = vec_b0;
        vec_lut[8] = _mm256_add_epi32(vec_lut[8], vec_b1);

        // 1 0
        vec_lut[7] = vec_b0;

        // 1 -1
        vec_lut[6] = vec_b0;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1);

        // 0 1
        vec_lut[5] = vec_b1;

        // 0 0
        vec_lut[4] = _mm256_setzero_si256();

        // 0 -1
        vec_lut[3] = _mm256_setzero_si256();
        vec_lut[3] = _mm256_sub_epi32(vec_lut[3], vec_b1);

        // -1 1
        vec_lut[2] = _mm256_setzero_si256();
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b0);
        vec_lut[2] = _mm256_add_epi32(vec_lut[2], vec_b1);

        // -1 0
        vec_lut[1] = _mm256_setzero_si256();
        vec_lut[1] = _mm256_sub_epi32(vec_lut[1], vec_b0);

        // -1 -1
        vec_lut[0] = _mm256_setzero_si256();
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b0);
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b1);

        __m256i ix[16];
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }
    *lut_scales = scales;
#endif
    return 0;
}

template<int batch_size, int K2>
inline int32_t two_tbl_impl(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);

    const int KK = BK2 / 2;

#pragma unroll
    for (int i = 0; i < BMGQA; i += 32) {
        // each 4 num / 4 * 8 = 32 num
        __m256i vec_as[KK / 2];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        // KK / 4 for 32 row each row 8index
        for (int k = 0; k < KK / 8; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                __m256i vec_a = vec_as[k * 4 + j];

                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BMGQA * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BMGQA * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BMGQA * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BMGQA * bs));

        // 8 * int32
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BMGQA * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BMGQA * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BMGQA * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BMGQA * bs), vec_gc3);
    }
    }
#endif
    return 0;
}

template<int BATCH_SIZE>
 int32_t three_qgemm_lut_k1536(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) uint32_t CBits[BATCH_SIZE * BMEMD];
  memset(&(CBits[0]), 0, BATCH_SIZE * BMEMD * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 16
  // 8 * BM = 96 / 3 * BM / 4
  for (int32_t k_outer = 0; k_outer < 1536 / BBKEMD; ++k_outer) {
    tbl_impl_EMD<BATCH_SIZE, 1536>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKEMD / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKEMD / 3 / 2 * BMEMD)])), (&(((uint8_t*)sign)[(k_outer * BBKEMD / 3 / 8 * BMEMD)])));
  }
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BMEMD; i++) {
    ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BMEMD]);
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

template<int BATCH_SIZE>
 int32_t three_qgemm_lut_k4032(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) uint32_t CBits[BATCH_SIZE * BMGQA];
  memset(&(CBits[0]), 0, BATCH_SIZE * BMGQA * sizeof(int32_t));
#pragma unroll
  // compute 96 nums in one loop
  // 96 = 8640 / 96
  // 16 * BM = 96 * BM / 3 / 2
  // 512 = 96 / 3 * 16
  // 8 * BM = 96 / 3 * BM / 4
  for (int32_t k_outer = 0; k_outer < 4032 / BBKGQA; ++k_outer) {
    tbl_impl_GQA<BATCH_SIZE, 4032>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBKGQA / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBKGQA / 3 / 2 * BMGQA)])), (&(((uint8_t*)sign)[(k_outer * BBKGQA / 3 / 8 * BMGQA)])));
  }
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BMGQA; i++) {
    ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BMGQA]);
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

template<int BATCH_SIZE>
 int32_t two_qgemm_lut_k64(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
  alignas(32) uint32_t CBits[BATCH_SIZE * BMGQA];
  memset(&(CBits[0]), 0, BATCH_SIZE * BMGQA * sizeof(int32_t));
#pragma unroll
  // compute 32 nums in one loop
  // 270 = 8640 / 32
  // 1280 = 32 * BM / 2 / 2
  // 128 = 32 / 2 * 8
  for (int32_t k_outer = 0; k_outer < 64 / 32; ++k_outer) {
    two_tbl_impl<BATCH_SIZE, 64>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BMGQA)])));
  }
  // ???
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BMGQA; i++) {
    ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BMGQA]);
    ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

template<int BATCH_SIZE>
 int32_t two_qgemm_lut_k0(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
#pragma unroll
  for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
  for (int i = 0; i < BMEMD; i++) {
    ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
  }
  }
  if (0 != 0) {
    return -1;
  }
  return 0;
}

void ggml_preprocessor(int bs, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));
  // 8640 / 24 == 200
    for (int32_t b = 0; b < bs; b++) {
        per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
        if (three_k == 1536) {
            three_lut_ctor<1536>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
        }
        else if (three_k == 4032) {
            three_lut_ctor<4032>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<64>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + 4032])), (&(((float*)LUT_Scales)[b])));
        }
    }
}

void ggml_qgemm_lut(int bs, int k, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    if (k == 1536) {
        if (bs == 1) {
            three_qgemm_lut_k1536<1>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 8) {
            three_qgemm_lut_k1536<8>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 32) {
            three_qgemm_lut_k1536<32>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 128) {
            three_qgemm_lut_k1536<128>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 256) {
            three_qgemm_lut_k1536<256>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 512) {
            three_qgemm_lut_k1536<512>(A, sign, LUT, Scales, LUT_Scales, C);
        }
    }
    else if (k == 4032) {
        if (bs == 1) {
            three_qgemm_lut_k4032<1>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 8) {
            three_qgemm_lut_k4032<8>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 32) {
            three_qgemm_lut_k4032<32>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 128) {
            three_qgemm_lut_k4032<128>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 256) {
            three_qgemm_lut_k4032<256>(A, sign, LUT, Scales, LUT_Scales, C);
        }else if (bs == 512) {
            three_qgemm_lut_k4032<512>(A, sign, LUT, Scales, LUT_Scales, C);
        }
    }
    else if (k == 64) {
        if (bs == 1) {
            two_qgemm_lut_k64<1>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 8) {
            two_qgemm_lut_k64<8>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 32) {
            two_qgemm_lut_k64<32>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 128) {
            two_qgemm_lut_k64<128>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 256) {
            two_qgemm_lut_k64<256>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 512) {
            two_qgemm_lut_k64<512>(A, LUT, Scales, LUT_Scales, C);
        }
    }
    else if (k == 0) {
        if (bs == 1) {
            two_qgemm_lut_k0<1>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 8) {
            two_qgemm_lut_k0<8>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 32) {
            two_qgemm_lut_k0<32>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 128) {
            two_qgemm_lut_k0<128>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 256) {
            two_qgemm_lut_k0<256>(A, LUT, Scales, LUT_Scales, C);
        } else if (bs == 512) {
            two_qgemm_lut_k0<512>(A, LUT, Scales, LUT_Scales, C);
        }
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
        type == GGML_TYPE_TL2) {
        return true;
    } else {
        return false;
    }
}

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        return true;
    }
    return false;
}

size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    
    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(tmac_float_type);
    if (sizeof(tmac_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(tmac_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

void ggml_tmac_transform_tensor(struct ggml_tensor * tensor) {
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {
        return;
    }

    int k = tensor->ne[0];
    int m = tensor->ne[1];  // `n` in llama.cpp

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
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}
#endif