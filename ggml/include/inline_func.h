#include <immintrin.h>

#define BMEMD 256
#define BBKEMD 96
template<int batch_size, int K3>
inline void tbl_impl_EMD(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    const int KK = BBKEMD / 3;
#pragma unroll
        for (int i = 0; i < BMEMD; i += 32) {
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
        for (int k = 0; k < KK / 8; k++) {
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
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BMEMD * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BMEMD * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BMEMD * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BMEMD * bs));
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BMEMD * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BMEMD * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BMEMD * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BMEMD * bs), vec_gc3);
    }
    }
#endif
}
    #include <immintrin.h>

#define BMGQA 128
#define BBKGQA 96
template<int batch_size, int K3>
inline void tbl_impl_GQA(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    const int KK = BBKEMD / 3;
#pragma unroll
        for (int i = 0; i < BMGQA; i += 32) {
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
        for (int k = 0; k < KK / 8; k++) {
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
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BMGQA * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BMGQA * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BMGQA * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BMGQA * bs));
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
}
    