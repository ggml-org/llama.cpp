import argparse

def gen_tbl_impl(pre, BM, BK, bm, by):

    kernel_code = "\
#include <immintrin.h>\n\
\n\
#define BM{0} {1}\n\
#define BBK{0} {2}\n\
template<int batch_size, int K3>\n\
inline void tbl_impl_{0}(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {{\n\
".format(pre, BM, BK)

    kernel_code = "".join([kernel_code, "\
#ifdef __AVX2__\n\
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);\n\
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);\n\
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);\n\
    const __m256i vec_one  = _mm256_set1_epi8(0xff);\n\
    const int KK = BBKEMD / 3;\n\
#pragma unroll\n\
        for (int i = 0; i < BM{0}; i += 32) {{\n\
        __m256i vec_as[KK / 2];\n\
        __m256i vec_signs[KK / 8];\n\
        #pragma unroll\n\
        for (int ai = 0; ai < KK / 2; ai++) {{\n\
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));\n\
        }}\n\
        #pragma unroll\n\
        for (int as = 0; as < KK / 8; as++) {{\n\
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));\n\
        }}\n\
#pragma unroll\n\
    for (int bs = 0; bs < batch_size; bs++) {{\n\
        __m256i vec_c0 = _mm256_setzero_si256();\n\
        __m256i vec_c1 = _mm256_setzero_si256();\n\
#pragma unroll\n\
        for (int k = 0; k < KK / 8; k++) {{\n\
            __m256i vec_sign = vec_signs[k];\n\
                __m256i vec_a_0 = vec_as[k * 4 + 0];\n\
                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);\n\
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);\n\
                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);\n\
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);\n\
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);\n\
                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);\n\
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);\n\
                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);\n\
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);\n\
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);\n\
                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);\n\
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);\n\
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);\n\
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0);\n\
                __m256i vec_a_1 = vec_as[k * 4 + 1];\n\
                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);\n\
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);\n\
                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);\n\
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);\n\
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);\n\
                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);\n\
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);\n\
                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);\n\
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);\n\
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);\n\
                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);\n\
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);\n\
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);\n\
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1);\n\
                __m256i vec_a_2 = vec_as[k * 4 + 2];\n\
                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);\n\
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);\n\
                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);\n\
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);\n\
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);\n\
                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);\n\
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);\n\
                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);\n\
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);\n\
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);\n\
                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);\n\
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);\n\
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);\n\
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2);\n\
                __m256i vec_a_3 = vec_as[k * 4 + 3];\n\
                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);\n\
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);\n\
                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);\n\
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);\n\
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);\n\
                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);\n\
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);\n\
                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);\n\
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);\n\
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);\n\
                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);\n\
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);\n\
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);\n\
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3);\n\
        }}\n\
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM{0} * bs));\n\
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{0} * bs));\n\
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{0} * bs));\n\
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{0} * bs));\n\
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n\
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n\
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n\
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM{0} * bs), vec_gc0);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{0} * bs), vec_gc1);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{0} * bs), vec_gc2);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{0} * bs), vec_gc3);\n\
    }}\n\
    }}\n\
#endif\n\
}}\n\
    ".format(pre)])
    return kernel_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gen impl')
    parser.add_argument('--BMEMD',default="input", type=int)
    parser.add_argument('--BKEMD',default="input", type=int)
    parser.add_argument('--bmEMD',default="input", type=int)
    parser.add_argument('--byEMD',default="input", type=int)
    parser.add_argument('--BMGQA',default="input", type=int)
    parser.add_argument('--BKGQA',default="input", type=int)
    parser.add_argument('--bmGQA',default="input", type=int)
    parser.add_argument('--byGQA',default="input", type=int)
    args = parser.parse_args()

    k1_code = gen_tbl_impl("EMD", args.BMEMD, args.BKEMD, args.bmEMD, args.byEMD)
    k2_code = gen_tbl_impl("GQA", args.BMGQA, args.BKGQA, args.bmGQA, args.byGQA)

    with open("ggml/include/inline_func.h", 'w') as f:      # 写文件，开始的时候会先清空原文件，参考w的用法。如果不用with open，只是open，要注意最后将文件关闭。
        f.write(''.join(k1_code))
        f.write(''.join(k2_code))