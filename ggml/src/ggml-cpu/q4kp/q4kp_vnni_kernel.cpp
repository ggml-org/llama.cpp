// P6 GEMV experiment: keep quant values and FP32 operations unchanged; replace
// the unsigned-byte/signed-byte dot stage with non-saturating VNNI dot-adds.
#include "q4kp_vnni_kernel.h"
#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "repack.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#define Q4KP_VNNI_TARGET __attribute__((target("avx2,bmi2,fma,f16c,avx512f,avx512vl,avx512vnni")))
#define UNUSED GGML_UNUSED
#ifndef GGML_F32Cx8_LOAD
#define GGML_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#endif
#define GGML_F32Cx8_REARRANGE_LOAD(x, mask) _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(x)), mask))

extern "C" int q4kp_vnni_supported(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512vnni");
}

static inline Q4KP_VNNI_TARGET __m128i q4kp_load_group(const uint8_t *code) {
    uint64_t first, last;
    // Both loads remain WITHIN the 12-byte group, including the final group
    // at the end of the metadata. memcpy supports unaligned addresses safely.
    std::memcpy(&first, code, 8);
    std::memcpy(&last, code + 4, 8);
    const uint64_t mask = 0x3f3f3f3f3f3f3f3full;
    const uint64_t scales = _pdep_u64(first, mask); // consumes only low 48 bits
    const uint64_t mins = _pdep_u64(last >> 16, mask);
    return _mm_set_epi64x(static_cast<int64_t>(mins), static_cast<int64_t>(scales));
}

extern "C" Q4KP_VNNI_TARGET void q4kp_vnni_gemv(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    if (n <= 0 || n % QK_K != 0 || nc <= 0 || nc % 8 != 0 || nr != 1 ||
        s == nullptr || vx == nullptr || vy == nullptr) {
        return;
    }
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    // Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Mask to extract nibbles from bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / QK_K;

    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 scale from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                const __m256i q8sums = _mm256_loadu_si256((const __m256i * )(a_ptr[b].bsums));
                __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
                q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

                // Processes two sub blocks from each Q4_K in each iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // 4-bit -> 8-bit
                    // Values of the first sub block of eight block_q4_K structures for the sb loop
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    // Values of the second sub block of eight block_q4_K structures when sb = 1
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    const __m128i decoded_0 = q4kp_load_group(b_ptr[b].scales + 24 * sb);
                    const __m128i decoded_1 = q4kp_load_group(b_ptr[b].scales + 24 * sb + 12);

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = decoded_0;
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Scales of second sub block in the sb loop
                    __m128i mins_and_scales_1 = decoded_1;
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                    __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
                    __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
                    __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

                    lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for first sub block and then for second sub block in each sb
                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // ...........................................................................
                    // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)


                    // Each old adjacent pair has the same six-bit scale. VNNI
                    // sums the same four products in int32, then multiplies by
                    // that scale. Even the original 16-bit partial sum is bounded
                    // by 16 * 15 * 128 = 30720, so neither form overflows or
                    // saturates. Four independent chains shorten dependencies;
                    // reassociation is integer-only, before unchanged FP32 FMAs.
                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();

                    __m256i partial_0_0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0));
                    partial_0_0 = _mm256_dpbusd_epi32(partial_0_0, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85));
                    __m256i partial_0_1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170));
                    partial_0_1 = _mm256_dpbusd_epi32(partial_0_1, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255));
                    __m256i partial_0_2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_02 ,_mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0));
                    partial_0_2 = _mm256_dpbusd_epi32(partial_0_2, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177) ,rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85));
                    __m256i partial_0_3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_03 ,_mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170));
                    partial_0_3 = _mm256_dpbusd_epi32(partial_0_3, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177) ,rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255));
                    iacc_0 = _mm256_add_epi32(_mm256_add_epi32(partial_0_0, partial_0_1),
                                               _mm256_add_epi32(partial_0_2, partial_0_3));
                    iacc_0 = _mm256_mullo_epi32(iacc_0, _mm256_srli_epi32(scales_0, 16));

                    __m256i partial_1_0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0));
                    partial_1_0 = _mm256_dpbusd_epi32(partial_1_0, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85));
                    __m256i partial_1_1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170));
                    partial_1_1 = _mm256_dpbusd_epi32(partial_1_1, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255));
                    __m256i partial_1_2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_12 ,_mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0));
                    partial_1_2 = _mm256_dpbusd_epi32(partial_1_2, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177) ,rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85));
                    __m256i partial_1_3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_blend_epi32(rhs_vec_0123_13 ,_mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170));
                    partial_1_3 = _mm256_dpbusd_epi32(partial_1_3, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177) ,rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255));
                    iacc_1 = _mm256_add_epi32(_mm256_add_epi32(partial_1_0, partial_1_1),
                                               _mm256_add_epi32(partial_1_2, partial_1_3));
                    iacc_1 = _mm256_mullo_epi32(iacc_1, _mm256_srli_epi32(scales_1, 16));

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                    // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
                    __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_bsrli_epi128(q8s, 4);

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Multiply-Add with scale values for the complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }

}
