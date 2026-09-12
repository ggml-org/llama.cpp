// P6 wide GEMV experiment. Low/high 256-bit halves compute independent
// eight-row groups. Q8 broadcasts are shared; the layout and FP32 order stay.
#include "q4kp_wide_kernel.h"
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
#define Q4KP_WIDE_TARGET __attribute__((target("avx2,bmi2,fma,f16c,avx512f,avx512vl,avx512bw,avx512vnni")))
#define UNUSED GGML_UNUSED
#ifndef GGML_F32Cx8_LOAD
#define GGML_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#endif
#define GGML_F32Cx8_REARRANGE_LOAD(x, mask) _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(x)), mask))
extern "C" int q4kp_wide_supported(void) {
    return q4kp_vnni_supported() && __builtin_cpu_supports("avx512bw");
}
static inline Q4KP_WIDE_TARGET __m512i q4kp_join256(__m256i a, __m256i b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}
static inline Q4KP_WIDE_TARGET __m512 q4kp_join_ps(__m256 a, __m256 b) {
    return _mm512_castsi512_ps(q4kp_join256(_mm256_castps_si256(a), _mm256_castps_si256(b)));
}
static inline Q4KP_WIDE_TARGET __m128i q4kp_wide_load_group(const uint8_t *code) {
    uint64_t first, last;
    std::memcpy(&first, code, 8);
    std::memcpy(&last, code + 4, 8);
    const uint64_t mask = 0x3f3f3f3f3f3f3f3full;
    return _mm_set_epi64x((int64_t)_pdep_u64(last >> 16, mask), (int64_t)_pdep_u64(first, mask));
}

extern "C" Q4KP_WIDE_TARGET void q4kp_wide_gemv(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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
    const __m512i finalpermutemask = _mm512_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15);

    // Mask to extract nibbles from bytes
    const __m512i m4b = _mm512_set1_epi8(0x0F);

    int64_t b_nb = n / QK_K;

    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 16; x++) {

            // Pointers to RHS blocks
            const block_q4_Kx8 * b_ptr_a = b_ptr_start + (2 * x * b_nb);
            const block_q4_Kx8 * b_ptr_b = b_ptr_a + b_nb;

            // Master FP accumulators
            __m512 acc_row = _mm512_setzero_ps();
            __m512 acc_min_rows = _mm512_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 scale from block_q8_K
                const __m512 row_scale_f32 = _mm512_set1_ps((a_ptr[b].d));

                // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m512 col_scale_f32 = q4kp_join_ps(GGML_F32Cx8_REARRANGE_LOAD(b_ptr_a[b].d, deltamask), GGML_F32Cx8_REARRANGE_LOAD(b_ptr_b[b].d, deltamask));
                const __m512 col_dmin_f32 = q4kp_join_ps(GGML_F32Cx8_LOAD(b_ptr_a[b].dmin), GGML_F32Cx8_LOAD(b_ptr_b[b].dmin));

                __m512i iacc_b = _mm512_setzero_si512();
                __m512i iacc_min_b = _mm512_setzero_si512();

                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)(a_ptr[b].bsums));
                const __m128i grouped_sums = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1));
                __m512i q8s = _mm512_broadcast_i32x4(grouped_sums);

                // Processes two sub blocks from each Q4_K in each iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m512i rhs_raw_vec_0123_0 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + sb * 256)));
                    const __m512i rhs_raw_vec_4567_0 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 32 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 32 + sb * 256)));
                    const __m512i rhs_raw_vec_0123_1 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 64 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 64 + sb * 256)));
                    const __m512i rhs_raw_vec_4567_1 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 96 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 96 + sb * 256)));
                    const __m512i rhs_raw_vec_0123_2 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 128 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 128 + sb * 256)));
                    const __m512i rhs_raw_vec_4567_2 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 160 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 160 + sb * 256)));
                    const __m512i rhs_raw_vec_0123_3 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 192 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 192 + sb * 256)));
                    const __m512i rhs_raw_vec_4567_3 = q4kp_join256(_mm256_loadu_si256((const __m256i * )(b_ptr_a[b].qs + 224 + sb * 256)), _mm256_loadu_si256((const __m256i * )(b_ptr_b[b].qs + 224 + sb * 256)));

                    // 4-bit -> 8-bit
                    // Values of the first sub block of eight block_q4_K structures for the sb loop
                    const __m512i rhs_vec_0123_00 = _mm512_and_si512(rhs_raw_vec_0123_0, m4b);
                    const __m512i rhs_vec_4567_00 = _mm512_and_si512(rhs_raw_vec_4567_0, m4b);
                    const __m512i rhs_vec_0123_01 = _mm512_and_si512(rhs_raw_vec_0123_1, m4b);
                    const __m512i rhs_vec_4567_01 = _mm512_and_si512(rhs_raw_vec_4567_1, m4b);
                    const __m512i rhs_vec_0123_02 = _mm512_and_si512(rhs_raw_vec_0123_2, m4b);
                    const __m512i rhs_vec_4567_02 = _mm512_and_si512(rhs_raw_vec_4567_2, m4b);
                    const __m512i rhs_vec_0123_03 = _mm512_and_si512(rhs_raw_vec_0123_3, m4b);
                    const __m512i rhs_vec_4567_03 = _mm512_and_si512(rhs_raw_vec_4567_3, m4b);

                    // Values of the second sub block of eight block_q4_K structures when sb = 1
                    const __m512i rhs_vec_0123_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m512i rhs_vec_4567_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m512i rhs_vec_0123_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m512i rhs_vec_4567_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m512i rhs_vec_0123_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m512i rhs_vec_4567_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m512i rhs_vec_0123_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m512i rhs_vec_4567_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    const __m128i a0 = q4kp_wide_load_group(b_ptr_a[b].scales + 24 * sb);
                    const __m128i a1 = q4kp_wide_load_group(b_ptr_a[b].scales + 24 * sb + 12);
                    const __m128i b0 = q4kp_wide_load_group(b_ptr_b[b].scales + 24 * sb);
                    const __m128i b1 = q4kp_wide_load_group(b_ptr_b[b].scales + 24 * sb + 12);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_set_m128i(_mm_shuffle_epi8(b0, scalemask), _mm_shuffle_epi8(a0, scalemask)));
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_set_m128i(_mm_shuffle_epi8(b1, scalemask), _mm_shuffle_epi8(a1, scalemask)));
                    const __m128i mins_a = _mm_unpacklo_epi8(_mm_shuffle_epi32(a0, 78), _mm_shuffle_epi32(a1, 78));
                    const __m128i mins_b = _mm_unpacklo_epi8(_mm_shuffle_epi32(b0, 78), _mm_shuffle_epi32(b1, 78));
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_set_m128i(mins_b, mins_a));

                    // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    const __m512i lhs_vec_00 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                    const __m512i lhs_vec_01 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64 + 16)));
                    const __m512i lhs_vec_10 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64 + 32)));
                    const __m512i lhs_vec_11 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64 + 48)));

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
                    __m512i iacc_0 = _mm512_setzero_si512();
                    __m512i iacc_1 = _mm512_setzero_si512();

                    __m512i partial_0_0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_00, _mm512_shuffle_epi32(rhs_vec_4567_00, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_00, (_MM_PERM_ENUM) 0));
                    partial_0_0 = _mm512_dpbusd_epi32(partial_0_0, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_00, (_MM_PERM_ENUM) 177), rhs_vec_4567_00), _mm512_shuffle_epi32(lhs_vec_00, (_MM_PERM_ENUM) 85));
                    __m512i partial_0_1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_01, _mm512_shuffle_epi32(rhs_vec_4567_01, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_00, (_MM_PERM_ENUM) 170));
                    partial_0_1 = _mm512_dpbusd_epi32(partial_0_1, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_01, (_MM_PERM_ENUM) 177), rhs_vec_4567_01), _mm512_shuffle_epi32(lhs_vec_00, (_MM_PERM_ENUM) 255));
                    __m512i partial_0_2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_02, _mm512_shuffle_epi32(rhs_vec_4567_02, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_01, (_MM_PERM_ENUM) 0));
                    partial_0_2 = _mm512_dpbusd_epi32(partial_0_2, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_02, (_MM_PERM_ENUM) 177), rhs_vec_4567_02), _mm512_shuffle_epi32(lhs_vec_01, (_MM_PERM_ENUM) 85));
                    __m512i partial_0_3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_03, _mm512_shuffle_epi32(rhs_vec_4567_03, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_01, (_MM_PERM_ENUM) 170));
                    partial_0_3 = _mm512_dpbusd_epi32(partial_0_3, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_03, (_MM_PERM_ENUM) 177), rhs_vec_4567_03), _mm512_shuffle_epi32(lhs_vec_01, (_MM_PERM_ENUM) 255));
                    iacc_0 = _mm512_add_epi32(_mm512_add_epi32(partial_0_0, partial_0_1),
                                               _mm512_add_epi32(partial_0_2, partial_0_3));
                    iacc_0 = _mm512_mullo_epi32(iacc_0, _mm512_srli_epi32(scales_0, 16));

                    __m512i partial_1_0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_10, _mm512_shuffle_epi32(rhs_vec_4567_10, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_10, (_MM_PERM_ENUM) 0));
                    partial_1_0 = _mm512_dpbusd_epi32(partial_1_0, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_10, (_MM_PERM_ENUM) 177), rhs_vec_4567_10), _mm512_shuffle_epi32(lhs_vec_10, (_MM_PERM_ENUM) 85));
                    __m512i partial_1_1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_11, _mm512_shuffle_epi32(rhs_vec_4567_11, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_10, (_MM_PERM_ENUM) 170));
                    partial_1_1 = _mm512_dpbusd_epi32(partial_1_1, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_11, (_MM_PERM_ENUM) 177), rhs_vec_4567_11), _mm512_shuffle_epi32(lhs_vec_10, (_MM_PERM_ENUM) 255));
                    __m512i partial_1_2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_12, _mm512_shuffle_epi32(rhs_vec_4567_12, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_11, (_MM_PERM_ENUM) 0));
                    partial_1_2 = _mm512_dpbusd_epi32(partial_1_2, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_12, (_MM_PERM_ENUM) 177), rhs_vec_4567_12), _mm512_shuffle_epi32(lhs_vec_11, (_MM_PERM_ENUM) 85));
                    __m512i partial_1_3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_mask_blend_epi32(43690, rhs_vec_0123_13, _mm512_shuffle_epi32(rhs_vec_4567_13, (_MM_PERM_ENUM) 177)), _mm512_shuffle_epi32(lhs_vec_11, (_MM_PERM_ENUM) 170));
                    partial_1_3 = _mm512_dpbusd_epi32(partial_1_3, _mm512_mask_blend_epi32(43690, _mm512_shuffle_epi32(rhs_vec_0123_13, (_MM_PERM_ENUM) 177), rhs_vec_4567_13), _mm512_shuffle_epi32(lhs_vec_11, (_MM_PERM_ENUM) 255));
                    iacc_1 = _mm512_add_epi32(_mm512_add_epi32(partial_1_0, partial_1_1),
                                               _mm512_add_epi32(partial_1_2, partial_1_3));
                    iacc_1 = _mm512_mullo_epi32(iacc_1, _mm512_srli_epi32(scales_1, 16));

                    // Accumulate the iacc value for one sb
                    __m512i iacc_sb = _mm512_add_epi32(iacc_0, iacc_1);

                    // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                    // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
                    __m512i q8s_sb = _mm512_shuffle_epi32(q8s, (_MM_PERM_ENUM) 0);
                    __m512i iacc_min_sb = _mm512_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm512_bsrli_epi128(q8s, 4);

                    // Accumulate for the complete block
                    iacc_b = _mm512_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm512_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Multiply-Add with scale values for the complete super block
                acc_row = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_b), _mm512_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_min_b), _mm512_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm512_permutexvar_ps(finalpermutemask, acc_row);
            _mm512_storeu_ps(s + (y * nr + x * 16), _mm512_sub_ps(acc_row, acc_min_rows));
        }
    }
    // The final eight output rows retain the frozen VNNI implementation.
    const int complete = (nc / 16) * 16;
    if (complete != nc) {
        const size_t block_offset = size_t(complete / 8) * size_t(nb);
        q4kp_vnni_gemv(n, s + complete, bs, b_ptr_start + block_offset, vy, 1, nc - complete);
    }

}
