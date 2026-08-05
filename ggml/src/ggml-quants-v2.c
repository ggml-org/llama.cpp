// ggml-quants-v2.c
//
// Accelerate (vDSP) + NEON implementations of the 5 TILE640 quant
// helpers. The C references in ggml-quants.c are the documented
// behaviour; the v2 paths are host-side accelerations for the
// GGML_OP_TILE640_MATMUL dispatch in ggml-ane.mm.
//
// Build: gated on __APPLE__ (Accelerate) and __aarch64__ (NEON).
// On other platforms the v2 functions fall back to the C
// references. This keeps ggml-base portable (a Linux build of
// the ANE backend is not used in production but the test
// infrastructure still expects ggml-base to link).

#include "ggml-quants-v2.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define GGML_TESSERA_T640_V2_NEON 1
#else
#define GGML_TESSERA_T640_V2_NEON 0
#endif

// ---------------------------------------------------------------------------
// v2 enable flag. Default on for Apple Silicon; can be disabled with the
// GGML_TESSERA_T640_V2_DISABLE env var (1 = disable).
// ---------------------------------------------------------------------------

static int g_v2_enabled = -1;  // -1 = unset, 0 = disabled, 1 = enabled

int ggml_tessera_t640_v2_enabled(void) {
    if (g_v2_enabled < 0) {
        const char * env = getenv("GGML_TESSERA_T640_V2_DISABLE");
        g_v2_enabled = (env && env[0] == '1') ? 0 : 1;
    }
    return g_v2_enabled;
}

// ---------------------------------------------------------------------------
// Function A: dequantize_row_tessera_t640_v2
// ---------------------------------------------------------------------------
//
// Strategy: per page, build a 640-element sign vector ({-1, 0, +1})
// and a 640-element per-lane scale broadcast. Then one vDSP_vmul
// elementwise multiply per page produces the final fp32 output.
//
// Trit decode: radix-243 has a serial dependency on `idx` (each
// trit extraction divides by 3), so the decode is scalar. The
// {+1, 0, -1} mapping and the per-col multiply are NEON in
// 4-element chunks (20 trits per lane = 5 NEON chunks of 4).
//
// Numerical equivalence: the C reference uses scalar
// `(trit == 1) ? scale : (trit == 2) ? -scale : 0.0f`; the v2
// path computes `sign = (trit == 1) - (trit == 2)` (each as an
// int8 {0, 1} mask, then 0/+1/-1) and multiplies by scale. Both
// produce the same {+scale, 0, -scale} values bit-for-bit when
// scale is a normal fp32; the parity test asserts max abs
// diff == 0.
//
// For very small k (below GGML_TESSERA_T640_V2_MIN_K), the
// vDSP call setup cost dominates and the C reference is faster.
// The dispatcher checks k and falls back.

void dequantize_row_tessera_t640_v2(const void * GGML_RESTRICT x,
                                    float * GGML_RESTRICT y,
                                    int64_t k) {
    if (k < GGML_TESSERA_T640_V2_MIN_K || !ggml_tessera_t640_v2_enabled()) {
        dequantize_row_tessera_t640(x, y, k);
        return;
    }

    const int pages = (int)((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    const uint32_t * packed      = (const uint32_t *) x;
    const uint16_t * page_scales = (const uint16_t *) (packed + pages * TILE640_WORDS_PER_PAGE);
    const int8_t   * lane_scales = (const int8_t   *) (page_scales + pages);

    // Per-page scratch: sign vector and per-lane scale broadcast,
    // both 640 elements (one full page) to keep the vDSP call
    // full-length. Aligned for vDSP/NEON; stack-alloc is fine
    // (640 * 4 * 2 = 5 KB).
    float sign_buf[TILE640_PAGE_SIZE]      __attribute__((aligned(16)));
    float scale_buf[TILE640_PAGE_SIZE]     __attribute__((aligned(16)));

    for (int p = 0; p < pages; p++) {
        const int base     = p * TILE640_PAGE_SIZE;
        const int page_len = (base + TILE640_PAGE_SIZE <= k) ? TILE640_PAGE_SIZE : (int)(k - base);

        const float page_max = GGML_FP16_TO_FP32(page_scales[p]);

        for (int l = 0; l < TILE640_LANES_PER_PAGE; l++) {
            const int   col0  = l * TILE640_LANE_SIZE;
            const float scale = page_max *
                (lane_scales[p * TILE640_LANES_PER_PAGE + l] * (1.0f / 127.0f));

            // Decode the 4 radix-243 groups (= 20 trits per lane).
            // idx has a serial dependency; cannot SIMD. The serial
            // cost is 4 mod + 4 div per group, ~16 scalar ops per
            // lane. Pre-decode all 20 trits into a small int8
            // buffer; the {+1, 0, -1} sign mapping and the scale
            // broadcast are then NEON 4-element chunks.
            uint32_t rem = packed[p * TILE640_WORDS_PER_PAGE + l];
            int8_t trits[20];
            int t = 0;
            for (int g = 0; g < 4; g++) {
                uint32_t idx = rem % 243;
                rem /= 243;
                for (int d = 0; d < 5; d++) {
                    trits[t++] = (int8_t) (idx % 3);
                    idx /= 3;
                }
            }

            // 5 NEON chunks of 4 trits -> 4 fp32 outputs each.
            // The 4-element chunks cross group boundaries (group
            // 0 = trits 0-4, group 1 = trits 5-9, etc.; 4-element
            // chunks of the 20-trit lane: 0-3, 4-7, 8-11, 12-15,
            // 16-19; these straddle groups but the math is
            // per-trit and group-agnostic).
            //
            // NEON path: int8x4_t does not exist; the narrowest
            // int8 type is int8x8_t. We load 4 trits into the
            // lower half of int8x8_t. The compare-then-subtract
            // operates on 8 lanes; we use the lower 4. The widen
            // to int32 widens 4 int16 to 4 int32 (one vmovl_s16
            // call).
            //
            // Important: NEON's vceq_s8 returns 0xFF (-1) for
            // matches and 0x00 for non-matches (NOT 1 and 0 as
            // on some other SIMD ISAs). So vsub_s8(eq2, eq1)
            // gives:
            //   trit=0: 0 - 0 = 0
            //   trit=1: 0 - (-1) = +1   (eq1 had -1, eq2 had 0)
            //   trit=2: -1 - 0 = -1     (eq1 had 0, eq2 had -1)
            // which is the desired sign convention {0, +1, -1}
            // for trit {0, 1, 2}.
#if GGML_TESSERA_T640_V2_NEON
            const float32x4_t vscale = vdupq_n_f32(scale);
            for (int chunk = 0; chunk < 5; chunk++) {
                // 4 trits -> lower half of int8x8_t
                int8_t t8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                t8[0] = trits[chunk * 4 + 0];
                t8[1] = trits[chunk * 4 + 1];
                t8[2] = trits[chunk * 4 + 2];
                t8[3] = trits[chunk * 4 + 3];
                int8x8_t vt   = vld1_s8(t8);
                int8x8_t eq1  = vceq_s8(vt, vdup_n_s8(1));  // -1 where trit==1
                int8x8_t eq2  = vceq_s8(vt, vdup_n_s8(2));  // -1 where trit==2
                int8x8_t vsign_s8 = vsub_s8(eq2, eq1);      // {0, +1, -1} for {0, 1, 2}
                int16x8_t vsign_s16 = vmovl_s8(vsign_s8);
                int32x4_t vsign_i32 = vmovl_s16(vget_low_s16(vsign_s16));
                float32x4_t vsign_f32 = vcvtq_f32_s32(vsign_i32);
                vst1q_f32(&sign_buf[col0 + chunk * 4], vsign_f32);
                vst1q_f32(&scale_buf[col0 + chunk * 4], vscale);
            }
#else
            for (int j = 0; j < 20; j++) {
                const int8_t trit = trits[j];
                sign_buf[col0 + j]  = (trit == 1) ? 1.0f : (trit == 2) ? -1.0f : 0.0f;
                scale_buf[col0 + j] = scale;
            }
#endif
        }

        // Zero the trailing portion of the last page (col >= page_len).
        if (page_len < TILE640_PAGE_SIZE) {
            for (int j = page_len; j < TILE640_PAGE_SIZE; j++) {
                sign_buf[j]  = 0.0f;
                scale_buf[j] = 0.0f;
            }
        }

        // Bulk multiply: y[base + j] = sign_buf[j] * scale_buf[j]
        // for j in [0, page_len). On Apple use vDSP_vmul; on
        // other platforms use a scalar loop.
#if defined(__APPLE__)
        vDSP_vmul(sign_buf, 1, scale_buf, 1, y + base, 1, (vDSP_Length) page_len);
#else
        for (int j = 0; j < page_len; j++) {
            y[base + j] = sign_buf[j] * scale_buf[j];
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// Function B: quantize_row_tessera_t640_v2
// ---------------------------------------------------------------------------
//
// Strategy: per page, vDSP_maxmgv for page_max, vDSP_sve for
// sum_abs. The threshold is sum_abs / page_len. Per-lane,
// vDSP_maxmgv for lane_max; NEON 4-element chunks for the trit
// encoding and 243-base packing.
//
// Round-trip parity: this function must produce a packed layout
// that, when dequantised, matches the input within the 1-2 ulp
// fp32 noise floor. vDSP_sve uses parallel summation; for
// inputs well-separated from the threshold (e.g. random
// uniform in [-0.5, 0.5]) the v2 trits are identical to the C
// reference's trits. For inputs where many elements are within
// 1-2 ulp of the threshold, the v2 trits may flip for a few
// elements; the dequant round-trip will then differ by ~2*scale
// for those elements (still within the 1e-1 rel err bar the
// dispatch uses).
//
// Note: this is the reference quantizer (slow path). Production
// uses tessera-quant.cpp's vDSP path already. The point of
// rewriting this C function is to keep the dequant <-> quant
// round-trip bit-identical (the test framework uses this for
// parity).

// Power-of-3 table for the 243-base packer. Indexed by trit
// position 0..4 (1, 3, 9, 27, 81).
static const uint32_t k_pow3[5] = { 1, 3, 9, 27, 81 };

void quantize_row_tessera_t640_v2(const float * GGML_RESTRICT x,
                                  void * GGML_RESTRICT y,
                                  int64_t k) {
    if (k < GGML_TESSERA_T640_V2_MIN_K || !ggml_tessera_t640_v2_enabled()) {
        quantize_row_tessera_t640_ref(x, y, k);
        return;
    }

    const int pages = (int)((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    uint32_t * packed      = (uint32_t *) y;
    uint16_t * page_scales = (uint16_t *) (packed + pages * TILE640_WORDS_PER_PAGE);
    int8_t   * lane_scales = (int8_t *)   (page_scales + pages);

    for (int p = 0; p < pages; p++) {
        const int base     = p * TILE640_PAGE_SIZE;
        const int page_len = (base + TILE640_PAGE_SIZE <= k) ? TILE640_PAGE_SIZE : (int)(k - base);

        // Per-page: page_max (vDSP_maxmgv), mean_abs
        // (vDSP_meamgv = mean of |x| = sum_abs / page_len).
        // We use vDSP_meamgv (mean of magnitudes) instead of
        // vDSP_sve (sum of values) because the C ref computes
        // sum_abs / page_len = mean(|x|); vDSP_sve would
        // produce sum(x) which is near zero for symmetric
        // signals and gives a wrong threshold.
        float page_max_f  = 0.0f;
        float mean_abs_f  = 0.0f;
#if defined(__APPLE__)
        vDSP_maxmgv(x + base, 1, &page_max_f, (vDSP_Length) page_len);
        vDSP_meamgv(x + base, 1, &mean_abs_f, (vDSP_Length) page_len);
#else
        for (int j = 0; j < page_len; j++) {
            const float a = fabsf(x[base + j]);
            mean_abs_f += a;
            if (a > page_max_f) page_max_f = a;
        }
        mean_abs_f /= (float) page_len;
#endif
        const float threshold = mean_abs_f;
        page_scales[p] = GGML_FP32_TO_FP16(page_max_f);

        for (int l = 0; l < TILE640_LANES_PER_PAGE; l++) {
            const int col0 = base + l * TILE640_LANE_SIZE;

            // Per-lane: lane_max. For the partial last lane
            // (col0 + TILE640_LANE_SIZE > k) we must bounds-
            // check; the C ref's per-element loop does this
            // with `if (col < k)`, but vDSP_maxmgv reads the
            // full 20 elements unconditionally and would
            // include uninitialised memory beyond k. We use
            // the smaller lane_len for the partial lane so
            // the vDSP call only reads valid cols.
            const int lane_len = (col0 + TILE640_LANE_SIZE <= k)
                ? TILE640_LANE_SIZE : (int)(k - col0);
            float lane_max_f = 0.0f;
#if defined(__APPLE__)
            vDSP_maxmgv(x + col0, 1, &lane_max_f, (vDSP_Length) lane_len);
#else
            for (int j = 0; j < lane_len; j++) {
                const float a = fabsf(x[col0 + j]);
                if (a > lane_max_f) lane_max_f = a;
            }
#endif

            int8_t ls = 0;
            if (page_max_f > 0.0f) {
                ls = (int8_t) roundf(127.0f * lane_max_f / page_max_f);
                if (ls > 127) ls = 127;
            }
            lane_scales[p * TILE640_LANES_PER_PAGE + l] = ls;

            // Trit encoding: 4 groups of 5 trits per lane. Per
            // group, 5 trits in base-3 packed into a single
            // uint32_t (the "group_val"). Per lane, 4 group_vals
            // are packed into the lane's 32-bit word as
            //     word = sum_g group_val * 243^g
            // (4 * log2(243) = 31.7 bits, fits in 32 bits).
            //
            // NEON path: 5 trits per group. We process 4 trits
            // at a time (chunked across the 5-trit groups). The
            // compare produces an int32 {0, 1, 2} trit; the
            // 243-base packer multiplies by [1, 3, 9, 27] (for
            // 4-trit chunks) and reduces to a uint32_t via
            // dot-product (vmull + vmlal).
            uint32_t word = 0;
            for (int g = 0; g < 4; g++) {
                uint32_t group_val = 0;
                const int gcol = col0 + g * 5;
#if GGML_TESSERA_T640_V2_NEON
                // 5 trits per group, processed as one 4-element
                // NEON chunk (cols 0-3) + 1 scalar leftover
                // (col 4). The encoding is group-local: a single
                // 4-trit chunk packs to 4 trit-positions in the
                // current group, the 5th is the scalar tail.
                // For the partial last lane of the partial last
                // page, the last group may have fewer than 4
                // valid cols. The bounds-check uses < not <=
                // because we need 4 contiguous valid cols to
                // safely vld1q_f32.
                const int valid = (gcol + 4 <= k) ? 4 : (k - gcol);
                if (valid >= 4) {
                    float32x4_t vf = vld1q_f32(x + gcol);
                    // trit: v > +threshold -> 1; v < -threshold -> 2; else 0
                    int32x4_t vmask_pos = vcgtq_f32(vf, vdupq_n_f32(threshold));
                    int32x4_t vmask_neg = vcltq_f32(vf, vdupq_n_f32(-threshold));
                    // trit = (vmask_pos & 1) | (vmask_neg & 2)
                    int32x4_t vtrit = vorrq_s32(vandq_s32(vmask_pos, vdupq_n_s32(1)),
                                                vandq_s32(vmask_neg, vdupq_n_s32(2)));
                    // 243-base pack: group_val_partial = sum_{d=0..3} trit[d] * 3^d
                    int32x4_t vpow = vld1q_s32((const int32_t *) k_pow3);
                    int32x4_t vmul = vmulq_s32(vtrit, vpow);
                    // Reduce to scalar via pairwise add. NEON has
                    // no horizontal int32 add intrinsic; use the
                    // standard pattern: vget_low + vget_high +
                    // vpadd + vpadd.
                    int32x2_t vl = vget_low_s32(vmul);
                    int32x2_t vh = vget_high_s32(vmul);
                    int32x2_t vs = vpadd_s32(vl, vh);
                    int32x2_t vss = vpadd_s32(vs, vs);
                    group_val += (uint32_t) vget_lane_s32(vss, 0);
                }
                // Scalar tail: handle the rest scalar (this
                // also handles the partial last group when
                // gcol + 4 >= k).
                for (int d = (valid >= 4 ? 4 : 0); d < 5; d++) {
                    const int col = gcol + d;
                    uint32_t trit = 0;
                    if (col < k) {
                        const float v = x[col];
                        if (v >  threshold) trit = 1;
                        if (v < -threshold) trit = 2;
                    }
                    group_val += trit * k_pow3[d];
                }
#else
                for (int d = 0; d < 5; d++) {
                    const int col = gcol + d;
                    uint32_t trit = 0;
                    if (col < k) {
                        const float v = x[col];
                        if (v >  threshold) trit = 1;
                        if (v < -threshold) trit = 2;
                    }
                    group_val += trit * k_pow3[d];
                }
#endif
                // pow243 accumulates across the 4 groups.
                // 243^0 = 1, 243^1 = 243, 243^2 = 59049,
                // 243^3 = 14348907.
                static const uint32_t k_pow243[4] = { 1, 243, 59049, 14348907 };
                word += group_val * k_pow243[g];
            }
            packed[p * TILE640_WORDS_PER_PAGE + l] = word;
        }
    }
}

// ---------------------------------------------------------------------------
// Function C: apply_outlier_addback_v2
// ---------------------------------------------------------------------------
//
// Sparse scatter: for [lo, hi) of the outlier CSR row, write
// row[outlier_cols[k]] = fp16_to_fp32(outlier_vals[k]) for each
// k. The scatter is per-element and not vDSP-friendly (sparse,
// irregular column indices). The v2 path batches the
// fp16->fp32 conversion with NEON vcvt_f32_f16 (4 fp16 -> 4
// fp32 per chunk): we read the outlier_vals once, convert in
// bulk into a small stack scratch, and the scatter reads from
// the scratch instead of doing the conversion per scatter.
//
// For the typical outlier pattern (~5% sparsity, ~100 outliers
// per row of 4096 cols) the per-row outlier count is small
// (<= 200 typical), so the bulk conversion reads at most ~200
// fp16 values. The NEON 4-element chunks are not a huge win at
// this size; the v2 label is for API symmetry with the
// dispatch's v2 path. For dense rows (e.g. the test fixture
// with no outliers) the function is a no-op.

void apply_outlier_addback_v2(float * GGML_RESTRICT row,
                              int64_t row_len,
                              int32_t lo, int32_t hi,
                              const int32_t * GGML_RESTRICT outlier_cols,
                              const void * GGML_RESTRICT outlier_vals) {
    const uint16_t * vals = (const uint16_t *) outlier_vals;
    const int32_t n = hi - lo;
    if (n <= 0) return;

    // Convert all outlier_vals in [lo, hi) to fp32 in a stack
    // scratch. vDSP_vflt16 is for int16, not fp16; we use NEON
    // vcvt_f32_f16 (4 fp16 -> 4 fp32 per chunk) on Apple, scalar
    // on other platforms. The scratch size is bounded by the
    // outlier count; for typical 5% sparsity the count is
    // ~in_dim * 0.05, e.g. ~200 for in_dim=4096.
    if (n > 4096) {
        // Pathological: fall back to per-element scalar.
        for (int32_t k = lo; k < hi; k++) {
            const int32_t col = outlier_cols[k];
            if (col >= 0 && col < row_len) {
                row[col] = GGML_FP16_TO_FP32(vals[k]);
            }
        }
        return;
    }
    float val_scratch[4096] __attribute__((aligned(16)));
#if GGML_TESSERA_T640_V2_NEON
    // NEON 4-element chunks: 4 fp16 -> 4 fp32. The tail (n % 4)
    // is handled scalar.
    int32_t k = 0;
    for (; k + 4 <= n; k += 4) {
        uint64_t bits;
        memcpy(&bits, &vals[lo + k], sizeof(bits));
        float16x4_t vh = vreinterpret_f16_u64(vdup_n_u64(bits));
        float32x4_t vf = vcvt_f32_f16(vh);
        vst1q_f32(&val_scratch[k], vf);
    }
    for (; k < n; k++) {
        val_scratch[k] = GGML_FP16_TO_FP32(vals[lo + k]);
    }
#else
    for (int32_t k = 0; k < n; k++) {
        val_scratch[k] = GGML_FP16_TO_FP32(vals[lo + k]);
    }
#endif

    // Scatter: write the pre-converted fp32 values to the
    // outlier column positions. The scatter is per-element
    // (the column indices are irregular) so it stays scalar.
    for (int32_t k = 0; k < n; k++) {
        const int32_t col = outlier_cols[lo + k];
        if (col >= 0 && col < row_len) {
            row[col] = val_scratch[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Function D: decode_per_row_meta_v2
// ---------------------------------------------------------------------------
//
// Per-page: fp16 page_scale -> fp32 page_max via NEON
// vcvt_f32_f16 (4 fp16 -> 4 fp32 per chunk). For pages < 4
// the tail is handled scalar.
//
// Per-lane: int8 lane_scale -> fp32 / 127. vDSP_vflt8 (int8 ->
// fp32, sign-extending) for the bulk conversion, vDSP_vsdiv
// (divide by 127) for the normalisation. Output is
// [n_pages * TILE640_LANES_PER_PAGE] floats.
//
// v2 is functionally identical to the C scalar loop in the
// dispatch (the scalar path reads each page_scale and
// lane_scale individually). The bulk conversion is the speedup.

void decode_per_row_meta_v2(const void * GGML_RESTRICT page_scales_packed,
                            const void * GGML_RESTRICT lane_scales_packed,
                            int64_t n_pages,
                            float * GGML_RESTRICT page_max_out,
                            float * GGML_RESTRICT lane_scale_out) {
    const uint16_t * ps = (const uint16_t *) page_scales_packed;
    const int8_t   * ls = (const int8_t   *) lane_scales_packed;
    const int64_t n_lanes = n_pages * TILE640_LANES_PER_PAGE;

    // page_max: NEON vcvt_f32_f16, 4 fp16 -> 4 fp32 per chunk.
    int64_t p = 0;
#if GGML_TESSERA_T640_V2_NEON
    for (; p + 4 <= n_pages; p += 4) {
        // Load 4 fp16 (each is a uint16_t) as a 64-bit value, then
        // cast to float16x4_t for vcvt.
        uint64_t bits;
        memcpy(&bits, &ps[p], sizeof(bits));
        float16x4_t vh = vreinterpret_f16_u64(vdup_n_u64(bits));
        float32x4_t vf = vcvt_f32_f16(vh);
        vst1q_f32(&page_max_out[p], vf);
    }
#endif
    for (; p < n_pages; p++) {
        page_max_out[p] = GGML_FP16_TO_FP32(ps[p]);
    }

    // lane_scale: int8 -> fp32, divide by 127.
#if defined(__APPLE__)
    // vDSP_vflt8: int8 -> float (sign-extending). Then /127
    // (vDSP_vsdiv divides by its scalar; the divisor is 127,
    // not 1/127).
    // vDSP_vflt8 takes const char * (sign-extending), so we
    // cast from int8_t * via a uintptr_t roundtrip to silence
    // the pointer-sign warning.
    vDSP_vflt8((const char *) ls, 1, lane_scale_out, 1, (vDSP_Length) n_lanes);
    float div127 = 127.0f;
    vDSP_vsdiv(lane_scale_out, 1, &div127, lane_scale_out, 1, (vDSP_Length) n_lanes);
#else
    for (int64_t i = 0; i < n_lanes; i++) {
        lane_scale_out[i] = ((float) ls[i]) / 127.0f;
    }
#endif
}

// ---------------------------------------------------------------------------
// Function E: apply_act_scale_v2
// ---------------------------------------------------------------------------
//
// y[i] *= fp16_to_fp32(act_scale[i]) (per-input-channel scale,
// n floats). vDSP_vmul elementwise multiply. The C version is
// a per-element scalar loop; v2 is the bulk vDSP call. For
// n <= 4096 the fp16->fp32 scratch is on the stack; for
// n > 4096 we fall back to a per-element scalar loop (the
// scratch would be too large for the stack).
//
// vDSP does not have an fp16->fp32 bulk conversion, so we
// convert with NEON vcvt_f32_f16 (4 fp16 -> 4 fp32 per chunk)
// on Apple, scalar on other platforms. The vDSP_vmul then
// does the bulk multiply into y.

void apply_act_scale_v2(float * GGML_RESTRICT y,
                        const void * GGML_RESTRICT act_scale_packed,
                        int64_t n) {
    const uint16_t * as = (const uint16_t *) act_scale_packed;
#if defined(__APPLE__)
    if (n <= 4096) {
        float scratch[4096] __attribute__((aligned(16)));
        // Convert fp16 act_scale -> fp32 in 4-element NEON
        // chunks; tail is scalar.
        int64_t i = 0;
#if GGML_TESSERA_T640_V2_NEON
        for (; i + 4 <= n; i += 4) {
            uint64_t bits;
            memcpy(&bits, &as[i], sizeof(bits));
            float16x4_t vh = vreinterpret_f16_u64(vdup_n_u64(bits));
            float32x4_t vf = vcvt_f32_f16(vh);
            vst1q_f32(&scratch[i], vf);
        }
#endif
        for (; i < n; i++) {
            scratch[i] = GGML_FP16_TO_FP32(as[i]);
        }
        // Bulk elementwise multiply: y[i] *= scratch[i].
        vDSP_vmul(scratch, 1, y, 1, y, 1, (vDSP_Length) n);
    } else {
        // For very long n, fall back to per-element scalar.
        for (int64_t i = 0; i < n; i++) {
            y[i] *= GGML_FP16_TO_FP32(as[i]);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        y[i] *= GGML_FP16_TO_FP32(as[i]);
    }
#endif
}
