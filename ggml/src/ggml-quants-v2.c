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
