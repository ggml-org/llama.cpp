// Gate for the BPOSIT8 fast dot (the type's default vec_dot): BPOSIT8 weights x BF16 activations,
// fp32-accumulated (AVX-512 BF16 / AVX-512 / NEON BF16 / scalar). Reference: a double-precision
// dot of the dequantised weights and the bf16-rounded activations. Also checks the lattice table
// shared with the GPU backends (ggml-common.h) against the type's own dequantisation.
// The exact W8A8 quire path (the deterministic profile's vec_dot) has its own golden gate
// (test-bposit8-quire.c).
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_IMPL_C
#include "../ggml/src/ggml-common.h"     // bp8_lut_f, the lattice table shared with the GPU backends
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static unsigned long long st = 0x2026091101ULL;
static float frand(void) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((float) ((st >> 33) & 0xFFFFF) / (float) 0x80000) * 2.0f - 2.0f;
}

int main(void) {
    ggml_cpu_init();
    const enum ggml_type T = GGML_TYPE_BPOSIT8;
    const struct ggml_type_traits     * tt  = ggml_get_type_traits(T);
    const struct ggml_type_traits_cpu * ttc = ggml_get_type_traits_cpu(T);
    if (!ttc || !ttc->vec_dot || !ttc->from_float || !tt || !tt->to_float) { printf("FAIL: BPOSIT8 traits missing\n"); return 1; }
    const struct ggml_type_traits_cpu * tty = ggml_get_type_traits_cpu(ttc->vec_dot_type);
    printf("vec_dot_type = %s (%s activations)\n", ggml_type_name(ttc->vec_dot_type), ttc->vec_dot_type == GGML_TYPE_BF16 ? "bf16" : "posit-coded");
    const int qk = ggml_blck_size(T);
    const size_t bsz = ggml_type_size(T);
    const size_t ysz = ggml_type_size(ttc->vec_dot_type);          // bytes per activation element (bf16: 2) or per block

    int fails = 0, checks = 0;
    {   // the shared lattice table (ggml-common.h, used by the Metal kernels) must equal the type's own dequantisation
        uint8_t blk[8 * 33]; float out[256];
        for (int b = 0; b < 8; b++) { blk[b * 33] = 0; for (int j = 0; j < 32; j++) blk[b * 33 + 1 + j] = (uint8_t) (b * 32 + j); }
        tt->to_float(blk, out, 256);
        int bad = 0;
        for (int c = 0; c < 256; c++) if (memcmp(&out[c], &bp8_lut_f[c], 4) != 0) { if (bad < 4) printf("  bp8_lut_f[%d] = %.9g, to_float gives %.9g\n", c, (double) bp8_lut_f[c], (double) out[c]); bad++; }
        printf("%s: shared lattice table bp8_lut_f vs to_float (%d/256 differ)\n", bad ? "FAIL" : "PASS", bad);
        checks += 256; fails += bad;
    }
    double worst = 0.0;
    for (int trial = 0; trial < 400; trial++) {
        const int nb = 1 + (trial % 64);                 // K = 32 .. 2048
        const int n  = nb * qk;
        const int pattern = trial % 4;                   // 0 plain, 1 scale-spread, 2 sparse, 3 tiny+huge blocks
        float * x0 = malloc(n * sizeof(float)), * x1 = malloc(n * sizeof(float));
        float * y0 = malloc(n * sizeof(float)), * y1 = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            float sx = 1.0f, sy = 1.0f;
            if (pattern == 1) { sx = ldexpf(1.0f, (i / qk) % 9 - 4); sy = ldexpf(1.0f, (i / qk) % 5 - 2); }
            if (pattern == 3) { sx = ((i / qk) & 1) ? 1e-12f : 1e6f; sy = ((i / qk) & 2) ? 3e-7f : 2e4f; }
            x0[i] = frand() * sx; x1[i] = frand() * sx; y0[i] = frand() * sy; y1[i] = frand() * sy;
            if (pattern == 2 && (i % 5)) { x0[i] = 0.0f; y1[i] = 0.0f; }
        }
        // weight rows contiguous at stride nb*bsz (the 2x2 tile addresses row 1 as row 0 + bx);
        // activation rows in the vec_dot_type at stride n*ysz (bf16) or nb*ysz (posit-coded)
        const size_t ystride = ttc->vec_dot_type == GGML_TYPE_BF16 ? (size_t) n * ysz : (size_t) nb * ysz;
        uint8_t * qx = malloc(2 * nb * bsz), * qy = malloc(2 * ystride);
        void * qx0 = qx, * qx1 = qx + nb * bsz, * qy0 = qy, * qy1 = qy + ystride;
        ttc->from_float(x0, qx0, n); ttc->from_float(x1, qx1, n);
        tty->from_float(y0, qy0, n); tty->from_float(y1, qy1, n);

        // reference in double: dequantised weights x the activations as the dot sees them
        float * dx0 = malloc(n * sizeof(float)), * dx1 = malloc(n * sizeof(float));
        float * dy0 = malloc(n * sizeof(float)), * dy1 = malloc(n * sizeof(float));
        tt->to_float(qx0, dx0, n); tt->to_float(qx1, dx1, n);
        ggml_get_type_traits(ttc->vec_dot_type)->to_float(qy0, dy0, n);
        ggml_get_type_traits(ttc->vec_dot_type)->to_float(qy1, dy1, n);
        double ref[4] = { 0, 0, 0, 0 }, absum[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < n; i++) {
            const double p00 = (double) dx0[i] * (double) dy0[i], p10 = (double) dx1[i] * (double) dy0[i], p01 = (double) dx0[i] * (double) dy1[i], p11 = (double) dx1[i] * (double) dy1[i];
            ref[0] += p00; ref[1] += p10; ref[2] += p01; ref[3] += p11;
            absum[0] += fabs(p00); absum[1] += fabs(p10); absum[2] += fabs(p01); absum[3] += fabs(p11);
        }

        // fast path: single dots and the 2x2 tile, s[r + bs*c]
        float one[4], tile[4] = { 0, 0, 0, 0 };
        ttc->vec_dot(n, &one[0], 0, qx0, 0, qy0, 0, 1);
        ttc->vec_dot(n, &one[1], 0, qx1, 0, qy0, 0, 1);
        ttc->vec_dot(n, &one[2], 0, qx0, 0, qy1, 0, 1);
        ttc->vec_dot(n, &one[3], 0, qx1, 0, qy1, 0, 1);
        const int have_tile = ttc->nrows >= 2;
        if (have_tile) {
            float st4[4];
            ttc->vec_dot(n, st4, 2, qx0, nb * bsz, qy0, ystride, 2);
            tile[0] = st4[0]; tile[1] = st4[1]; tile[2] = st4[2]; tile[3] = st4[3];
        }
        for (int k = 0; k < (have_tile ? 8 : 4); k++) {
            const float  got = k < 4 ? one[k] : tile[k - 4];
            const double r   = ref[k & 3];
            double scale = fabs(r); if (absum[k & 3] > scale) scale = absum[k & 3];
            const double err = fabs((double) got - r);
            const double rel = scale > 0 ? err / scale : err;
            if (rel > worst) worst = rel;
            checks++;
            // fp32 summation of up to 2048 exact products: error <= ~n * 2^-24 of the absolute-term sum
            if (rel > 4e-5 || isnan(got)) {
                if (fails < 8) printf("  trial %d k=%d n=%d pattern=%d: fast %.9g ref %.9g rel %.3g\n", trial, k, n, pattern, (double) got, r, rel);
                fails++;
            }
        }
        free(x0); free(x1); free(y0); free(y1); free(qx); free(qy); free(dx0); free(dx1); free(dy0); free(dy1);
    }
    printf("%s: %d checks, %d out of tolerance, worst relative error %.3g (tolerance 4e-5 of the absolute-term sum)\n",
           fails ? "FAIL" : "PASS", checks, fails, worst);
    return fails ? 1 : 0;
}
