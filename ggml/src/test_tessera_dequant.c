#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

// from ggml-quants.c
extern void quantize_row_tessera_t640_ref(const float * x, void * y, int64_t k);
extern void dequantize_row_tessera_t640(const void * x, float * y, int64_t k);

// minimal stubs for symbols referenced by other functions in ggml-quants.c
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "abort at %s:%d: %s\n", file, line, fmt);
    abort();
}
size_t ggml_type_size(enum ggml_type type) { (void)type; return 0; }
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void)type; (void)ne; return 0; }
const char * ggml_type_name(enum ggml_type type) { (void)type; return "unknown"; }

// bytes needed for one row of k elements in the packed format
static size_t tessera_t640_row_bytes(int64_t k) {
    const int pages = (int)((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    return (size_t)pages * TILE640_WORDS_PER_PAGE * sizeof(uint32_t)
         + (size_t)pages * sizeof(uint16_t)
         + (size_t)pages * TILE640_LANES_PER_PAGE * sizeof(int8_t);
}

int main(void) {
    const int64_t k = TILE640_PAGE_SIZE; // 640
    float * x = (float *) malloc(k * sizeof(float));
    float * y = (float *) malloc(k * sizeof(float));
    void  * packed = malloc(tessera_t640_row_bytes(k));

    // ternary-friendly signal: within each lane, values are 0 or +/- amp
    // two amplitude levels to exercise varying lane scales
    float amax = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        const int lane_idx = (int)(i / TILE640_LANE_SIZE);
        const int pos      = (int)(i % TILE640_LANE_SIZE);
        const float amp    = (lane_idx < 16) ? 2.5f : 4.0f;
        if (pos % 3 == 0) {
            x[i] = 0.0f;
        } else {
            x[i] = (pos % 2 == 0) ? amp : -amp;
        }
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }

    quantize_row_tessera_t640_ref(x, packed, k);
    memset(y, 0, k * sizeof(float));
    dequantize_row_tessera_t640(packed, y, k);

    // check reconstruction error
    float max_err = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        const float err = fabsf(y[i] - x[i]);
        if (err > max_err) max_err = err;
    }
    printf("max |x| = %f, max error = %f, ratio = %f\n", amax, max_err, max_err / amax);
    if (max_err > 0.1f * amax) {
        printf("FAIL: reconstruction error too large\n");
        return 1;
    }

    // verify all trits are in {0, 1, 2}
    const int pages = (int)((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    const uint32_t * words = (const uint32_t *) packed;
    for (int p = 0; p < pages; p++) {
        for (int l = 0; l < TILE640_WORDS_PER_PAGE; l++) {
            uint32_t rem = words[p * TILE640_WORDS_PER_PAGE + l];
            for (int g = 0; g < 4; g++) {
                uint32_t idx = rem % 243;
                rem /= 243;
                for (int d = 0; d < 5; d++) {
                    const uint32_t trit = idx % 3;
                    idx /= 3;
                    if (trit > 2) {
                        printf("FAIL: invalid trit %u at page %d lane %d group %d digit %d\n",
                               trit, p, l, g, d);
                        return 1;
                    }
                }
            }
            if (rem != 0) {
                printf("FAIL: leftover radix-243 remainder %u at page %d lane %d\n", rem, p, l);
                return 1;
            }
        }
    }

    printf("PASS\n");

    free(x);
    free(y);
    free(packed);
    return 0;
}
