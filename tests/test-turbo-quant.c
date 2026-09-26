#include "ggml.h"

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, int64_t k);
extern void dequantize_row_turbo3_0(const void * x, float * y, int64_t k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, int64_t k);
extern void dequantize_row_turbo4_0(const void * x, float * y, int64_t k);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

_Static_assert(GGML_TYPE_TURBO2_0 == 43 && GGML_TYPE_TURBO3_0 == 44 && GGML_TYPE_TURBO4_0 == 45 &&
               GGML_TYPE_TQ3_1S == 46 && GGML_TYPE_TQ4_1S == 47 && GGML_TYPE_Q8_CR == 48 &&
               GGML_TYPE_Q5_CR == 49 && GGML_TYPE_Q6_CR == 50, "stable serialized type IDs");

/* Must match GGML_TQ_DOT_CHUNK in ggml/src/ggml-cpu/ggml-cpu.c. */
#define TQ_DOT_CHUNK 256

/* Block-boundary invariant behind the chunked TurboQuant vec_dot kernels.
 *
 * ggml_vec_dot_{turbo,tq}_* stage their dequant through a fixed-size chunk
 * instead of a malloc'd whole-row buffer. That is only correct because every
 * TurboQuant row dequant is a pure per-block loop with no cross-block state:
 * dequantizing [0, k) in one call must be bit-identical to dequantizing it in
 * block-aligned pieces with the source pointer advanced by
 * (piece / blck_size) * type_size.
 *
 * This asserts exactly that, at row lengths that straddle the chunk size, so a
 * future row dequant that carries state across blocks -- or a wrong stride in
 * the vec_dot pointer advance -- fails here rather than silently corrupting a
 * CPU-offloaded matmul. */
static int check_chunked_dequant(enum ggml_type type, const char * name, int64_t k) {
    const struct ggml_type_traits * tr = ggml_get_type_traits(type);

    const int64_t blck = tr->blck_size;
    if (k % blck != 0) {
        printf("  %-9s k=%-5lld SKIP (not a whole number of blocks)\n", name, (long long) k);
        return 0;
    }

    float * src   = malloc((size_t) k * sizeof(float));
    void  * q     = malloc(ggml_row_size(type, k));
    float * whole = malloc((size_t) k * sizeof(float));
    float * piece = malloc((size_t) k * sizeof(float));

    assert(src && q && whole && piece);

    for (int64_t i = 0; i < k; i++) {
        src[i] = sinf((float) i * 0.037f) * 3.0f;
    }
    ggml_quantize_chunk(type, src, q, 0, 1, k, NULL);

    tr->to_float(q, whole, k);

    const int64_t chunk = (TQ_DOT_CHUNK / blck) * blck;
    const char * p = (const char *) q;
    for (int64_t i = 0; i < k; i += chunk) {
        const int64_t nc = (chunk < k - i) ? chunk : (k - i);
        tr->to_float(p, piece + i, nc);
        p += (nc / blck) * tr->type_size;
    }

    int64_t bad = -1;
    for (int64_t i = 0; i < k; i++) {
        if (memcmp(&whole[i], &piece[i], sizeof(float)) != 0) {
            bad = i;
            break;
        }
    }

    if (bad >= 0) {
        printf("  %-9s k=%-5lld FAIL at [%lld]: whole=%.9g chunked=%.9g\n",
               name, (long long) k, (long long) bad,
               (double) whole[bad], (double) piece[bad]);
    } else {
        printf("  %-9s k=%-5lld ok (blck=%lld, chunk=%lld)\n",
               name, (long long) k, (long long) blck, (long long) chunk);
    }

    free(src);
    free(q);
    free(whole);
    free(piece);
    return bad >= 0 ? 1 : 0;
}

int main(void) {
    const int d = 128;
    char buf[256];
    float input[128], output[128];
    float mse, cosv, ni, no;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    /* Test 1: basis vector
     *
     * dequantize_row_turbo3_0 leaves output in the WHT-rotated domain (Q is
     * also rotated by the graph, so <Q_rot, K_rot> yields correct attention
     * scores without an explicit inverse). To verify the round-trip, apply
     * the inverse WHT before comparing against the original input. */
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    printf("  In:  [%.6f, %.6f, %.6f, %.6f]\n",  (double)(input[0]), (double)(input[1]), (double)(input[2]), (double)(input[3]));
    printf("  Out: [%.6f, %.6f, %.6f, %.6f]\n",  (double)(output[0]), (double)(output[1]), (double)(output[2]), (double)(output[3]));
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f OutNorm=%.6f\n\n",  (double)(mse/d), (double)(ni > 0 && no > 0 ? cosv/sqrtf(ni)/sqrtf(no) : 0), (double)(sqrtf(no)));

    /* Test 2: large-norm vector */
    for (int i = 0; i < d; i++) input[i] = sinf(i*0.1f+0.5f) * 10.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 2 (turbo3): sin*10\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n",  (double)(input[0]), (double)(input[1]), (double)(input[2]), (double)(input[3]));
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n",  (double)(output[0]), (double)(output[1]), (double)(output[2]), (double)(output[3]));
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f InNorm=%.2f OutNorm=%.2f\n\n",  (double)(mse/d), (double)(cosv/sqrtf(ni)/sqrtf(no)), (double)(sqrtf(ni)), (double)(sqrtf(no)));

    /* Test 3: turbo4
     *
     * Same convention as turbo3: dequant leaves output in the rotated domain
     * (see comment in dequantize_row_turbo4_0 @ ggml-turbo-quant.c). Apply
     * the inverse WHT before comparing. */
    for (int i = 0; i < d; i++) input[i] = cosf(i*0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 3 (turbo4): cos*5\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n",  (double)(input[0]), (double)(input[1]), (double)(input[2]), (double)(input[3]));
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n",  (double)(output[0]), (double)(output[1]), (double)(output[2]), (double)(output[3]));
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f\n\n",  (double)(mse/d), (double)(cosv/sqrtf(ni)/sqrtf(no)));

    /* Test 4: chunk-boundary invariant for every type with a chunked vec_dot */
    printf("Test 4: chunked dequant == whole-row dequant\n");
    {
        const enum ggml_type types[] = {
            GGML_TYPE_TQ3_1S, GGML_TYPE_TQ4_1S,
            GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0,
        };
        const char * names[] = { "tq3_1s", "tq4_1s", "turbo2_0", "turbo3_0", "turbo4_0" };
        /* straddle TQ_DOT_CHUNK from both sides, plus a realistic row */
        const int64_t lens[] = { 128, 256, 384, 512, 4096 };

        int failures = 0;
        for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
            for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
                failures += check_chunked_dequant(types[t], names[t], lens[i]);
            }
        }
        printf("\n");
        if (failures) {
            printf("=== FAILED: %d chunk-boundary mismatches ===\n", failures);
            return 1;
        }
    }

    printf("=== Done ===\n");
    return 0;
}
