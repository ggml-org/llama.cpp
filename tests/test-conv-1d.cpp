// test-conv-1d.cpp: validate ggml_conv_1d against its own single batch result.
//
// conv_1d folds the batch into the rows of the GEMM, so the product is laid out
// as [OL, N, OC] and reaches the documented [OL, OC, N] only after a transpose
// of the two slowest dimensions. The test runs one batched convolution and N
// single batch convolutions sharing the same weight and the same input slices,
// on the CPU backend, and compares every channel of every batch in F32. A wrong
// layout keeps the shape intact and scrambles the values, so the comparison is
// on the contents.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// One geometry: kernel size, input and output channels, input length, batch,
// stride, padding, dilation
struct conv_1d_case {
    int64_t K;
    int64_t IC;
    int64_t OC;
    int64_t L;
    int64_t N;
    int     s0;
    int     p0;
    int     d0;
};

static const conv_1d_case CASES[] = {
    {  3,  4,  5, 17, 1, 1, 1, 1 },  // single batch, the shape every caller already exercises
    {  3,  4,  5, 17, 2, 1, 1, 1 },  // smallest batched case
    {  7,  8, 16, 64, 3, 2, 3, 1 },  // strided with padding, vocoder front end shape
    {  1,  3,  3, 11, 4, 1, 0, 1 },  // K = 1, pointwise mix
    {  5,  2,  7, 31, 5, 3, 2, 2 },  // dilation > 1, kernel not a multiple of stride
    {  4,  6,  1, 23, 2, 1, 0, 1 },  // OC = 1, mono output stage
    {  2,  1,  4,  9, 8, 2, 1, 1 },  // IC = 1, batch larger than every other dimension
};

// Deterministic LCG mapped to [-1, 1]
static uint64_t g_rng = 0x12345678ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

// NMSE of one batch of the packed output against its single batch reference
static double nmse(const float * y, const float * ref, int64_t n) {
    double num = 0.0;
    double den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double a = y[i];
        const double b = ref[i];
        num += (a - b) * (a - b);
        den += b * b;
    }
    return num / (den + 1e-30);
}

int main(void) {
    int fails = 0;

    for (const conv_1d_case & c : CASES) {
        const int64_t OL = (c.L + 2 * c.p0 - c.d0 * (c.K - 1) - 1) / c.s0 + 1;

        struct ggml_init_params params = {
            /* .mem_size   = */ (size_t) 64 << 20,
            /* .mem_base   = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        // One logical weight and one logical input feed both paths
        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.IC, c.OC);
        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.L, c.IC, c.N);
        for (int64_t i = 0; i < ggml_nelements(w); i++) {
            ((float *) w->data)[i] = frand();
        }
        for (int64_t i = 0; i < ggml_nelements(x); i++) {
            ((float *) x->data)[i] = frand();
        }

        struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, c.s0, c.p0, c.d0);

        GGML_ASSERT(y->ne[0] == OL && y->ne[1] == c.OC && y->ne[2] == c.N);
        GGML_ASSERT(ggml_is_contiguous(y));

        // Reference path: the same convolution on one batch at a time
        std::vector<struct ggml_tensor *> ref(c.N);
        for (int64_t n = 0; n < c.N; n++) {
            struct ggml_tensor * slice = ggml_cont(ctx,
                ggml_view_2d(ctx, x, c.L, c.IC, x->nb[1], n * x->nb[2]));
            ref[n] = ggml_conv_1d(ctx, w, slice, c.s0, c.p0, c.d0);
        }

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        for (int64_t n = 0; n < c.N; n++) {
            ggml_build_forward_expand(gf, ref[n]);
        }
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        // Worst batch decides the case
        double worst = 0.0;
        for (int64_t n = 0; n < c.N; n++) {
            const float * yn = (const float *) y->data + n * OL * c.OC;
            const double e = nmse(yn, (const float *) ref[n]->data, OL * c.OC);
            if (e > worst) {
                worst = e;
            }
        }

        // Both paths share the F16 im2col and the same dot products, so only
        // the row count of the GEMM differs
        const bool ok = worst <= 1e-7;
        if (!ok) {
            fails++;
        }
        printf("conv_1d K=%d IC=%2d OC=%2d L=%3d N=%d s0=%d p0=%d d0=%d: nmse=%.2e %s\n",
            (int) c.K, (int) c.IC, (int) c.OC, (int) c.L, (int) c.N, c.s0, c.p0, c.d0,
            worst, ok ? "OK" : "FAIL");

        ggml_free(ctx);
    }

    printf(fails == 0 ? "all conv_1d checks passed\n" : "%d conv_1d checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
