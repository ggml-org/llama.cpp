// test-col2im-1d.cpp: validate GGML_OP_COL2IM_1D against ggml_conv_transpose_1d.
//
// A ConvTranspose1d factorizes as a GEMM followed by an overlap-add:
//   conv_transpose_1d(w, x)  equals  col2im_1d(mul_mat(w_perm, x_t), s0, OC, p0)
// with w_perm the [IC, K*OC] permutation of the [K, OC, IC] kernel and x_t the
// [IC, T_in] transpose of the [T_in, IC] input. The test derives both alternative
// layouts from one logical weight and one logical input with graph ops only
// (permute + cont + reshape), runs the two paths on the CPU backend, and compares
// them in F32. The F16 and BF16 kernels are exercised by casting the column
// matrix before the scatter. Cropping (p0 > 0) is checked against the shifted
// slice of the uncropped reference, which conv_transpose_1d cannot express.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// One geometry: kernel size, output channels, input length, stride, crop
struct col2im_case {
    int64_t K;
    int64_t OC;
    int64_t T_in;
    int     s0;
    int     p0;
};

// Mirrors the eval grid of test-backend-ops
static const col2im_case CASES[] = {
    { 16, 32, 197, 8, 0 },  // kernel = 2*stride, DAC upsampling shape
    {  4,  3,   7, 2, 0 },
    {  1,  5,  13, 1, 0 },  // stride 1, no overlap
    {  6,  4,  11, 3, 1 },  // with cropping
    {  2,  3,   9, 3, 0 },  // kernel < stride, gap positions are zeroed
    {  5,  4,  11, 2, 0 },  // kernel not a multiple of stride, alternating overlap
    {  8,  4,  13, 4, 2 },  // padding = stride/2, DAC causal cropping
    {  4,  3,   1, 2, 0 },  // single column, pure kernel unfold
    { 16,  1, 197, 8, 0 },  // OC = 1, mono output stage
    {  1,  5,  13, 3, 0 },  // K = 1 with stride > 1, sparse scatter
    {  8,  2,   3, 2, 5 },  // cropping eats most of the signal, T_out = 2
};

// Input channels of the GEMM, shared by every case
static const int64_t IC = 7;

// Deterministic LCG mapped to [-1, 1]
static uint64_t g_rng = 0x12345678ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

// Read a F32/F16/BF16 tensor back as a flat F32 vector
static std::vector<float> tensor_to_f32(const struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) t->data)[i]);
        }
    } else {
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) t->data)[i]);
        }
    }
    return out;
}

// NMSE of the cropped output against the p0 shifted slice of the full reference
static double nmse_cropped(const float * y, const float * ref, int64_t T_out, int64_t T_ref, int64_t OC, int p0) {
    double num = 0.0;
    double den = 0.0;
    for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t t = 0; t < T_out; t++) {
            const double a = y  [t      + oc * T_out];
            const double b = ref[t + p0 + oc * T_ref];
            num += (a - b) * (a - b);
            den += b * b;
        }
    }
    return num / (den + 1e-30);
}

int main(void) {
    int fails = 0;

    for (const col2im_case & c : CASES) {
        const int64_t T_ref = (c.T_in - 1) * c.s0 + c.K;
        const int64_t T_out = T_ref - 2 * c.p0;

        struct ggml_init_params params = {
            /* .mem_size   = */ (size_t) 64 << 20,
            /* .mem_base   = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        // One logical weight and one logical input feed both paths
        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.OC, IC);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.T_in, IC);
        for (int64_t i = 0; i < ggml_nelements(w); i++) {
            ((float *) w->data)[i] = frand();
        }
        for (int64_t i = 0; i < ggml_nelements(x); i++) {
            ((float *) x->data)[i] = frand();
        }

        // Reference path: the native op, uncropped
        struct ggml_tensor * y_ref = ggml_conv_transpose_1d(ctx, w, x, c.s0, 0, 1);

        // Decomposed path: [K, OC, IC] -> [IC, K, OC] -> [IC, K*OC], k fastest inside each oc block
        struct ggml_tensor * w_perm = ggml_cont(ctx, ggml_permute(ctx, w, 1, 2, 0, 3));
        w_perm                      = ggml_reshape_2d(ctx, w_perm, IC, c.K * c.OC);
        struct ggml_tensor * x_t    = ggml_cont(ctx, ggml_transpose(ctx, x));
        struct ggml_tensor * col    = ggml_mul_mat(ctx, w_perm, x_t);
        struct ggml_tensor * y32    = ggml_col2im_1d(ctx, col, c.s0, (int) c.OC, c.p0);

        // Half precision kernels: the same columns cast before the scatter
        struct ggml_tensor * y16 = ggml_col2im_1d(ctx, ggml_cast(ctx, col, GGML_TYPE_F16),  c.s0, (int) c.OC, c.p0);
        struct ggml_tensor * ybf = ggml_col2im_1d(ctx, ggml_cast(ctx, col, GGML_TYPE_BF16), c.s0, (int) c.OC, c.p0);

        GGML_ASSERT(y_ref->ne[0] == T_ref && y_ref->ne[1] == c.OC);
        GGML_ASSERT(y32->ne[0] == T_out && y32->ne[1] == c.OC);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y_ref);
        ggml_build_forward_expand(gf, y32);
        ggml_build_forward_expand(gf, y16);
        ggml_build_forward_expand(gf, ybf);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        const std::vector<float> f32 = tensor_to_f32(y32);
        const std::vector<float> f16 = tensor_to_f32(y16);
        const std::vector<float> fbf = tensor_to_f32(ybf);
        const float * ref = (const float *) y_ref->data;

        const double e32 = nmse_cropped(f32.data(), ref, T_out, T_ref, c.OC, c.p0);
        const double e16 = nmse_cropped(f16.data(), ref, T_out, T_ref, c.OC, c.p0);
        const double ebf = nmse_cropped(fbf.data(), ref, T_out, T_ref, c.OC, c.p0);

        // Same thresholds as test-backend-ops: 1e-7 full precision, 5e-4 half
        const bool ok = e32 <= 1e-7 && e16 <= 5e-4 && ebf <= 5e-4;
        if (!ok) {
            fails++;
        }
        printf("col2im_1d K=%2d OC=%2d T_in=%3d s0=%d p0=%d: nmse f32=%.2e f16=%.2e bf16=%.2e %s\n",
            (int) c.K, (int) c.OC, (int) c.T_in, c.s0, c.p0, e32, e16, ebf, ok ? "OK" : "FAIL");

        ggml_free(ctx);
    }

    printf(fails == 0 ? "all col2im_1d checks passed\n" : "%d col2im_1d checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
