// test-gelu-quick.cpp: validate the F32 gelu_quick and geglu_quick CPU kernels against the exact formula.
//
// test-backend-ops uses the CPU backend as its reference, so it cannot see an error in the CPU kernels
// themselves. This test computes gelu_quick(x) = x / (1 + exp(-1.702 x)) in double precision and compares
// every F32 output of the CPU backend with it: dense ranges, values outside the F16 range, values near
// the float limits, and lengths that leave a tail after the vector loop.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Same coefficient as the CPU kernels
static const double GELU_QUICK_COEF_REF = (double) 1.702f;

static double gelu_quick_ref(double x) {
    return x / (1.0 + std::exp(-GELU_QUICK_COEF_REF * x));
}

// Deterministic LCG mapped to [-1, 1]
static uint64_t g_rng = 0x12345678ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

static std::vector<float> make_inputs(void) {
    std::vector<float> x;
    for (int i = 0; i <= 200000; i++) {
        x.push_back(-10.0f + 20.0f * (float) i / 200000.0f);
    }
    for (int i = 0; i <= 2000; i++) {
        x.push_back(-150.0f + 300.0f * (float) i / 2000.0f);
    }
    // outside the F16 range, near the float limits, and subnormal; both signs
    const float edge[] = { 65504.0f, 65520.0f, 1e5f, 1e10f, 1e30f, FLT_MAX, 1e-30f, FLT_MIN, FLT_MIN / 16, 0.0f };
    for (float v : edge) {
        x.push_back(v);
        x.push_back(-v);
    }
    return x;
}

static bool matches(float y, double ref) {
    if (std::isnan(ref)) {
        return std::isnan(y);
    }
    if (std::isinf(ref)) {
        return (double) y == ref;
    }
    return std::fabs((double) y - ref) <= 1e-6 + 1e-6 * std::fabs(ref);
}

// Runs gelu_quick and geglu_quick on x[0, n) and returns the number of outputs that differ from the reference
static int check(const float * x, const float * g, int64_t n) {
    struct ggml_init_params params = {
        /* .mem_size   = */ (size_t) 64 << 20,
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(a->data, x, n * sizeof(float));
    memcpy(b->data, g, n * sizeof(float));

    struct ggml_tensor * y_gelu  = ggml_gelu_quick(ctx, a);
    struct ggml_tensor * y_geglu = ggml_geglu_quick_split(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_gelu);
    ggml_build_forward_expand(gf, y_geglu);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    int    bad     = 0;
    double max_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double r0 = gelu_quick_ref(x[i]);
        const double r1 = r0 * g[i];
        const float  y0 = ((const float *) y_gelu->data)[i];
        const float  y1 = ((const float *) y_geglu->data)[i];
        if (std::isfinite(r0) && std::isfinite(y0)) {
            max_err = std::fmax(max_err, std::fabs((double) y0 - r0) / (1.0 + std::fabs(r0)));
        }
        if (!matches(y0, r0)) {
            if (bad < 3) {
                printf("  gelu_quick(%g) = %g, expected %g\n", x[i], y0, r0);
            }
            bad++;
        }
        if (!matches(y1, r1)) {
            if (bad < 3) {
                printf("  geglu_quick(%g, %g) = %g, expected %g\n", x[i], g[i], y1, r1);
            }
            bad++;
        }
    }
    printf("gelu_quick n=%6d: max err %.2e, %d mismatches %s\n", (int) n, max_err, bad, bad == 0 ? "OK" : "FAIL");

    ggml_free(ctx);
    return bad;
}

int main(void) {
    const std::vector<float> x = make_inputs();
    std::vector<float> g(x.size());
    for (float & v : g) {
        v = frand();
    }

    int fails = 0;

    // every input in one call
    fails += check(x.data(), g.data(), (int64_t) x.size()) != 0;

    // short lengths end on the edge values and leave a tail after the vector loop
    const int64_t lengths[] = { 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 23, 31, 33 };
    for (int64_t n : lengths) {
        const int64_t off = (int64_t) x.size() - n;
        fails += check(x.data() + off, g.data() + off, n) != 0;
    }

    printf(fails == 0 ? "all gelu_quick checks passed\n" : "%d gelu_quick checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
