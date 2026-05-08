// Test for ggml_conv_1d_grouped
//
// Verifies grouped 1D convolution by comparing against manual per-group computation.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static void fill_random_f16(ggml_fp16_t * data, int n) {
    for (int i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        data[i] = ggml_fp32_to_fp16(v);
    }
}

static void fill_random_f32(float * data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

static bool all_close(const float * a, const float * b, int n, float eps = 5e-3f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps) {
            fprintf(stderr, "    mismatch at [%d]: %.6f vs %.6f (diff=%.6f)\n",
                    i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

// Compute grouped conv1d on CPU naively for reference
// kernel (F16): [K, IC_G, OC], input (F32): [L, IC, N], output: [OL, OC, N]
static void conv1d_grouped_ref(
        const ggml_fp16_t * kernel, const float * input, float * output,
        int K, int IC, int OC, int L, int N, int groups, int stride, int padding) {
    int IC_G = IC / groups;
    int OC_G = OC / groups;
    int OL = (L + 2 * padding - K) / stride + 1;

    memset(output, 0, (size_t)OL * OC * N * sizeof(float));

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            for (int oc = 0; oc < OC_G; oc++) {
                int oc_global = g * OC_G + oc;
                for (int ol = 0; ol < OL; ol++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC_G; ic++) {
                        for (int k = 0; k < K; k++) {
                            int il = ol * stride + k - padding;
                            if (il >= 0 && il < L) {
                                int ic_global = g * IC_G + ic;
                                // kernel: [K, IC_G, OC] -> k + ic * K + oc_global * (IC_G * K)
                                float w = ggml_fp16_to_fp32(kernel[k + ic * K + oc_global * (IC_G * K)]);
                                // input: [L, IC, N] -> il + ic_global * L + n * (IC * L)
                                float x = input[il + ic_global * L + n * (IC * L)];
                                sum += w * x;
                            }
                        }
                    }
                    // output: [OL, OC, N] -> ol + oc_global * OL + n * (OC * OL)
                    output[ol + oc_global * OL + n * (OC * OL)] = sum;
                }
            }
        }
    }
}

static bool run_test(const char * label, int IC, int OC, int K, int L, int groups, int stride, int padding) {
    printf("  TEST: %s (IC=%d OC=%d K=%d L=%d G=%d s=%d p=%d)\n",
           label, IC, OC, K, L, groups, stride, padding);

    int IC_G = IC / groups;
    int OL = (L + 2 * padding - K) / stride + 1;

    size_t ctx_size = 256 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // kernel: [K, IC_G, OC] in F16 (like real models)
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, IC_G, OC);
    // input: [L, IC] in F32
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);

    fill_random_f16((ggml_fp16_t *)a->data, K * IC_G * OC);
    fill_random_f32((float *)b->data, L * IC);

    // reference
    std::vector<float> ref(OL * OC);
    conv1d_grouped_ref((ggml_fp16_t *)a->data, (float *)b->data, ref.data(),
                       K, IC, OC, L, 1, groups, stride, padding);

    // ggml
    struct ggml_tensor * result = ggml_conv_1d_grouped(ctx, a, b, stride, padding, 1, groups);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_graph_compute(backend, gf);

    bool ok = true;

    if (result->ne[0] != OL || result->ne[1] != OC) {
        fprintf(stderr, "    FAIL: shape [%lld, %lld], expected [%d, %d]\n",
                (long long)result->ne[0], (long long)result->ne[1], OL, OC);
        ok = false;
    }

    if (ok) {
        ok = all_close((float *)result->data, ref.data(), OL * OC);
    }

    printf("    %s\n", ok ? "PASS" : "FAIL");

    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    srand(42);

    printf("Testing ggml_conv_1d_grouped\n\n");

    int n_pass = 0, n_fail = 0;

    auto check = [&](const char * label, int IC, int OC, int K, int L, int G, int s, int p) {
        if (run_test(label, IC, OC, K, L, G, s, p)) { n_pass++; } else { n_fail++; }
    };

    check("groups=1 (standard conv1d)", 128, 256, 3, 32, 1, 1, 0);
    check("ZAYA1-8B exact params",      1280, 1280, 2, 16, 10, 1, 0);
    check("small 2 groups",             4, 4, 2, 8, 2, 1, 0);
    check("with padding",              8, 8, 2, 16, 4, 1, 1);
    check("IC != OC",                  12, 6, 3, 10, 3, 1, 0);
    check("stride=2",                  8, 8, 2, 16, 4, 2, 0);
    check("longer sequence",           1280, 1280, 2, 128, 10, 1, 0);

    printf("\nResult: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
