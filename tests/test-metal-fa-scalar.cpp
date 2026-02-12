// Unit test for Metal Flash Attention SCALAR implementation
// Validates GLSL→Metal translation correctness

#include "ggml.h"
#include "ggml-metal.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>

// Reference Flash Attention (CPU)
static void flash_attn_ref(
    const float * Q, const float * K, const float * V,
    float * O, float scale,
    int N, int KV, int D) {

    for (int i = 0; i < N; ++i) {
        float M = -INFINITY;
        float L = 0.0f;
        std::vector<float> O_acc(D, 0.0f);

        for (int j = 0; j < KV; ++j) {
            float S = 0.0f;
            for (int d = 0; d < D; ++d) {
                S += Q[i * D + d] * K[j * D + d];
            }
            S *= scale;

            float M_new = fmaxf(M, S);
            float exp_M = expf(M - M_new);
            float exp_S = expf(S - M_new);

            L = exp_M * L + exp_S;

            for (int d = 0; d < D; ++d) {
                O_acc[d] = exp_M * O_acc[d] + exp_S * V[j * D + d];
            }

            M = M_new;
        }

        for (int d = 0; d < D; ++d) {
            O[i * D + d] = O_acc[d] / L;
        }
    }
}

static bool compare_outputs(const float * ref, const float * test, int size, float tol = 1e-2f) {
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int num_errors = 0;

    for (int i = 0; i < size; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        float rel_diff = diff / (fabsf(ref[i]) + 1e-8f);

        max_diff = fmaxf(max_diff, diff);
        max_rel_diff = fmaxf(max_rel_diff, rel_diff);

        if (diff > tol) {
            if (num_errors < 10) {
                fprintf(stderr, "  Mismatch at [%d]: ref=%.6f test=%.6f diff=%.6f\n",
                        i, ref[i], test[i], diff);
            }
            num_errors++;
        }
    }

    fprintf(stderr, "  Max abs diff: %.6f, max rel diff: %.6f, errors: %d/%d (%.1f%%)\n",
            max_diff, max_rel_diff, num_errors, size, 100.0f * num_errors / size);

    return num_errors < (size / 100);
}

static bool test_flash_attn_scalar(int N, int KV, int D) {
    fprintf(stderr, "\n=== Testing FA SCALAR: N=%d, KV=%d, D=%d ===\n", N, KV, D);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q(N * D);
    std::vector<float> K(KV * D);
    std::vector<float> V(KV * D);
    std::vector<float> O_ref(N * D);
    std::vector<float> O_test(N * D);

    for (auto & x : Q) x = dist(rng);
    for (auto & x : K) x = dist(rng);
    for (auto & x : V) x = dist(rng);

    float scale = 1.0f / sqrtf((float)D);

    fprintf(stderr, "  Running CPU reference...\n");
    flash_attn_ref(Q.data(), K.data(), V.data(), O_ref.data(), scale, N, KV, D);

    fprintf(stderr, "  Running Metal SCALAR via ggml...\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "  ERROR: Failed to initialize ggml context\n");
        return false;
    }

    struct ggml_tensor * q_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    struct ggml_tensor * k_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, KV);
    struct ggml_tensor * v_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, KV);

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q_tensor, k_tensor, v_tensor,
                                                    NULL, scale, 0.0f, 0.0f);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "  ERROR: Failed to initialize Metal backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "  ERROR: Failed to allocate Metal buffer\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(q_tensor, Q.data(), 0, N * D * sizeof(float));
    ggml_backend_tensor_set(k_tensor, K.data(), 0, KV * D * sizeof(float));
    ggml_backend_tensor_set(v_tensor, V.data(), 0, KV * D * sizeof(float));

    fprintf(stderr, "  Computing on Metal...\n");
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  ERROR: Metal computation failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(out, O_test.data(), 0, N * D * sizeof(float));

    bool pass = compare_outputs(O_ref.data(), O_test.data(), N * D);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);

    fprintf(stderr, "  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    fprintf(stderr, "Metal Flash Attention SCALAR Translation Test\n");
    fprintf(stderr, "==============================================\n");

    bool all_pass = true;

    all_pass &= test_flash_attn_scalar(4, 64, 64);
    all_pass &= test_flash_attn_scalar(8, 128, 64);
    all_pass &= test_flash_attn_scalar(16, 256, 128);

    fprintf(stderr, "\n==============================================\n");
    fprintf(stderr, "Overall: %s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_pass ? 0 : 1;
}
