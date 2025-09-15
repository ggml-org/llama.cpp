// Deterministic MatMul invariance and cross-run tests for CUDA backend

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void set_env_deterministic() {
#if defined(_WIN32)
    SetEnvironmentVariableA("GGML_DETERMINISTIC", "1");
#else
    setenv("GGML_DETERMINISTIC", "1", 1);
#endif
}

enum class DType { F32, F16, BF16 };

struct MatOut {
    std::vector<float> data; // flattened [M,B] row-major by rows (row 0 first: M floats)
    int64_t M = 0;
    int64_t B = 0;
};

static void fill_weights(void *dst, DType dt, const std::vector<float> &w, int64_t M, int64_t K) {
    if (dt == DType::F32) {
        std::memcpy(dst, w.data(), sizeof(float)*w.size());
        return;
    }
    if (dt == DType::F16) {
        std::vector<ggml_fp16_t> tmp((size_t)M*K);
        for (int64_t r = 0; r < M; ++r) {
            ggml_fp32_to_fp16_row(&w[(size_t)r*K], &tmp[(size_t)r*K], K);
        }
        std::memcpy(dst, tmp.data(), tmp.size()*sizeof(tmp[0]));
        return;
    }
    // BF16
    std::vector<ggml_bf16_t> tmp((size_t)M*K);
    for (int64_t r = 0; r < M; ++r) {
        ggml_fp32_to_bf16_row(&w[(size_t)r*K], &tmp[(size_t)r*K], K);
    }
    std::memcpy(dst, tmp.data(), tmp.size()*sizeof(tmp[0]));
}

static MatOut run_matmul_graph(ggml_backend_t backend, DType dt_w, int64_t M, int64_t K, int64_t B,
                               const std::vector<float> &W_f32, const std::vector<float> &X_f32) {
    ggml_init_params ip = {
        ggml_tensor_overhead()*64 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_type tw = GGML_TYPE_F32;
    if (dt_w == DType::F16) tw = GGML_TYPE_F16;
    if (dt_w == DType::BF16) tw = GGML_TYPE_BF16;

    ggml_tensor * W = ggml_new_tensor_2d(ctx, tw, /*ne0=K*/K, /*ne1=M*/M);
    ggml_set_name(W, "W");
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=K*/K, /*ne1=B*/B);
    ggml_set_name(X, "X");
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X); // Y: [M,B]
    ggml_set_name(Y, "Y");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); throw std::runtime_error("alloc tensors failed"); }

    // Set weights and inputs
    const size_t nW = (size_t)M*K;
    const size_t nX = (size_t)K*B;
    if (W_f32.size() != nW || X_f32.size() != nX) {
        ggml_backend_buffer_free(buf); ggml_free(ctx);
        throw std::runtime_error("bad input sizes");
    }

    // Pack W into the tensor dtype layout
    std::vector<uint8_t> W_bytes(ggml_nbytes(W));
    fill_weights(W_bytes.data(), dt_w, W_f32, M, K);
    ggml_backend_tensor_set(W, W_bytes.data(), 0, W_bytes.size());
    ggml_backend_tensor_set(X, X_f32.data(), 0, X_f32.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }

    MatOut out; out.M = M; out.B = B; out.data.resize((size_t)M*B);
    ggml_backend_tensor_get(Y, out.data.data(), 0, out.data.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static bool bytes_equal(const float *a, const float *b, size_t n) {
    return std::memcmp(a, b, n*sizeof(float)) == 0;
}

// ---- MUL_MAT_ID (Mixture-of-Experts) helpers ----

struct MatIdOut {
    std::vector<float> data; // flattened [M, n_e_used, T]
    int64_t M=0, EU=0, T=0;
};

static MatIdOut run_matmul_id_graph(ggml_backend_t backend, DType dt_w, int64_t M, int64_t K, int64_t E, int64_t T, int64_t EU,
                                    const std::vector<float> &W_f32, const std::vector<float> &X_f32, const std::vector<int32_t> &ids_host) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_type tw = GGML_TYPE_F32;
    if (dt_w == DType::F16) tw = GGML_TYPE_F16;
    if (dt_w == DType::BF16) tw = GGML_TYPE_BF16;

    // as: [K, M, E]
    ggml_tensor * as = ggml_new_tensor_3d(ctx, tw, /*ne0=*/K, /*ne1=*/M, /*ne2=*/E);
    ggml_set_name(as, "as");
    // b: [K, EU, T]
    ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, /*ne0=*/K, /*ne1=*/EU, /*ne2=*/T);
    ggml_set_name(b, "b");
    // ids: [EU, T]
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, /*ne0=*/EU, /*ne1=*/T);
    ggml_set_name(ids, "ids");

    ggml_tensor * y = ggml_mul_mat_id(ctx, as, b, ids); // [M, EU, T]
    ggml_set_name(y, "y");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); throw std::runtime_error("alloc tensors failed"); }

    // pack weights
    std::vector<uint8_t> W_bytes(ggml_nbytes(as));
    // layout is expert-major by ne2, then rows by ne1, then cols by ne0
    // We supplied W_f32 as concatenated experts already
    fill_weights(W_bytes.data(), dt_w, W_f32, M*E, K); // treat [E*M, K]
    ggml_backend_tensor_set(as, W_bytes.data(), 0, W_bytes.size());

    // inputs and ids
    ggml_backend_tensor_set(b,   X_f32.data(),   0, X_f32.size()*sizeof(float));
    ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size()*sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx);
        throw std::runtime_error("graph compute failed (mul_mat_id)");
    }

    MatIdOut out; out.M = M; out.EU = EU; out.T = T; out.data.resize((size_t)M*EU*T);
    ggml_backend_tensor_get(y, out.data.data(), 0, out.data.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static int test_backend_matmul_id_invariance(ggml_backend_t backend) {
    std::mt19937 rng(4159);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int64_t M = 128;
    const int64_t K = 1024; // even K for MMVF
    const int64_t E = 4;    // num experts
    const int64_t EU = 2;   // experts used per token

    // Prepare per-expert weights W_f32 as [E * M, K] stacked by expert
    std::vector<float> W((size_t)E*M*K);
    for (float &v : W) v = dist(rng);

    const DType dtypes[] = { DType::F32, DType::F16, DType::BF16 };
    const int    Ts[]    = {1, 4, 9, 16};

    for (DType dt : dtypes) {
        // token 0 base inputs
        std::vector<float> xb0((size_t)K*EU); // [K, EU, 1]
        for (float &v : xb0) v = dist(rng);

        // ids for T=1 (select experts [0,1] for token 0)
        std::vector<int32_t> ids1((size_t)EU*1);
        ids1[0] = 0; ids1[1] = 1;
        auto y1 = run_matmul_id_graph(backend, dt, M, K, E, /*T=*/1, EU, W, xb0, ids1);

        for (int T : Ts) {
            // Build input b: [K, EU, T] with col0 matching xb0
            std::vector<float> Xb((size_t)K*EU*T);
            // copy token 0
            std::copy(xb0.begin(), xb0.end(), Xb.begin());
            // fill other tokens
            for (int t = 1; t < T; ++t) {
                for (int64_t eu = 0; eu < EU; ++eu) {
                    for (int64_t r = 0; r < K; ++r) {
                        Xb[(size_t)t*K*EU + eu*K + r] = dist(rng);
                    }
                }
            }
            // ids: [EU, T], token0 uses [0,1], others random in [0,E)
            std::vector<int32_t> ids((size_t)EU*T);
            ids[0] = 0; ids[1] = 1;
            for (int t = 1; t < T; ++t) {
                for (int eu = 0; eu < EU; ++eu) ids[t*EU + eu] = rng()%E;
            }

            auto yb = run_matmul_id_graph(backend, dt, M, K, E, T, EU, W, Xb, ids);
            // Compare the first token slice [M, EU, token0] with y1
            const float *a = y1.data.data();
            const float *b = yb.data.data();
            if (!bytes_equal(a, b, (size_t)M*EU)) {
                std::cerr << "[FAIL] mul_mat_id batch invariance: dt=" << (int)dt << " T=" << T << "\n";
                return 3;
            }
        }
    }

    return 0;
}

static int test_backend_matmul_invariance(ggml_backend_t backend) {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Shapes to probe (all even K to satisfy MMVF requirements)
    const int64_t Ms[] = {256, 512};
    const int64_t Ks[] = {1024, 4096};
    const int     Bs[] = {2, 4, 7, 8, 16, 17, 33, 64}; // add 17 to straddle mmf N<=16 threshold

    // Dtypes to test
    const DType dtypes[] = { DType::F32, DType::F16, DType::BF16 };

    for (DType dt : dtypes) {
        for (int64_t M : Ms) {
            for (int64_t K : Ks) {
                // Fixed weights per shape
                std::vector<float> W((size_t)M*K);
                for (float &v : W) v = dist(rng);

                // Base input column (B=1)
                std::vector<float> x0((size_t)K);
                for (float &v : x0) v = dist(rng);

                // B=1
                std::vector<float> X1 = x0; // [K,1]
                auto y1 = run_matmul_graph(backend, dt, M, K, /*B=*/1, W, X1);

                for (int B : Bs) {
                    std::vector<float> Xb((size_t)K*B);
                    std::copy(x0.begin(), x0.end(), Xb.begin());
                    for (int c = 1; c < B; ++c) {
                        for (int64_t r = 0; r < K; ++r) Xb[(size_t)c*K + r] = dist(rng);
                    }
                    auto yb = run_matmul_graph(backend, dt, M, K, B, W, Xb);
                    if (!bytes_equal(y1.data.data(), yb.data.data(), (size_t)M)) {
                        std::cerr << "[FAIL] batch invariance: dt=" << (int)dt
                                  << " M=" << M << " K=" << K << " B=" << B << " differ on col0\n";
                        return 1;
                    }
                }

                // Cross-run determinism for a tougher case
                {
                    const int B = 33;
                    std::vector<float> Xb((size_t)K*B);
                    rng.seed(2025 + (int)M + (int)K);
                    for (float &v : Xb) v = dist(rng);
                    auto a = run_matmul_graph(backend, dt, M, K, B, W, Xb);
                    auto b = run_matmul_graph(backend, dt, M, K, B, W, Xb);
                    if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
                        std::cerr << "[FAIL] cross-run determinism: dt=" << (int)dt
                                  << " M=" << M << " K=" << K << "\n";
                        return 2;
                    }
                }
            }
        }
    }

    return 0;
}

// Additional light-weight probes exercising odd-K and unsorted expert IDs for MUL_MAT_ID.
static int test_edge_probes_minimal(ggml_backend_t backend) {
    std::mt19937 rng(9001);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Near-odd K case (use even K to satisfy MMVF requirement) to exercise non-ideal alignment in deterministic fallback
    {
        const int64_t M = 256, K = 1538, B = 17; // even K required by MMVF
        std::vector<float> W((size_t)M*K), X((size_t)K*B);
        for (float &v : W) v = dist(rng);
        for (float &v : X) v = dist(rng);
        // Compare first column for B=1 vs B=17
        std::vector<float> X1((size_t)K);
        std::copy(X.begin(), X.begin()+K, X1.begin());
        auto y1 = run_matmul_graph(backend, DType::F32, M, K, 1, W, X1);
        auto yb = run_matmul_graph(backend, DType::F32, M, K, B, W, X);
        if (!bytes_equal(y1.data.data(), yb.data.data(), (size_t)M)) {
            std::cerr << "[FAIL] odd-K batch invariance: M=256 K=1537 B=17 differ on col0\n";
            return 20;
        }
        // Cross-run determinism on the same odd-K input
        auto a = run_matmul_graph(backend, DType::F32, M, K, B, W, X);
        auto b = run_matmul_graph(backend, DType::F32, M, K, B, W, X);
        if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
            std::cerr << "[FAIL] odd-K cross-run determinism differs\n";
            return 21;
        }
    }

    // Unsorted expert IDs for MUL_MAT_ID
    {
        const int64_t M = 128, K = 1024, E = 4, EU = 2;
        std::vector<float> W((size_t)E*M*K);
        for (float &v : W) v = dist(rng);

        // token0 ids unsorted [2,0]
        std::vector<int32_t> ids1 = {2, 0};
        std::vector<float> xb0((size_t)K*EU);
        for (float &v : xb0) v = dist(rng);
        auto y1 = run_matmul_id_graph(backend, DType::F32, M, K, E, /*T=*/1, EU, W, xb0, ids1);

        const int Ts[] = {4, 9};
        for (int T : Ts) {
            std::vector<float> Xb((size_t)K*EU*T);
            // token0 copy
            std::copy(xb0.begin(), xb0.end(), Xb.begin());
            // other tokens random
            for (int t = 1; t < T; ++t) {
                for (int64_t eu = 0; eu < EU; ++eu) {
                    for (int64_t r = 0; r < K; ++r) Xb[(size_t)t*K*EU + eu*K + r] = dist(rng);
                }
            }
            std::vector<int32_t> ids((size_t)EU*T);
            // token0 fixed unsorted
            ids[0] = 2; ids[1] = 0;
            for (int t = 1; t < T; ++t) {
                ids[t*EU + 0] = rng()%E;
                ids[t*EU + 1] = rng()%E;
            }
            auto yb = run_matmul_id_graph(backend, DType::F32, M, K, E, T, EU, W, Xb, ids);
            if (!bytes_equal(y1.data.data(), yb.data.data(), (size_t)M*EU)) {
                std::cerr << "[FAIL] mul_mat_id unsorted ids batch invariance: T=" << T << "\n";
                return 22;
            }
        }
    }

    return 0;
}

int main() {
    set_env_deterministic();
    ggml_backend_load_all();

    size_t n_dev = ggml_backend_dev_count();
    if (n_dev == 0) {
        std::cerr << "No backends available" << std::endl;
        return 0; // treat as skip
    }

    int n_ok = 0;
    bool ran_any = false;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        // Focus on CUDA backends only; skip others
        if (std::string(name).find("CUDA") == std::string::npos) {
            continue;
        }
        ran_any = true;
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) {
            std::cerr << "[SKIP] cannot init backend: " << name << std::endl;
            continue;
        }

        int rc = test_backend_matmul_invariance(backend);
        if (rc == 0) {
            rc = test_backend_matmul_id_invariance(backend);
        }
        if (rc == 0) {
            rc = test_edge_probes_minimal(backend);
        }
        if (rc == 0) {
            std::cout << "[OK] " << name << std::endl;
            n_ok++;
        } else {
            std::cerr << "[FAIL] " << name << " rc=" << rc << std::endl;
            ggml_backend_free(backend);
            ggml_quantize_free();
            return 1;
        }
        ggml_backend_free(backend);
    }
    ggml_quantize_free();

    if (!ran_any) {
        std::cerr << "[SKIP] No CUDA backend found" << std::endl;
        return 0;
    }
    std::cout << "CUDA backends passed: " << n_ok << std::endl;
    return 0;
}
