// Minimal batch-invariance and determinism checks for RMSNorm across backends

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
#include <thread>

static void set_env_deterministic() {
#if defined(_WIN32)
    SetEnvironmentVariableA("GGML_DETERMINISTIC", "1");
#else
    setenv("GGML_DETERMINISTIC", "1", 1);
#endif
}

struct GraphOut {
    std::vector<float> data; // flattened [B,H] row-major by rows (row 0 first)
    int64_t H = 0;
    int64_t B = 0;
};

static GraphOut run_rmsnorm_graph(ggml_backend_t backend, const std::vector<float> & xin, int64_t B, int64_t H, float eps) {
    ggml_init_params ip = {
        /* .mem_size = */ ggml_tensor_overhead()*32 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, B);
    ggml_set_name(x, "x");

    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    ggml_set_name(y, "y");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        throw std::runtime_error("alloc tensors failed");
    }

    // copy input
    if ((int64_t)xin.size() != B*H) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("bad xin size");
    }
    ggml_backend_tensor_set(x, xin.data(), 0, sizeof(float)*xin.size());

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }

    GraphOut out;
    out.B = B; out.H = H;
    out.data.resize((size_t)B*H);
    ggml_backend_tensor_get(y, out.data.data(), 0, sizeof(float)*out.data.size());

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static bool bytes_equal(const float *a, const float *b, size_t n) {
    return std::memcmp(a, b, n*sizeof(float)) == 0;
}

static int test_backend_rms_invariance(ggml_backend_t backend) {
    const int64_t H = 4096; // representative hidden dim
    const float eps = 1e-6f;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Base row data
    std::vector<float> row0(H);
    for (int64_t i = 0; i < H; ++i) row0[i] = dist(rng);

    // B=1 case
    std::vector<float> x1(H);
    std::copy(row0.begin(), row0.end(), x1.begin());
    auto out1 = run_rmsnorm_graph(backend, x1, /*B=*/1, H, eps);

    // batch sizes to probe
    const int Bs[] = {3, 8, 32};
    for (int B : Bs) {
        std::vector<float> xb((size_t)B*H);
        // row 0 identical to B=1 input
        std::copy(row0.begin(), row0.end(), xb.begin());
        // fill remaining rows with randoms
        for (int r = 1; r < B; ++r) {
            for (int64_t c = 0; c < H; ++c) xb[(size_t)r*H + c] = dist(rng);
        }
        auto outb = run_rmsnorm_graph(backend, xb, B, H, eps);

        // compare row 0 bitwise: outb[0] vs out1[0]
        const float *y1 = out1.data.data();
        const float *yb0 = outb.data.data(); // first row
        if (!bytes_equal(y1, yb0, (size_t)H)) {
            std::cerr << "[FAIL] batch invariance: B=1 vs B=" << B << " differ on row 0\n";
            return 1;
        }
    }

    // Cross-run determinism: run same B=8 twice
    {
        const int B = 8;
        // build a fixed input
        std::vector<float> xb((size_t)B*H);
        rng.seed(42);
        for (float &v : xb) v = dist(rng);
        auto a = run_rmsnorm_graph(backend, xb, B, H, eps);
        auto b = run_rmsnorm_graph(backend, xb, B, H, eps);
        if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
            std::cerr << "[FAIL] cross-run determinism: repeated run differs\n";
            return 2;
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
        return 1;
    }

    int n_ok = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) {
            std::cerr << "[SKIP] cannot init backend: " << name << std::endl;
            continue;
        }

        // Set a reasonable n_threads if supported (CPU backend)
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_threads) set_threads(backend, std::thread::hardware_concurrency());

        int rc = test_backend_rms_invariance(backend);
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

    std::cout << "Backends passed: " << n_ok << "/" << n_dev << std::endl;
    return 0;
}
