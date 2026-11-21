// A CUDA VMM pool OOM in MMQ must make graph_compute return GGML_STATUS_ALLOC_FAILED instead of aborting.
// After the memory is freed, the same graph must compute and match a run that did not fail.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Q4_0 weights use MMQ, which takes a few MiB from the pool for src1
static const int64_t K = 4096;
static const int64_t M = 4096;
static const int64_t N = 512;

struct mm_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         out = nullptr;
    ggml_cgraph *         gf  = nullptr;

    mm_graph(ggml_backend_t backend, const std::vector<uint8_t> & a, const std::vector<float> & b) {
        ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(params);
        ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, M);
        ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        out = ggml_mul_mat(ctx, ta, tb);
        gf  = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(ta, a.data(), 0, a.size());
        ggml_backend_tensor_set(tb, b.data(), 0, b.size()*sizeof(float));
    }

    ~mm_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    std::vector<float> get() const {
        std::vector<float> res(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        return res;
    }
};

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        printf("SKIP: no CUDA device\n");
        return 77;
    }

    std::vector<uint8_t> a(ggml_row_size(GGML_TYPE_Q4_0, K) * M);
    std::vector<float>   b(K * N);
    {
        std::vector<float> a_f32(K * M);
        for (size_t i = 0; i < a_f32.size(); ++i) {
            a_f32[i] = (float) ((i * 2654435761u) % 1000) / 500.0f - 1.0f;
        }
        ggml_quantize_chunk(GGML_TYPE_Q4_0, a_f32.data(), a.data(), 0, M, K, nullptr);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = (float) ((i * 40503u) % 1000) / 500.0f - 1.0f;
        }
    }

    // reference on another backend, this also loads the kernels before the device is full
    ggml_backend_t backend_ref = ggml_backend_dev_init(dev, nullptr);
    std::vector<float> ref;
    {
        mm_graph g(backend_ref, a, b);
        ggml_backend_graph_compute(backend_ref, g.gf);
        ref = g.get();
    }

    // the pool of a new backend is empty, so the compute must grow it
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_synchronize(backend); // creates the stream before the device is full
    bool ok;
    {
        mm_graph g(backend, a, b);

        // fill the device memory, the failed allocations are expected
        std::vector<ggml_backend_buffer_t> fill;
        ggml_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        for (size_t size = 1ull << 30; size >= 1ull << 20; ) {
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), size);
            if (buf) {
                fill.push_back(buf);
            } else {
                size /= 2;
            }
        }
        ggml_log_set(nullptr, nullptr);

        const ggml_status st_oom = ggml_backend_graph_compute(backend, g.gf);

        for (ggml_backend_buffer_t buf : fill) {
            ggml_backend_buffer_free(buf);
        }

        const ggml_status st = ggml_backend_graph_compute(backend, g.gf);
        ok = st_oom == GGML_STATUS_ALLOC_FAILED && st == GGML_STATUS_SUCCESS && g.get() == ref;
        printf("full device: %s, after free: %s\n", ggml_status_to_string(st_oom), ggml_status_to_string(st));
    }

    ggml_backend_free(backend);
    ggml_backend_free(backend_ref);

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
