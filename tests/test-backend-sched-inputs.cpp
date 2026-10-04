// Tensors the backend scheduler copies from the CPU to the device of the split that reads them.
// Checks that
//  - a user input is taken when the graph is queued: the caller overwrites its input data right
//    after ggml_backend_sched_graph_compute_async, and the result must not change,
//  - the copy is ordered against the device's queued work: a second evaluation queued before
//    the first one completed overwrites the same device copy of the inputs, and both results,
//    the first read back asynchronously before the second is queued, must be right,
//  - a tensor computed on the CPU reaches the device intact, for copies on both sides of the
//    backend's size limit for copies in its compute stream (the Vulkan backend's
//    GGML_VK_COMPUTE_STREAM_COPY_MAX, set here to 64 KiB unless the environment sets it;
//    with --measured, the backend measures its own).
// Run for each GPU device, with the CPU tensors in plain host memory and in the device's host
// buffer type when it has one, with tensors of 16 KiB and 4 MiB.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr int n_layers = 8;   // device multiplications per evaluation

struct model {
    int64_t n = 0;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr, buf_c = nullptr;
    ggml_tensor * w = nullptr;  // on the device
    ggml_tensor * c = nullptr;  // on the CPU: the first multiplication runs there
    std::vector<float> w_data, c_data;
};

// out = (... (((x * c) * w + y) * w + y) ...): x * c on the CPU, the rest on the device
static ggml_cgraph * build_graph(ggml_context * ctx, const model & m, ggml_tensor ** x, ggml_tensor ** y, ggml_tensor ** out) {
    *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m.n);
    *y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m.n);
    ggml_set_name(*x, "x");
    ggml_set_name(*y, "y");
    ggml_set_input(*x);
    ggml_set_input(*y);
    ggml_tensor * cur = ggml_mul(ctx, *x, m.c);
    ggml_set_name(cur, "x_c");
    for (int l = 0; l < n_layers; l++) {
        cur = ggml_add(ctx, ggml_mul(ctx, cur, m.w), *y);
    }
    *out = cur;
    ggml_set_name(*out, "out");
    ggml_set_output(*out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

static std::vector<float> reference(const model & m, const std::vector<float> & x, const std::vector<float> & y) {
    std::vector<float> r(m.n);
    for (int64_t i = 0; i < m.n; i++) {
        float cur = x[i] * m.c_data[i];
        for (int l = 0; l < n_layers; l++) {
            cur = cur * m.w_data[i] + y[i];
        }
        r[i] = cur;
    }
    return r;
}

static bool check(const char * what, const std::vector<float> & got, const std::vector<float> & want) {
    for (size_t i = 0; i < want.size(); i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f * (1.0f + std::fabs(want[i]))) {
            fprintf(stderr, "  FAIL %s: element %zu is %f, expected %f\n", what, i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

static std::vector<float> values(int64_t n, int seed) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = 0.25f * (float) ((i * 7 + seed * 13) % 9) - 1.0f;
    }
    return v;
}

static bool run(ggml_backend_dev_t dev, bool host_buffer, int64_t n) {
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) {
        fprintf(stderr, "  cannot initialize the backends\n");
        return false;
    }
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (host_buffer && !host_buft) {
        printf("  %s: no host buffer type, skipped\n", ggml_backend_dev_name(dev));
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return true;
    }
    ggml_backend_buffer_type_t cpu_buft = host_buffer ? host_buft : ggml_backend_get_default_buffer_type(cpu);

    model m;
    m.n = n;
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    m.ctx_w = ggml_init(wp);
    m.w = ggml_new_tensor_1d(m.ctx_w, GGML_TYPE_F32, n);
    m.buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, gpu);
    ggml_context * ctx_c = ggml_init(wp);
    m.c = ggml_new_tensor_1d(ctx_c, GGML_TYPE_F32, n);
    m.buf_c = ggml_backend_alloc_ctx_tensors_from_buft(ctx_c, cpu_buft);
    ggml_backend_buffer_set_usage(m.buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(m.buf_c, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.w_data.resize(n);
    m.c_data.resize(n);
    for (int64_t i = 0; i < n; i++) {
        m.w_data[i] = 0.5f + 0.125f * (float) (i % 5);
        m.c_data[i] = 1.0f + 0.25f * (float) (i % 3);
    }
    ggml_backend_tensor_set(m.w, m.w_data.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(m.c, m.c_data.data(), 0, n * sizeof(float));

    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu), cpu_buft };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);

    ggml_init_params gp = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x, * y, * out;
    ggml_cgraph * gf = build_graph(ctx, m, &x, &y, &out);

    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        fprintf(stderr, "  cannot allocate the graph\n");
    }
    if (ok && (ggml_backend_sched_get_tensor_backend(sched, out) != gpu ||
               ggml_backend_sched_get_tensor_backend(sched, ggml_graph_get_tensor(gf, "x_c")) != cpu)) {
        fprintf(stderr, "  the graph is not split between the CPU and %s\n", ggml_backend_dev_name(dev));
        ok = false;
    }

    const size_t bytes = n * sizeof(float);
    const std::vector<float> garbage(n, NAN);
    std::vector<float> got_a(n), got_b(n);
    for (int iter = 0; ok && iter < 4; iter++) {
        const std::vector<float> xa = values(n, 4 * iter), ya = values(n, 4 * iter + 1);
        const std::vector<float> xb = values(n, 4 * iter + 2), yb = values(n, 4 * iter + 3);

        // A: queue it, then overwrite the inputs at once
        ggml_backend_tensor_set(x, xa.data(), 0, bytes);
        ggml_backend_tensor_set(y, ya.data(), 0, bytes);
        ok = ggml_backend_sched_graph_compute_async(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_set(x, garbage.data(), 0, bytes);
        ggml_backend_tensor_set(y, garbage.data(), 0, bytes);
        ggml_backend_tensor_get_async(gpu, out, got_a.data(), 0, bytes);

        // B: queued behind A, into the same device copies of the inputs
        ggml_backend_tensor_set(x, xb.data(), 0, bytes);
        ggml_backend_tensor_set(y, yb.data(), 0, bytes);
        ok = ok && ggml_backend_sched_graph_compute_async(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_set(x, garbage.data(), 0, bytes);
        ggml_backend_tensor_set(y, garbage.data(), 0, bytes);
        ggml_backend_sched_synchronize(sched);
        ggml_backend_tensor_get(out, got_b.data(), 0, bytes);

        ok = ok && check("first evaluation", got_a, reference(m, xa, ya));
        ok = ok && check("second evaluation", got_b, reference(m, xb, yb));
    }

    printf("  %s, %lld KiB tensors, CPU tensors in %s: %s\n", ggml_backend_dev_name(dev), (long long) (bytes / 1024),
           host_buffer ? ggml_backend_buft_name(host_buft) : "host memory", ok ? "OK" : "FAIL");

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(m.buf_w);
    ggml_backend_buffer_free(m.buf_c);
    ggml_free(m.ctx_w);
    ggml_free(ctx_c);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return ok;
}

int main(int argc, char ** argv) {
    const bool measured = argc > 1 && std::string(argv[1]) == "--measured";
    // below and above this, CPU-to-device copies take the compute stream and the transfer queue
    if (!measured && !getenv("GGML_VK_COMPUTE_STREAM_COPY_MAX")) {
#ifdef _WIN32
        _putenv_s("GGML_VK_COMPUTE_STREAM_COPY_MAX", "65536");
#else
        setenv("GGML_VK_COMPUTE_STREAM_COPY_MAX", "65536", 0);
#endif
    }
    ggml_backend_load_all();

    int tested = 0;
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        for (int64_t n : { (int64_t) 4096, (int64_t) 1 << 20 }) {
            for (bool host_buffer : { false, true }) {
                ok = run(dev, host_buffer, n) && ok;
            }
        }
        tested++;
    }
    if (!tested) {
        printf("no GPU device: skipped\n");
        return 0;
    }
    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
