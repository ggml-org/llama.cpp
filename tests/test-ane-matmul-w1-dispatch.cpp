// W1 ANE matmul dispatch spike
//
// End-to-end test of MUL_MAT dispatch through ggml_backend_ane_graph_compute.
// Reuses the W0 spike's 256x256 fp16 matmul .mlmodelc (the bundle has the
// weight baked; for a real model the .mlmodelc is rebuilt with the
// model-specific weights at load time).
//
// What this validates:
//   - MUL_MAT is wired through ggml_ane_program_dispatch_op
//   - The ggml op's src[0] (activation) and src[1] (weight) feed the
//     bundle correctly
//   - The bundle's fp16 matmul output lands in op->data within the
//     1e-3 tolerance the W0 spike established
//   - supports_op advertises MUL_MAT so the scheduler routes it to ANE
//
// What this does NOT validate (W1+ follow-on work):
//   - Prefill (M>1) routing (the layer-slab function)
//   - Multi-projection multi-function bundles
//   - Lock-free IOSurface-backed weight (B in the W1 plan)

#include "ggml.h"
#include "ggml-ane.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kN = 256;
constexpr float    kTolerance = 1.0e-3f;
constexpr uint32_t kSeed = 0xA11Eu;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_W0_FIXTURE"); env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate / "tools/ane-mtp/fixtures/w0-matmul/w0-256x256.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr, "W0 fixture not found. Build it via:\n"
        "  python3 tools/ane-mtp/make-w0-matmul.py --n 256 "
        "--output tools/ane-mtp/fixtures/w0-matmul/\n");
    return {};
}

std::vector<float> make_input(uint32_t n) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<float> load_weight(const fs::path & mlmodelc_dir, uint32_t n) {
    fs::path weight_path = mlmodelc_dir.parent_path() /
                           (mlmodelc_dir.stem().string() + ".weight.bin");
    if (!fs::is_regular_file(weight_path)) {
        std::fprintf(stderr, "weight sidecar not found: %s\n", weight_path.string().c_str());
        return {};
    }
    std::ifstream f(weight_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "could not open weight file %s\n", weight_path.string().c_str());
        return {};
    }
    const size_t expected = n * n * sizeof(float);
    f.seekg(0, std::ios::end);
    if ((size_t) f.tellg() != expected) {
        std::fprintf(stderr, "weight file size mismatch (expected %zu, got %lld)\n",
                     expected, (long long) f.tellg());
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> w(n * n);
    f.read(reinterpret_cast<char *>(w.data()), expected);
    return w;
}

std::vector<float> cpu_reference_matmul(const std::vector<float> & x,
                                       const std::vector<float> & w,
                                       uint32_t n) {
    // CoreML innerProduct: y[i] = sum_j W[i, j] * x[j]. W is row-major.
    std::vector<float> y(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < n; ++j) {
            acc += x[j] * w[i * n + j];
        }
        y[i] = acc;
    }
    return y;
}

bool close_enough(const std::vector<float> & expected,
                  const float * actual, uint32_t n) {
    float max_abs_err = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float err = std::fabs(expected[i] - actual[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }
    std::printf("max |err| (ANE MUL_MAT vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

} // namespace

int main() {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("W1 ANE MUL_MAT dispatch: %s\n", fixture.string().c_str());

    // 1. Load the W0 bundle.
    ggml_backend_ane_program * program =
        ggml_backend_ane_program_load_from_dir(fixture.string().c_str(), nullptr);
    if (!program) {
        std::fprintf(stderr, "failed to load .mlmodelc\n");
        return 1;
    }

    // 2. Build the ggml graph: MUL_MAT(activation [K], weight [K, N]) -> [N].
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        ggml_backend_ane_program_free(program);
        return 1;
    }

    struct ggml_tensor * activation = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kN);
    ggml_set_name(activation, "activation");
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, kN);
    ggml_set_name(weight, "weight");
    struct ggml_tensor * out = ggml_mul_mat(ctx, activation, weight);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // 3. Allocate the graph on the CPU buffer. The W1 spike uses the
    // CPU buffer (the ANE backend's IOSurface arena would require the
    // lock-free path which is B in the W1 plan). The dispatch copies
    // activation bytes into the bundle's arena at run-time, which is
    // a CPU-side memcpy; the lock-free IOSurface path is the follow-on.
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, cpu_buft);
    if (!buf) {
        std::fprintf(stderr, "buffer alloc failed\n");
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    // 4. Fill the inputs. The activation is a deterministic pattern;
    // the weight is the W0 bundle's baked weight (read from the
    // sidecar file so the test does not need to share an RNG with the
    // Python builder).
    const std::vector<float> input = make_input(kN);
    const std::vector<float> w = load_weight(fixture, kN);
    if (w.empty()) {
        std::fprintf(stderr, "could not load reference weight\n");
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::memcpy(activation->data, input.data(), kN * sizeof(float));
    std::memcpy(weight->data, w.data(), kN * kN * sizeof(float));

    // 5. Bind the bundle to the ANE backend and dispatch the graph.
    // The ANE backend is registered as GGML_BACKEND_DEVICE_TYPE_ACCEL,
    // but other accelerators (CPU BLAS, etc.) also register as ACCEL;
    // we find the ANE device by its canonical name "ANE".
    ggml_backend_t ane_backend = nullptr;
    {
        const size_t n_devs = ggml_backend_dev_count();
        std::fprintf(stderr, "registered devices: %zu\n", n_devs);
        for (size_t i = 0; i < n_devs; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            std::fprintf(stderr, "  [%zu] %s (type=%d)\n",
                         i, ggml_backend_dev_name(d), (int) ggml_backend_dev_type(d));
        }
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("ANE");
        if (!dev) {
            std::fprintf(stderr, "no ANE device available (non-macOS?)\n");
            ggml_free(ctx);
            ggml_backend_ane_program_free(program);
            return 1;
        }
        ane_backend = ggml_backend_dev_init(dev, nullptr);
    }
    if (!ane_backend) {
        std::fprintf(stderr, "ANE backend init failed\n");
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    if (!ggml_backend_is_ane(ane_backend)) {
        std::fprintf(stderr, "backend is not ANE (got %s)\n",
                     ggml_backend_name(ane_backend));
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    if (!ggml_backend_ane_set_program(ane_backend, program)) {
        std::fprintf(stderr, "failed to bind W0 bundle to ANE backend\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::printf("W0 bundle bound to ANE backend\n");

    const enum ggml_status status = ggml_backend_graph_compute(ane_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed with status %d\n",
                     static_cast<int>(status));
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    // 6. Verify the output against the CPU reference.
    const std::vector<float> expected = cpu_reference_matmul(input, w, kN);
    const bool ok = close_enough(expected, (const float *) out->data, kN);
    if (!ok) {
        std::fprintf(stderr, "ANE MUL_MAT output disagrees with CPU fp32 reference\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    ggml_backend_free(ane_backend);
    ggml_free(ctx);
    ggml_backend_ane_program_free(program);
    std::printf("W1 ANE MUL_MAT dispatch: OK\n");
    return 0;
}
