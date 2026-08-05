// test-ane-softmax
//
// End-to-end parity test for the GGML_OP_SOFT_MAX dispatch path
// in ggml_ane_program_dispatch_op. The test loads the
// softmax-1x1024.mlmodelc fixture (single-function .mlmodelc,
// functionName "main", input x [1, 1024] fp32, output y [1, 1024]
// fp32), binds it to the ANE backend, builds a ggml graph with a
// single SoftMax op, and verifies the ANE output matches the
// ggml-cpu reference within 2e-3 (the fp16 internal precision
// dominates; vanilla softmax's normalize-by-sum keeps the error
// bounded).
//
// The test is the second of five Phase 1 body-op parity tests.
// Same shape contract as test-ane-rmsnorm.cpp: M=1 decode, fp32
// in/out, the bundle bakes scale=1 and max_bias=0 (the standard
// attention-softmax values for the per-head row).

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
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kN          = 1024;
constexpr float    kScale      = 1.0f;
constexpr float    kMaxBias    = 0.0f;
constexpr float    kTolerance  = 2.0e-3f;
constexpr uint32_t kSeed       = 0x50F7u;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_SOFTMAX_FIXTURE");
            env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate /
            "tools/ane-mtp/fixtures/softmax-1x1024/softmax-1x1024.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr, "softmax fixture not found. Build it via:\n"
        "  python3 tools/ane-mtp/build-softmax-fixture.py\n");
    return {};
}

std::vector<float> make_input(uint32_t n) {
    std::mt19937 rng(kSeed);
    // Widen the range a bit (x2) so the fp16 exp() is exercised
    // at values where the fp16 round-trip matters more.
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<float> cpu_reference_softmax(const std::vector<float> & x) {
    // Vanilla softmax; matches ggml_compute_forward_soft_max_f32
    // (ggml/src/ggml-cpu/ops.cpp) for the (scale=1, max_bias=0)
    // case this spike covers. y = exp(x - max(x)) / sum(exp(x - max(x))).
    std::vector<float> y(x.size());
    float mx = x[0];
    for (float v : x) {
        if (v > mx) {
            mx = v;
        }
    }
    double sum = 0.0;
    std::vector<double> ex(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ex[i] = std::exp(static_cast<double>(x[i]) - static_cast<double>(mx));
        sum += ex[i];
    }
    const double inv = 1.0 / sum;
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = static_cast<float>(ex[i] * inv);
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
    std::printf("max |err| (ANE SOFT_MAX vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

} // namespace

int main() {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("softmax fixture: %s\n", fixture.string().c_str());

    ggml_backend_ane_program * program =
        ggml_backend_ane_program_load_from_dir(fixture.string().c_str(), "main");
    if (!program) {
        std::fprintf(stderr, "failed to load softmax .mlmodelc\n");
        return 1;
    }

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

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, 1);
    ggml_set_name(input, "x");
    struct ggml_tensor * out = ggml_soft_max(ctx, input);
    ggml_set_name(out, "y");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, cpu_buft);
    if (!buf) {
        std::fprintf(stderr, "buffer alloc failed\n");
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    const std::vector<float> input_data = make_input(kN);
    std::memcpy(input->data, input_data.data(), kN * sizeof(float));
    (void) kScale;
    (void) kMaxBias;

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("ANE");
    if (!dev) {
        std::fprintf(stderr, "no ANE device available (non-macOS?)\n");
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    ggml_backend_t ane_backend = ggml_backend_dev_init(dev, nullptr);
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
        std::fprintf(stderr, "failed to bind softmax bundle to ANE backend\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::printf("softmax bundle bound to ANE backend\n");

    const enum ggml_status status = ggml_backend_graph_compute(ane_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed with status %d\n",
                     static_cast<int>(status));
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    const std::vector<float> expected = cpu_reference_softmax(input_data);
    const bool ok = close_enough(expected, (const float *) out->data, kN);
    if (!ok) {
        std::fprintf(stderr, "ANE SOFT_MAX output disagrees with CPU fp32 reference\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    ggml_backend_free(ane_backend);
    ggml_free(ctx);
    ggml_backend_ane_program_free(program);
    std::printf("ANE SOFT_MAX dispatch: OK\n");
    return 0;
}
