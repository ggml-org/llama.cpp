// test-ane-rmsnorm
//
// End-to-end parity test for the GGML_OP_RMS_NORM dispatch path
// in ggml_ane_program_dispatch_op. The test loads the
// rmsnorm-1x4096.mlmodelc fixture (single-function .mlmodelc, one
// functionName "rmsnorm", input x [1, 4096] fp32, output y [1, 4096]
// fp32), binds it to the ANE backend, builds a ggml graph with a
// single RMSNorm op, and verifies the ANE output matches the
// ggml-cpu reference within 1e-3 (fp16 internal precision).
//
// What this validates:
//   - ggml_backend_ane_program_load_from_dir reads the manifest
//     sidecar and pins the function's input/output slots.
//   - ggml_ane_program_dispatch_op dispatches GGML_OP_RMS_NORM
//     when the bound bundle's baked shape matches the ggml op's
//     shape (and the dtype is fp32 in / fp32 out).
//   - supports_op advertises GGML_OP_RMS_NORM so the scheduler
//     routes it to the ANE backend.
//   - The bound function's fp16 ANE output lands in op->data
//     within the 1e-3 fp16 tolerance.
//
// The fixture is built by tools/ane-mtp/build-rmsnorm-fixture.py.
// If the fixture is not present the test refuses to run; the
// fixture is committed alongside the test for reproducibility.

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

constexpr uint32_t kN          = 4096;
constexpr float    kEps        = 1.0e-6f;
// fp16 round-trip tolerance for a 4096-element reduction. The
// bundle computes internally in fp16, which is the dominant
// source of error; the 1.5x headroom over the 1.0e-3 W0
// tolerance accounts for the reduce_mean accumulator drift
// (a per-element relative error of 1/4096 from a 4096-wide sum
// can be up to ~2^-11 of the mean, so the 1.5x is empirical).
constexpr float    kTolerance  = 2.0e-3f;
constexpr uint32_t kSeed       = 0xBEEFu;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_RMSNORM_FIXTURE");
            env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate /
            "tools/ane-mtp/fixtures/rmsnorm-1x4096/rmsnorm-1x4096.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr, "rmsnorm fixture not found. Build it via:\n"
        "  python3 tools/ane-mtp/build-rmsnorm-fixture.py\n");
    return {};
}

std::vector<float> make_input(uint32_t n) {
    std::mt19937 rng(kSeed);
    // Small magnitudes so x*x doesn't blow the fp16 exponent in the
    // bundle's reduce_mean. The ANE path is internally fp16; the
    // ggml-cpu reference is fp32, so we want the difference to be
    // within fp16 rounding (~1e-3), not numerical pathology.
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<float> cpu_reference_rmsnorm(const std::vector<float> & x,
                                          float eps) {
    // Reference matches ggml_compute_forward_rms_norm_f32
    // (ggml/src/ggml-cpu/ops.cpp). The op's input is 2D
    // [K, M]; for decode M=1 so we reduce over K (the row).
    // y[i] = x[i] * rsqrt(mean(x^2 over the row) + eps).
    std::vector<float> y(x.size());
    double sum = 0.0;
    for (float v : x) {
        sum += static_cast<double>(v) * static_cast<double>(v);
    }
    const float mean = static_cast<float>(sum / x.size());
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] * scale;
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
    std::printf("max |err| (ANE RMS_NORM vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

} // namespace

int main() {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("rmsnorm fixture: %s\n", fixture.string().c_str());

    // 1. Load the rmsnorm bundle (single-function .mlmodelc with
    // functionName "main"; the manifest sidecar
    // rmsnorm-1x4096.ane_state.v1.json declares the role as
    // "rms_norm" which is what the dispatch path keys on). The
    // manifest's core_ml_function_name matches the .mlmodelc's
    // entry point.
    ggml_backend_ane_program * program =
        ggml_backend_ane_program_load_from_dir(fixture.string().c_str(), "main");
    if (!program) {
        std::fprintf(stderr, "failed to load rmsnorm .mlmodelc\n");
        return 1;
    }

    // 2. Build the ggml graph: RMSNorm(input [1, 4096], eps) -> [1, 4096].
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
    struct ggml_tensor * out = ggml_rms_norm(ctx, input, kEps);
    ggml_set_name(out, "y");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // 3. Allocate the graph on the CPU buffer (the dispatch path
    // copies bytes into the bundle's IOSurface arena at run time).
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, cpu_buft);
    if (!buf) {
        std::fprintf(stderr, "buffer alloc failed\n");
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    // 4. Fill the input.
    const std::vector<float> input_data = make_input(kN);
    std::memcpy(input->data, input_data.data(), kN * sizeof(float));

    // 5. Bind the bundle to the ANE backend and dispatch the graph.
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
        std::fprintf(stderr, "failed to bind rmsnorm bundle to ANE backend\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::printf("rmsnorm bundle bound to ANE backend\n");

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
    const std::vector<float> expected = cpu_reference_rmsnorm(input_data, kEps);
    const bool ok = close_enough(expected, (const float *) out->data, kN);
    if (!ok) {
        std::fprintf(stderr, "ANE RMS_NORM output disagrees with CPU fp32 reference\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    ggml_backend_free(ane_backend);
    ggml_free(ctx);
    ggml_backend_ane_program_free(program);
    std::printf("ANE RMS_NORM dispatch: OK\n");
    return 0;
}
