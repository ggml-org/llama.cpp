// test-ane-glu
//
// End-to-end parity test for the GGML_OP_GLU dispatch path
// (split form: ggml_glu_split with geglu) in
// ggml_ane_program_dispatch_op. The test loads the
// geglu-1x11008.mlmodelc fixture (single-function .mlmodelc,
// functionName "main", geglu activation baked), builds a ggml
// graph with one GLU op over [1, 11008], dispatches through
// the ANE backend, and verifies the output against a
// ggml-cpu reference within 2e-3 (the fp16 round-trip on
// the GELU + mul fusion is the dominant error source).
//
// Phase 1 ships geglu only (the gemma 4 FFN). swiglu, reglu,
// geglu_erf, geglu_quick, swiglu_oai fall through to the
// CPU/Accelerate path per the dispatch policy; a follow-on
// commit adds a second functionName per variant.
//
// The bundle uses the standard sigmoid-based GELU
// (0.5 * x * (1 + erf(x / sqrt(2)))), matching the ggml-cpu
// GELU_GATE path.

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

constexpr uint32_t kN          = 11008;
constexpr float    kTolerance  = 2.0e-3f;
constexpr uint32_t kSeed       = 0xFEEDu;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_GLU_FIXTURE");
            env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate /
            "tools/ane-mtp/fixtures/geglu-1x11008/geglu-1x11008.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr, "glu fixture not found. Build it via:\n"
        "  python3 tools/ane-mtp/build-glu-fixture.py\n");
    return {};
}

std::vector<float> make_input(uint32_t n) {
    std::mt19937 rng(kSeed);
    // Range [-1, 1]: GELU's curvature is meaningful in this
    // band; below -3 it's essentially zero, above +3 it's
    // essentially x.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<float> cpu_reference_geglu(const std::vector<float> & gate,
                                        const std::vector<float> & up) {
    // y = gelu(gate) * up
    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    std::vector<float> y(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        const float g = gate[i];
        const float gelu = 0.5f * g * (1.0f + std::erf(g * inv_sqrt2));
        y[i] = gelu * up[i];
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
    std::printf("max |err| (ANE GLU vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

} // namespace

int main() {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("glu fixture: %s\n", fixture.string().c_str());

    ggml_backend_ane_program * program =
        ggml_backend_ane_program_load_from_dir(fixture.string().c_str(), "main");
    if (!program) {
        std::fprintf(stderr, "failed to load glu .mlmodelc\n");
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

    struct ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, 1);
    ggml_set_name(gate, "gate");
    struct ggml_tensor * up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, 1);
    ggml_set_name(up, "up");
    struct ggml_tensor * out = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_GEGLU);
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

    const std::vector<float> gate_data = make_input(kN);
    // Use a different seed for up so the two inputs are
    // uncorrelated; the element-wise product would mask a
    // bug in the gate path if both were the same data.
    std::vector<float> up_data = make_input(kN);
    {
        std::mt19937 rng2(kSeed + 1);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (uint32_t i = 0; i < kN; ++i) {
            up_data[i] = dist(rng2);
        }
    }
    std::memcpy(gate->data, gate_data.data(), kN * sizeof(float));
    std::memcpy(up->data, up_data.data(), kN * sizeof(float));

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
        std::fprintf(stderr, "failed to bind glu bundle to ANE backend\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::printf("glu bundle bound to ANE backend\n");

    const enum ggml_status status = ggml_backend_graph_compute(ane_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed with status %d\n",
                     static_cast<int>(status));
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    const std::vector<float> expected = cpu_reference_geglu(gate_data, up_data);
    const bool ok = close_enough(expected, (const float *) out->data, kN);
    if (!ok) {
        std::fprintf(stderr, "ANE GLU output disagrees with CPU fp32 reference\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    ggml_backend_free(ane_backend);
    ggml_free(ctx);
    ggml_backend_ane_program_free(program);
    std::printf("ANE GLU dispatch: OK\n");
    return 0;
}
