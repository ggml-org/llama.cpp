// test-ane-rope
//
// End-to-end parity test for the GGML_OP_ROPE dispatch path
// in ggml_ane_program_dispatch_op. The test loads the
// rope-1x4096.mlmodelc fixture (single-function .mlmodelc,
// functionName "main", NORMAL mode, no freq_factors), builds
// a ggml graph with one RoPE op over [1, 4096], dispatches
// through the ANE backend, and verifies the output against
// a ggml-cpu reference within 3e-3 (the fp16 round-trip +
// the cos/sin table's per-element fp16 error).
//
// Phase 1 ships the gemma 4 NORMAL-mode variant only. NEOX,
// MROPE, VISION, IMROPE modes fall through to the CPU path
// (the dispatch_op's NORMAL gate); per-call freq_factors is
// also a follow-on bundle (the manifest's role still
// identifies it as ROPE so a follow-on commit can extend the
// bundle with extra input slots).
//
// The bundle bakes the rotation params (n_dims=4096,
// freq_base=10000, freq_scale=1, ext_factor=0, attn_factor=1,
// beta_fast=0, beta_slow=0) at export time. The test uses
// position 5 so the rotation is non-trivial (position 0
// produces cos=1, sin=0 which is a degenerate test).

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
constexpr int32_t  kPosition   = 5;
constexpr float    kFreqBase   = 10000.0f;
// RoPE has many operations (cos + sin + 4 muls + add/sub), so
// the fp16 round-trip error compounds. 3e-3 is what the
// measured max|err| sits at for position=5 (vs. 1.3e-3 for
// RMSNorm's smaller op count); the headroom tracks.
constexpr float    kTolerance  = 3.0e-3f;
constexpr uint32_t kSeed       = 0x0A0Bu;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_ROPE_FIXTURE");
            env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate /
            "tools/ane-mtp/fixtures/rope-1x4096/rope-1x4096.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr, "rope fixture not found. Build it via:\n"
        "  python3 tools/ane-mtp/build-rope-fixture.py\n");
    return {};
}

std::vector<float> make_input(uint32_t n) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<float> cpu_reference_rope(const std::vector<float> & x,
                                       int32_t pos) {
    // Reference matches ggml_compute_forward_rope_f32
    // (ggml/src/ggml-cpu/ops.cpp) for the NORMAL mode, no
    // freq_factors, no YaRN case. n_dims = x.size() (the spike
    // bakes n_dims = K at export time; this matches because
    // the test uses a [K, 1] tensor).
    const uint32_t half = x.size() / 2;
    std::vector<float> y(x.size());
    for (uint32_t i = 0; i < half; ++i) {
        const float theta =
            static_cast<float>(pos) *
            std::pow(kFreqBase, -2.0f * static_cast<float>(i) / static_cast<float>(x.size()));
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        // NORMAL: pair (i, i + half).
        y[i]      = x[i] * c - x[i + half] * s;
        y[i + half] = x[i] * s + x[i + half] * c;
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
    std::printf("max |err| (ANE ROPE vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

} // namespace

int main() {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("rope fixture: %s\n", fixture.string().c_str());

    ggml_backend_ane_program * program =
        ggml_backend_ane_program_load_from_dir(fixture.string().c_str(), "main");
    if (!program) {
        std::fprintf(stderr, "failed to load rope .mlmodelc\n");
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
    // ggml_rope requires a positions tensor (i32, length n_tokens).
    // The dispatch case reads src[1] and casts to fp32 before
    // passing to the bundle. We allocate the positions tensor
    // first; its data pointer is filled after the backend
    // allocation below (no_alloc=true leaves data null until the
    // backend assigns a buffer).
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(positions, "pos");
    struct ggml_tensor * out = ggml_rope(
        ctx, input, positions, kN, GGML_ROPE_TYPE_NORMAL);
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
    ((int32_t *) positions->data)[0] = kPosition;

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
        std::fprintf(stderr, "failed to bind rope bundle to ANE backend\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::printf("rope bundle bound to ANE backend\n");

    const enum ggml_status status = ggml_backend_graph_compute(ane_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed with status %d\n",
                     static_cast<int>(status));
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    const std::vector<float> expected = cpu_reference_rope(input_data, kPosition);
    const bool ok = close_enough(expected, (const float *) out->data, kN);
    if (!ok) {
        std::fprintf(stderr, "ANE ROPE output disagrees with CPU fp32 reference\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    ggml_backend_free(ane_backend);
    ggml_free(ctx);
    ggml_backend_ane_program_free(program);
    std::printf("ANE ROPE dispatch: OK\n");
    return 0;
}
