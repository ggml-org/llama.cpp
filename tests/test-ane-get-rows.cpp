// test-ane-get-rows
//
// End-to-end parity test for the GGML_OP_GET_ROWS dispatch path
// in ggml_ane_program_dispatch_op. The test loads the
// getrows-4x128x64.mlmodelc fixture (single-function .mlmodelc,
// functionName "main", table [128, 64] fp32, ids [4] i32, output
// [4, 64] fp32), builds a ggml graph with one GetRows op, and
// verifies the ANE output matches the ggml-cpu reference.
//
// Phase 1 ships the small-vocab case (vocab <= 128); the
// production gemma 4 vocab=~256k goes through the ggml-cpu
// memcpy path per the dispatch policy. The spike is the ANE
// path: a small embedding table and a small batch, where
// IOSurface-bound gather is competitive with a host memcpy.

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

constexpr uint32_t kBatch    = 4;
constexpr uint32_t kVocab    = 128;
constexpr uint32_t kHidden   = 64;
// Gather is a pure memory copy per row; fp16 round-trip is
// minimal (no compute on the per-row values). The tolerance
// is conservative at 1e-3 to leave headroom for the gather
// kernel's per-element conversion.
constexpr float    kTolerance = 1.0e-3f;
constexpr uint32_t kSeed     = 0xC0DEu;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_GETROWS_FIXTURE");
            env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate /
            "tools/ane-mtp/fixtures/getrows-4x128x64/getrows-4x128x64.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr, "getrows fixture not found. Build it via:\n"
        "  python3 tools/ane-mtp/build-get-rows-fixture.py\n");
    return {};
}

std::vector<float> make_table(uint32_t vocab, uint32_t hidden) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(vocab * hidden);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<int32_t> make_ids(uint32_t batch, uint32_t vocab) {
    std::mt19937 rng(kSeed + 1);
    std::uniform_int_distribution<int32_t> dist(0, (int32_t) vocab - 1);
    std::vector<int32_t> v(batch);
    for (uint32_t i = 0; i < batch; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<float> cpu_reference_get_rows(const std::vector<float> & table,
                                          const std::vector<int32_t> & ids,
                                          uint32_t hidden,
                                          uint32_t vocab) {
    // The bundle declares the table as [hidden, vocab] in
    // CoreML's row-major view (ggml's ne[0]=hidden, ne[1]=vocab),
    // so the flat data is laid out as vocab consecutive values
    // per hidden row. The reference matches the bundle's
    // gather-axis=1 math:
    //   for i in 0..hidden:
    //     for j in 0..batch:
    //       out[i, j] = table[i, ids[j]]  (row-major flat offset)
    //   flat offset for [hidden, vocab] row-major: i * vocab + v
    // The output is also [hidden, batch] in the same layout.
    std::vector<float> out(hidden * ids.size());
    for (uint32_t i = 0; i < hidden; ++i) {
        for (size_t j = 0; j < ids.size(); ++j) {
            const int32_t v = ids[j];
            // table[i, v] in [hidden, vocab] row-major: i * vocab + v.
            out[i * ids.size() + j] = table[i * vocab + v];
        }
    }
    return out;
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
    std::printf("max |err| (ANE GET_ROWS vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

} // namespace

int main() {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("getrows fixture: %s\n", fixture.string().c_str());

    ggml_backend_ane_program * program =
        ggml_backend_ane_program_load_from_dir(fixture.string().c_str(), "main");
    if (!program) {
        std::fprintf(stderr, "failed to load getrows .mlmodelc\n");
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

    // ggml_get_rows(a, b): a is the embedding table [vocab, hidden]
    // (or [hidden, vocab] depending on layout), b is the i32 ids
    // [batch]. The output is [batch, hidden] (or [hidden, batch]
    // depending on a's layout). For the spike we use the
    // [vocab, hidden] / [batch, hidden] convention.
    struct ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kVocab);
    ggml_set_name(table, "table");
    struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kBatch);
    ggml_set_name(ids, "ids");
    struct ggml_tensor * out = ggml_get_rows(ctx, table, ids);
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

    const std::vector<float> table_data = make_table(kVocab, kHidden);
    const std::vector<int32_t> ids_data  = make_ids(kBatch, kVocab);
    // ggml_get_rows: the table tensor is [hidden, vocab] in
    // ggml's column-major view (ne[0]=hidden, ne[1]=vocab);
    // the data is laid out with stride hidden per vocab row.
    // The test data above is built [vocab, hidden] in row-major
    // (the natural layout for make_table); we copy it through
    // directly because the flat data is the same regardless of
    // the row/col interpretation.
    std::memcpy(table->data, table_data.data(), kVocab * kHidden * sizeof(float));
    std::memcpy(ids->data, ids_data.data(), kBatch * sizeof(int32_t));

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
        std::fprintf(stderr, "failed to bind getrows bundle to ANE backend\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }
    std::printf("getrows bundle bound to ANE backend\n");

    const enum ggml_status status = ggml_backend_graph_compute(ane_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed with status %d\n",
                     static_cast<int>(status));
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    // The reference and the ANE output are both [hidden, batch]
    // in ggml's column-major view; the flat data is the same.
    const std::vector<float> expected = cpu_reference_get_rows(
        table_data, ids_data, kHidden, kVocab);
    const bool ok = close_enough(expected, (const float *) out->data, kHidden * kBatch);
    if (!ok) {
        std::fprintf(stderr, "ANE GET_ROWS output disagrees with CPU fp32 reference\n");
        ggml_backend_free(ane_backend);
        ggml_free(ctx);
        ggml_backend_ane_program_free(program);
        return 1;
    }

    ggml_backend_free(ane_backend);
    ggml_free(ctx);
    ggml_backend_ane_program_free(program);
    std::printf("ANE GET_ROWS dispatch: OK\n");
    return 0;
}
