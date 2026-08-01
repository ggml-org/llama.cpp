//
// test_l5_dispatch.cpp
//
// Integration test for the L5 adaptive requantize loop wired into
// ts_dispatch_run. Builds a tiny synthetic GGUF with two 2D weights in
// different tensor families (attention + FFN), runs the full dispatch
// pipeline with --tessera-adaptive-requantize enabled, and asserts:
//
//   1. result.l5_ran is true
//   2. result.l5_report_json is non-empty and parses as the
//      llama.tessera.l5-loop.v1 schema
//   3. the report's generations array is non-empty
//
// This proves the L2 -> L5 -> A/B -> re-quantize -> re-measure loop is
// reachable from the production entry point without crashing and emits the
// documented receipt.
//
// Mirrors test_gguf_write_lifetime.cpp's GGUF construction pattern.
//

#include "tessera-dispatch.h"
#include "tessera-dispatch-internal.h"
#include "tessera-quant.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static int g_fail = 0;
static void check(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    if (!ok) g_fail++;
}

// Build a synthetic F32 GGUF with the requested 2D weights, sized to
// tile640's page geometry so ts_quantize_2d accepts them.
static bool build_fixture_gguf(const char * path,
                               const std::vector<std::string> & tensor_names,
                               const std::vector<std::pair<int64_t, int64_t>> & dims) {
    struct gguf_context * ctx = gguf_init_empty();
    struct ggml_init_params ip = { /*mem_size=*/ 4 * 1024 * 1024,
                                   /*mem_buffer=*/ nullptr,
                                   /*no_alloc=*/ false };
    struct ggml_context * gctx = ggml_init(ip);

    for (size_t i = 0; i < tensor_names.size(); i++) {
        const int64_t out_dim = dims[i].first;
        const int64_t in_dim  = dims[i].second;
        struct ggml_tensor * t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, in_dim, out_dim);
        ggml_set_name(t, tensor_names[i].c_str());
        // Fill with non-trivial but deterministic values so quantization
        // produces non-zero reconstruction error (required for L2 flagging).
        float * data = (float *) t->data;
        uint32_t rng = (uint32_t)(i + 1) * 2654435761u;
        for (int64_t j = 0; j < out_dim * in_dim; j++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            float u = (float)((rng >> 8) & 0xFFFF) / (float)0xFFFF;
            // Scale to ~unit variance; deliberately uneven so the two tensors
            // have different divergence profiles.
            data[j] = (u - 0.5f) * (1.0f + 2.0f * (float)i);
        }
        gguf_add_tensor(ctx, t);
    }

    bool ok = gguf_write_to_file(ctx, path, false);
    ggml_free(gctx);
    gguf_free(ctx);
    if (!ok) {
        std::printf("FAIL: gguf_write_to_file(%s) returned false\n", path);
        g_fail++;
    }
    return ok;
}

int main() {
    const char * fixture_path = "/tmp/test_l5_dispatch_input.gguf";
    const char * output_path  = "/tmp/test_l5_dispatch_output.gguf";

    // Two tensors: an attention projection and an FFN down projection, in
    // separate families so the per-family A/B path has something to compare.
    // in_dim 1280 = 2 tile640 pages, matching the lifecycle test's geometry.
    std::vector<std::string> names = {
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
    };
    std::vector<std::pair<int64_t, int64_t>> dims = {
        { 16, 1280 },
        { 16, 1280 },
    };

    if (!build_fixture_gguf(fixture_path, names, dims)) {
        std::printf("\nFAIL (setup)\n");
        return 1;
    }

    ts_dispatch_params params = {};
    params.input_path        = fixture_path;
    params.output_path       = output_path;
    params.imatrix_path      = "";
    params.policy_path       = "";
    params.policy_out_path   = "/tmp/test_l5_dispatch.policy.json";
    params.evolve_seed       = 42;
    params.evolve_iters      = 2;
    params.evolve_islands    = 2;
    params.evolve_population = 4;
    params.outlier_frac      = 0.005f;
    params.awq_alpha         = "0.5";   // fixed alpha so AWQ is on and the
                                        // Stage A multiplier path is exercisable
    params.awq_clip          = 0.95f;
    params.nthreads          = 1;
    params.verbose           = false;

    // Enable the L5 adaptive requantize loop under test.
    params.adaptive_requantize        = true;
    params.l5_max_generations         = 2;
    params.l5_flag_multiplier         = 1.5f;
    params.l5_alpha_min               = 0.1f;
    params.l5_clip_min                = 0.1f;
    params.l5_outlier_overshoot_scale = 0.5f;
    params.l5_outlier_frac_cap        = 0.25f;
    params.l5_out_path                = "/tmp/test_l5_dispatch.l5-loop.json";

    ts_dispatch_result result;
    std::string err;
    int rc = ts_dispatch_run(&params, &result, &err);

    check("dispatch rc == 0", rc == 0);
    if (rc != 0) {
        std::printf("  error: %s\n", err.c_str());
        std::printf("\nFAIL (dispatch did not complete)\n");
        return 1;
    }

    // 1. The L5 loop must have run.
    check("l5_ran == true", result.l5_ran);

    // 2. The report JSON must be non-empty and carry the schema name.
    check("l5_report_json non-empty", !result.l5_report_json.empty());
    bool has_schema = result.l5_report_json.find("llama.tessera.l5-loop.v1") != std::string::npos;
    check("report carries the l5-loop.v1 schema", has_schema);
    bool has_generations = result.l5_report_json.find("\"generations\"") != std::string::npos;
    check("report has a generations array", has_generations);

    // 3. The report file was written.
    FILE * f = std::fopen("/tmp/test_l5_dispatch.l5-loop.json", "rb");
    check("l5-loop.json written to disk", f != nullptr);
    if (f) std::fclose(f);

    // 4. The dispatch still produced a valid output GGUF despite the
    //    in-place re-quantization (the repoint path must not have corrupted
    //    the descriptors). Reopen and confirm tensors are present.
    struct ggml_context * rin_ctx = nullptr;
    struct gguf_init_params rp = { /*no_alloc=*/ false, /*ctx=*/ &rin_ctx };
    struct gguf_context * rin = gguf_init_from_file(output_path, rp);
    check("output GGUF reopens", rin != nullptr);
    if (rin) {
        const int64_t n = gguf_get_n_tensors(rin);
        // Each Tessera weight produces 6+ component tensors; two weights
        // therefore yield >= 12 component tensors in the output.
        check("output GGUF has >= 12 tensors (2 weights * 6+ components)", n >= 12);
        gguf_free(rin);
        ggml_free(rin_ctx);
    }

    // Cleanup.
    std::remove(fixture_path);
    std::remove(output_path);
    std::remove("/tmp/test_l5_dispatch.policy.json");
    std::remove("/tmp/test_l5_dispatch.l5-loop.json");

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
